from common_header import *
import NNModels
import data
import Loss_Functions
import Plotter
import smt_verification
from dreal import *

# Load the configuration file
config_file = os.environ.get('CONFIG_FILE', 'config.json')
with open(config_file) as file:
    config = json.load(file)
if config["wandb"] == "None":
    wandb_name = "LASA"
else:
    wandb_name = config["wandb"]["name"]
wandb.init(project=wandb_name, config=config)
class MotionPlanner:
    def __init__(self):
        self.N_domain = config["domain"]["N"]
        self.hidden_neurons_f = config["model_f"]["hidden_neurons"]
        self.hidden_layers_f = config["model_f"]["layers"]
        self.device = config["device"]
        self.dim_in = config["dim_in"]
        # For Verification
        x1 = Variable("x1")
        x2 = Variable("x2")
        self.vars_ = [x1,x2]
        self.smt_config = Config()
        self.smt_config.use_polytope_in_forall = True
        self.smt_config.use_local_optimization = True
        self.smt_config.precision = 1e-5
        self.beta = 1e-3
        self.ball_lb = 1e-2
        self.ball_ub = 1.01
    
    def generateData(self):
        self.X_train, self.y_train, self.X_test, self.y_test = data.generateReferenceData()
        # Get initial set center
        self.x_domain = data.generateData(self.N_domain, "domain").to(self.device)
        if config["Barrier"]:
            # Get Init Data Points
            # Get initial set center
            N = config["dataset"]["datashape"]
            n = int(self.X_train.shape[0]/N)
            mean_point = [0,0]
            for i in range(n):
                mean_point += self.X_train[(i-1)*N]
            mean_point /= n
            self.initial_set_center = mean_point
            self.N_init = config["init"]["N"]
            self.x_init = data.generateData(self.N_init, "init", self.initial_set_center) 
            # Get Unsafe Data Points
            self.N_unsafe = config["unsafe"]["N"]
            self.x_unsafe = data.generateData(self.N_unsafe, "unsafe").to(self.device)
        else:
            # Get Init Data Points
            # Get initial set center
            N = config["dataset"]["datashape"]
            n = int(self.X_train.shape[0]/N)
            mean_point = [0,0]
            for i in range(n):
                mean_point += self.X_train[(i-1)*N]
            mean_point /= n
            self.initial_set_center = mean_point
            self.N_init = config["init"]["N"]
            self.x_init = data.generateData(self.N_init, "init", self.initial_set_center).to(self.device) 

    def trainInitialDynamics(self):
        self.hidden_f = [self.hidden_neurons_f] * self.hidden_layers_f
        self.model_f = NNModels.DyanmicsNet(self.dim_in, self.hidden_f).to(self.device)
        best_mse = np.inf   # init to infinity
        best_weights = None
        history = []
        loss_fn = nn.MSELoss()  # mean square error
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr = config["model_f"]["learning_rate"])
        for epoch in range(config["model_f"]["epochs_warm"]):
            self.model_f.train()
            # Calculate the loss
            y_pred = self.model_f(torch.tensor(self.X_train, dtype=torch.float32).to(self.device))
            loss = loss_fn(y_pred, torch.tensor(self.y_train, dtype=torch.float32).to(self.device))
            # backward pass
            optimizer_f.zero_grad()
            loss.backward()
            optimizer_f.step()
            #evaluate accuracy at end of each epoch
            self.model_f.eval()
            y_pred = self.model_f(torch.tensor(self.X_test, dtype=torch.float32).to(self.device))
            mse = loss_fn(y_pred, torch.tensor(self.y_test).to(self.device))
            mse = float(mse)
            history.append(mse)
            wandb.log({"DS_training_loss": loss.item()})
            if loss < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(self.model_f.state_dict())
            with torch.no_grad():
                torch.cuda.empty_cache()
        # restore model and return best accuracy
        self.model_f.load_state_dict(best_weights)
        print_info("MSE of Initial Estimate of Dynamical System: %.4f" % best_mse)
    
    def trainLyapunovFunction(self):
        hidden_neurons_v = config["model_v"]["hidden_neurons"]
        hidden_layers_v = config["model_v"]["layers"]
        hidden_v = [hidden_neurons_v] * hidden_layers_v
        self.model_v = NNModels.LyapunovNet(
            n_input=self.dim_in,
            hidden_v=hidden_v,
            hidden_f=self.hidden_f,
            model_f=self.model_f
        ).to(self.device)      
        max_iters = config["model_v"]["max_iters"]
        optimizer_v = torch.optim.Adam(self.model_v.parameters(), lr = config["model_v"]["learning_rate"])
        # Provide the start factor and end factor
        if config["model_v"]["scheduler"]["start_factor"] == "None":
            start_factor = 1.0
        else:
            start_factor = config["model_v"]["scheduler"]["start_factor"]
        if config["model_v"]["scheduler"]["end_factor"] == "None":
            end_factor = 0.0001
        else:
            end_factor = config["model_v"]["scheduler"]["end_factor"]
        scheduler_v = torch.optim.lr_scheduler.LinearLR(optimizer_v, start_factor=start_factor, end_factor=end_factor, total_iters=max_iters)
        self.model_v.train()
        # Starting with Sampling Based Verification
        start = timeit.default_timer()
        for i in range(max_iters):
            lyapunov_risk = Loss_Functions.loss_function_v(self.model_v, self.x_domain, torch.tensor(self.X_train,dtype=torch.float32).to(self.device), torch.tensor(self.y_train,dtype=torch.float32).to(self.device), i)
            optimizer_v.zero_grad()
            lyapunov_risk.backward()
            optimizer_v.step()
            scheduler_v.step()
            if i%200 == 0:
                self.x_domain, flag = Loss_Functions.lyapunovVerify(self.model_v, self.x_domain, i)    
                if flag:
                    print_info("Completed with Sampling Based Training. Proceeding with SMT Verification")
                    break
            if i%5000 == 0:
                Plotter.lyapunovBarrierPlot(mp.model_v, mp.X_train, mp.initial_set_center)
        stop_ = timeit.default_timer()
        print_info(f"Sampling Verification Time: {stop_ - start}")
        # SMT Verification
        optimizer_v = torch.optim.Adam(self.model_v.parameters(), lr = config["model_v"]["learning_rate"])
        scheduler_v = torch.optim.lr_scheduler.LinearLR(optimizer_v, start_factor=1.0, end_factor=0.01, total_iters=max_iters)
        self.model_v.train()
        verified_flag = False
        for i in range(max_iters):
            lyapunov_risk = Loss_Functions.loss_function_v(self.model_v, self.x_domain, torch.tensor(self.X_train,dtype=torch.float32).to(self.device), torch.tensor(self.y_train,dtype=torch.float32).to(self.device), i)
            optimizer_v.zero_grad()
            lyapunov_risk.backward()
            optimizer_v.step()
            scheduler_v.step()
            if i%200 == 0:
                # Finding the values for the new model function
                z = self.vars_
                for idx, layer in enumerate(self.model_v.layers_f[:-1]):
                    w = layer.weight.data.cpu().numpy()
                    b = layer.bias.data.cpu().numpy()
                    zhat = w @ z + b
                    z = smt_verification.hyper_tan_dr(zhat)
                last_layer = self.model_v.layers_f[-1].weight.data.cpu().numpy()
                z = last_layer @ z
                z += self.model_v.layers_f[-1].bias.data.cpu().numpy()
                f_learn = z
                # save the weights and biases
                z = self.vars_
                jacobian = np.eye(self.model_v.input_size, self.model_v.input_size)
                for idx, layer in enumerate(self.model_v.layers_v[:]):
                    w = layer.weight.data.cpu().numpy()
                    b = layer.bias.data.cpu().numpy()
                    zhat = w @ z + b
                    z = smt_verification.hyper_tan_dr(zhat)
                    # Vdot
                    jacobian = w @ jacobian
                    jacobian = np.diagflat(smt_verification.hyper_tan_der_dr(zhat)) @ jacobian
                V_learn = z[0]
                gradV = np.multiply(jacobian, np.broadcast_to(1, jacobian.shape))
                V_learn_dot = (gradV @ f_learn)[0]
                print_info('===========Verifying==========')
                start_ = timeit.default_timer()
                result = smt_verification.CheckLyapunov(self.vars_, f_learn, V_learn, V_learn_dot, self.ball_lb, self.ball_ub, self.smt_config, self.beta) # SMT solver
                if result:
                    x_domain = self.x_domain.to('cpu')
                    x_domain = smt_verification.AddCounterexamples(x_domain, result, 10)
                    self.x_domain = x_domain.float().to(self.device)
                else:
                    print_success("Satisfy conditions")
                    print_success(f"{V_learn} is a Lyapunov function with Epsilon: {self.beta}")
                    verified_flag = True
                    name = config["dataset"]["name"]
                    file_path = f"./models/{name}_model_v.pth"
                    torch.save(self.model_v.state_dict(), file_path)
                    wandb.save(file_path)
                    break
        stop = timeit.default_timer()
        print_info(f"Total Verification Time: {stop - start}")
        return verified_flag

    def trainBarrierCertificate(self):
        hidden_neurons_b = config["model_b"]["hidden_neurons"]
        hidden_layers_b = config["model_b"]["layers"]
        hidden_b = [hidden_neurons_b] * hidden_layers_b
        self.model_b = NNModels.BarrierNet(
            n_input=self.dim_in,
            hidden_b=hidden_b).to(self.device)      
        max_iters = config["model_b"]["max_iters"]
        optimizer_v = torch.optim.Adam(self.model_v.parameters(), lr = config["model_v"]["learning_rate"])
        scheduler_v = torch.optim.lr_scheduler.LinearLR(optimizer_v, start_factor=1.0, end_factor=0.001, total_iters=max_iters)
        optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr = config["model_b"]["learning_rate"])
        scheduler_b = torch.optim.lr_scheduler.LinearLR(optimizer_b, start_factor=1.0, end_factor=0.001, total_iters=max_iters)
        self.model_v.train()
        self.model_b.train()
        # Starting with Sampling Based Verification
        start = timeit.default_timer()
        for i in range(max_iters):
            lyapunov_risk = Loss_Functions.loss_function_v(self.model_v, self.x_domain, torch.tensor(self.X_train,dtype=torch.float32).to(self.device), torch.tensor(self.y_train,dtype=torch.float32).to(self.device), i)
            barrier_risk = Loss_Functions.loss_function_b(self.model_b, self.model_v, self.x_init, self.x_unsafe, self.x_domain, torch.tensor(self.X_train,dtype=torch.float32).to(self.device), i)
            total_loss = lyapunov_risk + barrier_risk
            optimizer_v.zero_grad()
            optimizer_b.zero_grad()
            total_loss.backward()
            optimizer_v.step()
            optimizer_b.step()
            scheduler_v.step()
            scheduler_b.step()
            if i%100 == 0:
                self.x_domain, flag_v = Loss_Functions.lyapunovVerify(self.model_v, self.x_domain, i)    
                self.x_init, self.x_unsafe, self.x_domain, flag_b = Loss_Functions.barrierVerify(self.model_v, self.model_b, self.x_domain, self.x_unsafe, self.x_init, self.initial_set_center)    
                if flag_v and flag_b:
                    print_info("Completed with Sampling Based Training. Proceeding with SMT Verification")
                    break
        stop_ = timeit.default_timer()
        print_info(f"Sampling Verification Time: {stop_ - start}")
        # SMT Verification
        optimizer_v = torch.optim.Adam(self.model_v.parameters(), lr = config["model_v"]["learning_rate"])
        scheduler_v = torch.optim.lr_scheduler.LinearLR(optimizer_v, start_factor=1.0, end_factor=0.001, total_iters=max_iters)
        optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr = config["model_b"]["learning_rate"])
        scheduler_b = torch.optim.lr_scheduler.LinearLR(optimizer_b, start_factor=1.0, end_factor=0.001, total_iters=max_iters)
        self.model_v.train()
        self.model_b.train()
        verified_flag = False
        for i in range(max_iters):
            lyapunov_risk = Loss_Functions.loss_function_v(self.model_v, self.x_domain, torch.tensor(self.X_train,dtype=torch.float32).to(self.device), torch.tensor(self.y_train,dtype=torch.float32).to(self.device), i)
            barrier_risk = Loss_Functions.loss_function_b(self.model_b, self.model_v, self.x_init, self.x_unsafe, self.x_domain, torch.tensor(self.X_train,dtype=torch.float32).to(self.device), i)
            total_loss = lyapunov_risk + barrier_risk
            optimizer_v.zero_grad()
            optimizer_b.zero_grad()
            total_loss.backward()
            optimizer_v.step()
            optimizer_b.step()
            scheduler_v.step()
            scheduler_b.step()
            if i%200 == 0:
                # Finding the values for the new model function
                z = self.vars_
                for idx, layer in enumerate(self.model_v.layers_f[:-1]):
                    w = layer.weight.data.cpu().numpy()
                    b = layer.bias.data.cpu().numpy()
                    zhat = w @ z + b
                    z = smt_verification.hyper_tan_dr(zhat)
                last_layer = self.model_v.layers_f[-1].weight.data.cpu().numpy()
                z = last_layer @ z
                z += self.model_v.layers_f[-1].bias.data.cpu().numpy()
                f_learn = z
                # save the weights and biases
                z = self.vars_
                jacobian = np.eye(self.model_v.input_size, self.model_v.input_size)
                for idx, layer in enumerate(self.model_v.layers_v[:]):
                    w = layer.weight.data.cpu().numpy()
                    b = layer.bias.data.cpu().numpy()
                    zhat = w @ z + b
                    z = smt_verification.hyper_tan_dr(zhat)
                    # Vdot
                    jacobian = w @ jacobian
                    jacobian = np.diagflat(smt_verification.hyper_tan_der_dr(zhat)) @ jacobian
                V_learn = z[0]
                gradV = np.multiply(jacobian, np.broadcast_to(1, jacobian.shape))
                V_learn_dot = (gradV @ f_learn)[0]
                # save the weights and biases
                z = self.vars_
                jacobian = np.eye(self.model_b.input_size, self.model_b.input_size)
                for idx, layer in enumerate(self.model_b.layers_b[:]):
                    w = layer.weight.data.cpu().numpy()
                    b = layer.bias.data.cpu().numpy()
                    zhat = w @ z + b
                    z = smt_verification.hyper_tan_dr(zhat)
                    # Vdot
                    jacobian = w @ jacobian
                    jacobian = np.diagflat(smt_verification.hyper_tan_der_dr(zhat)) @ jacobian
                B_learn = z[0]
                gradB = np.multiply(jacobian, np.broadcast_to(1, jacobian.shape))
                B_learn_dot = (gradB @ f_learn)[0]
                print('===========Verifying==========')
                start_ = timeit.default_timer()
                result = smt_verification.CheckLyapunov(self.vars_, f_learn, V_learn, V_learn_dot, self.ball_lb, self.ball_ub, self.smt_config, self.beta) # SMT solver
                if result:
                    x_domain = self.x_domain.to('cpu')
                    x_domain = smt_verification.AddCounterexamples(x_domain, result, 10)
                    self.x_domain = x_domain.float().to(self.device)
                else:
                    print_success("Satisfy conditions")
                    print_success(f"{V_learn} is a Lyapunov function with Epsilon: {self.beta}")
                init_ball = Expression(0)
                unsafe_ball = Expression(0)
                #init_ball = logical_and(self.vars_[0] >= INIT[0][0], self.vars_[0] <= INIT[0][1], self.vars_[1] >= INIT[1][0], self.vars_[1] <= INIT[1][1])
                initial_set_radius = config["init"]["radius"]
                init_ball = logical_and((self.vars_[0] - self.initial_set_center[0])**2 + (self.vars_[1] - self.initial_set_center[1])**2 <= initial_set_radius**2)
                #unsafe_ball = logical_and(self.vars_[0] >= UNSAFE[0][0], self.vars_[0] <= UNSAFE[0][1], self.vars_[1] >= UNSAFE[1][0], self.vars_[1] <= UNSAFE[1][1])
                unsafe_set_center = config["unsafe"]["centre"]
                unsafe_set_radius = config["unsafe"]["radius"]
                unsafe_ball = logical_and((self.vars_[0] - unsafe_set_center[0])**2 + (self.vars_[1] - unsafe_set_center[1])**2 <= unsafe_set_radius**2)
                # Constraint: x ∈ Ball → (B(c, xin) < 0 ∧ B(c, xun) >= 0)
                condition = logical_imply(init_ball, B_learn < 0)
                result = CheckSatisfiability(logical_not(condition),self.smt_config)
                if(result):
                    print_warning("Not a Barrier Function. Found counterexamples in Initial Domain: ")
                    print_warning(result)
                    x_init = self.x_init.to('cpu')
                    x_init = smt_verification.AddCounterexamples(x_init, result, 50)
                    self.x_init = x_init.float().to(self.device)
                else:
                    condition = logical_imply(unsafe_ball, B_learn >= 0)
                    result = CheckSatisfiability(logical_not(condition),self.smt_config)
                    if(result):
                        print_warning("Not a Barrier Function. Found counterexamples in Unsafe Domain: ")
                        print_warning(result)
                        x_unsafe = self.x_unsafe.to('cpu')
                        x_unsafe = smt_verification.AddCounterexamples(x_unsafe, result, 50)
                        self.x_unsafe = x_unsafe.to(self.device)
                    else:
                        epsi = 1e-3
                        DOMAIN = config["domain"]["range"]
                        domain_ball = logical_and(self.vars_[0] >= DOMAIN[0][0], self.vars_[0] <= DOMAIN[0][1], self.vars_[1] >= DOMAIN[1][0], self.vars_[1] <= DOMAIN[1][1])
                        condition = logical_imply(logical_and(B_learn <= epsi, B_learn >= -epsi, domain_ball), B_learn_dot <= 0)
                        result = CheckSatisfiability(logical_not(condition),self.smt_config)
                        if(result):
                            print_warning("Not a Barrier Function. Found counterexamples in Boundary Domain: ")
                            print_warning(result)
                            x_domain = self.x_domain.to('cpu')
                            x_domain = smt_verification.AddCounterexamples(x_domain, result, 50)
                            self.x_domain = x_domain.to(self.device)
                        else:
                            verified_flag = True
                            print_success("Satisfy conditions")
                            print_success(f"{B_learn} is a Barrier function with Epsilon: {self.beta}")
                            name = config["dataset"]["name"]
                            file_path = f"./models/{name}_model_v.pth"
                            torch.save(self.model_v.state_dict(), file_path)
                            wandb.save(file_path)
                            file_path = f"./models/{name}_model_b.pth"
                            torch.save(self.model_b.state_dict(), file_path)
                            wandb.save(file_path)
                            break
        stop = timeit.default_timer()
        print_info(f"Total Verification Time: {stop - start}")
        return verified_flag
    
if __name__ == "__main__":
    mp = MotionPlanner()
    mp.generateData()
    print_info("DYNAMICAL SYSTEM TRAINING")
    mp.trainInitialDynamics()
    Plotter.initialDSPlot(mp.model_f, mp.X_train, mp.initial_set_center)
    print_info("LYAPUNOV FUNCTION TRAINING")
    lyapunov_verified = mp.trainLyapunovFunction()
    check_barrier = config["Barrier"]
    if check_barrier:
        _ = mp.trainBarrierCertificate()
    Plotter.lyapunovBarrierPlot(mp.model_v, mp.X_train, mp.initial_set_center, mp.model_b)
    wandb.finish()

    