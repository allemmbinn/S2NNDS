from common_header import *
import NNModels
import data
import Loss_Functions
import Plotter
import smt_verification
from dreal import *

@dataclass
class ConfigFile:
    lasa_name : str = "Worm"
    dataset_type : str = "LASA"
        
class MotionPlanner:
    def __init__(self, args):
        self.args = args
        # Load the configuration file
        file_path = "./config_files/" + self.args.lasa_name + "_config.json"
        with open(file_path) as file:
            self.config = json.load(file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # wandb_name = self.config["plotting"]["name"]
        # wandb.init(project=wandb_name, config=self.config)
        
    def calculate_limits(self, data, x_limit_fact=1.0, y_limit_fact=1.0):
        x_min = min(data[0, :])
        x_max = max(data[0, :])
        x_range = x_max - x_min
        
        y_min = min(data[1, :])
        y_max = max(data[1, :])
        y_range = y_max - y_min
        
        xy_range = max(x_range, y_range)
        
        x_lowerlim = x_min - xy_range * x_limit_fact
        x_upperlim = x_max + xy_range * x_limit_fact
        y_lowerlim = y_min - xy_range * y_limit_fact
        y_upperlim = y_max + xy_range * y_limit_fact
        
        limits = [[x_lowerlim, x_upperlim], [y_lowerlim, y_upperlim]]
        return limits

    def generateData(self):
        if self.args.dataset_type == 'LASA':
            if self.args.lasa_name == "Angle":
                dataset = lasa.DataSet.Angle
            elif self.args.lasa_name == "Worm":
                dataset = lasa.DataSet.Worm
            elif self.args.lasa_name == "CShape":
                dataset = lasa.DataSet.CShape
            elif self.args.lasa_name == "DoubleBendedLine":
                dataset = lasa.DataSet.DoubleBendedLine
            elif self.args.lasa_name == "GShape":
                dataset = lasa.DataSet.GShape
            elif self.args.lasa_name == "WShape":
                dataset = lasa.DataSet.WShape
            elif self.args.lasa_name == "SShape":
                dataset = lasa.DataSet.SShape
            elif self.args.lasa_name == "Leaf_2":
                dataset = lasa.DataSet.Leaf_2
            elif self.args.lasa_name == "Sine":
                dataset = lasa.DataSet.Sine
            else:
                print_error("Invalid LASA Dataset has been choosen")
                raise NotImplementedError
            self.dt = dataset.dt
            demos = dataset.demos
            # Divide the data into training and testing
            total_demos = len(demos)
            self.dim_in = demos[0].pos.shape[0]
            self.vars_  = [Variable(f"x{i}") for i in range(self.dim_in)]
            train_size = int(5/7 * total_demos) # 5/7 datasets are used for training
            train_indices = random.sample(range(total_demos), train_size)
            test_indices = list(set(range(total_demos)) - set(train_indices))
            self.X_train = np.concatenate([demos[i].pos for i in train_indices], axis=1).T
            self.X_test = np.concatenate([demos[i].pos for i in test_indices], axis=1).T
            self.y_train = np.concatenate([demos[i].vel for i in train_indices], axis=1).T
            self.y_test = np.concatenate([demos[i].vel for i in test_indices], axis=1).T 
            # Convert to Pytorch Tensors
            self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
            self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
            self.y_test = torch.tensor(self.y_test, dtype=torch.float32)  
            assert self.X_train.shape[0] == self.y_train.shape[0], "Mismatch in number of samples between X_train and y_train"
            assert self.X_test.shape[0] == self.y_test.shape[0], "Mismatch in number of samples between X_test and y_test" 
            # Dataloader for X_TRAIN
            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
            test_dataset = torch.utils.data.TensorDataset(self.X_test, self.y_test)
            batch_size = self.config["model_f"]["batch_size"]
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            print_error("Non-LASA Dataset has been choosen")
        # Normalise the Trajectories to [-1, 1]
        pos_scaling = max(np.max(np.linalg.norm(self.X_train, axis=1)), np.max(np.linalg.norm(self.X_test, axis=1)))
        vel_scaling = max(np.max(np.linalg.norm(self.y_train, axis=1)), np.max(np.linalg.norm(self.y_test, axis=1)))
        self.X_train /= pos_scaling
        self.X_test /= pos_scaling
        self.y_train /= vel_scaling
        self.y_test /= vel_scaling
        # mean_pos = np.mean(total_data, axis=0)
        # std_dev_pos = np.std(total_data, axis=0)
        # mean_vel = np.mean(self.y_train, axis=0)
        # std_dev_vel = np.std(self.y_train, axis=0)
        # self.X_train = (self.X_train - mean_pos) / std_dev_pos
        # self.X_test = (self.X_test - mean_pos) / std_dev_pos
        # self.y_train = (self.y_train - mean_vel) / std_dev_vel
        # self.y_test = (self.y_test - mean_vel) / std_dev_vel
        # Check Limits
        self.limits = self.calculate_limits(self.X_train.numpy().T)
        # Finding the mean_point
        mean_point = np.mean([demos[i].pos[:,0] for i in range(total_demos)], axis=0)/pos_scaling
        # Get initial set center
        self.N_domain = self.config["domain"]["N"]
        # self.x_domain = data.generateRectangularData(self.N_domain, self.limits).to(self.device)
        self.x_domain = data.generateGridData(self.N_domain, self.limits)
        if self.config["Barrier"]:
            # Get Init Data Points
            self.initial_set_center = mean_point
            self.N_init = self.config["init"]["N"]
            init_range = [[ self.initial_set_center[i] - self.config["init"]["radius"], 
                            self.initial_set_center[i] + self.config["init"]["radius"]] for i in range(self.dim_in)]
            self.x_init = data.generateGridData(self.N_init, init_range)
            # self.x_init = data.generateCircularData(self.N_init, self.config["init"]["radius"], self.initial_set_center).to(self.device)            
            # Get Unsafe Data Points
            self.N_unsafe = self.config["unsafe"]["N"]
            if self.config["unsafe"]["shape"] == "Circle":
                self.x_unsafe = data.generateCircularData(self.N_unsafe, self.config["unsafe"]["radius"], self.config["unsafe"]["centre"]).to(self.device)
                # unsafe_range = [[ self.config["unsafe"]["centre"][i] - self.config["unsafe"]["radius"], 
                #                 self.config["unsafe"]["centre"][i] + self.config["unsafe"]["radius"]] for i in range(self.dim_in)]
                # self.x_unsafe = data.generateGridData(self.N_unsafe, unsafe_range)
            elif self.config["unsafe"]["shape"] == "Rectangle":
                unsafe_range = [[ self.config["unsafe"]["centre"][i] - self.config["unsafe"]["radius"], 
                self.config["unsafe"]["centre"][i] + self.config["unsafe"]["radius"]] for i in range(self.dim_in)]
                self.x_unsafe = data.generateGridData(self.N_unsafe, unsafe_range)

            else:
                print_error("Non-Circular Unsafe Set has been choosen") #TODO: Add code for rectangular data
                raise NotImplementedError
        else:
            # Get Init Data Points
            self.initial_set_center = mean_point
            self.N_init = self.config["init"]["N"]
            self.x_init = data.generateCircularData(self.N_init, self.config["init"]["radius"], self.initial_set_center)
        # DIMENSION
        self.dim_in = self.X_train.shape[1]

    def trainInitialDynamics(self):
        self.hidden_neurons_f = self.config["model_f"]["hidden_neurons"]
        self.hidden_layers_f = self.config["model_f"]["layers"]
        sigmoid_f = NNModels.assignActivationFunction(self.config['model_f']['activation_function'])
        self.hidden_f = [self.hidden_neurons_f] * self.hidden_layers_f
        self.model_f = NNModels.DyanmicsNet(self.dim_in, self.hidden_f, sigmoid_f).to(self.device)
        best_mse = np.inf   # init to infinity
        best_weights = None
        history = []
        loss_fn = nn.MSELoss()  # mean square error #TODO: Add Lyapunov, barrier, regularization loss
        optimizer_f = optimizer_f = torch.optim.Adam( self.model_f.parameters(), lr=self.config["model_f"]["learning_rate"],betas=(0.9, 0.999))
        for epoch in range(self.config["model_f"]["epochs_warm"]):
            total_loss = 0
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                self.model_f.train()
                # Calculate the loss
                y_pred = self.model_f(X_batch.float().to(self.device))
                loss = loss_fn(y_pred, y_batch.float().to(self.device))
                total_loss += loss.item()
                # backward pass
                optimizer_f.zero_grad()
                loss.backward()
                optimizer_f.step()
            # Log the Training Loss
            # wandb.log({"DS_training_loss": total_loss})
            #evaluate accuracy at end of each epoch           
            self.model_f.eval()
            for batch_idx, (X_batch, y_batch) in enumerate(self.test_loader):
                y_pred = self.model_f(X_batch.float().to(self.device))
                mse = loss_fn(y_pred, y_batch.float().to(self.device))
                mse = float(mse)
                history.append(mse)
                if loss < best_mse:
                    best_mse = mse
                    best_weights = copy.deepcopy(self.model_f.state_dict())
                with torch.no_grad():
                    torch.cuda.empty_cache()
        # restore model and return best accuracy
        self.model_f.load_state_dict(best_weights)
        print_info("MSE of Initial Estimate of Dynamical System: %.4f" % best_mse)
    
    def trainLyapunovFunction(self):
        # Defining the Lyapunov Function NN
        hidden_neurons_v = self.config["model_v"]["hidden_neurons"]
        hidden_layers_v = self.config["model_v"]["layers"]
        hidden_v = [hidden_neurons_v] * hidden_layers_v
        self.model_v = NNModels.LyapunovNet(
            n_input=self.dim_in,
            hidden_v=hidden_v,
            hidden_f=self.hidden_f,
            model_f=self.model_f,
            sigmoid_f=NNModels.assignActivationFunction(self.config['model_f']['activation_function']),
            sigmoid_v=NNModels.assignActivationFunction(self.config['model_v']['activation_function'])
        ).to(self.device)      
        max_iters = self.config["model_v"]["max_iters"]
        optimizer_v = torch.optim.Adam(self.model_v.parameters(), lr = self.config["model_v"]["learning_rate"])
        # SMT Verification
        self.ball_ub = max([x_range[1] for x_range in self.limits])
        # Provide the start factor and end factor
        try:
            start_factor = self.config["model_v"]["scheduler"]["start_factor"]
            end_factor   = self.config["model_v"]["scheduler"]["end_factor"]
        except:
            start_factor = 1.0
            end_factor = 0.0001
        scheduler_v = torch.optim.lr_scheduler.LinearLR(optimizer_v, start_factor=start_factor, end_factor=end_factor, total_iters=max_iters)
        # Setting up Training Dataloader
        domain_dataset = torch.utils.data.TensorDataset(self.x_domain)
        train_dataset  = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        batch_size = self.config["model_v"]["batch_size_training"]
        domain_loader = torch.utils.data.DataLoader(domain_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        """self.model_v.train() #I don't think we need dropout and batch normalizing- in fact, overfitting is preferred.
        Moreover we have already normalized the input data"""
        # Starting with Sampling Based Verification
        start = timeit.default_timer()
        for epoch in range(max_iters):
            for (x_domain_batch,), (X_train_batch, y_train_batch) in zip(domain_loader, train_loader):
                X_train_batch = X_train_batch.to(self.device)
                y_train_batch = y_train_batch.to(self.device)
                x_domain_batch = x_domain_batch.to(self.device)
                lyapunov_risk = Loss_Functions.loss_function_v(self.model_v, x_domain_batch, X_train_batch, y_train_batch, epoch, self.config)
                optimizer_v.zero_grad()
                lyapunov_risk.backward()
                optimizer_v.step()
                scheduler_v.step()
            if epoch % 500 == 0:
                print_info(f"Epoch: {epoch}, Loss: {lyapunov_risk.item()}")
                self.x_domain, flag = Loss_Functions.lyapunovVerify(self.model_v, self.x_domain, epoch, self.config, DOMAIN=self.limits)    
                if flag:
                    print_info("Completed with Sampling Based Training. Proceeding with SMT Verification")
                    Plotter.lyapunovBarrierPlot(mp.model_v, mp.X_train, mp.initial_set_center, self.config)
                    break
                domain_dataset = torch.utils.data.TensorDataset(self.x_domain)
                domain_loader = torch.utils.data.DataLoader(domain_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            # if epoch % 5000 == 0:
            #     Plotter.lyapunovBarrierPlot(mp.model_v, mp.X_train, mp.initial_set_center, self.config)
        stop_ = timeit.default_timer()
        print_info(f"Sampling Verification Time: {stop_ - start}")
        # SMT Verification
        optimizer_v = torch.optim.Adam(self.model_v.parameters(), lr = self.config["model_v"]["learning_rate"])
        scheduler_v = torch.optim.lr_scheduler.LinearLR(optimizer_v, start_factor=1.0, end_factor=0.01, total_iters=max_iters)
        self.model_v.train()
        verified_flag = False
        for epoch in range(max_iters):
            for (x_domain_batch,), (X_train_batch, y_train_batch) in zip(domain_loader, train_loader):
                lyapunov_risk = Loss_Functions.loss_function_v(self.model_v, x_domain_batch.to(self.device), X_train_batch.to(self.device), y_train_batch.to(self.device), epoch, self.config)
                optimizer_v.zero_grad()
                lyapunov_risk.backward()
                optimizer_v.step()
                scheduler_v.step()
            if epoch%200 == 0:
                # Finding the values for the new model function
                z = self.vars_  # Initial input
                # Dynamics Function (Fout)
                for idx, layer in enumerate(self.model_v.layers_f[:-1]):
                    w = layer.weight.data.cpu().numpy()
                    b = layer.bias.data.cpu().numpy()
                    zhat = w @ z + b
                    z = smt_verification.hyper_tan_dr(zhat)
                last_layer = self.model_v.layers_f[-1].weight.data.cpu().numpy()
                z = last_layer @ z
                z += self.model_v.layers_f[-1].bias.data.cpu().numpy()
                f_learn = z

                # Lyapunov Function (Vout)
                z = self.vars_
                jacobian = np.eye(self.model_v.input_size, self.model_v.input_size)
                for idx, layer in enumerate(self.model_v.layers_v[:-1]):
                    w = layer.weight.data.cpu().numpy()
                    b = layer.bias.data.cpu().numpy()
                    zhat = w @ z + b
                    z = smt_verification.hyper_tan_dr(zhat)
                    # Vdot computation
                    jacobian = w @ jacobian
                    jacobian = np.diagflat(smt_verification.hyper_tan_der_dr(zhat)) @ jacobian
                # Last layer for Lyapunov Function
                w_last = self.model_v.layers_v[-1].weight.data.cpu().numpy()
                b_last = self.model_v.layers_v[-1].bias.data.cpu().numpy()
                V_learn = (w_last @ z + b_last)[0]
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
                    name = self.args.lasa_name
                    folder_path = os.path.join(os.curdir, "models")
                    if os.path.isdir(folder_path):
                        torch.save(self.model_v.state_dict(), os.path.join(folder_path, f"{name}_model_v.pth"))
                    else:
                        try:
                            os.makedirs(folder_path)
                            torch.save(self.model_v.state_dict(), os.path.join(folder_path, f"{name}_model_v.pth"))
                        except OSError as e:
                            print_error(f"Error creating folder '{folder_path}': {e}")
                    break
                domain_dataset = torch.utils.data.TensorDataset(self.x_domain)
                train_dataset  = torch.utils.data.TensorDataset(self.X_train, self.y_train)
                domain_loader = torch.utils.data.DataLoader(domain_dataset, batch_size=batch_size, shuffle=True)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        stop = timeit.default_timer()
        print_info(f"Total Verification Time: {stop - start}")
        return verified_flag

    def trainBarrierCertificate(self):
        # Defining the Lyapunov Function NN
        hidden_neurons_v = self.config["model_v"]["hidden_neurons"]
        hidden_layers_v = self.config["model_v"]["layers"]
        hidden_v = [hidden_neurons_v] * hidden_layers_v
        self.model_v = NNModels.LyapunovNet(
            n_input=self.dim_in,
            hidden_v=hidden_v,
            hidden_f=self.hidden_f,
            model_f=self.model_f,
            sigmoid_f=NNModels.assignActivationFunction(self.config['model_f']['activation_function']),
            sigmoid_v=NNModels.assignActivationFunction(self.config['model_v']['activation_function'])
        ).to(self.device)      
        max_iters = self.config["model_b"]["max_iters"]
        optimizer_v = torch.optim.Adam(self.model_v.parameters(), lr = self.config["model_v"]["learning_rate"])
        # Provide the start factor and end factor
        try:
            start_factor = self.config["model_v"]["scheduler"]["start_factor"]
            end_factor   = self.config["model_v"]["scheduler"]["end_factor"]
        except:
            start_factor = 1.0
            end_factor = 1e-2
        scheduler_v = torch.optim.lr_scheduler.LinearLR(optimizer_v, start_factor=start_factor, end_factor=end_factor, total_iters=max_iters)
        # For the Barrier Function
        hidden_neurons_b = self.config["model_b"]["hidden_neurons"]
        hidden_layers_b  = self.config["model_b"]["layers"]
        hidden_b = [hidden_neurons_b] * hidden_layers_b
        self.model_b = NNModels.BarrierNet(
            n_input=self.dim_in,
            hidden_b=hidden_b,
            sigmoid_b=NNModels.assignActivationFunction(self.config['model_b']['activation_function'])).to(self.device)      
        max_iters = self.config["model_b"]["max_iters"]
        optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr = self.config["model_b"]["learning_rate"], )
        # Provide the start factor and end factor
        try:
            start_factor = self.config["model_b"]["scheduler"]["start_factor"]
            end_factor   = self.config["model_b"]["scheduler"]["end_factor"]
        except:
            start_factor = 1.0
            end_factor = 1e-2
        scheduler_b = torch.optim.lr_scheduler.LinearLR(optimizer_b, start_factor=start_factor, end_factor=end_factor, total_iters=max_iters)
        #self.model_v.train()
        #self.model_b.train()
        # Starting with Sampling Based Verification
        # Setting up Training Dataloader
        domain_dataset = torch.utils.data.TensorDataset(self.x_domain)
        init_dataset  = torch.utils.data.TensorDataset(self.x_init)
        unsafe_dataset  = torch.utils.data.TensorDataset(self.x_unsafe)
        train_dataset  = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        batch_size = self.config["model_v"]["batch_size_training"]
        domain_loader = torch.utils.data.DataLoader(domain_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        init_loader = torch.utils.data.DataLoader(init_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        unsafe_loader = torch.utils.data.DataLoader(unsafe_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        start = timeit.default_timer()
        for epoch in range(max_iters):
            if epoch % 20 == 19:
                for (x_domain_batch,), (X_train_batch, y_train_batch),(x_init_batch,),(x_unsafe_batch,) in zip(domain_loader, train_loader, init_loader, unsafe_loader):
                    x_domain_batch = x_domain_batch.to(self.device)
                    X_train_batch = X_train_batch.to(self.device)
                    y_train_batch = y_train_batch.to(self.device)
                    x_init_batch = x_init_batch.to(self.device)
                    x_unsafe_batch = x_unsafe_batch.to(self.device)
                    # lyapunov_risk = Loss_Functions.loss_function_v(self.model_v, x_domain_batch, X_train_batch, y_train_batch, i, self.config)
                    barrier_risk = Loss_Functions.loss_function_b(self.model_b, self.model_v, x_init_batch, x_unsafe_batch, x_domain_batch,  X_train_batch, epoch, self.config)
                    total_loss = barrier_risk
                    #total_loss = lyapunov_risk + barrier_risk 
                    optimizer_v.zero_grad()
                    optimizer_b.zero_grad()
                    total_loss.backward()
                    optimizer_v.step()
                    optimizer_b.step()
                    scheduler_v.step()
                    scheduler_b.step()
            else:
                for (x_domain_batch,), (X_train_batch, y_train_batch) in zip(domain_loader, train_loader):
                    x_domain_batch = x_domain_batch.to(self.device)
                    X_train_batch = X_train_batch.to(self.device)
                    y_train_batch = y_train_batch.to(self.device)
                    lyapunov_risk = Loss_Functions.loss_function_v(self.model_v, x_domain_batch, X_train_batch, y_train_batch, epoch, self.config)
                    total_loss = lyapunov_risk
                    optimizer_v.zero_grad()
                    total_loss.backward()
                    optimizer_v.step()
                    scheduler_v.step()
            if epoch % 500 == 0:
                self.x_domain, flag_v = Loss_Functions.lyapunovVerify(self.model_v, self.x_domain, epoch, self.config, DOMAIN=self.limits)    
                self.x_init, self.x_unsafe, self.x_domain, flag_b = Loss_Functions.barrierVerify(self.model_v, self.model_b, self.x_domain, self.x_unsafe, self.x_init, self.initial_set_center, self.config, DOMAIN=self.limits)    
                if flag_v and flag_b:
                    print_info("Completed with Sampling Based Training. Proceeding with SMT Verification")
                    break
                domain_dataset = torch.utils.data.TensorDataset(self.x_domain)
                init_dataset = torch.utils.data.TensorDataset(self.x_init)
                unsafe_dataset = torch.utils.data.TensorDataset(self.x_unsafe)
                domain_loader = torch.utils.data.DataLoader(domain_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
                init_loader = torch.utils.data.DataLoader(init_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
                unsafe_loader = torch.utils.data.DataLoader(unsafe_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        # for i in range(max_iters):
        #     for (x_domain_batch,), (X_train_batch, y_train_batch),(x_init_batch,),(x_unsafe_batch,) in zip(domain_loader, train_loader, init_loader, unsafe_loader):
        #         x_domain_batch = x_domain_batch.to(self.device)
        #         X_train_batch = X_train_batch.to(self.device)
        #         y_train_batch = y_train_batch.to(self.device)
        #         x_init_batch = x_init_batch.to(self.device)
        #         x_unsafe_batch = x_unsafe_batch.to(self.device)
        #         lyapunov_risk = Loss_Functions.loss_function_v(self.model_v, x_domain_batch, X_train_batch, y_train_batch, i, self.config)
        #         barrier_risk = Loss_Functions.loss_function_b(self.model_b, self.model_v, x_init_batch, x_unsafe_batch, x_domain_batch,  X_train_batch, i, self.config)
        #         total_loss = lyapunov_risk + barrier_risk 
        #         optimizer_v.zero_grad()
        #         optimizer_b.zero_grad()
        #         total_loss.backward()
        #         optimizer_v.step()
        #         optimizer_b.step()
        #         scheduler_v.step()
        #         scheduler_b.step()
        #     if i%500 == 0:
        #         self.x_domain, flag_v = Loss_Functions.lyapunovVerify(self.model_v, self.x_domain, i, self.config, DOMAIN=self.limits)    
        #         self.x_init, self.x_unsafe, self.x_domain, flag_b = Loss_Functions.barrierVerify(self.model_v, self.model_b, self.x_domain, self.x_unsafe, self.x_init, self.initial_set_center, self.config, DOMAIN=self.limits)    
        #         if flag_v and flag_b:
        #             print_info("Completed with Sampling Based Training. Proceeding with SMT Verification")
        #             break
        #         domain_dataset = torch.utils.data.TensorDataset(self.x_domain)
        #         init_dataset = torch.utils.data.TensorDataset(self.x_init)
        #         unsafe_dataset = torch.utils.data.TensorDataset(self.x_unsafe)
        #         domain_loader = torch.utils.data.DataLoader(domain_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        #         init_loader = torch.utils.data.DataLoader(init_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        #         unsafe_loader = torch.utils.data.DataLoader(unsafe_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            # if i%5000 == 0:
            #     Plotter.lyapunovBarrierPlot(mp.model_v, mp.X_train, mp.initial_set_center, self.config, mp.model_b)
        stop_ = timeit.default_timer()
        print_info(f"Sampling Verification Time: {stop_ - start}")
        # SMT Verification
        optimizer_v = torch.optim.Adam(self.model_v.parameters(), lr = self.config["model_v"]["learning_rate"])
        scheduler_v = torch.optim.lr_scheduler.LinearLR(optimizer_v, start_factor=1.0, end_factor=0.001, total_iters=max_iters)
        optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr = self.config["model_b"]["learning_rate"])
        scheduler_b = torch.optim.lr_scheduler.LinearLR(optimizer_b, start_factor=1.0, end_factor=0.001, total_iters=max_iters)
        #self.model_v.train()
        #self.model_b.train()
        verified_flag = False
        for i in range(max_iters):
            lyapunov_risk = Loss_Functions.loss_function_v(self.model_v, self.x_domain, torch.tensor(self.X_train,dtype=torch.float32).to(self.device), torch.tensor(self.y_train,dtype=torch.float32).to(self.device), i, self.config)
            barrier_risk = Loss_Functions.loss_function_b(self.model_b, self.model_v, self.x_init, self.x_unsafe, self.x_domain, torch.tensor(self.X_train,dtype=torch.float32).to(self.device), i, self.config)
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
                z = self.vars_  # Initial input
                # Dynamics Function (Fout)
                for idx, layer in enumerate(self.model_v.layers_f[:-1]):
                    w = layer.weight.data.cpu().numpy()
                    b = layer.bias.data.cpu().numpy()
                    zhat = w @ z + b
                    z = smt_verification.hyper_tan_dr(zhat)
                last_layer = self.model_v.layers_f[-1].weight.data.cpu().numpy()
                z = last_layer @ z
                z += self.model_v.layers_f[-1].bias.data.cpu().numpy()
                f_learn = z
                # Lyapunov Function (Vout)
                z = self.vars_
                jacobian = np.eye(self.model_v.input_size, self.model_v.input_size)
                for idx, layer in enumerate(self.model_v.layers_v[:-1]):
                    w = layer.weight.data.cpu().numpy()
                    b = layer.bias.data.cpu().numpy()
                    zhat = w @ z + b
                    z = smt_verification.hyper_tan_dr(zhat)
                    # Vdot computation
                    jacobian = w @ jacobian
                    jacobian = np.diagflat(smt_verification.hyper_tan_der_dr(zhat)) @ jacobian
                # Last layer for Lyapunov Function
                w_last = self.model_v.layers_v[-1].weight.data.cpu().numpy()
                b_last = self.model_v.layers_v[-1].bias.data.cpu().numpy()
                V_learn = (w_last @ z + b_last)[0]
                gradV = np.multiply(jacobian, np.broadcast_to(1, jacobian.shape))
                V_learn_dot = (gradV @ f_learn)[0]
                # save the weights and biases
                z = self.vars_
                jacobian = np.eye(self.model_b.input_size, self.model_b.input_size)
                for idx, layer in enumerate(self.model_b.layers_b[:-1]):
                    w = layer.weight.data.cpu().numpy()
                    b = layer.bias.data.cpu().numpy()
                    zhat = w @ z + b
                    z = smt_verification.hyper_tan_dr(zhat)
                    # Vdot
                    jacobian = w @ jacobian
                    jacobian = np.diagflat(smt_verification.hyper_tan_der_dr(zhat)) @ jacobian
                # For the final layer
                w_last = self.model_b.layers_b[-1].weight.data.cpu().numpy()
                b_last = self.model_b.layers_b[-1].bias.data.cpu().numpy()
                B_learn = (w_last @ z + b_last)[0]
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
                            name = self.args.lasa_name
                            folder_path = os.path.join(os.curdir, "models")
                            if os.path.isdir(folder_path):
                                torch.save(self.model_v.state_dict(), os.path.join(folder_path, f"{name}_model_v.pth"))
                                torch.save(self.model_b.state_dict(), os.path.join(folder_path, f"{name}_model_b.pth"))
                            else:
                                try:
                                    os.makedirs(folder_path)
                                    torch.save(self.model_v.state_dict(), os.path.join(folder_path, f"{name}_model_v.pth"))
                                    torch.save(self.model_b.state_dict(), os.path.join(folder_path, f"{name}_model_b.pth"))
                                except OSError as e:
                                    print_error(f"Error creating folder '{folder_path}': {e}")
                            break
        stop = timeit.default_timer()
        print_info(f"Total Verification Time: {stop - start}")
        return verified_flag
        return None
    
if __name__ == "__main__":
    # Settings Seeds for Reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    args = pyrallis.parse(ConfigFile)
    mp = MotionPlanner(args)
    mp.generateData()
    print_info("DYNAMICAL SYSTEM TRAINING")
    mp.trainInitialDynamics()
    Plotter.initialDSPlot(mp.model_f, mp.X_train, mp.initial_set_center)
    print_info("LYAPUNOV FUNCTION TRAINING")
    lyapunov_verified = mp.trainLyapunovFunction()
    check_barrier = mp.config["Barrier"]
    if check_barrier:
        _ = mp.trainBarrierCertificate()
        Plotter.lyapunovBarrierPlot(mp.model_v, mp.X_train, mp.initial_set_center, mp.config, mp.model_b)
    Plotter.lyapunovBarrierPlot(mp.model_v, mp.X_train, mp.initial_set_center, mp.config)
    # wandb.finish()

    