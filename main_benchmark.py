from common_header import *
import NNModels
import data_new as data
import Loss_Functions_new as Loss_Functions
import opt 
import Plotter
import verification

@dataclass
class ConfigFile:
    lasa_name : str = "Worm"
    dataset_type : str = "LASA"

def filter_args(args):
    known_args = ['--lasa_name', '--dataset_type']
    return [arg for arg in args if any(arg.startswith(known) for known in known_args)]

def save_seed(seed, seed_filepath):
    os.makedirs(os.path.dirname(seed_filepath), exist_ok=True)
    with open(seed_filepath, 'w') as file:
        json.dump({'seed': seed}, file)

def load_seed(seed_filepath):
    with open(seed_filepath, 'r') as file:
        seed_data = json.load(file)
    return seed_data['seed']    

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MotionPlanner:
    def __init__(self, args):
        self.args = args
        # Load the configuration file
        file_path = "./config_files/" + self.args.lasa_name + "_config_benchmark.json"
        print_info(f"Loading configuration file from {file_path}")
        with open(file_path) as file:
            self.config = json.load(file)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        #Initialize state dictionaries
        self.model_v_state_dict = None
        self.model_b_state_dict = None
        self.model_f_state_dict = None
        self.optimizer_v_state_dict = None
        self.optimizer_b_state_dict = None
        self.optimizer_f_state_dict = None
        self.scheduler_v_state_dict = None
        self.scheduler_b_state_dict = None
        self.scheduler_f_state_dict = None
        #initialize the learning rate
        self.lr_f = self.config["model_f"]["learning_rate"]
        self.lr_v = self.config["model_v"]["learning_rate"]
        self.lr_b = self.config["model_b"]["learning_rate"]
        self.counterexamples_added = True #Setting the counterexamples flag to true
        self.g = torch.Generator()
        self.g.manual_seed(0)    
        #verification flag

    def seed_worker(self,worker_id):
        np.random.seed(0)
        random.seed(0)

    def load_model_states(self):
        if self.model_v_state_dict is not None:
            self.model_v.load_state_dict(self.model_v_state_dict)
        if self.model_b_state_dict is not None:
            self.model_b.load_state_dict(self.model_b_state_dict)
        if self.model_f_state_dict is not None:
            self.model_f.load_state_dict(self.model_f_state_dict)
        if self.optimizer_v_state_dict is not None:
            self.optimizer_v.load_state_dict(self.optimizer_v_state_dict)
        if self.optimizer_b_state_dict is not None:
            self.optimizer_b.load_state_dict(self.optimizer_b_state_dict)
        if self.optimizer_f_state_dict is not None:
            self.optimizer_f.load_state_dict(self.optimizer_f_state_dict)
        if self.scheduler_v_state_dict is not None:
            self.scheduler_v.load_state_dict(self.scheduler_v_state_dict)
        if self.scheduler_b_state_dict is not None:
            self.scheduler_b.load_state_dict(self.scheduler_b_state_dict)
        if self.scheduler_f_state_dict is not None:
            self.scheduler_f.load_state_dict(self.scheduler_f_state_dict)

    def save_model(self, model, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model, model_path)
   
    def save_all_models(self):
        base_path = os.path.join('models', self.args.lasa_name)
        os.makedirs(base_path, exist_ok=True)  # Ensure the directory exists
        self.save_model(self.model_f, os.path.join(base_path, 'model_f.pth'))
        self.save_model(self.model_v, os.path.join(base_path, 'model_v.pth'))
        self.save_model(self.model_b, os.path.join(base_path, 'model_b.pth'))

    def export_onnx(self):
        base_path = os.path.join('models_onnx')
        os.makedirs(base_path, exist_ok=True)  # Ensure the directory exists
        self.model_f.eval()
        self.initial_point = self.initial_set_center[0].reshape(1,2).to(dtype=torch.float32)
        torch.onnx.export(
                self.model_f,
                self.initial_point,
                os.path.join(base_path, self.args.lasa_name + ".onnx"),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["position"],     # Name of the input tensor
                output_names=["velocity"],   # Name of the output tensor
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            )
        onnx_model = onnx.load(os.path.join(base_path, self.args.lasa_name + ".onnx"))
        onnx.checker.check_model(onnx_model)

    def generate_demo_data(self): #Trains data for MSE minimization, learning from demos
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
            elif self.args.lasa_name == "Sshape":
                dataset = lasa.DataSet.Sshape
            elif self.args.lasa_name == "Leaf_2":
                dataset = lasa.DataSet.Leaf_2
            elif self.args.lasa_name == "Sine":
                dataset = lasa.DataSet.Sine
            elif self.args.lasa_name == "heee":
                dataset = lasa.DataSet.heee
            elif self.args.lasa_name == "PShape":
                dataset = lasa.DataSet.PShape
            elif self.args.lasa_name == "NShape":
                dataset = lasa.DataSet.NShape
            else:
                print_error("Invalid LASA Dataset has been choosen")
                raise NotImplementedError
            self.dt = dataset.dt
            self.demos = dataset.demos
            # Divide the data into training and testing
            self.total_demos = len(self.demos)
            self.dim_in = self.demos[0].pos.shape[0]
            train_size = int(5/7 * self.total_demos) # 5/7 datasets are used for training
            train_indices = random.sample(range(self.total_demos), train_size)
            test_indices = list(set(range(self.total_demos)) - set(train_indices))
            self.X_train = np.concatenate([self.demos[i].pos for i in train_indices], axis=1).T
            self.X_test = np.concatenate([self.demos[i].pos for i in test_indices], axis=1).T
            self.y_train = np.concatenate([self.demos[i].vel for i in train_indices], axis=1).T
            self.y_test = np.concatenate([self.demos[i].vel for i in test_indices], axis=1).T 
            self.dim_in = self.X_train.shape[1] #Input dimension used later for constructing neural network
            # Convert to Pytorch Tensors
            self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
            self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
            self.y_test = torch.tensor(self.y_test, dtype=torch.float32)  
            assert self.X_train.shape[0] == self.y_train.shape[0], "Mismatch in number of samples between X_train and y_train"
            assert self.X_test.shape[0] == self.y_test.shape[0], "Mismatch in number of samples between X_test and y_test" 
            batch_size = self.config["model_f"]["batch_size"]
        else:
            print_error("Non-LASA Dataset has been choosen")

        # Normalise the Trajectories to [-1, 1] #Use the maximum value to normalize and scale the data.
        self.pos_scaling = torch.linalg.norm(torch.concatenate([abs(self.X_train), abs(self.X_test)]), ord=2, dim=1).max()	
        #torch.max(torch.concatenate([abs(self.X_train), abs(self.X_test)]))
        self.vel_scaling = torch.linalg.norm(torch.concatenate([abs(self.y_train), abs(self.y_test)]), ord=2, dim=1).max()	
        self.X_train /= self.pos_scaling
        self.X_test /= self.pos_scaling
        self.y_train /= self.vel_scaling
        self.y_test /= self.vel_scaling
        self.initial_set_center = (np.mean([self.demos[i].pos[:,0] for i in range(self.total_demos)], axis=0)/self.pos_scaling).reshape(1,2)
        train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        test_dataset = torch.utils.data.TensorDataset(self.X_test, self.y_test)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=self.seed_worker, generator=self.g)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=self.seed_worker, generator=self.g)

    def generate_domain_data(self):
        self.N_domain = self.config["domain"]["N"] #The number of samples we want in the domain region
        self.RANGE = self.config["domain"]["range"] #The number of samples we want in the domain region
        self.unsafe_domain = torch.empty(0,)
        self.init_domain = torch.empty(0,)
        while len(self.unsafe_domain) < 2 or len(self.init_domain) < 2:
            self.domain, _ = data.generateRandomData(self.N_domain, self.RANGE) #the domain is limited to [-1,1] due to normalization
            #Generate data for initial set - According to ABC-DS normalization
            self.init_rad = self.config["init"]["radius"]
            self.init_center = self.initial_set_center
            mask = (torch.linalg.norm(self.domain - self.init_center, dim =1) <= self.init_rad)
            self.init_domain = self.domain[mask]
            
            if self.config["unsafe"]["shape"] == 'Rectangle':
                self.unsafe = self.config["unsafe"]["range"]
                if not "unbounded" in self.config["unsafe"]:
                    self.unsafe_min = torch.tensor([self.unsafe[0][0],self.unsafe[1][0]])        
                    self.unsafe_max = torch.tensor([self.unsafe[0][1],self.unsafe[1][1]])
                else:
                    if self.config["unsafe"]["unbounded"] == 'x' and self.config["unsafe"]["max_min"] == 'min':
                        self.unsafe_min = torch.tensor([self.unsafe[0][0],self.unsafe[1][0]]) 
                        self.unsafe_max = torch.tensor([100, self.unsafe[1][1]])       
                    if self.config["unsafe"]["unbounded"] == 'x' and self.config["unsafe"]["max_min"] == 'max':
                        self.unsafe_min = torch.tensor([-100, self.unsafe[1][0]]) 
                        self.unsafe_max = torch.tensor([self.unsafe[0][0], self.unsafe[1][1]])       
                    if self.config["unsafe"]["unbounded"] == 'y' and self.config["unsafe"]["max_min"] == 'min':
                        self.unsafe_min = torch.tensor([self.unsafe[0][0], self.unsafe[1][0]]) 
                        self.unsafe_max = torch.tensor([self.unsafe[0][1], 100])       
                    if self.config["unsafe"]["unbounded"] == 'y' and self.config["unsafe"]["max_min"] == 'max':
                        self.unsafe_min = torch.tensor([self.unsafe[0][0], -100]) 
                        self.unsafe_max = torch.tensor([self.unsafe[0][1], self.unsafe[1][0]])       

                self.unsafe_domain = self.domain[((self.domain >= torch.tensor(self.unsafe_min)) & (self.domain <= torch.tensor(self.unsafe_max))).all(dim=1)]

            elif self.config["unsafe"]["shape"] == 'Circle':
                self.uns_center = torch.tensor(self.config["unsafe"]["center"])
                self.uns_rad = self.config["unsafe"]["radius"]
                mask = (torch.linalg.norm(self.domain - self.uns_center, dim =1) <= self.uns_rad )
                self.unsafe_domain = self.domain[mask]

            elif self.config["unsafe"]["shape"] == 'Custom':
                x= self.domain[:,0]
                y= self.domain[:,1]
                result = eval(self.config["unsafe"]["function"])
                mask = (result <= 0)
                self.unsafe_domain = self.domain[mask]

        #Dataset Generation and Shuffling
        num_rows = self.init_domain.size(0)
        random_index = torch.randint(0, num_rows, (1,)).item()
        self.initial_set_random = self.init_domain[random_index].reshape(1,2)
        self.initial_set_center = torch.cat([self.initial_set_center, self.initial_set_random])
        domain_dataset = torch.utils.data.TensorDataset(self.domain)
        init_dataset  = torch.utils.data.TensorDataset(self.init_domain)
        unsafe_dataset  = torch.utils.data.TensorDataset(self.unsafe_domain)
        train_dataset  = torch.utils.data.TensorDataset(self.X_train, self.y_train)

        total_size = len(self.domain) + len(self.init_domain) + len(self.unsafe_domain)

        self.batch_size = self.config["model_b"]["batch_size"]

        domain_batch_size = int(len(self.domain) / total_size * self.batch_size)	
        init_batch_size = max(2, int(len(self.init_domain) / total_size * self.batch_size))
        unsafe_batch_size = max(2, int(len(self.unsafe_domain) / total_size * self.batch_size))

        self.domain_loader = torch.utils.data.DataLoader(domain_dataset, batch_size=domain_batch_size, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=self.seed_worker, generator=self.g)
        self.init_loader = torch.utils.data.DataLoader(init_dataset, batch_size=init_batch_size,  shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=self.seed_worker, generator=self.g)
        self.unsafe_loader = torch.utils.data.DataLoader(unsafe_dataset, batch_size=unsafe_batch_size, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=self.seed_worker, generator=self.g)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config["model_b"]["batch_size"], shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=self.seed_worker, generator=self.g)

        self.N_cex_domain = self.config["counterex"]["N_cex_domain"] #The number of counterexample samples we want in the domain region        
        
        #Reset learning rate, will be called after training the initial dynamics
        if "learning_rate_cert" in self.config["model_f"]:
            if "reg_f" in self.config["hyperparameters"]:
                self.optimizer_f = torch.optim.Adam(self.model_f.parameters(),
                    lr=self.config["model_f"]["learning_rate_cert"], weight_decay = self.config["hyperparameters"]["reg_f"], betas=(0.9, 0.999))
            else:
                self.optimizer_f = torch.optim.Adam(self.model_f.parameters(),
                    lr=self.config["model_f"]["learning_rate_cert"], betas=(0.9, 0.999))
            self.optimizer_f_state_dict = self.optimizer_f.state_dict()
    
    def generate_counterexample_data(self):
        self.load_model_states()     
        input_domain, _ = data.generateRandomData(self.config["counterex"]["N_cex_domain"],self.RANGE) #the domain is limited to [-1,1] due to normalization
        mask = (torch.linalg.norm(input_domain - self.init_center, dim =1) <= self.init_rad)
        init_domain = input_domain[mask]
        
        if self.config["unsafe"]["shape"] == 'Rectangle':
            unsafe_domain =  input_domain[((input_domain >= torch.tensor(self.unsafe_min)) & (input_domain <= torch.tensor(self.unsafe_max))).all(dim=1)]        
        elif self.config["unsafe"]["shape"] == 'Circle':
            mask = (torch.linalg.norm(input_domain - self.uns_center, dim =1) <= self.uns_rad)
            unsafe_domain = input_domain[mask]
        elif self.config["unsafe"]["shape"] == 'Custom':
            x= input_domain[:,0]
            y = input_domain[:,1]
            result = eval(self.config["unsafe"]["function"])
            unsafe_domain = input_domain[result <= 0]


        counterexamples_domain = verification.verify_domain(self.model_v, self.model_b, self.model_f,                                                       input_domain, self.config)
        counterexamples_init = verification.verify_init(self.model_b, init_domain, self.config)
        counterexamples_unsafe = verification.verify_unsafe(self.model_b, unsafe_domain, self.config)
        add_data_domain = []
        add_data_init = []
        add_data_unsafe = []
        
        for counterexample in counterexamples_domain:
            #for _ in range(self.config["counterex"]["N"]):
                #random_point = counterexample + (torch.rand(counterexample.shape) - 0.5) * 2 * self.config["counterex"]["radius"]
            add_data_domain.append(counterexample)
            
        if len(add_data_domain) > 0:
            add_data_domain = torch.stack(add_data_domain).detach()
        else:
            add_data_domain = None
 
        for counterexample in counterexamples_init:
                add_data_init.append(counterexample)
        
        if len(add_data_init) > 0:
            add_data_init = torch.stack(add_data_init).detach()
        else:
            add_data_init = None

        for counterexample in counterexamples_unsafe:
                add_data_unsafe.append(counterexample)

        if len(add_data_unsafe) > 0:
             add_data_unsafe = torch.stack(add_data_unsafe).detach()
        else:
            add_data_unsafe = None

        if add_data_domain is not None:
             print_info(f"DOMAIN COUNTEREXAMPLES ADDED : {add_data_domain.shape[0]} CEs")
             self.domain = torch.unique(torch.cat([self.domain, add_data_domain], dim=0), dim = 0)
             mask = (torch.linalg.norm(self.domain - self.init_center, dim =1) <= self.init_rad) 
             self.init_domain = self.domain[mask]            
             #self.init_domain = self.domain[((self.domain >= torch.tensor(self.init_min)) & (self.domain <= torch.tensor(self.init_max))).all(dim=1)]
             if self.config["unsafe"]["shape"] == 'Rectangle':
                self.unsafe_domain = self.domain[((self.domain >= torch.tensor(self.unsafe_min)) & (self.domain <= torch.tensor(self.unsafe_max))).all(dim=1)]
             elif self.config["unsafe"]["shape"] == 'Circle':
                self.unsafe_domain = self.domain[(torch.linalg.norm(self.domain - self.uns_center, dim =1) <= self.uns_rad )]
        if add_data_init is not None:
            print_info(f"INIT COUNTEREXAMPLES ADDED : {add_data_init.shape[0]} CEs")
            self.init_domain = torch.unique(torch.cat([self.init_domain, add_data_init], dim=0), dim = 0)
            self.domain = torch.unique(torch.cat([self.domain, add_data_init], dim=0), dim = 0)
        
        if add_data_unsafe is not None:
            print_info(f"UNSAFE COUNTEREXAMPLES ADDED : {add_data_unsafe.shape[0]} CEs")
            self.unsafe_domain = torch.unique(torch.cat([self.unsafe_domain, add_data_unsafe], dim=0), dim =0)
            self.domain = torch.unique(torch.cat([self.domain, add_data_unsafe], dim=0), dim = 0)
                
        elif add_data_domain is None and add_data_init is None and add_data_unsafe is None:
            print_info("NO COUNTEREXAMPLES ADDED")
            self.counterexamples_added = False

        #Dataset Generation and Shuffling

        domain_dataset = torch.utils.data.TensorDataset(self.domain)
        init_dataset  = torch.utils.data.TensorDataset(self.init_domain)
        unsafe_dataset  = torch.utils.data.TensorDataset(self.unsafe_domain)

        total_size = len(self.domain) + len(self.init_domain) + len(self.unsafe_domain)

        domain_batch_size = int(len(self.domain) / total_size * self.batch_size)	
        init_batch_size = max(4, int(len(self.init_domain) / total_size * self.batch_size))
        unsafe_batch_size = max(3, int(len(self.unsafe_domain) / total_size * self.batch_size))

        self.domain_loader = torch.utils.data.DataLoader(domain_dataset, batch_size=domain_batch_size, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=self.seed_worker, generator=self.g)
        self.init_loader = torch.utils.data.DataLoader(init_dataset, batch_size=init_batch_size,  shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=self.seed_worker, generator=self.g)
        self.unsafe_loader = torch.utils.data.DataLoader(unsafe_dataset, batch_size=unsafe_batch_size, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=self.seed_worker, generator=self.g)

    def trainInitialDynamics(self):
        self.hidden_neurons_f = self.config["model_f"]["hidden_neurons"]
        self.hidden_layers_f = self.config["model_f"]["layers"]
        sigmoid_f = NNModels.assignActivationFunction(self.config['model_f']['activation_function'])
        self.hidden_f = [self.hidden_neurons_f] * self.hidden_layers_f
        self.model_f = NNModels.DyanmicsNet(self.dim_in, 
                                            self.hidden_f, 
                                            sigmoid_f).to(self.device)
        best_mse = np.inf   # init to infinity
        best_weights = None
        history = []
        loss_fn = nn.MSELoss()  # mean square error #TODO: Add Lyapunov, barrier, regularization loss
        self.optimizer_f = torch.optim.Adam( self.model_f.parameters(), lr=self.config["model_f"]["learning_rate"],betas=(0.9, 0.999))
        self.scheduler_f = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_f, mode='min', factor=self.config["model_f"]["lr_factor"], 
                                                                    patience=self.config["model_f"]["lr_patience"], verbose=True)
        for epoch in tqdm(range(self.config["model_f"]["epochs_warm"])):
            total_loss = 0
            # print(epoch)
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                self.model_f.train()
                # Calculate the loss
                x_val = X_batch.float().to(self.device)
                y_pred = self.model_f(x_val)
                #hyperparameter for l2 regularization
                loss_mse = loss_fn(y_pred, y_batch.float().to(self.device)) #+ DECAY_L2*sum(param.pow(2).sum() for param in self.model_f.parameters())
                #alpha = self.config["hyperparameters"]["alpha"]
                #dec = self.config["model_f"]["init_decay"]
                #loss_lyap = dec*torch.sum(F.leaky_relu(2*x_val*y_pred,alpha))
                loss = loss_mse #+ loss_lyap
                # backward pass
                self.optimizer_f.zero_grad()
                loss.backward()
                self.optimizer_f.step()
                #self.model_f.clip_weights()
                total_loss += loss.item()
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
        # Store the model state dictionary
        self.model_f_state_dict = best_weights
        self.optimizer_f_state_dict = self.optimizer_f.state_dict()

        print_info("MSE of Initial Estimate of Dynamical System: %.4f" % best_mse)

    def trainCertificate(self):
        if self.config["Barrier"]:
            #Building the Lyapunov Model
            hidden_neurons_v = self.config["model_v"]["hidden_neurons"]
            hidden_layers_v = self.config["model_v"]["layers"]
            hidden_v = [hidden_neurons_v] * hidden_layers_v            
            self.model_v = NNModels.LyapunovNet(
            n_input=self.dim_in,
            hidden_v=hidden_v,
            #thresholds=self.config["model_v"]["clip"],
            sigmoid_v=NNModels.assignActivationFunction(self.config['model_v']['activation_function'])).to(self.device)
            #Optimizer and Scheduler for Lyapunov Function
            self.optimizer_v = torch.optim.Adam(self.model_v.parameters(), lr = self.config["model_v"]["learning_rate"], weight_decay = self.config["hyperparameters"]["reg_v"])
            warmup_scheduler_v = opt.WarmUpLR(self.optimizer_v, self.config["model_v"]["warmup"], self.config["model_v"]["learning_rate"])
                
            self.scheduler_v = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_v, mode = 'min', factor = self.config["model_v"]["lr_factor"],
                                                                    patience = self.config["model_v"]["lr_patience"], verbose = True)
            
            #Building the Barrier Model
            hidden_neurons_b = self.config["model_b"]["hidden_neurons"]
            hidden_layers_b = self.config["model_b"]["layers"]
            hidden_b = [hidden_neurons_b] * hidden_layers_b
            self.model_b = NNModels.BarrierNet(
            n_input=self.dim_in,
            hidden_b=hidden_b,
            #thresholds = self.config["model_b"]["clip"],
            sigmoid_b=NNModels.assignActivationFunction(self.config['model_b']['activation_function'])).to(self.device)
            #Optimizer and Scheduler for Barrier Function
            self.optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr = self.config["model_b"]["learning_rate"], weight_decay = self.config["hyperparameters"]["reg_bar"])
            warmup_scheduler_b = opt.WarmUpLR(self.optimizer_b, self.config["model_b"]["warmup"], self.config["model_b"]["learning_rate"])
            
            self.scheduler_b = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_b, mode = 'min', factor = self.config["model_b"]["lr_factor"],
                                                                    patience = self.config["model_b"]["lr_patience"], verbose = True)

            # Load the stored model state dictionary if available
            
            self.load_model_states()        

            # Start Training
            start = timeit.default_timer() 
            max_iter = self.config["hyperparameters"]["max_iters"]

            for epoch in tqdm(range(max_iter)):
                cert_loss_b = 0
                cert_loss_v = 0
                dyn_loss = 0

                for batches in itertools.zip_longest(self.domain_loader, self.train_loader, self.init_loader, self.unsafe_loader, fillvalue=None):
                    self.batches= batches
                    if batches[0] is not None:
                        input_domain = batches[0][0].to(self.device)
                        self.optimizer_f.zero_grad()   
                        self.optimizer_v.zero_grad()
                        loss_domain_v, _ = Loss_Functions.loss_function_domain(self.model_v, self.model_b, self.model_f, input_domain, self.config)
                        loss_domain_v.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(self.model_v.parameters(), max_norm=1.0)
                        self.optimizer_v.step()
                        if "train_f_cert" not in self.config["model_f"]:    
                            self.optimizer_f.step()   
                        else:
                            if self.config["model_f"]["train_f_cert"] == False and epoch <= max_iter*self.config["model_f"]["cert_train_factor"]:
                                pass
                            else:
                                self.optimizer_f.step()

                        #self.model_v.clip_weights()
                    else:
                        loss_domain_v = torch.tensor(0.0, requires_grad = True)
                    
                    if batches[0] is not None or batches[2] is not None or batches[3] is not None:
                        self.optimizer_f.zero_grad()
                        self.optimizer_b.zero_grad()
                        loss_domain_b = torch.tensor(0.0, requires_grad=True)
                        loss_init_b = torch.tensor(0.0, requires_grad=True)
                        loss_unsafe_b = torch.tensor(0.0, requires_grad=True)

                        if batches[0] is not None:
                            _, loss_domain_b = Loss_Functions.loss_function_domain(self.model_v, self.model_b, self.model_f, input_domain, self.config)
                        if batches[2] is not None:
                            input_init = batches[2][0].to(self.device)
                            loss_init_b = Loss_Functions.loss_function_init(self.model_b, input_init, self.config)
                        if batches[3] is not None:
                            input_unsafe = batches[3][0].to(self.device)
                            loss_unsafe_b = Loss_Functions.loss_function_unsafe(self.model_b, input_unsafe, self.config)

                        loss_b = loss_domain_b + loss_init_b + loss_unsafe_b
                        loss_b.backward()
                        torch.nn.utils.clip_grad_norm_(self.model_b.parameters(), max_norm=1.0)
                        self.optimizer_b.step()
                        if "train_f_cert" not in self.config["model_f"]:    
                            self.optimizer_f.step()   
                        else:
                            if self.config["model_f"]["train_f_cert"] == False and epoch <= max_iter*self.config["model_f"]["cert_train_factor"]:
                                pass
                            else:
                                self.optimizer_f.step()

                        #self.model_b.clip_weights()
                    if batches[1] is not None:
                         input_train = batches[1][0].to(self.device)
                         output_train = batches[1][1].to(self.device)
                         self.optimizer_f.zero_grad()
                         loss_train = Loss_Functions.loss_function_dyn(self.model_f, input_train, output_train, self.config)
                         loss_init = torch.tensor(0.0, requires_grad=True)
                         if batches[2] is not None:
                            input_init = batches[2][0].to(self.device)
                            loss_init = Loss_Functions.loss_function_init_direction(self.model_f, input_train, output_train, input_init, self.initial_set_center[0], self.config)
                         loss_dyn = loss_train + loss_init
                         loss_dyn.backward()
                         torch.nn.utils.clip_grad_norm_(self.model_f.parameters(), max_norm=1.0)
                         self.optimizer_f.step()
                         #self.model_f.clip_weights()
 
                    else:
                        loss_dyn = torch.tensor(0.0, requires_grad=True)
                    
                    # Calculate average training loss for the epoch
                    cert_loss_b += loss_b
                    cert_loss_v += loss_domain_v
                    dyn_loss += loss_dyn

                    avg_loss_f = dyn_loss / len(self.train_loader)
                    # Step the scheduler with training loss
                    self.scheduler_f.step(avg_loss_f)
                    # Update learning rate with warm-up
                    if epoch < self.config["model_v"]["warmup"]:
                        warmup_scheduler_v.step()
                    else:
                        avg_loss_cert_v = cert_loss_v/len(self.domain_loader)
                        self.scheduler_v.step(avg_loss_cert_v)

                    if epoch < self.config["model_b"]["warmup"]:
                        warmup_scheduler_b.step()
                    else:
                        avg_loss_cert_b = cert_loss_b/len(self.domain_loader)
                        self.scheduler_b.step(avg_loss_cert_b)

                
                # Log the training loss
                decay=self.config["hyperparameters"]["decay_mse"]
                # print(f"Epoch {epoch + 1}/{max_iter}, MSE Loss: {dyn_loss.item()/decay}")
                # print(f"Epoch {epoch + 1}/{max_iter}, Certificate Loss: {cert_loss_v.item() + cert_loss_b.item()}")

            # Save the recent versions of model_v and model_b in memory
            self.model_v_state_dict = self.model_v.state_dict()
            self.model_b_state_dict = self.model_b.state_dict()
            self.model_f_state_dict = self.model_f.state_dict()
            self.final_mse_loss = dyn_loss.item()
            self.final_cert_loss = cert_loss_v.item() + cert_loss_b.item()

    def verifyCertificate(self):
        self.load_model_states()     
        input_domain, _ = data.generateRandomData(self.config["verification"]["N_conf"],self.RANGE) #the domain is limited to [-1,1] due to normalization
        mask = (torch.linalg.norm(input_domain - self.init_center, dim =1) <= self.init_rad)
        init_domain = input_domain[mask]
        
        if self.config["unsafe"]["shape"] == 'Rectangle':
            unsafe_domain =  input_domain[((input_domain >= torch.tensor(self.unsafe_min)) & (input_domain <= torch.tensor(self.unsafe_max))).all(dim=1)]        
        elif self.config["unsafe"]["shape"] == 'Circle':
            mask = (torch.linalg.norm(input_domain - self.uns_center, dim =1) <= self.uns_rad )
            unsafe_domain = input_domain[mask]
        elif self.config["unsafe"]["shape"] == 'Custom':
            x= input_domain[:,0]
            y = input_domain[:,1]
            result = eval(self.config["unsafe"]["function"])
            unsafe_domain = input_domain[result <= 0]

        q, beta = verification.conformal_prediction(self.model_v, self.model_b, self.model_f,input_domain, init_domain, unsafe_domain, self.config)
        if q <= 0:
            self.flag_verified = True
            conf = 1-self.config["verification"]["epsilon"]
            print(f"With a confidence of {1-beta}, conditions are valid with probability {conf}")
        else:
            self.flag_verified = False
            print(f"Verification failed with marginal safety error: {q}")

    def final_model_eval(self):
        self.model_f.eval()
        self.mse = 0
        total_samples = 0

        for batch_idx, (X_batch, y_batch) in enumerate(self.test_loader):
            y_pred = self.model_f(X_batch.float().to(self.device))
            loss_fn = nn.MSELoss(reduction = 'sum')
            batch_mse = loss_fn(y_pred, y_batch.float().to(self.device))
            self.mse += batch_mse.item()
            total_samples += X_batch.size(0)  
        self.mse = self.mse / total_samples

    def update_config(self):
        # Construct the init_range value
        file_path = "./config_files/" + self.args.lasa_name + "_config_benchmark.json"
        # init_range = [[self.init_min[0][0], self.init_max[0][0]], [self.init_min[0][1], self.init_max[0][1]]]
        initial_conditions = self.initial_set_center.tolist()

        # Load the existing configuration file
        with open(file_path, 'r') as config_file:
            config = json.load(config_file)

        # Update the 'plotting' key with the init_range
        if "plotting" not in config:
            config["plotting"] = {}
        # config["plotting"]["init_range"] = init_range
        config["plotting"]["initial_conditions"] = initial_conditions
        config["plotting"]["dt"] = self.dt

        # Save the updated configuration back to the file
        with open(file_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)

    def save_datasets(self):
        base_path = os.path.join('Datasets_2D', self.args.lasa_name)
        os.makedirs(base_path, exist_ok = True)
        torch.save(self.X_train, os.path.join(base_path,"X_train.pt"))
        torch.save(self.y_train, os.path.join(base_path,"y_train.pt"))
        torch.save(self.X_test, os.path.join(base_path,"X_test.pt"))
        torch.save(self.y_test, os.path.join(base_path,"y_test.pt"))

if __name__ == "__main__":
    # Settings Seeds for Reproducibility
    filtered_args = filter_args(sys.argv[1:])
    args = pyrallis.parse(ConfigFile, args=filtered_args)
    seed_filepath = f'seeds/{args.lasa_name}_seed_benchmark.json'
    #Check if the seed file exists
    try:
       seed = load_seed(seed_filepath)
    except FileNotFoundError:
       seed = random.randint(0, 100)  # seed value
    print(seed)
    set_seed(87)
    mp = MotionPlanner(args)
    print_info("OBTAINING DEMO DATA")
    mp.generate_demo_data()
    print_info("DYNAMICAL SYSTEM TRAINING")
    mp.trainInitialDynamics()
    Plotter.initialDSPlot(mp.model_f, mp.X_train, mp.initial_set_center, mp.dt)
    print_info("OBTAINING TRAINING DATA")
    mp.generate_domain_data()
    iters = 1
    print_info("CERTIFICATE TRAINING")
    mp.trainCertificate()
    trial = 1
    while trial < 500:
        print_info("ADDING COUNTEREXAMPLES")
        mp.generate_counterexample_data()
        print(f"Trial: {trial}")
        if trial % 5 == 0:
            Plotter.plotObstacle(mp.model_f, mp.model_b, mp.X_train, mp.initial_set_center[0], mp.config)
        if mp.counterexamples_added:
            mp.trainCertificate()
            trial += 1      
        else:
            print_info("SAMPLING-BASED VERIFICATION COMPLETE")
            Plotter.initialDSPlot(mp.model_f, mp.X_train, mp.initial_set_center, mp.dt)
            Plotter.lyapunovBarrierPlot(mp.model_v, mp.model_b, mp.model_f, mp.X_train, None,None,mp.config)
            # Plotter.plotLyapunov(mp.model_v)
            # Plotter.plotBarrier(mp.model_b)
            mp.verifyCertificate()
            if mp.flag_verified:
                import pdb; pdb.set_trace()
                mp.save_all_models()
                mp.final_model_eval()
                #print_info(f"MSE for test data after certificate training: {mp.mse}")
                save_seed(seed,seed_filepath)
                mp.export_onnx()
                #save initial set for plotting
                mp.update_config()
                #save datasets for results
                mp.save_datasets()
                break
    if trial == 100:
        print_error("MAXIMUM TRIALS EXCEEDED... SAMPLING VERIFICATION FAILED")
     