from common_header import *
import NNModels
import data
import Loss_Functions
import opt 
import Plotter
import verification

@dataclass
class ConfigFile:
    lasa_name : str = "Worm"
    dataset_type : str = "LASA" # This can also be 3D_Shapes
    name_3d: str = "Cshape_bottom"
    name_2d: str = "Five_Obstacle_DS"
    
def filter_args(args):
    known_args = ['--lasa_name', '--dataset_type', '--name_3d', '--name_2d']
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
        # Get the name of the Dataset
        if self.args.dataset_type == 'LASA':
            self.name = self.args.lasa_name
        elif self.args.dataset_type == '3D_Shapes':
            self.name = self.args.name_3d
        elif self.args.dataset_type == '2D_Shapes':
            self.name = self.args.name_2d
        # File location
        self.par_dir_path = os.path.dirname(os.path.realpath(__file__))
        # Load the configuration file
        # file_path = os.path.join(self.par_dir_path, "config_files", self.args.dataset_type, self.name + "_config.json")
        file_path = os.path.join("config_files", self.args.dataset_type, self.name + "_config.json")
        if os.path.exists(file_path):     
            with open(file_path) as file:
                self.config = json.load(file)
        else:
            print_error(f"Error: Configuration file '{file_path}' not found!")
            sys.exit(1)
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
        self.lr_f = self.config["model_f"].get("learning_rate", 1e-3)
        self.lr_v = self.config["model_v"].get("learning_rate", 1e-3)
        self.lr_b = self.config["model_b"].get("learning_rate", 1e-3)
        self.counterexamples_added = True #Setting the counterexamples flag to true
        self.g = torch.Generator()
        self.g.manual_seed(0)    

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
        full_path = os.path.join(self.par_dir_path, model_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        torch.save(model, full_path)

    def save_all_models(self):
        base_path = os.path.join(self.par_dir_path,'models', self.args.dataset_type, self.name)
        os.makedirs(base_path, exist_ok=True)  # Ensure the directory exists
        # Save the model states 
        self.save_model(self.model_f, os.path.join(base_path, 'model_f.pth'))
        self.save_model(self.model_v, os.path.join(base_path, 'model_v.pth'))
        self.save_model(self.model_b, os.path.join(base_path, 'model_b.pth'))

    def export_onnx(self):
        base_path = os.path.join(self.par_dir_path, 'models_onnx', self.args.dataset_type)
        os.makedirs(base_path, exist_ok=True)  # Ensure the directory exists
        file_path = os.path.join(base_path, self.name + ".onnx")
        self.model_f = self.model_f.to(self.device)
        self.model_f.eval()
        self.initial_point = self.initial_set_center[0].reshape(1,self.dim_in).to(device=self.device, dtype=torch.float32) 
        torch.onnx.export(
                self.model_f,
                self.initial_point,
                file_path,
                export_params=True,  
                opset_version=11,    
                do_constant_folding=True,  
                input_names=["position"],     
                output_names=["velocity"], 
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  
            )
        onnx_model = onnx.load(os.path.join(base_path, self.name + ".onnx"))
        onnx.checker.check_model(onnx_model)         
        
    def generate_demo_data(self): 
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
            elif self.args.lasa_name == "WShape":
                dataset = lasa.DataSet.WShape
            elif self.args.lasa_name == "Leaf_2":
                dataset = lasa.DataSet.Leaf_2
            elif self.args.lasa_name == "Sine":
                dataset = lasa.DataSet.Sine
            elif self.args.lasa_name == "Snake":
                dataset = lasa.DataSet.Snake
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
            self.total_demos = len(self.demos)
            self.dim_in = 2
        elif self.args.dataset_type == '3D_Shapes':
            self.dt = 0.01
            self.dim_in = 3
            folder_path = os.path.join(self.par_dir_path, os.getcwd(),"Datasets", "3D_Shapes")
            if not os.path.isdir(folder_path):
                print_error("Dataset not found!")
                sys.exit(1)
            path_name = os.path.join(folder_path, "3D_" + self.name + ".mat")
            if os.path.exists(path_name):
                mat = scipy.io.loadmat(path_name)
            else:
                print_error("Dataset not found!")
            data_value = np.squeeze(mat["data"])
            self.total_demos = len(data_value)
            self.demos = []
            for i in range(data_value.shape[0]):
                traj = data_value[i]
                pos = traj[:3, :]
                vel = traj[3:, :]
                # Move Trajectory to zero at origin
                pos = pos[:, :] - pos[:,-1][:,np.newaxis]
                self.demos.append(data.TrajectoryData(pos, vel))
        elif self.args.dataset_type == '2D_Shapes':
            self.dt = 0.01
            folder_path = os.path.join(self.par_dir_path, os.getcwd(), "Datasets", "2D_Shapes", self.name)
            if not os.path.isdir(folder_path):
                print_error(f"Error: Folder '{folder_path}' does not exist!")
                sys.exit(1)
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            self.demos = []
            subsample = subsample = self.config["hyperparameters"].get("subsample", 10)
            self.dim_in = 2
            for csv_file in csv_files:
                file_path = os.path.join(folder_path, csv_file)
                try:
                    df = pd.read_csv(file_path)
                    pos_array = np.vstack([np.array(df['x']), np.array(df['y'])]).T
                    vel_array = np.vstack([np.array(df['dx']), np.array(df['dy'])]).T
                    # After Subsampling
                    pos_array = pos_array[::subsample].T
                    pos_array = (pos_array[:,:] - pos_array[:,-1][:, np.newaxis])
                    vel_array = vel_array[::subsample].T
                    self.demos.append(data.TrajectoryData(pos_array, vel_array))
                except pd.errors.EmptyDataError:
                    print_error(f"Error: File '{file_path}' is empty!")
                    sys.exit(1)
            self.total_demos = len(self.demos)
        else:
            print_error("Non-LASA Dataset has been choosen")   
        self.demos = np.array(self.demos)         
        # Divide the data into training and testing
        train_size = int(5/7 * self.total_demos) # 5/7 datasets are used for training
        train_indices = random.sample(range(self.total_demos), train_size)
        test_indices = list(set(range(self.total_demos)) - set(train_indices))
        self.X_train = np.concatenate([self.demos[i].pos for i in train_indices], axis=1).T
        self.X_test = np.concatenate([self.demos[i].pos for i in test_indices], axis=1).T
        self.y_train = np.concatenate([self.demos[i].vel for i in train_indices], axis=1).T
        self.y_test = np.concatenate([self.demos[i].vel for i in test_indices], axis=1).T 
        # Convert to Pytorch Tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)  
        assert self.X_train.shape[0] == self.y_train.shape[0], "Mismatch in number of samples between X_train and y_train"
        assert self.X_test.shape[0] == self.y_test.shape[0], "Mismatch in number of samples between X_test and y_test" 
        self.initial_set_center = np.mean([self.demos[i].pos[:,0] for i in range(self.total_demos)], axis=0)
        # Normalise the Trajectories to [-1, 1] #Use the maximum value to normalize and scale the data.
        self.pos_scaling = torch.max(torch.concatenate([abs(self.X_train), abs(self.X_test)]))
        self.vel_scaling = torch.max(torch.concatenate([abs(self.y_train), abs(self.y_test)]))
        self.X_train /= self.pos_scaling
        self.X_test /= self.pos_scaling
        self.y_train /= self.vel_scaling
        self.y_test /= self.vel_scaling
        train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        test_dataset = torch.utils.data.TensorDataset(self.X_test, self.y_test)
        batch_size = self.config["model_f"].get("batch_size", 128)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=self.seed_worker, generator=self.g)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, worker_init_fn=self.seed_worker, generator=self.g)
        self.initial_set_center = (self.initial_set_center/self.pos_scaling).reshape(1,self.dim_in)

        # Rescaling the images
        for i in range(len(self.demos)):
            self.demos[i].pos = self.demos[i].pos/self.pos_scaling.cpu().detach().numpy()
            self.demos[i].vel = self.demos[i].vel/self.vel_scaling.cpu().detach().numpy()
            
    def generate_domain_data(self):
        self.N_domain = self.config["domain"].get("N", 10000) 
        self.RANGE = self.config["domain"].get("range", [[-1, 1]] * self.dim_in)
        self.domain, _ = data.generateRandomData(self.N_domain, self.RANGE, self.dim_in)
        #Generate data for initial set
        self.init_min = (np.min([self.demos[i].pos[:,0] for i in range(self.total_demos)], axis=0) - self.config["init"].get("radius", 0.01)).reshape(1,self.dim_in)
        self.init_min = np.where(self.init_min < -1, -1, self.init_min)
        self.init_max = (np.max([self.demos[i].pos[:,0] for i in range(self.total_demos)], axis=0) + self.config["init"].get("radius", 0.01)).reshape(1,self.dim_in)
        self.init_max = np.where(self.init_max > 1, 1, self.init_max)
        
        self.init_domain = self.domain[((self.domain >= torch.tensor(self.init_min)) & (self.domain <= torch.tensor(self.init_max))).all(dim=1)]
        #Dataset Generation and Shuffling
        num_rows = self.init_domain.size(0)
        random_index = torch.randint(0, num_rows, (1,)).item()
        self.initial_set_random = self.init_domain[random_index].reshape(1,self.dim_in)
        self.initial_set_center = torch.cat([self.initial_set_center, self.initial_set_random])

        #Generate data for unsafe set
        unsafe_config = self.config["unsafe"]
        shape = unsafe_config.get("shape", None)
        if shape == 'Rectangle':
            self.unsafe = self.config["unsafe"]["range"]
            if not "unbounded" in unsafe_config:
                if self.dim_in == 2:
                    self.unsafe_min = torch.tensor([self.unsafe[0][0],self.unsafe[1][0]])        
                    self.unsafe_max = torch.tensor([self.unsafe[0][1],self.unsafe[1][1]])
                elif self.dim_in == 3:
                    self.unsafe_min = torch.tensor([self.unsafe[0][0],self.unsafe[1][0], self.unsafe[2][0]])        
                    self.unsafe_max = torch.tensor([self.unsafe[0][1],self.unsafe[1][1], self.unsafe[2][1]])
            else:
                if unsafe_config.get("unbounded") == 'x' and unsafe_config.get("max_min") == 'min':
                    self.unsafe_min = torch.tensor([self.unsafe[0][0],self.unsafe[1][0]]) 
                    self.unsafe_max = torch.tensor([100, self.unsafe[1][1]])       
                if unsafe_config.get("unbounded") == 'x' and unsafe_config.get("max_min") == 'max':
                    self.unsafe_min = torch.tensor([-100, self.unsafe[1][0]]) 
                    self.unsafe_max = torch.tensor([self.unsafe[0][0], self.unsafe[1][1]])       
                if unsafe_config.get("unbounded") == 'y' and unsafe_config.get("max_min") == 'min':
                    self.unsafe_min = torch.tensor([self.unsafe[0][0], self.unsafe[1][0]]) 
                    self.unsafe_max = torch.tensor([self.unsafe[0][1], 100])       
                if unsafe_config.get("unbounded") == 'y' and unsafe_config.get("max_min") == 'max':
                    self.unsafe_min = torch.tensor([self.unsafe[0][0], -100]) 
                    self.unsafe_max = torch.tensor([self.unsafe[0][1], self.unsafe[1][0]])       
            self.unsafe_domain = self.domain[((self.domain >= self.unsafe_min.clone().detach()) & (self.domain <= self.unsafe_max.clone().detach())).all(dim=1)]
        
        elif shape == 'Circle':
            self.uns_center = torch.tensor(unsafe_config.get("center", [[0.0, 0.0]]*self.dim_in))
            self.uns_rad = unsafe_config.get("radius", 0.0)
            if isinstance(self.uns_center[0], (int, float)):
                mask = (torch.linalg.norm(self.domain - self.uns_center, dim =1) <= self.uns_rad )
                self.unsafe_domain = self.domain[mask]
            else:
                all_masks = torch.zeros(len(self.domain), dtype=torch.bool)
                for center in self.uns_center:
                    mask = (torch.linalg.norm(self.domain - center, dim =1) <= self.uns_rad )
                    all_masks = all_masks | mask
                self.unsafe_domain = self.domain[all_masks]

        elif shape == 'Custom':
            x = self.domain[:,0]
            y = self.domain[:,1]
            result = eval(unsafe_config.get("function"))
            mask = (result <= 0)
            self.unsafe_domain = self.domain[mask]                        
        
        #Dataset Generation and Shuffling
        if self.unsafe_domain is None:
            self.unsafe_domain = torch.empty((0, self.dim_in))

        domain_dataset = torch.utils.data.TensorDataset(self.domain)
        init_dataset  = torch.utils.data.TensorDataset(self.init_domain)
        unsafe_dataset  = torch.utils.data.TensorDataset(self.unsafe_domain)
        train_dataset  = torch.utils.data.TensorDataset(self.X_train, self.y_train)

        total_size = len(self.domain) + len(self.init_domain) + len(self.unsafe_domain)

        self.batch_size = self.config["model_b"].get("batch_size", 128)

        domain_batch_size = int(len(self.domain) / total_size * self.batch_size)	
        init_batch_size = max(2, int(len(self.init_domain) / total_size * self.batch_size))
        unsafe_batch_size = max(2, int(len(self.unsafe_domain) / total_size * self.batch_size))

        self.domain_loader = torch.utils.data.DataLoader(domain_dataset, batch_size=domain_batch_size, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=self.seed_worker, generator=self.g)
        self.init_loader = torch.utils.data.DataLoader(init_dataset, batch_size=init_batch_size,  shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=self.seed_worker, generator=self.g)
        self.unsafe_loader = torch.utils.data.DataLoader(unsafe_dataset, batch_size=unsafe_batch_size, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=self.seed_worker, generator=self.g)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config["model_b"].get("batch_size", 128), shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=self.seed_worker, generator=self.g)

        self.N_cex_domain = self.config["counterex"].get("N_cex_domain", 10000) #The number of counterexample samples we want in the domain region

        if "reg_f" in self.config["hyperparameters"]:
            self.optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=self.config["model_f"].get("learning_rate_cert", 1e-10), weight_decay = self.config["hyperparameters"]["reg_f"], betas=(0.9, 0.999))
            self.scheduler_f = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_f, mode='min', factor=self.config["model_f"].get("lr_factor", 1.0), patience=self.config["model_f"].get("lr_patience", 30), verbose=True)
            self.optimizer_f_state_dict = self.optimizer_f.state_dict()
        else:
            self.optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=self.config["model_f"].get("learning_rate_cert", 1e-10), betas=(0.9, 0.999))
            self.scheduler_f = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_f, mode='min', factor=self.config["model_f"].get("lr_factor", 1.0), patience=self.config["model_f"].get("lr_patience", 30), verbose=True)
            self.optimizer_f_state_dict = self.optimizer_f.state_dict()
        
    def generate_counterexample_data(self):
        self.load_model_states()     
        input_domain, _ = data.generateRandomData(self.config["counterex"].get("N_cex_domain", 1000), self.RANGE, self.dim_in)
        init_domain = input_domain[((input_domain >= torch.tensor(self.init_min)) & (input_domain <= torch.tensor(self.init_max))).all(dim=1)]
        
        if self.config["unsafe"]["shape"] == 'Rectangle':
            # unsafe_domain =  input_domain[((input_domain >= torch.tensor(self.unsafe_min)) & (input_domain <= torch.tensor(self.unsafe_max))).all(dim=1)]        
            unsafe_domain =  input_domain[((input_domain >= self.unsafe_min.clone().detach()) & (input_domain <= self.unsafe_max.clone().detach())).all(dim=1)]        
        elif self.config["unsafe"]["shape"] == 'Circle':
            if isinstance(self.uns_center[0], (int, float)):
                mask = (torch.linalg.norm(input_domain - self.uns_center, dim =1) <= self.uns_rad )
                self.unsafe_domain = input_domain[mask]
            else:
                all_masks = torch.zeros(len(input_domain), dtype=torch.bool)
                for center in self.uns_center:
                    mask = (torch.linalg.norm(input_domain - center, dim =1) <= self.uns_rad )
                    all_masks = all_masks | mask
                unsafe_domain = input_domain[all_masks]
        elif self.config["unsafe"]["shape"] == 'Custom':
            x = input_domain[:,0]
            y = input_domain[:,1]
            result = eval(self.config["unsafe"]["function"])
            unsafe_domain = input_domain[result <= 0]
            
        counterexamples_domain = verification.verify_domain(self.model_v, self.model_b, self.model_f, input_domain, self.config)
        counterexamples_init = verification.verify_init(self.model_b, init_domain, self.config)
        counterexamples_unsafe = verification.verify_unsafe(self.model_b, unsafe_domain, self.config)
        add_data_domain = []
        add_data_init = []
        add_data_unsafe = []
        
        for counterexample in counterexamples_domain:
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
            self.domain = torch.unique(torch.cat([self.domain, add_data_domain], dim=0), dim = 0)
            self.init_domain = self.domain[((self.domain >= torch.tensor(self.init_min)) & (self.domain <= torch.tensor(self.init_max))).all(dim=1)]
            if self.config["unsafe"]["shape"] == 'Rectangle':
                self.unsafe_domain = self.domain[((self.domain >= self.unsafe_min.clone().detach()) & (self.domain <= self.unsafe_max.clone().detach())).all(dim=1)]
            elif self.config["unsafe"]["shape"] == 'Circle':
                if isinstance(self.uns_center[0], (int, float)):
                    self.unsafe_domain = self.domain[(torch.linalg.norm(self.domain - self.uns_center, dim =1) <= self.uns_rad)]                
                else:
                    all_masks = torch.zeros(len(self.domain), dtype=torch.bool)
                    for center in self.uns_center:
                        mask = (torch.linalg.norm(self.domain - center, dim =1) <= self.uns_rad )
                        all_masks = all_masks | mask
                    self.unsafe_domain = self.domain[all_masks]
        
        if add_data_init is not None:
            self.init_domain = torch.unique(torch.cat([self.init_domain, add_data_init], dim=0), dim = 0)
            self.domain = torch.unique(torch.cat([self.domain, add_data_init], dim=0), dim = 0)
        
        if add_data_unsafe is not None:
            self.unsafe_domain = torch.unique(torch.cat([self.unsafe_domain, add_data_unsafe], dim=0), dim =0)
            self.domain = torch.unique(torch.cat([self.domain, add_data_unsafe], dim=0), dim = 0)
                
        elif add_data_domain is None and add_data_init is None and add_data_unsafe is None:
            self.counterexamples_added = False

        #Dataset Generation and Shuffling
        domain_dataset = torch.utils.data.TensorDataset(self.domain)
        init_dataset  = torch.utils.data.TensorDataset(self.init_domain)
        unsafe_dataset  = torch.utils.data.TensorDataset(self.unsafe_domain)

        total_size = len(self.domain) + len(self.init_domain) + len(self.unsafe_domain)

        domain_batch_size = int(len(self.domain) / total_size * self.batch_size)	
        init_batch_size = max(4, int(len(self.init_domain) / total_size * self.batch_size))
        unsafe_batch_size = max(3, int(len(self.unsafe_domain) / total_size * self.batch_size))

        self.domain_loader = torch.utils.data.DataLoader(domain_dataset, batch_size=domain_batch_size, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=self.seed_worker, generator=self.g)
        self.init_loader = torch.utils.data.DataLoader(init_dataset, batch_size=init_batch_size,  shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=self.seed_worker, generator=self.g)
        self.unsafe_loader = torch.utils.data.DataLoader(unsafe_dataset, batch_size=unsafe_batch_size, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=self.seed_worker, generator=self.g)

    def trainInitialDynamics(self):
        model_f_config = self.config["model_f"]
        self.hidden_neurons_f = model_f_config.get("hidden_neurons", 64)
        self.hidden_layers_f = model_f_config.get("layers", 2)
        sigmoid_f = NNModels.assignActivationFunction(model_f_config.get('activation_function', "Tanh"))
        self.hidden_f = [self.hidden_neurons_f] * self.hidden_layers_f
        self.model_f = NNModels.DyanmicsNet(self.dim_in, 
                                            self.hidden_f, 
                                            sigmoid_f).to(self.device)
        best_mse = np.inf  
        best_weights = None
        history = []
        loss_fn = nn.MSELoss()  
        self.optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=model_f_config.get("learning_rate", 1e-2), betas=(0.9, 0.999))
        self.scheduler_f = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_f, mode='min', factor=model_f_config.get("lr_factor", 1.0), patience=model_f_config.get("lr_patience", 30))

        for epoch in tqdm(range(model_f_config.get("epochs_warm", 1000))):
            total_loss = 0
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                self.model_f.train()
                # Calculate the loss
                x_val = X_batch.float().to(self.device)
                y_pred = self.model_f(x_val)
                loss_mse = loss_fn(y_pred, y_batch.float().to(self.device))
                loss = loss_mse 
                # backward pass
                self.optimizer_f.zero_grad()
                loss.backward()
                self.optimizer_f.step()
                total_loss += loss.item()
            with torch.no_grad():
                total_loss = 0
                for batch_idx, (X_batch, y_batch) in enumerate(self.test_loader):
                    y_pred = self.model_f(X_batch.float().to(self.device))
                    total_loss += loss_fn(y_pred, y_batch.float().to(self.device)).item()
                history.append(total_loss)
                if total_loss < best_mse:
                    best_mse = total_loss
                    best_weights = copy.deepcopy(self.model_f.state_dict())
                torch.cuda.empty_cache()
        # restore model and return best accuracy
        self.model_f.load_state_dict(best_weights)
        # Store the model state dictionary
        self.model_f_state_dict = best_weights
        self.optimizer_f_state_dict = self.optimizer_f.state_dict()
        print_info("MSE of Initial Estimate of Dynamical System: %.4f" % best_mse)
    
    def trainCertificate(self):
        if self.config["Barrier"]:
            model_v_config = self.config["model_v"]
            model_b_config = self.config["model_b"]

            hidden_neurons_v = model_v_config.get("hidden_neurons", 64)
            hidden_layers_v = model_v_config.get("layers", 2)
            hidden_v = [hidden_neurons_v] * hidden_layers_v
            self.model_v = NNModels.LyapunovNet(
                n_input=self.dim_in,
                hidden_v=hidden_v,
                sigmoid_v=NNModels.assignActivationFunction(model_v_config.get("activation_function", "eLU"))).to(self.device)

            self.optimizer_v = torch.optim.Adam(
                self.model_v.parameters(),
                lr=model_v_config.get("learning_rate", 1e-2),
                weight_decay=self.config["hyperparameters"].get("reg_v", 0.001))
            
            warmup_scheduler_v = opt.WarmUpLR(
                self.optimizer_v,
                model_v_config.get("warmup", 10),
                model_v_config.get("learning_rate", 1e-2))
            
            self.scheduler_v = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_v,
                mode='min',
                factor=model_v_config.get("lr_factor", 0.2),
                patience=model_v_config.get("lr_patience", 10))

            hidden_neurons_b = model_b_config.get("hidden_neurons", 64)
            hidden_layers_b = model_b_config.get("layers", 2)
            hidden_b = [hidden_neurons_b] * hidden_layers_b
            self.model_b = NNModels.BarrierNet(
                n_input=self.dim_in,
                hidden_b=hidden_b,
                sigmoid_b=NNModels.assignActivationFunction(model_b_config.get("activation_function", "eLU"))).to(self.device)

            self.optimizer_b = torch.optim.Adam(
                self.model_b.parameters(),
                lr=model_b_config.get("learning_rate", 1e-2),
                weight_decay=self.config["hyperparameters"].get("reg_bar", 0.001))
            warmup_scheduler_b = opt.WarmUpLR(
                self.optimizer_b,
                model_b_config.get("warmup", 10),
                model_b_config.get("learning_rate", 1e-2))
            self.scheduler_b = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_b,
                mode='min',
                factor=model_b_config.get("lr_factor", 0.2),
                patience=model_b_config.get("lr_patience", 10))

            # Load the stored model state dictionary if available
            self.load_model_states()        
            # Start Training
            max_iter = self.config["hyperparameters"]["max_iters"]
            for epoch in tqdm(range(max_iter)):
                cert_loss_b = 0
                cert_loss_v = 0
                dyn_loss = 0
                
                for batches in itertools.zip_longest(self.domain_loader, self.train_loader, self.init_loader, self.unsafe_loader, fillvalue=None):
                    if batches[0] is not None:
                        input_domain = batches[0][0].to(self.device)
                        self.optimizer_f.zero_grad()   
                        self.optimizer_v.zero_grad()
                        loss_domain_v, _ = Loss_Functions.loss_function_domain(self.model_v, self.model_b, self.model_f, input_domain, self.config)
                        loss_domain_v.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(self.model_v.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.model_f.parameters(), max_norm=1.0)
                        self.optimizer_v.step()
                        if "train_f_cert" not in self.config["model_f"]:    
                            self.optimizer_f.step()   
                        else:
                            if self.config["model_f"]["train_f_cert"] == False and epoch <= max_iter*self.config["model_f"]["cert_train_factor"]:
                                pass
                            else:
                                self.optimizer_f.step()
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
                        torch.nn.utils.clip_grad_norm_(self.model_f.parameters(), max_norm=1.0)
                        self.optimizer_b.step()
                        if "train_f_cert" not in self.config["model_f"]:    
                            self.optimizer_f.step()   
                        else:
                            if self.config["model_f"]["train_f_cert"] == False and epoch <= max_iter*self.config["model_f"]["cert_train_factor"]:
                                pass
                            else:
                                self.optimizer_f.step()

                    if batches[1] is not None:
                        input_train = batches[1][0].to(self.device)
                        output_train = batches[1][1].to(self.device)
                        self.optimizer_f.zero_grad()
                        loss_train = Loss_Functions.loss_function_dyn(self.model_f, input_train, output_train, self.config)
                        loss_train.backward()
                        torch.nn.utils.clip_grad_norm_(self.model_f.parameters(), max_norm=1.0)
                        self.optimizer_f.step()
                         
                    else:
                        loss_train = torch.tensor(0.0, requires_grad=True)
                    
                    # Calculate average training loss for the epoch
                    cert_loss_b += loss_b
                    cert_loss_v += loss_domain_v
                    dyn_loss += loss_train
                    avg_loss_f = dyn_loss / len(self.train_loader)
                    # Step the scheduler with training loss
                    self.scheduler_f.step(avg_loss_f)
                    # Update learning rate with warm-up
                    if epoch < model_v_config.get("warmup", 10):
                        warmup_scheduler_v.step()
                    else:
                        avg_loss_cert_v = cert_loss_v/len(self.domain_loader)
                        self.scheduler_v.step(avg_loss_cert_v)

                    if epoch < model_b_config.get("warmup", 10):
                        warmup_scheduler_b.step()
                    else:
                        avg_loss_cert_b = cert_loss_b/len(self.domain_loader)
                        self.scheduler_b.step(avg_loss_cert_v)
                        
            # Save the recent versions of model_v and model_b in memory
            self.model_v_state_dict = self.model_v.state_dict()
            self.model_b_state_dict = self.model_b.state_dict()
            self.model_f_state_dict = self.model_f.state_dict()
            self.final_mse_loss = dyn_loss.item()
            self.final_cert_loss = cert_loss_v.item() + cert_loss_b.item()

    def verifyCertificate(self):
        self.load_model_states()     
        input_domain, _ = data.generateRandomData(self.config["counterex"]["N_cex_domain"],self.RANGE, self.dim_in) #the domain is limited to [-1,1] due to normalization
        init_domain = input_domain[((input_domain >= torch.tensor(self.init_min)) & (input_domain <= torch.tensor(self.init_max))).all(dim=1)]
        
        if self.config["unsafe"]["shape"] == 'Rectangle':
            unsafe_domain =  input_domain[((input_domain >= self.unsafe_min.clone().detach()) & (input_domain <= self.unsafe_max.clone().detach())).all(dim=1)]        
        elif self.config["unsafe"]["shape"] == 'Circle':
            if isinstance(self.uns_center[0], (int, float)):
                mask = (torch.linalg.norm(input_domain - self.uns_center, dim =1) <= self.uns_rad )
                unsafe_domain = input_domain[mask]
            else:
                all_masks = torch.zeros(len(input_domain), dtype=torch.bool)
                for center in self.uns_center:
                    mask = (torch.linalg.norm(input_domain - center, dim =1) <= self.uns_rad )
                    all_masks = all_masks | mask
                unsafe_domain = input_domain[all_masks]
        elif self.config["unsafe"]["shape"] == 'Custom':
            x= input_domain[:,0]
            y = input_domain[:,1]
            result = eval(self.config["unsafe"]["function"])
            unsafe_domain = input_domain[result <= 0]
        beta, q = verification.conformal_prediction(self.model_v, self.model_b, self.model_f,input_domain, init_domain, unsafe_domain, self.config)
        if q <= 0:
            self.flag_verified = True
            conf = 1-self.config["verification"].get("epsilon", 0.0001)
            print_info(f"With a confidence of {1-beta}, conditions are valid with satisfaction level {conf}")
        else:
            self.flag_verified = False
            print_warning(f"Verification failed with marginal safety error: {q}")

    def final_model_eval(self):
        self.model_f = self.model_f.to(self.device)
        self.model_f.load_state_dict(self.model_f_state_dict)
        self.model_f.eval()
        self.mse = 0
        total_samples = 0
        for batch_idx, (X_batch, y_batch) in enumerate(self.test_loader):
            y_pred = self.model_f(X_batch.float().to(self.device))
            loss_fn = nn.MSELoss(reduction = 'mean')
            batch_mse = loss_fn(y_pred, y_batch.float().to(self.device))
            self.mse += batch_mse.item() * X_batch.size(0)  # Multiply by batch size to get total loss
            total_samples += X_batch.size(0)  
        self.mse = self.mse / total_samples

    def update_config(self):
        # Construct the init_range value
        file_path = os.path.join(self.par_dir_path, "config_files", self.args.dataset_type, self.name + "_config.json")
        init_range = [[self.init_min[0][i], self.init_max[0][i]] for i in range(self.dim_in)]
        initial_conditions = self.initial_set_center.tolist()

        # Load the existing configuration file
        with open(file_path, 'r') as config_file:
            config = json.load(config_file)

        # Update the 'plotting' key with the init_range
        if "plotting" not in config:
            config["plotting"] = {}
        config["plotting"]["init_range"] = init_range
        config["plotting"]["initial_conditions"] = initial_conditions
        config["plotting"]["dt"] = self.dt

        # Save the updated configuration back to the file
        with open(file_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)

    def save_datasets(self):
        if self.args.dataset_type == 'LASA':
            base_path = os.path.join(self.par_dir_path, 'Datasets', "LASA", self.args.lasa_name)
            os.makedirs(base_path, exist_ok = True)
            torch.save(self.X_train, os.path.join(base_path,"X_train.pt"))
            torch.save(self.y_train, os.path.join(base_path,"y_train.pt"))
            torch.save(self.X_test, os.path.join(base_path,"X_test.pt"))
            torch.save(self.y_test, os.path.join(base_path,"y_test.pt"))

if __name__ == "__main__":
    # Settings Seeds for Reproducibility
    filtered_args = filter_args(sys.argv[1:])
    args = pyrallis.parse(ConfigFile, args=filtered_args)
    if args.dataset_type == 'LASA':
        final_name = args.lasa_name
    elif args.dataset_type == '3D_Shapes':
        final_name = args.name_3d
    elif args.dataset_type == '2D_Shapes':
        final_name = args.name_2d
    seed_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'seeds', args.dataset_type, final_name + '_seed.json')
    #Check if the seed file exists
    try:
       seed = load_seed(seed_filepath)
    except FileNotFoundError:
       seed = random.randint(0, 100)  # seed value
    set_seed(seed)
    mp = MotionPlanner(args)
    print_info("OBTAINING DEMO DATA")
    mp.generate_demo_data()
    print_info("DYNAMICAL SYSTEM TRAINING")
    mp.trainInitialDynamics()
    Plotter.initialDSPlot(mp.model_f, mp.demos, mp.initial_set_center, mp.dim_in, mp.config)
    print_info("OBTAINING TRAINING DATA")
    mp.generate_domain_data()
    iters = 1
    print_info("CERTIFICATE TRAINING")
    mp.trainCertificate()
    trial = 1
    while trial < 1000:
        print_info("ADDING COUNTEREXAMPLES")
        mp.generate_counterexample_data()
        print_info(f"Trial: {trial}")
        if mp.counterexamples_added:
            if trial % 10 == 0:
                if mp.dim_in == 2:
                    fig = Plotter.initialDSPlot(mp.model_f, mp.demos, mp.initial_set_center, mp.dim_in, mp.config, mp.model_b)
                    plt.show()
                    Plotter.plotLyapunov(mp.model_v)
                    Plotter.plotBarrier(mp.model_b)
                elif mp.dim_in == 3:
                    fig = Plotter.final3DDSPlot(mp.model_f, mp.demos, mp.initial_set_center, mp.config)
                    plt.show()
                mp.verifyCertificate()
                if mp.flag_verified:
                    mp.save_all_models()
                    mp.final_model_eval()
                    print_info(f"MSE for test data after certificate training: {mp.mse}")
                    save_seed(seed,seed_filepath)
                    mp.export_onnx()
                    mp.update_config()
                    mp.save_datasets()
                    break    
            mp.trainCertificate()
            trial += 1      
        else:
            import pdb; pdb.set_trace()
            print_info("SAMPLING-BASED VERIFICATION COMPLETE")
            if mp.dim_in == 2:
                fig = Plotter.initialDSPlot(mp.model_f, mp.demos, mp.initial_set_center, mp.dim_in, mp.config, mp.model_b)
                plt.show()
                Plotter.plotLyapunov(mp.model_v)
                Plotter.plotBarrier(mp.model_b)
            elif mp.dim_in == 3:
                fig = Plotter.final3DDSPlot(mp.model_f, mp.demos, mp.initial_set_center, mp.config)
                plt.show()
            mp.verifyCertificate()
            if mp.flag_verified:
                mp.save_all_models()
                mp.final_model_eval()
                print_info(f"MSE for test data after certificate training: {mp.mse}")
                save_seed(seed,seed_filepath)
                mp.export_onnx()
                mp.update_config()
                mp.save_datasets()
                break
    if trial == 1000:
        print_error("MAXIMUM TRIALS EXCEEDED... SAMPLING VERIFICATION FAILED")
     
