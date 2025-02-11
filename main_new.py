from common_header import *
import NNModels_new as NNModels
import data_new as data
import Loss_Functions
import Plotter
import smt_verification
from dreal import *


@dataclass
class ConfigFile:
    lasa_name : str = "Worm"
    dataset_type : str = "LASA"

def filter_args(args):
    known_args = ['--lasa_name', '--dataset_type']
    return [arg for arg in args if any(arg.startswith(known) for known in known_args)]
        
class MotionPlanner:
    def __init__(self, args):
        self.args = args
        # Load the configuration file
        file_path = "./config_files/" + self.args.lasa_name + "_config2.json"
        with open(file_path) as file:
            self.config = json.load(file)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
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
            self.demos = dataset.demos
            # Divide the data into training and testing
            self.total_demos = len(self.demos)
            self.dim_in = self.demos[0].pos.shape[0]
            train_size = int(6/7 * self.total_demos) # 5/7 datasets are used for training
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
            train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
            test_dataset = torch.utils.data.TensorDataset(self.X_test, self.y_test)
            batch_size = self.config["model_f"]["batch_size"]
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            print_error("Non-LASA Dataset has been choosen")
        # Normalise the Trajectories to [-1, 1] #Use the maximum value to normalize and scale the data.
        self.pos_scaling = torch.max(torch.concatenate([abs(self.X_train), abs(self.X_test)]))
        self.vel_scaling = torch.max(torch.concatenate([abs(self.y_train), abs(self.y_test)]))
        self.X_train /= self.pos_scaling
        self.X_test /= self.pos_scaling
        self.y_train /= self.vel_scaling
        self.y_test /= self.vel_scaling


        self.initial_set_center = np.mean([self.demos[i].pos[:,0] for i in range(self.total_demos)], axis=0)/self.pos_scaling

    def generate_domain_data(self):
        self.N_domain = self.config["domain"]["N"] #The number of samples we want in the domain region
        self.RANGE = self.config["domain"]["range"] #The number of samples we want in the domain region
        self.domain = data.generateGridData(self.N_domain, self.RANGE) #the domain is limited to [-1,1] due to normalization

        if self.config["Barrier"]:
            #Generate data for initial set
            init_min = (np.min([self.demos[i].pos[:,0] for i in range(self.total_demos)], axis=0)/self.pos_scaling - self.config["init"]["radius"]).reshape(1,2)
            init_min = np.where(init_min < -1, -1, init_min)
            init_max = (np.max([self.demos[i].pos[:,0] for i in range(self.total_demos)], axis=0)/self.pos_scaling + self.config["init"]["radius"]).reshape(1,2)
            init_max = np.where(init_max > 1, 1, init_max)

            self.N_init = self.config["init"]["N"]
            self.init_range = ((np.concatenate((init_min, init_max), axis = 0)).transpose()).tolist()
            self.init_domain = data.generateGridData(self.N_init, self.init_range)

            #Generate data for unsafe set
            self.N_unsafe = self.config["unsafe"]["N"]
            self.unsafe = self.config["unsafe"]["range"]
            self.unsafe_domain = data.generateGridData(self.N_unsafe, self.unsafe)

            #TODO:Shuffle the data

    def trainInitialDynamics(self):
        self.hidden_neurons_f = self.config["model_f"]["hidden_neurons"]
        self.hidden_layers_f = self.config["model_f"]["layers"]
        sigmoid_f = NNModels.DynamicsNet.actFun(self.config['model_f']['activation_function'])
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







        
    

if __name__ == "__main__":
# Settings Seeds for Reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    filtered_args = filter_args(sys.argv[1:])
    args = pyrallis.parse(ConfigFile, args=filtered_args)
    #args = pyrallis.parse(ConfigFile)
    mp = MotionPlanner(args)
    mp = MotionPlanner(args)
    print_info("OBTAINING DEMO DATA")
    mp.generate_demo_data()
    print_info("DYNAMICAL SYSTEM TRAINING")
    mp.trainInitialDynamics()
    Plotter.initialDSPlot(mp.model_f, mp.X_train, mp.initial_set_center)
    print_info("OBTAINING TRAINING DATA")
    mp.generate_domain_data()
    print_info("CERTIFICATE TRAINING")




    