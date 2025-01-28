from common_header import *
import NNModels
import Plotter
import data as data

# Load the configuration file
config_file = os.environ.get('CONFIG_FILE', 'config.json')
with open(config_file) as file:
    config = json.load(file)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#For defining the layers
hidden_neurons_f = config["model_f"]["hidden_neurons"]
hidden_layers_f = config["model_f"]["layers"]
hidden_f = [hidden_neurons_f] * hidden_layers_f
hidden_neurons_v = config["model_v"]["hidden_neurons"]
hidden_layers_v = config["model_v"]["layers"]
hidden_v = [hidden_neurons_v] * hidden_layers_v
dim_in   = config["dim_in"]
model_f = NNModels.DyanmicsNet(dim_in,hidden_f).to(device)
model_v = NNModels.LyapunovNet(
            n_input=dim_in,
            hidden_v=hidden_v,
            hidden_f=hidden_f,
            model_f=model_f
)
dataset_name = config["dataset"]["name"]
path_model_v = "./models/" + dataset_name + "_model_v.pth"
#Getting the reference trajectories
X_train, y_train, X_test, y_test = data.generateReferenceData()
mean_point = [0,0]
N = config["dataset"]["datashape"]
n = int(X_train.shape[0]/N)
for i in range(n):
    mean_point += X_train[(i-1)*N]
mean_point /= n
# Importing the Model Path
model_v.load_state_dict(torch.load(path_model_v))

Plotter.lyapunovBarrierPlot(model_v, X_train, mean_point)
