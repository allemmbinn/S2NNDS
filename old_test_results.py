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
).to(device)
dataset_name = config["dataset"]["name"]
path_model_v = "./models/" + dataset_name + "_model_v.pth"
#Getting the reference trajectories
X_train, y_train, X_test, y_test = data.generateReferenceData()
# Convert the data to torch tensors
X_train = torch.Tensor(X_train).to(device)
X_test = torch.Tensor(X_test).to(device)
y_train = torch.Tensor(y_train).to(device)
y_test = torch.Tensor(y_test).to(device)
N = config["dataset"]["datashape"]
n_train = int(X_train.shape[0]/N)
n_test = int(X_test.shape[0]/N)
# Computing the mean error
MSE_train = np.zeros(n_train)
MSE_test = np.zeros(n_test)
for i in range(n_train):
    MSE_train[i] = torch.norm(model_f(X_train[i]).detach() - y_train[i]).item() 
for i in range(n_test):
    MSE_test[i]  = torch.norm(model_f(X_test[i]) - y_test[i]).item()

mean_train = np.mean(MSE_train)
std_dev_train = np.std(MSE_train)
mean_test = np.mean(MSE_test)
std_dev_test = np.std(MSE_test)
print(f"Mean Squared Error for Training Data: {mean_train} and std deviation: {std_dev_train}")
print(f"Mean Squared Error for Test Data: {mean_test} and std deviation: {std_dev_test}")
# for i in range(len(X_train)):
#     MSE_train += torch.norm(model_f(X_train[i]) - y_train[i]).item()
# # Importing the Model Path
# model_v.load_state_dict(torch.load(path_model_v))

# Plotter.lyapunovBarrierPlot(model_v, X_train, mean_point)
