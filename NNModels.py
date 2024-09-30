from common_header import *

# Load the configuration file
config_file = os.environ.get('CONFIG_FILE', 'config.json')
with open(config_file) as file:
    config = json.load(file)

# Activation Function
def assignActivationFunction(activation_function):
    if activation_function == 'Tanh':
        return nn.Tanh()

# Accessing configuration items
sigmoid_f = assignActivationFunction(config['model_f']['activation_function'])
sigmoid_v = assignActivationFunction(config['model_v']['activation_function'])
sigmoid_b = assignActivationFunction(config['model_b']['activation_function'])

#Class of Neural Network for Dynamics Function
class DyanmicsNet(nn.Module):
    def __init__(self,n_input, hidden_f):
        super(DyanmicsNet, self).__init__()
        self.input_size = n_input
        self.layers_f = nn.ModuleList()
        n_prev = self.input_size
        ## For the Dynamics Function
        for n_hid in hidden_f:
            layer = nn.Linear(n_prev, n_hid)
            nn.init.xavier_uniform_(layer.weight)
            self.layers_f.append(layer)
            n_prev = n_hid
        # Last Layer
        layer = nn.Linear(n_prev, n_input)
        nn.init.xavier_uniform_(layer.weight)
        self.layers_f.append(layer)

    # Forward Propogation
    def forward(self,x):
        # For the Dynamics Function
        y = x
        for idx, layer in enumerate(self.layers_f[:-1]):
            y = sigmoid_f(layer(y))
        Fout = self.layers_f[-1](y)
        return Fout
    
#Class of Neural Network for Lyapunov Function along with Dynamics Function
class LyapunovNet(nn.Module):
    def __init__(self, n_input, hidden_v, hidden_f, model_f):
        super(LyapunovNet, self).__init__()
        self.input_size = n_input
        self.layers_f = nn.ModuleList()
        self.layers_v = nn.ModuleList()
        ## For the Lyapunov Function
        n_prev = self.input_size
        for n_hid in hidden_v:
            layer = nn.Linear(n_prev, n_hid)
            nn.init.xavier_uniform_(layer.weight)
            self.layers_v.append(layer)
            n_prev = n_hid
        # Last Layer
        layer = nn.Linear(n_prev, 1)
        nn.init.xavier_uniform_(layer.weight)
        self.layers_v.append(layer)
        ## For the Dynamics Function
        n_prev = self.input_size
        for idx, ly in enumerate(model_f.layers_f[:-1]):
            layer = nn.Linear(n_prev, hidden_f[idx], bias=True)
            layer.weight = nn.Parameter(ly.weight.data.cpu())
            layer.bias = nn.Parameter(ly.bias.data.cpu())
            layer.bias.requires_grad = True
            self.layers_f.append(layer)
            n_prev = hidden_f[idx]
        # Last Layer
        layer = nn.Linear(n_prev, 1, bias=True)
        layer.weight = nn.Parameter(model_f.layers_f[-1].weight.data.cpu())
        layer.bias = nn.Parameter(model_f.layers_f[-1].bias.data.cpu())
        layer.bias.requires_grad = True
        self.layers_f.append(layer)

    # Forward Propogation
    def forward(self, x):
        # For the Dynamics Function
        y = x
        for idx, layer in enumerate(self.layers_f[:-1]):
            z = layer(y)
            y = sigmoid_f(z)
        Fout = self.layers_f[-1](y)
        # For the Lyapunov Function
        y = x
        for idx, layer in enumerate(self.layers_v[:]):
            z = layer(y)
            y = sigmoid_v(z)
        Vout = y
        return Vout, Fout
    
# General Class of Neural Network for the Barrier Certificate
class BarrierNet(nn.Module):
    def __init__(self, n_input, hidden_b):
        super(BarrierNet, self).__init__()
        self.input_size = n_input
        self.layers_b = nn.ModuleList()
        n_prev = self.input_size
        for n_hid in hidden_b:
            layer = nn.Linear(n_prev, n_hid, bias=True)
            nn.init.xavier_uniform_(layer.weight)
            self.layers_b.append(layer)
            n_prev = n_hid
        # Last Layer
        layer = nn.Linear(n_prev, 1, bias=True)
        nn.init.xavier_uniform_(layer.weight)
        self.layers_b.append(layer)

    # Forward Propogation
    def forward(self, x):
        y = x
        for idx, layer in enumerate(self.layers_b[:]):
            z = layer(y)
            y = sigmoid_b(z)
        return y