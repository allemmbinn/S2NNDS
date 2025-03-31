from common_header import *

# Activation Function
def assignActivationFunction(activation_function):
    if activation_function == 'Tanh':
        return nn.Tanh()
    elif activation_function == 'ReLU':
        return nn.ReLU()
    elif activation_function == 'eLU':
        return nn.ELU()    
    elif activation_function == 'softplus':
        return nn.Softplus()

#Class of Neural Network for Dynamics Function
class DyanmicsNet(nn.Module):
    def __init__(self,n_input, hidden_f, sigmoid_f=nn.Tanh()):
        super(DyanmicsNet, self).__init__()
        self.input_size = n_input
        self.layers_f = nn.ModuleList()
        n_prev = self.input_size
        ## For the Dynamics Function
        for n_hid in hidden_f:
            layer = nn.Linear(n_prev, n_hid, bias = False)
            nn.init.xavier_uniform_(layer.weight)
            self.layers_f.append(layer)
            n_prev = n_hid
        # Last Layer
        layer = nn.Linear(n_prev, n_input, bias = False)
        nn.init.xavier_uniform_(layer.weight)
        self.layers_f.append(layer)
        self.sigmoid_f = sigmoid_f
        self.output_act_f= nn.Tanh()

        
    # Forward Propogation
    def forward(self,x):
        # For the Dynamics Function
        y = x
        for idx, layer in enumerate(self.layers_f[:-1]):
            y = self.sigmoid_f(layer(y))
        Fout = self.output_act_f(self.layers_f[-1](y))
        return Fout
    
#Class of Neural Network for Lyapunov Function along with Dynamics Function
class LyapunovNet(nn.Module):
    def __init__(self, n_input, hidden_v, hidden_f, model_f, sigmoid_v=nn.Tanh(), sigmoid_f=nn.Tanh()):
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
            self.layers_f.append(layer)
            n_prev = hidden_f[idx]
        # Last Layer
        layer = nn.Linear(n_prev, 1, bias=True)
        layer.weight = nn.Parameter(model_f.layers_f[-1].weight.data.cpu())
        self.layers_f.append(layer)
        self.sigmoid_v = sigmoid_v
        self.sigmoid_f = sigmoid_f
        self.output_act_f= nn.Tanh()

    # Forward Propogation
    def forward(self, x):
        # For the Dynamics Function
        y = x
        for idx, layer in enumerate(self.layers_f[:-1]):
            z = layer(y)
            y = self.sigmoid_f(z)
        Fout = self.output_act_f(self.layers_f[-1](y))
        # For the Lyapunov Function
        y = x
        for idx, layer in enumerate(self.layers_v[:-1]):
            z = layer(y)
            y = self.sigmoid_v(z)
        Vout = self.layers_v[-1](y)
        return Vout, Fout
    
# General Class of Neural Network for the Barrier Certificate
class BarrierNet(nn.Module):
    def __init__(self, n_input, hidden_b, sigmoid_b=nn.Tanh()):
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
        self.sigmoid_b = sigmoid_b

    # Forward Propogation
    def forward(self, x):
        y = x
        for idx, layer in enumerate(self.layers_b[:-1]):
            z = layer(y)
            y = self.sigmoid_b(z)
        Bout = self.layers_b[-1](y) #Are you 
        return Bout