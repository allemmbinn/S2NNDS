from common_header import *

class SineActivation(nn.Module):
    def __init__(self):
        super(SineActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


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
    elif activation_function == 'Sine':
        return SineActivation()
    
# def spec_norm(layer, norm_value):
#     layer = spectral_norm(layer)
#     with torch.no_grad():
#         weight = layer.weight
#         u, s, v = torch.svd(weight)
#         max_singular_value = s[0]
#         scale_factor = 1 #norm_value / max_singular_value
#         layer.weight.data = weight * scale_factor
#     return layer

#Class of Neural Network for Dynamics Function
class DyanmicsNet(nn.Module):
    def __init__(self,n_input, hidden_f, sigmoid_f):
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
    

#Neural Network for Lyapunov Function
class LyapunovNet(nn.Module): 
    def __init__(self, n_input, hidden_v, sigmoid_v):
        super(LyapunovNet, self).__init__()
        self.input_size = n_input
        self.layers_v = nn.ModuleList()
        self.output_act_v= nn.Tanh()

        ## For the Lyapunov Function
        n_prev = self.input_size
        for n_hid in hidden_v:
            layer = nn.Linear(n_prev, n_hid, bias = False)
            nn.init.xavier_uniform_(layer.weight)
            self.layers_v.append(layer)
            n_prev = n_hid
        # Last Layer
        layer = nn.Linear(n_prev, 1, bias = False)
        nn.init.xavier_uniform_(layer.weight)
        self.layers_v.append(layer)
        self.sigmoid_v = sigmoid_v


    # Forward Propogation
    def forward(self, x):
        # For the Lyapunov Function
        y = x
        for idx, layer in enumerate(self.layers_v[:-1]):
            z = layer(y)
            y = self.sigmoid_v(z)
        Vout = self.layers_v[-1](y)
        return Vout
    
#Neural Network for Barrier Function
class BarrierNet(nn.Module): 
    def __init__(self, n_input, hidden_b, sigmoid_b):
        super(BarrierNet, self).__init__()
        self.input_size = n_input
        self.layers_b = nn.ModuleList()
        self.output_act_b= nn.Tanh()
        n_prev = self.input_size
        self.sigmoid_b = sigmoid_b
        if isinstance(self.sigmoid_b, nn.Softplus):
            self.shift = torch.log(torch.tensor(2.0))
        else:
            self.shift = 0

        for n_hid in hidden_b:
            layer = nn.Linear(n_prev, n_hid)
            nn.init.xavier_uniform_(layer.weight)
            self.layers_b.append(layer)
            n_prev = n_hid
        # Last Layer
        layer = nn.Linear(n_prev, 1)
        nn.init.xavier_uniform_(layer.weight)
        self.layers_b.append(layer)

    # Forward Propogation
    def forward(self, x):
        # For the Barrier Function
        y = x
        for idx, layer in enumerate(self.layers_b[:-1]):
            z = layer(y)
            y = self.sigmoid_b(z)
        Bout = self.layers_b[-1](y)
        return Bout
    