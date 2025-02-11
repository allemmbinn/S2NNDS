from common_header import *

# Activation Function
def assignActivationFunction(activation_function):
    if activation_function == 'Tanh':
        return nn.Tanh()
    elif activation_function == 'ReLU':
        return nn.ReLU()

#Class of Neural Network for Dynamics Function
class DyanmicsNet(nn.Module):
    def __init__(self,n_input, hidden_f, sigmoid_f=nn.Tanh()):
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
        self.sigmoid_f = sigmoid_f
        
    # Forward Propogation
    def forward(self,x):
        # For the Dynamics Function
        y = x
        for idx, layer in enumerate(self.layers_f[:-1]):
            y = self.sigmoid_f(layer(y))
        Fout = nn.Tanh(self.layers_f[-1](y))
        return Fout
    
