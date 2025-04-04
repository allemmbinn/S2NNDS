from common_header import *
from torch.utils.data import Dataset

def generateGridData(N, RANGE):
    y = torch.linspace(RANGE[1][0], RANGE[1][1], steps=N+1)
    epsilon_y = (RANGE[1][1]-RANGE[1][0])/N #Discretization parameter
    y = y[:-1] + epsilon_y/2
    #y +=  (RANGE[1][1]-RANGE[1][0])/N
    #y = torch.cat((torch.tensor([RANGE[1][0]]),y))
    x = torch.linspace(RANGE[0][0], RANGE[0][1], steps=N+1)
    epsilon_x = (RANGE[0][1]-RANGE[0][0])/N #Discretization parameter
    x = x[:-1] + epsilon_x/2
    #x +=  (RANGE[0][1]-RANGE[0][0])/N
    #x = torch.cat((torch.tensor([RANGE[0][0]]),x))
    X, Y = torch.meshgrid(x, y, indexing='ij')
    # Convert X and Y to torch tensors
    # X_tensor = torch.tensor(X, dtype=torch.float32)
    # Y_tensor = torch.tensor(Y, dtype=torch.float32)
    X_tensor = X.float()
    Y_tensor = Y.float()
    input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2)
    eps = np.sqrt(epsilon_x**2 + epsilon_y**2) / 2
    return input_data, eps

def generateRandomData(N, RANGE):
    uniform_dist_y = torch.distributions.Uniform(RANGE[1][0], RANGE[1][1])
    y = uniform_dist_y.sample((N,))
    uniform_dist_x = torch.distributions.Uniform(RANGE[0][0], RANGE[0][1])
    x = uniform_dist_x.sample((N,))
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X_tensor = X.float()
    Y_tensor = Y.float()
    input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2)
    return input_data, N

