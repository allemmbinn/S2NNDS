from common_header import *
from torch.utils.data import Dataset

def generateGridData(N, RANGE):
    y = torch.linspace(RANGE[1][0], RANGE[1][1], steps=N)
    y = y[:-1]
    y +=  (RANGE[1][1]-RANGE[1][0])/N
    y = torch.cat((torch.tensor([RANGE[1][0]]),y))
    x = torch.linspace(RANGE[0][0], RANGE[0][1], steps=N)
    x = x[:-1]
    x +=  (RANGE[0][1]-RANGE[0][0])/N
    x = torch.cat((torch.tensor([RANGE[0][0]]),x))
    X, Y = torch.meshgrid(x, y, indexing='ij')
    # Convert X and Y to torch tensors
    # X_tensor = torch.tensor(X, dtype=torch.float32)
    # Y_tensor = torch.tensor(Y, dtype=torch.float32)
    X_tensor = X.float()
    Y_tensor = Y.float()
    input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2)
    return input_data
