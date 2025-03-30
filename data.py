from common_header import *
from torch.utils.data import Dataset

def generateRectangularData(N, RANGE):
    return torch.cat([torch.Tensor(N, 1).uniform_(RANGE[i][0],RANGE[i][1]) for i in range(len(RANGE))],1)    
    
def generateCircularData(N, r, centre):
    border_batch = int(N / 10)
    internal_batch = N - border_batch
    angle = (2 * np.pi) * torch.rand(internal_batch, 1)
    radius = r * torch.rand(internal_batch, 1)
    x_coord = radius * np.cos(angle)
    y_coord = radius * np.sin(angle)
    offset = torch.cat([x_coord, y_coord], dim=1)
    angle = (2 * np.pi) * torch.rand(border_batch, 1)
    x_coord = r * np.cos(angle)
    y_coord = r * np.sin(angle)
    offset_border = torch.cat([x_coord, y_coord], dim=1)
    offset = torch.cat([offset, offset_border])
    return torch.tensor(centre) + offset



def generateGridData(N, RANGE):
    y = torch.linspace(RANGE[1][0], RANGE[1][1], steps=N)
    x = torch.linspace(RANGE[0][0], RANGE[0][1], steps=N)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    # Convert X and Y to torch tensors
    # X_tensor = torch.tensor(X, dtype=torch.float32)
    # Y_tensor = torch.tensor(Y, dtype=torch.float32)
    X_tensor = X.float()
    Y_tensor = Y.float()
    input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2)
    return input_data

class MultiVariableDataset(Dataset):
    def __init__(self, x_init, x_domain, x_unsafe):
        self.var1 = x_init
        self.var2 = x_domain
        self.var3 = x_unsafe
        self.length = len(x_init)  

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return a single sample as a tuple
        return self.var1[idx], self.var2[idx], self.var3[idx]
    
def collate_fn(batch):
    var1_batch = torch.stack([item[0] for item in batch])
    var2_batch = torch.stack([item[1] for item in batch])
    var3_batch = torch.stack([item[2] for item in batch])
    return var1_batch, var2_batch, var3_batch