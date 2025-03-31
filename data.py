from common_header import *

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
