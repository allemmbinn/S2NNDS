from common_header import *

def generateRandomData(N, RANGE):
    x = (RANGE[0][1] - RANGE[0][0])*torch.rand(N) + RANGE[0][0]
    y = (RANGE[1][1] - RANGE[1][0])*torch.rand(N) + RANGE[1][0]
    input_data = torch.stack([x, y], dim=1)  
    return input_data, N