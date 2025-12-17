from common_header import *

def generateRandomData(N, RANGE, dim_in=2):
    if dim_in == 2:
        x = (RANGE[0][1] - RANGE[0][0])*torch.rand(N) + RANGE[0][0]
        y = (RANGE[1][1] - RANGE[1][0])*torch.rand(N) + RANGE[1][0]
        input_data = torch.stack([x, y], dim=1)
    elif dim_in == 3:
        x = (RANGE[0][1] - RANGE[0][0])*torch.rand(N) + RANGE[0][0]
        y = (RANGE[1][1] - RANGE[1][0])*torch.rand(N) + RANGE[1][0]
        z = (RANGE[2][1] - RANGE[2][0])*torch.rand(N) + RANGE[2][0]
        input_data = torch.stack([x, y, z], dim=1)
    return input_data, N

# Trajectory Data Class
class TrajectoryData:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel