from common_header import *

def generateGridData(N, RANGE, dim_in=2):
    y = torch.linspace(RANGE[1][0], RANGE[1][1], steps=N+1)
    epsilon_y = (RANGE[1][1]-RANGE[1][0])/N #Discretization parameter
    y = y[:-1] + epsilon_y/2
    x = torch.linspace(RANGE[0][0], RANGE[0][1], steps=N+1)
    epsilon_x = (RANGE[0][1]-RANGE[0][0])/N #Discretization parameter
    x = x[:-1] + epsilon_x/2
    if dim_in == 2:
        # Create 2D meshgrid
        X, Y = torch.meshgrid(x, y, indexing='ij')
        # Convert to float tensors
        X_tensor = X.float()
        Y_tensor = Y.float()
        # Stack into input data points
        input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2)
        eps = np.sqrt(epsilon_x**2 + epsilon_y**2) / 2
    elif dim_in == 3:
        # Create 3D meshgrid
        z = torch.linspace(RANGE[2][0], RANGE[2][1], steps=N+1)
        epsilon_z = (RANGE[2][1]-RANGE[2][0])/N
        z = z[:-1] + epsilon_z/2
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        # Convert to float tensors
        X_tensor = X.float()
        Y_tensor = Y.float()
        Z_tensor = Z.float()
        # Stack into input data points
        input_data = torch.stack((X_tensor, Y_tensor, Z_tensor), dim=-1).reshape(-1, 3)
        eps = np.sqrt(epsilon_x**2 + epsilon_y**2 + epsilon_z**2) / 2
    return input_data, eps

# def generateRandomData(N, RANGE, dim_in=2):
#     uniform_dist_y = torch.distributions.Uniform(RANGE[1][0], RANGE[1][1])
#     y = uniform_dist_y.sample((N,))
#     uniform_dist_x = torch.distributions.Uniform(RANGE[0][0], RANGE[0][1])
#     x = uniform_dist_x.sample((N,))
#     if dim_in == 2:
#         # Create 2D meshgrid
#         X, Y = torch.meshgrid(x, y, indexing='ij')
#         # Convert to float tensors
#         X_tensor = X.float()
#         Y_tensor = Y.float()
#         # Stack into input data points
#         input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2)
#     elif dim_in == 3:
#         # Create 3D meshgrid
#         z = torch.distributions.Uniform(RANGE[2][0], RANGE[2][1]).sample((N,))
#         # Convert to float tensors
#         X_tensor = x.float()
#         Y_tensor = y.float()
#         Z_tensor = z.float()
#         # Stack into input data points
#         input_data = torch.stack((X_tensor, Y_tensor, Z_tensor), dim=-1).reshape(-1, 3)
#     return input_data, N

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

