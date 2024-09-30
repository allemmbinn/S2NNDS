from common_header import *

# Load the configuration file
config_file = os.environ.get('CONFIG_FILE', 'config.json')
with open(config_file) as file:
    config = json.load(file)

def generateReferenceData():
    dataset_type = config['dataset']['type']
    dataset_name = config['dataset']['name']
    core_path = config['dataset']['path']

    if dataset_type == 'LASA':
        # Training Data
        mat = scipy.io.loadmat(core_path + dataset_name + '_train.mat')
        datat = mat['Data']
        datat = np.transpose(datat)
        # Getting the Position and Velocity Seperately
        X_train = datat[:,:2]
        y_train = datat[:,2:]
        # Testing Data
        mat = scipy.io.loadmat(core_path + dataset_name + '_test.mat')
        data = mat['Data']
        data = np.transpose(data)
        # Getting the Position and Velocity Seperately
        X_test = data[:,:2]
        y_test = data[:,2:]
    else:
        print_error("Non-LASA Dataset has been choosen")

    ### NORMALISE THE TRAJECTORIES to [-1, 1]
    pos_scaling = np.max(np.linalg.norm(X_train, axis=1))
    vel_scaling = np.max(np.linalg.norm(y_train, axis=1))
    X_train /= pos_scaling
    X_test /= pos_scaling
    y_train /= vel_scaling
    y_test /= vel_scaling

    return X_train, y_train, X_test, y_test

# Generate Data for domain, init or unsafe sets
def generateData(N, setName, centre=None):
    shape = config[setName]["shape"]
    if shape == "Rectangle":
        dim_in = config["dim_in"]
        device = config["device"]
        if centre is None:
            RANGE = config[setName]["range"]
        else:
            r = config[setName]["radius"]
            RANGE = [[centre[i] - r, centre[i] + r ]for i in range(dim_in)]
        x_data = torch.cat([torch.Tensor(N, 1).uniform_(RANGE[i][0],RANGE[i][1]).to(device) for i in range(dim_in)],1)
        return x_data
    elif shape == "Circle":
        r = config[setName]["radius"]
        if centre is None:
            centre = config[setName]["centre"]
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

