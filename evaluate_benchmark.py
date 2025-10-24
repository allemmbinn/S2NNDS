from common_header import *
import Plotter
import main
from scipy.spatial.distance import cdist

@dataclass
class ConfigFile:
    lasa_name : str = "Sine"
    dataset_type : str = "LASA"  

def filter_args(args):
    known_args = ['--lasa_name', '--dataset_type']
    return [arg for arg in args if any(arg.startswith(known) for known in known_args)]

def load_config_models(model_name):
    # Construct the path to the configuration file
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(parent_dir, "config_files", "LASA", f"{model_name}_config_benchmark.json")
    model_path = os.path.join(parent_dir, "models", "LASA", f"{model_name}_benchmark")
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            model_v_path = os.path.join(model_path,'model_v.pth')
            model_b_path = os.path.join(model_path,'model_b.pth')
            model_f_path = os.path.join(model_path,'model_f.pth')
            model_v = torch.load(model_v_path)
            model_b = torch.load(model_b_path)
            model_f = torch.load(model_f_path)
        return config, model_v, model_b, model_f
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' or Models not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file '{config_path}'. {e}")
        return None
    
def compile_poly(expr: str):
    expr = expr.replace("^", "**")      
    expr = expr.replace("xi1", "x")
    expr = expr.replace("xi2", "y")
    code = compile(expr, "<expr>", "eval")
    return lambda x, y: eval(code, {"x": x, "y": y, "np": np})
    
if __name__ == "__main__":
    filtered_args = filter_args(sys.argv[1:])
    args = pyrallis.parse(ConfigFile, args=filtered_args)
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    model_name = args.lasa_name
    seed_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'seeds', args.dataset_type, model_name + '_seed.json')
    #Check if the seed file exists
    try:
       seed = main.load_seed(seed_filepath)
    except FileNotFoundError:
       seed = random.randint(0, 100)  # seed value
    main.set_seed(seed)
    # Get the dt time and test demo points
    mp = main.MotionPlanner(args)
    mp.generate_demo_data()
    # For Initi Set Centers of Test Trajectories
    train_size = int(5/7 * mp.total_demos)
    train_indices = random.sample(range(mp.total_demos), train_size)
    test_indices = list(set(range(mp.total_demos)) - set(train_indices))
    initial_set_center_test = np.array([mp.demos[i].pos[:,0]/np.array(mp.pos_scaling) for i in test_indices])
    # Obtain the models for S2-NNDS
    config, model_v, model_b, model_f = load_config_models(model_name)
    model_v = model_v.to('cpu')
    model_b = model_b.to('cpu')
    model_f = model_f.to('cpu')
    # Obtain the polynomials of ABC-DS
    abc_result_path = os.path.join(parent_dir, 'abc_ds_config', f"{model_name}_result_config.json")
    # Get the datasets
    dataset_path = os.path.join(parent_dir, 'Datasets', 'LASA', f"{model_name}_benchmark")
    X_test_tensor = torch.load(os.path.join(dataset_path, "X_test.pt"))
    y_test_tensor = torch.load(os.path.join(dataset_path, "y_test.pt"))
    X_test = X_test_tensor.cpu().numpy()
    y_test = y_test_tensor.cpu().numpy()
    X_train_tensor = torch.load(os.path.join(dataset_path, "X_train.pt"))
    y_train_tensor = torch.load(os.path.join(dataset_path, "y_train.pt"))
    X_train = X_train_tensor.numpy()
    y_train = y_train_tensor.numpy()
    try:
        with open(abc_result_path, 'r') as result_file:
            abc_data = json.load(result_file)
    except FileNotFoundError:
        print(f"Error: Result file '{abc_result_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file '{abc_result_path}'. {e}")
        sys.exit(1)
    """
        For S2NNDS
    """
    model_f.eval()
    model_f = model_f.to('cpu')    
    all_errors = []
    with torch.no_grad():
        for X, y in zip(X_test_tensor, y_test_tensor):
            pred = model_f(X.cpu().float()).cpu()
            error = (pred - y.cpu().float()).numpy()
            error_norm = np.linalg.norm(error)  # L2 error for this sample
            all_errors.append(error_norm)
    all_errors = np.array(all_errors)
    mse = np.mean(all_errors ** 2)
    sd = np.std(all_errors)
    print_success(f"S2-NNDS MSE: {mse:.6f}")
    print_success(f"S2-NNDS Standard Deviation: {sd:.6f}")
    # FOR DTW Distance
    dtw_dist = []
    for i in range(len(test_indices)):
        start_pos = initial_set_center_test[i]
        demo_traj = mp.demos[test_indices[i]].pos / np.array(mp.pos_scaling)
        s2nnds_traj = np.zeros_like(demo_traj)
        s2nnds_traj[:,0] = start_pos
        for k in range(1, demo_traj.shape[1]):
            s2nnds_traj[:,k] = s2nnds_traj[:,k-1] + mp.dt * model_f(torch.tensor(s2nnds_traj[:,k-1], dtype=torch.float32)).detach().cpu().numpy()
        dist_matrix = cdist(demo_traj.T, s2nnds_traj.T, metric='euclidean')
        rssd = np.sqrt(np.sum(np.min(np.square(dist_matrix),axis=1)))
        dtw_dist.append(rssd) 
    print_success(f"S2-NNDS DTW Distance: {np.mean(np.array(dtw_dist)):.6f}")
    # Area of Barrier Function
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    grid_resolution = 500
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T 
    with torch.no_grad():
        inputs = torch.tensor(grid_points, dtype=torch.float32)
        outputs = model_b(inputs).detach().cpu().numpy().flatten()
    region_mask = outputs < 0
    # Calculate area per grid cell
    delta_x = (x_max - x_min) / (grid_resolution - 1)
    delta_y = (y_max - y_min) / (grid_resolution - 1)
    area_per_cell = delta_x * delta_y
    area = np.sum(region_mask) * area_per_cell
    print_success(f"S2-NNDS Safe Region Area: {area:.6f}")
    """
        For ABC-DS
    """
    f1_str, f2_str = abc_data["f_fh_str_arr"]
    fx_poly, fy_poly = map(compile_poly, (f1_str, f2_str))
    abc_mse_test = 0
    total_samples = 0
    all_errors = []
    for X_batch, y_batch in zip(X_test, y_test):
        y_pred = np.array([fx_poly(X_batch[0], X_batch[1]), fy_poly(X_batch[0], X_batch[1])])
        error_norm = np.linalg.norm(y_pred - y_batch)
        all_errors.append(error_norm)
    all_errors = np.array(all_errors)
    mse = np.mean(all_errors ** 2)
    sd = np.std(all_errors)
    print_success(f"ABC-DS MSE: {mse:.6f}")
    print_success(f"ABC-DS Standard Deviation: {sd:.6f}")
    # FOR DTW Distance
    dtw_dist = []
    for i in range(len(test_indices)):
        start_pos = initial_set_center_test[i]
        demo_traj = mp.demos[test_indices[i]].pos / np.array(mp.pos_scaling)
        abcds_traj = np.zeros_like(demo_traj)
        abcds_traj[:,0] = start_pos
        for k in range(1, demo_traj.shape[1]):
            abcds_traj[0,k] = abcds_traj[0,k-1] + mp.dt * fx_poly(abcds_traj[0,k-1], abcds_traj[1,k-1])
            abcds_traj[1,k] = abcds_traj[1,k-1] + mp.dt * fy_poly(abcds_traj[0,k-1], abcds_traj[1,k-1])
        dist_matrix = cdist(demo_traj.T, abcds_traj.T, metric='euclidean')
        rssd = np.sqrt(np.sum(np.min(np.square(dist_matrix),axis=1)))
        dtw_dist.append(rssd) 
    print_success(f"ABC-DS DTW Distance: {np.mean(np.array(dtw_dist)):.6f}")
    # Area of Barrier Function
    b_str = abc_data["B_fh_str_arr"]
    b_poly = compile_poly(b_str)
    inputs = np.array(grid_points)
    b_values = b_poly(inputs[:,0], inputs[:,1])
    region_mask = (b_values < 0)
    area = np.sum(region_mask) * area_per_cell
    print_success(f"ABC-DS Safe Region Area: {area:.6f}")
