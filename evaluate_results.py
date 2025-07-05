from common_header import *
import Plotter

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
    config_path = os.path.join(parent_dir, "config_files", "LASA", f"{model_name}_config.json")
    model_path = os.path.join(parent_dir, "models_verified", "LASA", f"{model_name}")
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
        
if __name__ == "__main__":
    filtered_args = filter_args(sys.argv[1:])
    args = pyrallis.parse(ConfigFile, args=filtered_args)
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    model_name = args.lasa_name
    # Obtain the models for S2-NNDS
    config, model_v, model_b, model_f = load_config_models(model_name)
    # Get the datasets
    dataset_path = os.path.join(parent_dir, 'Datasets', 'LASA', f"{model_name}")
    X_test_tensor = torch.load(os.path.join(dataset_path, "X_test.pt"))
    y_test_tensor = torch.load(os.path.join(dataset_path, "y_test.pt"))
    # For S2-NNDS
    model_f.eval()
    model_f = model_f.to('cpu')    
    all_errors = []
    with torch.no_grad():
        for X, y in zip(X_test_tensor, y_test_tensor):
            pred = model_f(X.float())
            error = (pred - y.float()).cpu().numpy()
            error_norm = np.linalg.norm(error)  # L2 error for this sample
            all_errors.append(error_norm)
    all_errors = np.array(all_errors)
    mse = np.mean(all_errors ** 2)
    sd = np.std(all_errors)
    print_info(f"For S2-NNDS model: {model_name}")
    print_success(f"S2-NNDS MSE: {mse:.6f}")
    print_success(f"S2-NNDS Standard Deviation: {sd:.6f}")

    
    
    