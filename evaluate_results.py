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
    X_train_tensor = torch.load(os.path.join(dataset_path, "X_train.pt"))
    y_train_tensor = torch.load(os.path.join(dataset_path, "y_train.pt"))
    # For S2-NNDS
    model_f = model_f.to('cpu')
    model_f.eval()
    nnds_mse_test = 0
    total_samples = 0
    loss_fn = nn.MSELoss(reduction = 'sum')
    for X_batch, y_batch in zip(X_test_tensor, y_test_tensor):
        y_pred = model_f(X_batch.float())
        batch_mse = loss_fn(y_pred, y_batch.float())
        nnds_mse_test += batch_mse.item() * X_batch.size(0)  # Multiply by batch size to get total loss
        total_samples += X_batch.size(0)  
    nnds_mse_test /= total_samples
    
    nnds_mse_train = 0
    total_samples = 0
    loss_fn = nn.MSELoss(reduction = 'sum')
    for X_batch, y_batch in zip(X_train_tensor, y_train_tensor):
        y_pred = model_f(X_batch.float())
        batch_mse = loss_fn(y_pred, y_batch.float())
        nnds_mse_train += batch_mse.item() * X_batch.size(0)  # Multiply by batch size to get total loss
        total_samples += X_batch.size(0)  
    nnds_mse_train /= total_samples
    # Print the results
    print_success(f"NNDS MSE Train: {nnds_mse_train:.6f}")
    print_success(f"NNDS MSE Test: {nnds_mse_test:.6f}")
    
    
    