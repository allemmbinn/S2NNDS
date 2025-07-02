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
    # Obtain the models for S2-NNDS
    config, model_v, model_b, model_f = load_config_models(model_name)
    # Obtain the polynomials of ABC-DS
    abc_result_path = os.path.join(parent_dir, 'abc_ds_config', f"{model_name}_result_config.json")
    # Get the datasets
    dataset_path = os.path.join(parent_dir, 'Datasets', 'LASA', f"{model_name}_benchmark")
    X_test_tensor = torch.load(os.path.join(dataset_path, "X_test.pt"))
    y_test_tensor = torch.load(os.path.join(dataset_path, "y_test.pt"))
    X_test = X_test_tensor.numpy()
    y_test = y_test_tensor.numpy()
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
    # For ABC-DS
    f1_str, f2_str = abc_data["f_fh_str_arr"]
    fx_poly, fy_poly = map(compile_poly, (f1_str, f2_str))
    abc_mse_test = 0
    total_samples = 0
    for X_batch, y_batch in zip(X_test, y_test):
        y_pred = np.array([fx_poly(X_batch[0], X_batch[1]), fy_poly(X_batch[0], X_batch[1])])
        abc_mse_test += np.sum((y_pred - y_batch) ** 2)
        total_samples += 1
    abc_mse_test /= total_samples
    
    abc_mse_train = 0
    total_samples = 0
    for X_batch, y_batch in zip(X_train, y_train):
        y_pred = np.array([fx_poly(X_batch[0], X_batch[1]), fy_poly(X_batch[0], X_batch[1])])
        abc_mse_train += np.sum((y_pred - y_batch) ** 2)
        total_samples += 1
    abc_mse_train /= total_samples
    # Print the results
    # print_success(f"NNDS MSE Train: {nnds_mse_train:.6f}")
    # print_success(f"NNDS MSE Test: {nnds_mse_test:.6f}")
    # print_success(f"ABC-DS MSE Train: {abc_mse_train:.6f}")
    # print_success(f"ABC-DS MSE Test: {abc_mse_test:.6f}")
    
    model_f.eval()
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
    print_success(f"S2-NNDS MSE: {mse:.6f}")
    print_success(f"S2-NNDS Standard Deviation: {sd:.6f}")
    
    abc_mse_test = 0
    total_samples = 0
    all_errors = []
    for X_batch, y_batch in zip(X_test, y_test):
        y_pred = np.array([fx_poly(X_batch[0], X_batch[1]), fy_poly(X_batch[0], X_batch[1])])
        error_norm = np.linalg.norm(y_pred - y_batch)
        all_errors.append(error_norm)
        # abc_mse_test += np.sum((y_pred - y_batch) ** 2)
        # total_samples += 1
    # abc_mse_test /= total_samples
    all_errors = np.array(all_errors)
    mse = np.mean(all_errors ** 2)
    sd = np.std(all_errors)
    print_success(f"ABC-DS MSE: {mse:.6f}")
    print_success(f"ABC-DS Standard Deviation: {sd:.6f}")
