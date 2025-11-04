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
    model_path = os.path.join(parent_dir, "models_verified", "LASA", f"{model_name}_benchmark")
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            model_v_path = os.path.join(model_path,'model_v.pth')
            model_b_path = os.path.join(model_path,'model_b.pth')
            model_f_path = os.path.join(model_path,'model_f.pth')
            model_v = torch.load(model_v_path, map_location=torch.device('cpu'))
            model_b = torch.load(model_b_path, map_location=torch.device('cpu'))
            model_f = torch.load(model_f_path, map_location=torch.device('cpu'))
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_v = model_v.cpu().to(device)
    model_b = model_b.cpu().to(device)
    model_f = model_f.cpu().to(device)
    # Obtain the polynomials of ABC-DS
    abc_result_path = os.path.join(parent_dir, 'abc_ds_config', 'benchmark', f"{model_name}_result_config.json")
    try:
        with open(abc_result_path, 'r') as result_file:
            abc_data = json.load(result_file)
    except FileNotFoundError:
        print(f"Error: Result file '{abc_result_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file '{abc_result_path}'. {e}")
        sys.exit(1)    
    plot = Plotter.combinedBenchmarkPlot(abc_data, model_b, model_f, config, model_name)
    plt.show()
    #Saving the plots
    plot.savefig(os.path.join(parent_dir, 'results', 'LASA', model_name + '_combined_benchmark.svg'), format="svg", dpi=300)

