from common_header import *
import json
import os
import Plotter
from main import MotionPlanner

@dataclass
class ConfigFile:
    lasa_name : str = "NShape"
    dataset_type : str = "LASA" # This can also be 3D_Shapes
    name_3d: str = "Cshape_bottom"
    name_2d: str = "Five_Obstacle_DS"

def filter_args(args):
    known_args = ['--lasa_name', '--dataset_type', '--name_3d', '--name_2d']
    return [arg for arg in args if any(arg.startswith(known) for known in known_args)]

def load_config_models(args):
    # Get the name of the dataset
    if args.dataset_type == "LASA":
        model_name = args.lasa_name
    elif args.dataset_type == "3D_Shapes":
        model_name = args.name_3d
    else:
        model_name = args.name_2d
    # Construct the path to the configuration file
    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config_files')
    config_path = os.path.join(config_dir, args.dataset_name, f"{model_name}_config.json")

    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models_verified')
    os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists
    model_path = os.path.join(model_dir, args.dataset_name, model_name)        
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

# The main function starts here
mp = MotionPlanner()
mp.generate_demo_data()
filtered_args = filter_args(sys.argv[1:])
args = pyrallis.parse(ConfigFile, args=filtered_args)
config, model_v, model_b, model_f = load_config_models(args)
# TODO : Change this
plot = Plotter.lyapunovBarrierPlot(model_v, model_b, model_f, X_train, config)

#Saving the plots
plot.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)),'results', model_name + '_main.png'), format="png", dpi=300)  # Save as PNG with high resolution



