from common_header import *
import json
import os
import Plotter
from main import MotionPlanner

@dataclass
class ConfigFile:
    lasa_name : str = "CShape"
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
    config_path = os.path.join(config_dir, args.dataset_type, f"{model_name}_config.json")
    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists
    model_path = os.path.join(model_dir, args.dataset_type, model_name) 
       
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

# The main function starts here
filtered_args = filter_args(sys.argv[1:])
args = pyrallis.parse(ConfigFile, args=filtered_args)
mp = MotionPlanner(args)
mp.generate_demo_data()
config, model_v, model_b, model_f = load_config_models(args)
# name_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),"robot_demonstrations",mp.dataset_type,"Recording_"+mp.name+".csv")
# data_1 = pd.read_csv(name_file, header=1)
# plot = Plotter.finalDSPlot(model_f, model_v, model_b, mp.demos, mp.initial_set_center, mp.dim_in, config, data_1)
initial_set_center = torch.vstack([mp.initial_set_center, torch.tensor(mp.demos[3].pos[:, 0])])
if mp.dim_in == 2:
    plot = Plotter.lyapunovBarrierPlot(model_v, model_b, model_f, mp.demos, config)
    plt.show()
    # Plotter.plotLyapunov(mp.model_v)
    # Plotter.plotBarrier(mp.model_b)
elif mp.dim_in == 3:
    plot = Plotter.final3DDSPlot(model_f, mp.demos, initial_set_center, config)
    plt.show()
#Saving the plots
fig_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', args.dataset_type)
os.makedirs(fig_dir, exist_ok=True)
plot.savefig(os.path.join(fig_dir, mp.name + '_main.svg'), format="svg", dpi=300)



