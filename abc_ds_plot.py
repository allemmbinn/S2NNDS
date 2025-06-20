from common_header import *
from main import MotionPlanner
import Plotter

@dataclass
class ConfigFile:
    lasa_name : str = "Sine"
    dataset_type : str = "LASA"  # This can also be 3D_Shapes

def filter_args(args):
    known_args = ['--lasa_name', '--dataset_type']
    return [arg for arg in args if any(arg.startswith(known) for known in known_args)]

filtered_args = filter_args(sys.argv[1:])
args = pyrallis.parse(ConfigFile, args=filtered_args)
mp = MotionPlanner(args)
mp.generate_demo_data()
# Saving the plots for ABC-DS
abc_result_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'abc_ds_config', f"{args.lasa_name}_result_config.json")
config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config_files', 'LASA',  f"{args.lasa_name}_config_benchmark.json")

try:
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print(f"Error: Configuration file '{config_path}' not found.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Failed to parse JSON file '{config_path}'. {e}")
    sys.exit(1)
try:
    with open(abc_result_path, 'r') as result_file:
        result = json.load(result_file)
except FileNotFoundError:
    print(f"Error: Result file '{abc_result_path}' not found.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Failed to parse JSON file '{abc_result_path}'. {e}")
    sys.exit(1)    

plot = Plotter.abcdsPlot(result, mp.demos, config, args.lasa_name)
plt.show()
#Saving the plots
fig_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'ABC-DS', args.dataset_type)
os.makedirs(fig_dir, exist_ok=True)
plot.savefig(os.path.join(fig_dir, mp.name + '_main.svg'), format="svg", dpi=300)
