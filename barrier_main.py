from common_header import *
import NNModels
import data
import Loss_Functions
import Plotter
import smt_verification
import main
from dreal import *

@dataclass
class ConfigFile:
    lasa_name : str = "Worm"
    dataset_type : str = "LASA"
    
if __name__ == "__main__":
    args = pyrallis.parse(ConfigFile)
    mp = main.MotionPlanner(args)
    mp.generateData()
    check_barrier = mp.config["Barrier"]
    if check_barrier:
        print_info("Loading the Verified Lyapunov Function")
        hidden_neurons_f = mp.config["model_f"]["hidden_neurons"]
        hidden_layers_f = mp.config["model_f"]["layers"]
        hidden_f = [hidden_neurons_f] * hidden_layers_f
        hidden_neurons_v = mp.config["model_v"]["hidden_neurons"]
        hidden_layers_v = mp.config["model_v"]["layers"]
        hidden_v = [hidden_neurons_v] * hidden_layers_v
        model_f = NNModels.DyanmicsNet(mp.dim_in,hidden_f).to(mp.device)
        model_v = NNModels.LyapunovNet(
                    n_input=mp.dim_in,
                    hidden_v=hidden_v,
                    hidden_f=hidden_f,
                    model_f=model_f).to(mp.device)
        path_model_v = os.path.join("./models/", args.lasa_name + "_model_v.pth")
        model_v.load_state_dict(torch.load(path_model_v))
        mp.model_v = model_v
        _ = mp.trainBarrierCertificate()
        Plotter.lyapunovBarrierPlot(mp.model_v, mp.X_train, mp.initial_set_center, mp.config, mp.model_b)
    Plotter.lyapunovBarrierPlot(mp.model_v, mp.X_train, mp.initial_set_center, mp.config)
