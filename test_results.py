from common_header import *
import NNModels
import Plotter
import data as data
from main import MotionPlanner, filter_args, load_seed, set_seed

@dataclass
class ConfigFile:
    lasa_name : str = "Worm"
    dataset_type : str = "LASA" # This can also be 3D_DSOPT
    name_3d: str = "Cshape_bottom"

def load_model(model, optimizer, scheduler, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:  # in case you might not always have a scheduler
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return model, optimizer, scheduler

    
if __name__ == "__main__":
    # Settings Seeds for Reproducibility
    filtered_args = filter_args(sys.argv[1:])
    args = pyrallis.parse(ConfigFile, args=filtered_args)
    if args.dataset_type == '3D_DSOPT':
        seed_filepath = f'seeds/3D_Shapes/{args.name_3d}_seed.json'
    else:
        seed_filepath = f'seeds/LASA/{args.lasa_name}_seed.json'
    #Check if the seed file exists
    try:
       seed = load_seed(seed_filepath)
       set_seed(seed)
    except FileNotFoundError:
        print_error(f"Seed file {seed_filepath} not found. Unable to find models. Returning...")
        sys.exit(1)
    mp = MotionPlanner(args)
    print_info("OBTAINING DEMO DATA")
    mp.generate_demo_data()
    mp.createModels()
    if args.dataset_type == '3D_DSOPT':
        base_path = os.path.join(os.getcwd(), 'models', '3D_Shapes',args.name_3d)
    else:
        base_path = os.path.join(os.getcwd(), 'models', 'LASA',args.lasa_name)
    # Check if the folder exists
    if not os.path.exists(base_path):
        print_error(f"Folder {base_path} does not exist. Returning...")
        sys.exit(1)
    torch.load(os.path.join(base_path, 'model_f.pth'))
    torch.load(os.path.join(base_path, 'model_v.pth'))
    torch.load(os.path.join(base_path, 'model_b.pth'))
    mp.final_model_eval()
    print_info(f"MSE for test data after certificate training: {mp.mse}")
    if args.dataset_type == '3D_DSOPT':
        Plotter.initial3DDSPlot(mp.model_f, mp.demos/mp.pos_scaling, mp.initial_set_center)
    elif args.dataset_type == 'LASA':
        Plotter.plotFinalDS(mp.model_f, mp.X_train, mp.initial_set_center.squeeze(), mp.dt)
    Plotter.plotLyapunov(mp.model_v)
    Plotter.plotBarrier(mp.model_b)
