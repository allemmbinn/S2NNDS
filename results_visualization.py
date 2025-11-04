from common_header import *
import json
import pandas as pd
import os
import Plotter
from main import MotionPlanner
from NNModels import DyanmicsNet, LyapunovNet, BarrierNet

@dataclass
class ConfigFile:
    lasa_name : str = "NShape"
    dataset_type : str = "LASA" # This can also be 3D_Shapes
    name_3d: str = "Cshape_bottom"
    name_2d: str = "Five_Obstacle_DS"
    real_time: bool = False  # Set to True for real-time plotting
    perturbed: bool = False  # Set to True if you want to use perturbed data

def filter_args(args):
    known_args = ['--lasa_name', '--dataset_type', '--name_3d', '--name_2d', '--real_time', '--perturbed']
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
    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models_verified')
    os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists
    model_path = os.path.join(model_dir, args.dataset_type, model_name) 
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Robot_Data')
    data_path = os.path.join(data_dir, args.dataset_type)
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            model_v_path = os.path.join(model_path,'model_v.pth')
            model_b_path = os.path.join(model_path,'model_b.pth')
            model_f_path = os.path.join(model_path,'model_f.pth')
            model_v = torch.load(model_v_path, map_location=torch.device('cpu'))
            model_b = torch.load(model_b_path, map_location=torch.device('cpu'))
            model_f = torch.load(model_f_path, map_location=torch.device('cpu'))
            try:
                if args.perturbed:
                        robot_data = pd.read_csv(os.path.join(data_path, model_name+'_perturbed.csv'), header=1)
                else:
                        robot_data = pd.read_csv(os.path.join(data_path,model_name+'.csv'), header=1)
            except FileNotFoundError:
                robot_data = None
            return model_name, config, model_v, model_b, model_f, robot_data
            
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' or Models not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file '{config_path}'. {e}")
        return None
    
if __name__ == "__main__":
    # The main function starts here
    filtered_args = filter_args(sys.argv[1:])
    args = pyrallis.parse(ConfigFile, args=filtered_args)
    mp = MotionPlanner(args)
    mp.generate_demo_data()
    model_name, config, model_v, model_b, model_f, robot_data = load_config_models(args)
    initial_set_center = torch.tensor(config['plotting']['initial_conditions'])
    try:
        x_data = robot_data['x'].to_numpy()
        y_data = robot_data['y'].to_numpy()
    except Exception as e:
        x_data = None
        y_data = None
    if mp.dim_in == 2:
        plot = Plotter.lyapunovBarrierPlot(model_v, model_b, model_f, mp.demos, config, x_data, y_data)
        plt.show()
    elif mp.dim_in == 3:
        plot = Plotter.final3DDSPlot(model_f, mp.demos, initial_set_center, config, data_1=robot_data)
        plt.show()
    #Saving the plots
    fig_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', args.dataset_type)
    os.makedirs(fig_dir, exist_ok=True)
    plot.savefig(os.path.join(fig_dir, mp.name + '_main.svg'), format="svg", dpi=300)
    if args.real_time:
        save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'Videos',args.dataset_type)
        step = 10  # Downsampling step for real-time plotting
        x_data = robot_data['x'].to_numpy()[::step]
        y_data = robot_data['y'].to_numpy()[::step]
        if args.perturbed and 'cartesian_contact[0]' in robot_data.columns and 'cartesian_contact[1]' in robot_data.columns:
            x_contact = robot_data['cartesian_contact[0]'].to_numpy()[::step]
            y_contact = robot_data['cartesian_contact[1]'].to_numpy()[::step]
            z_contact = robot_data['cartesian_contact[2]'].to_numpy()[::step]
            assert len(x_data) == len(y_data) == len(x_contact) == len(y_contact), "Data length mismatch"
        fp_s = 1000/step
        if mp.dim_in == 2 and args.perturbed:
            fig, ani = Plotter.realTimePlot(model_v, model_b, model_f, mp.demos, config, x_data, y_data, x_contact, y_contact, z_contact)
            nframes = len(x_data)
            pbar = tqdm(total=nframes, desc="Saving video")
            def progress(i, n):
                pbar.update(i - pbar.n)
            ani.save(
                os.path.join(save_path, model_name + '.mp4'),
                writer='ffmpeg',
                fps=fp_s,
                progress_callback=progress
            )
        elif mp.dim_in == 2 and  not args.perturbed:
            fig, ani = Plotter.realTimePlot(model_v, model_b, model_f, mp.demos, config, x_data, y_data)
            nframes = len(x_data)
            pbar = tqdm(total=nframes, desc="Saving video")
            def progress(i, n):
                pbar.update(i - pbar.n)
            ani.save(
                os.path.join(save_path, model_name + '.mp4'),
                writer='ffmpeg',
                fps=fp_s,
                progress_callback=progress
            )
        if mp.dim_in == 3:
            z_data = robot_data['z'].to_numpy()[::step]
            fig, ani= Plotter.realTimePlot3D(model_f, mp.demos, initial_set_center, config, x_data, y_data, z_data)
            nframes = len(x_data)
            pbar = tqdm(total=nframes, desc="Saving video")
            def progress(i, n):
                pbar.update(i - pbar.n)
            ani.save(
                os.path.join(save_path, model_name + '.mp4'),
                writer='ffmpeg',
                fps=fp_s,
                progress_callback=progress
            )
            pbar.close()
