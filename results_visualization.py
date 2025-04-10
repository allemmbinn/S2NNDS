from common_header import *
import json
import os
import Plotter

def load_config_models(model_name):
    # Construct the path to the configuration file
    config_dir = "config_files"  # Directory where configuration files are stored
    config_path = os.path.join(config_dir, f"{model_name}_config2.json")

    model_dir = "models"
    model_path = os.path.join(model_dir, f"{model_name}")

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

model_name = "Worm"  # Replace with the results of the dataset you want to plot
config, model_v, model_b, model_f = load_config_models(model_name)
path = os.path.join('Datasets_2D', model_name)
X_train = torch.load(os.path.join(path, "X_train.pt"))
plot = Plotter.lyapunovBarrierPlot(model_v, model_b, model_f, X_train, config)

#Saving the plots
plot.savefig(os.path.join('results', model_name + '_main.png'), format="png", dpi=300)  # Save as PNG with high resolution



