from common_header import *
import pandas as pd
import json
import os
import Plotter

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
    
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    model_name = "Worm"  # Replace with the results of the dataset you want to plot
    config, model_v, model_b, model_f = load_config_models(model_name)
    path = os.path.join(parent_dir, 'Datasets', 'LASA', model_name + '_benchmark')
    X_train = torch.load(os.path.join(path, "X_train.pt"))
    plot = Plotter.benchmarkPlot(model_v, model_b, model_f, X_train, config)
    #Saving the plots
    plot.savefig(os.path.join(parent_dir, 'results', 'LASA', model_name + '_benchmark.svg'), format="svg", dpi=300)

