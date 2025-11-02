# Safe and Stable Neural Network Dynamical Systems for Robot Motion Planning

Official implementation of Safe and Stable Neural Network Dynamical Systems for Robot Motion
Planning with stability and safety certificates.

## Installation

### System Requirements
- Ubuntu 20.04/22.04 or Windows 10/11
- NVIDIA GPU with CUDA 11.8+ (optional but recommended)


## Install

### Create Conda Environment
 ```sh
conda create --name <your_env_name> python=3.8

conda activate <your_env_name>
 ```

### Install Dependencies
``` sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install termcolor scipy matplotlib onnx numpy pandas pyrallis pyLasaDataset tqdm 
```

## Datasets

1. **LASA Dataset**: This dataset is obtained directly from [pyLasaDataset](https://github.com/justagist/pyLasaDataset)

    We have tested and stored the models for the following datasets:
    
    - Angle
    - CShape
    - GShape
    - NShape
    - Sine
    - Sshape
    - Worm
    - PShape

2. **2D Dataset**: This 2D-dataset is robot trajectory directly obtained from robot demonstration of a Franka Emika Panda. This dataset has the following type which has been tested:
    - Five_Obstacle_DS
3. **3D Dataset**: This 3D-dataset is obtained directly from [ds-opt](https://github.com/nbfigueroa/ds-opt) 

    We have tested for the following datasets: 
    - 3D_CShape_bottom

## Quick Start

### Training Models

#### LASA Dataset
```bash
python main.py --dataset_type=LASA 
```

#### Robot Demonstration [2D Dataset]
```bash
python main.py --dataset_type=2DShapes 
```

#### 3D Dataset
```bash
python main.py --dataset_type=3DShapes
```


## Running the Code

### Visualising the Plots

The plots of the dynamical systems are stored in the `results` folder.

Use the arguments for `--dataset_type` and the name of the dataset as per shown above. Run the following command

```bash
python results_visualisation.py 
```

The models for these visulisation is found in the `models_verified` folder. Plese edit these models carefully. 

### Training new models
For training the models for the dynamical system and the certificates we can run the following command.

1. To use the **LASA Dataset**, run the following command:
    ```bash 
    python main.py --dataset_type=lasa --lasa_name=<name_from_dataset>
    ```

2. To use the **2D Dataset**, run the following command:
    ```bash
    python main.py --dataset_type=2D_Shapes --name_2d=<name_from_dataset>
    ```

3. To use the **3D Dataset**, run the following command:
    ```bash
    python main.py --dataset_type=3D_Shapes --name_3d=<name_from_dataset>
    ```

The hyperparameters for the config files can be changed by changing the parameters in the `config_files/`<dataset_type>`/`<dataset_name>`_config.json`

The verified models are stored in the `models` folder. We also have the `models_onnx` folder to store the onnx models for the dynamical system to be implemented on a Franka Panda robot arm.

### Validating the Results

For obtaining the MSE and SD for the errors, you can run the following command with the arguments corresponding to the dataset

```bash
python evaluate_results.py
```

## Benchmarking Results
The benchmarking has been performed for the followign LASA Datasets:
- Worm
- PShape
- Sshape
- Sine
### ABC-DS Polynomials
We use the best results reported in the [code](https://github.com/martinschonger/abc-ds). The config file containing these polynomials are stored in the folder `abc_ds_config`.

### Validating the Results

For obtaining the MSE and SD for the errors, you can run the following command with the arguments corresponding to the dataset

```bash
python evaluate_benchmark.py
```

### Plotting the Results

The plots of the trajectories of $S^2NN-DS$  and $ABC-DS$ are stored in the `results` folder.

To plot theses results, run the following code with the required arguments:
```bash
python benchmark_plot.py
```
