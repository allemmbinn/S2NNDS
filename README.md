# Neural Network Dynamical System (NNDS)

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

Official implementation of neural network dynamical systems with stability certificates, developed at MIRMI, Technical University of Munich. This repository contains code for learning stable dynamical systems from demonstrations using Lyapunov and barrier certificates.

## Installation

### System Requirements
- Ubuntu 20.04/22.04 or Windows 10/11 (WSL2 recommended)
- NVIDIA GPU with CUDA 11.8+ (optional but recommended)

### Conda Environment Setup




## Install

### Create Conda Environment
 ```sh
conda create --name <your_env_name> python=3.7

conda activate <your_env_name>
 ```

### Install Dependencies
``` sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install termcolor scipy matplotlib onnx numpy pandas pyrallis pyLasaDataset tqdm 
```

### Install this repo
```sh
git clone
```
## Datasets

1. **LASA Dataset**: This dataset is obtained directly from [pyLasaDataset](https://github.com/justagist/pyLasaDataset)

    We have tested and stored the models for the following datasets:
    
    - Angle
    - CShape
    - DoubleBendedLine
    - GShape
    - heee
    - Leaf_2
    - NShape
    - Sine
    - Sshape
    - Worm
    - WShape

2. **2D Dataset**: This 2D-dataset is robot trajectory directly obtained from robot demonstration of a Franka Emika Panda. This dataset has the following type which has been tested:
    - Five_Obstacle_DS
3. **3D Dataset**: This 3D-dataset is obtained directly from [ds-opt](https://github.com/nbfigueroa/ds-opt) 

    We have tested for the following datasets: 
    
    - 3D_CShape_bottom
    - 3D_CShape_top
    - 3D_sink


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

### Validating the Results

Use the arguments for `--dataset_type` and the name of the dataset as per shown above. Run the following command

```bash
python results_visualisation.py 
```

---

**CoRL Submission Notes**  
- Double-blind compliant: No author-identifying information in code  
- Video supplement: [Project Page](https://yourprojectpage.com)  
- Reviewers: See `config_files/` for hyperparameter details and `results_visualisation.py` for reproduction instructions
