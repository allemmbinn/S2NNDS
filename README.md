# Neural-Network-Dynamical-System

The official repository for the Neural Network Dynamical System based on my work at MIRMI, TUM under the guidance of Dr Abdalla Swikir.

## Using Config File

1. Tune the Hyperparameters for the various models in the function

## How to use

1. run `python main.py` for running the `main.py`. By default, we use LASA 2D dataset with Worm Shape being used.

    Other options:    
    a. `python main.py --lasa_name=Worm --dataset_type=LASA` : This is for LASA shapes which have been shown below.
    
    b. `python main.py --dsopt_name=CShape --dataset_type=3D_DSOPT` : This is for the 3D dataset from DSOPT. 

2. The Parameters available for LASA Datatype has been included in the `config_files` folder

## Datasets

1. **LASA Dataset**: This dataset is obtained directly from [pyLasaDataset](https://github.com/justagist/pyLasaDataset)

    Run the following command to install the datasets.
    ```
    python3 -m pip install pylasadataset 
    ```

    The following datasets are available:
    
    - Angle
    - CShape
    - DoubleBendedLine
    - GShape
    - Leaf_2
    - SShape
    - Sine
    - WShape

2. **3D Dataset**: This 3D-dataset was used in DS-OPT. The files for this can be imported by running the following script
    ```
    ./dsopt_dataset.sh
    ```
    The following datasets are available:
    
    - 3D_CShape_bottom
    - 3D_CShape_top
    - 3D_sink
    - 3D_viapoint_1
    - 3D_viapoint_2
    - 3D_viapoint_3
    - 3D-cube-pick
    - 3D-pick-box