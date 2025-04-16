# PyTorch Requirements
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
import torch.nn.utils as utils

import onnx

# Plotting Requirements
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from mpl_toolkits import mplot3d

# Data Processing Requirements
import numpy as np
import pandas as pd

# Import Python Essentials
import os
import json
import time
import math
from math import e
from termcolor import colored
import random
import sys
import argparse
from dataclasses import dataclass
import pyrallis
import scipy
import itertools
from tqdm import tqdm

# Import Model Requirement
import onnx
import copy

# Other Requirements
import scipy.special as sp
from scipy.linalg import solve_continuous_lyapunov

# Datasets
import pyLasaDataset as lasa

# Printing functions
def print_error(message):
    print(colored(message, 'red'))

def print_warning(message):
    print(colored(message, 'yellow'))

def print_info(message):
    print(colored(message, 'blue'))
    
def print_success(message):
    print(colored(message, 'green'))