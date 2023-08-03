import os, time
import torch
import numpy as np
import itk
import torch.optim as optim

import site
site.addsitedir("../lib")

import paths
from utils import get_logger
from signal_reader import read_signal
from main_calculator import MainCalculator
from config import parse_config

def datestr():
    now = time.localtime()
    return '{:04}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
print(datestr())

def main():

    config = parse_config()
    
    device = torch.device("cpu")

    
    # Read and preprocess (brain-region extraction, low-pass filtering) raw signal (size: (slice, row, column, time))
    RawPerfImg, mask, vessels, origin, spacing, direction = \
        read_signal(paths.FileName, paths.MaskName, paths.VesselName, ToTensor = config.to_tensor) 
    # Calculate perfusion parameters
    calculator = MainCalculator(RawPerfImg, mask, vessels, origin, spacing, direction, config, paths.SaveFolder, device)
    calculator.run()

########################################################################################################################

if __name__ == '__main__':
    main()

