import numpy as np
from scipy.integrate import trapz
import os
import torch
import logging
from scipy import signal
import itk
from builtins import object
from scipy import linalg
from ParamsCalculator.svd import DSC_mri_SVD

def cal(conc, aif, mask, size):
    cbf = DSC_mri_SVD(conc, aif['conc'], mask, size)
    return cbf
