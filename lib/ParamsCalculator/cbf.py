import numpy as np
from scipy.integrate import trapz
import os
import torch
import logging
from scipy import signal
import itk
from builtins import object
from scipy import linalg
from ParamsCalculator.svd import DSC_SVD

def cal(conc, aif, mask):
    cbf, tmax = DSC_SVD(conc, aif['gv'], mask)
    return cbf, tmax
