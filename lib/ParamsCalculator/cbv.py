import numpy as np
from scipy.integrate import trapz
import os
import torch
import logging
from scipy import signal
import itk
from builtins import object
from scipy import linalg

def cal(mask, conc, size, aif, config, device):
    kh = 1
    rho = 1
    TR = 30
    nT = 16
    time = np.arange(0, nT * TR, TR)
    cbv = np.zeros((size[0], size[1], size[2], size[3]))
    for s in range(size[0]):
          cbv[s, :, :, :] = (kh/ rho) * mask[s, :, :, :] * (trapz(conc[s, :, :, :], axis=1) / trapz((aif['time']), (aif['conc'])))

    return cbv
