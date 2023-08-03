import os
import torch
import logging
import numpy as np
from scipy import signal
import itk
from builtins import object
from scipy import linalg
import ParamsCalculator.cbv as cbv
import ParamsCalculator.cbf as cbf

def cal(cbv, cbf, mask):
  mtt = np.divide(cbv, cbf)
  return mtt
