import os
import torch
import logging
import numpy as np
from builtins import object
import itk


def brain_region(RawPerf, device, background = 0):
	
	mask = torch.zeros(RawPerf.size(), device = device)
	mask = torch.where(RawPerf != background, torch.tensor(1), mask)

	return mask
