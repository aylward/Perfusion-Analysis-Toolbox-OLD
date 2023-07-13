import os
import torch
import logging
import numpy as np
import itk
from tensorboardX import SummaryWriter

import utils
import ParamsCalculator.ctc as ctc
import ParamsCalculator.cbv as cbv
import ParamsCalculator.cbf as cbf
import ParamsCalculator.mask as mask
import ParamsCalculator.aif as aif


class MainCalculator:
    """Main calculator.
    Args:
    raw_perf: numpy array or torch tensor of size (slice, row, column, time)
    save_path: path/to/save/folder
    device: device currently working on
    logger: info logger
    """
    def __init__(self, raw_perf, origin, spacing, direction, config, save_path, device, logger = None):
        if logger is None:
            self.logger = utils.get_logger('MainCalculator', level = logging.DEBUG)
        else:
            self.logger = logger
        self.logger.info(f"Sending the raw perfusion image to '{device}'")
        self.raw_perf = raw_perf
        self.spacing = spacing
        self.config    = config
        self.itkinfo  = [origin, spacing, direction, save_path]
        self.device    = device
        self.nS        = self.raw_perf.size(0)
        self.nR        = self.raw_perf.size(1)
        self.nC        = self.raw_perf.size(2)
        self.nT        = self.raw_perf.size(3)
        self.size      = [self.nS, self.nR, self.nC, self.nT]


    def run(self):
        self.main_cal()


    def main_cal(self):
        mask_for_aif = mask.brain_region(self.raw_perf, self.device)
        print(self.itkinfo[1])
        print(self.spacing)
        CTC = ctc.cal(self.raw_perf, self.itkinfo, self.config, self.device) # dtype = torch.float
        AIF = aif.DSC_mri_aif(mask_for_aif, CTC, self.size, self.config, self.device)
        CBV = cbv.cal(mask_for_aif, CTC, self.size, AIF, self.config, self.device)
        CBF = cbf.cal(CTC, AIF, mask_for_aif, self.size)
        print("This is the CTC:", CTC)
        print("This is the AIF: ", AIF)
        print("This is the CBV: ", CBV)
        print("This is the CBF: ", CBF)

         
