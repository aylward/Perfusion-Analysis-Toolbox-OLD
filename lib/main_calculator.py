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
import ParamsCalculator.aif as aif

class MainCalculator:
    """Main calculator.
    Args:
    raw_perf: numpy array or torch tensor of size (slice, row, column, time)
    save_path: path/to/save/folder
    device: device currently working on
    logger: info logger
    """
    def __init__(self, raw_perf, mask, origin, spacing, direction, config, save_path, device, logger = None):
        if logger is None:
            self.logger = utils.get_logger('MainCalculator', level = logging.DEBUG)
        else:
            self.logger = logger
        self.logger.info(f"Sending the raw perfusion image to '{device}'")
        self.raw_perf = raw_perf.astype(int)
        self.spacing = spacing
        self.config    = config
        self.itkinfo  = [origin, spacing, direction, save_path]
        self.device    = device
        self.nS        = self.raw_perf.shape[0]
        self.nR        = self.raw_perf.shape[1]
        self.nC        = self.raw_perf.shape[2]
        self.nT        = self.raw_perf.shape[3]
        self.size      = [self.nS, self.nR, self.nC, self.nT]
        self.mask = mask

    def run(self):
        self.main_cal()


    def main_cal(self):
        CTC = ctc.DSC_mri_conc(self.raw_perf, self.mask) # dtype = torch.float
        AIF = aif.DSC_mri_aif(self.mask, CTC, self.size, self.config, self.device)
        # CBV = cbv.cal(self.mask, CTC, self.size, AIF, self.config, self.device)
        # CBF = cbf.cal(CTC, AIF, self.mask, self.size)

         
