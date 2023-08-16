#!/usr/bin/env python3

import argparse
import logging

import itk
import numpy.typing as npt
import torch

from . import utils
from .ParamsCalculator import ctc as ctc
from .ParamsCalculator import cbv as cbv
from .ParamsCalculator import cbf as cbf
from .ParamsCalculator import aif as aif
from .ParamsCalculator import mtt as mtt


class MainCalculator:
    """Main calculator."""

    def __init__(
        self,
        raw_perf: npt.ArrayLike,
        mask: npt.ArrayLike,
        vessels: npt.ArrayLike,
        origin: itk.Point,
        spacing: itk.Vector,
        direction: itk.Matrix,
        config: argparse.Namespace,
        save_path: str,
        device: torch.device,
        logger: logging.Logger = None,
    ):
        if raw_perf.ndim != 4:
            raise ValueError(
                "Expected input numpy array or torch tensor"
                "of size (slice, row, column, time)"
            )

        self.logger = (
            logger
            if logger
            else utils.get_logger("MainCalculator", level=logging.DEBUG)
        )
        self.logger.info(f"Sending the raw perfusion image to '{device}'")

        self.raw_perf = raw_perf.astype(int)
        self.spacing = spacing
        self.config = config
        self.itkinfo = [origin, spacing, direction, save_path]
        self.device = device
        self.nS = self.raw_perf.shape[0]
        self.nR = self.raw_perf.shape[1]
        self.nC = self.raw_perf.shape[2]
        self.nT = self.raw_perf.shape[3]
        self.size = [self.nS, self.nR, self.nC, self.nT]
        self.mask = mask
        self.savepath = save_path
        self.vessels = vessels

    def run(self):
        self.main_cal()

    def main_cal(self):
        mask = self.mask
        temp_mask = mask.copy()
        temp_mask1 = mask.copy()
        temp_mask2 = mask.copy()
        temp_mask3 = mask.copy()
        temp_mask4 = mask.copy()
        vessels = self.vessels
        CTC, BOLUS = ctc.DSC_conc(self.raw_perf, temp_mask)
        AIF = aif.DSC_aif(
            temp_mask1,
            CTC,
            BOLUS,
            self.size,
            self.config,
            self.device,
            vessels,
        )
        CBV = cbv.cal(temp_mask2, CTC, AIF, self.config, self.device)
        CBF, TMAX = cbf.cal(CTC, AIF, temp_mask3)
        MTT = mtt.cal(CBV, CBF, temp_mask4)

        mtt_img = itk.image_view_from_array(MTT)
        itk.imwrite(mtt_img, f"{self.savepath}/mtt.nii.gz")
        cbv_img = itk.image_view_from_array(CBV)
        itk.imwrite(cbv_img, f"{self.savepath}/cbv.nii.gz")
        cbf_img = itk.image_view_from_array(CBF)
        itk.imwrite(cbf_img, f"{self.savepath}/cbf.nii.gz")
        ctc_img = itk.image_view_from_array(CTC)
        itk.imwrite(ctc_img, f"{self.savepath}/ctc.nii.gz")
