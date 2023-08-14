import logging
import itk

import utils
import ParamsCalculator.ctc as ctc
import ParamsCalculator.cbv as cbv
import ParamsCalculator.cbf as cbf
import ParamsCalculator.aif as aif
import ParamsCalculator.mtt as mtt


class MainCalculator:
    """Main calculator.
    Args:
    raw_perf: numpy array or torch tensor of size (slice, row, column, time)
    save_path: path/to/save/folder
    device: device currently working on
    logger: info logger
    """

    def __init__(
        self,
        raw_perf,
        mask,
        vessels,
        origin,
        spacing,
        direction,
        config,
        save_path,
        device,
        logger=None,
    ):
        if logger is None:
            self.logger = utils.get_logger(
                "MainCalculator", level=logging.DEBUG
            )
        else:
            self.logger = logger
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
        mtt_img = itk.image_from_array(MTT)
        itk.imwrite(mtt_img, "mtt.nii")
        cbv_img = itk.image_from_array(CBV)
        itk.imwrite(cbv_img, "cbv.nii")
        cbf_img = itk.image_from_array(CBF)
        itk.imwrite(cbf_img, "cbf.nii")
        ctc_img = itk.image_from_array(CTC)
        itk.imwrite(ctc_img, "ctc.nii")
