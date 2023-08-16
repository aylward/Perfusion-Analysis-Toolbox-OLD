#!/usr/bin/env python3
from typing import Tuple

import numpy as np
import numpy.typing as npt
import itk


def read_signal(
    FileName: str, MaskName: str, VesselName: str
) -> Tuple[
    npt.ArrayLike,
    npt.ArrayLike,
    npt.ArrayLike,
    itk.Point,
    itk.Vector,
    itk.Matrix,
]:
    """
    Load computed tomography perfusion (CTP) image input from disk.
    """
    return read_ctp(FileName, MaskName, VesselName)


def read_ctp(FileName, MaskName, VesselName, ToTensor=True):
    img = itk.imread(FileName)
    img_arr = itk.GetArrayFromImage(img)
    img_arr = np.transpose(img_arr, (1, 2, 3, 0))
    nan_mask = np.isnan(img_arr)
    img_arr_nan = np.where(nan_mask is True)
    img_arr[img_arr_nan] = 0
    img = itk.GetImageFromArray(img_arr)
    mask = itk.imread(MaskName)
    mask_arr = itk.GetArrayFromImage(mask)
    mask = itk.GetImageFromArray(mask_arr)
    vessel = itk.imread(VesselName)
    vessel_arr = itk.GetArrayFromImage(vessel)
    return (
        img_arr,
        mask_arr,
        vessel_arr,
        img.GetOrigin(),
        img.GetSpacing(),
        img.GetDirection(),
    )
