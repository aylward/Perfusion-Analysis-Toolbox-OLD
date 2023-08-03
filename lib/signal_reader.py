import os
import torch 
import numpy as np
import itk
import scipy.ndimage as ndimage

from utils import cutoff_percentile

def read_signal(FileName, MaskName, VesselName, ToTensor = True):
        return read_ctp(FileName, MaskName, VesselName, ToTensor)


def read_ctp(FileName, MaskName, VesselName, ToTensor = True):

    img = itk.imread(FileName)
    img_arr = itk.GetArrayFromImage(img)
    img_arr = np.transpose(img_arr, (1, 2, 3, 0))
    nan_mask = np.isnan(img_arr)
    img_arr_nan = np.where(nan_mask==True)
    img_arr[img_arr_nan] = 0
    img = itk.GetImageFromArray(img_arr)
    mask = itk.imread(MaskName)
    mask_arr = itk.GetArrayFromImage(mask)
    mask = itk.GetImageFromArray(mask_arr)
    vessel = itk.imread(VesselName)
    vessel_arr = itk.GetArrayFromImage(vessel)
    return img_arr, mask_arr, vessel_arr, img.GetOrigin(), img.GetSpacing(), img.GetDirection()

