import os
import torch 
import numpy as np
import itk
import scipy.ndimage as ndimage

from utils import cutoff_percentile

def read_signal(FileName, MaskName, ToTensor = True):
        return read_ctp(FileName, MaskName, ToTensor)


def read_ctp(FileName, MaskName, ToTensor = True):
    '''
    Read MRP data, convert to target format

    *For MR Perfusion image:
    we need to convert signal of those voxels that are negative to 1, avoiding NaN issue when calculate CTC later
    '''

    img = itk.imread(FileName)
    img_arr = itk.GetArrayFromImage(img)
    img_arr = np.transpose(img_arr, (1, 2, 3, 0))
    nan_mask = np.isnan(img_arr)
    img_arr_nan = np.where(nan_mask==True)
    img_arr[img_arr_nan] = 0
    img = itk.GetImageFromArray(img_arr)
    print('  Raw signal image shape:', img_arr.shape)
    print('  Time points:', img_arr.shape[3])
    mask = itk.imread(MaskName)
    mask_arr = itk.GetArrayFromImage(mask)
    print(mask_arr.shape)
    mask = itk.GetImageFromArray(mask_arr)
    print('  Raw signal image shape:', mask_arr.shape)
    print('  Time points:', mask_arr.shape[0])
    # sig_masked_normalized = torch.from_numpy(sig_masked_normalized)
    return img_arr, mask_arr, img.GetOrigin(), img.GetSpacing(), img.GetDirection()

