import os
import torch 
import numpy as np
import itk
import scipy.ndimage as ndimage

from utils import cutoff_percentile

def read_signal(FileName, ImageType, ToTensor = True, Mask = [0]):

    print('Reading in %s image: %s' % (ImageType, os.path.basename(FileName)))
    if ImageType == 'MRP':
        return read_mrp(FileName, ToTensor, Mask[0])
    elif ImageType == 'CTP':
        return read_ctp(FileName, ToTensor, Mask)


def read_mrp(FileName, ToTensor = True, BackGround = 0):
    '''
    Read MRP data, convert to target format

    *For MR Perfusion image:
    we need to convert signal of those voxels that are negative to 1, avoiding NaN issue when calculate CTC later
    '''

    sig_raw = itk.ReadImage(FileName)
    print('  Raw signal image  Size:', sig_raw.shape, 'Depth/Height/Width:', sig_raw.GetWidth(), sig_raw.GetHeight(), sig_raw.GetDepth())
    print('  Time points:', sig_raw.shape[0])
    print('  Time points[3]:', sig_raw.shape[3])
    
        
    sig_nda = itk.GetArrayFromImage(sig_raw) # should be (slice, row, column, time)
    sig_nda = np.transpose(sig_nda, (1, 0, 2, 3))
    sig_nda = sig_nda.astype(float)
    print('  Raw signal array shape:', sig_nda.shape)

    # Extract brain region
    brain = np.where(sig_nda[..., 0] != BackGround)
    min_s = int(np.min(brain[0]))
    max_s = int(np.max(brain[0])) + 1
    min_r = int(np.min(brain[1]))
    max_r = int(np.max(brain[1])) + 1
    min_c = int(np.min(brain[2]))
    max_c = int(np.max(brain[2])) + 1
    brain_region = [[min_s, max_s], [min_r, max_r], [min_c, max_c]]
    print('  Extracted brain region:', brain_region)
    sig_resize = sig_nda[min_s:max_s, min_r:max_r, min_c:max_c, :]
    print('  Resized signal array shape:', sig_resize.shape)

    # Save resized signal image as image_resized.nii
    img_resize = itk.GetImageFromArray(sig_resize, isVector = True)
    img_resize.SetDirection(sig_raw.GetDirection())
    img_resize.SetSpacing(sig_raw.GetSpacing())
    # Be careful about the dimension correspondence (transpose) between sitk image and numpy array
    new_origin = sig_raw.GetOrigin() + np.array([min_c, min_r, min_s]) * sig_raw.GetSpacing()
    del sig_raw
    img_resize.SetOrigin(new_origin)
    ResizeFileName = '%s_resized.nii' % FileName[:-4]
    print('  Reized signal image saved as:', os.path.basename(ResizeFileName))
    itk.WriteImage(img_resize, ResizeFileName)

    # Convert signal of those voxels that are negative to 1
    sig_resize[sig_resize <= 0] = 1.0
    print('  Signal convertion for MRP image: <=0 -> 1')
    print('    Min and max for corrected MRP image: (%d, %d)' % (np.min(sig_resize), np.max(sig_resize)))

    # Save corrected MRP image (.nii) as RawName_corrected.nii
    CrtFileName = '%s_corrected.nii' % ResizeFileName[:-4]
    print('    Corrected MR Perfusion image saved as:', os.path.basename(CrtFileName))
    sig_img_crt = itk.GetImageFromArray(sig_resize, isVector = True)
    sig_img_crt.SetDirection(img_resize.GetDirection())
    sig_img_crt.SetOrigin(img_resize.GetOrigin())
    sig_img_crt.SetSpacing(img_resize.GetSpacing())
    itk.WriteImage(sig_img_crt, CrtFileName)

    if ToTensor:
        sig_resize = torch.from_numpy(sig_resize)

    return sig_resize, img_resize.GetOrigin(), img_resize.GetSpacing(), img_resize.GetDirection()


def read_ctp(FileName, ToTensor = True, BrainMask = []):
    '''
    Read MRP data, convert to target format

    *For MR Perfusion image:
    we need to convert signal of those voxels that are negative to 1, avoiding NaN issue when calculate CTC later
    '''

    sig_raw = itk.imread(FileName)
    sig_raw_arr = itk.GetArrayFromImage(sig_raw)
    print('  Raw signal image  Size:', sig_raw_arr.shape)
    print('  Time points:', sig_raw_arr.shape[0])
        
    sig_nda = itk.GetArrayFromImage(sig_raw) # should be (slice, row, column, time)
    #sig_nda = np.transpose(sig_nda, (1, 0, 2, 3))
    sig_nda = np.transpose(sig_nda, (1, 2, 3, 0))
    sig_nda = sig_nda.astype(float)
    print('  Raw signal array shape (18,138,138,16):', sig_nda.shape)
    
    # Crop brain region (for UNC CTP)
    if not len(BrainMask) == 3:
        raise ValueError("Mask list for CTP should have 3 sub-list element, \
            with value [[min_slice, max_slice], [min_row, max_row], [min_column, max_column]], \
                [] is designed for the entire-range selection")

    for i_boundary in range(len(BrainMask)):
        if len(BrainMask[i_boundary]) == 0:
            BrainMask[i_boundary] = [0, sig_nda.shape[i_boundary]]
    
    print('  Extracted brain region:', BrainMask)
    sig_resize = sig_nda[BrainMask[0][0] : BrainMask[0][1], BrainMask[1][0] : BrainMask[1][1], BrainMask[2][0] : BrainMask[2][1], :]
    print('  Resized signal array shape (4d):', sig_resize.shape)
 #   print(sig_resize)
    brain_nda = sig_resize[..., 0]
    print("Brain nda shape (should be 3d) = ", brain_nda.shape)
    mask = np.zeros(brain_nda.shape)
    brain = np.where(brain_nda > -900) # Aria - Changed based on arbitrary value
    mask[brain] = 1
    mask = ndimage.binary_fill_holes(mask)
    mask = np.repeat(mask[..., np.newaxis], sig_resize.shape[3], axis = 3)
    
    print('  Masked out non-brain region of raw CT perfusion signal image.')
    sig_masked = mask * sig_resize
    # Normalize masked CT over brain region
    CutOff = 2.0
    tmp = cutoff_percentile(sig_masked, mask, CutOff, 100.0 - CutOff)
    sig_masked_normalized = np.copy(sig_masked)
    sig_masked_normalized[mask] = (sig_masked[mask] - tmp[mask].mean()) / tmp[mask].std()
    # Save resized signal image as image_resized.nii
    img_resize = itk.GetImageFromArray(sig_resize)
    print("Image resize size (3D + vector) = ", itk.size(img_resize))
    print("Image resize Num Elements (16) = ", img_resize.GetNumberOfComponentsPerPixel())
    img_resize.SetDirection(sig_raw.GetDirection())
    img_resize.SetSpacing(sig_raw.GetSpacing())
    # Be careful about the dimension correspondence (transpose) between sitk image and numpy array
    new_origin = sig_raw.GetOrigin() + np.array([BrainMask[0][0], BrainMask[1][0], BrainMask[2][0], 0]) * sig_raw.GetSpacing()
    img_resize.SetOrigin(new_origin)
    ResizeFileName = '%s_resized.nii' % FileName[:-4]
    print('  Resized signal image saved as:', os.path.basename(ResizeFileName))
    itk.imwrite(img_resize, ResizeFileName)

    # Save masked signal image as image_masked.nii
    img_masked = itk.GetImageFromArray(sig_masked)
    img_masked.SetOrigin(img_resize.GetOrigin())
    img_masked.SetDirection(img_resize.GetDirection())
    img_masked.SetSpacing(img_resize.GetSpacing())
    MaskedFileName = '%s_masked.nii' % FileName[:-4]
    print('  Masked signal image saved as:', os.path.basename(MaskedFileName))
    itk.imwrite(img_masked, MaskedFileName) 
    
    # Save normalized signal image as image_normalized.nii
    img_normalized = itk.GetImageFromArray(sig_masked_normalized)
    img_normalized.SetOrigin(img_resize.GetOrigin())
    img_normalized.SetDirection(img_resize.GetDirection())
    img_normalized.SetSpacing(img_resize.GetSpacing())
    NormalizedFileName = '%s_normalized.nii' % FileName[:-4]
    print('  Normalized signal image saved as:', os.path.basename(NormalizedFileName))
    itk.imwrite(img_normalized, NormalizedFileName)
    sig_masked_normalized = torch.from_numpy(sig_masked_normalized)

    return sig_masked_normalized, img_resize.GetOrigin(), img_resize.GetSpacing(), img_resize.GetDirection()

