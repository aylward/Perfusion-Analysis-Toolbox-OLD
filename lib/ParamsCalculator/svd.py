import numpy as np
import os
import torch
import logging
from scipy import signal
import itk
from builtins import object
from scipy import linalg
from scipy.optimize import least_squares
from scipy.signal import convolve
from scipy.linalg import toeplitz

def DSC_mri_SVD(conc, aif, mask, size):
    # Function to compute parametric maps of Cerebral Blood Flow (CBF)
    # using the Singular Value Decomposition (SVD) method with truncation.
    #
    # Inputs:
    # - conc: 4D matrix containing DSC concentration time courses of all voxels.
    # - aif: Concentration time course in the chosen arterial site.
    # - mask: 3D matrix (Aria - 4D in our case) used to mask the brain volume for analysis.
   
    # 1) Create matrix G
    print("Shape of mask", mask.shape)
    print("AIFvector size:", size[3])
    aifVector = np.zeros((size[3], 1))
    aifVector[0] = aif[0]
    aifVector[size[3] - 1] = aif[size[3] - 1]

    for k in range(1, size[3]-1):
        aifVector[k] = (aif[k - 1] + 4 * aif[k] + aif[k + 1]) / 6
    
    print(aifVector[0])
    print(aifVector[0].shape)
    aif_temp = aifVector.reshape((len(aifVector), 1))
    F = np.concatenate((aif_temp, np.zeros((size[3] - 1, 1))))
    print(F.shape)
    G = toeplitz(aifVector, F)

    # 2) Apply SVD to calculate the inverse of G
    U, S, V = np.linalg.svd(G, full_matrices = False)
    print("This is the shape of S:", S.shape)
    print("This is the shape of U:", U.shape)
    print("This is the shape of V:", V.shape)
    eigenV = np.diag(S)
    threshold = 0.2 * np.max(eigenV)
    newEigen = np.zeros(eigenV.shape)
    print(newEigen.shape)
    for k in range(len(eigenV)):
        if eigenV[k][k] >= threshold:
            newEigen[k][k] = 1 / S[k]

    print("This is the shape of newEigen", (newEigen).shape)
    Ginv = U @ newEigen @ V

    # 3) Apply Ginv to calculate the residual function and CBF for each voxel
    res_svd_map = np.zeros((size[0], size[1], size[2]))
    res_svd_residual = np.zeros((size[0], size[1], size[2], size[3]))
    tr = 1.55
    for r in range(size[0]):
        for c in range(size[1]):
            for s in range(size[2]):
              for t in range(size[3]):
                if mask[r, c, s, t] != 0:
                    # Compute the residual function
                    vectorConc = conc[r, c, s, t].numpy()
                    vectorRes = (1 / tr) * Ginv * vectorConc
                    res_svd_map[r, c, s] = np.max(np.abs(vectorRes))
                    res_svd_residual[r, c, s, t] = vectorRes.flatten()[0]

    return res_svd_map, res_svd_residual

