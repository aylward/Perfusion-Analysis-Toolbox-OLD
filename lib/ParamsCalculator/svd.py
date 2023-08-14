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


def DSC_SVD(conc, aif, mask):
    # Function to compute parametric maps of Cerebral Blood Flow (CBF)
    # using the Singular Value Decomposition (SVD) method with truncation.
    #
    # Inputs:
    # - conc: 4D matrix containing DSC concentration time courses of all voxels.
    # - aif: Concentration time course in the chosen arterial site.
    # - mask: 3D matrix used to mask the brain volume for analysis.
   
    # 1) Create matrix G
    nS, nR, nC, nT = conc.shape
    aifVector = np.zeros((nT, 1))
    aifVector[0] = aif[0]
    aifVector[nT - 1] = aif[nT- 1]
    TR = 1.55;  # 1.5
    nT = 16
    time = np.arange(0, nT * TR, TR)
    for k in range(1, nT-1):
        aifVector[k] = (aif[k - 1] + 4 * aif[k] + aif[k + 1]) / 6

    aifVett = np.zeros(nT)
    aifVett[0] = aif[0]
    aifVett[-1] = aif[-1]

    for k in range(1, nT - 1):
        aifVett[k] = (aif[k - 1] + 4 * aif[k] + aif[k + 1]) / 6

    G = toeplitz(aifVett, ([aifVett[0]] + np.zeros(nT)))

    # 2) Apply SVD to calculate the inverse of G
    U, S, V = np.linalg.svd(G)
    eigenV = np.diag(S)
    threshold = 0.2000
    threshold = threshold * np.max(eigenV)

    newEigen = np.zeros((eigenV).shape)
    for r in range(eigenV.shape[0]):
      for c in range(eigenV.shape[1]):
        if eigenV[r, c] >= threshold:
            newEigen[r, c] = 1 / eigenV[r, c]

    Ginv = V.T @ np.diag(newEigen) @ U.T
    res_svd_residual = np.zeros((nS, nR, nC, nT))
    # 3) Apply Ginv to calculate the residual function and CBF for each voxel
    res_svd_map = np.zeros((nS, nR, nC))    
    for s in range(nS):
        for r in range(nR):
            for c in range(nC):
                if mask[s, r, c]:
                    vettConc = conc[s, r, c, :].flatten()
                    vettRes = (1 / TR) * Ginv * vettConc

                    res_svd_map[s, r, c] = np.max(np.abs(vettRes))
                    res_svd_residual[s, r, c, :] = vettRes

    tmax_svd = np.zeros((nS, nR, nC))
    for s in range(nS):
        for r in range(nR):
            for c in range(nC):
                argmax = np.argmax(res_svd_residual[s, r, c, :])
                tmax_svd[s, r, c] = argmax

    return res_svd_map, tmax_svd



