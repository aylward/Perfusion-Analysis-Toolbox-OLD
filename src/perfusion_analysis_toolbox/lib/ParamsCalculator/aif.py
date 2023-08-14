import numpy as np
from numpy.lib.function_base import trim_zeros
from scipy.integrate import trapz
import os
import math
import torch
import logging
from scipy import signal
import itk
import sys
from builtins import object
from scipy import linalg
from scipy.optimize import least_squares
from scipy.signal import convolve
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.optimize import least_squares

def DSC_aif(mask, conc, bolus, size, config, device, vessels):
    nSlice = 69
    conc_reshaped = conc[nSlice-1, :, :, :]
    aif= extract_AIF(conc_reshaped, mask[nSlice-1, :, :], vessels[nSlice-1, :, :], bolus[nSlice-1])
    return aif


def extract_AIF(AIFslice, mask, vesselSlice, bolusSlice):


    # Preparation of accessory variables and parameters
    semiaxisMag = 0.2500
    semiaxisMin = 0.1000
    pArea = 0.4000
    pTTP = 0.4000
    pReg = 0.0500
    nVoxelMax = 6
    nVoxelMin = 4
    peak_diff = 0.0400
    nR, nC, nT = AIFslice.shape
    TE = 0.025; # 25ms
    TR = 1.55;  # 1.55s
    nT = 16
    time = np.arange(0, nT * TR, TR)
    # Extracts the AIF from the provided input slice.
    # 1) Identification of the ROI containing the AIF
    # 1.1) Identification of the mask boundaries

    print('Brain bound detection')
    # Find minimum and maximum row indices with non-zero elements in the mask
    r = 0
    while True:
        if np.sum(mask[r, :]) != 0:
            minR = r
            break
        else:
            r += 1

    r = AIFslice.shape[0]-1
    while True:
        if np.sum(mask[r, :]) != 0:
            maxR = r
            break
        else:
            r -= 1

    # Find minimum and maximum column indices with non-zero elements in the mask
    c = 0
    while True:
        if np.sum(mask[:, c]) != 0:
            minC = c
            break
        else:
            c += 1
    c = AIFslice.shape[1] - 1
    while True:
        if np.sum(mask[:, c]) != 0:
            maxC = c
            break
        else:
            c -= 1

    print('Definition of the AIF extraction searching area')
    
    center = np.zeros(2)
    center[1] = 0.5 * (minR + maxR)  # Y coordinate of the center (calculated on the rows)
    center[0] = 0.6 * (minC + maxC)  # X coordinate of the center (calculated on the columns)

    semiaxisB = semiaxisMag * (maxC - minC)  # The major axis is along the anterior-posterior direction, i.e., left to right in the images
    semiaxisA = semiaxisMin * (maxR - minR)
    ROI = np.zeros((nR, nC))  # Mask containing the voxels of the ROI
    count = 0
    for r in range(nR):
        for c in range(nC):
            if ((r - center[1])**2 / (semiaxisA**2) + (c - center[0])**2 / (semiaxisB**2)) <= 1:
                ROI[r, c] = 1
                count +=1
    ROI = ROI * mask
    ROIinitial = ROI

    
    for r in range(nR):
        for c in range(nC):
            if (vesselSlice[r, c] == 1):
              ROI[r,c] = 1
            else:
              ROI[r, c] = 0
    ROI = ROI * mask

  
    data2D = np.zeros((((np.sum(np.sum(ROI))).astype(int)), nT))
    ind = np.where(ROI)

    for t in range(0, nT):
      data2D[:, t] = AIFslice[ind[0], ind[1], t]
    maskAIF = ROI
    

    AIFconc = np.zeros((nT))
    fit_Params = np.zeros((nR, nC, 4))
    count = 0
    for r in range(nR):
        for c in range(nC):
            if maskAIF[r, c] == 1:
                count +=1
                AIFconc = AIFslice[r, c, :]
                weights = 0.01 + np.exp(-AIFconc)  # Weights for the fit calculation.
                TTP = np.argmax(AIFconc)
                weights[TTP] = weights[TTP] / 10
                weights[TTP - 1] = weights[TTP - 1] / 5
                weights[TTP + 1] = weights[TTP + 1] / 2
                AIFconc_temp = AIFconc.copy()
                weights_temp = weights.copy()
                fitParameters_peak1 = fitGV_peak1(AIFconc_temp, weights_temp)
                fit_Params[r, c, :] = fitParameters_peak1[0:4]
    avg_alpha, avg_beta = find_avg_alpha_beta(maskAIF, fit_Params, count, nR, nC)

    
    nTrue = 0
    fitParams_bat = np.zeros((nR, nC, 2))
    fitParams_ab = np.zeros((nR, nC, 2))
    fit_Params_temp = np.zeros((nR, nC, 4))
    while True:
        nTrue += 1
        fitParams_avg = [avg_alpha, avg_beta]
        for r in range(nR):
          for c in range(nC):
              if maskAIF[r, c] == 1:
                AIFconc = AIFslice[r, c, :]
                weights = 0.01 + np.exp(-AIFconc)  # Weights for the fit calculation.
                TTP = np.argmax(AIFconc)
                weights[TTP] = weights[TTP] / 10
                weights[TTP - 1] = weights[TTP - 1] / 5
                weights[TTP + 1] = weights[TTP + 1] / 2
                fitParams_bat[r,c] = find_params_bat_optim(AIFconc, weights, fitParams_avg, fit_Params_temp[r, c, 0], fit_Params_temp[r, c, 3])
                fitParams_ab[r,c] = find_params_alpha_beta_optim(AIFconc, weights, fitParams_bat[r, c], avg_alpha, avg_beta)
                fit_Params_temp[r, c, 0] = fitParams_bat[r,c, 0]
                fit_Params_temp[r, c, 3] = fitParams_bat[r,c, 1]
                fit_Params_temp[r, c, 1] = fitParams_ab[r,c, 0]
                fit_Params_temp[r, c, 2] = fitParams_ab[r,c, 1]
        
        avg_alpha, avg_beta = find_avg_alpha_beta(maskAIF, fit_Params_temp, count, nR, nC)

        if nTrue>1000:
            break
    
    min_t0 = 100
    final_fitParameters = [0, 0, 0, 0]
    final_conc =  np.zeros((nT))
    for r in range(nR):
        for c in range(nC):
            if maskAIF[r, c] == 1:
                if(fit_Params_temp[r, c, 0]< min_t0):
                  final_fitParameters = fit_Params_temp[r, c, :]
                  final_conc = AIFslice[r,c,:]

    
    TR = 1.55
    nT = 16
    time = np.arange(0, nT * TR, TR)
    AIF_fit = {}
    AIF_fit['parameters'] = final_fitParameters
    AIF_fit['gv'] = GVfunction_peak1(final_fitParameters)
    AIF_fit['time'] = time
    AIF_fit['conc'] = final_conc
    hf_fit_aif_final = plt.figure()
    plt.plot(time, AIF_fit['conc'], 'bo', markersize=5)
    plt.plot(time, AIF_fit['gv'], 'k-', linewidth=2)
    plt.plot(time, AIF_fit['conc'], 'bo', markersize=5)
    plt.plot(time, AIF_fit['gv'], 'k-', linewidth=2)
    plt.xlabel('[s]', fontsize=10)
    plt.legend(['AIF samples', 'GV function'])
    plt.title('AIF', fontsize=12)
    plt.xlim(time[0], time[-1])
    plt.savefig('high-res-aif.png')
    print("This is AIF fit:", AIF_fit)
    return AIF_fit

def fitGV_peak1(data, weights):
    # Computes the fit of the first peak with a gamma-variant function.
    # The function is described by the formula:
    #
    # FP(t) = A * ((t - t0) ** alpha) * exp(-(t - t0) / beta) + S0
    #
    # c(t) = FP(t)
    #
    # Parameters: p = [t0 alpha beta A]

    # Estimator options
    nTrue = 0
    alpha_init = 5
    # Initial parameter estimates
    MCdata, TTPpos = np.max(data), np.argmax(data)
    TE = 0.025; # 25ms
    TR = 1.55;  # 1.55s
    nT = 16
    time = np.arange(0, nT * TR, TR)
    TTPdata = time[int(TTPpos)]
    indices = np.where(data[0:TTPpos] <= 0.05 * MCdata)[0]
    last_index = indices[-1]
    t0_init = time[last_index]
    beta_init = (TTPdata - t0_init) / alpha_init
    A_init = (MCdata / np.max(GVfunction_peak1([t0_init, alpha_init, beta_init, 1])))

    p0 = [t0_init, alpha_init, beta_init, A_init]  # Initial values
    lb = np.array(p0) * 0.1   # Lower bounds
    ub = np.array(p0) * 10   # Upper bounds

    # Check the data, ensure they are column vectors
    if data.shape[0] == 1:
        data = np.transpose(data)

    if weights.shape[0] == 1:
        weights = np.transpose(weights)

    # Increase the precision of the peak
    MC, TTP = np.max(data), np.argmax(data)
    weights[TTP] /= 10
    weights[TTP-1] /= 2
 
    # Find the end of the first peak (20% of the maximum value)
    i = TTP
    while i<nT and data[i] > (0.2 * data[TTP]):
        i += 1

    # Adapt the data for "only the first peak"
    data_peak1 = np.zeros_like(data)
    data_peak1[:i+1] = data[:i+1]

    weights_peak1 = 0.01 + np.zeros_like(weights)
    weights_peak1[:i+1] = weights[:i+1]

    # Estimator
    p = p0
    least_squares_output = least_squares(objFitGV_peak1, p, bounds=(lb, ub),max_nfev = 1000, xtol = 1e-8, gtol = 1e-8, ftol = 1e-8, args=(data_peak1, weights_peak1))
    GVparameter = np.transpose(least_squares_output.x)
    J = least_squares_output.jac
    covp = np.linalg.inv(np.transpose(J) @ J)
    var = np.diag(covp)
    sd = np.sqrt(var)
    cv_est_parGV = (sd / p * 100)

    return GVparameter

def objFitGV_peak1(p, data, weights):
    # Objective function to minimize for the fitGV_peak1 function
    vector = GVfunction_peak1(p)
    out = (vector - data) / weights
    return out


def GVfunction_peak1(p):
    # Calculates the gamma-variant function defined by the parameters in p
    # The gamma-variant function is defined by the formula:
    # GV(t) = A * ((t - t0) ** alpha) * exp(-(t - t0) / beta)
    t = 0

    t0 = p[0]    # t0
    alpha = p[1]    # alpha
    beta = p[2]    # beta
    A = p[3]    # A
    TR = 1.55;  # 1.55s
    nT = 16
    time = np.arange(0, nT * TR, TR)
    GV = np.zeros(nT)
    for cont in range(nT):
        t = time[cont]
        if t > t0:
            GV[cont] = A * ((t - t0) ** alpha) * np.exp(-(t - t0) / beta)

    return GV

def fitGV_peak1_2(data, weights, param1, param2, t0, A):
    nTrue = 0
    # Initial parameter estimates
    MCdata, TTPpos = np.max(data), np.argmax(data)
    TE = 0.025; # 25ms
    TR = 1.55;  # 1.55s
    nT = 16
    time = np.arange(0, nT * TR, TR)
    TTPdata = time[int(TTPpos)]
    indices = np.where(data[0:TTPpos] <= 0.05 * MCdata)[0]
    last_index = indices[-1]
    if(t0 == 0):
      t0_init = time[last_index]
    else:
      t0_init = t0
    if A == 0:
      A_init = (MCdata / np.max(GVfunction_peak1_2([t0_init, 1], param1, param2)))
    else:
      A_init = A

    p0 = [t0_init, A_init]  # Initial values
    lb = np.array(p0) * 0.1   # Lower bounds
    ub = np.array(p0) * 10   # Upper bounds

    # Check the data, ensure they are column vectors
    if data.shape[0] == 1:
        data = np.transpose(data)

    if weights.shape[0] == 1:
        weights = np.transpose(weights)

    # Increase the precision of the peak
    MC, TTP = np.max(data), np.argmax(data)
    weights[TTP] /= 10
    weights[TTP-1] /= 2

    # Find the end of the first peak (20% of the maximum value)
    i = TTP
    while i<nT and data[i] > (0.2 * data[TTP]):
        i += 1

    # Adapt the data for "only the first peak"
    data_peak1 = np.zeros_like(data)
    data_peak1[:i+1] = data[:i+1]

    weights_peak1 = 0.01 + np.zeros_like(weights)
    weights_peak1[:i+1] = weights[:i+1]

    # Estimator
    p = p0
    least_squares_output = least_squares(objFitGV_peak1_2, p, bounds=(lb, ub), xtol = 1e-8, gtol = 1e-8, ftol = 1e-8, max_nfev = 1000, args=(data_peak1, weights_peak1, param1, param2))
    GVparameter = np.transpose(least_squares_output.x)

    return GVparameter

def objFitGV_peak1_2(p, data, weights, param1, param2):
    # Objective function to minimize for the fitGV_peak1 function
    vector = GVfunction_peak1_2(p, param1, param2)
    out = (vector - data) / weights
    return out


def GVfunction_peak1_2(p, param1, param2):
    # Calculates the gamma-variant function defined by the parameters in p
    # The gamma-variant function is defined by the formula:
    # GV(t) = A * ((t - t0) ** alpha) * exp(-(t - t0) / beta)
    t = 0
    t0 = p[0]  # t0
    alpha = param1   # alpha
    beta = param2  # beta
    A = p[1]  # A
    TR = 1.55;  # 1.55s
    nT = 16
    time = np.arange(0, nT * TR, TR)
    GV = np.zeros(nT)
    for cont in range(nT):
        t = time[cont]
        if t > t0:
            GV[cont] = A * ((t - t0) ** alpha) * np.exp(-(t - t0) / beta)

    return GV

def find_params_bat_optim(avg_test, avg_weights, fitParams_test, t0, A ):
  param1 = (fitParams_test[0])
  param2 = (fitParams_test[1])
  temp1 = param1
  temp2 = param2
  fitParams_test = (fitGV_peak1_2(avg_test, avg_weights, param1, param2, t0, A))[0:2]
  return fitParams_test

def find_avg_alpha_beta(maskAIF, fit_Params, count, nR, nC):
    alpha_sum = 0
    beta_sum = 0
    for r in range(nR):
      for c in range(nC):
        if maskAIF[r, c] == 1:
          alpha_sum += fit_Params[r, c, 1]
          beta_sum += fit_Params[r, c, 2]

    avg_alpha = alpha_sum/count
    avg_beta = beta_sum/count
    return avg_alpha, avg_beta


def fitGV_peak1_1(data, weights, param1, param2, alpha, beta):
    # Computes the fit of the first peak with a gamma-variant function.
    # The function is described by the formula:
    #
    # FP(t) = A * ((t - t0) ** alpha) * exp(-(t - t0) / beta)
    #
    # c(t) = FP(t)
    #
    # Parameters: p = [t0 alpha beta A]

    # Estimator options
    nTrue = 0
    if alpha == 0:
      alpha_init = 5
    else:
      alpha_init = alpha
    # Initial parameter estimates
    MCdata, TTPpos = np.max(data), np.argmax(data)
    TE = 0.025; # 25ms
    TR = 1.55;  # 1.55s
    nT = 16
    time = np.arange(0, nT * TR, TR)
    TTPdata = time[int(TTPpos)]
    indices = np.where(data[0:TTPpos] <= 0.05 * MCdata)[0]
    last_index = indices[-1]
    t0_init = time[last_index]
    if beta == 0:
      beta_init = (TTPdata - t0_init) / alpha_init
    else:
      beta_init = beta
    p0 = [alpha_init, beta_init]  # Initial values
    lb = np.array(p0) * 0.1   # Lower bounds
    ub = np.array(p0) * 10   # Upper bounds

    # Check the data, ensure they are column vectors
    if data.shape[0] == 1:
        data = np.transpose(data)

    if weights.shape[0] == 1:
        weights = np.transpose(weights)

    # Increase the precision of the peak
    MC, TTP = np.max(data), np.argmax(data)
    weights[TTP] /= 10
    weights[TTP-1] /= 2

    # Find the end of the first peak (20% of the maximum value)
    i = TTP
    while i<nT and data[i] > (0.2 * data[TTP]):
        i += 1

    # Adapt the data for "only the first peak"
    data_peak1 = np.zeros_like(data)
    data_peak1[:i+1] = data[:i+1]

    weights_peak1 = 0.01 + np.zeros_like(weights)
    weights_peak1[:i+1] = weights[:i+1]

    # Estimator
    p = p0
    least_squares_output = least_squares(objFitGV_peak1_1, p, bounds=(lb, ub), xtol = 1e-8, gtol = 1e-8, ftol = 1e-8,max_nfev = 1000, args=(data_peak1, weights_peak1, param1, param2))
    GVparameter = np.transpose(least_squares_output.x)
    return GVparameter

def objFitGV_peak1_1(p, data, weights, param1, param2):
    # Objective function to minimize for the fitGV_peak1 function
    vector = GVfunction_peak1_1(p, param1, param2)
    out = (vector - data) / weights
    return out


def GVfunction_peak1_1(p, param1, param2):
    # Calculates the gamma-variant function defined by the parameters in p
    # The gamma-variant function is defined by the formula:
    # GV(t) = A * ((t - t0) ** alpha) * exp(-(t - t0) / beta)
    t = 0
    t0 = param1   # t0
    alpha = p[0]   # alpha
    beta = p[1]  # beta
    A = param2  # A
    TR = 1.55;  # 1.55s
    nT = 16
    time = np.arange(0, nT * TR, TR)
    GV = np.zeros(nT)
    for cont in range(nT):
        t = time[cont]
        if t > t0:
            GV[cont] = A * ((t - t0) ** alpha) * np.exp(-(t - t0) / beta)

    return GV

def find_params_alpha_beta_optim(avg_test, avg_weights, fitParams_test, alpha, beta ):
  param1 = (fitParams_test[0])
  param2 = (fitParams_test[1])
  temp1 = param1
  temp2 = param2
  fitParams_test = (fitGV_peak1_1(avg_test, avg_weights, param1, param2, alpha, beta))[0:2]
  return fitParams_test

