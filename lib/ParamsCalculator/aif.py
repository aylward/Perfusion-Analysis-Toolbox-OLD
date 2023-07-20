import numpy as np
from numpy.lib.function_base import trim_zeros
from scipy.integrate import trapz
import os
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

def DSC_mri_aif(mask, conc, size, config, device):
    # Identifies the AIF for DSC-MRI exams. The method is designed to identify the
    # middle cerebral artery (MCA) in slices just above the corpus callosum.
    #
    # Input parameters: 
    # - conc (4D matrix) containing the concentration time curves of all voxels
    # - mask (3D matrix) containing the mask for each slice. Note: the mask used here
    #   is not the one used for concentration calculations, but a restricted version
    #   (the 'fill' function was not used).
    # - options: the struct that contains the method's options. The significant ones are:
    nSlice = 9
    conc_reshaped = conc[nSlice-1, :, :, :]
    aif= extract_AIF(conc_reshaped, mask[nSlice-1, :, :])
    return aif


def extract_AIF(AIFslice, mask):
    # Extracts the AIF from the provided input slice.
    # 1) Identify the region containing the AIF
    # 2) Decimate the candidate voxels
    # 3) Apply the hierarchical clustering algorithm to identify arterial voxels
    # 4) Prepare the output

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

    print(minR)
    print(maxR)
    print(minC)
    print(maxC)
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

    
    xROI = np.arange(center[0] - semiaxisB, center[0] + semiaxisB, 0.01)
    nL = len(xROI)
    xROI = np.concatenate((xROI, np.zeros(nL)))
    yROI = np.zeros(2 * nL)
    for k in range(nL):
        yROI[k] = semiaxisA * ((1 - ((xROI[k] - center[0])**2) / (semiaxisB**2))**0.5) + center[1]

    for k in range(nL):
        xROI[nL + k] = xROI[nL - k]
        yROI[nL + k] = -semiaxisA * ((1 - ((xROI[nL + k] - center[0])**2) / (semiaxisB**2))**0.5) + center[1]
    yROI = np.real(yROI)

  # 2) DECIMATION OF CANDIDATE VOXELS
    print('   Candidate voxel analysis')

  # 2.1) Selection based on area under the curve.
    totalCandidates = np.sum(np.sum(ROI))
    totalCandidatesToKeep = np.ceil(totalCandidates * (1 - pArea))
    AUC = np.sum(AIFslice, axis=2)  # Calculate AUC for each voxel.
    AUC *= ROI
    AUCup =np.max(np.max(AUC))
    print(AUCup)
    AUC[np.where(np.isinf(AUC))] = 0

    cycle = True
    cycleCount = 0
    AUCdown =np.min(np.min(AUC))
    print(AUCdown)
    
    AUCup = np.max(np.max(AUC))
    print(AUCup)
    while cycle:
        cycleCount += 1
        threshold = 0.5 * (AUCup + AUCdown)
        numCandidates = np.sum(np.sum(AUC > threshold))

        if numCandidates == totalCandidatesToKeep:
            cycle = False
        elif numCandidates > totalCandidatesToKeep:
            AUCdown = threshold
        else:
            AUCup = threshold

        if ((AUCup - AUCdown) < 0.01) or (cycleCount > 100):
            cycle = False
    print(' Candidate voxel selection via AUC criteria')
    print('  Voxel initial amount:', totalCandidates)
    print('  Survived voxels:', np.sum(np.sum(ROI)))

    ROI = ROI * (AUC > threshold)
    # 2.2) Selection based on TTP
    totalCandidates = np.sum(np.sum(ROI))
    totalCandidatesToKeep = np.ceil(totalCandidates * (1 - pTTP))
    MC, TTP = np.max(AIFslice, axis=2), np.argmax(AIFslice, axis=2)
    TTP = TTP.astype(np.double) 
    print("This is TTP", np.max(TTP))
    TTP *= ROI
    print("This is TTP 2", np.max(TTP))
    cycle = True
    threshold = 1
    while cycle:
        if (np.sum(np.sum(TTP < threshold)) - np.sum(np.sum(TTP == 0))) >= totalCandidatesToKeep:
            cycle = False
        else:
            threshold += 1

    print(' ')
    print(' Candidate voxel selection via TTP criteria')
    print('  Voxel initial amount:', totalCandidates)
    print('  Survived voxels:', np.sum(np.sum(ROI)))

    ROI = ROI * (TTP < threshold)
    #Aria - Code works till here
    
    # 2.3) Selection based on irregularity index
    totalCandidates = np.sum(np.sum(ROI))
    candidatesToKeep = np.ceil(totalCandidates * (1 - pReg))
    AIF_temp = AIFslice.copy()
    time_temp = time.copy()
    ROI_temp = ROI.copy()
    REG = calculateReg(AIF_temp, time_temp, ROI_temp)

    loop = True
    nLoop = 0
    REGdown = np.min(np.min(REG))
    REGup = np.max(np.max(REG))
    while loop:
        nLoop += 1
        threshold = 0.5 * (REGup + REGdown)
        numCandidates = np.sum(np.sum(REG > threshold))

        if numCandidates == candidatesToKeep:
            loop = False
        elif numCandidates < candidatesToKeep:
            REGup = threshold
        else:
            REGdown = threshold

        if ((REGup - REGdown) < 0.001) or (nLoop >= 100):
            loop = False
    
    print(' ')
    print(' Candidate voxel selection via Ireg criteria')
    print('  Voxel initial amount:', totalCandidates)
    print('  Survived voxels:', np.sum(np.sum(ROI)))

    ROI = ROI * (REG > threshold)
    print('Arterial voxels extraction')
    # 3.1) Preparation of the matrix containing the data
    data2D = np.zeros((((np.sum(np.sum(ROI))).astype(int)), nT))
    ind = np.where(ROI)

    for t in range(0, nT):
      data2D[:, t] = AIFslice[ind[0], ind[1], t]
    maskAIF = ROI
    # 3.2) Applying the Hierarchical Cluster algorithm recursively
    nTrue = 0
    AIFslice = None

    while True:
        nTrue += 1
        print(' ------------------------------------')
        print('CYCLE N#', nTrue)

        # Applying the hierarchical cluster
        data2D_temp = data2D.copy()
        vectorCluster, centroid = clusterHierarchical(data2D_temp, 2)

        # Comparing the clusters and choosing which one to keep
        MC1, TTP1 = np.max(centroid[0, :]), np.argmax(centroid[0, :])
        MC2, TTP2 = np.max(centroid[1, :]), np.argmax(centroid[1, :])

        if (((max([MC1, MC2]) - min([MC1, MC2])) / max([MC1, MC2])) < peak_diff) and (TTP1 != TTP2):
            # The difference between the peaks is smaller than the threshold, choose
            # based on TTP
            clusterChoice = 1 + int(TTP2 < TTP1)  # Result is 1 if TTP1 < TTP2 and 2 if TTP2 < TTP1

            print('  Cluster selected via TTP criteria')
            print('   Selected cluster:', clusterChoice)

        else:
            # Choose based on the difference between peaks
            clusterChoice = 1+  int(MC2 > MC1)  # Result is 1 if MC1 > MC2 and 2 if MC2 > MC1
            print('  Cluster selected via MC criteria')
            print('   Selected cluster:', clusterChoice)
        if (np.sum(vectorCluster == clusterChoice) < nVoxelMin) and (np.sum(vectorCluster == (3 - clusterChoice)) >= nVoxelMin):
            # The populati        clusterChoice += 1on of the chosen cluster is less than the minimum number of
            # accepted voxels, while the other cluster has enough of them.
            # Choose the other cluster.
            clusterChoice = 3 - clusterChoice  # Invert the chosen cluster

            print('  Cluster selected switched because of minimum voxel bound')
            print('   Selected cluster:', clusterChoice)
        
        # Keep only the data related to the chosen cluster
        voxelChosen = 1 * (vectorCluster == clusterChoice)
        indMask = np.where(maskAIF)

        maskAIF[indMask] = voxelChosen
        indVoxel = np.where(voxelChosen)
        nL = len(indVoxel[0])
        data2Dold = data2D
        data2D = np.zeros((nL, nT))
        for t in range(nT):
            data2D[:, t] = data2Dold[indVoxel[0], t]
        print(' ')
        print(' Resume cycle n#', nTrue)
        print('  Voxel initial amount:', len(indMask[0]))
        print('  Survived voxels:', nL)
        print('  Cluster 1: MC', MC1)
        print('             TTP', TTP1)
        print('             voxel', np.sum(vectorCluster == 1))
        print('  Cluster 2: MC', MC2)
        print('             TTP', TTP2)
        print('             voxel', np.sum(vectorCluster == 2))
        print('  Selected cluster:', clusterChoice)

        # Checking the exit criteria
        if (nL <= nVoxelMax) or (nTrue >= 100):
            break;

    # 4) OUTPUT PREPARATION

    # 4.1) Save the search ROI
    AIF_ROI_ind = np.where(ROI)
    AIF_ROI_x = xROI
    AIF_ROI_y = yROI
    # 4.2) Save the position of the selected voxels and the average concentration
    AIFconc = centroid[clusterChoice-1, :]  # Concentration samples for the AIF.
    AIF_conc = AIFconc
    AIF_voxels = []
    for r in range(nR):
        for c in range(nC):
            if maskAIF[r, c] == 1:
                AIF_voxels.append([r, c])
    AIFvoxels = np.array(AIF_voxels)
    
    # 4.3) Calculate the arterial fit with gamma-variant (with recirculation)
    print('Gamma-variate fit computation')

    weights = 0.01 + np.exp(-AIFconc)  # Weights for the fit calculation.
    TTP = np.argmax(AIFconc)
    weights[TTP] = weights[TTP] / 10
    weights[TTP - 1] = weights[TTP - 1] / 5
    weights[TTP + 1] = weights[TTP + 1] / 2
    p = {'t0': '', 'alpha': '', 'beta': '', 'A': '', 'td': '', 'K': '', 'tao': '', 'ExitFlag': ''}
    nR, nC = data2D.shape
    AIFconc_temp = AIFconc.copy()
    weights_temp = weights.copy()
    fitParameters_peak1 = fitGV_peak1(AIFconc_temp, weights_temp)
    fitParameters = fitParameters_peak1[0:4]
    TR = 1.55
    nT = 16
    time = np.arange(0, nT * TR, TR)
    AIF_fit = {}
    AIF_fit['weights'] = weights
    AIF_fit['parameters'] = fitParameters
    fitParameters_temp = fitParameters.copy()
    AIF_fit['gv'] = GVfunction_peak1(fitParameters_temp)
    AIF_fit['time'] = time
    AIF_fit['conc'] = AIFconc
    print(AIF_fit['conc'].shape)
    print(AIF_fit['gv'].shape)
    print(nL)
    hf_fit_aif_final = plt.figure()
    plt.subplot(1, 2, 2)
    plt.plot(time, AIF_fit['conc'], 'bo', markersize=5)
    plt.plot(time, AIF_fit['gv'], 'k-', linewidth=2)
    for l in range(nL):
      plt.plot(time, data2D[l, :], 'r-')
    plt.plot(time, AIF_fit['conc'], 'bo', markersize=5)
    plt.plot(time, AIF_fit['gv'], 'k-', linewidth=2)
    plt.xlabel('[s]', fontsize=10)
    plt.legend(['AIF samples', 'GV function', 'Arterial voxels'])
    plt.title('AIF', fontsize=12)
    plt.xlim(time[0], time[-1])
    plt.show()
    print("This is AIF fit:", AIF_fit)
    return AIF_fit

def calculateReg(y, x, mask):
    # Calculate the irregularity index of the concentration curve for each voxel.
    # The index is calculated by normalizing the area to 1 in order not to penalize voxels with high areas.
    # The formula used to calculate the index is: CTC = integral((C''(t))^2 dt)
    td = x[1] - x[0]
    print(td)
    nR, nC, nT = y.shape
    AUC = np.sum(y, axis=2)
    AUC=AUC+(AUC==1);
    for t in range(nT):
      y[:,:,t]=y[:,:,t]/AUC;

    ind = np.where(np.isinf(y))
    y[ind[0], ind[1], ind[2]] = 0
    ind = np.where(np.isnan(y))
    y[ind[0], ind[1], ind[2]] = 0
    # Calculation of the second derivative
    second_derivative = np.zeros((nR, nC, nT))
    for r in range(nR):
        for c in range(nC):
            if mask[r, c] == 1:
                for t in range(nT):
                    if (t > 0) and (t < nT - 1):
                        # Standard case
                        second_derivative[r, c, t] = ((y[r, c, t+1] - y[r, c, t]) / (td)
                                             - (y[r, c, t] - y[r, c, t-1]) / (td)) / (td)
                    elif t == 0:
                        # Missing previous sample
                        second_derivative[r, c, t] = (y[r, c, t+1] - y[r, c, t]) / ((td)**2)
                    else:
                        # Missing next sample
                        second_derivative[r, c, t] = (y[r, c, t] - y[r, c, t-1]) / ((td)**2)

    # Calculation of the irregularity index
    second_derivative = second_derivative**2
    REG = trapz(second_derivative, x, axis=2)
    return REG


def clusterHierarchical(data, nCluster):
    # Applies the hierarchical clustering algorithm to the data and divides it into nCluster.
    # Returns a vector containing the cluster number assigned to each voxel and the centroid of each cluster.
    distance = pdist(data)
    tree = linkage(distance, 'ward')
    vector1 = fcluster(tree, nCluster, criterion='maxclust')
    nT = data.shape[1]
    centroid = np.zeros((nCluster, nT))
    for k in range(1, nCluster+1):
        ind = np.where(vector1 == k)[0]
        dataCluster = np.zeros((len(ind), nT))
        for t in range(nT):
            dataCluster[:, t] = data[ind, t]

        centroid[k-1, :] = np.mean(dataCluster, axis=0)
        
    return vector1, centroid

def fitGV_peak1(data, weights):
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
    # Initial parameter estimates
    alpha_init = 5
    MCdata, TTPpos = np.max(data), np.argmax(data)
    TE = 0.025; # 25ms
    TR = 1.55;  # 1.55s
    nT = 16
    time = np.arange(0, nT * TR, TR)
    TTPdata = time[int(TTPpos)]
    indices = np.where(data[0:TTPpos] <= 0.05 * MCdata)[0]
    last_index = indices[-1]
    print(last_index)
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
    while data[i] > (0.2 * data[TTP]):
        i += 1
    
    # Adapt the data for "only the first peak"
    data_peak1 = np.zeros_like(data)
    data_peak1[:i+1] = data[:i+1]
    
    weights_peak1 = 0.01 + np.zeros_like(weights)
    weights_peak1[:i+1] = weights[:i+1]
    
    # Estimator
    p = p0
    while True:
        nTrue += 1
        least_squares_output = least_squares(objFitGV_peak1, p, bounds=(lb, ub), args=(data_peak1, weights_peak1))
        
        if nTrue >= 4:
            break
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

# def fitGV_peak2(data, weights, cost_peak1):
#     TE = 0.025; # 25ms
#     tr = 1.55;  # 1.55s
#     nT = 16
#     time = np.arange(0, nT * tr, tr)
#     if np.size(time) == 1:
#         time = np.transpose(time)
#     if np.size(data) == 1:
#         data = np.transpose(data)
#     if np.size(weights) == 1:
#         weights = np.transpose(weights)
#     if np.size(cost_peak1) == 1:
#         cost_peak = np.transpose(cost_peak1)

#     peak1 = GVfunction_peak1(cost_peak1)
#     data_peak2 = data - peak1

#     weights_peak2 = np.ones(len(data_peak2))
#     posTaglioPesi = min([np.where(data > 0.4 * np.max(data))[0][-1], 3 + np.where(data == np.max(data))[0][0]])
#     weights_peak2[:posTaglioPesi] = 1

#     dati_x_stime_init = data_peak2.copy()
#     dati_x_stime_init[:posTaglioPesi] = 0
#     dati_x_stime_init[np.where(dati_x_stime_init < 0)] = 0
#     maxPeak2, TTPpeak2 = np.max(dati_x_stime_init), np.argmax(dati_x_stime_init)
#     t0peak2 = np.where(dati_x_stime_init[:TTPpeak2] < (0.1 * np.max(dati_x_stime_init)))[0][-1]
#     td_init = time[t0peak2] - cost_peak1[0]
#     tao_init = 40

#     ricircolo = GVfunction_ricircolo(np.concatenate((cost_peak1, [td_init, 1, tao_init])))
#     K_init = np.max(dati_x_stime_init) / np.max(ricircolo)

#     p = np.array([td_init, K_init, tao_init])

    
#     plt.figure()
#     plt.plot(time, data, 'ko', time, peak1, 'k-', time, GVfunction_ricircolo(np.concatenate((cost_peak1, p))), 'g-')
#     plt.title('Recirculation fit - initial values')

#     cycle = True
#     nCiclo = 0

#     while cycle:
#         ub = (p * np.array([10, 10, 10]))
#         lb = p / np.array([10, 10, 10])
#         nCiclo += 1
#         least_squares_output = least_squares(objFitGV_peak2, p, bounds=(lb, ub), args=(data_peak2, weights_peak2, cost_peak1))
        
#         if (nCiclo >= 4):
#             cycle = False

#     GVparameter = np.transpose(least_squares_output.x)
#     J = least_squares_output.jac
#     #Gives singular matrix without offset
#     covp = np.linalg.inv((J.T) @ (J))
#     var = np.diag(covp)
#     sd = np.sqrt(var)
#     cv_est_parGV = (sd / p * 100)

#     plt.figure()
#     plt.plot(time, GVfunction_ricircolo(np.concatenate((cost_peak1, p))), 'r-')
#     plt.title('Recirculation final fit')

#     return GVparameter, cv_est_parGV
    
# def objFitGV_peak2(p, data, weights, cost_peak1):
#     vector = GVfunction_ricircolo(np.concatenate((cost_peak1, p)))
#     out = (vector - data) / weights
#     return out

# def GVfunction_ricircolo(p):
#     t0 = p[0]
#     alpha = p[1]
#     beta = p[2]
#     A = p[3]
#     td = p[4]
#     K = p[5]
#     tao = p[6]
#     TE = 0.025; # 25ms
#     TR = 1.55;  # 1.55s
#     nT = 16
#     time = np.arange(0, nT * TR, TR)
#     Tmax = np.max(time)
#     Tmin = np.min(time)
#     TRfine = TR / 10
#     tGrid = np.arange(Tmin, 2 * Tmax + TRfine, TRfine)
#     nTfine = len(tGrid)
    

#     peak2 = np.zeros(nTfine)
#     disp = np.zeros(nTfine)

#     for cont in range(nTfine):
#         t = tGrid[cont]

#         if t > t0 + td:
#             peak2[cont] = K * ((t - t0 - td) ** alpha) * np.exp(-(t - t0 - td) / beta)

#         disp[cont] = np.exp(-t / tao)

#     ricircolo_fine = TRfine * np.convolve(peak2, disp, mode='full')[:nTfine]

#     ricircolo = np.interp(time, tGrid, ricircolo_fine)

#     return ricircolo

# def GVfunction_peak2(p):
#     FP = GVfunction_peak1(p[:4])
#     ricircolo = GVfunction_ricircolo(p)
#     GV = FP + ricircolo
#     return GV

# import numpy as np

# def GVfunction(p):
#     t0 = p[0]  # t0
#     alpha = p[1]  # alpha
#     beta = p[2]  # beta
#     A = p[3]  # A
#     td = p[4]  # td
#     K = p[5]  # K
#     tao = p[6]  # tao
#     TR = 1.55;  # 1.55s
#     nT = 16
#     time = np.arange(0, nT * TR, TR)
#     Tmax = np.max(time)
#     nT = len(time)

#     TRfine = TR / 10
#     tGrid = np.arange(0, 2 * Tmax, TRfine)
#     nTfine = len(tGrid)

#     picco1 = np.zeros(nTfine)
#     picco2 = np.zeros(nTfine)
#     disp = np.zeros(nTfine)

#     for cont in range(nTfine):
#         t = tGrid[cont]

#         if t > t0:
#             picco1[cont] = A * ((t - t0) ** alpha) * np.exp(-(t - t0) / beta)

#         if t > t0 + td:
#             picco2[cont] = K * ((t - t0 - td) ** alpha) * np.exp(-(t - t0 - td) / beta)

#         disp[cont] = np.exp(-t / tao)

#     ricircolo = TRfine * np.convolve(picco2, disp, mode='full')[:nTfine]
#     conc = picco1 + ricircolo

#     GV = np.zeros(nT)
#     for cont in range(nT):
#         pos = np.argmin(np.abs(tGrid - time[cont]))
#         GV[cont] = conc[pos]

#         if np.abs(tGrid[pos] - time[cont]) > 1:
#             print('WARNING: Poor approximation.')

#     return GV


