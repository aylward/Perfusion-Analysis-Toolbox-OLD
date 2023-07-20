import numpy as np
import os

def DSC_mri_conc(volumes, mask):
    S0map, bolus = DSC_ct_S0(volumes, mask)
    ind = np.where(S0map)
    conc = np.zeros(volumes.shape)
    nS = volumes.shape[0]
    nR = volumes.shape[1]
    nC = volumes.shape[2]
    nT = volumes.shape[3]
    conc_reshaped = conc.ravel()
    volumes_reshaped = volumes.ravel()
    for t in range(nT):
        conc[ind[0], ind[1], ind[2], t] = volumes[ind[0], ind[1], ind[2], t] - S0map[ind[0], ind[1], ind[2]]
    
    return np.real(conc)

def DSC_ct_S0(volumes, mask):
    nSamplesMin = 0
    nSamplesMax = 29
    thresh = 0.01
    nS = volumes.shape[0]
    nR = volumes.shape[1]
    nC = volumes.shape[2]
    nT = volumes.shape[3]
    sig_avg = np.zeros(nT)

    for t in range(nT):
        sig_avg[t] = np.nanmean(np.nanmean(np.nanmean(volumes[:, :, :, t])))
    
    #thresh = thresh * (np.max(sig_avg) - np.min(sig_avg))
    mean_signal = np.zeros((nS, nT))

    for s in range(nS):
        for t in range(nT):
            mean_signal[s, t] = (np.nanmean(np.nanmean(volumes[s, :, :, t])))
            
    flag = True
    start_s = 0
    while flag:
        if np.isnan(np.mean(mean_signal[start_s, :])):
            start_s += 1
        else:
            flag = False
    S0map = np.zeros(volumes.shape[:3])
    bolus = np.zeros((nS), dtype=int)
    count = 0
    for s in range(start_s, nS):
        ciclo = True
        pos = nSamplesMin

        while ciclo:
            count +=1
            mean_val = np.mean(mean_signal[s, 0:pos+1])
            if abs((mean_val - mean_signal[s, pos]) / mean_val) < thresh:
                pos += 1
            else:
                ciclo = False
                pos -= 1

            if pos == nSamplesMax or pos+1 > len(mean_signal[0, :]):
                ciclo = False
                pos -= 1
        S0map[s, :, :] = mask[s, :, :] * np.mean(volumes[s, :, :, 0:pos], axis=2)
        bolus[s] = pos

    return S0map, bolus

