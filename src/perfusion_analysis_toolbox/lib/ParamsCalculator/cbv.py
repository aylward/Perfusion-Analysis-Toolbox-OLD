import numpy as np


def cal(mask, conc, aif, config, device):
    kh = 1
    rho = 1
    TR = 1.55
    nT = 16
    time = np.arange(0, nT * TR, TR)
    nS, nR, nC = mask.shape
    cbv = np.zeros((nS, nR, nC))
    for s in range(nS):
        cbv[s, :, :] = (
            (kh / rho)
            * mask[s, :, :]
            * (
                np.trapz(conc[s, :, :, :], time, axis=2)
                / np.trapz(aif["gv"], time)
            )
        )

    return cbv
