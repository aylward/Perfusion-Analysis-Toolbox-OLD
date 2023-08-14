import numpy as np


def cal(cbv, cbf, mask):
    mtt = np.divide(cbv, cbf)
    return mtt
