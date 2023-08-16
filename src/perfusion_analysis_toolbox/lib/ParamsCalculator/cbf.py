from .svd import DSC_SVD


def cal(conc, aif, mask):
    cbf, tmax = DSC_SVD(conc, aif["gv"], mask)
    return cbf, tmax
