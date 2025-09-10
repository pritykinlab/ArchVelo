import numpy as np
def inverse_sigm(x, eps = 1e-40):
    x[x<=0.] = eps
    x[x>=1.]= 1-eps
    return np.log(x/(1-x))
