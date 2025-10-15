import numpy as np
import pandas as pd

import anndata

def cells_to_keep(cc, uu, ss,outl = 99.8):
    c_norm = cc - np.min(cc)
    u_norm = uu - np.min(uu)
    s_norm = ss - np.min(ss)
    non_zero = (np.ravel(c_norm > 0) | np.ravel(u_norm > 0) |
                         np.ravel(s_norm > 0))
    
    # remove outliers
    non_outlier = np.ravel(c_norm <= np.percentile(c_norm,outl))
    non_outlier &= np.ravel(u_norm <= np.percentile(u_norm, outl))
    non_outlier &= np.ravel(s_norm <= np.percentile(s_norm, outl))
    
    keep = non_zero & non_outlier & \
            (u_norm > 0.2 * np.percentile(u_norm, 99.5)) & \
            (s_norm > 0.2 * np.percentile(s_norm, 99.5))
    return keep

# minmax normalization factors for archetypes
def extract_minmax(smooth_arch):
    min_c = {}
    max_c = {}
    
    for i in range(smooth_arch.shape[1]):
        c_cur = smooth_arch[:,i].layers['Mc']
        min_c[i] = min(c_cur)[0]
        max_c[i] = max(c_cur)[0]
    max_c = np.ravel(pd.Series(max_c).values)
    min_c = np.ravel(pd.Series(min_c).values)
    return min_c, max_c


def minmax(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def print_vals(x, f, cont = 0):
    print(str(f))
    return
