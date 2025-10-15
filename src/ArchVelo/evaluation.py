import numpy as np
from scipy.spatial import KDTree

from scvelo.pl.simulation import compute_dynamics

import matplotlib.pyplot as plt

from .utils import cells_to_keep

def calc_lik_scvelo(g, 
                    adata_atac,
                    model_to_use,
                    plot = False):

    u_all = np.ravel(model_to_use[:,g].layers['Mu'].copy())
    s_all = np.ravel(model_to_use[:,g].layers['Ms'].copy())
    uu = u_all
    ss = s_all
    cc = adata_atac[:, g].layers['Mc'].A
    keep = cells_to_keep(cc, uu, ss)
    tt = np.ravel(model_to_use[:, g].layers['fit_t'])
    _, u_pred, s_pred = compute_dynamics(model_to_use, g)
    rev_srt = np.argsort(np.argsort(tt))
    u_pred = u_pred[rev_srt]
    s_pred = s_pred[rev_srt]
    
    std_u = np.std(uu)
    std_s = np.std(ss)

    scale_factor = [np.nan, 1/std_u, 1/std_s]
    
    n = 0.99*len(uu[keep])

    if plot:

        plt.figure()
        plt.scatter(tt, uu*scale_factor[1], label = 'uu')#uu/scale_u, label = 'uu')
        plt.scatter(tt, u_pred*scale_factor[1], linewidth=3,
                color='black', alpha=0.5, label = 'a_u')
        plt.legend()
        plt.figure()
        plt.scatter(tt, ss*scale_factor[2], label = 'ss')
        plt.scatter(tt, s_pred*scale_factor[2], linewidth=3,
                color='black', alpha=0.5, label = 'a_s')
        plt.legend()
        
    diff_u = (uu-u_pred)*scale_factor[1]#/scale_u
    diff_s = (ss-s_pred)*scale_factor[2]
    if keep is not None:
        diff_u = diff_u[keep]
        diff_s = diff_s[keep]
    
    dist_u = diff_u ** 2
    dist_s = diff_s ** 2
    var_u = np.var(diff_u)
    var_s = np.var(diff_s)
    nll_u = (0.5 * np.log(2 * np.pi * var_u) + 0.5 / n /
             var_u * np.sum(dist_u))
    nll_s = (0.5 * np.log(2 * np.pi * var_s) + 0.5 / n /
             var_s * np.sum(dist_s))
    nll = nll_u + nll_s
    likelihood_u = np.exp(-nll_u)
    likelihood_s = np.exp(-nll_s)
    likelihood = np.exp(-nll)
    return likelihood, likelihood_u, likelihood_s


def phase_multivelo(g, model_to_use):
    gene = g
    adata = model_to_use
    n_anchors = adata.uns['velo_s_params']['t']
    t_sw_array = np.array([adata[:, gene].var['fit_t_sw1'],
                           adata[:, gene].var['fit_t_sw2'],
                           adata[:, gene].var['fit_t_sw3']])
    t_sw_array = t_sw_array[t_sw_array < 20]
    min_idx = int(adata[:, gene].var['fit_anchor_min_idx'])
    max_idx = int(adata[:, gene].var['fit_anchor_max_idx'])
    old_t = np.linspace(0, 20, n_anchors)[min_idx:max_idx+1]
    new_t = old_t - np.min(old_t)
    new_t = new_t * 20 / np.max(new_t)
    a_c = adata[:, gene].varm['fit_anchor_c'].ravel()[min_idx:max_idx+1]
    a_u = adata[:, gene].varm['fit_anchor_u'].ravel()[min_idx:max_idx+1]
    a_s = adata[:, gene].varm['fit_anchor_s'].ravel()[min_idx:max_idx+1]
    new_t =new_t[0:new_t.shape[0]]

    tt = model_to_use[:,g].layers['fit_t']
    uu = np.ravel(adata[:,g].layers['Mu'])
    ss = np.ravel(adata[:,g].layers['Ms'])
    c_all = model_to_use[:, g].layers['ATAC']
    cc = np.ravel(c_all)
    
    new_t = new_t.reshape(-1,1)
    tree = KDTree(new_t)
    neighbor_dists, neighbor_indices = tree.query(tt.reshape(-1,1))
    c_pred = a_c[neighbor_indices]
    u_pred = a_u[neighbor_indices]
    s_pred = a_s[neighbor_indices]
    return tt, uu, ss, cc, new_t, u_pred, s_pred, c_pred

def calc_lik_multivelo(g, 
                       model_to_use,
                      plot = False):
    tt, uu, ss, cc, new_t, u_pred, s_pred, c_pred = phase_multivelo(g, model_to_use = model_to_use)    
    keep = cells_to_keep(cc, uu, ss)
    n = np.sum(keep)
    print('Num cells: ', n)
    
    std_u = np.std(uu)
    std_s = np.std(ss)
    std_c = np.std(cc)

    #print(scale_c, scale_u, scale_s)
    scale_factor = [1/std_c, 1/std_u, 1/std_s]
    n = len(uu[keep])

    uu*=scale_factor[1]
    u_pred*=scale_factor[1]
    ss*=scale_factor[2]
    s_pred*=scale_factor[2]
    if plot:

        plt.figure()
        plt.scatter(tt, uu, label = 'uu')#uu/scale_u, label = 'uu')
        plt.scatter(tt, u_pred, linewidth=3,
                color='black', alpha=0.5, label = 'a_u')
        plt.legend()
        plt.figure()
        plt.scatter(tt, ss, label = 'ss')
        plt.scatter(tt, s_pred, linewidth=3,
                color='black', alpha=0.5, label = 'a_s')
        plt.legend()
        print(len(tt))
        
    diff_u = (uu-u_pred)
    diff_s = (ss-s_pred)
    if keep is not None:
        diff_u = diff_u[keep]
        diff_s = diff_s[keep]
    
    dist_u = diff_u ** 2
    dist_s = diff_s ** 2

    var_u = np.var(diff_u)
    var_s = np.var(diff_s)


    nll_u = (0.5 * np.log(2 * np.pi * var_u) + 0.5 / n /
             var_u * np.sum(dist_u))
    nll_s = (0.5 * np.log(2 * np.pi * var_s) + 0.5 / n /
             var_s * np.sum(dist_s))
    nll = nll_u + nll_s

    likelihood_u = np.exp(-nll_u)
    likelihood_s = np.exp(-nll_s)
    likelihood = np.exp(-nll)

    return likelihood, likelihood_u, likelihood_s

def calc_lik_ArchVelo(g, 
                      adata_atac = None,
                      avel = None,
                      #genes = None, 
                      plot = False,
                      multivelo_cells = True):
    u_all = np.ravel(avel[:,g].layers['Mu'].copy())
    s_all = np.ravel(avel[:,g].layers['Ms'].copy())
    uu = u_all
    ss = s_all
    if multivelo_cells:
        cc = adata_atac[:, g].layers['Mc'].A
    
        keep = cells_to_keep(cc, uu, ss)
    else:
        keep = [True]*avel.shape[0]
    n = np.sum(keep)
    #print('Num cells: ', n)
    
    u = np.ravel(avel[:, g].layers['u']).copy()
    s = np.ravel(avel[:, g].layers['s']).copy()
    #print('s: ', s)
    tt = avel[:,g].layers['fit_t']
    
    std_u = np.std(uu)
    std_s = np.std(ss)

    scale_u = std_u
    scale_s = std_s

    u_all/=scale_u
    s_all/=scale_s

    u/=std_u
    s/=std_s

    u_pred = u
    s_pred = s
    #print('s: ', s)
    uu = u_all
    ss = s_all
    #print('s_all: ', s_all)
    if plot:
        plt.figure()
        plt.scatter(tt, np.ravel(u_all), label = 'uu')
        plt.scatter(tt, u, linewidth=3,
                color='black', alpha=0.5, label = 'a_u')
        plt.legend()
        plt.figure()
        plt.scatter(tt, np.ravel(s_all), label = 'ss')
        plt.scatter(tt, s, linewidth=3,
                color='black', alpha=0.5, label = 'a_s')
        plt.legend()

    diff_u = (uu-u_pred)
    diff_s = (ss-s_pred)
    if keep is not None:
        diff_u = diff_u[keep]
        diff_s = diff_s[keep]
    
    dist_u = diff_u ** 2
    dist_s = diff_s ** 2

    var_u = np.var(diff_u)
    var_s = np.var(diff_s)


    nll_u = (0.5 * np.log(2 * np.pi * var_u) + 0.5 / n /
             var_u * np.sum(dist_u))
    nll_s = (0.5 * np.log(2 * np.pi * var_s) + 0.5 / n /
             var_s * np.sum(dist_s))
    nll = nll_u + nll_s

    likelihood_u = np.exp(-nll_u)
    likelihood_s = np.exp(-nll_s)
    likelihood = np.exp(-nll)

    return likelihood, likelihood_u, likelihood_s