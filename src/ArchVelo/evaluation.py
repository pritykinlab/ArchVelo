import numpy as np
from scipy.spatial import KDTree

from scvelo.pl.simulation import compute_dynamics

import matplotlib.pyplot as plt

#from .utils import cells_to_keep

def calc_likelihood(uu, u_pred, ss, s_pred, cells, plot = False, tt = None):
    std_u = np.std(uu)
    std_s = np.std(ss)

    scale_factor = [1/std_u, 1/std_s]
    
    uu*=scale_factor[0]
    u_pred*=scale_factor[0]
    ss*=scale_factor[1]
    s_pred*=scale_factor[1]
    
    cells = cells
    n = np.sum(cells)
    diff_u = (uu-u_pred)
    diff_s = (ss-s_pred)
    if cells is not None:
        diff_u = diff_u[cells]
        diff_s = diff_s[cells]
    
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

    if plot:

        plt.figure()
        plt.scatter(tt, uu, label = 'Mu')#uu/scale_u, label = 'uu')
        plt.scatter(tt, u_pred, linewidth=3,
                color='black', alpha=0.5, label = 'u_pred')
        plt.legend()
        plt.figure()
        plt.scatter(tt, ss, label = 'Ms')
        plt.scatter(tt, s_pred, linewidth=3,
                color='black', alpha=0.5, label = 's_pred')
        plt.legend()
        
    return likelihood, likelihood_u, likelihood_s

def calc_lik_scvelo(avel,
                    g,
                    cells = None,
                    plot = False):

    uu = np.ravel(avel[:,g].layers['Mu'].copy())
    ss = np.ravel(avel[:,g].layers['Ms'].copy())
    # if multivelo_cells:
    #     cc = adata_atac[:, g].layers['Mc'].A
    #     cells = cells_to_keep(cc, uu, ss)
    # else:
    #     cells = [True]*avel.shape[0]
    if cells is None:
        cells = [True]*avel.shape[0]
    tt = np.ravel(avel[:, g].layers['fit_t'])
    _, u_pred, s_pred = compute_dynamics(avel, g)
    rev_srt = np.argsort(np.argsort(tt))
    u_pred = u_pred[rev_srt]
    s_pred = s_pred[rev_srt]
    
    lik, lik_u, lik_s = calc_likelihood(uu, u_pred, ss, s_pred, cells, plot = plot, tt = tt)
    
    return lik, lik_u, lik_s


def phase_multivelo(avel, g):
    gene = g
    adata = avel
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

    tt = avel[:,g].layers['fit_t']
    uu = np.ravel(adata[:,g].layers['Mu'])
    ss = np.ravel(adata[:,g].layers['Ms'])
    c_all = avel[:, g].layers['ATAC']
    cc = np.ravel(c_all)
    
    new_t = new_t.reshape(-1,1)
    tree = KDTree(new_t)
    neighbor_dists, neighbor_indices = tree.query(tt.reshape(-1,1))
    c_pred = a_c[neighbor_indices]
    u_pred = a_u[neighbor_indices]
    s_pred = a_s[neighbor_indices]
    return tt, uu, ss, cc, new_t, u_pred, s_pred, c_pred

def calc_lik_multivelo(avel, g,
                       cells = None,
                      plot = False):
    tt, uu, ss, cc, new_t, u_pred, s_pred, c_pred = phase_multivelo(avel, g)    
    #cells = cells_to_keep(cc, uu, ss)
    if cells is None:
        cells = [True]*avel.shape[0]
    lik, lik_u, lik_s = calc_likelihood(uu, u_pred, ss, s_pred, cells, plot = plot, tt = tt)

    return lik, lik_u, lik_s

def calc_lik_ArchVelo(avel, g,
                      plot = False,
                      cells = None):
    uu = np.ravel(avel[:,g].layers['Mu'].copy())
    ss = np.ravel(avel[:,g].layers['Ms'].copy())
    
    # if multivelo_cells:
    #     cc = adata_atac[:, g].layers['Mc'].A
    #     cells = cells_to_keep(cc, uu, ss)
    # else:
    #     cells = [True]*avel.shape[0]
    if cells is None:
        cells = [True]*avel.shape[0]
    
    u_pred = np.ravel(avel[:, g].layers['u']).copy()
    s_pred = np.ravel(avel[:, g].layers['s']).copy()

    tt = np.ravel(avel[:, g].layers['fit_t'])
        
    lik, lik_u, lik_s = calc_likelihood(uu, u_pred, ss, s_pred, cells, plot = plot, tt = tt)

    return lik, lik_u, lik_s

def set_likelihood(avel, cells = None, lik_cutoff = 0.05):
    avel.var['fit_likelihood'] = np.nan
    if cells is None:
        cells = {}
        for g in avel.var_names: 
            cells[g] = [True]*avel.shape[0]
    for g in avel.var_names:  
        liks = calc_lik_ArchVelo(avel, g,
                      cells = cells[g])
        avel.var['fit_likelihood'].loc[g] = liks[0]
    avel.var['velo_s_genes'] = (avel.var['fit_likelihood']>lik_cutoff)
    avel.var['velo_s_norm_genes'] = (avel.var['fit_likelihood']>lik_cutoff)
    