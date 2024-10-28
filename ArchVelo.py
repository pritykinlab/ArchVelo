import warnings
warnings.filterwarnings("ignore")

import multivelo as mv
import os
import scipy
import numpy as np

import pandas as pd
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt

from multivelo.dynamical_chrom_func import *

#### Optimization for chromatin

def minmax(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))


def solve_for_chromatin(times, pars):
    switch, alpha_c, scale_cc, c0= pars
    tau = times*(times<switch)+(times-switch)*(times>=switch)
    alpha_c_full = alpha_c*(times<switch)+alpha_c*scale_cc*(times>=switch)
    eat = np.exp(-alpha_c_full * tau)
    kc = (times<switch).astype(int)
    c = 1-(1-c0)*np.exp(-alpha_c*switch)
    c = c0+(c-c0)*(times>=switch).astype(int)
    # if chrom_open:
    #     kc = 1
    # else:
    #     kc = 0
    # alpha_c *= scale_cc
    return (kc - (kc - c) * eat)


def err(pars, c, times):
    sol = solve_for_chromatin(times, pars)
    return np.linalg.norm(c-sol)**2


#pars are switch, alpha_c, scale_cc, c0
def optimize_chromatin(times, c, seed=57):
    res = scipy.optimize.dual_annealing(err, #[5, 0.1, 0], 
                                  args = (c, times),
                                        seed = seed,
                                 bounds = [(0,20), (0, 10), (0.01,10), (0,1)], 
                                       maxiter = 1000)
        # scipy.optimize.differential_evolution(err, #[5, 0.1, 0], 
        #                               args = (c, times),
        #                              bounds = [(0,20), (0, 100), (0,1)])
    return res.x


#### Optimization

def predict_exp_mine(tau,
                c0,
                u0,
                s0,
                alpha_c,
                alpha,
                beta,
                gamma,
                scale_cc=1,
                pred_r=True,
                chrom_open=True,
                const_chrom = False,
                backward=False,
                rna_only=False):

    # check_params(alpha_c,
    #              alpha,
    #              beta,
    #              gamma,
    #              c0,
    #              u0,
    #              s0)

    if len(tau) == 0:
        return np.empty((0, 3))
    if backward:
        tau = -tau
    res = np.empty((len(tau), 3))
    #eat = np.exp(-alpha_c * tau)
    ebt = np.exp(-beta * tau)
    egt = np.exp(-gamma * tau)
    if rna_only:
        kc = 1
        c0 = 1
    else:
        if chrom_open:
            kc = 1
        else:
            kc = 0
            alpha_c *= scale_cc
    #this line was in the wrong spot
    eat = np.exp(-alpha_c * tau)
    const = (kc - c0) * alpha / (beta - alpha_c)
    #chromatin
    if not const_chrom:
        res[:, 0] = kc - (kc - c0) * eat
    else:
        res[:,0] = 0

    if pred_r:
        if not const_chrom:
            res[:, 1] = u0 * ebt + (alpha * kc / beta) * (1 - ebt)
            res[:, 1] += const * (ebt - eat)

            res[:, 2] = s0 * egt + (alpha * kc / gamma) * (1 - egt)
            res[:, 2] += ((beta / (gamma - beta)) *
                        ((alpha * kc / beta) - u0 - const) * (egt - ebt))
            res[:, 2] += (beta / (gamma - alpha_c)) * const * (egt - eat)
        else:
            res[:, 1] = u0*np.ones(len(tau))
            res[:, 2] = s0*np.ones(len(tau))

    else:
        res[:, 1] = np.zeros(len(tau))
        res[:, 2] = np.zeros(len(tau))
    return res


def generate_exp_mine(tau_list,
                 t_sw_array,
                 c0,
                 alpha_c,
                 alpha,
                 beta,
                 gamma,
                 scale_cc=1,
                 model=1,
                 rna_only=False):

    if beta == alpha_c:
        beta += 1e-3
    if gamma == beta or gamma == alpha_c:
        gamma += 1e-3
    switch = len(t_sw_array)
    if switch >= 1:
        tau_sw1 = np.array([t_sw_array[0]])
        if switch >= 2:
            tau_sw2 = np.array([t_sw_array[1] - t_sw_array[0]])
            if switch == 3:
                tau_sw3 = np.array([t_sw_array[2] - t_sw_array[1]])
    exp_sw1, exp_sw2, exp_sw3 = (np.empty((0, 3)),
                                 np.empty((0, 3)),
                                 np.empty((0, 3)))
    tau1 = tau_list[0]
    if switch >= 1:
        tau2 = tau_list[1]
        if switch >= 2:
            tau3 = tau_list[2]
            if switch == 3:
                tau4 = tau_list[3]
    exp1, exp2, exp3, exp4 = (np.empty((0, 3)), np.empty((0, 3)),
                              np.empty((0, 3)), np.empty((0, 3)))
    if model == 1:
        exp1 = predict_exp_mine(tau1, c0, 0, 0, alpha_c, alpha, beta, gamma,
                           pred_r=False, scale_cc=scale_cc, rna_only=rna_only)
        if switch >= 1:
            exp_sw1 = predict_exp_mine(tau_sw1, c0, 0, 0, alpha_c, alpha, beta,
                                  gamma, pred_r=False, scale_cc=scale_cc,
                                  rna_only=rna_only)
            exp2 = predict_exp_mine(tau2, exp_sw1[0, 0], exp_sw1[0, 1],
                               exp_sw1[0, 2], alpha_c, alpha, beta, gamma,
                               scale_cc=scale_cc, rna_only=rna_only)
            if switch >= 2:
                #print(tau_sw1, tau_sw2, tau_sw3)
                exp_sw2 = predict_exp_mine(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1],
                                      exp_sw1[0, 2], alpha_c, alpha, beta,
                                      gamma, scale_cc=scale_cc,
                                      rna_only=rna_only)
                exp3 = predict_exp_mine(tau3, exp_sw2[0, 0], exp_sw2[0, 1],
                                   exp_sw2[0, 2], alpha_c, alpha, beta, gamma,
                                   chrom_open=False, scale_cc=scale_cc,
                                   rna_only=rna_only)
                if switch == 3:
                    exp_sw3 = predict_exp_mine(tau_sw3, exp_sw2[0, 0],
                                          exp_sw2[0, 1], exp_sw2[0, 2],
                                          alpha_c, alpha, beta, gamma,
                                          chrom_open=False, scale_cc=scale_cc,
                                          rna_only=rna_only)
                    # print(c0, exp_sw1[0, 0], exp_sw2[0, 0], exp_sw3[0, 0])
                    # print(0, exp_sw1[0, 1], exp_sw2[0, 1], exp_sw3[0, 1])
                    # print(tau3[:4])
                    # print(exp3[:4,1])
                    exp4 = predict_exp_mine(tau4, exp_sw3[0, 0], exp_sw3[0, 1],
                                       exp_sw3[0, 2], alpha_c, 0, beta, gamma,
                                       chrom_open=False, scale_cc=scale_cc,
                                       rna_only=rna_only)
    elif model == 2:
        exp1 = predict_exp_mine(tau1, c0, 0, 0, alpha_c, alpha, beta, gamma,
                           pred_r=False, scale_cc=scale_cc, rna_only=rna_only)
        if switch >= 1:
            exp_sw1 = predict_exp_mine(tau_sw1, c0, 0, 0, alpha_c, alpha, beta,
                                  gamma, pred_r=False, scale_cc=scale_cc,
                                  rna_only=rna_only)
            exp2 = predict_exp_mine(tau2, exp_sw1[0, 0], exp_sw1[0, 1],
                               exp_sw1[0, 2], alpha_c, alpha, beta, gamma,
                               scale_cc=scale_cc, rna_only=rna_only)
            if switch >= 2:
                exp_sw2 = predict_exp_mine(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1],
                                      exp_sw1[0, 2], alpha_c, alpha, beta,
                                      gamma, scale_cc=scale_cc,
                                      rna_only=rna_only)
                exp3 = predict_exp_mine(tau3, exp_sw2[0, 0], exp_sw2[0, 1],
                                   exp_sw2[0, 2], alpha_c, 0, beta, gamma,
                                   scale_cc=scale_cc, rna_only=rna_only)
                if switch == 3:
                    exp_sw3 = predict_exp_mine(tau_sw3, exp_sw2[0, 0],
                                          exp_sw2[0, 1], exp_sw2[0, 2],
                                          alpha_c, 0, beta, gamma,
                                          scale_cc=scale_cc, rna_only=rna_only)
                    exp4 = predict_exp_mine(tau4, exp_sw3[0, 0], exp_sw3[0, 1],
                                       exp_sw3[0, 2], alpha_c, 0, beta, gamma,
                                       chrom_open=False, scale_cc=scale_cc,
                                       rna_only=rna_only)
    return (exp1, exp2, exp3, exp4), (exp_sw1, exp_sw2, exp_sw3)



def err_all(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = None, weight_c = 0.6, chrom_on = None, new = False, rna = None, gene_weights = None, max_c = None, min_c = None, full_res_denoised = None):
    u_all = rna[:,g].layers['Mu'].copy()
    s_all = rna[:,g].layers['Ms'].copy()
    #c_all = cur_prod.loc[:,g]
    #std_c = np.std(c_all)
    std_u = np.std(u_all)
    std_s = np.std(s_all)
    scale_u = std_u/std_s
    #scale_c = std_c/std_s
    u_all/=scale_u
    #c_all/=scale_c
    _, u,s = func_to_optimize(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = times, chrom_on = chrom_on, new = new, full_res_denoised = full_res_denoised)
    #c = c*(gene_weights.loc[:,g].values*(max_c-min_c))
    u = u*(gene_weights.loc[:,g].values*(max_c-min_c))
    s = s*(gene_weights.loc[:,g].values*(max_c-min_c))
    # offs_u = 0#pars[25]
    # offs_s = 0#pars[26]
    resc_u = pars[24]
    #c = np.ravel(np.sum(c,1))
    u = np.ravel(np.sum(u,1))
    s = np.ravel(np.sum(s,1))
    #weight_c**2*(np.linalg.norm(c-np.ravel(c_all))**2)+
    return (np.linalg.norm(u*resc_u-np.ravel(u_all))**2)+np.linalg.norm(s-np.ravel(s_all))**2     

def opt_all_pars(pars, g, chrom_switches, alpha_cs, scale_ccs, c0s, times = None, chrom_on = None, new = False, rna = None, gene_weights = None, max_c = None, min_c = None, full_res_denoised = None):
    #print(pars)
    e = err_all(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = times, weight_c = 0, chrom_on = chrom_on, new = new, rna = rna, gene_weights = gene_weights, max_c = max_c, min_c = min_c, full_res_denoised = full_res_denoised)
    #print(e)
    return e

def func_to_optimize(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, 
                     times = None, chrom_on = None, new = False, full_res_denoised = None):
    # pseudocode with multivelo
    alphas = pars[:8]
    t_sw1s = pars[8:16]
    t_sw_rnas = pars[16:24]
    #rescale_u = pars[24]
    beta = pars[25]
    gamma = pars[26]
    k = 8
    als = 1.#full_res_denoised.var['fit_alignment_scaling'].loc[g]
    #alpha = full_res.var['fit_alpha'].loc[g]*als
    #beta = full_res_denoised.var['fit_beta'].loc[g]*als
    #gamma = full_res_denoised.var['fit_gamma'].loc[g]*als
    #print(beta)
    #print(gamma)
    #t_ = full_res.var['fit_t_'].loc[g]
    #rescale_u = full_res.var['fit_rescale_u'].loc[g]
    if times is None:
        times = full_res_denoised[:,g].layers['fit_t']
    #ts_ = {}
    c = np.zeros((full_res_denoised.shape[0],8))
    u = np.zeros((full_res_denoised.shape[0],8))
    s = np.zeros((full_res_denoised.shape[0],8))
    for j in range(8):
        # c_cur = minmax(imputed_XC[:,j].layers['Mc'])
        # chrom_switches[j], alpha_cs[j], c0s[j] = optimize_chromatin(times, c_cur)
        t1 = t_sw1s[j]
        t2 = chrom_switches[j]
        t3 = t_sw_rnas[j]
        if new:
            cur_chrom_on = chrom_on[j]
            if t1<cur_chrom_on:
                t1 = cur_chrom_on+0.01
            if t2<cur_chrom_on:
                t2 = cur_chrom_on+0.01
            if t3<cur_chrom_on:
                t3 = cur_chrom_on+0.01
        if t3<t1:
            t3 = t1+0.01
        if t2<t1:
            t1 = t2-0.01
        #print(t1, t2, t3)
        if t2<=t3:
            t_sw_array = np.array([t1, t2, t3])
            model = 1
        else:
            t_sw_array = np.array([t1, t3, t2])
            model = 2
        n_anchors = 500
        anchor_time, tau_list = anchor_points(t_sw_array, 20,
                                              n_anchors, return_time=True)
        #print(t_sw_array)
        switch = np.sum(t_sw_array < 20)
        typed_tau_list = List()
        [typed_tau_list.append(x) for x in tau_list]
        #print(len(typed_tau_list))
        #print([len(l) for l in typed_tau_list])
        #
        if new:
            cur_chrom_on = chrom_on[j]
            #print(cur_chrom_on)
            #print(typed_tau_list[0])
            # if cur_chrom_on>t1:
            #     cur_chrom_on = t1-0.01
            aug = np.sum(typed_tau_list[0]<=cur_chrom_on)
            typed_tau_list[0] = typed_tau_list[0][typed_tau_list[0]>cur_chrom_on]
            aug_exp_list = np.zeros((aug, 3))
            
            #min_time = cur_chrom_on#min(np.concatenate([l for l in typed_tau_list]))
            if len(typed_tau_list[0])>0:
                min_time = min(typed_tau_list[0])
                
                typed_tau_list[0] -= min_time
            else:
                min_time = cur_chrom_on
            for i in range(len(t_sw_array)):
                t_sw_array[i] -= min_time
            #print(j,aug_exp_list.shape)
            #print(typed_tau_list)
        exp_list, exp_sw_list = generate_exp_mine(typed_tau_list,
                                             t_sw_array[:switch],
                                             c0s[j],
                                             alpha_cs[j],
                                             alphas[j],
                                             beta,
                                             gamma,
                                             scale_ccs[j],
                                             model = model)

        
        rescale_factor = np.array([1, 1,
                                   #rescale_u, 
                                   1.0])
        exp_list = [x*rescale_factor for x in exp_list]
        exp_sw_list = [x*rescale_factor for x in exp_sw_list]
        if new:
            exp_list.append(aug_exp_list)
            #print(j,exp_list[-1].shape)
        if new:
            anchor_c = np.ravel(np.concatenate([exp_list[x][:, 0]
                                     for x in range(-1,switch+1)]))
            anchor_u = np.ravel(np.concatenate([exp_list[x][:, 1]
                                         for x in range(-1,switch+1)]))
            anchor_s = np.ravel(np.concatenate([exp_list[x][:, 2]
                                         for x in range(-1,switch+1)]))
        else:
            anchor_c = np.ravel(np.concatenate([exp_list[x][:, 0]
                                     for x in range(switch+1)]))
            anchor_u = np.ravel(np.concatenate([exp_list[x][:, 1]
                                         for x in range(switch+1)]))
            anchor_s = np.ravel(np.concatenate([exp_list[x][:, 2]
                                         for x in range(switch+1)]))
        anchor_c = pd.DataFrame(anchor_c, index = anchor_time)
        anchor_u = pd.DataFrame(anchor_u, index = anchor_time)
        anchor_s = pd.DataFrame(anchor_s, index = anchor_time)
        from scipy.spatial import KDTree
        anchor_time = anchor_time.reshape(-1,1)
        tree = KDTree(anchor_time)
        neighbor_dists, neighbor_indices = tree.query(times.reshape(-1,1))

        interp_c = anchor_c.iloc[neighbor_indices,:]
        interp_u = anchor_u.iloc[neighbor_indices,:]
        interp_s = anchor_s.iloc[neighbor_indices,:]
        c[:,j] = np.ravel(interp_c.values)
        u[:,j] = np.ravel(interp_u.values)
        s[:,j]  = np.ravel(interp_s.values)
    return c, u,s


def optimize_pars(g, x0 = None, times = None, maxiter = 10, verbose = False, new = False, full_res_denoised = None, rna = None, gene_weights = None, max_c = None, min_c = None, unimputed_XC = None):
    if times is None:
        times = full_res_denoised[:,g].layers['fit_t']
    chrom_switches = np.zeros(8)
    alpha_cs = np.zeros(8)
    scale_ccs = np.zeros(8)
    c0s = np.zeros(8)
    if not new:
        chrom_on = None
    else:
        chrom_on = np.zeros(8)
    #pars are alphas, t_sw1s, t_sw_rnas, rescale_us
    for j in range(8):
        c_cur = minmax(unimputed_XC[:,j].layers['Mc'])
        if new:
            chrom_on[j], chrom_switches[j], alpha_cs[j], scale_ccs[j] = optimize_chromatin_new(times, c_cur)
        else:
            chrom_switches[j], alpha_cs[j], scale_ccs[j], c0s[j] = optimize_chromatin(times, c_cur)
    chrom_switches[chrom_switches<0.2] = 0.2
    chrom_switches[chrom_switches>19.8] = 19.8
    bnds = [(0.,300.)]*8+[(0.,chrom_switches[j]) for j in range(8)]+[(0., 20.)]*8+[(0.,2.)]+[(0., 5.)]+[(0.,20.)] #+[(-20,20)]*2
    mod = full_res_denoised[:,g].var['fit_model'].values[0]
    if x0 is None:
        if verbose:
            print('Init')
        alpha = full_res_denoised[:,g].var['fit_alpha'].values[0]
        beta = full_res_denoised[:,g].var['fit_beta'].values[0]
        gamma = full_res_denoised[:,g].var['fit_gamma'].values[0]
        resc = full_res_denoised[:,g].var['fit_rescale_u'].values[0]
        if mod == 1:
            sw_fin = full_res_denoised[:,g].var['fit_t_sw3'].values[0]
        else:
            sw_fin = full_res_denoised[:,g].var['fit_t_sw2'].values[0]
        x0 = np.array([alpha for j in range(8)]+#/(max_total-min_total)
                              [full_res_denoised[:,g].var['fit_t_sw1'].values[0] for j in range(8)]#[0.1]*8+#
                              +[sw_fin for j in range(8)]#[max(10, chrom_switches[j]+0.1) for j in range(8)]#
                              +[resc,beta,gamma])
        #print(x0)
        if verbose:
            print('Error: ', err_all(g, chrom_switches, alpha_cs, scale_ccs, c0s, x0, times = times, weight_c = 0, chrom_on = chrom_on, new = new, rna = rna, gene_weights = gene_weights, max_c = max_c, min_c = min_c, full_res_denoised = full_res_denoised))
    if verbose:
        cb = print_vals
    else:
        cb = None
        
    # res = scipy.optimize.dual_annealing(opt_all_pars, 
    #                               x0 = x0,# 0.1, 0], 
    #                               args = (g,chrom_switches, alpha_cs, scale_ccs, c0s, times, chrom_on, new),
    #                                     #polish = False,
    #                                     #tol = 0.01,
    #                                     bounds = bnds,
    #                                     seed = 57,
    #                                     visit = 2.9,
    #                                     #workers = 1,disp = True,
    #                                     maxiter = maxiter,
    #                                     callback = cb,
    #                                     no_local_search = True)#options = {'maxiter':maxiter}
    
    res = scipy.optimize.minimize(opt_all_pars, 
                                  x0 = x0,# 0.1, 0], 
                                  args = (g,chrom_switches, alpha_cs, scale_ccs, c0s, times, chrom_on, new, rna, gene_weights, max_c, min_c, full_res_denoised),
                                  method = 'Nelder-Mead', options = {'maxiter': maxiter},
                                        #polish = False,
                                        #tol = 0.01,
                                        bounds = bnds)#,
                                        #seed = 57,
                                        #visit = 2.9,
                                        #workers = 1,disp = True,
                                        #maxiter = maxiter,
                                        #callback = cb)#options = {'maxiter':maxiter}
    print('Minimized')
    return res.x, res.fun, chrom_switches, alpha_cs, scale_ccs, chrom_on, c0s



def optimize_all(g, maxiter1 = 30, max_outer_iter = 2, weight_c = 0.6, 
                 verbose = False, plot = False, new = False, full_res_denoised = None, rna = None, gene_weights = None, max_c = None, min_c = None, unimputed_XC = None):
    print('Fitting for '+str(g))
    u_all = np.ravel(rna[:,g].layers['Mu'].copy())
    s_all = np.ravel(rna[:,g].layers['Ms'].copy())
    c_all = pd.DataFrame(unimputed_XC.layers['Mc']).apply(minmax).values.copy()
    c_all = c_all*(gene_weights.loc[:,g].values*(max_c-min_c))
    #c_all = np.sum(c_all,1)
    #c_all = cur_prod.loc[:,g]
    #c_all = c_all-np.min(c_all)
    std_c = np.std(np.sum(c_all,1))
    std_u = np.std(u_all)
    std_s = np.std(s_all)
    scale_c = std_c/std_s
    scale_u = std_u/std_s
    u_all/=scale_u
    c_all/=scale_c
    #c_all*=std_s
    c_all*=weight_c
    
    new_times = full_res_denoised[:,g].layers['fit_t']
    x0 = None
    for i in range(max_outer_iter):
        if verbose:
            print('Outer iteration: '+str(i))
        times = new_times.copy()
        pars, val, chrom_switches, alpha_cs, scale_ccs, chrom_on, c0s = optimize_pars(g, 
                                                                       x0 = x0, 
                                                                       times = times, 
                                                                       maxiter = maxiter1,
                                                                       verbose = verbose, new = new, rna = rna, gene_weights = gene_weights, max_c = max_c, min_c = min_c, unimputed_XC = unimputed_XC, full_res_denoised = full_res_denoised)
                                                                                    
        if verbose:
        #print(chrom_switches, alpha_cs, scale_ccs, c0s)
            print('1', err_all(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = times, weight_c = 0, chrom_on = chrom_on, new = new, rna = rna, gene_weights = gene_weights, max_c = max_c, min_c = min_c, full_res_denoised = full_res_denoised))
        resc_u = pars[24]
        _, u,s = func_to_optimize(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = times, chrom_on = chrom_on, new = new, full_res_denoised = full_res_denoised)
        c = np.zeros(shape = (u.shape[0], 8))
        for i in range(8):
            c_cur = unimputed_XC[:,i].layers['Mc']
            c_cur = minmax(c_cur)
            if new:
                chrom_pars = optimize_chromatin_new(times, c_cur)
                c[:,i] = np.ravel(solve_for_chromatin_new(times, chrom_pars))
            else:
                chrom_pars = optimize_chromatin(times, c_cur)
                c[:,i] = np.ravel(solve_for_chromatin(times, chrom_pars))
        c = c*(gene_weights.loc[:,g].values*(max_c-min_c))
        u = u*(gene_weights.loc[:,g].values*(max_c-min_c))*resc_u#*(max_c-min_c)
        s = s*(gene_weights.loc[:,g].values*(max_c-min_c))
        #c = np.sum(c,1)
        c/=scale_c
        #c = c-np.min(c)
        #c*=std_s
        c*=weight_c
        u = np.sum(u,1)
        s = np.sum(s,1)
        if plot:
            
            plt.figure()
            for i in range(c_all.shape[1]):
                plt.scatter(times, c_all[:,i])
                plt.scatter(times, c[:,i])
            plt.show()
        tree = KDTree(np.concatenate([c, u.reshape(-1,1),s.reshape(-1,1)],1))
        neighbor_dists, neighbor_indices = tree.query(np.concatenate([c_all, u_all.reshape(-1,1),s_all.reshape(-1,1)],1))
        new_times = times[neighbor_indices]
        if plot:
            plt.figure()
            plt.scatter(s_all, u_all, c= times, s = 3)
            plt.scatter(s,u,s = 3)
            plt.show()
            plt.figure()
            plt.scatter(s_all, u_all, c= new_times, s = 3)
            plt.scatter(s,u,s = 3)
            plt.show()
            plt.figure()
            plt.scatter(times, new_times, s = 3)
            plt.show()
        x0 = pars
        if verbose:
            print('2', err_all(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = new_times, weight_c = 0, chrom_on = chrom_on, new = new, rna = rna, gene_weights = gene_weights, max_c = max_c, min_c = min_c, full_res_denoised = full_res_denoised))
        chrom_switches_new = np.zeros(8)
        alpha_cs_new = np.zeros(8)
        scale_ccs_new = np.zeros(8)
        c0s_new = np.zeros(8)
        if not new:
            chrom_on_new = None
        else:
            chrom_on_new = np.zeros(8)
        #pars are alphas, t_sw1s, t_sw_rnas, rescale_us
        for j in range(8):
            c_cur = minmax(unimputed_XC[:,j].layers['Mc'])
            if new:
                chrom_on_new[j], chrom_switches_new[j], alpha_cs_new[j], scale_ccs_new[j] = optimize_chromatin_new(new_times, c_cur)
            else:
                chrom_switches_new[j], alpha_cs_new[j], scale_ccs_new[j], c0s_new[j] = optimize_chromatin(new_times, c_cur)
        chrom_switches_new[chrom_switches_new<0.2] = 0.2
        chrom_switches_new[chrom_switches_new>19.8] = 19.8
        if verbose:
            #print(chrom_switches_new, alpha_cs_new, scale_ccs_new, c0s_new)
            print('3', err_all(g, chrom_switches_new, alpha_cs_new, scale_ccs_new, c0s_new, x0, times = new_times, weight_c = 0, chrom_on = chrom_on, new = new,
                              rna = rna, gene_weights = gene_weights, max_c = max_c, min_c = min_c, full_res_denoised = full_res_denoised))
    return pars, times, (chrom_switches, alpha_cs, scale_ccs, chrom_on, c0s)

#### Velocity

def print_vals(x, f, cont = 0):
    print(str(f))
    return

from multivelo.dynamical_chrom_func import velocity_equations
def compute_velocity_mine(t,
                     t_sw_array,
                     state,
                     c0,
                     alpha_c,
                     alpha,
                     beta,
                     gamma,
                     rescale_c,
                     rescale_u,
                     scale_cc=1,
                     model=1,
                     total_h=20,
                     rna_only=False):
    #print(t_sw_array)
    
    if state is None:
        state0 = t <= t_sw_array[0]
        state1 = (t_sw_array[0] < t) & (t <= t_sw_array[1])
        state2 = (t_sw_array[1] < t) & (t <= t_sw_array[2])
        state3 = t_sw_array[2] < t
    else:
        state0 = np.equal(state, 0)
        state1 = np.equal(state, 1)
        state2 = np.equal(state, 2)
        state3 = np.equal(state, 3)

    tau1 = t[state0]
    tau2 = t[state1] - t_sw_array[0]
    tau3 = t[state2] - t_sw_array[1]
    tau4 = t[state3] - t_sw_array[2]
    tau_list = [tau1, tau2, tau3, tau4]
    switch = np.sum(t_sw_array < total_h)
    typed_tau_list = List()
    [typed_tau_list.append(x) for x in tau_list]
    exp_list, exp_sw_list = generate_exp_mine(typed_tau_list,
                                         t_sw_array[:switch],
                                         c0,
                                         alpha_c,
                                         alpha,
                                         beta,
                                         gamma,
                                         model=model,
                                         scale_cc=scale_cc,
                                         rna_only=rna_only)

    c = np.empty(len(t))
    u = np.empty(len(t))
    s = np.empty(len(t))
    for i, ii in enumerate([state0, state1, state2, state3]):
        #print(i, np.sum(ii))
        if np.any(ii):
            c[ii] = exp_list[i][:, 0]
            u[ii] = exp_list[i][:, 1]
            s[ii] = exp_list[i][:, 2]

    vc_vec = np.zeros(len(u))
    vu_vec = np.zeros(len(u))
    vs_vec = np.zeros(len(u))

    if model == 0:
        if np.any(state0):
            vc_vec[state0], vu_vec[state0], vs_vec[state0] = \
                velocity_equations(c[state0], u[state0], s[state0], alpha_c,
                                   alpha, beta, gamma, pred_r=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
        if np.any(state1):
            vc_vec[state1], vu_vec[state1], vs_vec[state1] = \
                velocity_equations(c[state1], u[state1], s[state1], alpha_c,
                                   alpha, beta, gamma, pred_r=False,
                                   chrom_open=False, scale_cc=scale_cc,
                                   rna_only=rna_only)
        if np.any(state2):
            vc_vec[state2], vu_vec[state2], vs_vec[state2] = \
                velocity_equations(c[state2], u[state2], s[state2], alpha_c,
                                   alpha, beta, gamma, chrom_open=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
        if np.any(state3):
            vc_vec[state3], vu_vec[state3], vs_vec[state3] = \
                velocity_equations(c[state3], u[state3], s[state3], alpha_c, 0,
                                   beta, gamma, chrom_open=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
    elif model == 1:
        if np.any(state0):
            vc_vec[state0], vu_vec[state0], vs_vec[state0] = \
                velocity_equations(c[state0], u[state0], s[state0], alpha_c,
                                   alpha, beta, gamma, pred_r=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
        if np.any(state1):
            vc_vec[state1], vu_vec[state1], vs_vec[state1] = \
                velocity_equations(c[state1], u[state1], s[state1], alpha_c,
                                   alpha, beta, gamma, scale_cc=scale_cc,
                                   rna_only=rna_only)
        if np.any(state2):
            vc_vec[state2], vu_vec[state2], vs_vec[state2] = \
                velocity_equations(c[state2], u[state2], s[state2], alpha_c,
                                   alpha, beta, gamma, chrom_open=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
        if np.any(state3):
            vc_vec[state3], vu_vec[state3], vs_vec[state3] = \
                velocity_equations(c[state3], u[state3], s[state3], alpha_c, 0,
                                   beta, gamma, chrom_open=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
    elif model == 2:
        if np.any(state0):
            vc_vec[state0], vu_vec[state0], vs_vec[state0] = \
                velocity_equations(c[state0], u[state0], s[state0], alpha_c,
                                   alpha, beta, gamma, pred_r=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
        if np.any(state1):
            vc_vec[state1], vu_vec[state1], vs_vec[state1] = \
                velocity_equations(c[state1], u[state1], s[state1], alpha_c,
                                   alpha, beta, gamma, scale_cc=scale_cc,
                                   rna_only=rna_only)
        if np.any(state2):
            vc_vec[state2], vu_vec[state2], vs_vec[state2] = \
                velocity_equations(c[state2], u[state2], s[state2], alpha_c,
                                   0, beta, gamma, scale_cc=scale_cc,
                                   rna_only=rna_only)
        if np.any(state3):
            vc_vec[state3], vu_vec[state3], vs_vec[state3] = \
                velocity_equations(c[state3], u[state3], s[state3], alpha_c, 0,
                                   beta, gamma, chrom_open=False,
                                   scale_cc=scale_cc, rna_only=rna_only)
    return vc_vec * rescale_c, vu_vec * rescale_u, vs_vec


def velocity_full(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = None, full_res_denoised = None):
    # pseudocode with multivelo
    alphas = pars[:8]
    t_sw1s = pars[8:16]
    t_sw_rnas = pars[16:24]
    #rescale_u = pars[24]
    beta = pars[25]
    gamma = pars[26]
    k = 8
    als = 1.#full_res_denoised.var['fit_alignment_scaling'].loc[g]
    #alpha = full_res.var['fit_alpha'].loc[g]*als
    #beta = full_res_denoised.var['fit_beta'].loc[g]*als
    #gamma = full_res_denoised.var['fit_gamma'].loc[g]*als
    #print(beta)
    #print(gamma)
    #t_ = full_res.var['fit_t_'].loc[g]
    #rescale_u = full_res.var['fit_rescale_u'].loc[g]
    if times is None:
        times = full_res_denoised[:,g].layers['fit_t']
    #ts_ = {}
    vc = np.zeros((full_res_denoised.shape[0],8))
    vu = np.zeros((full_res_denoised.shape[0],8))
    vs = np.zeros((full_res_denoised.shape[0],8))
    for j in range(8):
        # c_cur = minmax(imputed_XC[:,j].layers['Mc'])
        # chrom_switches[j], alpha_cs[j], c0s[j] = optimize_chromatin(times, c_cur)
        t1 = t_sw1s[j]
        t2 = chrom_switches[j]
        t3 = t_sw_rnas[j]
        if t3<t1:
            t3 = t1+0.01
        if t2<t1:
            t1 = t2-0.01
        #print(t1, t2, t3)
        if t2<=t3:
            t_sw_array = np.array([t1, t2, t3])
            model = 1
        else:
            t_sw_array = np.array([t1, t3, t2])
            model = 2
        n_anchors = 500
        anchor_time, tau_list = anchor_points(t_sw_array, 20,
                                              n_anchors, return_time=True)
        #print(t_sw_array)
        switch = np.sum(t_sw_array < 20)
        # typed_tau_list = List()
        # [typed_tau_list.append(x) for x in tau_list]
        #print(typed_tau_list)
        #
    
        vc[:,j], vu[:,j], vs[:,j] = compute_velocity_mine(times,#typed_tau_list,
                                             t_sw_array,#[:switch],
                     None,
                     c0s[j],
                     alpha_cs[j],
                     alphas[j],
                     beta,
                     gamma,
                     1,
                     1,
                     scale_cc=scale_ccs[j],
                     model=model,
                     total_h=20,
                     rna_only=False)
    return vc, vu,vs


#### calculate errors

def phase_multivelo(g, model_to_use = None, rna = None):
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
    # if show_switches:
    tt = model_to_use[:,g].layers['fit_t']
    uu = np.ravel(rna[:,g].layers['Mu'])
    ss = np.ravel(rna[:,g].layers['Ms'])
    # c_all = pd.DataFrame(unimputed_XC.layers['Mc']).apply(minmax).values.copy()
    # c_all = c_all*(gene_weights.loc[:,g].values*(max_c-min_c))
    # c_all = np.sum(c_all,1)
    c_all = pd.DataFrame(prod.loc[:,g]).values.copy()#.apply(minmax).values.copy()
    cc = np.ravel(c_all)
    
    from scipy.spatial import KDTree
    new_t = new_t.reshape(-1,1)
    tree = KDTree(new_t)
    neighbor_dists, neighbor_indices = tree.query(tt.reshape(-1,1))
    c_pred = a_c[neighbor_indices]
    u_pred = a_u[neighbor_indices]
    s_pred = a_s[neighbor_indices]
    return tt, uu, ss, cc, new_t, u_pred, s_pred, c_pred

def calc_err_multivelo(g, 
                       model_to_use = None,
                      weight_c = 0.6,
                      plot = False):
    tt, uu, ss, cc, new_t, u_pred, s_pred, c_pred = phase_multivelo(g, model_to_use = model_to_use)
    std_u = np.std(uu)
    std_s = np.std(ss)
    std_c = np.std(cc)
    scale_c = std_c/std_s
    scale_u = std_u/std_s
    #print('Scale c: ', scale_c)
    #c_all*=std_s
    #c_all*=weight_c
    if plot:
        plt.figure()
        plt.scatter(tt, uu/scale_u, label = 'uu')
        plt.scatter(tt, u_pred/scale_u, linewidth=3,
                color='black', alpha=0.5, label = 'a_u')
        plt.legend()
        plt.figure()
        plt.scatter(tt, ss, label = 'ss')
        plt.scatter(tt, s_pred, linewidth=3,
                color='black', alpha=0.5, label = 'a_s')
        plt.legend()
        plt.figure()
        plt.scatter(tt, cc/scale_c, label = 'cc')
        plt.scatter(tt, c_pred/scale_c, linewidth=3,
                color='black', alpha=0.5, label = 'a_c')
        plt.legend()
        print(len(tt))
    # print(np.sum((uu-u_pred)**2)/(scale_u**2))
    # print(np.sum((ss-s_pred)**2))
    # print(cc.shape, c_pred.shape, uu.shape, u_pred.shape)
    # print(np.sum((cc/scale_c-c_pred/scale_c)**2))
    return np.sum((uu-u_pred)**2)/(scale_u**2)+np.sum((ss-s_pred)**2)+(weight_c**2)*np.sum((cc-c_pred)**2)/(scale_c**2)


def err_all_full(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = None, 
                 weight_c = 0.6, plot = False, chrom_on = None, new = False,
                rna = None, gene_weights = None, max_c = None, min_c = None,
                full_res_denoised = None):
    new_t = times
    tt = times
    u_all = rna[:,g].layers['Mu'].copy()
    s_all = rna[:,g].layers['Ms'].copy()
    
#     c_all = pd.DataFrame(unimputed_XC.layers['Mc']).apply(minmax).values.copy()
#     c_all = c_all*(gene_weights.loc[:,g].values*(max_c-min_c))
    
    max_c_total = max(prod.loc[:,g])
    min_c_total = min(prod.loc[:,g])
    c_all = pd.DataFrame(unimputed_XC.layers['Mc']).values.copy()
    c_all = c_all*(gene_weights.loc[:,g].values)
    
    
    #c_all = cur_prod.loc[:,g]
    #c_all = c_all-np.min(c_all)
    c, u,s = func_to_optimize(g, chrom_switches, alpha_cs, scale_ccs, c0s, 
                              pars, times = times, chrom_on = chrom_on, new = new, full_res_denoised = full_res_denoised)
    #c = c*(gene_weights.loc[:,g].values*(max_c-min_c))

    
    
    c = gene_weights.loc[:,g].values*(min_c+c*(max_c-min_c))
        
    u = u*(gene_weights.loc[:,g].values*(max_c-min_c))
    s = s*(gene_weights.loc[:,g].values*(max_c-min_c))
    
    if plot:
        for i in range(8):
            plt.figure()
            plt.scatter(tt, c_all[:,i], label = 'cc')
            plt.scatter(new_t, c[:,i], linewidth=3,
                    color='black', alpha=0.5, label = 'a_c')
            plt.legend()

    c_all = np.ravel(minmax(np.sum(c_all,1)))
    c = np.ravel((np.sum(c,1) -min_c_total)/(max_c_total-min_c_total))
    #c_all = np.sum(c_all,1)
    std_c = np.std(c_all)
    std_u = np.std(u_all)
    std_s = np.std(s_all)
    scale_c = std_c/std_s
    scale_u = std_u/std_s
    c_all/=scale_c
    u_all/=scale_u
    c/=scale_c
    # offs_u = 0#pars[25]
    # offs_s = 0#pars[26]
    resc_u = pars[24]
    #c = np.ravel(np.sum(c,1))
    u = np.ravel(np.sum(u,1))
    s = np.ravel(np.sum(s,1))
    new_t = times
    tt = times
    if plot:
        plt.figure()
        plt.scatter(tt, np.ravel(u_all), label = 'uu')
        plt.scatter(new_t, u*resc_u, linewidth=3,
                color='black', alpha=0.5, label = 'a_u')
        plt.legend()
        plt.figure()
        plt.scatter(tt, np.ravel(s_all), label = 'ss')
        plt.scatter(new_t, s, linewidth=3,
                color='black', alpha=0.5, label = 'a_s')
        plt.legend()
        
        plt.figure()
        plt.scatter(tt, c_all, label = 'cc')
        plt.scatter(new_t, c, s = 1,#linewidth=3,
                color='black', alpha=0.5, label = 'a_c')
        plt.legend()
    #weight_c**2*(np.linalg.norm(c-np.ravel(c_all))**2)+
    return (np.linalg.norm(u*resc_u-np.ravel(u_all))**2)+np.linalg.norm(s-np.ravel(s_all))**2 +(weight_c**2)*np.linalg.norm(c-np.ravel(c_all))**2    





def calc_err_ours(g, res = None, genes = None, weight_c = 0.6, 
                  plot = False, new = False,
                 rna = None, gene_weights = None, max_c = None, min_c = None,
                 full_res_denoised = None):
    i = np.where(genes == g)[0][0]
    pars, new_times, (chrom_switches, alpha_cs, scale_ccs, chrom_on, c0s) = res[i]
    return err_all_full(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = new_times, weight_c = weight_c, chrom_on = chrom_on, new = new, plot = plot,
                       rna = rna, gene_weights = gene_weights, max_c = max_c, min_c = min_c, full_res_denoised = full_res_denoised)

#### plotting

def plot_results(g, model_to_use = None,
                 pointsize = 2,
                 archevelo = False,
                 fig = None,
                 axs = None,
                 ax = None,
                 color = 'black',
                 lw = 2,
                 alpha= 0.6,
                 gray = True,
                 fsize = None,
                 res = None, genes = None, new = False, full_res_denoised = None):
    i = np.where(genes == g)[0][0]
    pars = res[i][0].copy()
    times = res[i][1].copy()
    (chrom_switches, alpha_cs, scale_ccs, chrom_on, c0s) = res[i][2]
    u_all = rna[:,g].layers['Mu'].copy()
    s_all = rna[:,g].layers['Ms'].copy()
    std_u = np.std(u_all)
    std_s = np.std(s_all)
    scale_u = std_u/std_s
    #u_all/=scale_u
    c,u,s = func_to_optimize(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = times,
                            chrom_on = chrom_on, new = new, full_res_denoised = full_res_denoised)
    std_c = np.std(np.sum(c,1))
    scale_c = std_c/std_s
    scale_u = std_u/std_s
    c/=scale_c
    c = c*(gene_weights.loc[:,g].values*(max_c-min_c))
    resc_u = pars[24]
    u = u*(gene_weights.loc[:,g].values*(max_c-min_c))*resc_u
    s = s*(gene_weights.loc[:,g].values*(max_c-min_c))
    #print('Gene: '+str(g))
    offs_u = 0#pars[25]
    offs_s = 0#pars[26]
    #plt.subplot(1,2,1)
    #plt.scatter(s_all, u_all*scale_u, s = 3)
    ordr = np.argsort(np.ravel(times))
    #ax = plt.subplot(1,2,2)
    if not archevelo:
        fig, axs = mv_scatter_plot_return(model_to_use, g, 
                                          pointsize = pointsize,
                                          linewidth=lw,
                                          colr = color,
                                          show_switches=False,
                                          alpha = alpha,
                                          #color_by = 'celltype'
                                          figsize = fsize,
                                          fig = fig,axs = axs
                                          )
        if gray:
            axs[0,0].scatter(s_all, u_all, c = 'darkgray', s = pointsize)
    else:
        ax.plot(np.sum(s,1)[ordr], (np.sum(u,1)*scale_u)[ordr], 
                     c = color, lw = lw,alpha =alpha)
        if gray:
            ax.scatter(s_all, u_all, c = 'darkgray', s = pointsize)
    # sc.pl.umap(rna, color = g)
    # sc.pl.umap(full_res_denoised, color = g, layer = 'fit_t')
    # plt.scatter(umap[:,0], umap[:, 1], c = times, s = 4)
    return fig, axs


def mv_scatter_plot_return(adata,
                 genes,
                 by='us',
                 color_by='state',
                 n_cols=5,
                 axis_on=True,
                 frame_on=True,
                 show_anchors=True,
                 show_switches=True,
                 show_all_anchors=False,
                 title_more_info=False,
                 velocity_arrows=False,
                 downsample=1,
                 figsize=None,
                 pointsize=2,
                 markersize=5,
                 linewidth=2,
                 colr = 'black',
                 alpha = 0.6,          
                 cmap='coolwarm',
                 view_3d_elev=None,
                 view_3d_azim=None,
                 full_name=False,
                           fig = None, axs = None
                 ):
    """Gene scatter plot.

    This function plots phase portraits of the specified plane.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        Anndata result from dynamics recovery.
    genes: `str`,  list of `str`
        List of genes to plot.
    by: `str` (default: `us`)
        Plot unspliced-spliced plane if `us`. Plot chromatin-unspliced plane
        if `cu`.
        Plot 3D phase portraits if `cus`.
    color_by: `str` (default: `state`)
        Color by the four potential states if `state`. Other common values are
        leiden, louvain, celltype, etc.
        If not `state`, the color field must be present in `.uns`, which can be
        pre-computed with `scanpy.pl.scatter`.
        For `state`, red, orange, green, and blue represent state 1, 2, 3, and
        4, respectively.
        When `by=='us'`, `color_by` can also be `c`, which displays the log
        accessibility on U-S phase portraits.
    n_cols: `int` (default: 5)
        Number of columns to plot on each row.
    axis_on: `bool` (default: `True`)
        Whether to show axis labels.
    frame_on: `bool` (default: `True`)
        Whether to show plot frames.
    show_anchors: `bool` (default: `True`)
        Whether to display anchors.
    show_switches: `bool` (default: `True`)
        Whether to show switch times. The three switch times and the end of
        trajectory are indicated by
        circle, cross, dismond, and star, respectively.
    show_all_anchors: `bool` (default: `False`)
        Whether to display full range of (predicted) anchors even for
        repression-only genes.
    title_more_info: `bool` (default: `False`)
        Whether to display model, direction, and likelihood information for
        the gene in title.
    velocity_arrows: `bool` (default: `False`)
        Whether to show velocity arrows of cells on the phase portraits.
    downsample: `int` (default: 1)
        How much to downsample the cells. The remaining number will be
        `1/downsample` of original.
    figsize: `tuple` (default: `None`)
        Total figure size.
    pointsize: `float` (default: 2)
        Point size for scatter plots.
    markersize: `float` (default: 5)
        Point size for switch time points.
    linewidth: `float` (default: 2)
        Line width for connected anchors.
    cmap: `str` (default: `coolwarm`)
        Color map for log accessibilities or other continuous color keys when
        plotting on U-S plane.
    view_3d_elev: `float` (default: `None`)
        Matplotlib 3D plot `elev` argument. `elev=90` is the same as U-S plane,
        and `elev=0` is the same as C-U plane.
    view_3d_azim: `float` (default: `None`)
        Matplotlib 3D plot `azim` argument. `azim=270` is the same as U-S
        plane, and `azim=0` is the same as C-U plane.
    full_name: `bool` (default: `False`)
        Show full names for chromatin, unspliced, and spliced rather than
        using abbreviated terms c, u, and s.
    """
    from pandas.api.types import is_numeric_dtype, is_categorical_dtype
    if by not in ['us', 'cu', 'cus']:
        raise ValueError("'by' argument must be one of ['us', 'cu', 'cus']")
    if color_by == 'state':
        types = [0, 1, 2, 3]
        colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']
    elif by == 'us' and color_by == 'c':
        types = None
    elif color_by in adata.obs and is_numeric_dtype(adata.obs[color_by]):
        types = None
        colors = adata.obs[color_by].values
    elif color_by in adata.obs and is_categorical_dtype(adata.obs[color_by]) \
            and color_by+'_colors' in adata.uns.keys():
        types = adata.obs[color_by].cat.categories
        colors = adata.uns[f'{color_by}_colors']
    else:
        raise ValueError('Currently, color key must be a single string of '
                         'either numerical or categorical available in adata'
                         ' obs, and the colors of categories can be found in'
                         ' adata uns.')

    if 'velo_s_params' not in adata.uns.keys() \
            or 'fit_anchor_s' not in adata.varm.keys():
        show_anchors = False
    if color_by == 'state' and 'fit_state' not in adata.layers.keys():
        raise ValueError('fit_state is not found. Please run '
                         'recover_dynamics_chrom function first or provide a '
                         'valid color key.')

    downsample = np.clip(int(downsample), 1, 10)
    genes = np.array(genes)
    missing_genes = genes[~np.isin(genes, adata.var_names)]
    if len(missing_genes) > 0:
        logg.update(f'{missing_genes} not found', v=0)
    genes = genes[np.isin(genes, adata.var_names)]
    gn = len(genes)
    if gn == 0:
        return
    if gn < n_cols:
        n_cols = gn
    if fig is None and axs is None:
        if by == 'cus':
            fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False,
                                    figsize=(3.2*n_cols, 2.7*(-(-gn // n_cols)))
                                    if figsize is None else figsize,
                                    subplot_kw={'projection': '3d'})
        else:
            fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False,
                                    figsize=(2.7*n_cols, 2.4*(-(-gn // n_cols)))
                                    if figsize is None else figsize)
    fig.patch.set_facecolor('white')
    count = 0
    for gene in genes:
        u = adata[:, gene].layers['Mu'].copy() if 'Mu' in adata.layers \
            else adata[:, gene].layers['unspliced'].copy()
        s = adata[:, gene].layers['Ms'].copy() if 'Ms' in adata.layers \
            else adata[:, gene].layers['spliced'].copy()
        u = u.A if sparse.issparse(u) else u
        s = s.A if sparse.issparse(s) else s
        u, s = np.ravel(u), np.ravel(s)
        if 'ATAC' not in adata.layers.keys() and \
                'Mc' not in adata.layers.keys():
            show_anchors = False
        elif 'ATAC' in adata.layers.keys():
            c = adata[:, gene].layers['ATAC'].copy()
            c = c.A if sparse.issparse(c) else c
            c = np.ravel(c)
        elif 'Mc' in adata.layers.keys():
            c = adata[:, gene].layers['Mc'].copy()
            c = c.A if sparse.issparse(c) else c
            c = np.ravel(c)

        if velocity_arrows:
            if 'velo_u' in adata.layers.keys():
                vu = adata[:, gene].layers['velo_u'].copy()
            elif 'velocity_u' in adata.layers.keys():
                vu = adata[:, gene].layers['velocity_u'].copy()
            else:
                vu = np.zeros(adata.n_obs)
            max_u = np.max([np.max(u), 1e-6])
            u /= max_u
            vu = np.ravel(vu)
            vu /= np.max([np.max(np.abs(vu)), 1e-6])
            if 'velo_s' in adata.layers.keys():
                vs = adata[:, gene].layers['velo_s'].copy()
            elif 'velocity' in adata.layers.keys():
                vs = adata[:, gene].layers['velocity'].copy()
            max_s = np.max([np.max(s), 1e-6])
            s /= max_s
            vs = np.ravel(vs)
            vs /= np.max([np.max(np.abs(vs)), 1e-6])
            if 'velo_chrom' in adata.layers.keys():
                vc = adata[:, gene].layers['velo_chrom'].copy()
                max_c = np.max([np.max(c), 1e-6])
                c /= max_c
                vc = np.ravel(vc)
                vc /= np.max([np.max(np.abs(vc)), 1e-6])

        row = count // n_cols
        col = count % n_cols
        ax = axs[row, col]
        if types is not None:
            for i in range(len(types)):
                if color_by == 'state':
                    filt = adata[:, gene].layers['fit_state'] == types[i]
                else:
                    filt = adata.obs[color_by] == types[i]
                filt = np.ravel(filt)
                if by == 'us':
                    if velocity_arrows:
                        ax.quiver(s[filt][::downsample], u[filt][::downsample],
                                  vs[filt][::downsample],
                                  vu[filt][::downsample], color=colors[i],
                                  alpha=0.5, scale_units='xy', scale=10,
                                  width=0.005, headwidth=4, headaxislength=5.5)
                    else:
                        ax.scatter(s[filt][::downsample],
                                   u[filt][::downsample], s=pointsize,
                                   c=colors[i], alpha=0.7)
                elif by == 'cu':
                    if velocity_arrows:
                        ax.quiver(u[filt][::downsample],
                                  c[filt][::downsample],
                                  vu[filt][::downsample],
                                  vc[filt][::downsample], color=colors[i],
                                  alpha=0.5, scale_units='xy', scale=10,
                                  width=0.005, headwidth=4, headaxislength=5.5)
                    else:
                        ax.scatter(u[filt][::downsample],
                                   c[filt][::downsample], s=pointsize,
                                   c=colors[i], alpha=0.7)
                else:
                    if velocity_arrows:
                        ax.quiver(s[filt][::downsample],
                                  u[filt][::downsample], c[filt][::downsample],
                                  vs[filt][::downsample],
                                  vu[filt][::downsample],
                                  vc[filt][::downsample],
                                  color=colors[i], alpha=0.4, length=0.1,
                                  arrow_length_ratio=0.5, normalize=True)
                    else:
                        ax.scatter(s[filt][::downsample],
                                   u[filt][::downsample],
                                   c[filt][::downsample], s=pointsize,
                                   c=colors[i], alpha=0.7)
        elif color_by == 'c':
            if 'velo_s_params' in adata.uns.keys() and \
                    'outlier' in adata.uns['velo_s_params']:
                outlier = adata.uns['velo_s_params']['outlier']
            else:
                outlier = 99.8
            non_zero = (u > 0) & (s > 0) & (c > 0)
            non_outlier = u < np.percentile(u, outlier)
            non_outlier &= s < np.percentile(s, outlier)
            non_outlier &= c < np.percentile(c, outlier)
            c -= np.min(c)
            c /= np.max(c)
            if velocity_arrows:
                ax.quiver(s[non_zero & non_outlier][::downsample],
                          u[non_zero & non_outlier][::downsample],
                          vs[non_zero & non_outlier][::downsample],
                          vu[non_zero & non_outlier][::downsample],
                          np.log1p(c[non_zero & non_outlier][::downsample]),
                          alpha=0.5,
                          scale_units='xy', scale=10, width=0.005,
                          headwidth=4, headaxislength=5.5, cmap=cmap)
            else:
                ax.scatter(s[non_zero & non_outlier][::downsample],
                           u[non_zero & non_outlier][::downsample],
                           s=pointsize,
                           c=np.log1p(c[non_zero & non_outlier][::downsample]),
                           alpha=0.8, cmap=cmap)
        else:
            if by == 'us':
                if velocity_arrows:
                    ax.quiver(s[::downsample], u[::downsample],
                              vs[::downsample], vu[::downsample],
                              colors[::downsample], alpha=0.5,
                              scale_units='xy', scale=10, width=0.005,
                              headwidth=4, headaxislength=5.5, cmap=cmap)
                else:
                    ax.scatter(s[::downsample], u[::downsample], s=pointsize,
                               c=colors[::downsample], alpha=0.7, cmap=cmap)
            elif by == 'cu':
                if velocity_arrows:
                    ax.quiver(u[::downsample], c[::downsample],
                              vu[::downsample], vc[::downsample],
                              colors[::downsample], alpha=0.5,
                              scale_units='xy', scale=10, width=0.005,
                              headwidth=4, headaxislength=5.5, cmap=cmap)
                else:
                    ax.scatter(u[::downsample], c[::downsample], s=pointsize,
                               c=colors[::downsample], alpha=0.7, cmap=cmap)
            else:
                if velocity_arrows:
                    ax.quiver(s[::downsample], u[::downsample],
                              c[::downsample], vs[::downsample],
                              vu[::downsample], vc[::downsample],
                              colors[::downsample], alpha=0.4, length=0.1,
                              arrow_length_ratio=0.5, normalize=True,
                              cmap=cmap)
                else:
                    ax.scatter(s[::downsample], u[::downsample],
                               c[::downsample], s=pointsize,
                               c=colors[::downsample], alpha=0.7, cmap=cmap)

        if show_anchors:
            min_idx = int(adata[:, gene].var['fit_anchor_min_idx'])
            max_idx = int(adata[:, gene].var['fit_anchor_max_idx'])
            a_c = adata[:, gene].varm['fit_anchor_c']\
                .ravel()[min_idx:max_idx+1].copy()
            a_u = adata[:, gene].varm['fit_anchor_u']\
                .ravel()[min_idx:max_idx+1].copy()
            a_s = adata[:, gene].varm['fit_anchor_s']\
                .ravel()[min_idx:max_idx+1].copy()
            if velocity_arrows:
                a_c /= max_c
                a_u /= max_u
                a_s /= max_s
            if by == 'us':
                ax.plot(a_s, a_u, linewidth=linewidth, color=colr,
                        alpha=alpha, zorder=1000)
            elif by == 'cu':
                ax.plot(a_u, a_c, linewidth=linewidth, color=colr,
                        alpha=alpha, zorder=1000)
            else:
                ax.plot(a_s, a_u, a_c, linewidth=linewidth, color=colr,
                        alpha=alpha, zorder=1000)
            if show_all_anchors:
                a_c_pre = adata[:, gene].varm['fit_anchor_c']\
                    .ravel()[:min_idx].copy()
                a_u_pre = adata[:, gene].varm['fit_anchor_u']\
                    .ravel()[:min_idx].copy()
                a_s_pre = adata[:, gene].varm['fit_anchor_s']\
                    .ravel()[:min_idx].copy()
                if velocity_arrows:
                    a_c_pre /= max_c
                    a_u_pre /= max_u
                    a_s_pre /= max_s
                if len(a_c_pre) > 0:
                    if by == 'us':
                        ax.plot(a_s_pre, a_u_pre, linewidth=linewidth/1.3,
                                color=colr, alpha=alpha, zorder=1000)
                    elif by == 'cu':
                        ax.plot(a_u_pre, a_c_pre, linewidth=linewidth/1.3,
                                color=colr, alpha=alpha, zorder=1000)
                    else:
                        ax.plot(a_s_pre, a_u_pre, a_c_pre,
                                linewidth=linewidth/1.3, color=colr,
                                alpha=alpha, zorder=1000)
            if show_switches:
                t_sw_array = np.array([adata[:, gene].var['fit_t_sw1']
                                      .values[0],
                                      adata[:, gene].var['fit_t_sw2']
                                      .values[0],
                                      adata[:, gene].var['fit_t_sw3']
                                      .values[0]])
                in_range = (t_sw_array > 0) & (t_sw_array < 20)
                a_c_sw = adata[:, gene].varm['fit_anchor_c_sw'].ravel().copy()
                a_u_sw = adata[:, gene].varm['fit_anchor_u_sw'].ravel().copy()
                a_s_sw = adata[:, gene].varm['fit_anchor_s_sw'].ravel().copy()
                if velocity_arrows:
                    a_c_sw /= max_c
                    a_u_sw /= max_u
                    a_s_sw /= max_s
                if in_range[0]:
                    c_sw1, u_sw1, s_sw1 = a_c_sw[0], a_u_sw[0], a_s_sw[0]
                    if by == 'us':
                        ax.plot([s_sw1], [u_sw1], "om", markersize=markersize,
                                zorder=2000)
                    elif by == 'cu':
                        ax.plot([u_sw1], [c_sw1], "om", markersize=markersize,
                                zorder=2000)
                    else:
                        ax.plot([s_sw1], [u_sw1], [c_sw1], "om",
                                markersize=markersize, zorder=2000)
                if in_range[1]:
                    c_sw2, u_sw2, s_sw2 = a_c_sw[1], a_u_sw[1], a_s_sw[1]
                    if by == 'us':
                        ax.plot([s_sw2], [u_sw2], "Xm", markersize=markersize,
                                zorder=2000)
                    elif by == 'cu':
                        ax.plot([u_sw2], [c_sw2], "Xm", markersize=markersize,
                                zorder=2000)
                    else:
                        ax.plot([s_sw2], [u_sw2], [c_sw2], "Xm",
                                markersize=markersize, zorder=2000)
                if in_range[2]:
                    c_sw3, u_sw3, s_sw3 = a_c_sw[2], a_u_sw[2], a_s_sw[2]
                    if by == 'us':
                        ax.plot([s_sw3], [u_sw3], "Dm", markersize=markersize,
                                zorder=2000)
                    elif by == 'cu':
                        ax.plot([u_sw3], [c_sw3], "Dm", markersize=markersize,
                                zorder=2000)
                    else:
                        ax.plot([s_sw3], [u_sw3], [c_sw3], "Dm",
                                markersize=markersize, zorder=2000)
                if max_idx > adata.uns['velo_s_params']['t'] - 4:
                    if by == 'us':
                        ax.plot([a_s[-1]], [a_u[-1]], "*m",
                                markersize=markersize, zorder=2000)
                    elif by == 'cu':
                        ax.plot([a_u[-1]], [a_c[-1]], "*m",
                                markersize=markersize, zorder=2000)
                    else:
                        ax.plot([a_s[-1]], [a_u[-1]], [a_c[-1]], "*m",
                                markersize=markersize, zorder=2000)

        if by == 'cus' and \
                (view_3d_elev is not None or view_3d_azim is not None):
            # US: elev=90, azim=270. CU: elev=0, azim=0.
            ax.view_init(elev=view_3d_elev, azim=view_3d_azim)
        title = gene
        if title_more_info:
            if 'fit_model' in adata.var:
                title += f" M{int(adata[:,gene].var['fit_model'].values[0])}"
            if 'fit_direction' in adata.var:
                title += f" {adata[:,gene].var['fit_direction'].values[0]}"
            if 'fit_likelihood' in adata.var \
                    and not np.all(adata.var['fit_likelihood'].values == -1):
                title += " "
                f"{adata[:,gene].var['fit_likelihood'].values[0]:.3g}"
        ax.set_title(f'{title}', fontsize=11)
        if by == 'us':
            ax.set_xlabel('spliced' if full_name else 's')
            ax.set_ylabel('unspliced' if full_name else 'u')
        elif by == 'cu':
            ax.set_xlabel('unspliced' if full_name else 'u')
            ax.set_ylabel('chromatin' if full_name else 'c')
        elif by == 'cus':
            ax.set_xlabel('spliced' if full_name else 's')
            ax.set_ylabel('unspliced' if full_name else 'u')
            ax.set_zlabel('chromatin' if full_name else 'c')
        if by in ['us', 'cu']:
            if not axis_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            if not frame_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.set_frame_on(False)
        elif by == 'cus':
            if not axis_on:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_zlabel('')
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
                ax.zaxis.set_ticklabels([])
            if not frame_on:
                ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.xaxis._axinfo['tick']['inward_factor'] = 0
                ax.xaxis._axinfo['tick']['outward_factor'] = 0
                ax.yaxis._axinfo['tick']['inward_factor'] = 0
                ax.yaxis._axinfo['tick']['outward_factor'] = 0
                ax.zaxis._axinfo['tick']['inward_factor'] = 0
                ax.zaxis._axinfo['tick']['outward_factor'] = 0
        count += 1
    for i in range(col+1, n_cols):
        fig.delaxes(axs[row, i])
    fig.tight_layout()
    return fig, axs