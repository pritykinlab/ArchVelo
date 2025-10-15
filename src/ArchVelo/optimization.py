import numpy as np
import pandas as pd

import scipy.optimize
from scipy.spatial import KDTree
from numba.typed import List

from multivelo.dynamical_chrom_func import anchor_points, velocity_equations

from .utils import minmax, print_vals

import matplotlib.pyplot as plt



def solve_for_chromatin(times, pars):
    switch, alpha_c, scale_cc, c0 = pars
    
    tau = times*(times<switch)+(times-switch)*(times>=switch)
    
    alpha_c_full = alpha_c*(times<switch)+alpha_c*scale_cc*(times>=switch)
    eat = np.exp(-alpha_c_full * tau)
    kc = (times<switch).astype(int)
    c = 1-(1-c0)*np.exp(-alpha_c*switch)
    c = c0+(c-c0)*(times>=switch).astype(int)

    return (kc - (kc - c) * eat)


#pars are switch, alpha_c, scale_cc, c0
def err_chrom(pars, c, times):
    sol = solve_for_chromatin(times, pars)
    return np.linalg.norm(c-sol)**2


def optimize_chromatin(times, c, seed=57):
    res = scipy.optimize.dual_annealing(err_chrom, 
                                        args = (c, times),
                                        seed = seed,
                                        bounds = [(0,20), (0, 10), (0.01,10), (0,1)], 
                                        maxiter = 1000)
    return res.x

def optimize_all(g, 
                 maxiter1 = 1500, 
                 max_outer_iter = 3, 
                 weight_c = 0.3, 
                 verbose = False,
                 plot = False, 
                 full_res_denoised = None, 
                 rna = None, 
                 gene_weights = None, 
                 max_c = None, 
                 min_c = None, arches = None):

    num_comps = gene_weights.shape[0]
    u_all = np.ravel(rna[:,g].layers['Mu'].copy())
    s_all = np.ravel(rna[:,g].layers['Ms'].copy())
    c_all = pd.DataFrame(arches.layers['Mc']).apply(minmax).values.copy()
    c_all = c_all*(gene_weights.loc[:,g].values*(max_c-min_c))

    std_c = np.std(np.sum(c_all,1))
    std_u = np.std(u_all)
    std_s = np.std(s_all)
    scale_c = std_c/std_s
    scale_u = std_u/std_s
    u_all/=scale_u
    c_all/=scale_c
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
                                                                       verbose = verbose, 
                                                                       rna = rna, 
                                                                       gene_weights = gene_weights,
                                                                       max_c = max_c, min_c = min_c, 
                                                                       arches = arches, 
                                                                       full_res_denoised = full_res_denoised)
                                                                                    
        if verbose:
            print('1', err_all(g, 
                               chrom_switches, 
                               alpha_cs, 
                               scale_ccs, 
                               c0s, 
                               pars, 
                               times = times, 
                               chrom_on = chrom_on, 
                               rna = rna, 
                               gene_weights = gene_weights, 
                               max_c = max_c, 
                               min_c = min_c, 
                               full_res_denoised = full_res_denoised))
        resc_u = pars[3*num_comps]
        _, u,s = func_to_optimize(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = times, chrom_on = chrom_on, full_res_denoised = full_res_denoised)
        c = np.zeros(shape = (u.shape[0], num_comps))
        for i in range(num_comps):
            c_cur = arches[:,i].layers['Mc']
            c_cur = minmax(c_cur)
            chrom_pars = optimize_chromatin(times, c_cur)
            c[:,i] = np.ravel(solve_for_chromatin(times, chrom_pars))
        c = c*(gene_weights.loc[:,g].values*(max_c-min_c))
        u = u*(gene_weights.loc[:,g].values*(max_c-min_c))*resc_u#*(max_c-min_c)
        s = s*(gene_weights.loc[:,g].values*(max_c-min_c))
        c/=scale_c
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
            print('2', err_all(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = new_times, chrom_on = chrom_on, rna = rna, gene_weights = gene_weights, max_c = max_c, min_c = min_c, full_res_denoised = full_res_denoised))
        chrom_switches_new = np.zeros(num_comps)
        alpha_cs_new = np.zeros(num_comps)
        scale_ccs_new = np.zeros(num_comps)
        c0s_new = np.zeros(num_comps)
        chrom_on_new = None
        #pars are alphas, t_sw1s, t_sw_rnas, rescale_us
        for j in range(num_comps):
            c_cur = minmax(arches[:,j].layers['Mc'])
            chrom_switches_new[j], alpha_cs_new[j], scale_ccs_new[j], c0s_new[j] = optimize_chromatin(new_times, c_cur)
        chrom_switches_new[chrom_switches_new<0.2] = 0.2
        chrom_switches_new[chrom_switches_new>19.8] = 19.8
        if verbose:
            print('3', err_all(g, chrom_switches_new, alpha_cs_new, scale_ccs_new, c0s_new, x0, times = new_times, chrom_on = chrom_on, rna = rna, gene_weights = gene_weights, max_c = max_c, min_c = min_c, full_res_denoised = full_res_denoised))
    return pars, times, (chrom_switches, alpha_cs, scale_ccs, chrom_on, c0s)

def optimize_pars(g, 
                  x0 = None, 
                  times = None, 
                  maxiter = 10, 
                  verbose = False, 
                  full_res_denoised = None, 
                  rna = None, 
                  gene_weights = None, 
                  max_c = None, 
                  min_c = None, 
                  arches = None):
    if times is None:
        times = full_res_denoised[:,g].layers['fit_t']
    num_comps = gene_weights.shape[0]
    chrom_switches = np.zeros(num_comps)
    alpha_cs = np.zeros(num_comps)
    scale_ccs = np.zeros(num_comps)
    c0s = np.zeros(num_comps)
    chrom_on = None
    #pars are alphas, t_sw1s, t_sw_rnas, rescale_us
    for j in range(num_comps):
        c_cur = minmax(arches[:,j].layers['Mc'])
        chrom_switches[j], alpha_cs[j], scale_ccs[j], c0s[j] = optimize_chromatin(times, c_cur)
    chrom_switches[chrom_switches<0.2] = 0.2
    chrom_switches[chrom_switches>19.8] = 19.8
    bnds = [(0.,300.)]*num_comps+[(0.,chrom_switches[j]) for j in range(num_comps)]+[(0., 20.)]*num_comps+[(0.,2.)]+[(0., 5.)]+[(0.,20.)] 
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
        x0 = np.array([alpha for j in range(num_comps)]+#/(max_total-min_total)
                              [full_res_denoised[:,g].var['fit_t_sw1'].values[0] for j in range(num_comps)]
                              +[sw_fin for j in range(num_comps)]
                              +[resc,beta,gamma])
        if verbose:
            print('Error: ', err_all(g, 
                                     chrom_switches, 
                                     alpha_cs, 
                                     scale_ccs, 
                                     c0s, 
                                     x0, 
                                     times = times, 
                                     chrom_on = chrom_on, 
                                     rna = rna, 
                                     gene_weights = gene_weights, 
                                     max_c = max_c, 
                                     min_c = min_c, 
                                     full_res_denoised = full_res_denoised))
    if verbose:
        cb = print_vals
    else:
        cb = None
            
    res = scipy.optimize.minimize(opt_all_pars, 
                                  x0 = x0,
                                  args = (g,chrom_switches, alpha_cs, scale_ccs, c0s, times, chrom_on, rna, gene_weights, max_c, min_c, full_res_denoised),
                                  method = 'Nelder-Mead', options = {'maxiter': maxiter},
                                        bounds = bnds)
    return res.x, res.fun, chrom_switches, alpha_cs, scale_ccs, chrom_on, c0s

def func_to_optimize(g, 
                     chrom_switches, 
                     alpha_cs, 
                     scale_ccs, 
                     c0s, 
                     pars, 
                     times = None, 
                     chrom_on = None, 
                     full_res_denoised = None):

    num_comps = int((len(pars)-3)/3)
    alphas = pars[:num_comps]
    t_sw1s = pars[num_comps:(2*num_comps)]
    t_sw_rnas = pars[(2*num_comps):(3*num_comps)]
    beta = pars[(3*num_comps)+1]
    gamma = pars[(3*num_comps)+2]
    k = num_comps
    als = 1.
    if times is None:
        times = full_res_denoised[:,g].layers['fit_t']
    c = np.zeros((full_res_denoised.shape[0],num_comps))
    u = np.zeros((full_res_denoised.shape[0],num_comps))
    s = np.zeros((full_res_denoised.shape[0],num_comps))
    for j in range(num_comps):
        t1 = t_sw1s[j]
        t2 = chrom_switches[j]
        t3 = t_sw_rnas[j]
        if t3<t1:
            t3 = t1+0.01
        if t2<t1:
            t1 = t2-0.01
        if t2<=t3:
            t_sw_array = np.array([t1, t2, t3])
            model = 1
        else:
            t_sw_array = np.array([t1, t3, t2])
            model = 2
        n_anchors = 500
        anchor_time, tau_list = anchor_points(t_sw_array, 20,
                                              n_anchors, return_time=True)
        switch = np.sum(t_sw_array < 20)
        typed_tau_list = List()
        [typed_tau_list.append(x) for x in tau_list]
        
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
        
        anchor_c = np.ravel(np.concatenate([exp_list[x][:, 0]
                                 for x in range(switch+1)]))
        anchor_u = np.ravel(np.concatenate([exp_list[x][:, 1]
                                     for x in range(switch+1)]))
        anchor_s = np.ravel(np.concatenate([exp_list[x][:, 2]
                                     for x in range(switch+1)]))
        anchor_c = pd.DataFrame(anchor_c, index = anchor_time)
        anchor_u = pd.DataFrame(anchor_u, index = anchor_time)
        anchor_s = pd.DataFrame(anchor_s, index = anchor_time)
        anchor_time = anchor_time.reshape(-1,1)
        tree = KDTree(anchor_time)
        neighbor_dists, neighbor_indices = tree.query(times.reshape(-1,1))

        interp_c = anchor_c.iloc[neighbor_indices,:]
        interp_u = anchor_u.iloc[neighbor_indices,:]
        interp_s = anchor_s.iloc[neighbor_indices,:]
        c[:,j] = np.ravel(interp_c.values)
        u[:,j] = np.ravel(interp_u.values)
        s[:,j]  = np.ravel(interp_s.values)
    return c, u, s

def opt_all_pars(pars, g, chrom_switches, alpha_cs, scale_ccs, c0s, times = None, chrom_on = None, rna = None, gene_weights = None, max_c = None, min_c = None, full_res_denoised = None):
    e = err_all(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = times, chrom_on = chrom_on, rna = rna, gene_weights = gene_weights, max_c = max_c, min_c = min_c, full_res_denoised = full_res_denoised)
    return e

def err_all(g, 
            chrom_switches, 
            alpha_cs, 
            scale_ccs, 
            c0s, 
            pars, 
            times = None, 
            chrom_on = None, 
            rna = None, 
            gene_weights = None, 
            max_c = None, 
            min_c = None, 
            full_res_denoised = None):
    
    num_comps = gene_weights.shape[0]
    
    u_all = rna[:,g].layers['Mu'].copy()
    s_all = rna[:,g].layers['Ms'].copy()
    
    std_u = np.std(u_all)
    std_s = np.std(s_all)
    scale_u = std_u/std_s
    
    u_all/=scale_u
    
    _, u,s = func_to_optimize(g, 
                              chrom_switches, 
                              alpha_cs, 
                              scale_ccs, 
                              c0s, 
                              pars, 
                              times = times, 
                              chrom_on = chrom_on, 
                              full_res_denoised = full_res_denoised)
    
    u = u*(gene_weights.loc[:,g].values*(max_c-min_c))
    s = s*(gene_weights.loc[:,g].values*(max_c-min_c))
    
    resc_u = pars[3*num_comps]
    
    u = np.ravel(np.sum(u,1))
    s = np.ravel(np.sum(s,1))

    return (np.linalg.norm(u*resc_u-np.ravel(u_all))**2)+np.linalg.norm(s-np.ravel(s_all))**2 

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
