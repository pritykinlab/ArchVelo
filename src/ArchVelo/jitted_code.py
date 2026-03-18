import numpy as np
import pandas as pd
import scipy.optimize
from scipy.spatial import KDTree
from numba import njit, float64
from numba.typed import List
from multivelo.dynamical_chrom_func import anchor_points, velocity_equations
from .utils import minmax, print_vals
import matplotlib.pyplot as plt

# --- SECTION 1: JITTED ENGINES ---

@njit(fastmath=True)
def solve_for_chromatin(times, pars):
    switch, alpha_c, scale_cc, c0 = pars
    tau = times*(times<switch)+(times-switch)*(times>=switch)
    alpha_c_full = alpha_c*(times<switch)+alpha_c*scale_cc*(times>=switch)
    eat = np.exp(-alpha_c_full * tau)
    kc = (times<switch).astype(np.float64)
    c_sw = 1.0-(1.0-c0)*np.exp(-alpha_c*switch)
    c_val = c0+(c_sw-c0)*(times>=switch).astype(np.float64)
    return (kc - (kc - c_val) * eat)

@njit(fastmath=True)
def err_chrom(pars, c, times):
    sol = solve_for_chromatin(times, pars)
    return np.sum((c - sol)**2)

# def err_chrom(pars, c, times):
#     sol = solve_for_chromatin(times, pars)
#     return np.linalg.norm(c-sol)**2

@njit(fastmath=True)
def predict_exp_mine(tau, c0, u0, s0, alpha_c, alpha, beta, gamma, 
                     scale_cc=1.0, pred_r=True, chrom_open=True, 
                     const_chrom=False, backward=False, rna_only=False):
    if len(tau) == 0: return np.empty((0, 3))
    if backward: tau = -tau
    res = np.empty((len(tau), 3))
    
    # Strictly scoped local rate selection
    ac_eff = alpha_c
    if rna_only:
        kc, c0_v = 1.0, 1.0
    else:
        if chrom_open:
            kc, c0_v = 1.0, c0
        else:
            kc, c0_v = 0.0, c0
            ac_eff = alpha_c * scale_cc
            
    eat = np.exp(-ac_eff * tau)
    ebt = np.exp(-beta * tau)
    egt = np.exp(-gamma * tau)
    
    # analytical lag constant
    denom1 = beta - ac_eff
    if abs(denom1) < 1e-12: denom1 = 1e-12
    const = (kc - c0_v) * alpha / denom1

    if not const_chrom:
        res[:, 0] = kc - (kc - c0_v) * eat
    else:
        res[:, 0] = 0.0

    if pred_r:
        if not const_chrom:
            res[:, 1] = u0 * ebt + (alpha * kc / beta) * (1 - ebt) + const * (ebt - eat)
            t1 = s0 * egt
            t2 = (alpha * kc / gamma) * (1 - egt)
            d3 = gamma - beta
            if abs(d3) < 1e-12: d3 = 1e-12
            t3 = (beta / d3) * ((alpha * kc / beta) - u0 - const) * (egt - ebt)
            d4 = gamma - ac_eff
            if abs(d4) < 1e-12: d4 = 1e-12
            t4 = (beta / d4) * const * (egt - eat)
            res[:, 2] = t1 + t2 + t3 + t4
        else:
            res[:, 1] = u0 * np.ones(len(tau))
            res[:, 2] = s0 * np.ones(len(tau))
    else:
        res[:, 1] = np.zeros(len(tau))
        res[:, 2] = np.zeros(len(tau))
    return res

@njit(fastmath=True)
def generate_exp_mine_njit(tau_list, t_sw_array, c0, alpha_c, alpha, beta, gamma, scale_cc, model, rna_only):
    # Literal: beta and gamma equality checks and shifts
    if beta == alpha_c:
        beta += 1e-3
    if gamma == beta or gamma == alpha_c:
        gamma += 1e-3
        
    switch = len(t_sw_array)
    
    # Initialize switching taus
    tau_sw1 = np.empty(0)
    tau_sw2 = np.empty(0)
    tau_sw3 = np.empty(0)
    if switch >= 1:
        tau_sw1 = np.array([t_sw_array[0]])
        if switch >= 2:
            tau_sw2 = np.array([t_sw_array[1] - t_sw_array[0]])
            if switch == 3:
                tau_sw3 = np.array([t_sw_array[2] - t_sw_array[1]])
                
    # Initialize empty arrays for all possible outputs
    exp_sw1 = exp_sw2 = exp_sw3 = np.empty((0, 3))
    exp1 = exp2 = exp3 = exp4 = np.empty((0, 3))
    
    tau1 = tau_list[0]
    tau2 = tau3 = tau4 = np.empty(0)
    if switch >= 1:
        tau2 = tau_list[1]
        if switch >= 2:
            tau3 = tau_list[2]
            if switch == 3:
                tau4 = tau_list[3]

    if model == 1:
        # Segment 1: pred_r=False
        exp1 = predict_exp_mine(tau1, c0, 0, 0, alpha_c, alpha, beta, gamma, 
                                pred_r=False, scale_cc=scale_cc, rna_only=rna_only)
        if switch >= 1:
            # Boundary 1: pred_r=False
            exp_sw1 = predict_exp_mine(tau_sw1, c0, 0, 0, alpha_c, alpha, beta, gamma, 
                                       pred_r=False, scale_cc=scale_cc, rna_only=rna_only)
            # Segment 2
            exp2 = predict_exp_mine(tau2, exp_sw1[0, 0], exp_sw1[0, 1], exp_sw1[0, 2], 
                                    alpha_c, alpha, beta, gamma, scale_cc=scale_cc, rna_only=rna_only)
            if switch >= 2:
                # Boundary 2: Logic lock - your code does NOT pass is_open here, it uses default (True)
                exp_sw2 = predict_exp_mine(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1], exp_sw1[0, 2], 
                                           alpha_c, alpha, beta, gamma, scale_cc=scale_cc, rna_only=rna_only)
                # Segment 3: chrom_open=False
                exp3 = predict_exp_mine(tau3, exp_sw2[0, 0], exp_sw2[0, 1], exp_sw2[0, 2], 
                                        alpha_c, alpha, beta, gamma, chrom_open=False, scale_cc=scale_cc, rna_only=rna_only)
                if switch == 3:
                    # Boundary 3: chrom_open=False
                    exp_sw3 = predict_exp_mine(tau_sw3, exp_sw2[0, 0], exp_sw2[0, 1], exp_sw2[0, 2], 
                                               alpha_c, alpha, beta, gamma, chrom_open=False, scale_cc=scale_cc, rna_only=rna_only)
                    # Segment 4: alpha=0, chrom_open=False
                    exp4 = predict_exp_mine(tau4, exp_sw3[0, 0], exp_sw3[0, 1], exp_sw3[0, 2], 
                                            alpha_c, 0, beta, gamma, chrom_open=False, scale_cc=scale_cc, rna_only=rna_only)
                                            
    elif model == 2:
        # Segment 1: pred_r=False
        exp1 = predict_exp_mine(tau1, c0, 0, 0, alpha_c, alpha, beta, gamma, 
                                pred_r=False, scale_cc=scale_cc, rna_only=rna_only)
        if switch >= 1:
            # Boundary 1: pred_r=False
            exp_sw1 = predict_exp_mine(tau_sw1, c0, 0, 0, alpha_c, alpha, beta, gamma, 
                                       pred_r=False, scale_cc=scale_cc, rna_only=rna_only)
            # Segment 2
            exp2 = predict_exp_mine(tau2, exp_sw1[0, 0], exp_sw1[0, 1], exp_sw1[0, 2], 
                                    alpha_c, alpha, beta, gamma, scale_cc=scale_cc, rna_only=rna_only)
            if switch >= 2:
                # Boundary 2
                exp_sw2 = predict_exp_mine(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1], exp_sw1[0, 2], 
                                           alpha_c, alpha, beta, gamma, scale_cc=scale_cc, rna_only=rna_only)
                # Segment 3: alpha=0
                exp3 = predict_exp_mine(tau3, exp_sw2[0, 0], exp_sw2[0, 1], exp_sw2[0, 2], 
                                        alpha_c, 0, beta, gamma, scale_cc=scale_cc, rna_only=rna_only)
                if switch == 3:
                    # Boundary 3: alpha=0
                    exp_sw3 = predict_exp_mine(tau_sw3, exp_sw2[0, 0], exp_sw2[0, 1], exp_sw2[0, 2], 
                                               alpha_c, 0, beta, gamma, scale_cc=scale_cc, rna_only=rna_only)
                    # Segment 4: alpha=0, chrom_open=False
                    exp4 = predict_exp_mine(tau4, exp_sw3[0, 0], exp_sw3[0, 1], exp_sw3[0, 2], 
                                            alpha_c, 0, beta, gamma, chrom_open=False, scale_cc=scale_cc, rna_only=rna_only)
                                            
    return (exp1, exp2, exp3, exp4), (exp_sw1, exp_sw2, exp_sw3)

# --- SECTION 2: WRAPPERS AND LOGIC ---

def optimize_chromatin(times, c, seed=57):
    t_np, c_np = np.asarray(times).ravel(), np.asarray(c).ravel()
    res = scipy.optimize.dual_annealing(err_chrom, args=(c_np, t_np), seed=seed, bounds=[(0,20), (0,10), (0.01,10), (0,1)], maxiter=1000)
    return res.x

@njit(fastmath=True)
def find_nearest_indices(array, values):
    # Find insertion points
    idx = np.searchsorted(array, values)
    # Clip to valid range
    idx = np.clip(idx, 1, len(array)-1)
    # Calculate distance to left (idx-1) and right (idx) neighbors
    left_dist = np.abs(values - array[idx-1])
    right_dist = np.abs(values - array[idx])
    # Choose closer neighbor
    # If left is closer, decrement index
    # (Using boolean subtraction: True=1, False=0)
    idx -= (left_dist <= right_dist)
    return idx

def func_to_optimize(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times=None, chrom_on=None, full_res_denoised=None):
    num_comps = int((len(pars)-3)/3)
    alphas, t_sw1s, d_sw_rnas = pars[:num_comps], pars[num_comps:(2*num_comps)], pars[(2*num_comps):(3*num_comps)]
    beta, gamma = pars[3*num_comps+1], pars[3*num_comps+2]
    if times is None: times = full_res_denoised[:,g].layers['fit_t']
    t_np = np.asarray(times).ravel()
    c_out, u_out, s_out = [np.zeros((len(t_np), num_comps)) for _ in range(3)]
    for j in range(num_comps):
        t1, t2 = t_sw1s[j], chrom_switches[j]
        if t2 < t1: t1 = t2 - 0.01
        t3 = t1 + d_sw_rnas[j]
        if t3 < t1: t3 = t1 + 0.01
        t_sw = np.array([t1, t2, t3]) if t2 <= t3 else np.array([t1, t3, t2])
        m_type = 1 if t2 <= t3 else 2
        anc_t, tau_raw = anchor_points(t_sw, 20, 500, return_time=True)
        t_l = List(); [t_l.append(x) for x in tau_raw]
        exp_l, _ = generate_exp_mine_njit(t_l, t_sw[t_sw < 20], c0s[j], alpha_cs[j], alphas[j], beta, gamma, scale_ccs[j], m_type, False)
        anc_c, anc_u, anc_s = [np.concatenate([x[:,i] for x in exp_l]) for i in range(3)]
        idx = find_nearest_indices(anc_t, t_np)
        c_out[:, j], u_out[:, j], s_out[:, j] = anc_c[idx], anc_u[idx], anc_s[idx]
    return c_out, u_out, s_out

def opt_all_pars(pars, g, chrom_switches, alpha_cs, scale_ccs, c0s, times, chrom_on, rna, gene_weights, max_c, min_c, full_res_denoised):
    return err_all(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times, chrom_on, rna, gene_weights, max_c, min_c, full_res_denoised)

def err_all(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times=None, chrom_on=None, rna=None, gene_weights=None, max_c=None, min_c=None, full_res_denoised=None):
    num_comps = gene_weights.shape[0]
    u_all = np.ravel(rna[:,g].layers['Mu'].copy())
    s_all = np.ravel(rna[:,g].layers['Ms'].copy())
    scale_u = np.std(u_all) / np.std(s_all)
    u_all /= scale_u
    _, u, s = func_to_optimize(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times, chrom_on, full_res_denoised)
    gn = gene_weights.loc[:,g].values * (max_c - min_c)
    u_sum = np.sum(u * gn, 1).ravel()
    s_sum = np.sum(s * gn, 1).ravel()
    resc_u = pars[3 * num_comps]
    return np.sum((u_sum * resc_u - u_all)**2) + np.sum((s_sum - s_all)**2)

def optimize_pars(g, x0=None, times=None, maxiter=10, verbose=False, full_res_denoised=None, rna=None, gene_weights=None, max_c=None, min_c=None, arches=None, method='Nelder-Mead'):
    num_comps = gene_weights.shape[0]
    t_np = np.asarray(times).ravel()
    c_sw, a_cs, s_ccs, c0s = [np.zeros(num_comps) for _ in range(4)]
    for j in range(num_comps):
        c_sw[j], a_cs[j], s_ccs[j], c0s[j] = optimize_chromatin(t_np, minmax(arches[:,j].layers['Mc']))
    c_sw = np.clip(c_sw, 0.2, 19.8)
    bnds = [(0.,300.)]*num_comps + [(0.,c_sw[j]) for j in range(num_comps)] + [(0., 20.)]*num_comps + [(0.,2.), (0., 5.), (0.,20.)] 
    if x0 is None:
        if verbose: print('Init')
        v = full_res_denoised[:,g].var
        sw_f = v['fit_t_sw3'].values[0] if v['fit_model'].values[0] == 1 else v['fit_t_sw2'].values[0]
        starts = [min(v['fit_t_sw1'].values[0], c_sw[j]-0.01) for j in range(num_comps)]
        fins = [sw_f - starts[j] for j in range(num_comps)]
        x0 = np.array([v['fit_alpha'].values[0]]*num_comps + starts + fins + [v['fit_rescale_u'].values[0], v['fit_beta'].values[0], v['fit_gamma'].values[0]])
        if verbose: print('Error: ', err_all(g, c_sw, a_cs, s_ccs, c0s, x0, t_np, None, rna, gene_weights, max_c, min_c, full_res_denoised))
    res = scipy.optimize.minimize(opt_all_pars, args=(g, c_sw, a_cs, s_ccs, c0s, t_np, None, rna, gene_weights, max_c, min_c, full_res_denoised), 
                                  x0=x0, method=method, options={'maxiter': maxiter}, bounds=bnds)
    return res.x, res.fun, c_sw, a_cs, s_ccs, None, c0s

def optimize_all(g, maxiter1=1500, max_outer_iter=3, weight_c=0.3, verbose=False, plot=False, full_res_denoised=None, rna=None, gene_weights=None, max_c=None, min_c=None, arches=None, method='Nelder-Mead'):
    num_comps = gene_weights.shape[0]
    u_all_orig, s_all_orig = np.ravel(rna[:,g].layers['Mu'].copy()), np.ravel(rna[:,g].layers['Ms'].copy())
    c_all = pd.DataFrame(arches.layers['Mc']).apply(minmax).values.copy()
    gn = gene_weights.loc[:,g].values * (max_c - min_c)
    c_all *= gn
    scale_c, scale_u = np.std(np.sum(c_all, 1)) / np.std(s_all_orig), np.std(u_all_orig) / np.std(s_all_orig)
    u_all_norm, c_all_scaled = u_all_orig / scale_u, (c_all / scale_c) * weight_c
    new_times = full_res_denoised[:,g].layers['fit_t']
    x0 = None
    for i in range(max_outer_iter):
        if verbose: print('Outer iteration: ' + str(i))
        times = new_times.copy()
        pars, val, c_sw, a_cs, s_ccs, c_on, c0_s = optimize_pars(g, x0, times, maxiter1, verbose, full_res_denoised, rna, gene_weights, max_c, min_c, arches, method)
        if verbose: print('1', err_all(g, c_sw, a_cs, s_ccs, c0_s, pars, times, c_on, rna, gene_weights, max_c, min_c, full_res_denoised))
        resc_u = pars[3*num_comps]
        c, u, s = func_to_optimize(g, c_sw, a_cs, s_ccs, c0_s, pars, times, c_on, full_res_denoised)
        # print(c_func[:30,0])
        # c = np.zeros((u.shape[0], num_comps))
        # for j in range(num_comps):
        #     chrom_pars = optimize_chromatin(times, minmax(arches[:, j].layers['Mc']))
        #     c[:, j] = np.ravel(solve_for_chromatin(times, chrom_pars))
        # print(c[:30,0])
        c = c * gn * weight_c / scale_c
        u = u * gn * resc_u
        s = s * gn
        u_sum, s_sum = np.sum(u, 1), np.sum(s, 1)
        tree = KDTree(np.concatenate([c, u_sum.reshape(-1,1), s_sum.reshape(-1,1)], 1))
        _, idx = tree.query(np.concatenate([c_all_scaled, u_all_norm.reshape(-1,1), s_all_orig.reshape(-1,1)], 1))
        print(np.concatenate([c, u_sum.reshape(-1,1), s_sum.reshape(-1,1)], 1)[:10, :])
        print(np.concatenate([c_all_scaled, u_all_norm.reshape(-1,1), s_all_orig.reshape(-1,1)], 1)[:10, :])
        print(idx)
        new_times = times[idx]
        x0 = pars
        if verbose: 
            print('2', err_all(g, c_sw, a_cs, s_ccs, c0_s, pars, new_times, c_on, rna, gene_weights, max_c, min_c, full_res_denoised))
            # c_sw_new, a_cs_new, s_ccs_new, c0s_new = [np.zeros(num_comps) for _ in range(4)]
            # for j in range(num_comps):
            #     c_sw_new[j], a_cs_new[j], s_ccs_new[j], c0s_new[j] = optimize_chromatin(new_times, minmax(arches[:, j].layers['Mc']))
            # #c_sw, a_cs, s_ccs, c0_s = np.clip(c_sw_new, 0.2, 19.8), a_cs_new, s_ccs_new, c0s_new
            # print('3', err_all(g, c_sw, a_cs, s_ccs, c0_s, x0, new_times, c_on, rna, gene_weights, max_c, min_c, full_res_denoised))
    return pars, times, (c_sw, a_cs, s_ccs, c_on, c0_s)

# def compute_velocity_mine(t, t_sw_array, state, c0, alpha_c, alpha, beta, gamma, rescale_c, rescale_u, scale_cc=1, model=1, total_h=20, rna_only=False):
#     t_np = np.asarray(t).ravel()
    
#     # State identification
#     if state is None:
#         state0 = t_np <= t_sw_array[0]
#         state1 = (t_sw_array[0] < t_np) & (t_np <= t_sw_array[1])
#         state2 = (t_sw_array[1] < t_np) & (t_np <= t_sw_array[2])
#         state3 = t_sw_array[2] < t_np
#     else:
#         state0, state1, state2, state3 = np.equal(state, 0), np.equal(state, 1), np.equal(state, 2), np.equal(state, 3)

#     tau_list = [t_np[state0], t_np[state1] - t_sw_array[0], t_np[state2] - t_sw_array[1], t_np[state3] - t_sw_array[2]]
#     switch = np.sum(t_sw_array < total_h)
    
#     # Typed list for Numba compatibility
#     t_l = List()
#     for x in tau_list:
#         t_l.append(x)
    
#     # Get trajectories
#     exp_list, _ = generate_exp_mine_njit(t_l, t_sw_array[:switch], c0, alpha_c, alpha, beta, gamma, scale_cc, model, rna_only)

#     c, u, s = np.zeros(len(t_np)), np.zeros(len(t_np)), np.zeros(len(t_np))
#     masks = [state0, state1, state2, state3]
    
#     # 1. Fill Trajectories - This is where the indexing error usually happens
#     for i in range(len(masks)):
#         mask = masks[i]
#         if np.any(mask):
#             segment = exp_list[i]
#             # Ensure the segment actually has the expected columns before indexing
#             if segment.ndim == 2 and segment.shape[1] == 3:
#                 c[mask] = segment[:, 0]
#                 u[mask] = segment[:, 1]
#                 s[mask] = segment[:, 2]

#     vc_vec, vu_vec, vs_vec = np.zeros(len(u)), np.zeros(len(u)), np.zeros(len(u))

#     # 2. Calculate Velocities using the velocity_equations engine
#     for i in range(len(masks)):
#         mask = masks[i]
#         if not np.any(mask): 
#             continue
        
#         # Determine the correct production (alpha) and chromatin state for each phase
#         # Phase 0: Always open, alpha on
#         # Phase 1: Model-dependent, alpha on
#         # Phase 2: Closed (if model 1) or Open (if model 2 with alpha=0), alpha on/off
#         # Phase 3: Always alpha off
        
#         if model == 0:
#             is_open, a_fact = (i == 0), (i < 3)
#         elif model == 1:
#             is_open, a_fact = (i < 2), (i < 3)
#         else: # model == 2
#             is_open, a_fact = (i < 3), (i < 2)
            
#         curr_alpha = alpha if a_fact else 0.0
        
#         # Call the differential equation system
#         vc, vu, vs = velocity_equations(
#             c[mask], u[mask], s[mask], alpha_c, curr_alpha, beta, gamma,
#             chrom_open=is_open, scale_cc=scale_cc, rna_only=rna_only
#         )
        
#         vc_vec[mask], vu_vec[mask], vs_vec[mask] = vc, vu, vs

#     return vc_vec * rescale_c, vu_vec * rescale_u, vs_vec

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
    exp_list, exp_sw_list = generate_exp_mine_njit(typed_tau_list,
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
