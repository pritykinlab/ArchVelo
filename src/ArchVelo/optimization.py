import numpy as np
import pandas as pd
import scipy.optimize
from scipy.spatial import KDTree
from numba import njit, float64
from numba.typed import List
from multivelo.dynamical_chrom_func import anchor_points
from .utils import minmax

# --- SECTION 1: JITTED ENGINES ---

@njit(fastmath=True)
def calculate_exact_gene_layers(times, pars, chrom_switches, alpha_c, scale_cc, c0):
    num_comps = len(chrom_switches)
    alphas = pars[:num_comps]
    t_sw1s = pars[num_comps : 2*num_comps]
    diff_sw_rnas = pars[2*num_comps : 3*num_comps]
    beta = pars[3*num_comps + 1]
    gamma = pars[3*num_comps + 2]
    if beta < 1e-12: beta = 1e-12
    if gamma < 1e-12: gamma = 1e-12
    
    n_cells = len(times)
    
    c_out = np.zeros((n_cells, num_comps))
    u_out = np.zeros((n_cells, num_comps))
    s_out = np.zeros((n_cells, num_comps))
    vc_out = np.zeros((n_cells, num_comps))
    vu_out = np.zeros((n_cells, num_comps))
    vs_out = np.zeros((n_cells, num_comps))
    
    for j in range(num_comps):
        # 1. SETUP SWITCHES
        t1 = t_sw1s[j]
        t2 = chrom_switches[j]
        if t2 < t1: t1 = t2 - 0.01
        t3 = t1 + diff_sw_rnas[j]
        if t3 < t1: t3 = t1 + 0.01
        
        m_type = 1
        if t2 <= t3:
            t_sw = np.array([t1, t2, t3])
            m_type = 1
        else:
            t_sw = np.array([t1, t3, t2])
            m_type = 2
            
        # 2. IDENTIFY STATES
        valid_cnt = 0
        for val in t_sw:
            if val < 20.0: valid_cnt += 1
            else: break
        state_indices = np.searchsorted(t_sw[:valid_cnt], times)
        
        # 3. PRE-CALCULATE BOUNDARIES
        val_sw1 = predict_exp_mine(np.array([t_sw[0]]), c0[j], 0, 0, alpha_c[j], alphas[j], beta, gamma, 
                                   pred_r=False, scale_cc=scale_cc[j], rna_only=False)
        c1, u1, s1 = val_sw1[0,0], val_sw1[0,1], val_sw1[0,2]
        
        c2, u2, s2 = c1, u1, s1
        if len(t_sw) >= 2:
            val_sw2 = predict_exp_mine(np.array([t_sw[1]-t_sw[0]]), c1, u1, s1, alpha_c[j], alphas[j], beta, gamma, 
                                       scale_cc=scale_cc[j], rna_only=False)
            c2, u2, s2 = val_sw2[0,0], val_sw2[0,1], val_sw2[0,2]

        c3, u3, s3 = c2, u2, s2
        if len(t_sw) >= 3:
            p2_open = True; p2_alpha = alphas[j]
            if m_type == 1: p2_open = False
            elif m_type == 2: p2_alpha = 0.0
            
            val_sw3 = predict_exp_mine(np.array([t_sw[2]-t_sw[1]]), c2, u2, s2, alpha_c[j], p2_alpha, beta, gamma, 
                                       chrom_open=p2_open, scale_cc=scale_cc[j], rna_only=False)
            c3, u3, s3 = val_sw3[0,0], val_sw3[0,1], val_sw3[0,2]

        # 4. CALCULATE EXACT VALUES PER CELL
        for s_idx in range(4):
            mask = (state_indices == s_idx)
            if not np.any(mask): continue
            
            t_subset = times[mask]
            if s_idx == 0: tau = t_subset
            elif s_idx == 1: tau = t_subset - t_sw[0]
            elif s_idx == 2: tau = t_subset - t_sw[1]
            elif s_idx == 3: tau = t_subset - t_sw[2]

            cur_c0, cur_u0, cur_s0 = c0[j], 0.0, 0.0
            cur_alpha = alphas[j]; cur_open = True; cur_pred_r = True

            if s_idx == 0: cur_pred_r = False
            elif s_idx == 1: cur_c0, cur_u0, cur_s0 = c1, u1, s1
            elif s_idx == 2:
                cur_c0, cur_u0, cur_s0 = c2, u2, s2
                if m_type == 1: cur_open = False
                if m_type == 2: cur_alpha = 0.0
            elif s_idx == 3:
                cur_c0, cur_u0, cur_s0 = c3, u3, s3
                cur_open = False; cur_alpha = 0.0

            res = predict_exp_mine(tau, cur_c0, cur_u0, cur_s0, alpha_c[j], cur_alpha, beta, gamma,
                                   chrom_open=cur_open, scale_cc=scale_cc[j], rna_only=False, pred_r=cur_pred_r)
            
            c_vals, u_vals, s_vals = res[:, 0], res[:, 1], res[:, 2]
            c_out[mask, j] = c_vals
            u_out[mask, j] = u_vals
            s_out[mask, j] = s_vals
            
            # Velocity Logic
            vel_alpha = alphas[j]; vel_open = True; vel_pred_r = True 
            if s_idx == 0: vel_pred_r = False
            elif s_idx == 1: pass
            elif s_idx == 2:
                if m_type == 1: vel_open = False
                elif m_type == 2: vel_alpha = 0.0
            elif s_idx == 3:
                vel_alpha = 0.0; vel_open = False

            vc, vu, vs = velocity_equations_njit(c_vals, u_vals, s_vals, 
                                                 alpha_c[j], vel_alpha, beta, gamma, 
                                                 chrom_open=vel_open, scale_cc=scale_cc[j], 
                                                 rna_only=False, pred_r=vel_pred_r)
            
            vc_out[mask, j] = vc
            vu_out[mask, j] = vu
            vs_out[mask, j] = vs
            
    return c_out, u_out, s_out, vc_out, vu_out, vs_out

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

@njit(fastmath=True)
def predict_exp_mine(tau, c0, u0, s0, alpha_c, alpha, beta, gamma, 
                     scale_cc=1.0, pred_r=True, chrom_open=True, 
                     const_chrom=False, backward=False, rna_only=False):
    if len(tau) == 0: return np.empty((0, 3))
    if backward: tau = -tau
    res = np.empty((len(tau), 3))
    
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

    denom1 = beta - ac_eff
    if abs(denom1) < 1e-12: denom1 = 1e-12
    const = (kc - c0_v) * alpha / denom1

    if not const_chrom:
        res[:, 0] = (kc - (kc - c0_v) * eat)
    else:
        res[:, 0] = 0.0

    if pred_r:
        if not const_chrom:
            res[:, 1] = (u0 * ebt + (alpha * kc / beta) * (1 - ebt) + const * (ebt - eat))
            t1 = s0 * egt
            if abs(gamma) < 1e-12: gamma = 1e-12
            t2 = (alpha * kc / gamma) * (1 - egt)
            d3 = gamma - beta
            if abs(d3) < 1e-12: d3 = 1e-12
            t3 = (beta / d3) * ((alpha * kc / beta) - u0 - const) * (egt - ebt)
            d4 = gamma - ac_eff
            if abs(d4) < 1e-12: d4 = 1e-12
            t4 = (beta / d4) * const * (egt - eat)
            res[:, 2] = (t1 + t2 + t3 + t4)
        else:
            res[:, 1] = u0 * np.ones(len(tau))
            res[:, 2] = s0 * np.ones(len(tau))
    else:
        res[:, 1] = np.zeros(len(tau))
        res[:, 2] = np.zeros(len(tau))
    return res

@njit(fastmath=True)
def generate_exp_mine_njit(tau_list, t_sw_array, c0, alpha_c, alpha, beta, gamma, scale_cc, model, rna_only):
    if beta == alpha_c:
        beta += 1e-3
    if gamma == beta or gamma == alpha_c:
        gamma += 1e-3
        
    switch = len(t_sw_array)
    tau_sw1 = tau_sw2 = tau_sw3 = np.empty(0)
    if switch >= 1:
        tau_sw1 = np.array([t_sw_array[0]])
        if switch >= 2:
            tau_sw2 = np.array([t_sw_array[1] - t_sw_array[0]])
            if switch == 3:
                tau_sw3 = np.array([t_sw_array[2] - t_sw_array[1]])
                
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
        exp1 = predict_exp_mine(tau1, c0, 0, 0, alpha_c, alpha, beta, gamma, pred_r=False, scale_cc=scale_cc, rna_only=rna_only)
        if switch >= 1:
            exp_sw1 = predict_exp_mine(tau_sw1, c0, 0, 0, alpha_c, alpha, beta, gamma, pred_r=False, scale_cc=scale_cc, rna_only=rna_only)
            exp2 = predict_exp_mine(tau2, exp_sw1[0, 0], exp_sw1[0, 1], exp_sw1[0, 2], alpha_c, alpha, beta, gamma, scale_cc=scale_cc, rna_only=rna_only)
            if switch >= 2:
                exp_sw2 = predict_exp_mine(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1], exp_sw1[0, 2], alpha_c, alpha, beta, gamma, scale_cc=scale_cc, rna_only=rna_only)
                exp3 = predict_exp_mine(tau3, exp_sw2[0, 0], exp_sw2[0, 1], exp_sw2[0, 2], alpha_c, alpha, beta, gamma, chrom_open=False, scale_cc=scale_cc, rna_only=rna_only)
                if switch == 3:
                    exp_sw3 = predict_exp_mine(tau_sw3, exp_sw2[0, 0], exp_sw2[0, 1], exp_sw2[0, 2], alpha_c, alpha, beta, gamma, chrom_open=False, scale_cc=scale_cc, rna_only=rna_only)
                    exp4 = predict_exp_mine(tau4, exp_sw3[0, 0], exp_sw3[0, 1], exp_sw3[0, 2], alpha_c, 0, beta, gamma, chrom_open=False, scale_cc=scale_cc, rna_only=rna_only)
                            
    elif model == 2:
        exp1 = predict_exp_mine(tau1, c0, 0, 0, alpha_c, alpha, beta, gamma, pred_r=False, scale_cc=scale_cc, rna_only=rna_only)
        if switch >= 1:
            exp_sw1 = predict_exp_mine(tau_sw1, c0, 0, 0, alpha_c, alpha, beta, gamma, pred_r=False, scale_cc=scale_cc, rna_only=rna_only)
            exp2 = predict_exp_mine(tau2, exp_sw1[0, 0], exp_sw1[0, 1], exp_sw1[0, 2], alpha_c, alpha, beta, gamma, scale_cc=scale_cc, rna_only=rna_only)
            if switch >= 2:
                exp_sw2 = predict_exp_mine(tau_sw2, exp_sw1[0, 0], exp_sw1[0, 1], exp_sw1[0, 2], alpha_c, alpha, beta, gamma, scale_cc=scale_cc, rna_only=rna_only)
                exp3 = predict_exp_mine(tau3, exp_sw2[0, 0], exp_sw2[0, 1], exp_sw2[0, 2], alpha_c, 0, beta, gamma, scale_cc=scale_cc, rna_only=rna_only)
                if switch == 3:
                    exp_sw3 = predict_exp_mine(tau_sw3, exp_sw2[0, 0], exp_sw2[0, 1], exp_sw2[0, 2], alpha_c, 0, beta, gamma, scale_cc=scale_cc, rna_only=rna_only)
                    exp4 = predict_exp_mine(tau4, exp_sw3[0, 0], exp_sw3[0, 1], exp_sw3[0, 2], alpha_c, 0, beta, gamma, chrom_open=False, scale_cc=scale_cc, rna_only=rna_only)
                                            
    return (exp1, exp2, exp3, exp4), (exp_sw1, exp_sw2, exp_sw3)

@njit(fastmath=True)
def find_nearest_indices(array, values):
    idx = np.searchsorted(array, values)
    idx = np.clip(idx, 1, len(array)-1)
    left_dist = np.abs(values - array[idx-1])
    right_dist = np.abs(values - array[idx])
    idx -= (left_dist <= right_dist)
    return idx

@njit(fastmath=True)
def velocity_equations_njit(c, u, s, alpha_c, alpha, beta, gamma, 
                            chrom_open=True, scale_cc=1.0, rna_only=False, pred_r=True):
    vc = np.zeros_like(c)
    vu = np.zeros_like(u)
    vs = np.zeros_like(s)

    if not rna_only:
        if chrom_open:
            kc, rate_c = 1.0, alpha_c
        else:
            kc, rate_c = 0.0, alpha_c * scale_cc
        vc = rate_c * (kc - c)
    
    if pred_r:
        vu = alpha * c - beta * u
        vs = beta * u - gamma * s
        
    return vc, vu, vs

def optimize_chromatin(times, c, seed=57):
    t_np, c_np = np.asarray(times).ravel(), np.asarray(c).ravel()
    res = scipy.optimize.dual_annealing(err_chrom, args=(c_np, t_np), seed=seed, bounds=[(0.,20), (0,10), (0.01,10), (0,1)],
                                        maxiter=1000)
    return res.x


# --- SECTION 2: DECOUPLED WRAPPERS ---

def func_to_optimize(chrom_switches, alpha_cs, scale_ccs, c0s, pars, times):
    num_comps = int((len(pars)-3)/3)
    alphas = pars[:num_comps]
    t_sw1s = pars[num_comps:(2*num_comps)]
    d_sw_rnas = pars[(2*num_comps):(3*num_comps)]
    beta = pars[3*num_comps+1]
    gamma = pars[3*num_comps+2]
    
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

def err_all(chrom_switches, alpha_cs, scale_ccs, c0s, pars, times, u_all_orig, s_all_orig, gn):
    num_comps = len(chrom_switches)
    scale_u = np.std(u_all_orig) / np.std(s_all_orig)
    u_all = u_all_orig / scale_u
    
    _, u, s = func_to_optimize(chrom_switches, alpha_cs, scale_ccs, c0s, pars, times)
    
    u_sum = np.sum(u * gn, 1).ravel()
    s_sum = np.sum(s * gn, 1).ravel()
    resc_u = pars[3 * num_comps]
    
    return np.sum((u_sum * resc_u - u_all)**2) + np.sum((s_sum - s_all_orig)**2)

def opt_all_pars(pars, chrom_switches, alpha_cs, scale_ccs, c0s, times, u_all_orig, s_all_orig, gn):
    return err_all(chrom_switches, alpha_cs, scale_ccs, c0s, pars, times, u_all_orig, s_all_orig, gn)

def optimize_pars(x0, times, maxiter, verbose, u_all_orig, s_all_orig, gn, c_all, var_row, num_comps, method):
    t_np = np.asarray(times).ravel()
    c_sw, a_cs, s_ccs, c0s, switch_neg = [np.zeros(num_comps) for _ in range(5)]
    
    for j in range(num_comps):
        c_sw[j], a_cs[j], s_ccs[j], c0s[j] = optimize_chromatin(t_np, c_all[:, j])
        if c0s[j] < 1 - 1e-12 and a_cs[j] > 1e-12:
            switch_neg[j] = max(np.log(1. - c0s[j]) / a_cs[j], -20)
        else:
            switch_neg[j] = -20
    c_sw = np.clip(c_sw, 0.2, 19.8)

    start = var_row['fit_t_sw1']
    bnds = [(0.,300)]*num_comps + [(switch_neg[j],c_sw[j]) for j in range(num_comps)] + [(0,20-switch_neg[j]) for j in range(num_comps)] + [(1e-12,2.), (1e-12, 5), (1e-12, 20)] 

    if x0 is None:
        sw_f = var_row['fit_t_sw3'] if var_row['fit_model'] == 1 else var_row['fit_t_sw2']
        starts = [min(start, c_sw[j]-0.01) for j in range(num_comps)]
        fins = [sw_f - start for j in range(num_comps)]
        x0 = np.array([var_row['fit_alpha']]*num_comps + starts + fins + [var_row['fit_rescale_u'], var_row['fit_beta'], var_row['fit_gamma']])
        
    res = scipy.optimize.minimize(opt_all_pars, args=(c_sw, a_cs, s_ccs, c0s, t_np, u_all_orig, s_all_orig, gn), 
                                  x0=x0, method=method, options={'maxiter': maxiter}, bounds=bnds)
    return res.x, res.fun, c_sw, a_cs, s_ccs, None, c0s

def optimize_all(u_all_orig, s_all_orig, c_all, new_times, gn, var_row, weight_c, maxiter1, max_outer_iter, method, update_mode, n_anchors=1000, verbose=False):
    num_comps = gn.shape[0]
    c_all_scaled_base = c_all * gn
    
    scale_c = np.std(np.sum(c_all_scaled_base, 1)) / np.std(s_all_orig)
    scale_u = np.std(u_all_orig) / np.std(s_all_orig)
    u_all_norm = u_all_orig / scale_u
    c_all_scaled = (c_all_scaled_base / scale_c) * weight_c
    
    x0 = None
    for i in range(max_outer_iter):
        times = new_times.copy()
        pars, val, c_sw, a_cs, s_ccs, c_on, c0_s = optimize_pars(
            x0, times, maxiter1, verbose, u_all_orig, s_all_orig, gn, c_all, var_row, num_comps, method
        )
        
        if i < max_outer_iter-1:
            resc_u = pars[3*num_comps]
            if update_mode == 'grid':
                anchor_times = np.linspace(0, 20.0, n_anchors)
                c, u, s = func_to_optimize(c_sw, a_cs, s_ccs, c0_s, pars, anchor_times)
                target_times = anchor_times
            else:
                c, u, s = func_to_optimize(c_sw, a_cs, s_ccs, c0_s, pars, times)
                target_times = times

            c = c * gn * weight_c / scale_c
            u = u * gn * resc_u
            s = s * gn
            
            u_sum, s_sum = np.sum(u, 1), np.sum(s, 1)
            tree = KDTree(np.concatenate([c, u_sum.reshape(-1,1), s_sum.reshape(-1,1)], 1))
            _, idx = tree.query(np.concatenate([c_all_scaled, u_all_norm.reshape(-1,1), s_all_orig.reshape(-1,1)], 1))
            
            new_times = target_times[idx]
            x0 = pars
            
    return pars, times, (c_sw, a_cs, s_ccs, c_on, c0_s)