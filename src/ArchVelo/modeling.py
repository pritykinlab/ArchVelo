import numpy as np
import pandas as pd
import multivelo as mv
import pickle
import os
import anndata
from joblib import Parallel, delayed

from .preprocessing import annotate_and_summarize, smooth_archetypes, extract_wnn_connectivities, gen_wnn
from .optimization import optimize_all, func_to_optimize, calculate_exact_gene_layers
from .evaluation import set_likelihood
from .utils import extract_minmax, minmax, cells_to_keep
from multivelo.dynamical_chrom_func import smooth_scale

# --- TOP-LEVEL PARALLEL WORKERS ---

def _extract_worker(i, Mu_arr_full, Ms_arr_full, c_all, fit_t_arr_full, gw_vals_full, var_row, weight_c, maxiter1, max_outer_iter, method, update_mode):
    # Slice inside the worker to prevent joblib from copying memory
    u_orig = Mu_arr_full[:, i]
    s_orig = Ms_arr_full[:, i]
    times = fit_t_arr_full[:, i]
    gn = gw_vals_full[:, i]
    
    return optimize_all(u_orig, s_orig, c_all, times, gn, var_row, weight_c, maxiter1, max_outer_iter, method, update_mode, 1000, False)

def _velocity_worker(i, Mu_arr_full, Ms_arr_full, entry, num_comps, gw_vals_full, rna_conn):
    try:
        # Slice inside the worker to prevent joblib from copying memory
        u_raw = Mu_arr_full[:, i]
        s_raw = Ms_arr_full[:, i]
        norm_const = gw_vals_full[:, i]
        
        std_u, std_s = np.std(u_raw), np.std(s_raw)
        if std_u == 0: std_u = 1.0
        
        pars = entry[0]
        times = np.ravel(entry[1])
        chrom_switches, alpha_cs, scale_ccs, _, c0s = entry[2]
        
        c, u, s, vc, vu, vs = calculate_exact_gene_layers(
            times, pars, chrom_switches, alpha_cs, scale_ccs, c0s
        )
        
        resc_u = pars[3*num_comps]
        u = u * resc_u / (std_s / std_u)
        
        c *= norm_const
        u *= norm_const
        s *= norm_const
        vs *= norm_const
        
        vs_smooth_comps = np.zeros_like(vs)
        for k in range(num_comps):
            if np.all(vs[:, k] == 0): continue
            vs_smooth_comps[:, k] = smooth_scale(rna_conn, vs[:, k])
        
        vs_sum_raw = vs.sum(1)
        vs_sum_smoothed = smooth_scale(rna_conn, vs_sum_raw) 
        
        s_sum = s.sum(1)
        u_sum = u.sum(1)
        c_sum = c.sum(1)
        
        
        return {
            'idx': i,
            's_comps': s, 'u_comps': u, 'c_comps': c, 
            'vs_comps': vs_smooth_comps,
            'vs_no_sm_comps': vs,
            's': s_sum, 'u': u_sum, 'c': c_sum,
            'vs': vs_sum_smoothed,
            'vs_no_sm': vs_sum_raw,
            'success': True
            }
    except Exception as e:
        return {'idx': i, 'success': False, 'error': str(e)}

# --- PIPELINE FUNCTIONS ---

def create_denoised_atac(adata_rna, gene_weights, XC_raw, nn_idx=None, nn_dist=None, model_outdir='modeling_results/', n_pcs=30, 
                         n_neighbors=30, random_state = 0):
    if nn_idx is None or nn_dist is None:
        XC_raw.columns = range(XC_raw.shape[1])
        prod_raw = XC_raw @ gene_weights.reset_index(drop=True)
        atac_AA_raw = anndata.AnnData(prod_raw.values, 
                                      obs=adata_rna.obs,
                                      var=pd.DataFrame(index=prod_raw.columns.values))
        del prod_raw 
        
        nn_idx, nn_dist = gen_wnn(adata_rna, atac_AA_raw, [n_pcs, n_pcs], n_neighbors, random_state = random_state)
        os.makedirs(model_outdir+"seurat_wnn/", exist_ok=True)
        np.savetxt(model_outdir+"seurat_wnn/nn_idx_arch.txt", nn_idx, delimiter=',')
        np.savetxt(model_outdir+"seurat_wnn/nn_dist_arch.txt", nn_dist, delimiter=',')
        
    to_smooth = anndata.AnnData(XC_raw.copy(), obs=adata_rna.obs)
    XC_smooth = smooth_archetypes(to_smooth, nn_idx, nn_dist, outdir=model_outdir)

    prod = XC_smooth @ gene_weights.reset_index(drop=True)
    atac_AA_denoised = anndata.AnnData(prod.values, 
                                       obs=adata_rna.obs,
                                       var=pd.DataFrame(index=prod.columns.values))
    del prod 
    
    atac_AA_denoised.layers['Mc'] = atac_AA_denoised.X
    atac_AA_denoised.write(model_outdir+'adata_atac_AA_denoised.h5ad')
    return atac_AA_denoised

def apply_MultiVelo_AA(adata_rna, XC_raw, S_raw, peak_annotation, nn_idx=None, nn_dist=None, gene_list=None, weight_c=0.6, n_jobs=-1, model_outdir='modeling_results/', n_pcs=30, n_neighbors=30):
    _, gene_weights = annotate_and_summarize(S_raw, peak_annotation, outdir=model_outdir)
    gene_weights = gene_weights.loc[:, adata_rna.var_names]
    
    atac_AA_denoised = create_denoised_atac(adata_rna, gene_weights, XC_raw, nn_idx=nn_idx, nn_dist=nn_dist, model_outdir=model_outdir, n_pcs=n_pcs, n_neighbors=n_neighbors)

    full_res_denoised = mv.recover_dynamics_chrom(adata_rna, 
                                                  atac_AA_denoised, 
                                                  gene_list=gene_list, 
                                                  weight_c=weight_c, 
                                                  n_jobs=n_jobs, 
                                                  n_neighbors=n_neighbors, n_pcs=n_pcs)
    return full_res_denoised

def apply_ArchVelo_full(adata_rna, atac_AA_denoised, smooth_arch, gene_weights, model_outdir, gene_list=None, method='Nelder-Mead', maxiter1=1500, max_outer_iter=3, update_mode='cells', n_jobs=-1, n_neighbors=50, n_pcs=50, verbose=False):
    full_res_denoised = mv.recover_dynamics_chrom(adata_rna.copy(), atac_AA_denoised, gene_list=gene_list, weight_c=0.6, n_jobs=n_jobs, n_neighbors=n_neighbors, n_pcs=n_pcs)
    full_res_denoised.write(model_outdir+'multivelo_result_denoised_chrom.h5ad')
    
    avel = apply_ArchVelo(adata_rna, full_res_denoised, smooth_arch, gene_weights, model_outdir, gene_list=gene_list, method=method, maxiter1=maxiter1, max_outer_iter=max_outer_iter, update_mode=update_mode, n_jobs=n_jobs, verbose=verbose)
    
    return avel

def apply_ArchVelo(adata_rna, full_res_denoised, smooth_arch, gene_weights, model_outdir, gene_list=None, method='Nelder-Mead', maxiter1=1500, max_outer_iter=3, update_mode='cells', n_jobs=-1, weight_c=0.3, verbose=False):
    if verbose: print('Applying ArchVelo')
    min_c, max_c = extract_minmax(smooth_arch)
    num_comps = smooth_arch.shape[1]
    
    av_pars = extract_ArchVelo_pars(adata_rna, full_res_denoised, smooth_arch, gene_weights, gene_list=gene_list, weight_c=weight_c, method=method, maxiter1=maxiter1, max_outer_iter=max_outer_iter, update_mode=update_mode, n_jobs=n_jobs, verbose=verbose)
    if verbose: print('Done')
    
    f = open(model_outdir+'archvelo_results_pars.p', 'wb')
    pickle.dump(av_pars, f)
    f.close()
    
    avel = velocity_result(adata_rna, full_res_denoised, gene_weights, min_c, max_c, av_pars, gene_list=gene_list, n_jobs=n_jobs)
    avel.uns['archvelo_params'] = {'num_comps': num_comps, 'weight_c': weight_c}
    cells_AA = {}
    for g in full_res_denoised.var_names: 
        uu = np.ravel(full_res_denoised[:,g].layers['Mu'])
        ss = np.ravel(full_res_denoised[:,g].layers['Ms'])
        cc = np.ravel(full_res_denoised[:, g].layers['ATAC'])
        cells_AA[g] = cells_to_keep(cc, uu, ss)
    set_likelihood(avel, cells = cells_AA)
    #avel = avel[:, avel.var['velo_s_genes']]
    return avel

def extract_ArchVelo_pars(adata_rna, full_res_denoised, smooth_arch, gene_weights, gene_list=None, weight_c=0.3, method='Nelder-Mead', maxiter1=1500, max_outer_iter=3, update_mode='cells', multiproc=True, n_jobs=-1, verbose=False):
    if gene_list is None:
        gene_list = full_res_denoised.var['fit_likelihood'].sort_values(ascending=False).index
    n_genes = len(gene_list)
    
    Mu_arr = adata_rna[:, gene_list].layers['Mu']
    Ms_arr = adata_rna[:, gene_list].layers['Ms']
    fit_t_arr = full_res_denoised[:, gene_list].layers['fit_t']
    
    if hasattr(Mu_arr, "toarray"): Mu_arr = Mu_arr.toarray()
    if hasattr(Ms_arr, "toarray"): Ms_arr = Ms_arr.toarray()
    if hasattr(fit_t_arr, "toarray"): fit_t_arr = fit_t_arr.toarray()

    c_all = pd.DataFrame(smooth_arch.layers['Mc']).apply(minmax).values.copy()
    
    var_cols = ['fit_t_sw1', 'fit_t_sw2', 'fit_t_sw3', 'fit_alpha', 'fit_rescale_u', 'fit_beta', 'fit_gamma', 'fit_model']
    var_df = full_res_denoised[:, gene_list].var[var_cols]
    
    gw_vals = gene_weights.loc[:, gene_list].values
    min_c, max_c = extract_minmax(smooth_arch)
    
    # BROADCASTING FIX: [:, None] forces alignment to multiply perfectly
    gw_vals_scaled = gw_vals * (max_c - min_c)[:, None]

    if verbose: print('Starting optimization')
    
    if not multiproc:
        final_results = [_extract_worker(i, Mu_arr, Ms_arr, c_all, fit_t_arr, gw_vals_scaled, var_df.iloc[i].to_dict(), weight_c, maxiter1, max_outer_iter, method, update_mode) for i in range(n_genes)]
    else:    
        # MEMORY FIX: Pass full arrays to joblib for memory mapping
        final_results = Parallel(n_jobs=n_jobs)(
            delayed(_extract_worker)(i, Mu_arr, Ms_arr, c_all, fit_t_arr, gw_vals_scaled, var_df.iloc[i].to_dict(), weight_c, maxiter1, max_outer_iter, method, update_mode) 
            for i in range(n_genes)
        )
    return final_results

def velocity_result(adata_rna, full_res_denoised, gene_weights, min_c, max_c, av_pars, gene_list=None, n_jobs=-1):
    if gene_list is None:
        gene_list = full_res_denoised.var['fit_likelihood'].sort_values(ascending=False).index
    else:
        gene_list = pd.Index(gene_list)
    
    n_genes = len(gene_list)
    num_comps = gene_weights.shape[0]
    
    Mu_arr = adata_rna[:, gene_list].layers['Mu']
    Ms_arr = adata_rna[:, gene_list].layers['Ms']
    if hasattr(Mu_arr, "toarray"): Mu_arr = Mu_arr.toarray()
    if hasattr(Ms_arr, "toarray"): Ms_arr = Ms_arr.toarray()

    rna_conn = full_res_denoised.obsp['_RNA_conn']
    gw_vals = gene_weights.loc[:, gene_list].values
    
    # BROADCASTING FIX: [:, None] forces alignment
    gw_vals_scaled = gw_vals * (max_c - min_c)[:, None]
    
    print(f"Calculating velocity layers for {n_genes} genes...")
    
    # MEMORY FIX: Pass full arrays to joblib for memory mapping
    results_gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_velocity_worker)(i, Mu_arr, Ms_arr, av_pars[i], num_comps, gw_vals_scaled, rna_conn) 
        for i in range(n_genes)
    )
    
    print("Assembling AnnData...")
    avel = adata_rna[:, gene_list].copy()
    shape = avel.shape
    
    avel.layers['s'] = np.zeros(shape, dtype=np.float32)
    avel.layers['u'] = np.zeros(shape, dtype=np.float32)
    avel.layers['c'] = np.zeros(shape, dtype=np.float32)
    avel.layers['velo_s'] = np.zeros(shape, dtype=np.float32)
    avel.layers['velo_s_no_sm'] = np.zeros(shape, dtype=np.float32)
    avel.layers['fit_t'] = np.zeros(shape, dtype=np.float32)
    
    comp_dicts = {'s_components': {}, 'u_components': {}, 'c_components': {}, 'velo_s_components': {}, 'velo_s_no_sm_components': {}}
    for k in range(num_comps):
        for key in comp_dicts:
            comp_dicts[key][k] = np.full(shape, np.nan, dtype=np.float32)

    for res in results_gen:
        if not res['success']: continue
        i = res['idx']
        
        avel.layers['s'][:, i] = res['s']
        avel.layers['u'][:, i] = res['u']
        avel.layers['c'][:, i] = res['c']
        avel.layers['velo_s'][:, i] = res['vs']
        avel.layers['velo_s_no_sm'][:, i] = res['vs_no_sm']
        avel.layers['fit_t'][:, i] = np.ravel(av_pars[i][1])
        
        for k in range(num_comps):
            comp_dicts['s_components'][k][:, i] = res['s_comps'][:, k]
            comp_dicts['u_components'][k][:, i] = res['u_comps'][:, k]
            comp_dicts['c_components'][k][:, i] = res['c_comps'][:, k]
            comp_dicts['velo_s_components'][k][:, i] = res['vs_comps'][:, k]
            comp_dicts['velo_s_no_sm_components'][k][:, i] = res['vs_no_sm_comps'][:, k]

    avel.uns.update(comp_dicts)
    
    np.nan_to_num(avel.layers['velo_s'], nan=0.0, copy=False)
    avel.uns['velo_s_params'] = full_res_denoised.uns.get('velo_s_params', {})
    avel.var['velo_s_genes'] = True
    avel.var["fit_likelihood"] = full_res_denoised.var['fit_likelihood']
    
    active_mask = np.abs(avel.layers['velo_s']).sum(0) > 0
    avel = avel[:, active_mask].copy()
    
    def safe_mean_abs(layer_name):
        arr = np.mean(np.abs(avel.layers[layer_name]), 0)
        arr[arr == 0] = 1.0
        return arr

    mean_abs_vs = safe_mean_abs('velo_s')
    
    for k in range(num_comps):
        suf = f'_comp_{k}'
        vs_comp = avel.uns['velo_s_components'][k][:, active_mask]
        vs_no_sm_comp = avel.uns['velo_s_no_sm_components'][k][:, active_mask]
        s_comp = avel.uns['s_components'][k][:, active_mask]
        u_comp = avel.uns['u_components'][k][:, active_mask]
        c_comp = avel.uns['c_components'][k][:, active_mask]
        
        avel.layers[f'velo_s{suf}'] = vs_comp
        avel.layers[f'velo_s_no_sm{suf}'] = vs_no_sm_comp
        avel.layers[f's{suf}'] = s_comp
        avel.layers[f'u{suf}'] = u_comp
        avel.layers[f'a{suf}'] = c_comp
        avel.layers[f'velo_s{suf}_norm'] = vs_comp / mean_abs_vs

    for k in comp_dicts.keys():
        if k in avel.uns: del avel.uns[k]
            
    return avel

def generate_decomposition(g, avel, us_only=True, smooth_arch=None, celltype_name='leiden'):
    num_comps = avel.uns['archvelo_params']['num_comps']
    df = pd.DataFrame(np.ravel(avel[np.argsort(np.ravel(avel[:,g].layers['fit_t'])),g].layers['Ms']))
    
    df.index.names = ['Time']
    df.columns = ['Ms']
    df = df.sort_values([ 'Time'])
    df['Ms_smooth'] =  df[['Ms']].rolling(100, min_periods=0, center=True).mean().values
    df = df.iloc[:, [-2, -1]]

    sub = np.argsort(np.ravel(avel[:,g].layers['fit_t']))
    df_u = pd.DataFrame(np.ravel(avel[sub,g].layers['Mu']))
    df_u.index.names = ['Time']
    df_u.columns = ['Mu']
    df_u = df_u.sort_values([ 'Time'])
    df_u['Mu_smooth'] =  df_u[['Mu']].rolling(100, min_periods=0, center=True).mean().values
    df_u = df_u.iloc[:, [-2, -1]]

    if not us_only and smooth_arch is not None:
        df_c = pd.DataFrame(smooth_arch[sub,:].X, columns=range(num_comps))
        df_c.index.names = ['Time']
        df_c.columns.names = ['Mc']
        df_c = df_c.sort_values([ 'Time'])
        
        df_c_pred = pd.DataFrame(np.stack([np.ravel(avel[sub,g].layers['a_comp_'+str(i)]) for i in range(num_comps)], axis=1), columns=range(num_comps))
        df_c_pred.index.names = ['Time']
        df_c_pred.columns.names = ['a']
        df_c_pred = df_c_pred.sort_values([ 'Time'])
    
    df_u_pred = pd.DataFrame(np.stack([np.ravel(avel[sub,g].layers['u_comp_'+str(i)]) for i in range(num_comps)], axis=1), columns=range(num_comps))
    df_u_pred.index.names = ['Time']
    df_u_pred.columns.names = ['u']
    df_u_pred['Total'] = np.ravel(avel[np.argsort(np.ravel(avel[:,g].layers['fit_t'])),g].layers['u'])
    df_u_pred = df_u_pred.sort_values([ 'Time'])

    df_s_pred = pd.DataFrame(np.stack([np.ravel(avel[sub,g].layers['s_comp_'+str(i)]) for i in range(num_comps)], axis=1), columns=range(num_comps))
    df_s_pred.index.names = ['Time']
    df_s_pred.columns.names = ['s']
    df_s_pred['Total'] = np.ravel(avel[sub,g].layers['s'])
    df_s_pred = df_s_pred.sort_values([ 'Time'])
    
    df_velo_s_pred = pd.DataFrame(np.stack([np.ravel(avel[sub,g].layers['velo_s_no_sm_comp_'+str(i)]) for i in range(num_comps)], axis=1), columns=range(num_comps))
    df_velo_s_pred.index.names = ['Time']
    df_velo_s_pred.columns.names = ['velo_s']
    df_velo_s_pred['Total'] = np.ravel(avel[sub,g].layers['velo_s_no_sm'])
    df_velo_s_pred = df_velo_s_pred.sort_values([ 'Time'])
    
    if not us_only and smooth_arch is not None:
        df = pd.concat([df, df_u, df_c, df_c_pred, df_s_pred, 
                        df_velo_s_pred, 
                        df_u_pred], keys=['data', 'data', 'data', 'a', 's', 
                                          'velo_s', 
                                          'u'], axis=1)
    else:
        df = pd.concat([df, df_u, df_s_pred, 
                        df_velo_s_pred, 
                        df_u_pred], keys=['data', 'data', 's', 
                                          'velo_s', 
                                          'u'], axis=1)

    df['Time'] = np.sort(np.ravel(avel[:,g].layers['fit_t']))
    df[celltype_name] = np.ravel(avel[sub,:].obs[celltype_name])
    df = df.set_index(['Time', celltype_name])
    df.columns.names = ['Variable', 'Archetype']
    df = df.stack(['Archetype', 'Variable']).reset_index(['Time','Archetype', 'Variable'])
    df.columns = ['Time', 'Archetype', 'Variable','Value']
    df_smooth = df.copy()
    df_smooth['Time'] = np.round(df_smooth['Time'], 3)
    df_smooth = df_smooth.groupby(['Time', 'Archetype', 'Variable']).mean().reset_index()
    return df, df_smooth

def velocity_graph(avel, vkey='velo_s', **kwargs):
    return mv.velocity_graph(avel, vkey=vkey)

def latent_time(avel, **kwargs):
    return mv.latent_time(avel, **kwargs)
    
def velocity_embedding_stream(avel, vkey='velo_s', show=False, color='celltype', title=False, **kwargs):
    return mv.velocity_embedding_stream(avel, vkey=vkey, show=show, color=color, title=title, **kwargs)