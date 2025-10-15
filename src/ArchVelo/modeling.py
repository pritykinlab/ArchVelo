import numpy as np
import pandas as pd

import multivelo as mv

import pickle

import anndata

from joblib import Parallel, delayed

from .preprocessing import annotate_and_summarize, smooth_archetypes
from .optimization import optimize_all, func_to_optimize, compute_velocity_mine
from .evaluation import calc_lik_ArchVelo
from .utils import extract_minmax

from multivelo.dynamical_chrom_func import anchor_points, smooth_scale
from multivelo.auxiliary import gen_wnn


def apply_MultiVelo_AA(adata_rna, obs_index, conn, XC_raw,
                                S_raw, peak_annotation,
                                nn_idx = None, nn_dist = None,
                                weight_c = 0.6,
                                n_jobs = -1,
                                data_outdir = 'processed_data/',
                                model_outdir = 'modeling_results/',
                                n_pcs=30, n_neighbors=30
                               ):

    #annotate S
    S, gene_weights = annotate_and_summarize(S_raw, 
                                             peak_annotation, 
                                             outdir = model_outdir)
    gene_weights = gene_weights.loc[:, adata_rna.var_names]
    if nn_idx is None or nn_dist is None:
        XC_raw.columns = range(XC_raw.shape[1])
        prod_raw = XC_raw @ gene_weights.reset_index(drop = True)
        atac_AA_raw = anndata.AnnData(prod_raw.values, 
                                       obs = obs_index,
                                  var = pd.DataFrame(index = prod_raw.columns.values))
        nn_idx, nn_dist = gen_wnn(adata_rna, atac_AA_raw, [n_pcs,n_pcs], n_neighbors)
        np.savetxt("seurat_wnn/nn_idx_ours.txt", nn_idx, delimiter=',')
        np.savetxt("seurat_wnn/nn_dist_ours.txt", nn_dist, delimiter=',')
        
    to_smooth = anndata.AnnData(XC_raw.copy(), 
                                obs = obs_index)
    # smooth archetypes
    XC_smooth = smooth_archetypes(to_smooth, 
                                  nn_idx, 
                                  nn_dist, 
                                  outdir = model_outdir)

    
    # Create ATAC matrix denoised via AA. Required for MultiVelo-AA
    prod = XC_smooth @ gene_weights.reset_index(drop = True)
    atac_AA_denoised = anndata.AnnData(prod.values, 
                                   obs = obs_index,
                              var = pd.DataFrame(index = prod.columns.values))
    atac_AA_denoised.layers['Mc'] = atac_AA_denoised.X
    
    #this is never used but required by MultiVelo
    atac_AA_denoised.obsp['connectivities'] = conn
    # Save the result for use later on
    atac_AA_denoised.write(model_outdir+'adata_atac_AA_denoised.h5ad')

    # Run MultiVelo-AA
    full_res_denoised = mv.recover_dynamics_chrom(adata_rna, 
                                                  atac_AA_denoised, 
                                                  weight_c = weight_c,
                                                  n_jobs = n_jobs,
                                                  n_neighbors = n_neighbors, 
                                                  n_pcs=n_pcs)
    
    return full_res_denoised

def apply_ArchVelo_full(adata_rna,
                        atac_AA_denoised,
                        adata_atac,
                        smooth_arch,
                        gene_weights,
                        model_outdir,
                        n_jobs = -1,
                        n_neighbors = 50,
                        n_pcs = 50,
                        lik_cutoff = 0.05):

    full_res_denoised = mv.recover_dynamics_chrom(adata_rna.copy(), 
                                                  atac_AA_denoised, 
                                                  weight_c = 0.6,
                                                  n_jobs = n_jobs,
                                                  n_neighbors = n_neighbors, 
                                                  n_pcs=n_pcs)
    full_res_denoised.write(model_outdir+'multivelo_result_denoised_chrom.h5ad')
    avel = apply_ArchVelo(adata_rna, 
                   full_res_denoised,
                   smooth_arch,
                   gene_weights,
                   model_outdir)
    avel.var['fit_likelihood'] = np.nan
    for g in avel.var_names:    
        liks = calc_lik_ArchVelo(g,
                      adata_atac = adata_atac,
                      avel = avel)
        avel.var['fit_likelihood'].loc[g] = liks[0]
    avel.var['velo_s_genes'] = (avel.var['fit_likelihood']>lik_cutoff)
    return avel

def apply_ArchVelo(adata_rna, 
                   full_res_denoised,
                   smooth_arch,
                   gene_weights,
                   model_outdir,
                   n_jobs = -1):
    min_c, max_c = extract_minmax(smooth_arch)
    av_pars = extract_ArchVelo_pars(adata_rna, 
                      full_res_denoised, 
                      smooth_arch,
                      gene_weights,
                      weight_c = 0.3, 
                      n_jobs = n_jobs)
    f = open(model_outdir+'archevelo_results_weight_c_0.3.p', 'wb')
    pickle.dump(av_pars, f)
    f.close()
    avel = velocity_result(adata_rna, 
                      full_res_denoised,
                      gene_weights,
                      min_c, max_c, 
                      av_pars)
    return avel

def extract_ArchVelo_pars(adata_rna, 
                          full_res_denoised,   
                          smooth_arch, 
                          gene_weights,
                          weight_c = 0.3, 
                          maxiter1 = 1500,
                          max_outer_iter = 3,
                          multiproc = True,
                          n_jobs = -1,
                          verbose = False):
    min_c, max_c = extract_minmax(smooth_arch)
    rna = adata_rna.copy()
    
    top_lik = full_res_denoised.var['fit_likelihood'].sort_values(ascending = False).index
    n_genes = len(top_lik)
    def process(i):
        #print(top_lik[i])
        return optimize_all(top_lik[i], 
                            maxiter1 = maxiter1, 
                            max_outer_iter = max_outer_iter, 
                            weight_c = weight_c, 
                            full_res_denoised = full_res_denoised, 
                            rna = rna, 
                            gene_weights = gene_weights, 
                            max_c = max_c, 
                            min_c = min_c, 
                            arches = smooth_arch,
                            verbose = verbose)

    if not multiproc:
        final_results = [process(i) for i in range(n_genes)]
    else:    
        # run with parallelization                
        final_results = Parallel(n_jobs=n_jobs)(delayed(process)(i) for i in range(n_genes))
    return final_results

def velocity_result(adata_rna, 
          full_res_denoised,
          gene_weights,
          min_c, max_c, 
          av_pars,
          n_jobs = -1):
    num_comps = gene_weights.shape[0]
    rna_conn = full_res_denoised.obsp['_RNA_conn']
    top_lik = full_res_denoised.var['fit_likelihood'].sort_values(ascending = False).index
    n_genes = len(top_lik)
    
    avel = adata_rna[:, top_lik].copy()
    avel.layers['velo_s'] = np.nan*np.zeros(avel.shape)
    avel.layers['velo_s_no_smooth'] = np.nan*np.zeros(avel.shape)
    avel.layers['s'] = np.nan*np.zeros(avel.shape)
    avel.layers['u'] = np.nan*np.zeros(avel.shape)
    avel.layers['c'] = np.nan*np.zeros(avel.shape)
    avel.uns['s_components'] = {}
    avel.uns['velo_s_no_smooth_components'] = {}
    avel.uns['velo_s_components'] = {}
    avel.uns['u_components'] = {}
    avel.uns['c_components'] = {}
    for comp in range(num_comps):
        avel.uns['s_components'][comp] = np.nan*np.zeros(avel.shape)
        avel.uns['velo_s_no_smooth_components'][comp] = np.nan*np.zeros(avel.shape)
        avel.uns['velo_s_components'][comp] = np.nan*np.zeros(avel.shape)
        avel.uns['u_components'][comp] = np.nan*np.zeros(avel.shape)
        avel.uns['c_components'][comp] = np.nan*np.zeros(avel.shape)
    def fill_vel(i):        
        try:
            g = top_lik[i]
            u_all = np.ravel(avel[:,g].layers['Mu'].copy())
            s_all = np.ravel(avel[:,g].layers['Ms'].copy())

            std_u = np.std(u_all)
            std_s = np.std(s_all)

            
            norm_const = gene_weights.loc[:,g].values*(max_c-min_c)
            pars, times, (chrom_switches, alpha_cs, scale_ccs, chrom_on, c0s) = av_pars[i]
            _, _, vs = velocity_full(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = np.ravel(times),
                                  full_res_denoised = full_res_denoised.copy())
            c, u,s = func_to_optimize(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = np.ravel(times), 
                                      chrom_on = chrom_on, full_res_denoised = full_res_denoised.copy())
        
            resc_u = pars[3*num_comps]
            u*=resc_u
            u/=(std_s/std_u)
            
            vs = (vs*norm_const)#.sum(1)
            #vu = (vu*norm_const)#.sum(1)
            #u*=resc_u
        
            s = (s*norm_const)#.sum(1)
            u = (u*norm_const)#.sum(1)
            c = (c*norm_const)#.sum(1)
            for comp in range(num_comps):
                avel.uns['velo_s_no_smooth_components'][comp][:, i] = vs[:,comp]
                
                avel.uns['velo_s_components'][comp][:, i] = smooth_scale(rna_conn, vs[:,comp])
                avel.uns['s_components'][comp][:, i] = s[:,comp]
                avel.uns['u_components'][comp][:, i] = u[:,comp]
                avel.uns['c_components'][comp][:, i] = c[:,comp]
            vs = vs.sum(1)
            #vu = vu.sum(1)
            s = s.sum(1)
            u = u.sum(1)
            c = c.sum(1)
            avel.layers['velo_s_no_smooth'][:, i] = vs
            vs = smooth_scale(rna_conn, vs)
            avel.layers['velo_s'][:, i] = vs
            avel.layers['s'][:, i] = s
            avel.layers['u'][:, i] = u
            avel.layers['c'][:, i] = c
        except:
            #print(g)
            pass
    for i in range(n_genes):
        fill_vel(i)
   # Parallel(n_jobs = n_jobs)(delayed(fill_vel)(i) for i in range(3))
    avel.layers['fit_t'] = np.stack([av_pars[i][1] for i in range(n_genes)])[:,:,0].T
    avel.layers['velo_s'] = np.nan_to_num(avel.layers['velo_s'],0)
    avel.layers['velo_s_no_smooth'] = np.nan_to_num(avel.layers['velo_s_no_smooth'],0)

    avel.uns['velo_s_params'] = full_res_denoised.uns['velo_s_params']
    avel.var['velo_s_genes'] = True#full_res_denoised.var['velo_s_genes']
    avel.uns['velo_s_no_smooth_params'] = full_res_denoised.uns['velo_s_params']
    avel.var['velo_s_no_smooth_genes'] = full_res_denoised.var['velo_s_genes']
    avel.var["fit_likelihood"] = full_res_denoised.var['fit_likelihood']

    avel = avel[:, np.abs(avel.layers['velo_s']).sum(0)>0]
    subs = [np.where(top_lik == x)[0][0] for x in avel.var_names]
    for comp in range(num_comps):
        avel.uns['velo_s_no_smooth_components'][comp] = avel.uns['velo_s_no_smooth_components'][comp][:, subs]
        avel.uns['velo_s_components'][comp] = avel.uns['velo_s_components'][comp][:, subs]    
        avel.uns['s_components'][comp] = avel.uns['s_components'][comp][:, subs]
        avel.uns['u_components'][comp] = avel.uns['u_components'][comp][:, subs]
        avel.uns['c_components'][comp] = avel.uns['c_components'][comp][:, subs]
        avel.uns['velo_s_components'][comp][:, np.abs(avel.uns['velo_s_components'][comp]).sum(0)==0] = np.nan   
    for comp in range(num_comps):
        suf = '_comp_'+str(comp)
        avel.layers['a'+suf] = avel.uns['c_components'][comp]
        avel.layers['u'+suf] = avel.uns['u_components'][comp]
        avel.layers['s'+suf] = avel.uns['s_components'][comp]
        avel.layers['velo_s'+suf] = avel.uns['velo_s_components'][comp]
        avel.layers['velo_s'+suf+'_no_smooth'] = avel.uns['velo_s_no_smooth_components'][comp]
        avel.layers['velo_s'+suf+'_norm'] = pd.DataFrame(avel.uns['velo_s_components'][comp]).div(np.mean(np.abs(avel.layers['velo_s']),axis = 0), 1).values
        avel.layers['s'+suf+'_norm'] = pd.DataFrame(avel.uns['s_components'][comp]).div(np.mean(np.abs(avel.layers['s']),axis = 0), 1).values
        avel.layers['u'+suf+'_norm'] = pd.DataFrame(avel.uns['u_components'][comp]).div(np.mean(np.abs(avel.layers['u']),axis = 0), 1).values
        avel.layers['velo_s'+suf+'_no_smooth_norm'] = pd.DataFrame(avel.uns['velo_s_no_smooth_components'][comp]).div(np.mean(np.abs(avel.layers['velo_s_no_smooth']),axis = 0), 1).values
    #vel = avel.copy()
    for kk in ['s_components', 'velo_s_no_smooth_components', 'velo_s_components', 'u_components', 'c_components']:
        avel.uns.pop(kk)
    return avel

def velocity_full(g, 
                  chrom_switches, 
                  alpha_cs, 
                  scale_ccs, 
                  c0s, 
                  pars, 
                  times = None, 
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
    #ts_ = {}
    vc = np.zeros((full_res_denoised.shape[0],num_comps))
    vu = np.zeros((full_res_denoised.shape[0],num_comps))
    vs = np.zeros((full_res_denoised.shape[0],num_comps))
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
    return vc, vu, vs

def generate_decomposition(g, 
                           avel, 
                           smooth_arch):
    num_comps = smooth_arch.shape[1]
    df = pd.DataFrame(np.ravel(avel[np.argsort(np.ravel(avel[:,g].layers['fit_t'])),g].layers['Ms']))
    
    df.index.names = ['Time']
    df.columns = ['Ms']
    df = df.sort_values([ 'Time'])
    df['Ms_smooth'] =  df[['Ms']].rolling(100, min_periods = 0, center = True).mean().values
    df = df.iloc[:, [-2, -1]]

    sub = np.argsort(np.ravel(avel[:,g].layers['fit_t']))
    
    df_u = pd.DataFrame(np.ravel(avel[sub,g].layers['Mu']))
    
    df_u.index.names = ['Time']
    df_u.columns = ['Mu']
    df_u = df_u.sort_values([ 'Time'])
    df_u['Mu_smooth'] =  df_u[['Mu']].rolling(100, min_periods = 0, center = True).mean().values
    df_u = df_u.iloc[:, [-2, -1]]

    df_c = pd.DataFrame(smooth_arch[sub,:].X,
                       columns = range(num_comps))
    
    df_c.index.names = ['Time']
    df_c.columns.names = ['Mc']
    df_c = df_c.sort_values([ 'Time'])
    
    df_c_pred = pd.DataFrame(np.stack([np.ravel(avel[sub,g].layers['a_comp_'+str(i)]) for i in range(num_comps)], axis = 1), 
                               columns = range(num_comps))
    df_c_pred.index.names = ['Time']
    df_c_pred.columns.names = ['a']
    df_c_pred = df_c_pred.sort_values([ 'Time'])
    
    
    df_u_pred = pd.DataFrame(np.stack([np.ravel(avel[sub,g].layers['u_comp_'+str(i)]) for i in range(num_comps)], axis = 1), 
                               columns = range(num_comps))
    df_u_pred.index.names = ['Time']
    df_u_pred.columns.names = ['u']
    df_u_pred['Total'] = np.ravel(avel[np.argsort(np.ravel(avel[:,g].layers['fit_t'])),g].layers['u'])
    df_u_pred = df_u_pred.sort_values([ 'Time'])


    df_s_pred = pd.DataFrame(np.stack([np.ravel(avel[sub,g].layers['s_comp_'+str(i)]) for i in range(num_comps)], axis = 1), 
                               columns = range(num_comps))
    df_s_pred.index.names = ['Time']
    df_s_pred.columns.names = ['s']
    df_s_pred['Total'] = np.ravel(avel[sub,g].layers['s'])
    df_s_pred = df_s_pred.sort_values([ 'Time'])
    
    df_velo_s_pred = pd.DataFrame(np.stack([np.ravel(avel[sub,g].layers['velo_s_comp_'+str(i)+'_no_smooth']) for i in range(num_comps)], axis = 1), 
                               columns = range(num_comps))
    df_velo_s_pred.index.names = ['Time']
    df_velo_s_pred.columns.names = ['velo_s']
    df_velo_s_pred['Total'] = np.ravel(avel[sub,g].layers['velo_s_no_smooth'])
    df_velo_s_pred = df_velo_s_pred.sort_values([ 'Time'])
    
    
    df = pd.concat([df, df_u, df_c, df_c_pred, 
                    df_s_pred, df_velo_s_pred,
                   df_u_pred], 
                   keys = ['data', 'data', 'data', 'a', 's', 'velo_s', 'u'], 
                   axis = 1)
    
    
    
    df['Time'] = np.sort(np.ravel(avel[:,g].layers['fit_t']))
    df['cell_type_abbr'] = np.ravel(avel[sub,:].obs['cell_type_abbr'])
    df = df.set_index(['Time', 'cell_type_abbr'])
    df.columns.names = ['Variable', 'Archetype']
    df = df.stack(['Archetype', 'Variable']).reset_index(['Time','Archetype', 'Variable'])
    df.columns = ['Time', 'Archetype', 'Variable','Value']
    df_smooth = df.copy()
    df_smooth['Time'] = np.round(df_smooth['Time'], 3)
    df_smooth = df_smooth.groupby(['Time', 'Archetype', 'Variable']).mean().reset_index()
    return df, df_u, df_smooth

def velocity_graph(avel, vkey = 'velo_s'):
    return mv.velocity_graph(avel, vkey = vkey)

def latent_time(avel):
    return mv.latent_time(avel)
    
def velocity_embedding_stream(avel, vkey = 'velo_s',
                             show=False, color = 'celltype', 
                             title = False):
    return mv.velocity_embedding_stream(avel, 
                             vkey = vkey,
                             show=show, 
                             color = color, 
                             title = title)

