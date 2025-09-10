import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


import multivelo as mv
import os
import scipy
import pickle


import anndata
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from multivelo.dynamical_chrom_func import *
from multivelo.auxiliary import *

from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal
from scvelo.tools.velocity_embedding import quiver_autoscale, velocity_embedding

def apply_ArchVelo_full(adata_rna,
                        atac_AA_denoised,
                        adata_atac,
                        smooth_arch,
                        gene_weights,
                        model_outdir,
                        n_jobs = -1,
                        n_neighbors = 50,
                        n_pcs = 50):

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
    lik_cutoff = 0.05
    avel.var['velo_s_genes'] = (avel.var['fit_likelihood']>lik_cutoff)
    return avel
    

def velo_on_grid(avel,
                 key_vel = 'velo_s',
                 celltype_name = 'leiden',
                 recalculate_graph = False):
    avel_copy = avel.copy()
    if recalculate_graph:
        avel_copy.var[key_vel+'_genes'] = True
        mv.velocity_graph(avel_copy, vkey = key_vel)
        mv.latent_time(avel_copy, vkey = key_vel)
    scv.pl.velocity_embedding_grid(avel_copy,
                                vkey = key_vel+'_norm',
                                show=False, 
                                color = celltype_name,
                                title = False, 
                                legend_loc = 'none', 
                                linewidth = 0.5,
                                arrow_length=2)
    plt.clf()
    adata = avel_copy
    xe = adata.obsm["X_umap"]
    ve = adata.obsm[key_vel+'_norm_umap']
    norms = np.linalg.norm(pd.DataFrame(avel.layers[key_vel+'_norm']).fillna(0), axis = 1)
    gp, vel, vel_n = compute_velocity_on_grid_with_norms(
    X_emb=xe,
    V_emb=ve,
    norms = norms,
    density = 1,
    )
    return gp, vel, vel_n

def vis_velo_on_grid(avel,
                     key_vel = 'velo_s',
                     gp = None, 
                     vel = None,
                     vel_n = None,
                     color_by = 'leiden',
                     title = '',
                     ax = None,
                     transparent = True,
                     recalculate_graph = False):
    if gp is None or vel is None or vel_n is None:
        gp, vel, vel_n = velo_on_grid(avel,
                 key_vel = key_vel,
                 celltype_name = 'leiden',
                 recalculate_graph = recalculate_graph)

    hl, hw, hal = 12,10,8
    scale = 0.5
    quiver_kwargs = {"angles": "xy", "scale_units": "xy", "edgecolors": "k"}
    quiver_kwargs.update({ "width": 0.001,"headlength": hl / 2})#"width": 0.001, 
    quiver_kwargs.update({"headwidth": hw / 2, "headaxislength": hal / 2})

    
    sns.set(style = 'white', font_scale = 1.5)

    if ax is None:
        fig, ax = plt.subplots(1,1, dpi = 500, figsize = (7,6))
    scv.pl.umap(avel, 
               ax = ax,
               alpha=0.3, 
               color=color_by,
               s = 300,
               layer = 'Ms',
               cmap = 'viridis',
               show = False,
               frameon = False,
               legend_loc = 'none')
    # Normalize norms to [0, 1] for alpha
    norm_min, norm_max = np.percentile(vel_n,5), np.percentile(vel_n,99)
    norm_maxscaled = (vel_n) / (norm_max + 1e-8)
    
    norm_proj_velos = np.linalg.norm(vel,axis = 1)
    max_norm_proj = np.max(norm_proj_velos*norm_maxscaled)
    for (x, y), (vx, vy), norm_proj, nnn in zip(gp, vel, 
                                                  norm_proj_velos,norm_maxscaled):
            proj_norm = nnn*norm_proj/max_norm_proj
            if transparent:
                alpha =min(proj_norm, 1)
            else:
                alpha = 1
            #if alpha*np.sqrt(vx**2+vy**2)>0.02:
            ax.quiver(
                    x, y, vx*nnn*2, vy*nnn*2, 
                    color="black",
                    linewidth = 0.5,
                    scale = 0.5,
                    alpha =alpha,
                    **quiver_kwargs
                )
    ax.axis('off')
    ax.set_title(title)
    return ax


def compute_velocity_on_grid_with_norms(
    X_emb,
    V_emb,
    norms,
    density=None,
    smooth=None,
    n_neighbors=None,
    min_mass=None,
    autoscale=True,
    adjust_for_stream=False,
    cutoff_perc=None,
):
    """TODO."""
    # remove invalid cells
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]
    norms = norms[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    norms_grid = (norms[neighs] * weight[:, :]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]
    norms_grid /= np.maximum(1, p_mass)[:]
    if min_mass is None:
        min_mass = 1

    if adjust_for_stream:
        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid**2).sum(0))
        min_mass = 10 ** (min_mass - 6)  # default min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)
        cutoff = mass.reshape(V_grid[0].shape) < min_mass

        if cutoff_perc is None:
            cutoff_perc = 5
        length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T
        length = length.reshape(ns, ns)
        cutoff |= length < np.percentile(length, cutoff_perc)

        V_grid[0][cutoff] = np.nan
    else:
        min_mass *= np.percentile(p_mass, 99) / 100
        X_grid, V_grid, norms_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass],norms_grid[p_mass > min_mass]

        if autoscale:
            V_grid /= 3 * quiver_autoscale(X_grid, V_grid)

    return X_grid, V_grid, norms_grid

def visualize_genes(gns, 
                    adata_rna, 
                    title = None, 
                    groups = 'cell_type', 
                    **kwargs):
    sns.set(style = 'white', font_scale = 1)
    rel_genes = gns

    cc = pd.DataFrame(adata_rna[:, rel_genes].layers['log1p'].todense().copy(), 
                      index = np.ravel(adata_rna.obs[groups]),
                      columns = np.ravel(rel_genes))
    cc.index.names = [groups]
    cc = cc.groupby(groups).mean()
    cc = (cc-cc.min(0)).div(cc.max(0)-cc.min(0), 1)
    sns.clustermap(cc, cmap = 'Greys', 
                  col_cluster = True,  **kwargs)
    if title is not None:
        plt.suptitle(title, y = 1.05, fontsize = 30)


def apply_km(vls):
    vls = vls.reshape(-1,1)

    km = KMeans(2)#GM(2, n_init = 3, init_params = 'random_from_data')#KMeans(2)
    km.fit(vls)
    lbs = km.predict(vls)
    centers = km.cluster_centers_#.means_
    args = np.argsort(np.argsort(np.ravel(centers)))
    lbs = np.array([args[lb] for lb in lbs])
    return lbs
    
def get_cells(avel,
             rna_conn,
             key_to_filter_cells,
             smooth_cells = False,
             km = True,
             thres = 0.1,
             plot = False):

    
    vls = np.nanmean(np.abs(avel.layers[key_to_filter_cells]), axis = 1)
    if smooth_cells:
        vls = smooth_scale(rna_conn, vls)
    if km:
        lbs = apply_km(vls)
        cells = (lbs>0)

        if plot:
            plt.figure()
            sns.distplot(vls)
            plt.axvline(np.max(vls[~cells]))
    else:
        cells = vls>thres

    print('Number of cells: ', np.sum(cells))
    return cells

def plot_phase(avel, 
               g, 
               cells = None, 
               color_by = 'leiden', 
               pal = None,
               s = 5,
               ax = None):
    if cells is None:
        cells = [True]*avel.shape[0]
    if ax is None:
        fig, ax = plt.subplots(1,1)
    uu = np.ravel(avel[cells, g].layers['Mu'])
    ss = np.ravel(avel[cells, g].layers['Ms'])
    u_pred = np.ravel(avel[:, g].layers['u'])
    s_pred = np.ravel(avel[:, g].layers['s'])
    full_uu = np.ravel(avel[:, g].layers['Mu'])
    full_ss = np.ravel(avel[:, g].layers['Ms'])
    std_s = np.std(full_ss)
    std_u = np.std(full_uu)
    tt = np.ravel(avel[:, g].layers['fit_t'])
    sns.scatterplot(y = uu, 
                    x = ss, 
                    ax = ax,
                    hue = avel[cells,:].obs[color_by], 
                    palette = pal,
                    legend = False, 
                    s = s)
    ax.plot(s_pred[np.argsort(tt)], 
            u_pred[np.argsort(tt)],
           c = 'black', lw = 4)
    ax.set_title(g)   
    return ax

def generate_decomposition(g, 
                           #adata_rna, 
                           avel, 
                           smooth_arch):
    num_comps = smooth_arch.shape[1]
    df = pd.DataFrame(np.ravel(avel[np.argsort(np.ravel(avel[:,g].layers['fit_t'])),g].layers['Ms']))
    
    df.index.names = ['Time']
    #df = df.reset_index()
    df.columns = ['Ms']
    df = df.sort_values([ 'Time'])
    df['Ms_smooth'] =  df[['Ms']].rolling(100, min_periods = 0, center = True).mean().values
    df = df.iloc[:, [-2, -1]]

    sub = np.argsort(np.ravel(avel[:,g].layers['fit_t']))
    
    df_u = pd.DataFrame(np.ravel(avel[sub,g].layers['Mu']))
    
    df_u.index.names = ['Time']
    #df = df.reset_index()
    df_u.columns = ['Mu']
    df_u = df_u.sort_values([ 'Time'])
    df_u['Mu_smooth'] =  df_u[['Mu']].rolling(100, min_periods = 0, center = True).mean().values
    df_u = df_u.iloc[:, [-2, -1]]

    df_c = pd.DataFrame(smooth_arch[sub,:].X,
                       columns = range(num_comps))
    
    df_c.index.names = ['Time']
    #df = df.reset_index()
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
    
    #print(df)
    # df[['a_'+str(i) for i in range(8)]] = df_c_pred.values
    # df[['s_'+str(i) for i in range(8)]+['s_pred']] = df_s_pred.values
    # df[['velo_s_'+str(i) for i in range(8)]+['velo_s_pred']] = df_velo_s_pred.values
    
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

from scvelo.pl.simulation import compute_dynamics, simulation
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
    # if show_switches:
    tt = model_to_use[:,g].layers['fit_t']
    uu = np.ravel(adata[:,g].layers['Mu'])
    ss = np.ravel(adata[:,g].layers['Ms'])
    # c_all = pd.DataFrame(unimputed_XC.layers['Mc']).apply(minmax).values.copy()
    # c_all = c_all*(gene_weights.loc[:,g].values*(max_c-min_c))
    # c_all = np.sum(c_all,1)
    c_all = model_to_use[:, g].layers['ATAC']#pd.DataFrame(prod.loc[:,g]).values.copy()#.apply(minmax).values.copy()
    cc = np.ravel(c_all)
    
    from scipy.spatial import KDTree
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
    
    # scale_c, scale_u, scale_s = np.max(c_norm-np.min(c_norm)), std_u/std_s, 1.0
    # c_norm /= scale_c
    # u_norm /= scale_u
    # s_norm /= scale_s
    keep = non_zero & non_outlier & \
            (u_norm > 0.2 * np.percentile(u_norm, 99.5)) & \
            (s_norm > 0.2 * np.percentile(s_norm, 99.5))
    return keep
    
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
    
    # will run ArchVelo on top MultiVelo likelihood genes
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



def multivelo_connectivities(adata_rna, n_neighbors = 30, n_pcs=30):
    from scanpy import Neighbors
    if ('connectivities' not in adata_rna.obsp.keys() or
            (adata_rna.obsp['connectivities'] > 0).sum(1).min()
            > (n_neighbors-1)):
        neighbors = Neighbors(adata_rna)
        neighbors.compute_neighbors(n_neighbors=n_neighbors, knn=True,
                                    n_pcs=n_pcs)
        rna_conn = neighbors.connectivities
        print('recalculating...')
    else:
        rna_conn = adata_rna.obsp['connectivities'].copy()
    rna_conn.setdiag(1)
    rna_conn = rna_conn.multiply(1.0 / rna_conn.sum(1)).tocsr()
    return rna_conn

def annotate_and_summarize(S, 
                           peak_annotation,
                           outdir = 'modeling_results/'):
    S = S.T
    S['gene'] = peak_annotation.loc[S.index,:]['gene']
    S.set_index('gene', append = True, inplace = True)
    S = S.T
    gene_weights = S.T.groupby('gene').mean().T
    gene_weights.to_csv(outdir+'gene_weights.csv')
    return S, gene_weights

def smooth_archetypes(to_smooth, 
                      nn_idx, 
                      nn_dist, 
                      outdir = 'modeling_results/'):
    # we smooth archetypes over Seurat wnn neighbors
    mv.knn_smooth_chrom(to_smooth, nn_idx, nn_dist)
    XC_smooth = pd.DataFrame(to_smooth.layers['Mc'], 
                 index = to_smooth.obs.index, 
                 columns = range(to_smooth.shape[1]))
    arches =  anndata.AnnData(XC_smooth)
    arches.layers['spliced'] = arches.X
    arches.layers['Mc'] = arches.X
    # Save the result for use later on
    arches.write(outdir+"arches.h5ad")
    return XC_smooth

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

def extract_wnn_connectivities(adata_atac_raw, nn_idx, nn_dist):
    mv.knn_smooth_chrom(adata_atac_raw, nn_idx, nn_dist)
    return adata_atac_raw.obsp['connectivities']

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

#### Optimization for chromatin

def minmax(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))


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
def err(pars, c, times):
    sol = solve_for_chromatin(times, pars)
    return np.linalg.norm(c-sol)**2


def optimize_chromatin(times, c, seed=57):
    res = scipy.optimize.dual_annealing(err, 
                                        args = (c, times),
                                        seed = seed,
                                        bounds = [(0,20), (0, 10), (0.01,10), (0,1)], 
                                        maxiter = 1000)
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

def opt_all_pars(pars, g, chrom_switches, alpha_cs, scale_ccs, c0s, times = None, chrom_on = None, rna = None, gene_weights = None, max_c = None, min_c = None, full_res_denoised = None):
    #print(pars)
    e = err_all(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = times, chrom_on = chrom_on, rna = rna, gene_weights = gene_weights, max_c = max_c, min_c = min_c, full_res_denoised = full_res_denoised)
    #print(e)
    return e

def func_to_optimize(g, 
                     chrom_switches, 
                     alpha_cs, 
                     scale_ccs, 
                     c0s, 
                     pars, 
                     times = None, 
                     chrom_on = None, 
                     full_res_denoised = None):
    # pseudocode with multivelo
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
        #print(x0)
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
    #print('Minimized')
    return res.x, res.fun, chrom_switches, alpha_cs, scale_ccs, chrom_on, c0s



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
    #print('Fitting for '+str(g))
    num_comps = gene_weights.shape[0]
    u_all = np.ravel(rna[:,g].layers['Mu'].copy())
    s_all = np.ravel(rna[:,g].layers['Ms'].copy())
    c_all = pd.DataFrame(arches.layers['Mc']).apply(minmax).values.copy()
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
                                                                       verbose = verbose, 
                                                                       rna = rna, 
                                                                       gene_weights = gene_weights,
                                                                       max_c = max_c, min_c = min_c, 
                                                                       arches = arches, 
                                                                       full_res_denoised = full_res_denoised)
                                                                                    
        if verbose:
        #print(chrom_switches, alpha_cs, scale_ccs, c0s)
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


def velocity_full(g, 
                  chrom_switches, 
                  alpha_cs, 
                  scale_ccs, 
                  c0s, 
                  pars, 
                  times = None, 
                  full_res_denoised = None):
    # pseudocode with multivelo
    num_comps = int((len(pars)-3)/3)
    alphas = pars[:num_comps]
    t_sw1s = pars[num_comps:(2*num_comps)]
    t_sw_rnas = pars[(2*num_comps):(3*num_comps)]
    #rescale_u = pars[3*num_comps]
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
    return vc, vu,vs


#### calculate errors

# def phase_multivelo(g, model_to_use = None, rna = None):
#     gene = g
#     adata = model_to_use
#     n_anchors = adata.uns['velo_s_params']['t']
#     t_sw_array = np.array([adata[:, gene].var['fit_t_sw1'],
#                            adata[:, gene].var['fit_t_sw2'],
#                            adata[:, gene].var['fit_t_sw3']])
#     t_sw_array = t_sw_array[t_sw_array < 20]
#     min_idx = int(adata[:, gene].var['fit_anchor_min_idx'])
#     max_idx = int(adata[:, gene].var['fit_anchor_max_idx'])
#     old_t = np.linspace(0, 20, n_anchors)[min_idx:max_idx+1]
#     new_t = old_t - np.min(old_t)
#     new_t = new_t * 20 / np.max(new_t)
#     a_c = adata[:, gene].varm['fit_anchor_c'].ravel()[min_idx:max_idx+1]
#     a_u = adata[:, gene].varm['fit_anchor_u'].ravel()[min_idx:max_idx+1]
#     a_s = adata[:, gene].varm['fit_anchor_s'].ravel()[min_idx:max_idx+1]
#     new_t =new_t[0:new_t.shape[0]]

#     tt = model_to_use[:,g].layers['fit_t']
#     uu = np.ravel(rna[:,g].layers['Mu'])
#     ss = np.ravel(rna[:,g].layers['Ms'])
    
#     c_all = pd.DataFrame(prod.loc[:,g]).values.copy()#.apply(minmax).values.copy()
#     cc = np.ravel(c_all)
    
#     from scipy.spatial import KDTree
#     new_t = new_t.reshape(-1,1)
#     tree = KDTree(new_t)
#     neighbor_dists, neighbor_indices = tree.query(tt.reshape(-1,1))
#     c_pred = a_c[neighbor_indices]
#     u_pred = a_u[neighbor_indices]
#     s_pred = a_s[neighbor_indices]
#     return tt, uu, ss, cc, new_t, u_pred, s_pred, c_pred

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
    return np.sum((uu-u_pred)**2)/(scale_u**2)+np.sum((ss-s_pred)**2)+(weight_c**2)*np.sum((cc-c_pred)**2)/(scale_c**2)


def err_all_full(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = None, 
                 weight_c = 0.6, plot = False, chrom_on = None, 
                rna = None, gene_weights = None, max_c = None, min_c = None,
                full_res_denoised = None):
    new_t = times
    tt = times

    num_comps = gene_weights.shape[0]
    
    u_all = rna[:,g].layers['Mu'].copy()
    s_all = rna[:,g].layers['Ms'].copy()
    
    max_c_total = max(prod.loc[:,g])
    min_c_total = min(prod.loc[:,g])
    c_all = pd.DataFrame(arches.layers['Mc']).values.copy()
    c_all = c_all*(gene_weights.loc[:,g].values)

    c, u,s = func_to_optimize(g, 
                              chrom_switches, 
                              alpha_cs, 
                              scale_ccs, 
                              c0s, 
                              pars, 
                              times = times, 
                              chrom_on = chrom_on, 
                              full_res_denoised = full_res_denoised)
  
    c = gene_weights.loc[:,g].values*(min_c+c*(max_c-min_c))
        
    u = u*(gene_weights.loc[:,g].values*(max_c-min_c))
    s = s*(gene_weights.loc[:,g].values*(max_c-min_c))
    
    if plot:
        for i in range(num_comps):
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
    resc_u = pars[3*num_comps]
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
    return (np.linalg.norm(u*resc_u-np.ravel(u_all))**2)+np.linalg.norm(s-np.ravel(s_all))**2 +(weight_c**2)*np.linalg.norm(c-np.ravel(c_all))**2    





# def calc_err_ArchVelo(g, res = None, genes = None, weight_c = 0.6, 
#                   plot = False, 
#                  rna = None, gene_weights = None, max_c = None, min_c = None,
#                  full_res_denoised = None):
#     i = np.where(genes == g)[0][0]
#     pars, new_times, (chrom_switches, alpha_cs, scale_ccs, chrom_on, c0s) = res[i]
#     return err_all_full(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = new_times, weight_c = weight_c, chrom_on = chrom_on, plot = plot,
#                        rna = rna, gene_weights = gene_weights, max_c = max_c, min_c = min_c, full_res_denoised = full_res_denoised)

#### plotting


def plot_results(g, 
                 model_to_use = None,
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
                 res = None, 
                 genes = None, 
                 full_res_denoised = None):
    i = np.where(genes == g)[0][0]
    pars = res[i][0].copy()
    num_comps = int((len(pars)-3)/3)
    times = res[i][1].copy()
    (chrom_switches, alpha_cs, scale_ccs, chrom_on, c0s) = res[i][2]
    u_all = rna[:,g].layers['Mu'].copy()
    s_all = rna[:,g].layers['Ms'].copy()
    std_u = np.std(u_all)
    std_s = np.std(s_all)
    scale_u = std_u/std_s
    #u_all/=scale_u
    c,u,s = func_to_optimize(g, chrom_switches, alpha_cs, scale_ccs, c0s, pars, times = times,
                            chrom_on = chrom_on, full_res_denoised = full_res_denoised)
    std_c = np.std(np.sum(c,1))
    scale_c = std_c/std_s
    scale_u = std_u/std_s
    c/=scale_c
    c = c*(gene_weights.loc[:,g].values*(max_c-min_c))
    resc_u = pars[3*num_comps]
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