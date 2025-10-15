import numpy as np
import pandas as pd


import scvelo as scv
import multivelo as mv

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from scipy.stats import norm as normal
from scvelo.tools.velocity_embedding import quiver_autoscale, velocity_embedding
#from multivelo import mv_logging as logg

from multivelo.dynamical_chrom_func import smooth_scale


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

    offs_u = 0
    offs_s = 0
    ordr = np.argsort(np.ravel(times))
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
    # if len(missing_genes) > 0:
    #     logg.update(f'{missing_genes} not found', v=0)
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
