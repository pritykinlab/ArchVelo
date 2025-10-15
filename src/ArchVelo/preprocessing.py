from scanpy import Neighbors

import numpy as np
import pandas as pd
import multivelo as mv

import anndata

def multivelo_connectivities(adata_rna, n_neighbors = 30, n_pcs=30):
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

def extract_wnn_connectivities(adata_atac_raw, nn_idx, nn_dist):
    mv.knn_smooth_chrom(adata_atac_raw, nn_idx, nn_dist)
    return adata_atac_raw.obsp['connectivities']