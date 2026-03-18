from scanpy import Neighbors

import numpy as np
import pandas as pd
import multivelo as mv

import anndata
import scanpy as sc
import multivelo as mv
import scipy

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


#mimicking multivelo
def gen_wnn(adata_rna, adata_adt, dims, nn, random_state=0):
    """Computes inputs for KNN smoothing.

    This function calculates the nn_idx and nn_dist matrices needed
    to run knn_smooth_chrom().

    Parameters
    ----------
    adata_rna: :class:`~anndata.AnnData`
        RNA anndata object.
    adata_atac: :class:`~anndata.AnnData`
        ATAC anndata object.
    dims: `List[int]`
        Dimensions of data for RNA (index=0) and ATAC (index=1)
    nn: `int` (default: `None`)
        Top N neighbors to extract for each cell in the connectivities matrix.

    Returns
    -------
    nn_idx: `np.darray` (default: `None`)
        KNN index matrix of size (cells, k).
    nn_dist: `np.darray` (default: `None`)
        KNN distance matrix of size (cells, k).
    """

    # make a copy of the original adata objects so as to keep them unchanged
    rna_copy = adata_rna.copy()
    adt_copy = adata_adt.copy()

    sc.tl.pca(rna_copy,
              n_comps=dims[0],
              random_state=np.random.RandomState(seed=42),
              use_highly_variable=True)  # run PCA on RNA

    lsi = scipy.sparse.linalg.svds(adt_copy.X, k=dims[1], random_state = random_state)  # run SVD on ADT

    # get the lsi result
    adt_copy.obsm['X_lsi'] = lsi[0]

    # add the PCA from adt to rna
    rna_copy.obsm['X_rna_pca'] = rna_copy.obsm.pop('X_pca')
    rna_copy.obsm['X_adt_lsi'] = adt_copy.obsm['X_lsi']

    # run WNN
    WNNobj = mv.pyWNN(rna_copy,
                      reps=['X_rna_pca', 'X_adt_lsi'],
                      npcs=dims,
                      n_neighbors=nn,
                      seed=42)

    adata_seurat = WNNobj.compute_wnn(rna_copy)

    # get the matrix storing the distances between each cell and its neighbors
    cx = scipy.sparse.coo_matrix(adata_seurat.obsp["WNN_distance"])

    # the number of cells
    cells = adata_seurat.obsp['WNN_distance'].shape[0]

    # define the shape of our final results
    # and make the arrays that will hold the results
    new_shape = (cells, nn)
    nn_dist = np.zeros(shape=new_shape)
    nn_idx = np.zeros(shape=new_shape)

    # new_col defines what column we store data in
    # our result arrays
    new_col = 0

    # loop through the distance matrices
    for i, j, v in zip(cx.row, cx.col, cx.data):

        # store the distances between neighbor cells
        nn_dist[i][new_col % nn] = v

        # for each cell's row, store the row numbers of its neighbor cells
        # (1-indexing instead of 0- is a holdover from R multimodalneighbors())
        nn_idx[i][new_col % nn] = int(j) + 1

        new_col += 1

    return nn_idx, nn_dist