#%%
"""
Evaluation utility functions.
This module contains util functions for computing evaluation scores.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wilcoxon
from statsmodels.sandbox.stats.multicomp import multipletests
def summary_scores(all_scores):
    """Summarize group scores.
    
    Args:
        all_scores (dict{str,list}): 
            {group name: score list of individual cells}.
    
    Returns:
        dict{str,float}: 
            Group-wise aggregation scores.
        float: 
            score aggregated on all samples
        
    """
    sep_scores = {k:np.mean(s) for k, s in all_scores.items() if s}
    overal_agg = np.mean([s for k, s in sep_scores.items() if s])
    return sep_scores, overal_agg

def keep_type(adata, nodes, target, k_cluster):
    """Select cells of targeted type
    
    Args:
        adata (Anndata): 
            Anndata object.
        nodes (list): 
            Indexes for cells
        target (str): 
            Cluster name.
        k_cluster (str): 
            Cluster key in adata.obs dataframe

    Returns:
        list: 
            Selected cells.

    """
    return nodes[adata.obs[k_cluster][nodes].values == target]

def cross_boundary_correctness(
    adata, 
    k_cluster, 
    k_velocity, 
    cluster_edges, 
    return_raw=False, 
    x_emb="X_umap"
):
    """Cross-Boundary Direction Correctness Score (A->B)
    
    Args:
        adata (Anndata): 
            Anndata object.
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str): 
            key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): 
            pairs of clusters has transition direction A->B
        return_raw (bool): 
            return aggregated or raw scores.
        x_emb (str): 
            key to x embedding for visualization.
        
    Returns:
        dict: 
            all_scores indexed by cluster_edges or mean scores indexed by cluster_edges
        float: 
            averaged score over all cells.
        
    """
    scores = {}
    all_scores = {}
    
    if x_emb == "X_umap":
        v_emb = adata.obsm['{}_umap'.format(k_velocity)]
    else:
        v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
        
    x_emb = adata.obsm[x_emb]
    
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel] # [n * 30]
        
        boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, k_cluster), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]
        
        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0: continue

            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1,-1)).flatten()
            type_score.append(np.mean(dir_scores))
        
        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score
        
    if return_raw:
        return all_scores 
    
    return scores, np.mean([sc for sc in scores.values()])

def create_relative_performance_table(df: pd.DataFrame, 
                                      control_method: str, 
                                      alpha: float = 0.05) -> pd.DataFrame:
    """
    Compares all methods against a specified control method for each Edge,
    using a three-marker system for relative performance.
    """
    
    # 1. Setup
    edges = df['Edge'].unique()
    methods = df['Method'].unique()
    other_methods = [m for m in methods if m != control_method]
    
    if control_method not in methods:
        print(f"Error: Control method '{control_method}' not found in the data.")
        return pd.DataFrame()

    results_table = pd.DataFrame(index=edges, columns=methods).fillna('')
    
    # 2. Comparison and Correction by Edge
    for edge in edges:
        edge_data = df[df['Edge'] == edge]
        
        # Data for the control method on this edge
        data_control = edge_data[edge_data['Method'] == control_method]['CBD'].values
        
        if len(data_control) < 2:
            results_table.loc[edge, :] = 'Insufficient data'
            continue

        p_values_for_fdr = []
        comparison_metadata = []

        # Pairwise comparison: Control vs. every Other Method
        for other_method in other_methods:
            data_other = edge_data[edge_data['Method'] == other_method]['CBD'].values
            
            if len(data_other) < 2:
                results_table.loc[edge, other_method] = 'ID'
                continue
            
            # Wilcoxon signed-rank test (two-sided)
            _, p_value = wilcoxon(data_control, data_other, alternative='two-sided')
            
            p_values_for_fdr.append(p_value)
            comparison_metadata.append(other_method)

        # 3. FDR Correction (Benjamini-Hochberg) for this EDGE only
        if not p_values_for_fdr:
            continue

        reject, _, _, _ = multipletests(
            p_values_for_fdr, 
            alpha=alpha, 
            method='fdr_bh'
        )

        # 4. Determine Markers based on FDR-corrected results
        median_control = np.median(data_control)
        #mean_control = np.mean(data_control)
        
        for i, other_method in enumerate(comparison_metadata):
            is_significant_difference = reject[i]
            
            data_other = edge_data[edge_data['Method'] == other_method]['CBD'].values
            median_other = np.median(data_other)
            #mean_other = np.mean(data_other)
            
            # Assuming higher CBD is better:
            is_control_better = median_control > median_other
            #is_control_better = mean_control > mean_other
            
            if is_significant_difference:
                if is_control_better:
                    # Control is significantly better
                    results_table.loc[edge, other_method] = '★'
                else:
                    # Other method is significantly better
                    results_table.loc[edge, other_method] = '-'
            else:
                # No significant difference
                results_table.loc[edge, other_method] = 'NS'

    return results_table

def evaluate(
    adata, 
    cluster_edges, 
    k_cluster, 
    k_velocity="velocity", 
    x_emb="X_umap", 
    verbose=True
):
    """Evaluate velocity estimation results using 5 metrics.
    
    Args:
        adata (Anndata): 
            Anndata object.
        cluster_edges (list of tuples("A", "B")): 
            pairs of clusters has transition direction A->B
        k_cluster (str): 
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str): 
            key to the velocity matrix in adata.obsm.
        x_emb (str): 
            key to x embedding for visualization.
        
    Returns:
        dict: 
            aggregated metric scores.
    
    """

    #from .eval_utils import cross_boundary_correctness
    #from .eval_utils import inner_cluster_coh
    crs_bdr_crc = cross_boundary_correctness(adata, k_cluster, k_velocity, cluster_edges, True, x_emb)
    ic_coh = inner_cluster_coh(adata, k_cluster, k_velocity, True)
    
    if verbose:
        print("# Cross-Boundary Direction Correctness (A->B)\n{}\nTotal Mean: {}".format(*summary_scores(crs_bdr_crc)))
        print("# In-cluster Coherence\n{}\nTotal Mean: {}".format(*summary_scores(ic_coh)))
    
    return {
        "Cross-Boundary Direction Correctness (A->B)": crs_bdr_crc,
        "In-cluster Coherence": ic_coh,
    }