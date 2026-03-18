from .archetypal_regression import *

"""
RNA velocity analysis from scRNA+ATAC-seq with ArchVelo.
"""

# A standard place for the package version
__version__ = "0.1.0"

# --- Utilities ---
from .utils import (
    cells_to_keep,
    extract_minmax,
    minmax,
    print_vals
)

# --- Preprocessing and Data Preparation ---
from .preprocessing import (
    multivelo_connectivities,
    annotate_and_summarize,
    smooth_archetypes,
    extract_wnn_connectivities,
    gen_wnn
)

# --- Core Modeling and Workflow Functions ---
from .modeling import (
    create_denoised_atac,
    apply_MultiVelo_AA,
    apply_ArchVelo,
    apply_ArchVelo_full,
    extract_ArchVelo_pars,
    velocity_result,
    generate_decomposition,
    velocity_graph,
    latent_time,
    velocity_embedding_stream
)

# --- Optimization ---
from .optimization import (
    optimize_all,
    optimize_pars,
    func_to_optimize,
    calculate_exact_gene_layers,
    err_all,
    generate_exp_mine_njit,
    predict_exp_mine,
    optimize_chromatin,
    solve_for_chromatin,
    err_chrom
)

# --- Plotting and Visualization ---
from .plotting import (
    vis_velo_on_grid,
    velo_on_grid,
    compute_velocity_on_grid_with_norms,
    visualize_genes,
    plot_phase,
    plot_fits,
    plot_velo,
    mv_scatter_plot_return,
    get_cells, 
    get_cells_and_genes,
    get_arch_latent_time,
    apply_km
)

# --- Model Evaluation ---
from .evaluation import (
    calc_lik_ArchVelo,
    calc_lik_scvelo,
    calc_lik_multivelo,
    calc_likelihood,
    set_likelihood, 
    phase_multivelo
)

from .metrics import (
     cross_boundary_correctness,
     create_relative_performance_table
)

# --- The __all__ variable: Defining the Public API ---
__all__ = [
    # Utilities
    "cells_to_keep",
    "extract_minmax",
    "minmax",
    "print_vals",
    
    # Preprocessing
    "multivelo_connectivities",
    "annotate_and_summarize",
    "smooth_archetypes",
    "extract_wnn_connectivities",
    "gen_wnn"

    # Modeling
    "create_denoised_atac",
    "apply_MultiVelo_AA",
    "apply_ArchVelo",
    "apply_ArchVelo_full",
    "extract_ArchVelo_pars",
    "velocity_result",
    "generate_decomposition",
    "velocity_graph",
    "latent_time",
    "velocity_embedding_stream",

    # Optimization
    "optimize_all",
    "optimize_pars",
    "func_to_optimize",
    "calculate_exact_gene_layers",
    "err_all",
    "generate_exp_mine_njit",
    "predict_exp_mine",
    "optimize_chromatin",
    "solve_for_chromatin",
    "err_chrom",

    # Plotting
    "vis_velo_on_grid",
    "velo_on_grid",
    "compute_velocity_on_grid_with_norms",
    "visualize_genes",
    "plot_phase",
    "plot_fits",
    "plot_velo",
    "mv_scatter_plot_return",
    "get_cells",
    "get_cells_and_genes",
    "get_arch_latent_time",
    "apply_km",

    # Evaluation
    "calc_lik_ArchVelo",
    "calc_lik_scvelo",
    "calc_lik_multivelo",
    "calc_likelihood",
    "set_likelihood",
    "phase_multivelo",

    #Metrics
    "cross_boundary_correctness",
    "create_relative_performance_table"
]