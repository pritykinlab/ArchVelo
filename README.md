<p align="center">
  <img src="archvelo_LOGO.jpg" alt="ArchVelo Logo" width="200"/>
</p>

ArchVelo is a method for modeling gene regulation and inferring cell trajectories using simultaneous single-cell chromatin accessibility and transcriptomic profiling (scRNA+ATAC-seq). ArchVelo extracts a set of shared **archetypal** chromatin accessibility profiles and models their dynamic influence on transcription. As a result, ArchVelo **improves the accuracy of trajectory inference** compared to previous methods and **decomposes the velocity field into components** driven by distinct regulatory programs.

This repository contains the source code for the `ArchVelo` Python package.

## Installation

ArchVelo requires Python 3.11 or newer. We strongly recommend creating a dedicated [virtual environment](https://docs.python.org/3/tutorial/venv.html) before installation.

The package and its dependencies can be installed with a single command directly from this GitHub repository:

```bash
pip install git+https://github.com/pritykinlab/ArchVelo.git
```

If you want to modify the source code, or run the demo, you should clone the repository and install it in the editable mode:
```bash
git clone https://github.com/pritykinlab/ArchVelo.git
cd ArchVelo
pip install -e .
```

## Tutorials

*   **Demo:** A demonstration of ArchVelo on a scRNA+ATAC-seq dataset for the mouse embryonic brain can be found in the `ArchVelo_demo.ipynb` notebook in this repository.

*   **Detailed End-to-End Analysis:** For a complete walkthrough of the ArchVelo analysis pipeline, please see our dedicated notebooks repository. It includes applications to the mouse embryonic brain, human hematopoietic stem cells, and CD8 T cells in acute and chronic viral infection conditions.
    *   **[ArchVelo Notebooks Repository](https://github.com/pritykinlab/ArchVelo_notebooks)**

## Minimal example
```python
import scanpy as sc
import ArchVelo as av
adata_rna = sc.read_h5ad(PATH_TO_RNA)
adata_atac_raw = sc.read_h5ad(PATH_TO_ATAC)
model_outdir = MODEL_OUTDIR
num_comps = NUM_COMPS
# peak_annotation should map each peak to one gene, containing adata_atac_raw.var_names in the index and the corresponding genes in the 'gene' column
peak_annotation = pd.read_csv(PATH_TO_PEAK_ANNOTATION, index_col = [0])

av.preprocess_rna(adata_rna)
av.preprocess_atac(adata_atac_raw)
adata_rna, adata_atac_raw = av.intersect_cells(adata_rna, adata_atac_raw)
adata_rna, adata_atac_raw = av.filter_genes_and_peaks(adata_rna, adata_atac_raw, peak_annotation)
XC_raw, S_raw = av.apply_AA_no_test(adata_atac_raw, k = num_comps, outdir = model_outdir)
_, gene_weights = av.annotate_and_summarize(S_raw, peak_annotation,
outdir = model_outdir)
atac_AA = av.create_denoised_atac(adata_rna, gene_weights, XC_raw, model_outdir = model_outdir, n_pcs=n_pcs, n_neighbors=n_neigh)
smooth_arch = sc.read_h5ad(model_outdir+"arches.h5ad")
avel = av.apply_ArchVelo_full(adata_rna, atac_AA, smooth_arch, gene_weights, model_outdir, n_jobs = -1)
av.velocity_graph(avel)
av.latent_time(avel)
av.velocity_embedding_stream(avel, show=False, color = 'celltype', title = 'ArchVelo result')
```

## Issues

If you encounter a bug or have trouble running the package, please open an issue on the [GitHub Issues page](https://github.com/pritykinlab/ArchVelo/issues).
