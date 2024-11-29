This folder contains processed Mouse Embryonic Brain data from https://www.10xgenomics.com/datasets/fresh-embryonic-e-18-mouse-brain-5-k-1-standard-1-0-0
The processing notebook can be found at ../ArchVelo_data_preparation.ipynb
Please see this notebook for more detailed instructions.

adata_rna.h5ad - RNA modality, cell x gene.
adata_atac.h5ad - ATAC modality, cell x gene, same cells and genes, processed for MultiVelo analysis
adata_atac_raw.h5ad - ATAC modality, cell x summits for the same cells and genes, processed for ArchVelo analysis.
