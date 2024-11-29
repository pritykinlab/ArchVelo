# ArchVelo
This repository contains the code for ArchVelo package and example notebooks demonstrating its application to the E18 mouse embryonic brain multi-omic dataset. This dataset is available through 10X (https://www.10xgenomics.com/datasets/fresh-embryonic-e-18-mouse-brain-5-k-1-standard-1-0-0). 
Additional notebook demonstrating benchmarking of ArchVelo against MultiVelo (https://www.nature.com/articles/s41587-022-01476-y) and scVelo (https://scvelo.readthedocs.io/en/stable/) are included.

ArchVelo is an algorithm for modeling multi-omic velocity from scRNA+ATAC data. ArchVelo code builds on the application of Archetypal Analysis (AA) to the ATAC modality of the dataset of interest. The AA component of the package is found in the archetypal_regression folder. This folder also contains the code for applying a regularized linear regression model for ATAC-to-RNA prediction on any multi-omic dataset.

Here is the structure of this repository.

|-- **processed_data**: processed data to run the example notebooks \
|-- **archetypal_regression**: Archetypal Analysis for the ATAC modality and ATAC-to-RNA regression code \
|&emsp; |-- **archetypes.py**: delta-AA analysis (see https://github.com/ulfaslak/py_pcha) \
|&emsp; |-- **archetypes_regression.py**: module for ATAC-to-RNA regression \
|&emsp; |-- **util.py**: utility methods \
|&emsp; |-- **util_atac.py**: utility methods for the ATAC component \
|&emsp; |-- **util_regression.py**: utility for the regression \
|-- **ArchVelo.py**: ArchVelo methods \
|-- **Test_AA.ipynb**: test notebook for archetypal analysis \
|-- **ArchVelo_data_preparation.ipynb**: process data for ArchVelo analysis \
|-- **Create_archetypes.ipynb**: apply AA to the dataset \
|-- **ArchVelo_apply.ipynb**: apply ArchVelo to the dataset \
|-- **ArchVelo_velocities.ipynb**: extract and analyze ArchVelo velocity fields \
|-- **Apply_multimodels.ipynb**: apply scVelo and MultiVelo to the dataset (for benchmarking purposes) \
|-- **Compare_latent_time_agreement.ipynb**: benchmark ArchVelo against MultiVelo and scVelo. \

