<p align="center">
  <img src="ArchVelo_logo.png" alt="ArchVelo Logo" width="400"/>
</p>

ArchVelo is a method for modeling gene regulation and inferring cell trajectories using simultaneous single-cell chromatin accessibility and transcriptomic profiling (scRNA+ATAC-seq). ArchVelo extracts a set of shared **archetypal** chromatin accessibility profiles and models their dynamic influence on transcription. As a result, ArchVelo **improves the accuracy of trajectory inference** compared to previous methods and **decomposes the velocity field into components** driven by distinct regulatory programs.

This repository contains the source code for the `ArchVelo` Python package.

## Installation

ArchVelo requires Python 3.11 or newer. We strongly recommend creating a dedicated [virtual environment](https://docs.python.org/3/tutorial/venv.html) before installation.

The package and its dependencies can be installed with a single command directly from this GitHub repository:

```bash
pip install git+https://github.com/pritykinlab/ArchVelo.git
```

## Tutorials

*   **Demo:** A demonstration of ArchVelo on a scRNA+ATAC-seq dataset for the mouse embryonic brain can be found in the `ArchVelo_demo.ipynb` notebook in this repository.

*   **Detailed End-to-End Analysis:** For a complete walkthrough of the ArchVelo analysis pipeline, please see our dedicated notebooks repository. It includes applications to the mouse embryonic brain, human hematopoietic stem cells, and CD8 T cells in acute and chronic viral infection conditions.
    *   **[ArchVelo Notebooks Repository](https://github.com/pritykinlab/ArchVelo_notebooks)**
 
## Issues

If you encounter a bug or have trouble running the package, please open an issue on the [GitHub Issues page](https://github.com/pritykinlab/ArchVelo/issues).