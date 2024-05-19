# IMLup

Repository for DSHARP IM Lup MPoL Analysis

# Directory Structure

    IMLup/
        casa_analysis/
            src/
            data/
                raw/
                temp/
            Snakefile
        mpol_analysis/
            src/
            data/
                raw/
                temp/
            Snakefile

The repository is configured the way it is because I need to approach this folder using two separate Docker images, mounting the `IMLup` folder as a volume. This is because (as of December 2023) the Docker images that will run CASA without bugs (Python 3.8) are so old that they won't run the optimized NVIDIA GPU files (Python 3.10, at oldest). 

The main idea is that Docker image #1 is non-GPU and configured with CASA modular and works with the scripts in `casa_analysis`. The goal of these scripts is to double-check we're understanding the DSHARP data correctly, (possibly) rescale the weights, and export the visibilities to a file format that does not depend on CASA, e.g. `casa_analysis/data/temp/data.asdf`.

Docker image #2 is configured with MPoL, other modern PyTorch analysis scripts, and picks up from the data exported from the CASA analysis (copied to `mpol_analysis/data/raw/data.asdf`).

The Dockerfiles are outside of the scope of this particular repository, but can be provided upon request, if useful.