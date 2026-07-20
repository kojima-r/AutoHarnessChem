# Sandbox image for AutoHarnessChem
# build: docker build -t autoharnesschem/sandbox:latest docker/
FROM condaforge/miniforge3:latest

RUN mamba install -y -n base -c conda-forge \
        python=3.11 \
        rdkit \
        pyscf \
        scikit-learn \
        pandas \
        matplotlib \
    && mamba clean -afy

WORKDIR /workspace
CMD ["python", "--version"]
