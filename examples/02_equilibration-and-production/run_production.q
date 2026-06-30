#!/bin/bash
#SBATCH -e 6e22_prod_rep1.err
#SBATCH -o 6e22_prod_rep1.out
#SBATCH --gres=gpu:1
#SBATCH --time=2-0
#SBATCH --partition=forli-pro,forli,alphafold
#SBATCH --exclude=nodea0110,nodea0111
#SBATCH --job-name="6e22_prod_rep1"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath

# Production MD, continuing from the equilibration checkpoint. Submit once per
# replica (change --replica and the SBATCH log/job names for rep2, rep3, ...).
python ap_production.py \
    --system-dir ../01_build_cosolvent_system/6E22_imidazole \
    --replica rep1
