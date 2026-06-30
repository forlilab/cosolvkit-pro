#!/bin/bash
#SBATCH -e 6e22_equil.err
#SBATCH -o 6e22_equil.out
#SBATCH --gres=gpu:1
#SBATCH --time=1-0
#SBATCH --partition=forli-pro,forli,alphafold
#SBATCH --exclude=nodea0110,nodea0111
#SBATCH --job-name="6e22_equil"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath

# Equilibrate the CosolvKit-built system (system.pdb + system.xml) produced by
# examples/01_build_cosolvent_system. Protein-only restraints (cosolvents free).
python ap_equilibration.py \
    --system-dir ../01_build_cosolvent_system/6E22_imidazole \
    --protocol cosolvent_equilibration.json
