#!/bin/bash
#SBATCH -e 6e22.err
#SBATCH -o 6e22.out
#SBATCH --gres=gpu:1#rtxa6000:1
#SBATCH --time=1-0
#SBATCH --partition=forli-pro,forli
#SBATCH --exclude=nodea0110,nodea0111
##SBATCH --exclusive
#SBATCH --job-name="6e22"

export OPENMM_CUDA_COMPILER=$(which nvcc)
nvidia-smi

source ~/.bashrc
micromamba activate autopath

create_cosolvent_system -c build_config.yaml