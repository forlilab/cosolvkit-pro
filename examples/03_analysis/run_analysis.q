#!/bin/bash
#SBATCH -e 6e22_analysis.err
#SBATCH -o 6e22_analysis.out
#SBATCH --time=8:00:00
#SBATCH --partition=forli-pro,forli
#SBATCH --cpus-per-task=4
#SBATCH --job-name="6e22_analysis"

# Analysis is CPU-only (no GPU). Runs in the cosolvkit env, where the
# analyze_cosolvent_simulation CLI lives.
source ~/.bashrc
micromamba activate autopath

analyze_cosolvent_simulation -cfg analysis.yaml
