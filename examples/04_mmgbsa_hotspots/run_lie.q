#!/bin/bash
#SBATCH -e 6e22_lie.err
#SBATCH -o 6e22_lie.out
#SBATCH --time=2:00:00
#SBATCH --partition=forli-pro,forli
#SBATCH --cpus-per-task=4
#SBATCH --job-name="6e22_lie"

# Occupancy annotation + LIE, in one pass. Unlike the MMGBSA leg there is no second
# stage and no --collect: compute_LIE runs pytraj in-process in about a second per
# molecule, so it finishes inside this job and the results are on the checkpoint when
# it exits.
#
# The annotation scan is the expensive part here, not the energetics.
source ~/.bashrc
micromamba activate autopath

python -m cosolvkit.cli.refine_hotspots --config ../03_analysis/analysis.yaml \
                --out results_lie \
                --stride 5 \
                --target binding_sites --top-n 3 \
                --mode lie
