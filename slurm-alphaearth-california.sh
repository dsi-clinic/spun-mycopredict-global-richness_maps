#!/bin/bash

#SBATCH --job-name=alphaearth-california
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=800G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# Run from submission directory.
cd "${SLURM_SUBMIT_DIR}"

eval $('/home/jpivarski/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)

PYTHONUNBUFFERED=1 python infer_alphaearth_california_tiles.py \
  --input-base ../alphaearth-inputs/ \
  --workers 120 \
  --ram-ceiling-gb 300
