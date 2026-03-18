#!/bin/bash

#SBATCH --job-name=original24-world
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=250
#SBATCH --mem=800G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# Run from submission directory.
cd "${SLURM_SUBMIT_DIR}"

eval $('/home/jpivarski/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)

export PYTHONUNBUFFERED=1

python infer_original_24_input_layers.py --input-tiff ../../environmental/original-24-input-layers.tif --output-dir output/original24-world --workers 120 --ram-ceiling-gb 300
