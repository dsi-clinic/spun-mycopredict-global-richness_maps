#!/bin/bash

# Driver script for training and evaluating fungal richness models
# This script performs cross-validation (both random and spatial) for AM and EcM fungal richness
# and outputs R² values for both types of cross-validation.

set -e  # Exit on error

# Configuration
CONDA_BASE=~/miniforge3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/output"
FUNCTIONS_DIR="${SCRIPT_DIR}/functions"
TEMP_DIR="${SCRIPT_DIR}/.temp_cv"

# Activate conda environment
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate base

echo "=================================="
echo "Fungal Richness Model Training and Evaluation"
echo "=================================="
echo ""

# Clean up any previous temporary files and outputs
echo "Cleaning up previous run artifacts..."
rm -rf "${TEMP_DIR}"
mkdir -p "${TEMP_DIR}"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Remove old grid search results to ensure fresh computation
# Use find to avoid glob expansion errors when no files exist
find "${OUTPUT_DIR}" -name "*_grid_search_results.csv" -type f -delete 2>/dev/null || true

echo "Done."
echo ""

# Function to run cross-validation for a given guild
run_cv() {
    local guild=$1
    local training_file=$2
    local random_points_file=$3
    local class_property=$4

    echo "=================================="
    echo "Processing ${guild} fungal richness"
    echo "=================================="
    echo ""

    # Check if training data exists
    if [ ! -f "${training_file}" ]; then
        echo "ERROR: Training data not found: ${training_file}"
        exit 1
    fi

    # Check if random points file exists (needed for spatial fold generation)
    if [ ! -f "${random_points_file}" ]; then
        echo "ERROR: Random points file not found: ${random_points_file}"
        exit 1
    fi

    # Create a copy of the training data to work with (in temp directory)
    local temp_training="${TEMP_DIR}/$(basename ${training_file})"
    cp "${training_file}" "${temp_training}"

    echo "Step 1: Checking for spatial CV folds..."
    echo "------------------------------------------------"

    # Check if spatial folds already exist in the training data
    if head -1 "${training_file}" | grep -q "knndmw_CV_folds"; then
        echo "Spatial folds already exist in training data (knndmw_CV_folds column found)."
        echo "Skipping KNNDM generation step."
    else
        echo "Spatial folds not found. Generating using KNNDM (this may take several minutes)..."

        # Generate spatial folds using the R script
        Rscript "${FUNCTIONS_DIR}/generateFoldsKNNDM.R" \
            --path_training "${temp_training}" \
            --ppoints "${random_points_file}" \
            --k 10 \
            --maxp 0.5 \
            --clustering "hierarchical" \
            --linkf "ward.D2" \
            --samplesize 1000 \
            --sampling "regular"

        echo "Spatial folds generated."

        # Save the spatial folds back to the original training data for future reuse
        echo "Saving spatial folds to original training data for future reuse..."
        cp "${temp_training}" "${training_file}"
        echo "Spatial folds saved to: ${training_file}"
    fi
    echo ""

    echo "Step 2: Running hyperparameter grid search with cross-validation..."
    echo "-------------------------------------------------------------------"

    # Run the PyTorch MLP grid search script
    python "${FUNCTIONS_DIR}/mlp_grid_search.py" \
        "${guild}" \
        "${temp_training}" \
        "${OUTPUT_DIR}/$(date +%Y%m%d)_${class_property}_grid_search_results.csv" \
        "${class_property}"

    echo ""
    echo "Cross-validation complete for ${guild}."
    echo ""
}

# # Process Arbuscular Mycorrhizal (AM) fungi
# echo ""
# echo "######################################"
# echo "# Arbuscular Mycorrhizal (AM) Fungi #"
# echo "######################################"
# echo ""

# run_cv "AM" \
#     "${DATA_DIR}/20260123_arbuscular_mycorrhizal_only_alphaearth_ball.csv" \
#     "${DATA_DIR}/filtered_randomPoints_AMF.csv" \
#     "arbuscular_mycorrhizal_richness"

# Process Ectomycorrhizal (EcM) fungi
echo ""
echo "######################################"
echo "# Ectomycorrhizal (EcM) Fungi       #"
echo "######################################"
echo ""

run_cv "EcM" \
    "${DATA_DIR}/20260123_ectomycorrhizal_only_alphaearth_ball.csv" \
    "${DATA_DIR}/filtered_randomPoints_ECM.csv" \
    "ectomycorrhizal_richness"

# Print final summary
echo ""
echo "=================================="
echo "All processing complete!"
echo "=================================="
echo ""
echo "Results have been saved to:"
echo ""

# Find and display the results files
for guild_type in "arbuscular_mycorrhizal" "ectomycorrhizal"; do
    result_file="${OUTPUT_DIR}/$(date +%Y%m%d)_${guild_type}_richness_grid_search_results.csv"
    if [ -f "${result_file}" ]; then
        echo "  ${guild_type}: ${result_file}"

        # Extract and display the best R² values
        if command -v python &> /dev/null; then
            python << PYEOF2
import pandas as pd
df = pd.read_csv("${result_file}")
if 'Mean_R2_Random' in df.columns:
    print("    Random CV  - Best R²: {:.4f} (Mean: {:.4f})".format(
        df['Mean_R2_Random'].max(), df['Mean_R2_Random'].mean()))
if 'Mean_R2_Spatial' in df.columns:
    print("    Spatial CV - Best R²: {:.4f} (Mean: {:.4f})".format(
        df['Mean_R2_Spatial'].max(), df['Mean_R2_Spatial'].mean()))
print("")
PYEOF2
        fi
    fi
done

# Clean up temporary directory
echo "Cleaning up temporary files..."
rm -rf "${TEMP_DIR}"

echo ""
echo "Done!"
echo ""
