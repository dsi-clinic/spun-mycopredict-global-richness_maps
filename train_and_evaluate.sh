#!/bin/bash

# Driver script for training and evaluating fungal richness models
# This script performs cross-validation (both random and spatial) for AM and EcM fungal richness
# and outputs R² values for both types of cross-validation.

set -e  # Exit on error

# Command-line arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <am_training_file> <em_training_file> <results_file>"
    exit 1
fi
AM_TRAINING_FILE="$1"
EM_TRAINING_FILE="$2"
RESULTS_FILE="$3"
echo "Running $1 $2 $3"

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

    # Create a Python script to run grid search
    cat > "${TEMP_DIR}/grid_search_${guild}.py" << 'PYEOF'
#!/usr/bin/env python
import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from contextlib import contextmanager
from functools import partial
from itertools import product
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    # For older sklearn versions
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)
import multiprocessing
import sys

# Get configuration from command line arguments
guild = sys.argv[1]
training_file = sys.argv[2]
class_property = sys.argv[3]

print(f"Loading training data from {training_file}...")
df = pd.read_csv(training_file)

# List of environmental covariates
covariateList = [
'A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09',
'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19',
'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29',
'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39',
'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49',
'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58', 'A59',
'A60', 'A61', 'A62', 'A63'
]

# Project-specific variables (primers, sequencing platforms, etc.)
if guild == "AM":
    project_vars = [
    'sequencing_platform454Roche',
    'sequencing_platformIllumina',
    'sample_typerhizosphere_soil',
    'sample_typesoil',
    'sample_typetopsoil',
    'primersAML1_AML2_then_AMV4_5NF_AMDGR',
    'primersAML1_AML2_then_NS31_AM1',
    'primersAML1_AML2_then_nu_SSU_0595_5__nu_SSU_0948_3_',
    'primersAMV4_5F_AMDGR',
    'primersAMV4_5NF_AMDGR',
    'primersGeoA2_AML2_then_NS31_AMDGR',
    'primersGeoA2_NS4_then_NS31_AML2',
    'primersGlomerWT0_Glomer1536_then_NS31_AM1A_and_GlomerWT0_Glomer1536_then_NS31_AM1B',
    'primersGlomerWT0_Glomer1536_then_NS31_AM1A__GlomerWT0_Glomer1536_then_NS31_AM1B',
    'primersNS1_NS4_then_AML1_AML2',
    'primersNS1_NS4_then_AMV4_5NF_AMDGR',
    'primersNS1_NS4_then_NS31_AM1',
    'primersNS1_NS41_then_AML1_AML2',
    'primersNS31_AM1',
    'primersNS31_AML2',
    'primersWANDA_AML2',
    'area_sampled',
    'extraction_dna_mass'
    ]
else:  # EcM
    project_vars = [
    'sequencing_platform454Roche',
    'sequencing_platformIllumina',
    'sequencing_platformIonTorrent',
    'sequencing_platformPacBio',
    'sample_typerhizosphere_soil',
    'sample_typesoil',
    'sample_typetopsoil',
    'primers5_8S_Fun_ITS4_Fun',
    'primersfITS7_ITS4',
    'primersfITS9_ITS4',
    'primersgITS7_ITS4',
    'primersgITS7_ITS4_then_ITS9_ITS4',
    'primersgITS7_ITS4_ITS4arch',
    'primersgITS7_ITS4m',
    'primersgITS7_ITS4ngs',
    'primersgITS7ngs_ITS4ngsUni',
    'primersITS_S2F___ITS3_mixed_1_1_ITS4',
    'primersITS1_ITS4',
    'primersITS1F_ITS4',
    'primersITS1F_ITS4_then_fITS7_ITS4',
    'primersITS1F_ITS4_then_ITS3_ITS4',
    'primersITS1ngs_ITS4ngs_or_ITS1Fngs_ITS4ngs',
    'primersITS3_KYO2_ITS4',
    'primersITS3_ITS4',
    'primersITS3ngs1_to_5___ITS3ngs10_ITS4ngs',
    'primersITS3ngs1_to_ITS3ngs11_ITS4ngs',
    'primersITS86F_ITS4',
    'primersITS9MUNngs_ITS4ngsUni',
    'area_sampled',
    'extraction_dna_mass'
    ]

# Create final list of covariates
covariateList = covariateList + project_vars

# Use random CV only
cv_col = "CV_Fold_Random"
if cv_col not in df.columns:
    print(f"ERROR: No random fold column found in training data: {cv_col}")
    print(f"Available columns: {df.columns.tolist()}")
    sys.exit(1)
else:
    print(f"Using random fold column: {cv_col}")

def gridSearch(params, X, y, df, cv_col, nTrees=250, random_seed=42):
    """Perform grid search for a given set of hyperparameters"""
    max_features, min_samples_leaf = params

    # Initialize the RandomForestRegressor
    rf = RandomForestRegressor(
        n_estimators=nTrees,
        random_state=random_seed,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        max_samples=0.632  # bag fraction
    )

    # Create custom CV folds
    unique_folds = df[cv_col].unique()
    cv_folds = [(np.where(df[cv_col] != fold)[0], np.where(df[cv_col] == fold)[0])
                for fold in unique_folds]

    # Perform cross-validation with custom folds
    scores = cross_validate(
        rf, X, y,
        cv=cv_folds,
        scoring={
            'r2': make_scorer(r2_score),
            'rmse': make_scorer(root_mean_squared_error),
            'mae': make_scorer(mean_absolute_error)
        },
        return_train_score=False,
        n_jobs=1
    )

    # Prepare the results
    model_name = f"{class_property}_rf_VPS{max_features}_LP{min_samples_leaf}_REGRESSION"
    suffix = "_Random" if cv_col == "CV_Fold_Random" else "_Spatial"
    results = {
        f'Mean_R2{suffix}': np.mean(scores['test_r2']),
        f'StDev_R2{suffix}': np.std(scores['test_r2']),
        f'Mean_RMSE{suffix}': np.mean(scores['test_rmse']),
        f'StDev_RMSE{suffix}': np.std(scores['test_rmse']),
        f'Mean_MAE{suffix}': np.mean(scores['test_mae']),
        f'StDev_MAE{suffix}': np.std(scores['test_mae']),
        'cName': model_name
    }
    print(f"  {model_name} ({cv_col}): R2 = {np.mean(scores['test_r2']):.4f}")
    return pd.DataFrame([results])

@contextmanager
def poolcontext(*args, **kwargs):
    """Context manager for multiprocessing pool"""
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

if __name__ == '__main__':
    # Get the regression matrix
    X = df[covariateList]
    y = df[class_property]

    print(f"Training data shape: {X.shape}")
    print(f"Target variable: {class_property}")
    print("")

    # Define hyperparameters for grid search
    # VPS (variables per split): 4-12 step 2
    # LP (min leaf population): 2-12 step 2
    param_grid = {
        'max_features': [12],
        'min_samples_leaf': [2],
    }

    # Create a list of all combinations of hyperparameters
    all_params = list(product(param_grid['max_features'], param_grid['min_samples_leaf']))

    print(f"Running grid search with {len(all_params)} parameter combinations...")
    print(f"Testing: max_features={param_grid['max_features']}, min_samples_leaf={param_grid['min_samples_leaf']}")
    print("")

    # Run only random CV
    print(f"Running Random cross-validation with column: {cv_col}")

    # Use multiprocessing to speed up grid search
    # Adjust number of processes based on available cores
    n_processes = min(multiprocessing.cpu_count() - 1, 8)
    n_processes = max(n_processes, 1)

    with poolcontext(processes=n_processes) as pool:
        results_list = pool.map(
            partial(gridSearch, X=X, y=y, df=df, cv_col=cv_col),
            all_params
        )

    spatial_df = pd.concat(results_list, ignore_index=True)
    best_spatial_r2 = float(spatial_df['Mean_R2_Random'].max())

    # Emit only the best spatial R2 to stdout for shell capture.
    print(f"{best_spatial_r2:.15g}")

PYEOF

    # Run the Python grid search script and capture best spatial R2.
    local best_spatial_r2
    best_spatial_r2=$(python "${TEMP_DIR}/grid_search_${guild}.py" \
        "${guild}" \
        "${temp_training}" \
        "${class_property}")

    echo ""
    echo "Spatial cross-validation complete for ${guild}. Best spatial R2: ${best_spatial_r2}"
    echo ""

    echo "${best_spatial_r2}"
}

# Process Arbuscular Mycorrhizal (AM) fungi
echo ""
echo "######################################"
echo "# Arbuscular Mycorrhizal (AM) Fungi #"
echo "######################################"
echo ""

AM_BEST_SPATIAL_R2=$(run_cv "AM" \
    "${AM_TRAINING_FILE}" \
    "${DATA_DIR}/filtered_randomPoints_AMF.csv" \
    "arbuscular_mycorrhizal_richness" | tail -n 1)

# Process Ectomycorrhizal (EcM) fungi
echo ""
echo "######################################"
echo "# Ectomycorrhizal (EcM) Fungi       #"
echo "######################################"
echo ""

EM_BEST_SPATIAL_R2=$(run_cv "EcM" \
    "${EM_TRAINING_FILE}" \
    "${DATA_DIR}/filtered_randomPoints_ECM.csv" \
    "ectomycorrhizal_richness" | tail -n 1)

# Print final summary
echo ""
echo "=================================="
echo "All processing complete!"
echo "=================================="
echo ""
printf "%s,%s\n" "${AM_BEST_SPATIAL_R2}" "${EM_BEST_SPATIAL_R2}" > "${RESULTS_FILE}"
echo "Wrote machine-readable output to: ${RESULTS_FILE}"

# Clean up temporary directory
echo "Cleaning up temporary files..."
rm -rf "${TEMP_DIR}"

echo ""
echo "Done!"
echo ""
