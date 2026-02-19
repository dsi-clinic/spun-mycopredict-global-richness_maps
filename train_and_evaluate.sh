#!/bin/bash

# Driver script for training and evaluating fungal richness models
# This script performs cross-validation (both random and spatial) for AM and EcM fungal richness
# and outputs R² values for both types of cross-validation.

set -e  # Exit on error

# Configuration
CONDA_BASE=~/miniforge3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
BOX_DATA_DIR="${HOME}/Box/dsi-core/11th-hour/spun/samples"
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
import re

# TEMPORARY DEBUG FILTERS
# Set these to False to restore full-data behavior.
DEBUG_ONLY_R0_ROWS = False
DEBUG_EXCLUDE_BC_COVARIATES = False

# Get configuration from command line arguments
guild = sys.argv[1]
training_file = sys.argv[2]
output_file = sys.argv[3]
class_property = sys.argv[4]

print(f"Loading training data from {training_file}...")
df = pd.read_csv(training_file)

if DEBUG_ONLY_R0_ROWS:
    if 'sample_id' not in df.columns:
        print("ERROR: DEBUG_ONLY_R0_ROWS is enabled, but 'sample_id' column is missing.")
        sys.exit(1)
    before_rows = len(df)
    df = df[df['sample_id'].astype(str).str.startswith('r0_')].copy()
    print(f"DEBUG: filtered to r0_ rows: {before_rows} -> {len(df)}")

if df.empty:
    print("ERROR: No rows remain after debug filtering.")
    sys.exit(1)

def build_alphaearth_covariates(columns):
    """Build and validate the expected 288 AlphaEarth predictor columns."""
    a_pattern = re.compile(r'^A_(\d+)$')
    b_pattern = re.compile(r'^B(\d+)_(\d+)$')
    c_pattern = re.compile(r'^C(\d+)_(\d+)$')

    a_covs = sorted(
        [c for c in columns if a_pattern.fullmatch(c)],
        key=lambda x: int(a_pattern.fullmatch(x).group(1))
    )
    b_covs = sorted(
        [c for c in columns if b_pattern.fullmatch(c)],
        key=lambda x: tuple(map(int, b_pattern.fullmatch(x).groups()))
    )
    c_covs = sorted(
        [c for c in columns if c_pattern.fullmatch(c)],
        key=lambda x: tuple(map(int, c_pattern.fullmatch(x).groups()))
    )

    if DEBUG_EXCLUDE_BC_COVARIATES:
        if len(a_covs) != 64:
            print(f"ERROR: Expected 64 A_* covariates in debug mode, found {len(a_covs)}.")
            sys.exit(1)
        print("DEBUG: using A_* covariates only (excluding B*_* and C*_*).")
        return a_covs

    expected_a = {f"A_{i}" for i in range(64)}
    expected_b = {f"B{i}_{j}" for i in range(16) for j in range(7)}
    expected_c = {f"C{i}_{j}" for i in range(28) for j in range(4)}
    expected_all = expected_a | expected_b | expected_c

    found_all = set(a_covs) | set(b_covs) | set(c_covs)
    missing = sorted(expected_all - found_all)
    extra = sorted(found_all - expected_all)

    if missing:
        print(f"ERROR: Missing expected AlphaEarth covariates ({len(missing)}).")
        print(f"First missing columns: {missing[:10]}")
        sys.exit(1)
    if extra:
        print(f"ERROR: Found unexpected A/B/C covariates ({len(extra)}).")
        print(f"First unexpected columns: {extra[:10]}")
        sys.exit(1)

    covariates = a_covs + b_covs + c_covs
    if len(covariates) != 288:
        print(f"ERROR: Expected 288 AlphaEarth covariates, found {len(covariates)}.")
        sys.exit(1)
    return covariates

covariateList = build_alphaearth_covariates(df.columns)

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

missing_project_vars = [col for col in project_vars if col not in df.columns]
if missing_project_vars:
    print(f"ERROR: Missing expected project variables ({len(missing_project_vars)}).")
    print(f"First missing columns: {missing_project_vars[:10]}")
    sys.exit(1)

covariateList = covariateList + project_vars
print(f"Using {len(covariateList)} total covariates ({len(covariateList) - len(project_vars)} AlphaEarth + {len(project_vars)} project vars).")

# Check for spatial fold column (should have been added by R script)
spatial_fold_col = None
for col in ['knndmw_CV_folds', 'CV_Fold_Spatial']:
    if col in df.columns:
        spatial_fold_col = col
        break

if spatial_fold_col is None:
    print("ERROR: No spatial fold column found in training data!")
    print(f"Available columns: {df.columns.tolist()}")
    sys.exit(1)
else:
    print(f"Using spatial fold column: {spatial_fold_col}")

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
    # VPS (variables per split): 48-144 step 24 (12x scaled from prior 4-12 range)
    # LP (min leaf population): 2-12 step 2
    param_grid = {
        'max_features': [96, 48, 24, 12, 8],
        'min_samples_leaf': [2, 6, 12],
    }

    # Create a list of all combinations of hyperparameters
    all_params = list(product(param_grid['max_features'], param_grid['min_samples_leaf']))

    print(f"Running grid search with {len(all_params)} parameter combinations...")
    print(f"Testing: max_features={param_grid['max_features']}, min_samples_leaf={param_grid['min_samples_leaf']}")
    print("")

    results_list = []

    # Run grid search for both random and spatial CV
    for cv_col in [spatial_fold_col]:
#    for cv_col in ["CV_Fold_Random", spatial_fold_col]:
        cv_type = "Random" if cv_col == "CV_Fold_Random" else "Spatial"
        print(f"Running {cv_type} cross-validation with column: {cv_col}")

        # Use multiprocessing to speed up grid search
        # Adjust number of processes based on available cores
        n_processes = multiprocessing.cpu_count() - 1

        with poolcontext(processes=n_processes) as pool:
            results = pool.map(
                partial(gridSearch, X=X, y=y, df=df, cv_col=cv_col),
                all_params
            )
            results_list.extend(results)

        print("")

    # Combine results from both random and spatial CV
    print("Merging results from random and spatial CV...")

    # Separate random and spatial results
    random_results = [r for r in results_list if 'Mean_R2_Random' in r.columns]
    spatial_results = [r for r in results_list if 'Mean_R2_Spatial' in r.columns]

    # Combine into single dataframes
    random_df = pd.concat(random_results, ignore_index=True)
    spatial_df = pd.concat(spatial_results, ignore_index=True)

    # Merge on cName
    results_df = pd.merge(random_df, spatial_df, on='cName', how='outer')

    # Save to CSV
    results_df.to_csv(output_file, index=False)

    print(f"Grid search results saved to: {output_file}")
    print("")

    # Print summary statistics
    print("Summary of cross-validation results:")
    print("=" * 60)
    print(f"Random CV  - Mean R²: {results_df['Mean_R2_Random'].mean():.4f} ± {results_df['Mean_R2_Random'].std():.4f}")
    print(f"             Best R²: {results_df['Mean_R2_Random'].max():.4f}")
    print(f"Spatial CV - Mean R²: {results_df['Mean_R2_Spatial'].mean():.4f} ± {results_df['Mean_R2_Spatial'].std():.4f}")
    print(f"             Best R²: {results_df['Mean_R2_Spatial'].max():.4f}")
    print("=" * 60)
    print("")

PYEOF

    # Run the Python grid search script
    python "${TEMP_DIR}/grid_search_${guild}.py" \
        "${guild}" \
        "${temp_training}" \
        "${OUTPUT_DIR}/$(date +%Y%m%d)_${class_property}_grid_search_results.csv" \
        "${class_property}"

    echo ""
    echo "Cross-validation complete for ${guild}."
    echo ""
}

# Process Ectomycorrhizal (EcM) fungi
echo ""
echo "######################################"
echo "# Ectomycorrhizal (EcM) Fungi       #"
echo "######################################"
echo ""

run_cv "EcM" \
    "${BOX_DATA_DIR}/20260218_ectomycorrhizal_alphaearth_bigpixels.csv" \
    "${DATA_DIR}/filtered_randomPoints_ECM.csv" \
    "ectomycorrhizal_richness"

# Process Arbuscular Mycorrhizal (AM) fungi
echo ""
echo "######################################"
echo "# Arbuscular Mycorrhizal (AM) Fungi #"
echo "######################################"
echo ""

run_cv "AM" \
    "${BOX_DATA_DIR}/20260218_arbuscular_mycorrhizal_alphaearth_bigpixels.csv" \
    "${DATA_DIR}/filtered_randomPoints_AMF.csv" \
    "arbuscular_mycorrhizal_richness"

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
print("    Random CV  - Best R²: {:.4f} (Mean: {:.4f})".format(
    df['Mean_R2_Random'].max(), df['Mean_R2_Random'].mean()))
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
