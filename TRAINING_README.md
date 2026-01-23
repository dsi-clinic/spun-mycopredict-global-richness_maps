# Fungal Richness Model Training and Evaluation

This document describes how to use the `train_and_evaluate.sh` script to train and evaluate fungal richness models with cross-validation.

## Overview

The `train_and_evaluate.sh` script is a driver script that orchestrates the complete training and evaluation pipeline for both Arbuscular Mycorrhizal (AM) and Ectomycorrhizal (EcM) fungal richness models. It uses the existing infrastructure in this repository to:

1. Generate spatial cross-validation folds using the KNNDM (k-fold Nearest Neighbor Distance Matching) algorithm
2. Perform hyperparameter grid search with both random and spatial cross-validation
3. Output R² (coefficient of determination) values for both types of cross-validation
4. Clean up intermediate files to ensure fresh results on each run

## Prerequisites

- Conda environment with GDAL, R, and required R packages (available in `~/miniforge3`)
- Python packages: pandas, numpy, scikit-learn, geopandas, shapely
- R packages: data.table, sf, tidyverse, FNN, twosamples

## Input Data

The script expects the following input files to be present in the `data/` directory:

1. **Training data:**
   - `data/20260123_arbuscular_mycorrhizal_only_alphaearth_center_pca.csv` - AM fungal richness training data (AlphaEarth variables only)
   - `data/20260123_ectomycorrhizal_only_alphaearth_center_pca.csv` - EcM fungal richness training data (AlphaEarth variables only)

2. **Random prediction points (for spatial fold generation):**
   - `data/filtered_randomPoints_AMF.csv` - Random points for AM fungi
   - `data/filtered_randomPoints_ECM.csv` - Random points for EcM fungi

### Training Data Format

The training data CSV files should contain:
- Environmental covariates (64 AlphaEarth variables):
  - A00, A01, A02, A03, A04, A05, A06, A07, A08, A09
  - A10, A11, A12, A13, A14, A15, A16, A17, A18, A19
  - A20, A21, A22, A23, A24, A25, A26, A27, A28, A29
  - A30, A31, A32, A33, A34, A35, A36, A37, A38, A39
  - A40, A41, A42, A43, A44, A45, A46, A47, A48, A49
  - A50, A51, A52, A53, A54, A55, A56, A57, A58, A59
  - A60, A61, A62, A63
  - These are AlphaEarth center-pixel features representing learned environmental representations

- Project-specific variables (primers, sequencing platforms, sample types, area_sampled, extraction_dna_mass)
- Resolve_Biome - Biome classification for stratification
- Target variable: `arbuscular_mycorrhizal_richness` or `ectomycorrhizal_richness`
- CV_Fold_Random - Random cross-validation fold assignments (1-10)
- Pixel_Long, Pixel_Lat - Geographic coordinates

## Usage

To run the complete training and evaluation pipeline:

```bash
./train_and_evaluate.sh
```

The script will:

1. Clean up any previous run artifacts (removes old grid search results and temporary files)
2. Process AM fungal richness:
   - Generate spatial CV folds using KNNDM
   - Run hyperparameter grid search (VPS: 4-12 step 2, LP: 2-12 step 2)
   - Evaluate with both random and spatial 10-fold cross-validation
3. Process EcM fungal richness:
   - Same steps as AM
4. Output results and clean up temporary files

## Output

The script generates grid search result files in the `output/` directory:

- `output/YYYYMMDD_arbuscular_mycorrhizal_richness_grid_search_results.csv`
- `output/YYYYMMDD_ectomycorrhizal_richness_grid_search_results.csv`

### Output File Format

Each results CSV contains:

- `cName` - Model configuration name (e.g., `arbuscular_mycorrhizal_richness_rf_VPS4_LP2_REGRESSION`)
- `Mean_R2_Random` - Mean R² from random cross-validation
- `StDev_R2_Random` - Standard deviation of R² from random CV
- `Mean_RMSE_Random` - Mean RMSE from random CV
- `StDev_RMSE_Random` - Standard deviation of RMSE from random CV
- `Mean_MAE_Random` - Mean MAE from random CV
- `StDev_MAE_Random` - Standard deviation of MAE from random CV
- `Mean_R2_Spatial` - Mean R² from spatial cross-validation
- `StDev_R2_Spatial` - Standard deviation of R² from spatial CV
- `Mean_RMSE_Spatial` - Mean RMSE from spatial CV
- `StDev_RMSE_Spatial` - Standard deviation of RMSE from spatial CV
- `Mean_MAE_Spatial` - Mean MAE from spatial CV
- `StDev_MAE_Spatial` - Standard deviation of MAE from spatial CV

The models are sorted by their performance, and you can identify the best hyperparameters by looking at the highest R² values.

## Algorithm Details

### Random Cross-Validation

- 10-fold cross-validation
- Stratified by biome (Resolve_Biome) to ensure even representation
- Folds are pre-assigned in the training data as `CV_Fold_Random`

### Spatial Cross-Validation

- 10-fold spatial cross-validation using KNNDM (k-fold Nearest Neighbor Distance Matching)
- Minimizes spatial autocorrelation between training and test sets
- Uses hierarchical clustering with Ward's linkage
- Matches nearest neighbor distance distributions between training-to-test and training-to-prediction spaces
- Generated using `functions/generateFoldsKNNDM.R`

**Performance Optimization:** The script automatically checks if spatial folds already exist in the training data (by looking for the `knndmw_CV_folds` column). If found, it skips the computationally expensive fold generation step. If not found, it generates the folds and saves them back to the training data file for future reuse. This can save several minutes per run, especially for large datasets like EcM.

### Hyperparameter Grid Search

The script tests the following hyperparameter combinations:

- **VPS (Variables Per Split / max_features):** 4, 6, 8, 10, 12
- **LP (min Leaf Population / min_samples_leaf):** 2, 4, 6, 8, 10, 12

Fixed hyperparameters:
- n_estimators: 250
- max_samples: 0.632 (bag fraction)
- random_state: 42

Total: 30 parameter combinations × 2 CV types = 60 model evaluations per guild

### Model

Random Forest Regressor (sklearn.ensemble.RandomForestRegressor)

## Performance

Expected runtime:
- Spatial fold generation: 2-5 minutes per guild
- Grid search with CV: 10-30 minutes per guild (depending on available CPU cores)
- Total: ~30-60 minutes for both AM and EcM

The script uses multiprocessing to parallelize grid search across parameter combinations.

## Cleaning Up

The script automatically:
- Removes old grid search results before running (ensures fresh computation)
- Creates a temporary directory `.temp_cv` for intermediate files
- Cleans up the temporary directory after completion

If the script is interrupted, you can manually clean up with:

```bash
rm -rf .temp_cv
```

### Regenerating Spatial Folds

If you want to force regeneration of spatial CV folds (e.g., to test different KNNDM parameters), you can:

1. **Remove the column from your training data:**
   ```bash
   # For AM data
   cut -d',' -f1-$(head -1 data/20260123_arbuscular_mycorrhizal_only_alphaearth_center_pca.csv | \
     tr ',' '\n' | grep -n "knndmw_CV_folds" | cut -d':' -f1 | \
     awk '{print $1-1}') data/20260123_arbuscular_mycorrhizal_only_alphaearth_center_pca.csv \
     > temp.csv && mv temp.csv data/20260123_arbuscular_mycorrhizal_only_alphaearth_center_pca.csv
   ```

2. **Or use a Python script:**
   ```python
   import pandas as pd
   df = pd.read_csv('data/20260123_arbuscular_mycorrhizal_only_alphaearth_center_pca.csv')
   df = df.drop(columns=['knndmw_CV_folds'], errors='ignore')
   df.to_csv('data/20260123_arbuscular_mycorrhizal_only_alphaearth_center_pca.csv', index=False)
   ```

The next run will automatically regenerate the spatial folds and save them back.

## Troubleshooting

### "Training data not found"
Ensure the input CSV files are in the correct location (`data/` directory) with the expected filenames.

### "Random points file not found"
The script needs random prediction points for spatial fold generation. Ensure the `filtered_randomPoints_*.csv` files exist in the `data/` directory.

### "No spatial fold column found"
The R script failed to generate spatial folds. Check that:
- R and required packages are installed
- The input data has `Pixel_Long` and `Pixel_Lat` columns
- The conda environment is properly activated

### Conda activation fails
Make sure the conda base directory is correct. If your miniforge3 is in a different location, edit the `CONDA_BASE` variable at the top of the script.

## Modifying the Script

You can customize the script by editing these parameters:

- **Conda environment location:** Change `CONDA_BASE` at the top of the script
- **Hyperparameter grid:** Modify `param_grid` in the Python section (around line 177)
- **Number of CV folds:** Change `--k 10` in the R script call (around line 48)
- **Number of processes:** Adjust `n_processes` calculation (around line 188)
- **Random seed:** Change `random_seed` parameter in `gridSearch` function

## References

The methodology is described in:
- Nature article: `nature-article.pdf` (van den Hoogen et al., 2025)
- KNNDM algorithm: Uses nearest neighbor distance matching for spatial cross-validation
- See `functions/generateFoldsKNNDM.R` for implementation details

## Contact

For questions about this script or the analysis pipeline, please refer to the main repository documentation or the published Nature article.
