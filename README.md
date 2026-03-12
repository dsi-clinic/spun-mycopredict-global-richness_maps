# SPUN Fungal Richness maps
This repository contains code and data to reproduce the Arbuscular Mycorrhizal (AM) and Ectomycorrhizal (ECM) richness maps from Van Nuland et al. 202x doi: 10.xxx.xxx

## Rarefaction
AM and ECM rarefied richness are obtained from [GlobalFungi](https://globalfungi.com/). Richness values are rarefied using the R package `iNEXT`. Code and data to reproduce this are found in the folder `xxx`

## Geospatial modeling
The geospatial modeling approach is divided in several parts, with the respecitve scripts numbered accordingly.
1. Covariate sampling. Here, we extract per-pixel environmental covariate data.
2. Data filtering. Removing outliers per biome.
3. One-hot encoding. Transforming the project-specific variables from multilevel categorical to binary format.
4. Modeling pipeline. The actual modeling approach. 

## Chunked GeoTIFF inference
Use [`infer_original_24_input_layers.py`](/Users/jpivarski/dsi/spun-mycopredict-global-richness_maps/infer_original_24_input_layers.py) to:

1. Read the best AM and EcM hyperparameters from the spatial cross-validation CSVs produced by `train_and_evaluate.sh`.
2. Retrain both random forest models on the full 2026 training CSVs.
3. Run inference against `../original-24-input-layers.tif` in bounded windows.
4. Write one 2-band Float32 GeoTIFF per non-empty window, preserving CRS and georeferencing for easy mosaicking.

The main scaling controls are:
1. `--workers` for the number of concurrent inference workers.
2. `--ram-ceiling-gb` for the approximate total memory budget used to choose window size automatically.
3. `--window-size` if you want to override the automatic chunk size directly.
4. `--mask-geojson data/california.geojson` if you want to limit processing to California.
