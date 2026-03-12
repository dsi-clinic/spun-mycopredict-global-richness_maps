#!/usr/bin/env python3
"""Train best spatial-CV models and run chunked inference on a 24-band GeoTIFF."""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, transform as window_transform
from sklearn.ensemble import RandomForestRegressor


ENV_FEATURES = [
    "CGIAR_PET",
    "CHELSA_BIO_Annual_Mean_Temperature",
    "CHELSA_BIO_Annual_Precipitation",
    "CHELSA_BIO_Max_Temperature_of_Warmest_Month",
    "CHELSA_BIO_Precipitation_Seasonality",
    "ConsensusLandCover_Human_Development_Percentage",
    "EarthEnvTexture_CoOfVar_EVI",
    "EarthEnvTexture_Correlation_EVI",
    "EarthEnvTexture_Homogeneity_EVI",
    "EarthEnvTopoMed_AspectCosine",
    "EarthEnvTopoMed_AspectSine",
    "EarthEnvTopoMed_Elevation",
    "EarthEnvTopoMed_Slope",
    "EarthEnvTopoMed_TopoPositionIndex",
    "EsaCci_BurntAreasProbability",
    "GHS_Population_Density",
    "GlobBiomass_AboveGroundBiomass",
    "MODIS_NPP",
    "SG_Depth_to_bedrock",
    "SG_Sand_Content_005cm",
    "SG_SOC_Content_005cm",
    "SG_Soil_pH_H2O_005cm",
    "plant_diversity",
    "climate_stability_index",
]

AM_PROJECT_DEFAULTS = {
    "sequencing_platform454Roche": 0.0,
    "sequencing_platformIllumina": 1.0,
    "sample_typerhizosphere_soil": 1.0,
    "sample_typesoil": 0.0,
    "sample_typetopsoil": 0.0,
    "primersAML1_AML2_then_AMV4_5NF_AMDGR": 0.0,
    "primersAML1_AML2_then_NS31_AM1": 0.0,
    "primersAML1_AML2_then_nu_SSU_0595_5__nu_SSU_0948_3_": 0.0,
    "primersAMV4_5F_AMDGR": 0.0,
    "primersAMV4_5NF_AMDGR": 1.0,
    "primersGeoA2_AML2_then_NS31_AMDGR": 0.0,
    "primersGeoA2_NS4_then_NS31_AML2": 0.0,
    "primersGlomerWT0_Glomer1536_then_NS31_AM1A_and_GlomerWT0_Glomer1536_then_NS31_AM1B": 0.0,
    "primersGlomerWT0_Glomer1536_then_NS31_AM1A__GlomerWT0_Glomer1536_then_NS31_AM1B": 0.0,
    "primersNS1_NS4_then_AML1_AML2": 0.0,
    "primersNS1_NS4_then_AMV4_5NF_AMDGR": 0.0,
    "primersNS1_NS4_then_NS31_AM1": 0.0,
    "primersNS1_NS41_then_AML1_AML2": 0.0,
    "primersNS31_AM1": 0.0,
    "primersNS31_AML2": 0.0,
    "primersWANDA_AML2": 0.0,
    "area_sampled": 100.0,
    "extraction_dna_mass": 0.5,
}

ECM_PROJECT_DEFAULTS = {
    "sequencing_platform454Roche": 0.0,
    "sequencing_platformIllumina": 1.0,
    "sequencing_platformIonTorrent": 0.0,
    "sequencing_platformPacBio": 0.0,
    "sample_typerhizosphere_soil": 0.0,
    "sample_typesoil": 1.0,
    "sample_typetopsoil": 0.0,
    "primers5_8S_Fun_ITS4_Fun": 0.0,
    "primersfITS7_ITS4": 0.0,
    "primersfITS9_ITS4": 0.0,
    "primersgITS7_ITS4": 0.0,
    "primersgITS7_ITS4_then_ITS9_ITS4": 0.0,
    "primersgITS7_ITS4_ITS4arch": 0.0,
    "primersgITS7_ITS4m": 0.0,
    "primersgITS7_ITS4ngs": 0.0,
    "primersgITS7ngs_ITS4ngsUni": 0.0,
    "primersITS_S2F___ITS3_mixed_1_1_ITS4": 0.0,
    "primersITS1_ITS4": 0.0,
    "primersITS1F_ITS4": 0.0,
    "primersITS1F_ITS4_then_fITS7_ITS4": 0.0,
    "primersITS1F_ITS4_then_ITS3_ITS4": 0.0,
    "primersITS1ngs_ITS4ngs_or_ITS1Fngs_ITS4ngs": 0.0,
    "primersITS3_KYO2_ITS4": 0.0,
    "primersITS3_ITS4": 1.0,
    "primersITS3ngs1_to_5___ITS3ngs10_ITS4ngs": 0.0,
    "primersITS3ngs1_to_ITS3ngs11_ITS4ngs": 0.0,
    "primersITS86F_ITS4": 0.0,
    "primersITS9MUNngs_ITS4ngsUni": 0.0,
    "area_sampled": 100.0,
    "extraction_dna_mass": 0.5,
}

AM_TARGET = "arbuscular_mycorrhizal_richness"
ECM_TARGET = "ectomycorrhizal_richness"

_AM_MODEL = None
_ECM_MODEL = None
_SRC = None
_BAND_INDEXES = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the best spatial-CV AM and EcM random forests, then predict "
            "chunked outputs for a single 24-band GeoTIFF."
        )
    )
    parser.add_argument(
        "--input-tiff",
        default="../original-24-input-layers.tif",
        help="Input 24-band GeoTIFF for inference (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="output/original_24_inference_tiles",
        help="Directory for chunked output GeoTIFFs and model artifacts.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(os.cpu_count() or 1, 8)),
        help="Number of parallel worker processes for inference.",
    )
    parser.add_argument(
        "--ram-ceiling-gb",
        type=float,
        default=50.0,
        help=(
            "Approximate total RAM ceiling in GB used to choose chunk size "
            "across all workers."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="Chunk width/height in pixels. Overrides --ram-ceiling-gb if > 0.",
    )
    parser.add_argument(
        "--am-training-csv",
        default="data/20260122_arbuscular_mycorrhizal_richness_training_data.csv",
        help="AM training CSV.",
    )
    parser.add_argument(
        "--ecm-training-csv",
        default="data/20260122_ectomycorrhizal_richness_training_data.csv",
        help="EcM training CSV.",
    )
    parser.add_argument(
        "--am-grid-search-results",
        default="",
        help=(
            "AM grid-search CSV from train_and_evaluate.sh. If omitted, the latest "
            "matching file under output/ is used."
        ),
    )
    parser.add_argument(
        "--ecm-grid-search-results",
        default="",
        help=(
            "EcM grid-search CSV from train_and_evaluate.sh. If omitted, the latest "
            "matching file under output/ is used."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output tiles and model artifacts.",
    )
    parser.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-row-off", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-col-off", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-height", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-width", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-am-model-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-ecm-model-path", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_grid_search_csv(path_arg: str, pattern: str) -> Path:
    if path_arg:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"Grid-search CSV not found: {path}")
        return path

    matches = sorted(Path("output").glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No files matched output/{pattern}. Run train_and_evaluate.sh first "
            "or pass --am-grid-search-results/--ecm-grid-search-results explicitly."
        )
    return matches[-1]


def parse_best_params(grid_search_csv: Path) -> tuple[int, int, str, float]:
    df = pd.read_csv(grid_search_csv)
    if "Mean_R2_Spatial" not in df.columns or "cName" not in df.columns:
        raise ValueError(f"{grid_search_csv} is missing Mean_R2_Spatial or cName.")

    best = df.sort_values(["Mean_R2_Spatial", "cName"], ascending=[False, True]).iloc[0]
    match = re.search(r"_VPS(\d+)_LP(\d+)_", str(best["cName"]))
    if match is None:
        raise ValueError(f"Could not parse VPS/LP from cName={best['cName']!r}.")
    return int(match.group(1)), int(match.group(2)), str(best["cName"]), float(best["Mean_R2_Spatial"])


def train_model(
    training_csv: Path,
    target: str,
    project_defaults: dict[str, float],
    max_features: int,
    min_samples_leaf: int,
) -> RandomForestRegressor:
    feature_names = ENV_FEATURES + list(project_defaults)
    usecols = feature_names + [target]
    df = pd.read_csv(training_csv, usecols=usecols)

    X = df[feature_names].copy()
    for column, value in project_defaults.items():
        X[column] = X[column].fillna(value)

    y = df[target].to_numpy(dtype=np.float32, copy=False)
    X_np = X.to_numpy(dtype=np.float32, copy=False)
    valid = np.isfinite(y) & np.isfinite(X_np).all(axis=1)
    if not np.any(valid):
        raise ValueError(f"No finite training rows available in {training_csv} for {target}.")

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        max_samples=0.632,
        n_jobs=-1,
    )
    model.fit(X_np[valid], y[valid])
    return model


def choose_window_size(ram_ceiling_gb: float, workers: int, n_features: int) -> int:
    bytes_per_pixel = 640 + (n_features * 8)
    total_budget_bytes = max(ram_ceiling_gb, 1.0) * (1024**3)
    in_flight_budget = total_budget_bytes * 0.60
    per_worker_budget = in_flight_budget / max(workers, 1)
    pixels_per_window = max(int(per_worker_budget / bytes_per_pixel), 1)
    side = int(math.sqrt(pixels_per_window))
    return max(256, min(2048, side))


def resolve_band_indexes(src: rasterio.io.DatasetReader) -> list[int]:
    descriptions = {
        description: index
        for index, description in enumerate(src.descriptions, start=1)
        if description
    }
    if all(name in descriptions for name in ENV_FEATURES):
        return [descriptions[name] for name in ENV_FEATURES]
    if src.count < len(ENV_FEATURES):
        raise ValueError(
            f"Input raster has {src.count} bands, expected at least {len(ENV_FEATURES)}."
        )
    return list(range(1, len(ENV_FEATURES) + 1))


def iter_windows(width: int, height: int, window_size: int) -> list[Window]:
    windows: list[Window] = []
    for row_off in range(0, height, window_size):
        win_height = min(window_size, height - row_off)
        for col_off in range(0, width, window_size):
            win_width = min(window_size, width - col_off)
            windows.append(
                Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=win_width,
                    height=win_height,
                )
            )
    return windows


def tile_filename(window: Window) -> str:
    return (
        f"tile_r{int(window.row_off):05d}_c{int(window.col_off):05d}"
        f"_h{int(window.height):04d}_w{int(window.width):04d}.tif"
    )


def output_is_complete(output_path: Path, window: Window, src: rasterio.io.DatasetReader) -> bool:
    if not output_path.exists() or output_path.stat().st_size <= 0:
        return False

    try:
        with rasterio.open(output_path) as ds:
            expected_transform = window_transform(window, src.transform)
            if ds.count != 2:
                return False
            if ds.width != int(window.width) or ds.height != int(window.height):
                return False
            if ds.crs != src.crs:
                return False
            if ds.transform != expected_transform:
                return False
            if tuple(ds.dtypes) != ("float32", "float32"):
                return False
            if ds.descriptions != ("am_richness", "ecm_richness"):
                return False
            _ = ds.read(1, window=Window(0, 0, 1, 1))
            _ = ds.read(2, window=Window(max(ds.width - 1, 0), max(ds.height - 1, 0), 1, 1))
    except Exception:
        return False

    return True


def init_worker(input_tiff: str, am_model_path: str, ecm_model_path: str) -> None:
    global _AM_MODEL, _ECM_MODEL, _SRC, _BAND_INDEXES
    _AM_MODEL = joblib.load(am_model_path)
    _ECM_MODEL = joblib.load(ecm_model_path)
    _SRC = rasterio.open(input_tiff)
    _BAND_INDEXES = resolve_band_indexes(_SRC)


def predict_window(task: tuple[int, int, int, int, str]) -> dict[str, object]:
    global _AM_MODEL, _ECM_MODEL, _SRC, _BAND_INDEXES
    row_off, col_off, height, width, output_path_str = task
    output_path = Path(output_path_str)
    window = Window(col_off=col_off, row_off=row_off, width=width, height=height)

    data = _SRC.read(indexes=_BAND_INDEXES, window=window, out_dtype=np.float32)
    flat_env = np.moveaxis(data, 0, -1).reshape(-1, len(ENV_FEATURES))
    valid = np.isfinite(flat_env).all(axis=1)
    valid_pixels = int(valid.sum())

    if valid_pixels == 0:
        return {
            "status": "empty",
            "output_path": str(output_path),
            "row_off": row_off,
            "col_off": col_off,
            "height": height,
            "width": width,
            "valid_pixels": 0,
        }

    am_pred = np.full(flat_env.shape[0], np.nan, dtype=np.float32)
    ecm_pred = np.full(flat_env.shape[0], np.nan, dtype=np.float32)
    valid_env = flat_env[valid]

    X_am_valid = np.empty(
        (valid_pixels, len(ENV_FEATURES) + len(AM_PROJECT_DEFAULTS)),
        dtype=np.float32,
    )
    X_am_valid[:, : len(ENV_FEATURES)] = valid_env
    for idx, value in enumerate(AM_PROJECT_DEFAULTS.values(), start=len(ENV_FEATURES)):
        X_am_valid[:, idx] = value

    X_ecm_valid = np.empty(
        (valid_pixels, len(ENV_FEATURES) + len(ECM_PROJECT_DEFAULTS)),
        dtype=np.float32,
    )
    X_ecm_valid[:, : len(ENV_FEATURES)] = valid_env
    for idx, value in enumerate(ECM_PROJECT_DEFAULTS.values(), start=len(ENV_FEATURES)):
        X_ecm_valid[:, idx] = value

    am_pred[valid] = _AM_MODEL.predict(X_am_valid).astype(np.float32, copy=False)
    ecm_pred[valid] = _ECM_MODEL.predict(X_ecm_valid).astype(np.float32, copy=False)

    profile = _SRC.profile.copy()
    profile.update(
        driver="GTiff",
        count=2,
        dtype="float32",
        nodata=np.nan,
        width=width,
        height=height,
        transform=window_transform(window, _SRC.transform),
        compress=profile.get("compress", "lzw"),
        predictor=3,
    )
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(am_pred.reshape(height, width), 1)
        dst.write(ecm_pred.reshape(height, width), 2)
        dst.set_band_description(1, "am_richness")
        dst.set_band_description(2, "ecm_richness")

    return {
        "status": "written",
        "output_path": str(output_path),
        "row_off": row_off,
        "col_off": col_off,
        "height": height,
        "width": width,
        "valid_pixels": valid_pixels,
    }


def write_manifest(output_dir: Path, records: list[dict[str, object]]) -> Path:
    manifest_path = output_dir / "tile_manifest.csv"
    df = pd.DataFrame(records).sort_values(["row_off", "col_off"]).reset_index(drop=True)
    df.to_csv(manifest_path, index=False)
    return manifest_path


def run_worker_subprocess(
    script_path: Path,
    input_tiff: Path,
    am_model_path: Path,
    ecm_model_path: Path,
    task: tuple[int, int, int, int, str],
) -> dict[str, object]:
    row_off, col_off, height, width, output_path = task
    cmd = [
        sys.executable,
        str(script_path),
        "--worker-mode",
        "--input-tiff",
        str(input_tiff),
        "--worker-row-off",
        str(row_off),
        "--worker-col-off",
        str(col_off),
        "--worker-height",
        str(height),
        "--worker-width",
        str(width),
        "--worker-output-path",
        output_path,
        "--worker-am-model-path",
        str(am_model_path),
        "--worker-ecm-model-path",
        str(ecm_model_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker failed for row={row_off} col={col_off}.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    marker = "WORKER_RESULT "
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(marker):
            parts = {}
            for token in line[len(marker) :].split():
                key, value = token.split("=", 1)
                parts[key] = value
            return {
                "status": parts["status"],
                "output_path": parts["output_path"],
                "row_off": int(parts["row_off"]),
                "col_off": int(parts["col_off"]),
                "height": int(parts["height"]),
                "width": int(parts["width"]),
                "valid_pixels": int(parts["valid_pixels"]),
            }

    raise RuntimeError(f"Worker did not emit WORKER_RESULT line.\nstdout:\n{proc.stdout}")


def main() -> int:
    args = parse_args()

    input_tiff = Path(args.input_tiff).resolve()
    output_dir = Path(args.output_dir).resolve()
    tiles_dir = output_dir / "tiles"
    models_dir = output_dir / "models"

    if args.worker_mode:
        init_worker(
            str(input_tiff),
            args.worker_am_model_path,
            args.worker_ecm_model_path,
        )
        record = predict_window(
            (
                args.worker_row_off,
                args.worker_col_off,
                args.worker_height,
                args.worker_width,
                args.worker_output_path,
            )
        )
        print(
            "WORKER_RESULT "
            f"status={record['status']} "
            f"output_path={record['output_path']} "
            f"row_off={record['row_off']} "
            f"col_off={record['col_off']} "
            f"height={record['height']} "
            f"width={record['width']} "
            f"valid_pixels={record['valid_pixels']}"
        )
        return 0

    if not input_tiff.exists():
        raise FileNotFoundError(f"Input TIFF not found: {input_tiff}")

    am_training_csv = Path(args.am_training_csv)
    ecm_training_csv = Path(args.ecm_training_csv)
    am_grid_search_csv = resolve_grid_search_csv(
        args.am_grid_search_results,
        "*arbuscular_mycorrhizal_richness_grid_search_results.csv",
    )
    ecm_grid_search_csv = resolve_grid_search_csv(
        args.ecm_grid_search_results,
        "*ectomycorrhizal_richness_grid_search_results.csv",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    am_max_features, am_min_samples_leaf, am_cname, am_score = parse_best_params(am_grid_search_csv)
    ecm_max_features, ecm_min_samples_leaf, ecm_cname, ecm_score = parse_best_params(ecm_grid_search_csv)

    print(
        f"AM best spatial-CV model: {am_cname} "
        f"(Mean_R2_Spatial={am_score:.4f}) from {am_grid_search_csv}"
    )
    print(
        f"EcM best spatial-CV model: {ecm_cname} "
        f"(Mean_R2_Spatial={ecm_score:.4f}) from {ecm_grid_search_csv}"
    )

    am_model_path = models_dir / f"am_spatial_best_vps{am_max_features}_lp{am_min_samples_leaf}.joblib"
    ecm_model_path = models_dir / f"ecm_spatial_best_vps{ecm_max_features}_lp{ecm_min_samples_leaf}.joblib"

    if args.overwrite or not am_model_path.exists():
        print("Training AM model on full training data...")
        am_model = train_model(
            am_training_csv,
            AM_TARGET,
            AM_PROJECT_DEFAULTS,
            am_max_features,
            am_min_samples_leaf,
        )
        joblib.dump(am_model, am_model_path)
    else:
        print(f"Reusing existing AM model: {am_model_path}")

    if args.overwrite or not ecm_model_path.exists():
        print("Training EcM model on full training data...")
        ecm_model = train_model(
            ecm_training_csv,
            ECM_TARGET,
            ECM_PROJECT_DEFAULTS,
            ecm_max_features,
            ecm_min_samples_leaf,
        )
        joblib.dump(ecm_model, ecm_model_path)
    else:
        print(f"Reusing existing EcM model: {ecm_model_path}")

    with rasterio.open(input_tiff) as src:
        if args.window_size > 0:
            window_size = args.window_size
        else:
            window_size = choose_window_size(
                args.ram_ceiling_gb,
                args.workers,
                len(ENV_FEATURES) + max(len(AM_PROJECT_DEFAULTS), len(ECM_PROJECT_DEFAULTS)),
            )
        band_indexes = resolve_band_indexes(src)
        windows = iter_windows(src.width, src.height, window_size)

        print(
            f"Input raster: {input_tiff} ({src.width}x{src.height}, bands used={band_indexes})"
        )
        print(
            f"Window size: {window_size}x{window_size} pixels, workers={args.workers}, "
            f"ram ceiling~{args.ram_ceiling_gb:.1f} GB, total windows={len(windows)}"
        )

        tasks: list[tuple[int, int, int, int, str]] = []
        skipped_existing = 0
        manifest_rows: list[dict[str, object]] = []
        for window in windows:
            output_path = tiles_dir / tile_filename(window)
            if not args.overwrite and output_is_complete(output_path, window, src):
                skipped_existing += 1
                manifest_rows.append(
                    {
                        "status": "existing",
                        "output_path": str(output_path),
                        "row_off": int(window.row_off),
                        "col_off": int(window.col_off),
                        "height": int(window.height),
                        "width": int(window.width),
                        "valid_pixels": -1,
                    }
                )
                continue
            tasks.append(
                (
                    int(window.row_off),
                    int(window.col_off),
                    int(window.height),
                    int(window.width),
                    str(output_path),
                )
            )

    if skipped_existing:
        print(f"Skipping {skipped_existing} already-complete output tiles.")
    if not tasks:
        manifest_path = write_manifest(output_dir, manifest_rows)
        print(f"Nothing to do. Manifest: {manifest_path}")
        return 0

    print(f"Running inference for {len(tasks)} candidate windows...")
    completed = 0
    empty = 0
    written = 0
    script_path = Path(__file__).resolve()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_worker_subprocess,
                script_path,
                input_tiff,
                am_model_path,
                ecm_model_path,
                task,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            record = future.result()
            manifest_rows.append(record)
            completed += 1
            if record["status"] == "empty":
                empty += 1
            else:
                written += 1
            print(
                f"[{completed}/{len(tasks)}] {record['status']}: {record['output_path']} "
                f"(valid_pixels={record['valid_pixels']})"
            )

    manifest_path = write_manifest(output_dir, manifest_rows)
    print(f"Wrote {written} non-empty tiles; skipped {empty} empty tiles.")
    print(f"Manifest: {manifest_path}")
    print(f"Models: {am_model_path}, {ecm_model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
