#!/usr/bin/env python3
"""Inference-only tile prediction for AM and EcM fungal richness.

This script trains two fixed-hyperparameter RandomForest models on the
AlphaEarth-centered training CSVs, then predicts AM/EcM richness for each
input GeoTIFF tile in parallel.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.ensemble import RandomForestRegressor


FEATURE_NAMES = [f"A{i:02d}" for i in range(64)]

AM_TARGET = "arbuscular_mycorrhizal_richness"
ECM_TARGET = "ectomycorrhizal_richness"

AM_TRAINING_CSV = Path("data/20260123_arbuscular_mycorrhizal_only_alphaearth_center.csv")
ECM_TRAINING_CSV = Path("data/20260123_ectomycorrhizal_only_alphaearth_center.csv")

BEST_MAX_FEATURES = 6
BEST_MIN_SAMPLES_LEAF = 4

_AM_MODEL = None
_ECM_MODEL = None
_BAND_INDEXES = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict AM/EcM richness for AlphaEarth California tiles and write "
            "2-band GeoTIFF outputs with mirrored directory structure."
        )
    )
    parser.add_argument(
        "--input-base",
        default="../spun-mycopredict-global/data/alpha_earth_california_2017",
        help="Base directory to mirror from (default: %(default)s).",
    )
    parser.add_argument(
        "--input-glob",
        default="raw_tiles/2017/*/*.tiff",
        help="Glob pattern relative to --input-base (default: %(default)s).",
    )
    parser.add_argument(
        "--output-base",
        default="data/alpha_earth_california_2017",
        help="Output base directory that will mirror --input-base (default: %(default)s).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel workers (default: CPU count).",
    )
    parser.add_argument(
        "--ram-ceiling-gb",
        type=float,
        default=50.0,
        help=(
            "Approximate total RAM ceiling (GB) used to auto-pick window size "
            "across all workers (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=0,
        help=(
            "Window width/height in pixels. If > 0, overrides auto sizing from "
            "--ram-ceiling-gb."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def train_fixed_model(training_csv: Path, target: str) -> RandomForestRegressor:
    usecols = FEATURE_NAMES + [target]
    df = pd.read_csv(training_csv, usecols=usecols)

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=False)
    y = df[target].to_numpy(dtype=np.float32, copy=False)
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        max_features=BEST_MAX_FEATURES,
        min_samples_leaf=BEST_MIN_SAMPLES_LEAF,
        max_samples=0.632,
        n_jobs=1,
    )
    model.fit(X[valid], y[valid])
    return model


def _resolve_band_indexes(src: rasterio.io.DatasetReader) -> list[int]:
    desc_to_index = {name: i for i, name in enumerate(src.descriptions, start=1) if name}
    if all(name in desc_to_index for name in FEATURE_NAMES):
        return [desc_to_index[name] for name in FEATURE_NAMES]
    if src.count < len(FEATURE_NAMES):
        raise ValueError(
            f"Tile has {src.count} bands, expected at least {len(FEATURE_NAMES)}."
        )
    return list(range(1, len(FEATURE_NAMES) + 1))


def _init_worker(am_model_path: str, ecm_model_path: str) -> None:
    global _AM_MODEL, _ECM_MODEL
    _AM_MODEL = joblib.load(am_model_path)
    _ECM_MODEL = joblib.load(ecm_model_path)


def _window_grid(width: int, height: int, window_size: int) -> list[Window]:
    windows: list[Window] = []
    for row_off in range(0, height, window_size):
        h = min(window_size, height - row_off)
        for col_off in range(0, width, window_size):
            w = min(window_size, width - col_off)
            windows.append(Window(col_off=col_off, row_off=row_off, width=w, height=h))
    return windows


def _predict_tile(input_tiff: str, output_tiff: str, window_size: int) -> tuple[str, int]:
    global _BAND_INDEXES

    input_path = Path(input_tiff)
    output_path = Path(output_tiff)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        if _BAND_INDEXES is None:
            _BAND_INDEXES = _resolve_band_indexes(src)

        profile = src.profile.copy()
        profile.update(count=2, dtype="float32", nodata=np.nan)

        with rasterio.open(output_path, "w", **profile) as dst:
            valid_total = 0
            for win in _window_grid(src.width, src.height, window_size):
                tile_data = src.read(indexes=_BAND_INDEXES, window=win, out_dtype=np.float32)
                valid_mask = src.read_masks(1, window=win) > 0

                # Build [n_pixels, n_features] matrix for this window only.
                flat = np.moveaxis(tile_data, 0, -1).reshape(-1, len(FEATURE_NAMES))
                valid_flat = valid_mask.reshape(-1) & np.isfinite(flat).all(axis=1)
                valid_total += int(valid_flat.sum())

                am_pred = np.full(flat.shape[0], np.nan, dtype=np.float32)
                ecm_pred = np.full(flat.shape[0], np.nan, dtype=np.float32)
                if np.any(valid_flat):
                    X_valid = flat[valid_flat]
                    am_pred[valid_flat] = _AM_MODEL.predict(X_valid).astype(
                        np.float32, copy=False
                    )
                    ecm_pred[valid_flat] = _ECM_MODEL.predict(X_valid).astype(
                        np.float32, copy=False
                    )

                dst.write(am_pred.reshape(int(win.height), int(win.width)), 1, window=win)
                dst.write(ecm_pred.reshape(int(win.height), int(win.width)), 2, window=win)

            dst.set_band_description(1, "am_richness")
            dst.set_band_description(2, "ecm_richness")

    return str(output_path), valid_total


def choose_window_size(ram_ceiling_gb: float, workers: int) -> int:
    # Conservative estimate of per-pixel transient memory in bytes while predicting:
    # - input features window (64 bands float32): 64 * 4
    # - reshaped/working feature matrix and sklearn internal copies (approx 3x):
    #   3 * (64 * 4)
    # - predictions/mask/temporary arrays overhead: ~16
    bytes_per_pixel = (4 * 64 * 4) + 16  # 1040 bytes/pixel

    total_budget_bytes = max(ram_ceiling_gb, 1.0) * (1024**3)
    budget_for_windows = total_budget_bytes * 0.60  # leave headroom for models + Python overhead
    per_worker_bytes = budget_for_windows / max(workers, 1)
    pixels_per_window = max(int(per_worker_bytes / bytes_per_pixel), 1)

    side = int(math.sqrt(pixels_per_window))
    # Clamp to practical bounds.
    side = max(256, min(4096, side))
    return side


def main() -> int:
    args = parse_args()

    input_base = Path(args.input_base).resolve()
    output_base = Path(args.output_base).resolve()

    input_tiles = sorted(input_base.glob(args.input_glob))
    if not input_tiles:
        raise FileNotFoundError(
            f"No input tiles found under {input_base} with pattern {args.input_glob!r}."
        )

    jobs: list[tuple[Path, Path]] = []
    for input_tile in input_tiles:
        rel = input_tile.relative_to(input_base)
        output_tile = output_base / rel
        if output_tile.exists() and not args.overwrite:
            continue
        jobs.append((input_tile, output_tile))

    print(f"Found {len(input_tiles)} input tiles.")
    if not jobs:
        print("All outputs already exist; nothing to do.")
        return 0

    if args.window_size > 0:
        window_size = args.window_size
    else:
        window_size = choose_window_size(args.ram_ceiling_gb, args.workers)

    est_windows_per_tile = math.ceil(8192 / window_size) ** 2
    print(
        f"Using window size {window_size}x{window_size} pixels "
        f"(ram ceiling ~{args.ram_ceiling_gb:.1f} GB, workers={args.workers}, "
        f"~{est_windows_per_tile} windows per 8192x8192 tile)."
    )

    print(
        f"Training AM and EcM models (VPS={BEST_MAX_FEATURES}, LP={BEST_MIN_SAMPLES_LEAF})..."
    )
    am_model = train_fixed_model(AM_TRAINING_CSV, AM_TARGET)
    ecm_model = train_fixed_model(ECM_TRAINING_CSV, ECM_TARGET)
    print("Model training complete.")

    with tempfile.TemporaryDirectory(prefix="fungal_models_") as tmpdir:
        am_model_path = Path(tmpdir) / "am_model.joblib"
        ecm_model_path = Path(tmpdir) / "ecm_model.joblib"
        joblib.dump(am_model, am_model_path)
        joblib.dump(ecm_model, ecm_model_path)

        total = len(jobs)
        print(f"Running inference on {total} tiles with {args.workers} workers...")

        completed = 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(str(am_model_path), str(ecm_model_path)),
        ) as executor:
            futures = {
                executor.submit(_predict_tile, str(inp), str(out), window_size): (inp, out)
                for inp, out in jobs
            }
            for future in as_completed(futures):
                completed += 1
                out_path, n_valid = future.result()
                print(f"[{completed}/{total}] wrote {out_path} (valid pixels: {n_valid})")

    print("Inference complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
