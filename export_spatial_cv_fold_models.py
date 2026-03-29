#!/usr/bin/env python3
"""Export one trained model per spatial CV fold plus train/test point tables."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from infer_original_24_input_layers import (
    AM_PROJECT_DEFAULTS,
    AM_TARGET,
    ECM_PROJECT_DEFAULTS,
    ECM_TARGET,
    ENV_FEATURES,
    parse_best_params,
    resolve_grid_search_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and save one AM and EcM model for each spatial CV fold, along with "
            "the train/test point assignments."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="output/spatial_cv_fold_models",
        help="Directory where fold subdirectories will be written.",
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
        help="AM grid-search CSV from train_and_evaluate.sh.",
    )
    parser.add_argument(
        "--ecm-grid-search-results",
        default="",
        help="EcM grid-search CSV from train_and_evaluate.sh.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing fold directories and models.",
    )
    return parser.parse_args()


def spatial_fold_column(df: pd.DataFrame) -> str:
    for column in ("knndmw_CV_folds", "CV_Fold_Spatial"):
        if column in df.columns:
            return column
    raise ValueError("No spatial fold column found.")


def train_fold_model(
    df: pd.DataFrame,
    target: str,
    project_defaults: dict[str, float],
    train_mask,
    max_features: int,
    min_samples_leaf: int,
) -> RandomForestRegressor:
    feature_names = ENV_FEATURES + list(project_defaults)
    train_df = df.loc[train_mask, feature_names + [target]].copy()
    for column, value in project_defaults.items():
        train_df[column] = train_df[column].fillna(value)

    X = train_df[feature_names].to_numpy(dtype="float32", copy=False)
    y = train_df[target].to_numpy(dtype="float32", copy=False)
    valid = pd.notna(train_df[target]).to_numpy() & pd.DataFrame(X).notna().all(axis=1).to_numpy()

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        max_samples=0.632,
        n_jobs=-1,
    )
    model.fit(X[valid], y[valid])
    return model


def split_points_table(df: pd.DataFrame, fold_col: str, held_out_fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = [column for column in ["sample_id", "Pixel_Long", "Pixel_Lat", fold_col] if column in df.columns]
    train_points = df.loc[df[fold_col] != held_out_fold, keep].copy()
    test_points = df.loc[df[fold_col] == held_out_fold, keep].copy()
    train_points["split"] = "train"
    test_points["split"] = "test"
    return train_points, test_points


def write_fold_artifacts(
    fold_dir: Path,
    guild: str,
    model: RandomForestRegressor,
    train_points: pd.DataFrame,
    test_points: pd.DataFrame,
    fold_value,
    fold_col: str,
    best_cname: str,
    best_score: float,
) -> None:
    model_path = fold_dir / f"{guild.lower()}_model.joblib"
    joblib.dump(model, model_path)
    train_points.to_csv(fold_dir / f"{guild.lower()}_train_points.csv", index=False)
    test_points.to_csv(fold_dir / f"{guild.lower()}_test_points.csv", index=False)
    metadata = {
        "guild": guild,
        "held_out_fold": int(fold_value),
        "spatial_fold_column": fold_col,
        "best_spatial_model_name": best_cname,
        "best_spatial_r2": best_score,
        "model_path": str(model_path),
        "train_points_csv": str(fold_dir / f"{guild.lower()}_train_points.csv"),
        "test_points_csv": str(fold_dir / f"{guild.lower()}_test_points.csv"),
    }
    (fold_dir / f"{guild.lower()}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    am_grid_csv = resolve_grid_search_csv(
        args.am_grid_search_results,
        "*arbuscular_mycorrhizal_richness_grid_search_results.csv",
    )
    ecm_grid_csv = resolve_grid_search_csv(
        args.ecm_grid_search_results,
        "*ectomycorrhizal_richness_grid_search_results.csv",
    )

    am_vps, am_lp, am_cname, am_score = parse_best_params(am_grid_csv)
    ecm_vps, ecm_lp, ecm_cname, ecm_score = parse_best_params(ecm_grid_csv)

    am_df = pd.read_csv(args.am_training_csv)
    ecm_df = pd.read_csv(args.ecm_training_csv)

    am_fold_col = spatial_fold_column(am_df)
    ecm_fold_col = spatial_fold_column(ecm_df)
    folds = sorted(set(am_df[am_fold_col].dropna().astype(int).unique()) | set(ecm_df[ecm_fold_col].dropna().astype(int).unique()))

    print(f"AM best spatial hyperparameters: VPS={am_vps}, LP={am_lp} from {am_grid_csv}")
    print(f"EcM best spatial hyperparameters: VPS={ecm_vps}, LP={ecm_lp} from {ecm_grid_csv}")
    print(f"Exporting {len(folds)} spatial folds to {output_dir}")

    for fold_value in folds:
        fold_dir = output_dir / f"fold_{int(fold_value):02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        am_model_path = fold_dir / "am_model.joblib"
        ecm_model_path = fold_dir / "ecm_model.joblib"
        if not args.overwrite and am_model_path.exists() and ecm_model_path.exists():
            print(f"Skipping existing fold {fold_value}: {fold_dir}")
            continue

        print(f"Training fold {fold_value} models...")
        am_model = train_fold_model(
            am_df,
            AM_TARGET,
            AM_PROJECT_DEFAULTS,
            am_df[am_fold_col] != fold_value,
            am_vps,
            am_lp,
        )
        ecm_model = train_fold_model(
            ecm_df,
            ECM_TARGET,
            ECM_PROJECT_DEFAULTS,
            ecm_df[ecm_fold_col] != fold_value,
            ecm_vps,
            ecm_lp,
        )

        am_train_points, am_test_points = split_points_table(am_df, am_fold_col, fold_value)
        ecm_train_points, ecm_test_points = split_points_table(ecm_df, ecm_fold_col, fold_value)

        write_fold_artifacts(
            fold_dir,
            "AM",
            am_model,
            am_train_points,
            am_test_points,
            fold_value,
            am_fold_col,
            am_cname,
            am_score,
        )
        write_fold_artifacts(
            fold_dir,
            "EcM",
            ecm_model,
            ecm_train_points,
            ecm_test_points,
            fold_value,
            ecm_fold_col,
            ecm_cname,
            ecm_score,
        )

    workflow = {
        "fold_directories_glob": str(output_dir / "fold_*"),
        "inference_example": (
            "python infer_original_24_input_layers.py "
            "--am-model-path output/spatial_cv_fold_models/fold_01/am_model.joblib "
            "--ecm-model-path output/spatial_cv_fold_models/fold_01/ecm_model.joblib "
            "--output-dir output/fold_01_inference"
        ),
    }
    (output_dir / "workflow.json").write_text(json.dumps(workflow, indent=2) + "\n")
    print(f"Wrote workflow guide: {output_dir / 'workflow.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
