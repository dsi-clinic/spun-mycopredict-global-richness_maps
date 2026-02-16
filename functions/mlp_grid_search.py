#!/usr/bin/env python
import datetime
import os
import sys
import multiprocessing
from dataclasses import dataclass
from itertools import product
import random

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


MAX_EPOCHS = int(os.getenv("MLP_MAX_EPOCHS", "220"))
PATIENCE = int(os.getenv("MLP_PATIENCE", "18"))


@dataclass(frozen=True)
class MLPConfig:
    num_layers: int
    hidden_dim: int
    dropout: float
    weight_decay: float
    learning_rate: float
    batch_size: int
    activation: str
    batch_norm: bool
    optimizer: str
    feature_set: str
    stage: str = "search"


class MLP(nn.Module):
    def __init__(
        self, input_dim, num_layers, hidden_dim, dropout, activation, batch_norm=False
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_factory(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def activation_factory(name):
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1)
    raise ValueError(f"Unsupported activation: {name}")


def make_dataloaders(x_train, y_train, x_val, y_val, batch_size, pin_memory):
    if isinstance(x_train, torch.Tensor):
        x_train_t = x_train.float()
        y_train_t = y_train.float().reshape(-1, 1)
        x_val_t = x_val.float()
        y_val_t = y_val.float().reshape(-1, 1)
    else:
        x_train_t = torch.from_numpy(x_train.astype(np.float32))
        y_train_t = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1))
        x_val_t = torch.from_numpy(x_val.astype(np.float32))
        y_val_t = torch.from_numpy(y_val.astype(np.float32).reshape(-1, 1))

    train_ds = TensorDataset(x_train_t, y_train_t)
    val_ds = TensorDataset(x_val_t, y_val_t)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )
    return train_loader, val_loader


def train_one_fold(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    y_scaler,
    config: MLPConfig,
    device,
    max_epochs=300,
    patience=20,
):
    input_dim = x_train.shape[1]
    model = MLP(
        input_dim=input_dim,
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        activation=config.activation,
        batch_norm=config.batch_norm,
    ).to(device)

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    criterion = nn.MSELoss()

    train_loader, val_loader = make_dataloaders(
        x_train,
        y_train,
        x_val,
        y_val,
        config.batch_size,
        pin_memory=(device.type == "cuda" and not isinstance(x_train, torch.Tensor)),
    )

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for _epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        if isinstance(x_test, torch.Tensor):
            x_test_t = x_test.to(device, non_blocking=True)
        else:
            x_test_t = torch.from_numpy(x_test.astype(np.float32)).to(
                device, non_blocking=True
            )
        preds_scaled = model(x_test_t).cpu().numpy().reshape(-1, 1)

    if isinstance(y_scaler, tuple):
        y_mean, y_scale = y_scaler
        preds = (preds_scaled * y_scale + y_mean).reshape(-1)
    else:
        preds = y_scaler.inverse_transform(preds_scaled).reshape(-1)

    r2 = r2_score(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    return r2, rmse, mae


def prepare_folds(
    X,
    y,
    df,
    cv_col,
    random_seed=42,
    use_torch_preprocess=False,
    fold_limit=0,
    train_subsample=0,
    val_subsample=0,
):
    unique_folds = sorted(df[cv_col].dropna().unique().tolist())
    if fold_limit and fold_limit > 0:
        unique_folds = unique_folds[:fold_limit]
    fold_data = []

    for fold in unique_folds:
        train_idx = np.where(df[cv_col] != fold)[0]
        test_idx = np.where(df[cv_col] == fold)[0]

        x_train_full = X[train_idx]
        y_train_full = y[train_idx]
        x_test = X[test_idx]
        y_test = y[test_idx]

        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=0.15,
            random_state=random_seed,
        )

        if train_subsample and train_subsample > 0 and len(x_train) > train_subsample:
            rng = np.random.default_rng(random_seed + int(fold) + 1000)
            idx = rng.choice(len(x_train), size=train_subsample, replace=False)
            x_train = x_train[idx]
            y_train = y_train[idx]
        if val_subsample and val_subsample > 0 and len(x_val) > val_subsample:
            rng = np.random.default_rng(random_seed + int(fold) + 2000)
            idx = rng.choice(len(x_val), size=val_subsample, replace=False)
            x_val = x_val[idx]
            y_val = y_val[idx]

        if use_torch_preprocess:
            fold_data.append((x_train, y_train, x_val, y_val, x_test, y_test))
        else:
            train_missing = np.isnan(x_train)
            val_missing = np.isnan(x_val)
            test_missing = np.isnan(x_test)

            medians = np.nanmedian(x_train, axis=0)
            medians = np.where(np.isnan(medians), 0.0, medians)

            x_train = np.where(np.isnan(x_train), medians, x_train)
            x_val = np.where(np.isnan(x_val), medians, x_val)
            x_test = np.where(np.isnan(x_test), medians, x_test)

            x_scaler = StandardScaler()
            x_train_scaled = x_scaler.fit_transform(x_train)
            x_val_scaled = x_scaler.transform(x_val)
            x_test_scaled = x_scaler.transform(x_test)

            x_train = np.hstack([x_train_scaled, train_missing.astype(np.float32)])
            x_val = np.hstack([x_val_scaled, val_missing.astype(np.float32)])
            x_test = np.hstack([x_test_scaled, test_missing.astype(np.float32)])

            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
            y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)

            fold_data.append(
                (x_train, y_train_scaled, x_val, y_val_scaled, x_test, y_test, y_scaler)
            )

    return fold_data


def load_or_create_tps_centers(df, k, centers_path):
    if os.path.exists(centers_path):
        centers_df = pd.read_csv(centers_path, usecols=["Pixel_Lat", "Pixel_Long"])
        centers = centers_df[["Pixel_Lat", "Pixel_Long"]].to_numpy()
        if len(centers) != k:
            raise ValueError(
                f"Expected {k} TPS centers in {centers_path}, found {len(centers)}."
            )
        return centers

    coords = df[["Pixel_Lat", "Pixel_Long"]].dropna().to_numpy()
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(coords)
    centers = kmeans.cluster_centers_
    pd.DataFrame(centers, columns=["Pixel_Lat", "Pixel_Long"]).to_csv(
        centers_path, index=False
    )
    print(f"Saved TPS centers to {centers_path}")
    return centers


def add_tps_features(df, k, centers_path):
    if "Pixel_Lat" not in df.columns or "Pixel_Long" not in df.columns:
        print("WARNING: Pixel_Lat/Pixel_Long missing; skipping TPS features.")
        return []

    centers = load_or_create_tps_centers(df, k, centers_path)
    coords = np.deg2rad(df[["Pixel_Lat", "Pixel_Long"]].to_numpy())
    centers_rad = np.deg2rad(centers)

    lat = coords[:, 0][:, None]
    lon = coords[:, 1][:, None]
    clat = centers_rad[:, 0][None, :]
    clon = centers_rad[:, 1][None, :]

    dlat = clat - lat
    dlon = clon - lon
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(clat) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    r = 6371.0 * c
    phi = r**2 * np.log(r + 1e-6)

    feature_names = []
    for i in range(k):
        name = f"spatial_tps_{i:02d}"
        df[name] = phi[:, i]
        feature_names.append(name)
    return feature_names


def _prepare_folds_torch(fold_data, device):
    processed = []
    for x_train, y_train, x_val, y_val, x_test, y_test in fold_data:
        x_train_t = torch.from_numpy(x_train.astype(np.float32)).to(device)
        x_val_t = torch.from_numpy(x_val.astype(np.float32)).to(device)
        x_test_t = torch.from_numpy(x_test.astype(np.float32)).to(device)

        train_missing = torch.isnan(x_train_t)
        val_missing = torch.isnan(x_val_t)
        test_missing = torch.isnan(x_test_t)

        if hasattr(torch, "nanmedian"):
            medians = torch.nanmedian(x_train_t, dim=0).values
        else:
            medians = torch.from_numpy(np.nanmedian(x_train, axis=0).astype(np.float32))
            medians = medians.to(device)
        medians = torch.where(torch.isnan(medians), torch.zeros_like(medians), medians)

        x_train_t = torch.where(train_missing, medians, x_train_t)
        x_val_t = torch.where(val_missing, medians, x_val_t)
        x_test_t = torch.where(test_missing, medians, x_test_t)

        x_mean = x_train_t.mean(dim=0)
        x_std = x_train_t.std(dim=0, unbiased=False)
        x_std = torch.where(x_std == 0, torch.ones_like(x_std), x_std)

        x_train_scaled = (x_train_t - x_mean) / x_std
        x_val_scaled = (x_val_t - x_mean) / x_std
        x_test_scaled = (x_test_t - x_mean) / x_std

        x_train_final = torch.cat([x_train_scaled, train_missing.float()], dim=1)
        x_val_final = torch.cat([x_val_scaled, val_missing.float()], dim=1)
        x_test_final = torch.cat([x_test_scaled, test_missing.float()], dim=1)

        y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
        y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)

        y_mean = y_train_t.mean()
        y_std = y_train_t.std(unbiased=False)
        if y_std == 0:
            y_std = torch.tensor(1.0, device=device)

        y_train_scaled = (y_train_t - y_mean) / y_std
        y_val_scaled = (y_val_t - y_mean) / y_std

        y_scaler = (float(y_mean.item()), float(y_std.item()))
        processed.append(
            (
                x_train_final,
                y_train_scaled,
                x_val_final,
                y_val_scaled,
                x_test_final,
                y_test,
                y_scaler,
            )
        )
    return processed


def evaluate_config(
    config: MLPConfig,
    fold_data,
    random_seed=42,
):
    set_seed(random_seed)
    torch.set_num_threads(1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    fold_scores = []

    for (
        x_train,
        y_train_scaled,
        x_val,
        y_val_scaled,
        x_test,
        y_test,
        y_scaler,
    ) in fold_data:
        scores = train_one_fold(
            x_train,
            y_train_scaled,
            x_val,
            y_val_scaled,
            x_test,
            y_test,
            y_scaler,
            config,
            device,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
        )
        fold_scores.append(scores)

    fold_scores = np.array(fold_scores)
    r2_mean = float(np.mean(fold_scores[:, 0]))
    r2_std = float(np.std(fold_scores[:, 0]))
    rmse_mean = float(np.mean(fold_scores[:, 1]))
    rmse_std = float(np.std(fold_scores[:, 1]))
    mae_mean = float(np.mean(fold_scores[:, 2]))
    mae_std = float(np.std(fold_scores[:, 2]))

    return r2_mean, r2_std, rmse_mean, rmse_std, mae_mean, mae_std


_FOLD_DATA = None
_CV_COL = None
_CLASS_PROPERTY = None
_USE_TORCH_PREPROCESS = False


def _init_worker(fold_data, cv_col, class_property, use_torch_preprocess):
    global _FOLD_DATA, _CV_COL, _CLASS_PROPERTY, _USE_TORCH_PREPROCESS
    _USE_TORCH_PREPROCESS = use_torch_preprocess
    if use_torch_preprocess:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _FOLD_DATA = _prepare_folds_torch(fold_data, device)
    else:
        _FOLD_DATA = fold_data
    _CV_COL = cv_col
    _CLASS_PROPERTY = class_property


def _evaluate_worker(config):
    fold_data = _FOLD_DATA
    cv_col = _CV_COL
    class_property = _CLASS_PROPERTY
    r2_mean, r2_std, rmse_mean, rmse_std, mae_mean, mae_std = evaluate_config(
        config=config,
        fold_data=fold_data,
    )
    suffix = "_Random" if cv_col == "CV_Fold_Random" else "_Spatial"
    row = {
        f"Mean_R2{suffix}": r2_mean,
        f"StDev_R2{suffix}": r2_std,
        f"Mean_RMSE{suffix}": rmse_mean,
        f"StDev_RMSE{suffix}": rmse_std,
        f"Mean_MAE{suffix}": mae_mean,
        f"StDev_MAE{suffix}": mae_std,
        "feature_set": config.feature_set,
        "num_layers": config.num_layers,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "weight_decay": config.weight_decay,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "activation": config.activation,
        "batch_norm": config.batch_norm,
        "optimizer": config.optimizer,
        "stage": config.stage,
        "cName": config_name(class_property, config),
    }
    print(f"  {row['cName']} ({cv_col}): R2 = {r2_mean:.4f}")
    return row


def config_name(class_property, config: MLPConfig):
    return (
        f"{class_property}_mlp_L{config.num_layers}_H{config.hidden_dim}"
        f"_DO{config.dropout}_WD{config.weight_decay}_LR{config.learning_rate}"
        f"_BS{config.batch_size}_ACT{config.activation}_BN{int(config.batch_norm)}"
        f"_OPT{config.optimizer}_{config.feature_set}"
    )


def sample_configs_for_feature_set(feature_set, max_trials, seed):
    param_grid = {
        "num_layers": [1, 2, 3, 4],
        "hidden_dim": [64, 128, 256, 384, 512, 768],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8],
        "weight_decay": [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate": [3e-5, 1e-4, 3e-4, 1e-3],
        "batch_size": [64, 128, 256],
        "activation": ["relu", "gelu", "silu", "tanh", "leakyrelu"],
        "batch_norm": [False, True],
        "optimizer": ["adamw", "adam"],
    }

    all_params = list(
        product(
            param_grid["num_layers"],
            param_grid["hidden_dim"],
            param_grid["dropout"],
            param_grid["weight_decay"],
            param_grid["learning_rate"],
            param_grid["batch_size"],
            param_grid["activation"],
            param_grid["batch_norm"],
            param_grid["optimizer"],
        )
    )

    anchors = [
        (3, 512, 0.8, 1e-2, 1e-4, 128, "relu", False, "adamw"),
        (3, 512, 0.3, 1e-4, 3e-4, 128, "relu", True, "adamw"),
        (2, 384, 0.2, 1e-4, 3e-4, 128, "gelu", True, "adamw"),
        (2, 256, 0.1, 1e-5, 1e-3, 128, "silu", True, "adam"),
        (4, 512, 0.2, 1e-4, 1e-4, 64, "relu", True, "adamw"),
    ]

    rng = random.Random(seed)
    selected = []
    seen = set()
    for params in anchors:
        if params in all_params and params not in seen:
            selected.append(params)
            seen.add(params)

    target_count = min(max_trials, len(all_params))
    if len(selected) > target_count:
        selected = selected[:target_count]
        seen = set(selected)

    remaining = [p for p in all_params if p not in seen]
    rng.shuffle(remaining)
    needed = max(0, target_count - len(selected))
    selected.extend(remaining[:needed])

    return [
        MLPConfig(
            num_layers=p[0],
            hidden_dim=p[1],
            dropout=p[2],
            weight_decay=p[3],
            learning_rate=p[4],
            batch_size=p[5],
            activation=p[6],
            batch_norm=p[7],
            optimizer=p[8],
            feature_set=feature_set,
        )
        for p in selected
    ]


def main():
    guild = sys.argv[1]
    training_file = sys.argv[2]
    output_file = sys.argv[3]
    class_property = sys.argv[4]

    print(f"Loading training data from {training_file}...")
    df = pd.read_csv(training_file)

    alpha_covariates = [f"A{i:02d}" for i in range(64)]
    missing_covariates = [col for col in alpha_covariates if col not in df.columns]
    if missing_covariates:
        print(
            "ERROR: Missing expected AlphaEarth covariates in training data: "
            + ", ".join(missing_covariates)
        )
        sys.exit(1)

    spatial_fold_col = None
    for col in ["knndmw_CV_folds", "CV_Fold_Spatial"]:
        if col in df.columns:
            spatial_fold_col = col
            break

    if spatial_fold_col is None:
        print("ERROR: No spatial fold column found in training data!")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    else:
        print(f"Using spatial fold column: {spatial_fold_col}")

    required_cols = [class_property, spatial_fold_col]
    before_rows = len(df)
    df = df.dropna(subset=required_cols)
    dropped = before_rows - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with NaN in target/fold columns.")

    search_seed = int(os.getenv("MLP_SEARCH_SEED", "42"))
    trials_per_feature_set = int(os.getenv("MLP_TRIALS_PER_FEATURE_SET", "24"))
    search_fold_limit = int(os.getenv("MLP_SEARCH_FOLD_LIMIT", "3"))
    search_train_subsample = int(os.getenv("MLP_SEARCH_TRAIN_SUBSAMPLE", "0"))
    search_val_subsample = int(os.getenv("MLP_SEARCH_VAL_SUBSAMPLE", "0"))
    refine_top_n = int(os.getenv("MLP_REFINE_TOP_N", "8"))
    n_processes = int(
        os.getenv(
            "MLP_SEARCH_PROCESSES",
            str(max(1, min(multiprocessing.cpu_count() - 1, 4))),
        )
    )
    tps_ks = [None, 10, 25, 50]

    feature_sets = [("alpha64", df.copy(), alpha_covariates.copy())]
    for k in tps_ks:
        if k is None:
            continue
        df_variant = df.copy()
        centers_path = f"data/tps_centers_k{k}.csv"
        tps_feature_names = add_tps_features(df_variant, k=k, centers_path=centers_path)
        feature_sets.append(
            (
                f"alpha64_tps{k}",
                df_variant,
                alpha_covariates + tps_feature_names,
            )
        )
    feature_set_map = {name: (dfv, covs) for name, dfv, covs in feature_sets}

    print(
        "Running EcM spatial search with "
        f"{trials_per_feature_set} trials per feature set, "
        f"{len(feature_sets)} feature sets, seed={search_seed}, "
        f"processes={n_processes}, search_fold_limit={search_fold_limit}, "
        f"search_train_subsample={search_train_subsample}, "
        f"search_val_subsample={search_val_subsample}, "
        f"refine_top_n={refine_top_n}"
    )
    print("")

    results_rows = []
    cv_col = spatial_fold_col
    cv_type = "Spatial"
    use_torch_preprocess = torch.cuda.is_available()

    def run_tasks(tasks, fold_data):
        if not tasks:
            return
        try:
            if torch.cuda.is_available():
                ctx = multiprocessing.get_context("spawn")
                with ctx.Pool(
                    processes=min(n_processes, len(tasks)),
                    initializer=_init_worker,
                    initargs=(fold_data, cv_col, class_property, use_torch_preprocess),
                ) as pool:
                    for row in pool.imap_unordered(_evaluate_worker, tasks):
                        results_rows.append(row)
                        pd.DataFrame(results_rows).sort_values(
                            "Mean_R2_Spatial", ascending=False
                        ).to_csv(output_file, index=False)
            else:
                with multiprocessing.Pool(
                    processes=min(n_processes, len(tasks)),
                    initializer=_init_worker,
                    initargs=(fold_data, cv_col, class_property, use_torch_preprocess),
                ) as pool:
                    for row in pool.imap_unordered(_evaluate_worker, tasks):
                        results_rows.append(row)
                        pd.DataFrame(results_rows).sort_values(
                            "Mean_R2_Spatial", ascending=False
                        ).to_csv(output_file, index=False)
        except PermissionError:
            print(
                "  Multiprocessing unavailable in this environment; "
                "falling back to serial evaluation."
            )
            _init_worker(fold_data, cv_col, class_property, use_torch_preprocess)
            for task in tasks:
                row = _evaluate_worker(task)
                results_rows.append(row)
                pd.DataFrame(results_rows).sort_values(
                    "Mean_R2_Spatial", ascending=False
                ).to_csv(output_file, index=False)

    for i, (feature_set_name, df_variant, covariate_list) in enumerate(
        feature_sets, start=1
    ):
        print(
            f"[Search {i}/{len(feature_sets)}] Feature set: {feature_set_name} "
            f"({len(covariate_list)} covariates)"
        )
        X = df_variant[covariate_list].to_numpy()
        y = df_variant[class_property].to_numpy()

        tasks = sample_configs_for_feature_set(
            feature_set=feature_set_name,
            max_trials=trials_per_feature_set,
            seed=search_seed + i * 1000,
        )
        print(
            f"  Running {len(tasks)} model configs on {cv_type} CV ({cv_col}) "
            f"with fold_limit={search_fold_limit}"
        )
        fold_data = prepare_folds(
            X,
            y,
            df_variant,
            cv_col,
            use_torch_preprocess=use_torch_preprocess,
            fold_limit=search_fold_limit,
            train_subsample=search_train_subsample,
            val_subsample=search_val_subsample,
        )
        run_tasks(tasks, fold_data)

        if results_rows:
            best_so_far = max(results_rows, key=lambda r: r.get("Mean_R2_Spatial", -1e9))
            print(
                "  Best so far: "
                f"{best_so_far['Mean_R2_Spatial']:.4f} ({best_so_far['cName']})"
            )
        print("")

    if refine_top_n > 0 and search_fold_limit > 0 and results_rows:
        print("Starting full-fold refinement on top search candidates...")
        search_df = pd.DataFrame(results_rows)
        search_df = search_df[search_df["stage"] == "search"]
        search_df = search_df.sort_values("Mean_R2_Spatial", ascending=False)
        top_df = search_df.drop_duplicates(subset=["cName"]).head(refine_top_n)

        for feature_set_name in top_df["feature_set"].unique():
            df_variant, covariate_list = feature_set_map[feature_set_name]
            X = df_variant[covariate_list].to_numpy()
            y = df_variant[class_property].to_numpy()
            fold_data = prepare_folds(
                X,
                y,
                df_variant,
                cv_col,
                use_torch_preprocess=use_torch_preprocess,
                fold_limit=0,
                train_subsample=0,
                val_subsample=0,
            )

            tasks = []
            for _, row in top_df[top_df["feature_set"] == feature_set_name].iterrows():
                tasks.append(
                    MLPConfig(
                        num_layers=int(row["num_layers"]),
                        hidden_dim=int(row["hidden_dim"]),
                        dropout=float(row["dropout"]),
                        weight_decay=float(row["weight_decay"]),
                        learning_rate=float(row["learning_rate"]),
                        batch_size=int(row["batch_size"]),
                        activation=str(row["activation"]),
                        batch_norm=bool(row["batch_norm"]),
                        optimizer=str(row["optimizer"]),
                        feature_set=str(row["feature_set"]),
                        stage="refine",
                    )
                )

            print(
                f"[Refine] Feature set: {feature_set_name} | "
                f"{len(tasks)} configs | full 10-fold spatial CV"
            )
            run_tasks(tasks, fold_data)
            if results_rows:
                best_so_far = max(
                    results_rows, key=lambda r: r.get("Mean_R2_Spatial", -1e9)
                )
                print(
                    "  Best so far after refine: "
                    f"{best_so_far['Mean_R2_Spatial']:.4f} ({best_so_far['cName']})"
                )
            print("")

    results_df = pd.DataFrame(results_rows).sort_values(
        "Mean_R2_Spatial", ascending=False
    )
    results_df.to_csv(output_file, index=False)

    print(f"Grid search results saved to: {output_file}")
    print("")
    print("Summary of cross-validation results:")
    print("=" * 60)
    if "Mean_R2_Spatial" in results_df.columns:
        print(
            f"Spatial CV - Mean R²: {results_df['Mean_R2_Spatial'].mean():.4f} ± "
            f"{results_df['Mean_R2_Spatial'].std():.4f}"
        )
        print(f"             Best R²: {results_df['Mean_R2_Spatial'].max():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
