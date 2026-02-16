#!/usr/bin/env python
import os
import sys
import math
import json
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


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
    raise ValueError(f"Unsupported activation: {name}")


@dataclass(frozen=True)
class ModelConfig:
    env_width: int
    env_depth: int
    env_dropout: float
    spatial_mode: str  # "none" or "rbf"
    spatial_k: int
    spatial_width: int
    spatial_depth: int
    spatial_dropout: float
    activation: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    huber_delta: float
    huber_mix: float
    alpha_l2: float
    max_epochs: int
    patience: int


class ResidualTower(nn.Module):
    def __init__(self, input_dim, width, depth, dropout, activation):
        super().__init__()
        self.inp = nn.Linear(input_dim, width)
        self.inp_norm = nn.LayerNorm(width)
        self.act = activation_factory(activation)
        self.drop = nn.Dropout(dropout)

        blocks = []
        for _ in range(depth):
            blocks.append(
                nn.ModuleDict(
                    {
                        "fc1": nn.Linear(width, width),
                        "ln1": nn.LayerNorm(width),
                        "fc2": nn.Linear(width, width),
                        "ln2": nn.LayerNorm(width),
                    }
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.drop(self.act(self.inp_norm(self.inp(x))))
        for block in self.blocks:
            z = self.drop(self.act(block["ln1"](block["fc1"](h))))
            z = self.drop(self.act(block["ln2"](block["fc2"](z))))
            h = h + z
        return self.head(h)


class SpatialCorrectionNet(nn.Module):
    def __init__(self, env_dim, spatial_dim, config: ModelConfig):
        super().__init__()
        self.env_tower = ResidualTower(
            input_dim=env_dim,
            width=config.env_width,
            depth=config.env_depth,
            dropout=config.env_dropout,
            activation=config.activation,
        )
        self.use_spatial = spatial_dim > 0
        if self.use_spatial:
            self.spatial_tower = ResidualTower(
                input_dim=spatial_dim,
                width=config.spatial_width,
                depth=config.spatial_depth,
                dropout=config.spatial_dropout,
                activation=config.activation,
            )
            self.raw_alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.spatial_tower = None
            self.raw_alpha = None

    def alpha(self):
        if self.raw_alpha is None:
            return torch.tensor(0.0)
        return torch.nn.functional.softplus(self.raw_alpha)

    def forward(self, x_env, x_spatial):
        env_pred = self.env_tower(x_env)
        if not self.use_spatial:
            zero = torch.zeros_like(env_pred)
            return env_pred, zero, env_pred
        spatial_pred = self.spatial_tower(x_spatial)
        yhat = env_pred + self.alpha() * spatial_pred
        return env_pred, spatial_pred, yhat


def load_or_create_knots(df, k, path):
    if os.path.exists(path):
        knots_df = pd.read_csv(path)
        knots = knots_df[["Pixel_Lat", "Pixel_Long"]].to_numpy()
        if len(knots) != k:
            raise ValueError(f"Expected {k} knots in {path}, found {len(knots)}")
        return knots

    coords = df[["Pixel_Lat", "Pixel_Long"]].dropna().to_numpy()
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(coords)
    knots = kmeans.cluster_centers_
    pd.DataFrame(knots, columns=["Pixel_Lat", "Pixel_Long"]).to_csv(path, index=False)
    return knots


def haversine_km(latlon_a, latlon_b):
    # a: (N,2), b: (M,2)
    a = np.deg2rad(latlon_a)
    b = np.deg2rad(latlon_b)
    lat1 = a[:, [0]]
    lon1 = a[:, [1]]
    lat2 = b[:, 0][None, :]
    lon2 = b[:, 1][None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    aa = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(aa, 0.0, 1.0)))
    return 6371.0 * c


def rbf_features(df, k, knots_path):
    if "Pixel_Lat" not in df.columns or "Pixel_Long" not in df.columns:
        return np.zeros((len(df), 0), dtype=np.float32)

    knots = load_or_create_knots(df, k, knots_path)
    coords = df[["Pixel_Lat", "Pixel_Long"]].to_numpy()
    d = haversine_km(coords, knots)

    kk = haversine_km(knots, knots)
    kk[kk == 0] = np.nan
    nn = np.nanmin(kk, axis=1)
    length_scale = float(np.nanmedian(nn))
    if not np.isfinite(length_scale) or length_scale <= 0:
        length_scale = 500.0

    phi = np.exp(-0.5 * (d / length_scale) ** 2)
    return phi.astype(np.float32)


def build_feature_matrices(df, config: ModelConfig):
    env_cols = [f"A{i:02d}" for i in range(64)]
    missing = [c for c in env_cols if c not in df.columns]
    if missing:
        raise ValueError("Missing AlphaEarth columns: " + ", ".join(missing))

    X_env = df[env_cols].to_numpy(dtype=np.float32)

    if config.spatial_mode == "none":
        X_spatial = np.zeros((len(df), 0), dtype=np.float32)
    elif config.spatial_mode == "rbf":
        knots_path = f"data/spatial_knots_rbf_k{config.spatial_k}.csv"
        X_spatial = rbf_features(df, config.spatial_k, knots_path)
    else:
        raise ValueError(f"Unknown spatial mode: {config.spatial_mode}")

    return X_env, X_spatial


def choose_spatial_val_fold(all_folds, test_fold):
    ordered = sorted(all_folds)
    idx = ordered.index(test_fold)
    # Deterministic neighboring fold as validation fold, excluding test fold.
    val_idx = (idx + 1) % len(ordered)
    return ordered[val_idx]


def preprocess_fold(X_env, X_spatial, y, train_idx, val_idx, test_idx):
    x_env_train = X_env[train_idx]
    x_env_val = X_env[val_idx]
    x_env_test = X_env[test_idx]

    x_sp_train = X_spatial[train_idx]
    x_sp_val = X_spatial[val_idx]
    x_sp_test = X_spatial[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # Impute missing using training medians.
    med = np.nanmedian(x_env_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    x_env_train = np.where(np.isnan(x_env_train), med, x_env_train)
    x_env_val = np.where(np.isnan(x_env_val), med, x_env_val)
    x_env_test = np.where(np.isnan(x_env_test), med, x_env_test)

    env_scaler = StandardScaler()
    x_env_train = env_scaler.fit_transform(x_env_train)
    x_env_val = env_scaler.transform(x_env_val)
    x_env_test = env_scaler.transform(x_env_test)

    if x_sp_train.shape[1] > 0:
        sp_scaler = StandardScaler()
        x_sp_train = sp_scaler.fit_transform(x_sp_train)
        x_sp_val = sp_scaler.transform(x_sp_val)
        x_sp_test = sp_scaler.transform(x_sp_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)

    return (
        x_env_train.astype(np.float32),
        x_sp_train.astype(np.float32),
        y_train_scaled.astype(np.float32),
        x_env_val.astype(np.float32),
        x_sp_val.astype(np.float32),
        y_val_scaled.astype(np.float32),
        x_env_test.astype(np.float32),
        x_sp_test.astype(np.float32),
        y_test.astype(np.float32),
        y_scaler,
    )


def make_fold_data(df, y, X_env, X_spatial, cv_col, fold_limit=0):
    all_folds = sorted(df[cv_col].dropna().unique().tolist())
    eval_folds = all_folds
    if fold_limit and fold_limit > 0:
        eval_folds = all_folds[:fold_limit]

    fold_data = []
    for test_fold in eval_folds:
        val_fold = choose_spatial_val_fold(all_folds, test_fold)
        train_idx = np.where((df[cv_col] != test_fold) & (df[cv_col] != val_fold))[0]
        val_idx = np.where(df[cv_col] == val_fold)[0]
        test_idx = np.where(df[cv_col] == test_fold)[0]

        fold_data.append(
            preprocess_fold(X_env, X_spatial, y, train_idx, val_idx, test_idx)
        )
    return fold_data


def train_one_fold(fold, config: ModelConfig, device):
    (
        x_env_train,
        x_sp_train,
        y_train,
        x_env_val,
        x_sp_val,
        y_val,
        x_env_test,
        x_sp_test,
        y_test,
        y_scaler,
    ) = fold

    x_env_train_t = torch.from_numpy(x_env_train).to(device)
    x_sp_train_t = torch.from_numpy(x_sp_train).to(device)
    y_train_t = torch.from_numpy(y_train).reshape(-1, 1).to(device)

    x_env_val_t = torch.from_numpy(x_env_val).to(device)
    x_sp_val_t = torch.from_numpy(x_sp_val).to(device)
    y_val_t = torch.from_numpy(y_val).reshape(-1, 1).to(device)

    x_env_test_t = torch.from_numpy(x_env_test).to(device)
    x_sp_test_t = torch.from_numpy(x_sp_test).to(device)

    train_ds = TensorDataset(x_env_train_t, x_sp_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    model = SpatialCorrectionNet(
        env_dim=x_env_train.shape[1],
        spatial_dim=x_sp_train.shape[1],
        config=config,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    mse_loss = nn.MSELoss()
    huber_loss = nn.HuberLoss(delta=config.huber_delta)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for _ in range(config.max_epochs):
        model.train()
        for be, bs, by in train_loader:
            optimizer.zero_grad()
            _, _, pred = model(be, bs)
            loss_h = huber_loss(pred, by)
            loss_m = mse_loss(pred, by)
            loss = config.huber_mix * loss_h + (1.0 - config.huber_mix) * loss_m
            if model.raw_alpha is not None and config.alpha_l2 > 0:
                loss = loss + config.alpha_l2 * model.alpha().pow(2)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            _, _, val_pred = model(x_env_val_t, x_sp_val_t)
            val_h = huber_loss(val_pred, y_val_t)
            val_m = mse_loss(val_pred, y_val_t)
            val_loss = float((config.huber_mix * val_h + (1.0 - config.huber_mix) * val_m).item())
            if model.raw_alpha is not None and config.alpha_l2 > 0:
                val_loss += float(config.alpha_l2 * model.alpha().pow(2).item())

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        _, _, pred_scaled = model(x_env_test_t, x_sp_test_t)

    pred_scaled = pred_scaled.detach().cpu().numpy().reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)

    r2 = r2_score(y_test, pred)
    rmse = root_mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    alpha_value = float(model.alpha().detach().cpu().item()) if model.raw_alpha is not None else 0.0
    return r2, rmse, mae, alpha_value


def evaluate_config(df, y, X_env, X_spatial, cv_col, config: ModelConfig, fold_limit=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_data = make_fold_data(df, y, X_env, X_spatial, cv_col, fold_limit=fold_limit)

    scores = []
    alpha_values = []
    for fold in fold_data:
        r2, rmse, mae, alpha_value = train_one_fold(fold, config, device)
        scores.append((r2, rmse, mae))
        alpha_values.append(alpha_value)

    arr = np.array(scores)
    return {
        "Mean_R2_Spatial": float(np.mean(arr[:, 0])),
        "StDev_R2_Spatial": float(np.std(arr[:, 0])),
        "Mean_RMSE_Spatial": float(np.mean(arr[:, 1])),
        "StDev_RMSE_Spatial": float(np.std(arr[:, 1])),
        "Mean_MAE_Spatial": float(np.mean(arr[:, 2])),
        "StDev_MAE_Spatial": float(np.std(arr[:, 2])),
        "Mean_Alpha": float(np.mean(alpha_values)),
    }


def config_name(class_property, c: ModelConfig):
    return (
        f"{class_property}_spatialnet_EW{c.env_width}_ED{c.env_depth}_EDO{c.env_dropout}"
        f"_SM{c.spatial_mode}_K{c.spatial_k}_SW{c.spatial_width}_SD{c.spatial_depth}"
        f"_SDO{c.spatial_dropout}_ACT{c.activation}_LR{c.learning_rate}_WD{c.weight_decay}"
        f"_BS{c.batch_size}_HM{c.huber_mix}_HD{c.huber_delta}_AL2{c.alpha_l2}"
    )


def build_search_configs(max_trials):
    grid = {
        "env_width": [192, 256, 384],
        "env_depth": [1, 2, 3],
        "env_dropout": [0.1, 0.2, 0.3],
        "spatial_mode": ["none", "rbf"],
        "spatial_k": [0, 8, 16],
        "spatial_width": [32, 64],
        "spatial_depth": [1, 2],
        "spatial_dropout": [0.0, 0.1],
        "activation": ["gelu", "silu"],
        "learning_rate": [1e-4, 3e-4, 8e-4],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "batch_size": [128, 256],
        "huber_delta": [1.0, 1.5],
        "huber_mix": [0.6, 0.8],
        "alpha_l2": [0.01, 0.05, 0.1],
    }

    all_cfgs = []
    for values in product(
        grid["env_width"],
        grid["env_depth"],
        grid["env_dropout"],
        grid["spatial_mode"],
        grid["spatial_k"],
        grid["spatial_width"],
        grid["spatial_depth"],
        grid["spatial_dropout"],
        grid["activation"],
        grid["learning_rate"],
        grid["weight_decay"],
        grid["batch_size"],
        grid["huber_delta"],
        grid["huber_mix"],
        grid["alpha_l2"],
    ):
        (
            env_width,
            env_depth,
            env_dropout,
            spatial_mode,
            spatial_k,
            spatial_width,
            spatial_depth,
            spatial_dropout,
            activation,
            learning_rate,
            weight_decay,
            batch_size,
            huber_delta,
            huber_mix,
            alpha_l2,
        ) = values

        if spatial_mode == "none":
            spatial_k_eff = 0
        else:
            if spatial_k == 0:
                continue
            spatial_k_eff = spatial_k

        all_cfgs.append(
            ModelConfig(
                env_width=env_width,
                env_depth=env_depth,
                env_dropout=env_dropout,
                spatial_mode=spatial_mode,
                spatial_k=spatial_k_eff,
                spatial_width=spatial_width,
                spatial_depth=spatial_depth,
                spatial_dropout=spatial_dropout,
                activation=activation,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                huber_delta=huber_delta,
                huber_mix=huber_mix,
                alpha_l2=alpha_l2,
                max_epochs=int(os.getenv("SPATIALNET_MAX_EPOCHS", "180")),
                patience=int(os.getenv("SPATIALNET_PATIENCE", "15")),
            )
        )

    anchors = [
        ModelConfig(256, 2, 0.2, "none", 0, 32, 1, 0.0, "gelu", 3e-4, 1e-4, 128, 1.0, 0.8, 0.05, int(os.getenv("SPATIALNET_MAX_EPOCHS", "180")), int(os.getenv("SPATIALNET_PATIENCE", "15"))),
        ModelConfig(384, 2, 0.2, "none", 0, 32, 1, 0.0, "silu", 3e-4, 1e-4, 128, 1.5, 0.8, 0.05, int(os.getenv("SPATIALNET_MAX_EPOCHS", "180")), int(os.getenv("SPATIALNET_PATIENCE", "15"))),
        ModelConfig(256, 2, 0.2, "rbf", 8, 32, 1, 0.0, "gelu", 3e-4, 1e-4, 128, 1.0, 0.8, 0.1, int(os.getenv("SPATIALNET_MAX_EPOCHS", "180")), int(os.getenv("SPATIALNET_PATIENCE", "15"))),
        ModelConfig(256, 2, 0.2, "rbf", 16, 64, 1, 0.1, "silu", 1e-4, 1e-4, 128, 1.5, 0.8, 0.1, int(os.getenv("SPATIALNET_MAX_EPOCHS", "180")), int(os.getenv("SPATIALNET_PATIENCE", "15"))),
    ]

    seen = set()
    selected = []

    def key(c):
        return (
            c.env_width,
            c.env_depth,
            c.env_dropout,
            c.spatial_mode,
            c.spatial_k,
            c.spatial_width,
            c.spatial_depth,
            c.spatial_dropout,
            c.activation,
            c.learning_rate,
            c.weight_decay,
            c.batch_size,
            c.huber_delta,
            c.huber_mix,
            c.alpha_l2,
        )

    for c in anchors:
        k = key(c)
        if k not in seen:
            selected.append(c)
            seen.add(k)

    rng = np.random.default_rng(int(os.getenv("SPATIALNET_SEARCH_SEED", "42")))
    idx = np.arange(len(all_cfgs))
    rng.shuffle(idx)

    for i in idx:
        c = all_cfgs[int(i)]
        k = key(c)
        if k in seen:
            continue
        selected.append(c)
        seen.add(k)
        if len(selected) >= max_trials:
            break

    return selected[:max_trials]


def main():
    guild = sys.argv[1]
    training_file = sys.argv[2]
    output_file = sys.argv[3]
    class_property = sys.argv[4]

    if guild != "EcM":
        print("Skipping non-EcM run in this branch while developing spatial model.")
        pd.DataFrame().to_csv(output_file, index=False)
        return

    set_seed(int(os.getenv("SPATIALNET_SEED", "42")))

    df = pd.read_csv(training_file)

    spatial_fold_col = None
    for col in ["knndmw_CV_folds", "CV_Fold_Spatial"]:
        if col in df.columns:
            spatial_fold_col = col
            break

    if spatial_fold_col is None:
        print("ERROR: No spatial fold column found in training data.")
        sys.exit(1)

    required = [class_property, spatial_fold_col]
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) != before:
        print(f"Dropped {before - len(df)} rows with missing target/fold values.")

    y = df[class_property].to_numpy(dtype=np.float32)

    max_trials = int(os.getenv("SPATIALNET_MAX_TRIALS", "24"))
    fold_limit = int(os.getenv("SPATIALNET_FOLD_LIMIT", "0"))
    configs = build_search_configs(max_trials=max_trials)

    print(
        f"Running spatial-only model search: trials={len(configs)}, "
        f"fold_limit={fold_limit}, fold_col={spatial_fold_col}"
    )

    rows = []
    for i, cfg in enumerate(configs, start=1):
        X_env, X_spatial = build_feature_matrices(df, cfg)
        metrics = evaluate_config(
            df=df,
            y=y,
            X_env=X_env,
            X_spatial=X_spatial,
            cv_col=spatial_fold_col,
            config=cfg,
            fold_limit=fold_limit,
        )

        row = {
            **metrics,
            "cName": config_name(class_property, cfg),
            "guild": guild,
            "feature_set": f"alpha64_{cfg.spatial_mode}_k{cfg.spatial_k}",
            "env_width": cfg.env_width,
            "env_depth": cfg.env_depth,
            "env_dropout": cfg.env_dropout,
            "spatial_mode": cfg.spatial_mode,
            "spatial_k": cfg.spatial_k,
            "spatial_width": cfg.spatial_width,
            "spatial_depth": cfg.spatial_depth,
            "spatial_dropout": cfg.spatial_dropout,
            "activation": cfg.activation,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "huber_delta": cfg.huber_delta,
            "huber_mix": cfg.huber_mix,
            "alpha_l2": cfg.alpha_l2,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
        }

        rows.append(row)
        df_out = pd.DataFrame(rows).sort_values("Mean_R2_Spatial", ascending=False)
        df_out.to_csv(output_file, index=False)

        best = df_out.iloc[0]
        print(
            f"[{i}/{len(configs)}] {row['Mean_R2_Spatial']:.4f} | "
            f"best={best['Mean_R2_Spatial']:.4f} | {row['cName']}"
        )

    final = pd.DataFrame(rows).sort_values("Mean_R2_Spatial", ascending=False)
    final.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    if len(final):
        print("=" * 60)
        print(
            "Spatial CV - Mean R²: "
            f"{final['Mean_R2_Spatial'].mean():.4f} ± {final['Mean_R2_Spatial'].std():.4f}"
        )
        print(f"             Best R²: {final['Mean_R2_Spatial'].max():.4f}")
        print("=" * 60)
        print("Best config:")
        print(json.dumps(final.iloc[0].to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
