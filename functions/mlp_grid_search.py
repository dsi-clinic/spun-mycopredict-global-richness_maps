#!/usr/bin/env python
import datetime
import sys
import multiprocessing
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import torch
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


@dataclass(frozen=True)
class MLPConfig:
    num_layers: int
    hidden_dim: int
    dropout: float
    weight_decay: float
    learning_rate: float
    batch_size: int


class MLP(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, dropout, activation):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
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


def make_dataloaders(x_train, y_train, x_val, y_val, batch_size, pin_memory):
    train_ds = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.float32).reshape(-1, 1)),
    )
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
    max_epochs=200,
    patience=10,
):
    input_dim = x_train.shape[1]
    activation = nn.ReLU()
    model = MLP(
        input_dim=input_dim,
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        activation=activation,
    ).to(device)

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
        pin_memory=(device.type == "cuda"),
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
        x_test_t = torch.from_numpy(x_test.astype(np.float32)).to(
            device, non_blocking=True
        )
        preds_scaled = model(x_test_t).cpu().numpy().reshape(-1, 1)

    preds = y_scaler.inverse_transform(preds_scaled).reshape(-1)

    r2 = r2_score(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    return r2, rmse, mae


def evaluate_config(
    config: MLPConfig,
    X,
    y,
    df,
    cv_col,
    random_seed=42,
):
    set_seed(random_seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    else:
        torch.set_num_threads(1)

    unique_folds = df[cv_col].unique()
    fold_scores = []

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


def _evaluate_worker(args):
    config, X, y, df, cv_col, class_property = args
    r2_mean, r2_std, rmse_mean, rmse_std, mae_mean, mae_std = evaluate_config(
        config=config,
        X=X,
        y=y,
        df=df,
        cv_col=cv_col,
    )
    suffix = "_Random" if cv_col == "CV_Fold_Random" else "_Spatial"
    row = {
        f"Mean_R2{suffix}": r2_mean,
        f"StDev_R2{suffix}": r2_std,
        f"Mean_RMSE{suffix}": rmse_mean,
        f"StDev_RMSE{suffix}": rmse_std,
        f"Mean_MAE{suffix}": mae_mean,
        f"StDev_MAE{suffix}": mae_std,
        "cName": config_name(class_property, config),
    }
    print(f"  {row['cName']} ({cv_col}): R2 = {r2_mean:.4f}")
    return row


def config_name(class_property, config: MLPConfig):
    return (
        f"{class_property}_mlp_L{config.num_layers}_H{config.hidden_dim}"
        f"_DO{config.dropout}_WD{config.weight_decay}_LR{config.learning_rate}"
        f"_BS{config.batch_size}"
    )


def main():
    guild = sys.argv[1]
    training_file = sys.argv[2]
    output_file = sys.argv[3]
    class_property = sys.argv[4]

    print(f"Loading training data from {training_file}...")
    df = pd.read_csv(training_file)

    covariateList = [
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

    if guild == "AM":
        project_vars = [
            "sequencing_platform454Roche",
            "sequencing_platformIllumina",
            "sample_typerhizosphere_soil",
            "sample_typesoil",
            "sample_typetopsoil",
            "primersAML1_AML2_then_AMV4_5NF_AMDGR",
            "primersAML1_AML2_then_NS31_AM1",
            "primersAML1_AML2_then_nu_SSU_0595_5__nu_SSU_0948_3_",
            "primersAMV4_5F_AMDGR",
            "primersAMV4_5NF_AMDGR",
            "primersGeoA2_AML2_then_NS31_AMDGR",
            "primersGeoA2_NS4_then_NS31_AML2",
            "primersGlomerWT0_Glomer1536_then_NS31_AM1A_and_GlomerWT0_Glomer1536_then_NS31_AM1B",
            "primersGlomerWT0_Glomer1536_then_NS31_AM1A__GlomerWT0_Glomer1536_then_NS31_AM1B",
            "primersNS1_NS4_then_AML1_AML2",
            "primersNS1_NS4_then_AMV4_5NF_AMDGR",
            "primersNS1_NS4_then_NS31_AM1",
            "primersNS1_NS41_then_AML1_AML2",
            "primersNS31_AM1",
            "primersNS31_AML2",
            "primersWANDA_AML2",
            "area_sampled",
            "extraction_dna_mass",
        ]
    else:
        project_vars = [
            "sequencing_platform454Roche",
            "sequencing_platformIllumina",
            "sequencing_platformIonTorrent",
            "sequencing_platformPacBio",
            "sample_typerhizosphere_soil",
            "sample_typesoil",
            "sample_typetopsoil",
            "primers5_8S_Fun_ITS4_Fun",
            "primersfITS7_ITS4",
            "primersfITS9_ITS4",
            "primersgITS7_ITS4",
            "primersgITS7_ITS4_then_ITS9_ITS4",
            "primersgITS7_ITS4_ITS4arch",
            "primersgITS7_ITS4m",
            "primersgITS7_ITS4ngs",
            "primersgITS7ngs_ITS4ngsUni",
            "primersITS_S2F___ITS3_mixed_1_1_ITS4",
            "primersITS1_ITS4",
            "primersITS1F_ITS4",
            "primersITS1F_ITS4_then_fITS7_ITS4",
            "primersITS1F_ITS4_then_ITS3_ITS4",
            "primersITS1ngs_ITS4ngs_or_ITS1Fngs_ITS4ngs",
            "primersITS3_KYO2_ITS4",
            "primersITS3_ITS4",
            "primersITS3ngs1_to_5___ITS3ngs10_ITS4ngs",
            "primersITS3ngs1_to_ITS3ngs11_ITS4ngs",
            "primersITS86F_ITS4",
            "primersITS9MUNngs_ITS4ngsUni",
            "area_sampled",
            "extraction_dna_mass",
        ]

    covariateList = covariateList + project_vars

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

    required_cols = [class_property, "CV_Fold_Random", spatial_fold_col]
    before_rows = len(df)
    df = df.dropna(subset=required_cols)
    dropped = before_rows - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with NaN in target/fold columns.")

    X = df[covariateList].to_numpy()
    y = df[class_property].to_numpy()

    param_grid = {
        "num_layers": [2, 3],
        "hidden_dim": [256, 384],
        "dropout": [0.5, 0.8],
        "weight_decay": [1e-3, 1e-2],
        "learning_rate": [1e-4, 3e-4],
        "batch_size": [128],
    }

    all_params = list(
        product(
            param_grid["num_layers"],
            param_grid["hidden_dim"],
            param_grid["dropout"],
            param_grid["weight_decay"],
            param_grid["learning_rate"],
            param_grid["batch_size"],
        )
    )

    print(f"Running grid search with {len(all_params)} parameter combinations...")
    print(
        "Testing: "
        f"layers={param_grid['num_layers']}, "
        f"hidden={param_grid['hidden_dim']}, "
        f"dropout={param_grid['dropout']}, "
        f"weight_decay={param_grid['weight_decay']}, "
        f"lr={param_grid['learning_rate']}, "
        f"batch_size={param_grid['batch_size']}"
    )
    print("")

    results_rows = []

    cv_columns = [spatial_fold_col]
    for cv_col in cv_columns:
        cv_type = "Random" if cv_col == "CV_Fold_Random" else "Spatial"
        print(f"Running {cv_type} cross-validation with column: {cv_col}")

        tasks = [
            (MLPConfig(*params), X, y, df, cv_col, class_property)
            for params in all_params
        ]

        if torch.cuda.is_available():
            for task in tasks:
                results_rows.append(_evaluate_worker(task))
        else:
            n_processes = max(1, min(multiprocessing.cpu_count() - 1, 4))
            with multiprocessing.Pool(processes=n_processes) as pool:
                for row in pool.imap_unordered(_evaluate_worker, tasks):
                    results_rows.append(row)

        print("")

    print("Merging results from random and spatial CV...")

    random_results = [r for r in results_rows if "Mean_R2_Random" in r]
    spatial_results = [r for r in results_rows if "Mean_R2_Spatial" in r]

    random_df = pd.DataFrame(random_results)
    spatial_df = pd.DataFrame(spatial_results)

    if len(random_df) and len(spatial_df):
        results_df = pd.merge(random_df, spatial_df, on="cName", how="outer")
    elif len(spatial_df):
        results_df = spatial_df
    else:
        results_df = random_df
    results_df.to_csv(output_file, index=False)

    print(f"Grid search results saved to: {output_file}")
    print("")
    print("Summary of cross-validation results:")
    print("=" * 60)
    if "Mean_R2_Random" in results_df.columns:
        print(
            f"Random CV  - Mean R²: {results_df['Mean_R2_Random'].mean():.4f} ± "
            f"{results_df['Mean_R2_Random'].std():.4f}"
        )
        print(f"             Best R²: {results_df['Mean_R2_Random'].max():.4f}")
    if "Mean_R2_Spatial" in results_df.columns:
        print(
            f"Spatial CV - Mean R²: {results_df['Mean_R2_Spatial'].mean():.4f} ± "
            f"{results_df['Mean_R2_Spatial'].std():.4f}"
        )
        print(f"             Best R²: {results_df['Mean_R2_Spatial'].max():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
