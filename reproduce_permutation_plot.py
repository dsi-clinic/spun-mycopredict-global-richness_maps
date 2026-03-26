#!/usr/bin/env python3

import argparse
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import haversine_distances
from sklearn.model_selection import cross_validate

EARTH_RADIUS_KM = 6371.0
ECOREGIONS_PATH = Path("~/Box/dsi-core/11th-hour/spun/Ecoregions2017.zip").expanduser()
DEFAULT_RADII_KM = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    20, 30, 40, 50, 60, 70, 80, 90, 100,
    200, 300, 400, 500, 600, 700, 800, 900, 1000,
    2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
    20000, 30000, 40000,
]


@dataclass(frozen=True)
class GuildConfig:
    name: str
    training_filename: str
    target_column: str
    project_columns: tuple[str, ...]


@dataclass(frozen=True)
class BranchConfig:
    name: str
    repo_dir: Path
    guilds: tuple[GuildConfig, GuildConfig]


BRANCHES = {
    "original24": BranchConfig(
        name="original24",
        repo_dir=Path("richness_maps-original24"),
        guilds=(
            GuildConfig(
                name="AM",
                training_filename="20260122_arbuscular_mycorrhizal_richness_training_data.csv",
                target_column="arbuscular_mycorrhizal_richness",
                project_columns=(
                    "sequencing_platform454Roche", "sequencing_platformIllumina",
                    "sample_typerhizosphere_soil", "sample_typesoil", "sample_typetopsoil",
                    "primersAML1_AML2_then_AMV4_5NF_AMDGR", "primersAML1_AML2_then_NS31_AM1",
                    "primersAML1_AML2_then_nu_SSU_0595_5__nu_SSU_0948_3_", "primersAMV4_5F_AMDGR",
                    "primersAMV4_5NF_AMDGR", "primersGeoA2_AML2_then_NS31_AMDGR",
                    "primersGeoA2_NS4_then_NS31_AML2",
                    "primersGlomerWT0_Glomer1536_then_NS31_AM1A_and_GlomerWT0_Glomer1536_then_NS31_AM1B",
                    "primersGlomerWT0_Glomer1536_then_NS31_AM1A__GlomerWT0_Glomer1536_then_NS31_AM1B",
                    "primersNS1_NS4_then_AML1_AML2", "primersNS1_NS4_then_AMV4_5NF_AMDGR",
                    "primersNS1_NS4_then_NS31_AM1", "primersNS1_NS41_then_AML1_AML2",
                    "primersNS31_AM1", "primersNS31_AML2", "primersWANDA_AML2",
                    "area_sampled", "extraction_dna_mass",
                ),
            ),
            GuildConfig(
                name="EcM",
                training_filename="20260122_ectomycorrhizal_richness_training_data.csv",
                target_column="ectomycorrhizal_richness",
                project_columns=(
                    "sequencing_platform454Roche", "sequencing_platformIllumina",
                    "sequencing_platformIonTorrent", "sequencing_platformPacBio",
                    "sample_typerhizosphere_soil", "sample_typesoil", "sample_typetopsoil",
                    "primers5_8S_Fun_ITS4_Fun", "primersfITS7_ITS4", "primersfITS9_ITS4",
                    "primersgITS7_ITS4", "primersgITS7_ITS4_then_ITS9_ITS4",
                    "primersgITS7_ITS4_ITS4arch", "primersgITS7_ITS4m", "primersgITS7_ITS4ngs",
                    "primersgITS7ngs_ITS4ngsUni", "primersITS_S2F___ITS3_mixed_1_1_ITS4",
                    "primersITS1_ITS4", "primersITS1F_ITS4", "primersITS1F_ITS4_then_fITS7_ITS4",
                    "primersITS1F_ITS4_then_ITS3_ITS4",
                    "primersITS1ngs_ITS4ngs_or_ITS1Fngs_ITS4ngs", "primersITS3_KYO2_ITS4",
                    "primersITS3_ITS4", "primersITS3ngs1_to_5___ITS3ngs10_ITS4ngs",
                    "primersITS3ngs1_to_ITS3ngs11_ITS4ngs", "primersITS86F_ITS4",
                    "primersITS9MUNngs_ITS4ngsUni", "area_sampled", "extraction_dna_mass",
                ),
            ),
        ),
    ),
    "alphaearth": BranchConfig(
        name="alphaearth",
        repo_dir=Path("richness_maps-alphaearth"),
        guilds=(
            GuildConfig(
                name="AM",
                training_filename="20260123_arbuscular_mycorrhizal_only_alphaearth_center.csv",
                target_column="arbuscular_mycorrhizal_richness",
                project_columns=(
                    "sequencing_platform454Roche", "sequencing_platformIllumina",
                    "sample_typerhizosphere_soil", "sample_typesoil", "sample_typetopsoil",
                    "primersAML1_AML2_then_AMV4_5NF_AMDGR", "primersAML1_AML2_then_NS31_AM1",
                    "primersAML1_AML2_then_nu_SSU_0595_5__nu_SSU_0948_3_", "primersAMV4_5F_AMDGR",
                    "primersAMV4_5NF_AMDGR", "primersGeoA2_AML2_then_NS31_AMDGR",
                    "primersGeoA2_NS4_then_NS31_AML2",
                    "primersGlomerWT0_Glomer1536_then_NS31_AM1A_and_GlomerWT0_Glomer1536_then_NS31_AM1B",
                    "primersGlomerWT0_Glomer1536_then_NS31_AM1A__GlomerWT0_Glomer1536_then_NS31_AM1B",
                    "primersNS1_NS4_then_AML1_AML2", "primersNS1_NS4_then_AMV4_5NF_AMDGR",
                    "primersNS1_NS4_then_NS31_AM1", "primersNS1_NS41_then_AML1_AML2",
                    "primersNS31_AM1", "primersNS31_AML2", "primersWANDA_AML2",
                    "area_sampled", "extraction_dna_mass",
                ),
            ),
            GuildConfig(
                name="EcM",
                training_filename="20260123_ectomycorrhizal_only_alphaearth_center.csv",
                target_column="ectomycorrhizal_richness",
                project_columns=(
                    "sequencing_platform454Roche", "sequencing_platformIllumina",
                    "sequencing_platformIonTorrent", "sequencing_platformPacBio",
                    "sample_typerhizosphere_soil", "sample_typesoil", "sample_typetopsoil",
                    "primers5_8S_Fun_ITS4_Fun", "primersfITS7_ITS4", "primersfITS9_ITS4",
                    "primersgITS7_ITS4", "primersgITS7_ITS4_then_ITS9_ITS4",
                    "primersgITS7_ITS4_ITS4arch", "primersgITS7_ITS4m", "primersgITS7_ITS4ngs",
                    "primersgITS7ngs_ITS4ngsUni", "primersITS_S2F___ITS3_mixed_1_1_ITS4",
                    "primersITS1_ITS4", "primersITS1F_ITS4", "primersITS1F_ITS4_then_fITS7_ITS4",
                    "primersITS1F_ITS4_then_ITS3_ITS4",
                    "primersITS1ngs_ITS4ngs_or_ITS1Fngs_ITS4ngs", "primersITS3_KYO2_ITS4",
                    "primersITS3_ITS4", "primersITS3ngs1_to_5___ITS3ngs10_ITS4ngs",
                    "primersITS3ngs1_to_ITS3ngs11_ITS4ngs", "primersITS86F_ITS4",
                    "primersITS9MUNngs_ITS4ngsUni", "area_sampled", "extraction_dna_mass",
                ),
            ),
        ),
    ),
}

PLOT_CONFIGS = (
    ("radius_km", "radius of permutation (km)", "permutation_radius_plot.png"),
    ("mean_permutation_distance_km", "mean distance of permutation (km)", "permutation_mean_distance_plot.png"),
    ("rms_permutation_distance_km", "\u221amean distance\u00b2 of permutation (km)", "permutation_rms_distance_plot.png"),
)
RESTRICTED_PLOT_CONFIGS = (
    ("mean_permutation_distance_km", "mean distance of permutation (km)", "permutation_mean_distance_plot.png"),
    ("rms_permutation_distance_km", "\u221amean distance\u00b2 of permutation (km)", "permutation_rms_distance_plot.png"),
)

REFERENCE_TICKS = [0.1, 1, 10, 100, 1000, 10000]
RESOLVE_COLUMNS = ["ECO_NAME", "BIOME_NAME", "geometry"]
PREPARED_SOURCE_LOCK = threading.Lock()
RESOLVE_LOCK = threading.Lock()
RESOLVE_GDF: gpd.GeoDataFrame | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce unrestricted permutation plots and save reusable intermediate results.")
    parser.add_argument("--output-dir", default="reproduction_outputs/unrestricted_permutation")
    parser.add_argument("--radii-km", nargs="+", type=float, default=DEFAULT_RADII_KM)
    parser.add_argument("--branches", nargs="+", choices=sorted(BRANCHES), default=["original24", "alphaearth"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--branch-workers", type=int, default=7, help="Parallel worker count per branch.")
    parser.add_argument("--restriction-mode", choices=["unrestricted", "ecoregion", "biome"], default="unrestricted")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--redraw-only", action="store_true", help="Skip evaluation and rebuild plots from saved per-radius summaries.")
    return parser.parse_args()


def locate_coordinate_columns(df: pd.DataFrame) -> tuple[str, str]:
    if {"Pixel_Long", "Pixel_Lat"}.issubset(df.columns):
        return "Pixel_Long", "Pixel_Lat"
    if {"longitude", "latitude"}.issubset(df.columns):
        return "longitude", "latitude"
    raise ValueError("Could not locate longitude/latitude columns.")


def environmental_columns(df: pd.DataFrame) -> list[str]:
    start_idx = 1
    stop_idx = df.columns.get_loc("sequencing_platform454Roche")
    if stop_idx <= start_idx:
        raise ValueError("Unexpected feature layout; no environmental columns detected.")
    return list(df.columns[start_idx:stop_idx])


def spatial_fold_column(df: pd.DataFrame) -> str:
    for name in ("knndmw_CV_folds", "CV_Fold_Spatial"):
        if name in df.columns:
            return name
    raise ValueError("No spatial fold column found.")


def stable_seed(*parts: object, base_seed: int) -> int:
    joined = "::".join(str(x) for x in (base_seed, *parts))
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def format_radius(radius_km: float) -> str:
    if radius_km.is_integer():
        return str(int(radius_km))
    return str(radius_km).replace(".", "p")


def radius_dir(output_root: Path, branch_name: str, radius_km: float) -> Path:
    return output_root / branch_name / f"radius_{format_radius(radius_km)}km"


def restriction_column(restriction_mode: str) -> str | None:
    if restriction_mode == "ecoregion":
        return "ECO_NAME"
    if restriction_mode == "biome":
        return "BIOME_NAME"
    return None


def plot_configs_for_mode(restriction_mode: str) -> tuple[tuple[str, str, str], ...]:
    return PLOT_CONFIGS if restriction_mode == "unrestricted" else RESTRICTED_PLOT_CONFIGS


def load_resolve_gdf() -> gpd.GeoDataFrame:
    global RESOLVE_GDF
    if RESOLVE_GDF is not None:
        return RESOLVE_GDF
    with RESOLVE_LOCK:
        if RESOLVE_GDF is None:
            RESOLVE_GDF = gpd.read_file(ECOREGIONS_PATH)[RESOLVE_COLUMNS]
        return RESOLVE_GDF


def prepared_source_path(output_root: Path, branch: BranchConfig, guild: GuildConfig) -> Path:
    return output_root / "prepared_sources" / branch.name / guild.training_filename


def prepare_source_table(output_root: Path, branch: BranchConfig, guild: GuildConfig) -> Path:
    cache_path = prepared_source_path(output_root, branch, guild)
    if cache_path.exists():
        return cache_path

    with PREPARED_SOURCE_LOCK:
        if cache_path.exists():
            return cache_path

        source_path = branch.repo_dir / "data" / guild.training_filename
        df = pd.read_csv(source_path)
        lon_col, lat_col = locate_coordinate_columns(df)
        points = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326",
        )
        resolve_gdf = load_resolve_gdf()
        joined = gpd.sjoin(points[["geometry"]], resolve_gdf, how="left", predicate="intersects")
        joined = joined[~joined.index.duplicated(keep="first")]
        annotated = df.copy()
        annotated["ECO_NAME"] = joined.reindex(df.index)["ECO_NAME"].to_numpy()
        annotated["BIOME_NAME"] = joined.reindex(df.index)["BIOME_NAME"].to_numpy()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.to_csv(cache_path, index=False)
        return cache_path


def pairwise_distances_km(df: pd.DataFrame) -> np.ndarray:
    lon_col, lat_col = locate_coordinate_columns(df)
    coords_radians = np.radians(np.column_stack((df[lat_col].to_numpy(), df[lon_col].to_numpy())))
    return EARTH_RADIUS_KM * haversine_distances(coords_radians)


def permute_environmental_features(
    df: pd.DataFrame,
    radius_km: float,
    seed: int,
    restriction_mode: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    env_cols = environmental_columns(df)
    distances_km = pairwise_distances_km(df)
    out = df.copy()
    category_col = restriction_column(restriction_mode)
    category_values = None
    if category_col is not None:
        category_values = df[category_col].astype(object).to_numpy(copy=True)
        for i, value in enumerate(category_values):
            if pd.isna(value):
                category_values[i] = f"__missing_row_{i}__"

    if radius_km == 0:
        chosen_indices = np.arange(len(df), dtype=int)
    else:
        rng = np.random.default_rng(seed)
        chosen_indices = np.empty(len(df), dtype=int)
        for i in range(len(df)):
            candidates = np.flatnonzero(distances_km[i] <= radius_km)
            if category_values is not None:
                candidates = candidates[category_values[candidates] == category_values[i]]
            if len(candidates) == 0:
                raise ValueError(f"No candidate points found within {radius_km} km for row {i}.")
            chosen_indices[i] = rng.choice(candidates)

    out.loc[:, env_cols] = df.iloc[chosen_indices][env_cols].to_numpy()
    chosen_distances_km = distances_km[np.arange(len(df)), chosen_indices]
    return out, chosen_distances_km


def evaluate_spatial_cv(training_path: Path, guild: GuildConfig) -> float:
    df = pd.read_csv(training_path)
    covariates = environmental_columns(df) + [col for col in guild.project_columns if col in df.columns]
    fold_col = spatial_fold_column(df)
    X = df[covariates]
    y = df[guild.target_column]
    unique_folds = pd.unique(df[fold_col])
    cv_folds = [(np.where(df[fold_col] != fold)[0], np.where(df[fold_col] == fold)[0]) for fold in unique_folds]

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        max_features=6,
        min_samples_leaf=4,
        max_samples=0.632,
        n_jobs=1,
    )
    scores = cross_validate(model, X, y, cv=cv_folds, scoring="r2", n_jobs=1, return_train_score=False)
    return float(np.mean(scores["test_score"]))


def radius_summary_path(output_root: Path, branch_name: str, radius_km: float) -> Path:
    return radius_dir(output_root, branch_name, radius_km) / "radius_summary.csv"


def write_radius_summary(
    branch: BranchConfig,
    radius_km: float,
    output_root: Path,
    base_seed: int,
    restriction_mode: str,
) -> pd.DataFrame:
    outdir = radius_dir(output_root, branch.name, radius_km)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for guild in branch.guilds:
        source_path = prepare_source_table(output_root, branch, guild) if restriction_mode != "unrestricted" else branch.repo_dir / "data" / guild.training_filename
        training_path = outdir / f"{guild.name.lower()}_training_permuted.csv"
        df = pd.read_csv(source_path)
        permuted_df, chosen_distances_km = permute_environmental_features(
            df=df,
            radius_km=radius_km,
            seed=stable_seed(branch.name, guild.name, radius_km, base_seed=base_seed),
            restriction_mode=restriction_mode,
        )
        permuted_df.to_csv(training_path, index=False)
        r2 = evaluate_spatial_cv(training_path, guild)
        rows.append(
            {
                "restriction_mode": restriction_mode,
                "branch": branch.name,
                "guild": guild.name,
                "radius_km": radius_km,
                "r2": r2,
                "mean_permutation_distance_km": float(np.mean(chosen_distances_km)),
                "rms_permutation_distance_km": float(np.sqrt(np.mean(np.square(chosen_distances_km)))),
                "training_path": str(training_path.resolve()),
            }
        )

    summary = pd.DataFrame(rows).sort_values("guild").reset_index(drop=True)
    summary.to_csv(radius_summary_path(output_root, branch.name, radius_km), index=False)
    return summary


def collect_saved_results(output_root: Path, branches: list[str] | None = None) -> pd.DataFrame:
    root = Path(output_root)
    summary_paths = sorted(root.glob("*/radius_*/radius_summary.csv"))
    if branches is not None:
        summary_paths = [path for path in summary_paths if path.parts[-3] in branches]
    frames = [pd.read_csv(path) for path in summary_paths]
    if not frames:
        return pd.DataFrame(
            columns=[
                "restriction_mode", "branch", "guild", "radius_km", "r2",
                "mean_permutation_distance_km", "rms_permutation_distance_km", "training_path",
            ]
        )
    results = pd.concat(frames, ignore_index=True)
    return results.sort_values(["branch", "guild", "radius_km"]).reset_index(drop=True)


def save_combined_results(output_root: Path, branches: list[str] | None = None) -> pd.DataFrame:
    results = collect_saved_results(output_root, branches=branches)
    results.to_csv(Path(output_root) / "permutation_r2_results.csv", index=False)
    return results


def make_plot(results: pd.DataFrame, x_col: str, x_label: str, plot_path: Path) -> None:
    if results.empty:
        return

    plt.figure(figsize=(6.5, 5))
    styles = {
        ("original24", "AM"): {"color": "tab:orange", "linestyle": "-", "label": "AM, original 24 vars"},
        ("original24", "EcM"): {"color": "tab:blue", "linestyle": "-", "label": "EcM, original 24 vars"},
        ("alphaearth", "AM"): {"color": "tab:orange", "linestyle": "--", "label": "AM, AlphaEarth"},
        ("alphaearth", "EcM"): {"color": "tab:blue", "linestyle": "--", "label": "EcM, AlphaEarth"},
    }

    for key, style in styles.items():
        branch, guild = key
        subset = results[(results["branch"] == branch) & (results["guild"] == guild)].sort_values(x_col)
        if subset.empty:
            continue
        x = subset[x_col].to_numpy(dtype=float)
        x_for_plot = np.where(x > 0, x, 0.1)
        plt.plot(x_for_plot, subset["r2"], marker="o", markersize=3, linewidth=1.4, **style)

    plt.xscale("log")
    max_x = float(results[x_col].max())
    plt.xlim(0.1, max(max_x, 1.0))
    plt.xticks(REFERENCE_TICKS)
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x):,}" if x >= 1000 else f"{x:g}")
    )
    plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    plt.gca().xaxis.set_minor_formatter(NullFormatter())
    plt.ylim(0.0, 0.30)
    plt.xlabel(x_label)
    plt.ylabel(r"$R^2$")
    plt.grid(True, which="major", linestyle=":", color="0.7")
    plt.legend(loc="upper right", framealpha=1.0)
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()


def make_all_plots(results: pd.DataFrame, output_root: Path, restriction_mode: str) -> None:
    for x_col, x_label, filename in plot_configs_for_mode(restriction_mode):
        make_plot(results, x_col, x_label, Path(output_root) / filename)


def process_radius(
    branch: BranchConfig,
    radius_km: float,
    output_root: Path,
    seed: int,
    overwrite: bool,
    print_lock: threading.Lock,
    restriction_mode: str,
) -> pd.DataFrame:
    summary_path = radius_summary_path(output_root, branch.name, radius_km)
    if summary_path.exists() and not overwrite:
        with print_lock:
            print(f"[{restriction_mode}:{branch.name}] radius={radius_km:g} km reused")
        return pd.read_csv(summary_path)

    with print_lock:
        print(f"[{restriction_mode}:{branch.name}] radius={radius_km:g} km started")
    summary = write_radius_summary(branch, radius_km, output_root, seed, restriction_mode)
    with print_lock:
        print(f"[{restriction_mode}:{branch.name}] radius={radius_km:g} km finished")
    return summary


def process_branch(
    branch: BranchConfig,
    radii_km: list[float],
    output_root: Path,
    seed: int,
    overwrite: bool,
    workers: int,
    print_lock: threading.Lock,
    restriction_mode: str,
) -> None:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_radius, branch, radius_km, output_root, seed, overwrite, print_lock, restriction_mode)
            for radius_km in radii_km
        ]
        for future in as_completed(futures):
            future.result()
            save_combined_results(output_root)


def main() -> None:
    args = parse_args()
    args.radii_km = [float(x) for x in args.radii_km]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.redraw_only:
        print_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=len(args.branches)) as branch_executor:
            futures = [
                branch_executor.submit(
                    process_branch,
                    BRANCHES[branch_name],
                    args.radii_km,
                    output_root,
                    args.seed,
                    args.overwrite,
                    args.branch_workers,
                    print_lock,
                    args.restriction_mode,
                )
                for branch_name in args.branches
            ]
            for future in as_completed(futures):
                future.result()

    results = save_combined_results(output_root, branches=args.branches)
    make_all_plots(results, output_root, args.restriction_mode)
    print(f"Wrote results to {(output_root / 'permutation_r2_results.csv').resolve()}")
    for _, _, filename in plot_configs_for_mode(args.restriction_mode):
        print(f"Wrote plot to {(output_root / filename).resolve()}")


if __name__ == "__main__":
    main()
