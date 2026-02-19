import pathlib

import numpy as np
import pandas as pd

DIRECTORY = pathlib.Path("~/Box/dsi-core/11th-hour/spun/patches/").expanduser()

category = np.array([
    [ 0,  0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,  0,  0,  0,  0],
    [ 0, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18,  3,  3,  3,  3,  4,  4,  4,  4,  0,  0,  0,  0],
    [ 0, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20,  5,  5,  5,  5],
    [ 0, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20,  5,  5,  5,  5],
    [ 0, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20,  5,  5,  5,  5],
    [ 0, 15, 15, 15, 15, 28, 28, 28, 28, 29, 29, 30, 30, 19, 19, 19, 19, 20, 20, 20, 20,  5,  5,  5,  5],
    [ 0, 15, 15, 15, 15, 28, 28, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 21, 21, 21, 21,  6,  6,  6,  6],
    [ 0, 15, 15, 15, 15, 28, 28, 28, 28, 40, 40, 41, 41, 31, 31, 32, 32, 21, 21, 21, 21,  6,  6,  6,  6],
    [ 0, 15, 15, 15, 15, 28, 28, 28, 28, 40, 40, 41, 41, 42, 42, 33, 33, 21, 21, 21, 21,  6,  6,  6,  6],
    [14, 14, 14, 14, 27, 27, 27, 27, 39, 39, 44, 44, 45, 42, 42, 33, 33, 21, 21, 21, 21,  6,  6,  6,  6],
    [14, 14, 14, 14, 27, 27, 27, 27, 39, 39, 44, 44, 43, 43, 34, 34, 22, 22, 22, 22,  7,  7,  7,  7,  0],
    [14, 14, 14, 14, 27, 27, 27, 27, 38, 38, 37, 37, 43, 43, 34, 34, 22, 22, 22, 22,  7,  7,  7,  7,  0],
    [14, 14, 14, 14, 27, 27, 27, 27, 38, 38, 37, 37, 36, 36, 35, 35, 22, 22, 22, 22,  7,  7,  7,  7,  0],
    [13, 13, 13, 13, 26, 26, 26, 26, 25, 25, 25, 25, 36, 36, 35, 35, 22, 22, 22, 22,  7,  7,  7,  7,  0],
    [13, 13, 13, 13, 26, 26, 26, 26, 25, 25, 25, 25, 24, 24, 24, 24, 23, 23, 23, 23,  8,  8,  8,  8,  0],
    [13, 13, 13, 13, 26, 26, 26, 26, 25, 25, 25, 25, 24, 24, 24, 24, 23, 23, 23, 23,  8,  8,  8,  8,  0],
    [13, 13, 13, 13, 26, 26, 26, 26, 25, 25, 25, 25, 24, 24, 24, 24, 23, 23, 23, 23,  8,  8,  8,  8,  0],
    [ 0,  0,  0,  0, 12, 12, 12, 12, 11, 11, 11, 11, 24, 24, 24, 24, 23, 23, 23, 23,  8,  8,  8,  8,  0],
    [ 0,  0,  0,  0, 12, 12, 12, 12, 11, 11, 11, 11, 10, 10, 10, 10,  9,  9,  9,  9,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0, 12, 12, 12, 12, 11, 11, 11, 11, 10, 10, 10, 10,  9,  9,  9,  9,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0, 12, 12, 12, 12, 11, 11, 11, 11, 10, 10, 10, 10,  9,  9,  9,  9,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10, 10, 10, 10,  9,  9,  9,  9,  0,  0,  0,  0,  0],
])

rot_names = ["r0_", "r1_", "r2_", "r3_", "f0_", "f1_", "f2_", "f3_"]

band_names = (
    [f"A_{j}" for j in range(64)] +
    [f"B{i}_{j}" for i in range(16) for j in range(7)] +
    [f"C{i}_{j}" for i in range(28) for j in range(4)]
)


def all_orientations(patches):
    all_patches = [
        patches,
        np.rot90(patches, k=1, axes=(1, 2)),
        np.rot90(patches, k=2, axes=(1, 2)),
        np.rot90(patches, k=3, axes=(1, 2)),
    ]
    patches = np.flip(patches, axis=2)
    all_patches.extend([
        patches,
        np.rot90(patches, k=1, axes=(1, 2)),
        np.rot90(patches, k=2, axes=(1, 2)),
        np.rot90(patches, k=3, axes=(1, 2)),
    ])
    return np.concatenate(all_patches, axis=0)


def pca(array):
    X = array.reshape(-1, 64)
    X_centered = X - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_pca = X_centered @ Vt.T
    return X_pca.reshape(array.shape)


def to_bigpixels(patches):
    bigpixels = np.zeros((len(patches), 46, 64), dtype=patches.dtype)
    np.add.at(
        bigpixels,
        (
            np.arange(len(patches))[:, np.newaxis, np.newaxis],
            category.flatten()[np.newaxis, :, np.newaxis],
            np.arange(64)[np.newaxis, np.newaxis, :],
        ),
        patches.reshape(len(patches), -1, 64),
    )
    square4x4 = pca(bigpixels[:, 1:29, :] / 4**2)
    square2x2 = pca(bigpixels[:, 29:45, :] / 2**2)
    square1x1 = bigpixels[:, 45:46, :]  # no centering, no PCA, no division (by 1)
    return np.concatenate([
        square1x1[:, :, :].reshape(len(bigpixels), -1),  # no channel truncation
        square2x2[:, :, :7].reshape(len(bigpixels), -1),
        square4x4[:, :, :4].reshape(len(bigpixels), -1),
    ], axis=1)


###################

am_alpha2017_file = np.load(DIRECTORY / "alpha_earth_AM_25x25_2017.npz")
if list(am_alpha2017_file["band_names"]) != [f"A{i:02d}" for i in range(64)]:
    raise ValueError("Unexpected AM band names")
if len(am_alpha2017_file["failed_indices"]) != 0:
    raise ValueError("AM file has failed indices")
am_alpha2017_bigpixels = to_bigpixels(all_orientations(am_alpha2017_file["patches"]))
am_alpha2017 = pd.DataFrame(
    {"sample_id": [r + x for r in rot_names for x in am_alpha2017_file["ids"]]} |
    {x: am_alpha2017_bigpixels[:, i] for i, x in enumerate(band_names)}
)
del am_alpha2017_file, am_alpha2017_bigpixels

am = pd.read_csv("data/20260122_arbuscular_mycorrhizal_richness_training_data.csv").merge(
    pd.read_csv("data/am_coordinates.csv"), on="sample_id"
)
am = pd.concat([am.assign(sample_id=x + am["sample_id"].str[:]) for x in rot_names])
am_alpha = am.merge(am_alpha2017, on="sample_id")
am_alpha = am_alpha[["sample_id"] + band_names + list(am.columns[25:])]
am_alpha.to_csv(
    DIRECTORY / ".." / "samples" / "20260218_arbuscular_mycorrhizal_alphaearth_bigpixels.csv",
    index=False,
)
del am_alpha, am_alpha2017

###################

em_alpha2017_file = np.load(DIRECTORY / "alpha_earth_EM_25x25_2017.npz")
if list(em_alpha2017_file["band_names"]) != [f"A{i:02d}" for i in range(64)]:
    raise ValueError("Unexpected EM band names")
if len(em_alpha2017_file["failed_indices"]) != 0:
    raise ValueError("EM file has failed indices")
em_alpha2017_bigpixels = to_bigpixels(all_orientations(em_alpha2017_file["patches"]))
em_alpha2017 = pd.DataFrame(
    {"sample_id": [r + x for r in rot_names for x in em_alpha2017_file["ids"]]} |
    {x: em_alpha2017_bigpixels[:, i] for i, x in enumerate(band_names)}
)
del em_alpha2017_file, em_alpha2017_bigpixels

em = pd.read_csv("data/20260122_ectomycorrhizal_richness_training_data.csv").merge(
    pd.read_csv("data/em_coordinates.csv"), on="sample_id"
)
em = pd.concat([em.assign(sample_id=x + em["sample_id"].str[:]) for x in rot_names])
em_alpha = em.merge(em_alpha2017, on="sample_id")
em_alpha = em_alpha[["sample_id"] + band_names + list(em.columns[25:])]
em_alpha.to_csv(
    DIRECTORY / ".." / "samples" / "20260218_ectomycorrhizal_alphaearth_bigpixels.csv",
    index=False,
)
del em_alpha, em_alpha2017
