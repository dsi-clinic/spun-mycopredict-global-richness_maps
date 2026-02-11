import pathlib

import numpy as np
import pandas as pd

DIRECTORY = pathlib.Path("~/Box/dsi-core/11th-hour/spun/patches/").expanduser()

am_alpha23_file = np.load(DIRECTORY / "alpha_earth_AM_25x25_2017.npz")
am_alpha23_patches = am_alpha23_file["patches"]
am_alpha23_ids = am_alpha23_file["ids"]
am_alpha23_band_names = am_alpha23_file["band_names"]
am_alpha23 = pd.DataFrame(
    {"sample_id": am_alpha23_ids}
    | {
        x: am_alpha23_patches[:, 12, 12, i] for i, x in enumerate(am_alpha23_band_names)
    },
)

em_alpha23_file = np.load(DIRECTORY / "alpha_earth_EM_25x25_2017.npz")
em_alpha23_patches = em_alpha23_file["patches"]
em_alpha23_ids = em_alpha23_file["ids"]
em_alpha23_band_names = em_alpha23_file["band_names"]
em_alpha23 = pd.DataFrame(
    {"sample_id": em_alpha23_ids}
    | {
        x: em_alpha23_patches[:, 12, 12, i] for i, x in enumerate(em_alpha23_band_names)
    },
)

am = pd.read_csv(
    "data/20260122_arbuscular_mycorrhizal_richness_training_data.csv"
).merge(pd.read_csv("data/am_coordinates.csv"), on="sample_id")
ecm = pd.read_csv("data/20260122_ectomycorrhizal_richness_training_data.csv").merge(
    pd.read_csv("data/em_coordinates.csv"), on="sample_id"
)

am_alpha = am.merge(am_alpha23, on="sample_id")
# am_alpha = am_alpha[["sample_id"] + list(am_alpha23_band_names) + list(am.columns[25:])]

ecm_alpha = ecm.merge(em_alpha23, on="sample_id")
# ecm_alpha = ecm_alpha[["sample_id"] + list(em_alpha23_band_names) + list(ecm.columns[25:])]

am_alpha.to_csv(
    "data/20260123_arbuscular_mycorrhizal_alphaearth_center.csv", index=False
)
ecm_alpha.to_csv("data/20260123_ectomycorrhizal_alphaearth_center.csv", index=False)
