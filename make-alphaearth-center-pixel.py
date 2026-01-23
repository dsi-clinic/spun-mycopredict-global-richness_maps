import pathlib

import numpy as np
import pandas as pd

DIRECTORY = pathlib.Path("~/Box/dsi-core/11th-hour/spun/patches/").expanduser()

alpha23_file = np.load(DIRECTORY / "alpha_earth_COMBINED_19x19_2023.npz")
alpha23_patches = alpha23_file["patches"]
alpha23_ids = alpha23_file["ids"]
alpha23_band_names = alpha23_file["band_names"]

alpha23 = pd.DataFrame(
    {"sample_id": alpha23_ids} |
    {x: alpha23_patches[:, 9, 9, i] for i, x in enumerate(alpha23_band_names)},
)

am = pd.read_csv("data/20260122_arbuscular_mycorrhizal_richness_training_data.csv")
ecm = pd.read_csv("data/20260122_ectomycorrhizal_richness_training_data.csv")

am_alpha = am.merge(alpha23, on="sample_id")
# am_alpha = am_alpha[["sample_id"] + list(alpha23_band_names) + list(am.columns[25:])]

ecm_alpha = ecm.merge(alpha23, on="sample_id")
# ecm_alpha = ecm_alpha[["sample_id"] + list(alpha23_band_names) + list(ecm.columns[25:])]

am_alpha.to_csv("data/20260123_arbuscular_mycorrhizal_alphaearth_center.csv", index=False)
ecm_alpha.to_csv("data/20260123_ectomycorrhizal_alphaearth_center.csv", index=False)
