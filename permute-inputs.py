import random
import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

EARTH_RADIUS_KM = 6371.0

DISTANCE_CUT, AM_OUTPUT, EM_OUTPUT = sys.argv[1:]
DISTANCE_CUT = float(DISTANCE_CUT)

print(f"Running {DISTANCE_CUT} {AM_OUTPUT} {EM_OUTPUT}")

am = pd.read_csv("data/20260123_arbuscular_mycorrhizal_only_alphaearth_center.csv")
if "longitude" in am.columns:
    longitude = am["longitude"]
    latitude = am["latitude"]
else:
    longitude = am["Pixel_Long"]
    latitude = am["Pixel_Lat"]

am_distances = EARTH_RADIUS_KM * haversine_distances(np.radians(np.column_stack((latitude, longitude))))
original_features = am.iloc[:, 1:list(am.columns).index("sequencing_platform454Roche")]
random_features = original_features.copy()
random_features.iloc[:, :] = np.nan
for i in range(len(am)):
    j = random.choice(np.nonzero(am_distances[i] <= DISTANCE_CUT)[0])
    random_features.iloc[i] = original_features.iloc[j]
am[am.columns[1:list(am.columns).index("sequencing_platform454Roche")]] = random_features
am.to_csv(AM_OUTPUT, index=False)

del am, longitude, latitude, am_distances, original_features, random_features, i, j

em = pd.read_csv("data/20260123_ectomycorrhizal_only_alphaearth_center.csv")
if "longitude" in em.columns:
    longitude = em["longitude"]
    latitude = em["latitude"]
else:
    longitude = em["Pixel_Long"]
    latitude = em["Pixel_Lat"]

em_distances = EARTH_RADIUS_KM * haversine_distances(np.radians(np.column_stack((latitude, longitude))))
original_features = em.iloc[:, 1:list(em.columns).index("sequencing_platform454Roche")]
random_features = original_features.copy()
random_features.iloc[:, :] = np.nan
for i in range(len(em)):
    j = random.choice(np.nonzero(em_distances[i] <= DISTANCE_CUT)[0])
    random_features.iloc[i] = original_features.iloc[j]
em[em.columns[1:list(em.columns).index("sequencing_platform454Roche")]] = random_features
em.to_csv(EM_OUTPUT, index=False)
