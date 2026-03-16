#!/usr/bin/env python3
"""Extract AlphaEarth bands A00-A63 at embedded California lon/lat points.

This is a one-off, single-file script intended to be copied to the machine that
holds the large GeoTIFF tiles. It keeps memory usage low by:

1. Reading only raster metadata while locating which tile contains each point.
2. Reading a single 1x1 pixel window for each matched point.
3. Never materializing a full tile or even a full raster block in memory.

Requirements on the target machine:
  - Python 3
  - rasterio
  - pyproj

Example:
  python extract_alphaearth_california_points.py \
    --zone10-dir /path/to/raw_tiles/2017/10N \
    --zone11-dir /path/to/raw_tiles/2017/11N \
    --output-csv alphaearth_california_points_64bands.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import pyproj
import rasterio
from rasterio.windows import Window


BAND_COUNT = 64
OUTPUT_COLUMNS = ["longitude", "latitude"] + [f"A{i:02d}" for i in range(BAND_COUNT)]
WGS84 = "EPSG:4326"
UTM10N = "EPSG:32610"
UTM11N = "EPSG:32611"


# Embedded from data/20260123_alphaearth_center_california_points.csv.
POINTS_LON_LAT = [
    (-120.58027780, 37.84888889),
    (-120.65777780, 40.66694444),
    (-122.24277780, 37.87055556),
    (-116.37500000, 33.65055556),
    (-117.24916670, 33.53194444),
    (-122.89019000, 38.09925000),
    (-119.93833000, 37.84261000),
    (-119.93833000, 37.84261000),
    (-119.49117000, 37.80889000),
    (-120.78000000, 39.26000000),
    (-119.26342000, 37.91069000),
    (-119.91283000, 37.81261000),
    (-121.04000000, 39.55000000),
    (-120.78000000, 39.26000000),
    (-121.04000000, 39.55000000),
    (-115.79572000, 33.75426200),
    (-122.89019000, 38.09925000),
    (-121.04000000, 39.55000000),
    (-119.44064000, 37.85158000),
    (-121.04000000, 39.55000000),
    (-122.89653000, 38.12031000),
    (-122.89653000, 38.12031000),
    (-120.78000000, 39.26000000),
    (-120.78000000, 39.26000000),
    (-119.93833000, 37.84261000),
    (-120.78000000, 39.26000000),
    (-119.55000000, 37.75000000),
    (-121.04000000, 39.55000000),
    (-122.87464000, 38.08867000),
    (-121.04000000, 39.55000000),
    (-120.78000000, 39.26000000),
    (-120.58000000, 39.06000000),
    (-117.63000000, 33.71000000),
    (-119.55345000, 37.74811000),
    (-119.49117000, 37.80889000),
    (-120.78000000, 39.26000000),
    (-119.89257810, 37.81383896),
    (-120.78000000, 39.26000000),
    (-118.00000000, 36.29000000),
    (-119.89000000, 37.81000000),
    (-120.78000000, 39.26000000),
    (-121.04000000, 39.55000000),
    (-117.32000000, 34.97000000),
    (-121.04000000, 39.55000000),
    (-121.04000000, 39.55000000),
    (-120.78000000, 39.26000000),
    (-120.78000000, 39.26000000),
    (-121.04000000, 39.55000000),
    (-121.04000000, 39.55000000),
    (-120.58000000, 37.85000000),
    (-121.04000000, 39.55000000),
    (-121.04000000, 39.55000000),
    (-120.66000000, 40.67000000),
    (-115.79572000, 33.75426200),
    (-120.78000000, 39.26000000),
    (-120.40000000, 39.20000000),
    (-120.78000000, 39.26000000),
    (-120.90000000, 41.08000000),
    (-115.81180300, 33.75066600),
    (-121.04000000, 39.55000000),
    (-121.04000000, 39.55000000),
    (-121.04000000, 39.55000000),
    (-116.43000000, 34.15000000),
    (-120.78000000, 39.26000000),
    (-120.78000000, 39.26000000),
    (-119.91283000, 37.81261000),
    (-115.79572000, 33.75426200),
    (-119.44064000, 37.85158000),
    (-120.78000000, 39.26000000),
    (-120.78000000, 39.26000000),
    (-117.69000000, 33.75000000),
    (-122.24000000, 37.87000000),
    (-121.04000000, 39.55000000),
    (-120.58050540, 37.84895325),
    (-120.93000000, 39.01000000),
    (-121.04000000, 39.55000000),
    (-120.78000000, 39.26000000),
    (-120.78000000, 39.26000000),
    (-120.78000000, 39.26000000),
    (-121.04000000, 39.55000000),
    (-121.04000000, 39.55000000),
    (-121.04000000, 39.55000000),
    (-121.04000000, 39.55000000),
    (-115.81180300, 33.75066600),
    (-120.07000000, 39.79000000),
]


@dataclass(frozen=True)
class PointRecord:
    point_id: int
    longitude: float
    latitude: float
    zone10_x: float
    zone10_y: float
    zone11_x: float
    zone11_y: float


@dataclass(frozen=True)
class TileMetadata:
    path: Path
    zone_name: str
    left: float
    bottom: float
    right: float
    top: float


@dataclass(frozen=True)
class TileHit:
    tile_path: Path
    zone_name: str
    x: float
    y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the nearest pixel values for bands 1-64 from AlphaEarth "
            "California GeoTIFF tiles at 85 embedded lon/lat points."
        )
    )
    parser.add_argument(
        "--zone10-dir",
        required=True,
        type=Path,
        help="Directory containing the UTM zone 10N GeoTIFF tiles.",
    )
    parser.add_argument(
        "--zone11-dir",
        required=True,
        type=Path,
        help="Directory containing the UTM zone 11N GeoTIFF tiles.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--tile-glob",
        default="*.tif*",
        help="Glob used to find tiles inside each zone directory (default: %(default)s).",
    )
    return parser.parse_args()


def build_point_records() -> list[PointRecord]:
    to_10 = pyproj.Transformer.from_crs(WGS84, UTM10N, always_xy=True)
    to_11 = pyproj.Transformer.from_crs(WGS84, UTM11N, always_xy=True)

    records: list[PointRecord] = []
    for point_id, (lon, lat) in enumerate(POINTS_LON_LAT):
        x10, y10 = to_10.transform(lon, lat)
        x11, y11 = to_11.transform(lon, lat)
        records.append(
            PointRecord(
                point_id=point_id,
                longitude=lon,
                latitude=lat,
                zone10_x=x10,
                zone10_y=y10,
                zone11_x=x11,
                zone11_y=y11,
            )
        )
    return records


def find_tiles(directory: Path, tile_glob: str) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(path for path in directory.glob(tile_glob) if path.is_file())


def load_tile_metadata(tile_paths: list[Path], zone_name: str) -> list[TileMetadata]:
    metadata: list[TileMetadata] = []
    for path in tile_paths:
        with rasterio.open(path) as dataset:
            bounds = dataset.bounds
            if dataset.count < BAND_COUNT:
                raise ValueError(
                    f"{path} has {dataset.count} bands; expected at least {BAND_COUNT}."
                )
            metadata.append(
                TileMetadata(
                    path=path,
                    zone_name=zone_name,
                    left=bounds.left,
                    bottom=bounds.bottom,
                    right=bounds.right,
                    top=bounds.top,
                )
            )
    return metadata


def point_in_bounds(x: float, y: float, meta: TileMetadata) -> bool:
    x_min = min(meta.left, meta.right)
    x_max = max(meta.left, meta.right)
    y_min = min(meta.bottom, meta.top)
    y_max = max(meta.bottom, meta.top)
    return x_min <= x <= x_max and y_min <= y <= y_max


def choose_nearest_pixel(dataset: rasterio.io.DatasetReader, x: float, y: float) -> tuple[int, int]:
    inv = ~dataset.transform
    col_float, row_float = inv * (x, y)

    # Pixel centers are at integer + 0.5, so nearest center is round(value - 0.5).
    col = int(round(col_float - 0.5))
    row = int(round(row_float - 0.5))

    col = max(0, min(dataset.width - 1, col))
    row = max(0, min(dataset.height - 1, row))
    return row, col

def locate_points(
    points: list[PointRecord], zone10_tiles: list[TileMetadata], zone11_tiles: list[TileMetadata]
) -> dict[int, TileHit]:
    hits: dict[int, TileHit] = {}

    for point in points:
        for meta in zone10_tiles:
            if point_in_bounds(point.zone10_x, point.zone10_y, meta):
                hits[point.point_id] = TileHit(
                    tile_path=meta.path,
                    zone_name=meta.zone_name,
                    x=point.zone10_x,
                    y=point.zone10_y,
                )
                break

        if point.point_id in hits:
            continue

        for meta in zone11_tiles:
            if point_in_bounds(point.zone11_x, point.zone11_y, meta):
                hits[point.point_id] = TileHit(
                    tile_path=meta.path,
                    zone_name=meta.zone_name,
                    x=point.zone11_x,
                    y=point.zone11_y,
                )
                break

    missing = [point.point_id for point in points if point.point_id not in hits]
    if missing:
        raise RuntimeError(
            "Some points were not covered by any tile. Missing point ids: "
            + ", ".join(str(x) for x in missing)
        )

    return hits


def extract_values(
    points: list[PointRecord], point_hits: dict[int, TileHit]
) -> list[list[float]]:
    grouped: dict[Path, list[tuple[PointRecord, TileHit]]] = {}
    for point in points:
        hit = point_hits[point.point_id]
        grouped.setdefault(hit.tile_path, []).append((point, hit))

    rows: list[list[float] | None] = [None] * len(points)

    for tile_path, assigned_points in grouped.items():
        with rasterio.open(tile_path) as dataset:
            for point, hit in assigned_points:
                row, col = choose_nearest_pixel(dataset, hit.x, hit.y)
                window = Window(col_off=col, row_off=row, width=1, height=1)
                pixel = dataset.read(
                    indexes=list(range(1, BAND_COUNT + 1)),
                    window=window,
                    out_dtype="float32",
                    masked=False,
                )
                if pixel.shape != (BAND_COUNT, 1, 1):
                    raise RuntimeError(
                        f"Unexpected read shape {pixel.shape} for {tile_path} at point {point.point_id}."
                    )

                pixel_values = [float(x) for x in pixel[:, 0, 0]]
                rows[point.point_id] = [point.longitude, point.latitude] + pixel_values

    unresolved = [i for i, row in enumerate(rows) if row is None]
    if unresolved:
        raise RuntimeError(
            "Extraction failed for point ids: " + ", ".join(str(x) for x in unresolved)
        )

    return [row for row in rows if row is not None]


def write_csv(output_path: Path, rows: list[list[float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    points = build_point_records()
    zone10_paths = find_tiles(args.zone10_dir, args.tile_glob)
    zone11_paths = find_tiles(args.zone11_dir, args.tile_glob)

    if not zone10_paths:
        raise RuntimeError(f"No tiles found in {args.zone10_dir}")
    if not zone11_paths:
        raise RuntimeError(f"No tiles found in {args.zone11_dir}")

    zone10_tiles = load_tile_metadata(zone10_paths, "10N")
    zone11_tiles = load_tile_metadata(zone11_paths, "11N")
    point_hits = locate_points(points, zone10_tiles, zone11_tiles)
    rows = extract_values(points, point_hits)
    write_csv(args.output_csv, rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Scanned {len(zone10_tiles)} tiles in 10N and {len(zone11_tiles)} tiles in 11N")


if __name__ == "__main__":
    main()
