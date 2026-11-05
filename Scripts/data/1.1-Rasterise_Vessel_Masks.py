
"""
1.1-Rasterise_Vessel_Masks.py
=============================
Burn vessel polygons into a label raster aligned pixel-for-pixel with the
scene's _norm.tif:
    land  -> 1    (LAND_NODATA, copied from the _norm.tif land/fill footprint)
    water -> 0
    ship  -> 255

Reads:  Processed_Scenes/<Scene>/<product>_ST_B10_norm.tif
        Full_Scenes/<Scene>/Shapefiles/*.gpkg (or .shp)
Writes: Processed_Scenes/<Scene>/<Scene>_mask.tif

    python 1.1-Rasterise_Vessel_Masks.py               # all scenes
    python 1.1-Rasterise_Vessel_Masks.py --scenes Qatar
"""

import argparse
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

import pipeline_config as cfg


def rasterise_vessels(norm_tif_path, vessel_path, out_path):
    with rasterio.open(norm_tif_path) as src:
        thermal = src.read(1)
        out_shape, transform, crs = (src.height, src.width), src.transform, src.crs
        profile = src.profile.copy()
        thermal_nodata = src.nodata

    # Land/fill = wherever the normalised image is NoData (0)
    land = (thermal == thermal_nodata) if thermal_nodata is not None else (thermal <= 0)

    vessels = gpd.read_file(vessel_path)
    vessels = vessels[vessels.geometry.notna() & ~vessels.geometry.is_empty]
    if vessels.crs is None:
        raise ValueError(f"{Path(vessel_path).name} has no CRS - cannot align to scene.")
    if vessels.crs != crs:
        vessels = vessels.to_crs(crs)
    print(f"  Loaded {len(vessels)} vessel polygons from {Path(vessel_path).name}")

    # all_touched=True: any pixel a polygon touches becomes ship, so vessels
    # smaller than a 30 m pixel are not lost.
    ship = rasterize([(g, 1) for g in vessels.geometry], out_shape=out_shape,
                     transform=transform, fill=0, dtype=np.uint8, all_touched=True)
    mask = (ship * cfg.SHIP_VALUE).astype(np.uint8)
    ship_px_before = int((mask == cfg.SHIP_VALUE).sum())

    # Land stamped LAST: a ship label on a masked (blank) image pixel would be
    # a label with no signal, so land wins.
    mask[land] = cfg.LAND_NODATA
    ship_px = int((mask == cfg.SHIP_VALUE).sum())

    profile.update(dtype='uint8', count=1, nodata=cfg.LAND_NODATA)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(mask, 1)

    lost = ship_px_before - ship_px
    print(f"  ship px: {ship_px:,}  water px: {int((mask == 0).sum()):,}  "
          f"land px: {int((mask == cfg.LAND_NODATA).sum()):,}")
    if ship_px_before and lost / ship_px_before > 0.05:
        print(f"  WARNING: {lost:,} ship px ({100*lost/ship_px_before:.0f}%) fell on "
              f"land/fill and were removed - check the land map or the labels.")
    print(f"  -> {Path(out_path).name}")


def run(scenes=None):
    cfg.check_splits()
    scenes = scenes or cfg.all_scenes()
    for scene in scenes:
        print(f"\n[1.1] {scene}  ({cfg.split_of(scene)})")
        p = cfg.processed_paths(scene)
        if not p['norm'].exists():
            raise FileNotFoundError(f"{p['norm']} missing - run 1.0 for {scene} first.")
        rasterise_vessels(p['norm'], cfg.find_scene_inputs(scene)['vessels'], p['mask'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='*', help='default: every scene in SPLITS')
    run(ap.parse_args().scenes)