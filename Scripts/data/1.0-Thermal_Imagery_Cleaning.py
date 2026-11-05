"""
1.0-Thermal_Imagery_Cleaning.py
===============================
Landsat ST_B10 -> Celsius -> fill + land masked -> WGS84 -> normalised uint8.

Reads:  Full_Scenes/<Scene>/  (B10, QA_PIXEL, MTL, Land Maps/)
Writes: Processed_Scenes/<Scene>/<product>_ST_B10_norm.tif   (input to 1.1/1.2)
                                 <product>_ST_B10_normparams.json
                                 <product>_ST_B10_celsius_wgs84_masked.tif

Run every scene listed in pipeline_config.SPLITS:
    python 1.0-Thermal_Imagery_Cleaning.py
Or only some:
    python 1.0-Thermal_Imagery_Cleaning.py --scenes Malacca Hormuz
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

import pipeline_config as cfg

TARGET_CRS   = CRS.from_epsg(4326)
NODATA_FLOAT = -9999.0


def parse_mtl(mtl_path):
    scale = offset = None
    with open(Path(mtl_path), 'r') as f:
        for line in f:
            line = line.strip()
            if 'TEMPERATURE_MULT_BAND_ST_B10' in line:
                scale = float(line.split('=')[1].strip())
            if 'TEMPERATURE_ADD_BAND_ST_B10' in line:
                offset = float(line.split('=')[1].strip())
    if scale is None or offset is None:
        raise ValueError(f"Could not find ST_B10 scale/offset in {mtl_path}")
    return scale, offset


def read_fill_mask_from_qa(qa_pixel_tif_path):
    """
    QA_PIXEL bit 0 = fill (no data). This is the ONLY quality bit masked.
    Cloud (bits 1-3) and cloud shadow (bit 4) are deliberately NOT masked:
    CFMask flags ships and their surroundings as cloud over uniform water,
    so masking them deletes real targets. Clouds are handled at labelling.
    Returns boolean: True = fill.
    """
    with rasterio.open(qa_pixel_tif_path) as src:
        qa = src.read(1)
    fill = (qa & (1 << 0)) > 0
    print(f"  QA fill mask: {fill.sum():,} px ({100*fill.mean():.1f}%)")
    return fill


def build_land_mask_from_waterlayer(celsius_wgs84_tif_path, water_layer_tif_paths,
                                    keep_values=cfg.WATER_LAYER_KEEP):
    """
    Land mask from OSM Water Layer tiles. All tiles are merged into one mosaic
    (handles scenes straddling 5-degree tile boundaries) and resampled onto the
    scene grid with nearest-neighbour so class labels stay integers.
    Classes: 0 land, 1 ocean, 2 large lake/river, 3 major river, 4 canal,
    5 small stream. Pixels in keep_values are water; everything else is land.
    """
    with rasterio.open(celsius_wgs84_tif_path) as ref:
        dst_transform, dst_crs = ref.transform, ref.crs
        dst_shape = (ref.height, ref.width)

    srcs = [rasterio.open(p) for p in water_layer_tif_paths]
    try:
        mosaic, mosaic_transform = merge(srcs)
        mosaic_crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()

    water_class = np.zeros(dst_shape, dtype=np.uint8)   # unfilled = 0 = land
    reproject(source=mosaic[0], destination=water_class,
              src_transform=mosaic_transform, src_crs=mosaic_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              resampling=Resampling.nearest)

    land_mask = ~np.isin(water_class, keep_values)
    print(f"  Water layer: {len(water_layer_tif_paths)} tile(s), "
          f"land = {100*land_mask.mean():.1f}% of grid")
    return land_mask


def reproject_to_wgs84(src_path, output_path):
    with rasterio.open(src_path) as src:
        if src.crs == TARGET_CRS:
            print("  Already WGS84, skipping reprojection.")
            return src_path
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds)
        profile = src.profile.copy()
        profile.update(crs=TARGET_CRS, transform=transform,
                       width=width, height=height, nodata=NODATA_FLOAT)
        with rasterio.open(output_path, 'w', **profile) as dst:
            reproject(source=rasterio.band(src, 1),
                      destination=rasterio.band(dst, 1),
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=TARGET_CRS,
                      src_nodata=NODATA_FLOAT, dst_nodata=NODATA_FLOAT,
                      resampling=Resampling.nearest)
    return output_path


def preprocess_b10(b10_tif_path, mtl_path, qa_pixel_path, land_map_paths, output_dir):
    b10_tif_path = Path(b10_tif_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = b10_tif_path.stem

    scale, offset = parse_mtl(mtl_path)
    print(f"  Scale={scale}, Offset={offset}")

    with rasterio.open(b10_tif_path) as src:
        raw = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    # DN -> Celsius (Collection 2 Level-2 surface temperature rescaling)
    celsius = raw * scale + offset - 273.15

    # Fill = QA bit 0, plus a secondary catch for DN==0 (-> -124.1 C), which
    # occasionally leaks through when QA bit 0 disagrees with ST_B10.
    invalid_mask = read_fill_mask_from_qa(qa_pixel_path) | (raw == 0)
    celsius[invalid_mask] = NODATA_FLOAT

    valid = celsius[~invalid_mask]
    if valid.size == 0:
        raise RuntimeError(f"{stem}: no valid pixels - scene fully masked.")
    print(f"  Valid temp range: {valid.min():.1f} to {valid.max():.1f} C")

    celsius_utm_path = output_dir / f'{stem}_celsius_utm.tif'
    profile.update(dtype='float32', nodata=NODATA_FLOAT)
    with rasterio.open(celsius_utm_path, 'w', **profile) as dst:
        dst.write(celsius, 1)

    celsius_wgs84_path = output_dir / f'{stem}_celsius_wgs84.tif'
    reproject_to_wgs84(celsius_utm_path, celsius_wgs84_path)

    with rasterio.open(celsius_wgs84_path) as src:
        celsius = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    # Sentinel detection with a tolerance, immune to float drift in reprojection
    invalid_mask = celsius <= NODATA_FLOAT + 1.0

    land_mask = build_land_mask_from_waterlayer(celsius_wgs84_path, land_map_paths)
    combined_mask = invalid_mask | land_mask
    celsius[combined_mask] = NODATA_FLOAT

    with rasterio.open(output_dir / f'{stem}_celsius_wgs84_masked.tif', 'w',
                       **{**profile, 'dtype': 'float32', 'nodata': NODATA_FLOAT}) as dst:
        dst.write(celsius, 1)

    # Percentile stretch computed from plausible water temperatures only
    water_mask = (~combined_mask) & (celsius > -2.0) & (celsius < 35.0)
    usable_frac = 100 * water_mask.sum() / celsius.size
    if water_mask.sum() < 1000:
        warnings.warn(f"{stem}: only {water_mask.sum()} water pixels "
                      f"({usable_frac:.1f}%). Likely unusable - review.")

    p2  = float(np.percentile(celsius[water_mask], 2))
    p98 = float(np.percentile(celsius[water_mask], 98))
    print(f"  Normalisation range: {p2:.1f} to {p98:.1f} C")

    normalised = np.clip((celsius - p2) / (p98 - p2), 0.0, 1.0)
    if cfg.RESERVE_ZERO_FOR_NODATA:
        # Valid pixels -> 1..255 so 0 means ONLY land/fill. Without this the
        # coldest ~2% of water (and cold cloud tops) clip to 0, become
        # indistinguishable from NoData, and 1.1 labels them as land.
        uint8_arr = (1 + normalised * 254).astype(np.uint8)
    else:
        uint8_arr = (normalised * 255).astype(np.uint8)      # old behaviour
    uint8_arr[combined_mask] = 0

    norm_path = output_dir / f'{stem}_norm.tif'
    with rasterio.open(norm_path, 'w',
                       **{**profile, 'dtype': 'uint8', 'nodata': 0}) as dst:
        dst.write(uint8_arr, 1)

    with open(output_dir / f'{stem}_normparams.json', 'w') as f:
        json.dump({'scene': stem, 'p2': p2, 'p98': p98,
                   'usable_water_pct': usable_frac}, f, indent=2)

    if not cfg.KEEP_INTERMEDIATES:
        celsius_utm_path.unlink(missing_ok=True)
        if celsius_wgs84_path != celsius_utm_path:
            celsius_wgs84_path.unlink(missing_ok=True)

    print(f"  Usable water: {usable_frac:.1f}%  ->  {norm_path.name}")
    return norm_path


def run(scenes=None):
    cfg.check_splits()
    scenes = scenes or cfg.all_scenes()
    for scene in scenes:
        print(f"\n[1.0] {scene}  ({cfg.split_of(scene)})")
        f = cfg.find_scene_inputs(scene)
        preprocess_b10(f['b10'], f['mtl'], f['qa'], f['land_maps'],
                       cfg.processed_dir(scene))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='*', help='default: every scene in SPLITS')
    run(ap.parse_args().scenes)