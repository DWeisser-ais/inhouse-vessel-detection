

"""
1.2-Tiling_scenes.py
====================
Cut each scene's _norm.tif + _mask.tif into aligned 512x512 tiles.

Reads:  Processed_Scenes/<Scene>/
Writes: Tiled_Inputs/<Split>/<Scene>/Tiles/<tile>.tif     (thermal input)
        Tiled_Inputs/<Split>/<Scene>/Labels/<tile>.tif    (0 water, 1 land, 255 ship)
        Tiled_Inputs/<Split>/<Scene>/manifest.csv         (one row per tile)

The split comes from SPLITS in pipeline_config.py, not from folder names.
Re-tiling a scene first deletes its old tiles in EVERY split, so moving a
scene between splits can never leave a stale copy behind (that would leak
data between train and test).

    python 1.2-Tiling_scenes.py               # all scenes
    python 1.2-Tiling_scenes.py --scenes Hormuz
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

import pipeline_config as cfg

EIGHT_CONN = np.ones((3, 3), dtype=int)


def remove_scene_tiles(scene):
    for split in (cfg.TRAIN, cfg.TRAIN_CRRP, cfg.VALIDATION, cfg.TEST):
        d = cfg.TILED_DIR / split / scene
        if d.exists():
            shutil.rmtree(d)


def tile_scene(norm_tif_path, mask_tif_path, scene, split,
               tile_size=cfg.TILE_SIZE, overlap=cfg.TILE_OVERLAP,
               min_water_frac=None):
    if min_water_frac is None:
        min_water_frac = cfg.MIN_WATER_FRAC[split]

    remove_scene_tiles(scene)
    img_out, lbl_out = cfg.tile_dirs(split, scene)
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    product_id = Path(norm_tif_path).stem.replace('_norm', '')
    stride = int(tile_size * (1 - overlap)) if overlap else tile_size

    rows, skipped = [], 0
    with rasterio.open(norm_tif_path) as isrc, rasterio.open(mask_tif_path) as msrc:
        if (isrc.width, isrc.height) != (msrc.width, msrc.height) \
                or isrc.transform != msrc.transform:
            raise ValueError(f"{scene}: image and mask grids differ - re-run 1.1.")

        H, W = isrc.height, isrc.width
        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                win = Window(x, y, tile_size, tile_size)
                img_tile = isrc.read(1, window=win)
                lbl_tile = msrc.read(1, window=win)

                water_frac = float((img_tile > 0).mean())
                ship = lbl_tile == cfg.SHIP_VALUE
                has_ship = bool(ship.any())

                # KEEP RULE: every tile with a ship, plus tiles with enough water
                if not (has_ship or water_frac >= min_water_frac):
                    skipped += 1
                    continue

                name = f"{product_id}_y{y:05d}_x{x:05d}.tif"
                iprof = {**isrc.profile, 'height': tile_size, 'width': tile_size,
                         'transform': isrc.window_transform(win)}
                mprof = {**msrc.profile, 'height': tile_size, 'width': tile_size,
                         'transform': msrc.window_transform(win),
                         'nodata': cfg.LAND_NODATA}
                with rasterio.open(img_out / name, 'w', **iprof) as d:
                    d.write(img_tile, 1)
                with rasterio.open(lbl_out / name, 'w', **mprof) as d:
                    d.write(lbl_tile, 1)

                n_ships = ndimage.label(ship, structure=EIGHT_CONN)[1] if has_ship else 0
                rows.append({'split': split, 'scene': scene, 'tile': name,
                             'has_ship': int(has_ship), 'n_ships': int(n_ships),
                             'ship_px': int(ship.sum()),
                             'water_frac': round(water_frac, 4)})

    cfg.write_manifest(rows, cfg.TILED_DIR / split / scene / 'manifest.csv')
    n_ship_tiles = sum(r['has_ship'] for r in rows)
    print(f"  {scene} [{split}] min_water={min_water_frac:.2f}: kept {len(rows)} "
          f"({n_ship_tiles} with ships), skipped {skipped}")
    return rows


def run(scenes=None):
    cfg.check_splits()
    scenes = scenes or cfg.all_scenes()
    for scene in scenes:
        print(f"\n[1.2] {scene}")
        p = cfg.processed_paths(scene)
        for k in ('norm', 'mask'):
            if not p[k].exists():
                raise FileNotFoundError(f"{p[k]} missing - run 1.0 and 1.1 first.")
        tile_scene(p['norm'], p['mask'], scene, cfg.split_of(scene))
    cfg.summarise_dataset()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='*', help='default: every scene in SPLITS')
    run(ap.parse_args().scenes)