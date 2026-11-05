"""
0.0-Import_Mediterranean_Scenes.py
==================================
One-off (re-runnable) import of the hand-labelled Mediterranean Landsat scenes
from the AI4COPSEC project into this project's standard raw-scene layout, so
the normal pipeline (1.0 -> 1.3) can process them like every other scene.

Source (AI4COPSEC, read only - nothing there is changed):
    Data/Landsat/Mediterranean/2020/09/<DD>/
        <pathrow>_<YYYYMMDD>_Shapes.gpkg          hand-labelled vessel polygons
        LC08_L2SP_<pathrow>_<date>_02_T1_ST_B10.TIF, _SR_B4/B3/B2.TIF, _ais.gpkg
    Data/Dataset_2/TIF_Files/<product>_QA_PIXEL.tif

A scene is imported only if its _Shapes.gpkg sits DIRECTLY in a day folder.
Anything in "Not Checked Yet", "Not Included" or "Still Working" is skipped,
the same rule AI4COPSEC's package_dataset.py used.

Written for each scene (Full_Scenes/Med_<YYYYMMDD>_<pathrow>/):
    <product>_ST_B10.TIF            copied byte-for-byte
    <product>_QA_PIXEL.TIF          copied from AI4COPSEC Dataset_2
    <product>_MTL.txt               copied from --mtl-dir, or downloaded from
                                    Microsoft Planetary Computer if missing
    RGB.tif                         3-band uint16 stack of SR_B4, SR_B3, SR_B2
    Land Maps/*.tif                 the OSM Water Layer tiles the scene touches
    Shapefiles/Vessels_<Scene>.gpkg vessel polygons ('Class': 1 small, 2 big;
                                    this project treats both as "vessel")
    AIS/<product>_ais.gpkg          AIS context only, never used as labels

Not copied: the _K / _K_masked / _K_norm rasters and _Ships.tif masks. They
are derived products; 1.0 and 1.1 regenerate them from the files above. The
polygons reproduce the _Ships.tif masks exactly (checked on import).

    python 0.0-Import_Mediterranean_Scenes.py --dry-run
    python 0.0-Import_Mediterranean_Scenes.py
    python 0.0-Import_Mediterranean_Scenes.py --mtl-dir <folder> --overwrite

Afterwards add the new scene names to SPLITS in pipeline_config.py and run
    python run_data_pipeline.py --scenes <new scenes>
"""

import argparse
import os
import math
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

import pipeline_config as cfg

# Local source folders. Override per machine with the environment variables
# AI4COPSEC_DATA and OSM_WATER_DIR, or with --src / --qa-dir / --osm-dir.
_HOME = Path.home()
AI4COPSEC = Path(os.environ.get('AI4COPSEC_DATA',
                                _HOME / 'Vessel_Detection' / 'AI4COPSEC' / 'Data'))
DEFAULT_SRC = AI4COPSEC / 'Landsat' / 'Mediterranean' / '2020' / '09'
DEFAULT_QA = AI4COPSEC / 'Dataset_2' / 'TIF_Files'
DEFAULT_OSM = Path(os.environ.get('OSM_WATER_DIR', _HOME / 'Documents' / 'OSM_water_tif'))

SCENE_PREFIX = 'Med'
SHAPES_RE = re.compile(r'^(\d{6})_(\d{8})_Shapes\.gpkg$', re.IGNORECASE)


# --------------------------------------------------------------------------
def find_labelled_scenes(src_root):
    """Every <pathrow>_<date>_Shapes.gpkg sitting directly in a day folder."""
    found = []
    for day in sorted(p for p in src_root.iterdir() if p.is_dir()):
        for shp in sorted(day.glob('*_Shapes.gpkg')):
            m = SHAPES_RE.match(shp.name)
            if not m:
                continue
            pathrow, date = m.groups()
            b10 = list(day.glob(f'LC0*_{pathrow}_{date}_*_ST_B10.TIF'))
            if len(b10) != 1:
                raise FileNotFoundError(f'Expected one ST_B10 for {shp.name} in {day}')
            product = b10[0].name[:-len('_ST_B10.TIF')]
            found.append({'scene': f'{SCENE_PREFIX}_{date}_{pathrow}', 'day_dir': day,
                          'product': product, 'shapes': shp, 'b10': b10[0],
                          'ships_tif': day / f'{pathrow}_{date}_Ships.tif'})
    return found


def osm_tiles_for(raster_path, osm_dir):
    """OSM Water Layer 5-degree tiles (named by SW corner) covering a raster."""
    with rasterio.open(raster_path) as s:
        minx, miny, maxx, maxy = transform_bounds(s.crs, 'EPSG:4326', *s.bounds)
    names = []
    for lat in range(int(math.floor(miny / 5) * 5), int(math.floor((maxy - 1e-9) / 5) * 5) + 1, 5):
        for lon in range(int(math.floor(minx / 5) * 5), int(math.floor((maxx - 1e-9) / 5) * 5) + 1, 5):
            names.append(f"{'n' if lat >= 0 else 's'}{abs(lat):02d}"
                         f"{'e' if lon >= 0 else 'w'}{abs(lon):03d}.tif")
    missing = [n for n in names if not (osm_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f'OSM water tiles missing from {osm_dir}: {missing}')
    return [osm_dir / n for n in names]


def get_mtl(product, mtl_dir, dest):
    if mtl_dir and (Path(mtl_dir) / f'{product}_MTL.txt').exists():
        shutil.copy2(Path(mtl_dir) / f'{product}_MTL.txt', dest)
        return 'copied'
    try:
        import planetary_computer as pc
        from pystac_client import Client
    except ImportError:
        raise RuntimeError(f'No MTL for {product} in --mtl-dir and planetary_computer '
                           f'is not installed in this environment to download it.')
    cat = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                      modifier=pc.sign_inplace)
    href = cat.get_collection('landsat-c2-l2').get_item(product).assets['mtl.txt'].href
    dest.write_bytes(urllib.request.urlopen(href, timeout=120).read())
    return 'downloaded'


def write_rgb(day_dir, product, dest):
    """Stack native SR_B4 (red), SR_B3 (green), SR_B2 (blue) into one uint16 file,
    matching the RGB.tif of the other scenes (values untouched)."""
    bands = [day_dir / f'{product}_SR_B{b}.TIF' for b in (4, 3, 2)]
    with rasterio.open(bands[0]) as s:
        profile = s.profile.copy()
    profile.update(count=3, dtype='uint16', compress='deflate', predictor=2,
                   tiled=True, blockxsize=512, blockysize=512, BIGTIFF='IF_SAFER')
    with rasterio.open(dest, 'w', **profile) as dst:
        for i, (p, name) in enumerate(zip(bands, ('Red', 'Green', 'Blue')), start=1):
            with rasterio.open(p) as s:
                if (s.transform, s.shape) != (profile['transform'], (profile['height'], profile['width'])):
                    raise ValueError(f'{p.name} is not on the same grid as SR_B4')
                dst.write(s.read(1), i)
            dst.set_band_description(i, f'{name} (SR_B{(4, 3, 2)[i - 1]})')


def check_labels(shapes, b10, ships_tif):
    """Polygons must reproduce the hand-checked AI4COPSEC _Ships.tif mask."""
    import geopandas as gpd
    from rasterio.features import rasterize
    with rasterio.open(b10) as s:
        shape, tf, crs = s.shape, s.transform, s.crs
    g = gpd.read_file(shapes)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].to_crs(crs)
    poly = rasterize(g.geometry, out_shape=shape, transform=tf, fill=0, dtype='uint8') > 0
    with rasterio.open(ships_tif) as s:
        tif = np.isin(s.read(1), (1, 2))
    if (tif & ~poly).any():
        raise ValueError(f'{ships_tif.name} has ship pixels with no polygon in {shapes.name}')
    return len(g), int(tif.sum()), int(poly.sum())


# --------------------------------------------------------------------------
def import_scene(s, qa_dir, osm_dir, mtl_dir, overwrite, dry_run):
    out = cfg.FULL_SCENES_DIR / s['scene']
    product = s['product']
    if out.exists() and not overwrite:
        print(f"  SKIP {s['scene']}: already exists (use --overwrite)")
        return
    qa = qa_dir / f'{product}_QA_PIXEL.tif'
    if not qa.exists():
        raise FileNotFoundError(qa)
    with rasterio.open(qa) as q, rasterio.open(s['b10']) as b:
        if (q.transform, q.shape, q.crs) != (b.transform, b.shape, b.crs):
            raise ValueError(f'{qa.name} is not on the ST_B10 grid')
    tiles = osm_tiles_for(s['b10'], osm_dir)
    n_poly, tif_px, poly_px = check_labels(s['shapes'], s['b10'], s['ships_tif'])
    print(f"  {s['scene']:<22} {product}  {n_poly} vessel polygons "
          f"(mask px {tif_px}, polygon px {poly_px}), land tiles {[t.name for t in tiles]}")
    if dry_run:
        return

    if out.exists():
        shutil.rmtree(out)
    lm, shp, ais = out / cfg.LAND_MAPS_SUBDIR, out / cfg.SHAPEFILES_SUBDIR, out / 'AIS'
    for d in (lm, shp, ais):
        d.mkdir(parents=True)

    shutil.copy2(s['b10'], out / f'{product}_ST_B10.TIF')
    shutil.copy2(qa, out / f'{product}_QA_PIXEL.TIF')
    how = get_mtl(product, mtl_dir, out / f'{product}_MTL.txt')
    write_rgb(s['day_dir'], product, out / 'RGB.tif')
    for t in tiles:
        shutil.copy2(t, lm / t.name)
    shutil.copy2(s['shapes'], shp / f"Vessels_{s['scene']}.gpkg")
    src_ais = s['day_dir'] / f'{product}_ais.gpkg'
    if src_ais.exists():
        shutil.copy2(src_ais, ais / src_ais.name)
    print(f"       -> {out}  (MTL {how})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=DEFAULT_SRC)
    ap.add_argument('--qa-dir', type=Path, default=DEFAULT_QA)
    ap.add_argument('--osm-dir', type=Path, default=DEFAULT_OSM)
    ap.add_argument('--mtl-dir', type=Path, help='folder of pre-downloaded <product>_MTL.txt')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    scenes = find_labelled_scenes(args.src)
    print(f'{len(scenes)} labelled scene(s) found under {args.src}\n')
    for s in scenes:
        import_scene(s, args.qa_dir, args.osm_dir, args.mtl_dir, args.overwrite, args.dry_run)
    print('\nDone. Scene names for pipeline_config.SPLITS:')
    print('  ' + ', '.join(repr(s['scene']) for s in scenes))
