"""
ONE place for every path, split and constant the data pipeline and the
training scripts share. Edit this file - not the individual scripts - when
you add a scene, move a scene between splits, or move the dataset.

Run it on its own to audit your raw scenes before processing:
    python pipeline_config.py

FOLDER LAYOUT (everything under DATASETS_ROOT; see Datasets/README.md)
    scene_catalogue.csv                 one row per scene: region, split, date...
    Full_Scenes/<Scene>/                raw inputs (never written to)
        *_ST_B10.TIF, *_QA_PIXEL.TIF, *_MTL.txt, RGB.tif (unused for now)
        Land Maps/*.tif                 OSM Water Layer tiles for this scene
        Shapefiles/*.gpkg (or .shp + .shx/.dbf/.prj)
        AIS/*.gpkg                      optional AIS context, never a label
    Processed_Scenes/<Scene>/           outputs of 1.0 and 1.1
        <product>_norm.tif, <product>_normparams.json,
        <product>_celsius_wgs84_masked.tif, <Scene>_mask.tif
    Tiled_Inputs/                       outputs of 1.2 and 1.3 (model input)
        Train/<Scene>/Tiles + Labels         real tiles
        Train_CRRP/<Scene>/Tiles + Labels    synthetic CRRP tiles (only if USE_CRRP)
        Validation/<Scene>/Tiles + Labels
        Test/<Scene>/Tiles + Labels
        <Split>/<Scene>/manifest.csv         one row per tile
        manifest.csv, dataset_summary.csv    whole-dataset tables
"""

import os
import csv
from pathlib import Path


# ===================================================================== #
#  PATHS
# ===================================================================== #
# Default: the Datasets/ folder at the repository root (two levels above this
# file), so the code works on any machine without editing. To keep the data
# elsewhere (Colab, OVHcloud, a shared drive) set VESSEL_DATASETS_ROOT.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = Path(os.environ.get('VESSEL_DATASETS_ROOT', REPO_ROOT / 'Datasets'))

FULL_SCENES_DIR = DATASETS_ROOT / 'Full_Scenes'
PROCESSED_DIR   = DATASETS_ROOT / 'Processed_Scenes'
TILED_DIR       = DATASETS_ROOT / 'Tiled_Inputs'

LAND_MAPS_SUBDIR  = 'Land Maps'
SHAPEFILES_SUBDIR = 'Shapefiles'


# ===================================================================== #
#  SPLITS - which scene goes where. Names must match the folder names in
#  Full_Scenes EXACTLY.
# ===================================================================== #
TRAIN       = 'Train'
TRAIN_CRRP  = 'Train_CRRP'
VALIDATION  = 'Validation'
TEST        = 'Test'

# Split rule: scenes whose footprints overlap always share a split, so no water
# appears in two splits. These groups must stay together: Qatar + Hormuz, the
# three NE USA coastal scenes, the Aegean scenes (except 180034), 176034-35,
# 174035-36, and 176038-39. The largest scenes train, so that no single scene
# dominates Validation or Test. Vessels: about 76% Train, 10% Validation, 14% Test.
SPLITS = {
    TRAIN:      ['Singapore', 'Shanghai',
                 'USA_June13_013031', 'USA_June13_013032', 'USA_June29',
                 'Med_20200901_182032', 'Med_20200901_182033',
                 'Med_20200901_182034', 'Med_20200903_180032',
                 'Med_20200910_181032'],
    VALIDATION: ['Rotterdam', 'Med_20200903_180034',
                 'Med_20200907_176034', 'Med_20200907_176035',
                 'Med_20200909_174035', 'Med_20200909_174036'],
    TEST:       ['Hormuz', 'Qatar', 'Malacca', 'USA_June09',
                 'Med_20200907_176038', 'Med_20200907_176039',
                 'Med_20200909_190030'],
}

# Which source dataset each scene came from (by folder-name prefix). Used
# only for the scene catalogue and for per-region reporting.
REGION_BY_PREFIX = {'USA_': 'NE USA', 'Med_': 'Mediterranean'}
DEFAULT_REGION = 'Major ports'


def region_of(scene):
    for prefix, region in REGION_BY_PREFIX.items():
        if scene.startswith(prefix):
            return region
    return DEFAULT_REGION

IMAGE_SUBDIR = 'Tiles'
LABEL_SUBDIR = 'Labels'


# ===================================================================== #
#  PROCESSING CONSTANTS
# ===================================================================== #
LAND_NODATA = 1          # label value for land (ignored by loss/metrics)
SHIP_VALUE  = 255        # label value for vessel
WATER_LAYER_KEEP = (1, 2, 3)   # OSM classes kept as water: ocean, lake/river, major river

KEEP_INTERMEDIATES = False     # True keeps the large _celsius_utm/_celsius_wgs84 rasters

# _norm.tif uses 0 as NoData. True scales valid water to 1..255 so 0 means
# ONLY land/fill. False reproduces the old 0..255 scaling, where the coldest
# ~2% of water (and cold cloud tops) also become 0 and get labelled as land.
RESERVE_ZERO_FOR_NODATA = True

TILE_SIZE    = 512
TILE_OVERLAP = 0.0
# Train drops tiles under 10% water; Val/Test keep nearly all water tiles so
# the false-alarm rate is measured on realistic empty sea. Ship tiles are
# always kept regardless of this threshold.
MIN_WATER_FRAC = {TRAIN: 0.10, VALIDATION: 0.01, TEST: 0.01}

# CRRP copy-paste augmentation. Off: step 1.3 is skipped by run_data_pipeline.py
# and train.py uses only real tiles. Set True to build and train with synthetic tiles.
USE_CRRP = False
CRRP_SETTINGS = dict(context_px=6, n_aug_per_tile=3,
                     ships_per_tile=(3, 12), seed=42)


# ===================================================================== #
#  HELPERS
# ===================================================================== #
def all_scenes():
    """Every scene listed in SPLITS, in split order."""
    return [s for split in (TRAIN, VALIDATION, TEST) for s in SPLITS[split]]


def split_of(scene):
    for split in (TRAIN, VALIDATION, TEST):
        if scene in SPLITS[split]:
            return split
    raise ValueError(f"'{scene}' is not listed in SPLITS (pipeline_config.py). "
                     f"Add it to Train, Validation or Test.")


def tile_dirs(split, scene):
    """(images_dir, labels_dir) for one scene in one split."""
    base = TILED_DIR / split / scene
    return base / IMAGE_SUBDIR, base / LABEL_SUBDIR


def processed_dir(scene):
    return PROCESSED_DIR / scene


def _one(matches, what, where):
    if len(matches) == 0:
        raise FileNotFoundError(f"No {what} found in {where}")
    if len(matches) > 1:
        names = ', '.join(p.name for p in matches)
        raise FileExistsError(f"More than one {what} in {where}: {names}")
    return matches[0]


def find_scene_inputs(scene):
    """
    Locate every raw input for one scene by pattern, so no filename is ever
    typed by hand. Raises a clear error naming whatever is missing.
    """
    sdir = FULL_SCENES_DIR / scene
    if not sdir.is_dir():
        raise FileNotFoundError(f"Scene folder not found: {sdir}")
    files = [p for p in sdir.iterdir() if p.is_file()]

    b10 = _one([p for p in files if p.name.upper().endswith('_ST_B10.TIF')],
               '*_ST_B10.TIF', sdir)
    product = b10.name[:-len('_ST_B10.TIF')]

    qa = _one([p for p in files if p.name.upper() == (product + '_QA_PIXEL.TIF').upper()],
              f'{product}_QA_PIXEL.TIF (must share the B10 file prefix)', sdir)
    mtl = _one([p for p in files if p.name.upper() == (product + '_MTL.TXT').upper()],
               f'{product}_MTL.txt (must share the B10 file prefix)', sdir)

    lm_dir = sdir / LAND_MAPS_SUBDIR
    land_maps = sorted(p for p in lm_dir.glob('*') if p.suffix.lower() in ('.tif', '.tiff')) \
        if lm_dir.is_dir() else []
    if not land_maps:
        raise FileNotFoundError(f"No .tif water-layer tiles in {lm_dir}")

    shp_dir = sdir / SHAPEFILES_SUBDIR
    if not shp_dir.is_dir():
        raise FileNotFoundError(f"Missing folder {shp_dir}")
    gpkgs = sorted(shp_dir.glob('*.gpkg'))
    if gpkgs:
        vessels = _one(gpkgs, 'vessel .gpkg', shp_dir)        # gpkg preferred
    else:
        shp = _one(sorted(shp_dir.glob('*.shp')), 'vessel .gpkg or .shp', shp_dir)
        missing = [ext for ext in ('.shx', '.dbf', '.prj')
                   if not shp.with_suffix(ext).exists()]
        if missing:
            raise FileNotFoundError(f"{shp.name} is missing sidecar file(s) "
                                    f"{missing} - a shapefile needs all of them.")
        vessels = shp

    return {'scene_dir': sdir, 'product': product, 'b10': b10, 'qa': qa,
            'mtl': mtl, 'land_maps': land_maps, 'vessels': vessels}


def processed_paths(scene):
    """Where 1.0 and 1.1 write their outputs for this scene."""
    product = find_scene_inputs(scene)['product']
    pdir = processed_dir(scene)
    stem = product + '_ST_B10'
    return {'dir': pdir,
            'norm': pdir / f'{stem}_norm.tif',
            'normparams': pdir / f'{stem}_normparams.json',
            'mask': pdir / f'{scene}_mask.tif'}


def check_splits():
    """Fail early on split mistakes that would silently corrupt the dataset."""
    seen = {}
    for split in (TRAIN, VALIDATION, TEST):
        for s in SPLITS[split]:
            if s in seen:
                raise ValueError(f"'{s}' is in both {seen[s]} and {split} - "
                                 f"that would leak data between splits.")
            seen[s] = split
    if FULL_SCENES_DIR.is_dir():
        on_disk = {p.name for p in FULL_SCENES_DIR.iterdir() if p.is_dir()}
        unlisted = sorted(on_disk - set(seen))
        if unlisted:
            print(f"  NOTE: scene folders not in SPLITS (ignored): {unlisted}")


# ===================================================================== #
#  DATASET SUMMARY (manifests written by 1.2 and 1.3)
# ===================================================================== #
MANIFEST_FIELDS = ['split', 'scene', 'tile', 'has_ship', 'n_ships',
                   'ship_px', 'water_frac']


def write_manifest(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def summarise_dataset():
    """Combine every per-scene manifest into Tiled_Inputs/manifest.csv and
    print + save a per-split, per-scene summary table."""
    splits = (TRAIN, TRAIN_CRRP, VALIDATION, TEST) if USE_CRRP else (TRAIN, VALIDATION, TEST)
    rows = []
    for split in splits:
        for m in sorted((TILED_DIR / split).glob('*/manifest.csv')):
            with open(m, newline='') as f:
                rows.extend(csv.DictReader(f))
    if not rows:
        print("  No manifests found - nothing to summarise.")
        return
    write_manifest(rows, TILED_DIR / 'manifest.csv')

    groups = {}
    for r in rows:
        g = groups.setdefault((r['split'], r['scene']),
                              {'tiles': 0, 'ship_tiles': 0, 'ships': 0, 'ship_px': 0})
        g['tiles'] += 1
        g['ship_tiles'] += int(r['has_ship'])
        g['ships'] += int(r['n_ships'])
        g['ship_px'] += int(r['ship_px'])

    summary = [{'split': k[0], 'scene': k[1], **v} for k, v in groups.items()]
    with open(TILED_DIR / 'dataset_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'scene', 'tiles', 'ship_tiles',
                                          'ships', 'ship_px'])
        w.writeheader()
        w.writerows(summary)

    print("\n=================== DATASET SUMMARY ===================")
    print(f"  {'split':<11}{'scene':<22}{'tiles':>7}{'ship tiles':>12}{'ships':>8}")
    for split in splits:
        sub = [s for s in summary if s['split'] == split]
        for s in sub:
            print(f"  {split:<11}{s['scene']:<22}{s['tiles']:>7}"
                  f"{s['ship_tiles']:>12}{s['ships']:>8}")
        if sub:
            print(f"  {'':<11}{'TOTAL ' + split:<22}{sum(s['tiles'] for s in sub):>7}"
                  f"{sum(s['ship_tiles'] for s in sub):>12}"
                  f"{sum(s['ships'] for s in sub):>8}")
    print("  (ships = connected ship blobs per tile; a ship cut by a tile")
    print("   border counts once in each tile)")
    print(f"  -> {TILED_DIR / 'manifest.csv'}")
    print(f"  -> {TILED_DIR / 'dataset_summary.csv'}")
    print("=======================================================")


CATALOGUE_FIELDS = ['scene', 'region', 'split', 'sensor', 'path_row',
                    'acquisition_date', 'product_id', 'n_vessel_polygons',
                    'land_tiles']


def write_scene_catalogue():
    """Datasets/scene_catalogue.csv: one row per raw scene, so anyone can see
    what is in the dataset and where it is used without opening any rasters."""
    import geopandas as gpd
    rows = []
    for scene in all_scenes():
        f = find_scene_inputs(scene)
        parts = f['product'].split('_')          # LC08_L2SP_PPPRRR_YYYYMMDD_...
        d = parts[3]
        rows.append({'scene': scene, 'region': region_of(scene),
                     'split': split_of(scene), 'sensor': parts[0],
                     'path_row': parts[2],
                     'acquisition_date': f'{d[:4]}-{d[4:6]}-{d[6:8]}',
                     'product_id': f['product'],
                     'n_vessel_polygons': len(gpd.read_file(f['vessels'])),
                     'land_tiles': ' '.join(p.name for p in f['land_maps'])})
    path = DATASETS_ROOT / 'scene_catalogue.csv'
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=CATALOGUE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  {'split':<11}{'region':<15}{'scenes':>7}{'vessels':>9}")
    for split in (TRAIN, VALIDATION, TEST):
        for region in sorted({r['region'] for r in rows}):
            sub = [r for r in rows if r['split'] == split and r['region'] == region]
            if sub:
                print(f"  {split:<11}{region:<15}{len(sub):>7}"
                      f"{sum(r['n_vessel_polygons'] for r in sub):>9}")
    print(f"  -> {path}")


# ===================================================================== #
#  Self-check: python pipeline_config.py
# ===================================================================== #
if __name__ == '__main__':
    print(f"DATASETS_ROOT: {DATASETS_ROOT}\n")
    check_splits()
    ok = 0
    for scene in all_scenes():
        try:
            f = find_scene_inputs(scene)
            print(f"  OK   {split_of(scene):<11}{scene:<22}"
                  f"{len(f['land_maps'])} land tile(s), vessels: {f['vessels'].name}")
            ok += 1
        except Exception as e:
            print(f"  FAIL {split_of(scene):<11}{scene:<22}{e}")
    print(f"\n{ok}/{len(all_scenes())} scenes ready.")
    if ok == len(all_scenes()):
        write_scene_catalogue()