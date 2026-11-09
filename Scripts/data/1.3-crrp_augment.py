"""
1.3-crrp_augment.py
===================
Copy-Rotate-Resize-Paste augmentation (MTU-Net paper, Sec. IV-E). TRAIN ONLY.

Reads:  Tiled_Inputs/Train/<Scene>/Tiles + Labels       (real tiles, untouched)
Writes: Tiled_Inputs/Train_CRRP/<Scene>/Tiles + Labels  (synthetic tiles)
        Tiled_Inputs/Train_CRRP/<Scene>/manifest.csv

Synthetic tiles live in their own folder rather than next to the real ones,
so the real dataset stays clean, and USE_CRRP in pipeline_config.py can switch
CRRP on/off (off by default) without deleting files. Train_CRRP is wiped and
rebuilt on every run, so changing CRRP_SETTINGS never leaves old tiles behind.

Ship patches are harvested from ALL training scenes pooled; validation and
test scenes are never read here, so no ship can leak out of the train split.

    python 1.3-crrp_augment.py
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import rasterio
from scipy import ndimage

import pipeline_config as cfg

EIGHT_CONN = np.ones((3, 3), dtype=int)


def harvest_ship_patches(train_dir_pairs, context_px=6, min_area=1, max_area=2000):
    """Every labelled ship (8-connected blob) as an (image, mask) patch with a
    context_px margin of surrounding water. Patches touching land are skipped."""
    patches = []
    for images_dir, labels_dir in train_dir_pairs:
        for lbl_path in sorted(Path(labels_dir).glob('*.tif')):
            img_path = Path(images_dir) / lbl_path.name
            if not img_path.exists():
                continue
            with rasterio.open(img_path) as s:
                img = s.read(1)
            with rasterio.open(lbl_path) as s:
                msk = s.read(1)

            ship_bin = (msk == cfg.SHIP_VALUE).astype(np.uint8)
            if ship_bin.sum() == 0:
                continue
            n, labels, stats, _ = cv2.connectedComponentsWithStats(ship_bin, connectivity=8)
            for i in range(1, n):
                if not (min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area):
                    continue
                x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                x0, y0 = x - context_px, y - context_px
                x1, y1 = x + w + context_px, y + h + context_px
                # Skip ships too close to the tile edge to get a full context
                # margin: a clipped margin lets rotation pull the black border
                # into the hull, creating ship labels with no thermal signal.
                if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
                    continue
                if (msk[y0:y1, x0:x1] == cfg.LAND_NODATA).any():
                    continue
                comp = (labels[y0:y1, x0:x1] == i).astype(np.uint8) * cfg.SHIP_VALUE
                patches.append((img[y0:y1, x0:x1].copy(), comp))
    print(f"  Harvested {len(patches)} ship patches from {len(train_dir_pairs)} train scene(s).")
    return patches


def transform_patch(img_patch, mask_patch, scale_range=(0.6, 1.6), rng=None):
    """Random rotation (0-360) and resize (0.6-1.6x). Bilinear for the image,
    nearest for the mask, then the mask is re-hardened to 0/255."""
    rng = rng or random
    h, w = img_patch.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rng.uniform(0, 360), 1.0)
    img_r = cv2.warpAffine(img_patch, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask_r = cv2.warpAffine(mask_patch, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    s = rng.uniform(*scale_range)
    nw, nh = max(3, int(round(w * s))), max(3, int(round(h * s)))
    img_s = cv2.resize(img_r, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask_s = cv2.resize(mask_r, (nw, nh), interpolation=cv2.INTER_NEAREST)
    mask_s = np.where(mask_s >= 128, cfg.SHIP_VALUE, 0).astype(np.uint8)
    mask_s[img_s == 0] = 0       # never label a pixel with no thermal signal
    return img_s, mask_s


def try_paste(img, msk, img_patch, mask_patch, rng=None, max_tries=30):
    """Paste ship pixels only where the whole patch footprint is plain water
    (label 0): never on land, never on another ship."""
    rng = rng or random
    H, W = img.shape
    ph, pw = img_patch.shape
    ship_px = mask_patch == cfg.SHIP_VALUE
    if ph >= H or pw >= W or not ship_px.any():
        return False
    for _ in range(max_tries):
        y, x = rng.randint(0, H - ph), rng.randint(0, W - pw)
        if np.all(msk[y:y+ph, x:x+pw] == 0):
            img[y:y+ph, x:x+pw][ship_px] = img_patch[ship_px]
            msk[y:y+ph, x:x+pw][ship_px] = cfg.SHIP_VALUE
            return True
    return False


def crrp_augment_scene(scene, patches, n_aug_per_tile=3, ships_per_tile=(3, 12), seed=42):
    rng = random.Random(seed)
    src_img, src_lbl = cfg.tile_dirs(cfg.TRAIN, scene)
    out_img, out_lbl = cfg.tile_dirs(cfg.TRAIN_CRRP, scene)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    originals = sorted(src_img.glob('*.tif'))
    rows = []
    for img_path in originals:
        lbl_path = src_lbl / img_path.name
        if not lbl_path.exists():
            continue
        with rasterio.open(img_path) as s:
            base_img, iprof = s.read(1), s.profile.copy()
        with rasterio.open(lbl_path) as s:
            base_msk, mprof = s.read(1), s.profile.copy()
        if (base_msk == 0).sum() < 500:          # no room to paste
            continue

        for k in range(n_aug_per_tile):
            img, msk = base_img.copy(), base_msk.copy()
            pasted = 0
            for _ in range(rng.randint(*ships_per_tile)):
                p_img, p_msk = rng.choice(patches)
                t_img, t_msk = transform_patch(p_img, p_msk, rng=rng)
                pasted += try_paste(img, msk, t_img, t_msk, rng=rng)
            if pasted == 0:
                continue

            name = f"{img_path.stem}_aug{k}.tif"
            with rasterio.open(out_img / name, 'w', **iprof) as d:
                d.write(img, 1)
            with rasterio.open(out_lbl / name, 'w', **mprof) as d:
                d.write(msk, 1)
            ship = msk == cfg.SHIP_VALUE
            rows.append({'split': cfg.TRAIN_CRRP, 'scene': scene, 'tile': name,
                         'has_ship': 1,
                         'n_ships': int(ndimage.label(ship, structure=EIGHT_CONN)[1]),
                         'ship_px': int(ship.sum()),
                         'water_frac': round(float((img > 0).mean()), 4)})

    cfg.write_manifest(rows, cfg.TILED_DIR / cfg.TRAIN_CRRP / scene / 'manifest.csv')
    print(f"  {scene}: {len(rows)} CRRP tiles from {len(originals)} real tiles")
    return len(rows)


def run(scenes=None):
    """CRRP always pools ALL train scenes (scenes arg ignored on purpose)."""
    cfg.check_splits()
    crrp_root = cfg.TILED_DIR / cfg.TRAIN_CRRP
    if crrp_root.exists():
        shutil.rmtree(crrp_root)

    print("\n[1.3] CRRP augmentation (train scenes only)")
    train_scenes = [s for s in cfg.SPLITS[cfg.TRAIN] if cfg.tile_dirs(cfg.TRAIN, s)[0].exists()]
    missing = set(cfg.SPLITS[cfg.TRAIN]) - set(train_scenes)
    if missing:
        print(f"  NOTE: no tiles yet for {sorted(missing)} - run 1.2 for them.")

    settings = cfg.CRRP_SETTINGS
    patches = harvest_ship_patches([cfg.tile_dirs(cfg.TRAIN, s) for s in train_scenes],
                                   context_px=settings['context_px'])
    if not patches:
        raise RuntimeError("No ship patches harvested - check the Train labels.")

    total = sum(crrp_augment_scene(s, patches,
                                   n_aug_per_tile=settings['n_aug_per_tile'],
                                   ships_per_tile=settings['ships_per_tile'],
                                   seed=settings['seed'])
                for s in train_scenes)
    print(f"  CRRP total: {total} synthetic tiles -> {crrp_root}")
    cfg.summarise_dataset()


if __name__ == '__main__':
    run()