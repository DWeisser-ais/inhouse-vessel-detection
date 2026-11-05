"""
 
OUTPUTS (all written to <out_dir>/):
    test_results.json                 headline IoU / Pd / Fa at the chosen threshold
    per_region_results.json           the same metrics split by region (ports / NE USA / Med)
    threshold_sweep.csv/.png          Pd & Fa across ALL thresholds + ROC-style curve
    probability_distributions.png     ship-pixel vs water-pixel score histograms
    prediction_stats.json             mean/median/max scores, ship-water separation
    per_tile_results.csv              one row per tile: ships, detected, Pd, Fa, IoU
    qualitative_thr*.png              test tiles: input | GT | probability | overlay
 
Usage:
    python evaluate.py --checkpoint <path to best.pth>
"""
 
import argparse
from pathlib import Path
import sys
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
 
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.mtu_net import MTUNet
from data.dataset import ShipTileDataset
from model.metrics import MetricAccumulator
from model.analysis import (threshold_sweep, probability_distributions, per_tile_table, confusion_matrix)
from training.visualise import visualise_tiles
from data.pipeline_config import SPLITS, TEST, VALIDATION, tile_dirs, region_of




# Scenes come from data/pipeline_config.py (SPLITS) - same source as train.py.

def make_dir_pairs(split, scenes):
    pairs = []
    for scene in scenes:
        img_dir, lbl_dir = tile_dirs(split, scene)
        for d in (img_dir, lbl_dir):
            if not d.exists():
                raise FileNotFoundError(f"Missing folder: {d} - run the data pipeline.")
        pairs.append((img_dir, lbl_dir))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/best.pth')
    ap.add_argument('--threshold',  type=float, default=0.3)
    ap.add_argument('--batch_size', type=int,   default=4)
    ap.add_argument('--out_dir',    default='analysis')
    ap.add_argument('--split', default=TEST, choices=[TEST, VALIDATION],
                    help='Test = final numbers (run once). Validation = for choosing the threshold.')
    ap.add_argument('--n_show',     type=int, default=8,
                    help='how many test tiles to draw in the qualitative fig')
    args = ap.parse_args()
 
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
 
    model = MTUNet(in_channels=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch','?')}")
 
    print(f"Evaluating on {args.split}: {SPLITS[args.split]}")
    test_ds = ShipTileDataset(make_dir_pairs(args.split, SPLITS[args.split]),
                              augment=False)
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
 

    # --- Run the model ONCE, cache everything for all downstream analysis ---
    names, images, probs, ships, valids = [], [], [], [], []
    tile_scenes = []          # scene of each tile, for the per-region breakdown
    with torch.no_grad():
        for bi, (img, ship, valid) in enumerate(loader):
            p = torch.sigmoid(model(img.to(device))).cpu().numpy()
            for b in range(p.shape[0]):
                gi = bi * args.batch_size + b
                names.append(test_ds.samples[gi][0].name)
                tile_scenes.append(test_ds.samples[gi][0].parent.parent.name)
                images.append(img[b, 0].numpy())
                probs.append(p[b, 0])
                ships.append(ship[b, 0].numpy())
                valids.append(valid[b, 0].numpy())
 
    print(f"Ran model on {len(probs)} test tiles.\n")
 


    # --- 1. Headline metrics at the chosen threshold ---
    acc = MetricAccumulator(threshold=args.threshold)
    for p, s, v in zip(probs, ships, valids):
        acc.update(p, s, v)
    r = acc.compute()
 
    print("================ TEST RESULTS ================")
    print(f"  threshold : {args.threshold}")
    print(f"  IoU : {r['IoU']*100:.2f} %")
    print(f"  Pd  : {r['Pd']*100:.2f} %   ({r['matched']}/{r['targets']} ships)")
    print(f"  Fa  : {r['Fa']*1e6:.2f} x 10^-6")
    print("==============================================")
    with open(out_dir / 'test_results.json', 'w') as f:
        json.dump({'threshold': args.threshold, **{k: float(v) if isinstance(v,(int,float)) else v
                                                    for k, v in r.items()}}, f, indent=2)
 



    # --- 1b. Same metrics per region (Major ports / NE USA / Mediterranean) ---
    per_region = {}
    for region in sorted({region_of(sc) for sc in tile_scenes}):
        ra = MetricAccumulator(threshold=args.threshold)
        for p, s, v, sc in zip(probs, ships, valids, tile_scenes):
            if region_of(sc) == region:
                ra.update(p, s, v)
        rr = ra.compute()
        per_region[region] = {k: float(v) if isinstance(v, (int, float)) else v
                              for k, v in rr.items()}
        print(f"  {region:<15} IoU {rr['IoU']*100:6.2f} %   Pd {rr['Pd']*100:6.2f} % "
              f"({rr['matched']}/{rr['targets']})   Fa {rr['Fa']*1e6:7.2f} x 10^-6")
    with open(out_dir / 'per_region_results.json', 'w') as f:
        json.dump({'threshold': args.threshold, **per_region}, f, indent=2)

    # --- 2. Threshold sweep ---
    print("\n--- Threshold sweep ---")
    threshold_sweep(probs, ships, valids, out_dir=out_dir)
 



    # --- 3. Probability distributions ---
    print("\n--- Prediction distributions ---")
    probability_distributions(probs, ships, valids, out_dir=out_dir)
 



    # --- 4. Per-tile table ---
    per_tile_table(names, probs, ships, valids,
                   threshold=args.threshold, out_dir=out_dir)
 



    # --- Confusion matrix  ---
    print("\n--- Confusion matrix ---")
    confusion_matrix(probs, ships, valids, threshold=args.threshold, out_dir=out_dir)
 



    # --- 5. Qualitative figures ---
    print("\n--- Qualitative figures ---")
    visualise_tiles(names, images, probs, ships, valids,
                    threshold=args.threshold, n_show=args.n_show,
                    out_dir=out_dir)
    # Also draw at a LOW threshold, so a weakly-responding model is still
    # visible. If max prob is ~0.04, a 0.5-threshold figure is all black.
    visualise_tiles(names, images, probs, ships, valids,
                    threshold=0.01, n_show=args.n_show, out_dir=out_dir)
 


    print(f"\nAll analysis written to: {out_dir.resolve()}")



 
 
if __name__ == '__main__':
    main()
 