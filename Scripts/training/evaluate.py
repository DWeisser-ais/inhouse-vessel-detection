"""
evaluate.py
===========
Final evaluation of a trained checkpoint.

    python evaluate.py --checkpoint checkpoints/best.pth

The decision threshold is chosen on the VALIDATION split (highest vessel F1)
and then applied unchanged to the split being reported, so the test set plays
no part in tuning. Run the Test split once, when the model is final.

Written to <checkpoint folder>/evaluation/<split>/ unless --out_dir is given:
    metrics.json               every metric: overall, per region, loss, threshold
    per_scene.csv              the main metrics for each scene
    threshold_sweep.csv        precision, recall and F1 at every threshold
    precision_recall.png       precision-recall curve and scores against threshold
    examples_detections.png    tiles where vessels were found
    examples_errors.png        tiles with false positives and missed vessels

Metric definitions are in model/metrics.py.
"""

import os

# Conda's MKL and pip's PyTorch each bring their own OpenMP runtime, and Windows
# aborts if both load. Single-threaded MKL never loads its one; PyTorch is unaffected.
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.dataset import ShipTileDataset
from data.pipeline_config import SPLITS, TEST, VALIDATION, region_of
from model.analysis import plot_precision_recall
from model.metrics import (DetectionMetrics, SIZE_CLASSES, average_precision, best_f1_threshold,
                           score_tile, threshold_sweep)
from model.mtu_net import MTUNet
from training.inference import make_dir_pairs, make_loss, predict_batches
from training.visualise import draw_examples, select_examples

SWEEP_COLUMNS = ['threshold', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn',
                 'fa_per_1000km2', 'iou']
SCENE_COLUMNS = ['region', 'scene', 'tiles', 'vessels', 'tp', 'fp', 'fn', 'precision', 'recall',
                 'f1', 'fa_per_1000km2', 'iou']


def collect_predictions(model, split, batch_size, device, criterion):
    """Run the model once over a split and keep what the metrics need."""
    dataset = ShipTileDataset(make_dir_pairs(split, SPLITS[split]), augment=False, name=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs, ships, valids, total_loss = [], [], [], 0.0
    for loss, p, s, v in predict_batches(model, loader, device, criterion):
        total_loss += loss * p.shape[0]
        probs.extend(p[:, 0])
        ships.extend(s[:, 0] > 0)
        valids.extend(v[:, 0] > 0)
    paths = [sample[0] for sample in dataset.samples]
    return {'dataset': dataset, 'probs': probs, 'ships': ships, 'valids': valids,
            'names': [p.name for p in paths],
            'scenes': [p.parent.parent.name for p in paths],
            'loss': total_loss / len(paths)}


def write_csv(path, columns, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def group_metrics(scores, groups):
    """Metrics for each distinct value in groups, in sorted order."""
    totals = {}
    for score, group in zip(scores, groups):
        totals.setdefault(group, DetectionMetrics()).add(score)
    return {group: totals[group].compute() for group in sorted(totals)}


def print_results(split, threshold, source, loss, ap, overall, by_region):
    print(f"\n================ {split.upper()} RESULTS ================")
    print(f"  Decision threshold : {threshold:.2f} ({source})")
    print(f"  Loss               : {loss:.4f}")
    print("\n  Vessels (object level)")
    print(f"    Precision        : {overall['precision']:.3f}")
    print(f"    Recall (Pd)      : {overall['recall']:.3f}   "
          f"({overall['tp']} of {overall['vessels']} vessels)")
    print(f"    F1               : {overall['f1']:.3f}")
    print(f"    AP               : {ap:.3f}")
    print(f"    False positives  : {overall['fp']}   "
          f"({overall['fa_per_1000km2']:.2f} per 1000 km2 of water)")
    print("    Recall by size   : " + "   ".join(
        f"{s} {overall[f'recall_{s}']:.3f} (n={overall[f'vessels_{s}']})" for s in SIZE_CLASSES))
    print("\n  Pixels (water only)")
    print(f"    IoU              : {overall['iou']:.3f}")
    print(f"    nIoU             : {overall['niou']:.3f}")
    print(f"    Dice             : {overall['dice']:.3f}")
    print(f"    False-alarm rate : {overall['fa_rate'] * 1e6:.1f} x 10^-6")
    print(f"\n  {'Region':<15}{'Vessels':>9}{'Precision':>11}{'Recall':>9}{'F1':>8}"
          f"{'IoU':>8}{'FA/1000km2':>12}")
    for region, m in by_region.items():
        print(f"  {region:<15}{m['vessels']:>9}{m['precision']:>11.3f}{m['recall']:>9.3f}"
              f"{m['f1']:>8.3f}{m['iou']:>8.3f}{m['fa_per_1000km2']:>12.2f}")
    print("=" * 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/best.pth')
    ap.add_argument('--split', default=TEST, choices=[TEST, VALIDATION])
    ap.add_argument('--threshold', type=float, default=None,
                    help='fixed threshold; default is best F1 on the validation split')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--n_show', type=int, default=8, help='tiles per example figure')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else (
        checkpoint_path.parent / 'evaluation' / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MTUNet(in_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Device: {device}   |   checkpoint epoch: {checkpoint.get('epoch', '?')}")
    criterion = make_loss()

    # --- Choose the threshold on validation, then report the requested split ---
    data = collect_predictions(model, VALIDATION, args.batch_size, device, criterion) \
        if args.threshold is None or args.split == VALIDATION else None
    if args.threshold is None:
        threshold = best_f1_threshold(threshold_sweep(data['probs'], data['ships'], data['valids']))
        source = 'best F1 on Validation'
    else:
        threshold, source = args.threshold, 'set with --threshold'
    if args.split != VALIDATION:
        del data
        data = collect_predictions(model, args.split, args.batch_size, device, criterion)

    # --- Metrics ---
    scores = [score_tile(p, s, v, threshold)
              for p, s, v in zip(data['probs'], data['ships'], data['valids'])]
    regions = [region_of(s) for s in data['scenes']]
    overall = group_metrics(scores, ['all'] * len(scores))['all']
    by_region = group_metrics(scores, regions)
    by_scene = group_metrics(scores, data['scenes'])

    sweep = threshold_sweep(data['probs'], data['ships'], data['valids'])
    ap_score = average_precision(sweep)
    print_results(args.split, threshold, source, data['loss'], ap_score, overall, by_region)

    # --- Files ---
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({'checkpoint': str(checkpoint_path), 'epoch': checkpoint.get('epoch'),
                   'split': args.split, 'threshold': threshold, 'threshold_source': source,
                   'loss': data['loss'], 'average_precision': ap_score,
                   'overall': overall, 'by_region': by_region}, f, indent=2)
    write_csv(out_dir / 'per_scene.csv', SCENE_COLUMNS,
              [{'region': region_of(s), 'scene': s, **m} for s, m in by_scene.items()])
    write_csv(out_dir / 'threshold_sweep.csv', SWEEP_COLUMNS, sweep)
    plot_precision_recall(sweep, threshold, ap_score, out_dir / 'precision_recall.png')

    detections, errors = select_examples(scores, regions, args.n_show)
    draw_examples(detections, data, scores, threshold, f'{args.split}: detected vessels',
                  out_dir / 'examples_detections.png')
    draw_examples(errors, data, scores, threshold, f'{args.split}: false positives and misses',
                  out_dir / 'examples_errors.png')
    print(f"\nAll evaluation output written to: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
