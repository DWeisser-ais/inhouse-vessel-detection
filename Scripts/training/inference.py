"""
inference.py
============
Running the model over a split. train.py and evaluate.py share this code, so
a loss or metric means exactly the same thing in both.
"""

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.pipeline_config import tile_dirs
from model.losses import FocalIoULoss
from model.metrics import DetectionMetrics, score_tile

FOCAL_GAMMA = 2.0


def make_loss():
    return FocalIoULoss(gamma=FOCAL_GAMMA)


def make_dir_pairs(split, scenes, required=True):
    """(Tiles, Labels) folder pairs for each scene in a split."""
    pairs = []
    for scene in scenes:
        img_dir, lbl_dir = tile_dirs(split, scene)
        if not (img_dir.exists() and any(img_dir.glob('*.tif'))):
            if required:
                raise FileNotFoundError(f"No tiles in {img_dir} - run the data pipeline.")
            print(f"  (skipping {split}/{scene}: no tiles)")
            continue
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Missing labels folder: {lbl_dir}")
        pairs.append((img_dir, lbl_dir))
    return pairs


@torch.no_grad()
def predict_batches(model, loader, device, criterion):
    """Yield (batch loss, probabilities, ship masks, water masks) as numpy."""
    model.eval()
    for img, ship, valid in loader:
        logits = model(img.to(device))
        loss = criterion(logits, ship.to(device), valid.to(device)).item()
        yield loss, torch.sigmoid(logits).cpu().numpy(), ship.numpy(), valid.numpy()


def evaluate_split(model, loader, device, criterion, threshold=None):
    """Mean loss over a split, plus all detection metrics if threshold is given."""
    total_loss, n_tiles = 0.0, 0
    metrics = DetectionMetrics() if threshold is not None else None
    for loss, probs, ships, valids in predict_batches(model, loader, device, criterion):
        batch = probs.shape[0]
        total_loss += loss * batch
        n_tiles += batch
        if metrics is not None:
            for b in range(batch):
                metrics.add(score_tile(probs[b, 0], ships[b, 0], valids[b, 0], threshold))
    result = {'loss': total_loss / max(n_tiles, 1)}
    if metrics is not None:
        result.update(metrics.compute())
    return result
