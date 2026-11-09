"""
train.py
========
    python train.py --epochs 300 --batch_size 4
    python train.py --epochs 300 --batch_size 4 --resume     # continue a stopped run

Written to --out_dir (default checkpoints/):
    best.pth, last.pth                 highest validation F1, and the latest epoch
    analysis/training_history.csv      losses and validation metrics per epoch
    analysis/loss_curves.png           train / validation / test loss against epoch
    analysis/validation_metrics.png    precision, recall, F1, IoU, false alarms
    analysis/model_size_report.json    parameters, weights, compute, peak RAM, latency

The test split is only used to log its loss. It never affects which checkpoint
is kept, when training stops, or the decision threshold.
"""

import os

# Conda's MKL and pip's PyTorch each bring their own OpenMP runtime, and Windows
# aborts if both load. Single-threaded MKL never loads its one; PyTorch is unaffected.
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import argparse
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.dataset import ShipTileDataset
from data.pipeline_config import SPLITS, TEST, TRAIN, TRAIN_CRRP, USE_CRRP, VALIDATION
from model.analysis import HistoryLogger
from model.deployment_metrics import model_size_report
from model.mtu_net import MTUNet
from training.inference import evaluate_split, make_dir_pairs, make_loss


# ===================================================================== #
#  CONFIGURATION - edit here
#  Paths and train/val/test scenes come from data/pipeline_config.py
# ===================================================================== #
VAL_THRESHOLD = 0.3      # used to monitor validation during training only
LEARNING_RATE = 0.05

DEFAULT_WORKERS = 0 if platform.system() == 'Windows' else 4
# ===================================================================== #


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(payload, path):
    """Write to a temporary file first, so a crash cannot corrupt the checkpoint."""
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def eval_loader(split, batch_size, workers):
    dataset = ShipTileDataset(make_dir_pairs(split, SPLITS[split]), augment=False, name=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs',      type=int, default=300)
    ap.add_argument('--batch_size',  type=int, default=4)
    ap.add_argument('--out_dir',     default='checkpoints')
    ap.add_argument('--patience',    type=int, default=40)
    ap.add_argument('--num_workers', type=int, default=DEFAULT_WORKERS)
    ap.add_argument('--seed',        type=int, default=42)
    ap.add_argument('--resume', action='store_true',
                    help='continue from last.pth in --out_dir, with the same --epochs')
    args = ap.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}   |   workers: {args.num_workers}   "
          f"|   monitor threshold: {VAL_THRESHOLD}   |   CRRP: {USE_CRRP}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Datasets & loaders ---
    train_pairs = make_dir_pairs(TRAIN, SPLITS[TRAIN])
    if USE_CRRP:
        train_pairs += make_dir_pairs(TRAIN_CRRP, SPLITS[TRAIN], required=False)
    train_ds = ShipTileDataset(train_pairs, augment=True, name=TRAIN)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = eval_loader(VALIDATION, args.batch_size, args.num_workers)
    test_loader = eval_loader(TEST, args.batch_size, args.num_workers)

    # --- Model ---
    model = MTUNet(in_channels=1).to(device)
    model_size_report(model, input_shape=(1, 1, 512, 512), device=device,
                      out_path=out_dir / 'analysis' / 'model_size_report.json')

    # Xavier init
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            if m.weight is not None and m.weight.dim() >= 2:
                torch.nn.init.xavier_uniform_(m.weight)
            if getattr(m, 'bias', None) is not None:
                torch.nn.init.zeros_(m.bias)

    criterion = make_loss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_epoch = 0
    epochs_since_improve = 0
    completed = 0
    if args.resume:
        state = torch.load(out_dir / 'last.pth', map_location=device)
        if state.get('total_epochs') != args.epochs:
            raise ValueError(f"Resume needs the original --epochs ({state.get('total_epochs')}), "
                             f"because the learning-rate schedule depends on it.")
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        completed = state['epoch']
        best_f1, best_epoch = state['best_f1'], state['best_epoch']
        epochs_since_improve = state['epochs_since_improve']
        print(f"Resuming after epoch {completed} (best validation F1 {best_f1:.3f}).")
    history = HistoryLogger(out_dir / 'analysis', resume_after_epoch=completed)

    for epoch in range(completed + 1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for img, ship, valid in pbar:
            img, ship, valid = img.to(device), ship.to(device), valid.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img), ship, valid)
            loss.backward()
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        train_loss = running / max(len(train_loader), 1)

        val = evaluate_split(model, val_loader, device, criterion, threshold=VAL_THRESHOLD)
        test_loss = evaluate_split(model, test_loader, device, criterion)['loss']
        epoch_seconds = time.perf_counter() - epoch_start

        print(f"Epoch {epoch:3d} | {epoch_seconds:6.0f}s | "
              f"loss train {train_loss:.4f}  val {val['loss']:.4f}  test {test_loss:.4f} | "
              f"val P {val['precision']:.3f}  R {val['recall']:.3f}  F1 {val['f1']:.3f}  "
              f"IoU {val['iou']:.3f}  FA/1000km2 {val['fa_per_1000km2']:.1f}")

        history.log(epoch=epoch, seconds=round(epoch_seconds, 1),
                    train_loss=train_loss, val_loss=val['loss'], test_loss=test_loss,
                    val_precision=val['precision'], val_recall=val['recall'], val_f1=val['f1'],
                    val_iou=val['iou'], val_fa_per_1000km2=val['fa_per_1000km2'])

        improved = val['f1'] > best_f1
        if improved:
            best_f1, best_epoch = val['f1'], epoch
            epochs_since_improve = 0
            save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'val': val},
                            out_dir / 'best.pth')
            print(f"           ^ new best (val F1 {best_f1:.3f}) saved.")
        else:
            epochs_since_improve += 1
        save_checkpoint({'epoch': epoch, 'total_epochs': args.epochs,
                         'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(), 'best_f1': best_f1,
                         'best_epoch': best_epoch,
                         'epochs_since_improve': epochs_since_improve}, out_dir / 'last.pth')

        if epochs_since_improve >= args.patience:
            print(f"No validation F1 improvement for {args.patience} epochs - stopping.")
            break

    print(f"\nDone. Best validation F1 {best_f1:.3f} at epoch {best_epoch}: {out_dir / 'best.pth'}")
    print("Next: python evaluate.py --checkpoint " + str(out_dir / 'best.pth'))


if __name__ == '__main__':
    main()
