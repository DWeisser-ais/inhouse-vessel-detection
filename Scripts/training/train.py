

"""
    python train.py --epochs 15 --batch_size 4
"""

import argparse
import random
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.mtu_net import MTUNet
from data.dataset import ShipTileDataset
from model.losses import FocalIoULoss
from model.metrics import MetricAccumulator
from model.deployment_metrics import full_report, edge_ai_assessment
from model.analysis import HistoryLogger
from training.timing import StageTimer
from data.pipeline_config import SPLITS, TRAIN, TRAIN_CRRP, VALIDATION, tile_dirs
from tqdm import tqdm


# ===================================================================== #
#  CONFIGURATION — edit here
# ===================================================================== #
# Paths and train/val/test scenes now come from data/pipeline_config.py

USE_CRRP = True          # include Tiled_Inputs/Train_CRRP synthetic tiles

VAL_THRESHOLD = 0.3      
LEARNING_RATE = 0.05     
FOCAL_GAMMA   = 2.0     

# num_workers is auto-chosen
DEFAULT_WORKERS = 0 if platform.system() == 'Windows' else 4

# ===================================================================== #





def make_dir_pairs(split, scenes, required=True):
    """(Tiles, Labels) folder pairs for each scene in a split."""
    pairs = []
    for scene in scenes:
        img_dir, lbl_dir = tile_dirs(split, scene)
        has_tiles = img_dir.exists() and any(img_dir.glob('*.tif'))
        if not has_tiles:
            if required:
                raise FileNotFoundError(f"No tiles in {img_dir} - run the data pipeline.")
            print(f"  (skipping {split}/{scene}: no tiles)")
            continue
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Missing labels folder: {lbl_dir}")
        pairs.append((img_dir, lbl_dir))
    return pairs





def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)





def evaluate(model, loader, device, threshold=VAL_THRESHOLD):
    model.eval()
    acc = MetricAccumulator(threshold=threshold)
    with torch.no_grad():
        for img, ship, valid in loader:
            img = img.to(device)
            probs = torch.sigmoid(model(img)).cpu().numpy()
            ships, valids = ship.numpy(), valid.numpy()
            for b in range(probs.shape[0]):
                acc.update(probs[b, 0], ships[b, 0], valids[b, 0])
    return acc.compute()





def main():
    ap = argparse.ArgumentParser()
    # Only the two things you genuinely vary between runs:
    ap.add_argument('--epochs',      type=int, default=300)
    ap.add_argument('--batch_size',  type=int, default=4)

    # Rare overrides (have sensible defaults; usually omitted):
    
    ap.add_argument('--out_dir',     default='checkpoints')
    ap.add_argument('--patience',    type=int, default=40)
    ap.add_argument('--num_workers', type=int, default=DEFAULT_WORKERS)
    ap.add_argument('--seed',        type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}   |   workers: {args.num_workers}   "
          f"|   val threshold: {VAL_THRESHOLD}   |   CRRP: {USE_CRRP}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- Datasets & loaders ---
    train_pairs = make_dir_pairs(TRAIN, SPLITS[TRAIN])
    if USE_CRRP:
        train_pairs += make_dir_pairs(TRAIN_CRRP, SPLITS[TRAIN], required=False)
    train_ds = ShipTileDataset(train_pairs, augment=True)
    val_ds   = ShipTileDataset(make_dir_pairs(VALIDATION, SPLITS[VALIDATION]),
                               augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    # --- Model ---
    model = MTUNet(in_channels=1).to(device)

    # One-time efficiency + edge-AI deployability report.
    print("\nMeasuring model efficiency (for deployment planning)...")
    dep = full_report(model, input_shape=(1, 1, 512, 512), device=device)
    edge_ai_assessment(dep)

    stage_timer = StageTimer()
    history = HistoryLogger(out_dir / 'analysis')

    # Xavier init
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            if m.weight is not None and m.weight.dim() >= 2:
                torch.nn.init.xavier_uniform_(m.weight)
            if getattr(m, 'bias', None) is not None:
                torch.nn.init.zeros_(m.bias)

    criterion = FocalIoULoss(gamma=FOCAL_GAMMA)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_pd = -1.0
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]",
                    leave=False)
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
        epoch_secs = time.perf_counter() - epoch_t0
        stage_timer.add('train_epoch', epoch_secs)

        val = evaluate(model, val_loader, device, threshold=VAL_THRESHOLD)
        print(f"Epoch {epoch:3d} | {epoch_secs:6.1f}s | Train loss {train_loss:.4f} | "
              f"val IoU {val['IoU']:.3f}  val Pd {val['Pd']:.3f}  "
              f"val Fa {val['Fa']*1e6:.2f}e-6 "
              f"({val['matched']}/{val['targets']} ships)")

        history.log(epoch, train_loss, val['IoU'], val['Pd'], val['Fa'],
                    val['matched'], val['targets'],
                    lr=optimizer.param_groups[0]['lr'])

        torch.save({'epoch': epoch, 'model': model.state_dict()},
                   out_dir / 'last.pth')
        if val['Pd'] > best_pd:
            best_pd = val['Pd']
            epochs_since_improve = 0
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val': val}, out_dir / 'best.pth')
            print(f"           ^ new best (Pd={best_pd:.3f}) saved.")
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= args.patience:
            print(f"No val improvement for {args.patience} epochs — stopping.")
            break

    history.save()
    stage_timer.summary(total_epochs_planned=args.epochs,
                        out_path=out_dir / 'analysis' / 'timing_summary.json')

    print(f"\nDone. Best val Pd = {best_pd:.3f}. Best checkpoint: {out_dir/'best.pth'}")
    print("Run evaluate.py on the TEST set ONCE with best.pth.")





if __name__ == '__main__':
    main()