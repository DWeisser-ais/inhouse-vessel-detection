

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless-safe backend
import matplotlib.pyplot as plt
import csv
 
 



#  1. TRAINING HISTORY — logging + curves

class HistoryLogger:
    """
    Records per-epoch metrics during training and writes them to CSV + JSON.
 
    Call .log(...) once per epoch from train.py, then .save() at the end.
    The CSV is what you'd paste into a report; the plots are generated from
    the same data so figures and tables can never disagree.
    """
 
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []
 
    def log(self, epoch, train_loss, val_iou, val_pd, val_fa,
            matched, targets, lr=None):
        self.rows.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_iou': float(val_iou),
            'val_pd': float(val_pd),
            'val_fa': float(val_fa),
            'matched': int(matched),
            'targets': int(targets),
            'lr': float(lr) if lr is not None else None,
        })
 
    def save(self):
        if not self.rows:
            return
        csv_path = self.out_dir / 'training_history.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)
        with open(self.out_dir / 'training_history.json', 'w') as f:
            json.dump(self.rows, f, indent=2)
        print(f"  History -> {csv_path}")
        self.plot()
 
    def plot(self):

        """Four-panel training curve figure."""
        
        e   = [r['epoch']      for r in self.rows]
        ls  = [r['train_loss'] for r in self.rows]
        iou = [r['val_iou']    for r in self.rows]
        pd_ = [r['val_pd']     for r in self.rows]
        fa  = [r['val_fa']*1e6 for r in self.rows]
 
        fig, ax = plt.subplots(2, 2, figsize=(11, 7))
 
        ax[0,0].plot(e, ls, 'o-', color='#2b6cb0')
        ax[0,0].set_title('Training loss')
        ax[0,0].set_xlabel('epoch'); ax[0,0].set_ylabel('FocalIoU loss')
        ax[0,0].grid(alpha=.3)
 
        ax[0,1].plot(e, pd_, 'o-', color='#2f855a')
        ax[0,1].set_title('Validation $P_d$ (probability of detection)')
        ax[0,1].set_xlabel('epoch'); ax[0,1].set_ylabel('$P_d$')
        ax[0,1].set_ylim(-0.02, 1.0); ax[0,1].grid(alpha=.3)
 
        ax[1,0].plot(e, iou, 'o-', color='#805ad5')
        ax[1,0].set_title('Validation IoU')
        ax[1,0].set_xlabel('epoch'); ax[1,0].set_ylabel('IoU')
        ax[1,0].set_ylim(-0.02, 1.0); ax[1,0].grid(alpha=.3)
 
        ax[1,1].plot(e, fa, 'o-', color='#c53030')
        ax[1,1].set_title('Validation $F_a$ (false alarm rate)')
        ax[1,1].set_xlabel('epoch'); ax[1,1].set_ylabel('$F_a$ ($\\times10^{-6}$)')
        ax[1,1].grid(alpha=.3)
 
        fig.suptitle('MTU-Net training history', fontsize=13)
        fig.tight_layout()
        p = self.out_dir / 'training_curves.png'
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"  Curves  -> {p}")
 
        # --- Dual-axis: train loss + val Pd vs epoch (the key diagnostic) ---
        # Loss alone flattens fast on imbalanced data and hides whether
        # detection is still improving. Overlaying Pd shows the real story.
        fig2, axL = plt.subplots(figsize=(8, 4.5))
        axL.plot(e, ls, 'o-', color='#2b6cb0', label='train loss')
        axL.set_xlabel('epoch'); axL.set_ylabel('train loss', color='#2b6cb0')
        axL.tick_params(axis='y', labelcolor='#2b6cb0'); axL.grid(alpha=.3)
        axR = axL.twinx()
        axR.plot(e, pd_, 's-', color='#2f855a', label='val Pd')
        axR.set_ylabel('validation $P_d$', color='#2f855a')
        axR.tick_params(axis='y', labelcolor='#2f855a'); axR.set_ylim(-0.02, 1.0)
        axL.set_title('Training loss vs validation detection ($P_d$)')
        fig2.tight_layout()
        p2 = self.out_dir / 'loss_vs_pd.png'
        fig2.savefig(p2, dpi=150); plt.close(fig2)
        print(f"  Loss+Pd -> {p2}")
 
 



#  2. THRESHOLD SWEEP — the key diagnostic

def threshold_sweep(probs_list, ships_list, valids_list,
                    thresholds=None, dist_thresh=3, out_dir='analysis'):
    """
    Recompute Pd / Fa / IoU across MANY decision thresholds.
 
    probs_list / ships_list / valids_list : lists of 2D numpy arrays
        (one per tile), as produced during evaluation.
    """
    from model.metrics import MetricAccumulator
 
    if thresholds is None:
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1,
                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
 
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in thresholds:
        acc = MetricAccumulator(threshold=t, dist_thresh=dist_thresh)
        for p, s, v in zip(probs_list, ships_list, valids_list):
            acc.update(p, s, v)
        r = acc.compute()
        rows.append({'threshold': t, 'IoU': r['IoU'], 'Pd': r['Pd'],
                     'Fa': r['Fa'], 'matched': r['matched'],
                     'targets': r['targets']})
        print(f"  thr={t:<6.3f}  Pd={r['Pd']:.3f}  "
              f"Fa={r['Fa']*1e6:8.2f}e-6  IoU={r['IoU']:.3f}  "
              f"({r['matched']}/{r['targets']})")
 
    # CSV table
    csv_path = out_dir / 'threshold_sweep.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
 
    # Plots: Pd & Fa vs threshold, and the ROC-style Pd vs Fa curve
    t   = [r['threshold'] for r in rows]
    pd_ = [r['Pd']        for r in rows]
    fa  = [r['Fa']*1e6    for r in rows]
 
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
 
    a = ax[0]
    a.plot(t, pd_, 'o-', color='#2f855a', label='$P_d$')
    a.set_xlabel('decision threshold'); a.set_ylabel('$P_d$', color='#2f855a')
    a.set_ylim(-0.02, 1.0); a.grid(alpha=.3)
    a2 = a.twinx()
    a2.plot(t, fa, 's--', color='#c53030', label='$F_a$')
    a2.set_ylabel('$F_a$ ($\\times10^{-6}$)', color='#c53030')
    a.set_title('Detection / false-alarm vs threshold')
 
    ax[1].plot(fa, pd_, 'o-', color='#2b6cb0')
    ax[1].set_xlabel('$F_a$ ($\\times10^{-6}$)'); ax[1].set_ylabel('$P_d$')
    ax[1].set_ylim(-0.02, 1.0); ax[1].grid(alpha=.3)
    ax[1].set_title('ROC-style curve ($P_d$ vs $F_a$)')
 
    fig.tight_layout()
    p = out_dir / 'threshold_sweep.png'
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Sweep   -> {csv_path}\n  Sweep   -> {p}")
    return rows
 
 



#  3. PREDICTION DISTRIBUTIONS — does the model separate ship from water?

def probability_distributions(probs_list, ships_list, valids_list,
                              out_dir='analysis'):
    """
    Histogram the model's output probability separately over
    SHIP pixels and WATER pixels.
 
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
 
    ship_p, water_p = [], []
    for p, s, v in zip(probs_list, ships_list, valids_list):
        valid = v > 0
        ship  = (s > 0) & valid
        water = (~(s > 0)) & valid
        ship_p.append(p[ship])
        water_p.append(p[water])
 
    ship_p  = np.concatenate(ship_p)  if ship_p  else np.array([])
    water_p = np.concatenate(water_p) if water_p else np.array([])
 
    if ship_p.size == 0:
        print("  No ship pixels found — cannot plot distributions.")
        return None
 
    stats = {
        'ship_mean': float(ship_p.mean()),
        'ship_median': float(np.median(ship_p)),
        'ship_max': float(ship_p.max()),
        'water_mean': float(water_p.mean()),
        'water_median': float(np.median(water_p)),
        'water_p999': float(np.percentile(water_p, 99.9)),
        'separation': float(ship_p.mean() - water_p.mean()),
    }
    print("\n  Prediction statistics:")
    for k, v in stats.items():
        print(f"    {k:<14s} {v:.6f}")
    print("    (separation > 0 means ships score HIGHER than water — good)")
 
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, max(ship_p.max(), np.percentile(water_p, 99.99)), 60)
    ax.hist(water_p, bins=bins, alpha=.6, label='water pixels',
            color='#4299e1', density=True, log=True)
    ax.hist(ship_p,  bins=bins, alpha=.6, label='ship pixels',
            color='#e53e3e', density=True, log=True)
    ax.set_xlabel('predicted probability'); ax.set_ylabel('density (log)')
    ax.set_title('Model output distribution: ship vs water pixels')
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    p = out_dir / 'probability_distributions.png'
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Dists   -> {p}")
 
    with open(out_dir / 'prediction_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    return stats
 
 



#  4. PER-TILE TABLE — where does it fail?

def per_tile_table(names, probs_list, ships_list, valids_list,
                   threshold=0.5, dist_thresh=3, out_dir='analysis'):
    """
    One row per test tile: ship count, detected count, Pd, Fa, IoU.
 
    WHY: an aggregate Pd of, say, 0.6 could mean "detects 60% of ships
    everywhere" or "detects 100% in open water and 0% in ports". Only a
    per-tile breakdown distinguishes those. Sort by Pd to immediately see
    your worst cases, then look at those tiles in the qualitative figure.
    """
    from model.metrics import MetricAccumulator, binarise, iou_score, pd_fa
 
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for nm, p, s, v in zip(names, probs_list, ships_list, valids_list):
        pred = binarise(p, threshold) * (v > 0).astype(np.uint8)
        gt   = (s > 0).astype(np.uint8)
        pd_, fa, matched, targets = pd_fa(pred, gt, (v > 0).astype(np.uint8),
                                          dist_thresh)
        rows.append({
            'tile': nm,
            'gt_ships': targets,
            'detected': matched,
            'Pd': round(pd_, 4) if pd_ is not None else '',
            'Fa_e6': round(fa * 1e6, 2),
            'IoU': round(iou_score(pred, gt), 4),
            'max_prob': round(float(p[(v > 0)].max()) if (v > 0).any() else 0, 4),
        })
 
    csv_path = out_dir / 'per_tile_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
 
    ship_tiles = [r for r in rows if r['gt_ships'] > 0]
    print(f"\n  Per-tile table -> {csv_path}")
    print(f"    tiles total: {len(rows)}, tiles with ships: {len(ship_tiles)}")
    if ship_tiles:
        detected = sum(r['detected'] for r in ship_tiles)
        total    = sum(r['gt_ships'] for r in ship_tiles)
        print(f"    ships detected: {detected}/{total}")
    return rows
 
 



#  6. CONFUSION MATRIX — pixel-level vessel vs water

def confusion_matrix(probs_list, ships_list, valids_list,
                     threshold=0.3, out_dir='analysis'):
    """
    Pixel-level 2x2 confusion matrix (water vs vessel) over VALID water.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    TP = TN = FP = FN = 0
    for p, s, v in zip(probs_list, ships_list, valids_list):
        valid = v > 0
        pred  = (p >= threshold) & valid
        gt    = (s > 0) & valid
        TP += int(( pred &  gt).sum())
        TN += int((~pred & ~gt & valid).sum())
        FP += int(( pred & ~gt).sum())
        FN += int((~pred &  gt).sum())
 
    cm = np.array([[TN, FP], [FN, TP]], dtype=np.float64)
    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)     # = pixel-level detection rate
    f1        = 2*precision*recall / (precision + recall + 1e-9)
 
    print("\n  Confusion matrix (pixels, valid water only):")
    print(f"                    pred water     pred vessel")
    print(f"    actual water    {TN:>12,}   {FP:>12,}")
    print(f"    actual vessel   {FN:>12,}   {TP:>12,}")
    print(f"    precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")
 
    # Row-normalised heatmap (so the tiny vessel row is visible)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.clip(row_sums, 1, None)
 
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, mat, title, fmt in [
        (ax[0], cm, 'Confusion matrix (raw counts)', ',.0f'),
        (ax[1], cm_norm, 'Row-normalised (per true class)', '.3f')]:
        im = a.imshow(mat, cmap='Blues')
        a.set_xticks([0,1]); a.set_yticks([0,1])
        a.set_xticklabels(['water','vessel']); a.set_yticklabels(['water','vessel'])
        a.set_xlabel('predicted'); a.set_ylabel('actual')
        a.set_title(title, fontsize=10)
        for i in range(2):
            for j in range(2):
                a.text(j, i, format(mat[i,j], fmt), ha='center', va='center',
                       color='black', fontsize=9)
    fig.tight_layout()
    p = out_dir / f'confusion_matrix_thr{threshold}.png'
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Confusion matrix -> {p}")
 
    stats = {'TP':TP,'TN':TN,'FP':FP,'FN':FN,
             'precision':precision,'recall':recall,'f1':f1,
             'threshold':threshold}
    with open(out_dir / 'confusion_matrix.json','w') as f:
        json.dump(stats, f, indent=2)
    return stats