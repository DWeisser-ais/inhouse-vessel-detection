""" 
    Colour convention 
      GREEN  = missed ship (in ground truth, not detected)
      RED    = correct detection (overlaps a true ship)
      YELLOW = false alarm (predicted ship where there is none)
"""
 
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import ndimage
 




def _centroids(binary):
    lab, n = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return []
    return ndimage.center_of_mass(binary, lab, range(1, n + 1))



 
 
def visualise_tiles(names, images, probs, ships, valids,
                    threshold=0.5, dist_thresh=3, n_show=8,
                    out_dir='analysis', prefer_ship_tiles=True):
    """
    Build a multi-row figure. Each row = one test tile, four panels:
        1. Input thermal image
        2. Ground-truth mask
        3. Model probability map (continuous, NOT thresholded)
        4. Overlay: detections / misses / false alarms on the image
 
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
 
    # Prefer tiles that actually contain ships
    idx = list(range(len(names)))
    if prefer_ship_tiles:
        with_ships = [i for i in idx if (ships[i] > 0).any()]
        without    = [i for i in idx if not (ships[i] > 0).any()]
        # Show mostly ship tiles, plus a couple of empties to inspect Fa.
        idx = with_ships[:max(1, n_show - 2)] + without[:2]
    idx = idx[:n_show]
    if not idx:
        print("  Nothing to visualise.")
        return None
 
    rows = len(idx)
    fig, axes = plt.subplots(rows, 4, figsize=(15, 3.4 * rows))
    if rows == 1:
        axes = axes[None, :]
 
    for r, i in enumerate(idx):
        img   = images[i]
        prob  = probs[i]
        gt    = (ships[i] > 0).astype(np.uint8)
        valid = (valids[i] > 0).astype(np.uint8)
        pred  = ((prob >= threshold).astype(np.uint8)) * valid
 
        # --- Panel 1: input ---
        axes[r,0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[r,0].set_title(f"input\n{names[i][:28]}", fontsize=8)
 
        # --- Panel 2: ground truth ---
        axes[r,1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[r,1].set_title(f"ground truth ({int(gt.sum())} px)", fontsize=8)
 
        # --- Panel 3: RAW probability map (not thresholded) ---
        im = axes[r,2].imshow(prob, cmap='inferno', vmin=0, vmax=max(prob.max(), 1e-6))
        axes[r,2].set_title(f"prediction (max={prob.max():.3f})", fontsize=8)
        plt.colorbar(im, ax=axes[r,2], fraction=.046)
 
        # --- Panel 4: overlay with paper's colour convention ---
        axes[r,3].imshow(img, cmap='gray', vmin=0, vmax=1)
        gt_c   = _centroids(gt)
        pred_c = _centroids(pred)
        used = set()
        for gc in gt_c:
            hit = False
            for j, pc in enumerate(pred_c):
                if j in used:
                    continue
                if np.hypot(gc[0]-pc[0], gc[1]-pc[1]) <= dist_thresh:
                    used.add(j); hit = True; break
            colour = 'lime' if hit else 'red'     # detected / missed
            axes[r,3].add_patch(Circle((gc[1], gc[0]), 7, fill=False,
                                        edgecolor=colour, lw=1.4))
        for j, pc in enumerate(pred_c):
            if j not in used:                      # unmatched prediction
                axes[r,3].add_patch(Circle((pc[1], pc[0]), 7, fill=False,
                                            edgecolor='yellow', lw=1.4))
        n_hit  = len(used)
        n_miss = len(gt_c) - n_hit
        n_fa   = len(pred_c) - n_hit
        axes[r,3].set_title(
            f"green=hit({n_hit}) red=miss({n_miss}) yellow=FA({n_fa})",
            fontsize=8)
 
        for c in range(4):
            axes[r,c].set_xticks([]); axes[r,c].set_yticks([])
 
    fig.suptitle(f'Test-set qualitative results (threshold={threshold})',
                 fontsize=13)
    fig.tight_layout()
    p = out_dir / f'qualitative_thr{threshold}.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f"  Qualitative -> {p}")
    return p