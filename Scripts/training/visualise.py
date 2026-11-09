"""
visualise.py
============
Example tiles at the evaluation threshold.

    examples_detections.png   tiles where the model found vessels, cleanest first
    examples_errors.png       tiles with positive predictions and mistakes:
                              half ranked by false positives, half by misses

Examples are taken from every region in turn, so one scene cannot fill a figure.
Markers differ in shape as well as colour:
    green circle = detected vessel (TP)
    orange square = missed vessel (FN)
    yellow triangle = false alarm (FP)
"""

import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MARKERS = (('tp_xy', 'o', '#1baf7a', 'Detected (TP)'),
           ('fn_xy', 's', '#eb6834', 'Missed (FN)'),
           ('fp_xy', '^', '#eda100', 'False alarm (FP)'))
_POSITION = re.compile(r'_y(\d+)_x(\d+)')


def _round_robin(candidates, key, regions, n):
    """Pick the n best-ranked tiles, taking one from each region in turn."""
    by_region = {}
    for i in sorted(candidates, key=key, reverse=True):
        by_region.setdefault(regions[i], []).append(i)
    picked = []
    while len(picked) < n and any(by_region.values()):
        for region in sorted(by_region):
            if by_region[region] and len(picked) < n:
                picked.append(by_region[region].pop(0))
    return picked


def select_examples(scores, regions, n):
    """(detection tile indices, error tile indices), up to n of each."""
    tiles = range(len(scores))
    detections = _round_robin([i for i in tiles if scores[i].tp > 0],
                              lambda i: scores[i].tp - scores[i].fp - scores[i].fn, regions, n)
    false_alarms = _round_robin([i for i in tiles if scores[i].fp > 0],
                                lambda i: scores[i].fp, regions, n // 2)
    misses = _round_robin([i for i in tiles if scores[i].fn > 0 and scores[i].tp > 0
                           and i not in false_alarms],
                          lambda i: scores[i].fn, regions, n - len(false_alarms))
    return detections, false_alarms + misses


def draw_examples(indices, data, scores, threshold, title, path):
    """One row per tile: input, ground truth, probability, marked-up result."""
    if not indices:
        print(f"  No tiles to draw for {path.name}.")
        return

    rows = len(indices)
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.5 * rows), squeeze=False)
    for r, i in enumerate(indices):
        image = data['dataset'][i][0][0].numpy()
        water = data['valids'][i]
        prob = np.ma.masked_where(~water, data['probs'][i])
        truth = np.where(water, 0.0, 0.35)
        truth[data['ships'][i]] = 1.0

        axes[r, 0].imshow(image, cmap='gray', vmin=0, vmax=1)
        position = _POSITION.search(data['names'][i])
        where = f"  y{int(position[1])} x{int(position[2])}" if position else ''
        axes[r, 0].set_title(f"{data['scenes'][i]}{where}", fontsize=9, loc='left')

        axes[r, 1].imshow(truth, cmap='gray', vmin=0, vmax=1)
        axes[r, 1].set_title(f"Ground truth: {scores[i].tp + scores[i].fn} vessels (land grey)",
                             fontsize=9, loc='left')

        shown = axes[r, 2].imshow(prob, cmap='magma', vmin=0, vmax=1)
        axes[r, 2].set_facecolor('#555555')
        axes[r, 2].set_title('Model probability', fontsize=9, loc='left')
        fig.colorbar(shown, ax=axes[r, 2], fraction=0.046, pad=0.02)

        axes[r, 3].imshow(image, cmap='gray', vmin=0, vmax=1)
        for attr, marker, colour, _ in MARKERS:
            xy = getattr(scores[i], attr)
            if len(xy):
                axes[r, 3].scatter(xy[:, 1], xy[:, 0], s=130, marker=marker, facecolors='none',
                                   edgecolors=colour, linewidths=1.6)
        axes[r, 3].set_xlim(0, image.shape[1] - 1)
        axes[r, 3].set_ylim(image.shape[0] - 1, 0)
        axes[r, 3].set_title(f"TP {scores[i].tp}   FN {scores[i].fn}   FP {scores[i].fp}",
                             fontsize=9, loc='left')

        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])

    handles = [Line2D([], [], marker=m, linestyle='', markerfacecolor='none', markeredgecolor=c,
                      markeredgewidth=1.6, markersize=9, label=label)
               for _, m, c, label in MARKERS]
    fig.suptitle(f'{title} (threshold {threshold:.2f})', fontsize=13, x=0.01, ha='left')
    fig.legend(handles=handles, loc='upper right', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.5 * rows)))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved -> {path}")
