"""
analysis.py
===========
Training history and the precision-recall figure.

Written during training (updated every epoch, so an interrupted run keeps them):
    training_history.csv     every value below, one row per epoch
    loss_curves.png          train, validation and test loss against epoch
    validation_metrics.png   validation detection quality against epoch

The test loss is plotted for information only. Nothing in training reads it,
so it cannot influence the checkpoint, early stopping or the threshold.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # headless-safe backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

INK ='#0b0b0b'
MUTED = '#52514e'
GRID = '#e5e4df'
SERIES = ('#2a78d6', '#eb6834', '#1baf7a')     # fixed order: 1st, 2nd, 3rd series


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, color=INK, loc='left')
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    if xlabel == 'Epoch':
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)


def _plot_lines(ax, x, series, end_labels=True):
    """Draw (label, values) lines with a legend and optionally a label at each end."""
    marker = 'o' if len(x) <= 30 else None
    for (label, values), colour in zip(series, SERIES):
        ax.plot(x, values, color=colour, linewidth=2, marker=marker, markersize=4, label=label)
    if len(series) > 1:
        ax.legend(frameon=False, labelcolor=MUTED, loc='best')
    ax.margins(x=0.08)
    if not end_labels:
        return
    # Direct labels, nudged apart so close lines do not overprint.
    span = max(max(v) for _, v in series) - min(min(v) for _, v in series) or 1.0
    used = []
    for label, values in sorted(series, key=lambda s: s[1][-1]):
        y = values[-1]
        for other in used:
            y = max(y, other + span * 0.045)
        used.append(y)
        ax.annotate(label, (x[-1], y), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=8, color=MUTED, annotation_clip=False)


class HistoryLogger:
    """One row per epoch, written to disk and re-plotted after every epoch."""

    def __init__(self, out_dir, resume_after_epoch=0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []
        if resume_after_epoch:
            with open(self.out_dir / 'training_history.csv', newline='') as f:
                for row in csv.DictReader(f):
                    if int(row['epoch']) <= resume_after_epoch:
                        self.rows.append({**{k: float(v) for k, v in row.items()},
                                          'epoch': int(row['epoch'])})

    def log(self, **row):
        self.rows.append(row)
        with open(self.out_dir / 'training_history.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.rows)
        self._plot()

    def _column(self, name):
        return [r[name] for r in self.rows]

    def _plot(self):
        epochs = self._column('epoch')
        f1 = self._column('val_f1')
        best_epoch = epochs[f1.index(max(f1))]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        _plot_lines(ax, epochs, [('Train', self._column('train_loss')),
                                 ('Validation', self._column('val_loss')),
                                 ('Test', self._column('test_loss'))])
        ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle='--')
        ax.annotate(f'best checkpoint (epoch {best_epoch})', (best_epoch, 1),
                    xycoords=('data', 'axes fraction'), xytext=(4, -10),
                    textcoords='offset points', fontsize=8, color=MUTED)
        _style(ax, 'Loss against epoch', 'Epoch', 'FocalIoU loss')
        fig.tight_layout()
        fig.savefig(self.out_dir / 'loss_curves.png', dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        _plot_lines(axes[0], epochs, [('Precision', self._column('val_precision')),
                                      ('Recall (Pd)', self._column('val_recall')),
                                      ('F1', f1)])
        axes[0].set_ylim(0, 1)
        _style(axes[0], 'Vessel detection', 'Epoch', 'Score')
        _plot_lines(axes[1], epochs, [('IoU', self._column('val_iou'))])
        axes[1].set_ylim(0, 1)
        _style(axes[1], 'Pixel IoU', 'Epoch', 'IoU')
        _plot_lines(axes[2], epochs, [('FA', self._column('val_fa_per_1000km2'))])
        axes[2].set_ylim(bottom=0)
        _style(axes[2], 'False alarms', 'Epoch', 'False detections per 1000 km²')
        for a in axes:
            a.axvline(best_epoch, color=MUTED, linewidth=1, linestyle='--')
        fig.suptitle(f'Validation metrics (best checkpoint: epoch {best_epoch})',
                     fontsize=12, color=INK, x=0.01, ha='left')
        fig.tight_layout()
        fig.savefig(self.out_dir / 'validation_metrics.png', dpi=150)
        plt.close(fig)


def plot_precision_recall(rows, threshold, ap, path):
    """Precision-recall curve, and the three scores against decision threshold."""
    point = next(r for r in rows if abs(r['threshold'] - threshold) < 0.011)
    # In threshold order, ignoring thresholds where the model predicts nothing.
    curve = [r for r in rows if r['tp'] + r['fp'] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot([r['recall'] for r in curve], [r['precision'] for r in curve],
                 color=SERIES[0], linewidth=2)
    axes[0].scatter([point['recall']], [point['precision']], s=60, color=SERIES[1],
                    edgecolor='white', linewidth=2, zorder=3)
    axes[0].annotate(f"threshold {threshold:.2f}\nF1 {point['f1']:.3f}",
                     (point['recall'], point['precision']), xytext=(10, 8),
                     textcoords='offset points', fontsize=9, color=MUTED)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    _style(axes[0], f'Vessel precision-recall (AP = {ap:.3f})', 'Recall (Pd)', 'Precision')

    t = [r['threshold'] for r in rows]
    _plot_lines(axes[1], t, [('Precision', [r['precision'] for r in rows]),
                             ('Recall (Pd)', [r['recall'] for r in rows]),
                             ('F1', [r['f1'] for r in rows])], end_labels=False)
    axes[1].axvline(threshold, color=MUTED, linewidth=1, linestyle='--')
    axes[1].set_ylim(0, 1.02)
    _style(axes[1], 'Scores against decision threshold', 'Threshold', 'Score')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
