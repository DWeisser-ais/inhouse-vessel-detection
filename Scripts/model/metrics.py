"""
metrics.py
==========
Every performance number in the project is computed here, so training,
validation and testing can never disagree about what a metric means.

OBJECT LEVEL: "did we find the vessels?" (the operational question)
    A vessel is a connected group of label pixels. A predicted object matches a
    vessel when their centroids are within MATCH_DISTANCE_PX, one-to-one.
    Precision   TP / (TP + FP)      how many detections are real vessels
    Recall (Pd) TP / (TP + FN)      how many real vessels were found
    F1          harmonic mean of the two. Used to pick the best checkpoint
                and the operating threshold.
    AP          area under the precision-recall curve (threshold independent)
    FA / 1000 km2   false detections per 1000 km2 of water searched
    Recall by size  small / medium / large vessels (bins below)

PIXEL LEVEL: "did we outline them?" (land is excluded everywhere)
    IoU     TP / (TP + FP + FN), pooled over all tiles
    nIoU    mean IoU over tiles that contain a vessel
    Dice    2TP / (2TP + FP + FN), pooled over all tiles
    Fa      false vessel pixels / water pixels, pooled over all tiles

Counts are pooled before dividing, so an empty tile cannot drag IoU to zero
and a tile that is mostly land cannot dominate the false-alarm rate.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

MATCH_DISTANCE_PX = 3          # standard for small-target detection (Pd/Fa)
PIXEL_AREA_KM2 = 0.03 * 0.03   # nominal Landsat 30 m pixel

# Vessel size bins in label pixels. Chosen from the labelled vessels in this
# dataset (median 10 px), so each bin holds a useful share of them.
SMALL_MAX_PX = 7
MEDIUM_MAX_PX = 15
SIZE_CLASSES = ('small', 'medium', 'large')

# Decision thresholds used for the precision-recall curve and threshold choice.
THRESHOLDS = tuple(round(float(t), 2) for t in np.arange(0.02, 0.99, 0.02))

_EIGHT_CONN = np.ones((3, 3), dtype=int)
_NONE_XY = np.empty((0, 2))


def _find_objects(binary):
    """(centroids as an N x 2 row/col array, areas in pixels) of each object."""
    labelled, n = ndimage.label(binary, structure=_EIGHT_CONN)
    if n == 0:
        return _NONE_XY, np.empty(0)
    idx = range(1, n + 1)
    return (np.array(ndimage.center_of_mass(binary, labelled, idx)),
            np.asarray(ndimage.sum(binary, labelled, idx)))


def _size_class(area):
    if area <= SMALL_MAX_PX:
        return 'small'
    return 'medium' if area <= MEDIUM_MAX_PX else 'large'


def _match(gt_xy, pred_xy):
    """One-to-one matching that finds as many pairs as possible within range."""
    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return np.empty(0, int), np.empty(0, int)
    dist = np.linalg.norm(gt_xy[:, None] - pred_xy[None, :], axis=2)
    gt_i, pred_i = linear_sum_assignment(np.where(dist <= MATCH_DISTANCE_PX, dist, 1e6))
    keep = dist[gt_i, pred_i] <= MATCH_DISTANCE_PX
    return gt_i[keep], pred_i[keep]


@dataclass
class TileScore:
    """Everything measured on one tile at one threshold."""
    pixel_tp: int
    pixel_fp: int
    pixel_fn: int
    water_px: int
    tp_xy: np.ndarray            # matched vessels (true positives)
    fn_xy: np.ndarray            # missed vessels (false negatives)
    fp_xy: np.ndarray            # predictions with no vessel (false positives)
    size_found: dict
    size_total: dict

    @property
    def tp(self):
        return len(self.tp_xy)

    @property
    def fn(self):
        return len(self.fn_xy)

    @property
    def fp(self):
        return len(self.fp_xy)


def score_tile(prob, gt, valid, threshold):
    """Score one tile. prob, gt and valid are H x W arrays."""
    valid = valid > 0
    gt = (gt > 0) & valid
    pred = (prob >= threshold) & valid

    size_found = dict.fromkeys(SIZE_CLASSES, 0)
    size_total = dict.fromkeys(SIZE_CLASSES, 0)
    gt_xy, gt_area = _find_objects(gt)
    pred_xy, _ = _find_objects(pred)
    gt_hit, pred_hit = _match(gt_xy, pred_xy)

    for i, area in enumerate(gt_area):
        size = _size_class(area)
        size_total[size] += 1
        size_found[size] += int(i in gt_hit)

    return TileScore(
        pixel_tp=int((pred & gt).sum()),
        pixel_fp=int((pred & ~gt).sum()),
        pixel_fn=int((~pred & gt).sum()),
        water_px=int(valid.sum()),
        tp_xy=gt_xy[gt_hit],
        fn_xy=np.delete(gt_xy, gt_hit, axis=0),
        fp_xy=np.delete(pred_xy, pred_hit, axis=0),
        size_found=size_found,
        size_total=size_total,
    )


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


class DetectionMetrics:
    """Adds up TileScores and turns the pooled counts into metrics."""

    def __init__(self):
        self.tiles = self.tp = self.fp = self.fn = 0
        self.pixel_tp = self.pixel_fp = self.pixel_fn = self.water_px = 0
        self.tile_iou_sum = 0.0
        self.tiles_with_vessels = 0
        self.size_found = dict.fromkeys(SIZE_CLASSES, 0)
        self.size_total = dict.fromkeys(SIZE_CLASSES, 0)

    def add(self, s):
        self.tiles += 1
        self.tp += s.tp
        self.fp += s.fp
        self.fn += s.fn
        self.pixel_tp += s.pixel_tp
        self.pixel_fp += s.pixel_fp
        self.pixel_fn += s.pixel_fn
        self.water_px += s.water_px
        if s.tp + s.fn:
            self.tiles_with_vessels += 1
            self.tile_iou_sum += _ratio(s.pixel_tp, s.pixel_tp + s.pixel_fp + s.pixel_fn)
        for size in SIZE_CLASSES:
            self.size_found[size] += s.size_found[size]
            self.size_total[size] += s.size_total[size]

    def compute(self):
        precision = _ratio(self.tp, self.tp + self.fp)
        recall = _ratio(self.tp, self.tp + self.fn)
        result = {
            'tiles': self.tiles,
            'vessels': self.tp + self.fn,
            'tp': self.tp, 'fp': self.fp, 'fn': self.fn,
            'precision': precision,
            'recall': recall,
            'f1': _ratio(2 * precision * recall, precision + recall),
            'fa_per_1000km2': _ratio(self.fp * 1000, self.water_px * PIXEL_AREA_KM2),
            'iou': _ratio(self.pixel_tp, self.pixel_tp + self.pixel_fp + self.pixel_fn),
            'niou': _ratio(self.tile_iou_sum, self.tiles_with_vessels),
            'dice': _ratio(2 * self.pixel_tp, 2 * self.pixel_tp + self.pixel_fp + self.pixel_fn),
            'fa_rate': _ratio(self.pixel_fp, self.water_px),
        }
        for size in SIZE_CLASSES:
            result[f'vessels_{size}'] = self.size_total[size]
            result[f'recall_{size}'] = _ratio(self.size_found[size], self.size_total[size])
        return result


def threshold_sweep(probs, ships, valids, thresholds=THRESHOLDS):
    """Metrics at every threshold. Returns one dict per threshold, ascending."""
    rows = []
    for t in thresholds:
        acc = DetectionMetrics()
        for p, s, v in zip(probs, ships, valids):
            acc.add(score_tile(p, s, v, t))
        rows.append({'threshold': t, **acc.compute()})
    return rows


def best_f1_threshold(rows):
    """Threshold with the highest object F1 in a sweep."""
    return max(rows, key=lambda r: r['f1'])['threshold']


def average_precision(rows):
    """Area under the object precision-recall curve (interpolated, as in COCO)."""
    pts = sorted((r['recall'], r['precision']) for r in rows)
    recall = np.array([0.0] + [p[0] for p in pts] + [1.0])
    precision = np.array([0.0] + [p[1] for p in pts] + [0.0])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(recall) * precision[1:]))
