
import numpy as np
from scipy import ndimage



 
 
def binarise(prob_map, threshold=0.5):
    return (prob_map >= threshold).astype(np.uint8)



 
 
def iou_score(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return float(inter) / float(union + 1e-9)





 
def centroids(binary_map):
    structure = np.ones((3, 3), dtype=int)      # 8-connectivity
    labelled, n = ndimage.label(binary_map, structure=structure)
    if n == 0:
        return []
    return ndimage.center_of_mass(binary_map, labelled, range(1, n + 1))



 
 
def pd_fa(pred_bin, gt_bin, valid, dist_thresh=3):
    gt_cents   = centroids(gt_bin)
    pred_cents = centroids(pred_bin)
    total_targets = len(gt_cents)
 
    # False-alarm pixels: predicted ship, not true ship, AND on valid water.
    false_pixels = int(np.logical_and(
        np.logical_and(pred_bin == 1, gt_bin == 0), valid == 1).sum())
    valid_pixels = int(valid.sum())
    fa = false_pixels / max(valid_pixels, 1)
 
    if total_targets == 0:
        return None, fa, 0, 0
 
    matched = 0
    used = set()
    for gc in gt_cents:
        for j, pc in enumerate(pred_cents):
            if j in used:
                continue
            if np.hypot(gc[0]-pc[0], gc[1]-pc[1]) <= dist_thresh:
                matched += 1
                used.add(j)
                break
    return matched / total_targets, fa, matched, total_targets
 



 
class MetricAccumulator:
    def __init__(self, threshold=0.5, dist_thresh=3):
        self.threshold = threshold
        self.dist_thresh = dist_thresh
        self.reset()
 
    def reset(self):
        self.iou_sum = 0.0; self.iou_n = 0
        self.matched = 0; self.targets = 0
        self.fa_sum = 0.0; self.fa_n = 0
 
    def update(self, prob_map, gt_bin, valid):
        pred_bin = binarise(prob_map, self.threshold)
        gt_bin   = (gt_bin > 0).astype(np.uint8)
        valid    = (valid > 0).astype(np.uint8)
 
        # Restrict predictions to valid water so land can't score.
        pred_bin = pred_bin * valid
 
        self.iou_sum += iou_score(pred_bin, gt_bin); self.iou_n += 1
        pd, fa, m, t = pd_fa(pred_bin, gt_bin, valid, self.dist_thresh)
        self.matched += m; self.targets += t
        self.fa_sum += fa; self.fa_n += 1
 
    def compute(self):
        return {
            'IoU': self.iou_sum / max(self.iou_n, 1),
            'Pd':  self.matched / max(self.targets, 1),
            'Fa':  self.fa_sum / max(self.fa_n, 1),
            'targets': self.targets, 'matched': self.matched,
        }