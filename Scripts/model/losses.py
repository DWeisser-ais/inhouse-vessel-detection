import torch
import torch.nn as nn



 
 
class FocalIoULoss(nn.Module):
    def __init__(self, gamma=2.0, smooth=1.0):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth
 
    def forward(self, logits, targets, valid=None):
        """
        logits  : (B,1,H,W) raw model outputs (pre-sigmoid)
        targets : (B,1,H,W) ship labels in {0,1}
        valid   : (B,1,H,W) water mask in {0,1}; if None, all pixels count
        """
        p = torch.sigmoid(logits)
        y = targets
        eps = 1e-6
        p = p.clamp(min=eps, max=1.0 - eps)
 
        if valid is None:
            valid = torch.ones_like(y)
 
        # --- Focal term, per pixel ---
        focal = -(y * (1 - p) ** self.gamma * torch.log(p)
                  + (1 - y) * p ** self.gamma * torch.log(1 - p))
        # Mask land out, then average over WATER pixels only.
        focal = (focal * valid).sum() / valid.sum().clamp(min=1.0)
 
        # --- SoftIoU term, over valid pixels only ---
        pv = p * valid
        yv = y * valid
        dims = (1, 2, 3)
        inter = (pv * yv).sum(dims)
        total = pv.sum(dims) + yv.sum(dims) - inter
        soft_iou = ((self.smooth + inter) / (self.smooth + total)).mean()
 
        # --- FocalIoU ---
        exponent = (1.0 + soft_iou) / 2.0
        return 2.0 * (1.0 - soft_iou) * (focal ** exponent)



 
 
if __name__ == '__main__':
    loss_fn = FocalIoULoss()
    logits = torch.randn(2,1,64,64)
    target = (torch.rand(2,1,64,64) > 0.95).float()
    valid  = torch.ones_like(target); valid[:,:, :10, :] = 0   # land strip
    bad  = loss_fn(logits, target, valid)
    good = loss_fn(target*10-5, target, valid)
    print(f"loss random: {bad.item():.4f}")
    print(f"loss good  : {good.item():.4f}")
    assert good < bad
    print("FocalIoU (masked) sanity check passed.")