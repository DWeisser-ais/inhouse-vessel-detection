from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import random
import tempfile
from rasterio.transform import from_origin
 
LAND_NODATA = 1
 



 
class ShipTileDataset(Dataset):
    
    def __init__(self, dir_pairs, augment=False, name="Dataset"):
        self.augment = augment
        self.samples = []
        for images_dir, masks_dir in dir_pairs:
            images_dir, masks_dir = Path(images_dir), Path(masks_dir)
            img_names  = {p.name for p in images_dir.glob('*.tif')}
            mask_names = {p.name for p in masks_dir.glob('*.tif')}
            paired  = sorted(img_names & mask_names)
            missing = img_names - mask_names
            if missing:
                raise ValueError(
                    f"{len(missing)} image tiles in {images_dir} have NO "
                    f"matching mask, e.g. {sorted(missing)[:3]}")
            if not paired:
                raise ValueError(f"No paired tiles in {images_dir}")
            for nm in paired:
                self.samples.append((images_dir / nm, masks_dir / nm))
        if not self.samples:
            raise ValueError("No paired tiles found across any location.")
        print(f"{name}: {len(self.samples)} paired tiles "
              f"({'augmented' if augment else 'clean'}) "
              f"from {len(dir_pairs)} location(s)")
 
    def __len__(self):
        return len(self.samples)
 
    def _read(self, path):
        with rasterio.open(path) as src:
            return src.read(1)
 
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img      = self._read(img_path).astype(np.float32)
        raw_mask = self._read(mask_path)
 
        img = np.clip(img, 0, 255) / 255.0
        ship  = (raw_mask == 255).astype(np.float32)
        valid = (raw_mask != LAND_NODATA).astype(np.float32)
 
        if self.augment:
            img, ship, valid = self._augment(img, ship, valid)
 
        img_t   = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)
        ship_t  = torch.from_numpy(np.ascontiguousarray(ship)).unsqueeze(0)
        valid_t = torch.from_numpy(np.ascontiguousarray(valid)).unsqueeze(0)
        return img_t, ship_t, valid_t
 
    def _augment(self, img, ship, valid):
        if random.random() < 0.5:
            img, ship, valid = img[:, ::-1], ship[:, ::-1], valid[:, ::-1]
        if random.random() < 0.5:
            img, ship, valid = img[::-1, :], ship[::-1, :], valid[::-1, :]
        k = random.randint(0, 3)
        if k:
            img, ship, valid = np.rot90(img,k), np.rot90(ship,k), np.rot90(valid,k)
        if random.random() < 0.2:
            noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
            img = np.clip(img + noise * valid, 0.0, 1.0)
        return (np.ascontiguousarray(img), np.ascontiguousarray(ship),
                np.ascontiguousarray(valid))



 
 
if __name__ == '__main__':

    tmp = Path(tempfile.mkdtemp())
    imgd, maskd = tmp / 'images', tmp / 'masks'
    imgd.mkdir(); maskd.mkdir()
    prof = dict(driver='GTiff', height=64, width=64, count=1,
                dtype='uint8', transform=from_origin(0,0,1,1))
    for i in range(3):
        nm = f"scene_y{i:05d}_x00000.tif"
        img = (np.random.rand(64,64)*200+30).astype('uint8'); img[:10,:]=0
        with rasterio.open(imgd/nm,'w',**prof) as d: d.write(img,1)
        m = np.zeros((64,64),'uint8'); m[:10,:]=LAND_NODATA; m[30:33,30:33]=255
        with rasterio.open(maskd/nm,'w',**prof) as d: d.write(m,1)
    ds = ShipTileDataset([(imgd,maskd)], augment=True)
    img, ship, valid = ds[0]
    print("image", tuple(img.shape), f"[{img.min():.2f},{img.max():.2f}]")
    print("ship uniques", torch.unique(ship).tolist())
    print("valid uniques", torch.unique(valid).tolist(),
          "water frac", round(valid.mean().item(),3))
    assert set(torch.unique(ship).tolist()) <= {0.0,1.0}
    assert set(torch.unique(valid).tolist()) <= {0.0,1.0}
    print("dataset self-test passed.")