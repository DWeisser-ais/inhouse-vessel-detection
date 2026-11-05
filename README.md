# In-house Vessel Detection

Detects vessels in Landsat 8/9 thermal imagery (ST_B10, surface temperature) with
**MTU-Net**, a segmentation network combining a ResNet-18 encoder, multi-level
vision-transformer branches and a U-shaped decoder. The task has two classes:
every pixel is either *vessel* or *not vessel*. Land is masked out and ignored.

Training data comes from three regions, all processed the same way:

| Region | Scenes | Source |
|---|---|---|
| Major ports | 6 | Hormuz, Malacca, Qatar, Rotterdam, Shanghai, Singapore |
| NE USA | 4 | AI4COPSEC, June 2020 |
| Mediterranean | 13 | AI4COPSEC, September 2020 |

See [`Datasets/README.md`](Datasets/README.md) for the full dataset description and
[`Datasets/scene_catalogue.csv`](Datasets/scene_catalogue.csv) for every scene and its split.

## Repository layout

```
.
├── README.md                  this file
├── environment.yml            conda environment (Python 3.11, rasterio, PyTorch ...)
├── pyproject.toml             project metadata and lint settings
├── .github/workflows/ci.yml   lint + compile check on every push and pull request
├── Datasets/                  DATA - not in git, except README.md and scene_catalogue.csv
└── Scripts/
    ├── data/                  data pipeline: raw Landsat scenes -> 512x512 training tiles
    │   ├── pipeline_config.py     ALL paths, splits and constants live here
    │   ├── run_data_pipeline.py   runs steps 1.0 -> 1.3 in order
    │   ├── 0.0-Import_Mediterranean_Scenes.py
    │   ├── 1.0-Thermal_Imagery_Cleaning.py   ST_B10 -> Celsius -> land-masked uint8
    │   ├── 1.1-Rasterise_Vessel_Masks.py     vessel polygons -> label raster
    │   ├── 1.2-Tiling_scenes.py              scene -> tiles, split by scene
    │   ├── 1.3-crrp_augment.py               copy-paste ship augmentation (train only)
    │   └── dataset.py                        PyTorch Dataset for the tiles
    ├── model/                 MTU-Net, loss, metrics, analysis plots
    └── training/              train.py, evaluate.py, visualisation
```

## Getting started

### 1. Environment

```
conda env create -f environment.yml
conda activate IHVesselDetection
```

Always run the scripts inside the activated environment. On Windows, calling the
environment's `python.exe` directly from Git Bash crashes GDAL with error `0xc06d007f`.

### 2. Data

The imagery is several gigabytes and is **not stored in git**. Get a copy of the
`Datasets/Full_Scenes/` folder from the team share and place it at `Datasets/Full_Scenes/`.
Alternatively, keep it anywhere and point the code at it:

```
set VESSEL_DATASETS_ROOT=D:\path\to\Datasets          (Windows cmd)
export VESSEL_DATASETS_ROOT=/path/to/Datasets         (Linux / Colab)
```

Then check the inputs and build the training tiles:

```
cd Scripts/data
python pipeline_config.py        # checks every scene, writes Datasets/scene_catalogue.csv
python run_data_pipeline.py      # builds Processed_Scenes/ and Tiled_Inputs/
```

### 3. Train and evaluate

```
cd Scripts/training
python train.py --epochs 300 --batch_size 4                 # writes checkpoints/
python evaluate.py --checkpoint checkpoints/best.pth --split Validation   # pick the threshold
python evaluate.py --checkpoint checkpoints/best.pth --split Test         # final numbers, once
```

`evaluate.py` reports IoU, probability of detection (Pd) and false-alarm rate (Fa),
overall and per region. Results go to `analysis/`.

## Changing the dataset

All scene lists live in `SPLITS` in [`Scripts/data/pipeline_config.py`](Scripts/data/pipeline_config.py).
Splits are made by scene, never by tile, so no part of a test scene is ever seen in
training. To add a scene or move one between splits, follow the steps in
[`Datasets/README.md`](Datasets/README.md#adding-a-new-scene).

## Trained models

Model weights (`*.pth`) are not committed. Publish a trained model as a
[GitHub Release](../../releases) and attach `best.pth`, together with the commit hash
and the `scene_catalogue.csv` it was trained on.

## Contributing

1. Create a branch from `main`, for example `feature/rgb-input` or `fix/land-mask`.
2. Keep data, weights and plots out of commits. `.gitignore` already excludes them.
3. Open a pull request. CI must pass before merging.

## Data sources and licences

- Landsat 8/9 Collection-2 Level-2 imagery: USGS, public domain.
- Land mask: OSM Water Layer, Yamazaki Lab, University of Tokyo, CC BY 4.0.
- Vessel labels: hand-labelled by the project team.
