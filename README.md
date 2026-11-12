# Aistech: In-house Vessel Detection Model

This repository houses the model and information relating to a vessel detection model built inhouse within Aistech Space, which uses Landsat 8/9 thermal imagery (ST_B10, surface temperature) alongside a **MTU-Net**, a segmentation network combining a ResNet-18 encoder, multi-lebel ViT branches and a U-shaped decoder.
The task has two classes: Every pixel is either *vessel* or *not vessel*.
Land has been masked out and hence ignored.

Training data comes from three regions, all processed the same way:

| Region | Scenes | Years |
|---|---|---|
| International ports | 6 | 2025 and 2026 |
| NE USA | 4 | June 2020 |
| Mediterranean | 13 | September 2020 |

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
    │   ├── pipeline_config.py                ALL paths, splits and constants live here
    │   ├── run_data_pipeline.py              runs steps 1.0 -> 1.2 (and 1.3 if enabled)
    │   ├── 0.0-Import_Mediterranean_Scenes.py
    │   ├── 1.0-Thermal_Imagery_Cleaning.py   ST_B10 -> Celsius -> land-masked uint8
    │   ├── 1.1-Rasterise_Vessel_Masks.py     vessel polygons -> label raster
    │   ├── 1.2-Tiling_scenes.py              scene -> tiles, split by scene
    │   ├── 1.3-crrp_augment.py               copy-paste ship augmentation (off by default)
    │   └── dataset.py                        PyTorch Dataset for the tiles
    ├── model/                                MTU-Net, loss, metrics, analysis plots
    └── training/                             train.py, evaluate.py, visualisation
```

## Getting started

### 1. Environment

```
conda env create -f environment.yml
conda activate IHVesselDetection
```

Always run the scripts inside the activated environment. 

### 2. Data

The imagery is several gigabytes and is **not stored in git**. 
Get a copy of the `Datasets/Full_Scenes/` folder from the team share and place it at `Datasets/Full_Scenes/`.
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
python train.py --epochs 300 --batch_size 4                   # writes checkpoints/
python evaluate.py --checkpoint checkpoints/best.pth --split Validation
python evaluate.py --checkpoint checkpoints/best.pth --split Test   # final numbers, once
```

Training records these in `checkpoints/analysis/`, updated every epoch:

| File | Contents |
|---|---|
| `loss_curves.png` | train, validation and test loss against epoch |
| `validation_metrics.png` | precision, recall, F1, IoU and false alarms against epoch |
| `training_history.csv` | every value above, one row per epoch |
| `model_size_report.json` | parameters, weights, MACs/FLOPs, peak RAM, latency |

The checkpoint with the best validation F1 is kept as `best.pth`. 

`evaluate.py` picks the decision threshold on the validation split (highest vessel F1),
applies it to the requested split and writes to `checkpoints/evaluation/<split>/`:

| File | Contents |
|---|---|
| `metrics.json` | every metric, overall and per region |
| `per_scene.csv` | main metrics for each scene |
| `threshold_sweep.csv` | precision, recall and F1 at every threshold |
| `precision_recall.png` | precision-recall curve and scores against threshold |
| `examples_detections.png` | tiles where vessels were found |
| `examples_errors.png` | tiles with false positives and missed vessels |

**Vessel Detection metrics:** precision, recall (Pd), F1, average precision, false alarms per
1000 km2, recall by vessel size. **Pixel metrics:** IoU, nIoU, Dice, false-alarm rate.
Definitions are at the top of `Scripts/model/metrics.py`.

Run `python Scripts/model/deployment_metrics.py` for the model-size report on its own.

## Changing the dataset

All scene lists live in `SPLITS` in [`Scripts/data/pipeline_config.py`](Scripts/data/pipeline_config.py).
Splits are made by scene, never by tile, so no part of a test scene is ever seen in
training. To add a scene or move one between splits, follow the steps in
[`Datasets/README.md`](Datasets/README.md#adding-a-new-scene).

## Trained models

Model weights (`*.pth`) are not committed. Publish a trained model as a
[GitHub Release](../../releases) and attach `best.pth`, together with the commit hash
and the `scene_catalogue.csv` it was trained on.

## Data sources and licences

- Landsat 8/9 Collection-2 Level-2 imagery: USGS, public domain.
- Land mask: OSM Water Layer, Yamazaki Lab, University of Tokyo, CC BY 4.0.
- Vessel labels: hand-labelled by the project team.
