"""
run_data_pipeline.py
====================
Runs the whole data pipeline in order: 1.0 -> 1.1 -> 1.2 -> 1.3.

    python run_data_pipeline.py                          # everything, all scenes
    python run_data_pipeline.py --scenes USA_June29      # 1.0-1.2 for one scene, then CRRP
    python run_data_pipeline.py --steps 1.2 1.3          # re-tile + re-augment only
                                                         # (skips the slow 1.0 step)

1.3 (CRRP) always rebuilds from ALL train scenes, because the harvested ship
pool is shared across them.
"""

import argparse
import importlib.util
import time
from pathlib import Path

import pipeline_config as cfg

HERE = Path(__file__).resolve().parent
STEPS = {
    '1.0': '1.0-Thermal_Imagery_Cleaning.py',
    '1.1': '1.1-Rasterise_Vessel_Masks.py',
    '1.2': '1.2-Tiling_scenes.py',
    '1.3': '1.3-crrp_augment.py',
}


def load(filename):
    # The step files have dots/hyphens in their names, so load them by path.
    spec = importlib.util.spec_from_file_location(filename.replace('.', '_').replace('-', '_'),
                                                  HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', nargs='*', default=list(STEPS), choices=list(STEPS))
    ap.add_argument('--scenes', nargs='*', help='default: every scene in SPLITS')
    args = ap.parse_args()

    cfg.check_splits()
    scenes = args.scenes or cfg.all_scenes()
    for s in scenes:
        cfg.split_of(s)                  # fail now, not halfway through
        cfg.find_scene_inputs(s)

    t_total = time.perf_counter()
    for step in sorted(args.steps):
        t0 = time.perf_counter()
        print(f"\n################ STEP {step}: {STEPS[step]} ################")
        load(STEPS[step]).run(scenes)
        print(f"  [step {step} took {time.perf_counter() - t0:.0f} s]")
    print(f"\nPipeline finished in {(time.perf_counter() - t_total)/60:.1f} min.")