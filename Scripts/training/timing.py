

import time
from contextlib import contextmanager
from pathlib import Path
import json





@contextmanager
def timer(label, store=None):
    """
    Context manager that times a block of code.

        with timer("epoch"):
            train_one_epoch()

    If `store` (a dict) is given, the elapsed seconds are appended under
    `label` so you can collect many timings.
    """
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"    [timing] {label}: {dt:.2f} s")
    if store is not None:
        store.setdefault(label, []).append(dt)





class StageTimer:

    def __init__(self):
        self.stages = {}

    def add(self, label, seconds):
        self.stages.setdefault(label, []).append(seconds)

    def summary(self, total_epochs_planned=None, gpu_speedup_range=(10, 40),
                out_path=None):
        print("\n============== TIMING SUMMARY (CPU) ==============")
        per_stage_mean = {}
        for label, times in self.stages.items():
            mean = sum(times) / len(times)
            per_stage_mean[label] = mean
            print(f"  {label:<28s} mean {mean:8.2f} s   "
                  f"(n={len(times)}, total {sum(times):.1f} s)")

        # The key extrapolation: epoch time -> full-run GPU estimate.
        epoch_key = next((k for k in per_stage_mean if 'epoch' in k.lower()), None)
        if epoch_key and total_epochs_planned:
            cpu_epoch = per_stage_mean[epoch_key]
            cpu_full  = cpu_epoch * total_epochs_planned
            print("\n  --- Extrapolation to a full run ---")
            print(f"  CPU time per epoch      : {cpu_epoch:.1f} s")
            print(f"  Planned epochs          : {total_epochs_planned}")
            print(f"  Full run on CPU (est)   : {cpu_full/3600:.1f} hours "
                  f"({cpu_full/86400:.1f} days)")
            lo, hi = gpu_speedup_range
            print(f"  Full run on GPU (est)   : "
                  f"{cpu_full/hi/3600:.1f} - {cpu_full/lo/3600:.1f} hours "
                  f"(assuming {lo}-{hi}x GPU speedup)")

        print("==================================================")

        return per_stage_mean