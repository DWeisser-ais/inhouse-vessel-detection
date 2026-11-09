"""
deployment_metrics.py
=====================
Measures how big the model is now. It only measures: nothing here changes the model.

    python deployment_metrics.py     (from Scripts/model/)

    Parameters     number of learnable weights
    Weights        storage for those weights: 32-bit float today, and an
                   estimate for 8-bit integer once quantised
    MACs / FLOPs   compute for one 512 x 512 tile (FLOPs = 2 x MACs)
    Peak RAM       largest amount of tensor memory alive at once during one
                   forward pass, excluding the weights
    Latency        measured forward-pass time on this machine
"""

import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn


def count_size(model):
    params = sum(p.numel() for p in model.parameters())
    bytes_fp32 = sum(t.numel() * t.element_size()
                     for t in list(model.parameters()) + list(model.buffers()))
    return {'parameters': params,
            'weights_MB_fp32': bytes_fp32 / 1e6,
            'weights_MB_int8_estimate': params / 1e6}


def _module_macs(module, inputs, output):
    if isinstance(module, nn.Conv2d):
        kernel = module.kernel_size[0] * module.kernel_size[1]
        return output.numel() * (module.in_channels // module.groups) * kernel
    if isinstance(module, nn.Linear):
        return module.in_features * output.numel()
    if isinstance(module, nn.MultiheadAttention):
        # Attention does its own matrix products, invisible to the Linear
        # hooks: four projections plus the score and value multiplications.
        batch, tokens, dim = inputs[0].shape
        return batch * (4 * tokens * dim * dim + 2 * tokens * tokens * dim)
    return 0


def count_macs(model, input_shape):
    """Total multiply-accumulate operations in one forward pass."""
    macs, handles = [], []
    attention = [m for m in model.modules() if isinstance(m, nn.MultiheadAttention)]
    inside_attention = {id(c) for a in attention for c in a.modules() if c is not a}

    def hook(module, inputs, output):
        macs.append(_module_macs(module, inputs, output))

    for module in model.modules():
        is_leaf = not list(module.children()) or isinstance(module, nn.MultiheadAttention)
        if is_leaf and id(module) not in inside_attention:
            handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(torch.zeros(input_shape))
    for handle in handles:
        handle.remove()
    return sum(macs)


def measure_peak_ram_mb(model, input_shape):
    """Peak tensor memory (MB) in one CPU forward pass, or None if unavailable."""
    x = torch.zeros(input_shape)
    try:
        with torch.autograd.profiler.profile(profile_memory=True) as prof:
            with torch.no_grad():
                model(x)
        events = sorted((e.start_ns(), e.nbytes()) for e in prof.kineto_results.events()
                        if e.name() == '[memory]')
    except (AttributeError, RuntimeError) as err:
        print(f"  (peak RAM could not be measured: {err})")
        return None
    alive = peak = 0
    for _, nbytes in events:
        alive += nbytes
        peak = max(peak, alive)
    return peak / 1e6


def measure_latency_ms(model, input_shape, device, warmup=3, runs=10):
    x = torch.zeros(input_shape, device=device)
    times = []
    with torch.no_grad():
        for i in range(warmup + runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[len(times) // 2]


def model_size_report(model, input_shape=(1, 1, 512, 512), device='cpu', out_path=None):
    """Measure a copy of the model, print the report and optionally save it."""
    cpu_model = copy.deepcopy(model).cpu().eval()
    macs = count_macs(cpu_model, input_shape)

    device_model = copy.deepcopy(model).to(device).eval()
    latency = measure_latency_ms(device_model, input_shape, device)

    rep = {'input_shape': list(input_shape), **count_size(cpu_model),
           'MACs': macs, 'GFLOPs': 2 * macs / 1e9,
           'peak_ram_MB_fp32': measure_peak_ram_mb(cpu_model, input_shape),
           'latency_ms': latency, 'latency_device': device,
           'throughput_tiles_per_s': 1000 / latency}

    int8_mb = rep['weights_MB_int8_estimate']
    peak = rep['peak_ram_MB_fp32']

    print("\n================= MODEL SIZE =================")
    print(f"  Input                 : {tuple(input_shape)}")
    print(f"  Parameters            : {rep['parameters']:,}")
    print(f"  Weights (float32)     : {rep['weights_MB_fp32']:.1f} MB")
    print(f"  Weights (int8, est.)  : {int8_mb:.1f} MB")
    print(f"  Compute per tile      : {macs / 1e9:.2f} GMACs ({rep['GFLOPs']:.2f} GFLOPs)")
    if peak is not None:
        print(f"  Peak RAM              : {peak:.1f} MB float32, about {peak / 4:.1f} MB int8")
    print(f"  Latency on {device:<11}: {latency:.1f} ms/tile "
          f"({rep['throughput_tiles_per_s']:.1f} tiles/s)")
    print("  (int8 figures are estimates, made before any conversion)")
    print("==============================================")

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(rep, f, indent=2)
        print(f"  Saved -> {out_path}")
    return rep


if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from model.mtu_net import MTUNet
    model_size_report(MTUNet(in_channels=1))
