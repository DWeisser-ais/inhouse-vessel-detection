"""
 
METRICS EXPLAINED:
 
  Parameters  : number of learnable weights. Determines STORAGE/MEMORY the
                model needs. At float32 (4 bytes/param), size_MB = params*4/1e6.
                Satellites have tiny memory budgets, so this decides whether
                the model even fits.
 
  FLOPs       : floating-point operations for ONE forward pass on ONE tile.
                The core compute-cost number. Lower = faster + less power.
                Reported as GFLOPs (billions). (Strictly we count multiply-adds
                and report both MACs and FLOPs = 2*MACs.)
 
  Activation  : peak intermediate-feature memory during a forward pass. Often
  memory       the REAL onboard bottleneck — the weights fit but the runtime
                buffers don't. Estimated from the largest feature maps.
 
  Latency     : measured wall-clock time for one forward pass (ms). The
                operational number: can it keep up with data collection?
                Measured on THIS machine (CPU here); onboard hardware differs,
                but relative latency between model variants is still informative.
 
  Throughput  : tiles per second = 1000 / latency_ms. Derived operational rate.
"""
 
import time
import torch
import torch.nn as nn



 
 
def count_parameters(model):

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / 1e6           # float32
    return {'params_total': total, 'params_trainable': trainable,
            'model_size_MB_fp32': size_mb,
            'model_size_MB_int8': total * 1 / 1e6}   # if quantised to int8



 
 
def count_flops(model, input_shape=(1, 1, 512, 512)):
    """
    Estimate FLOPs by hooking every Conv2d / Linear and summing multiply-adds.
    Returns MACs and FLOPs (=2*MACs). This is an approximation (covers the
    dominant conv/linear cost; ignores norms/activations, which are minor).
    """
    macs = [0]
    handles = []
 
    def conv_hook(module, inp, out):
        # MACs = out_elements * (in_channels/groups) * kernel_area
        out_dims = out.shape[2:]
        kernel = module.kernel_size
        in_ch = module.in_channels // module.groups
        n_out = out.shape[1] * torch.prod(torch.tensor(out_dims)).item()
        macs[0] += n_out * in_ch * kernel[0] * kernel[1]
 
    def linear_hook(module, inp, out):
        macs[0] += module.in_features * module.out_features * \
                   (out.numel() // out.shape[-1])
 
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
 
    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_shape))
 
    for h in handles:
        h.remove()
 
    return {'MACs': macs[0], 'FLOPs': 2 * macs[0],
            'GFLOPs': 2 * macs[0] / 1e9}




 
def measure_latency(model, input_shape=(1, 1, 512, 512),
                    warmup=3, runs=10, device='cpu'):
    """Median wall-clock forward-pass time (ms) and throughput (tiles/s)."""
    model.eval().to(device)
    x = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    median_ms = times[len(times)//2]
    return {'latency_ms_median': median_ms,
            'throughput_tiles_per_s': 1000.0 / median_ms}
 



 
def estimate_activation_memory(model, input_shape=(1, 1, 512, 512)):
    """
    Rough peak activation memory (MB) = sum of all forward output tensors
    at float32. Over-estimates true peak (not all coexist) but gives an
    order-of-magnitude sense of the runtime buffer requirement.
    """
    total_elems = [0]
    handles = []
 
    def hook(m, i, o):
        if isinstance(o, torch.Tensor):
            total_elems[0] += o.numel()
 
    for m in model.modules():
        if len(list(m.children())) == 0:      # leaf modules only
            handles.append(m.register_forward_hook(hook))
 
    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_shape))
    for h in handles:
        h.remove()
 
    return {'activation_MB_fp32': total_elems[0] * 4 / 1e6}
 



 
def full_report(model, input_shape=(1, 1, 512, 512), device='cpu'):
    """Run every measurement and return a combined dict + printed summary."""

    rep = {}
    rep.update(count_parameters(model))
    rep.update(count_flops(model, input_shape))
    rep.update(estimate_activation_memory(model, input_shape))
    rep.update(measure_latency(model, input_shape, device=device))
 
    print("\n========== DEPLOYMENT / EFFICIENCY REPORT ==========")
    print(f"  Input tile             : {input_shape}")
    print(f"  Parameters (total)     : {rep['params_total']:,}")
    print(f"  Model size (fp32)      : {rep['model_size_MB_fp32']:.1f} MB")
    print(f"  Model size (int8 quant): {rep['model_size_MB_int8']:.1f} MB")
    print(f"  Compute per tile       : {rep['GFLOPs']:.2f} GFLOPs")
    print(f"  Peak activations (est) : {rep['activation_MB_fp32']:.1f} MB")
    print(f"  Latency (median)       : {rep['latency_ms_median']:.1f} ms/tile")
    print(f"  Throughput             : {rep['throughput_tiles_per_s']:.1f} tiles/s")
    print("====================================================")
    return rep
 



 
if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from model.mtu_net import MTUNet
    m = MTUNet(in_channels=1)
    full_report(m)







# --------------------------------------------------------------------- #
#   Adjust to your actual target hardware.
#
#   model size   : how much non-volatile storage the weights need.
#   activations  : runtime RAM for intermediate feature maps (often the
#                  true bottleneck onboard).
#   compute/tile : GFLOPs the processor must do per tile; combined with the
#                  processor's GFLOP/s this sets throughput.


EDGE_BUDGETS = {
    'model_size_MB':    10.0,    # weights should fit in ~10 MB (post-quant)
    'activation_MB':    50.0,    # runtime buffers under ~50 MB
    'gflops_per_tile':  5.0,     # under ~5 GFLOPs/tile for reasonable rate
}
 



 
def edge_ai_assessment(report, budgets=None):
    """
    Compare a measured deployment `report` (from full_report) against edge
    budgets and print how far each metric is from target — i.e. the trim
    factor you need to reach deployability.
    """
    b = budgets or EDGE_BUDGETS
 
    # Use the int8 (quantised) size as the realistic deployment size, since
    # quantisation is the first and cheapest compression step.
    size_now  = report['model_size_MB_int8']
    act_now   = report['activation_MB_fp32']
    flops_now = report['GFLOPs']
 
    print("\n======== EDGE-AI DEPLOYABILITY ASSESSMENT ========")
 
    def line(name, now, budget, unit):
        ratio = now / budget
        verdict = "WITHIN budget" if ratio <= 1 else f"{ratio:.1f}x OVER — trim ~{ratio:.1f}x"
        print(f"  {name:<22s} {now:8.1f} {unit}  (budget {budget:.0f} {unit})  -> {verdict}")
        return ratio
 
    r1 = line("Model size (int8)", size_now,  b['model_size_MB'],   'MB')
    r2 = line("Peak activations",  act_now,   b['activation_MB'],   'MB')
    r3 = line("Compute per tile",  flops_now, b['gflops_per_tile'], 'GFLOPs')
 
    worst = max(r1, r2, r3)
    print(f"\n  Overall: the binding constraint is "
          f"{'activations' if r2==worst else 'model size' if r1==worst else 'compute'}, "
          f"~{worst:.1f}x over budget.")
    print("====================================================")
 
    return {'size_ratio': r1, 'activation_ratio': r2, 'flops_ratio': r3,
            'worst_ratio': worst}