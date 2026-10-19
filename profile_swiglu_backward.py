"""Attribute the unchanged FP8 SwiGLU backward and benchmark candidate primitives."""
import argparse
import cProfile
import hashlib
import json
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.testing

import mxtest
from benchmark_vs_fla import DEFAULT_CASES, params
from fp8_swiglu import FP8SwiGLU, _fp8_swiglu_fwd, _fp8_swiglu_bwd


def benchmark(fn, rep, repeats, setup=None):
    # Build retained autograd graphs on the capture stream as required by CUDA.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        if setup is not None:
            setup()
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        calls_per_replay = 16
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(calls_per_replay):
                fn()
        begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples = max(5, int(rep / max(begin.elapsed_time(end), 0.001)))
        times = []
        for _ in range(repeats):
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(samples)]
            for start, stop in events:
                start.record()
                graph.replay()
                stop.record()
            torch.cuda.synchronize()
            times.append(statistics.median(start.elapsed_time(stop) / calls_per_replay
                                           for start, stop in events))
    torch.cuda.current_stream().wait_stream(stream)
    return {"median_ms": statistics.median(times), "round_ms": times}


def classify(name):
    for prefix, label in (
        ("fp8gemm_bwd_swiglu_act", "down_dx_activation"),
        ("fp8_swiglu_bwd_dx", "gate_up_dx"),
        ("bf16_wgrad_dual", "gate_up_dw"),
        ("bf16_wgrad", "down_dw"),
    ):
        if name.startswith(prefix):
            return label
    if "fill" in name.lower():
        return "autograd_zero_fill"
    if "reduce" in name.lower():
        return "bias_reduce"
    if "copy" in name.lower():
        return "bias_cast"
    return "other"


def run_case(m, k, h, args):
    torch.manual_seed(23)
    layer = FP8SwiGLU(k, h, k, bias=False, device="cuda")
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    gw, gs = layer.quantized_gate_weight
    uw, us = layer.quantized_up_weight
    dw, ds = layer.quantized_down_weight
    with torch.no_grad():
        _, gate, up, hidden = _fp8_swiglu_fwd(
            x, *params(layer), gw, gs, uw, us, dw, ds, None, False)
    inputs = (dy, hidden, gate, up, dw, ds, gw, gs, uw, us, x)
    ref = mxtest.launch_fp8_down_swiglu_backward(*inputs, return_intermediate_grads=True)
    dg, du = torch.empty_like(gate), torch.empty_like(up)
    dx = torch.empty_like(x)

    def down_dx_activation():
        mxtest.fp8gemm_bwd_swiglu_act[
            lambda meta: (triton.cdiv(m, meta['BLOCK_M']) * triton.cdiv(h, meta['BLOCK_K']),)
        ](dy, dw, ds, gate, up, dg, du, m, k, h,
          dy.stride(0), dw.stride(0), ds.stride(0), gate.stride(0), up.stride(0), dg.stride(0), du.stride(0))

    def gate_up_dx():
        mxtest.fp8_swiglu_bwd_dx[
            lambda meta: (triton.cdiv(m, meta['BLOCK_M']) * triton.cdiv(k, meta['BLOCK_K']),)
        ](dg, du, gw, gs, uw, us, dx, m, h, k,
          dg.stride(0), du.stride(0), gw.stride(0), gs.stride(0), uw.stride(0), us.stride(0), dx.stride(0))

    down_dx_activation()
    gate_up_dx()
    for actual, expected in ((dg, ref[4]), (du, ref[5]), (dx, ref[0])):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    del ref
    stages = {
        "down_dx_activation": down_dx_activation,
        "gate_up_dx": gate_up_dx,
        "gate_up_dw": lambda: mxtest.launch_bf16_weight_grad_dual(dg, du, x),
        "down_dw": lambda: mxtest.launch_bf16_weight_grad(dy, hidden),
        "unused_bias": lambda: dy.float().sum(0),
    }
    y = layer(x)

    def autograd_backward():
        return torch.autograd.grad(y, (x,) + params(layer), dy, retain_graph=True)

    def setup_autograd():
        nonlocal y
        del y
        y = layer(x)

    def custom_backward():
        return _fp8_swiglu_bwd(dy, x, gate, up, hidden, gw, gs, uw, us, dw, ds, False, False)

    def raw_backward():
        return mxtest.launch_fp8_down_swiglu_backward(*inputs)

    def raw_with_bias():
        result = raw_backward()
        return *result, dy.float().sum(0)

    # Same inputs and FP32 outputs for vendor wgrad comparisons.
    candidates = {
        "gate_up_dw_separate_triton": lambda: (
            mxtest.launch_bf16_weight_grad(dg, x), mxtest.launch_bf16_weight_grad(du, x)),
        "gate_up_dw_cublas_fp32": lambda: (
            torch.mm(dg.T, x, out_dtype=torch.float32), torch.mm(du.T, x, out_dtype=torch.float32)),
        "down_dw_cublas_fp32": lambda: torch.mm(dy.T, hidden, out_dtype=torch.float32),
    }
    with torch.no_grad():
        expected = stages['gate_up_dw']()
        actual = candidates['gate_up_dw_cublas_fp32']()
        dw_expected = stages['down_dw']()
        dw_actual = candidates['down_dw_cublas_fp32']()
        candidate_errors = {}
        for label, a, b in zip(('dgate', 'dup', 'ddown'), (*actual, dw_actual), (*expected, dw_expected)):
            candidate_errors[label] = ((a-b).norm()/b.norm()).item()
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
        del expected, actual, dw_expected, dw_actual

    functions = {**stages, "raw_backward_no_bias": raw_backward,
                 "raw_backward_with_bias": raw_with_bias, "custom_op_backward": custom_backward,
                 "autograd_backward": autograd_backward, **candidates}
    timings = {}
    for name, fn in functions.items():
        with torch.no_grad() if name not in ('autograd_backward',) else torch.enable_grad():
            timings[name] = benchmark(fn, args.rep, args.repeats,
                                      setup_autograd if name == "autograd_backward" else None)
        print(f'{m} {k} {h} {name}: {timings[name]["median_ms"]*1000:.2f} us', flush=True)

    del y
    y = layer(x)  # Restore the default stream for the eager profiler.

    # Attribute actual kernels in the complete autograd backward. No timing is
    # inferred by subtracting forward from a forward+backward measurement.
    for _ in range(5):
        autograd_backward()
    torch.cuda.synchronize()
    trace_path = args.trace_dir / f'backward_{m}_{k}_{h}.json'
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(args.profile_iters):
            autograd_backward()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())
    kernel_totals, stage_totals, counts = defaultdict(float), defaultdict(float), defaultdict(int)
    for event in trace['traceEvents']:
        if event.get('cat') == 'kernel' and event.get('ph') == 'X':
            kernel_totals[event['name']] += event['dur'] / args.profile_iters
            stage_totals[classify(event['name'])] += event['dur'] / args.profile_iters
            counts[event['name']] += 1
    total = sum(stage_totals.values())
    profile = {name: {"us": value, "percent_gpu_kernel_time": value/total*100}
               for name, value in stage_totals.items()}
    print('PROFILE', json.dumps(profile), flush=True)

    # Uninstrumented steady-state batches, with synchronization only at edges.
    # Wall time includes GPU work and host dispatch; do not sum it with GPU time.
    eager_timings = {}
    eager_names = ('raw_backward_with_bias', 'custom_op_backward', 'autograd_backward')
    for name in eager_names:
        wall_ms = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                functions[name]()
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - start) * 10)
        eager_timings[name] = {"median_ms": statistics.median(wall_ms), "round_ms": wall_ms}
    print('EAGER_WALL', json.dumps(eager_timings), flush=True)

    # CPU diagnostic only: cProfile adds overhead. Synchronization is outside
    # the profiled loop; values below are not claimed as wall-clock latency.
    cpu_profiles = {}
    for name in ('autograd_backward', 'custom_op_backward', 'raw_backward_no_bias'):
        cp = cProfile.Profile()
        torch.cuda.synchronize()
        cp.enable()
        for _ in range(args.profile_iters):
            functions[name]()
        cp.disable()
        torch.cuda.synchronize()
        rows = []
        for entry in cp.getstats():
            if isinstance(entry.code, str):
                label = entry.code
            else:
                label = f'{entry.code.co_filename}:{entry.code.co_firstlineno}:{entry.code.co_name}'
            rows.append({'function': label, 'calls': entry.callcount,
                         'self_us_per_backward': entry.inlinetime*1e6/args.profile_iters,
                         'cumulative_us_per_backward': entry.totaltime*1e6/args.profile_iters})
        cpu_profiles[name] = sorted(rows, key=lambda r: r['self_us_per_backward'], reverse=True)[:25]
    configs = {name: str(getattr(getattr(mxtest, name), 'best_config', None)) for name in (
        'fp8gemm_bwd_swiglu_act', 'fp8_swiglu_bwd_dx', 'bf16_wgrad_dual', 'bf16_wgrad')}
    return {'shape': [m, k, h], 'graph_timings': timings, 'actual_backward_gpu_profile': profile,
            'profile_gpu_total_us': total,
            'profile_kernels': [{'name': n, 'us_per_backward': t, 'calls_total': counts[n]}
                                for n,t in kernel_totals.items()],
            'cpu_diagnostic': cpu_profiles, 'eager_wall_timings': eager_timings,
            'configs_last_run': configs,
            'candidate_relative_l2': candidate_errors, 'trace_path': str(trace_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--hidden', type=int)
    parser.add_argument('--rep', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--profile-iters', type=int, default=20)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/swiglu_backward_rtx5090.json'))
    parser.add_argument('--trace-dir', type=Path, default=Path('runs/swiglu_backward_traces'))
    args = parser.parse_args()
    dims = (args.m, args.k, args.hidden)
    if any(v is not None for v in dims) and not all(v is not None and v > 0 for v in dims):
        parser.error('--m, --k, --hidden must be supplied together and positive')
    if min(args.rep, args.repeats, args.profile_iters) <= 0:
        parser.error('timing counts must be positive')
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'environment': {'utc': datetime.now(timezone.utc).isoformat(),
                             'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                             'triton': triton.__version__,
                             'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
                             'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                               for p in ('mxtest.py','fp8_swiglu.py','profile_swiglu_backward.py')}},
              'settings': {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'method': {'graph': 'Standalone hot-cache CUDA graph replay, median of round medians; exact baseline tensors; FP32 dW',
                         'profile': 'Actual autograd backward with retained graph, sum of CUDA kernel durations; excludes CPU and idle gaps; instrumentation affects timings',
                         'eager_wall': 'Uninstrumented batches of 100 backward calls, edge synchronization; GPU plus host dispatch, retained graph, no L2 flush',
                         'cpu': 'cProfile self and cumulative times; diagnostic only, profiler adds overhead',
                         'scope': 'bias=False, fp8_wgrad=False; forward excluded; no optimizer or cache refresh'},
              'cases': []}
    for case in ([dims] if args.m is not None else DEFAULT_CASES):
        result['cases'].append(run_case(*case,args))
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        torch.cuda.empty_cache()
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
