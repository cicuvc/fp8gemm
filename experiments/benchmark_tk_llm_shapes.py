"""Local online-A FP8 forward throughput for large-M LLM shapes.

python experiments/benchmark_tk_llm_shapes.py
"""
import argparse
import csv
import gc
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mxtest
from experiments.tk_persistent_gemm import GemmLibrary, build, DEFAULT_OUTPUT, FLAGS
from experiments.benchmark_tk_forward_pipeline import measure
from experiments.benchmark_tk_persistent_gemm import quantized_a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m', type=int, nargs='+', default=[4096, 8192, 16384, 32768])
    parser.add_argument('--nk', type=int, nargs='+', default=[1024, 2048, 4096])
    parser.add_argument('--rounds', type=int, default=11)
    parser.add_argument('--repeats', type=int, default=64)
    parser.add_argument('--peak-tflops', type=float, default=419)
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'benchmarks/tk_llm_shapes_rtx5090.json')
    args = parser.parse_args()
    if min(args.m + args.nk) < 256 or any(v % 128 for v in args.m + args.nk):
        parser.error('This sweep uses dimensions >=256 divisible by 128')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_capability() == (12, 0)
    if not args.skip_build:
        build()
    lib = GemmLibrary()
    props = torch.cuda.get_device_properties(0)
    shapes = list(itertools.product(args.m, args.nk, args.nk))
    random.Random(123).shuffle(shapes)
    sources = ['cuda/persistent_fp8_gemm.cu', 'cuda/tma_ring_pipe.cuh',
               'experiments/tk_persistent_gemm.py', 'experiments/benchmark_tk_llm_shapes.py',
               'experiments/benchmark_tk_forward_pipeline.py', 'experiments/benchmark_tk_persistent_gemm.py',
               'mxtest.py']
    sources += [str(p.relative_to(ROOT)) for p in (ROOT/'third_party/thunderkittens/include').rglob('*.cuh')]
    result = {
        'gpu': props.name, 'sm_count': props.multi_processor_count,
        'torch': torch.__version__, 'peak_tflops': args.peak_tflops,
        'rounds': args.rounds, 'launches_per_graph': args.repeats,
        'config': {'id': 8, 'warp_tile': [16, 128], 'cta_tile': [128, 128], 'bk': 64,
                   'shared_stages': 2, 'raster_group': 1, 'prefetch': False},
        'method': 'Current default forward, BF16 A quantized online, FP8 E4M3 B with 32x32 E8M0 scales, BF16 output. '
                  'B quantization and allocations excluded. Warm CUDA graphs, median of per-graph CUDA-event timings, '
                  '0.5 second GPU warmup per shape, unlocked clocks, shuffled shape order. TFLOPS=2*M*N*K/time.',
        'shape_order_mnk': shapes, 'flags': FLAGS,
        'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'binary_sha256': hashlib.sha256((DEFAULT_OUTPUT/'libgemm.so').read_bytes()).hexdigest(),
        'nvidia_smi_before': subprocess.check_output(['nvidia-smi'], text=True), 'shapes': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        fields = ['M', 'N', 'K', 'median_us', 'min_us', 'max_us', 'tflops', 'mfu_percent',
                  'output_tiles', 'cta_waves', 'sample_relative_l2']
        with args.output.with_suffix('.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in sorted(result['shapes'], key=lambda v: (v['N'], v['K'], v['M'])):
                writer.writerow({k: r[k] for k in fields})

    for index, (m, n, k) in enumerate(shapes, 1):
        torch.manual_seed(123)
        a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        b, scales = mxtest.launch_weight_quantize(torch.randn((n, k), device='cuda'))
        out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        plan = lib.plan(a, b, 8, out=out, scales=scales, raster_group=1, prefetch=False)
        plan()
        assert torch.isfinite(out).all()
        # Sample across persistent CTA iterations and warp/CTA row boundaries.
        rows = torch.unique(torch.cat((torch.linspace(0, m-1, 96, device='cuda').long(),
                                       torch.tensor([15, 16, 127, 128, 255, 256], device='cuda'))))
        rows = rows[rows < m]
        reference = quantized_a(a[rows]) @ mxtest.dequantize_weight(b, scales).T
        rel = ((out[rows].float()-reference).norm()/reference.norm()).item()
        assert rel < .004, ((m, n, k), rel)
        del reference, rows
        timing = measure({'forward': plan}, 'forward', args.rounds, args.repeats, 123)['forward']
        tiles = (m//128)*(n//128)
        row = {'M': m, 'N': n, 'K': k, **timing,
               'tflops': 2*m*n*k/(timing['median_us']*1e6),
               'output_tiles': tiles, 'cta_waves': (tiles+props.multi_processor_count-1)//props.multi_processor_count,
               'sample_relative_l2': rel, 'all_finite': True, 'plan': plan.info}
        row['mfu_percent'] = 100*row['tflops']/args.peak_tflops
        result['shapes'].append(row)
        save()
        print(f'[{index:02d}/{len(shapes)}] M={m:5d} N={n:4d} K={k:4d}: '
              f'{row["median_us"]:9.3f} us, {row["tflops"]:7.2f} TFLOPS, '
              f'MFU {row["mfu_percent"]:5.1f}%', flush=True)
        del plan, a, b, scales, out
        gc.collect()
        torch.cuda.empty_cache()
    result['nvidia_smi_after'] = subprocess.check_output(['nvidia-smi'], text=True)
    save()


if __name__ == '__main__':
    main()
