"""Measure standalone B quantization + persistent A-online GEMM on equal inputs.

Uses the existing Triton row/K32 quantization kernel, refreshing B before EVERY
GEMM in the combined path. Separate timings are diagnostic: their sum is not
used as the combined latency because cache state and clocks can differ.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.tk_dual_online_gemm import DualOnlineLibrary, DEFAULT_OUTPUT
from experiments.benchmark_tk_dual_online_gemm import refresh_b
from experiments.benchmark_tk_persistent_gemm import graph_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--output', type=Path, default=ROOT/'benchmarks/tk_prequant_b_rtx5090.json')
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--repeats', type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(4)
    previous = json.loads((ROOT/'benchmarks/tk_dual_online_gemm_rtx5090.json').read_text())
    library = DualOnlineLibrary(args.library)
    files = [Path(__file__), ROOT/'experiments/tk_dual_online_gemm.py',
             ROOT/'experiments/benchmark_tk_dual_online_gemm.py',
             ROOT/'experiments/benchmark_tk_persistent_gemm.py',
             ROOT/'cuda/persistent_dual_online_gemm.cu', ROOT/'mxtest.py']
    result = {
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'quantizer': 'mxtest.fp8_row_quantize: BF16 B[N,K] -> E4M3 Bq[N,K], uint8 E8M0 scales[N,K/32]; BLOCK_N=64 BLOCK_K=64',
        'gemm': 'CUDA WASP persistent AB^T; BF16 A online row/K32 quantization, prequantized FP8 B and per-row scales',
        'timing': 'CUDA graphs; rotated order; warm buffers; preallocated memory; same stream; each combined invocation refreshes B, then GEMM; no locked clocks; no transpose',
        'selection': 'Use best online/cached configs from previous eight-config sweep, separately for each shape and output dtype',
        'rounds': args.rounds, 'repeats': args.repeats,
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'library_sha256': hashlib.sha256((args.library/'libdual.so').read_bytes()).hexdigest(),
        'benchmarks': [],
    }
    result['gpu_before'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,clocks.sm,power.limit', '--format=csv'], text=True)
    for old in previous['benchmarks']:
        m,n,k = old['shape_mnk']
        dtype = torch.float32 if old['dtype']=='torch.float32' else torch.bfloat16
        torch.manual_seed(123)
        a = torch.randn((m,k), device='cuda', dtype=torch.bfloat16)
        b = torch.randn((n,k), device='cuda', dtype=torch.bfloat16)
        q = torch.empty((n,k), device='cuda', dtype=torch.float8_e4m3fn)
        s = torch.empty((n,k//32), device='cuda', dtype=torch.uint8)
        online = library.plan(a,b,int(old['best_online'].split('_')[-1]), output_dtype=dtype)
        cached = library.plan(a,q,int(old['best_cached'].split('_')[-1]), output_dtype=dtype,b_scales=s)
        def quantize(): refresh_b(b,q,s)
        def combined(): quantize(); return cached()
        online()
        combined()
        rel = ((cached.out.float()-online.out.float()).norm()/online.out.float().norm()).item()
        assert rel < (5e-4 if dtype==torch.bfloat16 else 1e-6), (old['shape_mnk'],dtype,rel)
        # Force fresh B data through existing descriptors; a stale B cache
        # would leave the previous nonzero result in this check.
        original_b = b.clone()
        b.zero_()
        combined()
        assert torch.count_nonzero(cached.out).item()==0
        b.copy_(original_b)
        del original_b
        times = graph_benchmark({'quantize_b':quantize,'a_online_gemm':cached,
                                 'quantize_b_then_gemm':combined,'ab_online_gemm':online},
                                rounds=args.rounds,repeats=args.repeats)
        for name,t in times.items():
            if name!='quantize_b': t['effective_tflops']=2*m*n*k/(t['median_us']*1e6)
        total = times['quantize_b_then_gemm']['median_us']
        row = {'shape_mnk':[m,n,k], 'dtype':str(dtype), 'cached_config':cached.info,
               'online_config':online.info, 'relative_l2_vs_dual_online':rel, 'changed_b_refresh_passed':True,
               'timings':times, 'speedup_vs_dual_online':times['ab_online_gemm']['median_us']/total,
               'standalone_quantization_fraction_of_total':times['quantize_b']['median_us']/total}
        result['benchmarks'].append(row)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(f'{m,n,k} {dtype}: Q={times["quantize_b"]["median_us"]:.3f} us, '
              f'G={times["a_online_gemm"]["median_us"]:.3f}, Q->G={total:.3f}, '
              f'dual={times["ab_online_gemm"]["median_us"]:.3f}, speedup={row["speedup_vs_dual_online"]:.3f}x', flush=True)


if __name__=='__main__': main()
