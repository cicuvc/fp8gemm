"""Validate and sweep R/C tiles for the SM120a persistent online-FP8 GEMM."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.tk_persistent_gemm import CONFIGS, FLAGS, GemmLibrary, build


def quantized_a(a):
    """Independent tensor implementation of the existing finite BF16 bit rule."""
    x = a.reshape(a.shape[0], -1, 32)
    exponent_bits = (x.view(torch.int16).to(torch.int32) & 0x7f80).amax(-1)
    reciprocal = ((0x7f00 - exponent_bits) & 0x7f80).to(torch.int16).view(torch.bfloat16)
    q = (x.float() * reciprocal.float()[..., None]).bfloat16().to(torch.float8_e4m3fn).float()
    scale = torch.ldexp(torch.ones_like(exponent_bits, dtype=torch.float32), (exponent_bits >> 7) - 127)
    return (q * scale[..., None]).reshape_as(a)


def exact_inputs(m, n, k):
    i = torch.arange(m*k, dtype=torch.int64).reshape(m, k)
    a = torch.ldexp(((i*73+17)%255-127).float()/64, ((i//k+i//32)%5-2).int()).bfloat16()
    j = torch.arange(n*k, dtype=torch.int64).reshape(n, k)
    b = (((j*29+11)%17-8).float()/4).to(torch.float8_e4m3fn)
    return a, b


def validate(lib, smoke=False):
    rows = []
    shapes = [(1031, 192, 192)] if smoke else [(1, 64, 64), (137, 192, 64), (777, 320, 320), (1031, 192, 192)]
    for m, n, k in shapes:
        ac, bc = exact_inputs(m, n, k)
        ref = (quantized_a(ac).double() @ bc.double().T).bfloat16()
        a, b = ac.cuda(), bc.cuda()
        for config in range(len(CONFIGS)):
            out = torch.full((m, n), float('nan'), dtype=torch.bfloat16, device='cuda')
            p = lib.plan(a, b, config, out=out, sms=3)
            got = p().cpu()
            mismatches = (got.view(torch.int16) != ref.view(torch.int16)).sum().item()
            assert mismatches == 0, (config, (m, n, k), mismatches, (got.float()-ref.float()).abs().max().item())
            # Reuse the same descriptors with new data, including whole zero K32 blocks.
            a[:, ::32] = 0
            a[:, 32:64] = 0
            changed_ref = (quantized_a(a.cpu()).double() @ bc.double().T).bfloat16()
            got2 = p().cpu()
            assert torch.equal(got2, changed_ref), (config, 'reused plan')
            a.copy_(ac)
            rows.append({'shape': [m, n, k], 'config': config, **p.info,
                         'mismatches': mismatches, 'values': 2*m*n})
        print(f'PASS exact/reuse M,N,K={m,n,k}, all {len(CONFIGS)} configurations', flush=True)
    return rows


def graph_benchmark(functions, rounds=5, repeats=32):
    graphs = {}
    # Capture pre-created plans: descriptor creation, allocation, compilation
    # and Python launch overhead are excluded for every implementation.
    for name, fn in functions.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                fn()
        graphs[name] = graph
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    timings = {name: [] for name in graphs}
    names = list(graphs)
    for rnd in range(rounds):
        for name in names[rnd % len(names):] + names[:rnd % len(names)]:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end)*1000/repeats)
    return {name: {'median_us': statistics.median(v), 'min_us': min(v), 'max_us': max(v), 'round_us': v}
            for name, v in timings.items()}


def assembly(output, cuda):
    binary = output / 'libgemm.so'
    sass = subprocess.check_output([str(cuda/'bin/cuobjdump'), '--dump-sass', str(binary)], text=True)
    (output/'sass.log').write_text(sass)
    with (output/'ptx_build.log').open('w') as log:
        subprocess.run([str(cuda/'bin/nvcc'), *FLAGS, '-ptx', 'cuda/persistent_fp8_gemm.cu',
                        '-o', str(output/'kernel.ptx')], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = {}
    for section in sass.split('Function : ')[1:]:
        name = section.splitlines()[0]
        ops = Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)', section))
        for prefix in ['UTMALDG', 'UTMASTG', 'LDSM', 'STSM', 'QMMA.SF', 'USETMAXREG']:
            assert any(op.startswith(prefix) for op in ops), (name, prefix)
        assert not any(op.startswith(('LDG', 'STG', 'LDL', 'STL', 'CALL')) for op in ops), name
        # Check the generated physical B pairs: both words must come directly
        # from one LDSM destination tuple, with no intervening register MOV.
        # This is a lexical def-use audit of the emitted unrolled instruction
        # sequence, not a general SASS control-flow analyzer.
        definitions, qmma_count, direct_pairs = {}, 0, 0
        for line in section.splitlines():
            match = re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)\s+([^;]+);', line)
            if not match:
                continue
            pc, op, body = match.groups()
            operands = body.split(',')
            if op.startswith('QMMA'):
                b = int(operands[2].strip()[1:])
                origins = [definitions.get(b), definitions.get(b+1)]
                qmma_count += 1
                if all(o and o[0] == 'LDSM' for o in origins) and origins[0][1] == origins[1][1]:
                    direct_pairs += 1
            if re.fullmatch(r'R\d+', operands[0].strip()):
                base = int(operands[0].strip()[1:])
                count = 4 if op.startswith(('LDSM', 'QMMA')) else 2 if '.64' in op else 1
                for i in range(count):
                    definitions[base+i] = ('LDSM' if op.startswith('LDSM') else op, pc)
        assert direct_pairs == qmma_count > 0, (name, direct_pairs, qmma_count)
        result[name] = {'opcodes': dict(ops), 'qmma_instructions': qmma_count,
                        'b_pairs_direct_from_ldsm': direct_pairs}
    assert len(result) == len(CONFIGS)
    return result


def sanitize(args):
    summaries = {}
    for tool in ['memcheck', 'synccheck', 'initcheck', 'racecheck']:
        print(f'Running {tool}...', flush=True)
        with (args.output/f'{tool}.log').open('w') as log:
            subprocess.run([str(args.cuda/'bin/compute-sanitizer'), '--tool', tool, '--error-exitcode', '1',
                            sys.executable, str(Path(__file__).resolve()), '--skip-build', '--validate-only',
                            '--smoke', '--output', str(args.output)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
        text = (args.output/f'{tool}.log').read_text()
        summary = ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool == 'racecheck'
                   else 'ERROR SUMMARY: 0 errors')
        assert summary in text and 'all 10 configurations' in text, text[-4000:]
        summaries[tool] = summary
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/tk_persistent_gemm')
    parser.add_argument('--cuda', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=32)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if not args.skip_build:
        build(args.output, args.cuda)
    lib = GemmLibrary(args.output)
    validation = validate(lib, args.smoke)
    if args.validate_only:
        (args.output/('smoke.json' if args.smoke else 'validation.json')).write_text(json.dumps(validation, indent=2)+'\n')
        if args.sanitize:
            (args.output/'sanitizers.json').write_text(json.dumps(sanitize(args), indent=2)+'\n')
        return
    result = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'flags': FLAGS, 'configs': CONFIGS, 'validation': validation,
              'rounds': args.rounds, 'launches_per_graph': args.repeats,
              'timing': 'CUDA graphs; 32 launches/graph by default; rotated configuration order; warm resident inputs; no locked clocks',
              'peak_tflops': 419.0, 'sass': assembly(args.output, args.cuda), 'benchmarks': []}
    import mxtest
    shapes = [(256, 2048, 512), (1024, 8192, 1024), (2048, 16384, 2048),
              (4096, 4096, 4096), (8192, 8192, 4096)]
    for m, n, k in shapes:
        torch.manual_seed(123)
        a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
        b = torch.randn((n, k), device='cuda').to(torch.float8_e4m3fn)
        ref = (quantized_a(a) @ b.float().T).bfloat16()
        unquantized_ref = (a.float() @ b.float().T).bfloat16()
        plans = {f'cuda_{i}': lib.plan(a, b, i) for i in range(len(CONFIGS))}
        errors = {}
        for name, p in plans.items():
            got = p()
            diff = got.float()-ref.float()
            rel = (diff.norm()/ref.float().norm()).item()
            assert torch.isfinite(got).all() and rel < 5e-4, (name, (m, n, k), rel)
            errors[name] = {'relative_l2_vs_quantized_ref': rel, 'max_abs_vs_quantized_ref': diff.abs().max().item(),
                            'relative_l2_vs_unquantized_a': ((got.float()-unquantized_ref.float()).norm()/unquantized_ref.float().norm()).item()}
        scales = torch.full(((n+31)//32, k//32), 127, dtype=torch.uint8, device='cuda')
        triton_fn = lambda: mxtest.launch_fp8gemm(a, b, scales)
        triton_result = triton_fn()  # autotune outside timing
        assert torch.allclose(triton_result.float(), ref.float(), atol=.1, rtol=.01)
        times = graph_benchmark({**plans, 'triton': triton_fn}, args.rounds, args.repeats)
        for name, row in times.items():
            row['tflops'] = 2*m*n*k/(row['median_us']*1e6)
            row['mfu_pct'] = row['tflops']/419*100
            if name in plans:
                row.update(plans[name].info)
                row.update(errors[name])
        best = min(plans, key=lambda name: times[name]['median_us'])
        result['benchmarks'].append({'shape_mnk': [m, n, k], 'best_cuda': best, 'timings': times})
        print(f'{m,n,k}: best {best} {times[best]["median_us"]:.2f} us, '
              f'{times[best]["tflops"]:.2f} TFLOP/s; Triton {times["triton"]["median_us"]:.2f} us', flush=True)
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        del plans, a, b, ref, unquantized_ref
    files = [ROOT/'cuda/persistent_fp8_gemm.cu', ROOT/'cuda/tma_ring_pipe.cuh', Path(__file__).resolve(),
             ROOT/'experiments/tk_persistent_gemm.py', *sorted((ROOT/'third_party/thunderkittens/include').rglob('*.cuh'))]
    result['source_sha256'] = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    if args.sanitize:
        result['sanitizers'] = sanitize(args)
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
