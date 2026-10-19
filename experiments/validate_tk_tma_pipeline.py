"""Validate standard TK BF16/FP8 TMA tiles and the DISM WASP ring on SM120a.

Run with --sanitize for memcheck, synccheck, initcheck and racecheck. This is a
correctness test, not a GEMM benchmark or a register-redistribution experiment.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/tk_tma_pipeline')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    binary = out / 'test'
    source = ROOT / 'tests/cuda/test_tk_tma_pipeline.cu'
    flags = ['-std=c++20', '-gencode=arch=compute_120a,code=sm_120a',
             '-DKITTENS_RTX_BLACKWELL', '-Ithird_party/thunderkittens/include',
             '--expt-relaxed-constexpr', '--extended-lambda', '-O3', '-lineinfo']

    def run(name, command, timeout=180):
        print(f'Running {name}...', flush=True)
        with (out / f'{name}.log').open('w') as log:
            proc = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout)
        text = (out / f'{name}.log').read_text()
        if proc.returncode:
            raise RuntimeError(f'{name} failed ({proc.returncode}):\n{text[-8000:]}')
        return text

    build = run('build', [args.cuda / 'bin/nvcc', *flags, '-Xptxas=-v', source,
                          '-lcuda', '-o', binary])
    numerical = run('numerical', [binary])
    match = re.search(r'PASS ALL: (\d+) cases, (\d+) values', numerical)
    assert match and int(match[1]) == 112, numerical
    print(match[0], flush=True)
    run('ptx_build', [args.cuda / 'bin/nvcc', *flags, '-ptx', source,
                      '-o', out / 'kernel.ptx'])
    sass = run('sass', [args.cuda / 'bin/cuobjdump', '--dump-sass', binary])
    counts = {}
    for section in sass.split('Function : ')[1:]:
        name = section.splitlines()[0]
        ops = Counter(re.findall(
            r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)', section))
        assert any(op.startswith('UTMALDG') for op in ops), name
        assert any(op.startswith('LDSM') for op in ops), name
        assert any(op.startswith('SYNCS') for op in ops), name
        assert not any(op.startswith(('CALL', 'LDL', 'STL')) for op in ops), name
        if 'mixed_pipe' in name:
            assert any(op.startswith('QMMA') and '.SF' in op for op in ops), name
        counts[name] = dict(ops)
    assert len(counts) == 22, len(counts)
    spills = re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', build)
    assert len(spills) == 22 and all(int(x) == 0 for row in spills for x in row), spills
    sanitizers = {}
    if args.sanitize:
        for tool in ['memcheck', 'synccheck', 'initcheck', 'racecheck']:
            log = run(tool, [args.cuda / 'bin/compute-sanitizer', '--tool', tool,
                             '--error-exitcode', '1', binary, '--quick'])
            summary = ('RACECHECK SUMMARY: 0 hazards displayed' if tool == 'racecheck'
                       else 'ERROR SUMMARY: 0 errors')
            assert summary in log, log[-8000:]
            assert 'PASS ALL: 22 cases,' in log
            sanitizers[tool] = summary
    headers = sorted((ROOT / 'third_party/thunderkittens/include').rglob('*.cuh'))
    files = [source, ROOT / 'tests/cuda/tma_ring_pipe.cuh', ROOT / 'cuda/tma_ring_pipe.cuh', Path(__file__).resolve(), *headers]
    result = {
        'nvcc': subprocess.check_output([str(args.cuda / 'bin/nvcc'), '--version'], text=True),
        'gpu': numerical.splitlines()[0], 'flags': flags,
        'cases': int(match[1]), 'values': int(match[2]), 'mismatches': 0,
        'coverage': {
            'formats': ['bf16', 'e4m3', 'e5m2'], 'stages': [1, 2, 3],
            'consumer_warps': [4, 8], 'producer_warps': '4 allocated, first warp active',
            'swizzle_bytes': [32, 64, 128], 'steps': [0, 1, 2, 3, 7, 17, 33],
            'tasks_per_cta': '[steps, 0, 1, steps + 2], phases retained across tasks',
            'loads': 'standard gl<T,-1,-1,-1,-1,st<T,R,C>>, no custom traits',
            'bounds': 'row tails, panel-aligned column tails, full OOB tiles; padded strides; batch/depth indexing',
            'stress': 'producer and consumer nanosleeps; 6 CTAs; ring wraps and early buffer release',
            'mixed': 'BF16 A + FP8 B share a completion barrier; A online quantization then native block-scaled MMA',
            'oracle': 'bitwise copy; independent host exponent/quantization/dot-product reference with exact dyadic sums',
        },
        'sanitizers': sanitizers, 'sanitizer_cases_per_tool': 22 if args.sanitize else 0,
        'sass_static_instruction_counts': counts, 'stack_spill_bytes': 0,
        'limitations': [
            'No throughput measurement or setmaxnreg redistribution.',
            'Standard swizzled descriptors round columns to panels; sub-panel tails need physical padding.',
        ],
        'source_sha256': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {out / "result.json"}')


if __name__ == '__main__':
    main()
