"""Build and validate the SM120 ThunderKittens block-scaled MMA primitives.

python experiments/validate_tk_block_scaled.py --sanitize
Build products, SASS, logs and the numerical summary go under runs/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--output', type=Path, default=ROOT/'runs/tk_block_scale')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    compiler = args.cuda/'bin/nvcc'
    binary = out/'test_tk_block_scaled'
    # Explicit gencode avoids nvcc's -arch=sm_120a generic sm_120 PTX fallback,
    # which cannot represent the architecture-specific block-scale instruction.
    flags = ['-std=c++20', '-gencode=arch=compute_120a,code=sm_120a',
             '-DKITTENS_RTX_BLACKWELL', '-Ithird_party/thunderkittens/include',
             '--expt-relaxed-constexpr', '--extended-lambda', '-O3', '-lineinfo', '-Xptxas=-v']

    def run(label, command):
        print(f'Running {label}', flush=True)
        result = subprocess.run(list(map(str,command)), cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out/f'{label}.log').write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f'{label} failed ({result.returncode}):\n{result.stdout[-6000:]}')
        return result.stdout

    build = run('build', [compiler,*flags,'tests/cuda/test_tk_block_scaled.cu','-o',binary])
    assert not re.search(r'\b[1-9]\d* bytes (?:spill stores|spill loads)',build), 'Unexpected test-kernel spills'
    numerical = run('numerical', [binary])
    match = re.search(r'PASS ALL: (\d+) cases, (\d+) values', numerical)
    assert match, numerical
    print(match[0], flush=True)
    sass = run('sass', [args.cuda/'bin/cuobjdump','--dump-sass',binary])
    formats = sorted(set(re.findall(r'QMMA\.SF\.16832\.F32\.(E\dM\d\.E\dM\d)\.E8',sass)))
    assert len(formats)==4, formats
    assert not re.search(r'\b(?:CALL|LDL|STL|LDS|STS)(?:\.|\s)',sass), 'Unexpected call/local/shared memory instruction'
    sanitizers = {}
    if args.sanitize:
        for tool in ('memcheck','synccheck','initcheck'):
            log = run(tool, [args.cuda/'bin/compute-sanitizer','--tool',tool,
                             '--error-exitcode','1',binary])
            assert 'ERROR SUMMARY: 0 errors' in log, log[-2000:]
            sanitizers[tool] = '0 errors'
            print(f'{tool}: 0 errors',flush=True)
    source = [ROOT/'tests/cuda/test_tk_block_scaled.cu',ROOT/'experiments/validate_tk_block_scaled.py']
    source += sorted((ROOT/'third_party/thunderkittens/include').rglob('*.cuh'))
    result = {
        'gpu': numerical.splitlines()[0],
        'nvcc': subprocess.check_output([str(compiler),'--version'],text=True).strip(),
        'flags': flags, 'cases': int(match[1]), 'checked_values': int(match[2]),
        'max_abs_error': 0.0, 'oracle': 'CPU FP64 on decoded FP8 and E8M0, rounded to FP32; exact dyadic sums and one-hot finite-encoding sweeps',
        'sass_qmma_sf_formats': formats, 'spills': 0,
        'sass_has_calls_local_or_shared_instructions': False, 'sanitizers': sanitizers,
        'source_sha256': {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
    }
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Saved {out/"result.json"}')


if __name__=='__main__':
    main()
