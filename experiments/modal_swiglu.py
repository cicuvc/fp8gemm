"""Run the SwiGLU comparison on Modal RTX PRO 6000 and save results locally.

python -m modal run experiments/modal_swiglu.py::main
python -m modal run experiments/modal_swiglu.py::dump
"""
import hashlib
import json
from pathlib import Path
import subprocess

import modal

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    'mxtest.py', 'fp8_linear.py', 'fp8_swiglu.py', 'benchmark_vs_fla.py',
    'profile_swiglu_backward.py', 'experiments/__init__.py',
    'experiments/evaluate_fp8_wgrad.py', 'experiments/cached_fp8_wgrad.py',
    'experiments/cached_fp8_wgrad_fused.py',
    'experiments/benchmark_cached_fp8_wgrad.py', 'tests/test_cached_fp8_wgrad.py',
    'experiments/dump_forward_asm.py',
]
image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.13.0', 'triton==3.7.1',
                 extra_index_url='https://download.pytorch.org/whl/cu130')
    .pip_install('flash-linear-attention==0.4.2', 'transformers==5.15.0',
                 'einops==0.8.2', 'pytest==9.0.2')
    .env({'PYTHONPATH': '/workspace', 'TRITON_CACHE_DIR': '/cache/triton',
          'PYTHONUNBUFFERED': '1'})
)
for source in SOURCES:
    image = image.add_local_file(ROOT / source, '/workspace/' + source)

app = modal.App('fp8-swiglu-rtx-pro-6000')
cache = modal.Volume.from_name('fp8-swiglu-triton-cache', create_if_missing=True)


@app.function(image=image, gpu='RTX-PRO-6000', cpu=8, memory=32768,
              timeout=1800, volumes={'/cache': cache})
def forward_asm():
    import os
    import io
    import zipfile
    from experiments.dump_forward_asm import dump_forward
    os.chdir('/workspace')
    directory = Path('/cache/forward_asm')
    dump_forward(directory)
    cache.commit()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(directory))
    return buffer.getvalue()


@app.local_entrypoint()
def dump(output: str = 'benchmarks/ptx_rtx_pro_6000'):
    import io
    import zipfile
    blob = forward_asm.remote()
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        archive.extractall(directory)
    print(f'Saved forward assembly to {directory}')


@app.function(image=image, gpu='RTX-PRO-6000', cpu=8, memory=32768,
              timeout=3600, volumes={'/cache': cache})
def measure(manifest, rep: int, repeats: int):
    import os
    import sys
    from datetime import datetime, timezone
    import torch
    import triton
    import fla

    os.chdir('/workspace')
    os.environ['FP8GEMM_BASELINE_COMMIT'] = manifest['commit']
    for name, expected in manifest['source_sha256'].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        assert actual == expected, f'Source hash mismatch: {name}'
    metadata = {
        'utc': datetime.now(timezone.utc).isoformat(), 'source': manifest,
        'gpu': torch.cuda.get_device_name(),
        'capability': torch.cuda.get_device_capability(),
        'torch': torch.__version__, 'triton': triton.__version__,
        'fla': fla.__version__, 'cuda': torch.version.cuda,
        'nvidia_smi_before': subprocess.check_output(['nvidia-smi'], text=True),
    }
    print(json.dumps(metadata, indent=2), flush=True)
    # Keep logs/results in a persistent volume even if the local client exits.
    outdir = Path('/cache/results') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    outdir.mkdir(parents=True, exist_ok=True)
    commands = [
        ('tests', [sys.executable, '-m', 'pytest', '-q', 'tests/test_cached_fp8_wgrad.py']),
        ('benchmark', [sys.executable, 'experiments/benchmark_cached_fp8_wgrad.py',
                       '--fusions', '--baselines', '--rep', str(rep), '--repeats', str(repeats),
                       '--output', str(outdir / 'benchmark.json')]),
    ]
    logs = {}
    for label, command in commands:
        print(f'Running {label}', flush=True)
        lines = []
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                lines.append(line)
            code = process.wait()
        logs[label] = ''.join(lines)
        (outdir / f'{label}.log').write_text(logs[label])
        cache.commit()
        if code:
            raise RuntimeError(f'{label} exited {code}; logs saved in {outdir}')
    result = json.loads((outdir / 'benchmark.json').read_text())
    result['modal_environment'] = metadata
    result['modal_environment']['nvidia_smi_after'] = subprocess.check_output(['nvidia-smi'], text=True)
    result['mfu'] = {
        'fp8_a32_peak_tflops_range': [877.9, 1007.6], 'peak_source': 'user supplied',
        'definition': 'Dense GEMM FLOPs / elapsed seconds / peak. All timings include packing. BF16 providers and BF16 dW are normalized to the same FP8 peak, not dtype-adjusted.',
    }
    for case in result['cases']:
        m, k, h = case['shape']
        for phases in case['timings'].values():
            for phase, values in phases.items():
                flops = {'training_forward': 6, 'backward': 12, 'forward_backward': 18}[phase] * m * k * h
                tflops = flops / values['median_ms'] / 1e9
                values['effective_tflops'] = tflops
                values['fp8_peak_mfu_percent_range'] = [100*tflops/1007.6, 100*tflops/877.9]
    (outdir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    cache.commit()
    return result, logs


@app.local_entrypoint()
def main(rep: int = 50, repeats: int = 5,
         output: str = 'benchmarks/swiglu_rtx_pro_6000.json'):
    manifest = {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip(),
        'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
    }
    result, logs = measure.remote(manifest, rep, repeats)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    for label, log in logs.items():
        path.with_suffix(f'.{label}.log').write_text(log)
    print(f'Saved {path}')
