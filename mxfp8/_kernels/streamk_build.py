"""Bounded, Torch-free CUDA extension build shared by Stream-K variants."""
import subprocess
from pathlib import Path
from .._core import ROOT, nvcc_path


def build_shared(source, output, *, flags):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.log').open('w') as log:
        subprocess.run([nvcc_path(), *flags, '-Xptxas=-v', '-shared',
                        '-Xcompiler=-fPIC', str(source), '-lcuda', '-o', str(output)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=600)
