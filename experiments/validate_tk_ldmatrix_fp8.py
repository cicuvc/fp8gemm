"""Measure byte coordinates before/after swapping LDSM address-provider groups."""
import ctypes as ct
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.tk_persistent_gemm import FLAGS


def main():
    output = ROOT / 'runs/tk_ldmatrix_fp8'
    output.mkdir(parents=True, exist_ok=True)
    binary = output / 'probe.so'
    subprocess.run(['/usr/local/cuda/bin/nvcc', *FLAGS, '-shared', '-Xcompiler=-fPIC',
                    'tests/cuda/ldmatrix_fp8_layout.cu', '-o', str(binary)],
                   cwd=ROOT, check=True)
    lib = ct.CDLL(str(binary))
    lib.probe_fp8.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p]
    lib.probe_fp8.restype = ct.c_int
    out = torch.empty((32, 4, 4), device='cuda', dtype=torch.uint8)
    result = {'gpu': torch.cuda.get_device_name(), 'cases': []}
    for swap in (False, True):
        for columns in (False, True):
            assert lib.probe_fp8(out.data_ptr(), columns, swap,
                                torch.cuda.current_stream().cuda_stream) == 0
            got = out.cpu()
            expected = torch.empty_like(got)
            for lane in range(32):
                for reg in range(4):
                    for byte in range(4):
                        row = 4*(lane % 4) + 16*(reg // 2) + byte
                        if swap:
                            row = (row & 7) | ((row & 8) << 1) | ((row & 16) >> 1)
                        expected[lane, reg, byte] = lane // 4 + 8*(reg % 2) if columns else row
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
            result['cases'].append({'swap_address_rows_8_15_and_16_23': swap,
                                    'coordinate': 'N' if columns else 'K',
                                    'register_bytes_by_lane': got.tolist()})
    result['all_coordinates_exact'] = True
    result['conclusion'] = (
        'Native output pairs split N8, while QMMA B pairs split K16. '
        'Swapping row address providers permutes K inside words; it cannot '
        'exchange the fixed N8 grouping of the transposing instruction.')
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(binary)], text=True)
    (output / 'probe.sass').write_text(sass)
    print(f'All 4 coordinate probes passed; results in {output}')


if __name__ == '__main__':
    main()
