"""Measure LDSM byte coordinates and verify their use by native block-scaled QMMA."""
import ctypes as ct
import hashlib
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
    lib.probe_fp8_qmma.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p]
    lib.probe_fp8_qmma.restype = ct.c_int
    torch.manual_seed(12)
    weight = torch.randint(-8, 9, (32, 16), device='cuda').to(torch.float8_e4m3fn)
    product = torch.full((32, 16), float('nan'), device='cuda', dtype=torch.float32)
    reference = weight.float()
    result['qmma_identity_cases'] = []
    for mode in range(4):
        assert lib.probe_fp8_qmma(weight.data_ptr(), product.data_ptr(), mode,
                                 torch.cuda.current_stream().cuda_stream) == 0
        mismatches = (product != reference).sum().item()
        max_abs = (product-reference).abs().max().item()
        if mode == 0:
            torch.testing.assert_close(product, reference, rtol=0, atol=0)
        else:
            assert mismatches > 0
        result['qmma_identity_cases'].append({
            'swap_address_rows_8_15_and_16_23': bool(mode & 2),
            'b_register_pairs': [[0, 1], [2, 3]] if mode & 1 else [[0, 2], [1, 3]],
            'values': product.numel(), 'mismatches': mismatches, 'max_abs': max_abs})
    result['qmma_reference'] = (
        'Two m16n8k32 instructions per CTA; two CTAs select K0..15 and K16..31 '
        'using identity A. E4M3 W is random integers -8..8, seed12. '
        'All UE8M0 scales are 127. Output must equal W exactly.')
    result['source_sha256'] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT/'tests/cuda/ldmatrix_fp8_layout.cu']}
    result['conclusion'] = (
        'Native output pairs split N8, while QMMA B pairs split K16. '
        'Swapping row address providers permutes K inside words; it cannot '
        'exchange the fixed N8 grouping of the transposing instruction.')
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(binary)], text=True)
    (output / 'probe.sass').write_text(sass)
    print(f'All 4 coordinate probes passed; QMMA identity results:')
    for case in result['qmma_identity_cases']:
        print(case)
    print(f'Results in {output}')


if __name__ == '__main__':
    main()
