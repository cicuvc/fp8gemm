"""Small fragment/quantization/epilogue probes for swapped FP8 linear dX."""
import argparse
from collections import Counter
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.tk_persistent_gemm import FLAGS
from experiments.benchmark_tk_persistent_gemm import quantized_a
from experiments.benchmark_tk_dual_online_gemm import exact_input

OUTPUT = ROOT/'runs/tk_swap_ab'
SHAPES = [(16,32,32), (32,64,32), (16,96,64), (64,128,32), (32,64,64)]


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'build.log').open('w') as log:
        subprocess.run(['/usr/local/cuda/bin/nvcc', *FLAGS, '-Xptxas=-v', '-shared',
                        '-Xcompiler=-fPIC', 'tests/cuda/probe_tk_swap_ab.cu',
                        '-o', str(OUTPUT/'probe.so')],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert not re.search(r'\b[1-9]\d* bytes (?:spill stores|spill loads)',
                         (OUTPUT/'build.log').read_text())


def validate():
    torch.set_num_threads(4)
    lib = ct.CDLL(str(OUTPUT/'probe.so'))
    lib.probe_swap_ab.argtypes = [ct.c_void_p]*5 + [ct.c_int]*3 + [ct.c_void_p]
    lib.probe_swap_ab.restype = ct.c_int
    results = []
    batches = 7
    for config, (m,n,k) in enumerate(SHAPES):
        for pattern in ['dyadic', 'zero_blocks', 'fp8_encoding_sweep']:
            y = exact_input(batches*m, n, 73).reshape(batches,m,n)
            index = torch.arange(batches*n*k).reshape(batches,n,k)
            w = (((index*29+11)%17-8).float()/4).to(torch.float8_e4m3fn)
            scales = (125+(torch.arange(batches)[:,None,None]*7 +
                           torch.arange(n//32)[None,:,None]*3 +
                           torch.arange(k//32)[None,None,:])%5).to(torch.uint8)
            if pattern == 'zero_blocks':
                y[:, :8, :] = 0
                y[:, :, :32] = 0
                w.view(torch.uint8)[:, :, :32] = 0
            elif pattern == 'fp8_encoding_sweep':
                bits = (index%256).to(torch.uint8)
                bits[(bits & 127) == 127] = 0  # Remove both E4M3 NaN encodings.
                w = bits.view(torch.float8_e4m3fn)
                y.zero_()
                for b in range(batches):
                    for row in range(m):
                        y[b,row,(b*m+row)%n] = 1
            decoded = w.double()*torch.exp2(scales.double()-127).repeat_interleave(
                32,1).repeat_interleave(32,2)
            quant_y = quantized_a(y.reshape(batches*m,n)).reshape(batches,m,n)
            ref = (quant_y.double() @ decoded).float()
            inputs = [t.cuda() for t in [y,w,scales]]
            reference_dx = None
            for swap in [False, True]:
                raw = torch.full((batches,k,m) if swap else (batches,m,k),
                                 float('nan'), device='cuda', dtype=torch.float32)
                dx = torch.full((batches,m,k), float('nan'), device='cuda', dtype=torch.bfloat16)
                error = lib.probe_swap_ab(
                    *[t.data_ptr() for t in inputs], raw.data_ptr(), dx.data_ptr(),
                    config, int(swap), batches, torch.cuda.current_stream().cuda_stream)
                assert error == 0, error
                actual = raw.cpu().transpose(1,2) if swap else raw.cpu()
                actual_dx = dx.cpu()
                torch.testing.assert_close(actual, ref, rtol=0, atol=0)
                torch.testing.assert_close(actual_dx, ref.bfloat16(), rtol=0, atol=0)
                if reference_dx is None:
                    reference_dx = actual_dx
                else:
                    torch.testing.assert_close(actual_dx, reference_dx, rtol=0, atol=0)
                results.append({'shape_mnk':[m,n,k], 'batches':batches,
                                'pattern':pattern, 'swap_ab':swap,
                                'fp32_values':ref.numel(), 'bf16_values':ref.numel(),
                                'max_abs_fp32':0.0, 'bf16_mismatches':0})
        print(f'PASS M,N,K={m,n,k}: both orientations, 3 patterns, FP32 + BF16', flush=True)
    return results


def assembly():
    sass = subprocess.check_output(
        ['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(OUTPUT/'probe.so')], text=True)
    (OUTPUT/'probe.sass').write_text(sass)
    kernels = []
    for section in sass.split('Function : ')[1:]:
        name = section.splitlines()[0]
        swap = 'ELb1E' in name
        definitions, qmma = {}, []
        ops = Counter()
        for line in section.splitlines():
            match = re.search(
                r'/\*([0-9a-f]+)\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)\s+([^;]+);',
                line)
            if not match:
                continue
            pc, op, body = match.groups()
            operands = [x.strip() for x in body.split(',')]
            ops[op] += 1
            if op.startswith('QMMA'):
                base = int(operands[1 if swap else 2][1:])
                width = 4 if swap else 2
                origins = [definitions.get(base+i) for i in range(width)]
                direct = all(d and d[1] == 'LDSM.8.MT1616.2' for d in origins)
                direct = direct and len({d[0] for d in origins}) == 1
                qmma.append({'pc':pc, 'weight_registers':list(range(base,base+width)),
                             'definitions':origins, 'weight_direct_from_ldsm':direct})
            dest = operands[1] if op.startswith('SHFL') else operands[0]
            if re.fullmatch(r'R\d+', dest):
                base = int(dest[1:])
                width = (4 if op.startswith(('QMMA','LDSM')) or '.128' in op else
                         2 if '.64' in op or op.startswith(('IMAD.WIDE','LEA.WIDE')) else 1)
                for i in range(width):
                    definitions[base+i] = (pc,op)
        assert qmma
        assert not any(op.startswith(('LDL','STL','CALL')) for op in ops)
        if swap:
            assert all(q['weight_direct_from_ldsm'] for q in qmma), (name,qmma)
            assert 'STSM.16.MT88.4' in ops, (name,ops)
        kernels.append({'name':name, 'swap_ab':swap, 'opcodes':dict(ops),
                        'qmma_count':len(qmma),
                        'direct_weight_operands':sum(q['weight_direct_from_ldsm'] for q in qmma),
                        'qmma':qmma})
    assert len(kernels) == 2*len(SHAPES)
    return kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    if not args.skip_build:
        build()
    result = {'gpu':torch.cuda.get_device_name(), 'numerical':validate()}
    if args.validate_only:
        return
    result['assembly'] = assembly()
    result['sanitizers'] = {}
    if args.sanitize:
        for tool in ['memcheck','synccheck','initcheck','racecheck']:
            print(f'Running {tool}', flush=True)
            log = OUTPUT/f'{tool}.log'
            with log.open('w') as f:
                subprocess.run(
                    ['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,
                     '--error-exitcode','1',sys.executable,str(Path(__file__).resolve()),
                     '--skip-build','--validate-only'],
                    cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
            result['sanitizers'][tool] = log.read_text().strip().splitlines()[-1]
    result['scope'] = (
        'Small one-warp CTA probes with standard swizzled TK shared tiles. '
        'No global transpose, TMA, persistence, or end-to-end performance claim. '
        'FP8 W is transposed by native byte LDSM into A. Original BF16 dY '
        'is row-loaded and quantized with unchanged row/N32 scales into B. '
        'FP32 output is logically transposed via tile indexing; BF16 conversion '
        'plus STSM trans produces the original [M,K] output layout.')
    result['oracle'] = 'Independent CPU FP64 GEMM on decoded FP8 weight and row/N32-quantized BF16 dY; exact dyadic sums.'
    result['assembly_method'] = 'Lexical last-definition audit of each unrolled QMMA weight operand; not a general control-flow analyzer.'
    files = [Path(__file__),ROOT/'tests/cuda/probe_tk_swap_ab.cu']
    files += sorted((ROOT/'third_party/thunderkittens/include').rglob('*.cuh'))
    result['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result['binary_sha256'] = hashlib.sha256((OUTPUT/'probe.so').read_bytes()).hexdigest()
    (OUTPUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Saved {OUTPUT / "result.json"}')


if __name__ == '__main__':
    main()
