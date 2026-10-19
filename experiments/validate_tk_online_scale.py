"""Validate register-only BF16 -> MXFP8 quantization against Triton online_scale.

Run with the blkw Python environment. --sanitize also checks both FP8 formats
and the quantize -> block-scaled MMA composition with compute-sanitizer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mxtest import online_scale, u16_max


@triton.jit
def reference(X, Q, S, M: tl.constexpr, K: tl.constexpr, E5: tl.constexpr):
    batch = tl.program_id(0)
    offsets = tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :]
    x = tl.load(X + batch*M*K + offsets)
    if E5:
        # Exact copy of online_scale; only the final output type differs.
        xx = tl.reshape(x, (M, K//32, 32))
        ex = tl.inline_asm_elementwise('and.b32 $0, $1, 0x7F807F80;', '=r,r', [xx], tl.uint16, True, 2)
        es = tl.reduce(ex, -1, u16_max)
        rcp = tl.inline_asm_elementwise('sub.u32 $0, 0x7F007F00, $1; and.b32 $0, $0, 0x7F807F80;',
                                      '=r,r', [es], tl.bfloat16, True, 2)
        q = (xx * rcp[:, :, None]).to(tl.float8e5).reshape(M, K)
        s = (es >> 7).to(tl.uint8)
    else:
        q, s = online_scale(x)
    tl.store(Q + batch*M*K + offsets, q)
    so = tl.arange(0, M)[:, None] * (K//32) + tl.arange(0, K//32)[None, :]
    tl.store(S + batch*M*K//32 + so, s)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda', type=Path, default=Path('/usr/local/cuda'))
    p.add_argument('--output', type=Path, default=ROOT/'runs/tk_online_scale')
    p.add_argument('--sanitize', action='store_true')
    p.add_argument('--probe-cuda', type=Path, default=Path('/usr/local/cuda-13.4'),
                   help='Optional PTX 9.4 toolkit for the scaled-cvt compile-only probe')
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    flags = ['-std=c++20', '-gencode=arch=compute_120a,code=sm_120a', '-DKITTENS_RTX_BLACKWELL',
             '-Ithird_party/thunderkittens/include', '--expt-relaxed-constexpr', '--extended-lambda',
             '-O3', '-lineinfo', '-Xptxas=-v']
    binary = out/'test_tk_online_scale'

    def run(name, command):
        print(f'Running {name}', flush=True)
        r = subprocess.run(list(map(str, command)), cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out/f'{name}.log').write_text(r.stdout)
        if r.returncode:
            raise RuntimeError(f'{name}: exit {r.returncode}\n{r.stdout[-5000:]}')
        return r.stdout

    build = run('build', [args.cuda/'bin/nvcc', *flags, 'tests/cuda/test_tk_online_scale.cu', '-o', binary])
    assert not re.search(r'\b[1-9]\d* bytes (?:spill stores|spill loads)', build)
    # Keep the PTX to prove mul.bf16x2 and native packed cvt are used.
    run('ptx', [args.cuda/'bin/nvcc', *flags[:-1], '-ptx', 'tests/cuda/test_tk_online_scale.cu', '-o', out/'kernel.ptx'])
    sass = run('sass', [args.cuda/'bin/cuobjdump', '--dump-sass', binary])
    assert not re.search(r'\b(?:CALL|LDL|STL|LDS|STS|LDSM|LDGSTS)(?:\.|\s)', sass)
    assert 'HMUL2.BF16' in sass and 'F2FP.SATFINITE.E4M3.BF16' in sass and 'F2FP.SATFINITE.E5M2.BF16' in sass
    ptx = (out/'kernel.ptx').read_text()
    assert 'mul.bf16x2' in ptx
    probes = {}
    probe_nvcc = args.probe_cuda/'bin/nvcc'
    if probe_nvcc.exists():
        for fmt in ['e4', 'e5']:
            command = [str(probe_nvcc), '-gencode=arch=compute_120a,code=sm_120a', '-c',
                       'tests/cuda/probe_tk_scaled_cvt.cu', '-o', str(out/f'probe_{fmt}.o')]
            if fmt=='e5':
                command.append('-DPROBE_E5')
            r = subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (out/f'scaled_cvt_probe_{fmt}.log').write_text(r.stdout)
            assert r.returncode and "Illegal modifier '.scaled::n1::ue8m0'" in r.stdout, r.stdout
            probes[fmt] = 'rejected for sm_120a: Illegal modifier scaled::n1::ue8m0'
    torch.manual_seed(123)
    cases = []
    sanitize_commands = []
    for m,k in [(16,32),(16,64),(32,64),(32,96),(64,128)]:
        x = (torch.randn(13,m,k) * 3).to(torch.bfloat16)
        cases.append((f'random_{m}_{k}', x, False))
        # Exactly representable dyadic products/sums; distinct row/K32 scales.
        x = torch.randint(-8,9,(13,m,k)).float()/4
        shifts = torch.randint(-1,2,(13,m,k//32,1))
        x = (x.reshape(13,m,k//32,32) * (2.**shifts)).reshape(13,m,k).to(torch.bfloat16)
        cases.append((f'gram_{m}_{k}', x, True))
    raw = torch.arange(65536,dtype=torch.int32)
    raw[((raw >> 7) & 255)==255] = 0  # finite BF16 contract, includes +/-0 and all subnormals
    cases.append(('all_finite_bf16_encodings',raw.to(torch.uint16).view(torch.bfloat16).reshape(128,16,32),False))
    x = torch.zeros(13,16,32,dtype=torch.bfloat16)
    x[:,::2] = -0.
    cases.append(('signed_zero',x,False))
    # Every position of the maximum, all mantissas and rounding ties near 1.
    x = torch.arange(128,dtype=torch.int32).repeat(128).reshape(32,16,32) + (127<<7)
    for b in range(32):
        x[b,:,b] = (130<<7)+127
    x[:,::2] |= 0x8000
    cases.append(('max_positions_and_mantissas',x.to(torch.uint16).view(torch.bfloat16),False))
    # Finite blocks with very different exponents, including FP8 underflow.
    exponents = torch.randint(1,254,(13,32,2,1)) - torch.randint(0,180,(13,32,2,32))
    bits = (exponents.clamp_min(0)<<7) | torch.randint(0,128,exponents.shape)
    bits |= torch.randint(0,2,exponents.shape)<<15
    cases.append(('mixed_exponents',bits.to(torch.uint16).view(torch.bfloat16).reshape(13,32,64),False))
    results = []
    for name, x, gram in cases:
        batches,m,k = x.shape
        input_path = out/f'{name}.bf16'
        input_path.write_bytes(x.view(torch.uint16).numpy().tobytes())
        for fmt,dtype in [('e4',torch.float8_e4m3fn),('e5',torch.float8_e5m2)]:
            label = f'{name}_{fmt}'
            output_path = out/f'{label}.bin'
            command = [binary,m,k,batches,fmt,int(gram),input_path,output_path]
            run(label, command)
            n=x.numel()
            data=output_path.read_bytes()
            actual_q=torch.from_numpy(np.frombuffer(data[:n],dtype=np.uint8).copy()).reshape_as(x)
            actual_s=torch.from_numpy(np.frombuffer(data[n:n+n//32],dtype=np.uint8).copy()).reshape(batches,m,k//32)
            # K=96 is not a Triton power-of-two shape: run each independent K32 block.
            xr=x.cuda()
            if k==96:
                xr=xr.reshape(batches*m,k//32,32).transpose(0,1).contiguous().reshape(-1,16,32)
            qr=torch.empty(xr.shape,device='cuda',dtype=dtype)
            sr=torch.empty((*xr.shape[:-1],xr.shape[-1]//32),device='cuda',dtype=torch.uint8)
            ref_kernel=reference[(xr.shape[0],)](xr,qr,sr,xr.shape[1],xr.shape[2],fmt=='e5',num_warps=4)
            if k==96:
                qr=qr.reshape(k//32,batches*m,32).transpose(0,1).reshape(batches,m,k)
                sr=sr.reshape(k//32,batches*m).transpose(0,1).reshape(batches,m,k//32)
            expected_q=qr.view(torch.uint8).cpu()
            expected_s=sr.cpu()
            qb=int((actual_q!=expected_q).sum())
            sb=int((actual_s!=expected_s).sum())
            # Independent host exponent oracle (includes all-zero scale byte 0).
            bits=x.view(torch.uint16).to(torch.int32)
            host_s=((bits>>7)&255).reshape(batches,m,k//32,32).amax(-1).to(torch.uint8)
            assert torch.equal(actual_s,host_s), label
            # Independent value oracle reproducing the legacy BF16 reciprocal.
            reciprocal_bits=(((254-host_s.to(torch.int32))<<7)&0x7f80).to(torch.uint16)
            reciprocal=reciprocal_bits.view(torch.bfloat16).double()[...,None]
            host_q=(x.reshape(batches,m,k//32,32).double()*reciprocal).to(torch.bfloat16).to(dtype).view(torch.uint8).reshape_as(x)
            hb=int((actual_q!=host_q).sum())
            result={'case':label,'values':n,'scales':n//32,'triton_fp8_bit_mismatches':qb,
                    'triton_scale_mismatches':sb,'host_fp8_bit_mismatches':hb}
            if qb or sb or hb:
                indices=(actual_q!=expected_q).nonzero()[:8]
                result['first_differences']=[{'index':ix.tolist(),'input':float(x[tuple(ix)]),
                    'cuda':int(actual_q[tuple(ix)]),'triton':int(expected_q[tuple(ix)])} for ix in indices]
            if gram:
                actual_g=torch.from_numpy(np.frombuffer(data[n+n//32:],dtype=np.float32).copy()).reshape(batches,m,m)
                dequant=actual_q.view(dtype).double().reshape(batches,m,k//32,32)*torch.pow(2.,actual_s.double()-127)[...,None]
                dequant=dequant.reshape(batches,m,k)
                expected_g=(dequant@dequant.transpose(-1,-2)).float()
                err=(actual_g-expected_g).abs().max().item()
                result['gram_max_abs_error']=err
                assert torch.equal(actual_g,expected_g), result
                if (m,k)==(32,96):
                    sanitize_commands.append((fmt,command))
            print(json.dumps(result),flush=True)
            results.append(result)
            (out/'comparisons.json').write_text(json.dumps(results,indent=2)+'\n')
            assert qb==sb==hb==0, result
            if name=='random_16_32':
                (out/f'triton_{fmt}.ptx').write_text(ref_kernel.asm['ptx'])
    sanitizers={}
    if args.sanitize:
        for fmt,command in sanitize_commands:
            for tool in ['memcheck','synccheck','initcheck']:
                log=run(f'{tool}_{fmt}',[args.cuda/'bin/compute-sanitizer','--tool',tool,'--error-exitcode','1',*command])
                assert 'ERROR SUMMARY: 0 errors' in log
                sanitizers[f'{tool}_{fmt}']='0 errors'
    sources=[ROOT/'tests/cuda/test_tk_online_scale.cu', ROOT/'tests/cuda/probe_tk_scaled_cvt.cu',
             Path(__file__),ROOT/'mxtest.py']
    sources+=sorted((ROOT/'third_party/thunderkittens/include').rglob('*.cuh'))
    result={'gpu':torch.cuda.get_device_name(),'nvcc':subprocess.check_output([str(args.cuda/'bin/nvcc'),'--version'],text=True),
            'flags':flags,'cases':results,'total_fp8_values':sum(r['values'] for r in results),
            'total_scale_values':sum(r['scales'] for r in results),'sanitizers':sanitizers,
            'sass_has_calls_local_or_shared_instructions':False,'spills':0,
            'scaled_cvt_sm120_probe':probes,
            'source_sha256':{str(f.resolve().relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
            'limitations':['Finite BF16 only','Preserves Triton reciprocal exponent-254 degeneration to zero',
                           'SM120: explicit mul.bf16x2 + unscaled native FP8 cvt; scaled::n1 unsupported']}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS {len(results)} cases; saved {out/"result.json"}',flush=True)


if __name__=='__main__':
    main()
