"""Validate and inspect factored BF16/FP8 STMATRIX shared addresses."""
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
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.tk_persistent_gemm import FLAGS

OUT=ROOT/'runs/tk_stmatrix_address'
CASES=[(16,32,1),(16,64,1),(16,128,1),(16,32,4),(16,64,4),
       (16,128,4),(16,256,4),(32,64,4),(32,128,4),(64,32,4),(64,64,4)]
KINDS=['bf16','fp8','fp8_movmatrix','fp32_to_bf16']


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'build.log').open('w') as log:
        subprocess.run(['/usr/local/cuda/bin/nvcc',*FLAGS,'-shared','-Xcompiler=-fPIC',
                        '-Xptxas=-v','cuda/probe_stmatrix_address.cu','-o',str(OUT/'libprobe.so')],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def assembly():
    raw=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',
                                 str(OUT/'libprobe.so')],text=True)
    (OUT/'probe.sass').write_text(raw)
    result={}
    for part in raw.split('Function : ')[1:]:
        name=part.splitlines()[0]
        ins=[(int(pc,16),op,args) for pc,op,args in re.findall(
            r'/\*([0-9a-f]+)\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)\s*([^;]*);',part)]
        assert not any(op.startswith(('LDL','STL')) for _,op,_ in ins),name
        loops=[]
        for pc,op,args in ins:
            target=re.search(r'0x([0-9a-f]+)',args)
            if op.startswith('BRA') and target and int(target[1],16)<pc:
                block=[v for v in ins if int(target[1],16)<=v[0]<=pc]
                stores=[v for v in block if v[1].startswith('STSM')]
                if stores:
                    loops.append({'opcodes':dict(Counter(v[1] for v in block)),
                                  'addresses':[v[2].split(']',1)[0]+']' for v in stores],
                                  'instructions':[f'{a:04x}: {b} {c};' for a,b,c in block]})
        assert len(loops)==1,(name,loops)
        if 'Lb1E' in name:
            assert not any(op.startswith('SHF.') for op in loops[0]['opcodes']),name
        result[name]=loops[0]
    assert len(result)==len(CASES)*8
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--skip-build',action='store_true')
    ap.add_argument('--output',type=Path,default=ROOT/'benchmarks/tk_stmatrix_address_probe_rtx5090.json')
    args=ap.parse_args()
    if not args.skip_build:build()
    torch.set_num_threads(4)
    lib=ct.CDLL(str(OUT/'libprobe.so'))
    lib.stsm_probe.argtypes=[ct.c_int]*7+[ct.c_void_p]*3
    lib.stsm_probe_error.restype=ct.c_char_p
    result={'gpu':torch.cuda.get_device_name(),'cases':[],
            'scope':'One-CTA address/layout probe. Compare all physical shared bytes of two buffers; not a GEMM benchmark.'}
    for r,c,gw in CASES:
        for kind in range(4):
            size=2 if kind in [0,3] else 1
            sb=min(128,c*size);alignment=8*sb
            outputs=[torch.empty(2*gw*r*c*size//4,device='cuda',dtype=torch.int32) for _ in range(2)]
            cycles=[torch.empty(gw*32,device='cuda',dtype=torch.int64) for _ in range(2)]
            def launch(mode,it,off):
                status=lib.stsm_probe(r,c,gw,kind,mode,it,off,outputs[mode].data_ptr(),
                                      cycles[mode].data_ptr(),torch.cuda.current_stream().cuda_stream)
                if status:raise RuntimeError(lib.stsm_probe_error().decode())
            assert lib.stsm_probe(r,c,gw,kind,1,1,16,outputs[1].data_ptr(),
                                  cycles[1].data_ptr(),torch.cuda.current_stream().cuda_stream)!=0
            comparisons=0
            for off in [0,alignment,2*alignment,4*alignment]:
                for it in [1,2,3,17]:
                    for mode in range(2):launch(mode,it,off)
                    torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
                    comparisons+=1
            result['cases'].append({'R':r,'C':c,'GW':gw,'kind':KINDS[kind],
                                    'bitwise_comparisons':comparisons,'misaligned_base_rejected':True})
        print((r,c,gw),'PASS',flush=True)
    result['sass']=assembly()
    result['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
        'cuda/shared_store.cuh','cuda/mxfp8_epilogue.cuh','cuda/probe_stmatrix_address.cu',
        'experiments/probe_tk_stmatrix.py']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
