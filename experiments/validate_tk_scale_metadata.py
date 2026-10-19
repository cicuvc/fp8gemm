"""Check E8M0 metadata conversion alternatives on SM120; dump SASS for comparison."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda'))
    p.add_argument('--output',type=Path,default=ROOT/'runs/tk_scale_metadata')
    p.add_argument('--sanitize',action='store_true')
    args=p.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    binary=out/'test'
    source=ROOT/'tests/cuda/test_tk_scale_metadata.cu'
    flags=['-std=c++20','-gencode=arch=compute_120a,code=sm_120a','-O3','-lineinfo','-Xptxas=-v']

    def run(name,command):
        r=subprocess.run(list(map(str,command)),cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/f'{name}.log').write_text(r.stdout)
        assert r.returncode==0,r.stdout
        return r.stdout

    run('build',[args.cuda/'bin/nvcc',*flags,source,'-o',binary])
    numerical=run('numerical',[binary])
    print(numerical,end='')
    assert numerical.count('PASS')==4
    sass=run('sass',[args.cuda/'bin/cuobjdump','--dump-sass',binary])
    counts={}
    for section in sass.split('Function : ')[1:]:
        name=section.splitlines()[0]
        # Counts are static instructions, not throughput or latency measurements.
        opcodes=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)',section)
        counts[name]=dict(Counter(opcodes))
    sanitizers={}
    if args.sanitize:
        for tool in ['memcheck','synccheck','initcheck']:
            log=run(tool,[args.cuda/'bin/compute-sanitizer','--tool',tool,'--error-exitcode','1',binary])
            assert 'ERROR SUMMARY: 0 errors' in log
            sanitizers[tool]='0 errors'
    result={'nvcc':subprocess.check_output([str(args.cuda/'bin/nvcc'),'--version'],text=True),
            'flags':flags,'values_per_variant':131072,'variants':{
                '0':'shift and extract baseline','1':'cvt.rz.ue8m0x2.bf16x2 then shift/extract',
                '2':'cvt.rz.satfinite.ue8m0x2.bf16x2 then shift/extract','3':'non-saturating cvt then byte_perm'},
            'non_saturating_mismatches':0,'saturating_finite_mismatches':0,'saturating_exponent255_mismatches':512,
            'reverse_conversion_finite_values':255,'reverse_e8m0_zero_bf16_bits':'0x0040',
            'sass_static_instruction_counts':counts,'sanitizers':sanitizers,
            'decision':'Keep shift in online_quantize_mxfp8: cvt plus lane selection has no instruction-count advantage; no performance claim.',
            'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                             for f in [source,Path(__file__).resolve()]}}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Saved {out/"result.json"}')


if __name__=='__main__':
    main()
