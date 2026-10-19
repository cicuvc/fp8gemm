"""Cross-check warp accumulator ownership in dumped TTGIR, PTX and SASS."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess


def analyze(directory):
    import triton
    root = Path(directory)
    dump = json.loads((root/'summary.json').read_text())
    nvdisasm = Path(triton.__file__).parent/'backends/nvidia/bin/nvdisasm'
    records = []
    for kernel in dump['kernels']:
        stem = root/kernel['files']
        ptx = stem.with_suffix('.ptx').read_text()
        wm, wn = map(int, re.search(r'warpsPerCTA = \[(\d+), (\d+)\]', kernel['mma_layout'][0]).groups())
        config = kernel['config']
        bm, bn, bk = [config[k] for k in ('BLOCK_M','BLOCK_N','BLOCK_K')]
        assert wm*wn == config['num_warps']
        assert bm % wm == bn % wn == 0
        tm, tn = bm//wm, bn//wn
        projections = 2 if kernel['stage'].startswith('gate_up') else 1
        expected_acc = tm*tn*projections//32
        mma = [line.strip() for line in ptx.splitlines() if line.strip().startswith('mma.sync.')]
        dest = [re.search(r'\{([^}]+)\}', line)[1] for line in mma]
        acc_regs = {int(reg) for group in dest for reg in re.findall(r'%r(\d+)', group)}
        expected_mma = (tm//16)*(tn//8)*(bk//32)*projections
        assert len(acc_regs) == expected_acc, kernel['files']
        assert len(mma) == expected_mma, kernel['files']
        sass = subprocess.check_output([str(nvdisasm),str(stem.with_suffix('.cubin'))],text=True)
        stem.with_suffix('.sass').write_text(sass)
        qmma = re.findall(r'\bQMMA\.SF\.16832\.F32\.E4M3\.E4M3\.E8 R(\d+),',sass)
        physical_acc = {base+i for base in map(int,qmma) for i in range(4)}
        assert len(qmma) == expected_mma, kernel['files']
        # ptxas can relocate an accumulator between K slices (Rd != Rc),
        # so the union of physical destinations can exceed its live size.
        assert len(physical_acc) >= expected_acc, kernel['files']
        # All these kernels have a single K loop; count its static instructions.
        blocks = list(re.finditer(r'^(\$L__BB\w+):[^\n]*Inner Loop Header[^\n]*$',ptx,re.M))
        loop_counts = {}
        if len(blocks) == 1:
            header = blocks[0]
            backedge = re.search(r'bra\s+'+re.escape(header[1])+r';',ptx[header.end():])
            if backedge:
                body = ptx[header.end():header.end()+backedge.end()]
                loop_counts = dict(Counter(re.findall(r'^\s*(?:@!?%p\d+\s+)?([a-z][\w.:]*)\s+[^\n]*;',body,re.M)))
        records.append({
            'kernel': kernel['files'], 'cta_m_n_k': [bm,bn,bk],
            'warps_m_n': [wm,wn], 'warp_logical_acc_m_n_per_projection': [tm,tn],
            'projections': projections, 'fp32_acc_values_per_thread': expected_acc,
            'ptx_acc_virtual_registers': sorted(acc_regs),
            'sass_acc_destination_register_union': sorted(physical_acc),
            'mma_per_warp_per_k_step': expected_mma,
            'total_registers_per_thread': kernel['registers_per_thread'],
            'dynamic_shared_bytes': kernel['metadata']['shared'],
            'spill_count': kernel['spill_count'],
            'occupancy_max_blocks_per_sm': kernel.get('occupancy_max_blocks_per_sm'),
            'occupancy_max_warps_per_sm': kernel.get('occupancy_max_warps_per_sm'),
            'occupancy_max_percent': kernel.get('occupancy_max_percent'),
            'ptx_loop_static_instruction_counts': loop_counts,
        })
        print(kernel['files'],f'warp acc {tm}x{tn} x{projections}; acc/thread {expected_acc}; '
              f'total regs {kernel["registers_per_thread"]}; MMA/K-step {expected_mma}')
    result = {'notes': [
        'Warp logical tile dimensions count owned rows/columns; ownership may be interleaved, not a contiguous rectangle.',
        'PTX register names are virtual; total registers/thread and SASS register numbers are physical.',
        'The union of SASS accumulator destinations can exceed live accumulator size due to register relocation.',
        'Occupancy is a CUDA Driver resource-limit calculation, not measured achieved occupancy.',
        'Static instruction counts are not cycle counts or time proportions.',
    ], 'kernels': records}
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', default='benchmarks/ptx_rtx_pro_6000')
    analyze(parser.parse_args().directory)
