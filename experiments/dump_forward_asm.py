"""Capture the compiled kernels returned by actual forward launches."""
import hashlib
import json
from pathlib import Path


def dump_forward(output):
    import torch
    import triton
    import mxtest
    from cuda.bindings import driver as cu
    from fp8_swiglu import FP8SwiGLU
    from experiments.cached_fp8_wgrad import CachedWgradSwiGLU
    from experiments.cached_fp8_wgrad_fused import _swiglu_fwd_save_hidden

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tuners = {'gate_up': mxtest.fp8_swiglu_fwd,
              'down': mxtest.fp8gemm_fwd,
              'gate_up_save_hidden': _swiglu_fwd_save_hidden}
    captured, disk_hits, originals = {}, {}, []
    for name, tuner in tuners.items():
        old_run = tuner.fn.run
        old_disk = tuner.check_disk_cache

        def run(*args, _name=name, _old=old_run, **kwargs):
            compiled = _old(*args, **kwargs)
            captured[_name] = compiled
            return compiled

        def disk(*args, _name=name, _old=old_disk, **kwargs):
            hit = _old(*args, **kwargs)
            disk_hits[_name] = hit
            return hit

        originals.append((tuner, old_run, old_disk))
        tuner.fn.run = run
        tuner.check_disk_cache = disk
    summary = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
               'triton': triton.__version__, 'capability': torch.cuda.get_device_capability(),
               'device_properties': str(torch.cuda.get_device_properties(0)),
               'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in ('mxtest.py', 'experiments/cached_fp8_wgrad_fused.py')},
               'kernels': []}
    def checked(result):
        error, value = result
        if error != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'CUDA query failed: {error}')
        return int(value)
    summary['sm_resources'] = {
        name: checked(cu.cuDeviceGetAttribute(getattr(cu.CUdevice_attribute, 'CU_DEVICE_ATTRIBUTE_'+attr), 0))
        for name, attr in (
            ('max_threads', 'MAX_THREADS_PER_MULTIPROCESSOR'),
            ('registers_32bit', 'MAX_REGISTERS_PER_MULTIPROCESSOR'),
            ('shared_bytes', 'MAX_SHARED_MEMORY_PER_MULTIPROCESSOR'),
        )
    }
    try:
        for m, k, h in ((256,512,1024), (1024,1024,4096), (2048,2048,8192)):
            torch.manual_seed(23)
            x = torch.randn(m,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            for variant, cls, options in (
                ('default', FP8SwiGLU, {}),
                ('cached_hidden', CachedWgradSwiGLU, {'fuse_hidden_pack': True}),
            ):
                layer = cls(k,h,k,bias=False,device='cuda',**options)
                captured.clear()
                y = layer(x)
                torch.cuda.synchronize()
                for name, kernel in captured.items():
                    stem = f'{m}_{k}_{h}/{variant}_{name}'
                    path = output / stem
                    path.parent.mkdir(parents=True, exist_ok=True)
                    for ext in ('ptx','ttgir','ttir','llir','cubin'):
                        data = kernel.asm.get(ext)
                        if data is not None:
                            if isinstance(data, bytes):
                                path.with_suffix('.'+ext).write_bytes(data)
                            else:
                                path.with_suffix('.'+ext).write_text(data)
                    config = tuners[name].best_config
                    record = {
                        'shape_m_k_h': [m,k,h], 'variant': variant, 'stage': name,
                        'files': stem, 'config': config.all_kwargs(),
                        'autotune_disk_hit': disk_hits.get(name),
                        'kernel_hash': kernel.hash,
                        'metadata': kernel.metadata._asdict(),
                        'registers_per_thread': kernel.n_regs,
                        'spill_count': kernel.n_spills,
                        'occupancy_max_blocks_per_sm': checked(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                            cu.CUfunction(kernel.function), kernel.metadata.num_warps*32, kernel.metadata.shared)),
                        'mma_layout': [s for s in kernel.asm['ttgir'].splitlines()
                                       if 'nvidia_mma' in s],
                    }
                    record['occupancy_max_warps_per_sm'] = record['occupancy_max_blocks_per_sm'] * kernel.metadata.num_warps
                    record['occupancy_max_percent'] = 100 * record['occupancy_max_warps_per_sm'] * 32 / summary['sm_resources']['max_threads']
                    path.with_suffix('.json').write_text(json.dumps(record,indent=2,default=str)+'\n')
                    summary['kernels'].append(record)
                    print(json.dumps({key: record[key] for key in (
                        'files','config','autotune_disk_hit','registers_per_thread','spill_count','mma_layout')},default=str),flush=True)
                del y, layer
            del x
            torch.cuda.empty_cache()
    finally:
        for tuner, run, disk in originals:
            tuner.fn.run = run
            tuner.check_disk_cache = disk
    (output/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    return summary
