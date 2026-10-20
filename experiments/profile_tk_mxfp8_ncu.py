"""Profile one prepared pure MXFP8 forward launch with --profile-from-start off."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mxtest
from experiments.benchmark_tk_forward_pipeline import measure
from experiments.tk_mxfp8_gemm import Library, OUTPUT, pack_a_scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, default=[4096, 4096, 4096])
    parser.add_argument("--config", type=int, default=8)
    parser.add_argument("--raster-group", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(123)
    m, n, k = args.shape
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    a, a_scales = mxtest.launch_row_quantize(x)
    a_scales = pack_a_scales(a_scales)
    b, b_scales = mxtest.launch_weight_quantize(torch.randn((n, k), device="cuda"))
    plan = Library().plan(a, b, a_scales, b_scales, config=args.config,
                          raster_group=args.raster_group)
    timing = measure({"pure": plan}, "pure", rounds=5, repeats=64, seed=123)
    reference = plan().clone()
    for _ in range(32):
        plan()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    plan()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    torch.testing.assert_close(plan.out, reference, rtol=0, atol=0)
    assert torch.isfinite(plan.out).all().item()
    files = [Path(__file__), ROOT / "cuda/persistent_mxfp8_gemm.cu",
             ROOT / "cuda/tma_ring_pipe.cuh", ROOT / "cuda/linear_tma_buffer.cuh",
             ROOT / "experiments/tk_mxfp8_gemm.py", OUTPUT / "libgemm.so"]
    result = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "shape_mnk": args.shape, "config": args.config, "plan": plan.info,
        "unprofiled_graph_timing": timing,
        "profile_region": "One pure MXFP8 forward GEMM; all quantization, warmup and graph timing precede cudaProfilerStart.",
        "output_bitwise_equal_after_profile": True,
        "sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in files},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
