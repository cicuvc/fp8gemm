"""Complete FP8Linear forward/backward plans: CUDA GEMMs versus Triton.

The native CUDA forward has no bias epilogue: the primary comparison is
bias=False. Bias=True backward is measured separately. Weight refresh remains
Triton and is reported both alone and in a full refresh+forward+backward step.
This benchmarks preallocated kernel plans, not Python/autograd overhead.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mxtest
from experiments.tk_linear_backward import LinearBackwardPlan
from experiments.benchmark_tk_persistent_gemm import graph_benchmark


def error(got, ref, tolerance):
    assert torch.isfinite(got).all().item()
    diff = got.float() - ref.float()
    relative = (diff.norm() / ref.float().norm().clamp_min(1e-20)).item()
    assert relative < tolerance, (relative, tolerance)
    return {"relative_l2": relative, "max_abs": diff.abs().max().item(),
            "bitwise_equal": torch.equal(got, ref)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--shape", nargs=3, type=int, action="append")
    parser.add_argument("--peak", nargs=2, type=float, default=[877.9, 1007.6])
    parser.add_argument("--output", type=Path, default=ROOT / "runs/tk_linear_training/result.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    result = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "rounds": args.rounds, "repeats": args.repeats,
        "peak_fp8a32_tflops_range": args.peak,
        "method": "Preallocated CUDA graphs, rotated measurement order, warm buffers, unlocked clocks. "
                  "All dY preparation, X quantization, dX/dW and final output handling included in backward. "
                  "Cached FP8 weights unless the phase explicitly includes refresh. No optimizer update.",
        "scope": "Native CUDA forward + CUDA-backward plan, bias=False. The public FP8Linear module "
                 "still uses Triton forward; its CUDA-backward combination is also measured. "
                 "Weight refresh and backward X quantization intentionally remain Triton.",
        "mfu_definition": "forward=2MNK, backward=4MNK, forward_backward=6MNK; "
                          "refresh-inclusive MFU uses 6MNK (quantization work excluded from numerator).",
        "cases": [],
    }
    for m, n, k in args.shape or [(256, 2048, 512), (1024, 8192, 1024),
                                  (2048, 16384, 2048), (4096, 4096, 4096),
                                  (8192, 8192, 4096)]:
        torch.manual_seed(123)
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        dy = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
        master = torch.randn((n, k), device="cuda", dtype=torch.float32)
        w, ws = mxtest.launch_weight_quantize(master)
        bp = LinearBackwardPlan(dy, x, w, ws, bias=False)
        bias_bp = LinearBackwardPlan(dy, x, w, ws, bias=True)
        gemm = bp.dx_plan.owner
        tasks_c128 = triton.cdiv(m, 128) * triton.cdiv(n, 128)
        fconfig = 7 if tasks_c128 < result["sm_count"] else 8
        fp = gemm.plan(x, w, fconfig, scales=ws)
        out = torch.empty_like(fp.out)
        tdx, tdw = torch.empty_like(bp.dx), torch.empty_like(bp.dw)
        tdb = torch.empty_like(bias_bp.db)

        def refresh():
            mxtest.fp8_weight_block_quantize[(triton.cdiv(n, 32) * triton.cdiv(k, 32),)](
                master, w, ws, n, k, k, k, ws.stride(0), 32, 32)

        def tfwd():
            mxtest.fp8gemm_fwd[lambda c: (triton.cdiv(m, c["BLOCK_M"]) *
                                         triton.cdiv(n, c["BLOCK_N"]),)](
                x, w, ws, ws, out, m, n, k, k, k, ws.stride(0), n, False)

        def t_dx():
            mxtest.fp8gemm_bwd[lambda c: (triton.cdiv(m, c["BLOCK_M"]) *
                                         triton.cdiv(k, c["BLOCK_K"]),)](
                dy, w, ws, tdx, m, n, k, n, k, ws.stride(0), k)

        def t_dw():
            mxtest.fp8gemm_wgrad[lambda c: (triton.cdiv(n, c["BLOCK_N"]) *
                                           triton.cdiv(k, c["BLOCK_K"]),)](
                dy, x, tdw, m, n, k, n, k, k)

        def tbwd():
            t_dx()
            t_dw()

        def t_db():
            torch.sum(dy, dim=0, dtype=torch.float32, out=tdb)

        def tbwd_bias():
            tbwd()
            t_db()

        def cuda_fb():
            fp()
            bp()

        def triton_fb():
            tfwd()
            tbwd()

        def hybrid_fb():
            tfwd()
            bp()

        def cuda_step():
            refresh()
            cuda_fb()

        def triton_step():
            refresh()
            triton_fb()

        # Compile/autotune everything before graph capture.
        refresh()
        tfwd()
        tbwd_bias()
        cuda_fb()
        bias_bp()
        errors = {"forward": error(fp.out, out, 5e-4),
                  "dx": error(bp.dx, tdx, 5e-4), "dw": error(bp.dw, tdw, 1e-6),
                  "bias_dx": error(bias_bp.dx, tdx, 5e-4),
                  "bias_dw": error(bias_bp.dw, tdw, 1e-6),
                  "db": error(bias_bp.db, tdb, 2e-6)}
        configs = {"forward": str(mxtest.fp8gemm_fwd.best_config),
                   "dx": str(mxtest.fp8gemm_bwd.best_config),
                   "dw": str(mxtest.fp8gemm_wgrad.best_config)}
        # Sustained warm-up before comparing implementations.
        warm = torch.cuda.CUDAGraph()
        with torch.cuda.graph(warm):
            for _ in range(32):
                cuda_fb()
        until = time.monotonic() + .4
        while time.monotonic() < until:
            warm.replay()
            torch.cuda.synchronize()
        functions = {
            "cuda_forward": fp, "triton_forward": tfwd,
            "cuda_backward": bp, "triton_backward": tbwd,
            "cuda_forward_backward": cuda_fb, "triton_forward_backward": triton_fb,
            "module_triton_forward_cuda_backward": hybrid_fb,
            "cuda_refresh_forward_backward": cuda_step,
            "triton_refresh_forward_backward": triton_step,
            "cuda_backward_bias": bias_bp, "triton_backward_bias": tbwd_bias,
        }
        timings = graph_benchmark(functions, args.rounds, args.repeats)
        components = graph_benchmark({
            "weight_refresh_triton": refresh, "prepare_dy": bp.prepare_dy,
            "dx_cuda": bp.dx_plan, "quantize_x_triton": bp.quantize_x,
            "dw_cuda": bp.dw_plan, "prepare_dy_bias": bias_bp.prepare_dy,
            "finish_bias": bias_bp.finish, "dx_triton": t_dx, "dw_triton": t_dw,
            "db_triton": t_db}, args.rounds, args.repeats)
        for name, timing in timings.items():
            factor = 6 if "forward_backward" in name or name.startswith("module_") else (
                2 if "forward" in name else 4)
            tflops = factor * m * n * k / timing["median_us"] / 1e6
            timing.update(effective_tflops=tflops,
                          mfu_percent_range=[100 * tflops / max(args.peak),
                                             100 * tflops / min(args.peak)])
        for name, factor in [("dx_cuda", 2), ("dw_cuda", 2), ("dx_triton", 2), ("dw_triton", 2)]:
            tflops = factor * m * n * k / components[name]["median_us"] / 1e6
            components[name].update(effective_tflops=tflops,
                                   mfu_percent_range=[100*tflops/max(args.peak),
                                                      100*tflops/min(args.peak)])
        row = {"shape_mnk": [m, n, k], "forward_config": fconfig,
               "forward_plan": fp.info, "backward_plan": bp.info,
               "errors_vs_triton": errors, "triton_configs": configs,
               "timings": timings, "components": components,
               "speedup": {phase: timings["triton_" + phase]["median_us"] /
                           timings["cuda_" + phase]["median_us"]
                           for phase in ["forward", "backward", "forward_backward",
                                         "refresh_forward_backward", "backward_bias"]}}
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{m,n,k}: " + "; ".join(
            f"{phase} CUDA={timings['cuda_'+phase]['median_us']:.3f}us "
            f"Triton={timings['triton_'+phase]['median_us']:.3f}us "
            f"({row['speedup'][phase]:.3f}x)"
            for phase in ["forward", "backward", "forward_backward"]), flush=True)
    files = [Path(__file__), ROOT / "mxtest.py", ROOT / "cuda/persistent_fp8_gemm.cu",
             ROOT / "cuda/persistent_dual_online_gemm.cu", ROOT / "cuda/linear_backward_prepare.cu",
             ROOT / "experiments/tk_linear_backward.py"]
    result["sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in files}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
