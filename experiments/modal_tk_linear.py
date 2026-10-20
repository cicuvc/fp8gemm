"""Build/test CUDA FP8Linear kernels on Modal RTX PRO 6000, then benchmark.

python -m modal run experiments/modal_tk_linear.py
python -m modal run experiments/modal_tk_linear.py --tune-dw --rounds 5 --repeats 16
python -m modal run experiments/modal_tk_linear.py --forward-pipeline --rounds 5 --repeats 16
python -m modal run experiments/modal_tk_linear.py --cublas
python -m modal run experiments/modal_tk_linear.py --mxfp8
"""
import hashlib
import json
from pathlib import Path
import subprocess

import modal

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "mxtest.py", "fp8_linear.py", "experiments/__init__.py",
    "experiments/tk_persistent_gemm.py", "experiments/tk_dual_online_gemm.py",
    "experiments/tk_linear_backward.py", "experiments/benchmark_tk_persistent_gemm.py",
    "experiments/benchmark_tk_dual_online_gemm.py",
    "experiments/benchmark_tk_linear_training.py",
    "experiments/benchmark_tk_dw_tuning.py",
    "experiments/benchmark_tk_forward_pipeline.py",
    "experiments/benchmark_cublaslt.py", "experiments/cublaslt_gemm.py",
    "cuda/cublaslt_benchmark.cu",
    "cuda/persistent_mxfp8_gemm.cu", "cuda/mxfp8_configs.inc", "experiments/tk_mxfp8_gemm.py",
    "experiments/benchmark_tk_mxfp8.py", "tests/test_tk_mxfp8_gemm.py",
    "cuda/linear_tma_buffer.cuh", "cuda/fp8_shared_load.cuh",
    "tests/test_tk_forward_pipeline.py",
    "tests/test_tk_linear_backward.py", "tests/test_tk_group_raster.py",
    "cuda/persistent_fp8_gemm.cu", "cuda/persistent_dual_online_gemm.cu",
    "cuda/linear_backward_prepare.cu", "cuda/tma_ring_pipe.cuh",
]
image = (
    modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .pip_install("torch==2.13.0", "triton==3.7.1",
                 extra_index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("pytest==9.0.2", "numpy")
    .env({"PYTHONPATH": "/workspace", "TRITON_CACHE_DIR": "/cache/triton",
          "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor-linear",
          "PYTHONUNBUFFERED": "1"})
)
for source in SOURCES:
    image = image.add_local_file(ROOT / source, "/workspace/" + source)
image = image.add_local_dir(ROOT / "third_party/thunderkittens",
                            "/workspace/third_party/thunderkittens")
app = modal.App("fp8-linear-cuda-rtx-pro-6000")
cache = modal.Volume.from_name("fp8-swiglu-triton-cache", create_if_missing=True)


@app.function(image=image, gpu="RTX-PRO-6000", cpu=8, memory=32768,
              timeout=3600, volumes={"/cache": cache})
def measure(manifest, rounds, repeats, tune_dw=False, groups=None, configs=None, backward_sweep=False,
            forward_pipeline=False, baseline_sources=None, cublas=False, mxfp8=False):
    import os
    import sys
    from datetime import datetime, timezone
    import torch
    import triton

    os.chdir("/workspace")
    for name, expected in manifest["sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    assert torch.cuda.get_device_capability() == (12, 0)
    assert "RTX PRO 6000" in torch.cuda.get_device_name()
    outdir = Path("/cache/linear_cuda_results") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir.mkdir(parents=True)
    metadata = {
        "utc": datetime.now(timezone.utc).isoformat(), "source": manifest,
        "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__, "triton": triton.__version__,
        "torch_cuda": torch.version.cuda,
        "nvcc": subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True),
        "nvidia_smi_before": subprocess.check_output(["nvidia-smi"], text=True),
        "volume_output": str(outdir),
    }
    print(json.dumps(metadata, indent=2), flush=True)
    (outdir / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logs = {}

    def run(label, command):
        print(f"Running {label}", flush=True)
        with (outdir / f"{label}.log").open("w") as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = process.wait()
        logs[label] = (outdir / f"{label}.log").read_text()
        if label == "build" or code:
            for build_log in Path("runs").glob("*/build.log"):
                key = build_log.parent.name + "_build"
                logs[key] = build_log.read_text()
                (outdir / f"{key}.log").write_text(logs[key])
                if code:
                    print(logs[key][:12000] + "\n...\n" + logs[key][-4000:], flush=True)
        cache.commit()
        if code:
            raise RuntimeError(f"{label} failed ({code}), logs: {outdir}")

    run("build", [sys.executable, "-c",
                  "from experiments.tk_linear_backward import libraries; libraries(); print('CUDA libraries ready')"])
    if cublas:
        run("cublas_build", [sys.executable, "-c",
                            "from experiments.cublaslt_gemm import build; build()"])
    if mxfp8:
        run("mxfp8_build", [sys.executable, "-c",
                           "from experiments.tk_mxfp8_gemm import build; build()"])
    if forward_pipeline:
        from experiments.tk_persistent_gemm import FLAGS
        baseline_dir = Path("runs/tk_forward_pipeline/old")
        baseline_dir.mkdir(parents=True, exist_ok=True)
        for name, content in baseline_sources.items():
            (baseline_dir / name).write_text(content)
        run("baseline_build", ["/usr/local/cuda/bin/nvcc", *FLAGS, "-Xptxas=-v",
                               "-shared", "-Xcompiler=-fPIC",
                               str(baseline_dir / "persistent_fp8_gemm.cu"), "-lcuda",
                               "-o", str(baseline_dir / "libgemm.so")])
    for name in ["tk_persistent_gemm", "tk_dual_online_gemm", "tk_linear_backward", "tk_mxfp8_gemm"]:
        path = Path("runs") / name / "build.log"
        if path.exists():
            logs[name + "_build"] = path.read_text()
            (outdir / f"{name}_build.log").write_text(logs[name + "_build"])
    test_files = ["tests/test_tk_linear_backward.py", "tests/test_tk_group_raster.py"]
    if forward_pipeline:
        test_files.append("tests/test_tk_forward_pipeline.py")
    if mxfp8:
        test_files.append("tests/test_tk_mxfp8_gemm.py")
    run("tests", [sys.executable, "-m", "pytest", "-q", *test_files])
    if not tune_dw and not forward_pipeline and not cublas and not mxfp8:
        run("forward_validation", [sys.executable, "experiments/benchmark_tk_persistent_gemm.py",
                                  "--skip-build", "--validate-only"])
    script = ("benchmark_tk_mxfp8.py" if mxfp8 else
              "benchmark_cublaslt.py" if cublas else
              "benchmark_tk_forward_pipeline.py" if forward_pipeline else
              "benchmark_tk_dw_tuning.py" if tune_dw else "benchmark_tk_linear_training.py")
    command = [sys.executable, "experiments/" + script,
               "--rounds", str(rounds), "--repeats", str(repeats),
               "--output", str(outdir / "benchmark.json")]
    if forward_pipeline or mxfp8:
        command += ["--peak-tflops", "877.9", "1007.6"]
    if tune_dw or forward_pipeline:
        if backward_sweep:
            command += ["--backward-sweep"]
        if groups:
            command += ["--groups", *map(str, groups)]
        if configs:
            command += ["--configs", *map(str, configs)]
    run("benchmark", command)
    result = json.loads((outdir / "benchmark.json").read_text())
    metadata["nvidia_smi_after"] = subprocess.check_output(["nvidia-smi"], text=True)
    result["modal_environment"] = metadata
    result["validation"] = {
        "tests": logs["tests"],
    }
    if not tune_dw and not forward_pipeline and not cublas and not mxfp8:
        result["validation"]["forward"] = json.loads(
            Path("runs/tk_persistent_gemm/validation.json").read_text())
    (outdir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    cache.commit()
    return result, logs


@app.local_entrypoint()
def main(rounds: int = 9, repeats: int = 64,
         output: str = "", tune_dw: bool = False, groups: str = "", configs: str = "",
         backward_sweep: bool = False, forward_pipeline: bool = False, cublas: bool = False,
         mxfp8: bool = False):
    if sum([tune_dw, forward_pipeline, cublas, mxfp8]) > 1:
        raise ValueError("Choose one benchmark mode")
    if backward_sweep and not tune_dw:
        raise ValueError("--backward-sweep requires --tune-dw")
    if (groups or configs) and not (tune_dw or forward_pipeline):
        raise ValueError("--groups/--configs require a tuning mode")
    files = [ROOT / p for p in SOURCES]
    files += [p for p in (ROOT / "third_party/thunderkittens").rglob("*") if p.is_file()]
    manifest = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    baseline_sources = None
    if forward_pipeline:
        baseline_commit = "6400254"
        baseline_sources = {name: subprocess.check_output(
            ["git", "show", f"{baseline_commit}:cuda/{name}"], cwd=ROOT, text=True)
            for name in ["persistent_fp8_gemm.cu", "tma_ring_pipe.cuh"]}
        manifest["baseline"] = {
            "commit": baseline_commit,
            "sha256": {name: hashlib.sha256(content.encode()).hexdigest()
                       for name, content in baseline_sources.items()}}
    result, logs = measure.remote(manifest, rounds, repeats, tune_dw,
                                 [int(x) for x in groups.split(",")] if groups else None,
                                 [int(x) for x in configs.split(",")] if configs else None,
                                 backward_sweep, forward_pipeline, baseline_sources, cublas, mxfp8)
    path = Path(output or ("benchmarks/tk_mxfp8_rtx_pro_6000.json" if mxfp8 else
                           "benchmarks/cublaslt_rtx_pro_6000.json" if cublas else
                           "benchmarks/tk_forward_pipeline_rtx_pro_6000.json" if forward_pipeline else
                           "benchmarks/tk_dw_tuning_rtx_pro_6000.json" if tune_dw else
                           "benchmarks/tk_linear_training_rtx_pro_6000.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    for label, log in logs.items():
        path.with_suffix(f".{label}.log").write_text(log)
    print(f"Saved {path}")
