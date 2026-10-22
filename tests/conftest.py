"""Shared CUDA binaries for the pure-MXFP8 test suite."""

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def compiled_mxfp8_kernels():
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (12, 0)
    ):
        return

    from experiments import (
        tk_mxfp8_dgrad,
        tk_mxfp8_gemm,
        tk_mxfp8_lse,
        tk_mxfp8_swiglu,
        tk_mxfp8_swiglu_dgrad,
        tk_mxfp8_wgrad,
        tk_silu_bwd,
        tk_weight_quantize,
    )

    build_specs = [
        (
            tk_mxfp8_gemm.build,
            tk_mxfp8_gemm.OUTPUT / "libgemm.so",
            [
                tk_mxfp8_gemm.ROOT / "cuda/persistent_mxfp8_gemm.cu",
                tk_mxfp8_gemm.ROOT / "cuda/mxfp8_configs.inc",
                tk_mxfp8_gemm.ROOT / "cuda/mxfp8_epilogue.cuh",
                tk_mxfp8_gemm.ROOT / "cuda/mxfp8_lse.cuh",
                tk_mxfp8_gemm.ROOT
                / "third_party/thunderkittens/include/ops/group/register/tile/online_scale.cuh",
            ],
        ),
        (
            tk_mxfp8_lse.build,
            tk_mxfp8_lse.OUTPUT / "liblse.so",
            [
                tk_mxfp8_lse.ROOT / "cuda/persistent_mxfp8_lse.cu",
                tk_mxfp8_lse.ROOT / "cuda/persistent_mxfp8_gemm.cu",
                tk_mxfp8_lse.ROOT / "cuda/mxfp8_lse.cuh",
                tk_mxfp8_lse.ROOT / "cuda/mxfp8_lse_configs.inc",
                tk_mxfp8_lse.ROOT / "cuda/mxfp8_epilogue.cuh",
            ],
        ),
        (
            tk_mxfp8_dgrad.build,
            tk_mxfp8_dgrad.OUTPUT / "libdgrad.so",
            [
                tk_mxfp8_dgrad.ROOT / "cuda/persistent_mxfp8_dgrad.cu",
                tk_mxfp8_dgrad.ROOT / "cuda/mxfp8_dgrad_configs.inc",
            ],
        ),
        (
            tk_mxfp8_wgrad.build,
            tk_mxfp8_wgrad.OUTPUT / "libwgrad.so",
            [
                tk_mxfp8_wgrad.ROOT / "cuda/persistent_mxfp8_wgrad.cu",
                tk_mxfp8_wgrad.ROOT / "cuda/mxfp8_wgrad_configs.inc",
            ],
        ),
        (
            tk_mxfp8_swiglu.build,
            tk_mxfp8_swiglu.OUTPUT / "libswiglu.so",
            [
                tk_mxfp8_swiglu.ROOT / "cuda/persistent_mxfp8_swiglu.cu",
                tk_mxfp8_swiglu.ROOT / "cuda/mxfp8_swiglu_configs.inc",
            ],
        ),
        (
            tk_mxfp8_swiglu_dgrad.build,
            tk_mxfp8_swiglu_dgrad.OUTPUT / "libswd.so",
            [
                tk_mxfp8_swiglu_dgrad.ROOT
                / "cuda/persistent_mxfp8_swiglu_dgrad.cu",
                tk_mxfp8_swiglu_dgrad.ROOT
                / "cuda/mxfp8_swiglu_dgrad_configs.inc",
            ],
        ),
        (
            tk_silu_bwd.build,
            tk_silu_bwd.OUTPUT / "libsilu_bwd_quantize.so",
            [
                tk_silu_bwd.ROOT / "cuda/silu_bwd_elementwise.cu",
                tk_silu_bwd.ROOT / "cuda/silu_bwd_quantize.cu",
                tk_silu_bwd.ROOT / "cuda/silu_bwd_common.cuh",
                tk_silu_bwd.ROOT / "cuda/mxfp8_quantize.cuh",
            ],
        ),
        (
            tk_weight_quantize.build,
            tk_weight_quantize.DEFAULT_OUTPUT / "libquant.so",
            [
                tk_weight_quantize.ROOT / "cuda/persistent_weight_quantize.cu",
                tk_weight_quantize.ROOT / "cuda/mxfp8_quantize.cuh",
            ],
        ),
    ]
    for build, binary, sources in build_specs:
        if not binary.exists() or any(
            source.stat().st_mtime > binary.stat().st_mtime
            for source in sources
            if source.exists()
        ):
            build()
