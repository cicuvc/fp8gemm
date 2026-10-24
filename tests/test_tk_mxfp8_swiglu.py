"""Correctness and replay tests for prequantized-MXFP8 fused SwiGLU."""

from pathlib import Path

import pytest
import torch

from mxfp8._kernels.swiglu import (
    CONFIGS,
    MXFP8SwiGLULibrary,
    OUTPUT,
    ROOT,
    build,
)
from mxfp8._kernels.gemm import pack_a_scales

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


@pytest.fixture(scope="module", autouse=True)
def compiled():
    binary = OUTPUT / "libswiglu.so"
    sources = [
        ROOT / "cuda/persistent_mxfp8_swiglu.cu",
        ROOT / "cuda/mxfp8_swiglu_configs.inc",
        ROOT / "cuda/fp8_shared_load.cuh",
        ROOT / "cuda/tma_ring_pipe.cuh",
        *ROOT.glob("third_party/thunderkittens/include/**/*.cuh"),
    ]
    if not binary.exists() or any(path.stat().st_mtime > binary.stat().st_mtime for path in sources):
        build()


def make_operands(m, n, k, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = (
        torch.randint(0, 15, (m, k), device="cuda", generator=generator) - 7
    ).float().div(8).to(torch.float8_e4m3fn)
    gate = (
        torch.randint(0, 15, (n, k), device="cuda", generator=generator) - 7
    ).float().div(k**0.5 * 8).to(torch.float8_e4m3fn)
    up = (
        torch.randint(0, 15, (n, k), device="cuda", generator=generator) - 7
    ).float().div(k**0.5 * 8).to(torch.float8_e4m3fn)
    x_scale = torch.randint(
        124, 131, (m, k // 32), device="cuda", generator=generator, dtype=torch.int32
    ).to(torch.uint8)
    gate_scale = torch.randint(
        124,
        131,
        (n // 32, k // 32),
        device="cuda",
        generator=generator,
        dtype=torch.int32,
    ).to(torch.uint8)
    up_scale = torch.randint(
        124,
        131,
        (n // 32, k // 32),
        device="cuda",
        generator=generator,
        dtype=torch.int32,
    ).to(torch.uint8)
    if m >= 3:
        x_scale[-3:] = 0
        x[-3:] = 0
    return x, pack_a_scales(x_scale), gate, gate_scale, up, up_scale, x_scale


def reference_accumulators(m, n, k, x, gate, up, x_scale, gate_scale, up_scale):
    x_exp = torch.pow(2.0, (x_scale.double() - 127)[:, :, None])
    dequant_x = (
        x.double().reshape(m, k // 32, 32) * x_exp.repeat_interleave(32, dim=2)
    ).reshape(m, k)

    def dequant_weight(weight, scales):
        scale = torch.pow(2.0, (scales.double() - 127)[:, None, :, None])
        return (
            weight.double().reshape(n // 32, 32, k // 32, 32) * scale
        ).reshape(n, k)

    gate_acc = dequant_x @ dequant_weight(gate, gate_scale).T
    up_acc = dequant_x @ dequant_weight(up, up_scale).T
    return gate_acc, up_acc


def reference(m, n, k, x, gate, up, x_scale, gate_scale, up_scale):
    gate_acc, up_acc = reference_accumulators(
        m, n, k, x, gate, up, x_scale, gate_scale, up_scale
    )
    return (gate_acc * torch.sigmoid(gate_acc) * up_acc).float().bfloat16()


def error(got, expected):
    delta = got.float() - expected.float()
    return {
        "relative_l2": (delta.norm() / expected.float().norm()).item(),
        "max_abs": delta.abs().max().item(),
    }


@pytest.mark.parametrize(
    "m,n,k",
    [(1, 64, 128), (13, 64, 32), (17, 64, 64), (137, 192, 128), (260, 320, 256)],
)
def test_all_configurations_and_tails(m, n, k):
    operands = make_operands(m, n, k, seed=m + n + k)
    x, x_scale, gate, gate_scale, up, up_scale, raw_x_scale = operands
    expected = reference(m, n, k, x, gate, up, raw_x_scale, gate_scale, up_scale)
    gate_expected, up_expected = reference_accumulators(
        m, n, k, x, gate, up, raw_x_scale, gate_scale, up_scale
    )
    gate_expected = (0.5 * gate_expected).float().bfloat16()
    up_expected = up_expected.float().bfloat16()
    lib = MXFP8SwiGLULibrary()
    for config, (_, _, _, block_k, _) in enumerate(CONFIGS):
        if k % block_k:
            continue
        for store_aux in [False, True]:
            out = torch.full(
                (m, n), float("nan"), device="cuda", dtype=torch.bfloat16
            )
            gate_out = (
                torch.full_like(out, float("nan")) if store_aux else None
            )
            up_out = torch.full_like(out, float("nan")) if store_aux else None
            plan = lib.plan(
                x,
                x_scale,
                gate,
                gate_scale,
                up,
                up_scale,
                config=config,
                out=out,
                gate_out=gate_out,
                up_out=up_out,
                sms=3,
            )
            plan()
            torch.cuda.synchronize()
            metrics = error(out, expected)
            assert metrics["relative_l2"] < 1e-4, (
                m,
                n,
                k,
                config,
                store_aux,
                plan.info,
                metrics,
            )
            assert torch.isfinite(out).all()
            if store_aux:
                gate_metrics = error(gate_out, gate_expected)
                up_metrics = error(up_out, up_expected)
                assert gate_metrics["relative_l2"] < 1e-4, (
                    m,
                    n,
                    k,
                    config,
                    gate_metrics,
                )
                assert up_metrics["relative_l2"] < 1e-4, (
                    m,
                    n,
                    k,
                    config,
                    up_metrics,
                )
                assert torch.isfinite(gate_out).all()
                assert torch.isfinite(up_out).all()
            if m >= 3:
                assert (out[-3:] == 0).all()
            print((m, n, k), config, "aux" if store_aux else "fused", plan.info, "PASS")


def test_register_accumulation_matches_scalar_activation():
    m, n, k = 17, 64, 64
    x = torch.zeros((m, k), device="cuda", dtype=torch.float8_e4m3fn)
    x[:, 0] = 1
    x_scale = pack_a_scales(
        torch.full((m, k // 32), 127, device="cuda", dtype=torch.uint8)
    )
    scales = torch.full((n // 32, k // 32), 127, device="cuda", dtype=torch.uint8)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    lib = MXFP8SwiGLULibrary()
    for gate_value, up_value in [(2.0, 3.0), (-2.0, 0.5), (8.0, 1.0)]:
        gate = torch.zeros((n, k), device="cuda", dtype=torch.float8_e4m3fn)
        up = torch.zeros_like(gate)
        gate[:, 0] = gate_value
        up[:, 0] = up_value
        lib.plan(x, x_scale, gate, scales, up, scales, config=0, out=out, sms=3)()
        torch.cuda.synchronize()
        expected = (
            gate_value * torch.sigmoid(torch.tensor(gate_value)) * up_value
        ).bfloat16()
        assert torch.equal(out, torch.full_like(out, expected))


@pytest.mark.parametrize("config", [1, 3])
def test_reuse_stream_and_cuda_graph(config):
    m, n, k = 137, 192, 128
    operands = make_operands(m, n, k, seed=991)
    x, x_scale, gate, gate_scale, up, up_scale, _ = operands
    gate_out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    up_out = torch.empty_like(gate_out)
    plan = MXFP8SwiGLULibrary().plan(
        x,
        x_scale,
        gate,
        gate_scale,
        up,
        up_scale,
        config=config,
        gate_out=gate_out,
        up_out=up_out,
        sms=3,
    )
    baseline = plan().clone()
    gate_baseline = gate_out.clone()
    up_baseline = up_out.clone()

    saved_x = x.clone()
    x.zero_()
    plan()
    torch.cuda.synchronize()
    assert torch.count_nonzero(plan.out) == 0
    assert torch.count_nonzero(gate_out) == 0
    assert torch.count_nonzero(up_out) == 0
    x.copy_(saved_x)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    graph.replay()
    torch.testing.assert_close(plan.out, baseline, rtol=0, atol=0)
    torch.testing.assert_close(gate_out, gate_baseline, rtol=0, atol=0)
    torch.testing.assert_close(up_out, up_baseline, rtol=0, atol=0)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(plan.out, baseline, rtol=0, atol=0)
    torch.testing.assert_close(gate_out, gate_baseline, rtol=0, atol=0)
    torch.testing.assert_close(up_out, up_baseline, rtol=0, atol=0)
