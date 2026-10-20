"""Correctness, replay, and codegen tests for the all-BF16 SiLU backward."""

from collections import Counter
from pathlib import Path
import re
import subprocess

import pytest
import torch

from experiments.tk_silu_bwd import OUTPUT, ROOT, SiluBwdLibrary, build

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


@pytest.fixture(scope="module", autouse=True)
def compiled():
    binary = OUTPUT / "libsilu_bwd.so"
    source = ROOT / "cuda/silu_bwd_elementwise.cu"
    if not binary.exists() or source.stat().st_mtime > binary.stat().st_mtime:
        build()


def reference(dy, half_gate, up):
    z = half_gate.double()
    u = up.double()
    g = z * 2
    tanh = torch.tanh(z)
    a = z + z * tanh
    b = 0.5 - 0.5 * tanh
    c = 0.5 + 0.5 * tanh
    return (
        (dy.double() * u * (a * b + c)).float().bfloat16(),
        (dy.double() * a).float().bfloat16(),
    )


def metrics(got, expected):
    delta = got.float() - expected.float()
    expected_f = expected.float()
    return {
        "relative_l2": (delta.norm() / expected_f.norm().clamp_min(1e-30)).item(),
        "max_abs": delta.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            got.float().flatten(), expected_f.flatten(), dim=0
        ).item(),
    }


@pytest.mark.parametrize("scale", [0.1, 1.0, 2.0, 4.0, 8.0])
def test_precision(scale):
    shape = (1024, 1024)
    torch.manual_seed(1009 + int(scale * 10))
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    half_gate = (torch.randn(shape, device="cuda") * scale).bfloat16()
    up = (torch.randn(shape, device="cuda") * scale).bfloat16()
    expected_gate, expected_up = reference(dy, half_gate, up)
    dgate, dup = SiluBwdLibrary().plan(dy, half_gate, up)()
    torch.cuda.synchronize()
    gate_metrics = metrics(dgate, expected_gate)
    up_metrics = metrics(dup, expected_up)
    gate_limit = 0.01 if scale <= 2 else 0.035 if scale <= 4 else 0.06
    assert gate_metrics["relative_l2"] < gate_limit, gate_metrics
    assert up_metrics["relative_l2"] < 0.005, up_metrics
    gate_cosine_limit = 0.9995 if scale >= 4 else 0.9998
    assert gate_metrics["cosine"] > gate_cosine_limit, gate_metrics
    assert up_metrics["cosine"] > 0.9998, up_metrics
    assert torch.isfinite(dgate).all() and torch.isfinite(dup).all()


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (17, 129)])
def test_small_shapes(shape):
    torch.manual_seed(shape[0] * 1009 + shape[1])
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    half_gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    expected_gate, expected_up = reference(dy, half_gate, up)
    dgate, dup = SiluBwdLibrary().plan(dy, half_gate, up)()
    torch.cuda.synchronize()
    assert metrics(dgate, expected_gate)["relative_l2"] < 0.1
    assert metrics(dup, expected_up)["relative_l2"] < 0.1
    assert torch.isfinite(dgate).all() and torch.isfinite(dup).all()


def test_extreme_values_and_tail():
    half_gate = torch.tensor(
        [-16, -8, -4, -2, -1, -0.0, 0.0, 1, 2, 4, 8, 16, -0.5, 0.5, 3],
        device="cuda",
        dtype=torch.bfloat16,
    )
    up = torch.linspace(
        -2, 2, half_gate.numel(), device="cuda", dtype=torch.bfloat16
    )
    dy = torch.linspace(
        2, -2, half_gate.numel(), device="cuda", dtype=torch.bfloat16
    )
    expected_gate, expected_up = reference(dy, half_gate, up)
    dgate, dup = SiluBwdLibrary().plan(dy, half_gate, up)()
    torch.cuda.synchronize()
    assert metrics(dgate, expected_gate)["relative_l2"] < 0.035
    assert metrics(dup, expected_up)["relative_l2"] < 0.005
    assert torch.isfinite(dgate).all() and torch.isfinite(dup).all()


def test_reuse_stream_and_cuda_graph():
    torch.manual_seed(991)
    shape = (257, 1031)
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    half_gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    plan = SiluBwdLibrary().plan(dy, half_gate, up)
    baseline_gate, baseline_up = (value.clone() for value in plan())

    saved_gate = half_gate.clone()
    half_gate.zero_()
    plan()
    torch.cuda.synchronize()
    expected_gate, expected_up = reference(dy, half_gate, up)
    assert metrics(plan.dgate, expected_gate)["relative_l2"] < 0.01
    assert torch.count_nonzero(plan.dup) == 0
    half_gate.copy_(saved_gate)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    graph.replay()
    torch.testing.assert_close(plan.dgate, baseline_gate, rtol=0, atol=0)
    torch.testing.assert_close(plan.dup, baseline_up, rtol=0, atol=0)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(plan.dgate, baseline_gate, rtol=0, atol=0)
    torch.testing.assert_close(plan.dup, baseline_up, rtol=0, atol=0)


def test_codegen_uses_packed_bf16_instructions():
    sass = subprocess.check_output(
        [
            "/usr/local/cuda/bin/cuobjdump",
            "--dump-sass",
            str(OUTPUT / "libsilu_bwd.so"),
        ],
        text=True,
    )
    section = sass.split("Function : ")[1]
    instructions = Counter(
        re.findall(
            r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)",
            section,
        )
    )
    assert instructions["HFMA2.BF16_V2"] >= 16
    assert instructions["HMUL2.BF16_V2"] >= 8
    assert instructions["MUFU.TANH.BF16"] >= 8
    assert "LDL" not in " ".join(instructions)
    assert "STL" not in " ".join(instructions)
