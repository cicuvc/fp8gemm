"""Correctness, replay, and codegen tests for the all-BF16 SiLU backward."""

from collections import Counter
from pathlib import Path
import re
import subprocess

import pytest
import torch

from experiments.tk_silu_bwd import (
    OUTPUT,
    ROOT,
    SiluBwdLibrary,
    SiluBwdQuantizeLibrary,
    build,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


@pytest.fixture(scope="module", autouse=True)
def compiled():
    binaries = [OUTPUT / "libsilu_bwd.so", OUTPUT / "libsilu_bwd_quantize.so"]
    sources = [
        ROOT / "cuda/silu_bwd_common.cuh",
        ROOT / "cuda/silu_bwd_elementwise.cu",
        ROOT / "cuda/silu_bwd_quantize.cu",
        ROOT / "cuda/mxfp8_quantize.cuh",
    ]
    if any(
        not binary.exists()
        or any(source.stat().st_mtime > binary.stat().st_mtime for source in sources)
        for binary in binaries
    ):
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
    torch.cuda.synchronize()


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
            r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Za-z0-9_.]+)",
            section,
        )
    )
    assert instructions["HFMA2.BF16_V2"] >= 16
    assert instructions["HMUL2.BF16_V2"] >= 8
    assert instructions["MUFU.TANH.BF16"] >= 8
    assert "LDL" not in " ".join(instructions)
    assert "STL" not in " ".join(instructions)


@pytest.mark.parametrize("shape", [(1, 32), (17, 128), (137, 256)])
def test_quantized_outputs_match_existing_quantizers(shape):
    torch.cuda.synchronize()
    m, n = shape
    torch.manual_seed(173)
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    half_gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    reference_gate, reference_up = SiluBwdLibrary().plan(dy, half_gate, up)()
    torch.cuda.synchronize()

    expected = []
    for reference in [reference_gate, reference_up]:
        (
            expected_row,
            expected_row_scale,
            expected_col,
            expected_col_scale,
        ) = SiluBwdQuantizeLibrary().plan_direct(reference, transpose=True)()
        torch.cuda.synchronize()
        expected.append(
            (expected_row, expected_row_scale, expected_col, expected_col_scale)
        )

    plan = SiluBwdQuantizeLibrary().plan(dy, half_gate, up)
    (
        gate_row,
        gate_row_scale,
        up_row,
        up_row_scale,
        gate_col,
        gate_col_scale,
        up_col,
        up_col_scale,
    ) = (output.clone() for output in plan())
    torch.cuda.synchronize()
    for branch, outputs, expected_outputs in [
        (
            "gate",
            (gate_row, gate_row_scale, gate_col, gate_col_scale),
            expected[0],
        ),
        (
            "up",
            (up_row, up_row_scale, up_col, up_col_scale),
            expected[1],
        ),
    ]:
        for name, actual, expected_value in zip(
            ["row", "row_scale", "col", "col_scale"],
            outputs,
            expected_outputs,
        ):
            if actual.dtype == torch.float8_e4m3fn:
                actual = actual.view(torch.uint8)
                expected_value = expected_value.view(torch.uint8)
            torch.testing.assert_close(
                actual,
                expected_value,
                rtol=0,
                atol=0,
                msg=lambda message: f"{branch}.{name}: {message}",
            )


def test_quantized_reuse_and_cuda_graph():
    shape = (137, 128)
    torch.manual_seed(117)
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    half_gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    plan = SiluBwdQuantizeLibrary().plan(dy, half_gate, up)
    baseline = [output.clone() for output in plan()]

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    saved_dy = dy.clone()
    dy.zero_()
    graph.replay()
    torch.cuda.synchronize()
    for output in [
        plan.gate_row,
        plan.up_row,
        plan.gate_col,
        plan.up_col,
        plan.gate_row_scale,
        plan.up_row_scale,
        plan.gate_col_scale,
        plan.up_col_scale,
    ]:
        assert torch.count_nonzero(output) == 0
    dy.copy_(saved_dy)
    graph.replay()
    for actual, expected in zip(plan(), baseline):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_quantized_codegen():
    sass = subprocess.check_output(
        [
            "/usr/local/cuda/bin/cuobjdump",
            "--dump-sass",
            str(OUTPUT / "libsilu_bwd_quantize.so"),
        ],
        text=True,
    )
    section = next(
        block
        for block in sass.split("Function : ")[1:]
        if "ILb0ELb0EEE" in block.splitlines()[0]
    )
    instructions = Counter(
        re.findall(
            r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Za-z0-9_.]+)",
            section,
        )
    )
    for opcode in [
        "HFMA2.BF16_V2",
        "HMUL2.BF16_V2",
        "MUFU.TANH.BF16",
        "REDUX.MAX",
        "VIMNMX.U16x2",
        "VIADD.16x2",
    ]:
        assert instructions[opcode] > 0
    assert any(opcode.startswith("LDGSTS.") for opcode in instructions)
    assert not any(op.startswith(("LDL", "STL")) for op in instructions)
    direct = next(
        block
        for block in sass.split("Function : ")[1:]
        if "ILb1ELb0EEE" in block.splitlines()[0]
    )
    direct_instructions = Counter(
        re.findall(
            r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Za-z0-9_.]+)",
            direct,
        )
    )
    assert any(op.startswith("LDGSTS.") for op in direct_instructions)
    assert direct_instructions["MUFU.TANH.BF16"] == 0


@pytest.mark.parametrize("shape", [(1, 32), (17, 128), (137, 192)])
def test_direct_cuda_dual_quantizer(shape):
    torch.manual_seed(66)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    plan = SiluBwdQuantizeLibrary().plan_direct(x)
    transposed = SiluBwdQuantizeLibrary().plan_direct(x, transpose=True)
    row, row_scale, column, column_scale = plan()
    expected_row, expected_row_scale, expected_column, expected_column_scale = (
        transposed()
    )
    torch.testing.assert_close(
        row.view(torch.uint8), expected_row.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(row_scale, expected_row_scale, rtol=0, atol=0)
    torch.testing.assert_close(
        column.view(torch.uint8),
        expected_column.T.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(column_scale, expected_column_scale, rtol=0, atol=0)
