"""Exact and replay tests for fused dual-gradient MXFP8 dX."""

from collections import Counter
import re
import subprocess

import pytest
import torch

from mxfp8._kernels.gemm import pack_a_scales
from mxfp8._kernels.swiglu_dgrad import (
    CONFIGS,
    OUTPUT,
    ROOT,
    SwiGLUDgradLibrary,
    build,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


@pytest.fixture(scope="module", autouse=True)
def compiled():
    binary = OUTPUT / "libswd.so"
    sources = [
        ROOT / "cuda/persistent_mxfp8_swiglu_dgrad.cu",
        ROOT / "cuda/mxfp8_swiglu_dgrad_configs.inc",
        ROOT / "cuda/fp8_shared_load.cuh",
        ROOT / "cuda/tma_ring_pipe.cuh",
        *ROOT.glob("third_party/thunderkittens/include/**/*.cuh"),
    ]
    if not binary.exists() or any(path.stat().st_mtime > binary.stat().st_mtime for path in sources):
        build()


def exact_inputs(m, n, k):
    gate_index = torch.arange(m * n).reshape(m, n)
    up_index = torch.arange(m * n).reshape(m, n)
    weight_gate_index = torch.arange(n * k).reshape(n, k)
    weight_up_index = torch.arange(n * k).reshape(n, k)
    gate = ((gate_index * 13 % 17 - 8).float() / 4).to(torch.float8_e4m3fn)
    up = ((up_index * 7 % 17 - 8).float() / 4).to(torch.float8_e4m3fn)
    weight_gate = (
        (weight_gate_index * 11 % 17 - 8).float() / 4
    ).to(torch.float8_e4m3fn)
    weight_up = (
        (weight_up_index * 5 % 17 - 8).float() / 4
    ).to(torch.float8_e4m3fn)
    gate_scale = (
        (torch.arange(m)[:, None] * 3 + torch.arange(n // 32)[None, :] * 7) % 5
        + 125
    ).byte()
    up_scale = (
        (torch.arange(m)[:, None] * 5 + torch.arange(n // 32)[None, :] * 3) % 5
        + 125
    ).byte()
    weight_gate_scale = (
        (torch.arange(n // 32)[:, None] * 7 + torch.arange(k // 32)[None, :] * 3)
        % 5
        + 125
    ).byte()
    weight_up_scale = (
        (torch.arange(n // 32)[:, None] * 3 + torch.arange(k // 32)[None, :] * 7)
        % 5
        + 125
    ).byte()
    gate_dequant = torch.ldexp(
        gate.double(), (gate_scale.int() - 127).repeat_interleave(32, 1)
    )
    up_dequant = torch.ldexp(
        up.double(), (up_scale.int() - 127).repeat_interleave(32, 1)
    )

    def dequant_weight(weight, scales):
        scale = torch.ldexp(
            torch.ones(
                (n // 32, 1, k // 32, 1),
                dtype=torch.float64,
                device=weight.device,
            ),
            (scales.int() - 127)[:, None, :, None],
        )
        return (
            weight.double().reshape(n // 32, 32, k // 32, 32) * scale
        ).reshape(n, k)

    reference = (
        gate_dequant @ dequant_weight(weight_gate, weight_gate_scale)
        + up_dequant @ dequant_weight(weight_up, weight_up_scale)
    ).bfloat16()
    return {
        "gate": gate.cuda(),
        "up": up.cuda(),
        "weight_gate": weight_gate.cuda(),
        "weight_up": weight_up.cuda(),
        "gate_scale": pack_a_scales(gate_scale.cuda()),
        "up_scale": pack_a_scales(up_scale.cuda()),
        "weight_gate_scale": weight_gate_scale.cuda(),
        "weight_up_scale": weight_up_scale.cuda(),
        "reference": reference.cuda(),
    }


@pytest.mark.parametrize("shape", [(1, 128, 128), (137, 256, 192), (260, 320, 256)])
def test_all_configurations_exact(shape):
    m, n, k = shape
    values = exact_inputs(m, n, k)
    reference = values.pop("reference")
    lib = SwiGLUDgradLibrary()
    for config, (_, _, output_columns, block_n, _) in enumerate(CONFIGS):
        if n % block_n or k % min(128, output_columns):
            continue
        out = torch.full_like(reference, float("nan"))
        plan = lib.plan(*values.values(), config=config, out=out, sms=3)
        torch.testing.assert_close(plan(), reference, rtol=0, atol=0)
        assert torch.equal(out, reference)
        print(shape, config, plan.info, "PASS")


def test_reuse_stream_and_cuda_graph():
    m, n, k = 137, 256, 256
    values = exact_inputs(m, n, k)
    reference = values.pop("reference")
    plan = SwiGLUDgradLibrary().plan(*values.values(), config=6, sms=3)
    baseline = plan().clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    saved_gate = plan.gate.clone()
    saved_up = plan.up.clone()
    plan.gate.zero_()
    plan.up.zero_()
    graph.replay()
    assert torch.count_nonzero(plan.out) == 0
    plan.gate.copy_(saved_gate)
    plan.up.copy_(saved_up)
    graph.replay()
    torch.testing.assert_close(plan.out, reference, rtol=0, atol=0)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(plan.out, baseline, rtol=0, atol=0)


def test_codegen():
    sass = subprocess.check_output(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(OUTPUT / "libswd.so")],
        text=True,
    )
    sections = sass.split("Function : ")[1:]
    assert len(sections) == len(CONFIGS)
    for section in sections:
        instructions = Counter(
            re.findall(
                r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)",
                section,
            )
        )
        assert instructions["QMMA.SF.16832.F32.E4M3.E4M3.E8"] > 0
        assert instructions["UTMALDG.5D"] >= 6
        assert not any(op.startswith(("LDL", "STL")) for op in instructions)
