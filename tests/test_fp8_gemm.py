"""FP32 1x128 / 128x128 scales, K-block promotion, and SM120 execution."""

import pytest
import torch

from mxfp8 import FP8GemmPlan, fp8_gemm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


def operands(m, n, k, *, exact=False):
    generator = torch.Generator(device="cuda").manual_seed(1234 + m + n + k)
    a = (
        torch.randint(-8, 9, (m, k), generator=generator, device="cuda").float() / 4
    ).to(torch.float8_e4m3fn)
    b = (
        torch.randint(-8, 9, (n, k), generator=generator, device="cuda").float() / 4
    ).to(torch.float8_e4m3fn)
    sa = torch.rand((m, k // 128), generator=generator, device="cuda") * 1.7 + 0.13
    sb = (
        torch.rand((n // 128, k // 128), generator=generator, device="cuda") * 1.3
        + 0.21
    )
    if exact:
        sa = (
            torch.randint(-2, 3, sa.shape, generator=generator, device="cuda").float()
            / 2
        )
        sb = (
            torch.randint(-2, 3, sb.shape, generator=generator, device="cuda").float()
            / 2
        )
    return a, b, sa, sb


def reference(a, b, sa, sb):
    # FP64 represented operands, independent of the kernel's accumulator layout.
    ad = a.double() * sa.double().repeat_interleave(128, 1)
    bd = b.double() * sb.double().repeat_interleave(128, 0).repeat_interleave(128, 1)
    return ad @ bd.T


@pytest.mark.parametrize("config", range(4))
@pytest.mark.parametrize("shape", [(1, 128, 128), (65, 256, 384), (257, 384, 640)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_exact(config, shape, dtype):
    args = operands(*shape, exact=True)
    plan = FP8GemmPlan(*args, config=config, dtype=dtype, sms=2, raster_group=3)
    torch.testing.assert_close(plan(), reference(*args).to(dtype), rtol=0, atol=0)
    assert plan.info["local_bytes"] == 0


@pytest.mark.parametrize("config", range(4))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_all_finite_fp8_encodings(config, dtype):
    # Constant rows exercise every pair of finite FP8 values, including
    # subnormals. K128 sums and the chosen scale products are exact in FP32.
    bits = torch.arange(256, device="cuda").byte()
    bits[(bits & 127) == 127] = 0
    values = bits.view(torch.float8_e4m3fn)
    a = values[:, None].repeat(1, 256)
    b = values.flip(0)[:, None].repeat(1, 256)
    sa = torch.tensor([1.0, 2.0], device="cuda").repeat(256, 1)
    sb = torch.tensor([[1.0, 0.25], [2.0, 0.5]], device="cuda")
    actual = FP8GemmPlan(a, b, sa, sb, config=config, dtype=dtype)()
    torch.testing.assert_close(
        actual, reference(a, b, sa, sb).to(dtype), rtol=0, atol=0
    )


@pytest.mark.parametrize("config", range(4))
def test_non_power_of_two_scales(config):
    args = operands(137, 384, 1152)
    ref = reference(*args)
    plan = FP8GemmPlan(*args, config=config, dtype=torch.float32)
    torch.testing.assert_close(plan(), ref.float(), rtol=2e-5, atol=2e-4)
    # BF16 rounding can straddle a midpoint because the reference uses FP64.
    rounded = FP8GemmPlan(*args, config=config)()
    torch.testing.assert_close(rounded.float(), ref.float(), rtol=4e-3, atol=2e-4)


def test_k_blocks_are_scaled_before_accumulating():
    a = torch.ones((17, 256), device="cuda").to(torch.float8_e4m3fn)
    b = torch.ones((256, 256), device="cuda").to(torch.float8_e4m3fn)
    sa = torch.tensor([0.5, 2.0], device="cuda").repeat(17, 1)
    sb = torch.tensor([[3.0, 0.25], [0.125, 4.0]], device="cuda")
    out = fp8_gemm(a, b, sa, sb, dtype=torch.float32)
    expected = torch.cat(
        (torch.full((17, 128), 256.0), torch.full((17, 128), 1032.0)), 1
    ).cuda()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_training_range_and_unaligned_scale_storage():
    torch.manual_seed(4321)
    m, n, k = 37, 256, 4096
    a = (torch.randn((m, k), device="cuda") * 40).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device="cuda") * 40).to(torch.float8_e4m3fn)
    sa = (torch.rand(m * (k // 128) + 1, device="cuda") * 0.02 + 0.01)[1:].reshape(
        m, -1
    )
    sb = (torch.rand((n // 128) * (k // 128) + 1, device="cuda") * 0.02 + 0.01)[
        1:
    ].reshape(n // 128, -1)
    ref = reference(a, b, sa, sb).float()
    actual = fp8_gemm(a, b, sa, sb, dtype=torch.float32)
    torch.testing.assert_close(actual, ref, rtol=3e-5, atol=2e-4)


def test_graph_reads_updated_scales_and_operands():
    args = operands(129, 256, 384, exact=True)
    out = torch.empty((129, 256), device="cuda", dtype=torch.float32)
    plan = FP8GemmPlan(*args, out=out, sms=1)
    assert plan() is out
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    for index in (2, 3):
        args[index].mul_(0.5)
        graph.replay()
        torch.testing.assert_close(out, reference(*args).float(), rtol=0, atol=0)
    args[0].zero_()
    graph.replay()
    assert torch.count_nonzero(out) == 0


def test_nondefault_stream():
    args = operands(33, 128, 256, exact=True)
    plan = FP8GemmPlan(*args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        result = plan()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(result, reference(*args).bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind", ["shape", "dtype", "noncontiguous", "grad", "n", "k", "alias"]
)
def test_reject_invalid_inputs(kind):
    a, b, sa, sb = operands(128, 128, 256)
    options = {}
    if kind == "shape":
        sa = sa[:, :1].contiguous()
    elif kind == "dtype":
        sb = sb.bfloat16()
    elif kind == "noncontiguous":
        a = a.T.contiguous().T
    elif kind == "grad":
        sa.requires_grad_()
    elif kind == "n":
        b = b[:64]
    elif kind == "k":
        a, b = a[:, :96].contiguous(), b[:, :96].contiguous()
    elif kind == "alias":
        options["out"] = a.view(torch.bfloat16)
    with pytest.raises(ValueError):
        FP8GemmPlan(a, b, sa, sb, **options)
