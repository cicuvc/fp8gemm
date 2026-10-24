"""Smoke and architecture tests for the post-RMSNorm FP8 decoder."""

import gc

import pytest
import torch
import torch.nn.functional as F

from experiments.fp8_decoder_postnorm_transformer import (
    FP8PostNormDecoderTransformer,
)
from fp8_adamw import FusedAdamWMXFP8


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 and flash-attn required",
)


@pytest.fixture(autouse=True)
def cleanup_model_state():
    yield
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def make_model():
    return FP8PostNormDecoderTransformer(
        vocab_size=257,
        d_model=64,
        n_layers=2,
        n_heads=4,
        mlp_hidden=128,
        context_length=32,
        qkv_forward_config=0,
        qkv_dx_config=0,
        qkv_dw_config=0,
        swiglu_forward_config=1,
        swiglu_dx_config=5,
        swiglu_dw_config=0,
        down_forward_config=0,
        down_dx_config=2,
        down_dw_config=0,
        device="cuda",
    )


def test_postnorm_output_is_normalized_and_backward_runs():
    torch.manual_seed(401)
    model = make_model()
    tokens = torch.tensor(
        [1, 5, 9, 13, 17, 23, 29, 31, 2, 4, 8, 16, 32, 64, 128],
        device="cuda",
    )
    cu_seqlens = torch.tensor([0, 8, 15], device="cuda", dtype=torch.int32)

    hidden = model.forward_hidden(tokens, cu_seqlens, 8)
    rms = hidden.float().square().mean(-1).sqrt()
    torch.testing.assert_close(
        rms,
        torch.ones_like(rms),
        rtol=2e-2,
        atol=2e-2,
    )

    logits = F.linear(hidden, model.embedding)
    loss = F.cross_entropy(logits.float(), tokens)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_postnorm_fused_optimizer_step():
    torch.manual_seed(402)
    model = make_model()
    optimizer = FusedAdamWMXFP8(model.parameters(), lr=1e-4)
    model.bind_optimizer(optimizer)
    assert len(optimizer._fp8_outputs) == 12

    cu_seqlens = torch.tensor([0, 8, 15], device="cuda", dtype=torch.int32)
    tokens = torch.randint(1, 257, (15,), device="cuda")
    logits = model(tokens, cu_seqlens, 8)
    loss = F.cross_entropy(logits.float(), tokens)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)

    with torch.no_grad():
        logits = model(tokens, cu_seqlens, 8)
    assert torch.isfinite(logits).all()
