"""End-to-end smoke test for the FP8 prenorm decoder Transformer."""

import pytest
import torch
import torch.nn.functional as F

flash_attn = pytest.importorskip("flash_attn", reason="SM120 decoder requires flash-attn")

from experiments.fp8_decoder_transformer import (  # noqa: E402
    FP8DecoderTransformer,
    RotaryEmbedding,
)
from fp8_adamw import FusedAdamWMXFP8  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 and flash-attn required",
)


def test_forward_backward_and_optimizer_step():
    torch.manual_seed(202)
    model = FP8DecoderTransformer(
        vocab_size=257,
        d_model=64,
        n_layers=2,
        n_heads=4,
        mlp_hidden=128,
        context_length=32,
        attention_window=16,
        device="cuda",
        qkv_forward_config=0,
        qkv_dx_config=0,
        qkv_dw_config=0,
        swiglu_forward_config=1,
        swiglu_dx_config=5,
        swiglu_dw_config=0,
        down_forward_config=0,
        down_dx_config=2,
        down_dw_config=0,
    )
    tokens = torch.tensor(
        [1, 5, 9, 13, 17, 23, 29, 31, 2, 4, 8, 16, 32, 64, 128],
        device="cuda",
        dtype=torch.long,
    )
    cu_seqlens = torch.tensor([0, 8, 15], device="cuda", dtype=torch.int32)
    logits = model(tokens, cu_seqlens, 8)
    assert logits.shape == (tokens.numel(), 257)
    assert logits.dtype == torch.bfloat16
    loss = F.cross_entropy(logits.float(), tokens)
    loss.backward()

    required = [
        model.embedding,
        model.blocks[0].attention.q_weight,
        model.blocks[0].attention.k_weight,
        model.blocks[0].attention.v_weight,
        model.blocks[0].attention.norm_weight,
        model.blocks[0].attention.out_proj.weight,
        model.blocks[0].mlp.gate_weight,
        model.blocks[0].mlp.up_weight,
        model.blocks[0].mlp.norm_weight,
        model.blocks[0].mlp.down_proj.master_weight,
    ]
    for parameter in required:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    out_proj = model.blocks[0].attention.out_proj
    assert isinstance(out_proj, torch.nn.Linear)
    assert out_proj.weight.dtype == torch.bfloat16

    for module in model.modules():
        assert getattr(module, "bias", None) is None

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    model.bind_optimizer(optimizer)
    optimizer.step()
    model.refresh_fp8_weights()
    with torch.no_grad():
        second_logits = model(tokens, cu_seqlens, 8)
    assert torch.isfinite(second_logits).all()


def test_variable_sequence_boundaries():
    torch.manual_seed(203)
    model = FP8DecoderTransformer(
        vocab_size=31,
        d_model=64,
        n_layers=1,
        n_heads=4,
        mlp_hidden=128,
        context_length=8,
        device="cuda",
        qkv_forward_config=0,
        qkv_dx_config=0,
        qkv_dw_config=0,
        swiglu_forward_config=1,
        swiglu_dx_config=5,
        swiglu_dw_config=0,
        down_forward_config=0,
        down_dx_config=2,
        down_dw_config=0,
    )
    tokens = torch.randint(0, 31, (7,), device="cuda")
    cu_seqlens = torch.tensor([0, 1, 4, 7], device="cuda", dtype=torch.int32)
    logits = model(tokens, cu_seqlens, 3)
    assert logits.shape == (7, 31)
    assert torch.isfinite(logits).all()


def test_repeated_steps_reuse_plans_and_remain_finite():
    torch.manual_seed(204)
    model = FP8DecoderTransformer(
        vocab_size=257,
        d_model=64,
        n_layers=2,
        n_heads=4,
        mlp_hidden=128,
        context_length=32,
        device="cuda",
        qkv_forward_config=0,
        qkv_dx_config=0,
        qkv_dw_config=0,
        swiglu_forward_config=1,
        swiglu_dx_config=5,
        swiglu_dw_config=0,
        down_forward_config=0,
        down_dx_config=2,
        down_dw_config=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.bind_optimizer(optimizer)
    cu_seqlens = torch.tensor([0, 8, 15], device="cuda", dtype=torch.int32)
    cached_plan_ids = []

    for _ in range(3):
        tokens = torch.randint(1, 257, (15,), device="cuda")
        loss = model.forward_loss(
            tokens,
            tokens.clone(),
            cu_seqlens,
            8,
            num_chunks=2,
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        step_plan_ids = []
        for block in model.blocks:
            states = [
                block.attention._forward_states,
                block.attention._backward_states,
                block.mlp._forward_states,
                block.mlp._backward_states,
                block.mlp.down_proj._forward_states,
                block.mlp.down_proj._backward_states,
            ]
            assert all(len(cache) == 1 for cache in states)
            step_plan_ids.extend(id(next(iter(cache.values()))) for cache in states)
        cached_plan_ids.append(step_plan_ids)
        optimizer.step()

    assert cached_plan_ids[0] == cached_plan_ids[1] == cached_plan_ids[2]


def test_fused_adamw_updates_model_fp8_cache_directly():
    torch.manual_seed(206)
    model = FP8DecoderTransformer(
        vocab_size=257,
        d_model=64,
        n_layers=2,
        n_heads=4,
        mlp_hidden=128,
        context_length=32,
        device="cuda",
        qkv_forward_config=0,
        qkv_dx_config=0,
        qkv_dw_config=0,
        swiglu_forward_config=1,
        swiglu_dx_config=5,
        swiglu_dw_config=0,
        down_forward_config=0,
        down_dx_config=2,
        down_dw_config=0,
    )
    optimizer = FusedAdamWMXFP8(model.parameters(), lr=1e-4)
    model.bind_optimizer(optimizer)
    assert len(optimizer._fp8_outputs) == 12

    attention = model.blocks[0].attention
    previous_fp8 = attention._q_fp8.clone()
    tokens = torch.randint(1, 257, (15,), device="cuda")
    cu_seqlens = torch.tensor([0, 8, 15], device="cuda", dtype=torch.int32)
    loss = model.forward_loss(
        tokens,
        tokens.clone(),
        cu_seqlens,
        8,
        num_chunks=2,
    )
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    assert attention._synced_versions[0] == attention.q_weight._version
    assert not torch.equal(
        previous_fp8.view(torch.uint8), attention._q_fp8.view(torch.uint8)
    )
    with torch.no_grad():
        logits = model(tokens, cu_seqlens, 8)
    assert torch.isfinite(logits).all()


def test_fla_rope_matches_interleaved_reference():
    torch.manual_seed(205)
    tokens, heads, head_dim = 15, 4, 16
    q = torch.randn((tokens, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    cu_seqlens = torch.tensor([0, 8, 15], device="cuda", dtype=torch.int32)
    rope = RotaryEmbedding(head_dim, 32, device="cuda")
    actual_q, actual_k = rope(q, k, cu_seqlens, 8)

    positions = torch.cat(
        (
            torch.arange(8, device="cuda"),
            torch.arange(7, device="cuda"),
        )
    )
    frequencies = torch.pow(
        10000.0,
        -torch.arange(0, head_dim, 2, dtype=torch.float32, device="cuda")
        / head_dim,
    )
    angles = torch.outer(positions.float(), frequencies)
    cos = angles.cos().bfloat16()[:, None, :]
    sin = angles.sin().bfloat16()[:, None, :]

    def reference(x):
        even = x[..., 0::2]
        odd = x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)

    torch.testing.assert_close(actual_q, reference(q), rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(actual_k, reference(k), rtol=2e-2, atol=2e-3)
