"""FLA fused LM-head/cross-entropy integration.

Kept in a final-order module because FLA's fused kernels alter allocator state
used by the standalone CUDA Graph isolation tests.
"""

import pytest
import torch
import torch.nn.functional as F

from experiments.fp8_decoder_transformer import FP8DecoderTransformer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 and FLA required",
)


def test_fused_linear_cross_entropy_matches_logits_path():
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
    tokens = torch.randint(0, 257, (15,), device="cuda")
    cu_seqlens = torch.tensor([0, 8, 15], device="cuda", dtype=torch.int32)

    logits = model(tokens, cu_seqlens, 8)
    reference_loss = F.cross_entropy(logits.float(), tokens)
    reference_loss.backward()
    reference_grads = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    model.zero_grad(set_to_none=True)
    fused_loss = model.forward_loss(tokens, tokens, cu_seqlens, 8)
    fused_loss.backward()

    torch.testing.assert_close(fused_loss, reference_loss, rtol=1e-5, atol=1e-5)
    for name, parameter in model.named_parameters():
        if name == "embedding":
            delta = (parameter.grad.float() - reference_grads[name].float()).norm()
            relative = delta / reference_grads[name].float().norm()
            assert relative < 5e-3
        else:
            torch.testing.assert_close(
                parameter.grad,
                reference_grads[name],
                rtol=0,
                atol=0,
            )
