"""Prenorm decoder Transformer built from the repository's FP8 components."""

import math

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from torch import nn

from experiments.mxfp8_core import bind_fp8_optimizer, quantize_weight
from experiments.tk_mxfp8_dgrad import Library as DgradLibrary
from experiments.tk_mxfp8_down import MXFP8DownProjection
from experiments.tk_mxfp8_gemm import Library as GemmLibrary
from experiments.tk_mxfp8_linear import MXFP8Linear
from experiments.tk_mxfp8_swiglu import MXFP8SwiGLULibrary
from experiments.tk_mxfp8_swiglu_backward import MXFP8SwiGLUBackward
from experiments.tk_mxfp8_swiglu_dgrad import SwiGLUDgradLibrary
from experiments.tk_mxfp8_wgrad import Library as WgradLibrary
from experiments.tk_rmsnorm import (
    FusedAddRMSNormBackwardPlan,
    FusedAddRMSNormPlan,
    RMSNormBackwardPlan,
    RMSNormDualQuantPlan,
)
from experiments.tk_silu_bwd import SiluBwdQuantizeLibrary


class RMSNorm(nn.Module):
    """BF16-compatible prenorm with an FP32 RMS reduction."""

    def __init__(self, hidden_size, eps=1e-6, device=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
        )
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.float()
        inverse = torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + self.eps)
        return (x_fp32 * inverse * self.weight.float()).to(torch.bfloat16)


class _FusedAddRMSNormBF16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, factor, eps):
        plan = FusedAddRMSNormPlan(
            x,
            residual,
            factor,
            eps=eps,
            y_mode="bf16",
        )
        inverse_rms, _, hidden, _, _, _, _ = plan()
        ctx.save_for_backward(x, residual, factor, inverse_rms)
        return hidden

    @staticmethod
    def backward(ctx, dy):
        x, residual, factor, inverse_rms = ctx.saved_tensors
        plan = FusedAddRMSNormBackwardPlan(
            dy.contiguous().to(torch.bfloat16),
            x,
            residual,
            factor,
            inverse_rms,
        )
        dx, dresidual, factor_grad = plan()
        return dx, dresidual, factor_grad.bfloat16(), None


class RotaryEmbedding(nn.Module):
    """FLA fused interleaved RoPE for packed variable-length sequences."""

    def __init__(self, head_dim, max_seq_len, theta=10000.0, device=None):
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        from fla.modules.rotary import RotaryEmbedding as FLARotaryEmbedding

        self.max_seq_len = max_seq_len
        self._rope = FLARotaryEmbedding(
            dim=head_dim,
            base=theta,
            interleaved=True,
            pos_idx_in_fp32=True,
            device=device,
        )

    def forward(self, q, k, cu_seqlens, max_seqlen):
        q, k = self._rope(
            q.unsqueeze(0),
            k.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            # FLA varlen requires its cache to cover the packed input extent.
            max_seqlen=max(max_seqlen, q.shape[0]),
        )
        return q.squeeze(0), k.squeeze(0)


class _QKVForwardState:
    def __init__(
        self,
        x,
        residual_delta,
        factor,
        weights,
        scales,
        config,
        raster_group,
    ):
        self.x = x
        self.residual_delta = residual_delta
        if residual_delta is None:
            self.norm = RMSNormDualQuantPlan(x, factor)
        else:
            self.norm = FusedAddRMSNormPlan(
                x,
                residual_delta,
                factor,
                write_sum=True,
                y_mode="mxfp8",
            )
        self.q = GemmLibrary().plan(
            self.norm.row_fp8,
            weights[0],
            self.norm.row_scale,
            scales[0],
            config=config,
            raster_group=raster_group,
        )
        self.k = GemmLibrary().plan(
            self.norm.row_fp8,
            weights[1],
            self.norm.row_scale,
            scales[1],
            config=config,
            raster_group=raster_group,
        )
        self.v = GemmLibrary().plan(
            self.norm.row_fp8,
            weights[2],
            self.norm.row_scale,
            scales[2],
            config=config,
            raster_group=raster_group,
        )

    def __call__(self):
        residual_out = self.x
        self.norm()
        if self.residual_delta is not None:
            residual_out = self.norm.sum_out
        return self.q(), self.k(), self.v(), residual_out


class _PrenormQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual_delta, factor, qw, kw, vw, module):
        module._sync_if_needed()
        state = module._forward_state(x, residual_delta, factor)
        q, k, v, residual_out = state()
        ctx.module = module
        ctx.residual_delta = residual_delta
        ctx.save_for_backward(
            x,
            factor,
            qw,
            kw,
            vw,
            module._q_fp8,
            module._q_scale,
            module._k_fp8,
            module._k_scale,
            module._v_fp8,
            module._v_scale,
            state.norm.column_fp8,
            state.norm.column_scale,
            state.norm.inverse_rms,
        )
        return q, k, v, residual_out

    @staticmethod
    def backward(ctx, dq, dk, dv, dresidual_out):
        (
            x,
            factor,
            _,
            _,
            _,
            q_fp8,
            q_scale,
            k_fp8,
            k_scale,
            v_fp8,
            v_scale,
            x_column,
            x_column_scale,
            inverse_rms,
        ) = ctx.saved_tensors
        module = ctx.module
        quantizers = []
        weight_grads = []
        for dy in [dq, dk, dv]:
            dy = dy.contiguous().to(torch.bfloat16)
            quantizers.append(
                SiluBwdQuantizeLibrary().plan_direct(dy, transpose=True)
            )
        pair_dx = SwiGLUDgradLibrary().plan(
            quantizers[0].row_fp8,
            quantizers[1].row_fp8,
            q_fp8,
            k_fp8,
            quantizers[0].row_scale,
            quantizers[1].row_scale,
            q_scale,
            k_scale,
            config=module.dx_config,
            raster_group=module.raster_group,
        )
        v_dx = DgradLibrary().plan(
            quantizers[2].row_fp8,
            v_fp8,
            quantizers[2].row_scale,
            v_scale,
            config=module.dx_config,
            raster_group=module.raster_group,
        )
        for quantize, weight_scale in [
            (quantizers[0], q_scale),
            (quantizers[1], k_scale),
            (quantizers[2], v_scale),
        ]:
            weight_grads.append(
                WgradLibrary().plan(
                    quantize.column_fp8,
                    x_column,
                    quantize.column_scale,
                    x_column_scale,
                    config=module.dw_config,
                    raster_group=module.raster_group,
                )
            )
        for quantize in quantizers:
            quantize()
        dx_norm = torch.add(pair_dx(), v_dx())
        weight_grads = [plan() for plan in weight_grads]
        if ctx.residual_delta is None:
            norm_backward = RMSNormBackwardPlan(dx_norm, x, factor, inverse_rms)
            dx, factor_grad = norm_backward()
            if dresidual_out is not None:
                dx = torch.add(dx, dresidual_out)
            dresidual_delta = None
        else:
            norm_backward = FusedAddRMSNormBackwardPlan(
                dx_norm,
                x,
                ctx.residual_delta,
                factor,
                inverse_rms,
                dsum=dresidual_out,
            )
            dx, dresidual_delta, factor_grad = norm_backward()
        return (
            dx,
            dresidual_delta,
            factor_grad.bfloat16(),
            weight_grads[0].float(),
            weight_grads[1].float(),
            weight_grads[2].float(),
            None,
        )


class FP8Attention(nn.Module):
    """Separate pure-MXFP8 Q/K/V projections with fused prenorm."""

    def __init__(
        self,
        d_model,
        n_heads,
        max_seq_len,
        *,
        qkv_forward_config=8,
        dx_config=6,
        dw_config=11,
        raster_group=13,
        attention_window=-1,
        device=None,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_window = attention_window
        self.qkv_forward_config = qkv_forward_config
        self.dx_config = dx_config
        self.dw_config = dw_config
        self.raster_group = raster_group
        self.norm_weight = nn.Parameter(
            torch.ones(d_model, dtype=torch.bfloat16, device=device)
        )
        self.q_weight = nn.Parameter(
            torch.empty((d_model, d_model), dtype=torch.float32, device=device)
        )
        self.k_weight = nn.Parameter(
            torch.empty((d_model, d_model), dtype=torch.float32, device=device)
        )
        self.v_weight = nn.Parameter(
            torch.empty((d_model, d_model), dtype=torch.float32, device=device)
        )
        for name in ["q", "k", "v"]:
            self.register_buffer(
                f"_{name}_fp8",
                torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
                persistent=False,
            )
            self.register_buffer(
                f"_{name}_scale",
                torch.empty(0, dtype=torch.uint8, device=device),
                persistent=False,
            )
        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        self.rope = RotaryEmbedding(
            self.head_dim, max_seq_len, device=device
        )
        self._synced_versions = None
        self._fp8_dirty = False
        self._forward_states = {}
        self.reset_projection_parameters()
        self._sync_if_needed()

    def reset_projection_parameters(self):
        nn.init.kaiming_uniform_(self.q_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_weight, a=math.sqrt(5))
        nn.init.ones_(self.norm_weight)

    def refresh_fp8_weights(self):
        with torch.no_grad():
            self._q_fp8, self._q_scale = quantize_weight(self.q_weight)
            self._k_fp8, self._k_scale = quantize_weight(self.k_weight)
            self._v_fp8, self._v_scale = quantize_weight(self.v_weight)
        self._synced_versions = (
            self.q_weight._version,
            self.k_weight._version,
            self.v_weight._version,
        )
        self._fp8_dirty = False
        self._forward_states.clear()
        return self

    def mark_fp8_dirty(self):
        self._fp8_dirty = True
        return self

    def _sync_if_needed(self):
        versions = (
            self.q_weight._version,
            self.k_weight._version,
            self.v_weight._version,
        )
        if self._fp8_dirty or self._synced_versions != versions:
            self.refresh_fp8_weights()

    def _forward_state(self, x, residual_delta, factor):
        key = (
            x.data_ptr(),
            0 if residual_delta is None else residual_delta.data_ptr(),
            factor.data_ptr(),
            tuple(x.shape),
            self._q_fp8.data_ptr(),
            self._k_fp8.data_ptr(),
            self._v_fp8.data_ptr(),
            self.qkv_forward_config,
            self.raster_group,
        )
        state = self._forward_states.get(key)
        if state is None:
            state = _QKVForwardState(
                x,
                residual_delta,
                factor,
                (self._q_fp8, self._k_fp8, self._v_fp8),
                (self._q_scale, self._k_scale, self._v_scale),
                self.qkv_forward_config,
                self.raster_group,
            )
            self._forward_states = {key: state}
        return state

    def forward(
        self,
        x,
        cu_seqlens,
        max_seqlen,
        residual_delta=None,
    ):
        q, k, v, residual_out = _PrenormQKV.apply(
            x,
            residual_delta,
            self.norm_weight,
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self,
        )
        shape = (q.shape[0], self.n_heads, self.head_dim)
        q, k = self.rope(
            q.reshape(shape),
            k.reshape(shape),
            cu_seqlens,
            max_seqlen,
        )
        v = v.reshape(shape)
        window = (
            (self.attention_window, -1)
            if self.attention_window > 0
            else (-1, -1)
        )
        attended = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=True,
            window_size=window,
        )
        return self.out_proj(attended.reshape(x.shape)), residual_out


class _GateUpForwardState:
    def __init__(
        self,
        x,
        residual_delta,
        factor,
        gate_fp8,
        gate_scale,
        up_fp8,
        up_scale,
        forward_config,
        raster_group,
    ):
        self.norm = FusedAddRMSNormPlan(
            x,
            residual_delta,
            factor,
            write_sum=True,
            y_mode="mxfp8",
        )
        shape = (x.shape[0], gate_fp8.shape[0])
        self.gate_linear = torch.empty(
            shape, device=x.device, dtype=torch.bfloat16
        )
        self.up_linear = torch.empty_like(self.gate_linear)
        self.forward = MXFP8SwiGLULibrary().plan(
            self.norm.row_fp8,
            self.norm.row_scale,
            gate_fp8,
            gate_scale,
            up_fp8,
            up_scale,
            config=forward_config,
            raster_group=raster_group,
            gate_out=self.gate_linear,
            up_out=self.up_linear,
        )

    def __call__(self):
        self.norm()
        return self.forward(), self.norm.sum_out


class _PrenormGateUp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        residual_delta,
        factor,
        gate_weight,
        up_weight,
        module,
    ):
        module._sync_if_needed()
        state = module._forward_state(x, residual_delta, factor)
        hidden, residual_out = state()
        ctx.module = module
        ctx.x = x
        ctx.residual_delta = residual_delta
        ctx.save_for_backward(
            x,
            factor,
            gate_weight,
            up_weight,
            module._gate_fp8,
            module._gate_scale,
            module._up_fp8,
            module._up_scale,
            state.gate_linear,
            state.up_linear,
            state.norm.column_fp8,
            state.norm.column_scale,
            state.norm.inverse_rms,
        )
        return hidden, residual_out

    @staticmethod
    def backward(ctx, dy, dresidual_out):
        (
            x,
            factor,
            _,
            _,
            gate_fp8,
            gate_scale,
            up_fp8,
            up_scale,
            gate_linear,
            up_linear,
            x_column,
            x_column_scale,
            inverse_rms,
        ) = ctx.saved_tensors
        module = ctx.module
        fused = MXFP8SwiGLUBackward(
            dy.contiguous().to(torch.bfloat16),
            gate_linear,
            up_linear,
            x,
            gate_fp8,
            up_fp8,
            gate_scale,
            up_scale,
            x_column,
            x_column_scale,
            dx_config=module.dx_config,
            wgrad_config=module.wgrad_config,
            raster_group=module.raster_group,
        )
        dx_norm, gate_grad, up_grad = fused()
        norm_backward = FusedAddRMSNormBackwardPlan(
            dx_norm,
            ctx.x,
            ctx.residual_delta,
            factor,
            inverse_rms,
            dsum=dresidual_out,
        )
        dx, dresidual_delta, factor_grad = norm_backward()
        return (
            dx,
            dresidual_delta,
            factor_grad.bfloat16(),
            gate_grad.float(),
            up_grad.float(),
            None,
        )


class PrenormSwiGLU(nn.Module):
    """Pure-MXFP8 SwiGLU consuming fused prenorm dual quantization."""

    def __init__(
        self,
        d_model,
        hidden_features,
        *,
        device=None,
        forward_config=1,
        dx_config=5,
        wgrad_config=12,
        down_forward_config=70,
        down_dx_config=12,
        down_dw_config=12,
        raster_group=13,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_features = hidden_features
        self.forward_config = forward_config
        self.dx_config = dx_config
        self.wgrad_config = wgrad_config
        self.raster_group = raster_group
        self.norm_weight = nn.Parameter(
            torch.ones(d_model, dtype=torch.bfloat16, device=device)
        )
        self.gate_weight = nn.Parameter(
            torch.empty((hidden_features, d_model), dtype=torch.float32, device=device)
        )
        self.up_weight = nn.Parameter(
            torch.empty((hidden_features, d_model), dtype=torch.float32, device=device)
        )
        for name in ["gate", "up"]:
            self.register_buffer(
                f"_{name}_fp8",
                torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
                persistent=False,
            )
            self.register_buffer(
                f"_{name}_scale",
                torch.empty(0, dtype=torch.uint8, device=device),
                persistent=False,
            )
        self.down_proj = MXFP8DownProjection(
            hidden_features,
            d_model,
            device=device,
            forward_config=down_forward_config,
            dx_config=down_dx_config,
            dw_config=down_dw_config,
            raster_group=raster_group,
        )
        self._synced_versions = None
        self._fp8_dirty = False
        self._forward_states = {}
        self.reset_parameters()
        self._sync_if_needed()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_weight, a=math.sqrt(5))
        nn.init.ones_(self.norm_weight)

    def refresh_fp8_weights(self):
        with torch.no_grad():
            self._gate_fp8, self._gate_scale = quantize_weight(
                self.gate_weight
            )
            self._up_fp8, self._up_scale = quantize_weight(
                self.up_weight
            )
        self.down_proj.refresh_fp8_weight()
        self._synced_versions = (
            self.gate_weight._version,
            self.up_weight._version,
        )
        self._fp8_dirty = False
        self._forward_states.clear()
        return self

    def mark_fp8_dirty(self):
        self._fp8_dirty = True
        self.down_proj.mark_fp8_dirty()
        return self

    def _sync_if_needed(self):
        versions = (self.gate_weight._version, self.up_weight._version)
        if self._fp8_dirty or self._synced_versions != versions:
            self.refresh_fp8_weights()

    def _forward_state(self, x, residual_delta, factor):
        key = (
            x.data_ptr(),
            residual_delta.data_ptr(),
            factor.data_ptr(),
            tuple(x.shape),
            self._gate_fp8.data_ptr(),
            self._up_fp8.data_ptr(),
            self.forward_config,
            self.raster_group,
        )
        state = self._forward_states.get(key)
        if state is None:
            state = _GateUpForwardState(
                x,
                residual_delta,
                factor,
                self._gate_fp8,
                self._gate_scale,
                self._up_fp8,
                self._up_scale,
                self.forward_config,
                self.raster_group,
            )
            self._forward_states = {key: state}
        return state

    def forward(self, x, residual_delta):
        hidden, residual_out = _PrenormGateUp.apply(
            x,
            residual_delta,
            self.norm_weight,
            self.gate_weight,
            self.up_weight,
            self,
        )
        return self.down_proj(hidden), residual_out


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        mlp_hidden,
        max_seq_len,
        *,
        qkv_forward_config=8,
        qkv_dx_config=6,
        qkv_dw_config=11,
        swiglu_forward_config=1,
        swiglu_dx_config=5,
        swiglu_dw_config=12,
        attention_window=-1,
        device=None,
        down_forward_config=70,
        down_dx_config=12,
        down_dw_config=12,
    ):
        super().__init__()
        self.attention = FP8Attention(
            d_model,
            n_heads,
            max_seq_len,
            qkv_forward_config=qkv_forward_config,
            dx_config=qkv_dx_config,
            dw_config=qkv_dw_config,
            attention_window=attention_window,
            device=device,
        )
        self.mlp = PrenormSwiGLU(
            d_model,
            mlp_hidden,
            device=device,
            forward_config=swiglu_forward_config,
            dx_config=swiglu_dx_config,
            wgrad_config=swiglu_dw_config,
            down_forward_config=down_forward_config,
            down_dx_config=down_dx_config,
            down_dw_config=down_dw_config,
        )

    def forward(
        self,
        x,
        cu_seqlens,
        max_seqlen,
        residual_delta=None,
    ):
        attention_delta, x = self.attention(
            x,
            cu_seqlens,
            max_seqlen,
            residual_delta=residual_delta,
        )
        mlp_delta, x = self.mlp(x, attention_delta)
        return x, mlp_delta


class FP8DecoderTransformer(nn.Module):
    """Causal decoder with prenorm attention and pure-MXFP8 SwiGLU FFN.

    ``forward`` materializes logits for inference. ``forward_loss`` is the
    training path and uses FLA fused linear cross-entropy by default.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        mlp_hidden,
        context_length,
        *,
        qkv_forward_config=8,
        qkv_dx_config=6,
        qkv_dw_config=11,
        swiglu_forward_config=1,
        swiglu_dx_config=5,
        swiglu_dw_config=12,
        attention_window=-1,
        device=None,
        down_forward_config=70,
        down_dx_config=12,
        down_dw_config=12,
    ):
        super().__init__()
        if d_model % 64 or mlp_hidden % 64:
            raise ValueError("d_model and mlp_hidden must be multiples of 64")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_length = context_length
        self.embedding = nn.Parameter(
            torch.empty(vocab_size, d_model, dtype=torch.bfloat16, device=device)
        )
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    n_heads,
                    mlp_hidden,
                    context_length,
                    qkv_forward_config=qkv_forward_config,
                    qkv_dx_config=qkv_dx_config,
                    qkv_dw_config=qkv_dw_config,
                    swiglu_forward_config=swiglu_forward_config,
                    swiglu_dx_config=swiglu_dx_config,
                    swiglu_dw_config=swiglu_dw_config,
                    attention_window=attention_window,
                    device=device,
                    down_forward_config=down_forward_config,
                    down_dx_config=down_dx_config,
                    down_dw_config=down_dw_config,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model, device=device)
        self.apply(self._init_weights)
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward_hidden(
        self,
        tokens,
        cu_seqlens,
        max_seqlen,
        *,
        position_ids=None,
    ):
        del position_ids
        if cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(torch.int32)
        x = F.embedding(tokens, self.embedding)
        residual_delta = None
        for block in self.blocks:
            x, residual_delta = block(
                x,
                cu_seqlens,
                max_seqlen,
                residual_delta=residual_delta,
            )
        return _FusedAddRMSNormBF16.apply(
            x,
            residual_delta,
            self.final_norm.weight,
            self.final_norm.eps,
        )

    def forward(self, tokens, cu_seqlens, max_seqlen):
        hidden = self.forward_hidden(tokens, cu_seqlens, max_seqlen)
        return F.linear(hidden, self.embedding)

    def forward_loss(
        self,
        tokens,
        targets,
        cu_seqlens,
        max_seqlen,
        *,
        position_ids=None,
        fused_linear_ce=True,
        reduction="mean",
        ignore_index=-100,
        num_chunks=2,
    ):
        """Return training loss, optionally fusing lm head and cross-entropy."""
        if targets.numel() != tokens.numel():
            raise ValueError("targets and tokens must contain the same number of elements")
        hidden = self.forward_hidden(
            tokens,
            cu_seqlens,
            max_seqlen,
            position_ids=position_ids,
        )
        if fused_linear_ce:
            if reduction not in ("mean", "sum"):
                raise ValueError("FLA fused linear CE supports mean or sum reduction")
            from fla.modules.fused_linear_cross_entropy import (
                fused_linear_cross_entropy_loss,
            )

            return fused_linear_cross_entropy_loss(
                hidden,
                targets,
                self.embedding,
                ignore_index=ignore_index,
                reduction=reduction,
                num_chunks=num_chunks,
            )
        logits = F.linear(hidden, self.embedding)
        return F.cross_entropy(
            logits.float(),
            targets,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def refresh_fp8_weights(self):
        for module in self.modules():
            refresh = getattr(module, "refresh_fp8_weights", None)
            if module is not self and callable(refresh):
                refresh()
        return self

    def bind_optimizer(self, optimizer):
        return bind_fp8_optimizer(self, optimizer)
