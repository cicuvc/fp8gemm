"""Prenorm decoder Transformer built from the repository's FP8 components."""

import math

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from torch import nn

from fp8_linear import FP8Linear
from experiments.tk_mxfp8_dgrad import Library as DgradLibrary, RowQuantizePlan
from experiments.tk_mxfp8_down import MXFP8DownProjection
from experiments.tk_mxfp8_gemm import Library as GemmLibrary
from experiments.tk_mxfp8_swiglu import MXFP8SwiGLULibrary
from experiments.tk_mxfp8_swiglu_backward import MXFP8SwiGLUBackward
from experiments.tk_mxfp8_wgrad import ColumnQuantizePlan, Library as WgradLibrary
from experiments.tk_rmsnorm import RMSNormBackwardPlan, RMSNormDualQuantPlan


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


class RotaryEmbedding(nn.Module):
    """Torch compatibility RoPE for packed variable-length sequences."""

    def __init__(self, head_dim, max_seq_len, theta=10000.0, device=None):
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        frequencies = torch.pow(
            theta,
            -torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim,
        )
        angles = torch.outer(positions, frequencies)
        self.register_buffer("cos", angles.cos().bfloat16()[None, :, :], persistent=False)
        self.register_buffer("sin", angles.sin().bfloat16()[None, :, :], persistent=False)

    def forward(self, x, position_ids):
        cos = self.cos[0].index_select(0, position_ids)[:, None, :]
        sin = self.sin[0].index_select(0, position_ids)[:, None, :]
        even = x[..., 0::2]
        odd = x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)


class _QKVForwardState:
    def __init__(
        self,
        x,
        factor,
        weights,
        scales,
        config,
        raster_group,
    ):
        self.norm = RMSNormDualQuantPlan(x, factor)
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
        self.norm()
        return self.q(), self.k(), self.v()


class _PrenormQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor, qw, kw, vw, module):
        module._sync_if_needed()
        state = module._forward_state(x, factor)
        q, k, v = state()
        ctx.module = module
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
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
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
        dx_parts = []
        weight_grads = []
        for dy, weight, weight_scale in [
            (dq, q_fp8, q_scale),
            (dk, k_fp8, k_scale),
            (dv, v_fp8, v_scale),
        ]:
            dy = dy.contiguous().to(torch.bfloat16)
            row = RowQuantizePlan(dy)
            column = ColumnQuantizePlan(dy, transpose=True)
            dx_part = DgradLibrary().plan(
                row.out,
                weight,
                row.scales,
                weight_scale,
                config=module.dx_config,
                raster_group=module.raster_group,
            )
            dw_part = WgradLibrary().plan(
                column.out,
                x_column,
                column.scales,
                x_column_scale,
                config=module.dw_config,
                raster_group=module.raster_group,
            )
            row()
            column()
            dx_parts.append(dx_part())
            weight_grads.append(dw_part())
        dx_norm = torch.add(dx_parts[0], dx_parts[1])
        dx_norm = torch.add(dx_norm, dx_parts[2])
        norm_backward = RMSNormBackwardPlan(dx_norm, x, factor, inverse_rms)
        dx, factor_grad = norm_backward()
        return (
            dx,
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
        dx_config=12,
        dw_config=12,
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
        self.out_proj = FP8Linear(
            d_model,
            d_model,
            bias=False,
            device=device,
            backward_backend="cuda",
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
        import mxtest

        with torch.no_grad():
            self._q_fp8, self._q_scale = mxtest.launch_weight_quantize(self.q_weight)
            self._k_fp8, self._k_scale = mxtest.launch_weight_quantize(self.k_weight)
            self._v_fp8, self._v_scale = mxtest.launch_weight_quantize(self.v_weight)
        self.out_proj.refresh_fp8_weight()
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
        self.out_proj.mark_fp8_dirty()
        return self

    def _sync_if_needed(self):
        versions = (
            self.q_weight._version,
            self.k_weight._version,
            self.v_weight._version,
        )
        if self._fp8_dirty or self._synced_versions != versions:
            self.refresh_fp8_weights()

    def _forward_state(self, x, factor):
        key = (
            x.data_ptr(),
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
                factor,
                (self._q_fp8, self._k_fp8, self._v_fp8),
                (self._q_scale, self._k_scale, self._v_scale),
                self.qkv_forward_config,
                self.raster_group,
            )
            self._forward_states = {key: state}
        return state

    def forward(self, x, cu_seqlens, max_seqlen, position_ids):
        q, k, v = _PrenormQKV.apply(
            x,
            self.norm_weight,
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self,
        )
        shape = (q.shape[0], self.n_heads, self.head_dim)
        q = self.rope(q.reshape(shape), position_ids)
        k = self.rope(k.reshape(shape), position_ids)
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
        return self.out_proj(attended.reshape(x.shape))


class _GateUpForwardState:
    def __init__(
        self,
        x,
        factor,
        gate_fp8,
        gate_scale,
        up_fp8,
        up_scale,
        forward_config,
        raster_group,
    ):
        self.norm = RMSNormDualQuantPlan(x, factor)
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
        return self.forward()


class _PrenormGateUp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor, gate_weight, up_weight, module):
        module._sync_if_needed()
        state = module._forward_state(x, factor)
        hidden = state()
        ctx.module = module
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
        return hidden

    @staticmethod
    def backward(ctx, dy):
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
        norm_backward = RMSNormBackwardPlan(dx_norm, x, factor, inverse_rms)
        dx, factor_grad = norm_backward()
        return (
            dx,
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
        wgrad_config=6,
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
        import mxtest

        with torch.no_grad():
            self._gate_fp8, self._gate_scale = mxtest.launch_weight_quantize(
                self.gate_weight
            )
            self._up_fp8, self._up_scale = mxtest.launch_weight_quantize(
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

    def _forward_state(self, x, factor):
        key = (
            x.data_ptr(),
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

    def forward(self, x):
        hidden = _PrenormGateUp.apply(
            x,
            self.norm_weight,
            self.gate_weight,
            self.up_weight,
            self,
        )
        return self.down_proj(hidden)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        mlp_hidden,
        max_seq_len,
        *,
        qkv_forward_config=8,
        qkv_dx_config=12,
        qkv_dw_config=12,
        swiglu_forward_config=1,
        swiglu_dx_config=5,
        swiglu_dw_config=6,
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

    def forward(self, x, cu_seqlens, max_seqlen, position_ids):
        x = x + self.attention(
            x,
            cu_seqlens,
            max_seqlen,
            position_ids,
        )
        x = x + self.mlp(x)
        return x


class FP8DecoderTransformer(nn.Module):
    """Causal decoder with prenorm attention and pure-MXFP8 SwiGLU FFN."""

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
        qkv_dx_config=12,
        qkv_dw_config=12,
        swiglu_forward_config=1,
        swiglu_dx_config=5,
        swiglu_dw_config=6,
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

    @staticmethod
    def _position_ids(cu_seqlens, total_tokens):
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        offsets = torch.repeat_interleave(cu_seqlens[:-1], lengths)
        return torch.arange(total_tokens, device=cu_seqlens.device) - offsets

    def forward(self, tokens, cu_seqlens, max_seqlen):
        if cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(torch.int32)
        position_ids = self._position_ids(cu_seqlens, tokens.numel())
        x = F.embedding(tokens, self.embedding)
        for block in self.blocks:
            x = block(x, cu_seqlens, max_seqlen, position_ids)
        hidden = self.final_norm(x)
        return F.linear(hidden, self.embedding)

    def refresh_fp8_weights(self):
        for module in self.modules():
            refresh = getattr(module, "refresh_fp8_weights", None)
            if module is not self and callable(refresh):
                refresh()
        return self

    def bind_optimizer(self, optimizer):
        from fp8_linear import bind_fp8_optimizer

        return bind_fp8_optimizer(self, optimizer)
