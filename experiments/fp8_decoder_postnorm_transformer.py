"""Post-RMSNorm decoder built from the repository's MXFP8 components."""

import math

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from torch import nn

from experiments.fp8_decoder_transformer import (
    RotaryEmbedding,
    _FusedAddRMSNormBF16,
)
from experiments.mxfp8_core import (
    bind_fp8_optimizer,
    iter_fp8_modules,
    quantize_weight,
)
from experiments.tk_mxfp8_dgrad import Library as DgradLibrary
from experiments.tk_mxfp8_gemm import Library as GemmLibrary
from experiments.tk_mxfp8_swiglu_dgrad import SwiGLUDgradLibrary
from experiments.tk_mxfp8_swiglu_layer import MXFP8SwiGLU
from experiments.tk_mxfp8_wgrad import Library as WgradLibrary
from experiments.tk_silu_bwd import SiluBwdQuantizeLibrary


class _PostQKVForwardState:
    def __init__(
        self,
        x,
        weights,
        scales,
        config,
        raster_group,
    ):
        self.quantize = SiluBwdQuantizeLibrary().plan_direct(x)
        self.projections = tuple(
            GemmLibrary().plan(
                self.quantize.row_fp8,
                weight,
                self.quantize.row_scale,
                scale,
                config=config,
                raster_group=raster_group,
            )
            for weight, scale in zip(weights, scales)
        )

    def __call__(self, x, weights, scales):
        self.quantize.rebind(x)
        for projection, weight, scale in zip(
            self.projections, weights, scales
        ):
            projection.rebind(
                self.quantize.row_fp8,
                weight,
                self.quantize.row_scale,
                scale,
            )
        self.quantize()
        return tuple(projection() for projection in self.projections)


class _PostQKVBackwardState:
    def __init__(
        self,
        dq,
        dk,
        dv,
        x,
        weights,
        scales,
        x_column,
        x_column_scale,
        *,
        dx_config,
        dw_config,
        raster_group,
    ):
        self.quantizers = tuple(
            SiluBwdQuantizeLibrary().plan_direct(dy, transpose=True)
            for dy in (dq, dk, dv)
        )
        self.pair_dx = SwiGLUDgradLibrary().plan(
            self.quantizers[0].row_fp8,
            self.quantizers[1].row_fp8,
            weights[0],
            weights[1],
            self.quantizers[0].row_scale,
            self.quantizers[1].row_scale,
            scales[0],
            scales[1],
            config=dx_config,
            raster_group=raster_group,
        )
        self.v_dx = DgradLibrary().plan(
            self.quantizers[2].row_fp8,
            weights[2],
            self.quantizers[2].row_scale,
            scales[2],
            config=dx_config,
            raster_group=raster_group,
        )
        wgrad_library = WgradLibrary()
        self.weight_grads = tuple(
            wgrad_library.plan(
                quantize.column_fp8,
                x_column,
                quantize.column_scale,
                x_column_scale,
                config=dw_config,
                raster_group=raster_group,
            )
            for quantize in self.quantizers
        )
        self.dx_shape = tuple(x.shape)
        self.weight_grad_shapes = tuple(weight.shape for weight in weights)
        self.device = x.device

    def __call__(
        self,
        dq,
        dk,
        dv,
        weights,
        scales,
        x_column,
        x_column_scale,
    ):
        for quantize, dy in zip(self.quantizers, (dq, dk, dv)):
            quantize.reallocate_outputs()
            quantize.rebind(dy)
        self.pair_dx.rebind(
            self.quantizers[0].row_fp8,
            self.quantizers[1].row_fp8,
            weights[0],
            weights[1],
            self.quantizers[0].row_scale,
            self.quantizers[1].row_scale,
            scales[0],
            scales[1],
            out=torch.empty(
                self.dx_shape, device=self.device, dtype=torch.bfloat16
            ),
        )
        self.v_dx.rebind(
            self.quantizers[2].row_fp8,
            weights[2],
            self.quantizers[2].row_scale,
            scales[2],
            out=torch.empty(
                self.dx_shape, device=self.device, dtype=torch.bfloat16
            ),
        )
        for plan, quantize, shape in zip(
            self.weight_grads, self.quantizers, self.weight_grad_shapes
        ):
            plan.rebind(
                quantize.column_fp8,
                x_column,
                quantize.column_scale,
                x_column_scale,
                out=torch.empty(
                    shape, device=self.device, dtype=torch.bfloat16
                ),
            )
        for quantize in self.quantizers:
            quantize()
        dx = torch.empty(
            self.dx_shape, device=self.device, dtype=torch.bfloat16
        )
        torch.add(self.pair_dx(), self.v_dx(), out=dx)
        weight_grads = tuple(plan() for plan in self.weight_grads)
        for quantize in self.quantizers:
            quantize.release()
        self.pair_dx.release()
        self.v_dx.release()
        for plan in self.weight_grads:
            plan.release()
        return dx, weight_grads


class _PostQKVProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q_weight, k_weight, v_weight, module):
        module._sync_if_needed()
        state = module._forward_state(x)
        q, k, v = state(
            x,
            (module._q_fp8, module._k_fp8, module._v_fp8),
            (module._q_scale, module._k_scale, module._v_scale),
        )
        ctx.module = module
        ctx.save_for_backward(
            x,
            q_weight,
            k_weight,
            v_weight,
            module._q_fp8,
            module._q_scale,
            module._k_fp8,
            module._k_scale,
            module._v_fp8,
            module._v_scale,
            state.quantize.column_fp8,
            state.quantize.column_scale,
        )
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        (
            x,
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
        ) = ctx.saved_tensors
        dq, dk, dv = (
            dy.contiguous().to(torch.bfloat16) for dy in (dq, dk, dv)
        )
        module = ctx.module
        state = module._backward_state(
            dq,
            dk,
            dv,
            x,
            (q_fp8, k_fp8, v_fp8),
            (q_scale, k_scale, v_scale),
            x_column,
            x_column_scale,
        )
        dx, weight_grads = state(
            dq,
            dk,
            dv,
            (q_fp8, k_fp8, v_fp8),
            (q_scale, k_scale, v_scale),
            x_column,
            x_column_scale,
        )
        return (
            dx,
            weight_grads[0].float(),
            weight_grads[1].float(),
            weight_grads[2].float(),
            None,
        )


class PostNormAttention(nn.Module):
    """Separate MXFP8 Q/K/V projections followed by post-attention RMSNorm."""

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
        eps=1e-6,
        device=None,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_forward_config = qkv_forward_config
        self.dx_config = dx_config
        self.dw_config = dw_config
        self.raster_group = raster_group
        self.attention_window = attention_window
        self.eps = eps
        self.q_weight = nn.Parameter(
            torch.empty((d_model, d_model), dtype=torch.float32, device=device)
        )
        self.k_weight = nn.Parameter(
            torch.empty((d_model, d_model), dtype=torch.float32, device=device)
        )
        self.v_weight = nn.Parameter(
            torch.empty((d_model, d_model), dtype=torch.float32, device=device)
        )
        self.norm_weight = nn.Parameter(
            torch.ones(d_model, dtype=torch.bfloat16, device=device)
        )
        for name in ("q", "k", "v"):
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
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, device=device)
        self._synced_versions = None
        self._fp8_dirty = False
        self._forward_states = {}
        self._backward_states = {}
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
        return self

    def mark_fp8_dirty(self):
        self._fp8_dirty = True
        return self

    def mark_fp8_synced(self):
        self._synced_versions = (
            self.q_weight._version,
            self.k_weight._version,
            self.v_weight._version,
        )
        self._fp8_dirty = False
        return self

    def fp8_parameter_bindings(self):
        return (
            (self.q_weight, self._q_fp8, self._q_scale),
            (self.k_weight, self._k_fp8, self._k_scale),
            (self.v_weight, self._v_fp8, self._v_scale),
        )

    def _sync_if_needed(self):
        versions = (
            self.q_weight._version,
            self.k_weight._version,
            self.v_weight._version,
        )
        if self._fp8_dirty or self._synced_versions != versions:
            self.refresh_fp8_weights()

    def _forward_state(self, x):
        key = (
            tuple(x.shape),
            self.qkv_forward_config,
            self.raster_group,
        )
        state = self._forward_states.get(key)
        if state is None:
            state = _PostQKVForwardState(
                x,
                (self._q_fp8, self._k_fp8, self._v_fp8),
                (self._q_scale, self._k_scale, self._v_scale),
                self.qkv_forward_config,
                self.raster_group,
            )
            self._forward_states[key] = state
        return state

    def _backward_state(
        self,
        dq,
        dk,
        dv,
        x,
        weights,
        scales,
        x_column,
        x_column_scale,
    ):
        key = (
            tuple(x.shape),
            self.dx_config,
            self.dw_config,
            self.raster_group,
        )
        state = self._backward_states.get(key)
        if state is None:
            state = _PostQKVBackwardState(
                dq,
                dk,
                dv,
                x,
                weights,
                scales,
                x_column,
                x_column_scale,
                dx_config=self.dx_config,
                dw_config=self.dw_config,
                raster_group=self.raster_group,
            )
            self._backward_states[key] = state
        return state

    def forward(self, x, cu_seqlens, max_seqlen):
        q, k, v = _PostQKVProjection.apply(
            x,
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self,
        )
        shape = (x.shape[0], self.n_heads, self.head_dim)
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
        attention_output = self.out_proj(attended.reshape(x.shape))
        return _FusedAddRMSNormBF16.apply(
            x,
            attention_output,
            self.norm_weight,
            self.eps,
        )


class PostNormSwiGLU(nn.Module):
    """Fused MXFP8 SwiGLU followed by post-FFN RMSNorm."""

    def __init__(
        self,
        d_model,
        hidden_features,
        *,
        eps=1e-6,
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
        self.eps = eps
        self.mlp = MXFP8SwiGLU(
            d_model,
            hidden_features,
            d_model,
            device=device,
            forward_config=forward_config,
            dx_config=dx_config,
            wgrad_config=wgrad_config,
            down_forward_config=down_forward_config,
            down_dx_config=down_dx_config,
            down_dw_config=down_dw_config,
            raster_group=raster_group,
        )
        self.norm_weight = nn.Parameter(
            torch.ones(d_model, dtype=torch.bfloat16, device=device)
        )

    def forward(self, x):
        mlp_output = self.mlp(x)
        return _FusedAddRMSNormBF16.apply(
            x,
            mlp_output,
            self.norm_weight,
            self.eps,
        )


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
        self.attention = PostNormAttention(
            d_model,
            n_heads,
            max_seq_len,
            qkv_forward_config=qkv_forward_config,
            dx_config=qkv_dx_config,
            dw_config=qkv_dw_config,
            attention_window=attention_window,
            device=device,
        )
        self.mlp = PostNormSwiGLU(
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

    def forward(self, x, cu_seqlens, max_seqlen):
        x = self.attention(x, cu_seqlens, max_seqlen)
        return self.mlp(x)


class FP8PostNormDecoderTransformer(nn.Module):
    """Causal post-RMSNorm decoder with MXFP8 attention and SwiGLU."""

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
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward_hidden(self, tokens, cu_seqlens, max_seqlen):
        if cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(torch.int32)
        x = F.embedding(tokens, self.embedding)
        for block in self.blocks:
            x = block(x, cu_seqlens, max_seqlen)
        return x

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
        del position_ids
        if targets.numel() != tokens.numel():
            raise ValueError(
                "targets and tokens must contain the same number of elements"
            )
        hidden = self.forward_hidden(tokens, cu_seqlens, max_seqlen)
        if fused_linear_ce:
            if reduction not in ("mean", "sum"):
                raise ValueError(
                    "FLA fused linear CE supports mean or sum reduction"
                )
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
        for module in iter_fp8_modules(self):
            module.refresh_fp8_weights()
        return self

    def bind_optimizer(self, optimizer):
        return bind_fp8_optimizer(self, optimizer)
