"""Chunked MXFP8 LM head and cross entropy on SM120.

By default returns a scalar SUM of non-ignored token losses. balance="token"
averages valid tokens; balance="document" averages the per-document valid-token
means, excluding empty/all-ignored documents. No bias or smoothing option.
With autograd enabled, forward computes first-order dX/dW
chunk by chunk; backward only scales the saved gradients. Z is BF16, LSE/loss
are FP32. Forward and dW use MXFP8; dX uses BF16 dZ @ original BF16 W.
Forward W has one E8M0 scale per embedding and 32 hidden features.

Example::

    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=1024, balance="token")
    loss = criterion(hidden_bf16, targets_int64, lm_head_weight)
    loss.backward()

Weights may be FP32 masters or BF16 (including a tied embedding parameter).
Optional quantized_weight=(E4M3[V,H], uint8[V,H/32]) avoids requantizing W;
the caller must keep that rowwise cache synchronized. Use
quantize_weight_rowwise(), not the blockwise quantize_weight(), to create it.
The approximate first-order backward uses original W rounded to BF16 for dX,
and independently quantized dZ/X for dW.
Build CUDA dependencies with :func:`mxfp8.build_extensions`.
"""

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .._kernels.rowwise_quantize import quantize_weight_rowwise
from .._kernels.lse import CONFIGS, Library as LSELibrary
from .._kernels.silu import SiluBwdQuantizeLibrary
from .._kernels.wgrad import Library as WgradLibrary
from .balance import document_weights, reduce_balanced_loss, validate_cu_seqlens
from .ce import (
    CEPlan,
    _validate_ignore_index,
    _validate_softcap,
    _validate_target,
    merge_lse,
)


class _ChunkState:
    def __init__(self, x, w, ws, module, need_grad):
        self.xq = module._quant.plan_direct(x)
        self.gemm = module._gemm.plan(
            self.xq.row_fp8,
            w,
            self.xq.row_scale,
            ws,
            config=module.forward_config,
            raster_group=module.raster_group,
            softcap=module.softcap,
            rowwise_weight=True,
        )
        m = x.shape[0]
        v = w.shape[0]
        self.lse = torch.empty(m, device=x.device, dtype=torch.float32)
        self.ce = self.dx = self.dw = None
        if not need_grad:
            return
        self.ce = CEPlan(m, v, x.device, bf16_dgrad=True, inplace=True)
        self.dx = torch.empty_like(x)
        self.dw = module._wgrad.plan(
            self.ce.column_fp8,
            self.xq.column_fp8,
            self.ce.column_scale,
            self.xq.column_scale,
            config=module.dw_config,
            raster_group=module.raster_group,
        )

    def __call__(
        self,
        x,
        target,
        w,
        ws,
        ignore_index,
        loss,
        row_weight=None,
        nll=None,
        weight_bf16=None,
    ):
        self.xq.rebind(x)
        self.gemm.rebind(self.xq.row_fp8, w, self.xq.row_scale, ws)
        try:
            self.xq()
            logits, partial = self.gemm()
            merge_lse(
                logits,
                partial,
                target,
                lse=self.lse,
                loss=loss,
                nll=nll,
                ignore_index=ignore_index,
                row_weight=row_weight,
                softcap=self.gemm.softcap,
            )
            if self.ce is None:
                return None, None
            # Loss/LSE/NLL have consumed logits. CE now overwrites this BF16
            # workspace with dZ; softcap's derivative is computed before stores.
            self.ce(
                logits,
                self.lse,
                target,
                ignore_index=ignore_index,
                row_weight=row_weight,
                softcap=self.gemm.softcap,
            )
            # Explicit output keeps this BF16 GEMM independent of autocast.
            torch.mm(self.ce.row_bf16, weight_bf16, out=self.dx)
            return self.dx, self.dw()
        finally:
            # Cached workspaces must not retain a previous activation graph or W.
            # The descriptors are rebound before their next launch.
            self.xq.x = None
            self.gemm.refs = ()


@torch.no_grad()
def _forward(x, target, weight, wq, ws, module, cu_seqlens, return_nll, need_grad):
    dx = torch.empty_like(x) if need_grad else None
    dw = torch.zeros_like(weight, dtype=torch.float32) if need_grad else None
    # Outputs own their storage independently of reusable chunk workspaces.
    losses = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    separate_nll = return_nll and module.balance == "document"
    nll = torch.empty_like(losses) if separate_nll else losses if return_nll else None
    row_weight = doc_counts = None
    with torch.cuda.device(x.device):
        # Convert once per invocation, outside the chunk loop. BF16 parameters
        # (including tied embeddings) are used directly. Never use dequantized Wq.
        weight_bf16 = weight.to(torch.bfloat16) if need_grad else None
        if module.balance == "document":
            row_weight, doc_counts = document_weights(
                target,
                cu_seqlens,
                ignore_index=module.ignore_index,
            )
        for start in range(0, x.shape[0], module.chunk_size):
            end = min(start + module.chunk_size, x.shape[0])
            c_x = x[start:end]
            state = module._state(c_x, wq, ws, need_grad=need_grad)
            c_weight = None if row_weight is None else row_weight[start:end]
            c_nll = nll[start:end] if separate_nll else None
            c_dx, c_dw = state(
                c_x,
                target[start:end],
                wq,
                ws,
                module.ignore_index,
                losses[start:end],
                c_weight,
                c_nll,
                weight_bf16,
            )
            if need_grad:
                dx[start:end].copy_(c_dx)
                dw.add_(c_dw)
        if module.balance is None:
            loss, inv = losses.sum(), None
        else:
            loss, inv = reduce_balanced_loss(
                losses,
                target,
                balance=module.balance,
                doc_counts=doc_counts,
                ignore_index=module.ignore_index,
            )
    # Accumulate dW in FP32 and cast only after all chunks have contributed.
    return loss, nll, dx, dw.to(weight.dtype) if need_grad else None, inv


class _FusedCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, weight, wq, ws, module, cu_seqlens, return_nll):
        loss, nll, dx, dw, inv = _forward(
            x,
            target,
            weight,
            wq,
            ws,
            module,
            cu_seqlens,
            return_nll,
            need_grad=True,
        )
        ctx.save_for_backward(dx, dw, inv)
        if return_nll:
            ctx.mark_non_differentiable(nll)
            ctx.set_materialize_grads(False)
            return loss, nll
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_nll=None):
        dx, dw, inv = ctx.saved_tensors
        # No host read of grad_output, and no mutation of saved tensors: CUDA
        # Graph capture and repeated backward with different scalars both work.
        scale = grad_output if inv is None else grad_output * inv
        return dx * scale, None, dw * scale, None, None, None, None, None


class MXFP8FusedLinearCrossEntropy(nn.Module):
    """Reusable sequential-stream criterion, with bounded per-chunk workspaces.

    X must be contiguous BF16 [M,H], W contiguous FP32/BF16 [V,H], and targets
    contiguous int64 [M] in [0,V) or equal to ignore_index (default -100).
    Ignored rows contribute zero loss and gradients; all-ignored inputs return
    zero loss and gradients as well. H/V must be multiples of 64 and satisfy the
    selected GEMM panels. Default configs require multiples of 128. A final
    short token chunk is supported via the quantizer's zero padding.

    balance=None preserves summed loss. "token" divides by valid token count;
    "document" uses cu_seqlens[D+1] (CUDA int32/int64) to weight each nonempty
    document equally by its valid-token mean. Boundaries cover [0,M], are
    nondecreasing, and describe documents, not arbitrary attention segments.
    Repeated boundaries (empty documents) are allowed. Normalization is over
    this invocation, not across gradient-accumulation steps or distributed ranks.

    softcap=None disables softcapping. A positive finite constant s applies
    s*tanh(l/s) in FP32 to the raw BF16 logits l. CUDA and Triton both use
    tanh.approx.f32. The chain-rule factor (1-tanh(l/s)**2) is applied before
    BF16 rounding of dZ and MXFP8 column quantization for dW.

    dX uses BF16 dZ @ W.bfloat16(); forward and dW use MXFP8. The legacy
    dx_config argument is accepted for source compatibility and has no effect.
    After loss/LSE/NLL are computed, BF16 dZ overwrites the logits workspace;
    column-quantized dZ for dW retains its separate FP8 buffer.

    forward(..., return_nll=True) returns (loss, nll), with FP32 nll[M] before
    token/document balancing and zero at ignored positions. NLL reflects softcap
    and is non-differentiable; the scalar loss retains its usual gradients.
    With no_grad/inference_mode, or neither X nor W requiring gradients, forward
    skips dZ/dX/dW computation and allocation. eval() alone does not disable gradients.

    Warm up once for each shape before CUDA Graph capture. This object owns
    mutable scratch buffers and must not be called concurrently on two streams.
    Only first-order differentiation is supported, as in the existing linear.
    """

    def __init__(
        self,
        *,
        chunk_size=1024,
        forward_config=8,
        dx_config=12,
        dw_config=12,
        raster_group=13,
        ignore_index=-100,
        balance=None,
        softcap=None,
    ):
        super().__init__()
        self.softcap = _validate_softcap(softcap)
        if balance not in (None, "token", "document"):
            raise ValueError("balance must be None, 'token', or 'document'")
        self.balance = balance
        _validate_ignore_index(ignore_index)
        self.ignore_index = ignore_index
        if type(chunk_size) is not int or chunk_size <= 0 or chunk_size % 128:
            raise ValueError("chunk_size must be a positive multiple of 128")
        if type(forward_config) is not int or forward_config not in CONFIGS:
            raise ValueError("Unsupported forward LSE config")
        self.chunk_size = chunk_size
        self.forward_config, self.dx_config, self.dw_config = (
            forward_config,
            dx_config,
            dw_config,
        )
        self.raster_group = raster_group
        self._states = {}
        self._quant = self._gemm = self._wgrad = None

    def _state(self, x, w, ws, *, need_grad=True):
        if self._gemm is None:
            self._quant, self._gemm = SiluBwdQuantizeLibrary(), LSELibrary()
        if need_grad and self._wgrad is None:
            self._wgrad = WgradLibrary()
        key = (
            x.device,
            tuple(x.shape),
            tuple(w.shape),
            self.forward_config,
            self.dw_config,
            self.raster_group,
            self.softcap,
            need_grad,
        )
        if key not in self._states:
            self._states[key] = _ChunkState(x, w, ws, self, need_grad)
        return self._states[key]

    def clear_workspace(self):
        """Drop cached per-shape workspaces outside CUDA Graph capture/replay."""
        self._states.clear()

    def forward(
        self,
        x,
        target,
        weight,
        *,
        quantized_weight=None,
        cu_seqlens=None,
        return_nll=False,
    ):
        if type(return_nll) is not bool:
            raise ValueError("return_nll must be a bool")
        if (
            x.ndim != 2
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or not x.is_contiguous()
            or min(x.shape) <= 0
        ):
            raise ValueError("Expected nonempty contiguous CUDA BF16 X[M,H]")
        if (
            weight.ndim != 2
            or weight.shape[1] != x.shape[1]
            or weight.shape[0] <= 0
            or weight.dtype not in (torch.bfloat16, torch.float32)
            or weight.device != x.device
            or not weight.is_contiguous()
        ):
            raise ValueError("Expected contiguous FP32/BF16 W[V,H] on X's device")
        if weight.shape[0] % 64 or x.shape[1] % 64:
            raise ValueError("V and H must be multiples of 64")
        if torch.cuda.get_device_capability(x.device) != (12, 0):
            raise RuntimeError("SM120 required")
        _validate_target(target, x.shape[0], x.device)
        if self.balance == "document" or cu_seqlens is not None:
            validate_cu_seqlens(cu_seqlens, x.device)
        if quantized_weight is None:
            with torch.no_grad():
                wq, ws = quantize_weight_rowwise(weight.detach())
        else:
            wq, ws = quantized_weight
            if (
                wq.shape != weight.shape
                or wq.dtype != torch.float8_e4m3fn
                or ws.shape != (weight.shape[0], weight.shape[1] // 32)
                or ws.dtype != torch.uint8
                or wq.device != x.device
                or ws.device != x.device
                or not wq.is_contiguous()
                or not ws.is_contiguous()
            ):
                raise ValueError(
                    "Invalid rowwise quantized weight cache; expected E4M3[V,H], uint8[V,H/32]"
                )
        if not torch.is_grad_enabled() or not (x.requires_grad or weight.requires_grad):
            loss, nll, _, _, _ = _forward(
                x,
                target,
                weight,
                wq,
                ws,
                self,
                cu_seqlens,
                return_nll,
                need_grad=False,
            )
            return (loss, nll) if return_nll else loss
        return _FusedCEFunction.apply(
            x, target, weight, wq, ws, self, cu_seqlens, return_nll
        )


def fused_linear_cross_entropy_loss(
    x,
    target,
    weight,
    *,
    chunk_size=1024,
    forward_config=8,
    dx_config=12,
    dw_config=12,
    raster_group=13,
    quantized_weight=None,
    ignore_index=-100,
    balance=None,
    cu_seqlens=None,
    softcap=None,
    return_nll=False,
):
    """One-shot CE, optionally (loss, non-differentiable FP32 NLL[M])."""
    return MXFP8FusedLinearCrossEntropy(
        chunk_size=chunk_size,
        forward_config=forward_config,
        dx_config=dx_config,
        dw_config=dw_config,
        raster_group=raster_group,
        ignore_index=ignore_index,
        balance=balance,
        softcap=softcap,
    )(
        x,
        target,
        weight,
        quantized_weight=quantized_weight,
        cu_seqlens=cu_seqlens,
        return_nll=return_nll,
    )
