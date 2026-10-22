"""Experimental chunked MXFP8 LM head + cross entropy on SM120.

Returns a scalar SUM of token losses; divide it externally for a mean. No bias,
ignore_index, smoothing, or reduction option. Forward computes first-order dX/dW
chunk by chunk; backward only scales the saved gradients. Z is BF16, LSE/loss
are FP32, and dZ is independently MXFP8-quantized along both GEMM reduction axes.

Example::

    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=1024)
    loss = criterion(hidden_bf16, targets_int64, lm_head_weight) / hidden_bf16.shape[0]
    loss.backward()

Weights may be FP32 masters or BF16 (including a tied embedding parameter).
Optional quantized_weight=(E4M3, E8M0) avoids requantizing W; the caller must keep
that cache synchronized. Quantization uses the existing linear's STE convention.
Build CUDA dependencies as for MXFP8Linear, plus python -m experiments.tk_mxfp8_lse.
"""

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from experiments.mxfp8_core import quantize_weight
from experiments.mxfp8_ce import CEPlan, merge_lse, _validate_target
from experiments.tk_mxfp8_lse import Library as LSELibrary, CONFIGS
from experiments.tk_mxfp8_dgrad import Library as DgradLibrary
from experiments.tk_mxfp8_wgrad import Library as WgradLibrary
from experiments.tk_silu_bwd import SiluBwdQuantizeLibrary


class _ChunkState:
    def __init__(self, x, w, ws, module):
        self.xq = module._quant.plan_direct(x)
        self.gemm = module._gemm.plan(
            self.xq.row_fp8, w, self.xq.row_scale, ws,
            config=module.forward_config, raster_group=module.raster_group,
        )
        m, h = x.shape
        v = w.shape[0]
        self.lse = torch.empty(m, device=x.device, dtype=torch.float32)
        self.loss = torch.empty_like(self.lse)
        self.ce = CEPlan(m, v, x.device)
        self.dx = module._dgrad.plan(
            self.ce.row_fp8, w, self.ce.row_scale, ws,
            config=module.dx_config, raster_group=module.raster_group,
        )
        self.dw = module._wgrad.plan(
            self.ce.column_fp8, self.xq.column_fp8,
            self.ce.column_scale, self.xq.column_scale,
            config=module.dw_config, raster_group=module.raster_group,
        )

    def __call__(self, x, target, w, ws):
        self.xq.rebind(x)
        self.gemm.rebind(self.xq.row_fp8, w, self.xq.row_scale, ws)
        self.dx.rebind(self.ce.row_fp8, w, self.ce.row_scale, ws)
        try:
            self.xq()
            logits, partial = self.gemm()
            merge_lse(logits, partial, target, lse=self.lse, loss=self.loss)
            self.ce(logits, self.lse, target)
            return self.loss, self.dx(), self.dw()
        finally:
            # Cached workspaces must not retain a previous activation graph or W.
            # The descriptors are rebound before their next launch.
            self.xq.x = None
            self.gemm.refs = ()
            self.dx.refs = []


class _FusedCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, weight, wq, ws, module):
        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        losses = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        with torch.cuda.device(x.device):
            for start in range(0, x.shape[0], module.chunk_size):
                end = min(start + module.chunk_size, x.shape[0])
                c_x = x[start:end]
                state = module._state(c_x, wq, ws)
                c_loss, c_dx, c_dw = state(c_x, target[start:end], wq, ws)
                losses[start:end].copy_(c_loss)
                dx[start:end].copy_(c_dx)
                dw.add_(c_dw)
        # Independent storage: future forwards may immediately reuse every plan.
        # Accumulate dW in FP32 and cast only after all chunks have contributed.
        ctx.save_for_backward(dx, dw.to(weight.dtype))
        return losses.sum()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        dx, dw = ctx.saved_tensors
        # No host read of grad_output, and no mutation of saved tensors: CUDA
        # Graph capture and repeated backward with different scalars both work.
        return dx * grad_output, None, dw * grad_output, None, None, None


class MXFP8FusedLinearCrossEntropy(nn.Module):
    """Reusable sequential-stream criterion, with bounded per-chunk workspaces.

    X must be contiguous BF16 [M,H], W contiguous FP32/BF16 [V,H], and targets
    contiguous int64 [M] in [0,V). H/V must be multiples of 64 and satisfy the
    selected GEMM panels. Default configs require multiples of 128. A final
    short token chunk is supported via the quantizer's zero padding.

    Warm up once for each shape before CUDA Graph capture. This object owns
    mutable scratch buffers and must not be called concurrently on two streams.
    Only first-order differentiation is supported, as in the existing linear.
    """
    def __init__(self, *, chunk_size=1024, forward_config=8, dx_config=12,
                 dw_config=12, raster_group=13):
        super().__init__()
        if type(chunk_size) is not int or chunk_size <= 0 or chunk_size % 128:
            raise ValueError("chunk_size must be a positive multiple of 128")
        if type(forward_config) is not int or forward_config not in CONFIGS:
            raise ValueError("Unsupported forward LSE config")
        self.chunk_size = chunk_size
        self.forward_config, self.dx_config, self.dw_config = forward_config, dx_config, dw_config
        self.raster_group = raster_group
        self._states = {}
        self._quant = self._gemm = self._dgrad = self._wgrad = None

    def _state(self, x, w, ws):
        if self._gemm is None:
            self._quant, self._gemm = SiluBwdQuantizeLibrary(), LSELibrary()
            self._dgrad, self._wgrad = DgradLibrary(), WgradLibrary()
        key = (x.device, tuple(x.shape), tuple(w.shape), self.forward_config,
               self.dx_config, self.dw_config, self.raster_group)
        if key not in self._states:
            self._states[key] = _ChunkState(x, w, ws, self)
        return self._states[key]

    def clear_workspace(self):
        """Drop cached per-shape workspaces outside CUDA Graph capture/replay."""
        self._states.clear()

    def forward(self, x, target, weight, *, quantized_weight=None):
        if (x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda or
                not x.is_contiguous() or min(x.shape) <= 0):
            raise ValueError("Expected nonempty contiguous CUDA BF16 X[M,H]")
        if (weight.ndim != 2 or weight.shape[1] != x.shape[1] or weight.shape[0] <= 0 or
                weight.dtype not in (torch.bfloat16, torch.float32) or
                weight.device != x.device or not weight.is_contiguous()):
            raise ValueError("Expected contiguous FP32/BF16 W[V,H] on X's device")
        if weight.shape[0] % 64 or x.shape[1] % 64:
            raise ValueError("V and H must be multiples of 64")
        if torch.cuda.get_device_capability(x.device) != (12, 0):
            raise RuntimeError("SM120 required")
        _validate_target(target, x.shape[0], x.device)
        if quantized_weight is None:
            with torch.no_grad():
                wq, ws = quantize_weight(weight.detach().float())
        else:
            wq, ws = quantized_weight
            if (wq.shape != weight.shape or wq.dtype != torch.float8_e4m3fn or
                    ws.shape != (weight.shape[0] // 32, weight.shape[1] // 32) or
                    ws.dtype != torch.uint8 or wq.device != x.device or ws.device != x.device or
                    not wq.is_contiguous() or not ws.is_contiguous()):
                raise ValueError("Invalid quantized weight cache")
        return _FusedCEFunction.apply(x, target, weight, wq, ws, self)


def fused_linear_cross_entropy_loss(x, target, weight, *, chunk_size=1024,
                                    forward_config=8, dx_config=12, dw_config=12,
                                    raster_group=13, quantized_weight=None):
    """One-shot scalar summed CE; use the module to reuse plans across calls."""
    return MXFP8FusedLinearCrossEntropy(
        chunk_size=chunk_size, forward_config=forward_config, dx_config=dx_config,
        dw_config=dw_config, raster_group=raster_group,
    )(x, target, weight, quantized_weight=quantized_weight)
