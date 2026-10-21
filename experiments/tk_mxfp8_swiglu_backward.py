"""End-to-end MXFP8 SwiGLU backward wiring using existing CUDA kernels.

The fused elementwise kernel computes and quantizes dG/dU in both directions.
The dual-gradient kernel consumes the row-wise outputs for dX, while the
existing MXFP8 wgrad kernel consumes the column-wise outputs and a cached
column quantization of X for dWg/dWu.
"""

import torch

from experiments.tk_mxfp8_swiglu_dgrad import SwiGLUDgradLibrary
from experiments.tk_silu_bwd import SiluBwdQuantizeLibrary


class MXFP8SwiGLUBackward:
    """Reusable and CUDA-graph-safe backward plan for fixed allocations."""

    def __init__(
        self,
        dy,
        half_gate,
        up,
        x,
        weight_gate,
        weight_up,
        weight_gate_scale,
        weight_up_scale,
        x_column=None,
        x_column_scale=None,
        *,
        dx_config=6,
        wgrad_config=12,
        raster_group=13,
    ):
        from experiments.tk_mxfp8_wgrad import ColumnQuantizePlan, Library as WgradLibrary

        if (
            x.ndim != 2
            or dy.ndim != 2
            or dy.shape[0] != x.shape[0]
            or weight_gate.shape != (dy.shape[1], x.shape[1])
            or weight_up.shape != weight_gate.shape
        ):
            raise ValueError("Expected X[M,K], dY[M,H], and weights [H,K]")
        self.dy, self.half_gate, self.up, self.x = dy, half_gate, up, x
        self.weight_gate, self.weight_up = weight_gate, weight_up
        self.weight_gate_scale = weight_gate_scale
        self.weight_up_scale = weight_up_scale

        self.quantize = SiluBwdQuantizeLibrary().plan(dy, half_gate, up)
        if (x_column is None) != (x_column_scale is None):
            raise ValueError("x_column and x_column_scale must be supplied together")
        mp = (x.shape[0] + 127) // 128 * 128
        if x_column is not None:
            if x_column.shape != (mp, x.shape[1]):
                raise ValueError("Invalid prequantized X column buffer")
            if x_column_scale.shape != (
                mp // 128,
                (x.shape[1] + 15) // 16 * 16,
            ):
                raise ValueError("Invalid prequantized X column scales")
        self.x_column, self.x_column_scale = x_column, x_column_scale
        self.quantize_x = None if x_column is not None else ColumnQuantizePlan(x)
        self.dx_plan = SwiGLUDgradLibrary().plan(
            self.quantize.gate_row,
            self.quantize.up_row,
            weight_gate,
            weight_up,
            self.quantize.gate_row_scale,
            self.quantize.up_row_scale,
            weight_gate_scale,
            weight_up_scale,
            config=dx_config,
            raster_group=raster_group,
        )
        wgrad_library = WgradLibrary()
        self.gate_weight_grad = wgrad_library.plan(
            self.quantize.gate_col,
            x_column if x_column is not None else self.quantize_x.out,
            self.quantize.gate_col_scale,
            x_column_scale if x_column_scale is not None else self.quantize_x.scales,
            config=wgrad_config,
            raster_group=raster_group,
        )
        self.up_weight_grad = wgrad_library.plan(
            self.quantize.up_col,
            x_column if x_column is not None else self.quantize_x.out,
            self.quantize.up_col_scale,
            x_column_scale if x_column_scale is not None else self.quantize_x.scales,
            config=wgrad_config,
            raster_group=raster_group,
        )

    def __call__(self):
        self.quantize()
        if self.quantize_x is not None:
            self.quantize_x()
        dx = self.dx_plan()
        gate_weight_grad = self.gate_weight_grad()
        up_weight_grad = self.up_weight_grad()
        return dx, gate_weight_grad, up_weight_grad


def build():
    from experiments.tk_mxfp8_swiglu_dgrad import build as build_dx
    from experiments.tk_mxfp8_wgrad import build as build_wgrad
    from experiments.tk_silu_bwd import build as build_elementwise

    build_elementwise()
    build_dx()
    build_wgrad()
