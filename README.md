# fp8gemm

An experimental MXFP8 training stack for NVIDIA GeForce RTX 50-series gaming
GPUs (Blackwell, SM120). The goal is to unlock the full MXFP8 Tensor Core
training throughput of these cards (bf16a32 and fp8a32 are limited while mxfp8a32 are 
not on these GPUs), covering both forward and backward matrix multiplications, 
including activation and weight gradients.

The project combines custom CUDA kernels with PyTorch autograd modules,
quantization, fused loss computation, and optimizer integration. It is intended
for training workloads that can adopt its BF16 activation and MXFP8 compute
path. Full Tensor Core utilization is the optimization target; achieved
end-to-end throughput depends on model shapes, memory traffic, and the rest of
the training pipeline.

## Usage

`mxfp8` is the public entry point for training code:

```python
import torch
from mxfp8 import MXFP8Linear, MXFP8RMSNorm, MXFP8SwiGLU

linear = MXFP8Linear(1024, 4096, device="cuda")
norm = MXFP8RMSNorm(1024, device="cuda")
mlp = MXFP8SwiGLU(1024, 4096, 1024, device="cuda")

x = torch.randn((256, 1024), device="cuda", dtype=torch.bfloat16)
y = linear(x)
```

## Requirements

- SM120 hardware is required; CUDA kernels target `compute_120a`.
- Input activations must be contiguous CUDA BF16 tensors.
- Linear layers and MLPs are bias-free and use FP32 master weights.
- The CUDA toolkit's `nvcc` must be available. The default lookup checks
  `$CUDA_HOME/bin/nvcc` and `/usr/local/cuda/bin/nvcc`.
- Missing or outdated CUDA extensions in `runs/` are built on demand when
  modules are first created. For ahead-of-time builds, call
  `mxfp8.build_extensions()` before creating modules and set
  `MXFP8_NO_AUTO_BUILD=1` to disable automatic builds at runtime.

## Public components

- `MXFP8Linear`
- `MXFP8DownProjection`
- `MXFP8SwiGLU` / `MXFP8MLP`
- `MXFP8RMSNorm`
- `MXFP8FusedLinearCrossEntropy`
- `FusedAdamWMXFP8`
- `quantize_weight`
- `quantize_weight_rowwise`

## Optional MXFP8 Stream-K

The persistent MXFP8 kernels prefetch scales in the producer and use factored
transposing shared-memory loads. Tensor maps are encoded by the CUDA driver
when plans are created/rebound, including for the K128 FP32-scale kernels.

Stream-K is opt-in; existing modules keep their persistent routing by default:

```python
from mxfp8 import MXFP8SwiGLU, configure_ffn_streamk

mlp = MXFP8SwiGLU(1024, 2944, 1024, device="cuda")
report = configure_ffn_streamk(mlp, reduction="atomic")  # before first forward
# To allow Stream-K down forward too:
# report = configure_ffn_streamk(mlp, reduction="atomic", down_forward="streamk")
```

`reduction="deterministic"` uses exclusive FP32 partials and increasing-K
fixup; `"atomic"` uses TMA reduce-add into one slot per split tile. Atomic
arrival order is not deterministic, and neither mode promises bitwise equality
with an unsplit dot product. The adapter preserves parameters, autograd and
optimizer bindings. Down forward stays persistent unless explicitly enabled;
other forward/dX plans use a conservative wave-efficiency heuristic, and dW
uses Stream-K. Shapes outside the fixed Stream-K panels retain the original
persistent plan. Changing the configuration after installation is rejected.

Warm up each shape before CUDA Graph capture. All adapted layers share one
scratch allocation and must execute in order on one stream; concurrent model
executions must not share this adapter. Every atomic replay includes its clear
and fixup. The report records routes and used/allocated scratch bytes.
`build_extensions("streamk")` builds the ten Stream-K variants; normal module
construction builds their persistent dependencies. Low-level family plans are
available in `mxfp8._kernels.streamk` and `streamk_wgrad` with explicit shape,
SM-count, tail-wave and reduction settings.

These changes were ported from dism-core's `42a199c`, `a0893a0`, `85d003d`
and its 2026-11-24 working-tree FFN integration/TMA fix. No historical kernels
are retained. CUDA device translation units remain Torch-free. The source
project observed downstream training differences when changing forward
reduction order; Stream-K is therefore an execution option, not a guarantee of
training equivalence.

## FP8 forward GEMM with FP32 block scales

`fp8_gemm` also provides a forward-only SM120 GEMM for prequantized E4M3
operands with FP32 dequantization scales at granularity 128:

| Argument | Shape | Dtype |
| --- | --- | --- |
| `a` | `[M,K]` | `torch.float8_e4m3fn` |
| `b` | `[N,K]` | `torch.float8_e4m3fn` |
| `sfA` | `[M,K//128]` | `torch.float32` |
| `sfB` | `[N//128,K//128]` | `torch.float32` |
| output | `[M,N]` | BF16 by default; optional FP32 |

All tensors must be contiguous on the same SM120 GPU. M is any positive
integer; N and K must be positive multiples of 128. Scales multiply the
stored FP8 values; they are not inverse scales. For each K128 block `g`,
the kernel computes an FP32 partial dot product, multiplies it by
`sfA[m,g] * sfB[n//128,g]`, and adds it to a separate FP32 accumulator.
The result is `A_dequantized @ B_dequantized.T`, subject to floating-point
rounding. FP32 scale values are used directly, without E8M0 conversion.
The kernel computes the unscaled dot products with `QMMA.SF` and constant
internal E8M0 scales of 127 (1.0), using the same ThunderKittens
block-scaled MMA and shared-memory loader as the MXFP8 kernels.

```python
from mxfp8 import FP8GemmPlan, fp8_gemm

# a, b, sfA, sfB are already quantized/prepared in the layouts above.
y = fp8_gemm(a, b, sfA, sfB)
y_fp32 = fp8_gemm(a, b, sfA, sfB, dtype=torch.float32)

# Reuse descriptors/output for repeated execution or CUDA Graph capture.
plan = FP8GemmPlan(a, b, sfA, sfB)
y = plan()
print(plan.info)
```

This API does not quantize inputs or implement backward; inputs requiring
gradients are rejected. Construct and warm up plans before graph capture.
Calls read current input/scale contents and overwrite the same output;
keep tensor storage and metadata fixed and serialize calls to each plan.
An optional `out=` tensor selects BF16 or FP32 output through its dtype.
Build ahead of time with `mxfp8.build_extensions("fp8_gemm")`, or let the
first plan build the extension. Four configurations cover N tiles of
64/128 and 2/3 TMA stages, selected with `FP8GemmPlan(..., config=0..3)`.
The default is config 0: a `128x64x128` CTA tile with two TMA stages.

Run `python benchmarks/bench_fp8_gemm.py` for warm CUDA Graph timings of
each configuration and a BF16 `torch.mm` latency baseline. Quantization
and plan setup are excluded.

## FP8 SwiGLU training with FP32 block scales

`FP8Linear`, `FP8DownProjection`, and `FP8SwiGLU` (`FP8MLP`) extend the
K128 FP32-scale path to forward and backward:

```python
from mxfp8 import FP8SwiGLU, bind_optimizer

mlp = FP8SwiGLU(1024, 4096, 1024, device="cuda")
optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
hook = bind_optimizer(mlp, optimizer)
x = torch.randn(129, 1024, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
y = mlp(x)
y.float().square().mean().backward()
optimizer.step()
```

Inputs are contiguous BF16 `[M,K]`. Feature dimensions are positive multiples
of 128; any positive M is supported. Parameters and weight gradients are FP32;
outputs and input gradients are BF16. Master parameters use the same names as
the MXFP8 modules: `gate_weight`, `up_weight`, and `down_proj.master_weight`.

| Operation | Quantization / implementation |
| --- | --- |
| Activation and gradient quantization | Independent rowwise and columnwise groups of 128; column representation stored transposed, M padded with zeros |
| Weight quantization | FP32 master → E4M3, one FP32 scale per 128×128 block; transpose and transposed scales cached without requantization |
| Gate/up forward | Fused CUDA GEMM sharing A, with SiLU×up epilogue |
| Down forward and dHidden | Rowwise activation/gradient × blockwise weight GEMM |
| SiLU backward | FP32 elementwise derivative, fused with row/column quantization of dGate and dUp |
| Gate/up dX | Fused sum of two GEMMs, rounded to BF16 once |
| All three dW | Both operands column-quantized; rowwise scales on both sides of GEMM, FP32 output |

All GEMMs use the existing TMA ring pipeline, shared-memory loader, and unit
E8M0 `QMMA.SF` instructions, with external FP32 scaling after each K128 block.
Quantization uses `scale = max(abs(block))/448` (1 for an all-zero block,
clamped to the smallest normal FP32 value otherwise). Inputs should be finite.
Gate/up projections are saved as BF16; SiLU uses those values in FP32 and
rounds its output to BF16. Its backward similarly rounds dGate/dUp to BF16
before quantization. The custom backward is a first-order quantized training
approximation; double backward is unsupported.

`FP8GemmPlan(..., b_scale_mode="row", dtype=torch.float32)` exposes the dW
primitive: B's scale shape is `[N,K//128]`. It currently uses default launch
settings (`config=0`, `sms=0`, `raster_group=8`). Fused forward/dX also use
128×64×128 tiles and two TMA stages. These are initial configurations, not
an autotuned FFN implementation.

Caches refresh automatically after detected eager parameter updates. Use
`bind_optimizer` for graph-captured optimizer steps: refresh operations are
then part of the graph and cache addresses remain stable. `FusedAdamWMXFP8`
also works; its K128 FP32-scale caches are refreshed by separate quantization
kernels after the update. Its existing E8M0 fused epilogue remains for MXFP8.
Warm up forward **and backward** for each shape on the capture stream before
capture. Workspaces preserve independent live forwards and retained autograd
graphs; returned outputs and gradients are not overwritten by later calls.

Run `python benchmarks/bench_fp8_training.py` for forward and forward+backward
CUDA Graph timings including activation/gradient quantization, excluding
optimizer updates and weight-cache refresh.

## Execution and optimizer integration

Modules own execution plans and workspaces for fixed shapes and should be
called sequentially on a single CUDA stream. Warm up each new shape before
CUDA Graph capture. When using a standard optimizer, call
`mxfp8.bind_optimizer(model, optimizer)` to register FP8 weight-cache refreshes
for all nested modules. `FusedAdamWMXFP8` updates MXFP8 caches directly;
K128 FP32-scale caches use a separate refresh. Both keep buffer addresses
stable for CUDA Graph replay.

## Fused LM-head cross entropy

`MXFP8FusedLinearCrossEntropy` uses MXFP8 for `X @ W.T` and `dZ.T @ X`,
and BF16 for `dX = dZ @ W`. Forward weights have one E8M0 scale per embedding
and 32 hidden features: for `W[V,D]`, scales are `uint8[V,D/32]`. V and D
must still satisfy the selected kernels' alignment requirements. Other layers
continue to use the existing 32x32 blockwise weight quantization.

```python
from mxfp8 import MXFP8FusedLinearCrossEntropy, quantize_weight_rowwise

criterion = MXFP8FusedLinearCrossEntropy(chunk_size=1024, balance="token")
# hidden: contiguous BF16 [tokens,D]; weight: contiguous FP32/BF16 [V,D]
loss = criterion(hidden, targets, weight)
loss.backward()

# Optional forward cache; regenerate after each weight update.
cache = quantize_weight_rowwise(weight)
loss = criterion(hidden, targets, weight, quantized_weight=cache)
```

`quantize_weight()` and `FusedAdamWMXFP8`'s bound outputs retain their blockwise
layout and cannot supply this CE cache. dX always reads the original weight
rounded to BF16, including when a forward cache is supplied. FP32 weights are
converted once per invocation; BF16 weights (including tied embeddings) are
used directly. CE computes dZ in FP32 before rounding it to BF16 for dX and
independently quantizing it along tokens for dW. This is an approximate
first-order backward; it does not differentiate the quantizer. dW is accumulated
across chunks in FP32 and then converted to the parameter dtype.

The legacy `dx_config` argument remains accepted but has no effect on the BF16
GEMM. After loss/LSE/NLL are computed, BF16 dZ overwrites the logits workspace.
Column-quantized dZ has a separate FP8 buffer; there is no separate BF16 dZ,
row-quantized dZ, or second FP8 weight copy. Loss, softcap, balancing,
ignored targets, and optional NLL keep their existing semantics. Warm up the
criterion before CUDA Graph capture, and refresh both the original weight and
any supplied forward cache before replaying an updated training step.

Implementations live in `mxfp8/`: `_kernels/` contains CUDA adapters, and
`_impl/` contains internal autograd functions and composed execution plans.
The public API is bias-free and is not a drop-in replacement for
`torch.nn.Linear`'s parameter layout or optional bias.
