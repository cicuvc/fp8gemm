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

## Execution and optimizer integration

Modules own execution plans and workspaces for fixed shapes and should be
called sequentially on a single CUDA stream. Warm up each new shape before
CUDA Graph capture. When using a standard optimizer, call
`mxfp8.bind_optimizer(model, optimizer)` to register FP8 weight-cache refreshes
for all nested modules. `FusedAdamWMXFP8` updates the caches directly while
keeping buffer addresses stable for CUDA Graph replay.

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
