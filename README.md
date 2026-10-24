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

Modules own execution plans and workspaces for fixed shapes and should be
called sequentially on a single CUDA stream. Warm up each new shape before
CUDA Graph capture. When using a standard optimizer, call
`mxfp8.bind_optimizer(model, optimizer)` to register FP8 weight-cache refreshes
for all nested modules. `FusedAdamWMXFP8` updates the caches directly while
keeping buffer addresses stable for CUDA Graph replay.

Implementations live in `mxfp8/`: `_kernels/` contains CUDA adapters, and
`_impl/` contains internal autograd functions and composed execution plans.
The public API is bias-free and is not a drop-in replacement for
`torch.nn.Linear`'s parameter layout or optional bias.
