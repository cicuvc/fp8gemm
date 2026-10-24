# fp8gemm

SM120（Blackwell RTX）MXFP8 训练原型。`mxfp8` 是面向训练代码的公开入口：

```python
import torch
from mxfp8 import MXFP8Linear, MXFP8RMSNorm, MXFP8SwiGLU

linear = MXFP8Linear(1024, 4096, device="cuda")
norm = MXFP8RMSNorm(1024, device="cuda")
mlp = MXFP8SwiGLU(1024, 4096, 1024, device="cuda")

x = torch.randn((256, 1024), device="cuda", dtype=torch.bfloat16)
y = linear(x)
```

## 要求

- 计算能力严格为 SM120（`compute_120a`）。
- 输入 activation 为 contiguous CUDA BF16。
- 线性层和 MLP 为 bias-free，权重为 FP32 master。
- CUDA toolkit 中的 `nvcc` 必须可用；默认查找 `$CUDA_HOME/bin/nvcc`
  和 `/usr/local/cuda/bin/nvcc`。
- 首次创建模块时会按需编译缺少或过期的 `runs/` CUDA 扩展。需要预构建时
  设置 `MXFP8_NO_AUTO_BUILD=1` 并先调用 `mxfp8.build_extensions()`。

## 公开组件

- `MXFP8Linear`
- `MXFP8DownProjection`
- `MXFP8SwiGLU` / `MXFP8MLP`
- `MXFP8RMSNorm`
- `MXFP8FusedLinearCrossEntropy`
- `FusedAdamWMXFP8`
- `quantize_weight`

模块持有固定 shape 的执行计划和 workspace，只应在一个 CUDA stream 上顺序调用。
每捕获一个新的 shape 前先运行一次 warmup。使用普通 optimizer 时通过
`mxfp8.bind_optimizer(model, optimizer)` 刷新所有嵌套的 FP8 权重缓存；
`FusedAdamWMXFP8` 会直接更新缓存并保持 CUDA Graph 使用的缓冲区地址稳定。

底层实验和完整 decoder 仍位于 `experiments/`。公开 API 保持当前无 bias
语义，不尝试兼容 `torch.nn.Linear` 的参数布局或可选 bias。
