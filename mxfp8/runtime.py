"""Runtime checks and extension build management for the public MXFP8 API."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from ._core import ROOT


@dataclass(frozen=True)
class ExtensionSpec:
    name: str
    builder: Callable[[], None]
    binaries: tuple[Path, ...]
    sources: tuple[Path, ...]


_BUILD_LOCK = threading.RLock()


def _path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def _extension_spec(name: str) -> ExtensionSpec:
    if name == "fp8_gemm":
        from ._kernels import fp8_gemm

        return ExtensionSpec(
            name,
            fp8_gemm.build,
            (fp8_gemm.OUTPUT / "libfp8_gemm.so",),
            (
                _path("cuda/persistent_fp8_gemm.cu"),
                _path("cuda/fp8_configs.inc"),
                _path("cuda/fp8_shared_load.cuh"),
                _path("cuda/tma_ring_pipe.cuh"),
                *tuple(_path("third_party/thunderkittens/include").rglob("*.cuh")),
            ),
        )
    if name == "quantize":
        from ._kernels import weight_quantize

        return ExtensionSpec(
            name,
            weight_quantize.build,
            (weight_quantize.DEFAULT_OUTPUT / "libquant.so",),
            (
                _path("cuda/persistent_weight_quantize.cu"),
                _path("cuda/mxfp8_quantize.cuh"),
                _path("cuda/linear_tma_buffer.cuh"),
                _path("cuda/tma_ring_pipe.cuh"),
            ),
        )
    if name == "silu":
        from ._kernels import silu

        return ExtensionSpec(
            name,
            silu.build,
            (
                silu.OUTPUT / "libsilu_bwd.so",
                silu.OUTPUT / "libsilu_bwd_quantize.so",
            ),
            (
                _path("cuda/silu_bwd_elementwise.cu"),
                _path("cuda/silu_bwd_quantize.cu"),
                _path("cuda/silu_bwd_common.cuh"),
                _path("cuda/mxfp8_quantize.cuh"),
            ),
        )
    if name == "gemm":
        from ._kernels import gemm

        return ExtensionSpec(
            name,
            gemm.build,
            (gemm.OUTPUT / "libgemm.so",),
            (
                _path("cuda/persistent_mxfp8_gemm.cu"),
                _path("cuda/mxfp8_configs.inc"),
                _path("cuda/mxfp8_epilogue.cuh"),
                _path("cuda/mxfp8_lse.cuh"),
                _path("cuda/tma_ring_pipe.cuh"),
                _path("cuda/fp8_shared_load.cuh"),
            ),
        )
    if name == "dgrad":
        from ._kernels import dgrad

        return ExtensionSpec(
            name,
            dgrad.build,
            (dgrad.OUTPUT / "libdgrad.so",),
            (
                _path("cuda/persistent_mxfp8_dgrad.cu"),
                _path("cuda/mxfp8_dgrad_configs.inc"),
                _path("cuda/tma_ring_pipe.cuh"),
                _path("cuda/fp8_shared_load.cuh"),
            ),
        )
    if name == "wgrad":
        from ._kernels import wgrad

        return ExtensionSpec(
            name,
            wgrad.build,
            (wgrad.OUTPUT / "libwgrad.so",),
            (
                _path("cuda/persistent_mxfp8_wgrad.cu"),
                _path("cuda/mxfp8_wgrad_configs.inc"),
                _path("cuda/tma_ring_pipe.cuh"),
                _path("cuda/fp8_shared_load.cuh"),
            ),
        )
    if name == "swiglu":
        from ._kernels import swiglu

        return ExtensionSpec(
            name,
            swiglu.build,
            (swiglu.OUTPUT / "libswiglu.so",),
            (
                _path("cuda/persistent_mxfp8_swiglu.cu"),
                _path("cuda/mxfp8_swiglu_configs.inc"),
                _path("cuda/tma_ring_pipe.cuh"),
                _path("cuda/fp8_shared_load.cuh"),
            ),
        )
    if name == "swiglu_dgrad":
        from ._kernels import swiglu_dgrad

        return ExtensionSpec(
            name,
            swiglu_dgrad.build,
            (swiglu_dgrad.OUTPUT / "libswd.so",),
            (
                _path("cuda/persistent_mxfp8_swiglu_dgrad.cu"),
                _path("cuda/mxfp8_swiglu_dgrad_configs.inc"),
                _path("cuda/tma_ring_pipe.cuh"),
                _path("cuda/fp8_shared_load.cuh"),
            ),
        )
    if name == "lse":
        from ._kernels import lse

        return ExtensionSpec(
            name,
            lse.build,
            (lse.OUTPUT / "liblse.so",),
            (
                _path("cuda/persistent_mxfp8_lse.cu"),
                _path("cuda/persistent_mxfp8_gemm.cu"),
                _path("cuda/mxfp8_configs.inc"),
                _path("cuda/mxfp8_lse_configs.inc"),
                _path("cuda/mxfp8_epilogue.cuh"),
                _path("cuda/tma_ring_pipe.cuh"),
                _path("cuda/fp8_shared_load.cuh"),
            ),
        )
    raise ValueError(f"unknown MXFP8 extension group: {name!r}")


_GROUPS = {
    "linear": ("quantize", "silu", "gemm", "dgrad", "wgrad"),
    "mlp": (
        "quantize",
        "silu",
        "gemm",
        "dgrad",
        "wgrad",
        "swiglu",
        "swiglu_dgrad",
    ),
    "loss": ("silu", "lse", "wgrad"),
}
_EXTENSIONS = (
    "fp8_gemm",
    "quantize",
    "silu",
    "gemm",
    "dgrad",
    "wgrad",
    "swiglu",
    "swiglu_dgrad",
    "lse",
)
_GROUPS["all"] = _EXTENSIONS


def _expand(names: tuple[str, ...]) -> tuple[str, ...]:
    expanded: list[str] = []
    for name in names or ("all",):
        if name in _GROUPS:
            candidates = _GROUPS[name]
        elif name in _EXTENSIONS:
            candidates = (name,)
        else:
            valid = ", ".join(sorted((*_GROUPS, *_EXTENSIONS)))
            raise ValueError(
                f"unknown MXFP8 extension group {name!r}; choose from {valid}"
            )
        for candidate in candidates:
            if candidate not in expanded:
                expanded.append(candidate)
    return tuple(expanded)


def _is_current(spec: ExtensionSpec) -> bool:
    if not all(binary.is_file() for binary in spec.binaries):
        return False
    oldest_binary = min(binary.stat().st_mtime_ns for binary in spec.binaries)
    return all(
        not source.exists() or source.stat().st_mtime_ns <= oldest_binary
        for source in spec.sources
    )


def extension_status() -> dict[str, bool]:
    """Return whether each built extension exists and is newer than its sources."""
    return {name: _is_current(_extension_spec(name)) for name in _EXTENSIONS}


def build_extensions(*names: str, force: bool = False) -> None:
    """Build one or more extension groups.

    Valid groups are ``linear``, ``mlp``, ``loss``, and ``all``; individual
    builds may also be named explicitly (for example ``gemm`` or ``lse``).
    Build products stay under ``runs/`` and are reused until a source changes.
    """
    if type(force) is not bool:
        raise TypeError("force must be a bool")
    requested = _expand(names)
    with _BUILD_LOCK:
        for name in requested:
            spec = _extension_spec(name)
            if force or not _is_current(spec):
                spec.builder()


def ensure_extensions(*names: str) -> None:
    """Build requested extensions if needed unless auto-build is disabled.

    Set ``MXFP8_NO_AUTO_BUILD=1`` for deployments that build artifacts ahead
    of time. Missing artifacts then produce an actionable error instead of
    invoking NVCC from model construction.
    """
    requested = _expand(names)
    with _BUILD_LOCK:
        pending = tuple(
            name for name in requested if not _is_current(_extension_spec(name))
        )
        if not pending:
            return
        if os.environ.get("MXFP8_NO_AUTO_BUILD", "").lower() in {"1", "true", "yes"}:
            missing = ", ".join(pending)
            raise RuntimeError(
                f"MXFP8 extensions are not built ({missing}); "
                "call mxfp8.build_extensions() in the target environment"
            )
        for name in pending:
            _extension_spec(name).builder()


def require_sm120(device: torch.device | str | int | None = None) -> torch.device:
    """Validate that ``device`` is an SM120 GPU and return a torch device."""
    if not torch.cuda.is_available():
        raise RuntimeError("the MXFP8 modules require an available CUDA device")
    if device is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    else:
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
    if resolved.type != "cuda":
        raise ValueError(f"MXFP8 modules require a CUDA device, got {resolved}")
    capability = torch.cuda.get_device_capability(resolved)
    if capability != (12, 0):
        raise RuntimeError(
            "MXFP8 modules require SM120 (compute capability 12.0); "
            f"{resolved} reports {capability[0]}.{capability[1]}"
        )
    return resolved
