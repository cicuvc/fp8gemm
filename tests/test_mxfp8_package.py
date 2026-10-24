"""Public package contract tests that do not require a GPU."""

import inspect

import pytest
import torch

import mxfp8
from mxfp8 import MXFP8Linear, MXFP8RMSNorm
from mxfp8._core import cuda_tool_path
from mxfp8.runtime import _GROUPS, _EXTENSIONS


def test_public_api_is_importable():
    for name in mxfp8.__all__:
        assert hasattr(mxfp8, name)


def test_bias_is_not_part_of_public_linear_or_mlp_api():
    assert "bias" not in inspect.signature(MXFP8Linear).parameters
    assert "bias" not in inspect.signature(mxfp8.MXFP8SwiGLU).parameters


def test_extension_groups_only_name_known_extensions():
    for extensions in _GROUPS.values():
        assert set(extensions) <= set(_EXTENSIONS)


def test_unknown_extension_group_is_rejected():
    with pytest.raises(ValueError, match="unknown MXFP8 extension group"):
        mxfp8.build_extensions("not-a-kernel")


def test_cuda_tool_path_prefers_cuda_home(monkeypatch, tmp_path):
    compiler = tmp_path / "bin" / "nvcc"
    compiler.parent.mkdir()
    compiler.write_text("")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    assert cuda_tool_path("nvcc") == str(compiler)


def test_rmsnorm_can_be_constructed_before_moving_to_cuda():
    norm = MXFP8RMSNorm(64)
    assert norm.weight.dtype == torch.bfloat16
    assert norm.weight.shape == (64,)
    assert torch.equal(norm.weight, torch.ones(64, dtype=torch.bfloat16))
    assert set(norm.state_dict()) == {"weight"}


def test_rmsnorm_rejects_unsupported_width():
    with pytest.raises(ValueError, match="multiple of 32"):
        MXFP8RMSNorm(48)


def test_sm120_check_fails_cleanly_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="available CUDA device"):
        mxfp8.require_sm120()
