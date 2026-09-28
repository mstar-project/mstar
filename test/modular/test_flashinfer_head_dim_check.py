"""FlashInfer instantiates its paged prefill/decode kernels for head dims 64, 128
and 256. Other sizes either raise inside FlashInfer (32, 48) or run and return
wrong values (72: max |diff| 2.25 against a torch reference where 128 gives
0.004; FlashInfer 0.6.18 on H100). The engine must refuse them before any plan."""
import pytest
import torch

from mstar.engine.resources.attn.wrappers import (
    FLASHINFER_HEAD_DIMS,
    FlashInferDecodeWrapper,
    FlashInferPrefillWrapper,
    check_flashinfer_head_dim,
)

_CUDA = torch.device("cuda")     # a device object only; the check runs before anything is allocated


@pytest.mark.parametrize("head_dim", sorted(FLASHINFER_HEAD_DIMS))
def test_supported_head_dims_pass(head_dim):
    check_flashinfer_head_dim(head_dim)


@pytest.mark.parametrize("head_dim", [32, 48, 72, 80, 96, 192])
def test_other_head_dims_are_refused_with_padding_advice(head_dim):
    with pytest.raises(ValueError, match=rf"head_dim {head_dim} .*Zero-pad"):
        check_flashinfer_head_dim(head_dim)


@pytest.mark.parametrize("cls", [FlashInferPrefillWrapper, FlashInferDecodeWrapper])
def test_wrappers_refuse_before_touching_flashinfer(cls):
    with pytest.raises(ValueError, match="head_dim 72"):
        cls(workspace_buffer=torch.empty(0), num_qo_heads=16, num_kv_heads=16, head_dim=72, page_size=16, device=_CUDA)


def test_non_cuda_devices_are_not_checked():
    """CPU tests drive the wrappers with stand-in FlashInfer modules and toy
    head dims; FlashInfer itself never runs there."""
    check_flashinfer_head_dim(3, torch.device("cpu"))
    check_flashinfer_head_dim(8, torch.device("meta"))
    with pytest.raises(ValueError):
        check_flashinfer_head_dim(72, _CUDA)
