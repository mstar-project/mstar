"""``copy_blocks``, the Ulysses exchange's packing copy, against torch's copy of the same strided views."""

import pytest
import torch

from mstar.distributed.row_copy import copy_blocks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("batch,world,rows,width", [(2, 2, 37, 1536), (1, 3, 5, 2560), (2, 2, 300, 7), (3, 4, 1, 384)])
def test_copy_blocks_packs_and_unpacks_like_torch(batch, world, rows, width, dtype):
    x = torch.randn(batch, rows, world * width, device="cuda").to(dtype)
    src = x.unflatten(2, (world, width)).transpose(1, 2)          # [B, P, n, width], each row a strided run
    packed = torch.empty(batch, world, rows, width, device="cuda", dtype=dtype)
    copy_blocks(packed, src)
    assert torch.equal(packed, src.contiguous())
    back = torch.empty_like(x)
    copy_blocks(back.unflatten(2, (world, width)).transpose(1, 2), packed)
    assert torch.equal(back, x)


def test_copy_blocks_refuses_mismatched_views():
    a = torch.empty(1, 2, 3, 4, device="cuda")
    with pytest.raises(ValueError):
        copy_blocks(a, torch.empty(1, 2, 3, 5, device="cuda"))
