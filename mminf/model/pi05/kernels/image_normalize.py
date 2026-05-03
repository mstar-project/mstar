"""GPU-side image range normalization without CPU–GPU sync.

Replaces the two blocking transfers in _prepare_one():

    img_min = float(images.min())   # GPU → CPU (sync)
    img_max = float(images.max())   # GPU → CPU (sync)
    if img_min >= -1e-4 and img_max <= 1.0 + 1e-4:
        images = images * 2.0 - 1.0

with three GPU kernel launches and zero CPU transfers:

    1. torch.min  → GPU scalar tensor (no .item())
    2. torch.max  → GPU scalar tensor (no .item())
    3. Triton kernel reads both scalars from device memory,
       applies x*2-1 if the range says [0,1], identity otherwise.

The Triton kernel avoids materialising a CPU-visible boolean by
keeping the "needs_rescale" predicate entirely in registers and
using tl.where to select between the two outcomes per element.

Falls back to the original sync-based path on CPU tensors or when
Triton is not installed.
"""

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _normalize_range_kernel(
        in_ptr,
        out_ptr,
        img_min_ptr,
        img_max_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Rescale float32 elements from [0,1] to [-1,1] if the global
        min/max (read from GPU memory) indicates the image is in that range.

        Reads ``img_min`` and ``img_max`` from device pointers so the
        decision is made entirely on the GPU — no host branch, no sync.
        """
        g_min = tl.load(img_min_ptr).to(tl.float32)
        g_max = tl.load(img_max_ptr).to(tl.float32)
        # True when pixel values are in the [0, 1] range (data_worker path).
        needs_rescale = (g_min >= -1e-4) & (g_max <= 1.0 + 1e-4)

        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(needs_rescale, x * 2.0 - 1.0, x)
        tl.store(out_ptr + offsets, x, mask=mask)


def normalize_float_images(images: torch.Tensor) -> torch.Tensor:
    """Detect and rescale float32 images from [0,1] to [-1,1], sync-free.

    Intended to replace the inline range-check in _prepare_one() which
    performs two CPU–GPU synchronisations via float(images.min()) and
    float(images.max()). This function computes those reductions on the
    GPU and feeds the result directly to a Triton kernel; the values
    never surface to the CPU.

    Args:
        images: float32 tensor on CUDA, any shape (typically
                (num_cameras, 3, H, W)).

    Returns:
        float32 tensor, same shape and device. Pixels in [0,1] are
        remapped to [-1,1]; pixels already in [-1,1] are unchanged.
    """
    if images.numel() == 0:
        return images

    if not _TRITON_AVAILABLE or not images.is_cuda:
        # CPU path or no Triton: fall back to the original sync-based code.
        img_min = float(images.min())
        img_max = float(images.max())
        if img_min >= -1e-4 and img_max <= 1.0 + 1e-4:
            images = images * 2.0 - 1.0
        return images

    flat = images.contiguous().view(-1)
    img_min = flat.min()   # GPU scalar tensor — no CPU transfer
    img_max = flat.max()   # GPU scalar tensor — no CPU transfer

    out = torch.empty_like(flat)
    n = flat.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _normalize_range_kernel[grid](
        flat, out, img_min, img_max, n, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.view(images.shape)
