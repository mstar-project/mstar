"""The ``flashinfer.utils`` helpers mstar reads."""

from __future__ import annotations

import torch

_cache_buf: dict = {}


def _ceil_pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def _get_range_buf(seq_len: int, device) -> torch.Tensor:
    """``arange(seq_len)`` int32, cached per (power-of-two size, device)."""
    key = (f"range_{_ceil_pow2(seq_len)}", str(device))
    buf = _cache_buf.get(key)
    if buf is None:
        buf = _cache_buf[key] = torch.arange(_ceil_pow2(seq_len), device=device, dtype=torch.int32)
    return buf[:seq_len]


def get_device_sm_count(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _get_trtllm_gen_multi_ctas_kv_counter_buffer(
    batch_size: int, num_qo_heads: int, sm_count: int, device,
) -> torch.Tensor:
    """A zeroed semaphore buffer; the stand-in decode never reads it."""
    return torch.zeros(max(1, batch_size * num_qo_heads), dtype=torch.int32, device=device)
