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


def determine_attention_backend(device, pos_encoding_mode: int, use_fp16_qk_reductions: bool, use_custom_mask: bool,
                                dtype_q: torch.dtype, dtype_kv: torch.dtype, *, head_dim_qk: int | None = None,
                                head_dim_vo: int | None = None) -> str:
    """FlashInfer's answer to ``backend="auto"``: fa3 on Hopper where its kernels serve the
    call, else fa2."""
    fp8 = {torch.float8_e4m3fn, torch.float8_e5m2}
    if (torch.cuda.get_device_capability(device)[0] != 9 or use_custom_mask or pos_encoding_mode != 0
            or use_fp16_qk_reductions or (dtype_kv in fp8 and dtype_q not in fp8) or dtype_kv == torch.uint8):
        return "fa2"
    if head_dim_qk is None or head_dim_vo is None:
        return "fa3"
    if head_dim_qk == head_dim_vo:
        return "fa3" if head_dim_qk in (64, 128, 256) else "fa2"
    return "fa3" if (head_dim_qk, head_dim_vo) == (192, 128) else "fa2"


def get_device_sm_count(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _get_trtllm_gen_multi_ctas_kv_counter_buffer(
    batch_size: int, num_qo_heads: int, sm_count: int, device,
) -> torch.Tensor:
    """A zeroed semaphore buffer; the stand-in decode never reads it."""
    return torch.zeros(max(1, batch_size * num_qo_heads), dtype=torch.int32, device=device)
