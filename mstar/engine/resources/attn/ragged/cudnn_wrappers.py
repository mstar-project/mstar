"""Cacheless self-attention through a dense kernel, torch's cuDNN SDPA or FlashAttention 4,
for a layout of equal-length segments.

A DiT's batch packs one span per request, every row (padding rows included) the
bucket's length, so its layout is ``n`` segments of ``S`` tokens: the packed
``[n * S, H, D]`` tensors are a ``[n, S, H, D]`` batch as they stand, which a
dense kernel takes without a copy. torch's varlen cuDNN path fails under
stream capture (torch 2.12), so this is the cuDNN kernel a captured graph can
replay. FA4's dense call is statically persistent where its varlen one is not.
"""

import torch

from mstar.engine.resources.attn.ragged.wrappers import RaggedPrefillWrapper


def uniform_segments(cu_seqlens: torch.Tensor) -> tuple[int, int] | None:
    """``(segments, length)`` when every non-empty segment of ``cu_seqlens`` has one
    length and the empty ones (a graph wrapper's padding) all come last; else None."""
    lens = torch.diff(cu_seqlens.to(device="cpu", dtype=torch.int64)).tolist()
    n = len(lens)
    while n and lens[n - 1] == 0:
        n -= 1
    if n == 0 or any(length != lens[0] for length in lens[:n]):
        return None
    return n, int(lens[0])


class _UniformSegmentsPrefillWrapper(RaggedPrefillWrapper):
    """Self-attention within each of ``n`` equal-length segments, through a dense
    kernel over the ``[n, S, H, D]`` view (``_attend``).

    A captured graph bakes ``n`` and ``S`` in, so a graph-mode wrapper replays
    only the layout it was first planned with and refuses any other. An eager
    one takes any layout of equal-length segments; the manager hands a layout
    of unequal ones to another wrapper.
    """

    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int, **kwargs):
        if num_qo_heads != num_kv_heads:
            raise ValueError(f"{type(self).__name__} runs MHA only, got {num_qo_heads} q and {num_kv_heads} kv heads")
        super().__init__(num_qo_heads, num_kv_heads, head_dim, **kwargs)
        # (segments, length, causal) this wrapper runs
        self._layout: tuple[int, int, bool] | None = None

    def _plan_kernel(self, cu: torch.Tensor, kv_cu: torch.Tensor, total_tokens: int, causal: bool) -> None:
        segments = uniform_segments(cu)
        if self.accelerator_graph:
            layout = None if segments is None else (*segments, causal)
            if self._layout is not None and layout != self._layout:
                raise ValueError(
                    f"{type(self).__name__}: a captured graph replays the {self._layout[0]} segments of "
                    f"{self._layout[1]} tokens it was planned with, not {self._describe(cu)}"
                )
            if layout is None:
                raise ValueError(f"{type(self).__name__} needs equal-length segments, got {self._describe(cu)}")
            self._layout = layout
            return
        if segments is None:
            raise ValueError(f"{type(self).__name__} needs equal-length segments, got {self._describe(cu)}")
        self._layout = (*segments, causal)

    @staticmethod
    def _describe(cu: torch.Tensor) -> str:
        return f"segments {torch.diff(cu.cpu()).tolist()}"

    @torch.compiler.disable
    def run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self._layout is not None, f"{type(self).__name__}: run before plan"
        n, seq, causal = self._layout
        rows = n * seq
        if self.accelerator_graph:
            self._check_rows(q.shape[0])
        q4, k4, v4 = (t[:rows].to(self.q_data_type).view(n, seq, t.shape[1], t.shape[2]) for t in (q, k, v))
        out, lse = self._attend(q4, k4, v4, causal, return_lse)
        out = out.reshape(rows, self.num_qo_heads, self.head_dim)
        if rows < q.shape[0]:
            # rows past the layout read back zero, as on the other backends
            out = torch.cat([out, out.new_zeros(q.shape[0] - rows, *out.shape[1:])])
        if not return_lse:
            return out
        lse = lse.transpose(1, 2).reshape(rows, self.num_qo_heads).float()
        if rows < q.shape[0]:
            lse = torch.cat([lse, lse.new_zeros(q.shape[0] - rows, lse.shape[1])])
        return out, lse

    def _attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, return_lse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(out [n, S, H, D], lse [n, H, S] or None)`` of ``[n, S, H, D]`` q / k / v."""
        raise NotImplementedError


class CudnnRaggedPrefillWrapper(_UniformSegmentsPrefillWrapper):
    """Equal-length segments through ``aten._scaled_dot_product_cudnn_attention``
    over the ``[n, H, S, D]`` view."""

    def _attend(self, q, k, v, causal, return_lse):
        out, lse = torch.ops.aten._scaled_dot_product_cudnn_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), None, return_lse, 0.0, causal, False,
            scale=self.sm_scale,
        )[:2]
        return out.transpose(1, 2), lse


class Fa4UniformPrefillWrapper(_UniformSegmentsPrefillWrapper):
    """Equal-length segments through FlashAttention 4's dense forward (``flash_attn.cute``),
    one call over the ``[n, S, H, D]`` batch, which it takes as laid out.

    ``precision="fp8"`` is lossy and opt-in (``RaggedAttnBackend.FA4_FP8``): q, k and v go
    to e4m3 with one scale per (segment, head) and FA4's FP8 kernel runs them, the output
    staying in the activation dtype.
    """

    def __init__(self, *args, precision: str = "bf16", **kwargs):
        if precision not in ("bf16", "fp8"):
            raise ValueError(f"{type(self).__name__}: precision is 'bf16' or 'fp8', not {precision!r}")
        super().__init__(*args, **kwargs)
        self.precision = precision

    def _attend(self, q, k, v, causal, return_lse):
        from flash_attn.cute import flash_attn_func

        kwargs = {}
        if self.precision == "fp8":
            from flash_attn.cute.interface import TensorwiseScales
            from mstar.engine.resources.attn.ragged.fp8_quant import quantize_per_segment_head

            n, seq, heads, dim = q.shape
            quantized = [quantize_per_segment_head(t.reshape(n * seq, heads, dim), n) for t in (q, k, v)]
            q, k, v = (y.view(n, seq, heads, dim) for y, _ in quantized)
            kwargs["scales"] = TensorwiseScales(*(scale for _, scale in quantized))
        result = flash_attn_func(q, k, v, softmax_scale=self.sm_scale, causal=causal, return_lse=return_lse, **kwargs)
        if not return_lse:
            return (result[0] if isinstance(result, tuple) else result), None
        return result[0], result[1]
