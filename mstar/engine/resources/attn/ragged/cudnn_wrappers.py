"""Cacheless self-attention through torch's cuDNN SDPA, for a layout of
equal-length segments.

A DiT's batch packs one span per request, every row (padding rows included) the
bucket's length, so its layout is ``n`` segments of ``S`` tokens: the packed
``[n * S, H, D]`` tensors are a ``[n, S, H, D]`` batch as they stand, which the
dense cuDNN kernel takes without a copy. torch's varlen cuDNN path fails under
stream capture (torch 2.12), so this is the cuDNN kernel a captured graph can
replay.
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


class CudnnRaggedPrefillWrapper(RaggedPrefillWrapper):
    """Self-attention within each of ``n`` equal-length segments, through
    ``aten._scaled_dot_product_cudnn_attention`` over the ``[n, H, S, D]`` view.

    A captured graph bakes ``n`` and ``S`` in, so a graph-mode wrapper replays
    only the layout it was first planned with and refuses any other. An eager
    one takes any layout of equal-length segments; the manager hands a layout
    of unequal ones to another wrapper.
    """

    # the dense kernels take the head count off the tensors
    any_heads = True

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
        q4, k4, v4 = (
            t[:rows].to(self.q_data_type).view(n, seq, t.shape[1], t.shape[2]).transpose(1, 2) for t in (q, k, v)
        )
        out, lse = torch.ops.aten._scaled_dot_product_cudnn_attention(
            q4, k4, v4, None, return_lse, 0.0, causal, False, scale=self.sm_scale,
        )[:2]
        out = out.transpose(1, 2).reshape(rows, q.shape[1], self.head_dim)
        if rows < q.shape[0]:
            # rows past the layout read back zero, as on the other backends
            out = torch.cat([out, out.new_zeros(q.shape[0] - rows, *out.shape[1:])])
        if not return_lse:
            return out
        lse = lse.transpose(1, 2).reshape(rows, q.shape[1]).float()
        if rows < q.shape[0]:
            lse = torch.cat([lse, lse.new_zeros(q.shape[0] - rows, lse.shape[1])])
        return out, lse
