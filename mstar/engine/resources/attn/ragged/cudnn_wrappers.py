"""Cacheless self-attention through torch's cuDNN SDPA, for a layout of
equal-length segments (and, eagerly where a row asks, of a few runs of them).

A DiT's batch packs one span per request, every row (padding rows included) the
bucket's length, so its layout is ``n`` segments of ``S`` tokens: the packed
``[n * S, H, D]`` tensors are a ``[n, S, H, D]`` batch as they stand, which the
dense cuDNN kernel takes without a copy. torch's varlen cuDNN path fails under
stream capture (torch 2.12), so this is the cuDNN kernel a captured graph can
replay.

A row with ``{"segments": "runs"}`` also takes, eagerly, a layout of unequal
segments: one call per run of consecutive equal-length segments (MiniMax-H3's
request rows, then its padding rows to the compile bucket).
"""

import torch

from mstar.engine.resources.attn.ragged.wrappers import RaggedPrefillWrapper


def segment_runs(cu_seqlens: torch.Tensor) -> list[tuple[int, int]]:
    """``[(segments, length), ...]``: the non-empty segments of ``cu_seqlens`` as runs of
    consecutive equal lengths, in order; the trailing empty ones (a graph wrapper's padding) dropped."""
    lens = torch.diff(cu_seqlens.to(device="cpu", dtype=torch.int64)).tolist()
    n = len(lens)
    while n and lens[n - 1] == 0:
        n -= 1
    runs: list[tuple[int, int]] = []
    for length in lens[:n]:
        if runs and runs[-1][1] == length:
            runs[-1] = (runs[-1][0] + 1, length)
        else:
            runs.append((1, int(length)))
    return runs


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
    one takes any layout of equal-length segments, and with ``runs`` any layout,
    one call per run of equal lengths; without it the manager hands a layout of
    unequal ones to another wrapper.
    """

    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int, runs: bool = False, **kwargs):
        if num_qo_heads != num_kv_heads:
            raise ValueError(f"{type(self).__name__} runs MHA only, got {num_qo_heads} q and {num_kv_heads} kv heads")
        super().__init__(num_qo_heads, num_kv_heads, head_dim, **kwargs)
        if runs and self.accelerator_graph:
            raise ValueError(f"{type(self).__name__}: a captured graph runs one layout of equal-length segments")
        self._runs_ok = runs
        # (segments, length, causal): the first run's, for a captured graph the only one
        self._layout: tuple[int, int, bool] | None = None
        # [(segments, length), ...] of the planned layout, in order
        self._runs: list[tuple[int, int]] = []

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
            self._layout, self._runs = layout, [segments]
            return
        runs = [segments] if segments is not None else segment_runs(cu) if self._runs_ok else []
        if not runs:
            raise ValueError(f"{type(self).__name__} needs equal-length segments, got {self._describe(cu)}")
        self._runs, self._layout = runs, (*runs[0], causal)

    @staticmethod
    def _describe(cu: torch.Tensor) -> str:
        return f"segments {torch.diff(cu.cpu()).tolist()}"

    @torch.compiler.disable
    def run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self._layout is not None, f"{type(self).__name__}: run before plan"
        causal = self._layout[2]
        if self.accelerator_graph:
            self._check_rows(q.shape[0])
        outs, lses, rows = [], [], 0
        for n, seq in self._runs:
            out, lse = self._run_segments(q, k, v, rows, n, seq, causal, return_lse)
            outs.append(out)
            lses.append(lse)
            rows += n * seq
        if rows < q.shape[0]:
            # rows past the layout read back zero, as on the other backends
            outs.append(outs[0].new_zeros(q.shape[0] - rows, *outs[0].shape[1:]))
            lses.append(None if not return_lse else lses[0].new_zeros(q.shape[0] - rows, lses[0].shape[1]))
        out = outs[0] if len(outs) == 1 else torch.cat(outs)
        if not return_lse:
            return out
        return out, (lses[0] if len(lses) == 1 else torch.cat(lses))

    def _run_segments(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int, n: int, seq: int, causal: bool,
        return_lse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``n`` segments of ``seq`` rows from row ``start`` as one ``[n, H, seq, D]`` batch: the output
        ``[n * seq, H, D]``, and with ``return_lse`` the log-sum-exp ``[n * seq, H]`` (fp32)."""
        rows = n * seq
        q4, k4, v4 = (
            t[start:start + rows].to(self.q_data_type).view(n, seq, t.shape[1], t.shape[2]).transpose(1, 2)
            for t in (q, k, v)
        )
        out, lse = torch.ops.aten._scaled_dot_product_cudnn_attention(
            q4, k4, v4, None, return_lse, 0.0, causal, False, scale=self.sm_scale,
        )[:2]
        out = out.transpose(1, 2).reshape(rows, self.num_qo_heads, self.head_dim)
        if not return_lse:
            return out, None
        return out, lse.transpose(1, 2).reshape(rows, self.num_qo_heads).float()
