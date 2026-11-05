"""``ragged_attention``: cacheless self-attention over equal-length segments, swept per segment length.

Each bucket times every implementation through the resource's own captured wrapper
(``SelectedRaggedPrefillWrapper`` with a one-row table naming it), as a captured denoise step runs
it: at one segment and at a batch of them (``SEGMENT_TOKENS`` tokens, at most ``MAX_SEGMENTS``
segments), and a row's time is the sum of the two. Inputs are unit-variance q / k / v, as a q/k
RMSNorm hands them over. A candidate competes only when segment 0's output is within the
implementation's tolerance of fp32 attention (TF32 off).
"""
from __future__ import annotations

from collections.abc import Mapping

import torch

from benchmark.kernels.sweep import BucketResult
from benchmark.kernels.timing import time_graph
from mstar.engine.resources.attn.ragged.select import RAGGED_ATTENTION, SelectedRaggedPrefillWrapper
from mstar.engine.resources.attn.workspace import Workspace
from mstar.kernels import Choice, TableEntry

# Image DiT segments: Z-Image's caption buckets, then 256^2..2048^2 images with their text
DEFAULT_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 4224, 4608, 8704, 16384)
SEGMENT_TOKENS = 36864
MAX_SEGMENTS = 8
CANDIDATES = (Choice("flashinfer"), Choice("cudnn"))


def _reference(q, k, v, seq: int) -> torch.Tensor:
    """fp32 attention of the first segment, [seq, H, D]."""
    qf, kf, vf = (t[:seq].float() for t in (q, k, v))
    out = torch.empty_like(qf)
    for h in range(qf.shape[1]):
        out[:, h] = torch.softmax((qf[:, h] @ kf[:, h].T) * qf.shape[-1] ** -0.5, -1) @ vf[:, h]
    return out


class RaggedAttentionSweep:
    op = "ragged_attention"
    default_buckets = DEFAULT_BUCKETS

    def __init__(self, shape: Mapping[str, int], dtype: torch.dtype, device: torch.device, quick: bool = False):
        self.H, self.H_kv, self.D = (shape[k] for k in ("H", "H_kv", "D"))
        self.dtype, self.device, self.quick = dtype, device, quick
        self.workspace = Workspace(torch.empty(256 << 20, dtype=torch.uint8, device=device), "sweep")
        torch.backends.cuda.matmul.allow_tf32 = False

    def check(self, entry: TableEntry) -> None:
        """Refuse an entry the resource would refuse: an implementation it can't run here."""
        target = RAGGED_ATTENTION.target(self.dtype, dict(entry.shape), self.device)
        for _, choice in entry.rows:
            reason = RAGGED_ATTENTION.impl(choice.impl).unavailable(target)
            if reason is not None:
                raise ValueError(f"row {choice.describe()} can't run here: {reason}")

    def _time(self, choice: Choice, seq: int, segments: int) -> tuple[float, float]:
        """(error against fp32 as a fraction of the tolerance, us per call) at ``segments`` x ``seq``."""
        g = torch.Generator(device=self.device).manual_seed(seq)
        q = torch.randn(segments * seq, self.H, self.D, device=self.device, dtype=self.dtype, generator=g)
        k, v = (torch.randn(segments * seq, self.H_kv, self.D, device=self.device, dtype=self.dtype, generator=g)
                for _ in range(2))
        wrapper = SelectedRaggedPrefillWrapper(
            lambda: self.workspace, self.H, self.H_kv, self.D, rows=((None, choice),), device=self.device,
            accelerator_graph=True, max_num_segments=segments, max_total_tokens=segments * seq,
            q_data_type=self.dtype,
        )
        wrapper.plan(torch.arange(0, (segments + 1) * seq, seq, dtype=torch.int32))
        out = wrapper.run(q, k, v)
        error = RAGGED_ATTENTION.impl(choice.impl).tolerance.error(out[:seq], _reference(q, k, v, seq))
        us = time_graph(lambda: wrapper.run(q, k, v), replays=5 if self.quick else 20)
        return error, us

    def run_bucket(self, seq: int, current: Choice | None) -> BucketResult:
        """``current`` is the table's row for ``seq``; None, FlashInfer (what runs without a table)."""
        default = CANDIDATES[0]
        current = default if current is None else current
        batch = max(1, min(MAX_SEGMENTS, SEGMENT_TOKENS // seq))
        times: dict[Choice, float] = {}
        rejected = 0
        for choice in dict.fromkeys((*CANDIDATES, current)):
            try:
                (e1, t1), (e8, t8) = self._time(choice, seq, 1), self._time(choice, seq, batch)
            except Exception:  # noqa: BLE001 - a kernel that can't run this size
                rejected += 1
                continue
            if max(e1, e8) > 1.0:
                rejected += 1
                continue
            times[choice] = t1 + t8
        nan = float("nan")
        best = min(times, key=times.get) if times else None
        result = BucketResult(bucket=seq, current=current, current_us=times.get(current, nan),
                              default_us=times.get(default, nan), best=best,
                              best_us=times[best] if best is not None else nan,
                              timed=len(times), rejected=rejected)
        for choice, us in times.items():
            name = choice.impl if choice.config is None else choice.describe().replace(" ", "")
            result.impl_us[name] = min(us, result.impl_us.get(name, us))
        return result
