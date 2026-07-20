"""Rolling metrics for the soak driver.

Everything is time-windowed: a multi-hour soak is about *drift*, so the headline
numbers are moving averages over the last ``window_s`` seconds rather than
run-cumulative means. Per output modality:

  - text            → tok/s (windowed system throughput) + per-request tok/s
  - audio           → audio-seconds generated per wall-second (a realtime factor
                      >1 means the server keeps up) + per-request RTF (e2e/dur)
  - image / video   → e2e latency percentiles (throughput is naturally low; the
                      SLO is per-request latency)

Plus run-wide counters (ok / failed / timed-out / in-flight / backlog) and the
client-side **admission delay** — how long an arrival waited for an in-flight
slot. Admission delay climbing while the server's own queue stays flat is the
client-visible signature of backpressure, the exact series the arena reviewer
asked to watch.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from benchmark.base import Status
from benchmark.request import RequestMetrics, _audio_duration_seconds


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class _Sample:
    t: float  # completion time (monotonic)
    e2e: float
    ttft: float
    tokens: int = 0
    audio_s: float = 0.0
    nbytes: int = 0


class _TimeWindow:
    """Fixed-duration sliding window of completion samples."""

    def __init__(self, window_s: float):
        self.window_s = window_s
        self.samples: deque[_Sample] = deque()

    def add(self, s: _Sample) -> None:
        self.samples.append(s)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self.samples and self.samples[0].t < cutoff:
            self.samples.popleft()

    def snapshot(self, now: float) -> list[_Sample]:
        self._trim(now)
        return list(self.samples)


class SoakMetrics:
    """Thread/async-safe rolling metrics collector.

    All mutation goes through a lock so a same-thread ``stats()`` poll (or a
    separate reporter task) sees a consistent snapshot.
    """

    def __init__(self, window_s: float, modalities: list[str]):
        self._lock = threading.Lock()
        self.window_s = window_s
        self.start_t = time.monotonic()

        # Run-wide counters.
        self.launched = 0  # arrivals generated
        self.admitted = 0  # passed the in-flight cap
        self.completed_ok = 0
        self.failed = 0
        self.timed_out = 0
        self.in_flight = 0  # admitted, not yet done
        self.backlog = 0  # arrived, awaiting an in-flight slot

        # Per-output-modality windows + cumulative totals.
        self._win: dict[str, _TimeWindow] = {
            m: _TimeWindow(window_s) for m in modalities
        }
        self.cum_tokens: dict[str, int] = {m: 0 for m in modalities}
        self.cum_audio_s: dict[str, float] = {m: 0.0 for m in modalities}
        # Per-mixture-entry completion counts (labelled) for a load-mix sanity check.
        self.by_label: dict[str, int] = {}
        self.errors_by_label: dict[str, int] = {}

        # Admission-delay window (client-side backpressure).
        self._adm: deque[tuple[float, float]] = deque()  # (t_admit, delay)

    # -- lifecycle hooks (called by the driver) --------------------------

    def on_arrival(self) -> None:
        with self._lock:
            self.launched += 1
            self.backlog += 1

    def on_admit(self, delay: float) -> None:
        now = time.monotonic()
        with self._lock:
            self.backlog -= 1
            self.admitted += 1
            self.in_flight += 1
            self._adm.append((now, delay))
            cutoff = now - self.window_s
            while self._adm and self._adm[0][0] < cutoff:
                self._adm.popleft()

    def on_complete(
        self, label: str, out_mod: str, metrics: RequestMetrics,
        timed_out: bool,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            self.in_flight -= 1
            self.by_label[label] = self.by_label.get(label, 0) + 1
            if timed_out:
                self.timed_out += 1
                self.errors_by_label[label] = self.errors_by_label.get(label, 0) + 1
                return
            if metrics.error is not None or metrics.status != Status.SUCCESS:
                self.failed += 1
                self.errors_by_label[label] = self.errors_by_label.get(label, 0) + 1
                return

            # Known limitation: this only distinguishes *empty* streams from
            # healthy ones. record_completion flags a request FAILED when an
            # expected output modality is entirely missing (received []), but a
            # stream that returns *some* output and is then truncated under
            # overload still passes that check and lands in completed_ok. So
            # ok/failed detect total drops, not partial/short completions —
            # watch the throughput moving averages for the latter.
            self.completed_ok += 1
            e2e = metrics.e2e_latency or 0.0
            ttft = metrics.ttft.get(out_mod, e2e)
            tokens = int(metrics.output_text_tokens or 0)
            audio_bytes = metrics.output_bytes.get("audio", 0)
            audio_s = _audio_duration_seconds(audio_bytes) if audio_bytes else 0.0
            nbytes = sum(metrics.output_bytes.values())

            win = self._win.get(out_mod)
            if win is not None:
                win.add(_Sample(
                    t=now, e2e=e2e, ttft=ttft, tokens=tokens,
                    audio_s=audio_s, nbytes=nbytes,
                ))
            self.cum_tokens[out_mod] = self.cum_tokens.get(out_mod, 0) + tokens
            self.cum_audio_s[out_mod] = self.cum_audio_s.get(out_mod, 0.0) + audio_s

    # -- reporting -------------------------------------------------------

    def snapshot(self) -> dict:
        """A JSON-serialisable moving-average snapshot for the report line and
        the JSONL time series."""
        now = time.monotonic()
        with self._lock:
            elapsed = now - self.start_t
            adm_delays = [d for _, d in self._adm]
            out: dict = {
                "t_wall": time.time(),
                "elapsed_s": round(elapsed, 2),
                "launched": self.launched,
                "admitted": self.admitted,
                "in_flight": self.in_flight,
                "backlog": self.backlog,
                "ok": self.completed_ok,
                "failed": self.failed,
                "timed_out": self.timed_out,
                "admission_delay_p50_s": round(_pct(adm_delays, 50), 4),
                "admission_delay_p95_s": round(_pct(adm_delays, 95), 4),
                "modalities": {},
            }
            w = self.window_s
            for mod, win in self._win.items():
                s = win.snapshot(now)
                if not s:
                    continue
                e2es = [x.e2e for x in s]
                ttfts = [x.ttft for x in s]
                m: dict = {
                    "n_window": len(s),
                    "e2e_p50_s": round(_pct(e2es, 50), 3),
                    "e2e_p95_s": round(_pct(e2es, 95), 3),
                    "ttft_p50_s": round(_pct(ttfts, 50), 3),
                }
                if mod == "text":
                    tok_win = sum(x.tokens for x in s)
                    m["tok_per_s_window"] = round(tok_win / w, 2)
                    per_req = [
                        x.tokens / max(x.e2e - x.ttft, 1e-6)
                        for x in s if x.tokens > 1
                    ]
                    m["tok_per_s_per_req_p50"] = round(_pct(per_req, 50), 2)
                elif mod == "audio":
                    audio_win = sum(x.audio_s for x in s)
                    m["audio_s_per_s_window"] = round(audio_win / w, 3)
                    rtf = [
                        x.e2e / x.audio_s for x in s if x.audio_s > 0
                    ]
                    m["rtf_e2e_over_dur_p50"] = round(_pct(rtf, 3), 3)
                out["modalities"][mod] = m
            return out

    def format_line(self, snap: dict) -> str:
        head = (
            f"[t=+{snap['elapsed_s']:>7.1f}s] "
            f"launched={snap['launched']} admitted={snap['admitted']} "
            f"inflight={snap['in_flight']} backlog={snap['backlog']} "
            f"ok={snap['ok']} fail={snap['failed']} timeout={snap['timed_out']} "
            f"| adm-delay p50={snap['admission_delay_p50_s']:.3f}s "
            f"p95={snap['admission_delay_p95_s']:.3f}s"
        )
        lines = [head]
        for mod, m in snap["modalities"].items():
            if mod == "text":
                lines.append(
                    f"    text : {m['tok_per_s_window']:8.1f} tok/s(win)  "
                    f"per-req p50 {m['tok_per_s_per_req_p50']:.1f} tok/s  "
                    f"ttft p50 {m['ttft_p50_s']:.2f}s  n={m['n_window']}"
                )
            elif mod == "audio":
                lines.append(
                    f"    audio: {m['audio_s_per_s_window']:8.2f} audio-s/s(win)  "
                    f"rtf p50 {m['rtf_e2e_over_dur_p50']:.2f}  "
                    f"ttft p50 {m['ttft_p50_s']:.2f}s  n={m['n_window']}"
                )
            else:
                lines.append(
                    f"    {mod:5s}: e2e p50 {m['e2e_p50_s']:7.2f}s "
                    f"p95 {m['e2e_p95_s']:7.2f}s  n={m['n_window']}"
                )
        return "\n".join(lines)

    def final_summary(self) -> dict:
        with self._lock:
            elapsed = time.monotonic() - self.start_t
            total = self.completed_ok + self.failed + self.timed_out
            return {
                "elapsed_s": round(elapsed, 2),
                "launched": self.launched,
                "completed": total,
                "ok": self.completed_ok,
                "failed": self.failed,
                "timed_out": self.timed_out,
                "failure_rate": round((self.failed + self.timed_out) / total, 4)
                if total else 0.0,
                "cum_tokens": dict(self.cum_tokens),
                "cum_audio_s": {k: round(v, 1) for k, v in self.cum_audio_s.items()},
                "by_label": dict(self.by_label),
                "errors_by_label": dict(self.errors_by_label),
            }
