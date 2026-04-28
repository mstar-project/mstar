import base64
import io
import json
import mimetypes
import os
import statistics
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp

from benchmark.base import Model, RequestType, Status
from benchmark.utils import _write_wav


@dataclass
class LatencyStats:
    mean: Optional[float]
    p50: Optional[float]
    higher_is_better: bool = False
    # Populated when higher_is_better=False
    p95: Optional[float] = None
    p99: Optional[float] = None
    # Populated when higher_is_better=True
    p05: Optional[float] = None
    p10: Optional[float] = None

    def __str__(self) -> str:
        def fmt(v: Optional[float]) -> str:
            return f"{v:.3f}s" if v is not None else "n/a"

        if self.higher_is_better:
            return (
                f"mean={fmt(self.mean)}  p50={fmt(self.p50)}"
                f"  p05={fmt(self.p05)}  p10={fmt(self.p10)}"
                f"  (higher is better)"
            )
        return f"mean={fmt(self.mean)}  p50={fmt(self.p50)}  p95={fmt(self.p95)}  p99={fmt(self.p99)}"


def _latency_stats(values: list[float], higher_is_better: bool = False) -> LatencyStats:
    if not values:
        return LatencyStats(mean=None, p50=None, higher_is_better=higher_is_better)

    sorted_vals = sorted(values)

    def percentile(p: float) -> float:
        idx = (p / 100) * (len(sorted_vals) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)

    if higher_is_better:
        return LatencyStats(
            mean=statistics.mean(values),
            p50=percentile(50),
            p05=percentile(5),
            p10=percentile(10),
            higher_is_better=True,
        )
    return LatencyStats(
        mean=statistics.mean(values),
        p50=percentile(50),
        p95=percentile(95),
        p99=percentile(99),
    )


@dataclass
class RequestMetrics:
    request_id: str
    type: RequestType
    expected_output_modalities: Optional[list[str]] = None
    # Anchored at HTTP send by the adapter (see Fix 5). None until the adapter
    # calls `metrics.start_time = time.monotonic()` immediately before
    # `session.post(...)`.
    start_time: Optional[float] = None
    status: Status = Status.PROGRESS

    # Per-modality TTFT: e.g. {"text": 0.12, "audio": 0.34}
    ttft: dict[str, float] = field(default_factory=dict)

    # Per-modality monotonic timestamps for every received chunk.
    # Used to derive inter-chunk gaps and (for text) per-token-normalised ITL.
    chunk_arrivals: dict[str, list[float]] = field(default_factory=dict)

    e2e_latency: Optional[float] = None
    error: Optional[str] = None

    # Per-modality count of server-emitted framing units (one per record_token
    # call). For text these are SSE / NDJSON messages, NOT tokens —
    # vllm-omni and OurSystem emit one token per chunk while sglang-omni
    # chunk-bursts ~5 tokens per message. For the actual text-token count use
    # `output_text_tokens` (set from `usage.completion_tokens`).
    response_chunks: dict[str, int] = field(default_factory=dict)
    # Per-modality byte counts (raw bytes received)
    output_bytes: dict[str, int] = field(default_factory=dict)
    # Text token count. Per-chunk increments accumulate while streaming and are
    # overwritten by `usage.completion_tokens` from the final SSE chunk for
    # vllm-omni / sglang-omni (see Fix 3).
    output_text_tokens: int = 0
    # Prompt-token count from `usage.prompt_tokens` when emitted by the server.
    input_tokens: int = 0

    _output_modalities_recvd: list[str] = field(default_factory=list, repr=False)

    # Streaming viability tracking (audio only).
    # _audio_chunk_log tracks (inter_chunk_latency, prev_chunk_duration), from
    # which streaming viability is computed.
    _audio_chunk_log: list[tuple[float, float]] = field(default_factory=list, repr=False)
    _last_audio_chunk_time: Optional[float] = field(default=None, repr=False)
    _last_audio_chunk_duration: Optional[float] = field(default=None, repr=False)

    # For storing outputs of different modalities
    _text_chunks: list[str] = field(default_factory=list, repr=False)
    _image_chunks: list[bytes] = field(default_factory=list, repr=False)
    _audio_pcm: io.BytesIO = field(default_factory=io.BytesIO, repr=False)

    def __post_init__(self):
        # `start_time` is intentionally NOT set here. The adapter must set it
        # immediately before `session.post(...)` so TTFT/E2E exclude pre-send
        # work (form/payload building, file I/O, base64 encoding). See Fix 5.
        pass

    def record_output_chunk(
        self,
        modality: str,
        data_b64: str,
        n_tokens: int = 1,
        arrival_time: Optional[float] = None,
    ):
        """
        Decode a base64 chunk, record timing/byte metrics, and buffer content
        for final output assembly. Call this for every streamed chunk.
        """
        data = base64.b64decode(data_b64)
        if not data:
            return

        self._output_modalities_recvd.append(modality)

        self.record_token(
            modality=modality,
            nbytes=len(data),
            arrival_time=arrival_time,
            n_tokens=n_tokens,
        )

        if modality == "text":
            self._text_chunks.append(data.decode("utf-8", errors="replace"))
        elif modality == "image":
            self._image_chunks.append(data)
        elif modality == "audio":
            self._audio_pcm.write(data)

    def record_token(self, modality: str, nbytes: int, arrival_time: float | None = None, n_tokens: int = 1):
        """
        Record a received chunk for the given modality.
        Possible modalities: "text", "image", "audio".
        nbytes is the raw byte size of the chunk.
        """
        if self.start_time is None:
            raise RuntimeError(
                f"record_token called before start_time was set on request "
                f"{self.request_id}. The adapter must set "
                f"`metrics.start_time = time.monotonic()` immediately before "
                f"`session.post(...)`."
            )
        now = time.monotonic() if arrival_time is None else arrival_time

        if modality not in self.ttft:
            self.ttft[modality] = now - self.start_time
        self.chunk_arrivals.setdefault(modality, []).append(now)

        self.response_chunks[modality] = self.response_chunks.get(modality, 0) + 1
        self.output_bytes[modality] = self.output_bytes.get(modality, 0) + nbytes

        if modality == "text":
            self._record_text_tokens(n_tokens)
        elif modality == "audio":
            chunk_duration = self._pcm_duration_bytes(nbytes)
            if self._last_audio_chunk_time is not None and self._last_audio_chunk_duration is not None:
                inter_chunk_latency = now - self._last_audio_chunk_time
                self._audio_chunk_log.append((inter_chunk_latency, self._last_audio_chunk_duration))
            self._last_audio_chunk_time = now
            self._last_audio_chunk_duration = chunk_duration

    def _record_text_tokens(self, n_tokens: int):
        """Accumulate decoded token count for the text modality."""
        self.output_text_tokens += n_tokens

    def record_completion(self):
        """
        Finalise timing and assemble output_content from buffered chunks.
        Optionally write audio to a WAV file at output_path.
        """
        if self.start_time is None:
            # Adapter exited before reaching `session.post(...)` cleanly.
            # Mark as failed so the modality-completeness check below doesn't
            # silently pass on a request that never actually ran.
            self.e2e_latency = 0.0
            self.status = Status.FAILED
            self.error = self.error or "start_time was never set (no HTTP send)"
            return
        self.e2e_latency = time.monotonic() - self.start_time

        if self.expected_output_modalities and any(
            mod not in self._output_modalities_recvd for mod in self.expected_output_modalities
        ):
            print(
                f"ERROR [req {self.request_id}]: expected modalities "
                f"{self.expected_output_modalities}, received {self._output_modalities_recvd}",
                file=sys.stderr,
            )
            self.status = Status.FAILED
            return
        self.status = Status.SUCCESS

    def write_files(self, output_dir: str):
        n_outputs = 0
        if self._text_chunks:
            path = os.path.join(output_dir, f"req_{self.request_id}.txt")
            output_content = "".join(self._text_chunks)
            with open(path, "w") as f:
                f.write(output_content)
            n_outputs += 1
        if self._image_chunks:
            output_content = b"".join(self._image_chunks)
            path = os.path.join(output_dir, f"req_{self.request_id}.png")
            with open(path, "wb") as f:
                f.write(output_content)
            n_outputs += 1
        if self._audio_pcm.tell() > 0:
            output_path = os.path.join(output_dir, f"req_{self.request_id}.wav")
            pcm_bytes = self._audio_pcm.getvalue()
            if output_path is not None:
                _write_wav(pcm_bytes, output_path)
            n_outputs += 1
        return n_outputs

    def record_error(self, msg: str):
        if self.start_time is None:
            self.e2e_latency = 0.0
        else:
            self.e2e_latency = time.monotonic() - self.start_time
        self.status = Status.FAILED
        self.error = msg

    @property
    def chunk_gaps(self) -> dict[str, list[float]]:
        """Per-modality inter-chunk arrival gaps in seconds."""
        out: dict[str, list[float]] = {}
        for modality, arrivals in self.chunk_arrivals.items():
            if len(arrivals) < 2:
                continue
            out[modality] = [arrivals[i + 1] - arrivals[i] for i in range(len(arrivals) - 1)]
        return out

    def itl_per_token_text(self, tokenizer=None) -> Optional[list[float]]:
        """Per-token-normalised inter-token latency for the text modality.

        Mirrors sglang.bench_serving's `--accept-length` path: re-tokenize each
        chunk individually, divide that chunk's gap by its actual token count,
        then replicate the result by the token count so percentiles are
        weighted by tokens (not by chunks). This makes sglang-omni's
        chunk-bursting comparable to systems that stream one token per chunk
        (OurSystem, vllm-omni) — both collapse to "per-token gap".

        If `tokenizer` is None, falls back to raw per-chunk gaps (matches
        sglang.bench_serving's default ITL path; not cross-system comparable
        for sglang-omni vs others).
        """
        gaps = self.chunk_gaps.get("text")
        if not gaps:
            return None
        # gap[i] is the time from chunk i's arrival to chunk i+1's arrival; we
        # attribute it to chunk i+1 (the one that "took" that long to arrive).
        if tokenizer is None or len(self._text_chunks) < 2:
            return list(gaps)
        out: list[float] = []
        for i, gap in enumerate(gaps):
            chunk_idx = i + 1
            if chunk_idx >= len(self._text_chunks):
                continue
            chunk_text = self._text_chunks[chunk_idx]
            try:
                n_tokens = len(tokenizer.encode(chunk_text, add_special_tokens=False))
            except Exception:
                n_tokens = 1
            if n_tokens <= 0:
                continue
            per_token = gap / n_tokens
            out.extend([per_token] * n_tokens)
        return out if out else None

    @property
    def streaming_viability(self) -> Optional[float]:
        """
        Fraction of audio chunks where inter-chunk latency < duration of the
        previous chunk. Only defined when at least one inter-chunk gap exists.
        A value of 1.0 means perfectly continuous audio; lower means dropouts.
        """
        if not self._audio_chunk_log:
            return None
        viable = sum(1 for gap, prev_dur in self._audio_chunk_log if gap < prev_dur)
        return viable / len(self._audio_chunk_log)

    def _pcm_duration_bytes(self, nbytes: int) -> float:
        """Returns duration of a PCM audio chunk in seconds (24kHz, 16-bit, mono)."""
        sample_rate = 24000
        bytes_per_sample = 2
        channels = 1
        return nbytes / (sample_rate * bytes_per_sample * channels)


@dataclass
class AggregateMetrics:
    n_requests: int
    n_success: int
    wall_time: float
    ttft: dict[str, LatencyStats]
    e2e_latency: LatencyStats
    itl: dict[str, LatencyStats]
    streaming_viability: Optional[LatencyStats]
    type_counts: dict[str, int]
    total_response_chunks: dict[str, int] = field(default_factory=dict)
    total_output_bytes: dict[str, int] = field(default_factory=dict)
    total_text_tokens: int = 0
    mean_text_tokens: Optional[float] = None
    online: bool = False
    batch_size: int = 1
    rate: Optional[float] = None
    # Firing-mode metadata (Fix 9): set by aggregate_metrics() so the header
    # reflects the actual mode (offline / closed_loop / online).
    profiling_type: Optional[str] = None
    max_concurrency: Optional[int] = None

    # Fix 8 — per-modality throughput, audio RTF, and prompt-token totals.
    total_input_tokens: int = 0
    request_throughput: Optional[float] = None  # successful req / wall_time
    text_token_throughput: Optional[float] = None  # text tokens / wall_time
    audio_seconds_throughput: Optional[float] = None  # synthesized audio sec / wall_time
    rtf: Optional[LatencyStats] = None  # per-request E2E / audio_duration
    audio_duration_mean_s: Optional[float] = None  # mean synthesized audio duration

    def __str__(self) -> str:
        if not self.ttft:
            return "ERROR: benchmark produced no results."
        if self.profiling_type == "closed_loop":
            header = (
                f"Closed-Loop Benchmark Results ({self.n_requests} requests, max_concurrency={self.max_concurrency})"
            )
        elif self.online:
            header = f"Online Benchmark Results ({self.n_requests} requests, rate={self.rate} req/s)"
        else:
            header = f"Offline Benchmark Results ({self.n_requests} requests, batch={self.batch_size})"
        header += "\n" + ("─" * 50)

        max_len = max([len(m) for m in self.ttft])
        ttft_lines = "\n".join(
            f"TTFT ({m})" + (" " * (max_len - len(m))) + f"  : {s}" for m, s in sorted(self.ttft.items())
        )
        itl_lines = "\n".join(
            f"ITL  ({m})" + (" " * (max_len - len(m))) + f"  : {s}" for m, s in sorted(self.itl.items())
        )
        sv_line = (
            f"Audio SV {' ' * max_len}: {self.streaming_viability}\n" if self.streaming_viability is not None else ""
        )

        # Throughput block: req/s always; per-modality rates depending on outputs.
        tpt_lines = ""
        if self.request_throughput is not None:
            tpt_lines += f"Throughput: {self.request_throughput:.2f} req/s (successful only)\n"
        if self.text_token_throughput is not None and self.total_text_tokens > 0:
            tpt_lines += f"Throughput: {self.text_token_throughput:.2f} text tok/s\n"
        if self.audio_seconds_throughput is not None and self.audio_seconds_throughput > 0:
            tpt_lines += (
                f"Throughput: {self.audio_seconds_throughput:.2f} audio sec/s (synthesized audio per wall second)\n"
            )

        # Audio block: RTF + mean audio duration when audio output is present.
        # RTF is dimensionless (E2E / audio_duration); render without the "s"
        # suffix that LatencyStats.__str__ appends.
        audio_lines = ""
        if self.rtf is not None:

            def _fmt(v):
                return f"{v:.3f}" if v is not None else "n/a"

            audio_lines += (
                f"RTF      {' ' * max_len}: "
                f"mean={_fmt(self.rtf.mean)}  p50={_fmt(self.rtf.p50)}  "
                f"p95={_fmt(self.rtf.p95)}  p99={_fmt(self.rtf.p99)}  (lower is better; <1.0 = real-time)\n"
            )
        if self.audio_duration_mean_s is not None:
            audio_lines += f"Audio dur{' ' * (max_len - 1)}: mean={self.audio_duration_mean_s:.3f}s\n"

        # Token block.
        tok_lines = ""
        if self.total_text_tokens > 0:
            avg = f"{self.mean_text_tokens:.1f}" if self.mean_text_tokens is not None else "n/a"
            tok_lines += f"Text tokens: {self.total_text_tokens} total ({avg} avg/req)\n"
        if self.total_input_tokens > 0:
            tok_lines += f"Prompt tokens: {self.total_input_tokens} total\n"
        for modality, total_bytes in sorted(self.total_output_bytes.items()):
            chunks = self.total_response_chunks.get(modality, 0)
            tok_lines += f"Output bytes ({modality}): {total_bytes} total ({chunks} chunks)\n"

        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(self.type_counts.items()))

        return (
            f"{header}\n"
            f"Request type breakdown: {breakdown}\n"
            f"Requests : {self.n_success}/{self.n_requests} succeeded\n"
            f"{ttft_lines}\n"
            f"E2E      {' ' * max_len}: {self.e2e_latency}\n"
            f"{itl_lines}\n"
            f"{sv_line}"
            f"{audio_lines}"
            f"{tok_lines}"
            f"{tpt_lines}"
            f"Total wall time: {self.wall_time:.2f}s"
        )


def _audio_duration_seconds(nbytes: int) -> float:
    """Duration of a 24kHz int16 mono PCM stream of `nbytes`. Hardcoded to the
    Qwen3-Omni audio format; revisit if we ever benchmark a different rate."""
    sample_rate = 24000
    bytes_per_sample = 2
    channels = 1
    return nbytes / (sample_rate * bytes_per_sample * channels)


def aggregate_metrics(
    requests: list[RequestMetrics],
    wall_time: float,
    online: bool = False,
    batch_size: int = 1,
    rate: Optional[float] = None,
    max_concurrency: Optional[int] = None,
    profiling_type: Optional[str] = None,
    model: Optional[Model] = None,
) -> AggregateMetrics:
    n_success = sum(1 for r in requests if r.status == Status.SUCCESS)

    ttft_by_modality: dict[str, list[float]] = {}
    for r in requests:
        for modality, t in r.ttft.items():
            ttft_by_modality.setdefault(modality, []).append(t)

    # ITL: per-chunk audio gaps and per-token-normalised text gaps, flattened
    # across all requests. The text normalisation re-tokenises each chunk
    # (matching sglang.bench_serving's --accept-length path) and replicates
    # the per-token gap by the chunk's token count, weighting percentiles by
    # tokens. For OurSystem and vllm-omni (one token per chunk) this
    # collapses to raw per-chunk gaps; for sglang-omni it un-bursts the
    # chunk-burst into per-token equivalents — making all three comparable.
    tokenizer = None
    if model is not None:
        try:
            tokenizer = model.get_tokenizer()
        except Exception as e:
            print(
                f"WARNING: failed to load tokenizer ({e}); falling back to raw "
                f"per-chunk text ITL (sglang default; not cross-system "
                f"comparable for sglang-omni vs others).",
                file=sys.stderr,
            )
    itl_text_per_token: list[float] = []
    itl_audio_per_chunk: list[float] = []
    for r in requests:
        text_itl = r.itl_per_token_text(tokenizer)
        if text_itl:
            itl_text_per_token.extend(text_itl)
        audio_gaps = r.chunk_gaps.get("audio")
        if audio_gaps:
            itl_audio_per_chunk.extend(audio_gaps)
    itl_by_modality: dict[str, LatencyStats] = {}
    if itl_text_per_token:
        itl_by_modality["text"] = _latency_stats(itl_text_per_token)
    if itl_audio_per_chunk:
        itl_by_modality["audio"] = _latency_stats(itl_audio_per_chunk)

    e2e_vals = [r.e2e_latency for r in requests if r.e2e_latency is not None]
    sv_vals = [r.streaming_viability for r in requests if r.streaming_viability is not None]

    total_chunks: dict[str, int] = {}
    total_bytes: dict[str, int] = {}
    for r in requests:
        for modality, n in r.response_chunks.items():
            total_chunks[modality] = total_chunks.get(modality, 0) + n
        for modality, n in r.output_bytes.items():
            total_bytes[modality] = total_bytes.get(modality, 0) + n

    total_text_tokens = sum(r.output_text_tokens for r in requests)
    text_token_counts = [r.output_text_tokens for r in requests if r.output_text_tokens > 0]
    total_input_tokens = sum(r.input_tokens for r in requests)

    # Per-request audio duration paired with the same request's E2E so RTF
    # (E2E / audio_duration) is computed consistently across requests.
    audio_pairs: list[tuple[RequestMetrics, float]] = []
    for r in requests:
        audio_bytes = r.output_bytes.get("audio", 0)
        if audio_bytes <= 0:
            continue
        dur = _audio_duration_seconds(audio_bytes)
        if dur <= 0:
            continue
        audio_pairs.append((r, dur))
    audio_durations = [d for _, d in audio_pairs]
    audio_duration_mean_s = statistics.mean(audio_durations) if audio_durations else None
    audio_seconds_throughput = sum(audio_durations) / wall_time if wall_time > 0 and audio_durations else None
    rtf_vals = [r.e2e_latency / dur for r, dur in audio_pairs if r.e2e_latency is not None]
    rtf_stats = _latency_stats(rtf_vals) if rtf_vals else None

    request_throughput = n_success / wall_time if wall_time > 0 else None
    text_token_throughput = total_text_tokens / wall_time if wall_time > 0 and total_text_tokens > 0 else None

    type_counts: dict[str, int] = {}
    for r in requests:
        type_counts[r.type.value] = type_counts.get(r.type.value, 0) + 1

    return AggregateMetrics(
        n_requests=len(requests),
        n_success=n_success,
        ttft={m: _latency_stats(vals) for m, vals in ttft_by_modality.items()},
        e2e_latency=_latency_stats(e2e_vals),
        itl=itl_by_modality,
        streaming_viability=_latency_stats(sv_vals, higher_is_better=True) if sv_vals else None,
        wall_time=wall_time,
        online=online,
        batch_size=batch_size,
        rate=rate,
        profiling_type=profiling_type,
        max_concurrency=max_concurrency,
        type_counts=type_counts,
        total_response_chunks=total_chunks,
        total_output_bytes=total_bytes,
        total_text_tokens=total_text_tokens,
        mean_text_tokens=statistics.mean(text_token_counts) if text_token_counts else None,
        total_input_tokens=total_input_tokens,
        request_throughput=request_throughput,
        text_token_throughput=text_token_throughput,
        audio_seconds_throughput=audio_seconds_throughput,
        rtf=rtf_stats,
        audio_duration_mean_s=audio_duration_mean_s,
    )


@dataclass
class RequestInput:
    req_type: RequestType
    prompt: str

    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None

    # Fix 6 — pre-loaded media. Populated by `__post_init__` when paths are
    # provided so adapters never re-read or re-encode per request. Keeping
    # both raw bytes (for OurSystem multipart uploads) and base64 strings
    # (for vllm-omni data: URIs) avoids re-encoding at send time.
    _image_bytes: Optional[bytes] = field(default=None, repr=False)
    _audio_bytes: Optional[bytes] = field(default=None, repr=False)
    _video_bytes: Optional[bytes] = field(default=None, repr=False)
    _image_b64: Optional[str] = field(default=None, repr=False)
    _audio_b64: Optional[str] = field(default=None, repr=False)
    _video_b64: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if self.image_path and self._image_bytes is None:
            self._image_bytes = Path(self.image_path).read_bytes()
            self._image_b64 = base64.b64encode(self._image_bytes).decode()
        if self.audio_path and self._audio_bytes is None:
            self._audio_bytes = Path(self.audio_path).read_bytes()
            self._audio_b64 = base64.b64encode(self._audio_bytes).decode()
        if self.video_path and self._video_bytes is None:
            self._video_bytes = Path(self.video_path).read_bytes()
            self._video_b64 = base64.b64encode(self._video_bytes).decode()

    def get_all_filepaths(self) -> dict[str, str]:
        res = {}
        if self.image_path:
            res["image"] = self.image_path
        if self.audio_path:
            res["audio"] = self.audio_path
        if self.video_path:
            res["video"] = self.video_path
        return res

    def get_bytes(self, modality: str) -> Optional[bytes]:
        return {
            "image": self._image_bytes,
            "audio": self._audio_bytes,
            "video": self._video_bytes,
        }.get(modality)

    def get_b64(self, modality: str) -> Optional[str]:
        return {
            "image": self._image_b64,
            "audio": self._audio_b64,
            "video": self._video_b64,
        }.get(modality)

    def get_filename(self, modality: str) -> str:
        path = {
            "image": self.image_path,
            "audio": self.audio_path,
            "video": self.video_path,
        }.get(modality)
        return os.path.basename(path) if path else ""


class InferenceSystem(ABC):
    @abstractmethod
    async def send_request(
        self,
        session: aiohttp.ClientSession,
        req_input: RequestInput,
        base_url: str,
        request_id: int,
        model: Model,
        additional_model_kwargs: dict = {},
    ) -> RequestMetrics:
        pass


class OurSystem(InferenceSystem):
    async def send_request(
        self,
        session: aiohttp.ClientSession,
        req_input: RequestInput,
        base_url: str,
        request_id: int,
        model: Model,
        additional_model_kwargs: dict = {},
    ) -> RequestMetrics:
        req_type = req_input.req_type
        model_kwargs = json.dumps(
            {
                **model.get_model_kwargs(req_type),
                **additional_model_kwargs,
            }
        )
        output_mod = req_type.get_output_modalities()

        metrics = RequestMetrics(
            request_id=request_id,
            type=req_type,
            expected_output_modalities=[output_mod],
        )

        try:
            form = aiohttp.FormData()
            form.add_field("text", req_input.prompt)
            form.add_field("model_kwargs", model_kwargs)
            form.add_field("output_modalities", output_mod)

            for modality in req_input.get_all_filepaths():
                file_content = req_input.get_bytes(modality)
                if file_content is None:
                    continue
                form.add_field(
                    "files",
                    file_content,
                    filename=req_input.get_filename(modality),
                    content_type="application/octet-stream",
                )

            metrics.start_time = time.monotonic()
            async with session.post(f"{base_url}/generate", data=form, read_bufsize=2**24) as resp:
                resp.raise_for_status()
                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not msg.get("data"):
                        continue
                    arrival_time = time.monotonic()
                    mod = msg.get("modality")
                    data_b64 = msg.get("data", "")

                    metrics.record_output_chunk(
                        modality=mod,
                        data_b64=data_b64,
                        arrival_time=arrival_time,
                        n_tokens=1,  # mminf server emits one token per chunk
                    )

        except Exception as e:
            metrics.record_error(str(e))
        else:
            metrics.record_completion()

        return metrics


class VoxServe(InferenceSystem):
    async def send_request(
        self,
        session: aiohttp.ClientSession,
        req_input: RequestInput,
        base_url: str,
        request_id: int,
        model: Model,
        additional_model_kwargs: dict = {},
    ) -> RequestMetrics:
        req_type = req_input.req_type
        assert req_type == RequestType.T2S, "vox-serve only supports text-to-speech requests"
        metrics = RequestMetrics(
            request_id=request_id,
            type=req_type,
            expected_output_modalities=["audio"],
        )

        try:
            form = aiohttp.FormData()
            form.add_field("text", req_input.prompt)
            form.add_field("streaming", "true")

            metrics.start_time = time.monotonic()
            async with session.post(
                f"{base_url}/generate",
                data=form,
                timeout=aiohttp.ClientTimeout(total=None, sock_read=30),
            ) as resp:
                resp.raise_for_status()

                chunk_count = 0
                async for chunk in resp.content.iter_any():
                    if not chunk:
                        break

                    chunk_count += 1
                    if chunk_count == 1:
                        # Skip WAV header — vox-serve sends it as the first chunk,
                        # our metrics abstraction doesn't expect it
                        continue

                    arrival_time = time.monotonic()
                    data_b64 = base64.b64encode(chunk)

                    metrics.record_output_chunk(
                        modality="audio",
                        data_b64=data_b64,
                        arrival_time=arrival_time,
                        n_tokens=1,
                    )

        except Exception as e:
            metrics.record_error(str(e))
        else:
            metrics.record_completion()

        return metrics


# ---------------------------------------------------------------------------
# vllm-omni and sglang-omni: OpenAI-compatible chat-completions adapters
# ---------------------------------------------------------------------------

# vllm-omni accepts standard OpenAI multimodal content parts (image_url,
# audio_url, video_url) inside `messages[].content`. Its system prompt is
# therefore wrapped as a content-parts list to match the user-message shape.
_SYSTEM_MESSAGE = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": (
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                "capable of perceiving auditory and visual inputs, as well as generating text and speech."
            ),
        }
    ],
}

# sglang-omni does NOT parse OpenAI-style multimodal content parts; multimodal
# media must arrive in top-level `images`/`audios`/`videos` fields and message
# content must be plain text. See sglang-omni's openai_api.py:_build_chat_generate_request
# and protocol.py:ChatCompletionRequest (Bug 1 in sglang-omni_v1_profile.md).
_SYSTEM_MESSAGE_PLAINTEXT = {
    "role": "system",
    "content": (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    ),
}


def _build_openai_user_message(prompt: str, input_modality: str, req_input: RequestInput) -> dict:
    """Build a vllm-omni-compatible user message with OpenAI content parts.

    Always wraps text in a `[{"type": "text", "text": prompt}]` list — this is
    what vllm-omni's own bench sends (see `_get_chat_content` in
    `vllm/benchmarks/lib/endpoint_request_func.py`). Plain-string content
    appears to bypass omni multimodal routing on the server side.

    For multimodal inputs (image/audio/video), prepends the media as the
    first content part using pre-encoded base64 from `req_input.get_b64(...)`
    (Fix 6).
    """
    content: list[dict] = []
    if input_modality != "text":
        media_path = req_input.get_all_filepaths().get(input_modality)
        b64 = req_input.get_b64(input_modality)
        if media_path is not None and b64 is not None:
            mime = (
                mimetypes.guess_type(media_path)[0]
                or {
                    "image": "image/jpeg",
                    "audio": "audio/wav",
                    "video": "video/mp4",
                }[input_modality]
            )
            content.append(
                {
                    "type": f"{input_modality}_url",
                    f"{input_modality}_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def _parse_sse_chunk(raw_line: bytes) -> Optional[dict]:
    """Parse one SSE line and return its JSON payload, or None for non-data lines.

    Returns the sentinel dict {"_done": True} for the [DONE] terminator.
    """
    line = raw_line.strip()
    if not line or not line.startswith(b"data:"):
        return None
    payload_str = line[len(b"data:") :].strip()
    if payload_str == b"[DONE]":
        return {"_done": True}
    try:
        return json.loads(payload_str)
    except json.JSONDecodeError:
        return None


def _record_vllm_omni_chunk(chunk: dict, metrics: "RequestMetrics") -> None:
    """Extract content from one vllm-omni SSE chunk and feed it to metrics.

    vllm-omni's chunks have a top-level `modality` field ("text" / "audio" /
    "image") and audio bytes typically arrive as base64 in
    `choices[0].delta.content` (with `delta.audio.data` as a fallback path).
    Critically, audio chunks ALSO carry `usage` with `completion_tokens=0`,
    so we must process choices unconditionally rather than skipping after
    reading usage.
    """
    arrival_time = time.monotonic()

    usage = chunk.get("usage")
    if usage:
        ct = usage.get("completion_tokens")
        if ct is not None:
            metrics.output_text_tokens = ct
        pt = usage.get("prompt_tokens")
        if pt is not None:
            metrics.input_tokens = pt

    chunk_modality = chunk.get("modality")
    for choice in chunk.get("choices", []):
        delta = choice.get("delta") or {}
        content = delta.get("content")
        audio_obj = delta.get("audio") if isinstance(delta.get("audio"), dict) else None
        audio_b64_field = audio_obj.get("data") if audio_obj else None

        if content and chunk_modality == "audio":
            modality, data_b64 = "audio", content
        elif content and chunk_modality == "image":
            modality, data_b64 = "image", content
        elif content:
            modality = "text"
            data_b64 = base64.b64encode(content.encode()).decode()
        elif audio_b64_field:
            modality, data_b64 = "audio", audio_b64_field
        else:
            continue

        metrics.record_output_chunk(
            modality=modality,
            data_b64=data_b64,
            arrival_time=arrival_time,
            n_tokens=1,
        )


class VLLMOmni(InferenceSystem):
    """
    Benchmark adapter for the vllm-omni OpenAI-compatible server.

    Uses native aiohttp + iter_any()-based SSE parsing. Two important
    implementation details that took a while to track down:

    1. vllm-omni's audio SSE chunks can be very large (~200KB base64 WAV per
       chunk). aiohttp's default line iterator (`async for line in
       resp.content`) has an 8 KB max-line limit and silently drops oversized
       lines. We use `iter_any()` + manual `\\n\\n` message buffering instead
       (matches `vllm_omni/benchmarks/patch/patch.py:391`).
    2. vllm-omni's audio chunks carry BOTH a populated `usage` field (with
       `completion_tokens=0`) AND `choices[0].delta.content` with the audio
       bytes. We must NOT `continue` after reading usage — that drops the
       audio. Read usage AND process choices on every chunk.

    `modalities=[output_mod]` only when `output_mod != "text"` (matches
    vllm-omni's expected behavior: `["audio"]` streams audio-only; absence
    streams text+audio because the talker runs unconditionally).
    """

    async def send_request(
        self,
        session: aiohttp.ClientSession,
        req_input: RequestInput,
        base_url: str,
        request_id: int,
        model: Model,
        additional_model_kwargs: dict = {},
    ) -> RequestMetrics:
        req_type = req_input.req_type
        output_mod = req_type.get_output_modalities()  # "text", "audio", "image"
        input_mod = req_type.get_input_modalities()  # "text", "image", "audio", "video"

        metrics = RequestMetrics(
            request_id=request_id,
            type=req_type,
            expected_output_modalities=[output_mod],
        )

        try:
            user_message = _build_openai_user_message(req_input.prompt, input_mod, req_input)
            # Always send `modalities` explicitly. Omitting the field makes
            # vllm-omni's server fall back to text+audio output even for
            # text-only requests (the talker runs unconditionally), inflating
            # work and emitting unwanted audio chunks. Sending ["text"] tells
            # the server to skip the talker.
            modalities_arg = [output_mod]

            payload: dict = {
                "model": model.get_hf_url(),
                "messages": [_SYSTEM_MESSAGE, user_message],
                "temperature": 0.0,  # match vllm-omni's bench (`patch.py:336`)
                "stream": True,
                "stream_options": {"include_usage": True},
                **model.get_model_kwargs(req_type),
                **additional_model_kwargs,
            }
            payload["modalities"] = modalities_arg

            # Match vllm-omni bench headers (`patch.py:354-358`).
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            }

            metrics.start_time = time.monotonic()
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                read_bufsize=2**24,
                timeout=aiohttp.ClientTimeout(total=None, sock_read=120),
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}: {await resp.text()}")

                buffer = b""
                async for raw_bytes in resp.content.iter_any():
                    if not raw_bytes:
                        continue
                    buffer += raw_bytes
                    while b"\n\n" in buffer:
                        message, buffer = buffer.split(b"\n\n", 1)
                        message = message.strip()
                        if not message:
                            continue
                        chunk = _parse_sse_chunk(message)
                        if chunk is None:
                            continue
                        if chunk.get("_done"):
                            # vllm-omni may emit data after [DONE]; keep reading
                            # until the stream is naturally closed.
                            continue
                        _record_vllm_omni_chunk(chunk, metrics)

                # Trailing-buffer flush: catch a final chunk if the server
                # closed without a final `\n\n`.
                tail = buffer.strip()
                if tail.startswith(b"data:"):
                    payload_str = tail[len(b"data:") :].strip()
                    if payload_str and payload_str != b"[DONE]":
                        try:
                            json.loads(payload_str)
                            chunk = _parse_sse_chunk(tail)
                            if chunk is not None and not chunk.get("_done"):
                                _record_vllm_omni_chunk(chunk, metrics)
                        except json.JSONDecodeError:
                            pass

        except Exception as e:
            metrics.record_error(str(e))
        else:
            metrics.record_completion()

        return metrics


class SGLangOmni(InferenceSystem):
    """
    Benchmark adapter for the sglang-omni OpenAI-compatible server.

    Per Fix 10:
      (a) Multimodal media goes in top-level `images`/`audios`/`videos` fields
          (sglang-omni does NOT parse OpenAI content parts); message content
          must be plain text.
      (b) Audio chunks arrive in `delta.audio.data`, NOT in `delta.content`.

    See sglang-omni's `sglang_omni/serve/openai_api.py:_build_chat_generate_request`
    and `protocol.py:ChatCompletionRequest` for the request schema; see
    `protocol.py:ChatCompletionStreamDelta` and the playground frontend
    (`playground/web/frontend/app.js:1257`) for the audio chunk shape.
    """

    async def send_request(
        self,
        session: aiohttp.ClientSession,
        req_input: RequestInput,
        base_url: str,
        request_id: int,
        model: Model,
        additional_model_kwargs: dict = {},
    ) -> RequestMetrics:
        req_type = req_input.req_type
        output_mod = req_type.get_output_modalities()
        input_mod = req_type.get_input_modalities()

        metrics = RequestMetrics(
            request_id=request_id,
            type=req_type,
            expected_output_modalities=[output_mod],
        )

        try:
            # Same logic as VLLMOmni: audio output needs ["text", "audio"] so
            # the talker has the thinker's text to consume. Matches sglang-omni's
            # own TTS task convention (`benchmarks/tasks/tts.py:907`).
            if output_mod == "audio":
                modalities_arg = ["text", "audio"]
            elif output_mod != "text":
                modalities_arg = [output_mod]
            else:
                modalities_arg = None
            payload: dict = {
                "model": model.get_hf_url(),
                "messages": [
                    _SYSTEM_MESSAGE_PLAINTEXT,
                    {"role": "user", "content": req_input.prompt},
                ],
                "temperature": 0.0,  # match VLLMOmni — greedy for cross-system parity
                "stream": True,
                "stream_options": {"include_usage": True},
                **model.get_model_kwargs(req_type),
                **additional_model_kwargs,
            }
            # Top-level multimodal fields (Fix 10a). sglang-omni reads files
            # from disk on the server side; benchmarks running on the same
            # host as the server work directly with paths.
            if input_mod == "image" and req_input.image_path:
                payload["images"] = [req_input.image_path]
            elif input_mod == "audio" and req_input.audio_path:
                payload["audios"] = [req_input.audio_path]
            elif input_mod == "video" and req_input.video_path:
                payload["videos"] = [req_input.video_path]
            if modalities_arg is not None:
                payload["modalities"] = modalities_arg
                if output_mod == "audio":
                    payload["audio"] = {"format": "wav"}

            metrics.start_time = time.monotonic()
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                read_bufsize=2**24,
                timeout=aiohttp.ClientTimeout(total=None, sock_read=120),
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}: {await resp.text()}")

                async for raw_line in resp.content:
                    chunk = _parse_sse_chunk(raw_line)
                    if chunk is None:
                        continue
                    if chunk.get("_done"):
                        break

                    arrival_time = time.monotonic()

                    # Read usage if present, but DON'T `continue` — sglang-omni
                    # may also emit chunks that carry both usage and audio (the
                    # same dual-payload pattern we hit with vllm-omni). Process
                    # choices unconditionally below.
                    usage = chunk.get("usage")
                    if usage:
                        ct = usage.get("completion_tokens")
                        if ct is not None:
                            metrics.output_text_tokens = ct
                        pt = usage.get("prompt_tokens")
                        if pt is not None:
                            metrics.input_tokens = pt

                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        text_content = delta.get("content")
                        audio_obj = delta.get("audio") if isinstance(delta.get("audio"), dict) else None
                        audio_b64 = audio_obj.get("data") if audio_obj else None

                        # sglang-omni emits text and audio in separate chunks;
                        # accept both fields defensively in case a chunk ever
                        # carries both.
                        if text_content:
                            metrics.record_output_chunk(
                                modality="text",
                                data_b64=base64.b64encode(text_content.encode()).decode(),
                                arrival_time=arrival_time,
                                n_tokens=1,
                            )
                        if audio_b64:
                            metrics.record_output_chunk(
                                modality="audio",
                                data_b64=audio_b64,
                                arrival_time=arrival_time,
                                n_tokens=1,
                            )

        except Exception as e:
            metrics.record_error(str(e))
        else:
            metrics.record_completion()

        return metrics
