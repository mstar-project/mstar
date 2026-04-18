
import base64
import io
import json
import mimetypes
import os
import statistics
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
        return (
            f"mean={fmt(self.mean)}  p50={fmt(self.p50)}"
            f"  p95={fmt(self.p95)}  p99={fmt(self.p99)}"
        )


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
    start_time: Optional[float] = None
    status: Status = Status.PROGRESS

    # Per-modality TTFT: e.g. {"text": 0.12, "audio": 0.34}
    ttft: dict[str, float] = field(default_factory=dict)

    e2e_latency: Optional[float] = None
    error: Optional[str] = None

    # Per-modality chunk counts (one per record_token call)
    output_chunks: dict[str, int] = field(default_factory=dict)
    # Per-modality byte counts (raw bytes received)
    output_bytes: dict[str, int] = field(default_factory=dict)
    # Text token count (set via record_text_tokens)
    output_text_tokens: int = 0
    
    _output_modalities_recvd: list[str] = field(default_factory=list, repr=False)

    # Streaming viability tracking (audio only)
    # _audio_chunk_log tracks (inter_chunk_latency, prev_chunk_length), from which streaming
    # viability cann be directly computed
    _audio_chunk_log: list[tuple[float, float]] = field(default_factory=list, repr=False)
    _last_audio_chunk_time: Optional[float] = field(default=None, repr=False)
    _last_audio_chunk_duration: Optional[float] = field(default=None, repr=False)

    # For storing outputs of different modalities
    _text_chunks: list[str] = field(default_factory=list, repr=False)
    _image_chunks: list[bytes] = field(default_factory=list, repr=False)
    _audio_pcm: io.BytesIO = field(default_factory=io.BytesIO, repr=False)

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.monotonic()

    def record_output_chunk(
        self, modality: str,
        data_b64: str,
        n_tokens: int=1,
        arrival_time: Optional[float] = None
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
            modality=modality, nbytes=len(data), 
            arrival_time=arrival_time,
            n_tokens=n_tokens
        )

        if modality == "text":
            self._text_chunks.append(data.decode("utf-8", errors="replace"))
        elif modality == "image":
            self._image_chunks.append(data)
        elif modality == "audio":
            self._audio_pcm.write(data)


    def record_completion(self):
        """
        Finalise timing and assemble output_content from buffered chunks.
        Optionally write audio to a WAV file at output_path.
        """
        self.e2e_latency = time.monotonic() - self.start_time

        if self.expected_output_modalities and any([
            mod not in self._output_modalities_recvd for mod in self.expected_output_modalities
        ]):
            print(f"ERROR: Expected {self.expected_output_modalities} output but got modalities: {self._output_modalities_recvd}")
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

    def record_token(self, modality: str, nbytes: int, arrival_time: float | None=None, n_tokens: int=1):
        """
        Record a received chunk for the given modality.
        Possible modalities: "text", "image", "audio".
        nbytes is the raw byte size of the chunk.
        For text token counts, use record_text_tokens() instead.
        """
        now = time.monotonic()
        if arrival_time is not None:
            now = arrival_time

        if modality not in self.ttft:
            self.ttft[modality] = now - self.start_time

        self.output_chunks[modality] = self.output_chunks.get(modality, 0) + 1
        self.output_bytes[modality] = self.output_bytes.get(modality, 0) + nbytes

        if modality == "text":
            self._record_text_tokens(n_tokens)
        elif modality == "audio":
            chunk_duration = self._pcm_duration_bytes(nbytes)
            if (self._last_audio_chunk_time is not None
                    and self._last_audio_chunk_duration is not None):
                inter_chunk_latency = now - self._last_audio_chunk_time
                self._audio_chunk_log.append(
                    (inter_chunk_latency, self._last_audio_chunk_duration)
                )
            self._last_audio_chunk_time = now
            self._last_audio_chunk_duration = chunk_duration

    def _record_text_tokens(self, n_tokens: int):
        """Accumulate decoded token count for the text modality."""
        self.output_text_tokens += n_tokens

    def record_completion(self):
        self.e2e_latency = time.monotonic() - self.start_time
        self.status = Status.SUCCESS

    def record_error(self, msg: str):
        self.e2e_latency = time.monotonic() - self.start_time
        self.status = Status.FAILED
        self.error = msg

    @property
    def mean_itl(self) -> dict[str, Optional[float]]:
        """
        Per-modality mean inter-token latency: (E2E - TTFT) / (output_chunks - 1).
        Only defined for modalities with > 1 chunk and a recorded TTFT.
        """
        if self.e2e_latency is None:
            return {}
        result = {}
        for modality, n_chunks in self.output_chunks.items():
            ttft = self.ttft.get(modality)
            if ttft is not None and n_chunks > 1:
                result[modality] = (self.e2e_latency - ttft) / (n_chunks - 1)
        return result

    @property
    def streaming_viability(self) -> Optional[float]:
        """
        Fraction of audio chunks where inter-chunk latency < duration of the
        previous chunk. Only defined when at least one inter-chunk gap exists.
        A value of 1.0 means perfectly continuous audio; lower means dropouts.
        """
        if not self._audio_chunk_log:
            return None
        viable = sum(
            1 for gap, prev_dur in self._audio_chunk_log if gap < prev_dur
        )
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
    total_output_chunks: dict[str, int] = field(default_factory=dict)
    total_output_bytes: dict[str, int] = field(default_factory=dict)
    total_text_tokens: int = 0
    mean_text_tokens: Optional[float] = None
    online: bool = False
    batch_size: int = 1
    rate: Optional[float] = None

    def __str__(self) -> str:
        if self.online:
            header = f"Online Benchmark Results ({self.n_requests} requests, rate={self.rate} req/s)"
        else:
            header = f"Offline Benchmark Results ({self.n_requests} requests, batch={self.batch_size})"
        header += "\n" + ("─" * 50)

        tpt = ""
        if self.wall_time > 0:
            throughput = self.n_success / self.wall_time
            tpt = f"Throughput: {throughput:.2f} req/s (successful only)\n"

        tok_lines = ""
        if self.total_text_tokens > 0:
            avg = f"{self.mean_text_tokens:.1f}" if self.mean_text_tokens is not None else "n/a"
            tok_lines += f"Text tokens: {self.total_text_tokens} total ({avg} avg/req)\n"
        for modality, total_bytes in sorted(self.total_output_bytes.items()):
            chunks = self.total_output_chunks.get(modality, 0)
            tok_lines += f"Output bytes ({modality}): {total_bytes} total ({chunks} chunks)\n"

        ttft_lines = "\n".join(
            f"TTFT ({m})  : {s}" for m, s in sorted(self.ttft.items())
        )
        itl_lines = "\n".join(
            f"ITL  ({m})  : {s}" for m, s in sorted(self.itl.items())
        )
        sv_line = (
            f"Audio SV   : {self.streaming_viability}\n"
            if self.streaming_viability is not None
            else ""
        )

        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(self.type_counts.items()))

        return (
            f"{header}\n"
            f"Request type breakdown: {breakdown}\n"
            f"Requests : {self.n_success}/{self.n_requests} succeeded\n"
            f"{ttft_lines}\n"
            f"E2E      : {self.e2e_latency}\n"
            f"{itl_lines}\n"
            f"{sv_line}"
            f"{tok_lines}"
            f"{tpt}"
            f"Total wall time: {self.wall_time:.2f}s"
        )


def aggregate_metrics(
    requests: list[RequestMetrics],
    wall_time: float,
    online: bool = False,
    batch_size: int = 1,
    rate: Optional[float] = None,
) -> AggregateMetrics:
    n_success = sum(1 for r in requests if r.status == Status.SUCCESS)

    ttft_by_modality: dict[str, list[float]] = {}
    for r in requests:
        for modality, t in r.ttft.items():
            ttft_by_modality.setdefault(modality, []).append(t)

    itl_by_modality: dict[str, list[float]] = {}
    for r in requests:
        for modality, itl in r.mean_itl.items():
            if itl is not None:
                itl_by_modality.setdefault(modality, []).append(itl)

    e2e_vals = [r.e2e_latency for r in requests if r.e2e_latency is not None]
    sv_vals = [r.streaming_viability for r in requests if r.streaming_viability is not None]

    total_chunks: dict[str, int] = {}
    total_bytes: dict[str, int] = {}
    for r in requests:
        for modality, n in r.output_chunks.items():
            total_chunks[modality] = total_chunks.get(modality, 0) + n
        for modality, n in r.output_bytes.items():
            total_bytes[modality] = total_bytes.get(modality, 0) + n

    total_text_tokens = sum(r.output_text_tokens for r in requests)
    text_token_counts = [r.output_text_tokens for r in requests if r.output_text_tokens > 0]

    type_counts: dict[str, int] = {}
    for r in requests:
        type_counts[r.type.value] = type_counts.get(r.type.value, 0) + 1

    return AggregateMetrics(
        n_requests=len(requests),
        n_success=n_success,
        ttft={m: _latency_stats(vals) for m, vals in ttft_by_modality.items()},
        e2e_latency=_latency_stats(e2e_vals),
        itl={m: _latency_stats(vals) for m, vals in itl_by_modality.items()},
        streaming_viability=_latency_stats(sv_vals, higher_is_better=True) if sv_vals else None,
        wall_time=wall_time,
        online=online,
        batch_size=batch_size,
        rate=rate,
        type_counts=type_counts,
        total_output_chunks=total_chunks,
        total_output_bytes=total_bytes,
        total_text_tokens=total_text_tokens,
        mean_text_tokens=statistics.mean(text_token_counts) if text_token_counts else None,
    )


@dataclass
class RequestInput:
    req_type: RequestType
    prompt: str
    image_path: Optional[str]=None


class InferenceSystem(ABC):
    @abstractmethod
    async def send_request(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        request_id: int,
        req_type: RequestType,
        model: Model,
        prompt: str,
        image_path: Optional[str] = None,
        additional_model_kwargs: dict = {},
    ) -> RequestMetrics:
        pass


class OurSystem(InferenceSystem):
    async def send_request(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        request_id: int,
        req_type: RequestType,
        model: Model,
        prompt: str,
        image_path: Optional[str] = None,
        additional_model_kwargs: dict = {},
    ) -> RequestMetrics:
        model_kwargs = json.dumps({
            **model.get_model_kwargs(req_type),
            **additional_model_kwargs
        })
        output_mod = req_type.get_output_modalities()

        metrics = RequestMetrics(
            request_id=request_id, type=req_type,
            expected_output_modalities=[output_mod]
        )

        try:
            form = aiohttp.FormData()
            form.add_field("text", prompt)
            form.add_field("model_kwargs", model_kwargs)
            form.add_field("output_modalities", output_mod)

            if image_path is not None:
                path = Path(image_path)
                file_bytes = path.read_bytes()
                form.add_field("files", file_bytes, filename=path.name, content_type="application/octet-stream")

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
                        arrival_time=arrival_time
                    )

        except Exception as e:
            metrics.record_error(str(e))
        else:
            metrics.record_completion()

        return metrics


class VLLMOmni(InferenceSystem):
    async def send_request(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        request_id: int,
        req_type: RequestType,
        model: Model,
        prompt: str,
        image_path: Optional[str] = None,
        additional_model_kwargs: dict = {},
    ) -> RequestMetrics:
        metrics = RequestMetrics(request_id=request_id, type=req_type)
        try:
            if req_type == RequestType.T2I:
                await self._chat(
                    session, base_url, model, prompt, None, metrics,
                    additional_model_kwargs, output_modality="image",
                )
            elif req_type == RequestType.I2I:
                if image_path is None:
                    raise ValueError("image_path is required for I2I requests")
                await self._chat(
                    session, base_url, model, prompt, image_path, metrics,
                    additional_model_kwargs, output_modality="image",
                )
            elif req_type in (RequestType.T2T, RequestType.I2T):
                await self._chat(
                    session, base_url, model, prompt, image_path, metrics,
                    additional_model_kwargs, output_modality="text",
                )
            else:
                raise ValueError(f"Unsupported request type: {req_type}")
        except Exception as e:
            metrics.record_error(str(e))
        else:
            metrics.record_completion()
        return metrics

    async def _chat(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        model: Model,
        prompt: str,
        image_path: Optional[str],
        metrics: RequestMetrics,
        additional_model_kwargs: dict,
        output_modality: str = "text",
    ) -> None:
        if image_path is not None:
            image_bytes = Path(image_path).read_bytes()
            b64 = base64.b64encode(image_bytes).decode()
            content_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{content_type};base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": f"<|im_start|>{prompt}<|im_end|>",
                },
            ]
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": f"<|im_start|>{prompt}<|im_end|>"}]

        extra_body = {k: v for k, v in additional_model_kwargs.items()}

        payload = {
            "model": model.get_hf_url(),
            "messages": messages,
            "max_tokens": 2048,
        }
        # Only send modalities for image output — vLLM-Omni returns empty
        # content for I2T when modalities: ["text"] is explicitly set
        if output_modality != "text":
            payload["modalities"] = [output_modality]
        if extra_body:
            payload["extra_body"] = extra_body
        
        # TODO refactor. figure out streaming for this!

        async with session.post(
            f"{base_url}/v1/chat/completions", json=payload, read_bufsize=2**24
        ) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}: {await resp.text()}")

            resp_json = await resp.json()
            choices = resp_json.get("choices", [])
            if not choices:
                raise Exception(f"No choices in response: {resp_json}")

            # Extract token count from usage if available
            usage = resp_json.get("usage", {})
            metrics.output_tokens = usage.get("completion_tokens", 0)

            msg = choices[0].get("message", {})
            content = msg.get("content", "")

            if isinstance(content, list):
                for chunk in content:
                    if chunk.get("type") == "image_url":
                        metrics.record_token()
                        # Extract image bytes
                        url = chunk.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            _, b64_data = url.split(",", 1)
                            metrics.output_content = base64.b64decode(b64_data)
                    elif chunk.get("type") == "text":
                        metrics.record_token()
            elif content:
                metrics.record_token()
                metrics.output_content = content
