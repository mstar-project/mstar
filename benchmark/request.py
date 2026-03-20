
import base64
from dataclasses import dataclass
import json
import mimetypes
from pathlib import Path
import time
import statistics
from typing import Optional
import aiohttp

from benchmark.base import Model, RequestType, Status


@dataclass
class RequestMetrics:
    request_id: str
    type: RequestType
    start_time: Optional[float]=None
    status: Status=Status.PROGRESS,
    ttft: Optional[float]=None
    e2e_latency: Optional[float]=None
    error: Optional[str]=None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.monotonic()
    
    def record_token(self):
        if self.ttft is None:
            self.ttft = time.monotonic() - self.start_time
    
    def record_completion(self):
        self.e2e_latency = time.monotonic() - self.start_time
        self.status = Status.SUCCESS
    
    def record_error(self, msg: str):
        self.e2e_latency = time.monotonic() - self.start_time
        self.status = Status.FAILED
        self.error = msg


@dataclass
class LatencyStats:
    mean: Optional[float]
    p50: Optional[float]
    p95: Optional[float]
    p99: Optional[float]

    def __str__(self) -> str:
        def fmt(v: Optional[float]) -> str:
            return f"{v:.3f}s" if v is not None else "n/a"
        return f"mean={fmt(self.mean)}  p50={fmt(self.p50)}  p95={fmt(self.p95)}  p99={fmt(self.p99)}"


@dataclass
class AggregateMetrics:
    n_requests: int
    n_success: int
    wall_time: float
    ttft: LatencyStats
    e2e_latency: LatencyStats
    type_counts: dict[str, int]
    rate: Optional[float]=None

    def __str__(self) -> str:
        if self.rate is not None:
            header = f"Benchmark Results ({self.n_requests} requests, rate={self.rate} req/s)"
        else:
            header = f"Benchmark Results ({self.n_requests} requests, sequential)"
        header += "\n" + ("\u2500" * 50)

        tpt = ""
        if self.wall_time > 0:
            throughput = self.n_success / self.wall_time
            tpt = f"Throughput: {throughput:.2f} req/s (successful only)\n"
        
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(self.type_counts.items()))

        return (
            f"{header}\n"
            f"Request type breakdown: {breakdown}\n"
            f"Requests : {self.n_success}/{self.n_requests} succeeded\n"
            f"TTFT     : {self.ttft}\n"
            f"E2E      : {self.e2e_latency}\n"
            f"{tpt}"
            f"Total wall time: {self.wall_time:.2f}s"
        )


def _latency_stats(values: list[float]) -> LatencyStats:
    if not values:
        return LatencyStats(mean=None, p50=None, p95=None, p99=None)

    sorted_vals = sorted(values)

    def percentile(p: float) -> float:
        idx = (p / 100) * (len(sorted_vals) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)

    return LatencyStats(
        mean=statistics.mean(values),
        p50=percentile(50),
        p95=percentile(95),
        p99=percentile(99),
    )


def aggregate_metrics(
    requests: list[RequestMetrics],
    wall_time: float,
    rate: Optional[float]=None,
) -> AggregateMetrics:
    n_success = sum(1 for r in requests if r.status == Status.SUCCESS)
    ttft_vals = [r.ttft for r in requests if r.ttft is not None]
    e2e_vals = [r.e2e_latency for r in requests if r.e2e_latency is not None]

    type_counts: dict[str, int] = {}
    for r in requests:
        type_counts[r.type.value] = type_counts.get(r.type.value, 0) + 1
    
    return AggregateMetrics(
        n_requests=len(requests),
        n_success=n_success,
        ttft=_latency_stats(ttft_vals),
        e2e_latency=_latency_stats(e2e_vals),
        wall_time=wall_time,
        rate=rate,
        type_counts=type_counts
    )


@dataclass
class RequestInput:
    req_type: RequestType
    prompt: str
    image_path: Optional[str]=None


from abc import ABC, abstractmethod


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

        metrics = RequestMetrics(request_id=request_id, type=req_type)
        try:
            form = aiohttp.FormData()
            form.add_field("text", prompt)
            form.add_field("model_kwargs", model_kwargs)
            form.add_field("output_modalities", output_mod)

            if image_path is not None:
                path = Path(image_path)
                file_bytes = path.read_bytes()
                form.add_field("files", file_bytes, filename=path.name, content_type="application/octet-stream")

            output_modalities_recvd = []

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
                    metrics.record_token()
                    mod = msg.get("modality")
                    if mod == "image":
                        decoded = base64.b64decode(msg.get("data"))
                        assert decoded, "Image output unable to be decoded"
                    output_modalities_recvd.append(mod)

            if output_mod not in output_modalities_recvd:
                raise Exception(f"Expected {output_mod} output but got modalities: {output_modalities_recvd}")

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
                await self._chat(session, base_url, model, prompt, None, metrics, additional_model_kwargs, output_modality="image")
            elif req_type == RequestType.I2I:
                if image_path is None:
                    raise ValueError("image_path is required for I2I requests")
                await self._chat(session, base_url, model, prompt, image_path, metrics, additional_model_kwargs, output_modality="image")
            elif req_type in (RequestType.T2T, RequestType.I2T):
                await self._chat(session, base_url, model, prompt, image_path, metrics, additional_model_kwargs, output_modality="text")
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
        # if output_modality != "text":
        #     extra_body.setdefault("modalities", [output_modality])

        payload = {
            "model": model.get_hf_url(),
            "messages": messages,
        }
        if extra_body:
            payload["extra_body"] = extra_body

        async with session.post(
            f"{base_url}/v1/chat/completions", json=payload, read_bufsize=2**24
        ) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}: {await resp.text()}")

            resp_json = await resp.json()
            choices = resp_json.get("choices", [])
            if not choices:
                raise Exception(f"No choices in response: {resp_json}")

            msg = choices[0].get("message", {})
            content = msg.get("content", [])

            # content is a list of modality chunks for image output, plain string for text
            if isinstance(content, list):
                for chunk in content:
                    if chunk.get("type") == "image_url" or chunk.get("type") == "text":
                        metrics.record_token()
            elif content:
                metrics.record_token()