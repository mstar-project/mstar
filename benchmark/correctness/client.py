"""Send a sample set to a running server and keep everything it returns.

Native /generate (NDJSON, one modality per chunk) or an OpenAI JSON route.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp


@dataclass
class Request:
    item_id: str
    route: str = "generate"  # "generate" or an OpenAI path such as "/v1/audio/speech"
    text: str | None = None
    files: list[Path] = field(default_factory=list)
    input_modalities: str | None = None
    output_modalities: str = "text"
    model_kwargs: dict = field(default_factory=dict)
    json_body: dict | None = None  # OpenAI routes
    request_id: str | None = None


@dataclass
class Response:
    item_id: str
    ok: bool = False
    error: str | None = None
    status: int | None = None
    latency_s: float | None = None
    ttft_s: float | None = None
    chunks: dict[str, int] = field(default_factory=dict)
    # One entry per text chunk. Most models send decoded UTF-8, Whisper and Higgs send raw BPE bytes,
    # so join the bytes before decoding.
    text_chunks: list[bytes] = field(default_factory=list)
    audio: bytearray = field(default_factory=bytearray)  # raw bytes as sent (PCM16 on /generate)
    images: list[bytes] = field(default_factory=list)
    action: bytearray = field(default_factory=bytearray)
    video: bytearray = field(default_factory=bytearray)
    other: dict[str, bytearray] = field(default_factory=dict)
    metadata: list[dict] = field(default_factory=list)
    body: bytes = b""  # whole body of a non-streamed OpenAI response
    content_type: str = ""

    @property
    def text(self) -> str:
        return b"".join(self.text_chunks).decode("utf-8", errors="replace")

    def audio_sample_rate(self, default: int = 24000) -> int:
        for m in self.metadata:
            if m.get("modality") == "audio" and m.get("sample_rate"):
                return int(m["sample_rate"])
        return default

    def summary(self) -> dict:
        return {
            "id": self.item_id, "ok": self.ok, "error": self.error, "status": self.status,
            "latency_s": self.latency_s, "ttft_s": self.ttft_s, "chunks": self.chunks,
            "text_chars": len(self.text), "audio_bytes": len(self.audio), "images": len(self.images),
            "action_bytes": len(self.action), "video_bytes": len(self.video), "body_bytes": len(self.body),
            "content_type": self.content_type,
            "metadata": self.metadata[:3] + (self.metadata[-1:] if len(self.metadata) > 3 else []),
        }


def _record_chunk(resp: Response, msg: dict) -> None:
    mod = msg.get("modality")
    data = base64.b64decode(msg.get("data") or "")
    meta = msg.get("metadata")
    if meta and not (mod in ("audio", "text") and resp.chunks.get(mod)):
        # Metadata repeats on every chunk, keep the first one per modality.
        resp.metadata.append({"modality": mod, **meta} if isinstance(meta, dict) else {"modality": mod, "m": meta})
    if mod == "error":
        resp.error = data.decode("utf-8", errors="replace") or json.dumps(meta)
        return
    resp.chunks[mod] = resp.chunks.get(mod, 0) + 1
    if mod == "text":
        resp.text_chunks.append(data)
    elif mod == "audio":
        resp.audio += data
    elif mod == "image":
        resp.images.append(data)
    elif mod == "action":
        resp.action += data
    elif mod in ("video", "video_frame"):
        resp.video += data
    else:
        resp.other.setdefault(str(mod), bytearray()).extend(data)


async def _send_generate(session: aiohttp.ClientSession, base_url: str, req: Request, resp: Response) -> None:
    form = aiohttp.FormData()
    if req.text is not None:
        form.add_field("text", req.text)
    form.add_field("model_kwargs", json.dumps(req.model_kwargs))
    form.add_field("output_modalities", req.output_modalities)
    if req.input_modalities:
        form.add_field("input_modalities", req.input_modalities)
    if req.request_id:
        form.add_field("request_id", req.request_id)
    for path in req.files:
        form.add_field("files", Path(path).read_bytes(), filename=Path(path).name,
                       content_type="application/octet-stream")
    t0 = time.monotonic()
    async with session.post(f"{base_url}/generate", data=form, read_bufsize=2**24) as r:
        resp.status = r.status
        if r.status != 200:
            resp.error = f"HTTP {r.status}: {(await r.text())[:2000]}"
            return
        async for raw in r.content:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not msg.get("data") and msg.get("modality") != "error":
                continue
            if resp.ttft_s is None:
                resp.ttft_s = time.monotonic() - t0
            _record_chunk(resp, msg)


async def _send_openai(session: aiohttp.ClientSession, base_url: str, req: Request, resp: Response) -> None:
    t0 = time.monotonic()
    async with session.post(f"{base_url}{req.route}", json=req.json_body or {}, read_bufsize=2**24,
                            headers={"Authorization": "Bearer EMPTY"}) as r:
        resp.status = r.status
        resp.content_type = r.headers.get("Content-Type", "")
        buf = io.BytesIO()
        async for chunk in r.content.iter_any():
            if resp.ttft_s is None and chunk:
                resp.ttft_s = time.monotonic() - t0
            buf.write(chunk)
        resp.body = buf.getvalue()
        if r.status != 200:
            resp.error = f"HTTP {r.status}: {resp.body[:2000].decode('utf-8', errors='replace')}"


async def _send_one(session, base_url: str, req: Request) -> Response:
    resp = Response(req.item_id)
    t0 = time.monotonic()
    try:
        if req.route == "generate":
            await _send_generate(session, base_url, req, resp)
        else:
            await _send_openai(session, base_url, req, resp)
    except Exception as e:  # recorded per item; the run continues
        resp.error = f"{type(e).__name__}: {e}"
    resp.latency_s = time.monotonic() - t0
    resp.ok = resp.error is None
    return resp


async def _send_all(base_url: str, requests: list[Request], concurrency: int, timeout_s: float,
                    progress: bool) -> list[Response]:
    sem = asyncio.Semaphore(max(1, concurrency))
    done = 0

    async def limited(req):
        nonlocal done
        async with sem:
            r = await _send_one(session, base_url, req)
        done += 1
        if progress:
            status = "ok" if r.ok else f"ERROR {r.error[:160]}"
            print(f"  [{done}/{len(requests)}] {req.item_id} {r.latency_s:.2f}s {status}", flush=True)
        return r

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s),
                                     connector=aiohttp.TCPConnector(limit=max(100, concurrency + 10))) as session:
        return list(await asyncio.gather(*(limited(r) for r in requests)))


def send_all(base_url: str, requests: list[Request], concurrency: int = 1, timeout_s: float = 1800.0,
             progress: bool = True) -> list[Response]:
    """Send every request. Results come back in request order."""
    return asyncio.run(_send_all(base_url.rstrip("/"), requests, concurrency, timeout_s, progress))


def warmup(base_url: str, requests: list[Request], n: int) -> None:
    """Throwaway requests so first-request compiles happen before the real ones."""
    if n > 0 and requests:
        send_all(base_url, [requests[i % len(requests)] for i in range(n)], concurrency=1, progress=False)
