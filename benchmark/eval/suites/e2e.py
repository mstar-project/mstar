"""e2e: the request kinds the speech suites do not cover, for ``compare``'s base vs head matrix.

One ``--kind`` per run: greedy chat (text, image, audio or video in, optionally speech out), seeded image and
video generation, voice-clone uploads, VLA actions and streamed world-model frames. Requests are seeded and greedy
where the route allows it, so two servers given the same requests should return the same outputs; the kept
repeat's outputs are saved under the run's audio directory (``<id>.txt``, ``.png``, ``.mp4``, ``.f32``, ``.u8``,
``.pcm``) for ``compare`` to diff. Per repeat: latency percentiles, time to first output, requests/s.

Inputs come from ``--media`` (``python -m benchmark.eval data e2e``); every kind but ``gen_clone`` (which takes a
voice-clone ``--data`` manifest) builds its own prompts, so ``--data`` is optional.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import aiohttp

from benchmark.eval import stats
from benchmark.eval.datasets import Item
from benchmark.eval.suites.base import Suite, error_row, exception_row, parse_extra

KINDS = ("chat_text", "chat_image", "chat_audio", "chat_video", "chat_count", "image", "video", "video_i2v",
         "gen_clone", "pi05", "frames", "world")
MEDIA_KINDS = {"chat_image", "chat_audio", "chat_video", "video_i2v", "pi05", "frames", "world"}
PROMPTS = [
    "Describe this in two sentences.",
    "What is the main subject here? Answer briefly.",
    "List three notable details.",
    "What colors stand out? Answer in one sentence.",
]
AUDIO_PROMPTS = [
    "Transcribe or describe what you hear.",
    "Summarize what is said in one sentence.",
    "What language is spoken, and what is the topic?",
    "List two key points from this audio.",
]
TEXT_PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a haiku about a lighthouse.",
    "Name three prime numbers greater than 50 and explain how you checked.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
]
COUNT_PROMPT = "Count from 1 to 400, separated by commas, with nothing else."
GEN_PROMPTS = [
    "a red fox sitting in fresh snow at sunrise, photorealistic",
    "a bowl of ramen on a wooden table, studio lighting",
    "a lighthouse on a cliff during a storm, oil painting",
    "a small robot watering plants in a greenhouse",
]
PI05_TASKS = ["pick up the red block", "open the drawer", "push the cup left", "stack the blocks"]
WORLD_PROMPT = "a quiet street at dusk, the camera moving slowly forward"
WORLD_FRAMES = 29                         # two 4-latent-frame chunks at a temporal VAE stride of 4
WORLD_INTRINSICS = [500.0, 500.0, 416.0, 240.0]   # fx, fy, cx, cy at 832x480
EXT = {"image": "png", "video": "mp4", "video_i2v": "mp4", "pi05": "f32", "frames": "u8", "world": "u8"}
DEFAULT_ITEMS = 16


def camera_path(n: int, step: float, yaw_deg: float) -> list[list[list[float]]]:
    """``n`` OpenCV camera-to-world poses: forward along +z by ``step`` per frame while turning ``yaw_deg`` per
    frame about the y axis. A seeded, fixed trajectory, so two servers get the same request."""
    import math

    poses = []
    for k in range(n):
        a = math.radians(yaw_deg * k)
        c, s = math.cos(a), math.sin(a)
        poses.append([[c, 0.0, s, step * k * s], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, step * k * c],
                      [0.0, 0.0, 0.0, 1.0]])
    return poses


class E2ESuite(Suite):
    name = "e2e"
    help = "chat, image / video generation, voice-clone uploads, VLA actions and streamed frames (for compare)"

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("e2e")
        g.add_argument("--kind", choices=KINDS, required=True)
        g.add_argument("--media", default=None, help="the media/ directory of `data e2e` (image, audio, video inputs)")
        g.add_argument("--max-tokens", type=int, default=96)
        g.add_argument("--audio-out", action="store_true", help="chat: ask for text and speech")
        g.add_argument("--size", default="1024x1024", help="image / video / frames size, WxH")
        g.add_argument("--extra", action="append", default=[], metavar="KEY=JSON",
                       help="extra request field (chat, image, video) or model_kwargs entry (/generate)")
        g.add_argument("--sample-rate", type=int, default=24000, help="rate of PCM16 audio in the responses")
        g.add_argument("--timeout", type=float, default=1800.0, help="per-request timeout, seconds")

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.extra = parse_extra(args.extra)
        if args.kind in MEDIA_KINDS and not args.media:
            raise SystemExit(f"--kind {args.kind} needs --media (python -m benchmark.eval data e2e)")
        self.media = Path(args.media) if args.media else None
        self._cache: dict[str, bytes] = {}

    def default_items(self) -> list[Item] | None:
        if self.args.kind == "gen_clone":
            return None
        return [Item(id=f"{i:03d}") for i in range(DEFAULT_ITEMS)]

    def check_items(self, items: list[Item]) -> None:
        super().check_items(items)
        if self.args.kind == "gen_clone" and not all(it.text and it.ref_audio for it in items):
            raise ValueError("--kind gen_clone needs a manifest whose items have text and ref_audio (clone set)")

    def config(self) -> dict[str, Any]:
        a = self.args
        return {"kind": a.kind, "max_tokens": a.max_tokens, "audio_out": a.audio_out, "size": a.size,
                "extra": self.extra, "sample_rate": a.sample_rate}

    def _bytes(self, name: str) -> bytes:
        if name not in self._cache:
            self._cache[name] = (self.media / name).read_bytes()
        return self._cache[name]

    def _b64(self, name: str) -> str:
        return base64.b64encode(self._bytes(name)).decode()

    def _chat_body(self, i: int) -> dict[str, Any]:
        kind = self.args.kind
        if kind == "chat_text":
            content: Any = TEXT_PROMPTS[i % len(TEXT_PROMPTS)]
        elif kind == "chat_count":
            content = COUNT_PROMPT
        else:
            content = []
            if kind == "chat_image":
                content.append({"type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + self._b64("image.png")}})
            elif kind == "chat_audio":
                content.append({"type": "input_audio",
                                "input_audio": {"data": self._b64("audio.wav"), "format": "wav"}})
            else:
                content.append({"type": "video_url",
                                "video_url": {"url": "data:video/mp4;base64," + self._b64("video.mp4")}})
            prompts = AUDIO_PROMPTS if kind == "chat_audio" else PROMPTS
            content.append({"type": "text", "text": prompts[i % len(prompts)]})
        body = {"model": self.args.model_name, "messages": [{"role": "user", "content": content}], "temperature": 0.0,
                "max_tokens": self.args.max_tokens, "stream": True, "seed": 1234, **self.extra}
        if self.args.audio_out:
            body["modalities"] = ["text", "audio"]
        return body

    def _form(self, i: int, item: Item) -> aiohttp.FormData:
        kind = self.args.kind
        form = aiohttp.FormData()
        if kind == "gen_clone":
            form.add_field("files", Path(item.ref_audio).read_bytes(), filename="ref.wav", content_type="audio/wav")
            form.add_field("text", item.text)
            form.add_field("input_modalities", "text,audio")
            form.add_field("output_modalities", "audio")
            kwargs: dict[str, Any] = {"ref_text": item.ref_text or "", "seed": 1000 + i}
        elif kind == "pi05":
            for view in ("base", "left", "right"):
                form.add_field("files", self._bytes(f"pi05_{view}.png"), filename=f"{view}_{i % 4}.png",
                               content_type="image/png")
            form.add_field("text", PI05_TASKS[i % 4])
            form.add_field("input_modalities", "image,text")
            form.add_field("output_modalities", "action")
            kwargs = {"robot_state": [0.01 * (j + i % 4) for j in range(32)], "seed": 1000 + i}
        elif kind == "world":
            form.add_field("files", self._bytes("i2v.png"), filename="seed.png", content_type="image/png")
            form.add_field("text", WORLD_PROMPT)
            form.add_field("input_modalities", "image,text")
            form.add_field("output_modalities", "video_frame")
            yaw = [0.0, 1.5, -1.5, 0.75][i % 4]
            kwargs = {"camera_poses": camera_path(WORLD_FRAMES, 0.05, yaw), "intrinsics": WORLD_INTRINSICS,
                      "num_frames": WORLD_FRAMES, "seed": 7}
        else:
            form.add_field("files", self._bytes("waypoint_seed.png"), filename="seed.png", content_type="image/png")
            form.add_field("text", "")
            form.add_field("output_modalities", "video_frame")
            mouse = [-24, 24, 0, 12][i % 4]
            kwargs = {"num_steps": 2, "seed": 7, "actions": [{"mouse": [mouse, 0], "buttons": [], "scroll": 0}] * 2}
        form.add_field("streaming", "true")
        form.add_field("model_kwargs", json.dumps({**kwargs, **self.extra}))
        return form

    async def request(
        self, session: aiohttp.ClientSession, url: str, index: int, item: Item, t0: float
    ) -> dict[str, Any]:
        a = self.args
        first: list[float] = []
        text: list[str] = []
        audio = bytearray()
        blob: bytes | None = None
        timeout = aiohttp.ClientTimeout(total=a.timeout)

        def mark() -> None:
            if not first:
                first.append(time.perf_counter() - t0)

        try:
            if a.kind.startswith("chat_"):
                async with session.post(f"{url}/v1/chat/completions", json=self._chat_body(index),
                                        timeout=timeout) as r:
                    if r.status != 200:
                        return error_row(item, f"HTTP {r.status}", await r.text())
                    finished = False
                    async for msg in _lines(r, b"data:"):
                        if msg is DONE:
                            finished = True
                            continue
                        if msg.get("error"):
                            return error_row(item, "stream error", json.dumps(msg["error"])[:500])
                        for ch in msg.get("choices", []):
                            finished = finished or ch.get("finish_reason") is not None
                            d = ch.get("delta", {})
                            if d.get("content"):
                                mark()
                                text.append(d["content"])   # one delta per decode step: the token count
                            if (d.get("audio") or {}).get("data"):
                                mark()
                                audio.extend(base64.b64decode(d["audio"]["data"]))
                    if not finished:   # the connection ended before the reply did
                        return error_row(item, "incomplete stream", "no finish_reason or [DONE]")
            elif a.kind in ("image", "video", "video_i2v"):
                body = {"model": a.model_name, "prompt": GEN_PROMPTS[index % len(GEN_PROMPTS)], "size": a.size,
                        "seed": 1000 + index}
                if a.kind == "image":
                    body |= {"n": 1, "response_format": "b64_json"}
                if a.kind == "video_i2v":
                    body["image"] = "data:image/png;base64," + self._b64("i2v.png")
                body |= self.extra
                route = "images" if a.kind == "image" else "videos"
                async with session.post(f"{url}/v1/{route}/generations", json=body, timeout=timeout) as r:
                    js = await r.json(content_type=None)
                    if r.status != 200:
                        return error_row(item, f"HTTP {r.status}", json.dumps(js)[:500])
                blob = base64.b64decode(js["data"][0]["b64_json"])
            else:
                chunks: list[bytes] = []
                async with session.post(f"{url}/generate", data=self._form(index, item), timeout=timeout) as r:
                    if r.status != 200:
                        return error_row(item, f"HTTP {r.status}", await r.text())
                    async for msg in _lines(r):
                        if msg is DONE:
                            continue
                        if msg.get("modality") == "error":
                            return error_row(item, "stream error", str(msg)[:500])
                        if msg.get("data"):
                            mark()
                            data = base64.b64decode(msg["data"])
                            if msg.get("modality") == "audio":
                                audio.extend(data)
                            else:
                                chunks.append(data)
                blob = b"".join(chunks) or None
        except Exception as exc:  # noqa: BLE001 - every failure is an error row
            return exception_row(item, exc)
        e2e = time.perf_counter() - t0
        if not text and not audio and not blob:
            return error_row(item, "empty output", e2e_s=e2e)
        out = "".join(text).encode() + bytes(audio) + (blob or b"")
        return {"id": item.id, "ok": True, "error": None, "e2e_s": e2e, "ttft_s": first[0] if first else None,
                "text": "".join(text), "n_chunks": len(text), "audio_s": len(audio) / 2 / a.sample_rate,
                "out_sha": hashlib.sha256(out).hexdigest()[:16], "_audio": bytes(audio), "_blob": blob}

    def score(self, items: list[Item], rows: list[dict[str, Any]], wall_s: float, audio_dir: Path | None) -> dict:
        ok = [r for r in rows if r["ok"]]
        if audio_dir is not None:
            save_outputs(ok, audio_dir, self.args.kind)
        return {
            "requests": len(rows),
            "errors": len(rows) - len(ok),
            "lat_p50_s": stats.percentile([r["e2e_s"] for r in ok], 50),
            "lat_p90_s": stats.percentile([r["e2e_s"] for r in ok], 90),
            "lat_mean_s": stats.mean(r["e2e_s"] for r in ok),
            "ttft_p50_s": stats.percentile([r["ttft_s"] for r in ok], 50),
            "req_per_s": len(ok) / wall_s if wall_s > 0 else None,
            "audio_s_total": sum(r["audio_s"] for r in ok),
        }


async def _lines(resp: aiohttp.ClientResponse, prefix: bytes = b""):
    """JSON messages of an SSE (``prefix=b"data:"``) or NDJSON stream. Lines are split by hand: one audio chunk can
    be a single line of several MB, past aiohttp's readline limit."""
    buf = b""
    async for chunk in resp.content.iter_any():
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            msg = _parse(line, prefix)
            if msg is not None:
                yield msg
    msg = _parse(buf, prefix)
    if msg is not None:
        yield msg


DONE: dict = {}   # sentinel for an SSE stream's [DONE]


def _parse(line: bytes, prefix: bytes) -> dict | None:
    line = line.strip()
    if not line or (prefix and not line.startswith(prefix)):
        return None
    line = line[len(prefix):].strip()
    return DONE if line == b"[DONE]" else json.loads(line)


def save_outputs(rows: list[dict[str, Any]], d: Path, kind: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r["text"]:
            (d / f"{r['id']}.txt").write_text(r["text"], encoding="utf-8")
        if r["_blob"]:
            (d / f"{r['id']}.{EXT.get(kind, 'bin')}").write_bytes(r["_blob"])
        if r["_audio"]:
            (d / f"{r['id']}.pcm").write_bytes(r["_audio"])
