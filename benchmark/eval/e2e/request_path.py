"""Request-path cases (multipart, urlencoded and JSON parsing, h11, media decoding) and served invariants: one
result per case, compared base against head by status code and identity flags (``checks.rp_facts``).

python -m benchmark.eval.e2e.request_path --url http://127.0.0.1:PORT --suite SUITE \\
    --inputs DATA_DIR/rp --out rp.json

``whisper`` and ``omni`` exercise the request path; ``chat`` and ``speech`` are the served invariants (AGENTS.md):
identical requests give identical output, streamed equals non-streamed, an edge budget is honoured, and a
malformed request is a 4xx after which the server still serves. ``cancel_speech`` and ``cancel_chat`` drop a
streamed request's connection mid-transfer (after 3 chunks, and after 1.5 s) and check the server still serves the
same reply: on a multi-worker config the cancelled request's tensors are still moving between workers.
``timeout_speech`` and ``timeout_chat`` run against a server whose per-request ``--timeout`` is shorter than one of
their requests, so the server itself cancels it mid-transfer (#572's trigger).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import socket
import tempfile
import time
import wave
from pathlib import Path

import msgpack
import requests
import websockets

GREEDY = {"thinker_temperature": 0.0, "max_output_tokens": 48, "seed": 1234}
INVARIANT_PROMPT = "Name three prime numbers greater than 50."


def h(b: bytes | str) -> str:
    return hashlib.sha256(b.encode() if isinstance(b, str) else b).hexdigest()[:12]


def summarize(r: requests.Response) -> dict:
    ct = r.headers.get("content-type", "")
    out: dict = {"status": r.status_code, "content_type": ct.split(";")[0]}
    body = r.content
    if "json" in ct and "ndjson" not in ct:
        try:
            js = r.json()
            if isinstance(js, dict):
                out["text"] = js.get("text")
                if js.get("error"):
                    err = js["error"]
                    out["error"] = str(err.get("message", err) if isinstance(err, dict) else err)[:300]
                if js.get("detail"):
                    out["error"] = str(js["detail"])[:300]
                if "outputs" in js:
                    out["text"] = "".join(base64.b64decode(c.get("data", "")).decode("utf-8", "replace")
                                          for c in js["outputs"].get("text", []))
                if js.get("choices"):
                    out["text"] = js["choices"][0]["message"].get("content")
        except Exception:  # noqa: BLE001
            out["body"] = body[:200].decode("utf-8", "replace")
    elif "event-stream" in ct:
        events = [ln for ln in body.decode("utf-8", "replace").splitlines() if ln.startswith("data:")]
        out["sse_events"] = len(events)
        out["sse_done"] = any(ln.strip() == "data: [DONE]" for ln in events)
        texts = []
        for ln in events:
            try:
                js = json.loads(ln[5:])
            except Exception:  # noqa: BLE001
                continue
            err = js.get("error") if isinstance(js, dict) else None
            if err or js.get("type") == "error":   # an error event on an HTTP 200 stream
                err = err if isinstance(err, dict) else {"message": err or js.get("message")}
                out["error"] = str(err.get("message", err))[:300]
                out["error_status"] = err.get("code") or err.get("status") or "error"
            if js.get("type") == "transcript.text.done":
                out["text"] = js.get("text")
            if js.get("type") == "transcript.text.delta":
                texts.append(js.get("delta", ""))
            texts += [(ch.get("delta") or {}).get("content") or "" for ch in js.get("choices", []) or []]
        out.setdefault("text", "".join(texts))
    elif "ndjson" in ct:
        texts = []
        for ln in body.splitlines():
            try:
                js = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if js.get("modality") == "text":
                texts.append(base64.b64decode(js.get("data", "")).decode("utf-8", "replace"))
            if js.get("modality") == "error":
                out["error"] = base64.b64decode(js.get("data", "")).decode("utf-8", "replace")[:300]
                out["error_status"] = (js.get("metadata") or {}).get("status")
            elif js.get("error"):
                out["error"] = str(js)[:300]
        out["text"] = "".join(texts)
    else:
        out["body"] = body[:200].decode("utf-8", "replace")
    return out


def call(name: str, results: dict, fn) -> None:
    t0 = time.perf_counter()
    try:
        res = fn()
    except Exception as e:  # noqa: BLE001
        res = {"exception": type(e).__name__, "detail": str(e)[:200]}
    res["s"] = round(time.perf_counter() - t0, 2)
    results[name] = res
    print(name, json.dumps({k: v for k, v in res.items() if k != "s"})[:300], flush=True)


def raw_http(url: str, raw: bytes) -> dict:
    """Send raw bytes on a socket (h11 parser paths) and return the status line."""
    host, port = url.split("//")[1].rsplit(":", 1)
    s = socket.create_connection((host, int(port)), timeout=30)
    s.sendall(raw)
    data = b""
    try:
        while b"\r\n" not in data:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
    except TimeoutError:
        pass
    s.close()
    return {"status_line": data.split(b"\r\n", 1)[0].decode("latin-1") or "<closed>"}


def h11_cases(url: str, r: dict) -> None:
    call("h11_ok_get", r, lambda: raw_http(url, b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n"))
    p = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
    call("h11_bad_chunk", r, lambda: raw_http(url, p + b"Transfer-Encoding: chunked\r\n\r\n5\r\nhelloXX0\r\n\r\n"))
    te = b"Transfer-Encoding: chunked\r\n\r\n"
    call("h11_chunk_ext_bare_lf", r, lambda: raw_http(url, p + te + b"2;x\nab\r\n0\r\n\r\n"))
    call("h11_good_chunked", r, lambda: raw_http(url, p + b"Transfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n"))
    call("h11_cl_te_both", r, lambda: raw_http(url, p + b"Content-Length: 4\r\n" + te + b"0\r\n\r\n"))
    call("h11_bad_header", r, lambda: raw_http(url, b"GET /health HTTP/1.1\r\nHost: x\r\nBad Header: y\r\n\r\n"))


def whisper(url: str, inputs: Path, scratch: Path) -> dict:
    r: dict = {}
    t = f"{url}/v1/audio/transcriptions"
    wav = (inputs / "speech.wav").read_bytes()

    def tr(files, data=None, stream=False):
        return summarize(requests.post(t, files=files, data=data or {}, stream=stream, timeout=900))

    for ext in ("wav", "mp3", "flac", "ogg", "m4a"):
        p = inputs / f"speech.{ext}"
        call(f"asr_{ext}", r, lambda p=p: tr({"file": (p.name, p.read_bytes(), "application/octet-stream")}))
    call("asr_json_fmt", r, lambda: tr({"file": ("a.wav", wav)}, {"response_format": "json", "language": "en"}))
    call("asr_verbose", r, lambda: tr({"file": ("a.wav", wav)}, {"response_format": "verbose_json"}))
    call("asr_text_fmt", r, lambda: tr({"file": ("a.wav", wav)}, {"response_format": "text"}))
    call("asr_list_field", r, lambda: tr({"file": ("a.wav", wav)},
                                          [("timestamp_granularities[]", "segment"), ("language", "en")]))
    call("asr_stream_sse", r, lambda: tr({"file": ("a.wav", wav)}, {"stream": "true"}, stream=True))
    call("asr_long_20min", r, lambda: tr({"file": ("long.wav", (inputs / "long.wav").read_bytes())}))
    call("asr_empty_file", r, lambda: tr({"file": ("a.wav", b"")}))
    call("asr_no_file", r, lambda: summarize(requests.post(t, data={"language": "en"}, timeout=60)))
    call("asr_urlencoded", r, lambda: summarize(requests.post(t, data={"file": "x", "language": "en"}, timeout=60)))
    call("asr_garbage_audio", r, lambda: tr({"file": ("a.wav", os.urandom(50000))}))
    call("asr_2mb_text_field", r, lambda: tr({"file": ("a.wav", wav)}, {"prompt": "a" * (2 * 2**20)}))
    call("asr_1100_fields", r, lambda: tr({"file": ("a.wav", wav)}, [(f"f{i}", "1") for i in range(1100)]))
    call("asr_1100_files", r, lambda: tr([("file", ("a.wav", wav))]
                                         + [(f"x{i}", ("x.bin", b"0")) for i in range(1100)]))
    big = scratch / "big.wav"
    with open(big, "wb") as f:
        f.truncate(513 * 2**20)   # sparse: one MiB over the multipart file limit
    call("asr_over_limit_513mb", r, lambda: tr({"file": ("big.wav", open(big, "rb"))}))  # noqa: SIM115
    call("asr_bad_boundary", r, lambda: summarize(requests.post(
        t, data=b"--zzz\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nabc",
        headers={"content-type": "multipart/form-data; boundary=zzz"}, timeout=60)))
    call("asr_no_boundary", r, lambda: summarize(requests.post(
        t, data=b"abc", headers={"content-type": "multipart/form-data"}, timeout=60)))

    async def realtime():
        """The event types the server sends after a session update (up to the update's acknowledgement), and an
        error if one of them is an error event."""
        events: list[str] = []
        out: dict = {}
        try:
            async with websockets.connect(url.replace("http", "ws") + "/v1/realtime?intent=transcription",
                                          max_size=None) as ws:
                await ws.send(json.dumps({"type": "transcription_session.update", "session": {}}))
                try:
                    while len(events) < 4 and not any(e.endswith(".updated") for e in events):
                        msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
                        events.append(str(msg.get("type")))
                        if msg.get("type") == "error" or msg.get("error"):
                            out["error"] = str(msg.get("error") or msg)[:200]
                            break
                except websockets.ConnectionClosed as e:
                    out["close_code"] = e.rcvd.code if e.rcvd else None
                except (TimeoutError, ValueError):
                    pass
        except Exception as e:  # noqa: BLE001
            return {"exception": type(e).__name__, "detail": str(e)[:200]}
        return {**out, "events": ",".join(events)}

    call("realtime_ws", r, lambda: asyncio.run(realtime()))
    h11_cases(url, r)
    return r


def omni(url: str, inputs: Path, scratch: Path) -> dict:  # noqa: ARG001
    r: dict = {}
    g = f"{url}/generate"
    mk = json.dumps(GREEDY)
    img = (inputs / "img.png").read_bytes()

    def gen(files=None, data=None, **kw):
        d = {"text": "Describe the input in one sentence.", "model_kwargs": mk, **(data or {})}
        return summarize(requests.post(g, files=files, data=d, timeout=900, **kw))

    def one(name: str, mime: str, fname: str | None = None, data: dict | None = None):
        return gen([("files", (fname or name, (inputs / name).read_bytes(), mime))], data)

    for fmt, mime in (("png", "image/png"), ("jpg", "image/jpeg"), ("webp", "image/webp"), ("gif", "image/gif")):
        call(f"gen_img_{fmt}", r, lambda fmt=fmt, mime=mime: one(f"img.{fmt}", mime))
    call("gen_img_png16", r, lambda: one("img16.png", "image/png"))
    call("gen_img_rgba", r, lambda: one("rgba.png", "image/png"))
    call("gen_img_cmyk_jpg", r, lambda: one("cmyk.jpg", "image/jpeg"))
    call("gen_img_anim_gif", r, lambda: one("anim.gif", "image/gif"))
    call("gen_img_exif_rot", r, lambda: one("exif_rot.jpg", "image/jpeg"))
    call("gen_img_truncated", r, lambda: gen([("files", ("trunc.png", img[:20000], "image/png"))]))
    call("gen_img_4k_noise", r, lambda: one("noise4k.png", "image/png"))
    call("gen_audio_wav", r, lambda: one("speech.wav", "audio/wav", data={"text": "What is said in the audio?"}))
    call("gen_audio_mp3", r, lambda: one("speech.mp3", "audio/mpeg", data={"text": "What is said in the audio?"}))
    call("gen_video_mp4", r, lambda: one("video.mp4", "video/mp4", data={"text": "What happens in the video?"}))
    call("gen_multi_files", r, lambda: gen([("files", ("img.png", img, "image/png")),
                                            ("files", ("a.wav", (inputs / "speech.wav").read_bytes(), "audio/wav"))],
                                           {"text": "Describe the image and the audio."}))
    call("gen_nonstream", r, lambda: gen(data={"streaming": "false", "text": "Name three colors."}))
    call("gen_unknown_ext", r, lambda: gen([("files", ("x.xyz", b"abc", "application/octet-stream"))]))
    # urlencoded (no files): starlette's form parser
    call("urlenc_small", r, lambda: summarize(requests.post(g, data={"text": "Name three colors.", "model_kwargs": mk},
                                                             timeout=300)))
    call("urlenc_1_5mb_field", r, lambda: summarize(requests.post(g, data={"text": "hi " * 500000, "model_kwargs": mk},
                                                                   timeout=300)))
    call("urlenc_1100_fields", r, lambda: summarize(requests.post(
        g, data=[("text", "Name three colors."), ("model_kwargs", mk)] + [(f"f{i}", "1") for i in range(1100)],
        timeout=300)))
    call("multipart_1_5mb_text", r, lambda: gen([("files", ("img.png", img, "image/png"))], {"text": "hi " * 500000}))
    call("multipart_1100_fields", r, lambda: summarize(requests.post(
        g, files=[("files", ("img.png", img, "image/png"))],
        data=[("text", "x"), ("model_kwargs", mk)] + [(f"f{i}", "1") for i in range(1100)], timeout=300)))
    c = f"{url}/v1/chat/completions"

    def chat(content, stream=False):
        body = {"messages": [{"role": "user", "content": content}], "temperature": 0.0, "max_tokens": 48,
                "seed": 1234, "stream": stream}
        return summarize(requests.post(c, json=body, timeout=900, stream=stream))

    call("chat_json", r, lambda: chat("Name three colors."))
    call("chat_sse", r, lambda: chat("Name three colors.", stream=True))
    big = base64.b64encode((inputs / "noise4k.png").read_bytes()).decode()
    call("chat_large_dataurl", r, lambda: chat([
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big}"}},
        {"type": "text", "text": "Describe briefly."}]))
    call("chat_bad_json", r, lambda: summarize(requests.post(c, data=b"{not json", timeout=60,
                                                             headers={"content-type": "application/json"})))
    call("chat_bad_dataurl", r, lambda: chat([{"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!"}},
                                              {"type": "text", "text": "x"}]))

    def speech_json():
        resp = requests.post(f"{url}/v1/audio/speech", json={"input": "Hello there.", "response_format": "wav"},
                             timeout=900)
        return {"status": resp.status_code, "bytes_gt_1k": len(resp.content) > 1000,
                "content_type": resp.headers.get("content-type", "").split(";")[0]}

    call("speech_json", r, speech_json)

    async def gen_ws():
        out = {}
        async with websockets.connect(url.replace("http", "ws") + "/generate/ws", max_size=None) as ws:
            await ws.send(json.dumps({"request_id": "j1", "text": "Name three colors.", "model_kwargs": GREEDY}))
            await ws.send(msgpack.packb({"request_id": "m1", "text": "Describe the image in one sentence.",
                                         "files": [{"name": "img.png", "data": img}], "model_kwargs": GREEDY},
                                        use_bin_type=True))
            texts: dict[str, list[str]] = {"j1": [], "m1": []}
            done = set()
            while len(done) < 2:
                msg = await asyncio.wait_for(ws.recv(), 300)
                m = msgpack.unpackb(msg, raw=False) if isinstance(msg, bytes) else json.loads(msg)
                rid = m.get("request_id")
                if m.get("finish") or m.get("error"):
                    done.add(rid)
                    if m.get("error"):
                        out[f"{rid}_error"] = str(m["error"])[:200]
                    continue
                if m.get("modality") == "text":
                    d = m.get("data")
                    texts[rid].append(d.decode() if isinstance(d, bytes) else base64.b64decode(d).decode())
        out.update({k: "".join(v) for k, v in texts.items()})
        return out

    call("generate_ws", r, lambda: asyncio.run(gen_ws()))
    h11_cases(url, r)
    return r


def _chat(url: str, stream: bool = False, **kw) -> tuple[int, str, dict]:
    body = {"model": "default", "messages": [{"role": "user", "content": INVARIANT_PROMPT}], "temperature": 0.0,
            "max_tokens": 32, "seed": 1234, "stream": stream, **kw}
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=600, stream=stream)
    if r.status_code != 200:
        return r.status_code, "", {}
    if not stream:
        js = r.json()
        return 200, js["choices"][0]["message"].get("content") or "", js.get("usage") or {}
    text = []
    for line in r.iter_lines():
        if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
            text += [(ch.get("delta") or {}).get("content") or "" for ch in json.loads(line[5:]).get("choices", [])]
    return 200, "".join(text), {}


def chat_invariants(url: str, inputs: Path, scratch: Path) -> dict:  # noqa: ARG001
    r: dict = {}
    s1, t1, _ = _chat(url)
    s2, t2, _ = _chat(url)
    r["same_twice"] = {"status": s1, "identical": s1 == s2 == 200 and t1 == t2, "sha": h(t1)}
    s3, t3, _ = _chat(url, stream=True)
    r["stream_equals_nonstream"] = {"status": s3, "identical": s3 == 200 and t3 == t1}
    s4, t4, usage = _chat(url, max_tokens=1)
    r["max_tokens_1"] = {"status": s4, "completion_tokens": usage.get("completion_tokens"),
                         "ok": s4 == 200 and usage.get("completion_tokens") in (1, None) and len(t4) < 40}
    bad = requests.post(f"{url}/v1/chat/completions", data=b"{not json", timeout=60,
                        headers={"content-type": "application/json"})
    r["malformed_json"] = {"status": bad.status_code}
    bad2 = requests.post(f"{url}/v1/chat/completions", json={"model": "default", "messages": "hello"}, timeout=60)
    r["bad_messages_field"] = {"status": bad2.status_code}
    s5, t5, _ = _chat(url)
    r["serves_after_errors"] = {"status": s5, "identical": t5 == t1}
    return r


def _speech(url: str) -> tuple[int, bytes]:
    body = {"model": "default", "input": "The train leaves tomorrow morning.", "response_format": "wav", "seed": 7}
    resp = requests.post(f"{url}/v1/audio/speech", json=body, timeout=600)
    if resp.status_code != 200:
        return resp.status_code, b""
    try:
        with wave.open(io.BytesIO(resp.content)) as w:
            return 200, w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        return 200, b""


def speech_invariants(url: str, inputs: Path, scratch: Path) -> dict:  # noqa: ARG001
    r: dict = {}
    s1, a1 = _speech(url)
    s2, a2 = _speech(url)
    r["same_twice"] = {"status": s1, "identical": s1 == s2 == 200 and a1 == a2}
    bad = requests.post(f"{url}/v1/audio/speech", data=b"{not json", timeout=60,
                        headers={"content-type": "application/json"})
    r["malformed_json"] = {"status": bad.status_code}
    bad2 = requests.post(f"{url}/v1/audio/speech", json={"model": "default", "input": 5}, timeout=60)
    r["bad_input_field"] = {"status": bad2.status_code}
    s3, a3 = _speech(url)
    r["serves_after_errors"] = {"status": s3, "identical": a3 == a1}
    return r


CANCEL_TEXT = ("The quick brown fox jumps over the lazy dog, then runs across the field, past the old barn and the "
               "river, until the sun goes down behind the hills and the stars come out one by one.")
SHORT_TEXT = "The train leaves tomorrow morning."
STORY = [{"role": "user", "content": "Tell a long story about a lighthouse keeper and a storm."}]
SETTLE_S = 2.0           # after a cancel, before the follow-up request: teardown of the cancelled request is in flight
TIMEOUT_SETTLE_S = 6.0   # #572: the worker died 1 to 5 s after the timed-out request was cancelled
VOICE: str | None = None   # --voice: the speech suites' voice (the model's default when unset)


def _status(fn) -> tuple[int | str, str]:
    """(status, reply hash) of one request; a refused or reset connection (a server that died) is a status too."""
    try:
        return fn()
    except requests.RequestException as e:
        return type(e).__name__, ""


def _cut_stream(url: str, path: str, body: dict, chunks: int = 3, after_s: float | None = None) -> dict:
    """Start a streamed request and close the socket after ``chunks`` body chunks (or ``after_s`` seconds), without
    a cancel: the server learns from the dropped connection while tensors for the request are still moving."""
    t0 = time.perf_counter()
    got, cut = 0, False
    try:
        with requests.post(f"{url}{path}", json=body, stream=True, timeout=120) as r:
            status = r.status_code
            if status == 200:
                for chunk in r.iter_content(chunk_size=None):
                    got += bool(chunk)
                    if got >= chunks or (after_s is not None and time.perf_counter() - t0 >= after_s):
                        cut = True
                        break
    except requests.RequestException as e:
        return {"status": type(e).__name__, "cut": False}
    return {"status": status, "cut": cut}   # cut False: the stream ended before the cut, nothing was cancelled


def _cancel_suite(url: str, path: str, body: dict, reference) -> dict:
    """The same request served whole, cut after 3 chunks, cut after 1.5 s, then whole again: the server must
    still serve, and give the same reply as before the cuts."""
    r: dict = {}
    s1, ref = _status(reference)
    r["reference"] = {"status": s1}
    r["disconnect_after_3_chunks"] = _cut_stream(url, path, body, chunks=3)
    time.sleep(SETTLE_S)
    r["disconnect_after_1_5_s"] = _cut_stream(url, path, body, chunks=10**9, after_s=1.5)
    time.sleep(SETTLE_S)
    s2, again = _status(reference)
    r["serves_after_cancels"] = {"status": s2, "identical": s1 == s2 == 200 and again == ref}
    cuts = (r["disconnect_after_3_chunks"], r["disconnect_after_1_5_s"])
    if not any(c["cut"] for c in cuts):
        r["_unexercised"] = "both streams ended before their cut"
        r["_completed"] = all(c["status"] == 200 for c in cuts)
    return r


def _timeout_suite(reference, long_request) -> dict:
    """On a server started with a per-request ``--timeout`` shorter than ``long_request``: a short request served
    (once to warm up), the long one timed out by the server (which cancels it mid-transfer), then the short one
    again after the window in which #572's worker crashed. The short reply must not change."""
    _status(reference)
    s1, ref = _status(reference)
    r: dict = {"reference": {"status": s1}}
    s, said = _status(long_request)
    timed_out = s == 500 and "timed out" in said.lower()   # the API server's own reply (entrypoint.collect_results)
    r["times_out"] = {"status": s, "ok": timed_out}
    if not timed_out:
        r["_unexercised"] = f"the long request was not timed out by the server (status {s})"
        r["_completed"] = s == 200
    time.sleep(TIMEOUT_SETTLE_S)
    s2, again = _status(reference)
    r["serves_after_timeout"] = {"status": s2, "identical": s1 == s2 == 200 and again == ref}
    return r


def _speech_req(url: str, text: str, **kw):
    """(status, reply hash), or the error text when the status is not 200."""
    voice = {"voice": VOICE} if VOICE else {}

    def fn() -> tuple[int, str]:
        resp = requests.post(f"{url}/v1/audio/speech", json={"model": "default", "input": text, "seed": 7,
                                                             "response_format": "wav", **voice, **kw}, timeout=600)
        return resp.status_code, h(resp.content) if resp.status_code == 200 else resp.text[:300]

    return fn


def cancel_speech(url: str, inputs: Path, scratch: Path) -> dict:  # noqa: ARG001
    body = {"model": "default", "input": CANCEL_TEXT, "stream": True, "seed": 7, **({"voice": VOICE} if VOICE else {})}
    return _cancel_suite(url, "/v1/audio/speech", body, _speech_req(url, CANCEL_TEXT))


def cancel_chat(url: str, inputs: Path, scratch: Path) -> dict:  # noqa: ARG001
    body = {"model": "default", "messages": STORY, "temperature": 0.0, "max_tokens": 96, "seed": 1234,
            "stream": True, "modalities": ["text", "audio"]}

    def whole() -> tuple[int, str]:
        s, text, _ = _chat(url, stream=True, messages=STORY, max_tokens=48)
        return s, h(text)

    return _cancel_suite(url, "/v1/chat/completions", body, whole)


def timeout_speech(url: str, inputs: Path, scratch: Path) -> dict:  # noqa: ARG001
    return _timeout_suite(_speech_req(url, SHORT_TEXT), _speech_req(url, " ".join([CANCEL_TEXT] * 4)))


def timeout_chat(url: str, inputs: Path, scratch: Path) -> dict:  # noqa: ARG001
    def short() -> tuple[int, str]:
        s, text, _ = _chat(url, max_tokens=16)
        return s, h(text)

    def long() -> tuple[int, str]:
        body = {"model": "default", "messages": STORY, "temperature": 0.0, "max_tokens": 512, "seed": 1234,
                "modalities": ["text", "audio"]}
        resp = requests.post(f"{url}/v1/chat/completions", json=body, timeout=600)
        return resp.status_code, "" if resp.status_code == 200 else resp.text[:300]

    return _timeout_suite(short, long)


SUITES = {"whisper": whisper, "omni": omni, "chat": chat_invariants, "speech": speech_invariants,
          "cancel_speech": cancel_speech, "cancel_chat": cancel_chat, "timeout_speech": timeout_speech,
          "timeout_chat": timeout_chat}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--url", required=True)
    p.add_argument("--suite", required=True, choices=SUITES)
    p.add_argument("--inputs", required=True, help="the rp/ directory of `data e2e`")
    p.add_argument("--out", required=True)
    p.add_argument("--voice", default=None, help="voice of the cancel and timeout speech suites")
    a = p.parse_args(argv)
    global VOICE
    VOICE = a.voice
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mstar-rp-") as scratch:
        res = SUITES[a.suite](a.url.rstrip("/"), Path(a.inputs), Path(scratch))
    res["_elapsed_s"] = round(time.perf_counter() - t0, 1)
    tmp = Path(a.out).with_suffix(".tmp")   # whole or not at all
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
