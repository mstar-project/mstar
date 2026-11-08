"""Live checks of a running Muse Glimmer server through its OpenAI API.

Against a server started with ``mstar serve muse_glimmer``; exits nonzero on any
failed check:

  stream      a greedy answer streamed equals the same answer non-streamed
  repeat      two identical greedy requests return identical text
  image       an image's color is named; a two-image prompt answers (the checkpoint
              itself calls a red and a blue square "both red": HF does too)
  bad input   an unknown reasoning effort and an unsupported video part are 400s,
              and the server answers normally afterwards
  abort       a stream dropped after a few chunks leaves the server serving

  python benchmark/muse_glimmer/serve_checks.py --port 8000
"""
import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request

MODEL = "muse_glimmer"
GREEDY = {"temperature": 0.0, "reasoning_effort": "low"}


def _png(color: str) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (96, 64), color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _mp4() -> str:
    """A real 8-frame clip, so the request reaches the model's own refusal."""
    import av
    import numpy as np

    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=8)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for i in range(8):
            frame = np.full((64, 64, 3), 30 * i, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return "data:video/mp4;base64," + base64.b64encode(buf.getvalue()).decode()


class Client:
    def __init__(self, port: int):
        self.url = f"http://localhost:{port}/v1/chat/completions"

    def _request(self, body: dict):
        req = urllib.request.Request(
            self.url, json.dumps({"model": MODEL, **body}).encode(),
            {"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=600)

    def chat(self, content, **body) -> str:
        with self._request({"messages": [{"role": "user", "content": content}], **body}) as r:
            return json.load(r)["choices"][0]["message"]["content"] or ""

    def stream(self, content, max_chunks: int | None = None, **body) -> list[str]:
        chunks = []
        with self._request({"messages": [{"role": "user", "content": content}], "stream": True, **body}) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[5:])["choices"][0].get("delta", {}).get("content")
                if delta:
                    chunks.append(delta)
                if max_chunks is not None and len(chunks) >= max_chunks:
                    break   # leaving the `with` drops the connection mid-stream
        return chunks

    def status_of(self, content, **body) -> int:
        try:
            with self._request({"messages": [{"role": "user", "content": content}], **body}) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    c = Client(args.port)
    failures = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            failures.append(name)

    q = "Name three primary colors."
    whole = c.chat(q, max_tokens=64, **GREEDY)
    streamed = "".join(c.stream(q, max_tokens=64, **GREEDY))
    check("stream", whole == streamed and bool(whole), f"{whole[:60]!r} vs {streamed[:60]!r}")
    check("repeat", c.chat(q, max_tokens=64, **GREEDY) == whole, "same request, same text")

    for color in ("red", "blue"):
        one = c.chat([{"type": "image_url", "image_url": {"url": _png(color)}},
                      {"type": "text", "text": "What color is this image? One word."}], max_tokens=64, **GREEDY)
        check(f"image {color}", f"to=user<|message|>{color}" in one.lower(), repr(one[-60:]))
    two = c.chat([{"type": "image_url", "image_url": {"url": _png("red")}},
                  {"type": "image_url", "image_url": {"url": _png("blue")}},
                  {"type": "text", "text": "Name the colors of the two images."}], max_tokens=64, **GREEDY)
    check("two images", bool(two.strip()), repr(two[-60:]))

    check("bad reasoning effort", c.status_of(q, max_tokens=4, temperature=0.0, reasoning_effort="ultra") == 400,
          "reasoning_effort='ultra'")
    video = [{"type": "video_url", "video_url": {"url": _mp4()}},
             {"type": "text", "text": "What happens?"}]
    check("video refused", c.status_of(video, max_tokens=4, **GREEDY) == 400, "video part")
    check("serves after bad input", c.chat(q, max_tokens=64, **GREEDY) == whole, "the stream check's answer")

    c.stream("Count from 1 to 200, one number per line.", max_chunks=3, max_tokens=512, **GREEDY)
    check("serves after abort", c.chat(q, max_tokens=64, **GREEDY) == whole, "the stream check's answer")

    print("SERVE CHECKS", "FAIL " + ", ".join(failures) if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
