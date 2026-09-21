#!/usr/bin/env python3
"""API-surface probe for the image routes (FLUX.2 [klein] / Z-Image servers).

    python benchmark/flux2_klein/api_probe.py --url http://localhost:8000 --model flux2_klein --out-dir results/c5

Exercises, against a running server, what the parity and benchmark runs do not: ``output_format`` jpeg / webp and
the PNG knobs (extra JSON fields), ``n=2`` with a seed (image 0 must equal the ``n=1`` image byte for byte), the error
paths (size not a multiple of 16, oversize, empty prompt, zero steps — expected: a 4xx, not a 500), a prompt far
beyond 512 tokens (truncation), a client that disconnects mid-request followed by a normal request, and the edit
route with RGBA and palette PNG uploads and a ``size`` field. Prints one line per check and a PASS / FAIL summary;
writes the JSON report.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

PROMPT = "a cat holding a sign that says hello world, studio lighting, detailed fur"
MAGIC = {"png": b"\x89PNG", "jpeg": b"\xff\xd8\xff", "webp": b"RIFF"}


def post_json(url: str, body: dict, timeout: float = 600) -> tuple[int, dict | None, float]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode())
        except Exception:  # noqa: BLE001
            payload = None
        return e.code, payload, time.perf_counter() - t0


def post_multipart(url: str, fields: dict[str, str], files: list[tuple[str, str, bytes]],
                   timeout: float = 600) -> tuple[int, dict | None, float]:
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    for name, value in fields.items():
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    for name, filename, data in files:
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                   f"Content-Type: image/png\r\n\r\n".encode())
        body.write(data)
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(url, data=body.getvalue(),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode())
        except Exception:  # noqa: BLE001
            payload = None
        return e.code, payload, time.perf_counter() - t0


def images_of(payload: dict | None) -> list[bytes]:
    return [base64.b64decode(d["b64_json"]) for d in (payload or {}).get("data", [])]


def psnr(a: bytes, b: bytes) -> float:
    x = np.array(Image.open(io.BytesIO(a)).convert("RGB")).astype(np.float64)
    y = np.array(Image.open(io.BytesIO(b)).convert("RGB")).astype(np.float64)
    if x.shape != y.shape:
        return float("nan")
    mse = float(np.mean((x - y) ** 2))
    return math.inf if mse == 0 else 20 * math.log10(255.0) - 10 * math.log10(mse)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--edits", action="store_true", default=True)
    ap.add_argument("--no-edits", dest="edits", action="store_false", help="model without /v1/images/edits")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_url, edit_url = f"{args.url}/v1/images/generations", f"{args.url}/v1/images/edits"
    base = {"model": args.model, "prompt": PROMPT, "size": "1024x1024", "n": 1, "response_format": "b64_json",
            "num_inference_steps": args.steps, "seed": 3}
    report: dict[str, dict] = {}
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str, **extra) -> None:
        report[name] = {"ok": bool(ok), "detail": detail, **extra}
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            failures.append(name)

    status, payload, wall = post_json(gen_url, base)
    ref = images_of(payload)
    check("baseline png", status == 200 and len(ref) == 1 and ref[0][:4] == MAGIC["png"],
          f"status {status}, {len(ref)} image(s), {wall * 1000:.0f} ms")
    if ref:
        (out / "baseline.png").write_bytes(ref[0])

    for fmt in ("jpeg", "webp"):
        status, payload, wall = post_json(gen_url, {**base, "output_format": fmt, "output_compression": 90})
        imgs = images_of(payload)
        ok = status == 200 and imgs and imgs[0][:len(MAGIC[fmt])] == MAGIC[fmt]
        detail = (f"status {status}, magic {imgs[0][:4]!r}, {len(imgs[0])} bytes, {wall * 1000:.0f} ms" if imgs
                  else f"status {status}")
        check(f"output_format {fmt}", bool(ok), detail)
        if imgs:
            (out / f"baseline.{fmt}").write_bytes(imgs[0])
    status, payload, wall = post_json(gen_url, {**base, "png_compress_level": 0})
    imgs = images_of(payload)
    same_pixels = bool(imgs) and bool(ref) and psnr(ref[0], imgs[0]) == math.inf
    check("png_compress_level 0", status == 200 and same_pixels and imgs[0][:4] == MAGIC["png"],
          f"status {status}, {len(imgs[0]) if imgs else 0} bytes vs {len(ref[0]) if ref else 0} at level 1, "
          f"same pixels: {same_pixels}")

    status, payload, wall = post_json(gen_url, {**base, "n": 2})
    imgs = images_of(payload)
    same0 = bool(imgs) and bool(ref) and imgs[0] == ref[0]
    check("n=2 seeded", status == 200 and len(imgs) == 2 and same0 and imgs[1] != imgs[0],
          f"status {status}, {len(imgs)} images, image 0 identical to n=1: {same0}, {wall * 1000:.0f} ms")

    for name, bad in (("size 1000x1000", {"size": "1000x1000"}), ("size 8192x8192", {"size": "8192x8192"}),
                      ("empty prompt", {"prompt": ""}), ("zero steps", {"num_inference_steps": 0}),
                      ("size garbage", {"size": "big"})):
        status, payload, wall = post_json(gen_url, {**base, **bad}, timeout=120)
        msg = (payload or {}).get("error", payload)
        check(f"error path: {name}", 400 <= status < 500, f"status {status}: {str(msg)[:120]}", status=status)

    long_prompt = " ".join(["a very detailed painting of a cat"] * 400)
    status, payload, wall = post_json(gen_url, {**base, "prompt": long_prompt})
    imgs = images_of(payload)
    check("prompt beyond 512 tokens", status == 200 and bool(imgs),
          f"status {status}, {wall * 1000:.0f} ms (truncated prompt)")

    try:
        post_json(gen_url, {**base, "seed": 99}, timeout=0.05)
    except (urllib.error.URLError, TimeoutError, OSError) as e:  # the client gives up while the server works
        report["disconnect"] = {"client_error": type(e).__name__}
    status, payload, wall = post_json(gen_url, {**base, "seed": 100})
    check("request after a client disconnect", status == 200 and bool(images_of(payload)),
          f"status {status}, {wall * 1000:.0f} ms (a healthy server answers in about one generation)")

    if args.edits and ref:
        rgb = ref[0]
        img = Image.open(io.BytesIO(rgb)).convert("RGB")
        rgba_buf, pal_buf = io.BytesIO(), io.BytesIO()
        img.convert("RGBA").save(rgba_buf, format="PNG")
        img.convert("P", palette=Image.ADAPTIVE, colors=256).save(pal_buf, format="PNG")
        fields = {"model": args.model, "prompt": "make the sign say goodbye, watercolor style", "seed": "11",
                  "num_inference_steps": str(args.steps)}
        results = {}
        for name, data in (("rgb", rgb), ("rgba", rgba_buf.getvalue()), ("palette", pal_buf.getvalue())):
            status, payload, wall = post_multipart(edit_url, fields, [("image", f"ref_{name}.png", data)])
            imgs = images_of(payload)
            results[name] = imgs[0] if imgs else None
            if imgs:
                (out / f"edit_{name}.png").write_bytes(imgs[0])
            check(f"edit upload {name}", status == 200 and bool(imgs), f"status {status}, {wall * 1000:.0f} ms")
        if results.get("rgb") and results.get("rgba"):
            check("edit rgba equals rgb", psnr(results["rgb"], results["rgba"]) == math.inf,
                  f"PSNR {psnr(results['rgb'], results['rgba'])}")
        if results.get("rgb") and results.get("palette"):
            v = psnr(results["rgb"], results["palette"])
            check("edit palette vs rgb (quantized reference, expected to differ)", not math.isnan(v),
                  f"PSNR {v:.2f} dB")
        status, payload, wall = post_multipart(edit_url, {**fields, "size": "768x1024"}, [("image", "ref.png", rgb)])
        imgs = images_of(payload)
        size = Image.open(io.BytesIO(imgs[0])).size if imgs else None
        check("edit with size 768x1024", status == 200 and size == (768, 1024),
              f"status {status}, output {size}, {wall * 1000:.0f} ms")
        status, payload, wall = post_multipart(edit_url, {**fields, "size": "1000x1000"}, [("image", "ref.png", rgb)],
                                               timeout=120)
        check("edit error path: size 1000x1000", 400 <= status < 500,
              f"status {status}: {str((payload or {}).get('error', payload))[:120]}")

    verdict = "PASS" if not failures else "FAIL"
    print(f"{verdict}: {len(report) - len(failures)} ok, {len(failures)} failing: {failures}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
