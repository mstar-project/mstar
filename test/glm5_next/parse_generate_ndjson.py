"""Best-effort text + chunk-count extraction from mstar-serve ``/generate``
NDJSON replies saved by the glm53 serve smokes (``gen-*.ndjson`` in a dir).
"""

from __future__ import annotations

import glob
import json
import os
import sys

_TEXT_KEYS = ("text", "delta", "token", "content")


def extract(path: str) -> tuple[str, int]:
    """(text, chunk count); tolerates several chunk shapes: ``{"text":..}`` /
    ``{"delta":..}`` / ``{"outputs":{"text":..}}`` / ``{"token":..}``."""
    txt, n = "", 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if not isinstance(o, dict):
            continue
        for k in _TEXT_KEYS:
            v = o.get(k)
            if isinstance(v, str):
                txt += v
                n += 1
                break
        else:
            outs = o.get("outputs")
            if isinstance(outs, dict) and isinstance(outs.get("text"), str):
                txt += outs["text"]
                n += 1
    return txt, n


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python env/parse_generate_ndjson.py <out_dir> [elapsed_s ...]")
        return 2
    out = argv[1]
    elapsed = [float(x) for x in argv[2:]]
    paths = sorted(glob.glob(os.path.join(out, "gen-*.ndjson")))
    tot_n = tot_t = 0.0
    for i, path in enumerate(paths):
        txt, n = extract(path)
        head = f"=== {os.path.basename(path)} ({n} chunks"
        if i < len(elapsed) and elapsed[i] > 0:
            head += f", {elapsed[i]:.2f}s, {n / elapsed[i]:.1f} chunks/s"
            tot_n += n
            tot_t += elapsed[i]
        print(f"\n{head}) ===")
        if txt.strip():
            print(txt.strip()[:500])
        else:
            print("(parser found no text -- raw head:)")
            with open(path, errors="replace") as fh:
                print(fh.read()[:600])
    if tot_t > 0:
        print(f"\nAGGREGATE: {tot_n:.0f} chunks in {tot_t:.2f}s = {tot_n / tot_t:.1f} chunks/s "
              "(one chunk per token when streaming=true)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
