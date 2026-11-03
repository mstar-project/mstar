"""Checkpoint access, sustained timing and capture for the megakernel benchmarks."""
from __future__ import annotations

import glob
import json
import statistics as st
from functools import cache

import torch


def snapshot(size: str = "0.8B") -> str:
    return glob.glob(f"/root/hf/hub/models--Qwen--Qwen3.5-{size}/snapshots/*/")[0]


@cache
def text_config(size: str = "0.8B") -> dict:
    cfg = json.load(open(snapshot(size) + "config.json"))
    return cfg.get("text_config", cfg)


class Checkpoint:
    """Lazy tensor access to one safetensors snapshot."""

    def __init__(self, size: str = "0.8B"):
        from safetensors import safe_open

        d = snapshot(size)
        idx = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
        self._files = {}
        self._where = idx
        self._dir = d
        self._open = safe_open

    def get(self, key: str, device="cuda", dtype=None) -> torch.Tensor:
        f = self._where[key]
        if f not in self._files:
            self._files[f] = self._open(self._dir + f, framework="pt", device="cpu")
        t = self._files[f].get_tensor(key)
        return t.to(device=device, dtype=dtype or t.dtype)

    def layer(self, i: int, name: str, **kw) -> torch.Tensor:
        return self.get(f"model.language_model.layers.{i}.{name}", **kw)


def timed_us(fn, seconds: float = 1.0, inner: int = 5) -> float:
    """Back-to-back calls for about ``seconds``; median of the second half (clocks settle)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out, total = [], 0.0
    while total < seconds * 1000:
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        e.synchronize()
        ms = s.elapsed_time(e)
        total += ms
        out.append(ms / inner * 1000)
    return st.median(out[len(out) // 2:])


def capture(fn, warm: int = 3) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g
