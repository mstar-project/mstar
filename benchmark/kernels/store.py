"""The results store: one JSON record per (kernel, shape, dtype), keyed by
what decides a kernel's speed.

    <root>/<arch>/<product>/<stack>/<kernel>/<shape>-<dtype>.json

``stack`` is the torch / HIP or CUDA / Triton versions, since a kernel's
timings and code generation change with them, and products of one arch are
kept apart (MI300X and MI325X share code objects, not timings). A record
holds every candidate measured for that shape, newest run per candidate, with
its timing, its percent of the roofline floor, its oracle verdict, its ISA
statistics and the job and commit that produced it. It is the search's
memory: earlier winners seed the next search, and ``promote`` writes the
winners per shape class as a shipped config.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from mstar.kernels.configs import CONFIG_ROOT, config_path
from mstar.platforms.facts import DeviceFacts

DEFAULT_ROOT = Path(__file__).resolve().parent / "results"


def stack_key() -> str:
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = "none"
    toolkit = f"hip-{torch.version.hip}" if torch.version.hip else f"cuda-{torch.version.cuda}"
    return f"torch-{torch.__version__}-{toolkit}-triton-{triton_version}".replace("+", "_")


def shape_key(shape: Mapping[str, int], dtype: torch.dtype) -> str:
    dims = "-".join(f"{k}{v}" for k, v in shape.items())
    return f"{dims}-{str(dtype).removeprefix('torch.')}"


class ResultsStore:
    def __init__(self, root: str | os.PathLike = DEFAULT_ROOT, stack: str | None = None):
        self.root = Path(root)
        self.stack = stack or stack_key()

    def kernel_dir(self, facts: DeviceFacts, kernel: str) -> Path:
        return self.root / (facts.arch or "unknown") / (facts.product or "unknown") / self.stack / kernel

    def path(self, facts: DeviceFacts, kernel: str, shape: Mapping[str, int], dtype: torch.dtype) -> Path:
        return self.kernel_dir(facts, kernel) / f"{shape_key(shape, dtype)}.json"

    def read(self, facts, kernel, shape, dtype) -> dict[str, Any] | None:
        try:
            with open(self.path(facts, kernel, shape, dtype)) as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def write(self, facts: DeviceFacts, kernel: str, shape, dtype, record: dict[str, Any]) -> Path:
        """Merge ``record`` into the stored one: its candidates replace theirs
        by name, the rest are kept; ``best`` is recomputed."""
        path = self.path(facts, kernel, shape, dtype)
        stored = self.read(facts, kernel, shape, dtype) or {}
        candidates = {**stored.get("candidates", {}), **record.get("candidates", {})}
        merged = {**stored, **record, "candidates": candidates, "best": best_name(candidates)}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(merged, indent=1, sort_keys=True, default=str))
        os.replace(tmp, path)
        return path

    def records(self, facts: DeviceFacts, kernel: str) -> list[dict[str, Any]]:
        directory = self.kernel_dir(facts, kernel)
        if not directory.is_dir():
            return []
        return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]

    def seeds(self, facts: DeviceFacts, kernel: str, family: str) -> list[dict[str, Any]]:
        """Knobs of this family's winners on any stored shape, most frequent first."""
        counts: dict[str, tuple[int, dict]] = {}
        for record in self.records(facts, kernel):
            for name, cand in record.get("candidates", {}).items():
                if cand.get("family") == family and name == record.get("best"):
                    key = json.dumps(cand.get("knobs", {}), sort_keys=True)
                    counts[key] = (counts.get(key, (0, {}))[0] + 1, cand.get("knobs", {}))
        return [knobs for _, knobs in sorted(counts.values(), key=lambda item: -item[0])]

    def promote(
        self,
        facts: DeviceFacts,
        kernel: str,
        family: str,
        shape_class: Callable[[Mapping[str, int]], str | None],
        root: Path | None = None,
        per_product: bool = False,
        provenance: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write ``family``'s winning knobs per shape class as ``kernel``'s
        config for this arch (or product): in each class, the knobs that won
        the most shapes. Only after a model-level A/B says the family should
        ship."""
        votes: dict[str, dict[str, tuple[int, dict]]] = {}
        for record in self.records(facts, kernel):
            passing = {n: c for n, c in record["candidates"].items() if c.get("family") == family and _ok(c)}
            if not passing:
                continue
            winner = passing[best_name(passing)]
            klass = shape_class(record["shape"]) or "default"
            key = json.dumps(winner.get("knobs", {}), sort_keys=True)
            count, knobs = votes.setdefault(klass, {}).get(key, (0, winner.get("knobs", {})))
            votes[klass][key] = (count + 1, knobs)
        if not votes:
            raise ValueError(f"no passing {family} results for {kernel} on {facts.arch}/{facts.product}")
        classes = {k: max(v.values(), key=lambda item: item[0])[1] for k, v in votes.items()}
        config = {
            "kernel": kernel, "arch": facts.arch, "product": facts.product if per_product else None,
            "classes": classes, "provenance": {"stack": self.stack, "family": family, **(provenance or {})},
        }
        path = config_path(kernel, facts.arch, facts.product if per_product else None, root or CONFIG_ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")
        return path


def _ok(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("oracle", {}).get("ok")) and isinstance(candidate.get("us"), (int, float))


def best_name(candidates: Mapping[str, Mapping[str, Any]]) -> str | None:
    """The fastest candidate that passed the oracle."""
    passing = [(c["us"], name) for name, c in candidates.items() if _ok(c)]
    return min(passing)[1] if passing else None
