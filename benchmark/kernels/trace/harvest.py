"""Harvest the shapes a kernel spec runs from what a served model computes.

    python -m benchmark.kernels.trace.harvest --model NAME [--hf-config config.json --rows 1 2 4 ...]
        [--server-log server.log ...] [--trace trace.json ...] --out shapes/NAME.json

Three sources, merged with a count per shape:

- ``--hf-config``: a Qwen3.5 text config (``model_type`` qwen3_5*), from which
  the decode step's linears and norms follow for each of ``--rows``;
- ``--server-log``: the "skinny GEMM: rows=.. k=.. n=.." lines the ROCm GEMM
  pass logs, one per compiled decode linear;
- ``--trace``: a chrome trace taken with ``KTRACE_SHAPES=1`` (``proftrigger``),
  whose CPU ops carry their input shapes (eager steps only: a replayed graph
  launches kernels without them).

The output is ``{"model", "sources", "ops": {op: [{dim: value, ..., "count": n}]}}``,
read by ``spec.load_shapes``.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from collections.abc import Iterable

_SKINNY = re.compile(r"skinny GEMM: rows=(\d+) k=(\d+) n=(\d+)")


def qwen3_5_decode(config: dict, rows: Iterable[int]) -> dict[str, collections.Counter]:
    """Linears and RMSNorms of one Qwen3.5 decode step at each row count:
    the gated-delta layers' fused input and output projections, attention's
    fused q(+gate)/k/v and o, the MLP's fused gate/up and down, the vocab head;
    two norms per layer and the final one, and per-head q/k norms."""
    text = config.get("text_config", config)
    if not str(text.get("model_type", "")).startswith("qwen3_5"):
        raise ValueError(f"not a Qwen3.5 text config: {text.get('model_type')!r}")
    hidden, inter, vocab = text["hidden_size"], text["intermediate_size"], text["vocab_size"]
    heads, kv_heads, head_dim = text["num_attention_heads"], text["num_key_value_heads"], text["head_dim"]
    v_heads = text["linear_num_value_heads"]
    key = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value = v_heads * text["linear_value_head_dim"]
    layers = text["layer_types"]
    gdn, attn = layers.count("linear_attention"), layers.count("full_attention")
    q_out = heads * head_dim * (2 if text.get("attn_output_gate") else 1)
    linears = [
        # the fused [q|k|v|z|a|pad|b] projection; the pad aligns b to 32 bytes
        (2 * key + 2 * value + 2 * v_heads + (-v_heads % 16), hidden, gdn),
        (hidden, value, gdn),
        (q_out + 2 * kv_heads * head_dim, hidden, attn),
        (hidden, heads * head_dim, attn),
        (2 * inter, hidden, len(layers)),
        (hidden, inter, len(layers)),
        (vocab, hidden, 1),
    ]
    ops = {"linear": collections.Counter(), "rms_norm": collections.Counter()}
    for r in rows:
        for n, k, count in linears:
            ops["linear"][(("rows", r), ("n", n), ("k", k))] += count
        ops["rms_norm"][(("rows", r), ("hidden", hidden))] += 2 * len(layers) + 1
        ops["rms_norm"][(("rows", r * heads), ("hidden", head_dim))] += attn
        ops["rms_norm"][(("rows", r * kv_heads), ("hidden", head_dim))] += attn
    return ops


def from_server_log(text: str) -> dict[str, collections.Counter]:
    linear = collections.Counter()
    for match in _SKINNY.finditer(text):
        rows, k, n = (int(g) for g in match.groups())
        linear[(("rows", rows), ("n", n), ("k", k))] += 1
    return {"linear": linear}


def _prod(dims) -> int:
    return math.prod(dims) if dims else 1


def from_trace(events: list[dict]) -> dict[str, collections.Counter]:
    """Shapes of the linears and RMSNorms among a trace's CPU ops."""
    ops = {"linear": collections.Counter(), "rms_norm": collections.Counter()}
    for e in events:
        if e.get("cat") != "cpu_op":
            continue
        dims = e.get("args", {}).get("Input Dims")
        name = e.get("name", "")
        if not dims:
            continue
        try:
            if name == "aten::mm":
                (m, k), (_, n) = dims[0], dims[1]
            elif name == "aten::addmm":
                (m, k), (_, n) = dims[1], dims[2]
            elif name == "aten::linear":
                x, w = dims[0], dims[1]
                m, k, n = _prod(x[:-1]), x[-1], w[0]
            elif name == "mstar::rms_norm":
                x = dims[0]
                ops["rms_norm"][(("rows", _prod(x[:-1])), ("hidden", x[-1]))] += 1
                continue
            else:
                continue
        except (ValueError, IndexError, TypeError):
            continue
        ops["linear"][(("rows", m), ("n", n), ("k", k))] += 1
    return ops


def describe(path: str) -> str:
    """A source as provenance without the host's paths: ``Qwen/Qwen3.5-9B@<sha>``
    for a Hugging Face snapshot, else the last three path components."""
    match = re.search(r"models--([^/]+)--([^/]+)/snapshots/([0-9a-f]+)/", path)
    if match:
        return f"{match.group(1)}/{match.group(2)}@{match.group(3)[:12]}"
    return "/".join(path.split("/")[-3:])


def merge(*harvests: dict[str, collections.Counter]) -> dict[str, list[dict]]:
    merged: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for harvest in harvests:
        for op, counter in harvest.items():
            merged[op].update(counter)
    return {op: [{**dict(key), "count": count} for key, count in sorted(counter.items())]
            for op, counter in sorted(merged.items())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--hf-config")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--server-log", nargs="*", default=[])
    parser.add_argument("--trace", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    harvests, sources = [], []
    if args.hf_config:
        with open(args.hf_config) as f:
            harvests.append(qwen3_5_decode(json.load(f), args.rows))
        sources.append({"hf_config": describe(args.hf_config), "rows": args.rows})
    for path in args.server_log:
        with open(path, errors="replace") as f:
            harvests.append(from_server_log(f.read()))
        sources.append({"server_log": describe(path)})
    for path in args.trace:
        with open(path) as f:
            harvests.append(from_trace(json.load(f).get("traceEvents", [])))
        sources.append({"trace": describe(path)})
    if not harvests:
        parser.error("name at least one source")
    out = {"model": args.model, "sources": sources, "ops": merge(*harvests)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"{args.out}: " + ", ".join(f"{op} {len(shapes)} shapes" for op, shapes in out["ops"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
