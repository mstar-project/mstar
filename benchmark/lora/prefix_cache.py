"""Adapter-keyed prefix caching, served: hits only within an adapter's namespace.

GPU only; nothing here runs in CI. Needs a model that both declares a prefix-cache
stream (``prefix_key_streams``) and serves LoRA adapters (``get_lora_specs``). Orpheus
does both (speech out, through ``/generate`` with the adapter in ``model_kwargs``); a
text model goes through the chat endpoint (``--endpoint chat``). Qwen3.5 serves adapters
but keys no stream (its gated DeltaNet state cannot skip a cached prefix).

One server, one long prompt, sent one request at a time in this order::

    base, a, b, a, base, b

with greedy decoding. Each request's ``KV <cache>: <id>/<label> matched N of its M pages``
line in the server log says how much it was served from the cache. Pass when:

* the first request in each namespace (``base``, ``a``, ``b``) matched 0 pages, so no
  namespace was served another's KV;
* every repeat matched more than 0, so the cache was on and the misses above mean
  something;
* every output equals the same request sent with ``prefix_cache: false`` (per request,
  in ``extra_body``) on the same server: the cached KV is the KV the request would have
  computed.

    python -m benchmark.lora.prefix_cache --model orpheus --config configs/orpheus_colocated.yaml \\
        --adapter a=<Tachyeon/orpheus-3b-sft-lora>/adapter_model.safetensors \\
        --adapter b=<derekurban2001/orpheus-jarvis-voice-lora>/adapter_model.safetensors \\
        --max-rank 64 --out ~/lora_prefix
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ORDER = ("base", "a", "b", "a", "base", "b")
MATCHED = re.compile(r"KV \S+: (\S+)/\S+ matched (\d+) of its (\d+) pages")
# long enough to fill several pages at any page size the configs use
PROMPT = " ".join(
    f"Paragraph {i}: the quick brown fox jumps over the lazy dog while the archivist "
    f"records entry {i * 7} in the ledger." for i in range(60)
) + " Summarize the ledger in one sentence."


def _preflight(model: str) -> None:
    from mstar.model.base import Model
    from mstar.model.registry import get_model_class

    cls = get_model_class(model)
    missing = [
        name for name in ("prefix_key_streams", "get_lora_specs")
        if getattr(cls, name) is getattr(Model, name)
    ]
    if missing:
        raise SystemExit(
            f"{model}: {cls.__name__} does not override {missing}; this check needs a model "
            "that keys a prefix-cache stream and serves LoRA adapters"
        )


def _serving_config(args, workdir: Path) -> Path:
    stock = Path(args.config) if args.config else (
        Path(__file__).resolve().parents[2] / "configs" / f"{args.model}.yaml"
    )
    config = yaml.safe_load(stock.read_text())
    config.setdefault("resources", {})["lora"] = {
        "max_slots": 2, "max_rank": args.max_rank, "adapters": dict(args.adapters),
    }
    config.pop("warmup_requests", None)
    path = workdir / f"{args.model}_lora_prefix.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _chat(url: str, model: str, max_tokens: int, cached: bool) -> dict:
    body = {
        "model": model, "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0, "max_tokens": max_tokens,
    }
    if not cached:
        body["prefix_cache"] = False
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{model}: HTTP {e.code}: {e.read()[:500]!r}") from e
    return {"id": out["id"], "text": out["choices"][0]["message"]["content"]}


def _speak(url: str, adapter: str | None, max_tokens: int, cached: bool, rid: str) -> dict:
    """``/generate`` for a speech model: greedy, the adapter in ``model_kwargs``; the
    output is the audio bytes, which a request served from the cache must reproduce."""
    import urllib.parse

    kwargs = {"voice": "tara", "temperature": 0.0, "max_tokens": max_tokens}
    if adapter is not None:
        kwargs["lora_adapter"] = adapter
    if not cached:
        kwargs["prefix_cache"] = False
    body = urllib.parse.urlencode({
        "text": PROMPT, "output_modalities": "audio", "model_kwargs": json.dumps(kwargs),
        "request_id": rid,
    }).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{url}/generate", data=body), timeout=600) as r:
            audio = b"".join(
                msg.get("data", "").encode() for msg in map(json.loads, filter(None, r.read().splitlines()))
                if msg.get("modality") == "audio"
            )
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{adapter}: HTTP {e.code}: {e.read()[:500]!r}") from e
    return {"id": rid, "text": audio.decode()}


def _ask(args, url: str, ns: str, names: dict, cached: bool, n: int) -> dict:
    if args.endpoint == "chat":
        return _chat(url, names[ns], args.max_tokens, cached)
    return _speak(url, None if ns == "base" else ns, args.max_tokens, cached, f"prefix-{ns}-{n}")


def run(args) -> int:
    from benchmark.eval.server import ManagedServer

    _preflight(args.model)
    log = args.out / "server.log"
    names = {"base": args.model, **{k: k for k in args.adapters}}
    rows = []
    with tempfile.TemporaryDirectory(prefix="lora-prefix-") as tmp:
        config = _serving_config(args, Path(tmp))
        with ManagedServer(args.model, config=str(config), log_path=log) as server:
            # uncached first: they leave nothing in the index (an opted-out request
            # neither reads nor files pages), so the cached sequence starts cold
            reference = {
                ns: _ask(args, server.url, ns, names, False, -1)["text"]
                for ns in dict.fromkeys(ORDER)
            }
            for n, ns in enumerate(ORDER):
                rows.append({"namespace": ns, **_ask(args, server.url, ns, names, True, n)})
    matched = {m.group(1): (int(m.group(2)), int(m.group(3)))
               for m in MATCHED.finditer(log.read_text(errors="replace"))}
    failures, seen = [], set()
    for row in rows:
        pages, total = matched.get(row["id"], (None, None))
        row.update(matched_pages=pages, pages=total, same_as_uncached=row["text"] == reference[row["namespace"]])
        if pages is None:
            failures.append(f"{row['id']} ({row['namespace']}): no 'matched' line; was the cache open?")
        elif row["namespace"] not in seen and pages:
            failures.append(f"{row['namespace']}: first request matched {pages} pages of another namespace")
        elif row["namespace"] in seen and not pages:
            failures.append(f"{row['namespace']}: a repeat matched nothing; the cache did not serve it")
        if not row["same_as_uncached"]:
            failures.append(f"{row['namespace']} ({row['id']}): output differs from the uncached run")
        seen.add(row["namespace"])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "prefix_cache.json").write_text(json.dumps(
        {"rows": rows, "reference": reference, "failures": failures}, indent=1,
    ))
    for row in rows:
        print(f"{row['namespace']:>5} {row['id']}: matched {row['matched_pages']}/{row['pages']} "
              f"same_as_uncached={row['same_as_uncached']}")
    for failure in failures:
        print("FAIL", failure)
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="a registered mstar model name")
    p.add_argument("--config", help="serving config to start from (default configs/<model>.yaml)")
    p.add_argument("--adapter", action="append", required=True, metavar="a|b=PATH",
                   help="exactly two: a=... and b=..., safetensors adapter files")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--max-rank", type=int, default=16)
    p.add_argument("--endpoint", choices=("chat", "speech"), default=None,
                   help="chat completions, or /generate with audio out (default: speech for orpheus)")
    args = p.parse_args()
    args.endpoint = args.endpoint or ("speech" if args.model == "orpheus" else "chat")
    args.adapters = dict(a.split("=", 1) for a in args.adapter)
    if set(args.adapters) != {"a", "b"}:
        p.error("pass --adapter a=PATH and --adapter b=PATH")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
