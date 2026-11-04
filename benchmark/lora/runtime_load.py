"""Loading and unloading LoRA adapters while Qwen3.5 serves base-model traffic.

GPU only; nothing here runs in CI. Needs the ``reference`` phase of
``benchmark.lora.qwen3_5_parity`` in the same ``--out`` (``reference.json``, HF + PEFT
greedy outputs with their top-k), produced with the same ``--max-tokens``.

One server, started with a ``resources: lora:`` block that names no adapter, and
``--base-concurrency`` closed-loop base-model requests running from start to end. While
they run:

1. ``load``: two adapters loaded with ``POST /v1/load_lora_adapter`` (vLLM's body,
   ``lora_path`` the PEFT directory). Each must answer 200 and appear in ``/v1/models``;
   loading one again must answer 400.
2. ``parity``: every prompt of the parity script on each adapter, checked against the
   reference by vLLM's greedy rule (``qwen3_5_parity._agree``), and a control that each
   adapter changes at least one output against the base model.
3. ``unload``: ``--held`` long requests (``ignore_eos``) started on the first adapter,
   then, once they are running, ``POST /v1/unload_lora_adapter``. The unload must answer
   200, a new request on the adapter must answer 400, and every held request must finish
   with its full length (they run on the old weights to the end).
4. ``reload``: the same adapter loaded again from the same directory (retried while the
   workers still hold it for the requests that just finished: a 400 saying so), then the
   parity prompts again, which must equal the first load's outputs token for token.

Pass (exit 0): every check above, and ``failed == 0`` for the base traffic. Results:
``runtime_load.json`` and ``server_runtime_load.log`` under ``--out`` (suffixed
``_tp<N>_async<0|1>`` under ``--tp``).

Qwen3.5 keys no prefix-cache stream, so no adapter gets a prefix-cache root here; that a
reload of the same weights keeps its root is covered on CPU
(``test_lora_api.py::test_a_runtime_load_and_unload_end_to_end``).

    python -m benchmark.lora.runtime_load --model qwen3_5_0.8b --out ~/lora_parity
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from benchmark.lora.qwen3_5_parity import (
    BASE,
    DEFAULT_ADAPTERS,
    HF_REPOS,
    PROMPTS,
    _adapter_dir,
    _agree,
    _chat,
    _post,
    _write,
)


def _config(model: str, workdir: Path, max_slots: int, tp: int) -> Path:
    stock = Path(__file__).resolve().parents[2] / "configs" / f"{model}.yaml"
    config = yaml.safe_load(stock.read_text())
    if tp > 1:
        # as the parity script's TP phase: the LLM node over ranks 0..tp-1
        config["node_groups"] = [
            {"node_names": ["vision_encoder"], "ranks": [0]},
            {"node_names": ["LLM"], "ranks": list(range(tp)), "tp_size": tp},
        ]
    # LoRA on, no adapter named: every adapter arrives at runtime
    config.setdefault("resources", {})["lora"] = {
        "max_slots": max_slots, "max_rank": 16, "runtime_updates": True,
    }
    config.pop("warmup_requests", None)
    path = workdir / f"{model}_lora_runtime.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _post_text(url: str, body: dict, timeout: float = 600) -> tuple[int, str]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


class _BaseTraffic:
    """``concurrency`` threads sending base-model requests back to back until stopped.
    Each request is logged as (start, end, error) on the monotonic clock."""

    def __init__(self, url: str, model: str, concurrency: int, max_tokens: int):
        self._url, self._model, self._max_tokens = url, model, max_tokens
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.log: list[tuple[float, float, str | None]] = []
        self._threads = [threading.Thread(target=self._run, args=(i,), daemon=True)
                         for i in range(concurrency)]

    def _run(self, i: int) -> None:
        from benchmark.lora.load import PROMPTS as LOAD_PROMPTS
        from benchmark.lora.load import _chat as chat_tokens

        n = 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            error = None
            try:
                chat_tokens(self._url, self._model, LOAD_PROMPTS[(i + n) % len(LOAD_PROMPTS)],
                            self._max_tokens, 300.0)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                error = str(exc)
            with self._lock:
                self.log.append((t0, time.monotonic(), error))
            n += 1

    def __enter__(self):
        for t in self._threads:
            t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=600)

    def summary(self, windows: dict[str, tuple[float, float]]) -> dict:
        lat = sorted(end - start for start, end, err in self.log if err is None)
        errors = [err for _, _, err in self.log if err is not None]

        def pct(xs, p):
            return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

        out = {
            "requests": len(self.log), "failed": len(errors), "errors": errors[:5],
            "latency_p50_s": pct(lat, 0.5), "latency_p99_s": pct(lat, 0.99),
            "latency_max_s": lat[-1] if lat else None,
        }
        # requests in flight during each admin call: the read is off the workers' main
        # loops, so a latency bump here is host contention (worth knowing, not a pass/fail)
        for name, (a, b) in windows.items():
            overlap = sorted(end - start for start, end, err in self.log
                             if err is None and start < b and end > a)
            out[f"during_{name}"] = {"requests": len(overlap), "latency_max_s": overlap[-1] if overlap else None}
        return out


def run(args) -> int:
    from transformers import AutoTokenizer

    from benchmark.eval.server import ManagedServer

    ref = json.loads((args.out / "reference.json").read_text())
    tok = AutoTokenizer.from_pretrained(HF_REPOS[args.model])
    names = list(args.adapters)[:2]
    if len(names) < 2:
        raise SystemExit("needs two adapters")
    dirs = {n: str(_adapter_dir(args.adapters[n])) for n in names}
    first = names[0]
    tag = "" if args.tp == 1 else f"_tp{args.tp}_async{args.async_sched}"
    log = args.out / f"server_runtime_load{tag}.log"
    failed: list[str] = []
    result: dict = {"adapters": dirs}
    windows: dict[str, tuple[float, float]] = {}

    def check(ok: bool, what: str) -> None:
        print(("ok   " if ok else "FAIL ") + what, flush=True)
        if not ok:
            failed.append(what)

    def admin(path: str, body: dict, label: str) -> tuple[int, str]:
        t0 = time.monotonic()
        status, text = _post_text(f"{url}{path}", body)
        windows[label] = (t0, time.monotonic())
        return status, text

    def parity(tag: str) -> dict:
        outputs, rows = {}, {}
        for name in names:
            for i, prompt in enumerate(PROMPTS):
                text = _chat(url, name, prompt, args.max_tokens)
                outputs.setdefault(name, {})[str(i)] = text
                ids = tok(text, add_special_tokens=False).input_ids
                rows[f"{name}/{i}"] = r = _agree(ids, ref[name][str(i)])
                check(r["pass"], f"{tag}: {name}/{i} against the reference (match {r['match']})")
        for name in names:
            check(any(outputs[name][str(i)] != base_out[str(i)] for i in range(len(PROMPTS))),
                  f"{tag}: control, {name} changes some output against the base model")
        return {"outputs": outputs, "rows": rows}

    with tempfile.TemporaryDirectory(prefix="lora-runtime-") as tmp:
        config = _config(args.model, Path(tmp), args.max_slots, args.tp)
        env = {"MSTAR_TP_ASYNC_SCHED": args.async_sched} if args.tp > 1 else {}
        with ManagedServer(args.model, config=str(config), env=env, log_path=log) as server:
            url = server.url
            base_out = {str(i): _chat(url, args.model, p, args.max_tokens) for i, p in enumerate(PROMPTS)}
            with urllib.request.urlopen(f"{url}/v1/models") as r:
                check([m["id"] for m in json.loads(r.read())["data"]] == [args.model],
                      "/v1/models lists the base model alone before any load")
            status, _ = _post(f"{url}/v1/chat/completions", {
                "model": first, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
            })
            check(status == 400, f"a request on {first} before it is loaded: HTTP {status}")

            with _BaseTraffic(url, args.model, args.base_concurrency, args.base_max_tokens) as base:
                time.sleep(args.settle_s)

                # 1. load two adapters mid-traffic
                for name in names:
                    status, text = admin("/v1/load_lora_adapter",
                                         {"lora_name": name, "lora_path": dirs[name]}, f"load_{name}")
                    check(status == 200, f"load {name}: HTTP {status} {text[:200]}")
                with urllib.request.urlopen(f"{url}/v1/models") as r:
                    listed = [(m["id"], m.get("parent")) for m in json.loads(r.read())["data"]]
                check(listed == [(args.model, None)] + [(n, args.model) for n in names],
                      f"/v1/models after the loads: {listed}")
                status, text = admin("/v1/load_lora_adapter",
                                     {"lora_name": first, "lora_path": dirs[first]}, "load_again")
                check(status == 400, f"loading {first} again: HTTP {status} {text[:200]}")

                # 2. parity against HF + PEFT
                result["first"] = parity("first load")

                # 3. unload while requests on it run
                def held(i: int) -> tuple[int, int | None, str]:
                    status, body = _post(f"{url}/v1/chat/completions", {
                        "model": first, "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
                        "temperature": 0, "max_tokens": args.held_tokens, "ignore_eos": True,
                        "enable_thinking": False,
                    })
                    if status != 200:
                        return status, None, str(body)[:200]
                    # the non-streaming usage is not filled in on every engine: count the text
                    choice = body["choices"][0]
                    text = choice["message"]["content"] or ""
                    return status, len(tok(text, add_special_tokens=False).input_ids), choice.get("finish_reason")

                with concurrent.futures.ThreadPoolExecutor(args.held) as pool:
                    futures = [pool.submit(held, i) for i in range(args.held)]
                    time.sleep(args.held_lead_s)
                    status, text = admin("/v1/unload_lora_adapter", {"lora_name": first}, "unload")
                    check(status == 200, f"unload {first} with {args.held} requests on it: HTTP {status} {text[:200]}")
                    status, _ = _post(f"{url}/v1/chat/completions", {
                        "model": first, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
                    })
                    check(status == 400, f"a new request on {first} after its unload: HTTP {status}")
                    check(not any(f.done() for f in futures),
                          "the held requests were still running when the unload answered")
                    held_results = [f.result() for f in futures]
                result["held"] = held_results
                # re-tokenized text can merge or split a few tokens at the boundaries
                check(all(s == 200 and n is not None and n >= 0.9 * args.held_tokens and why == "length"
                          for s, n, why in held_results),
                      f"every held request ran to its {args.held_tokens} tokens (re-tokenized >= 90%, "
                      f"finish_reason length): {[(s, n, why) for s, n, why in held_results]}")
                status, _ = admin("/v1/unload_lora_adapter", {"lora_name": first}, "unload_again")
                check(status == 404, f"unloading {first} again: HTTP {status}")
                with urllib.request.urlopen(f"{url}/v1/models") as r:
                    listed = [m["id"] for m in json.loads(r.read())["data"]]
                check(first not in listed, f"/v1/models after the unload: {listed}")

                # 4. reload the same weights
                deadline = time.monotonic() + 60
                while True:
                    status, text = admin("/v1/load_lora_adapter",
                                         {"lora_name": first, "lora_path": dirs[first]}, "reload")
                    if status == 200 or time.monotonic() > deadline or "still running" not in text:
                        break
                    time.sleep(1)
                check(status == 200, f"reload {first}: HTTP {status} {text[:200]}")
                result["reload"] = parity("reload")
                same = [k for k in result["first"]["outputs"][first]
                        if result["first"]["outputs"][first][k] == result["reload"]["outputs"][first][k]]
                check(len(same) == len(PROMPTS),
                      f"{first} reloaded from the same weights: {len(same)}/{len(PROMPTS)} outputs identical")
            result["base_traffic"] = base.summary(windows)
    b = result["base_traffic"]
    check(b["failed"] == 0 and b["requests"] > 0,
          f"base traffic: {b['requests']} requests, {b['failed']} failed {b['errors']}")
    text = log.read_text(errors="replace")
    result["log_lines"] = [line for line in text.splitlines() if "LoRA" in line][-200:]
    result["failed"] = failed
    _write(args.out / f"runtime_load{tag}.json", result)
    print(f"base traffic p99 {b['latency_p99_s']} s, max {b['latency_max_s']} s; "
          f"during loads: {[b.get(f'during_load_{n}') for n in names]}")
    print(f"{len(failed)} failure(s); results in {args.out / f'runtime_load{tag}.json'}")
    return 1 if failed else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3_5_0.8b", choices=sorted(HF_REPOS))
    p.add_argument("--adapter", action="append", default=[], metavar="NAME=REPO_OR_DIR",
                   help="the first two are used; default: the parity script's")
    p.add_argument("--out", type=Path, required=True, help="holds reference.json")
    p.add_argument("--max-tokens", type=int, default=48, help="as the reference was generated")
    p.add_argument("--max-slots", type=int, default=2)
    p.add_argument("--base-concurrency", type=int, default=32)
    p.add_argument("--base-max-tokens", type=int, default=128)
    p.add_argument("--held", type=int, default=8, help="requests on the adapter when it is unloaded")
    # long enough that the held requests are still decoding when the unload answers
    p.add_argument("--held-tokens", type=int, default=4096)
    p.add_argument("--held-lead-s", type=float, default=1.0,
                   help="how long the held requests run before the unload")
    p.add_argument("--settle-s", type=float, default=10.0,
                   help="base traffic alone before the first load")
    p.add_argument("--tp", type=int, default=1, help="the LLM node's TP size (that many GPUs)")
    p.add_argument("--async-sched", default="0", help="MSTAR_TP_ASYNC_SCHED under --tp")
    args = p.parse_args()
    args.adapters = dict(a.split("=", 1) for a in args.adapter) or DEFAULT_ADAPTERS.get(args.model)
    if not args.adapters:
        p.error(f"no default adapters for {args.model}; pass --adapter name=repo")
    assert BASE not in args.adapters
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
