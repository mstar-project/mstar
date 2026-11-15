#!/usr/bin/env python3
"""Ripple-65B's GPU correctness gate: drive a running server, then judge it against the oracle.

Against ``mstar serve ripple_65b``; every subcommand exits nonzero on any failed check.

    prompts   the eight parity prompts, with the ids mstar's own chat path renders
    collect   greedy replies to them through /v1/chat/completions, saved under a tag
    load      the ~4.4K prompt while at least four other requests decode
    gate      the parity criteria from record_oracle.py's output

The gate pairs its main tag with a second collect on the same server (``<main>b``), the other TP
degree, and ``eager`` if one was collected (a tree whose LLM declares no captures). The oracle's
``--control-tag`` must be the gate's main tag.

    python test/ripple/run_parity.py prompts --ckpt <snapshot> --out out
    python test/ripple/run_parity.py collect --port 8000 --tag tp2 --out out    # TP2 server
    python test/ripple/run_parity.py collect --port 8000 --tag tp2b --out out
    python test/ripple/run_parity.py collect --port 8001 --tag tp4 --out out    # TP4 server
    python test/ripple/record_oracle.py ... --runs out/runs --control-tag tp2 --out out/oracle_bf16.json
    python test/ripple/run_parity.py gate --out out --tags tp2,tp4
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

MODEL = "ripple_65b"
MAX_TOKENS = 128
TOP_K = 5
# the gate's top-1/top-2 logit gap. The largest flip seen: 0.19 at TP2, 0.125 at TP4, TP8, eager and beside
# decoding streams; the reference against itself, under 0.03
NEAR_TIE = 0.25
WINDOW, CHUNK = 1024, 2048
REPO = Path(__file__).resolve().parents[2]

NEEDLE = "Before anything else: the vault code for this document is 7391-ALPHA.\n\n"
ASK = ("\n\nFirst state the vault code given at the very start of this document, then say in one sentence "
       "what the text above is.")
BACKGROUND = [
    "Write a detailed essay on how bridges are designed, of at least 600 words.",
    "Write a long story about a lighthouse keeper and a storm, of at least 600 words.",
    "Explain in depth how a compiler works, stage by stage, in at least 600 words.",
    "Describe the history of the printing press in at least 600 words.",
    "Explain how vaccines train the immune system, in at least 600 words.",
]


def user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def parity_messages() -> list[list[dict]]:
    # the license doesn't change between commits, so neither do the prompts; twice, for the longest one
    corpus = (REPO / "LICENSE").read_text() * 2
    return [
        user("What is the capital of Australia, and why was it chosen over Sydney and Melbourne?"),
        user("Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."),
        user("A train leaves at 3:15 pm and travels 180 km at 72 km/h. When does it arrive? Show the arithmetic."),
        user("Translate into French and Japanese: 'The library closes early on Fridays.'"),
        [{"role": "system", "content": "You are a terse assistant. Answer in one sentence."},
         *user("Name a prime number greater than 50."),
         {"role": "assistant", "content": "53 is a prime number greater than 50."},
         *user("Name the next one.")],
        user("Summarize in two sentences:\n\n" + corpus[:1400]),
        user(NEEDLE + corpus[:7100] + ASK),   # past the 1024 window
        user(NEEDLE + corpus[:21600] + ASK),  # past the 2048 chunk budget
    ]


def post(port: int, body: dict, timeout: float = 1800):
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        json.dumps({"model": MODEL, **body}).encode(), {"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def greedy(max_tokens: int = MAX_TOKENS) -> dict:
    return {"temperature": 0.0, "max_tokens": max_tokens}


def chat(port: int, messages: list[dict], **body) -> str:
    with post(port, {"messages": messages, **greedy(), **body}) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def stream(port: int, messages: list[dict], on_delta=None, **body) -> str:
    text = []
    with post(port, {"messages": messages, "stream": True, **greedy(), **body}) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            if "error" in event:
                raise RuntimeError(event["error"])
            delta = event["choices"][0].get("delta", {}).get("content")
            if delta:
                text.append(delta)
                if on_delta:
                    on_delta()
    return "".join(text)


def rows_path(out: Path, tag: str) -> Path:
    (out / "runs").mkdir(parents=True, exist_ok=True)
    return out / "runs" / f"{tag}.json"


def load_prompts(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "prompts.jsonl").read_text().splitlines()]


def row(prompt: dict, text: str, max_tokens: int = MAX_TOKENS) -> dict:
    return {**prompt, "text": text, "max_tokens": max_tokens}


def cmd_prompts(a) -> int:
    from mstar.api_server.openai.adapters import ADAPTER_REGISTRY
    from mstar.api_server.openai.protocol import ChatCompletionRequest
    from mstar.model.ripple.ripple_model import RippleModel

    model = RippleModel(a.ckpt)
    adapter = ADAPTER_REGISTRY[MODEL]

    def ids(messages: list[dict]) -> list[int]:
        # the server's own path: the adapter's SubmitArgs into process_prompt
        args = adapter.chat_to_request(ChatCompletionRequest(messages=messages, model=MODEL), Path("/tmp"))
        prompt = model.process_prompt(args.text, args.input_modalities, args.output_modalities,
                                      prompt_parts=args.prompt_parts, **(args.model_kwargs or {}))
        return prompt.new_input_tensors["text_inputs"][0].tolist()

    a.out.mkdir(parents=True, exist_ok=True)
    prompts = [{"messages": m, "prompt_ids": ids(m)} for m in parity_messages()]
    background = [{"messages": user(t), "prompt_ids": ids(user(t))} for t in BACKGROUND]
    (a.out / "prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
    (a.out / "background.jsonl").write_text("".join(json.dumps(p) + "\n" for p in background))
    lengths = [len(p["prompt_ids"]) for p in prompts]
    print("prompt tokens:", lengths)
    return 0 if 1.2 * WINDOW < lengths[-2] < CHUNK and lengths[-1] > 2 * CHUNK else 1


def cmd_collect(a) -> int:
    rows = []
    for p in load_prompts(a.out):
        text = chat(a.port, p["messages"])
        rows.append(row(p, text))
        print(f"[{a.tag}] {len(p['prompt_ids'])} tokens: {text[:100]!r}", flush=True)
    rows_path(a.out, a.tag).write_text(json.dumps(rows, indent=1))
    return 0 if all(r["text"] for r in rows) else 1


def cmd_load(a) -> int:
    """The chunked prompt must really run beside live decode rows: every background stream is
    going when it is sent, and at least four keep producing tokens during its prefill."""
    prompts = [json.loads(line) for line in (a.out / "background.jsonl").read_text().splitlines()]
    long_prompt = load_prompts(a.out)[-1]
    stamps: list[list[float]] = [[] for _ in prompts]
    done: list[float] = [0.0] * len(prompts)
    texts: list[str] = [""] * len(prompts)

    def background(i: int) -> None:
        texts[i] = stream(a.port, prompts[i]["messages"], lambda: stamps[i].append(time.time()), max_tokens=a.bg_tokens)
        done[i] = time.time()

    threads = [threading.Thread(target=background, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    while min(map(len, stamps)) < 16 and all(t.is_alive() for t in threads):
        time.sleep(0.05)
    t0, first = time.time(), []
    text = stream(a.port, long_prompt["messages"], lambda: first.append(time.time()))
    for t in threads:
        t.join()
    during = [sum(t0 < s < first[0] for s in st) for st in stamps]
    alive = sum(d >= 1 for d, end in zip(during, done, strict=True) if end > first[0])
    print(f"prefill of {len(long_prompt['prompt_ids'])} tokens took {first[0] - t0:.2f}s; "
          f"tokens each background stream emitted during it: {during}")
    rows_path(a.out, a.tag).write_text(json.dumps([row(long_prompt, text)], indent=1))
    rows_path(a.out, f"{a.tag}_bg").write_text(json.dumps(
        [row(p, t, a.bg_tokens) for p, t in zip(prompts, texts, strict=True)], indent=1))
    print(f"reply: {text[:160]!r}")
    return 0 if alive >= 4 and text else 1


def mismatches(entry: dict) -> list[str]:
    """Criterion 2 at one reply: each position's served token is in the oracle's top-5, and a
    position whose argmax differs counts unless it is a near tie."""
    bad = []
    for k, (rank, gap) in enumerate(zip(entry["rank"], entry["gap12"], strict=True)):
        if rank >= TOP_K:
            bad.append(f"pos {k}: rank {rank}")
        elif rank > 0 and gap > NEAR_TIE:
            bad.append(f"pos {k}: rank {rank}, top-2 gap {gap:.2f}")
    return bad


def summary(entries: list[dict]) -> tuple[str, int]:
    positions = sum(len(e["rank"]) for e in entries)
    agree = sum(r == 0 for e in entries for r in e["rank"])
    flips = [g for e in entries for r, g in zip(e["rank"], e["gap12"], strict=True) if r > 0]
    bad = sum(len(mismatches(e)) for e in entries)
    return (f"argmax {agree}/{positions}, max rank {max(max(e['rank'], default=0) for e in entries)}, "
            f"flip gaps <= {max(flips, default=0):.2f}, stops {sum(e['stopped'] for e in entries)}, "
            f"artifacts {sum(e['retokenized'] for e in entries)}, mismatches {bad}"), bad


def pair(a_rows: list[dict], b_rows: list[dict]) -> tuple[str, int]:
    """Two served runs against each other. Each already passes criterion 2 alone; here, the first
    token where they part must be a near tie in the oracle, on the shared prefix. ``served`` ends
    on the stop token of a reply that stopped, so one that stops where the other goes on parts there."""
    same, bad, gaps = 0, 0, []
    for x, y in zip(a_rows, b_rows, strict=True):
        xs, ys = x["served"], y["served"]
        k = next((i for i in range(min(len(xs), len(ys))) if xs[i] != ys[i]), None)
        if k is None:
            same += 1
            continue
        logit = dict(zip(x["top5"][k], x["top5_logits"][k], strict=True))
        if ys[k] not in logit or xs[k] not in logit:
            bad += 1
            continue
        gaps.append(abs(logit[xs[k]] - logit[ys[k]]))
        bad += gaps[-1] > NEAR_TIE
    return f"identical {same}/{len(a_rows)}, gaps where they part {[round(g, 2) for g in gaps]}, bad {bad}", bad


def cmd_gate(a) -> int:
    report = json.loads((a.out / "oracle_bf16.json").read_text())
    base, failed = report["base"], []
    for tag, entries in base.items():
        text, bad = summary(entries)
        print(f"{tag:>10}: {text}")
        for i, e in enumerate(entries):
            for m in mismatches(e)[:4]:
                print(f"            row {i} {m}")
        if bad:
            failed.append(tag)
    main, other = a.tags.split(",")
    lengths = [e["prompt_tokens"] for e in base[main]]
    print("prompt tokens:", lengths)
    if not (lengths[-2] > WINDOW and lengths[-1] > CHUNK):
        failed.append("the long prompts are too short")
    for name in ("no_gate", "no_valres"):
        text, bad = summary(report[name][main])
        print(f"control {name}: {text}")
        if not bad:
            failed.append(f"control {name} did not fail")
    for peer in (f"{main}b", "eager", other):
        if peer not in base:
            # eager needs a tree whose LLM declares no captures; the rest this script collects
            if peer != "eager":
                failed.append(f"{peer} was not collected")
            print(f"pair {main}/{peer}: not collected")
            continue
        text, bad = pair(base[main], base[peer])
        print(f"pair {main}/{peer}: {text}")
        if bad:
            failed.append(f"{main}/{peer}")
    print("FAILED:" if failed else "PASS", failed or "")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    commands = {"prompts": cmd_prompts, "collect": cmd_collect, "load": cmd_load, "gate": cmd_gate}
    for name in commands:
        s = sub.add_parser(name)
        s.add_argument("--out", type=Path, required=True)
        if name not in ("prompts", "gate"):
            s.add_argument("--port", type=int, default=8000)
        if name == "prompts":
            s.add_argument("--ckpt", required=True)
        if name in ("collect", "load"):
            s.add_argument("--tag", required=True)
        if name == "load":
            s.add_argument("--bg-tokens", type=int, default=384)
        if name == "gate":
            s.add_argument("--tags", default="tp2,tp4", help="the gate's main tag and the other TP degree")
    a = ap.parse_args()
    return commands[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
