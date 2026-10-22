"""Per-request multi-LoRA serving on Qwen3.5-0.8B: mstar against vLLM and SGLang.

GPU only; nothing here runs in CI. One server at a time on ``--gpu`` (default 0), each
engine from its own venv (``--mstar-python``, ``--vllm-python``, ``--sglang-python``).
Stages, each resumable (a finished cell or file is skipped on a rerun):

``correctness``
    HF + PEFT greedy reference (``qwen3_5_parity``'s ``reference`` phase, in a child
    process), then per engine and adapter set (attention + MLP on ``fit``, all-linear on
    ``alllinear``, each with its own reference) one LoRA server whose greedy answers for
    the parity prompts on the base model and on each adapter, one at a time (``single``)
    and all at once (``mixed``), are scored by ``qwen3_5_parity``'s rule (equal up to the first
    difference, where the engine's token is in the reference's top-k), with the control
    that each adapter changes at least one answer. Per-engine pass counts side by side.
``perf``
    ``--rounds`` rounds; each starts every server afresh and rotates the engine order.
    Four server kinds per engine: ``off`` (no LoRA), ``fit`` (LoRA, ``--slots-fit``
    slots, so every adapter is resident), ``churn`` (LoRA, ``--slots-churn`` slots,
    fewer than adapters) and ``alllinear`` (``--slots-fit`` slots, the all-linear
    adapters, which also touch the gated delta-net projections). Scenarios, each at every
    ``--concurrency``:

    ================== ========= ==================================================
    a_base             fit       LoRA on, every request on the base model
    b_one              fit       every request on one adapter
    c_mixed            fit       the adapters round-robin, all resident
    d_churn            churn     the adapters round-robin over fewer slots than adapters
    e_off              off       LoRA off; the anchor every other scenario is divided by
    f_all_linear_one   alllinear every request on one all-linear adapter
    g_all_linear_mixed alllinear two all-linear adapters and the base round-robin
    ================== ========= ==================================================

    (f) and (g) need mstar's LoRA on the delta-net (``lora/gdn`` or later); on a branch
    without it mstar refuses the adapters and they are recorded as not supported there.
    Before anything starts, each adapter's modules are read from its file and checked
    against what each engine applies; an engine that would apply only some of them
    (SGLang has no ``in_proj_a``/``in_proj_b`` target on Qwen3.5) is excluded from the
    cells and correctness rows on that adapter, with the reason.

    Closed loop, streamed, warmup waves discarded, ``ignore_eos`` with a seeded
    per-request output length that is the same on every engine. A cell counts only
    with ``failed == 0``; a request whose output is short of its ``max_tokens`` fails.
``profile``
    mstar only, untimed: one ``MSTAR_PHASE_TIMING`` server per kind, each scenario at
    ``--profile-concurrency`` (1 and 64), rendered with ``benchmark.worker_phases``.

Matched on every engine: bf16 weights, the same slot count (vLLM ``--max-loras S``,
mstar ``max_slots: S``, SGLang ``--max-loras-per-batch S+1`` because SGLang counts the
base model as one of its adapters per batch, but ``S`` on the churn server, whose
scenario has no base rows and so would leave SGLang's base buffer to an adapter), the max
rank of the loaded adapters (their ``r``), every adapter held in host memory, prefix
caching off (mstar's Qwen3.5 has none;
vLLM ``--no-enable-prefix-caching``, SGLang ``--disable-radix-cache``), context length and
max running requests from the mstar config, fp32 GDN state, thinking off in the chat template, greedy, CUDA graphs on.

An engine that fails to start, or rejects LoRA or an adapter, is recorded as not
supported with an excerpt of its error, and the run goes on without it.

    python -m benchmark.lora.compare_engines --out ~/lora_compare \\
        --vllm-python ~/venvs/vllm/bin/python --sglang-python ~/venvs/sglang/bin/python
    python -m benchmark.lora.compare_engines --out ~/lora_compare --stage perf --first-round 2 --rounds 1
    python -m benchmark.lora.compare_engines --out ~/lora_compare --summary
    python -m benchmark.lora.compare_engines --out /tmp/x --dry-run     # the plan, nothing started
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from benchmark.lora.load import PROMPTS as LOAD_PROMPTS
from benchmark.lora.qwen3_5_parity import (
    BASE,
    DEFAULT_ADAPTERS,
    HF_REPOS,
    PROMPTS,
    _adapter_dir,
    _agree,
    _messages,
    _write,
)

REPO = Path(__file__).resolve().parents[2]
ENGINES = ("mstar", "vllm", "sglang")
KINDS = ("off", "fit", "churn", "alllinear")
CORRECTNESS_KINDS = ("fit", "alllinear")
ATTN_MLP = frozenset({"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})
# Hub adapters that also touch the gated delta-net (`linear_attn.*`); the same three as
# `qwen3_5_parity.DELTA_NET_ADAPTERS` on lora/gdn. The first two are PEFT all-linear, rank
# 16; `medvqa` is rank 8 on in_proj_qkv / in_proj_z / out_proj only (correctness only).
ALL_LINEAR_ADAPTERS = {
    "qwen3_5_0.8b": {
        "transaction": "indrapurnayasa/transaction-qwen3.5-0.8b-lora",
        "atemoko": "BurnyCoder/qwen3.5-0.8b-atemokoloporos-positive-expanded",
        "medvqa": "thealper2/qwen3.5-0.8b-slake-medvqa-lora",
    },
}
# a PEFT module -> SGLang's Qwen3.5 LoRA target that holds it (SGLang packs like vLLM)
SGLANG_NAMES = {
    "q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj", "o_proj": "o_proj",
    "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj", "down_proj": "down_proj",
    "in_proj_qkv": "in_proj_qkvz", "in_proj_z": "in_proj_qkvz",
    "in_proj_a": "in_proj_ba", "in_proj_b": "in_proj_ba", "out_proj": "out_proj",
}
# SGLang's Qwen3.5 `supported_lora_modules` (no in_proj_ba); --sglang-lora-modules overrides
SGLANG_QWEN3_5_LORA_MODULES = ("qkv_proj", "o_proj", "out_proj", "in_proj_qkvz", "gate_up_proj",
                               "down_proj", "lm_head")


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str  # the server it runs on
    models: str  # "base" | "one" | "mixed" | "mixed+base"
    what: str


SCENARIOS = (
    Scenario("a_base", "fit", "base", "LoRA on, base-model requests"),
    Scenario("b_one", "fit", "one", "one adapter"),
    Scenario("c_mixed", "fit", "mixed", "adapters round-robin, all resident"),
    Scenario("d_churn", "churn", "mixed", "adapters round-robin, fewer slots than adapters"),
    Scenario("e_off", "off", "base", "LoRA off (anchor)"),
    Scenario("f_all_linear_one", "alllinear", "one", "one all-linear adapter (delta-net too)"),
    Scenario("g_all_linear_mixed", "alllinear", "mixed+base",
             "two all-linear adapters and the base round-robin, all resident"),
)
ANCHOR = "e_off"
# the worker phases the summary quotes (benchmark/worker_phases/README.md)
PROFILE_PHASES = ("iter_total", "worker.postprocess.event_sync", "worker.gpu_thread.exec",
                  "engine.admit", "res.plan.lora", "res.admit.lora")


# --- engines -------------------------------------------------------------------------

def _slots(args, kind: str) -> int | None:
    return {"off": None, "fit": args.slots_fit, "churn": args.slots_churn, "alllinear": args.slots_fit}[kind]


def _kind_dirs(args, kind: str) -> dict[str, Path]:
    """The adapters a server kind loads."""
    return {"off": {}, "fit": args.adapter_dirs, "churn": args.adapter_dirs,
            "alllinear": args.all_linear_dirs}[kind]


def _kind_specs(args, kind: str) -> dict[str, str]:
    return {"fit": args.adapters, "alllinear": args.all_linear_adapters}[kind]


def _scenarios(args) -> list[Scenario]:
    """The scenarios whose server kind has adapters (none for (f)-(g) without defaults),
    narrowed to ``--scenarios`` when given."""
    return [s for s in SCENARIOS if (s.models == "base" or _kind_dirs(args, s.kind))
            and (not args.scenarios or s.name in args.scenarios)]


def _mstar_limits(args) -> tuple[int, int]:
    """Context length and max running requests, from the stock config, for all engines."""
    config = yaml.safe_load((REPO / "configs" / f"{args.model}.yaml").read_text())
    return int(config["max_seq_len"]), int(config["resources"]["gdn_state"]["max_slots"]) - 1


def _mstar_config(args, kind: str) -> Path:
    config = yaml.safe_load((REPO / "configs" / f"{args.model}.yaml").read_text())
    # text-only load; the vision warmup is not what is measured
    config.pop("warmup_requests", None)
    slots = _slots(args, kind)
    if slots is not None:
        config.setdefault("resources", {})["lora"] = {
            "max_slots": slots, "max_rank": args.max_rank,
            "adapters": {n: str(d / "adapter_model.safetensors") for n, d in _kind_dirs(args, kind).items()},
            **{k: yaml.safe_load(v) for k, v in (o.split("=", 1) for o in getattr(args, "mstar_lora_opt", ()))},
        }
    path = args.out / "configs" / f"mstar_{kind}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config))
    return path


def _target_modules(args, kind: str = "fit") -> list[str]:
    """SGLang's ``--lora-target-modules``: the union of the kind's adapters' targets. The
    attention + MLP set must be just that; the all-linear set is mapped to SGLang's names,
    delta-net included, so a module SGLang has no target for is its to refuse."""
    if kind == "alllinear":
        if args.sglang_alllinear_target_modules:
            return args.sglang_alllinear_target_modules
        return sorted({SGLANG_NAMES.get(m, m) for n in args.all_linear_dirs for m in args.coverage[n]["modules"]})
    if args.sglang_target_modules:
        return args.sglang_target_modules
    targets: set[str] = set()
    for name, d in args.adapter_dirs.items():
        found = json.loads((d / "adapter_config.json").read_text()).get("target_modules")
        if not isinstance(found, list):
            raise SystemExit(f"{name}: target_modules is {found!r}; pass --sglang-target-modules")
        targets |= {t.rsplit(".", 1)[-1] for t in found}
    if not targets <= ATTN_MLP:
        raise SystemExit(f"adapters target {sorted(targets - ATTN_MLP)}, not attention + MLP only")
    return sorted(targets)


def _venv_env(python: str, args) -> dict[str, str]:
    """The venv's bin/ first on PATH (FlashInfer's JIT runs its ninja); one GPU; no Hub;
    nothing of this checkout on the import path."""
    bin_dir = str(Path(python).parent)
    return {
        "PATH": os.pathsep.join([bin_dir, os.environ.get("PATH", "")]),
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "HF_HUB_OFFLINE": "1",
        "PYTHONPATH": "",
    }


def _server(args, engine: str, kind: str, log: Path, extra_env: dict[str, str] | None = None):
    from benchmark.eval.server import ManagedServer

    max_len, max_seqs = _mstar_limits(args)
    slots = _slots(args, kind)
    adapters = [f"{n}={d}" for n, d in _kind_dirs(args, kind).items()]
    if engine == "mstar":
        env = {"MSTAR_SHM_ARENA": "1", "MSTAR_REQUIRE_CUDA_GRAPHS": "1", "HF_HUB_OFFLINE": "1",
               **(extra_env or {})}
        return ManagedServer(
            args.model, config=str(_mstar_config(args, kind)), env=env, log_path=log,
            python=args.mstar_python, extra_args=["--gpus", args.gpu],
            startup_timeout_s=args.health_timeout, cwd=REPO,
        )
    if engine == "vllm":
        cmd = [str(Path(args.vllm_python).parent / "vllm"), "serve", args.hf_repo,
               "--host", "127.0.0.1", "--port", "{port}", "--dtype", "bfloat16", "--seed", "0",
               "--max-model-len", str(max_len), "--max-num-seqs", str(max_seqs),
               "--no-enable-prefix-caching"]
        if slots is not None:
            cmd += ["--enable-lora", "--max-loras", str(slots), "--max-lora-rank", str(args.max_rank),
                    # every adapter in host memory, as mstar's pool and SGLang hold them
                    "--max-cpu-loras", str(max(slots, len(adapters))),
                    "--lora-modules", *adapters]
        cmd += args.vllm_arg
        env = _venv_env(args.vllm_python, args)
    elif engine == "sglang":
        cmd = [args.sglang_python, "-m", "sglang.launch_server", "--model-path", args.hf_repo,
               "--host", "127.0.0.1", "--port", "{port}", "--dtype", "bfloat16", "--random-seed", "0",
               "--context-length", str(max_len), "--max-running-requests", str(max_seqs),
               "--disable-radix-cache", "--mamba-ssm-dtype", "float32"]
        if slots is not None:
            # SGLang counts base-model rows as one of the batch's adapters, but evicts the
            # base model's buffer when no base row runs: the churn scenario has none, so
            # slots + 1 buffers would hold one adapter more than the others' slots
            buffers = slots if kind == "churn" else slots + 1
            cmd += ["--enable-lora", "--lora-paths", *adapters, "--max-loras-per-batch", str(buffers),
                    "--max-lora-rank", str(args.max_rank), "--lora-target-modules", *_target_modules(args, kind)]
        cmd += args.sglang_arg
        env = _venv_env(args.sglang_python, args)
    else:
        raise ValueError(engine)
    # cwd outside the checkout, so `python -m` does not put it first on sys.path
    return ManagedServer(engine, command=cmd, env={**env, **(extra_env or {})}, log_path=log,
                         startup_timeout_s=args.health_timeout, cwd=args.out)


def _describe(args, engine: str, kind: str) -> str:
    if engine == "mstar":
        return (f"{args.mstar_python} -m mstar.cli.main serve {args.model} --gpus {args.gpu} "
                f"--config {args.out / 'configs' / f'mstar_{kind}.yaml'}  (slots={_slots(args, kind)})")
    srv = _server(args, engine, kind, args.out / "servers" / "dry.log")
    return " ".join(srv.command())


def wire_model(args, engine: str, label: str) -> str:
    """The OpenAI ``model`` field that puts a request on ``label`` (an adapter or BASE)."""
    if engine == "mstar":
        return args.model if label == BASE else label
    if label == BASE:
        return args.hf_repo
    if engine == "vllm":
        return label
    return args.sglang_adapter_model.format(base=args.hf_repo, adapter=label)


def chat_body(args, engine: str, label: str, prompt: str, max_tokens: int, *, stream: bool,
              ignore_eos: bool) -> dict:
    body = {"model": wire_model(args, engine, label), "messages": _messages(prompt),
            "temperature": 0.0, "max_tokens": max_tokens, "stream": stream}
    if ignore_eos:
        body["ignore_eos"] = True
    if engine == "mstar":
        body["enable_thinking"] = False
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}
        if stream:
            body["stream_options"] = {"include_usage": True}
    return body


# --- support records, GPU, versions ---------------------------------------------------

def _support_path(args, engine: str, kind: str) -> Path:
    return args.out / "support" / f"{engine}_{kind}.json"


def _support(args, engine: str, kind: str) -> dict | None:
    path = _support_path(args, engine, kind)
    return json.loads(path.read_text()) if path.exists() else None


def _excerpt(log: Path, limit: int = 700) -> str:
    """The error lines at the end of a server log, else its last lines."""
    try:
        lines = log.read_text(errors="replace").splitlines()[-300:]
    except OSError:
        return "(no log)"
    hits = [ln for ln in lines if re.search(r"Error|error|not supported|NotImplemented|Traceback|raise ", ln)]
    text = " | ".join(ln.strip() for ln in (hits[-6:] or lines[-6:]))
    return text[-limit:]


def wait_gpu_free(args, timeout: float = 300.0) -> None:
    """One server at a time: refuse to start while anything holds the GPU."""
    deadline = time.monotonic() + timeout
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                              "-i", args.gpu], capture_output=True, text=True, check=True).stdout
        used = int(out.split()[0])
        if used <= 2048:
            return
        if time.monotonic() > deadline:
            raise SystemExit(f"GPU {args.gpu} still holds {used} MiB; not sharing it")
        time.sleep(5)


_PROBE = """
import importlib.metadata as m, json
out = {}
for d in ("mstar-ai", "vllm", "sglang", "sgl-kernel", "torch", "transformers", "flashinfer-python",
          "flashinfer", "triton"):
    try:
        out[d] = m.version(d)
    except m.PackageNotFoundError:
        pass
print(json.dumps(out))
"""


def record_versions(args, engine: str) -> None:
    path = args.out / "versions.json"
    versions = json.loads(path.read_text()) if path.exists() else {}
    if engine in versions:
        return
    python = {"mstar": args.mstar_python, "vllm": args.vllm_python, "sglang": args.sglang_python}[engine]
    probe = subprocess.run([python, "-c", _PROBE], capture_output=True, text=True, check=False)
    info: dict = {"python": python}
    try:
        info["packages"] = json.loads(probe.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        info["packages"] = {"error": probe.stderr[-300:]}
    if engine == "mstar":
        from benchmark.eval.schema import git_info

        info["git"] = git_info(REPO)
    if "gpu" not in versions:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader",
                              "-i", args.gpu], capture_output=True, text=True, check=False).stdout.strip()
        versions["gpu"] = gpu
    versions[engine] = info
    _write(path, versions)


# --- HTTP ----------------------------------------------------------------------------

def _post(url: str, body: dict, timeout: float = 600.0) -> tuple[int, dict | str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:500]
    except (urllib.error.URLError, OSError, ValueError) as e:
        return 0, f"{type(e).__name__}: {e}"


def _answer(args, url: str, engine: str, label: str, prompt: str, max_tokens: int) -> str | dict:
    """Greedy text, or ``{"error": ...}``."""
    status, body = _post(f"{url}/v1/chat/completions",
                         chat_body(args, engine, label, prompt, max_tokens, stream=False, ignore_eos=False))
    if status != 200 or not isinstance(body, dict):
        return {"error": f"HTTP {status}: {body}"}
    msg = body["choices"][0]["message"]
    return msg.get("content") or ""


@dataclass(slots=True)
class Outcome:
    """One streamed request."""

    label: str
    prompt: int
    max_tokens: int
    ok: bool = False
    error: str | None = None
    ttft_s: float | None = None
    latency_s: float | None = None
    text: str = ""
    usage_completion: int | None = None
    usage_prompt: int | None = None
    tokens: int = 0


async def _stream_one(session, url: str, body: dict, out: Outcome) -> None:
    import aiohttp

    t0 = time.perf_counter()
    parts: list[str] = []
    try:
        async with session.post(f"{url}/v1/chat/completions", json=body) as resp:
            if resp.status != 200:
                out.error = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                return
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    out.error = f"in-stream error: {str(event['error'])[:300]}"
                    return
                usage = event.get("usage")
                if usage:
                    out.usage_completion = usage.get("completion_tokens") or None
                    out.usage_prompt = usage.get("prompt_tokens") or None
                for choice in event.get("choices") or ():
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if piece:
                        if out.ttft_s is None:
                            out.ttft_s = time.perf_counter() - t0
                        parts.append(piece)
    except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
        out.error = f"{type(exc).__name__}: {exc}"
        return
    out.latency_s = time.perf_counter() - t0
    out.text = "".join(parts)
    if out.ttft_s is None:
        out.error = "no content streamed"
    else:
        out.ok = True


async def _closed_loop(url: str, bodies: list[dict], outcomes: list[Outcome], concurrency: int,
                       timeout: float) -> float:
    import aiohttp

    pending = iter(range(len(bodies)))
    connector = aiohttp.TCPConnector(limit=concurrency + 8)
    async with aiohttp.ClientSession(connector=connector,
                                     timeout=aiohttp.ClientTimeout(total=timeout)) as session:

        async def worker() -> None:
            for k in pending:  # one event loop: the shared iterator needs no lock
                await _stream_one(session, url, bodies[k], outcomes[k])

        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        return time.perf_counter() - t0


# --- cells -------------------------------------------------------------------------------

def _labels(args, scenario: Scenario) -> list[str]:
    if scenario.models == "base":
        return [BASE]
    adapters = args.all_linear_perf if scenario.kind == "alllinear" else args.perf_adapters
    if scenario.models == "one":
        return [adapters[0]]
    if scenario.models == "mixed+base":
        return [BASE, *adapters]
    return list(adapters)


def _requests(c: int) -> int:
    return 8 if c == 1 else 5 * c


def _jobs(args, labels: list[str], n: int, first: int) -> list[Outcome]:
    """Request ``first + i``: the same prompt, adapter and output length on every engine."""
    out = []
    for i in range(first, first + n):
        rng = random.Random(args.seed * 1_000_003 + i)
        out.append(Outcome(label=labels[i % len(labels)], prompt=i % len(LOAD_PROMPTS),
                           max_tokens=rng.randint(args.output_len_min, args.output_len_max)))
    return out


def _tokenizer(args):
    if getattr(args, "_tok", None) is None:
        from transformers import AutoTokenizer

        args._tok = AutoTokenizer.from_pretrained(args.hf_repo)
    return args._tok


def _prompt_lengths(args) -> list[int]:
    tok = _tokenizer(args)
    return [
        len(tok(tok.apply_chat_template(_messages(p), tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False), add_special_tokens=False).input_ids)
        for p in LOAD_PROMPTS
    ]


def _check_lengths(args, outcomes: list[Outcome]) -> None:
    """Equal work, counted alike on every engine: a request is credited the tokens its
    streamed text re-tokenizes to (mstar streams no usage, so the engines' own counts
    cannot be the measure for all three), and fails if that is under 90% of
    ``max_tokens``. Where an engine reports its count (vLLM, SGLang) it must also be
    exactly ``max_tokens``: a check, not the credit."""
    tok = _tokenizer(args)
    for o in outcomes:
        if not o.ok:
            continue
        if o.usage_completion is not None and o.usage_completion != o.max_tokens:
            o.ok, o.error = False, f"completion_tokens {o.usage_completion} != max_tokens {o.max_tokens}"
            continue
        n = len(tok(o.text, add_special_tokens=False).input_ids)
        if n < 0.9 * o.max_tokens:
            o.ok, o.error = False, f"short output: {n} re-tokenized tokens of max_tokens {o.max_tokens}"
        else:
            o.tokens = n


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def _ms(x: float | None) -> float | None:
    return None if x is None else 1000.0 * x


def _loadavg() -> float | None:
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


def run_cell(args, url: str, engine: str, scenario: Scenario, c: int, warmup_waves: int) -> dict:
    labels = _labels(args, scenario)
    warm = _jobs(args, labels, max(warmup_waves * c, len(labels)), first=1_000_000)
    jobs = _jobs(args, labels, _requests(c), first=0)

    def bodies(js):
        return [chat_body(args, engine, j.label, LOAD_PROMPTS[j.prompt], j.max_tokens, stream=True,
                          ignore_eos=True) for j in js]

    asyncio.run(_closed_loop(url, bodies(warm), warm, c, args.request_timeout))
    load_before = _loadavg()
    wall = asyncio.run(_closed_loop(url, bodies(jobs), jobs, c, args.request_timeout))
    load_after = _loadavg()
    _check_lengths(args, jobs)
    ok = [o for o in jobs if o.ok]
    tokens = sum(o.tokens for o in ok)
    ttft = [o.ttft_s for o in ok]
    tpot = [(o.latency_s - o.ttft_s) / (o.tokens - 1) for o in ok if o.tokens > 1]
    expected = _prompt_lengths(args)
    prompt_mismatch = sum(1 for o in ok if o.usage_prompt is not None and o.usage_prompt != expected[o.prompt])
    errors = [f"{o.label}: {o.error}" for o in jobs if not o.ok]
    return {
        "engine": engine, "scenario": scenario.name, "concurrency": c, "labels": labels,
        "requests": len(jobs), "completed": len(ok), "failed": len(jobs) - len(ok),
        "warmup_requests": len(warm), "warmup_failed": sum(1 for o in warm if not o.ok),
        "wall_s": wall, "output_tokens": tokens,
        "tok_s": tokens / wall if wall else 0.0, "req_s": len(ok) / wall if wall else 0.0,
        "ttft_p50_ms": _ms(_pct(ttft, 0.5)), "ttft_p99_ms": _ms(_pct(ttft, 0.99)),
        "tpot_p50_ms": _ms(_pct(tpot, 0.5)), "tpot_p99_ms": _ms(_pct(tpot, 0.99)),
        "output_len": [args.output_len_min, args.output_len_max], "seed": args.seed,
        "tokens_from_usage": sum(1 for o in ok if o.usage_completion is not None),
        "prompt_tokens_expected": expected, "prompt_tokens_mismatch": prompt_mismatch,
        "load_before": load_before, "load_after": load_after,
        "errors": errors[:5],
        "per_request": [{k: v for k, v in asdict(o).items() if k != "text"} for o in jobs],
    }


def _cell_dir(args, scenario: str, engine: str, c: int, rnd: int) -> Path:
    return args.out / "cells" / scenario / engine / f"c{c}_round{rnd}"


# --- servers -------------------------------------------------------------------------

def start(args, engine: str, kind: str, label: str, extra_env: dict[str, str] | None = None):
    """A healthy, smoked server, or None (and a support record saying why)."""
    from benchmark.eval.server import ServerError

    wait_gpu_free(args)
    log = args.out / "servers" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    srv = _server(args, engine, kind, log, extra_env)
    print(f"== {label}: starting {engine} ({kind})", flush=True)
    srv.start()
    try:
        srv.wait_ready()
    except (ServerError, KeyboardInterrupt) as exc:
        srv.stop()
        if isinstance(exc, KeyboardInterrupt):
            raise
        _write(_support_path(args, engine, kind), {"ok": False, "error": _excerpt(log), "log": str(log)})
        print(f"!! {label}: not supported: {_excerpt(log, 300)}", flush=True)
        return None
    # one short request per model the server serves; an adapter it rejects is recorded
    labels = [BASE, *_kind_dirs(args, kind)]
    rejected = {}
    for name in labels:
        got = _answer(args, srv.url, engine, name, "Say hello.", 8)
        if isinstance(got, dict) or not got.strip():
            rejected[name] = (got.get("error") if isinstance(got, dict) else "empty answer")[:400]
    record = {"ok": BASE not in rejected, "rejected": rejected, "startup_s": srv.startup_s, "log": str(log)}
    if BASE in rejected:
        record["error"] = rejected[BASE]
    _write(_support_path(args, engine, kind), record)
    record_versions(args, engine)
    if not record["ok"]:
        srv.stop()
        print(f"!! {label}: the base model fails: {rejected[BASE]}", flush=True)
        return None
    for name, why in rejected.items():
        print(f"!! {label}: adapter {name} rejected: {why}", flush=True)
    return srv


def stop(args, srv, engine: str, label: str) -> None:
    srv.stop()
    if engine == "mstar":
        # MSTAR_REQUIRE_CUDA_GRAPHS fails startup on a dropped bucket; count any other fallback
        text = (args.out / "servers" / f"{label}.log").read_text(errors="replace")
        n = text.count("Failed to capture")
        if n:
            print(f"!! {label}: {n} 'Failed to capture' lines", flush=True)


def _adapter_modules(d: Path) -> list[str]:
    """The linears (leaf names) an adapter file holds LoRA weights for, from the
    safetensors header, so nothing imports torch."""
    with open(d / "adapter_model.safetensors", "rb") as f:
        header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
    return sorted({k.split(".lora_A.")[0].rsplit(".", 1)[-1] for k in header if ".lora_A." in k})


def module_coverage(args) -> dict:
    """Per adapter, the modules each engine applies. mstar (``LoraSpec.check_adapter``)
    and vLLM (its loader's expected-modules check) refuse an adapter with a module they
    have no target for, so they apply all of it or none. SGLang applies the modules whose
    target it has; one without may be refused or dropped, so it is counted as dropped."""
    out = {}
    for name, d in {**args.adapter_dirs, **args.all_linear_dirs}.items():
        modules = _adapter_modules(d)
        engines = {}
        for e in args.engines:
            drops = ([m for m in modules if SGLANG_NAMES.get(m, m) not in args.sglang_lora_modules]
                     if e == "sglang" else [])
            engines[e] = {"applies": [m for m in modules if m not in drops], "drops": drops}
        config = json.loads((d / "adapter_config.json").read_text()).get("target_modules")
        out[name] = {"target_modules": config, "modules": modules, "engines": engines}
    _write(args.out / "support" / "adapter_modules.json", out)
    return out


def _excluded(args, engine: str, labels: list[str]) -> str | None:
    """Why ``engine`` must not be compared on ``labels``: it would apply only part of one."""
    for n in labels:
        rec = args.coverage.get(n, {}).get("engines", {}).get(engine)
        if rec and rec["drops"]:
            return (f"excluded: {engine} has no target for {n}'s {', '.join(rec['drops'])} "
                    f"(refused or silently dropped), so it would do less work")
    return None


def _skip_reason(args, engine: str, kind: str, labels: list[str]) -> str | None:
    if why := _excluded(args, engine, labels):
        return why
    rec = _support(args, engine, kind)
    if rec is None:
        return None
    # mstar serves LoRA on the delta-net from lora/gdn on; before that it refuses these
    head = "mstar: not supported on this branch" if engine == "mstar" and kind == "alllinear" else "not supported"
    if not rec.get("ok"):
        return f"{head} ({rec.get('error', '')[:300]})"
    bad = [n for n in labels if n in rec.get("rejected", {})]
    if bad:
        return f"{head} (adapter {bad[0]} rejected: {rec['rejected'][bad[0]][:300]})"
    return None


# --- stages --------------------------------------------------------------------------

def _correctness_dir(args, kind: str) -> Path:
    cdir = args.out / "correctness"
    return cdir if kind == "fit" else cdir / kind


def stage_correctness(args) -> None:
    for kind in CORRECTNESS_KINDS:
        if not _kind_dirs(args, kind):
            continue
        cdir = _correctness_dir(args, kind)
        cdir.mkdir(parents=True, exist_ok=True)
        if not (cdir / "reference.json").exists():
            wait_gpu_free(args)
            cmd = [args.mstar_python, "-m", "benchmark.lora.qwen3_5_parity", "--phase", "reference",
                   "--model", args.model, "--out", str(cdir), "--max-tokens", str(args.parity_max_tokens),
                   "--topk", str(args.topk)]
            for name, spec in _kind_specs(args, kind).items():
                cmd += ["--adapter", f"{name}={spec}"]
            print(f"== correctness ({kind}): HF + PEFT reference", flush=True)
            subprocess.run(cmd, check=True, cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu})
        for engine in args.engines:
            path = cdir / f"{engine}.json"
            if path.exists():
                continue
            label = f"correctness_{engine}" if kind == "fit" else f"correctness_{engine}_{kind}"
            srv = start(args, engine, kind, label)
            if srv is None:
                _write(path, {"unsupported": _skip_reason(args, engine, kind, [BASE])})
                continue
            # an adapter the engine rejects, or would apply only part of, is not scored
            skipped = {n: why for n in _kind_dirs(args, kind) if (why := _skip_reason(args, engine, kind, [n]))}
            names = [BASE, *(n for n in _kind_dirs(args, kind) if n not in skipped)]
            try:
                url = srv.url
                jobs = [(n, i) for n in names for i in range(len(PROMPTS))]

                def ask(job, url=url, engine=engine):
                    return _answer(args, url, engine, job[0], PROMPTS[job[1]], args.parity_max_tokens)

                single, mixed = {}, {}
                for n, i in jobs:
                    single.setdefault(n, {})[str(i)] = ask((n, i))
                with concurrent.futures.ThreadPoolExecutor(len(jobs)) as pool:
                    for (n, i), got in zip(jobs, pool.map(ask, jobs), strict=True):
                        mixed.setdefault(n, {})[str(i)] = got
            finally:
                stop(args, srv, engine, label)
            _write(path, {"single": single, "mixed": mixed, "skipped": skipped})
        score_correctness(args, kind)


def score_correctness(args, kind: str = "fit") -> dict:
    cdir = _correctness_dir(args, kind)
    ref_path = cdir / "reference.json"
    if not ref_path.exists():
        return {}
    ref = json.loads(ref_path.read_text())
    tok = _tokenizer(args)
    scores: dict = {}
    for engine in args.engines:
        path = cdir / f"{engine}.json"
        if not path.exists():
            continue
        got = json.loads(path.read_text())
        if "unsupported" in got:
            scores[engine] = {"unsupported": got["unsupported"]}
            continue
        names = list(got["single"])  # the base model and every adapter not skipped
        rows: dict[str, dict] = {}
        for mode in ("single", "mixed"):
            for n in names:
                for i, text in got[mode][n].items():
                    if isinstance(text, dict):
                        rows[f"{mode}/{n}/{i}"] = {"pass": False, "error": text["error"]}
                    else:
                        ids = tok(text, add_special_tokens=False).input_ids
                        rows[f"{mode}/{n}/{i}"] = _agree(ids, ref[n][i])
        counts = {
            f"{mode}/{n}": {
                "pass": sum(1 for k, r in rows.items() if k.startswith(f"{mode}/{n}/") and r["pass"]),
                "identical": sum(1 for k, r in rows.items()
                                 if k.startswith(f"{mode}/{n}/") and "error" not in r and not r["diverged"]),
                "total": len(got[mode][n]),
            }
            for mode in ("single", "mixed") for n in names
        }
        # each adapter must change some answer, or "passing" may mean it never ran
        control = {n: any(got["single"][n][i] != got["single"][BASE][i] for i in got["single"][n])
                   for n in names if n != BASE}
        scores[engine] = {"rows": rows, "counts": counts, "adapter_changes_output": control,
                          "skipped": got.get("skipped", {})}
    # rows the engines judge differently
    keys = sorted({k for s in scores.values() for k in s.get("rows", {})})
    scores["disagreements"] = [
        {"row": k, **{e: s["rows"][k]["pass"] for e, s in scores.items() if k in s.get("rows", {})}}
        for k in keys
        if len({s["rows"][k]["pass"] for s in scores.values() if k in s.get("rows", {})}) > 1
    ]
    _write(cdir / "scores.json", scores)
    return scores


def stage_perf(args) -> None:
    concs = args.concurrency
    for rnd in range(args.first_round, args.first_round + args.rounds):
        shift = (rnd - 1) % len(args.engines)
        order = list(args.engines[shift:]) + list(args.engines[:shift])
        for kind in KINDS:
            scenarios = [s for s in _scenarios(args) if s.kind == kind]
            for engine in order:
                pending = [(s, c) for s in scenarios for c in concs
                           if not (_cell_dir(args, s.name, engine, c, rnd) / "result.json").exists()]
                if not pending:
                    continue
                excluded = [_excluded(args, engine, _labels(args, s)) for s, _ in pending]
                if all(excluded):
                    print(f"== round {rnd} {engine} {kind}: {excluded[0]}")
                    continue
                if args.dry_run:
                    print(f"round {rnd} {engine} {kind}: {_describe(args, engine, kind)}")
                    for s, c in pending:
                        print(f"    {s.name} c={c}: {_requests(c)} requests on {_labels(args, s)}")
                    continue
                # a server that failed, or rejected every pending cell's adapters, is not restarted
                reasons = [_skip_reason(args, engine, kind, _labels(args, s)) for s, _ in pending]
                if all(reasons) and not args.retry_unsupported:
                    print(f"== round {rnd} {engine} {kind}: skipped, {reasons[0]}")
                    continue
                label = f"perf_{engine}_{kind}_round{rnd}"
                srv = start(args, engine, kind, label)
                if srv is None:
                    continue
                try:
                    first = True
                    for s, c in pending:
                        why = _skip_reason(args, engine, kind, _labels(args, s))
                        if why:
                            print(f"   {s.name} c={c}: {why}", flush=True)
                            continue
                        waves = args.warmup_first if first else args.warmup_rest
                        first = False
                        print(f"== round {rnd} {engine} {s.name} c={c} (warmup {waves} waves)", flush=True)
                        result = run_cell(args, srv.url, engine, s, c, waves)
                        result.update(round=rnd, server_log=str(args.out / "servers" / f"{label}.log"))
                        _write(_cell_dir(args, s.name, engine, c, rnd) / "result.json", result)
                        flag = "" if result["failed"] == 0 else f"  !! {result['failed']} FAILED {result['errors'][:2]}"
                        print(f"   {result['tok_s']:.0f} tok/s  ttft p50 {result['ttft_p50_ms'] or 0:.1f} ms  "
                              f"tpot p50 {result['tpot_p50_ms'] or 0:.2f} ms{flag}", flush=True)
                except Exception as exc:  # noqa: BLE001 - one engine failing must not cost the others their run
                    print(f"!! {label}: {exc!r}; its remaining cells are left for a rerun", flush=True)
                finally:
                    stop(args, srv, engine, label)


def stage_profile(args) -> None:
    from benchmark.worker_phases.phases import parse_log, render, segment

    if "mstar" not in args.engines:
        return
    pdir = args.out / "profile"
    for kind in KINDS:
        scenarios = [s for s in _scenarios(args) if s.kind == kind]
        pending = [(s, c) for s in scenarios for c in args.profile_concurrency
                   if not (pdir / f"{s.name}_c{c}.txt").exists()]
        if not pending:
            continue
        if args.dry_run:
            print(f"profile mstar {kind} (MSTAR_PHASE_TIMING={args.period}): "
                  + ", ".join(f"{s.name} c={c}" for s, c in pending))
            continue
        if all(_skip_reason(args, "mstar", kind, _labels(args, s)) for s, _ in pending):
            continue
        label = f"profile_mstar_{kind}"
        srv = start(args, "mstar", kind, label, {"MSTAR_PHASE_TIMING": str(args.period)})
        if srv is None:
            continue
        log = args.out / "servers" / f"{label}.log"
        try:
            first = True
            for s, c in pending:
                if _skip_reason(args, "mstar", kind, _labels(args, s)):
                    continue
                offset = log.stat().st_size
                result = run_cell(args, srv.url, "mstar", s, c, args.warmup_first if first else args.warmup_rest)
                first = False
                time.sleep(3)  # the last records flush
                segs = segment(parse_log(log.read_bytes()[offset:].decode("utf-8", "replace")))
                # the longest segment near this concurrency is its steady state
                near = [g for g in segs if abs(g.mean_bs - c) <= max(1.0, 0.2 * c)]
                steady = max(near, key=lambda g: len(g.records)) if near else None
                result["steady"] = None if steady is None else {
                    "batch": steady.mean_bs,
                    "mean_ms": {p: v["mean_ms"] for p, v in steady.summary().items() if p in PROFILE_PHASES},
                }
                _write(pdir / f"{s.name}_c{c}.json", {k: v for k, v in result.items() if k != "per_request"})
                (pdir / f"{s.name}_c{c}.txt").write_text(render(segs))
                print(f"== profile {s.name} c={c}: {pdir / f'{s.name}_c{c}.txt'}", flush=True)
        finally:
            stop(args, srv, "mstar", label)


# --- summary -------------------------------------------------------------------------

def _cells(args) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((args.out / "cells").glob("*/*/*/result.json"))]


def _fmt(x: float | None, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def _show(label: str) -> str:
    return "base" if label == BASE else label  # "<base>" is an HTML tag in markdown


def summary(args) -> str:
    cells = _cells(args)
    counted = [r for r in cells if r["failed"] == 0]
    by: dict[tuple, list[dict]] = {}
    for r in counted:
        by.setdefault((r["scenario"], r["concurrency"], r["engine"]), []).append(r)

    def tok_s(scn, c, e):
        rows = by.get((scn, c, e), [])
        return statistics.fmean(r["tok_s"] for r in rows) if rows else None

    def cell_tok(scn, c, e):
        rows = by.get((scn, c, e), [])
        if not rows:
            s = next(s for s in SCENARIOS if s.name == scn)
            why = _skip_reason(args, e, s.kind, _labels(args, s))
            return "x" if why is None else ("excl." if why.startswith("excluded") else "n/s")
        sd = statistics.stdev(r["tok_s"] for r in rows) if len(rows) > 1 else 0.0
        return f"{tok_s(scn, c, e):.0f}±{sd:.0f} ({len(rows)})"

    def med(scn, c, e, key):
        vals = [r[key] for r in by.get((scn, c, e), []) if r[key] is not None]
        return statistics.median(vals) if vals else None

    engines = args.engines
    head = (["scenario", "c"] + [f"{e} tok/s" for e in engines] + [f"{e} TTFT p50 ms" for e in engines]
            + [f"{e} TPOT p50 ms" for e in engines] + [f"{e} vs own e_off" for e in engines])
    lines = [
        "# Multi-LoRA serving: mstar vs vLLM vs SGLang (Qwen3.5-0.8B)", "",
        f"tok/s: mean±sd over counted rounds (count); TTFT/TPOT: median over rounds of each round's p50. "
        f"Output length per request seeded uniform {args.output_len_min}-{args.output_len_max}, ignore_eos. "
        f"Slots: fit={args.slots_fit}, churn={args.slots_churn}, alllinear={args.slots_fit}. `n/s` = not "
        f"supported, `excl.` = excluded because the engine would apply only part of an adapter, "
        f"`x` = no counted cell.",
        "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head),
    ]
    for s in _scenarios(args):
        for c in args.concurrency:
            row = [s.name, str(c)]
            row += [cell_tok(s.name, c, e) for e in engines]
            row += [_fmt(med(s.name, c, e, "ttft_p50_ms"), ".1f") for e in engines]
            row += [_fmt(med(s.name, c, e, "tpot_p50_ms"), ".2f") for e in engines]
            for e in engines:
                a, b = tok_s(s.name, c, e), tok_s(ANCHOR, c, e)
                row.append("anchor" if s.name == ANCHOR else ("-" if a is None or b is None
                                                                else f"{100 * (a / b - 1):+.1f}%"))
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Scenarios: " + "; ".join(f"`{s.name}` {s.what}" for s in _scenarios(args))]
    lines += _all_linear_lines(args)
    lines += ["", "## Every metric (median over counted rounds)", "",
              "| scenario | c | engine | rounds | tok/s | req/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | "
              "failed (all rounds) |", "|" + "---|" * 11]
    for s in _scenarios(args):
        for c in args.concurrency:
            for e in engines:
                failed = sum(r["failed"] for r in cells
                             if (r["scenario"], r["concurrency"], r["engine"]) == (s.name, c, e))
                lines.append(f"| {s.name} | {c} | {e} | {len(by.get((s.name, c, e), []))} | "
                             + " | ".join(_fmt(med(s.name, c, e, k), f) for k, f in (
                                 ("tok_s", ".0f"), ("req_s", ".2f"), ("ttft_p50_ms", ".1f"), ("ttft_p99_ms", ".1f"),
                                 ("tpot_p50_ms", ".2f"), ("tpot_p99_ms", ".2f")))
                             + f" | {failed} |")

    for kind in CORRECTNESS_KINDS:
        lines += _correctness_lines(args, kind)

    unsupported = []
    for p in sorted((args.out / "support").glob("*.json")):
        rec = json.loads(p.read_text())
        if "ok" not in rec:
            continue  # adapter_modules.json
        head = "mstar: not supported on this branch" if p.stem == "mstar_alllinear" else "not supported"
        if not rec.get("ok"):
            unsupported.append(f"- {p.stem}: {head} ({rec.get('error', '')[:300]})")
        for n, why in rec.get("rejected", {}).items():
            if n != BASE:
                unsupported.append(f"- {p.stem}: {head}, adapter {n} rejected ({why[:300]})")
    if unsupported:
        lines += ["", "## Not supported", "", *unsupported]
    excluded = [r for r in cells if r["failed"]]
    if excluded:
        lines += ["", "## Excluded cells (failed > 0)", ""]
        lines += [f"- {r['scenario']} {r['engine']} c={r['concurrency']} round {r['round']}: "
                  f"{r['failed']} failed, e.g. {r['errors'][:1]}" for r in excluded]
    mismatch = [r for r in counted if r.get("prompt_tokens_mismatch")]
    if mismatch:
        lines += ["", "Prompt-length mismatches against the HF chat template (input not matched): "
                  + ", ".join(f"{r['engine']}/{r['scenario']}/c{r['concurrency']}" for r in mismatch)]

    lines += _profile_lines(args)
    vpath = args.out / "versions.json"
    if vpath.exists():
        lines += ["", "## Versions", "", "```", vpath.read_text().strip(), "```"]
    return "\n".join(lines)


def _correctness_lines(args, kind: str) -> list[str]:
    scores = score_correctness(args, kind)
    if not scores:
        return []
    names = [BASE, *_kind_dirs(args, kind)]
    which = "" if kind == "fit" else ", all-linear adapters"
    lines = ["", f"## Correctness{which} (greedy vs HF + PEFT, top-{args.topk} rule; pass/total, identical)", "",
             "| engine | " + " | ".join(f"{_show(n)} single | {_show(n)} mixed" for n in names)
             + " | adapter changes output |",
             "|" + "---|" * (2 * len(names) + 2)]
    skipped = []
    for e in args.engines:
        s = scores.get(e)
        if s is None:
            continue
        if "unsupported" in s:
            lines.append(f"| {e} | {s['unsupported']} |")
            continue
        vals = []
        for n in names:
            for mode in ("single", "mixed"):
                k = s["counts"].get(f"{mode}/{n}")
                vals.append("skip" if k is None else f"{k['pass']}/{k['total']} ({k['identical']})")
        ctrl = ", ".join(f"{n}:{'yes' if v else 'NO'}" for n, v in s["adapter_changes_output"].items())
        lines.append(f"| {e} | " + " | ".join(vals) + f" | {ctrl} |")
        skipped += [f"- {e} {n}: {why}" for n, why in s.get("skipped", {}).items()]
    if skipped:
        lines += ["", "Skipped:", *skipped]
    dis = scores.get("disagreements", [])
    lines += ["", f"Rows the engines judge differently: {len(dis)}"]
    for d in dis[:20]:
        verdicts = ", ".join(f"{e}={'pass' if v else 'FAIL'}" for e, v in d.items() if e != "row")
        lines.append(f"- {d['row'].replace(BASE, 'base')}: {verdicts}")
    return lines


def _all_linear_lines(args) -> list[str]:
    """Per all-linear adapter, the modules each engine applies, and whether it served it."""
    if not args.all_linear_dirs:
        return []
    lines = ["", "## All-linear adapters: modules each engine applies", "",
             f"SGLang `--lora-target-modules {' '.join(_target_modules(args, 'alllinear'))}`. "
             "mstar and vLLM refuse an adapter with a module they have no target for; SGLang's "
             "missing targets count as dropped, which excludes it.", ""]
    for n in args.all_linear_dirs:
        rec = args.coverage[n]
        parts = []
        for e in args.engines:
            why = _skip_reason(args, e, "alllinear", [n])
            applied = ", ".join(rec["engines"][e]["applies"]) or "none"
            parts.append(f"{e}: {applied}" + (f" ({why})" if why else ""))
        lines.append(f"- `{n}` (file: {', '.join(rec['modules'])}): " + "; ".join(parts))
    return lines


def _profile_lines(args) -> list[str]:
    pdir = args.out / "profile"
    if not pdir.exists():
        return []
    out = ["", "## mstar worker phases (mean ms, steady segment; event_sync near 0 = CPU-bound)", "",
           "| scenario | c | batch | " + " | ".join(PROFILE_PHASES) + " |", "|---" * (3 + len(PROFILE_PHASES)) + "|"]
    for s in _scenarios(args):
        for c in args.profile_concurrency:
            path = pdir / f"{s.name}_c{c}.json"
            steady = json.loads(path.read_text()).get("steady") if path.exists() else None
            if not steady:
                continue
            ms = steady["mean_ms"]
            out.append(f"| {s.name} | {c} | {steady['batch']:.1f} | "
                       + " | ".join(_fmt(ms.get(p), ".3f") for p in PROFILE_PHASES) + " |")
    out.append(f"\nFull tables: `{pdir}/<scenario>_c<c>.txt` (see benchmark/worker_phases/README.md).")
    return out


# --- main ------------------------------------------------------------------------------

def _max_rank(dirs: dict[str, Path]) -> int:
    """Every engine's max LoRA rank, one for the whole run: the largest rank (``r``, or a
    ``rank_pattern`` entry) among the adapters it loads."""
    ranks = []
    for name, d in dirs.items():
        config = json.loads((d / "adapter_config.json").read_text())
        r = config.get("r")
        if not isinstance(r, int) or r <= 0:
            raise SystemExit(f"{name}: adapter_config.json gives no rank (r = {r!r})")
        ranks += [r, *(config.get("rank_pattern") or {}).values()]
    return max(ranks, default=16)


def _adapter_setup(args, log: bool = False) -> None:
    args.adapter_dirs = {n: _adapter_dir(s) for n, s in args.adapters.items()}
    args.all_linear_dirs = {n: _adapter_dir(s) for n, s in args.all_linear_adapters.items()}
    args.coverage = module_coverage(args)
    args.max_rank = _max_rank({**args.adapter_dirs, **args.all_linear_dirs})
    if not log:
        return
    print("== adapter modules, and which each engine applies", flush=True)
    for n, rec in args.coverage.items():
        print(f"   {n}: target_modules {rec['target_modules']}; file holds {rec['modules']}", flush=True)
        for e, r in rec["engines"].items():
            note = f"; DROPS {r['drops']}: excluded on {n}" if r["drops"] else ""
            print(f"      {e}: applies {r['applies']}{note}", flush=True)
    if "sglang" in args.engines and args.all_linear_dirs:
        print(f"   sglang alllinear --lora-target-modules {_target_modules(args, 'alllinear')}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage", choices=("all", "correctness", "perf", "profile"), default="all")
    p.add_argument("--summary", action="store_true", help="only render the summary of what has finished")
    p.add_argument("--dry-run", action="store_true", help="print the servers and cells; start nothing")
    p.add_argument("--engines", default=",".join(ENGINES))
    p.add_argument("--mstar-python", default=sys.executable)
    p.add_argument("--vllm-python", default=str(Path.home() / "venvs/vllm/bin/python"))
    p.add_argument("--sglang-python", default=str(Path.home() / "venvs/sglang/bin/python"))
    p.add_argument("--model", default="qwen3_5_0.8b", choices=sorted(HF_REPOS))
    p.add_argument("--adapter", action="append", default=[], metavar="NAME=REPO_OR_DIR")
    p.add_argument("--perf-adapters", default=None,
                   help="comma-separated subset for (b)-(d); (b) uses the first. Default: every adapter")
    p.add_argument("--all-linear-adapter", action="append", default=[], metavar="NAME=REPO_OR_DIR",
                   help="the alllinear server's adapters (repeatable); default ALL_LINEAR_ADAPTERS")
    p.add_argument("--all-linear-perf-adapters", default=None,
                   help="comma-separated subset for (f)-(g); (f) uses the first. Default: the first two")
    p.add_argument("--gpu", default="0")
    p.add_argument("--scenarios", nargs="+", default=None, choices=[s.name for s in SCENARIOS],
                   help="run only these scenarios (default: all)")
    p.add_argument("--slots-fit", type=int, default=4)
    p.add_argument("--slots-churn", type=int, default=2)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 32, 64])
    p.add_argument("--profile-concurrency", type=int, nargs="+", default=[1, 64])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--first-round", type=int, default=1, help="to add rounds without overwriting earlier ones")
    p.add_argument("--warmup-first", type=int, default=5, help="warmup waves before a server's first cell")
    p.add_argument("--warmup-rest", type=int, default=1, help="warmup waves before each later cell")
    p.add_argument("--output-len-min", type=int, default=256)
    p.add_argument("--output-len-max", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parity-max-tokens", type=int, default=48)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--period", type=int, default=100, help="MSTAR_PHASE_TIMING iterations per record")
    p.add_argument("--health-timeout", type=float, default=3600.0)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--retry-unsupported", action="store_true",
                   help="start an (engine, server kind) again even if it failed before")
    p.add_argument("--sglang-adapter-model", default="{base}:{adapter}",
                   help="the OpenAI model field that selects an SGLang adapter")
    p.add_argument("--sglang-target-modules", nargs="+", default=None,
                   help="default: the union of the adapters' target_modules")
    p.add_argument("--sglang-alllinear-target-modules", nargs="+", default=None,
                   help="the alllinear server's; default: the all-linear adapters' modules in SGLang's names")
    p.add_argument("--sglang-lora-modules", nargs="+", default=list(SGLANG_QWEN3_5_LORA_MODULES),
                   help="the LoRA targets SGLang's Qwen3.5 has; a module outside them excludes SGLang")
    p.add_argument("--mstar-lora-opt", action="append", default=[], metavar="KEY=VALUE",
                   help="extra key of mstar's `resources: lora:` block, e.g. slot_wait_steps=64 (repeatable)")
    p.add_argument("--vllm-arg", action="append", default=[], help="extra vllm serve argument (repeatable)")
    p.add_argument("--sglang-arg", action="append", default=[], help="extra launch_server argument (repeatable)")
    args = p.parse_args()
    args.engines = tuple(e for e in args.engines.split(",") if e)
    if unknown := set(args.engines) - set(ENGINES):
        p.error(f"unknown engines {sorted(unknown)}")
    args.hf_repo = HF_REPOS[args.model]
    args.adapters = dict(a.split("=", 1) for a in args.adapter) or DEFAULT_ADAPTERS.get(args.model, {})
    if not args.adapters:
        p.error(f"no default adapters for {args.model}; pass --adapter NAME=REPO_OR_DIR")
    args.perf_adapters = args.perf_adapters.split(",") if args.perf_adapters else list(args.adapters)
    args.all_linear_adapters = (dict(a.split("=", 1) for a in args.all_linear_adapter)
                                or ALL_LINEAR_ADAPTERS.get(args.model, {}))
    args.all_linear_perf = (args.all_linear_perf_adapters.split(",") if args.all_linear_perf_adapters
                            else list(args.all_linear_adapters)[:2])
    if args.all_linear_adapters and (len(args.all_linear_perf) < 2
                                     or not set(args.all_linear_perf) <= set(args.all_linear_adapters)):
        p.error("--all-linear-perf-adapters must name two or more --all-linear-adapter names")
    if args.slots_fit < len(args.all_linear_perf):
        p.error("--slots-fit must hold every all-linear perf adapter")
    if args.slots_churn >= len(args.perf_adapters):
        p.error("--slots-churn must be below the number of adapters, or (d) does not churn")
    if args.slots_fit < len(args.perf_adapters):
        p.error("--slots-fit must hold every adapter")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.summary:
        _adapter_setup(args)
        text = summary(args)
        (args.out / "summary.md").write_text(text)
        print(text)
        return 0
    from huggingface_hub import snapshot_download

    snapshot_download(args.hf_repo)  # servers run with HF_HUB_OFFLINE=1
    _adapter_setup(args, log=True)
    if "sglang" in args.engines:
        _target_modules(args)  # refuse adapters outside attention + MLP before any server starts
    if args.dry_run:
        for kind in KINDS:
            _mstar_config(args, kind)
    t0 = time.monotonic()
    if args.stage in ("all", "correctness"):
        if args.dry_run:
            print(f"correctness: HF + PEFT reference, then {', '.join(args.engines)} on the 'fit' server each")
        else:
            stage_correctness(args)
    if args.stage in ("all", "perf"):
        stage_perf(args)
    if args.stage in ("all", "profile"):
        stage_profile(args)
    if args.dry_run:
        return 0
    text = summary(args)
    (args.out / "summary.md").write_text(text)
    print(text)
    print(f"done in {(time.monotonic() - t0) / 60:.0f} min; results in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
