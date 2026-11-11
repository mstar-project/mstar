"""Qwen3-8B with the DSpark drafter, served from real weights (GPU).

Serves configs/qwen3_8b.yaml and variants of each drafter's config (``DRAFTERS``) with
``test/e2e``'s server harness, collecting greedy token ids (``return_token_ids``)
and the ``mstar_spec_*`` counters at ``/metrics``. Each served config answers
the prompts twice: one request at a time (every step a batch of one, so two
runs are comparable token for token) and all at once (batched).

What is compared, and how:

- exactly, only between sequential runs that differ in nothing but the mode, at
  compile level 0: synchronous and asynchronous verify on one GPU, and on TP2;
- the KV-pressure run by the fp32 screen (fp32_screen.py): ``ROUNDS`` rounds of
  32 requests in seeded arrival orders against plain under the same pressure,
  judged by transformers' Qwen3-8B in fp32 teacher-forced on each served reply;
  the TP2 runs by the screen too, against plain one request at a time and at once;
- everything else (plain vs speculative, FA3 vs FlashInfer, concurrent runs) by
  the margin rule: where two replies
  first differ, the plain model's top-2 log-prob margin at that position, from
  transformers' Qwen3-8B on the shared prefix, must be at most ``MARGIN`` (a
  bf16 near-tie a different kernel or batch shape may flip);
- acceptance: mean accepted length (tau) above 2, equal between modes and
  between captured and uncaptured runs within noise, TP2 within 10% of one GPU,
  a repeated prompt as accepted as its first run, and a sampled run at the
  checkpoint's defaults above 1.5;
- a repeated prompt hits the prefix cache (16-token pages, so the chat prompts
  fill whole pages; the hit counter must grow), and a KV-pressure run offloads.

The module never downloads weights: it skips a drafter unless every shard of its
checkpoint and the target's is in a standard Hugging Face cache, and serves with
``HF_HUB_OFFLINE=1``. It needs CUDA and FA3's paged kernel (Hopper); the TP2
cases need two GPUs, the margin rule needs transformers and the screen scipy too.

    pytest test/integration/test_qwen3_dspark_real_weights.py -q
"""
from __future__ import annotations

import base64
import importlib.util
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import pytest
import requests
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
TARGET = "Qwen/Qwen3-8B"


class Drafter(NamedTuple):
    """A drafter under test: its method, serving config, checkpoint and its own attention binding."""

    id: str
    method: str
    config: str
    checkpoint: str
    attn_key: str | None


DRAFTERS = [
    Drafter("dspark", "dspark", "configs/qwen3_8b_dspark.yaml", "deepseek-ai/dspark_qwen3_8b_block7", "dspark_attn"),
]
MARGIN = 0.25
PAGE = 16
# the pressure group's rounds, each in its own seeded arrival order
ROUNDS = 4
# a pool of 16-token pages too small for 32 requests at once
PRESSURE_PAGES = 128

TOKENS = 128


def _sibling(name: str):
    """A helper module of this directory, by path (the integration tests are no package)."""
    key = f"_integration_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(f"{name}.py"))
        sys.modules[key] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[key])
    return sys.modules[key]


fp32_screen = _sibling("fp32_screen")
# the deficit tripwire is uncalibrated for Qwen3-8B, so the rank tripwire alone
GATES = fp32_screen.Gates(p_fail=1e-3, trip_rank=5, trip_deficit=None, burst=(3, 16))
spec_served = _sibling("spec_served")
_snapshot, _server_module, _totals, _counters, _tau = (
    spec_served._snapshot, spec_served._server_module, spec_served._totals, spec_served._counters, spec_served._tau,
)
PROMPTS = spec_served.PROMPTS


def _fa3_reason() -> str | None:
    from mstar.engine.resources.attn.fa3_paged import fa3_paged_unavailable_reason

    return fa3_paged_unavailable_reason(torch.device("cuda"), 128, torch.bfloat16)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not _snapshot(TARGET), reason=f"{TARGET} is not fully cached"),
    pytest.mark.skipif(torch.cuda.is_available() and _fa3_reason() is not None,
                       reason="FA3's paged kernel doesn't run here"),
]
needs_two = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")


@pytest.fixture(scope="module", params=DRAFTERS, ids=lambda d: d.id)
def drafter(request) -> Drafter:
    """Every test runs against each drafter; one whose checkpoint isn't fully cached is skipped."""
    if not _snapshot(request.param.checkpoint):
        pytest.skip(f"{request.param.checkpoint} is not fully cached")
    return request.param


# ── serving ──


def _config(tmp: Path, name: str, drafter: Drafter, *, draft_mode: str | None, verify_backend: str | None = None,
            tp: int = 1,
            captured: bool = True, max_num_pages: int | None = None) -> Path:
    base = drafter.config if draft_mode else "configs/qwen3_8b.yaml"
    config = yaml.safe_load((REPO / base).read_text())
    resources = config.setdefault("resources", {})
    if draft_mode:
        config["model_kwargs"]["draft"]["mode"] = draft_mode
    if verify_backend:
        resources.update({key: {"verify_backend": verify_backend} for key in ("attn", drafter.attn_key) if key})
    if not captured:
        config["model_kwargs"]["accelerator_graph"] = False
    # pages of 16 tokens, so a 23-56-token chat prompt fills whole pages a repeat can hit; as many
    # tokens as the shipped configs' 128-token pages hold
    kv = resources.setdefault("kv_cache", {})
    kv["page_size"] = PAGE
    kv["max_num_pages"] = kv.get("max_num_pages", 1024) * (128 // PAGE)
    if max_num_pages:
        # pages run out under load: requests are offloaded and reloaded
        kv.update(max_num_pages=max_num_pages, cpu_offload_pages=4 * max_num_pages)
    if tp > 1:
        config["node_groups"] = [{"node_names": ["LLM"], "ranks": list(range(tp)), "tp_size": tp}]
    path = tmp / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _serve(tmp_path_factory, name: str, drafter: Drafter, compile_level: int | None = None, **kw):
    tmp = tmp_path_factory.mktemp(name)
    server = _server_module()
    command = [
        sys.executable, "-u", "-m", "mstar.api_server.entrypoint", "--config", str(_config(tmp, name, drafter, **kw)),
        "--host", server.HOST, "--port", "{port}", "--socket-path-prefix", str(tmp / "ipc"),
        "--upload-dir", str(tmp / "upload"), "--tensor-comm-protocol", "SHM", "--enable-metrics",
    ]
    # asctime in UTC, which the reload-time parse (`fp32_screen.events`) and the client's clock agree on
    env = {"HF_HUB_OFFLINE": "1", "TZ": "UTC"}
    if compile_level is not None:
        env["MSTAR_COMPILE_LEVEL"] = str(compile_level)
    return server.Server(REPO, tmp / "config.yaml", tmp / "server.log", command=command, env=env)


def _reply(url: str, prompt: str, rid: str | None = None, sampled: bool = False) -> dict:
    """One streamed reply, greedy unless ``sampled`` (then the checkpoint's defaults): its token ids, and each
    chunk's ids with the client's receive time."""
    kwargs = {"max_output_tokens": TOKENS, "enable_thinking": False, "return_token_ids": True}
    if not sampled:
        kwargs["temperature"] = 0
    data = {"text": prompt, "output_modalities": "text", "streaming": "true", "model_kwargs": json.dumps(kwargs)}
    if rid is not None:
        data["request_id"] = rid
    r = requests.post(url + "/generate", data=data, stream=True, timeout=900)
    r.raise_for_status()
    chunks = []
    for line in r.iter_lines():
        if line:
            chunk = json.loads(line)
            if chunk["modality"] == "error":
                raise AssertionError(base64.b64decode(chunk["data"]).decode(errors="replace"))
            ids = (chunk.get("metadata") or {}).get("token_ids", [])
            if ids:
                chunks.append([time.time(), ids])
    return {"rid": rid, "chunks": chunks, "ids": [t for _, ids in chunks for t in ids]}


def _ids(url: str, prompt: str, sampled: bool = False) -> list[int]:
    return _reply(url, prompt, sampled=sampled)["ids"]


def _all_at_once(url: str, prompts: list[str], sampled: bool = False) -> list[list[int]]:
    with ThreadPoolExecutor(len(prompts)) as pool:
        return list(pool.map(lambda p: _ids(url, p, sampled), prompts))


RUNS: dict[str, dict] = {}


def _run(tmp_path_factory, drafter: Drafter, name: str, *, sampled_too: bool = False, repeat: bool = False,
         pressure: bool = False, **kw) -> dict:
    """Serve ``name`` once and keep what every test reads of it."""
    # a plain run is every drafter's baseline; a speculative one is its drafter's
    name = f"{drafter.id}_{name}" if kw.get("draft_mode") else name
    if name in RUNS:
        return RUNS[name]
    run: dict = {}
    server = _serve(tmp_path_factory, name, drafter, **kw)
    with server as url:
        speculative = bool(kw.get("draft_mode"))
        if pressure:
            start = _counters(url, wait=False) if speculative else None
            run["rounds"] = fp32_screen.rounds(_reply, url, name, PROMPTS, 4, ROUNDS)
            if speculative:
                run["tau"] = _tau(start, _counters(url))
            # the worker logs each request it offloads for pages (`Admission.relieve`) and reloads
            log = Path(server.log_path).read_text()
            run["offloads"] = log.count("offloaded victim=")
            run["events"] = fp32_screen.events(log)
        else:
            start = _counters(url, wait=False) if speculative else None
            run["sequential"] = [_ids(url, p) for p in PROMPTS]
            seq = _counters(url) if speculative else None
            run["concurrent"] = _all_at_once(url, PROMPTS)
            conc = _counters(url) if speculative else None
            if speculative:
                run["tau"], run["tau_concurrent"] = _tau(start, seq), _tau(seq, conc)
            if repeat:
                # the same prompts again: their prompt pages, the drafter's layers too, are cached
                (hits,) = _totals(url, "mstar_prefix_cache_hit_tokens_total")
                run["repeated"] = [_ids(url, p) for p in PROMPTS]
                run["hit_tokens"] = _totals(url, "mstar_prefix_cache_hit_tokens_total")[0] - hits
                run["tau_repeated"] = _tau(conc, _counters(url))
                conc = _counters(url)
            if sampled_too:
                run["sampled"] = _all_at_once(url, PROMPTS, sampled=True)
                run["tau_sampled"] = _tau(conc, _counters(url))
    if kw.get("draft_mode"):
        # the mode the node runs in, as its startup line states it (`_set_up_draft`): a silent
        # fallback from asynchronous to synchronous would pass the sync == async comparison
        log = Path(server.log_path).read_text()
        modes = set(re.findall(rf"speculative decoding on LLM: {drafter.method}, k=\d+, (\w+) mode", log))
        assert modes == {kw["draft_mode"]}, (name, modes)
    RUNS[name] = run
    return run


# ── the margin rule ──


class _Margins:
    """``margins(prompt_index, ids)``: plain Qwen3-8B's top-2 log-prob margin for
    the token after ``ids`` of that prompt's reply, from transformers in float32 (a bf16 reference
    disagrees with itself by more than the margin rule allows).
    Loaded only once the servers it is compared against have stopped, and
    released after each comparison so the next server has the GPU."""

    def __init__(self, transformers, repo: str = TARGET):
        from mstar.model.qwen3.qwen3_model import Qwen3Model

        self.transformers, self.snap, self.lm = transformers, _snapshot(repo), None
        model = Qwen3Model(model_path_hf=str(self.snap))
        self.stop_ids = model.config.stop_token_ids
        self.prompts = [
            model.process_prompt(p, ["text"], ["text"], enable_thinking=False)
            .new_input_tensors["text_inputs"][0].tolist()
            for p in PROMPTS
        ]

    def load(self):
        if self.lm is None:
            self.lm = self.transformers.AutoModelForCausalLM.from_pretrained(
                self.snap, dtype=torch.float32).cuda().eval()
        return self.lm

    def __call__(self, i: int, ids: list[int]) -> float:
        self.load()
        with torch.no_grad():
            seq = torch.tensor([self.prompts[i % len(self.prompts)] + ids], device="cuda")
            top = self.lm(seq).logits[0, -1].float().log_softmax(-1).topk(2).values
        return float(top[0] - top[1])

    def release(self) -> None:
        self.lm = None
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def margin_of():
    return _Margins(pytest.importorskip("transformers"))


class _Fp32(fp32_screen.Fp32Reference):
    """transformers' Qwen3-8B in fp32, a prompt chat-rendered as M* renders it."""

    def __init__(self, transformers):
        from mstar.model.qwen3.qwen3_model import Qwen3Model

        super().__init__()
        self.transformers, self.snap = transformers, _snapshot(TARGET)
        self.model = Qwen3Model(model_path_hf=str(self.snap))

    def _load(self):
        return self.transformers.AutoModelForCausalLM.from_pretrained(self.snap, dtype=torch.float32).cuda().eval()

    def _inputs(self, prompt: str, image: bool) -> dict:
        ids = self.model.process_prompt(prompt, ["text"], ["text"], enable_thinking=False)
        return {"input_ids": ids.new_input_tensors["text_inputs"][0].reshape(1, -1)}


@pytest.fixture(scope="module")
def fp32():
    # the screen's exact tests
    pytest.importorskip("scipy")
    return _Fp32(pytest.importorskip("transformers"))


def _check_margins(plain: list[list[int]], other: list[list[int]], margin_of, label: str) -> None:
    try:
        for i, (a, b) in enumerate(zip(plain, other, strict=True)):
            n = min(len(a), len(b))
            at = next((j for j in range(n) if a[j] != b[j]), None)
            if at is None:
                if len(a) == len(b):
                    continue
                at = n  # one stopped where the other went on
            m = margin_of(i, a[:at])
            assert m <= MARGIN, f"{label}: prompt {i} first differs at token {at} where plain's margin is {m:.3f}"
    finally:
        margin_of.release()


# ── tests: run with -x ──


def test_the_drafter_is_accepted_captured_and_uncaptured(tmp_path_factory, drafter):
    """Mean accepted length above 2 (a mismatched drafter sits near 1), and the
    same with no captured graphs."""
    sync = _run(tmp_path_factory, drafter, "sync", draft_mode="synchronous", repeat=True)
    eager = _run(tmp_path_factory, drafter, "sync_eager", draft_mode="synchronous", captured=False)
    assert sync["tau"] > 2.0 and eager["tau"] > 2.0, (sync["tau"], eager["tau"])
    assert eager["tau"] == pytest.approx(sync["tau"], rel=0.1)


def test_speculative_greedy_tokens_are_plain_greedy_tokens(tmp_path_factory, drafter, margin_of):
    plain = _run(tmp_path_factory, drafter, "plain", draft_mode=None)
    sync = _run(tmp_path_factory, drafter, "sync", draft_mode="synchronous", repeat=True)
    async_ = _run(tmp_path_factory, drafter, "async", draft_mode="asynchronous", repeat=True, sampled_too=True)
    eager = _run(tmp_path_factory, drafter, "sync_eager", draft_mode="synchronous", captured=False)
    for name, run in (("sync", sync), ("async", async_), ("sync_eager", eager)):
        _check_margins(plain["sequential"], run["sequential"], margin_of, f"{name} sequential")
        _check_margins(plain["sequential"], run["concurrent"], margin_of, f"{name} concurrent")


def test_synchronous_and_asynchronous_verify_agree_exactly(tmp_path_factory, drafter):
    """One request at a time, both on FA3: the same tokens and the same acceptance. Both servers compile at level 0:
    per-process coordinate-descent tuning moves the drafter's low bits between servers, so at the default level
    sync and async tau can differ on a draft near-tie."""
    sync = _run(tmp_path_factory, drafter, "sync_o0", draft_mode="synchronous", compile_level=0)
    async_ = _run(tmp_path_factory, drafter, "async_o0", draft_mode="asynchronous", compile_level=0)
    assert sync["sequential"] == async_["sequential"]
    assert sync["tau"] == pytest.approx(async_["tau"], rel=1e-6)
    assert sync["tau_concurrent"] == pytest.approx(async_["tau_concurrent"], rel=0.05)


def test_flashinfer_and_fa3_paged_kernels_agree(tmp_path_factory, drafter, margin_of):
    plain = _run(tmp_path_factory, drafter, "plain", draft_mode=None)
    fi = _run(tmp_path_factory, drafter, "sync_flashinfer", draft_mode="synchronous", verify_backend="flashinfer")
    _check_margins(plain["sequential"], fi["sequential"], margin_of, "flashinfer verify")
    assert fi["tau"] > 2.0


def test_sampled_requests_at_the_checkpoint_defaults(tmp_path_factory, drafter):
    """No request option set: Qwen3-8B's generation_config (T=0.6, top_k=20, top_p=0.95) verifies."""
    run = _run(tmp_path_factory, drafter, "async", draft_mode="asynchronous", repeat=True, sampled_too=True)
    assert all(len(ids) > 0 for ids in run["sampled"])
    assert run["tau_sampled"] > 1.5, run["tau_sampled"]


def test_a_repeated_prompt_hits_the_prefix_cache_and_drafts_as_well(tmp_path_factory, drafter, margin_of):
    """The repeat's prompt pages come from the cache, the drafter's KV layers with
    them: the hit counter grows, and the replies and acceptance hold."""
    plain = _run(tmp_path_factory, drafter, "plain", draft_mode=None)
    for name, mode in (("sync", "synchronous"), ("async", "asynchronous")):
        run = _run(tmp_path_factory, drafter, name, draft_mode=mode, repeat=True, sampled_too=(name == "async"))
        assert run["hit_tokens"] > 0, "the repeat hit no cached page"
        _check_margins(plain["sequential"], run["repeated"], margin_of, f"{name} repeated")
        assert run["tau_repeated"] == pytest.approx(run["tau"], rel=0.1)


def test_kv_pressure_offloads_and_completes(tmp_path_factory, drafter, fp32):
    """A pool too small for every request at once (``PRESSURE_PAGES``), ``ROUNDS`` rounds of 32 requests in
    shuffled arrival orders: requests are offloaded and reloaded under asynchronous verify, and the replies pass
    the fp32 screen against plain under the same pressure and arrival orders. A reply that ends where plain's
    goes on, on no stop token (truncated or empty), trips it."""
    plain = _run(tmp_path_factory, drafter, "plain_pressure", draft_mode=None, max_num_pages=PRESSURE_PAGES,
                 pressure=True)
    run = _run(tmp_path_factory, drafter, "async_pressure", draft_mode="asynchronous",
               max_num_pages=PRESSURE_PAGES, pressure=True)
    assert run["offloads"] > 0, "the pool never ran out of pages"
    reloads = sum(e[1] == "reload" for e in run["events"])
    print(f"{drafter.id} pressure: offloads {run['offloads']}, reloads {reloads} (plain: offloads "
          f"{plain['offloads']}), tau {run['tau']:.3f}")
    try:
        failed = fp32_screen.screen(f"{drafter.id} pressure", [
            (fp32_screen.pairs_rounds(plain, run, PROMPTS), None, run["events"])], fp32, GATES)
    finally:
        fp32.release()
    assert not failed, "\n".join(failed)
    assert run["tau"] > 2.0


def _pairs(plain: list[list[int]], other: list[list[int]], label: str) -> list:
    """The screen's pairs of two runs' replies to PROMPTS, one at a time or at once (ids only)."""
    return [fp32_screen.Pair(PROMPTS[i % len(PROMPTS)], False, {"ids": a, "rid": None, "chunks": []},
                             {"ids": b, "rid": None, "chunks": []}, f"{label} p{i}")
            for i, (a, b) in enumerate(zip(plain, other, strict=True))]


@needs_two
def test_tp2_verifies_as_one_gpu_does(tmp_path_factory, drafter, fp32):
    """On a TP2 node, at compile level 0, synchronous and asynchronous verify agree exactly one request at a time
    (level 0 for near-ties, as on one GPU: a hypothesis, not measured on TP2). At the default level, both pass the
    fp32 screen against plain one request at a time and at once, and acceptance is within 10% of one GPU."""
    exact_sync = _run(tmp_path_factory, drafter, "tp2_sync_o0", draft_mode="synchronous", tp=2, compile_level=0)
    exact_async = _run(tmp_path_factory, drafter, "tp2_async_o0", draft_mode="asynchronous", tp=2, compile_level=0)
    assert exact_sync["sequential"] == exact_async["sequential"]
    tp_sync = _run(tmp_path_factory, drafter, "tp2_sync", draft_mode="synchronous", tp=2)
    tp_async = _run(tmp_path_factory, drafter, "tp2_async", draft_mode="asynchronous", tp=2)
    one = _run(tmp_path_factory, drafter, "async", draft_mode="asynchronous", repeat=True, sampled_too=True)
    plain = _run(tmp_path_factory, drafter, "plain", draft_mode=None)
    try:
        failed = [r for name, run in (("tp2_sync", tp_sync), ("tp2_async", tp_async))
                  for r in fp32_screen.screen(f"{drafter.id} {name}", [
                      (_pairs(plain["sequential"], run["sequential"], "seq"), None, None),
                      (_pairs(plain["concurrent"], run["concurrent"], "conc"), None, None)], fp32, GATES)]
    finally:
        fp32.release()
    assert not failed, "\n".join(failed)
    assert tp_async["tau"] == pytest.approx(one["tau"], rel=0.1), (tp_async["tau"], one["tau"])
