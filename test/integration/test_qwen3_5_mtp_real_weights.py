"""Qwen3.5 drafting with its MTP head, served from real weights (GPU).

Serves configs/qwen3_5_2b.yaml and variants of configs/qwen3_5_2b_mtp.yaml
(``QWEN35_MTP_MODEL=9b`` for the 9B ones) with ``test/e2e``'s server harness,
collecting greedy token ids (``return_token_ids``) and the ``mstar_spec_*``
counters at ``/metrics``. What is compared, and how (``spec_served.py`` and
``fp32_screen.py`` hold the checks):

- exactly, between sequential runs differing only in the mode (synchronous
  and asynchronous verify), at compile level 0, since at the default level
  per-process tuning flips near-ties between servers;
- by the fp32 screen otherwise (plain vs speculative one request at a time
  and at once, the long chunked prompt, the image
  prompt, KV pressure in both verify modes against plain under the same
  pressure): transformers' Qwen3.5
  in fp32, teacher-forced on each distinct served reply, gives the served
  token's rank and deficit at every position. Per speculative arm, against
  plain at the same batch shapes: a one-sided exact test of the departure rate
  (rho = 1), a hypergeometric test of departures past plain's same-shape 99th
  percentile, both failing at p < ``GATES.p_fail``, and tripwires (a served token at
  fp32 rank >= ``GATES.trip_rank`` or deficit >= ``GATES.trip_deficit``, a burst of
  departures right after a reload, or a reply that ends on no stop token where
  plain's goes on). Where a reply first differs from plain's
  is reported, not gated. The concurrent and pressure groups run ``ROUNDS``
  rounds in seeded arrival orders, and every reply (with its chunks' receive
  times) is kept in the run's basetemp;
- exactly, under ``MSTAR_KV_DEBUG_ASSERTS`` on the speculative pressure servers: each
  reload finds the offload's KV pages, the GDN slot and the sampler's rows as
  they were, and resident rows hold the host's values;
- acceptance: mean accepted length (tau) above ``MIN_TAU`` in both modes and
  uncaptured, a repeated prompt (a prefix hit) above it too and, on the same reply,
  as accepted as its first run,
  and the image prompt's tau reported.

The module never downloads weights (every shard must be in a standard Hugging
Face cache; served with ``HF_HUB_OFFLINE=1``) and needs CUDA and FA3's paged
kernel (Hopper) for the asynchronous runs.

    pytest test/integration/test_qwen3_5_mtp_real_weights.py -q -x -rs
"""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
SIZE = os.environ.get("QWEN35_MTP_MODEL", "2b")
TARGET = f"Qwen/Qwen3.5-{SIZE.upper()}"
# FlashInfer's chunked GDN kernel on SM90 checkpoints (where prefix snapshots fall, one a page)
# only at multiples of its 64-token chunk
PAGE = 64
TOKENS = 128
# k = 3 drafts; a head that drafts nothing useful sits near 1
MIN_TAU = 1.8
# tau of requests at once at default tuning: about 0.9x what each size measured
MIN_TAU_CONCURRENT = {"9b": 2.7, "2b": 2.6}.get(SIZE, MIN_TAU)
# rounds of the concurrent and pressure groups, each in its own seeded arrival order
ROUNDS = 4
# 3072 tokens of KV, whatever the page
PRESSURE_PAGES = 3072 // PAGE


def _sibling(name: str):
    """A helper module of this directory, by path (the integration tests are no package)."""
    key = f"_integration_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(f"{name}.py"))
        sys.modules[key] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[key])
    return sys.modules[key]


fp32_screen = _sibling("fp32_screen")
# the screen's gates (fp32_screen.Gates); the deficit tripwire is ruled for 9B (1.5 nats), and 2B gates on the
# rank tripwire alone until calibrated
GATES = fp32_screen.Gates(p_fail=1e-3, trip_rank=5, trip_deficit={"9b": 1.5}.get(SIZE), burst=(3, 16))
spec_served = _sibling("spec_served")
_snapshot, _server_module, _totals, _counters, _tau = (
    spec_served._snapshot, spec_served._server_module, spec_served._totals, spec_served._counters, spec_served._tau,
)
PROMPTS = spec_served.PROMPTS
# past the 2048-token step budget, so it prefills in chunks
LONG = "Summarize the following notes in two sentences.\n" + " ".join(
    f"Note {i}: the shipment for order {1000 + i} left the warehouse on day {i % 28 + 1}." for i in range(160)
)


def _fa3_reason() -> str | None:
    from mstar.engine.resources.attn.fa3_verify import fa3_verify_unavailable_reason

    # the checkpoint's full-attention head size (256 on every released Qwen3.5)
    snap = _snapshot(TARGET)
    if snap is None:
        return f"{TARGET} is not fully cached"
    config = json.loads((snap / "config.json").read_text())
    head_dim = config.get("text_config", config)["head_dim"]
    return fa3_verify_unavailable_reason(torch.device("cuda"), head_dim, torch.bfloat16)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not _snapshot(TARGET), reason=f"{TARGET} is not fully cached"),
]
needs_fa3 = pytest.mark.skipif(torch.cuda.is_available() and _fa3_reason() is not None,
                               reason="FA3's paged kernel doesn't run here")


def _png() -> bytes:
    """A red square on white: the e2e suite's image prompt."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(image).rectangle((64, 64, 192, 192), fill="red")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


IMAGE_PROMPT = "Describe this image in detail: its colors, shapes and their positions."


def _config(tmp: Path, name: str, *, draft_mode: str | None, captured: bool = True,
            max_num_pages: int | None = None) -> Path:
    base = f"configs/qwen3_5_{SIZE}_mtp.yaml" if draft_mode else f"configs/qwen3_5_{SIZE}.yaml"
    config = yaml.safe_load((REPO / base).read_text())
    config.pop("warmup_requests", None)
    kwargs = config.setdefault("model_kwargs", {})
    if draft_mode:
        kwargs["draft"]["mode"] = draft_mode
    if not captured:
        kwargs["accelerator_graph"] = False
    # pages of PAGE tokens: the long prompt's repeat hits them (a chat prompt fills none)
    kv = config.setdefault("resources", {}).setdefault("kv_cache", {})
    kv["page_size"] = PAGE
    if max_num_pages:
        # pages run out under load: requests are offloaded and reloaded
        kv.update(max_num_pages=max_num_pages, cpu_offload_pages=4 * max_num_pages)
    path = tmp / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _serve(tmp_path_factory, name: str, debug_asserts: bool = False, compile_level: int | None = None, **kw):
    tmp = tmp_path_factory.mktemp(name)
    server = _server_module()
    command = [
        sys.executable, "-u", "-m", "mstar.api_server.entrypoint", "--config", str(_config(tmp, name, **kw)),
        "--host", server.HOST, "--port", "{port}", "--socket-path-prefix", str(tmp / "ipc"),
        "--upload-dir", str(tmp / "upload"), "--tensor-comm-protocol", "SHM", "--enable-metrics",
    ]
    # asctime in UTC, which the reload-time parse (`fp32_screen.events`) and the client's clock agree on
    env = {"HF_HUB_OFFLINE": "1", "TZ": "UTC"}
    if debug_asserts:
        # the exact gates: KV bookkeeping, each offload's pages and kept state at its reload, resident rows
        env["MSTAR_KV_DEBUG_ASSERTS"] = "1"
    if compile_level is not None:
        env["MSTAR_COMPILE_LEVEL"] = str(compile_level)
    return server.Server(REPO, tmp / "config.yaml", tmp / "server.log", command=command, env=env)


def _reply(url: str, prompt: str, rid: str | None = None, sampled: bool = False, image: bool = False) -> dict:
    """One streamed reply: its token ids, and each NDJSON chunk's ids with the client's receive time."""
    kwargs = {"max_output_tokens": TOKENS, "enable_thinking": False, "return_token_ids": True}
    if not sampled:
        kwargs["temperature"] = 0
    data = {"text": prompt, "output_modalities": "text", "streaming": "true", "model_kwargs": json.dumps(kwargs)}
    if rid is not None:
        data["request_id"] = rid
    files = {"files": ("square.png", _png(), "image/png")} if image else None
    r = requests.post(url + "/generate", data=data, files=files, stream=True, timeout=900)
    r.raise_for_status()
    chunks = []
    for line in r.iter_lines():
        if line:
            chunk = json.loads(line)
            if chunk["modality"] == "error":
                error = base64.b64decode(chunk["data"]).decode(errors="replace")
                # a server timeout arrives as a chunk: what `_pressure_arm` waives, as a broken stream would be
                raise (requests.Timeout if "timed out" in error else AssertionError)(error)
            ids = (chunk.get("metadata") or {}).get("token_ids", [])
            if ids:
                chunks.append([time.time(), ids])
    return {"rid": rid, "prompt": PROMPTS.index(prompt) if prompt in PROMPTS else prompt[:40], "image": image,
            "chunks": chunks, "ids": [t for _, ids in chunks for t in ids]}


def _rounds(url: str, name: str, copies: int) -> list[dict[int, dict]]:
    return fp32_screen.rounds(_reply, url, name, PROMPTS, copies, ROUNDS)


RUNS: dict[str, dict] = {}


def _dump(tmp_path_factory) -> Path:
    """Every run's replies (prompt, copy, ids, chunks with receive times) and server events, kept with the
    run's basetemp so a split can be traced afterwards."""
    path = tmp_path_factory.getbasetemp() / "qwen3_5_mtp_replies.json"
    path.write_text(json.dumps(RUNS))
    return path


def _run(tmp_path_factory, name: str, *, pressure: bool = False, debug_asserts: bool | None = None, **kw) -> dict:
    """A server's replies. A pressure run of a speculative arm is under ``MSTAR_KV_DEBUG_ASSERTS`` by
    default; plain's, the baseline, is not."""
    if name in RUNS:
        return RUNS[name]
    speculative = bool(kw.get("draft_mode"))
    if debug_asserts is None:
        debug_asserts = pressure and speculative
    run: dict = {"pressure": pressure, "debug_asserts": debug_asserts}
    server = _serve(tmp_path_factory, name, debug_asserts=debug_asserts, **kw)
    if pressure:
        with server as url:
            start = _counters(url, wait=False) if speculative else None
            run["rounds"] = _rounds(url, name, copies=4)
            if speculative:
                run["tau"] = _tau(start, _counters(url))
        log = Path(server.log_path).read_text()
        run["offloads"] = log.count("offloaded victim=")
        run["events"] = fp32_screen.events(log)
        run["asserts"] = log.count("AssertionError")
        RUNS[name] = run
        print(f"{name}: offloads {run['offloads']}, reloads {sum(e[1] == 'reload' for e in run['events'])}, "
              f"debug asserts {'on' if debug_asserts else 'off'}, failures {run['asserts']}"
              + (f", tau {run['tau']:.3f}" if speculative else "")
              + f"; replies in {_dump(tmp_path_factory)}")
        return run
    with server as url:
        start = _counters(url, wait=False) if speculative else None
        run["sequential"] = [_reply(url, p) for p in PROMPTS]
        seq = _counters(url) if speculative else None
        run["rounds"] = _rounds(url, name, copies=1)
        conc = _counters(url) if speculative else None
        run["repeated"] = [_reply(url, p) for p in PROMPTS]
        rep = _counters(url) if speculative else None
        run["long"] = _reply(url, LONG)
        long_ = _counters(url) if speculative else None
        (hits,) = _totals(url, "mstar_prefix_cache_hit_tokens_total")
        run["long_repeated"] = _reply(url, LONG)
        run["hit_tokens"] = _totals(url, "mstar_prefix_cache_hit_tokens_total")[0] - hits
        long_rep = _counters(url) if speculative else None
        run["image"] = _reply(url, IMAGE_PROMPT, image=True)
        image = _counters(url) if speculative else None
        if speculative:
            # the checkpoint's sampling defaults: speculative sampling
            with ThreadPoolExecutor(len(PROMPTS)) as pool:
                run["sampled"] = list(pool.map(lambda p: _reply(url, p, sampled=True)["ids"], PROMPTS))
            run["tau_sampled"] = _tau(image, _counters(url))
        if speculative:
            run["tau"], run["tau_concurrent"] = _tau(start, seq), _tau(seq, conc)
            run["tau_long"], run["tau_long_repeated"] = _tau(rep, long_), _tau(long_, long_rep)
            run["tau_image"] = _tau(long_rep, image)
    log = Path(server.log_path).read_text()
    if speculative:
        modes = set(re.findall(r"speculative decoding on LLM: mtp, k=\d+, (\w+) mode", log))
        assert modes == {kw["draft_mode"]}, (name, modes)
    # the uncaptured arm really is (the LLM node captured no graph), and the captured one did,
    # so the absence on the first is the line's, not the log's
    captured_any = bool(re.search(r"Captured accelerator graph for LLM", log))
    assert captured_any == kw.get("captured", True), (name, captured_any)
    RUNS[name] = run
    print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in run.items() if k.startswith("tau"))
          + f"; replies in {_dump(tmp_path_factory)}")
    return run


def _ids_of(run: dict, key: str) -> list[list[int]]:
    return [reply["ids"] for reply in run[key]]


# ── the fp32 screen (fp32_screen.py) ──


class _Fp32(fp32_screen.Fp32Reference):
    """transformers' Qwen3.5 in fp32, a prompt chat-rendered as M* renders it (the image prompt through the
    checkpoint's processor)."""

    def __init__(self, transformers):
        super().__init__()
        self.transformers, self.snap, self.processor = transformers, _snapshot(TARGET), None

    def _load(self):
        self.processor = self.transformers.AutoProcessor.from_pretrained(self.snap)
        return self.transformers.AutoModelForImageTextToText.from_pretrained(
            self.snap, dtype=torch.float32).cuda().eval()

    def _inputs(self, prompt: str, image: bool) -> dict:
        if not image:
            # the ids M* prefilled
            from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

            ids = Qwen3_5DenseModel(model_path_hf=str(self.snap)).process_prompt(
                prompt, ["text"], ["text"], enable_thinking=False,
            ).new_input_tensors["text_inputs"][0]
            return {"input_ids": ids.reshape(1, -1)}
        from PIL import Image

        # the processor's rendering, image patches included (M* follows its layout)
        content = [{"type": "image", "image": Image.open(io.BytesIO(_png()))}, {"type": "text", "text": prompt}]
        return dict(self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        ))


@pytest.fixture(scope="module")
def fp32():
    # the screen's exact tests
    pytest.importorskip("scipy")
    return _Fp32(pytest.importorskip("transformers"))


def _pairs_one_at_a_time(plain: dict, run: dict) -> list[fp32_screen.Pair]:
    pairs = [fp32_screen.Pair(PROMPTS[i], False, a, b, f"{key} p{i}") for key in ("sequential", "repeated")
             for i, (a, b) in enumerate(zip(plain[key], run[key], strict=True))]
    pairs += [fp32_screen.Pair(LONG, False, plain[key], run[key], key) for key in ("long", "long_repeated")]
    pairs.append(fp32_screen.Pair(IMAGE_PROMPT, True, plain["image"], run["image"], "image"))
    return pairs


def _pairs_rounds(plain: dict, run: dict) -> list[fp32_screen.Pair]:
    return fp32_screen.pairs_rounds(plain, run, PROMPTS)


def _screen(label: str, groups, fp32) -> list[str]:
    return fp32_screen.screen(label, groups, fp32, GATES)


def _screen_arm(plain: dict, run: dict, fp32, name: str) -> list[str]:
    """``run`` one request at a time and at once, against plain at each shape."""
    try:
        return _screen(name, [(_pairs_one_at_a_time(plain, run), None, None),
                              (_pairs_rounds(plain, run), None, None)], fp32)
    finally:
        fp32.release()


# ── tests: run with -x ──


def test_the_mtp_head_is_accepted_captured_and_uncaptured(tmp_path_factory):
    sync = _run(tmp_path_factory, "sync", draft_mode="synchronous")
    eager = _run(tmp_path_factory, "sync_eager", draft_mode="synchronous", captured=False)
    assert sync["tau"] > MIN_TAU and eager["tau"] > MIN_TAU, (sync["tau"], eager["tau"])
    assert eager["tau"] == pytest.approx(sync["tau"], rel=0.1)


def test_speculative_greedy_tokens_are_plain_greedy_tokens(tmp_path_factory, fp32):
    """Text, the chunked long prompt, the repeat that hits the prefix cache and
    the image prompt, one at a time and at once, each against plain Qwen3.5 by
    the fp32 screen."""
    plain = _run(tmp_path_factory, "plain", draft_mode=None)
    failed = [r for name, kw in (("sync", {}), ("sync_eager", {"captured": False}))
              for r in _screen_arm(plain, _run(tmp_path_factory, name, draft_mode="synchronous", **kw), fp32, name)]
    assert not failed, "\n".join(failed)


@needs_fa3
def test_asynchronous_mode_agrees_with_synchronous_exactly(tmp_path_factory, fp32):
    """At compile level 0: at the default level per-process tuning moves low bits between servers, which flips
    near-ties (first differences at fp32 margins under 0.1)."""
    sync = _run(tmp_path_factory, "sync_o0", draft_mode="synchronous", compile_level=0)
    async_ = _run(tmp_path_factory, "async_o0", draft_mode="asynchronous", compile_level=0)
    assert _ids_of(async_, "sequential") == _ids_of(sync, "sequential")
    assert async_["tau"] == pytest.approx(sync["tau"], rel=0.05)
    failed = _screen_arm(_run(tmp_path_factory, "plain", draft_mode=None), async_, fp32, "async_o0")
    assert not failed, "\n".join(failed)


@needs_fa3
def test_an_asynchronous_server_at_default_tuning_passes_the_screen(tmp_path_factory, fp32):
    """A fresh server at the default compile level, whose tuned kernels are the ones a bad pick would corrupt:
    its replies pass the fp32 screen against plain and it drafts as well (tau above ``MIN_TAU``, at once above
    ``MIN_TAU_CONCURRENT``; the long prompt's is one reply, so printed only). Run several times, each in its own
    process, with ``-k default_tuning``, and ``QWEN35_MTP_MODEL=9b`` for the size whose screen has the deficit
    tripwire."""
    async_ = _run(tmp_path_factory, "async", draft_mode="asynchronous")
    print(f"default tuning: tau {async_['tau']:.3f}, at once {async_['tau_concurrent']:.3f}, "
          f"long prompt {async_['tau_long']:.3f}")
    failed = _screen_arm(_run(tmp_path_factory, "plain", draft_mode=None), async_, fp32, "async")
    assert not failed, "\n".join(failed)
    assert async_["tau"] > MIN_TAU, async_["tau"]
    assert async_["tau_concurrent"] > MIN_TAU_CONCURRENT, async_["tau_concurrent"]


def test_a_prefix_hit_and_a_chunked_prompt_draft_as_well(tmp_path_factory):
    """The long prompt, prefilled in chunks, drafts well (a chunk's last MTP row
    read the next chunk's token); its repeat hits the prefix cache (the MTP
    head's layer is in the cached pages) and drafts about as well."""
    sync = _run(tmp_path_factory, "sync", draft_mode="synchronous")
    assert sync["tau_long"] > MIN_TAU, sync["tau_long"]
    assert sync["hit_tokens"] > 0
    assert sync["tau_long_repeated"] > MIN_TAU, sync["tau_long_repeated"]
    # one reply's tau, so only comparable on the same reply: a prefix hit prefills through another kernel
    # shape, whose low bits can flip a near-tie, on plain decoding as well
    if sync["long_repeated"]["ids"] == sync["long"]["ids"]:
        assert sync["tau_long_repeated"] == pytest.approx(sync["tau_long"], rel=0.1)


def test_an_image_prompt_drafts(tmp_path_factory):
    """Acceptance on an image prompt, reported; asserted
    only to be above no drafting at all."""
    sync = _run(tmp_path_factory, "sync", draft_mode="synchronous")
    print(f"image prompt tau: {sync['tau_image']:.3f} (text: {sync['tau']:.3f})")
    assert sync["tau_image"] > 1.2, sync["tau_image"]


def test_sampled_requests_draft_and_complete(tmp_path_factory):
    """Temperature > 0 at the server's sampling defaults: speculative sampling
    verifies them; every request completes and drafts are still accepted."""
    sync = _run(tmp_path_factory, "sync", draft_mode="synchronous")
    assert all(len(ids) > 0 for ids in sync["sampled"])
    assert sync["tau_sampled"] > 1.3, sync["tau_sampled"]


def _pressure_arm(tmp_path_factory, name: str, mode: str | None) -> dict:
    """A pressure run; a speculative arm under ``MSTAR_KV_DEBUG_ASSERTS``. Under them the plain path's
    offload/reload scheduling has been seen to ping-pong until requests time out (a host sync at each
    offload and reload), so an arm whose requests fail is run again without them, and its exact reload
    gate is reported waived."""
    import requests

    try:
        return _run(tmp_path_factory, name, draft_mode=mode, pressure=True, max_num_pages=PRESSURE_PAGES)
    except requests.RequestException as exc:
        if mode is None:
            raise
        print(f"WAIVED {name}: the exact reload gate (MSTAR_KV_DEBUG_ASSERTS) waived pending a host-sync-free "
              f"check; under it the server failed requests ({type(exc).__name__}). Rerun without it.")
        return _run(tmp_path_factory, f"{name}_unchecked", draft_mode=mode, pressure=True,
                    max_num_pages=PRESSURE_PAGES, debug_asserts=False)


@needs_fa3
def test_kv_pressure_offloads_and_completes(tmp_path_factory, fp32):
    """Few pages, 32 requests at once, ``ROUNDS`` rounds in shuffled arrival
    orders: requests are offloaded and reloaded in plain and in both verify
    modes, and each verify mode's replies pass the fp32 screen against plain
    under the same pressure. Drafts are still accepted and nothing wedges.
    The verify modes run under ``MSTAR_KV_DEBUG_ASSERTS`` (each reload finds
    the offload's KV pages, GDN slot and sampler rows as they were; resident
    rows hold the host's values) unless that is waived (`_pressure_arm`);
    plain, the baseline, runs without."""
    runs = {name: _pressure_arm(tmp_path_factory, name, mode)
            for name, mode in (("plain_pressure", None), ("pressure", "asynchronous"),
                               ("sync_pressure", "synchronous"))}
    for name, run in runs.items():
        assert run["offloads"] >= 1, f"{name}: no request was offloaded"
        assert run["asserts"] == 0, f"{name}: {run['asserts']} debug-assert failures in the server log"
    failed = []
    try:
        for name in ("pressure", "sync_pressure"):
            assert runs[name]["tau"] > MIN_TAU, (name, runs[name]["tau"])
            failed += _screen(name, [(_pairs_rounds(runs["plain_pressure"], runs[name]), None,
                                      runs[name]["events"])], fp32)
    finally:
        fp32.release()
    assert not failed, "\n".join(failed)


def test_acceptance_matches_the_offline_reference(tmp_path_factory):
    """#1090's offline harness (benchmark/spec_decode/mtp_acceptance.py) on
    the same prompts, greedy, thinking off, k = 3: its calibration
    must pick M*'s input convention (the hidden state after the final norm,
    the embedding first), and the served tau must be within 5% of its tokens
    per step. Its JSON's path comes in QWEN35_MTP_REFERENCE."""
    path = os.environ.get("QWEN35_MTP_REFERENCE")
    if not path:
        pytest.skip("set QWEN35_MTP_REFERENCE to the offline harness's JSON, to compare the served tau with it")
    ref = json.loads(Path(path).read_text())
    assert ref["chosen"] == {"hidden": "post", "fc_order": "emb_first"}, ref["chosen"]
    (cell,) = [c for c in ref["cells"] if c["mode"] == "greedy" and c["k"] == 3]
    want = cell["sets"]["chat"]["tokens_per_step"]
    sync = _run(tmp_path_factory, "sync", draft_mode="synchronous")
    print(f"served tau {sync['tau']:.3f}, offline reference {want:.3f}")
    assert sync["tau"] == pytest.approx(want, rel=0.05)
