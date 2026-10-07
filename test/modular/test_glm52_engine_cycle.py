"""GLM-5.2 on the resource-pools engine, CPU, reduced config."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _cpu_rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


def _cpu_flashinfer() -> types.ModuleType:
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    return fi


try:
    import flashinfer  # noqa: F401  (the real one stays for later test files)
except ImportError:
    sys.modules["flashinfer"] = _cpu_flashinfer()


@pytest.fixture(autouse=True)
def _force_cpu_flashinfer(monkeypatch):
    # forced per test: on a box with real flashinfer an earlier import would
    # otherwise route CPU tensors into GPU kernels
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())


from mstar.engine.resources import StepContext  # noqa: E402
from mstar.model.glm52._testing import build_cpu_resources  # noqa: E402
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.config import (  # noqa: E402
    KV_RESOURCE,
    SAMPLER_RESOURCE,
    Glm52ModelConfig,
)
from mstar.model.glm52.quantization import process_weights_after_loading  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402


def _cfg(mla_absorb: bool) -> Glm52ModelConfig:
    cfg = Glm52ModelConfig.reduced()
    cfg.num_hidden_layers = 4
    cfg.mla_absorb = mla_absorb
    return cfg


def _fwd_info(rid: str, max_tokens: int, ignore_eos: bool) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=rid,
        max_tokens=max_tokens,
        resource_configs={SAMPLER_RESOURCE: SimpleNamespace(
            ignore_eos=ignore_eos, temperature=0.0, repetition_penalty=1,
        )},
        dynamic_loop_iter_counts={},
    )


class _Driver:
    """The engine's per-step cycle for one node, over real resources."""

    def __init__(self, sub: Glm52LLMSubmodule, cfg: Glm52ModelConfig, rids: list[str]):
        self.sub = sub
        self.resources, self.runner = build_cpu_resources(cfg, rids)
        sub.bind_node_resources(self.resources)

    def step(self, walk: str, batch: dict[str, tuple[SimpleNamespace, torch.Tensor]]):
        rids = list(batch)
        inputs = [
            self.sub.prepare_inputs(walk, info, {"text_inputs": [text]})
            for info, text in batch.values()
        ]
        step = self.sub.declare_step(walk, rids, inputs)
        ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False)
        step.set_ctx(ctx)
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        engine_inputs = ModelInputsFromEngine(
            request_ids=rids,
            per_request_info={rid: info for rid, (info, _) in batch.items()},
            resources=self.resources,
            step=step,
        )
        kw = self.sub.preprocess(walk, engine_inputs, inputs)
        outs = self.sub.forward_batched(walk, engine_inputs, **kw)
        self.runner.commit(step)
        for rid, (info, _) in batch.items():
            self.sub.postprocess(rid, info, outs[rid])
        return outs


def _drive(driver: _Driver, prompt: torch.Tensor, info, max_steps=64) -> torch.Tensor:
    rid = info.request_id
    emitted = []
    walk, text, decode_step = "prefill", prompt, 0
    for _ in range(max_steps):
        out = driver.step(walk, {rid: (info, text)})[rid]
        emitted.append(out["new_token"][0])
        if walk == "decode":
            info.dynamic_loop_iter_counts["decode_loop"] = decode_step
            decode_step += 1
        if driver.sub.check_stop(rid, info, out):
            break
        walk, text = "decode", out["text_inputs"][0]
    return torch.cat(emitted)


def _model(cfg: Glm52ModelConfig, seed: int = 0) -> Glm52ForCausalLM:
    """Every parameter randomized: the MoE expert containers are raw
    ``torch.empty`` at construction (the loader fills them), and garbage
    there NaNs the logits — a NaN model emits an all-zero argmax stream on
    every path, which makes a stream comparison vacuous.
    """
    torch.manual_seed(seed)
    model = Glm52ForCausalLM(cfg)
    for name, p in model.named_parameters():
        if "norm" in name:
            p.data.normal_(1.0, 0.02)
        elif name.endswith("gate.weight") or "e_score_correction_bias" in name:
            p.data.normal_(0, 1.0)
        else:
            p.data.normal_(0, 0.05)
    process_weights_after_loading(model, torch.device("cpu"))
    return model.eval()


def test_kv_length_tracks_the_stream():
    cfg = _cfg(True)
    model = _model(cfg)
    prompt = torch.arange(5, dtype=torch.long) + 3
    sub = Glm52LLMSubmodule(model, cfg)
    driver = _Driver(sub, cfg, ["r0"])
    info = _fwd_info("r0", 20, True)
    kv = driver.resources[KV_RESOURCE]
    out = driver.step("prefill", {"r0": (info, prompt)})["r0"]
    total = prompt.numel()
    assert kv.stored_len("r0") == total
    text = out["text_inputs"][0]
    for _ in range(4):
        out = driver.step("decode", {"r0": (info, text)})["r0"]
        total += 1
        assert kv.stored_len("r0") == total
        assert out["text_inputs"][0].numel() == 1
        text = out["text_inputs"][0]


@pytest.mark.parametrize("mla_absorb", [True, False], ids=["absorbed", "naive"])
def test_batch_of_two_matches_single_streams(mla_absorb):
    """Two requests in one batch (different prompts) emit what each emits
    alone."""
    cfg = _cfg(mla_absorb)
    model = _model(cfg)
    prompts = {"a": torch.arange(5, dtype=torch.long) + 3, "b": torch.arange(3, dtype=torch.long) + 9}
    singles = {}
    for rid, prompt in prompts.items():
        sub = Glm52LLMSubmodule(model, cfg)
        singles[rid] = _drive(_Driver(sub, cfg, [rid]), prompt, _fwd_info(rid, 14, True))
    # a real stream, not a degenerate one (an uninitialized model emits 0s)
    assert len(set(singles["a"].tolist())) > 3

    sub = Glm52LLMSubmodule(model, cfg)
    driver = _Driver(sub, cfg, list(prompts))
    infos = {rid: _fwd_info(rid, 14, True) for rid in prompts}
    emitted = {rid: [] for rid in prompts}
    texts = {}
    for rid, prompt in prompts.items():
        out = driver.step("prefill", {rid: (infos[rid], prompt)})[rid]
        emitted[rid].append(out["new_token"][0])
        texts[rid] = out["text_inputs"][0]
    live = set(prompts)
    steps = dict.fromkeys(prompts, 0)
    for _ in range(20):
        if not live:
            break
        outs = driver.step("decode", {rid: (infos[rid], texts[rid]) for rid in sorted(live)})
        for rid in sorted(live):
            emitted[rid].append(outs[rid]["new_token"][0])
            texts[rid] = outs[rid]["text_inputs"][0]
            infos[rid].dynamic_loop_iter_counts["decode_loop"] = steps[rid]
            steps[rid] += 1
            if sub.check_stop(rid, infos[rid], outs[rid]):
                live.discard(rid)
    for rid in prompts:
        got = torch.cat(emitted[rid])
        assert torch.equal(got, singles[rid]), f"{rid}: {got.tolist()} vs {singles[rid].tolist()}"


def test_declare_step_shapes():
    cfg = _cfg(True)
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    inputs = [sub.prepare_inputs("prefill", _fwd_info("r0", 8, True), {"text_inputs": [torch.tensor([1, 2, 3])]})]
    step = sub.declare_step("prefill", ["r0"], inputs)
    assert set(step.keys()) == {"kv", "attn", "sampler"}
    assert step.segments[0].span == 3


def test_last_rows_slices_padded_prefill_to_real_requests():
    # A captured prefill plan pads qo_indptr to the bucket and tail-fills it
    # with the last real offset. _last_rows must return one row per real
    # request, not one per bucket slot: a padded slot duplicates the final
    # request's row and hands the sampler more logit rows than it has
    # per-request params (an out-of-bounds read on the eager sampler).
    cfg = _cfg(mla_absorb=True)
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    # three real requests (lengths 5, 7, 3) captured in a bucket of four
    qo_indptr = torch.tensor([0, 5, 12, 15, 15])
    hidden = torch.randn(15, cfg.hidden_size)
    sub._attn = lambda _ei: SimpleNamespace(qo_indptr_buf=lambda _slot: qo_indptr)
    engine_inputs = SimpleNamespace(request_ids=["a", "b", "c"])
    rows = sub._last_rows(engine_inputs, hidden, {})
    assert rows.shape[0] == 3
    assert torch.equal(rows, hidden[[4, 11, 14]])


# ── the context limit ────────────────────────────────────────────────────


def _window_cfg(limit: int) -> Glm52ModelConfig:
    """Reduced config whose context limit (index_topk, DSA off) is ``limit``
    rows: two CPU pages of 8, so a row past the limit needs a third page."""
    cfg = _cfg(mla_absorb=True)
    cfg.index_topk = limit
    return cfg


def test_stream_stops_cleanly_at_the_context_limit():
    """A request whose max_tokens outruns the context limit stops one row short
    of it: the prefill's preprocess caps its budget at limit - prompt, so the
    guard never fires inside the batch and no third KV page is taken."""
    limit, prompt_len = 16, 5
    cfg = _window_cfg(limit)
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    driver = _Driver(sub, cfg, ["r0"])
    prompt = torch.arange(prompt_len, dtype=torch.long) + 3
    stream = _drive(driver, prompt, _fwd_info("r0", 40, True))
    kv = driver.resources[KV_RESOURCE]
    expect_tokens = limit - prompt_len
    assert sub._token_budget["r0"] == expect_tokens
    assert stream.numel() == expect_tokens
    assert kv.stored_len("r0") == limit - 1
    assert len(kv._streams["r0"]["main"].page_indices) == limit // 8


def test_the_step_after_the_context_stop_still_fits():
    # the worker may launch step N+1 before check_stop(N) lands; its row must fit
    limit, prompt_len = 16, 5
    cfg = _window_cfg(limit)
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    driver = _Driver(sub, cfg, ["r0"])
    info = _fwd_info("r0", 40, True)
    out = driver.step("prefill", {"r0": (info, torch.arange(prompt_len) + 3)})["r0"]
    step = 0
    while not sub.check_stop("r0", info, out):
        out = driver.step("decode", {"r0": (info, out["text_inputs"][0])})["r0"]
        info.dynamic_loop_iter_counts["decode_loop"] = step
        step += 1
    driver.step("decode", {"r0": (info, out["text_inputs"][0])})
    assert driver.resources[KV_RESOURCE].stored_len("r0") == limit


def test_a_prompt_without_room_fails_alone():
    cfg = _window_cfg(16)
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    fits, full = torch.arange(15) + 3, torch.arange(16) + 3
    sub.prepare_inputs("prefill", _fwd_info("a", 8, True), {"text_inputs": [fits]})
    with pytest.raises(RuntimeError, match="context limit"):
        sub.prepare_inputs("prefill", _fwd_info("b", 8, True), {"text_inputs": [full]})


def test_budget_within_the_window_is_untouched():
    cfg = _window_cfg(64)
    model = _model(cfg)
    sub = Glm52LLMSubmodule(model, cfg)
    driver = _Driver(sub, cfg, ["r0"])
    prompt = torch.arange(5, dtype=torch.long) + 3
    stream = _drive(driver, prompt, _fwd_info("r0", 9, True))
    assert sub._token_budget["r0"] == 9
    assert stream.numel() == 9
