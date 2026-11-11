"""Qwen3 with a separate draft model one token behind it (``method: draft_model``) on the device: a toy target and a
toy draft model of another KV geometry (its own storage in the pool), drafting in one-row passes.

- Greedy tokens are plain greedy decoding's, synchronous on FlashInfer (eager and captured) and asynchronous on FA3.
- Each verify step's drafts are the draft model's own greedy chain over the true history, recomputed densely in
  float64: a lagged write, extent or position off by one changes a draft, so this checks the lag and the row-0 sink
  end to end (a near-tie in bf16 ends a chain's comparison). The drafts are read through a copy inside the sampler's
  ``agree``, which a captured graph replays. The toy draft model's drafts are almost never accepted, so a second
  chain case drafts with the target itself: nearly every block is accepted in full, and row 0's K/V is kept
  (`DRAFT_KEEP_FIRST`) on every step after the first.
- Eager only, where a Python wrapper can swap the drafts: an oracle (the target's own next tokens) is accepted in full,
  so row 0 is kept every step after the first; a refuser is never accepted. Both still decode plainly. (Swapped
  drafts leave the drafter's KV holding its own drafts, so these check acceptance and output, not the chain.)
"""
from __future__ import annotations

import json

import pytest
import torch
from qwen3_cpu import (
    STOP,
    Driver,
    _DenseAttention,
    _DensePositions,
    copy_weights,
    device_specs,
    engine,
    make_model,
    patch_cpu,
    reference_greedy,
    write_checkpoints,
)
from qwen3_cpu import TARGET as TOY

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.components.draft_model import DraftModelDrafter
from mstar.model.components.qwen3_lm import Qwen3DenseLM, Qwen3LMConfig
from mstar.model.qwen3.submodules import Qwen3LMSubmodule

VOCAB = 512
TARGET = dict(hidden_size=256, num_attention_heads=2, num_key_value_heads=1, head_dim=128, intermediate_size=512,
              vocab_size=VOCAB, torch_dtype="bfloat16")
# another KV geometry (2 heads of 64 against the target's 1 of 128): a storage of its own over the pool's pages
DRAFT = dict(architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=128, num_hidden_layers=2,
             num_attention_heads=4, num_key_value_heads=2, head_dim=64, intermediate_size=256, vocab_size=VOCAB,
             rms_norm_eps=1e-6, rope_theta=10_000.0, max_position_embeddings=512, tie_word_embeddings=True,
             torch_dtype="bfloat16", use_sliding_window=False)
# the target toy itself as a draft model: its config here, its weights by `_same_weights`
SELF = {**TOY, **TARGET}
PROMPTS = {1: [5, 9, 2, 7, 11, 3], 2: [17], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4: [88, 90]}
MAX = 24
K = 3
NEAR_TIE = 0.1
# (mode, captured, the drafter binding's verify backend)
SETUPS = [("synchronous", False, "flashinfer"), ("synchronous", True, "flashinfer"), ("asynchronous", True, "fa3")]


def _reason(mode: str) -> str | None:
    if not torch.cuda.is_available():
        return "needs CUDA"
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return "needs flashinfer"
    if mode == "asynchronous":
        from mstar.engine.resources.attn.fa3_paged import fa3_paged_unavailable_reason

        return fa3_paged_unavailable_reason(torch.device("cuda"), 64, torch.bfloat16)
    return None


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    patch_cpu(monkeypatch)
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    monkeypatch.setattr(Qwen3LMSubmodule, "DECODE_CAPTURE_BATCH_SIZES", [1, 2, 4, 8])
    monkeypatch.setattr(Qwen3LMSubmodule, "PREFILL_CAPTURE_BATCH_SIZES", [1, 2, 4, 8])
    monkeypatch.setattr(Qwen3LMSubmodule, "PREFILL_TOKEN_BUCKETS", [64, 128])


def _model(root, mode: str, captured: bool, config: dict = DRAFT):
    target, _ = write_checkpoints(root, target=TARGET)
    draft = root / "draft"
    draft.mkdir(parents=True, exist_ok=True)
    (draft / "config.json").write_text(json.dumps(config))
    return make_model(target, {"method": "draft_model", "model": str(draft), "num_draft_tokens": K, "mode": mode},
                      accelerator_graph=captured)


def _engine(model, backend: str, captured: bool, prepare=None):
    specs = device_specs(model)
    for spec in specs:
        if spec.resource_key == model.drafter_decl.attn_key:
            # parsed into a VerifyBackend, as a deployment's override is
            spec.apply_yaml_overrides(verify_backend=backend)
    return engine(model, device="cuda", dtype=torch.bfloat16, warmup=captured, specs=specs, prepare=prepare)


def _same_weights(sub) -> None:
    copy_weights(sub.drafter.lm, sub.model)


class _Probe:
    """Each driver verify step's rows, their anchors (the bonus token's index in the reply) and the drafts, copied out
    inside ``agree``: a captured graph replays that copy, as it replays the drafter."""

    def __init__(self, monkeypatch, rows: int = 8):
        self.buf = torch.zeros(rows, K, dtype=torch.long, device="cuda")
        self.steps: list[list[tuple[int, int, list[int]]]] = []
        real_agree, real_step, probe = SamplerResource.agree, Driver.step, self

        @torch.compiler.disable
        def agree(self, values):
            out = real_agree(self, values)
            if out.dim() == 2 and out.shape[1] == K:
                probe.buf[: out.shape[0]].copy_(out)
            return out

        def step(self, rids):
            anchors = {rid: len(self.out[rid]) - 1 for rid in rids}
            stopped = real_step(self, rids)
            drafts = probe.buf[: len(rids)].tolist()
            probe.steps.append([(rid, anchors[rid], drafts[i]) for i, rid in enumerate(rids)])
            return stopped

        monkeypatch.setattr(SamplerResource, "agree", agree)
        monkeypatch.setattr(Driver, "step", step)


def _swap(monkeypatch, replace):
    """Eager steps only: the drafter's passes run, then ``replace(rid, anchor)`` stands in for its drafts."""
    real_propose, real_step = DraftModelDrafter.propose, Driver.step
    anchors: dict[int, int] = {}

    def propose(self, inputs):
        drafts = real_propose(self, inputs)
        out = drafts.clone()
        for i, rid in enumerate(inputs.request_ids):
            if rid in anchors:
                out[i] = torch.tensor(replace(rid, anchors[rid]), device=out.device)
        return out

    def step(self, rids):
        anchors.clear()
        anchors.update({rid: len(self.out[rid]) - 1 for rid in rids})
        return real_step(self, rids)

    monkeypatch.setattr(DraftModelDrafter, "propose", propose)
    monkeypatch.setattr(Driver, "step", step)


def _run(model, eng):
    driver = Driver(eng, model)
    for rid, prompt in PROMPTS.items():
        driver.add(rid, prompt, MAX)
    driver.prefill(list(PROMPTS), PROMPTS)
    driver.run(list(PROMPTS))
    return driver


def _plain(model, eng, extra: int = 0) -> dict[int, tuple[list[int], list[float]]]:
    """Each prompt's plain greedy reply recomputed densely in float64, with each token's top-2 log-prob margin."""
    sub = eng._submodules["LLM"].submodule
    out = {}
    for rid, prompt in PROMPTS.items():
        margins: list[float] = []
        out[rid] = (reference_greedy(model, prompt, MAX + extra, like=sub, margins=margins), margins)
    return out


def _check_plain(plain, replies) -> None:
    """Each reply is the plain one up to a bf16 near-tie: where they first differ, the reference's margin is under
    ``NEAR_TIE`` (the margin rule)."""
    for rid, reply in replies.items():
        ref, margins = plain[rid]
        n = min(len(ref), len(reply))
        at = next((j for j in range(n) if ref[j] != reply[j]), None)
        assert at is None or margins[at] <= NEAR_TIE, (rid, at, ref, reply)


def _dense_draft(eng, config: dict):
    """The draft model's own weights in float64 on the CPU, over dense causal attention."""
    sub = eng._submodules["LLM"].submodule
    lm = Qwen3DenseLM(Qwen3LMConfig.from_hf(config), attn_key="attn", kv_key="kv_cache", pos_key="rope")
    have = dict(sub.drafter.lm.named_parameters())
    for name, p in lm.named_parameters():
        p.data = have[name].detach().to("cpu", torch.float64)
    attn, pos = _DenseAttention(), _DensePositions()
    for layer in lm.layers:
        layer.self_attn.bind_resources({"attn": attn, "kv_cache": attn, "rope": pos})
    return lm.double().eval()


def _check_drafts(eng, out, steps, config: dict = DRAFT) -> int:
    """Each step's drafts against the draft model's dense greedy chain from the row's history up to its bonus token;
    a chain compares up to its first near-tie. Returns the drafts compared."""
    lm = _dense_draft(eng, config)
    compared = 0
    for rows in steps:
        for rid, anchor, drafts in rows:
            seq = PROMPTS[rid] + out[rid][: anchor + 1]
            with torch.no_grad():
                for j in range(K):
                    logp = lm.logits(lm(lm.embed(torch.tensor(seq)), label="main"))[-1].log_softmax(-1)
                    top = logp.topk(2)
                    if float(top.values[0] - top.values[1]) < NEAR_TIE:
                        break
                    want = int(top.indices[0])
                    assert drafts[j] == want, (rid, anchor, j, drafts, want)
                    compared += 1
                    seq = seq + [want]
    return compared


@pytest.mark.parametrize("mode, captured, backend", SETUPS, ids=str)
def test_greedy_tokens_are_plain_and_drafts_are_the_draft_models_own_chain(tmp_path, monkeypatch, mode, captured,
                                                                            backend):
    if (reason := _reason(mode)) is not None:
        pytest.skip(reason)
    model = _model(tmp_path, mode, captured)
    # before the engine captures: its graphs record the probe's copy
    probe = _Probe(monkeypatch)
    eng = _engine(model, backend, captured)
    driver = _run(model, eng)
    _check_plain(_plain(model, eng), driver.out)
    assert _check_drafts(eng, driver.out, probe.steps) >= len(PROMPTS) * K
    if backend == "flashinfer":
        # every pass of a slot plans on that slot's one float workspace
        held = eng._resources[model.drafter_decl.attn_key]._workspaces._workspaces
        assert not [key for key in held if ".lag" in key and not key.startswith("main.lag_")], sorted(held)


@pytest.mark.parametrize("mode, captured, backend", SETUPS, ids=str)
def test_a_draft_model_that_is_the_target_keeps_row_0_on_its_own_chain(tmp_path, monkeypatch, mode, captured,
                                                                         backend):
    """The draft model is the target toy (its config and weights): nearly every block is accepted in full, so
    `DRAFT_KEEP_FIRST` keeps row 0's K/V (the token before the block, which no earlier block wrote) on every step
    after the first, and each step's drafts are still the dense chain's. A keep flag that sends that write to the
    sink leaves the position unwritten, and the chain leaves the dense one."""
    if (reason := _reason(mode)) is not None:
        pytest.skip(reason)
    model = _model(tmp_path, mode, captured, SELF)
    probe = _Probe(monkeypatch)
    eng = _engine(model, backend, captured, prepare=_same_weights)
    driver = _run(model, eng)
    _check_plain(_plain(model, eng), driver.out)
    assert _check_drafts(eng, driver.out, probe.steps, SELF) >= len(PROMPTS) * K
    rows = [(rid, at, drafts) for step in probe.steps for rid, at, drafts in step]
    full = sum(drafts == driver.out[rid][at + 1: at + 1 + K] for rid, at, drafts in rows)
    # all but a bf16 near-tie's block, and each row's last, which may stop mid-block
    assert full >= 0.8 * len(rows), (full, len(rows))


def test_an_oracle_is_accepted_in_full_and_a_refuser_never(tmp_path, monkeypatch):
    """Eager, synchronous on FlashInfer, under the KV debug asserts: an oracle's drafts (the target's own next
    tokens) are nearly all accepted, a refuser's never, and both decode plainly. Acceptance and output only: the
    swapped drafts leave the drafter's KV holding its own, so row 0's K/V is checked by the chain cases."""
    if (reason := _reason("synchronous")) is not None:
        pytest.skip(reason)
    model = _model(tmp_path / "oracle", "synchronous", False)
    eng = _engine(model, "flashinfer", False)
    plain = _plain(model, eng, extra=K + 1)
    ahead = {rid: reply + [STOP] * (K + 1) for rid, (reply, _) in plain.items()}
    with monkeypatch.context() as patch:
        _swap(patch, lambda rid, at: ahead[rid][at + 1: at + 1 + K])
        driver = _run(model, eng)
    _check_plain(plain, driver.out)
    draft = eng._drafts["LLM"]
    # nearly all: a row's last step may stop mid-block, and past a bf16 near-tie flip the float64 oracle's drafts are
    # the other branch's
    assert draft.accepted >= 0.8 * draft.drafted, (draft.accepted, draft.drafted)

    model = _model(tmp_path / "refuser", "synchronous", False)
    eng = _engine(model, "flashinfer", False)
    plain = _plain(model, eng, extra=K + 1)
    with monkeypatch.context() as patch:
        # a first draft one past the target's next token is never accepted, whatever follows it
        _swap(patch, lambda rid, at: [((plain[rid][0] + [STOP])[at + 1] + 1) % VOCAB] * K)
        driver = _run(model, eng)
    _check_plain(plain, driver.out)
    assert eng._drafts["LLM"].accepted == 0
