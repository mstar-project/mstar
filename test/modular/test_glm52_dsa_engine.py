"""DSA engine half: per-request state, guard states, IndexShare threading and
the prefix property across index_topk — over the real ``KVManager`` (MLA
latent layout), the index-key store and the MLA attention resource on CPU.
"""
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
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())


from mstar.engine.resources import StepContext  # noqa: E402
from mstar.engine.resources.attn.flashinfer_mla import flashinfer_mla_supports  # noqa: E402
from mstar.model.glm52 import dsa_paged  # noqa: E402
from mstar.model.glm52._testing import (  # noqa: E402
    build_cpu_resources,
    build_random_model,
)
from mstar.model.glm52.components.attention import Glm52MLAAttention  # noqa: E402
from mstar.model.glm52.config import (  # noqa: E402
    ATTN_RESOURCE,
    INDEX_KV_RESOURCE,
    KV_RESOURCE,
    SAMPLER_RESOURCE,
    Glm52ModelConfig,
)
from mstar.model.glm52.dsa_paged import Glm52DsaPagedContext  # noqa: E402
from mstar.model.glm52.glm52_model import Glm52Model  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine  # noqa: E402


def _ctx(needs_selection: bool) -> Glm52DsaPagedContext:
    """One decode row at position 6 of a single-page request."""
    one = torch.ones(1, dtype=torch.int32)
    return Glm52DsaPagedContext(
        row_req=one - 1, lens=one * 7, host_lens=[7], kv_table=one[None], index_table=one[None],
        spans=[(0, 1, 0)], width=7, page_size=8, topk=4, needs_selection=needs_selection)


def test_cleanup_request_drops_per_request_state():
    """Retirement leaves NO per-request growth: the engine's request removal
    calls cleanup_request on every managed submodule, which must drop the
    prefix-hit mark, the MTP counters and the token budget."""
    sub = object.__new__(Glm52LLMSubmodule)
    sub.config = Glm52ModelConfig.reduced()
    sub.request_states = {}
    sub._prefix_hits = {"r0"}
    sub._mtp_emitted = {"r0": 7}
    sub._mtp_checked = {"r0": 7}
    sub._mtp_max_tokens = {"r0": 32}
    sub._mtp_ignore_eos = {"r0": False}
    sub._token_budget = {"r0": 20}
    sub._mtp_pending, sub._mtp_verdict, sub._mtp_trimmed = {"r0": (3, 0, 0)}, {"r0": ()}, {"r0": 3}
    sub._mtp_slot, sub._mtp_free_slots = {"r0": 5}, []

    sub.request_state("r0")  # base per-request state exists too
    sub.cleanup_request("r0")
    assert sub._prefix_hits == set()
    assert sub.request_states == {}
    assert sub._mtp_emitted == {}
    assert sub._mtp_max_tokens == {}
    assert sub._mtp_ignore_eos == {}
    assert sub._token_budget == {}
    assert not (sub._mtp_pending or sub._mtp_verdict or sub._mtp_trimmed or sub._mtp_slot)
    assert sub._mtp_free_slots == [5]  # the seed slot goes back
    sub.cleanup_request("r0")  # idempotent, like the base hook


# ---------------------------------------------------------------------------
# preprocess guard, both flag states
# ---------------------------------------------------------------------------

class _FakeKV:
    def __init__(self, starts, page_size=8):
        self._starts = starts
        self.config = SimpleNamespace(page_size=page_size)

    def stored_len(self, rid, label="main"):
        return self._starts[rid]


def _fake_step(tables):
    """This step's plans of the latent cache and the index store, as preprocess reads them:
    one view per request, positionally; an eager step."""
    views = [SimpleNamespace(page_idxs=t) for t in tables]
    plan = {key: {"main": SimpleNamespace(views=views)} for key in (KV_RESOURCE, INDEX_KV_RESOURCE)}
    return SimpleNamespace(ctx=SimpleNamespace(
        plan_results=plan, capture=False, slot_lease=None, slot=0))


def _make_submodule(config) -> Glm52LLMSubmodule:
    sub = object.__new__(Glm52LLMSubmodule)
    sub.config = config
    sub._prefix_hits = set()
    return sub


def _preprocess(sub, starts, seq_len, tables=None):
    sub.get_device = lambda: torch.device("cpu")
    inputs = [
        ARNodeInputs(
            input_ids=torch.zeros(seq_len, dtype=torch.long),
            input_seq_len=seq_len,
        )
        for _ in starts
    ]
    engine_inputs = SimpleNamespace(
        request_ids=list(starts),
        resources={key: _FakeKV(starts) for key in (KV_RESOURCE, INDEX_KV_RESOURCE)},
        step=_fake_step(tables or [[0, 1, 2] for _ in starts]),
    )
    return sub.preprocess("prefill", engine_inputs, inputs)


def test_guard_flag_off_unchanged_and_no_ctx():
    cfg = Glm52ModelConfig()
    assert cfg.dsa_long_context is False
    sub = _make_submodule(cfg)
    out = _preprocess(sub, {"r0": 2032}, seq_len=16)  # exactly topk: allowed
    assert Glm52LLMSubmodule._dsa_ctx(out) is None
    with pytest.raises(RuntimeError, match="dsa_long_context"):
        _preprocess(sub, {"r0": 2040}, seq_len=16)


def test_guard_flag_on_lifts_cap_to_max_seq_len():
    cfg = Glm52ModelConfig()
    cfg.dsa_long_context = True
    cfg.max_seq_len = 4096
    sub = _make_submodule(cfg)
    topk = cfg.index_topk

    # Decode one past topk: allowed, and the batch is marked for selection.
    ctx = Glm52LLMSubmodule._dsa_ctx(_preprocess(sub, {"r0": topk}, seq_len=1))
    assert isinstance(ctx, Glm52DsaPagedContext)
    assert ctx.needs_selection is True
    assert ctx.host_lens == [topk + 1]
    assert ctx.last_selection is None  # transient starts clean every forward

    # Prefill rows past topk select too: each over its own causal prefix.
    ctx = Glm52LLMSubmodule._dsa_ctx(_preprocess(sub, {"r0": topk}, seq_len=16))
    assert ctx.needs_selection is True
    assert ctx.host_lens == list(range(topk + 1, topk + 17))

    # The new cap is max_seq_len, not topk.
    with pytest.raises(RuntimeError, match="max_seq_len"):
        _preprocess(sub, {"r0": 4096}, seq_len=1)


def test_flag_on_identity_batch_builds_spans_without_selection():
    cfg = Glm52ModelConfig()
    cfg.dsa_long_context = True
    sub = _make_submodule(cfg)
    tables = [[0, 1, 2], [5]]
    ctx = Glm52LLMSubmodule._dsa_ctx(
        _preprocess(sub, {"r0": 0, "r1": 5}, seq_len=4, tables=tables))

    assert ctx.needs_selection is False  # everything fits topk: identity
    assert ctx.spans == [(0, 4, 0), (4, 4, 1)]
    assert ctx.host_lens == [1, 2, 3, 4, 6, 7, 8, 9]
    assert ctx.row_req.tolist() == [0] * 4 + [1] * 4
    assert ctx.kv_table.tolist() == [[0, 1, 2], [5, 0, 0]]
    # Page tables are snapshots: later plan growth must not alias.
    tables[0].append(99)
    assert ctx.kv_table.tolist() == [[0, 1, 2], [5, 0, 0]]


# ---------------------------------------------------------------------------
# Model kwarg plumbing + CUDA-graph gate
# ---------------------------------------------------------------------------

def test_model_kwarg_dsa_long_context():
    m = Glm52Model(model_path_hf="", dsa_long_context=True, max_seq_len=4096)
    assert m.config.dsa_long_context is True
    assert m.config.max_seq_len == 4096

    default = Glm52Model(model_path_hf="", dsa_long_context=True)
    assert default.config.max_seq_len == 8192  # checkpoint generation default

    off = Glm52Model(model_path_hf="")
    assert off.config.dsa_long_context is False
    assert off.config.max_seq_len == off.config.index_topk == 2048

    with pytest.raises(ValueError, match="mla_absorb"):
        Glm52Model(model_path_hf="", config_variant="reduced", dsa_long_context=True)
    mtp = Glm52Model(model_path_hf="", dsa_long_context=True, mtp_num_draft_tokens=2)
    assert mtp.config.dsa_long_context and mtp.config.mtp_num_draft_tokens == 2


def test_model_kwarg_dsa_long_context_refuses_old_flashinfer(monkeypatch):
    monkeypatch.setitem(sys.modules, "flashinfer", types.SimpleNamespace(__version__="0.6.17"))
    with pytest.raises(ValueError, match="FlashInfer >= 0.6.18"):
        Glm52Model(model_path_hf="", dsa_long_context=True)


def test_reduced_variant_grows_trunk_for_mtp():
    # reduced() sizes 2 trunk layers, landing the MTP position on a SHARED
    # indexer slot; a reduced-variant serve yaml with MTP on must still
    # construct. Real variants keep the loud SHARED guard in the constructor.
    from mstar.model.glm52.components.indexer import is_full_indexer_layer
    from mstar.model.glm52.components.mtp import Glm52MTPModule

    off = Glm52Model(model_path_hf="", config_variant="reduced_fp8")
    assert off.config.num_hidden_layers == 2  # untouched without MTP

    for variant in ("reduced", "reduced_fp8"):
        m = Glm52Model(
            model_path_hf="", config_variant=variant, mtp_num_draft_tokens=2)
        pos = m.config.num_hidden_layers
        assert is_full_indexer_layer(m.config, pos), variant
        Glm52MTPModule(m.config)  # the SHARED-slot guard no longer fires


# ---------------------------------------------------------------------------
# The real resources: reduced absorbed model driven a step at a time
# ---------------------------------------------------------------------------

class _Driver:
    """The engine's per-step cycle for one request over the real resources:
    declare -> admit -> plan -> preprocess (builds the DSA inputs exactly as
    serving would, page tables from the live plans) -> forward -> commit."""

    def __init__(self, sub: Glm52LLMSubmodule, cfg: Glm52ModelConfig, rid="r0", page_size=8):
        self.sub, self.rid = sub, rid
        self.resources, self.runner = build_cpu_resources(cfg, [rid], page_size=page_size)
        sub.bind_node_resources(self.resources)

    def step(self, walk: str, ids: torch.Tensor) -> torch.Tensor:
        ar = ARNodeInputs(input_ids=ids, input_seq_len=ids.shape[0])
        step = self.sub.declare_step(walk, [self.rid], [ar])
        step.set_ctx(StepContext(request_ids=(self.rid,), graph_walk=walk, slot=0, capture=False))
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        engine_inputs = ModelInputsFromEngine(
            request_ids=[self.rid], per_request_info={}, resources=self.resources, step=step)
        packed = self.sub.preprocess(walk, engine_inputs, [ar])
        with torch.no_grad():
            logits = self.sub.forward(walk, engine_inputs, **packed)["logits"][0]
        self.runner.commit(step)
        return logits


def _serve_steps(driver: _Driver, prompt, decode_tokens):
    """Prefill + teacher-forced decode. Returns the per-step logits."""
    logits_per_step = []
    for i, ids in enumerate([prompt, *[t.view(1) for t in decode_tokens]]):
        logits_per_step.append(driver.step("prefill" if i == 0 else "decode", ids))
    return logits_per_step


def _longctx_reduced_cfg(topk):
    cfg = Glm52ModelConfig.reduced()
    cfg.mla_absorb = True
    cfg.dsa_long_context = True
    cfg.index_topk = topk
    return cfg


def test_declared_step_under_dsa_adds_the_index_store():
    cfg = _longctx_reduced_cfg(topk=4)
    sub = Glm52LLMSubmodule(language_model=build_random_model(cfg, seed=0), config=cfg)
    ar = ARNodeInputs(input_ids=torch.tensor([1, 2, 3]), input_seq_len=3)
    step = sub.declare_step("prefill", ["r0"], [ar])
    assert set(step.keys()) == {KV_RESOURCE, ATTN_RESOURCE, INDEX_KV_RESOURCE, SAMPLER_RESOURCE}
    assert step.segments[0].span == 3


def test_full_to_shared_threading_order(monkeypatch):
    """Layer 1 (SHARED, no indexer weights) must consume the selection layer
    0 (FULL) published THIS forward — the IndexShare reuse window."""
    cfg = _longctx_reduced_cfg(topk=4)
    model = build_random_model(cfg, seed=0)
    sub = Glm52LLMSubmodule(language_model=model, config=cfg)
    driver = _Driver(sub, cfg)

    trace = []

    def spy(self, dsa_ctx, selection, q_nope, q_pe, kv_c, k_pe):
        trace.append((self.layer_idx, selection))
        return torch.zeros_like(q_nope)

    monkeypatch.setattr(Glm52MLAAttention, "_run_sparse", spy)

    select_calls = []
    original = dsa_paged.select

    def counting(*args, **kwargs):
        result = original(*args, **kwargs)
        select_calls.append(result)
        return result

    monkeypatch.setattr(dsa_paged, "select", counting)

    prompt = torch.randint(0, cfg.vocab_size, (4,))  # ctx 4 == topk: identity
    decode = [torch.tensor(5)]  # ctx 5 > topk: selection engages
    _serve_steps(driver, prompt, decode)

    # One FULL computation, consumed IN ORDER by layer 0 then layer 1, and
    # both consumed the very tensor the FULL layer produced.
    assert len(select_calls) == 1
    assert [layer for layer, _ in trace] == [0, 1]
    assert trace[0][1] is select_calls[0]
    assert trace[1][1] is select_calls[0]

    # The FULL layer stored prompt + decode keys; the store has no layer for a SHARED one.
    index = driver.resources[INDEX_KV_RESOURCE]
    assert index.stored_len("r0") == 5
    assert index.config.num_layers == 1
    assert driver.resources[KV_RESOURCE].stored_len("r0") == 5


def test_shared_layer_without_full_selection_raises():
    """A SHARED layer reached in selection mode with no published rows is a
    threading bug and must fail loudly, not attend densely off-spec."""
    cfg = _longctx_reduced_cfg(topk=4)
    model = build_random_model(cfg, seed=1)
    attn1 = model.model.layers[1].self_attn  # SHARED
    with pytest.raises(RuntimeError, match="no FULL layer ran"):
        attn1(torch.randn(1, cfg.hidden_size) * 0.1, torch.tensor([6]), dsa_ctx=_ctx(True))


def test_naive_path_refuses_dsa_ctx():
    cfg = Glm52ModelConfig.reduced()  # mla_absorb False
    attn = Glm52MLAAttention(cfg, layer_idx=0)
    with pytest.raises(RuntimeError, match="mla_absorb"):
        attn(torch.randn(2, cfg.hidden_size), torch.arange(2), dsa_ctx=_ctx(False))


# ---------------------------------------------------------------------------
# CPU decode across the topk boundary: prefix property + real divergence
# ---------------------------------------------------------------------------

def test_decode_across_topk_prefix_property_and_divergence():
    """Teacher-forced decode with topk=6 vs a dense comparator (topk lifted so selection
    never engages).
    """
    topk = 6
    cfg = _longctx_reduced_cfg(topk)
    model = build_random_model(cfg, seed=3)
    sub = Glm52LLMSubmodule(language_model=model, config=cfg)

    torch.manual_seed(7)
    prompt = torch.randint(0, cfg.vocab_size, (3,))
    decode = list(torch.randint(0, cfg.vocab_size, (6,)))
    # Step i covers context 3 + 1 + i tokens after it: steps 0..2 end at
    # ctx 4, 5, 6 (identity); steps 3..5 end at 7, 8, 9 (sparse).

    driver = _Driver(sub, cfg, page_size=4)  # the sparse steps span 3 pages
    assert not flashinfer_mla_supports(cfg.kv_lora_rank, cfg.qk_rope_head_dim)
    sparse_logits = _serve_steps(driver, prompt, decode)
    assert driver.resources[INDEX_KV_RESOURCE].stored_len("r0") == 3 + len(decode)
    assert driver.resources[KV_RESOURCE].stored_len("r0") == 3 + len(decode)
    sub.cleanup_request("r0")

    cfg.index_topk = 512  # dense comparator: identity regime end to end
    driver2 = _Driver(sub, cfg, page_size=4)  # fresh cache, same weights
    dense_logits = _serve_steps(driver2, prompt, decode)
    sub.cleanup_request("r0")

    for i in range(4):  # prefill + decode steps 0..2
        assert torch.equal(sparse_logits[i], dense_logits[i]), f"step {i}"
    for i in range(4, 7):  # decode steps 3..5: beyond topk
        assert not torch.equal(sparse_logits[i], dense_logits[i]), f"step {i}"
        assert torch.isfinite(sparse_logits[i]).all()
