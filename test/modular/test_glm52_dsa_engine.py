"""DSA engine half: k-store lifecycle, guard states, IndexShare threading,
and the sparse gather path vs an independent masked reference — over the
real ``KVManager`` (MLA latent layout) and MLA attention resource on CPU.
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


if "flashinfer" not in sys.modules:
    sys.modules["flashinfer"] = _cpu_flashinfer()


@pytest.fixture(autouse=True)
def _force_cpu_flashinfer(monkeypatch):
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())


from mstar.engine.resources import StepContext  # noqa: E402
from mstar.model.glm52._testing import build_cpu_resources  # noqa: E402
from mstar.model.glm52.components.attention import Glm52MLAAttention  # noqa: E402
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.config import (  # noqa: E402
    ATTN_RESOURCE,
    KV_RESOURCE,
    SAMPLER_RESOURCE,
    Glm52ModelConfig,
)
from mstar.model.glm52.dsa import (  # noqa: E402
    Glm52DsaForwardContext,
    Glm52DsaKStore,
    Glm52DsaRequestSpan,
)
from mstar.model.glm52.glm52_model import Glm52Model  # noqa: E402
from mstar.model.glm52.quantization import process_weights_after_loading  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Glm52DsaKStore
# ---------------------------------------------------------------------------

def test_kstore_append_history_grow_and_evict():
    store = Glm52DsaKStore()
    d = 16
    first = torch.randn(3, d)
    second = torch.randn(70, d)  # crosses the initial 64-row capacity

    store.append("r0", 0, first, start_pos=0)
    store.append("r0", 0, second, start_pos=3)
    assert store.tokens("r0", 0) == 73
    hist = store.history("r0", 0, 73)
    torch.testing.assert_close(hist, torch.cat([first, second]), rtol=0, atol=0)
    # Windowed view (what a mid-sequence selection scores against).
    torch.testing.assert_close(store.history("r0", 0, 2), first[:2], rtol=0, atol=0)

    # Growth doubled capacity but content survived the buffer swap above;
    # a request/layer never touched reports 0, not KeyError.
    assert store.tokens("r0", 1) == 0
    assert store.tokens("nope", 0) == 0

    store.evict("r0")
    assert store.tracked_requests() == set()
    assert store.tokens("r0", 0) == 0
    store.evict("r0")  # idempotent


def test_kstore_desync_and_overrun_raise():
    store = Glm52DsaKStore()
    store.append("r0", 0, torch.randn(4, 8), start_pos=0)
    with pytest.raises(RuntimeError, match="desync"):
        store.append("r0", 0, torch.randn(1, 8), start_pos=3)  # overlap
    with pytest.raises(RuntimeError, match="desync"):
        store.append("r0", 0, torch.randn(1, 8), start_pos=5)  # gap
    with pytest.raises(RuntimeError, match="holds 4"):
        store.history("r0", 0, 5)  # window beyond stored history


def test_cleanup_request_evicts_kstore():
    """Retirement leaves NO per-request growth: the engine's request removal
    calls cleanup_request on every managed submodule, which must drop the
    k-history and the MTP counters."""
    sub = object.__new__(Glm52LLMSubmodule)
    sub.config = Glm52ModelConfig.reduced()
    sub.request_states = {}
    sub._dsa_k_store = Glm52DsaKStore()
    sub._mtp_emitted = {"r0": 7}
    sub._mtp_max_tokens = {"r0": 32}
    sub._mtp_ignore_eos = {"r0": False}

    sub._dsa_k_store.append("r0", 0, torch.randn(5, 16), start_pos=0)
    sub.request_state("r0")  # base per-request state exists too
    sub.cleanup_request("r0")
    assert sub._dsa_k_store.tracked_requests() == set()
    assert sub.request_states == {}
    assert sub._mtp_emitted == {}
    assert sub._mtp_max_tokens == {}
    assert sub._mtp_ignore_eos == {}
    sub.cleanup_request("r0")  # idempotent, like the base hook


# ---------------------------------------------------------------------------
# preprocess guard, both flag states
# ---------------------------------------------------------------------------

class _FakeKV:
    def __init__(self, starts):
        self._starts = starts

    def stored_len(self, rid, label="main"):
        return self._starts[rid]


class _FakeAttn:
    """The attention resource's current plan, as preprocess reads it: one
    page table per request, positionally."""

    def __init__(self, tables):
        self._tables = tables

    def page_tables(self, label="main"):
        return self._tables


def _make_submodule(config) -> Glm52LLMSubmodule:
    sub = object.__new__(Glm52LLMSubmodule)
    sub.config = config
    sub._dsa_k_store = Glm52DsaKStore()
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
        resources={
            KV_RESOURCE: _FakeKV(starts),
            ATTN_RESOURCE: _FakeAttn(tables or [[0, 1, 2] for _ in starts]),
        },
    )
    return sub.preprocess("prefill", engine_inputs, inputs)


def test_guard_flag_off_unchanged_and_no_ctx():
    cfg = Glm52ModelConfig()
    assert cfg.dsa_long_context is False
    sub = _make_submodule(cfg)
    out = _preprocess(sub, {"r0": 2032}, seq_len=16)  # exactly topk: allowed
    assert out["dsa_ctx"] is None
    with pytest.raises(RuntimeError, match="dsa_long_context"):
        _preprocess(sub, {"r0": 2040}, seq_len=16)


def test_guard_flag_on_lifts_cap_to_max_seq_len():
    cfg = Glm52ModelConfig()
    cfg.dsa_long_context = True
    cfg.max_seq_len = 4096
    sub = _make_submodule(cfg)

    # Decode one past topk: allowed, and the batch is marked for selection.
    out = _preprocess(sub, {"r0": cfg.index_topk}, seq_len=1)
    ctx = out["dsa_ctx"]
    assert isinstance(ctx, Glm52DsaForwardContext)
    assert ctx.needs_selection is True
    assert ctx.k_store is sub._dsa_k_store
    assert ctx.last_selection is None  # transient starts clean every forward

    # The new cap is max_seq_len, not topk.
    with pytest.raises(RuntimeError, match="max_seq_len"):
        _preprocess(sub, {"r0": 4096}, seq_len=1)


def test_guard_flag_on_prefill_beyond_topk_is_refused_decode_first():
    cfg = Glm52ModelConfig()
    cfg.dsa_long_context = True
    cfg.max_seq_len = 4096  # inside the lifted cap: the topk rule must fire
    sub = _make_submodule(cfg)
    with pytest.raises(RuntimeError, match="decode-only"):
        _preprocess(sub, {"r0": cfg.index_topk}, seq_len=16)


def test_flag_on_identity_batch_builds_spans_without_selection():
    cfg = Glm52ModelConfig()
    cfg.dsa_long_context = True
    sub = _make_submodule(cfg)
    tables = [[0, 1, 2], [5]]
    out = _preprocess(sub, {"r0": 0, "r1": 5}, seq_len=4, tables=tables)

    ctx = out["dsa_ctx"]
    assert ctx.needs_selection is False  # everything fits topk: identity
    (s0, s1) = ctx.spans
    assert (s0.request_id, s0.q_start, s0.q_len, s0.ctx_start) == ("r0", 0, 4, 0)
    assert (s1.request_id, s1.q_start, s1.q_len, s1.ctx_start) == ("r1", 4, 4, 5)
    assert s1.page_indices == [5]
    # Page tables are snapshots: later plan growth must not alias.
    tables[0].append(99)
    assert s0.page_indices == [0, 1, 2]


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
    with pytest.raises(ValueError, match="mutually exclusive"):
        Glm52Model(model_path_hf="", dsa_long_context=True, mtp_num_draft_tokens=2)


def test_no_cuda_graphs_when_long_context():
    cfg = Glm52ModelConfig.reduced()  # would otherwise register 2 configs
    cfg.dsa_long_context = True
    sub = _make_submodule(cfg)
    assert sub.get_cuda_graph_configs(torch.device("cpu")) == []
    assert sub.get_piecewise_cuda_graph_configs(torch.device("cpu"), torch.float32) == {}


# ---------------------------------------------------------------------------
# The real resources: reduced absorbed model driven a step at a time
# ---------------------------------------------------------------------------

def _build_absorbed_model(cfg, seed):
    """Randomize EVERY parameter (MoE expert containers are raw
    ``torch.empty`` at construction — garbage in them NaNs the logits)."""
    from mstar.model.glm52.components.moe import Glm52SparseMoeBlock

    torch.manual_seed(seed)
    model = Glm52ForCausalLM(cfg)
    model.model.embed_tokens.weight.data.normal_(0, 0.05)
    model.model.norm.weight.data.normal_(1.0, 0.02)
    model.lm_head.weight.data.normal_(0, 0.02)
    for layer in model.model.layers:
        a = layer.self_attn
        for lin in (a.q_a_proj, a.q_b_proj, a.kv_a_proj_with_mqa,
                    a.kv_b_proj, a.o_proj):
            lin.weight.data.normal_(0, 0.03)
        for norm in (a.q_a_layernorm, a.kv_a_layernorm):
            norm.weight.data.normal_(1.0, 0.02)
        if a.indexer is not None:
            a.indexer.wq_b.weight.data.normal_(0, 0.05)
            a.indexer.wk.weight.data.normal_(0, 0.05)
            a.indexer.weights_proj.weight.data.normal_(0, 0.05)
            a.indexer.k_norm.weight.data.normal_(1.0, 0.02)
            a.indexer.k_norm.bias.data.normal_(0, 0.02)
        layer.input_layernorm.weight.data.normal_(1.0, 0.02)
        layer.post_attention_layernorm.weight.data.normal_(1.0, 0.02)
        mlp = layer.mlp
        if isinstance(mlp, Glm52SparseMoeBlock):
            mlp.gate.weight.data.normal_(0, 1)
            mlp.gate.e_score_correction_bias.data = torch.randn(
                cfg.n_routed_experts, dtype=torch.float32)
            mlp.experts.gate_up_proj.data.normal_(0, 0.05)
            mlp.experts.down_proj.data.normal_(0, 0.05)
            mlp.shared_expert.gate_up_proj.weight.data.normal_(0, 0.05)
            mlp.shared_expert.down_proj.weight.data.normal_(0, 0.05)
        else:
            mlp.gate_up_proj.weight.data.normal_(0, 0.05)
            mlp.down_proj.weight.data.normal_(0, 0.05)
    process_weights_after_loading(model, torch.device("cpu"))
    return model.eval()


class _Driver:
    """The engine's per-step cycle for one request over the real resources:
    declare -> admit -> plan -> preprocess (builds the dsa_ctx exactly as
    serving would, page tables from the live plan) -> forward -> commit."""

    def __init__(self, sub: Glm52LLMSubmodule, cfg: Glm52ModelConfig, rid="r0", page_size=8):
        self.sub, self.rid = sub, rid
        self.resources, self.runner = build_cpu_resources(cfg, [rid], page_size=page_size)
        sub.bind_node_resources(self.resources)
        self.engine_inputs = ModelInputsFromEngine(
            request_ids=[rid], per_request_info={}, resources=self.resources)

    def step(self, walk: str, ids: torch.Tensor) -> torch.Tensor:
        ar = ARNodeInputs(input_ids=ids, input_seq_len=ids.shape[0])
        step = self.sub.declare_step(walk, [self.rid], [ar])
        step.set_ctx(StepContext(request_ids=(self.rid,), graph_walk=walk, slot=0, capture=False))
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        packed = self.sub.preprocess(walk, self.engine_inputs, [ar])
        with torch.no_grad():
            logits = self.sub.forward(walk, self.engine_inputs, **packed)["logits"][0]
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


def test_declared_step_under_dsa_is_the_plain_one():
    cfg = _longctx_reduced_cfg(topk=4)
    sub = Glm52LLMSubmodule(language_model=_build_absorbed_model(cfg, seed=0), config=cfg)
    ar = ARNodeInputs(input_ids=torch.tensor([1, 2, 3]), input_seq_len=3)
    step = sub.declare_step("prefill", ["r0"], [ar])
    assert set(step.keys()) == {KV_RESOURCE, ATTN_RESOURCE, SAMPLER_RESOURCE}
    assert step.segments[0].span == 3


def test_full_to_shared_threading_order(monkeypatch):
    """Layer 1 (SHARED, no indexer weights) must consume the selection layer
    0 (FULL) published THIS forward — the IndexShare reuse window."""
    cfg = _longctx_reduced_cfg(topk=4)
    model = _build_absorbed_model(cfg, seed=0)
    sub = Glm52LLMSubmodule(language_model=model, config=cfg)
    driver = _Driver(sub, cfg)

    trace = []

    def spy(self, dsa_ctx, selection, q_nope, q_pe, kv_c, k_pe):
        trace.append((self.layer_idx, selection))
        return torch.zeros_like(q_nope)

    monkeypatch.setattr(Glm52MLAAttention, "_run_sparse_absorbed", spy)

    compute_calls = []
    layer0_indexer = model.model.layers[0].self_attn.indexer
    original = layer0_indexer.compute_selection

    def counting(*args, **kwargs):
        result = original(*args, **kwargs)
        compute_calls.append(result)
        return result

    monkeypatch.setattr(layer0_indexer, "compute_selection", counting)

    prompt = torch.randint(0, cfg.vocab_size, (4,))  # ctx 4 == topk: identity
    decode = [torch.tensor(5)]  # ctx 5 > topk: selection engages
    _serve_steps(driver, prompt, decode)

    # One FULL computation, consumed IN ORDER by layer 0 then layer 1, and
    # both consumed the very tensor the FULL layer produced.
    assert len(compute_calls) == 1
    assert [layer for layer, _ in trace] == [0, 1]
    assert trace[0][1] is compute_calls[0]
    assert trace[1][1] is compute_calls[0]

    # FULL layer appended prompt + decode token; SHARED layer appended nothing.
    assert sub._dsa_k_store.tokens("r0", 0) == 5
    assert sub._dsa_k_store.tokens("r0", 1) == 0
    assert driver.resources[KV_RESOURCE].stored_len("r0") == 5

    sub.cleanup_request("r0")
    assert sub._dsa_k_store.tracked_requests() == set()


def test_shared_layer_without_full_selection_raises():
    """A SHARED layer reached in selection mode with no published rows is a
    threading bug and must fail loudly, not attend densely off-spec."""
    cfg = _longctx_reduced_cfg(topk=4)
    model = _build_absorbed_model(cfg, seed=1)
    attn1 = model.model.layers[1].self_attn  # SHARED
    ctx = Glm52DsaForwardContext(
        spans=[Glm52DsaRequestSpan("r0", 0, 1, 6, [0])],
        k_store=Glm52DsaKStore(), needs_selection=True,
    )
    with pytest.raises(RuntimeError, match="no FULL layer ran"):
        attn1(torch.randn(1, cfg.hidden_size) * 0.1, torch.tensor([6]), dsa_ctx=ctx)


def test_naive_path_refuses_dsa_ctx():
    cfg = Glm52ModelConfig.reduced()  # mla_absorb False
    attn = Glm52MLAAttention(cfg, layer_idx=0)
    ctx = Glm52DsaForwardContext(spans=[], k_store=Glm52DsaKStore(),
                                 needs_selection=False)
    with pytest.raises(RuntimeError, match="mla_absorb"):
        attn(torch.randn(2, cfg.hidden_size), torch.arange(2), dsa_ctx=ctx)


# ---------------------------------------------------------------------------
# Sparse gather path vs independent mask-based reference
# ---------------------------------------------------------------------------

def _independent_masked_reference(latent_cache, pages, page_size, positions,
                                  selection_row, q_all, scale, latent_dim):
    """Dense MQA over ALL cached positions with an additive mask keeping only
    the selected ones — written from scratch, sharing no code with
    ``_run_sparse_absorbed``."""
    pos = torch.arange(positions)
    flat = latent_cache[pages[pos // page_size], pos % page_size].float()
    scores = q_all.float() @ flat.T * scale  # (H, positions)
    mask = torch.full((positions,), float("-inf"))
    mask[selection_row[selection_row >= 0].long()] = 0.0
    weights = (scores + mask).softmax(-1)
    return weights @ flat[:, :latent_dim]


def _bound_attention(cfg, page_size=4):
    """A FULL-layer attention bound to a real MLA cache (no plan needed: the
    sparse path addresses pages by the span's page table directly)."""
    attn = Glm52MLAAttention(cfg, layer_idx=0)
    for lin in (attn.q_a_proj, attn.q_b_proj, attn.kv_a_proj_with_mqa,
                attn.kv_b_proj, attn.o_proj):
        lin.weight.data.normal_(0, 0.03)
    attn.process_weights_after_loading()
    resources, _ = build_cpu_resources(cfg, page_size=page_size)
    attn.bind_resources(resources)
    kv = resources[KV_RESOURCE]
    assert kv.config.page_size == page_size
    return attn, kv


@pytest.mark.parametrize("selection_row", [
    [9, 2, 7, 0],      # full topk row, unsorted, includes self (position 9)
    [5, 9, -1, -1],    # padded row: -1 entries must be excluded, not scored
])
def test_sparse_gather_matches_masked_reference(selection_row):
    torch.manual_seed(2)
    cfg = _longctx_reduced_cfg(topk=4)
    page_size = 4
    attn, kv = _bound_attention(cfg, page_size)
    latent_dim = cfg.kv_lora_rank
    width = latent_dim + cfg.qk_rope_head_dim
    cache = kv.layer_view(0)  # (pages, page_size, width), the real latent layer
    assert cache.shape[1:] == (page_size, width)
    pages = torch.tensor([3, 1, 6], dtype=torch.long)  # non-monotonic on purpose

    current = 9  # decode position; prefix 0..8 already cached
    for pos in range(current):
        cache[pages[pos // page_size], pos % page_size] = torch.randn(width)

    ctx = Glm52DsaForwardContext(
        spans=[Glm52DsaRequestSpan("r0", 0, 1, current, pages.tolist())],
        k_store=Glm52DsaKStore(), needs_selection=True,
    )
    h = attn.num_heads
    q_nope = torch.randn(1, h, latent_dim)
    q_pe = torch.randn(1, h, cfg.qk_rope_head_dim)
    kv_c = torch.randn(1, 1, latent_dim)
    k_pe = torch.randn(1, 1, cfg.qk_rope_head_dim)
    selection = torch.tensor([selection_row], dtype=torch.int32)

    out = attn._run_sparse_absorbed(ctx, selection, q_nope, q_pe, kv_c, k_pe)

    # The scatter half: position 9's slot now holds this step's latent.
    written = cache[pages[current // page_size], current % page_size]
    torch.testing.assert_close(
        written, torch.cat([kv_c, k_pe], dim=-1).squeeze(), rtol=0, atol=0)

    q_all = torch.cat([q_nope, q_pe], dim=-1)[0]
    ref = _independent_masked_reference(
        cache, pages, page_size, current + 1, selection[0], q_all,
        attn.softmax_scale, latent_dim)
    torch.testing.assert_close(out[0], ref, rtol=1e-6, atol=1e-6)

    # Discrimination control: attending the WHOLE prefix must not match —
    # otherwise this test could not catch a sparse path that ignores the
    # selection.
    dense_row = torch.arange(current + 1, dtype=torch.int32)
    dense_ref = _independent_masked_reference(
        cache, pages, page_size, current + 1, dense_row, q_all,
        attn.softmax_scale, latent_dim)
    assert not torch.allclose(out[0], dense_ref, rtol=1e-3, atol=1e-3)


def test_sparse_path_refuses_prefill_shape():
    cfg = _longctx_reduced_cfg(topk=4)
    attn, _ = _bound_attention(cfg)
    ctx = Glm52DsaForwardContext(
        spans=[Glm52DsaRequestSpan("r0", 0, 2, 5, [1, 2])],
        k_store=Glm52DsaKStore(), needs_selection=True,
    )
    h = attn.num_heads
    with pytest.raises(RuntimeError, match="decode-only"):
        attn._run_sparse_absorbed(
            ctx, torch.zeros(2, 4, dtype=torch.int32),
            torch.randn(2, h, cfg.kv_lora_rank),
            torch.randn(2, h, cfg.qk_rope_head_dim),
            torch.randn(2, 1, cfg.kv_lora_rank),
            torch.randn(2, 1, cfg.qk_rope_head_dim))


# ---------------------------------------------------------------------------
# CPU decode across the topk boundary: prefix property + real divergence
# ---------------------------------------------------------------------------

def test_decode_across_topk_prefix_property_and_divergence():
    """Teacher-forced decode with topk=6 vs a dense comparator (topk lifted so selection
    never engages).
    """
    topk = 6
    cfg = _longctx_reduced_cfg(topk)
    model = _build_absorbed_model(cfg, seed=3)
    sub = Glm52LLMSubmodule(language_model=model, config=cfg)

    torch.manual_seed(7)
    prompt = torch.randint(0, cfg.vocab_size, (3,))
    decode = list(torch.randint(0, cfg.vocab_size, (6,)))
    # Step i covers context 3 + 1 + i tokens after it: steps 0..2 end at
    # ctx 4, 5, 6 (identity); steps 3..5 end at 7, 8, 9 (sparse).

    driver = _Driver(sub, cfg, page_size=4)  # the sparse steps span 3 pages
    assert not driver.resources[ATTN_RESOURCE].uses_kernel
    sparse_logits = _serve_steps(driver, prompt, decode)
    assert sub._dsa_k_store.tokens("r0", 0) == 3 + len(decode)
    assert driver.resources[KV_RESOURCE].stored_len("r0") == 3 + len(decode)
    sub.cleanup_request("r0")
    assert sub._dsa_k_store.tracked_requests() == set()

    cfg.index_topk = 512  # dense comparator: identity regime end to end
    driver2 = _Driver(sub, cfg, page_size=4)  # fresh cache, same weights
    dense_logits = _serve_steps(driver2, prompt, decode)
    sub.cleanup_request("r0")

    for i in range(4):  # prefill + decode steps 0..2
        assert torch.equal(sparse_logits[i], dense_logits[i]), f"step {i}"
    for i in range(4, 7):  # decode steps 3..5: beyond topk
        assert not torch.equal(sparse_logits[i], dense_logits[i]), f"step {i}"
        assert torch.isfinite(sparse_logits[i]).all()
