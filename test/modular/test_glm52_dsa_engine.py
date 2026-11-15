"""GLM-5.2 on the DSA resource: guard states, the declared step, IndexShare threading and
decode across ``index_topk`` — over the real ``KVManager`` (MLA latent layout), MLA attention
resource and DSA resource on CPU. The resource's own math is test_dsa_resource.py's.
"""
import logging
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


@pytest.fixture(autouse=True)
def _force_cpu_flashinfer(monkeypatch):
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())
    install_cpu_rope(monkeypatch)


from mstar.engine.resources import StepContext  # noqa: E402
from mstar.engine.resources.attn.flashinfer_mla import flashinfer_mla_supports  # noqa: E402
from mstar.engine.resources.dsa.manager import DSAManager  # noqa: E402
from mstar.engine.state import sample  # noqa: E402
from mstar.engine.state.analysis import state_declaration_problems  # noqa: E402
from mstar.model.glm52.components.attention import Glm52MLAAttention  # noqa: E402
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.config import (  # noqa: E402
    ATTN,
    INDEX_KEYS,
    KV,
    SAMPLER,
    Glm52ModelConfig,
)
from mstar.model.glm52.glm52_model import Glm52Model  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402
from mstar.model.glm52.tests.testing import (  # noqa: E402
    POSITIONS,
    bind_cpu_state,
    build_cpu_resources,
    close_step,
    install_cpu_rope,
    open_step,
)
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine, engine_step  # noqa: E402
from mstar.model.utils import process_weights_after_loading  # noqa: E402

# ---------------------------------------------------------------------------
# preprocess guard, both flag states
# ---------------------------------------------------------------------------

class _FakePositions:
    def __init__(self, starts):
        self._starts = starts

    def position(self, rid, label):
        return self._starts[rid]


def _make_submodule(config) -> Glm52LLMSubmodule:
    sub = object.__new__(Glm52LLMSubmodule)
    sub.config = config
    return sub


def _preprocess(sub, starts, seq_len):
    sub.get_device = lambda: torch.device("cpu")
    sub.kv = SimpleNamespace(positions=_FakePositions(starts))
    inputs = [
        ARNodeInputs(
            input_ids=torch.zeros(seq_len, dtype=torch.long),
            input_seq_len=seq_len,
        )
        for _ in starts
    ]
    engine_inputs = SimpleNamespace(request_ids=list(starts))
    return sub.preprocess("prefill", engine_inputs, inputs)


def test_guard_flag_off_holds_context_to_topk():
    cfg = Glm52ModelConfig()
    assert cfg.dsa_long_context is False
    sub = _make_submodule(cfg)
    _preprocess(sub, {"r0": 2032}, seq_len=16)  # exactly topk: allowed
    with pytest.raises(RuntimeError, match="dsa_long_context"):
        _preprocess(sub, {"r0": 2040}, seq_len=16)


def test_guard_flag_on_lifts_cap_to_max_seq_len():
    cfg = Glm52ModelConfig()
    cfg.dsa_long_context = True
    cfg.max_seq_len = 4096
    sub = _make_submodule(cfg)

    # Decode one past topk: allowed (the DSA resource selects for it).
    _preprocess(sub, {"r0": cfg.index_topk}, seq_len=1)

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
    # the verify step would settle the latent array and leave the index keys a block ahead
    mtp = Glm52Model(model_path_hf="", dsa_long_context=True, mtp_num_draft_tokens=2)
    assert any(f"Array {INDEX_KEYS!r} has other verify_span" in each for each in state_declaration_problems(mtp))


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


def test_the_runner_drops_every_capture_under_long_context(monkeypatch, caplog):
    """The submodule offers its configs, and the runner drops each one, since every step holds the index keys,
    which the DSA resource can't capture."""
    from unittest.mock import Mock

    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner

    cfg = _longctx_reduced_cfg(topk=4)
    sub = Glm52LLMSubmodule(language_model=_build_absorbed_model(cfg, seed=0), config=cfg)
    resources, _ = build_cpu_resources(cfg)
    bind_cpu_state(sub, cfg, resources)
    for key, resource in resources.items():
        if key != INDEX_KEYS:
            monkeypatch.setattr(resource, "can_capture", lambda: True)
    runner = AcceleratorGraphRunner(
        submodule_name="LLM", submodule=sub, resources=resources, step_runner=Mock(),
        device=torch.device("cpu"), autocast_dtype=None, joint_comm_group=SimpleNamespace(world_size=1),
        num_slots=1, head_shards=1,
    )
    assert {spec.bucket.graph_walk for spec in runner._get_slot_specs()} == {"prefill", "decode"}
    with caplog.at_level(logging.ERROR):
        assert runner.prepare_for_capture() == []
    assert f"resources cannot capture: ['{INDEX_KEYS}']" in caplog.text


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
    declare -> admit -> plan (the DSA resource plans its rows off the live KV
    plan, as serving would) -> preprocess -> forward -> commit."""

    def __init__(self, sub: Glm52LLMSubmodule, cfg: Glm52ModelConfig, rid="r0", page_size=8):
        self.sub, self.rid = sub, rid
        self.resources, self.runner = build_cpu_resources(cfg, [rid], page_size=page_size)
        bind_cpu_state(sub, cfg, self.resources)

    def step(self, walk: str, ids: torch.Tensor) -> torch.Tensor:
        ar = ARNodeInputs(input_ids=ids, input_seq_len=ids.shape[0])
        step = engine_step(self.sub, graph_walk=walk, request_ids=[self.rid], inputs=[ar])
        ctx = StepContext(request_ids=(self.rid,), graph_walk=walk, slot=0, capture=False)
        assert open_step(self.runner, step, ctx).ok
        engine_inputs = ModelInputsFromEngine(
            request_ids=[self.rid], per_request_info={}, resources=self.resources, step=step)
        packed = self.sub.preprocess(walk, engine_inputs, [ar])
        with torch.no_grad():
            logits = self.sub.forward(walk, engine_inputs, **packed)["logits"][0]
        # the step's one draw, which forward_batched takes and the logits forward leaves
        sample(self.sub.sampler, logits)
        close_step(self.runner, step)
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


def test_declared_step_under_dsa_adds_the_dsa_step():
    cfg = _longctx_reduced_cfg(topk=4)
    sub = Glm52LLMSubmodule(language_model=_build_absorbed_model(cfg, seed=0), config=cfg)
    bind_cpu_state(sub, cfg, build_cpu_resources(cfg)[0])
    ar = ARNodeInputs(input_ids=torch.tensor([1, 2, 3]), input_seq_len=3)
    step = engine_step(sub, graph_walk="prefill", request_ids=["r0"], inputs=[ar])
    assert set(step.steps) == {KV, ATTN, POSITIONS, SAMPLER, INDEX_KEYS}
    assert step.segments[0].span == 3


def test_full_to_shared_threading_order(monkeypatch):
    """Layer 1 (SHARED, no indexer weights) must consume the pick layer 0 (FULL) carried
    THIS forward — the IndexShare reuse window."""
    cfg = _longctx_reduced_cfg(topk=4)
    model = _build_absorbed_model(cfg, seed=0)
    sub = Glm52LLMSubmodule(language_model=model, config=cfg)
    driver = _Driver(sub, cfg)
    dsa = driver.resources[INDEX_KEYS]

    trace = []
    layer = {}

    def spy(self, selection, q_nope, q_pe, latent_layer, sm_scale, label="main"):
        trace.append((layer["idx"], selection))
        return torch.zeros_like(q_nope)

    monkeypatch.setattr(DSAManager, "attend", spy)
    index_pass = Glm52MLAAttention._index_pass

    def tagged(self, *args, **kwargs):
        layer["idx"] = self.layer_idx
        return index_pass(self, *args, **kwargs)

    monkeypatch.setattr(Glm52MLAAttention, "_index_pass", tagged)

    picks = []
    select = DSAManager.select

    def counting(self, *args, **kwargs):
        result = select(self, *args, **kwargs)
        picks.append(result)
        return result

    monkeypatch.setattr(DSAManager, "select", counting)

    prompt = torch.randint(0, cfg.vocab_size, (4,))  # ctx 4 == topk: identity
    decode = [torch.tensor(5)]  # ctx 5 > topk: selection engages
    _serve_steps(driver, prompt, decode)

    # One FULL computation, consumed IN ORDER by layer 0 then layer 1, and
    # both consumed the very tensor the FULL layer produced.
    assert len(picks) == 1
    assert [idx for idx, _ in trace] == [0, 1]
    assert trace[0][1] is picks[0]
    assert trace[1][1] is picks[0]

    # FULL layer stored prompt + decode token; SHARED layer stored nothing.
    assert dsa.keys.tokens("r0", 0) == 5
    assert dsa.keys.tokens("r0", 1) == 0
    assert driver.resources[KV]._streams["r0"]["main"].stored_len == 5

    driver.runner.remove_request("r0")
    assert dsa.keys.tracked_requests() == set()


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
    assert not flashinfer_mla_supports(cfg.kv_lora_rank, cfg.qk_rope_head_dim)
    sparse_logits = _serve_steps(driver, prompt, decode)
    dsa = driver.resources[INDEX_KEYS]
    assert dsa.keys.tokens("r0", 0) == 3 + len(decode)
    assert driver.resources[KV]._streams["r0"]["main"].stored_len == 3 + len(decode)
    driver.runner.remove_request("r0")
    assert dsa.keys.tracked_requests() == set()

    cfg.index_topk = 512  # dense comparator: identity regime end to end
    driver2 = _Driver(sub, cfg, page_size=4)  # fresh cache, same weights
    dense_logits = _serve_steps(driver2, prompt, decode)
    driver2.runner.remove_request("r0")

    for i in range(4):  # prefill + decode steps 0..2
        assert torch.equal(sparse_logits[i], dense_logits[i]), f"step {i}"
    for i in range(4, 7):  # decode steps 3..5: beyond topk
        assert not torch.equal(sparse_logits[i], dense_logits[i]), f"step {i}"
        assert torch.isfinite(sparse_logits[i]).all()
