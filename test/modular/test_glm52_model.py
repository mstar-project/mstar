"""Glm52Model contract on the resource-pools engine: registry, graph walks,
the prefill -> decode transition, the node's resource declaration, the
per-request sampling config, and the submodule's stop / preprocess guards.
No weights, no GPU.
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


from mstar.conductor.request_info import (  # noqa: E402
    CurrentForwardConductorMetadata,
    CurrentForwardPassInfo,
)
from mstar.engine.resources import (  # noqa: E402
    AttentionSpec,
    KVLayout,
    KVSpec,
    MlaAttentionSpec,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import Loop  # noqa: E402
from mstar.model.glm52.config import (  # noqa: E402
    ATTN_RESOURCE,
    KV_RESOURCE,
    SAMPLER_RESOURCE,
    Glm52ModelConfig,
)
from mstar.model.glm52.dsa import Glm52DsaKStore  # noqa: E402
from mstar.model.glm52.glm52_model import Glm52Model  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402


def _make_model(config: Glm52ModelConfig | None = None) -> Glm52Model:
    # Skip __init__ (tokenizer download); the contract under test only
    # needs the config.
    model = object.__new__(Glm52Model)
    model.config = config or Glm52ModelConfig()
    return model


def _make_model_k(k: int) -> Glm52Model:
    cfg = Glm52ModelConfig()
    cfg.mtp_num_draft_tokens = k
    return _make_model(cfg)


def test_glm52_registered():
    # The registry is lazy (module path, class name) so importing it pulls
    # no model code; resolve the entry and check it is the class.
    from mstar.model import registry

    assert registry.MODEL_REGISTRY["glm52"] == (
        "mstar.model.glm52.glm52_model", "Glm52Model")
    assert registry.get_model_class("glm52") is Glm52Model
    assert registry.HF_MODELS["glm52"]["model_path_hf"] == "zai-org/GLM-5.2-FP8"


def test_glm52_graph_walks_are_one_node():
    model = _make_model()
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {"prefill", "decode"}
    assert isinstance(walks["decode"], Loop)
    # every walk runs the one fat LLM node, and every resource is declared
    # for exactly that node
    assert {n for g in walks.values() for n in g.get_nodes()} == {"LLM"}
    assert all(spec.nodes == {"LLM"} for spec in model.get_node_resources())


# ── resource declaration ──


def _specs(config):
    kv, attn, sampler = _make_model(config).get_node_resources()
    assert isinstance(kv, KVSpec) and kv.resource_key == KV_RESOURCE
    assert isinstance(sampler, SamplerSpec) and sampler.resource_key == SAMPLER_RESOURCE
    assert attn.resource_key == ATTN_RESOURCE
    return kv, attn, sampler


def test_glm52_resources_absorbed_is_mla_latent_layout():
    cfg = Glm52ModelConfig()  # full model, mla_absorb default True
    kv, attn, sampler = _specs(cfg)
    # MLA: one shared latent "head" of width kv_lora_rank + qk_rope_head_dim,
    # not num_kv_heads x head_dim.
    assert kv.config.layout == KVLayout.MLA
    assert kv.config.num_layers == 78
    assert kv.config.num_kv_heads == 1
    assert kv.config.head_dim == 512 + 64
    assert kv.config.num_qo_heads == 64
    assert kv.config.max_seq_len == cfg.max_seq_len == 2048
    assert isinstance(attn, MlaAttentionSpec)
    assert attn.config.kv_cache == KV_RESOURCE
    assert attn.config.softmax_scale == cfg.qk_head_dim ** -0.5  # 256**-0.5, no mscale
    assert attn.config.ckv_dim == cfg.kv_lora_rank == 512
    assert attn.depends_on() == {KV_RESOURCE}
    assert sampler.vocab_size == cfg.vocab_size


def test_glm52_resources_mtp_adds_the_plane_layer():
    # the layer-78 draft module keeps its KV in one extra plane on the
    # trunk's page table
    kv, _, _ = _specs(_make_model_k(2).config)
    assert kv.config.num_layers == 79
    kv0, _, _ = _specs(_make_model_k(0).config)
    assert kv0.config.num_layers == 78


def test_glm52_resources_flag_off_is_naive():
    cfg = Glm52ModelConfig.reduced()  # mla_absorb False
    kv, attn, _ = _specs(cfg)
    assert kv.config.layout == KVLayout.NHD
    assert kv.config.num_kv_heads == cfg.num_attention_heads == 4
    assert kv.config.head_dim == cfg.padded_head_dim == 64  # qk 24 -> FlashInfer 64
    assert kv.config.num_qo_heads == 4
    assert isinstance(attn, AttentionSpec) and not isinstance(attn, MlaAttentionSpec)
    assert attn.config.kv_cache == KV_RESOURCE


# ── conductor state machine ──


def test_glm52_initial_forward_pass_args_seed_prefill_from_the_prompt():
    signals = {"text_inputs": ["PROMPT"]}
    args = _make_model().get_initial_forward_pass_args(
        "default", ["text"], ["text"], signals)
    assert args.full_metadata.graph_walk == "prefill"
    assert args.full_metadata.is_prefill is True
    assert args.step_metadata == {"is_prefill": True}
    (edge,) = args.inputs
    assert (edge.next_node, edge.name) == ("LLM", "text_inputs")
    assert edge.tensor_info == ["PROMPT"]
    assert args.unpersist_tensors == ["PROMPT"]


def test_glm52_prefill_transitions_to_decode():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"],
        output_modalities=["text"],
        graph_walk="prefill",
        is_prefill=True,
    )

    result = model.get_partition_forward_pass_args(
        partition_name="default",
        partition_metadata=metadata,
        persist_signals={"new_token": []},
    )

    assert result.full_metadata.graph_walk == "decode"
    assert result.step_metadata["is_prefill"] is False
    assert result.request_done is False


def test_glm52_prefill_drafts_default_on_and_escape_hatch(monkeypatch):
    """The prefill-draft edge ships ON as of 2026-08-19 (arm L: 78.53 tok/s,
    3264 bit-exact, forced n_acc=0 bin 143 -> 129). The 2026-08-10 regression
    that kept it off (33.02 / p1 0.18) was the `text_inputs` name collision,
    fixed since. MSTAR_GLM52_MTP_PREFILL_DRAFTS=0 must still drop the edge.

    delenv, not ambient env: reading whatever the process happens to have
    set is the exact dependence that let 8 tests rot silently in this
    suite."""
    from mstar.model.glm52.submodules import MTP_DRAFT_BUNDLE

    monkeypatch.delenv("MSTAR_GLM52_MTP_PREFILL_DRAFTS", raising=False)
    prefill = _make_model_k(2).get_graph_walk_graphs()["prefill"]
    assert [e.name for e in prefill.outputs] == ["new_token", MTP_DRAFT_BUNDLE]
    monkeypatch.setenv("MSTAR_GLM52_MTP_PREFILL_DRAFTS", "0")
    prefill = _make_model_k(2).get_graph_walk_graphs()["prefill"]
    assert [e.name for e in prefill.outputs] == ["new_token"]


def test_glm52_prefill_persists_drafts_only_under_mtp(monkeypatch):
    """The MTP prefill computes [emitted, k drafts]. An output with no
    declared edge is UNROUTED — the worker drops it — so without this edge
    the prefill's whole sync+draft pass was wasted TTFT work and the first
    decode step ran unspeculated, injecting one artificial n_acc=0 per
    request into the acceptance histogram. k=0 must keep the byte-identical
    old walk."""
    from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
    from mstar.model.glm52.submodules import MTP_DRAFT_BUNDLE

    monkeypatch.setenv("MSTAR_GLM52_MTP_PREFILL_DRAFTS", "1")
    prefill_k0 = _make_model_k(0).get_graph_walk_graphs()["prefill"]
    assert [e.name for e in prefill_k0.outputs] == ["new_token"]

    prefill_k2 = _make_model_k(2).get_graph_walk_graphs()["prefill"]
    by_name = {e.name: e for e in prefill_k2.outputs}
    assert set(by_name) == {"new_token", MTP_DRAFT_BUNDLE}
    # Persisted, not emitted: the conductor seeds decode from it, and it must
    # never reach the client as output text.
    drafts = by_name[MTP_DRAFT_BUNDLE]
    assert drafts.persist is True
    assert drafts.next_node == EMPTY_DESTINATION
    assert by_name["new_token"].next_node == EMIT_TO_CLIENT


def test_glm52_decode_never_reseeds_from_the_prompt_signal(monkeypatch):
    """REGRESSION (2026-08-10, cost a 27-min box run to find).

    The conductor seeds persist_signals from initial_signals, and this
    model's initial signal is named "text_inputs" — the PROMPT. A transition
    that reads that key hands decode the entire prompt back as its first
    step (measured: a 17-row decode step with no capture bucket, and with
    the prefill-draft edge on, p1 acceptance 0.76 -> 0.18). The draft bundle
    travels under a dedicated name that cannot collide; feed the prompt in
    under BOTH names to prove the transition ignores the prompt one."""
    from mstar.model.glm52.submodules import MTP_DRAFT_BUNDLE

    assert MTP_DRAFT_BUNDLE != "text_inputs"

    def _seed_from(persist_signals):
        # Fresh metadata per call: the transition MUTATES it (prefill ->
        # decode), so a reused object takes the request-done branch.
        metadata = CurrentForwardConductorMetadata(
            input_modalities=["text"], output_modalities=["text"],
            graph_walk="prefill", is_prefill=True,
        )
        return _make_model_k(2).get_partition_forward_pass_args(
            partition_name="default", partition_metadata=metadata,
            persist_signals=persist_signals,
        )

    prompt_and_token = {"text_inputs": ["PROMPT"], "new_token": ["tok"]}
    for flag in ("0", "1"):
        # the guarantee is the NAME, not the gating
        monkeypatch.setenv("MSTAR_GLM52_MTP_PREFILL_DRAFTS", flag)
        res = _seed_from(prompt_and_token)
        assert res.inputs[0].tensor_info == ["tok"], (
            f"flag={flag}: decode was seeded with the PROMPT instead of the "
            "emitted token")
        assert "PROMPT" not in res.unpersist_tensors


def test_glm52_decode_ignores_a_persisted_bundle_when_the_flag_is_off(monkeypatch):
    """The READ gate. get_graph_walk_graphs is evaluated independently in
    the conductor and in every worker, so a split-flag deployment can have
    a worker persisting a bundle that a conductor with the flag OFF would
    consume. Gating only the write path is what regressed the "off" arm on
    2026-08-10."""
    from mstar.model.glm52.submodules import MTP_DRAFT_BUNDLE

    monkeypatch.setenv("MSTAR_GLM52_MTP_PREFILL_DRAFTS", "0")
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"], output_modalities=["text"],
        graph_walk="prefill", is_prefill=True,
    )
    res = _make_model_k(2).get_partition_forward_pass_args(
        partition_name="default", partition_metadata=metadata,
        persist_signals={"new_token": ["tok"], MTP_DRAFT_BUNDLE: ["bundle"]},
    )
    assert res.inputs[0].tensor_info == ["tok"], (
        "flag off, but the persisted bundle was consumed anyway — the read "
        "path is ungated")
    assert "bundle" not in res.unpersist_tensors


def test_glm52_decode_seeds_from_drafts_when_mtp_persisted_them(monkeypatch):
    """The prefill->decode handoff must prefer the persisted draft bundle
    over the bare new_token, and must unpersist BOTH so no per-request
    tensor outlives the transition."""
    monkeypatch.setenv("MSTAR_GLM52_MTP_PREFILL_DRAFTS", "1")

    def _transition(model, persist_signals):
        metadata = CurrentForwardConductorMetadata(
            input_modalities=["text"], output_modalities=["text"],
            graph_walk="prefill", is_prefill=True,
        )
        return model.get_partition_forward_pass_args(
            partition_name="default", partition_metadata=metadata,
            persist_signals=persist_signals,
        )

    # k=0 (and any step where MTP persisted nothing): seed from new_token.
    res = _transition(_make_model_k(0), {"new_token": ["tok"]})
    assert res.inputs[0].tensor_info == ["tok"]
    assert res.unpersist_tensors == ["tok"]

    # MTP: seed from the draft bundle, and consume new_token alongside it.
    from mstar.model.glm52.submodules import MTP_DRAFT_BUNDLE

    res = _transition(
        _make_model_k(2),
        {"new_token": ["tok"], MTP_DRAFT_BUNDLE: ["bundle"]})
    assert res.inputs[0].name == "text_inputs"
    assert res.inputs[0].tensor_info == ["bundle"]
    assert set(res.unpersist_tensors) == {"bundle", "tok"}


def test_glm52_decode_loop_cap_stays_below_the_context_guard():
    """The decode loop cap must NOT be raised to max_seq_len.

    The context-window check lives in preprocess, which is BATCH-level and
    raises, so a request that iterates into it fails every CO-BATCHED
    request. A cap at max_seq_len lets a long request reach that guard,
    converting a silent per-request truncation into a batch kill. Tried
    2026-08-10 and reverted; this pins the revert."""
    model = _make_model_k(0)
    cfg = model.config
    decode = model.get_graph_walk_graphs()["decode"]
    # The guard's bound is whichever limit preprocess actually compares
    # against — index_topk with DSA off, max_seq_len with it on.
    guard = cfg.max_seq_len if cfg.dsa_long_context else cfg.index_topk
    assert decode.max_iters < guard, (
        f"loop cap {decode.max_iters} can reach the batch-killing context "
        f"guard at {guard}")
    # check_stop is the only thing enforcing the real per-request budget,
    # because the decode edge carries no conductor_new_token.
    assert not any(e.conductor_new_token for e in decode.section.outputs)


def test_glm52_decode_completion_marks_done():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"],
        output_modalities=["text"],
        graph_walk="decode",
        is_prefill=False,
    )

    result = model.get_partition_forward_pass_args(
        partition_name="default",
        partition_metadata=metadata,
        persist_signals={},
    )

    assert result.request_done is True
    assert result.full_metadata.kwargs["decode_finished"] is True


# ── per-request sampling config ──


def test_glm52_request_config_is_the_sampler_config():
    model = _make_model_k(0)
    cfg = model.config
    configs = model.get_request_resource_configs({})
    assert set(configs) == {SAMPLER_RESOURCE}
    sampling = configs[SAMPLER_RESOURCE]
    assert isinstance(sampling, SamplingReqConfig)
    # the config's generation defaults, unless the request overrides them
    assert sampling.temperature == cfg.temperature
    assert sampling.top_p == cfg.top_p
    assert sampling.repetition_penalty == cfg.repetition_penalty
    assert sampling.ignore_eos is cfg.ignore_eos
    over = model.get_request_resource_configs(
        {}, {"top_p": 0.5, "ignore_eos": True, "repetition_penalty": 1.1})[SAMPLER_RESOURCE]
    assert (over.top_p, over.ignore_eos, over.repetition_penalty) == (0.5, True, 1.1)
    assert over.temperature == cfg.temperature


def test_glm52_mtp_declares_greedy_default_but_honors_explicit_asks():
    """MTP v1 decode is raw argmax. A bare request must serve coherently
    (greedy declared) rather than inherit config temperature=1.0 and be
    refused by prepare_inputs; an EXPLICIT non-greedy ask must survive to
    that refusal, because silently ignoring an ask is the failure mode."""
    k2 = _make_model_k(2)
    assert k2.get_request_resource_configs({})[SAMPLER_RESOURCE].temperature == 0.0
    assert k2.get_request_resource_configs(
        {}, {"temperature": 0.7})[SAMPLER_RESOURCE].temperature == 0.7
    # k=0 keeps the model default — speculation is what forces greedy.
    k0 = _make_model_k(0)
    assert k0.get_request_resource_configs({})[SAMPLER_RESOURCE].temperature == k0.config.temperature


# ── config ──


def test_glm52_config_sanity():
    cfg = Glm52ModelConfig()
    assert cfg.cache_latent_dim == 576  # 512 latent + 64 decoupled rope
    assert cfg.qk_head_dim == cfg.padded_head_dim == 256
    assert cfg.num_dense_layers == 3

    from mstar.model.glm52.components.language_model import is_moe_layer

    assert [is_moe_layer(cfg, i) for i in (0, 1, 2)] == [False, False, False]
    assert is_moe_layer(cfg, 3) and is_moe_layer(cfg, 77)

    reduced = Glm52ModelConfig.reduced()
    assert not is_moe_layer(reduced, 0) and is_moe_layer(reduced, 1)
    assert reduced.qk_head_dim == 24 and reduced.padded_head_dim == 64


# ── submodule: check_stop ──


def _make_submodule(config) -> Glm52LLMSubmodule:
    # check_stop / preprocess only read self.config; skip nn.Module init.
    sub = object.__new__(Glm52LLMSubmodule)
    sub.config = config
    sub._dsa_k_store = Glm52DsaKStore()
    return sub


def _fwd_info(max_tokens=100, ignore_eos=False, iters=0) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id="r0",
        graph_walk="decode",
        fwd_index=0,
        random_seed=0,
        max_tokens=max_tokens,
        resource_configs={SAMPLER_RESOURCE: SamplingReqConfig(ignore_eos=ignore_eos)},
        dynamic_loop_iter_counts={"decode_loop": iters},
    )


@pytest.mark.parametrize("eos", [154820, 154827, 154829])
def test_glm52_check_stop_on_each_eos_id(eos):
    sub = _make_submodule(Glm52ModelConfig())
    outputs = {"new_token": [torch.tensor([eos])]}
    assert sub.check_stop("r0", _fwd_info(), outputs) == {"decode_loop"}


def test_glm52_check_stop_continues_on_normal_token():
    sub = _make_submodule(Glm52ModelConfig())
    outputs = {"new_token": [torch.tensor([42])]}
    assert sub.check_stop("r0", _fwd_info(), outputs) == set()


def test_glm52_check_stop_ignore_eos_runs_to_max_tokens():
    sub = _make_submodule(Glm52ModelConfig())
    outputs = {"new_token": [torch.tensor([154820])]}
    assert sub.check_stop("r0", _fwd_info(ignore_eos=True), outputs) == set()
    # max_tokens counts TOTAL generated (vLLM semantics): 1 prefill token +
    # iters+1 decode tokens. For max 8 the stop fires at decode iter 6
    # (8 total), not 7 (which produced the measured off-by-one).
    assert sub.check_stop(
        "r0", _fwd_info(max_tokens=8, ignore_eos=True, iters=5), outputs,
    ) == set()
    assert sub.check_stop(
        "r0", _fwd_info(max_tokens=8, ignore_eos=True, iters=6), outputs,
    ) == {"decode_loop"}


# ── submodule: graph configs ──


def test_glm52_no_cuda_graphs_under_reference_dispatch():
    # The reference MoE dispatch (.nonzero()/host loop) cannot be stream-
    # captured; registering graph configs would fail every capture and then
    # break eager prefill. Reference modes must register none.
    fp8 = _make_submodule(Glm52ModelConfig.reduced_fp8())
    assert fp8.get_cuda_graph_configs(torch.device("cpu")) == []
    bf16_tp = _make_submodule(Glm52ModelConfig.reduced())
    assert bf16_tp.get_cuda_graph_configs(torch.device("cpu"), tp_world_size=8) == []
    # bf16 TP=1 uses the capture-safe fused kernel on GPU: graphs stay.
    bf16 = _make_submodule(Glm52ModelConfig.reduced())
    assert len(bf16.get_cuda_graph_configs(torch.device("cpu"))) == 2


def test_glm52_graph_compile_env_escape_hatch(monkeypatch):
    # MSTAR_GLM52_GRAPH_COMPILE=0 captures the eager forward (both walks) —
    # the escape hatch for the Inductor-subprocess Triton crash that failed
    # all 296 captures on 08-07. Default stays compile-on, in the cuBLAS
    # ("default") mode.
    sub = _make_submodule(Glm52ModelConfig.reduced())
    configs = sub.get_cuda_graph_configs(torch.device("cpu"))
    assert all(c.compile for c in configs)
    assert all(c.compile_mode == "default" for c in configs)
    monkeypatch.setenv("MSTAR_GLM52_GRAPH_COMPILE", "0")
    assert not any(
        c.compile for c in sub.get_cuda_graph_configs(torch.device("cpu"))
    )


# ── submodule: preprocess ──


class _FakeKV:
    """Only what preprocess reads off the KV resource: the stored length."""

    def __init__(self, starts):
        self._starts = starts

    def stored_len(self, rid, label="main"):
        return self._starts[rid]


def _preprocess(sub, starts, seq_len):
    from mstar.model.submodule_base import ARNodeInputs

    sub.get_device = lambda: torch.device("cpu")
    inputs = [
        ARNodeInputs(
            input_ids=torch.zeros(seq_len, dtype=torch.long),
            input_seq_len=seq_len,
        )
        for _ in starts
    ]
    engine_inputs = SimpleNamespace(
        request_ids=list(starts), resources={KV_RESOURCE: _FakeKV(starts)},
    )
    return sub.preprocess("prefill", engine_inputs, inputs)


def test_glm52_preprocess_supplies_eager_last_token_indices():
    sub = _make_submodule(Glm52ModelConfig())
    out = _preprocess(sub, {"r0": 0, "r1": 0}, seq_len=16)
    assert torch.equal(out["last_token_indices"], torch.tensor([15, 31]))
    assert out["seq_lens"] == [16, 16]
    assert out["dsa_ctx"] is None
    assert out["mtp_runners"] == {}


def test_glm52_preprocess_positions_continue_from_the_stored_length():
    sub = _make_submodule(Glm52ModelConfig())
    out = _preprocess(sub, {"r0": 5, "r1": 0}, seq_len=3)
    assert out["position_ids"].tolist() == [5, 6, 7, 0, 1, 2]


def test_glm52_preprocess_refuses_context_beyond_dsa_window():
    # Dense MLA == DSA only within the top-2048 window; beyond it needs the
    # DSA engine path (dsa_long_context).
    sub = _make_submodule(Glm52ModelConfig())
    _preprocess(sub, {"r0": 2032}, seq_len=16)  # exactly 2048: allowed
    with pytest.raises(RuntimeError, match="dsa_long_context"):
        _preprocess(sub, {"r0": 2040}, seq_len=16)


# ── prompt / output processing ──


def test_glm52_process_prompt_byte_mode_never_touches_tokenizer():
    m = object.__new__(Glm52Model)
    m.config = Glm52ModelConfig.reduced()  # vocab 256
    m._tokenizer_mode = "byte"
    m._tokenizer = None
    out = m.process_prompt("Hi", ["text"], ["text"])
    assert set(out) == {"text_inputs"}
    (ids,) = out["text_inputs"]
    assert ids.dtype == torch.long and ids.tolist() == [72, 105]
    # bytes past the vocabulary clip to the last id; an empty prompt is one
    # pad-ish token, not an empty prefill
    m.config.vocab_size = 100
    assert m.process_prompt("z", ["text"], ["text"])["text_inputs"][0].tolist() == [99]
    assert m.process_prompt("", ["text"], ["text"])["text_inputs"][0].tolist() == [0]
    assert m.process_prompt(None, ["text"], ["text"]) == {}
    assert m._tokenizer is None  # no lazy HF download triggered


def test_glm52_postprocess_byte_mode_never_touches_tokenizer():
    m = object.__new__(Glm52Model)
    m._tokenizer_mode = "byte"
    m._tokenizer = None
    out = m.postprocess(torch.tensor([72, 105]), "text")
    assert out == b"Hi"
    assert m._tokenizer is None  # no lazy HF download triggered
    with pytest.raises(ValueError, match="Unsupported modality"):
        m.postprocess(torch.tensor([1]), "audio")
