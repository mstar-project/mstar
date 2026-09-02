import sys

sys.path.insert(0, ".")


import pytest
import torch

from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    CurrentForwardPassInfo,
)
from mstar.engine.base import EngineType
from mstar.graph.base import Loop
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.glm52_model import Glm52Model
from mstar.model.glm52.submodules import Glm52LLMSubmodule
from mstar.utils.sampling import MultiSamplingConfig, SamplingConfig


def _make_model() -> Glm52Model:
    # Skip __init__ (tokenizer download); the contract under test only
    # needs the config.
    model = object.__new__(Glm52Model)
    model.config = Glm52ModelConfig()
    return model


def test_glm52_registered():
    # The registry imports every model; qwen3_omni pulls GPU-only deps
    # (flashinfer/triton) at module level, so skip where those aren't
    # installed (macOS dev) and assert fully on CI / the cluster. Plain
    # importorskip is not enough: the conftest triton stub makes
    # transformers' find_spec probe raise ValueError, not
    # ModuleNotFoundError, on dev machines.
    try:
        from mstar.model import registry
    except (ImportError, ValueError) as e:
        pytest.skip(f"full registry import needs GPU-only deps: {e}")

    assert registry.MODEL_REGISTRY["glm52"] is Glm52Model
    assert registry.HF_MODELS["glm52"]["model_path_hf"] == "zai-org/GLM-5.2-FP8"


def test_glm52_graph_walks_match_engine_types():
    model = _make_model()
    walks = model.get_graph_walk_graphs()
    engine_types = model.get_node_engine_types()

    assert set(walks) == {"prefill", "decode"}
    assert isinstance(walks["decode"], Loop)
    assert engine_types == {"LLM": EngineType.KV_CACHE}


def test_glm52_kv_cache_is_mla_latent_layout():
    model = _make_model()
    (kv,) = model.get_kv_cache_config()

    assert kv.num_layers == 78
    # MLA: one shared latent "head" of width kv_lora_rank + qk_rope_head_dim,
    # not num_kv_heads x head_dim.
    assert kv.num_kv_heads == 1
    assert kv.head_dim == 512 + 64
    assert kv.num_qo_heads == 64


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


def _make_model_k(k: int) -> Glm52Model:
    model = object.__new__(Glm52Model)
    cfg = Glm52ModelConfig()
    cfg.mtp_num_draft_tokens = k
    model.config = cfg
    return model


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
    """The MTP prefill computes [emitted, k drafts] and returns it as
    "text_inputs". An output with no declared edge is UNROUTED — the worker
    drops it — so without this edge the prefill's whole sync+draft pass was
    wasted TTFT work and the first decode step ran unspeculated, injecting
    one artificial n_acc=0 per request into the acceptance histogram (it
    deflates measured p1). k=0 must keep the byte-identical old walk."""
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
    model's initial signal is named "text_inputs" — the PROMPT. So at the
    prefill->decode transition, persist_signals["text_inputs"] is already
    populated with the prompt, and a transition that reads that key hands
    decode the entire prompt back as its first step: measured as a 17-row
    decode step with no capture bucket (eager trunk + a diverged token
    stream), and with the prefill-draft edge also on, a p1 acceptance
    collapse 0.76 -> 0.18.

    The draft bundle therefore travels under a dedicated name that cannot
    collide. This test feeds the prompt in under BOTH names to prove the
    transition ignores the prompt one."""
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

    # exactly what the conductor holds at the transition: the prompt is
    # already sitting under "text_inputs", seeded from initial_signals.
    prompt_and_token = {"text_inputs": ["PROMPT"], "new_token": ["tok"]}
    for flag in ("0", "1"):
        # Assert under BOTH flag states: the guarantee is the NAME, not the
        # gating. A collision would be a bug even with the feature on.
        monkeypatch.setenv("MSTAR_GLM52_MTP_PREFILL_DRAFTS", flag)
        res = _seed_from(prompt_and_token)
        assert res.inputs[0].tensor_info == ["tok"], (
            f"flag={flag}: decode was seeded with the PROMPT instead of the "
            "emitted token")
        assert "PROMPT" not in res.unpersist_tensors


def test_glm52_decode_ignores_a_persisted_bundle_when_the_flag_is_off(monkeypatch):
    """The READ gate, which is the whole point of gating both halves.

    get_graph_walk_graphs is evaluated independently in the conductor and in
    every worker, so a split-flag deployment can have a worker persisting a
    bundle that a conductor with the flag OFF would consume. Gating only the
    write path is what regressed the "off" arm on 2026-08-10; without this
    test that lesson is documented but unenforced — reverting the read gate
    alone leaves the rest of the suite green."""
    from mstar.model.glm52.submodules import MTP_DRAFT_BUNDLE

    monkeypatch.setenv("MSTAR_GLM52_MTP_PREFILL_DRAFTS", "0")  # default is ON since 08-19
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
    # The bundle arrives under its dedicated name; the decode node still
    # consumes it as its "text_inputs" input.
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
    raises; kv_cache_engine catches only AllocationFailedError, so the
    escape reaches _handle_main_loop_error and fails every CO-BATCHED
    request. A cap at max_seq_len lets a long request iterate until it trips
    that guard, converting a silent per-request truncation into a batch
    kill. Tried 2026-08-10 and reverted; this pins the revert.

    (Asserting the cap is strictly below the guard is the property. Merely
    asserting it equals whatever the code sets would restate the line under
    test and pass either way — which is what the first version of this test
    did.)"""
    model = _make_model_k(0)
    cfg = model.config
    decode = model.get_graph_walk_graphs()["decode"]
    # The guard's bound is whichever limit preprocess actually compares
    # against — index_topk with DSA off, max_seq_len with it on. Asserting
    # index_topk unconditionally checks a field the guard does not use under
    # dsa_long_context, and happens to look right today only because both
    # default to 2048.
    guard = cfg.max_seq_len if cfg.dsa_long_context else cfg.index_topk
    assert decode.max_iters < guard, (
        f"loop cap {decode.max_iters} can reach the batch-killing context "
        f"guard at {guard}")
    # check_stop is the only thing enforcing the real per-request budget,
    # because the decode edge carries no conductor_new_token.
    assert not any(e.conductor_new_token for e in decode.section.outputs)


def test_glm52_mtp_declares_greedy_default_but_honors_explicit_asks():
    """MTP v1 decode is raw argmax. A bare request must serve coherently
    (greedy declared) rather than inherit config temperature=1.0 and be
    refused by prepare_inputs; an EXPLICIT non-greedy ask must survive to
    that refusal, because silently ignoring an ask is the failure mode."""
    assert _make_model_k(2).get_sampling_config("LLM").temperature == 0.0
    assert _make_model_k(2).get_sampling_config(
        "LLM", {"temperature": 0.7}).temperature == 0.7
    # k=0 keeps the model default — speculation is what forces greedy.
    k0 = _make_model_k(0)
    assert k0.get_sampling_config("LLM").temperature == k0.config.temperature


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


def _make_submodule(config) -> Glm52LLMSubmodule:
    # check_stop only reads self.config; skip nn.Module init / weights.
    sub = object.__new__(Glm52LLMSubmodule)
    sub.config = config
    return sub


def _fwd_info(max_tokens=100, ignore_eos=False, iters=0) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id="r0",
        graph_walk="decode",
        requires_cfg=False,
        fwd_index=0,
        random_seed=0,
        max_tokens=max_tokens,
        sampling_config={"LLM": MultiSamplingConfig(
            main=SamplingConfig(ignore_eos=ignore_eos))},
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
    # all 296 captures on 08-07. Default stays compile-on.
    sub = _make_submodule(Glm52ModelConfig.reduced())
    assert all(
        c.compile for c in sub.get_cuda_graph_configs(torch.device("cpu"))
    )
    monkeypatch.setenv("MSTAR_GLM52_GRAPH_COMPILE", "0")
    assert not any(
        c.compile for c in sub.get_cuda_graph_configs(torch.device("cpu"))
    )


class _FakeSeqState:
    def __init__(self, start):
        self.position_id_start = start


class _FakeCacheManager:
    def __init__(self, starts):
        self._starts = starts
        self.request_ids = list(starts)

    def set_active_label(self, label):
        pass

    def plan_attention(self, **kwargs):
        pass

    def plan_rope(self, **kwargs):
        pass

    def _get_state(self, rid, label):
        return _FakeSeqState(self._starts[rid])


class _FakeEngineInputs:
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager


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
    engine_inputs = _FakeEngineInputs(_FakeCacheManager(starts))
    return sub.preprocess("prefill", engine_inputs, inputs)


def test_glm52_preprocess_supplies_eager_last_token_indices():
    sub = _make_submodule(Glm52ModelConfig())
    out = _preprocess(sub, {"r0": 0, "r1": 0}, seq_len=16)
    assert torch.equal(out["last_token_indices"], torch.tensor([15, 31]))


def test_glm52_preprocess_refuses_context_beyond_dsa_window():
    # Dense MLA == DSA only within the top-2048 window (Phase C lifts this).
    sub = _make_submodule(Glm52ModelConfig())
    _preprocess(sub, {"r0": 2032}, seq_len=16)  # exactly 2048: allowed
    with pytest.raises(RuntimeError, match="Phase C"):
        _preprocess(sub, {"r0": 2040}, seq_len=16)


def test_glm52_postprocess_byte_mode_never_touches_tokenizer():
    m = object.__new__(Glm52Model)
    m._tokenizer_mode = "byte"
    m._tokenizer = None
    out = m.postprocess(torch.tensor([72, 105]), "text")
    assert out == b"Hi"
    assert m._tokenizer is None  # no lazy HF download triggered
