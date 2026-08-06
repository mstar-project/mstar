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
from mstar.utils.sampling import SamplingConfig


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
        sampling_config={"LLM": SamplingConfig(ignore_eos=ignore_eos)},
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
