import sys

sys.path.insert(0, ".")


import pytest

from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.base import EngineType
from mstar.graph.base import Loop
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.glm52_model import Glm52Model


def _make_model() -> Glm52Model:
    # Skip __init__ (tokenizer download); the contract under test only
    # needs the config.
    model = object.__new__(Glm52Model)
    model.config = Glm52ModelConfig()
    return model


def test_glm52_registered():
    # The registry imports every model; qwen3_omni pulls GPU-only deps
    # (flashinfer) at module level, so skip where those aren't installed
    # (macOS dev) and assert fully on CI / the cluster.
    registry = pytest.importorskip(
        "mstar.model.registry",
        reason="full registry import needs GPU-only deps (flashinfer)",
    )

    assert registry.MODEL_REGISTRY["glm52"] is Glm52Model
    assert registry.HF_MODELS["glm52"]["model_path_hf"] == "zai-org/GLM-5.2"


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
