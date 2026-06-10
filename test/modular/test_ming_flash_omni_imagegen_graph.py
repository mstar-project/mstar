"""Tests for the ImageGen graph walk + partition wiring (step 9b).

Covers the imagegen-enabled graph topology: the ``imagegen`` walk, the ImageGen
partition, the Thinker->ImageGen streaming connection, the STATELESS engine
type, and the unchanged paths when no image_gen config is present.

All tests build a bare MingFlashOmniModel via __new__ + injected config — no
checkpoint load, no diffusers.
"""

from __future__ import annotations

from mminf.engine.base import EngineType
from mminf.graph.base import GraphNode
from mminf.model.ming_omni_flash.config import (
    AudioEncoderConfig,
    ImageGenConfig,
    MingFlashOmniModelConfig,
    ThinkerLLMConfig,
    VisionEncoderConfig,
)
from mminf.model.ming_omni_flash.ming_omni_flash_model import MingFlashOmniModel


def _model(with_imagegen: bool) -> MingFlashOmniModel:
    inst = MingFlashOmniModel.__new__(MingFlashOmniModel)
    inst.config = MingFlashOmniModelConfig(
        local_dir="",
        mlp_depth=2,
        thinker_llm=ThinkerLLMConfig(),
        vision=VisionEncoderConfig(),
        audio_encoder=AudioEncoderConfig(),
        image_gen=ImageGenConfig() if with_imagegen else None,
    )
    inst._submodule_cache = {}
    return inst


# ---------------------------------------------------------------------------
# ImageGen absent — unchanged
# ---------------------------------------------------------------------------


def test_no_imagegen_walk_when_config_absent() -> None:
    walks = _model(with_imagegen=False).get_graph_walk_graphs()
    assert "imagegen" not in walks


def test_no_imagegen_partition_when_config_absent() -> None:
    parts = [p.name for p in _model(with_imagegen=False).get_partitions()]
    assert "ImageGen" not in parts


def test_no_imagegen_engine_type_when_config_absent() -> None:
    types = _model(with_imagegen=False).get_node_engine_types()
    assert "ImageGen" not in types


# ---------------------------------------------------------------------------
# ImageGen enabled — graph structure
# ---------------------------------------------------------------------------


def test_imagegen_walk_present_and_emits_image() -> None:
    walks = _model(with_imagegen=True).get_graph_walk_graphs()
    assert "imagegen" in walks
    node = walks["imagegen"]
    assert isinstance(node, GraphNode)
    assert node.name == "ImageGen"
    assert set(node.input_names) == {"thinker_hidden_states"}
    assert len(node.outputs) == 1
    assert node.outputs[0].name == "image"
    assert node.outputs[0].output_modality == "image"


def test_imagegen_partition_listed_with_producer() -> None:
    parts = {p.name: p for p in _model(with_imagegen=True).get_partitions()}
    assert "ImageGen" in parts
    ig = parts["ImageGen"]
    assert ig.graph_walks == {"imagegen"}
    assert ig.initial_walk is None
    assert ig.producer_partitions == ["Thinker"]


def test_imagegen_topology_connects_thinker_to_imagegen() -> None:
    topo = _model(with_imagegen=True).get_partition_topology()
    assert set(topo.partitions) == {"Thinker", "ImageGen"}
    assert len(topo.connections) == 1
    conn = topo.connections[0]
    assert conn.from_partition == "Thinker"
    assert conn.to_partition == "ImageGen"
    assert conn.edge_name == "thinker_hidden_states"
    # The chunk policy must keep the consumer alive past producer-done.
    policy = conn.chunk_policy_factory()
    assert policy.continue_after_producer_done() is True


def test_node_engine_types_registers_imagegen_stateless() -> None:
    types = _model(with_imagegen=True).get_node_engine_types()
    assert types["ImageGen"] == EngineType.STATELESS


def test_imagegen_does_not_disturb_thinker_walks() -> None:
    walks = _model(with_imagegen=True).get_graph_walk_graphs()
    # All five thinker walks remain present.
    for w in ("prefill_text", "prefill_audio", "prefill_vision", "prefill_video", "thinker_decode"):
        assert w in walks


def test_imagegen_and_talker_coexist_when_both_absent_is_thinker_only() -> None:
    """Sanity: with neither talker nor imagegen, topology is single-partition."""
    topo = _model(with_imagegen=False).get_partition_topology()
    assert topo.partitions == ["Thinker"]
    assert topo.connections == []


# ---------------------------------------------------------------------------
# get_submodule dispatch (no load — just the unknown-node error path)
# ---------------------------------------------------------------------------


def test_get_submodule_unknown_node_lists_imagegen() -> None:
    import pytest

    model = _model(with_imagegen=True)
    with pytest.raises(ValueError, match="ImageGen"):
        model.get_submodule("NotARealNode")
