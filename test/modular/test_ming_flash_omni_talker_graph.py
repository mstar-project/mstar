"""Tests for the talker graph walk + Thinker->Talker bridge (step 6e-3).

Covers the talker-enabled graph topology: the `talker` walk, the
Talker partition, the Thinker->Talker streaming connection, the
audio sample rate, and the Talker partition state machine. Plus the
thinker-only path stays unchanged (talker config absent → no Talker
partition / walk).

All tests build a bare MingFlashOmniModel via __new__ + injected
config — no checkpoint load. The detokenize/re-tokenize text bridge
is exercised with stub tokenizers.
"""

from __future__ import annotations

import pytest

from mminf.conductor.request_info import CurrentForwardConductorMetadata
from mminf.engine.base import EngineType
from mminf.graph.base import GraphNode, Loop
from mminf.streaming.topology import StreamingGraphEdge
from mminf.model.ming_omni_flash.config import (
    AudioEncoderConfig,
    AudioVAEConfig,
    DiTBlockConfig,
    MingFlashOmniModelConfig,
    TalkerConfig,
    TalkerLLMConfig,
    ThinkerLLMConfig,
    VisionEncoderConfig,
)
from mminf.model.ming_omni_flash.ming_omni_flash_model import MingFlashOmniModel


def _talker_config() -> TalkerConfig:
    return TalkerConfig(
        steps=2, patch_size=2, history_patch_size=2, cfg_strength=2.0,
        llm=TalkerLLMConfig(hidden_size=32, num_hidden_layers=1),
        flowmodel=DiTBlockConfig(depth=1, hidden_size=32, num_heads=2, in_channels=4),
        aggregator=DiTBlockConfig(depth=1, hidden_size=32, num_heads=2, in_channels=4),
        vae=AudioVAEConfig(sample_rate=44100, patch_size=-1, latent_dim=4),
    )


def _model(with_talker: bool) -> MingFlashOmniModel:
    inst = MingFlashOmniModel.__new__(MingFlashOmniModel)
    inst.config = MingFlashOmniModelConfig(
        local_dir="",
        mlp_depth=2,
        thinker_llm=ThinkerLLMConfig(),
        vision=VisionEncoderConfig(),
        audio_encoder=AudioEncoderConfig(),
        talker=_talker_config() if with_talker else None,
    )
    inst._submodule_cache = {}
    return inst


# ---------------------------------------------------------------------------
# Thinker-only (talker absent) — unchanged from step 5c
# ---------------------------------------------------------------------------


def test_thinker_only_no_talker_walk() -> None:
    walks = _model(with_talker=False).get_graph_walk_graphs()
    assert "talker" not in walks
    assert set(walks) == {
        "prefill_text", "prefill_audio", "prefill_vision",
        "prefill_video", "thinker_decode",
    }


def test_thinker_only_decode_has_no_streaming_edge() -> None:
    """Without a talker, the decode loop emits only text edges (no Talker stream)."""
    walks = _model(with_talker=False).get_graph_walk_graphs()
    loop = walks["thinker_decode"]
    assert isinstance(loop, Loop)
    assert not any(
        isinstance(e, StreamingGraphEdge) for e in loop.section.outputs
    )


def test_thinker_only_single_partition() -> None:
    parts = _model(with_talker=False).get_partitions()
    assert [p.name for p in parts] == ["Thinker"]


def test_thinker_only_topology_no_connections() -> None:
    topo = _model(with_talker=False).get_partition_topology()
    assert topo.partitions == ["Thinker"]
    assert topo.connections == []


# ---------------------------------------------------------------------------
# Talker enabled — graph structure
# ---------------------------------------------------------------------------


def test_talker_walk_present_and_emits_audio() -> None:
    walks = _model(with_talker=True).get_graph_walk_graphs()
    assert "talker" in walks
    node = walks["talker"]
    assert isinstance(node, GraphNode)
    assert node.name == "Talker"
    assert set(node.input_names) == {"thinker_tokens"}
    assert len(node.outputs) == 1
    assert node.outputs[0].name == "audio_chunk"
    assert node.outputs[0].output_modality == "audio"


def test_decode_loop_streams_thinker_tokens_to_talker() -> None:
    walks = _model(with_talker=True).get_graph_walk_graphs()
    loop = walks["thinker_decode"]
    stream_edges = [
        e for e in loop.section.outputs if isinstance(e, StreamingGraphEdge)
    ]
    assert len(stream_edges) == 1
    assert stream_edges[0].name == "thinker_tokens"
    assert stream_edges[0].target_partition == "Talker"
    # Text edges still present (client text + decode loopback).
    text_edges = [e.name for e in loop.section.outputs if not isinstance(e, StreamingGraphEdge)]
    assert "new_token" in text_edges
    assert "text_inputs" in text_edges


def test_talker_partition_listed_with_producer() -> None:
    parts = {p.name: p for p in _model(with_talker=True).get_partitions()}
    assert set(parts) == {"Thinker", "Talker"}
    talker = parts["Talker"]
    assert talker.graph_walks == {"talker"}
    assert talker.initial_walk is None
    assert talker.producer_partitions == ["Thinker"]


def test_talker_topology_connects_thinker_to_talker() -> None:
    topo = _model(with_talker=True).get_partition_topology()
    assert set(topo.partitions) == {"Thinker", "Talker"}
    assert len(topo.connections) == 1
    conn = topo.connections[0]
    assert conn.from_partition == "Thinker"
    assert conn.to_partition == "Talker"
    assert conn.edge_name == "thinker_tokens"
    # The chunk policy must keep the consumer alive past producer-done.
    policy = conn.chunk_policy_factory()
    assert policy.continue_after_producer_done() is True


def test_node_engine_types_registers_talker_stateless() -> None:
    types = _model(with_talker=True).get_node_engine_types()
    assert types["Talker"] == EngineType.STATELESS


# ---------------------------------------------------------------------------
# Output sample rate
# ---------------------------------------------------------------------------


def test_output_sample_rate_uses_talker_vae() -> None:
    assert _model(with_talker=True).get_output_sample_rate("audio") == 44100


def test_output_sample_rate_falls_back_without_talker() -> None:
    # Base class default (no talker) — just assert it doesn't raise and
    # returns a positive int.
    sr = _model(with_talker=False).get_output_sample_rate("audio")
    assert isinstance(sr, int) and sr > 0


# ---------------------------------------------------------------------------
# Talker partition state machine
# ---------------------------------------------------------------------------


class _Conn:
    """Stub StreamingConnectionState."""
    def __init__(self, producer_done: bool) -> None:
        self.producer_done = producer_done
        self.token_count = 0
        self.consumed_count = 0


def test_talker_initial_args_audio_output_keeps_partition_alive() -> None:
    model = _model(with_talker=True)
    args = model.get_initial_forward_pass_args(
        partition_name="Talker",
        input_modalities=["text"],
        output_modalities=["audio"],
        input_signals={},
    )
    assert args.full_metadata.graph_walk == "talker"
    assert args.request_done is False


def test_talker_initial_args_no_audio_output_done_immediately() -> None:
    model = _model(with_talker=True)
    args = model.get_initial_forward_pass_args(
        partition_name="Talker",
        input_modalities=["text"],
        output_modalities=["text"],   # no audio requested
        input_signals={},
    )
    assert args.request_done is True


def test_talker_forward_waits_for_producer_done() -> None:
    model = _model(with_talker=True)
    meta = CurrentForwardConductorMetadata(
        input_modalities=["text"], output_modalities=["audio"],
        graph_walk="talker", is_prefill=False,
    )
    # Producer still running → no-op step (no fire, not done).
    args = model.get_partition_forward_pass_args(
        partition_name="Talker", partition_metadata=meta,
        persist_signals={}, new_tokens={},
        incoming_connections=[_Conn(producer_done=False)],
    )
    assert args.request_done is False
    assert args.inputs == []


def test_talker_forward_fires_once_then_done() -> None:
    model = _model(with_talker=True)
    meta = CurrentForwardConductorMetadata(
        input_modalities=["text"], output_modalities=["audio"],
        graph_walk="talker", is_prefill=False,
    )
    # Producer done → fire the talker walk.
    args1 = model.get_partition_forward_pass_args(
        partition_name="Talker", partition_metadata=meta,
        persist_signals={}, new_tokens={},
        incoming_connections=[_Conn(producer_done=True)],
    )
    assert args1.full_metadata.graph_walk == "talker"
    assert len(args1.inputs) == 1
    assert args1.inputs[0].name == "thinker_tokens"
    assert args1.request_done is False
    # Next invocation → already fired → done.
    args2 = model.get_partition_forward_pass_args(
        partition_name="Talker", partition_metadata=args1.full_metadata,
        persist_signals={}, new_tokens={},
        incoming_connections=[_Conn(producer_done=True)],
    )
    assert args2.request_done is True


# ---------------------------------------------------------------------------
# Thinker->Talker text bridge
# ---------------------------------------------------------------------------


class _StubThinkerTok:
    def decode(self, ids, skip_special_tokens=True):
        # Toy: join ids as chars.
        return "".join(chr(65 + (i % 26)) for i in ids)


class _StubTalkerTok:
    def __call__(self, text, return_tensors="pt"):
        import torch
        ids = torch.tensor([[ord(c) for c in text]], dtype=torch.long)
        return type("O", (), {"input_ids": ids})()


def test_text_bridge_decodes_then_reencodes() -> None:
    import torch
    model = _model(with_talker=True)
    model.tokenizer = _StubThinkerTok()
    model._talker_tokenizer = _StubTalkerTok()
    thinker_ids = torch.tensor([0, 1, 2])   # -> "ABC"
    out = model.thinker_text_to_talker_inputs(thinker_ids)
    assert out.tolist() == [ord("A"), ord("B"), ord("C")]


def test_text_bridge_raises_without_thinker_tokenizer() -> None:
    import torch
    model = _model(with_talker=True)
    model.tokenizer = None
    model._talker_tokenizer = _StubTalkerTok()
    with pytest.raises(RuntimeError, match="thinker tokenizer not loaded"):
        model.thinker_text_to_talker_inputs(torch.tensor([1, 2]))
