"""Structural tests for the Nemotron-Duplex M* engine integration: the four-partition
full-duplex walk graphs, topology, aux sampling, and forward-pass-args routing.

These validate everything the engine needs to *wire* the model (no weights / GPU):
``get_worker_graphs`` resolves every edge route + cross-partition streaming connection,
so a dangling edge name or unresolved partition fails loudly here.
"""
from pathlib import Path

from mstar.engine.base import EngineType
from mstar.model.nemotron_duplex.config import NemotronDuplexConfig
from mstar.model.nemotron_duplex.nemotron_duplex_model import NemotronDuplexModel

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "nemotron_duplex.yaml"

WALKS = {"encode", "prefill_text", "decode", "talker_decode", "codec_chunk"}


def _make_model() -> NemotronDuplexModel:
    model = object.__new__(NemotronDuplexModel)
    model.config = NemotronDuplexConfig()
    model._submodule_cache = {}
    return model


def test_duplex_declares_all_walks():
    assert set(_make_model().get_graph_walk_graphs()) == WALKS


def test_duplex_engine_types():
    assert _make_model().get_node_engine_types() == {
        "conformer_encoder": EngineType.STATELESS,
        "nano_llm": EngineType.KV_CACHE,
        "eartts_talker": EngineType.KV_CACHE,
        "audio_codec": EngineType.STATELESS,
    }


def test_duplex_partition_producer_chain():
    parts = {p.name: p for p in _make_model().get_partitions()}
    assert set(parts) == {"Encoder", "LLM", "Talker", "Codec"}
    assert parts["Encoder"].producer_partitions == []
    assert parts["LLM"].producer_partitions == ["Encoder"]
    assert parts["Talker"].producer_partitions == ["LLM"]
    assert parts["Codec"].producer_partitions == ["Talker"]


def test_duplex_topology_routes():
    conns = {(c.from_partition, c.to_partition, c.edge_name)
             for c in _make_model().get_partition_topology().connections}
    assert conns == {
        ("Encoder", "LLM", "audio_frame"),
        ("LLM", "Talker", "new_token"),
        ("Talker", "Codec", "codec_tokens"),
    }


def test_duplex_nano_has_function_aux_channel():
    assert "function" in _make_model().get_aux_sampling_configs("nano_llm")
    assert _make_model().get_aux_sampling_configs("audio_codec") == {}


def test_duplex_worker_graphs_derive_all_walks():
    model = _make_model()
    worker_graphs = model.get_worker_graphs(str(CONFIG_PATH))
    by_walk = {next(iter(wg.graph_walks)): wg for wg in worker_graphs}
    assert set(by_walk) == WALKS
    assert by_walk["decode"].consumes_stream is True
    assert by_walk["codec_chunk"].consumes_stream is True


def test_duplex_initial_partition_routing():
    model = _make_model()
    sig = {"audio_features": ["a"], "text_inputs": ["t"]}
    expected = {"Encoder": "encode", "LLM": "prefill_text",
                "Talker": "talker_decode", "Codec": "codec_chunk"}
    for pname, walk in expected.items():
        fpa = model.get_initial_forward_pass_args(pname, ["audio"], ["audio", "text"], sig)
        assert fpa.full_metadata.graph_walk == walk
    # no system prompt -> LLM starts straight in the decode loop
    fpa = model.get_initial_forward_pass_args("LLM", ["audio"], ["audio"], {"audio_features": ["a"]})
    assert fpa.full_metadata.graph_walk == "decode"
    # audio not requested -> streaming output partitions are immediately done
    assert model.get_initial_forward_pass_args("Codec", ["audio"], ["text"], sig).request_done


def test_duplex_llm_prefill_to_decode_transition():
    model = _make_model()
    meta = model._meta(["audio"], ["audio"], "prefill_text", True)
    fpa = model.get_partition_forward_pass_args("LLM", meta, {})
    assert fpa.full_metadata.graph_walk == "decode"
    assert fpa.full_metadata.is_prefill is False
