"""Structural tests for the Nemotron-Duplex M* integration: the four-partition
full-duplex walk graphs, topology, declared resources, per-request configs and
forward-pass-args routing.

These validate everything the engine needs to *wire* the model (no weights / GPU):
``get_worker_graphs`` resolves every edge route + cross-partition streaming connection,
so a dangling edge name or unresolved partition fails loudly here.
"""
from pathlib import Path

import torch

from mstar.engine.resources import (
    AttentionSpec,
    KVReqConfig,
    KVSpec,
    PositionSpec,
    SamplerSpec,
    SamplingReqConfig,
    resolve_spec_dependencies,
)
from mstar.engine.resources.linear_attn.config import LinearAttnSpec, LinearAttnVariant
from mstar.engine.resources.recurrent import Mamba2Geometry, RecurrentStateSpec
from mstar.model.nemotron_duplex.config import (
    MAMBA,
    MAMBA_STATE,
    NANO_ATTN,
    NANO_KV,
    NANO_SAMPLER,
    TALKER_ATTN,
    TALKER_KV,
    TALKER_POS,
    NemotronDuplexConfig,
)
from mstar.model.nemotron_duplex.nemotron_duplex_model import NemotronDuplexModel
from mstar.model.registry import HF_MODELS, get_model_class
from mstar.streaming.chunk_policy import FixedChunkPolicy

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "nemotron_duplex.yaml"

WALKS = {"encode", "prefill_text", "decode", "talker_decode", "codec_chunk"}
NODES = {"conformer_encoder", "nano_llm", "eartts_talker", "audio_codec"}


def _make_model() -> NemotronDuplexModel:
    model = object.__new__(NemotronDuplexModel)
    model.config = NemotronDuplexConfig()
    model._submodule_cache = {}
    return model


def test_duplex_is_registered():
    assert get_model_class("nemotron_duplex") is NemotronDuplexModel
    assert HF_MODELS["nemotron_duplex"]["model_path_hf"] == "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"


def test_duplex_declares_all_walks_and_nodes():
    model = _make_model()
    assert set(model.get_graph_walk_graphs()) == WALKS
    assert set(model.nodes) == NODES


def test_duplex_node_resources():
    """The nano declares a paged KV over its 4 attention layers with attention
    planned on it (NoPE: no position resource), a recurrent pool holding the 27
    Mamba-2 layers' conv + SSM state with the Mamba-2 resource planned on it,
    and one text sampler; the talker a paged KV over its 28 layers with attention
    and RoPE positions. The encoder and the codec own no engine resource."""
    model = _make_model()
    specs = model.get_node_resources()
    by_key = resolve_spec_dependencies(specs)          # unique keys, dependencies satisfied
    assert set(by_key) == {NANO_KV, NANO_ATTN, MAMBA_STATE, MAMBA, NANO_SAMPLER, TALKER_KV, TALKER_ATTN, TALKER_POS}
    nano_keys = {NANO_KV, NANO_ATTN, MAMBA_STATE, MAMBA, NANO_SAMPLER}
    assert all(spec.nodes == ({"nano_llm"} if spec.resource_key in nano_keys else {"eartts_talker"}) for spec in specs)

    eartts = model.config.eartts
    tkv = by_key[TALKER_KV]
    assert isinstance(tkv, KVSpec)
    # the pool's head dim is the talker's 72 zero-padded to a size FlashInfer computes exactly
    assert (tkv.config.num_layers, tkv.config.num_kv_heads, tkv.config.head_dim) == (28, 16, 128)
    assert tkv.config.head_dim == eartts.kv_head_dim > eartts.head_dim
    assert tkv.config.max_seq_len > eartts.sliding_window + 37   # window + speaker warm-up
    assert isinstance(by_key[TALKER_ATTN], AttentionSpec) and by_key[TALKER_ATTN].config.kv_cache == TALKER_KV
    tpos = by_key[TALKER_POS]
    assert isinstance(tpos, PositionSpec) and tpos.config.kv_cache == TALKER_KV
    assert tpos.config.rotary_dim == eartts.head_dim and tpos.config.interleave is False

    nano = model.config.nano
    pool = by_key[MAMBA_STATE]
    assert isinstance(pool, RecurrentStateSpec)
    assert pool.config.num_layers == nano.num_mamba_layers == 27
    geom = Mamba2Geometry.from_blocks(pool.config.blocks)
    dims = (geom.num_heads, geom.head_dim, geom.state_size, geom.n_groups, geom.conv_kernel_size)
    assert dims == (128, 80, 128, 8, 4)
    assert pool.config.blocks["ssm"].dtype is torch.float32 and pool.config.blocks["conv"].shape == (12288, 3)
    assert pool.config.usable_slots == model.DEFAULT_MAMBA_SLOTS      # the sink is extra; padding rows take none
    mamba = by_key[MAMBA]
    assert isinstance(mamba, LinearAttnSpec) and mamba.config.variant is LinearAttnVariant.MAMBA2
    assert mamba.config.recurrent_state == MAMBA_STATE and mamba.depends_on() == {MAMBA_STATE}

    kv = by_key[NANO_KV]
    assert isinstance(kv, KVSpec)
    nano = model.config.nano
    assert kv.config.num_layers == nano.num_attention_layers == 4
    assert kv.config.num_kv_heads == nano.num_key_value_heads
    assert kv.config.num_qo_heads == nano.num_attention_heads
    assert kv.config.head_dim == nano.head_dim

    attn = by_key[NANO_ATTN]
    assert isinstance(attn, AttentionSpec) and attn.config.kv_cache == NANO_KV
    sampler = by_key[NANO_SAMPLER]
    assert isinstance(sampler, SamplerSpec) and sampler.vocab_size == model.config.vocab_size


def test_duplex_request_resource_configs():
    model = _make_model()
    cfg = model.get_request_resource_configs({}, None)
    assert set(cfg) == {NANO_SAMPLER, TALKER_KV}
    assert isinstance(cfg[TALKER_KV], KVReqConfig)
    assert cfg[TALKER_KV].get_labels("eartts_talker", "talker_decode") == ["main", "uncond"]
    text = cfg[NANO_SAMPLER]
    assert isinstance(text, SamplingReqConfig)
    assert text.temperature == model.config.temperature and text.top_p == model.config.top_p
    # request knobs override the config defaults
    greedy = model.get_request_resource_configs({}, {"temperature": 0.0, "ignore_eos": True})[NANO_SAMPLER]
    assert greedy.temperature == 0.0 and greedy.ignore_eos is True


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
    fpa = model.get_partition_forward_pass_args("LLM", meta, {"prev_text": ["pt"], "prev_func": ["pf"]})
    assert fpa.full_metadata.graph_walk == "decode"
    assert fpa.full_metadata.is_prefill is False
    # the primed carry-in tokens persisted by prefill_text feed decode iteration 0
    assert {e.name for e in fpa.inputs} == {"prev_text", "prev_func"}


def test_duplex_stream_connection_policies():
    """The duplex loops are 1:1 stream-driven and terminate on stream end:
    Encoder→LLM and LLM→Talker release one item per step with
    continue_after_done=False (else the loops would spin past their input);
    Talker→Codec releases NON-overlapping chunks of ``codec_chunk_frames`` (the
    codec keeps its own left-context, so re-delivering overlap balloons the
    audio ~window/chunk×)."""
    model = _make_model()
    conns = {c.edge_name: c.chunk_policy_factory()
             for c in model.get_partition_topology().connections}

    for edge in ("audio_frame", "new_token"):
        pol = conns[edge]
        assert isinstance(pol, FixedChunkPolicy)
        assert pol.next_chunk_size(10) == 1
        assert pol.continue_after_producer_done() is False

    codec = conns["codec_tokens"]
    assert isinstance(codec, FixedChunkPolicy)
    assert codec.next_chunk_size(100) == model.config.eartts.codec_chunk_frames
    assert codec.continue_after_producer_done() is False


def test_duplex_process_prompt_seeds_frame0_feedback():
    """The frame-0 decode step lists prev_text / prev_func as inputs but has no
    prior sampled token; process_prompt must seed them (BOS / PAD) or the decode
    loop never becomes ready (the original hang)."""
    model = _make_model()
    out = model.process_prompt(
        None, ["audio"], ["audio", "text"],
        tensors={"audio_inputs": [torch.zeros(16000)]},
    )
    assert "audio_features" in out
    assert int(out["prev_text"][0].item()) == model.config.text_bos_id
    assert int(out["prev_func"][0].item()) == model.config.text_pad_id


def test_duplex_no_prompt_seeds_initial_decode_inputs():
    """Audio-only start (no system prompt): the LLM partition jumps straight to
    the decode loop and its initial inputs carry the seeded prev_text / prev_func
    so iteration 0's readiness gate fires."""
    model = _make_model()
    sig = {"audio_features": ["a"], "prev_text": ["pt"], "prev_func": ["pf"]}
    fpa = model.get_initial_forward_pass_args("LLM", ["audio"], ["audio"], sig)
    assert fpa.full_metadata.graph_walk == "decode"
    assert {e.name for e in fpa.inputs} == {"prev_text", "prev_func"}
