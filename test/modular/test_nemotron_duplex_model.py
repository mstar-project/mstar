"""Structural tests for the Nemotron-Duplex M* integration: the four-partition
full-duplex walk graphs, topology, declared resources, per-request configs and
forward-pass-args routing.

These validate everything the engine needs to *wire* the model (no weights / GPU):
``get_worker_graphs`` resolves every edge route + cross-partition streaming connection,
so a dangling edge name or unresolved partition fails loudly here.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
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
from mstar.model.base import ClientChunk
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
from mstar.streaming.topology import client_stream_limits
from mstar.validation.client_streams import check_client_streams

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "nemotron_duplex.yaml"

WALKS = {"encode", "decode", "talker_decode", "codec_chunk"}
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
        ("client", "Encoder", "audio_in"),
        ("Encoder", "LLM", "audio_frame"),
        ("LLM", "Talker", "new_token"),
        ("Talker", "Codec", "codec_tokens"),
    }


def test_duplex_worker_graphs_derive_all_walks():
    model = _make_model()
    worker_graphs = model.get_worker_graphs(str(CONFIG_PATH))
    by_walk = {next(iter(wg.graph_walks)): wg for wg in worker_graphs}
    assert set(by_walk) == WALKS
    assert by_walk["encode"].consumes_stream is True
    assert by_walk["decode"].consumes_stream is True
    assert by_walk["codec_chunk"].consumes_stream is True
    # never a speculation target (its input is the client's); async scheduling
    # lets the worker overlap another node's step with it
    encoder_node = by_walk["encode"].section.section                 # Loop -> GraphNode
    assert encoder_node.name == "conformer_encoder" and encoder_node.enable_async_scheduling is True
    # the talker's next step reads the codes its previous step stored in
    # postprocess: it must not be built speculatively while that step runs
    talker_node = by_walk["talker_decode"].section.section          # Loop -> GraphNode
    assert talker_node.name == "eartts_talker" and talker_node.enable_async_scheduling is False
    assert by_walk["decode"].section.section.enable_async_scheduling is True


def test_duplex_initial_partition_routing():
    model = _make_model()
    sig = {"text_inputs": ["t"], "has_prompt": ["h"], "prev_text": ["p"], "prev_func": ["f"]}
    expected = {"Encoder": "encode", "LLM": "decode",
                "Talker": "talker_decode", "Codec": "codec_chunk"}
    for pname, walk in expected.items():
        fpa = model.get_initial_forward_pass_args(pname, ["audio"], ["audio", "text"], sig)
        assert fpa.full_metadata.graph_walk == walk
        if pname != "LLM":
            # self-triggered: the Encoder by the client's audio stream
            assert fpa.inputs == [] and fpa.request_done is False
    # with a system prompt the LLM's first step is prompt + frame: flagged so the
    # worker keeps it off the fixed-shape decode capture; the prompt rides the loop
    fpa = model.get_initial_forward_pass_args("LLM", ["audio", "text"], ["audio", "text"], sig)
    assert fpa.step_metadata == {"prompt_pending": True} and fpa.full_metadata.is_prefill is True
    assert {e.name for e in fpa.inputs} == {"prev_text", "prev_func", "text_inputs"}
    # no system prompt -> same walk, nothing pending
    no_prompt = {"text_inputs": ["s"]}
    fpa = model.get_initial_forward_pass_args("LLM", ["audio"], ["audio"], no_prompt)
    assert fpa.full_metadata.graph_walk == "decode" and fpa.step_metadata == {"prompt_pending": False}


def test_duplex_llm_after_the_first_step_nothing_is_pending():
    model = _make_model()
    meta = model._meta(["audio", "text"], ["audio"], "decode", True)
    fpa = model.get_partition_forward_pass_args("LLM", meta, {"prev_text": ["pt"], "prev_func": ["pf"]})
    assert fpa.full_metadata.graph_walk == "decode"
    assert fpa.full_metadata.is_prefill is False
    assert fpa.step_metadata == {"prompt_pending": False}
    assert {e.name for e in fpa.inputs} == {"prev_text", "prev_func"}
    # the decode node lists the prompt as a fed-back input, so a re-injected copy
    # of the seed can never outrank the node's own sentinel
    decode = model.get_graph_walk_graphs()["decode"]
    node = next(iter(decode.section.get_nodes().values()))
    assert "text_inputs" in node.input_names
    assert any(e.name == "text_inputs" and e.next_node == "nano_llm" for e in node.outputs)


def test_duplex_stream_connection_policies():
    """The duplex loops are 1:1 stream-driven and terminate on stream end:
    client→Encoder, Encoder→LLM and LLM→Talker release one item per step with
    continue_after_done=False (else the loops would spin past their input);
    Talker→Codec releases NON-overlapping chunks of ``codec_chunk_frames`` (the
    codec keeps its own left-context, so re-delivering overlap balloons the
    audio ~window/chunk×)."""
    model = _make_model()
    topology = model.get_partition_topology()
    conns = {c.edge_name: c.chunk_policy_factory() for c in topology.connections}
    (client,) = topology.client_connections()
    assert (client.edge_name, client.to_partition) == ("audio_in", "Encoder")

    for edge in ("audio_in", "audio_frame", "new_token"):
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
    ).new_input_tensors
    assert "audio_in" in out                     # an uploaded clip is the stream's first chunk
    assert int(out["prev_text"][0].item()) == model.config.text_bos_id
    assert int(out["prev_func"][0].item()) == model.config.text_pad_id
    assert out["text_inputs"][0].tolist() == [-1] and "has_prompt" not in out   # no prompt: the sentinel
    # after a system prompt the agent channel is held at PAD across the prompt, so
    # the first audio frame's previous agent token is PAD, not BOS (reference)
    model._tokenizer = lambda prompt, return_tensors: SimpleNamespace(input_ids=torch.tensor([[5, 6, 7]]))
    out_p = model.process_prompt(
        "You are NVIDIA Voice Chat.", ["audio", "text"], ["audio", "text"],
        tensors={"audio_inputs": [torch.zeros(16000)]},
    ).new_input_tensors
    assert int(out_p["prev_text"][0].item()) == model.config.text_pad_id
    assert out_p["text_inputs"][0].tolist() == [5, 6, 7] and "has_prompt" in out_p


def test_duplex_process_prompt_leaves_the_session_cap_to_audio_in():
    """Any length of upload is a first chunk (the encoder waits for enough audio,
    and fails a call that ends without it). Files that together pass the session
    pass through: audio_in's max_items, which the data worker charges every
    uploaded sample to, refuses them."""
    model = _make_model()
    sr = model.config.stt.sample_rate
    out = model.process_prompt(
        None, ["audio"], ["audio", "text"], tensors={"audio_inputs": [torch.zeros(320)]},
    ).new_input_tensors
    assert out["audio_in"][0].numel() == 320
    half = int(model.max_session_s * sr / 2) + 1
    out = model.process_prompt(None, ["audio"], ["audio", "text"],
                               tensors={"audio_inputs": [torch.zeros(half), torch.zeros(half)]}).new_input_tensors
    limits = client_stream_limits(model.get_partition_topology(), None)["audio_in"]
    assert sum(limits.item_size(w) for w in out["audio_in"]) > limits.max_items


def test_duplex_process_prompt_rejects_bad_requests():
    """Requests the duplex loop cannot serve are a ValueError (a 400), not a 500
    or a silently wrong reply: non-finite audio, or no audio output. A call may
    open with no audio: it streams in as appends."""
    model = _make_model()
    ok = {"audio_inputs": [torch.zeros(16000)]}
    model._tokenizer = lambda prompt, return_tensors: SimpleNamespace(input_ids=torch.tensor([[5, 6, 7]]))
    for tensors in (None, {}, {"audio_inputs": []}):
        out = model.process_prompt("hi", ["text"], ["audio", "text"], tensors=tensors).new_input_tensors
        assert "audio_in" not in out and out["text_inputs"][0].tolist() == [5, 6, 7]
    for bad in (float("nan"), float("inf"), -float("inf")):
        wav = torch.zeros(16000)
        wav[100] = bad
        with pytest.raises(ValueError, match="NaN or inf"):
            model.process_prompt(None, ["audio"], ["audio", "text"], tensors={"audio_inputs": [wav]})
    for omod in (["text"], ["image"], ["audio", "image"], []):
        with pytest.raises(ValueError, match="output modalities"):
            model.process_prompt(None, ["audio"], omod, tensors=ok)
    for omod in (["audio"], ["text", "audio"]):
        model.process_prompt(None, ["audio"], omod, tensors=ok)


def test_duplex_process_prompt_scales_down_over_full_scale_audio():
    """Float files can exceed full scale; past it the agent goes silent, so the
    peak is brought back to 1.0. Audio within full scale is left untouched."""
    model = _make_model()
    wav = torch.sin(torch.arange(16000) / 10.0)
    out = model.process_prompt(
        None, ["audio"], ["audio", "text"], tensors={"audio_inputs": [wav * 1e6]},
    ).new_input_tensors
    assert torch.allclose(out["audio_in"][0], wav, atol=1e-6)
    quiet = wav * 0.5
    out = model.process_prompt(None, ["audio"], ["audio", "text"], tensors={"audio_inputs": [quiet]}).new_input_tensors
    assert torch.equal(out["audio_in"][0], quiet)


def test_duplex_process_prompt_scales_a_multi_file_upload_as_one():
    """The files of an upload are one stretch of the call: one gain for all,
    so a loud file doesn't end up level with a quiet one."""
    model = _make_model()
    wav = torch.sin(torch.arange(16000) / 10.0)
    out = model.process_prompt(None, ["audio"], ["audio", "text"],
                               tensors={"audio_inputs": [wav * 4.0, wav * 0.5]}).new_input_tensors["audio_in"]
    assert torch.allclose(torch.cat(out), torch.cat([wav, wav / 8.0]), atol=1e-6)


def _append(tensors, input_metadata=None, text=None):
    """An append of ``tensors`` as the data worker hands it to the hook."""
    return ClientChunk(
        text=text, tensors=tensors, input_metadata=input_metadata or {},
        streams={key: [None] * len(items) for key, items in tensors.items()},
        model_kwargs={}, request_kwargs={},
    )


def test_duplex_audio_in_caps_a_call_at_the_session_in_samples():
    """The client stream, not the encoder, enforces the session cap, so a live
    call past it gets a 400 naming the limit; a deployment may lower the cap
    there but not raise it past what the talker KV and the loops were sized for."""
    model = _make_model()
    model.max_session_s = 2.0
    (client,) = model.get_partition_topology().client_connections()
    limits = client_stream_limits(model.get_partition_topology(), None)["audio_in"]
    assert limits == client.limits
    assert (limits.max_items, limits.unit) == (2 * 16000, "samples")
    assert limits.item_size(torch.zeros(1280)) == 1280
    lowered = client_stream_limits(model.get_partition_topology(),
                                   {"client_streams": {"audio_in": {"max_items": 16000}}})["audio_in"]
    assert lowered.max_items == 16000
    (problem,) = check_client_streams(model, {"client_streams": {"audio_in": {"max_items": None}}})
    assert "may lower" in problem


def test_duplex_appended_audio_becomes_stream_items_in_order():
    """Each appended file streams as ``audio_in`` items, checked like an upload
    but with no length limit of its own: a live chunk can be shorter than a frame,
    and audio_in's max_items caps the call's samples as it grows."""
    model = _make_model()
    short, loud = torch.zeros(320), torch.full((1280,), 4.0)
    items = model.process_stream_chunk(_append({"audio_inputs": [short, loud]}, {"audio_inputs": [{}, {}]}))["audio_in"]
    assert [w.numel() for w in items] == [320, 1280]
    assert float(items[1].abs().max()) == 1.0           # clamped, not rescaled: no seam
    wav = torch.zeros(16)
    wav[3] = float("nan")
    with pytest.raises(ValueError, match="NaN or inf"):
        model.process_stream_chunk(_append({"audio_inputs": [wav]}))
    with pytest.raises(ValueError, match="16000 Hz"):
        model.process_stream_chunk(_append({"audio_inputs": [short]},
                                           {"audio_inputs": [{"source_sample_rate": 48000}]}))
    with pytest.raises(ValueError, match="audio appends only"):
        model.process_stream_chunk(_append({"image_inputs": [torch.zeros(3, 4, 4)]}))
    with pytest.raises(ValueError, match="audio appends only; got text"):
        model.process_stream_chunk(_append({}, text="hello"))


def test_duplex_an_upload_or_append_is_one_stream_item():
    """A file streams whole, so a one-file upload is one encoder step, as the
    offline encoder ran it (the encoder runs a long item in a pass of its own)."""
    model = _make_model()
    long = torch.rand(20 * model.config.stt.sample_rate + 777) - 0.5
    for items in (
        model.process_prompt(
            None, ["audio"], ["audio", "text"], tensors={"audio_inputs": [long]},
        ).new_input_tensors["audio_in"],
        model.process_stream_chunk(_append({"audio_inputs": [long]}, {"audio_inputs": [{}]}))["audio_in"],
    ):
        assert len(items) == 1 and torch.equal(items[0], long)


def test_duplex_load_audio_refuses_a_file_longer_than_a_session_without_decoding_it_all(tmp_path, monkeypatch):
    """A compressed file can decode to far more than its bytes: a file longer
    than the session cap is refused after decoding one frame past the cap."""
    sf = pytest.importorskip("soundfile")
    model = _make_model()
    model.max_session_s = 1.0
    path = str(tmp_path / "two_seconds.flac")
    sf.write(path, torch.zeros(32000).numpy(), 16000)
    ok = str(tmp_path / "one_second.flac")
    sf.write(ok, torch.zeros(16000).numpy(), 16000)
    assert model.load_audio(ok, "cpu").data.numel() == 16000

    asked = []
    read = sf.SoundFile.read
    monkeypatch.setattr(sf.SoundFile, "read", lambda f, frames=-1, **kw: asked.append(frames) or read(f, frames, **kw))
    with pytest.raises(ValueError, match="longest session"):
        model.load_audio(path, "cpu")
    assert asked == [16001]


@pytest.mark.parametrize("channels,rate,error", [
    (3, 16000, "at most 2 channels"),
    (1, 48001, "unsupported sample rate 48001 Hz"),
])
def test_duplex_load_audio_refuses_wide_audio_before_decoding_it(tmp_path, monkeypatch, channels, rate, error):
    """A file of many channels, or at a rate whose resampling kernel is
    unbounded, is refused before anything is read."""
    sf = pytest.importorskip("soundfile")
    model = _make_model()
    path = str(tmp_path / "wide.flac")
    sf.write(path, torch.zeros(rate // 10, channels).numpy(), rate)
    read = sf.SoundFile.read
    asked = []
    monkeypatch.setattr(sf.SoundFile, "read", lambda f, *a, **kw: asked.append(a) or read(f, *a, **kw))
    with pytest.raises(ValueError, match=error):
        model.load_audio(path, "cpu")
    assert asked == []


def test_duplex_load_audio_refuses_a_file_past_the_session(tmp_path):
    """The session cap counts the frames decoded, at the file's own rate."""
    sf = pytest.importorskip("soundfile")
    model = _make_model()
    model.max_session_s = 1.0
    ok = str(tmp_path / "stereo48k.flac")
    sf.write(ok, torch.zeros(48000, 2).numpy(), 48000)
    assert model.load_audio(ok, "cpu").data.numel() == 16000
    long = str(tmp_path / "long.flac")
    sf.write(long, torch.zeros(19200).numpy(), 16000)
    with pytest.raises(ValueError, match="longest session"):
        model.load_audio(long, "cpu")


def test_duplex_load_audio_does_not_trust_the_header_length(tmp_path, monkeypatch):
    """A VBR MP3 without its Xing tag reports a length from its first frame's
    bitrate, several times its real one: a file within the cap still loads."""
    sf = pytest.importorskip("soundfile")
    model = _make_model()
    model.max_session_s = 2.0
    path = str(tmp_path / "one_second.flac")
    sf.write(path, torch.zeros(16000).numpy(), 16000)
    frames = sf.SoundFile.frames
    monkeypatch.setattr(sf.SoundFile, "frames", property(lambda f: frames.fget(f) * 4))
    assert model.load_audio(path, "cpu").data.numel() == 16000


def test_duplex_encoder_partition_ends_with_the_client_stream():
    model = _make_model()
    meta = model._meta(["audio"], ["audio", "text"], "encode", False)
    fpa = model.get_partition_forward_pass_args("Encoder", meta, {})
    assert fpa.request_done is False and fpa.inputs == []


def test_duplex_no_prompt_seeds_initial_decode_inputs():
    """Audio-only start (no system prompt): the LLM partition starts in the
    decode loop and its initial inputs carry the seeded prev_text / prev_func
    and the no-prompt text_inputs sentinel so iteration 0's readiness gate fires."""
    model = _make_model()
    sig = {"prev_text": ["pt"], "prev_func": ["pf"], "text_inputs": ["s"]}
    fpa = model.get_initial_forward_pass_args("LLM", ["audio"], ["audio"], sig)
    assert fpa.full_metadata.graph_walk == "decode"
    assert {e.name for e in fpa.inputs} == {"prev_text", "prev_func", "text_inputs"}


def _live_client(monkeypatch):
    """test/nemotron_duplex/live_ws_request.py, loaded by path with its own
    helpers: another suite's ``_env`` may be loaded already. Whatever was under
    the helpers' names is put back afterwards."""
    pytest.importorskip("msgpack")
    here = Path(__file__).parents[1] / "nemotron_duplex"
    monkeypatch.syspath_prepend(str(here))
    for helper in ("_env", "_audio_prep"):
        monkeypatch.setitem(sys.modules, helper, None)
        del sys.modules[helper]
    spec = importlib.util.spec_from_file_location("nemotron_live_ws_request", here / "live_ws_request.py")
    live = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, live)     # its dataclasses look it up
    spec.loader.exec_module(live)
    return live


def test_live_client_charges_each_frame_to_the_append_that_makes_it_final(monkeypatch):
    """Frame k is final once sample 1280 k + 255 has arrived (frame 0 once
    sample 256 has, for the featurizer's reflection): its lag runs from the
    append that carried that sample."""
    live = _live_client(monkeypatch)
    # 16 ms appends (256 samples), ten of them, sent at t = 1..10
    session = live.Session(request_id="x", sent_at=[float(t) for t in range(1, 11)],
                           tokens=[(20.0, "a"), (21.0, "b"), (22.0, "c")])
    # frame 0: sample 256 is in append 1; frame 1: sample 1535 in append 5;
    # frame 2: sample 2815 is past the last append, charged to it
    assert live.unit_lags(session, 16.0) == [20.0 - 2, 21.0 - 6, 22.0 - 10]
    assert live.unit_lags(live.Session(request_id="y", tokens=[(1.0, "a")]), 80.0) == []


def test_live_client_refuses_zero_sessions(monkeypatch):
    live = _live_client(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["live_ws_request.py", "--sessions", "0"])
    with pytest.raises(SystemExit):
        live.main()


def test_live_client_reports_where_a_call_departs_from_the_one_shot_text(monkeypatch):
    """--compare-one-shot gates on the whole-clip append only, which must match
    the upload save its last frame (either side may encode that frame in a pass
    of its own); for either call it names the first frame whose token differs."""
    live = _live_client(monkeypatch)
    ref = ["a", "b", "c", "d"]
    assert live._compare("whole_clip_append", list(ref), ref)["identical"]
    last = live._compare("whole_clip_append", ["a", "b", "c", "x"], ref)
    assert not last["identical"] and last["identical_but_last_frame"]
    row = live._compare("paced", ["a", "x", "c", "d"], ref)
    assert not row["identical"] and row["first_differing_frame"] == 1 and not row["identical_but_last_frame"]
    assert row["live_from_there"] == "xcd" and row["one_shot_from_there"] == "bcd"
    longer = live._compare("paced", ref + ["e"], ref)
    assert longer["first_differing_frame"] == 4 and not longer["identical_but_last_frame"]
    assert not live._compare("paced", ref[:3], ref)["identical_but_last_frame"]
