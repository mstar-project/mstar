"""LTX-2.5: graph, resources, request handling and the dit's step declaration, on CPU.

Dummy mode (``skip_weight_loading``): no weights, no GPU, no network. The config is
read from the checkpoint's own config files, copied under ``test/ltx2_5/fixtures``.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import AttentionStep, RaggedCrossAttentionSpec, RaggedCrossAttentionStep
from mstar.graph.base import GraphNode, Loop, Parallel, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.components.diffusion.attention import sdpa_attention
from mstar.model.ltx2_5.components.transformer import (
    LTX2Attends,
    LTX2DiT,
    audio_positions,
    build_rope,
    video_positions,
)
from mstar.model.ltx2_5.config import (
    DENOISE_LOOP,
    DISTILLED_SIGMA_VALUES,
    LTX25_REPO,
    SNAPSHOT_PATTERNS,
    LTX2TransformerConfig,
    LTX25Config,
)
from mstar.model.ltx2_5.ltx2_5_model import (
    ENCODE_TEXT_WALK,
    GENERATE_AUDIO_WALK,
    GENERATE_AV_WALK,
    GENERATE_VIDEO_WALK,
    LTX25Model,
)
from mstar.model.ltx2_5.submodules import (
    AUDIO_ATTN,
    AUDIO_XATTN,
    VIDEO_ATTN,
    VIDEO_XATTN,
    LTXDenoiseSubmodule,
    LTXShape,
    distilled_schedule,
    ltx_step,
    pack_audio,
    pack_video,
    shape_from_metadata,
    unpack_audio,
    unpack_video,
)
from mstar.model.submodule_base import NodeInputs

CONFIG_PATH = "configs/ltx2_5.yaml"


class _StubEncoding:
    def __init__(self, ids):
        self.ids = ids


class _StubTokenizer:
    """Bytes as ids; the real tokenizer is checked against the oracle on GPU."""

    def encode(self, text):
        return _StubEncoding(list(text.encode("utf-8")))


def _make_model(**kwargs) -> LTX25Model:
    model = LTX25Model(model_path_hf="test/ltx2_5", skip_weight_loading=True, **kwargs)
    model.set_config(LTX25Config())
    model.tokenizer = _StubTokenizer()
    return model


def _info(name: str, dims: list[int]) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=dims, dtype="torch.float32", nbytes=4, address=0, stride=[1] * len(dims),
        uuid=f"uuid-{name}", source_session_id="test:0", source_entity="test",
    )


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------


def test_default_config_is_ltx2_5():
    cfg = LTX25Config()
    t = cfg.transformer
    assert (t.num_layers, t.inner_dim, t.audio_inner_dim) == (48, 4096, 2048)
    assert t.use_prompt_adaln_single and t.rope_type == "split" and not t.ff_bias
    assert cfg.text.layer_types.count("full_attention") == 8 and cfg.text.global_head_dim == 512
    assert cfg.text.global_partial_rotary_factor == 0.25
    assert cfg.connectors.video_connector_num_layers == 8 and cfg.connectors.text_proj_in_factor == 49
    geo = cfg.geometry
    assert (geo.spatial_compression, geo.temporal_compression, geo.output_sample_rate) == (32, 8, 48000)
    # 121 frames at 24 fps: 16 latent frames, and LTX2Pipeline's round(5.04 s * 25/s) = 126 audio frames
    assert geo.latent_frames(121) == 16 and geo.audio_frames(121, 24.0) == 126


def test_config_rejects_an_unported_transformer():
    raw = dataclasses.asdict(LTX2TransformerConfig())
    assert LTX2TransformerConfig.from_dict(raw) == LTX2TransformerConfig()
    with pytest.raises(NotImplementedError, match="rope_type"):
        LTX2TransformerConfig.from_dict({**raw, "rope_type": "interleaved"})


def test_defaults_match_the_checkpoint_when_cached():
    """The dummy-mode tests run on the defaults; they must be what the checkpoint says."""
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(LTX25_REPO, allow_patterns=SNAPSHOT_PATTERNS, local_files_only=True))
        real = LTX25Config.from_snapshot(snapshot)
    except Exception:  # noqa: BLE001
        pytest.skip("LTX-2.5 checkpoint configs not in the local HF cache")
    assert real == LTX25Config()


# ----------------------------------------------------------------------
# graph and resources
# ----------------------------------------------------------------------


def test_graph_walks_and_nodes():
    model = _make_model()
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {ENCODE_TEXT_WALK, GENERATE_AV_WALK, GENERATE_VIDEO_WALK, GENERATE_AUDIO_WALK}
    assert model.nodes == ["audio_decoder", "dit", "text_encoder", "vae_decoder"]
    text = walks[ENCODE_TEXT_WALK]
    assert isinstance(text, GraphNode) and text.input_names == {"text_inputs"}
    assert {(e.name, e.persist, e.next_node) for e in text.outputs} == {
        ("text_video", True, EMPTY_DESTINATION), ("text_audio", True, EMPTY_DESTINATION),
    }


@pytest.mark.parametrize(
    "walk,decoders",
    [(GENERATE_AV_WALK, {"vae_decoder", "audio_decoder"}), (GENERATE_VIDEO_WALK, {"vae_decoder"}),
     (GENERATE_AUDIO_WALK, {"audio_decoder"})],
)
def test_generate_walk_structure(walk, decoders):
    model = _make_model()
    section = model.get_graph_walk_graphs()[walk]
    assert isinstance(section, Sequential)
    loop, tail = section.sections
    assert isinstance(loop, Loop) and loop.name == DENOISE_LOOP
    assert loop.max_iters == len(DISTILLED_SIGMA_VALUES)
    dit = loop.section
    assert dit.input_names == {"text_video", "text_audio", "latents", "audio_latents"}
    assert {(e.name, e.next_node) for e in dit.outputs} == {("latents", "dit"), ("audio_latents", "dit")}
    nodes = tail.sections if isinstance(tail, Parallel) else [tail]
    assert {n.name for n in nodes} == decoders
    for node in nodes:
        (emit,) = node.outputs
        assert emit.next_node == EMIT_TO_CLIENT
        assert emit.output_modality == ("video" if node.name == "vae_decoder" else "audio")


def test_resources_follow_the_attention_backend():
    specs = {s.resource_key: s for s in _make_model(attention_backend="flashinfer").get_node_resources()}
    assert set(specs) == {VIDEO_ATTN, AUDIO_ATTN, VIDEO_XATTN, AUDIO_XATTN}
    assert {k for k, s in specs.items() if isinstance(s, RaggedCrossAttentionSpec)} == {VIDEO_XATTN, AUDIO_XATTN}
    assert (specs[VIDEO_XATTN].config.head_dim, specs[AUDIO_XATTN].config.head_dim) == (128, 64)
    assert all(s.nodes == {"dit"} for s in specs.values())
    assert (specs[VIDEO_ATTN].config.num_qo_heads, specs[VIDEO_ATTN].config.head_dim) == (32, 128)
    assert (specs[AUDIO_ATTN].config.num_qo_heads, specs[AUDIO_ATTN].config.head_dim) == (32, 64)
    assert _make_model(attention_backend="sdpa").get_node_resources() == []
    with pytest.raises(ValueError):
        _make_model(attention_backend="flash")


def test_worker_graphs_from_yaml():
    walks = set()
    for wg in _make_model().get_worker_graphs(CONFIG_PATH):
        walks |= wg.graph_walks
    assert walks == {ENCODE_TEXT_WALK, GENERATE_AV_WALK, GENERATE_VIDEO_WALK, GENERATE_AUDIO_WALK}


def test_dummy_mode_returns_no_submodules():
    model = _make_model()
    assert all(model.get_submodule(node) is None for node in model.nodes)


def test_autocast_is_off_so_numerics_follow_the_checkpoint():
    assert _make_model().get_autocast_dtype() is None


# ----------------------------------------------------------------------
# requests
# ----------------------------------------------------------------------


def test_process_prompt_tokenizes_the_stripped_prompt():
    out = _make_model().process_prompt("  a dog ", ["text"], ["video", "audio"])
    assert out["text_inputs"][0].tolist() == list(b"a dog")


@pytest.mark.parametrize(
    "kwargs,message",
    [({"height": 500}, "multiple of 32"), ({"width": 0}, "positive"), ({"num_frames": 120}, "8k\\+1"),
     ({"fps": 0}, "fps must be positive"), ({"height": 2048, "width": 4096, "num_frames": 257}, "max_video_tokens")],
)
def test_process_prompt_rejects_bad_geometry(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _make_model().process_prompt("a dog", ["text"], ["video", "audio"], **kwargs)


def test_process_prompt_rejects_missing_prompt_or_modality():
    model = _make_model()
    with pytest.raises(ValueError, match="non-empty"):
        model.process_prompt("  ", ["text"], ["video"])
    with pytest.raises(ValueError, match="video and/or audio"):
        model.process_prompt("a dog", ["text"], ["image"])
    with pytest.raises(ValueError, match="text prompt only"):
        model.process_prompt("a dog", ["text", "image"], ["video"])


@pytest.mark.parametrize(
    "outputs,walk",
    [(["video", "audio"], GENERATE_AV_WALK), (["video"], GENERATE_VIDEO_WALK), (["audio"], GENERATE_AUDIO_WALK)],
)
def test_schedule_encode_text_then_generate_then_done(outputs, walk):
    model = _make_model()
    args = model.get_initial_forward_pass_args(
        "default", ["text"], outputs, {"text_inputs": [_info("t", [36])]},
        model_kwargs={"height": 544, "width": 960, "num_frames": 121, "fps": 24},
    )
    md = args.full_metadata
    assert md.graph_walk == ENCODE_TEXT_WALK and md.kwargs["walk_schedule"] == [ENCODE_TEXT_WALK, walk]
    assert args.step_metadata == {"is_prefill": True, "height": 544, "width": 960, "num_frames": 121, "fps": 24.0}
    persist = {"text_video": [_info("v", [1024, 4096])], "text_audio": [_info("a", [1024, 2048])]}
    nxt = model.get_partition_forward_pass_args("default", md, persist)
    assert nxt.full_metadata.graph_walk == walk and not nxt.request_done
    by_name = {e.name: e for e in nxt.inputs}
    assert set(by_name) == {"text_video", "text_audio", "latents", "audio_latents"}
    assert by_name["latents"].tensor_info == [] and by_name["audio_latents"].tensor_info == []
    done = model.get_partition_forward_pass_args("default", nxt.full_metadata, {})
    assert done.request_done and done.inputs == []


def test_postprocess_audio_is_interleaved_pcm16():
    model = _make_model()
    wave = torch.stack([torch.full((5,), 0.5), torch.full((5,), -1.5)])
    pcm = torch.frombuffer(bytearray(model.postprocess(wave, "audio")), dtype=torch.int16).view(5, 2)
    assert pcm[:, 0].tolist() == [16384] * 5 and pcm[:, 1].tolist() == [-32767] * 5
    assert model.get_output_sample_rate() == 48000 and model.get_output_audio_channels() == 2


def test_postprocess_video_is_an_mp4():
    frames = torch.randint(0, 255, (3, 9, 64, 64), dtype=torch.uint8)
    data = _make_model().postprocess(frames, "video", {"fps": 24})
    assert data[4:8] == b"ftyp"


# ----------------------------------------------------------------------
# the dit node
# ----------------------------------------------------------------------


def _dit(config: LTX25Config, use_ragged: bool = True) -> LTXDenoiseSubmodule:
    return LTXDenoiseSubmodule(nn_stub(), config, loop_name=DENOISE_LOOP, use_ragged_attention=use_ragged)


def nn_stub():
    stub = torch.nn.Linear(1, 1)
    stub.dtype = torch.bfloat16
    return stub


def test_shape_from_step_metadata():
    cfg = LTX25Config()
    shape = shape_from_metadata(cfg, {"height": 544, "width": 960, "num_frames": 121, "fps": 24.0})
    assert shape == LTXShape(frames=16, height=17, width=30, audio_frames=126, fps=24.0, text_len=1024)
    assert shape.video_tokens == 8160 and shape.total_tokens == 8160 + 126 + 1024


def test_seed_draws_video_then_audio_noise_from_one_generator():
    """LTX2Pipeline draws the video noise first, then the audio noise, both fp32 in
    their unpacked layouts; the same seed must reproduce both."""
    cfg = LTX25Config()
    shape = LTXShape(frames=2, height=3, width=4, audio_frames=5, fps=24.0, text_len=1024)
    seeds = _dit(cfg).seed_loop_back(None, shape, torch.Generator().manual_seed(7))
    gen = torch.Generator().manual_seed(7)
    video = torch.randn((1, 128, 2, 3, 4), generator=gen)
    audio = torch.randn((1, 8, 5, 16), generator=gen)
    assert torch.equal(seeds["latents"], pack_video(video)[0])
    assert torch.equal(seeds["audio_latents"], pack_audio(audio)[0])
    assert seeds["latents"].shape == (24, 128) and seeds["audio_latents"].shape == (5, 128)


def test_pack_unpack_round_trip():
    shape = LTXShape(frames=2, height=3, width=4, audio_frames=5, fps=24.0, text_len=8)
    video = torch.randn(1, 128, 2, 3, 4)
    assert torch.equal(unpack_video(pack_video(video), shape), video)
    audio = torch.randn(1, 8, 5, 16)
    assert torch.equal(unpack_audio(pack_audio(audio), 16), audio)


def test_declare_step_spans_and_pairs():
    cfg = LTX25Config()
    shape = LTXShape(frames=2, height=3, width=4, audio_frames=5, fps=24.0, text_len=16)
    inputs = [NodeInputs(resource_step_info=shape) for _ in range(2)]
    step = _dit(cfg).declare_step("generate_av", [11, 12], inputs)
    video, audio = step.steps[VIDEO_ATTN], step.steps[AUDIO_ATTN]
    assert isinstance(video, AttentionStep) and not video.causal and not audio.causal
    assert {s.label for s in video.segments} == {"video"} and {s.label for s in audio.segments} == {"audio"}
    video_x, audio_x = step.steps[VIDEO_XATTN], step.steps[AUDIO_XATTN]
    assert isinstance(video_x, RaggedCrossAttentionStep) and video_x.pairs == (("video", "text"),)
    assert set(audio_x.pairs) == {("audio", "text"), ("video", "audio"), ("audio", "video")}
    spans = {(s.request_id, s.label): s.span for s in audio_x.segments}
    assert spans == {(r, label): n for r in (11, 12) for label, n in (("video", 24), ("audio", 5), ("text", 16))}
    assert step.cg_key_info == shape
    assert _dit(cfg, use_ragged=False).declare_step("generate_av", [11], inputs[:1]) is None


def test_distilled_schedule_matches_the_scheduler_recipe():
    sched = distilled_schedule()
    assert sched.num_steps == 8
    assert sched.sigmas[-1] == 0 and sched.sigmas[:-1].tolist() == pytest.approx(list(DISTILLED_SIGMA_VALUES))
    assert torch.equal(sched.timesteps, sched.sigmas[:-1] * 1000)


def test_step_is_the_reference_round_trip_then_euler():
    x = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    sigma, sigma_next = torch.tensor(0.725), torch.tensor(0.421875)
    out = ltx_step(x, v, sigma, sigma_next)
    vf = v.float()
    expected = x + (sigma_next - sigma) * ((x - (x - vf * sigma)) / sigma)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-6)


# ----------------------------------------------------------------------
# the DiT, at toy size
# ----------------------------------------------------------------------


def _toy_config() -> LTX25Config:
    cfg = LTX25Config()
    toy = dataclasses.replace(
        cfg.transformer, num_layers=2, num_attention_heads=2, attention_head_dim=16,
        audio_num_attention_heads=2, audio_attention_head_dim=8, cross_attention_dim=32,
        audio_cross_attention_dim=16, in_channels=8, out_channels=8, audio_in_channels=8, audio_out_channels=8,
    )
    return dataclasses.replace(cfg, transformer=toy)


def test_rope_positions_match_the_reference_layout():
    t = LTX25Config().transformer
    pos = video_positions(t, 2, 1, 2, 24.0, "cpu")[0]
    # first latent frame is causally shifted to start at 0 s; frame 1 covers pixels 1..8 -> 9..16
    assert pos[:, 0].tolist() == pytest.approx([0.5 / 24, 0.5 / 24, 5 / 24, 5 / 24])
    assert pos[:, 2].tolist() == pytest.approx([16.0, 48.0, 16.0, 48.0])
    assert audio_positions(t, 2, "cpu")[0, :, 0].tolist() == pytest.approx([0.005, 0.03])


@torch.inference_mode()
def test_toy_dit_forward_shapes_and_row_independence():
    torch.manual_seed(0)
    cfg = _toy_config()
    dit = LTX2DiT(cfg.transformer)
    for p in dit.parameters():
        torch.nn.init.normal_(p, std=0.05)
    rope = build_rope(cfg.transformer, 2, 2, 3, 24.0, 5, "cpu")
    attends = LTX2Attends(*([sdpa_attention] * 6))

    def run(rows):
        g = torch.Generator().manual_seed(1)
        x = torch.randn(2, 12, 8, generator=g)[rows]
        a = torch.randn(2, 5, 8, generator=g)[rows]
        t = torch.randn(2, 7, 32, generator=g)[rows]
        at = torch.randn(2, 7, 16, generator=g)[rows]
        return dit(x, a, t, at, torch.tensor([1000.0, 500.0])[rows], rope, attends)

    video, audio = run(slice(0, 2))
    assert video.shape == (2, 12, 8) and audio.shape == (2, 5, 8)
    # a row of a batch equals the same request run alone
    v1, a1 = run(slice(1, 2))
    torch.testing.assert_close(video[1:], v1, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(audio[1:], a1, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()
