"""Contract tests for the LingBot-World-V2 serving shell. CPU-only, no
checkpoint: the schedule tables, the rolling K/V eviction, the walk wiring,
the stop/veto, request and config validation, and that the per-chunk
condition and Plucker rays equal the whole-rollout ones. Numerics against the
reference are the oracle-gated parity harness's job
(``test/lingbot_world/run_parity.py``).
"""

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.engine.resources import RaggedCrossAttentionSpec, RaggedCrossAttentionStep, cross_label  # noqa: E402
from mstar.graph.base import GraphNode, Loop  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT  # noqa: E402
from mstar.model.lingbot_world.components.dit import (  # noqa: E402
    REFERENCE_ATTENTION,
    RollingKVCache,
    attended_frames,
)
from mstar.model.lingbot_world.components.schedule import make_fast_schedule  # noqa: E402
from mstar.model.lingbot_world.lingbot_world_model import LingBotWorldModel  # noqa: E402
from mstar.model.lingbot_world.submodules import (  # noqa: E402
    DIT_ATTN,
    ROLLOUT_LOOP_NAME,
    ROLLOUT_WALK,
    ChunkSpans,
    EncoderStream,
    LingBotDitSubmodule,
    encode_condition_chunk,
    first_frame_mask,
)
from mstar.model.submodule_base import NodeInputs  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return LingBotWorldModel(skip_weight_loading=True)


def test_schedule_picks_the_reference_timesteps():
    schedule = make_fast_schedule(5.0, (0, 250, 500, 750))
    assert [int(t) for t in schedule.step_timesteps] == [999, 937, 833, 624]
    assert schedule.sigmas.dtype == torch.float32 and schedule.sigmas[-1] == 0
    assert schedule.timesteps.dtype == torch.int64 and len(schedule.timesteps) == 1000


def test_colliding_timesteps_resolve_first_for_x0_and_second_for_noise():
    schedule = make_fast_schedule(5.0, (0, 250, 500, 750))
    t = schedule.step_timesteps[1]
    matches = (schedule.timesteps == t).nonzero().flatten().tolist()
    assert len(matches) > 1, "the test needs a timestep that repeats"
    assert schedule.x0_sigma(t).item() == schedule.sigmas[matches[0]].double().item()
    assert schedule.noise_sigma(t).item() == schedule.sigmas[matches[1]].item()


def _simulate(cache: RollingKVCache, frame_seqlen: int, chunk: int, num_chunks: int, passes: int):
    """Write each chunk ``passes`` times (denoise steps, then the commit), tagging
    every token with its absolute latent frame, as the self-attention does."""
    for c in range(num_chunks):
        start = c * chunk * frame_seqlen
        tags = torch.arange(c * chunk, (c + 1) * chunk).repeat_interleave(frame_seqlen).float()
        for _ in range(passes):
            plan = cache.plan(start, chunk * frame_seqlen)
            cache.write(0, tags.view(1, -1, 1, 1), tags.view(1, -1, 1, 1), plan)
            cache.global_end, cache.local_end = plan.current_end, plan.local_end
        yield c, cache.k[0][0, :cache.local_end, 0, 0]


def test_rolling_cache_keeps_the_sink_and_the_latest_window():
    frame_seqlen, chunk, window, sink = 3, 4, 18, 6
    cache = RollingKVCache.allocate(1, window * frame_seqlen, 1, 1, sink_tokens=sink * frame_seqlen,
                                    windowed=True, device="cpu", dtype=torch.float32)
    for c, held in _simulate(cache, frame_seqlen, chunk, num_chunks=8, passes=5):
        frames = held.view(-1, frame_seqlen)[:, 0].long().tolist()
        end = (c + 1) * chunk
        if end <= window:
            assert frames == list(range(end))
        else:
            assert frames == list(range(sink)) + list(range(end - (window - sink), end))


def test_unwindowed_cache_appends_without_eviction():
    cache = RollingKVCache.allocate(1, 24 * 2, 1, 1, sink_tokens=0, windowed=False, device="cpu", dtype=torch.float32)
    *_, (_, held) = _simulate(cache, 2, 4, num_chunks=6, passes=5)
    assert held.view(-1, 2)[:, 0].long().tolist() == list(range(24))


def test_first_frame_mask_marks_only_the_first_video_frame():
    msk = first_frame_mask(3, 2, 2, "cpu")
    assert msk.shape == (4, 3, 2, 2)
    assert bool((msk[:, 0] == 1).all()) and bool((msk[:, 1:] == 0).all())


def test_walks_run_text_then_a_chunk_loop_that_emits_frames(model):
    walks = model.get_graph_walk_graphs()
    assert list(walks) == ["encode_text", ROLLOUT_WALK]
    loop = walks[ROLLOUT_WALK]
    assert isinstance(loop, Loop) and loop.name == ROLLOUT_LOOP_NAME
    node = loop.section
    assert isinstance(node, GraphNode) and node.name == "dit"
    assert "image_inputs" in node.input_names
    emits = [e for e in node.outputs if e.next_node == EMIT_TO_CLIENT]
    assert [(e.name, e.output_modality) for e in emits] == [("video_output", "video_frame")]
    assert any(e.next_node == "dit" and e.name == "clock" for e in node.outputs)


def _info(num_chunks: int, done: int) -> CurrentForwardPassInfo:
    info = CurrentForwardPassInfo(
        request_id="r", graph_walk=ROLLOUT_WALK, fwd_index=0, random_seed=0, max_tokens=0,
        step_metadata={"num_chunks": num_chunks}, rid_handle=0,
    )
    info.dynamic_loop_iter_counts[ROLLOUT_LOOP_NAME] = done
    return info


def test_check_stop_fires_on_the_last_chunk_and_the_overshoot_is_vetoed(model):
    sub = LingBotDitSubmodule.__new__(LingBotDitSubmodule)
    torch.nn.Module.__init__(sub)
    sub.request_states = {}
    assert [bool(sub.check_stop("r", _info(3, k), {})) for k in range(3)] == [False, False, True]
    sub.request_state(0).add("chunk", 3)
    assert sub.prepare_inputs(ROLLOUT_WALK, _info(3, 3), {}) is None


# The reference's sqrt gives 479.99999999999994 for an exact 480x832 image and
# floors it to 58 latent rows, not 60; matching it is what keeps parity.
@pytest.mark.parametrize("image_hw,expected", [
    ((2894, 5145), (58, 104)), ((480, 832), (58, 104)), ((832, 480), (104, 60)),
])
def test_geometry_follows_the_image_aspect_at_the_requested_area(model, image_hw, expected):
    assert model.geometry(image_hw, 480, 832) == expected


@pytest.mark.parametrize("num_frames,num_poses,expected", [(93, 269, 24), (81, 269, 20), (361, 269, 68), (17, 17, 4)])
def test_rollout_length_is_clamped_to_the_trajectory_and_floored_to_chunks(model, num_frames, num_poses, expected):
    assert model.num_latent_frames(num_frames, num_poses) == expected


def test_process_prompt_rejects_requests_it_cannot_serve(model):
    image = torch.zeros(3, 480, 832, dtype=torch.uint8)
    poses = np.tile(np.eye(4), (33, 1, 1))
    ok = dict(camera_poses=poses, intrinsics=[415.0, 415.0, 416.0, 240.0], num_frames=33)
    with pytest.raises(ValueError, match="video_frame"):
        model.process_prompt("p", ["text", "image"], ["video"], tensors={"image_inputs": [image]}, **ok)
    with pytest.raises(ValueError, match="prompt"):
        model.process_prompt("", ["text", "image"], ["video_frame"], tensors={"image_inputs": [image]}, **ok)
    with pytest.raises(ValueError, match="image"):
        model.process_prompt("p", ["text"], ["video_frame"], tensors={}, **ok)
    with pytest.raises(ValueError, match="camera_poses"):
        model.process_prompt("p", ["text", "image"], ["video_frame"], tensors={"image_inputs": [image]},
                             **{**ok, "camera_poses": poses[:5]})
    with pytest.raises(ValueError, match="intrinsics"):
        model.process_prompt("p", ["text", "image"], ["video_frame"], tensors={"image_inputs": [image]},
                             **{**ok, "intrinsics": [1.0, 2.0]})


@pytest.mark.parametrize("frame", [0, 32])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_process_prompt_refuses_a_non_finite_pose_or_intrinsic(model, frame, bad):
    """Frame 32 is past the frames a 33-pose, 17-frame request samples."""
    image = torch.zeros(3, 480, 832, dtype=torch.uint8)
    poses = np.tile(np.eye(4), (33, 1, 1))
    poses[frame, 0, 3] = bad
    kwargs = dict(camera_poses=poses, intrinsics=[415.0, 415.0, 416.0, 240.0], num_frames=17)
    with pytest.raises(ValueError, match="finite"):
        model.process_prompt("p", ["text", "image"], ["video_frame"], tensors={"image_inputs": [image]}, **kwargs)
    kwargs.update(camera_poses=np.tile(np.eye(4), (33, 1, 1)), intrinsics=[415.0, bad, 416.0, 240.0])
    with pytest.raises(ValueError, match="finite"):
        model.process_prompt("p", ["text", "image"], ["video_frame"], tensors={"image_inputs": [image]}, **kwargs)


@pytest.mark.parametrize("size", [
    {"height": 100_000, "width": 100_000}, {"height": 1281, "width": 720}, {"height": -480}, {"width": 0},
])
def test_the_output_area_is_bounded_before_the_resize(model, size, monkeypatch):
    """``height`` / ``width`` are client knobs: past ``max_area`` the bicubic
    resize alone would allocate the requested area on the data worker."""
    def no_resize(*args, **kwargs):
        raise AssertionError("resized an out-of-range request")

    monkeypatch.setattr(torch.nn.functional, "interpolate", no_resize)
    with pytest.raises(ValueError, match="height and width"):
        model._preprocess_image(torch.zeros(3, 48, 83, dtype=torch.uint8), size)


def test_the_largest_area_and_an_extreme_aspect(model):
    image = torch.zeros(3, 72, 128, dtype=torch.uint8)
    assert model._preprocess_image(image, {"height": 720, "width": 1280}).shape == (3, 720, 1280)
    with pytest.raises(ValueError, match="cannot fit"):
        model._preprocess_image(torch.zeros(3, 1, 4000, dtype=torch.uint8), {})


def _png16_rgb(path, rgb16: np.ndarray):
    """A 16-bit-per-channel RGB PNG (Pillow cannot write one)."""
    import struct
    import zlib

    h, w, _ = rgb16.shape
    rows = b"".join(b"\x00" + rgb16[y].astype(">u2").tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def test_a_16_bit_png_loads_as_the_8_bit_image_the_reference_reads(model, tmp_path):
    """torchvision decodes a 16-bit PNG to uint16, which _preprocess_image would
    take as [0, 65535] instead of [0, 255]; PIL (the reference's reader) keeps
    the high byte."""
    from PIL import Image

    rgb16 = np.random.default_rng(0).integers(0, 65536, size=(24, 40, 3), dtype=np.uint16)
    path = tmp_path / "rgb16.png"
    _png16_rgb(path, rgb16)
    image = model.load_image(str(path), "cpu").data
    assert image.dtype == torch.uint8 and image.shape == (3, 24, 40)
    expected = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).permute(2, 0, 1)
    assert torch.equal(image, expected)


def test_prompt_clean_is_the_reference_tokenizers_whitespace_clean(model):
    # ftfy straightens curly quotes; the reference T5 tokenizer cleans every prompt this way.
    assert model.prompt_clean("  A \u201ctree\u201d &amp;amp; its\u00a0 reflection\n") == 'A "tree" & its reflection'


def test_one_chunk_needs_thirteen_poses(model, monkeypatch):
    image = torch.zeros(3, 480, 832, dtype=torch.uint8)
    seen = {}

    class Tok:
        def __call__(self, text, **kwargs):
            seen["text"] = text
            return type("Ids", (), {"input_ids": [0, 1]})()

    monkeypatch.setattr(model, "_ensure_tokenizer", lambda: None)
    monkeypatch.setattr(model, "tokenizer", Tok())
    req = dict(intrinsics=[415.0, 415.0, 416.0, 240.0], num_frames=13)
    with pytest.raises(ValueError, match="F >= 13"):
        model.process_prompt("p", ["text", "image"], ["video_frame"], tensors={"image_inputs": [image]},
                             camera_poses=np.tile(np.eye(4), (12, 1, 1)), **req)
    out = model.process_prompt("p", ["text", "image"], ["video_frame"], tensors={"image_inputs": [image]},
                               camera_poses=np.tile(np.eye(4), (13, 1, 1)), **req)
    assert seen["text"] == "p"
    assert out["camera_poses"][0].shape[0] == 4  # one chunk of latent frames


def test_mstar_serve_knows_the_model():
    from pathlib import Path

    from mstar.cli.main import DEFAULT_CONFIGS, _resolve_config

    assert DEFAULT_CONFIGS["lingbot_world_1_3b"] == "lingbot_world_1_3b.yaml"
    assert Path(_resolve_config("lingbot_world_1_3b", None)).name == "lingbot_world_1_3b.yaml"


def test_dit_loader_takes_each_tensor_only_from_its_indexed_shard(tmp_path):
    """A shard that lacks an indexed tensor must fail the load, not leave it as
    ``to_empty``'s uninitialized memory; a stale copy in another shard must not
    overwrite or stand in for it."""
    import json

    from safetensors.torch import save_file

    from mstar.model.lingbot_world.components.dit import LingBotCausalDiT
    from mstar.model.lingbot_world.config import LingBotWorldConfig
    from mstar.model.lingbot_world.weight_loader import build_lingbot_dit

    cfg = LingBotWorldConfig(dim=12, ffn_dim=24, num_heads=1, num_layers=1, freq_dim=8, text_dim=8)
    tensors = {name: torch.ones_like(p) for name, p in LingBotCausalDiT(cfg).named_parameters()}
    weight_map = {name: "a.safetensors" for name in tensors}
    weight_map["patch_embedding.bias"] = "b.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    def write(a, b):
        save_file(a, str(tmp_path / "a.safetensors"))
        save_file(b, str(tmp_path / "b.safetensors"))

    weight, bias = tensors.pop("patch_embedding.weight"), tensors.pop("patch_embedding.bias")
    write({**tensors, "patch_embedding.weight": weight}, {"patch_embedding.bias": bias})
    assert build_lingbot_dit(cfg, str(tmp_path), device="cpu").patch_embedding.weight.eq(1).all()

    write(tensors, {"patch_embedding.bias": bias})
    with pytest.raises(ValueError, match="missing.*patch_embedding.weight"):
        build_lingbot_dit(cfg, str(tmp_path), device="cpu")
    for stale in (weight, weight * 9):
        write({**tensors, "patch_embedding.weight": weight},
              {"patch_embedding.bias": bias, "patch_embedding.weight": stale})
        with pytest.raises(ValueError, match="index puts in a.safetensors"):
            build_lingbot_dit(cfg, str(tmp_path), device="cpu")
    write(tensors, {"patch_embedding.bias": bias, "patch_embedding.weight": weight})
    with pytest.raises(ValueError, match="index puts in a.safetensors"):
        build_lingbot_dit(cfg, str(tmp_path), device="cpu")


def test_the_parity_gate_fails_on_any_differing_point():
    """Every flag in the parity report must hold, and the rollout must stop on
    its last chunk only; anything else fails the run."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "lingbot_world" / "run_parity.py"
    spec = importlib.util.spec_from_file_location("lingbot_run_parity", path)
    run_parity = importlib.util.module_from_spec(spec)
    precision = torch.get_float32_matmul_precision()
    try:
        spec.loader.exec_module(run_parity)
    finally:
        torch.set_float32_matmul_precision(precision)

    report = {
        "tokenizer_ids_equal": True, "text_embeds_tail_zero": True, "mask_equal": True,
        "image": {"bit_exact": True, "max_abs": 0.0}, "dit_calls_bit_exact": {"x_in": True, "flow": True},
        "e2e_chunks": [{"latent": {"bit_exact": True}, "frames": {"bit_exact": True}}] * 2,
        "check_stop_flags": [False, False, True], "overshoot_vetoed": True, "dit_first_divergent_call": None,
    }
    assert run_parity.mismatches(report) == []
    for key, value, point in [
        ("text_embeds_tail_zero", False, ".text_embeds_tail_zero"),
        ("dit_calls_bit_exact", {"x_in": True, "flow": False}, ".dit_calls_bit_exact.flow"),
        ("e2e_chunks", [{"latent": {"bit_exact": True}, "frames": {"bit_exact": False}}],
         ".e2e_chunks[0].frames.bit_exact"),
        ("image", {"bit_exact": False, "shape_mismatch": [[1], [2]]}, ".image.bit_exact"),
        ("check_stop_flags", [False, True, True], ".check_stop_flags"),
        ("check_stop_flags", [False, False, False], ".check_stop_flags"),
    ]:
        assert run_parity.mismatches({**report, key: value}) == [point]


def test_poses_may_drop_the_constant_bottom_row(model, monkeypatch):
    """``[F, 3, 4]`` poses fit 4,077 frames in the 1 MiB ``model_kwargs`` form
    part; they must give the same request as the ``[F, 4, 4]`` ones."""
    class Tok:
        def __call__(self, text, **kwargs):
            return type("Ids", (), {"input_ids": [0, 1]})()

    monkeypatch.setattr(model, "_ensure_tokenizer", lambda: None)
    monkeypatch.setattr(model, "tokenizer", Tok())
    rng = np.random.default_rng(0)
    poses = np.tile(np.eye(4), (33, 1, 1))
    poses[:, :3, 3] = rng.normal(size=(33, 3)).cumsum(0)
    image = torch.zeros(3, 480, 832, dtype=torch.uint8)
    req = dict(intrinsics=[415.0, 415.0, 416.0, 240.0], num_frames=33)
    full = model.process_prompt("p", ["text", "image"], ["video_frame"], tensors={"image_inputs": [image]},
                                camera_poses=poses.tolist(), **req)
    top = model.process_prompt("p", ["text", "image"], ["video_frame"], tensors={"image_inputs": [image]},
                               camera_poses=poses[:, :3].tolist(), **req)
    assert torch.equal(full["camera_poses"][0], top["camera_poses"][0])
    with pytest.raises(ValueError, match=r"\[F, 3, 4\]"):
        model.process_prompt("p", ["text", "image"], ["video_frame"], tensors={"image_inputs": [image]},
                             camera_poses=poses[:, :2].tolist(), **req)


def test_per_chunk_plucker_rays_are_the_whole_trajectorys():
    from mstar.model.lingbot_world.components.camera import plucker_embedding

    torch.manual_seed(0)
    c2ws = torch.eye(4).repeat(8, 1, 1)
    c2ws[:, :3, 3] = torch.randn(8, 3)
    c2ws[:, :3, :3] = torch.linalg.qr(torch.randn(8, 3, 3)).Q
    intr = torch.tensor([20.0, 21.0, 16.0, 8.0])
    whole = plucker_embedding(c2ws, intr, 16, 32, 2, 4)
    chunks = [plucker_embedding(c2ws[k:k + 4], intr, 16, 32, 2, 4) for k in (0, 4)]
    assert torch.equal(torch.cat(chunks, dim=2), whole)


@pytest.fixture
def one_thread():
    # oneDNN's conv results can depend on the intra-op thread count.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_the_condition_chunk_by_chunk_is_the_whole_clip_encode(one_thread):
    """The streamed encode the dit runs per chunk against the reference's one
    ``vae.encode`` of the image followed by black frames, on a tiny Wan VAE,
    long enough that the black frames drive the cache to its fixed point."""
    from diffusers import AutoencoderKLWan

    from mstar.model.lingbot_world.config import LingBotWorldConfig

    torch.manual_seed(0)
    z = 4
    vae = AutoencoderKLWan(base_dim=8, z_dim=z, num_res_blocks=1, latents_mean=[0.1] * z,
                           latents_std=[2.0] * z).eval()
    cfg = LingBotWorldConfig(vae_z_dim=z)
    image = torch.rand(3, 16, 24) * 2 - 1
    lat_f = 16 * cfg.chunk_size
    num_frames = (lat_f - 1) * cfg.vae_stride[0] + 1
    with torch.inference_mode():
        clip = torch.concat([image[:, None], torch.zeros(3, num_frames - 1, 16, 24)], dim=1)
        latent = vae.encode(clip[None]).latent_dist.mode().float()
        latent = ((latent - 0.1) / 2.0)[0]
    whole = torch.concat([first_frame_mask(lat_f, 2, 3, "cpu"), latent])

    stream = EncoderStream(vae)
    chunks = [encode_condition_chunk(vae, image, stream, k * cfg.chunk_size, cfg.chunk_size, cfg)
              for k in range(lat_f // cfg.chunk_size)]
    assert stream.steady is not None and stream.cache is None
    torch.testing.assert_close(torch.cat(chunks, dim=1), whole, rtol=0, atol=0)


def test_a_cache_slot_written_in_place_never_counts_as_unchanged():
    from mstar.model.lingbot_world.submodules import _same_slot

    t = torch.zeros(2)
    assert not _same_slot(t, t)
    assert _same_slot(t, t.clone()) and not _same_slot(t, t + 1)
    assert _same_slot(None, None) and not _same_slot(None, t)


def test_the_text_encoder_runs_eager_attention_as_the_reference_t5(monkeypatch):
    """transformers 5 defaults UMT5 to SDPA, whose text embeds differ from the
    reference's eager T5."""
    import transformers

    from mstar.model.wan22 import submodules as wan22_submodules

    seen = {}

    class Encoder(torch.nn.Module):
        def eval(self):
            return self

    def from_pretrained(*args, **kwargs):
        seen.update(kwargs)
        return Encoder()

    monkeypatch.setattr(transformers.UMT5EncoderModel, "from_pretrained", from_pretrained)
    monkeypatch.setattr(wan22_submodules, "Wan22TextEncoderSubmodule", lambda encoder, cfg: encoder)
    LingBotWorldModel().get_submodule("text_encoder")
    assert seen["attn_implementation"] == "eager"


def test_the_reference_attention_dit_fails_to_build_without_flash_attn(monkeypatch):
    """Not the first request: with flash-attn missing the server must not
    come up healthy and then fail every request."""
    from mstar.model.lingbot_world import weight_loader

    def no_weights(*args, **kwargs):
        raise AssertionError("loaded weights before checking flash-attn")

    monkeypatch.setitem(sys.modules, "flash_attn", None)
    monkeypatch.setattr(weight_loader, "build_lingbot_dit", no_weights)
    with pytest.raises(ImportError, match="install flash-attn"):
        LingBotWorldModel(reference_attention=True).get_submodule("dit")


def test_the_engine_attention_dit_builds_without_flash_attn(monkeypatch):
    from mstar.model.lingbot_world import weight_loader

    class BuiltError(Exception):
        pass

    def no_weights(*args, **kwargs):
        raise BuiltError

    monkeypatch.setitem(sys.modules, "flash_attn", None)
    monkeypatch.setattr(weight_loader, "build_lingbot_dit", no_weights)
    with pytest.raises(BuiltError):
        LingBotWorldModel().get_submodule("dit")


def test_the_dit_declares_one_ragged_cross_attention(model):
    (spec,) = model.get_node_resources()
    assert isinstance(spec, RaggedCrossAttentionSpec)
    assert spec.resource_key == DIT_ATTN and spec.nodes == {"dit"}
    assert (spec.config.num_qo_heads, spec.config.num_kv_heads, spec.config.head_dim) == (12, 12, 128)
    assert spec.config.dtype == torch.bfloat16
    assert LingBotWorldModel(skip_weight_loading=True, reference_attention=True).get_node_resources() == []


@pytest.mark.parametrize("value", ["false", 0, None])
def test_reference_attention_must_be_a_bool(value):
    """A yaml "false" is truthy; read as on it would serve the other attention."""
    with pytest.raises(ValueError, match="reference_attention"):
        LingBotWorldModel(skip_weight_loading=True, reference_attention=value)


def _bare_dit_submodule(model, resources):
    sub = LingBotDitSubmodule.__new__(LingBotDitSubmodule)
    torch.nn.Module.__init__(sub)
    sub.request_states = {}
    sub.config = model.config
    sub._attention = None
    sub.node_resources = resources
    return sub


def test_a_chunks_step_pairs_its_queries_with_its_window_and_the_text(model):
    sub = _bare_dit_submodule(model, {DIT_ATTN: object()})
    frame = 58 * 104 // 4  # 464x832
    windows = [sub.chunk_spans(c, (464, 832)).window // frame for c in range(7)]
    assert windows == [4, 8, 12, 16, 18, 18, 18]  # sink included, capped at local_attn_size
    inputs = [NodeInputs(resource_step_info=sub.chunk_spans(5, (464, 832)))]
    step = sub.declare_step(ROLLOUT_WALK, ["r"], inputs).steps[DIT_ATTN]
    assert isinstance(step, RaggedCrossAttentionStep)
    assert [(s.label, s.span) for s in step.segments] == [("chunk", 4 * frame), ("window", 18 * frame), ("text", 512)]
    assert step.pairs == (("chunk", "window"), ("chunk", "text"))


def test_without_the_resource_the_dit_runs_the_reference_attention(model):
    sub = _bare_dit_submodule(model, {})
    assert sub.attention() is REFERENCE_ATTENTION
    assert sub.declare_step(ROLLOUT_WALK, ["r"], [NodeInputs(resource_step_info=ChunkSpans(1, 1, 1))]) is None


def test_the_engine_attention_binds_both_pairs(model):
    calls = []

    class Resource:
        def run(self, q, k, v, label):
            calls.append((label, q.shape[0], k.shape[0]))
            return q

    sub = _bare_dit_submodule(model, {DIT_ATTN: Resource()})
    attention = sub.attention()
    q, k = torch.zeros(1, 6, 2, 4), torch.zeros(1, 10, 2, 4)
    assert attention.window(q, k, k).shape == q.shape and attention.text(q, k[:, :3], k[:, :3]).shape == q.shape
    assert calls == [(cross_label("chunk", "window"), 6, 10), (cross_label("chunk", "text"), 6, 3)]
    assert sub.attention() is attention  # one set of callables per binding


def test_the_declared_window_is_what_the_rolling_cache_attends(model):
    cfg, frame_seqlen = model.config, 3
    cache = RollingKVCache.allocate(1, cfg.local_attn_size * frame_seqlen, 1, 1,
                                    sink_tokens=cfg.sink_size * frame_seqlen, windowed=True, device="cpu",
                                    dtype=torch.float32)
    for c in range(9):
        plan = cache.plan(c * cfg.chunk_size * frame_seqlen, cfg.chunk_size * frame_seqlen)
        window = cache.window(plan)
        assert window.stop - window.start == attended_frames(cfg, c) * frame_seqlen
        cache.global_end, cache.local_end = plan.current_end, plan.local_end


def test_the_shipped_config_serves_one_rollout_at_a_time(model):
    from pathlib import Path

    import yaml

    path = Path(__file__).parents[2] / "configs" / "lingbot_world_1_3b.yaml"
    model.validate_config_yaml(yaml.safe_load(path.read_text()), str(path))
    for limit in (None, 0, 2, 16, True, "1"):
        config = {"max_concurrent_requests": limit} if limit is not None else {}
        with pytest.raises(ValueError, match="max_concurrent_requests: 1") as err:
            model.validate_config_yaml(config, "lingbot.yaml")
    assert "4.7 GiB at 480x832" in str(err.value) and "up to 11.2 GiB" in str(err.value)
