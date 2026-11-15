"""Contract tests for the LingBot-World-V2 serving shell. CPU-only, no
checkpoint: the schedule tables, the rolling K/V eviction, the walk wiring,
the stop/veto, request and config validation, and that the per-chunk
condition and Plucker rays equal the whole-rollout ones. Numerics against the
reference are the oracle-gated parity harness's job
(``test/lingbot_world/run_parity.py``).
"""

import math
import sys
from unittest import mock

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources import RaggedCrossAttentionSpec, RaggedCrossAttentionStep, cross_label  # noqa: E402
from mstar.engine.resources.kv.rolling import (  # noqa: E402
    RollingKVConfig,
    RollingKVSpec,
    RollingKVStep,
)
from mstar.engine.resources.kv.rolling.manager import RollingKVManager  # noqa: E402
from mstar.engine.resources.request_pool import (  # noqa: E402
    RequestPoolConfig,
    RequestPoolReqConfig,
    RequestPoolSpec,
    RequestPoolStep,
)
from mstar.engine.resources.request_pool.manager import RequestPool  # noqa: E402
from mstar.engine.resources.step import Segment, StepContext  # noqa: E402
from mstar.graph.base import GraphNode, Loop, TensorPointerInfo  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT  # noqa: E402
from mstar.model.components import wan_vae  # noqa: E402
from mstar.model.lingbot_world import submodules  # noqa: E402
from mstar.model.lingbot_world.components.dit import (  # noqa: E402
    REFERENCE_ATTENTION,
    ChunkOnlyStore,
)
from mstar.model.lingbot_world.components.schedule import make_fast_schedule  # noqa: E402
from mstar.model.lingbot_world.lingbot_world_model import (  # noqa: E402
    DIT_KV_TOKENS,
    DIT_LAT_HW,
    REFERENCE_MODEL_KWARGS,
    LingBotWorldModel,
)
from mstar.model.lingbot_world.submodules import (  # noqa: E402
    DIT_ATTN,
    DIT_KV,
    DIT_POOL,
    ENCODER_OUT,
    ROLLOUT_LOOP_NAME,
    ROLLOUT_WALK,
    TEXT_KV,
    EncoderStream,
    LingBotDitSubmodule,
    encode_condition_chunk,
    first_frame_mask,
    rollout_state_buffers,
)
from mstar.model.submodule_base import NodeInputs, PerRequestState  # noqa: E402


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


def _pool_and_window(num_layers: int, tokens: int, num_heads: int = 1, head_dim: int = 1, dtype=torch.float32):
    """A CPU request pool holding one rollout's K/V buffer of ``tokens`` a layer, the
    rolling window over it, and the rollout (request 7) opened in it."""
    window = RollingKVManager(RollingKVConfig(num_layers=num_layers, num_heads=num_heads, head_dim=head_dim,
                                              pool=DIT_POOL, dtype=dtype), torch.device("cpu"), name=DIT_KV)
    wants = RequestPoolReqConfig.of({DIT_KV: window.config.buffer(tokens)})
    pool = RequestPool(RequestPoolConfig(pool_bytes=wants.nbytes()), torch.device("cpu"), name=DIT_POOL)
    window.bind_dependencies({DIT_POOL: pool})
    pool.ingest_request(7, wants)
    assert pool.reserve(7, "dit", ROLLOUT_WALK, None) is None
    return pool, window


def _step_chunk(sub, spans, pool, window, rid: int = 7):
    """Admit and plan the chunk the dit declares, pool first, as the engine does."""
    steps = sub.declare_step(ROLLOUT_WALK, [rid], [NodeInputs(resource_step_info=spans)]).steps
    ctx = StepContext(request_ids=(rid,), graph_walk=ROLLOUT_WALK, slot=0, capture=False)
    assert pool.admit(steps[DIT_POOL], ctx).ok and window.admit(steps[DIT_KV], ctx).ok
    pool.plan(steps[DIT_POOL], ctx)
    window.plan(steps[DIT_KV], ctx)
    return steps[DIT_KV], ctx


def _rollout_frames(model, image_hw, num_chunks: int, passes: int = 5):
    """Step a CPU store as the dit declares its chunks, writing each chunk
    ``passes`` times (denoise steps, then the commit) tagged with its latent
    frames; yields the latent frames each chunk attends."""
    cfg = model.config
    sub = _bare_dit_submodule(model, {})
    spans = [sub.chunk_spans(c, image_hw, num_chunks * cfg.chunk_size) for c in range(num_chunks)]
    fs = spans[0].chunk // cfg.chunk_size
    tokens = max(s.position.sink + s.position.window for s in spans)
    pool, store = _pool_and_window(cfg.num_layers, tokens)
    sub.node_resources = {DIT_POOL: pool, DIT_KV: store}
    for c, s in enumerate(spans):
        step, ctx = _step_chunk(sub, s, pool, store)
        tags = torch.arange(c * cfg.chunk_size, (c + 1) * cfg.chunk_size).repeat_interleave(fs).float()
        for _ in range(passes):
            store.write(0, tags.view(1, -1, 1, 1), tags.view(1, -1, 1, 1))
        store.commit(step, ctx)
        held = store.window(0)[0].flatten()
        assert held.numel() == s.window
        yield c, held.view(-1, fs)[:, 0].long().tolist()


def test_the_store_keeps_the_sink_and_the_latest_window():
    model = LingBotWorldModel(skip_weight_loading=True)
    cfg, (window, sink) = model.config, (model.config.local_attn_size, model.config.sink_size)
    for c, frames in _rollout_frames(model, (48, 64), num_chunks=8):
        end = (c + 1) * cfg.chunk_size
        if end <= window:
            assert frames == list(range(end))
        else:
            assert frames == list(range(sink)) + list(range(end - (window - sink), end))


def test_without_a_window_the_store_keeps_every_frame():
    model = LingBotWorldModel(skip_weight_loading=True)
    model.config.local_attn_size = -1
    *_, (c, frames) = _rollout_frames(model, (48, 64), num_chunks=6)
    assert frames == list(range(24))


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


def _pointer(*dims: int) -> TensorPointerInfo:
    return TensorPointerInfo(dims=list(dims), dtype=torch.float32, nbytes=0, address=0, stride=[], uuid=0,
                             source_session_id="", source_entity="")


def test_the_plan_runs_both_walks_with_the_rollouts_chunk_count(model):
    signals = {"text_inputs": [_pointer(512)], "image_inputs": [_pointer(3, 480, 832)],
               "camera_poses": [_pointer(8, 6, 60, 104)], "intrinsics": [_pointer(4)]}
    plan = model.walk_plan("default", ["text", "image"], ["video_frame"], signals, {})
    assert [step.walk for step in plan] == ["encode_text", ROLLOUT_WALK] and not any(step.bind for step in plan)
    assert all(step.step_metadata == {"guidance_scale": 1.0, "num_latent_frames": 8} for step in plan)
    with pytest.raises(ValueError, match="intrinsics"):
        model.walk_plan("default", ["text", "image"], ["video_frame"], {**signals, "intrinsics": []}, {})


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


def test_an_rgba_png_drops_its_alpha_as_the_reference_reads_it(model, tmp_path):
    """The reference drops the alpha channel rather than compositing on white."""
    from PIL import Image

    rgba = np.random.default_rng(0).integers(0, 256, size=(24, 40, 4), dtype=np.uint8)
    path = tmp_path / "rgba.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    image = model.load_image(str(path), "cpu").data
    assert image.dtype == torch.uint8 and image.shape == (3, 24, 40)
    assert torch.equal(image, torch.from_numpy(rgba[..., :3].copy()).permute(2, 0, 1))


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
    assert out.new_input_tensors["camera_poses"][0].shape[0] == 4  # one chunk of latent frames


@pytest.mark.parametrize("num_frames,num_poses", [(13, 13), (93, 269), (361, 269)])
def test_the_rollout_loop_is_capped_at_the_chunks_process_prompt_lays_out(model, monkeypatch, num_frames, num_poses):
    """The cap alone ends the rollout: one iteration per chunk of the poses the DiT reads."""
    monkeypatch.setattr(model, "_ensure_tokenizer", lambda: None)
    monkeypatch.setattr(model, "tokenizer", lambda text, **kwargs: type("Ids", (), {"input_ids": [0, 1]})())
    poses = np.tile(np.eye(4), (num_poses, 1, 1))
    req = dict(intrinsics=[415.0, 415.0, 416.0, 240.0], num_frames=num_frames, camera_poses=poses)
    out = model.process_prompt("p", ["text", "image"], ["video_frame"],
                               tensors={"image_inputs": [torch.zeros(3, 480, 832, dtype=torch.uint8)]}, **req)
    chunks = out.new_input_tensors["camera_poses"][0].shape[0] // model.config.chunk_size
    assert model.get_loop_caps(**req) == {ROLLOUT_LOOP_NAME: chunks}


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
    """Every flag in the parity report must hold; anything else fails the run."""
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
        "dit_first_divergent_call": None,
    }
    assert run_parity.mismatches(report) == []
    for key, value, point in [
        ("text_embeds_tail_zero", False, ".text_embeds_tail_zero"),
        ("dit_calls_bit_exact", {"x_in": True, "flow": False}, ".dit_calls_bit_exact.flow"),
        ("e2e_chunks", [{"latent": {"bit_exact": True}, "frames": {"bit_exact": False}}],
         ".e2e_chunks[0].frames.bit_exact"),
        ("image", {"bit_exact": False, "shape_mismatch": [[1], [2]]}, ".image.bit_exact"),
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
    assert torch.equal(full.new_input_tensors["camera_poses"][0], top.new_input_tensors["camera_poses"][0])
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


@pytest.mark.parametrize("pooled", [False, True])
def test_the_condition_chunk_by_chunk_is_the_whole_clip_encode(one_thread, pooled):
    """The streamed encode the dit runs per chunk against the reference's one
    ``vae.encode`` of the image followed by black frames, on a tiny Wan VAE,
    long enough that the black frames drive the cache to its fixed point.
    ``pooled`` encodes the dit's way: the encoder saves uncopied into a
    ``PoolFeatCache`` that compares each slot as it overwrites it; it settles on
    the same chunk as a plain cache."""
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

    def encode(stream):
        chunks, settled = [], None
        for k in range(lat_f // cfg.chunk_size):
            chunks.append(encode_condition_chunk(vae, image, stream, k * cfg.chunk_size, cfg.chunk_size, cfg))
            if settled is None and stream.steady is not None:
                settled = k
        return chunks, settled

    plain, settled = encode(EncoderStream(vae))
    chunks = plain
    if pooled:
        probe = EncoderStream(vae)
        with torch.inference_mode():
            vae.encoder(image[None, :, None], feat_cache=probe.cache, feat_idx=[0])
            vae.encoder(torch.zeros(1, 3, 4, 16, 24), feat_cache=probe.cache, feat_idx=[0])
        shapes = [tuple(t.shape) if torch.is_tensor(t) else None for t in probe.cache]
        wan_vae.encoder_saves_uncopied(vae)
        with torch.inference_mode():
            stream = EncoderStream.over(wan_vae.PoolFeatCache.fresh(shapes, torch.float32, "cpu"), None)
        chunks, pooled_settled = encode(stream)
        assert pooled_settled == settled
    assert settled is not None and settled < lat_f // cfg.chunk_size - 1
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


def test_the_reference_numerics_dit_fails_to_build_without_flash_attn(monkeypatch):
    """Not the first request: with flash-attn missing the server must not
    come up healthy and then fail every request."""
    from mstar.model.lingbot_world import weight_loader

    def no_weights(*args, **kwargs):
        raise AssertionError("loaded weights before checking flash-attn")

    monkeypatch.setitem(sys.modules, "flash_attn", None)
    monkeypatch.setattr(weight_loader, "build_lingbot_dit", no_weights)
    with pytest.raises(ImportError, match="install flash-attn"):
        LingBotWorldModel(reference_numerics=True).get_submodule("dit")


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


def test_the_dit_declares_a_request_pool_the_rolling_kv_over_it_and_a_ragged_cross_attention(model):
    pool, kv, attn = model.get_node_resources()
    assert isinstance(pool, RequestPoolSpec) and pool.resource_key == DIT_POOL and pool.nodes == {"dit"}
    assert pool.config.pool_bytes == model.default_pool_bytes() and pool.config.inference_tensors
    assert isinstance(kv, RollingKVSpec) and kv.resource_key == DIT_KV and kv.nodes == {"dit"}
    assert (kv.config.num_layers, kv.config.num_heads, kv.config.head_dim) == (30, 12, 128)
    assert kv.config.dtype == torch.bfloat16 and kv.config.pool == DIT_POOL and kv.depends_on() == {DIT_POOL}
    assert isinstance(attn, RaggedCrossAttentionSpec)
    assert attn.resource_key == DIT_ATTN and attn.nodes == {"dit"}
    assert (attn.config.num_qo_heads, attn.config.num_kv_heads, attn.config.head_dim) == (12, 12, 128)
    assert attn.config.dtype == torch.bfloat16
    # the reference's FA2 attends the same store
    (ref_pool, ref_kv) = LingBotWorldModel(skip_weight_loading=True, reference_numerics=True).get_node_resources()
    assert isinstance(ref_pool, RequestPoolSpec) and isinstance(ref_kv, RollingKVSpec) and ref_kv.resource_key == DIT_KV


def test_the_default_pool_holds_one_rollout_at_the_largest_area_or_two_at_the_default(model):
    """A rollout keeps its rolling K/V (18 latent frames of at most max_area / 256 tokens,
    whatever the aspect), its text K/V and its VAE caches in the pool. With a bf16 decode
    the default pool, 20.1 GiB, holds one at 720x1280 (11.1 GiB of K/V, 4.8 GiB of
    encoder cache, 4.1 GiB of decoder cache) or two at the default 464x832."""
    cfg = model.config
    assert model.max_window_tokens() == 18 * (720 * 1280 // 256)
    sub = _bare_dit_submodule(model, {})
    largest = model.largest_latent()
    for image_hw in [(720, 1280), (1280, 720), (960, 960), (408, 2256), (2256, 408), (480, 832)]:
        lat = model.geometry(image_hw, 720, 1280)
        last = sub.chunk_spans(cfg.max_chunks - 1, (lat[0] * 8, lat[1] * 8)).position
        assert last.sink + last.window <= model.max_window_tokens(), image_hw
        assert lat[0] * lat[1] <= largest[0] * largest[1] == 90 * 160, image_hw  # the cache scales with pixels
    with mock.patch.object(submodules, "wan_vae_decode_dtype", return_value=torch.bfloat16):
        assert round(model.default_pool_bytes() / 2**30, 1) == 20.1
        first = sub.chunk_spans(0, (464, 832)).position
        default = RequestPoolReqConfig.of(rollout_state_buffers(cfg, (58, 104), first.sink + first.window)).nbytes()
        assert model.default_pool_bytes() // default == 2


def test_a_rollouts_state_is_its_kv_text_kv_and_every_vae_cache_slot(model):
    """At 464x832 (a 58x104 latent): 4.7 GiB of rolling K/V, 90 MiB of text K/V, 2.0 GiB
    of encoder cache (fp32) and 1.7 GiB of decoder cache (bf16), as measured on an H100."""
    layout = submodules.encoder_cache_layout()
    assert len(layout.slots) == 26 and sum(s is not None for s in layout.slots) == 24
    decoder = submodules.decoder_cache_layout()
    assert len(decoder) == 33 and sum(s is not None for s in decoder) == 32
    with mock.patch.object(submodules, "wan_vae_decode_dtype", return_value=torch.bfloat16):
        buffers = rollout_state_buffers(model.config, (58, 104), 27144)
    gib = {prefix: sum(math.prod(shape) * dtype.itemsize for name, (shape, dtype) in buffers.items()
                       if name.startswith(prefix)) / 2**30 for prefix in (DIT_KV, TEXT_KV, "enc.", "dec.")}
    assert {k: round(v, 2) for k, v in gib.items()} == {DIT_KV: 4.66, TEXT_KV: 0.09, "enc.": 2.02, "dec.": 1.7}
    assert buffers["dec.32" if decoder[32] else "dec.31"][0][-2:] == (464, 832)
    assert buffers[ENCODER_OUT] == ((1, 32, 1, 58, 104), torch.float32)
    assert all(dtype == torch.float32 for name, (_, dtype) in buffers.items() if name.startswith("enc."))


def test_the_decoder_cache_follows_its_region_when_the_pool_moves_it(model):
    """Kept between chunks as views of the pool, and taken again, with what it holds,
    after a rearrange has slid the region to another place."""
    buffers = {name: (shape, torch.float32) for name, (shape, _) in
               rollout_state_buffers(model.config, (2, 2), 1).items() if name.startswith("dec.")}
    wants = RequestPoolReqConfig.of(buffers)
    pool = RequestPool(RequestPoolConfig(pool_bytes=3 * wants.nbytes(), inference_tensors=True),
                       torch.device("cpu"), name=DIT_POOL)
    sub = _bare_dit_submodule(model, {DIT_POOL: pool})

    def step(*rids):
        segs = tuple(Segment(request_id=rid, label="chunk", span=1) for rid in rids)
        ctx = StepContext(request_ids=rids, graph_walk=ROLLOUT_WALK, slot=0, capture=False)
        assert pool.admit(RequestPoolStep(segments=segs), ctx).ok
        pool.plan(RequestPoolStep(segments=segs), ctx)

    for rid in (1, 2):
        pool.ingest_request(rid, wants)
        assert pool.reserve(rid, "dit", ROLLOUT_WALK, None) is None
    step(1, 2)
    step(2)  # the cache is request 2's
    state = PerRequestState()
    state.add_all(decoder_cache=None, decoder_cache_at=None)
    cache = sub._decoder_cache(state)
    filled = [i for i, b in enumerate(cache.buffers) if b is not None]
    with torch.inference_mode():  # as the dit decodes
        for i in filled:
            cache[i] = torch.full(cache.buffers[i].shape, float(i))
    assert sub._decoder_cache(state) is cache  # nothing moved
    pool.remove_request(1)
    pool.ingest_request(3, RequestPoolReqConfig.of({"big": ((2 * wants.nbytes() // 4,), torch.float32)}))
    assert pool.reserve(3, "dit", ROLLOUT_WALK, None) is None
    step(3)  # only fits once request 2 slides to the start
    step(2)
    moved = sub._decoder_cache(state)
    assert moved is not cache and state["decoder_cache"] is moved
    for i in filled:
        assert moved[i].data_ptr() == pool.view(f"dec.{i}").data_ptr() != cache[i].data_ptr()
        assert torch.equal(moved[i], torch.full(cache.buffers[i].shape, float(i)))


def test_the_encode_stream_follows_its_region_when_the_pool_moves_it(model):
    """Kept between chunks as views of the pool, and taken again, with what it holds,
    after a rearrange has slid the region to another place; once settled, the one
    output it keeps moves with it too."""
    lat_hw = (2, 2)
    buffers = {name: spec for name, spec in rollout_state_buffers(model.config, lat_hw, 1).items()
               if name.startswith("enc.")}
    wants = RequestPoolReqConfig.of(buffers)
    pool = RequestPool(RequestPoolConfig(pool_bytes=3 * wants.nbytes(), inference_tensors=True),
                       torch.device("cpu"), name=DIT_POOL)
    sub = _bare_dit_submodule(model, {DIT_POOL: pool})

    def step(*rids):
        segs = tuple(Segment(request_id=rid, label="chunk", span=1) for rid in rids)
        ctx = StepContext(request_ids=rids, graph_walk=ROLLOUT_WALK, slot=0, capture=False)
        assert pool.admit(RequestPoolStep(segments=segs), ctx).ok
        pool.plan(RequestPoolStep(segments=segs), ctx)

    def slide_request_2():
        pool.remove_request(1)
        pool.ingest_request(3, RequestPoolReqConfig.of({"big": ((2 * wants.nbytes() // 4,), torch.float32)}))
        assert pool.reserve(3, "dit", ROLLOUT_WALK, None) is None
        step(3)  # only fits once request 2 slides to the start
        step(2)

    for rid in (1, 2):
        pool.ingest_request(rid, wants)
        assert pool.reserve(rid, "dit", ROLLOUT_WALK, None) is None
    step(1, 2)
    step(2)  # the stream is request 2's
    state = PerRequestState()
    state.add_all(encoder_stream=None, encoder_stream_at=None)
    stream = sub._encoder_stream(state)
    cache = stream.cache
    filled = [i for i, b in enumerate(cache.buffers) if b is not None]
    with torch.inference_mode():  # as the dit encodes
        for i in filled:
            cache[i] = torch.full(cache.buffers[i].shape, float(i))
    assert sub._encoder_stream(state) is stream  # nothing moved
    slide_request_2()
    moved = sub._encoder_stream(state)
    assert moved is not stream and state["encoder_stream"] is moved
    for i in filled:
        assert moved.cache[i].data_ptr() == pool.view(f"enc.{i}").data_ptr() != cache[i].data_ptr()
        assert torch.equal(moved.cache[i], torch.full(cache.buffers[i].shape, float(i)))
    # settled: the output is kept in the pool, once, and follows the region from then on
    out = torch.randn(1, 32, 1, *lat_hw)
    with torch.inference_mode():
        sub._keep_settled_output(state, EncoderStream.over(None, out))
    kept = state["encoder_stream"].steady
    assert kept.data_ptr() == pool.view(ENCODER_OUT).data_ptr() and torch.equal(kept, out)
    pool.remove_request(3)
    pool.ingest_request(1, wants)
    assert pool.reserve(1, "dit", ROLLOUT_WALK, None) is None
    step(1)
    step(2)
    assert torch.equal(sub._encoder_stream(state).steady, out)


@pytest.mark.parametrize("image_hw", [(464, 832), (720, 1280)])
def test_a_rollout_declares_its_rolling_kv_so_it_waits_in_line_for_the_pool(model, monkeypatch, image_hw):
    """The sink and window its steps will keep, at the output size process_prompt
    settles: the engine queues the request until the pool has that room, and
    refuses one that can never fit before it reaches a worker."""
    monkeypatch.setattr(model, "_ensure_tokenizer", lambda: None)
    monkeypatch.setattr(model, "tokenizer", lambda text, **kwargs: type("Ids", (), {"input_ids": [0, 1]})())
    req = dict(intrinsics=[415.0, 415.0, 416.0, 240.0], camera_poses=np.tile(np.eye(4), (93, 1, 1)),
               height=image_hw[0], width=image_hw[1])
    out = model.process_prompt("p", ["text", "image"], ["video_frame"],
                               tensors={"image_inputs": [torch.zeros(3, *image_hw, dtype=torch.uint8)]}, **req)
    assert tuple(out.new_input_tensors["image_inputs"][0].shape[1:]) == image_hw
    # the data worker folds the metadata into the request's kwargs, which the conductor resolves configs from
    configs = model.get_request_resource_configs(["text", "image"], ["video_frame"], {**req, **out.metadata})
    first = _bare_dit_submodule(model, {}).chunk_spans(0, image_hw).position
    lat_hw = (image_hw[0] // 8, image_hw[1] // 8)
    wants = RequestPoolReqConfig.of(rollout_state_buffers(model.config, lat_hw, first.sink + first.window))
    assert configs == {DIT_POOL: wants}
    (pool, _, _) = model.get_node_resources()
    configs[DIT_POOL].validate(pool)
    pool.apply_yaml_overrides(pool_bytes=configs[DIT_POOL].nbytes() - 1)
    with pytest.raises(ValueError, match="pool_bytes"):
        configs[DIT_POOL].validate(pool)


def test_only_process_prompt_sets_the_region_size():
    """A client can't size its own region: the data worker drops a model's
    INTERNAL_KWARGS from the request before process_prompt sets them."""
    assert {DIT_KV_TOKENS, DIT_LAT_HW} <= set(LingBotWorldModel.INTERNAL_KWARGS)


SWITCHES = ("reference_numerics", "dit_compile_autotune")


@pytest.mark.parametrize("name", SWITCHES)
@pytest.mark.parametrize("value", ["false", 0, None])
def test_a_model_kwargs_switch_must_be_a_bool(name, value):
    """A yaml "false" is truthy; read as on it would serve the opposite setting."""
    with pytest.raises(ValueError, match=name):
        LingBotWorldModel(skip_weight_loading=True, **{name: value})


def test_model_kwargs_switches_set_the_config():
    default = LingBotWorldModel(skip_weight_loading=True).config
    assert [getattr(default, n) for n in SWITCHES] == [False, False]
    for name in SWITCHES:
        flipped = LingBotWorldModel(skip_weight_loading=True, **{name: not getattr(default, name)}).config
        assert getattr(flipped, name) is not getattr(default, name)
        assert all(getattr(flipped, n) is getattr(default, n) for n in SWITCHES if n != name)
    reference = LingBotWorldModel(skip_weight_loading=True, **REFERENCE_MODEL_KWARGS).config
    assert [getattr(reference, n) for n in SWITCHES] == [True, False]


def test_an_unknown_model_kwarg_is_refused():
    """A switch this one replaced, or a typo, fails at load instead of serving the default."""
    with pytest.raises(ValueError, match="vae_decode_fp32"):
        LingBotWorldModel(skip_weight_loading=True, vae_decode_fp32=True)


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
    # with the store bound, the same step declares the chunk's write and its place in the stream
    sub.node_resources = {DIT_ATTN: object(), DIT_KV: object()}
    kv = sub.declare_step(ROLLOUT_WALK, ["r"], inputs).steps[DIT_KV]
    assert isinstance(kv, RollingKVStep) and [(s.label, s.span) for s in kv.segments] == [("chunk", 4 * frame)]
    assert kv.positions["r"] == (5 * 4 * frame, 6 * frame, 12 * frame)


def test_without_the_resource_only_the_reference_numerics_attend(model):
    sub = _bare_dit_submodule(model, {})
    with pytest.raises(RuntimeError, match=DIT_ATTN):
        sub.attention()  # a silent FA2 fallback would hide a deployment without the resource
    assert sub.declare_step(ROLLOUT_WALK, ["r"], [NodeInputs(resource_step_info=sub.chunk_spans(0, (16, 16)))]) is None
    with pytest.raises(RuntimeError, match=DIT_KV):
        sub.kv_store()  # the rolling K/V live only in the engine's store
    reference = _bare_dit_submodule(LingBotWorldModel(skip_weight_loading=True, reference_numerics=True), {})
    assert reference.attention() is REFERENCE_ATTENTION


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


def test_the_fast_config_only_adds_the_autotuned_compile():
    from pathlib import Path

    import yaml

    configs = Path(__file__).parents[2] / "configs"
    base = yaml.safe_load((configs / "lingbot_world_1_3b.yaml").read_text())
    fast = yaml.safe_load((configs / "lingbot_world_1_3b_fast.yaml").read_text())
    assert fast.pop("model_kwargs") == {"dit_compile_autotune": True}
    assert fast == base
    LingBotWorldModel(skip_weight_loading=True).validate_config_yaml(fast, "lingbot_world_1_3b_fast.yaml")


def test_the_shipped_config_lets_the_pool_decide_how_many_rollouts_run(model):
    """Every rollout's state is in the request pool, which queues a rollout until it
    has room, so the config sets no conductor limit and keeps the default pool."""
    from pathlib import Path

    import yaml

    path = Path(__file__).parents[2] / "configs" / "lingbot_world_1_3b.yaml"
    config = yaml.safe_load(path.read_text())
    assert "max_concurrent_requests" not in config and not config.get("resources")
    model.validate_config_yaml(config, str(path))
    (pool, _, _) = model.get_node_resources()
    assert pool.config.pool_bytes == model.default_pool_bytes()


def _compiled_rollout_graphs(num_layers: int, chunks: int) -> int:
    from mstar.model.lingbot_world.components.dit import DitAttention, LingBotCausalDiT
    from mstar.model.lingbot_world.config import LingBotWorldConfig

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    cfg = LingBotWorldConfig(dim=32, ffn_dim=64, num_heads=2, num_layers=num_layers, freq_dim=16, text_dim=8,
                             text_len=8, rope_max_frames=64, local_attn_size=8, sink_size=2)
    dit = LingBotCausalDiT(cfg).eval()
    dit.compile_blocks(backend="eager", fullgraph=False, dynamic=False)

    @torch.compiler.disable
    def attend(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    attention = DitAttention(window=attend, text=attend)
    frame_seqlen, cs = 4, cfg.chunk_size
    text_kv = dit.text_kv(torch.randn(3, 8))
    # warmup traces against a store that keeps nothing, as the submodule's does at load
    conditioning = dit.chunk_conditioning(torch.zeros(1, 384, cs, 4, 4), 0)
    dit(torch.randn(16, cs, 4, 4), torch.tensor([500.0]), text_kv, torch.randn(20, cs, 4, 4),
        conditioning, ChunkOnlyStore(), attention)
    pool, store = _pool_and_window(num_layers, cfg.local_attn_size * frame_seqlen, cfg.num_heads, cfg.head_dim)
    sub = _bare_dit_submodule(LingBotWorldModel(skip_weight_loading=True), {DIT_POOL: pool, DIT_KV: store})
    sub.config = cfg
    for chunk in range(chunks):
        spans = sub.chunk_spans(chunk, (32, 32))
        step, ctx = _step_chunk(sub, spans, pool, store)
        conditioning = dit.chunk_conditioning(torch.zeros(1, 384, cs, 4, 4), chunk * cs)
        for _ in range(2):
            dit(torch.randn(16, cs, 4, 4), torch.tensor([500.0]), text_kv, torch.randn(20, cs, 4, 4),
                conditioning, store, attention)
        store.commit(step, ctx)
    return torch._dynamo.utils.counters["stats"]["unique_graphs"]


def test_compiled_blocks_trace_once_for_every_layer_and_chunk():
    """The store's offsets and the layer index stay out of the compiled blocks,
    so a rollout compiles them once, not once per layer or per chunk (five
    chunks run past the first eviction), and the warmup's scratch store traces
    the same graphs as the engine's."""
    one_chunk = _compiled_rollout_graphs(num_layers=2, chunks=1)
    assert one_chunk > 0
    assert _compiled_rollout_graphs(num_layers=4, chunks=5) == one_chunk

