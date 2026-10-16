"""Contract tests for the LingBot-World-V2 serving shell. CPU-only, no
checkpoint: the schedule tables, the rolling K/V eviction, the walk wiring,
the stop/veto, and request validation. Numerics against the reference are
the oracle-gated parity harness's job (``test/lingbot_world/run_parity.py``).
"""

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.graph.base import GraphNode, Loop  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT  # noqa: E402
from mstar.model.lingbot_world.components.dit import RollingKVCache  # noqa: E402
from mstar.model.lingbot_world.components.schedule import make_fast_schedule  # noqa: E402
from mstar.model.lingbot_world.lingbot_world_model import LingBotWorldModel  # noqa: E402
from mstar.model.lingbot_world.submodules import (  # noqa: E402
    ROLLOUT_LOOP_NAME,
    ROLLOUT_WALK,
    LingBotDitSubmodule,
    first_frame_mask,
)


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


def test_walks_run_text_then_image_then_a_chunk_loop_that_emits_frames(model):
    walks = model.get_graph_walk_graphs()
    assert list(walks) == ["encode_text", "encode_image", ROLLOUT_WALK]
    loop = walks[ROLLOUT_WALK]
    assert isinstance(loop, Loop) and loop.name == ROLLOUT_LOOP_NAME
    node = loop.section
    assert isinstance(node, GraphNode) and node.name == "dit"
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
