"""MiniMax-H3-Image in dummy mode, on a pinned config: no weights, GPU or network."""

from __future__ import annotations

import dataclasses
import io

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mstar.graph.base import GraphNode, Loop, Parallel, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.minimax_h3.components.layout import patchify_video
from mstar.model.minimax_h3.components.transformer import H3DiT, H3Inputs
from mstar.model.minimax_h3.config import DENOISE_LOOP, TEXT_TAG, VIDEO_TAG
from mstar.model.minimax_h3.submodules import (
    COND_LATENTS,
    LATENTS,
    TEXT_EMBEDS,
    TEXT_INPUTS,
    VISION_PIXELS,
    H3TextEncoderSubmodule,
)
from mstar.model.minimax_h3.tests.test_model import _Tokenizer
from mstar.model.minimax_h3.tests.tiny_checkpoint import CFG, random_checkpoint
from mstar.model.minimax_h3_image import weight_loader
from mstar.model.minimax_h3_image.config import SIGMAS, H3ImageConfig, bound_short_edge, canvas_size
from mstar.model.minimax_h3_image.layout import (
    T_GENERATED,
    T_REFERENCE,
    H3ImageLayout,
    modality_tags,
    position_ids,
    timestep_classes,
)
from mstar.model.minimax_h3_image.minimax_h3_image_model import (
    ENCODE_REFERENCES_WALK,
    ENCODE_TEXT_WALK,
    GENERATE_EDIT_WALK,
    GENERATE_WALK,
    H3ImageModel,
)
from mstar.model.minimax_h3_image.submodules import (
    REFERENCE_PIXELS,
    H3ImageDenoiseSubmodule,
    H3ImageSchedule,
    h3_image_schedule,
    layout_from_metadata,
)

TINY = H3ImageConfig(transformer=dataclasses.replace(CFG, audio_enabled=False, num_output_heads=3),
                     sigmas=(1.0, 0.75, 0.5, 0.0))


def _model(**kwargs) -> H3ImageModel:
    model = H3ImageModel(model_path_hf="test/minimax_h3_image", skip_weight_loading=True, **kwargs)
    model.set_config(H3ImageConfig())
    model.tokenizer = _Tokenizer()
    return model


def _info(dims: list[int]) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=dims, dtype="torch.int64", nbytes=8, address=0, stride=[1] * len(dims),
        uuid="uuid", source_session_id="test:0", source_entity="test",
    )


class _ImageProcessor:
    """Qwen3-VL's processor as far as the model reads it: 16-px patches, merged 2x2."""

    merge_size = 2

    def __call__(self, images, return_tensors="pt"):
        grids = torch.tensor([[1, image.size[1] // 16, image.size[0] // 16] for image in images])
        rows = int(sum(int(g.prod()) for g in grids))
        return {"pixel_values": torch.arange(rows * 4, dtype=torch.float32).view(rows, 4), "image_grid_thw": grids}


def test_graph_walks():
    walks = _model().get_graph_walk_graphs()
    assert set(walks) == {ENCODE_TEXT_WALK, ENCODE_REFERENCES_WALK, GENERATE_WALK, GENERATE_EDIT_WALK}
    references = walks[ENCODE_REFERENCES_WALK]
    assert isinstance(references, Parallel)
    assert {n.name: set(n.input_names) for n in references.sections} == {
        "text_encoder": {TEXT_INPUTS, VISION_PIXELS}, "vae_encoder": {REFERENCE_PIXELS}}
    assert isinstance(walks[ENCODE_TEXT_WALK], GraphNode)
    for walk, edit in ((GENERATE_WALK, False), (GENERATE_EDIT_WALK, True)):
        loop, decoder = walks[walk].sections
        assert isinstance(walks[walk], Sequential) and isinstance(loop, Loop) and loop.name == DENOISE_LOOP
        assert (COND_LATENTS in loop.section.input_names) == edit
        assert [(e.next_node, e.output_modality) for e in decoder.outputs] == [(EMIT_TO_CLIENT, "image")]


def test_canvas_and_reference_sizes():
    # below the default 4 MP generation area a request renders on a larger canvas, same aspect
    assert canvas_size(1024, 1024, 2048 * 2048) == (2048, 2048)
    assert canvas_size(1024, 1024, 1024) == (1024, 1024)
    assert canvas_size(3552, 4736, 2048 * 2048) == (3552, 4736)
    assert canvas_size(720, 1280, 2048 * 2048) == (1536, 2720)
    # a short edge inside the bounds keeps the original pixels, else it lands on the nearer bound
    assert bound_short_edge(1500, 1100, 1024, 2048) == (1500, 1100)
    assert bound_short_edge(800, 600, 1024, 2048) == (1376, 1024)
    assert bound_short_edge(4000, 3000, 1024, 1024) == (1376, 1024)


def test_layout_rows_positions_and_tags():
    layout = H3ImageLayout(text_len=3, latent_height=4, latent_width=4, references=((4, 8),),
                           text_image_spans=((1, 2),))
    assert (layout.num_condition_rows, layout.num_video_rows, layout.total_rows) == (8, 4, 15)
    pos = position_ids(layout)
    assert pos[:3].tolist() == [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    # every image is one slot on the rotary clock after the text: the reference, then the target
    assert pos[3:11, 0].unique().tolist() == [3.0] and pos[11:, 0].unique().tolist() == [4.0]
    # a square grid spans [0, 32) per axis; a 1:2 one is centred on the same unit square
    assert pos[11:, 1:].tolist() == [[0, 0], [0, 16], [16, 0], [16, 16]]
    width_axis = (np.linspace(-(2 ** 0.5 - 1) / 2, 1 + (2 ** 0.5 - 1) / 2, 4, endpoint=False) * 32).tolist()
    assert pos[3:7, 2].tolist() == pytest.approx(width_axis, abs=1e-12)
    assert modality_tags(layout).tolist() == [TEXT_TAG, VIDEO_TAG, VIDEO_TAG] + [VIDEO_TAG] * 12
    assert timestep_classes(layout).tolist() == [T_GENERATED] * 3 + [T_REFERENCE] * 8 + [T_GENERATED] * 4


def test_schedule_is_the_released_grid():
    sched = h3_image_schedule(SIGMAS)
    assert isinstance(sched, H3ImageSchedule) and sched.num_steps == 6
    timesteps = torch.tensor([1.0 - s for s in SIGMAS[:-1]], dtype=torch.float32)
    assert torch.equal(sched.timesteps, timesteps)
    # the sigmas the update steps on are the fp32 round trip, not the config's
    assert torch.equal(sched.sigmas, torch.cat([1.0 - timesteps, torch.zeros(1)]))
    assert sched.heads.tolist() == list(range(6)) and sched.reference_timesteps.unique().tolist() == [
        pytest.approx(0.999)]


def test_seed_draws_follow_the_reference_order():
    """posterior samples (drawn by the vae_encoder on the same seed), the target's noise,
    then each reference's, all from one CPU generator."""
    meta = {"text_len": 2, "latent_height": 4, "latent_width": 6, "reference_latent_grids": [[5, 7]]}
    layout = layout_from_metadata(meta)
    assert layout.references == ((4, 6),)  # cropped to whole patches
    assert layout.reference_grids == ((5, 7),)  # the draws' shapes, in the bucket key
    sub = H3ImageDenoiseSubmodule.__new__(H3ImageDenoiseSubmodule)
    sub.config = H3ImageConfig()
    rows = sub.seed_loop_back(layout, torch.Generator().manual_seed(7))[LATENTS]
    gen = torch.Generator().manual_seed(7)
    torch.randn((1, 24, 5, 7), generator=gen)
    target = torch.randn((1, 24, 1, 4, 6), generator=gen)
    reference = torch.randn((1, 24, 4, 6), generator=gen)
    want = torch.cat([patchify_video(reference[:, :, None][0], (2, 2)), patchify_video(target[0], (2, 2))])
    assert torch.equal(rows, want)


def test_dit_without_audio_runs_the_step_head():
    torch.manual_seed(0)
    cfg = TINY.transformer
    dit = H3DiT(cfg).float()
    for p in dit.parameters():
        torch.nn.init.normal_(p, std=0.1)
    assert dit.audio_patch_proj is None and dit.final_layer.audio_out is None
    layout = H3ImageLayout(text_len=3, latent_height=4, latent_width=4, references=((2, 4),))
    sub = H3ImageDenoiseSubmodule(dit, TINY, loop_name="denoise_loop")
    step = sub.build_layout(layout, "cpu")
    video = torch.randn(1, layout.num_condition_rows + layout.num_video_rows, cfg.patch_dim)
    inputs = H3Inputs(video=video, audio=None, text=torch.randn(1, 3, cfg.hidden_size),
                      timesteps=torch.tensor([[0.25, 0.999]]))
    with torch.no_grad():
        outs = [dit(inputs, step, head=torch.tensor([k]))[0] for k in range(3)]
    assert outs[0].shape == video.shape
    # head k is the plain linear of its own weights
    heads = dit.final_layer.video_out
    with torch.no_grad():
        swapped = heads.weight.data[[1, 0, 2]].clone(), heads.bias.data[[1, 0, 2]].clone()
        heads.weight.data, heads.bias.data = swapped
        again = dit(inputs, step, head=torch.tensor([0]))[0]
    assert torch.equal(again, outs[1]) and not torch.equal(outs[0], outs[1])


def test_the_dits_attention_settings():
    """The DiT's read pipelines its Ulysses exchange over 2 head groups and, compiled, traces as
    the resource's op; the eager refiner's read keeps the defaults."""
    from mstar.model.minimax_h3.config import DIT_ATTN, REFINER_ATTN

    eager, compiled = _model().kernel_settings(), _model(compile=True).kernel_settings()
    assert eager[DIT_ATTN] == {"max_segments_per_request": 2, "workspace_mb": 128, "compile_op": False,
                               "ulysses_head_groups": 2}
    assert compiled[DIT_ATTN]["compile_op"] is True
    assert eager[REFINER_ATTN] == compiled[REFINER_ATTN] == {"workspace_mb": 128}


def test_text_cache_keys_the_prompt_and_its_references():
    """With the cache on, process_prompt keys the text node by the prompt's ids and the vision
    tower's pixels, and the walk plan carries the key to every step; off, nothing changes."""
    from mstar.model.base import ProcessPromptOutput
    from mstar.model.minimax_h3.submodules import TEXT_CACHE_KEY

    assert not isinstance(_model().process_prompt("a cat", ["text"], ["image"]), ProcessPromptOutput)
    model = _model(text_cache_entries=4)
    model._image_processor = _ImageProcessor()
    one, again, other = (model.process_prompt(p, ["text"], ["image"]) for p in ("a cat", "a cat", "a dog"))
    assert one.metadata[TEXT_CACHE_KEY] == again.metadata[TEXT_CACHE_KEY] != other.metadata[TEXT_CACHE_KEY]
    image = torch.zeros(3, 1024, 1024, dtype=torch.uint8)
    edits = [model.process_prompt("a cat", ["text", "image"], ["image"], tensors={"image_inputs": [im]})
             for im in (image, image.clone(), torch.zeros(3, 1024, 1280, dtype=torch.uint8))]
    keys = [e.metadata[TEXT_CACHE_KEY] for e in edits]
    assert keys[0] == keys[1] and len({keys[0], keys[2], one.metadata[TEXT_CACHE_KEY]}) == 3
    signals = {TEXT_INPUTS: [_info(list(one.new_input_tensors[TEXT_INPUTS][0].shape))]}
    plan = model.walk_plan("", ["text"], ["image"], signals, one.metadata)
    assert all(step.step_metadata[TEXT_CACHE_KEY] == one.metadata[TEXT_CACHE_KEY] for step in plan)
    with pytest.raises(ValueError, match="text_cache_entries"):
        _model(text_cache_entries=-1)


def test_text_node_serves_a_cached_prompt_without_encoding():
    """A hit returns the stored rows and runs nothing; the least recently used entry goes first."""
    from types import SimpleNamespace

    from mstar.model.minimax_h3.submodules import TEXT_CACHE_KEY, TextStep

    calls = []

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))  # the node reads its device off a parameter

        def forward(self, ids, mask, padded=False):
            calls.append(int(ids.shape[1]))
            return ids[..., None].float()

    sub = H3TextEncoderSubmodule(Encoder(), lambda hidden: hidden * 2, cache_entries=2)
    ids = torch.tensor([1, 2, 3])
    info = SimpleNamespace(step_metadata={TEXT_CACHE_KEY: "k"})
    assert sub.prepare_inputs("w", info, {TEXT_INPUTS: [ids]}).resource_step_info.cache_key == "k"

    def run(key, n=3):
        return sub.forward("w", None, torch.arange(n), TextStep(n, (), key))[TEXT_EMBEDS][0]

    first = run("a")
    assert run("a") is first and calls == [3]
    run("b", 4), run("a"), run("c", 5)  # "b" is the least recently used when "c" arrives
    assert list(sub._cache) == ["a", "c"] and calls == [3, 4, 5]
    run(None), run(None)
    assert calls == [3, 4, 5, 3, 3] and list(sub._cache) == ["a", "c"]  # no key: always computed


def test_process_prompt_and_walk_plan_for_an_edit():
    model = _model()
    model._image_processor = _ImageProcessor()
    small = torch.zeros(3, 600, 800, dtype=torch.uint8)  # short edge below 1024: scaled up for both copies
    kept = torch.zeros(3, 1100, 1500, dtype=torch.uint8)  # kept for the VAE, cropped to whole cells
    out = model.process_prompt("make it so", ["text", "image"], ["image"],
                               tensors={"image_inputs": [small, kept]}, height=1024, width=1024)
    assert [tuple(p.shape) for p in out[REFERENCE_PIXELS]] == [(3, 1024, 1376), (3, 1088, 1488)]
    # the text encoder's copies sit at a 1024 short edge; one [h, w, patch] view each
    assert [tuple(p.shape[:2]) for p in out[VISION_PIXELS]] == [(64, 86), (64, 88)]
    ids = out[TEXT_INPUTS][0]
    tokens = [64 * 86 // 4, 64 * 88 // 4]
    label = [ord(c) % 800 for c in "Picture 1: "]
    assert ids[:len(label)].tolist() == label and ids[len(label)] == 900
    assert len(ids) == 2 * len(label) + sum(tokens) + 4 + len("make it so")
    signals = {name: [_info(list(t.shape)) for t in out[name]] for name in (TEXT_INPUTS, VISION_PIXELS,
                                                                             REFERENCE_PIXELS)}
    plan = model.walk_plan("", ["text", "image"], ["image"], signals, {"height": 1024, "width": 1024})
    assert [step.walk for step in plan] == [ENCODE_REFERENCES_WALK, GENERATE_EDIT_WALK]
    meta = plan[1].step_metadata
    assert (meta["canvas_height"], meta["canvas_width"]) == (2048, 2048)
    assert meta["reference_latent_grids"] == [[64, 86], [68, 93]]
    assert meta["keyframe_grids"] == [[1, 64, 86], [1, 64, 88]]
    layout = layout_from_metadata(meta)
    assert layout.references == ((64, 86), (68, 92))
    assert layout.text_image_spans == ((len(label), tokens[0] + 2), (2 * len(label) + tokens[0] + 2, tokens[1] + 2))
    assert layout.total_rows == len(ids) + 32 * 43 + 34 * 46 + 64 * 64


def test_requests_refused_at_the_seam():
    model = _model(max_packed_rows=5000)
    with pytest.raises(ValueError, match="max_packed_rows"):
        model.process_prompt("a cat", ["text"], ["image"], height=4096, width=4096)
    with pytest.raises(ValueError, match="min_generation_area"):
        model.process_prompt("a cat", ["text"], ["image"], min_generation_area=512)
    with pytest.raises(ValueError, match="together"):
        model.process_prompt("a cat", ["text"], ["image"], height=1024)
    with pytest.raises(ValueError, match="only generates images"):
        model.process_prompt("a cat", ["text"], ["video"])
    with pytest.raises(ValueError, match="32 pixels"):
        model.process_prompt("a cat", ["text", "image"], ["image"],
                             tensors={"image_inputs": [torch.zeros(3, 16, 64, dtype=torch.uint8)]})


def test_postprocess_resizes_the_canvas_to_the_request():
    from PIL import Image

    canvas = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    png = _model().postprocess(canvas, "image", {"height": 32, "width": 48, "min_generation_area": 1024})
    image = Image.open(io.BytesIO(png))
    want = Image.fromarray(canvas.permute(1, 2, 0).numpy()).resize((48, 32), Image.Resampling.LANCZOS)
    assert image.size == (48, 32) and np.array_equal(np.asarray(image.convert("RGB")), np.asarray(want))


def test_transformer_loads_the_step_heads(tmp_path):
    state = {k: v for k, v in random_checkpoint(CFG).items() if "audio" not in k}
    (tmp_path / "transformer").mkdir()
    save_file(state, str(tmp_path / "transformer" / "model.safetensors"))
    heads = {"video_out.weight": torch.randn(3, CFG.patch_dim, CFG.hidden_size),
             "video_out.bias": torch.randn(3, CFG.patch_dim)}
    save_file(heads, str(tmp_path / "pdd_fused_heads.safetensors"))
    dit = weight_loader.build_transformer(TINY, tmp_path, "cpu", dtype=torch.float32)
    assert torch.equal(dit.final_layer.video_out.weight, heads["video_out.weight"])
    # the transformer's own single head is not what serves
    assert not hasattr(dit.final_layer.video_out, "in_features")
    save_file({"video_out.weight": heads["video_out.weight"]}, str(tmp_path / "pdd_fused_heads.safetensors"))
    with pytest.raises(RuntimeError, match="unloaded"):
        weight_loader.build_transformer(TINY, tmp_path, "cpu", dtype=torch.float32)


def _tiny_decode_case():
    """A tiny image VAE with 64-px tiles and a batch of two requests' rows: a 10x6 latent grid
    makes 3 x 2 tiles."""
    from mstar.model.minimax_h3.config import H3VideoVAEConfig
    from mstar.model.minimax_h3_image.components.image_vae import H3ImageDecoder

    torch.manual_seed(0)
    vcfg = H3VideoVAEConfig(latent_channels=4, decoder_layers=1, decoder_heads=2, decoder_head_dim=32,
                            decoder_rope_dim_ratio=0.5, decoder_2d=True, tile_size=64, tile_overlap_min=16,
                            latents_mean=(0.1,) * 4, latents_std=(0.9,) * 4)
    vae = H3ImageDecoder(vcfg).eval()
    with torch.no_grad():
        for param in vae.parameters():
            param.normal_(0, 0.2)
    layout = H3ImageLayout(text_len=3, latent_height=10, latent_width=6, references=((2, 2),))
    rows = torch.randn(2, layout.num_condition_rows + layout.num_video_rows, 4 * 2 * 2)
    return vae, H3ImageConfig(vae=vcfg), layout, rows


def _sp_decode(rank: int, world: int, store: str, out: str) -> None:
    import torch.distributed as dist

    from mstar.distributed.communication import CommGroup
    from mstar.model.minimax_h3.config import VAE_SHARD
    from mstar.model.minimax_h3.tests.test_model import _plan, _sp_resources
    from mstar.model.minimax_h3_image.submodules import IMAGE_OUTPUT, H3ImageDecoderSubmodule

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        torch.set_num_threads(1)
        group = CommGroup(rank, rank, list(range(world)))
        group.device_group, group.initialized = dist.group.WORLD, True
        vae, config, layout, rows = _tiny_decode_case()
        calls = []
        decode = vae.decoder.forward
        vae.decoder.forward = lambda *args, **kwargs: calls.append(1) or decode(*args, **kwargs)
        with torch.no_grad():
            split = H3ImageDecoderSubmodule(vae, config)
            model = _model()
            model.set_config(config)
            _plan(split, _sp_resources(model, [VAE_SHARD], group), ["r0", "r1"], layout)
            image = split.run_batch(rows, layout)[IMAGE_OUTPUT]
            torch.save(len(calls), f"{out}.{rank}.calls")
            if rank == 0:  # at the same thread count: CPU reductions round by it
                torch.save(H3ImageDecoderSubmodule(vae, config).run_batch(rows, layout)[IMAGE_OUTPUT], f"{out}.one")
        torch.save(image, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_sequence_parallel_decode_gives_every_rank_the_one_nodes_image(tmp_path, world):
    """``vae_decoder`` on SP ranks, its sequence shard splitting each request's tiles: each
    rank's share, gathered, against one decode; 4 ranks over 6 tiles take 2, 2, 1 and 1."""
    import torch.multiprocessing as mp

    out = str(tmp_path / "image")
    mp.spawn(_sp_decode, args=(world, str(tmp_path / "store"), out), nprocs=world)
    want = torch.load(f"{out}.one")
    assert want.shape == (2, 3, 160, 96)
    for rank in range(world):
        assert torch.equal(torch.load(f"{out}.{rank}"), want)
    calls = [torch.load(f"{out}.{rank}.calls") for rank in range(world)]
    assert calls == ([3, 3] if world == 2 else [2, 2, 1, 1]), calls  # one decoder call per tile
