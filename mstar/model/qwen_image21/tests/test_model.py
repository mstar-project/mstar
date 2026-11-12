"""Qwen-Image-2.1's intake and plan on CPU, with a stand-in Qwen3-VL processor: the prompt's spans,
their KV layout and prefix keys, the walk plan built from them, and the refusals (each a 400). Then
the dit node over a tiny random DiT: a cache hit and a batch land where a request alone does."""

from __future__ import annotations

import re

import pytest
import torch

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.model.qwen_image21.config import (
    KV,
    MAIN,
    MAX_REFERENCE_IMAGES,
    SYSTEM_PROMPT,
    UNCOND,
    QwenImage21Config,
)
from mstar.model.qwen_image21.layout import PREFIX_IDS, RequestLayout
from mstar.model.qwen_image21.qwen_image21_model import QwenImage21Model, calculate_dimensions


@pytest.fixture(autouse=True)
def _threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # bit-exact CPU comparisons
    yield
    torch.set_num_threads(threads)


SPECIAL = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|image_pad|>": 151655, "<|vision_start|>": 151652,
           "<|vision_end|>": 151653}


class _Tokenizer:
    """Special tokens whole, every other character its own id."""

    def __call__(self, text: str) -> dict:
        ids = []
        for piece in re.split(r"(<\|[a-z_]+\|>)", text):
            ids += [SPECIAL[piece]] if piece in SPECIAL else [ord(c) + 1000 for c in piece]
        return {"input_ids": ids}


class _ImageProcessor:
    """Qwen3-VL's grid for an image already on the 32-px grid within its pixel bounds."""

    def __call__(self, images, return_tensors="pt") -> dict:
        grids = [(1, image.size[1] // 16, image.size[0] // 16) for image in images]
        return {"pixel_values": torch.zeros(sum(h * w for _, h, w in grids), 1536),
                "image_grid_thw": torch.tensor(grids)}


class _Processor:
    tokenizer = _Tokenizer()
    image_processor = _ImageProcessor()


def _model(**config) -> QwenImage21Model:
    model = QwenImage21Model(skip_weight_loading=True)
    model.set_config(QwenImage21Config(**config))
    model._processor = _Processor()
    model._drop_idx = len(_Tokenizer()(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")["input_ids"])
    return model


def _image(width: int, height: int) -> torch.Tensor:
    return torch.randint(0, 256, (4, height, width), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))


def test_an_edit_under_cfg_lays_out_its_spans_keys_and_plan():
    model = _model()
    images = [_image(1200, 800), _image(600, 900)]
    kwargs = {"guidance_scale": 4.0, "negative_prompt": "blurry", "output_resolution": 512}
    out = model.process_prompt("make it night", ["text", "image"], ["image"], {"image_inputs": images}, **kwargs)
    tensors, meta = out.new_input_tensors, out.metadata
    layout = RequestLayout.from_metadata(meta["layout"])

    grids = [calculate_dimensions(512 * 512, 1200 / 800), calculate_dimensions(512 * 512, 600 / 900)]
    assert [tuple(t.shape) for t in tensors["ref_pixels"]] == [(4, h, w) for w, h in grids]
    assert [span.is_text for span in layout.spans] == [True, False, True, False, True]
    assert [(s.height, s.width) for s in layout.spans if not s.is_text] == [(h // 16, w // 16) for w, h in grids]
    # positions: text advances by its length, an image by max(h, w) of its latent grid
    for before, after in zip(layout.spans[:-1], layout.spans[1:], strict=True):
        assert after.position == before.position + before.advance
    # the output takes the last reference's aspect ratio
    assert layout.target == tuple(d // 16 for d in reversed(calculate_dimensions(512 * 512, 600 / 900)))
    lengths = meta["prompt_lengths"]
    assert sum(lengths) == tensors["text_inputs"][0].numel() and len(lengths) == 2
    assert layout.frames()[0] != layout.frames()[1]  # the negative tail is shorter

    kv = meta["kv_layout"][KV]
    assert kv[MAIN].keyed_count() == 4 and kv[MAIN].scratch == layout.target[0] * layout.target[1]
    assert kv[UNCOND].spans[:4] == kv[MAIN].spans[:4] and kv[UNCOND].spans[4].length == layout.uncond_tail[1]
    # the preprocess worker can key the spans through the last image from the tensors returned
    keys = model.kv_labels()[KV][MAIN].keys
    pages, _ = PreprocessWorkerThread._page_layout(
        None, keys, kv[MAIN].spans[:4], [None, b"a", None, b"b"], {PREFIX_IDS: tensors[PREFIX_IDS]}, 16,
    )
    assert sum(map(len, pages)) == sum(span.length for span in layout.spans[:4] if span.is_text)

    plan = model.walk_plan("default", ["text", "image"], ["image"], {}, {**kwargs, **meta})
    assert [step.walk for step in plan] == [
        "encode_images", "prefix_text", "prefix_image", "prefix_text", "prefix_image", "prefix_text", "image_gen",
    ]
    assert [step.step_metadata.get("span") for step in plan[1:-1]] == [0, 1, 2, 3, 4]
    assert plan[-1].step_metadata["guidance_scale"] == 4.0
    assert model.get_request_resource_configs(["text", "image"], ["image"], kwargs)[KV].needed_labels == [MAIN, UNCOND]
    assert model.get_loop_caps(**kwargs) == {"denoise_loop": 40}


def test_text_to_image_is_one_span_and_keys_nothing():
    model = _model()
    # a scale without a negative prompt runs no CFG: the dit's guidance stays off
    kwargs = {"height": 640, "width": 1000, "guidance_scale": 4.0}
    tensors, meta, _ = model.process_prompt("a cat", ["text"], ["image"], {}, **kwargs)
    layout = RequestLayout.from_metadata(meta["layout"])
    assert len(layout.spans) == 1 and layout.target == (640 // 16, 992 // 16)
    assert meta["kv_layout"][KV][MAIN].keyed_count() == 0 and not tensors[PREFIX_IDS]
    plan = model.walk_plan("default", ["text"], ["image"], {}, {**kwargs, **meta})
    assert [step.walk for step in plan] == ["encode", "prefix_text", "image_gen"]
    assert plan[-1].step_metadata["guidance_scale"] == 1.0
    assert model.get_request_resource_configs(["text"], ["image"], kwargs)[KV].needed_labels == [MAIN]


@pytest.mark.parametrize("kwargs, images, match", [
    ({"true_cfg_scale": 4.0}, 0, "guidance_scale"),
    ({"guidance_scale": float("inf"), "negative_prompt": "x"}, 0, "finite"),
    ({"output_format": "jpeg"}, 0, "RGBA"),
    ({"num_inference_steps": 1}, 0, "num_inference_steps"),
    # a reference resized under Qwen3-VL's min_pixels would be resized again by its processor
    ({"output_resolution": 128}, 1, "min_pixels|outside"),
    ({"negative_prompt": "<|image_pad|>", "guidance_scale": 2.0}, 0, "vision tokens"),
    # ids_for turns a typed placeholder into an image pad; the negative prompt's tail is never laid out
    ({"negative_prompt": "x<|placeholder|>", "guidance_scale": 2.0}, 1, "placeholder"),
    # a side past 2048 latents would wrap onto the rope tables' positive rows (or index past them)
    ({"height": 32, "width": 33024}, 0, "px per side"),
])
def test_refusals(kwargs, images, match):
    with pytest.raises(ValueError, match=match):
        _model().process_prompt("p", ["text"] + ["image"] * bool(images), ["image"],
                                {"image_inputs": [_image(64, 64)] * images}, **kwargs)


def test_a_negative_prompt_past_max_seq_len_is_refused():
    """``uncond`` holds its own tail: a long negative prompt can pass ``max_seq_len`` while ``main`` fits."""
    model = _model()
    model.max_seq_len = 512
    kwargs = {"height": 256, "width": 256, "guidance_scale": 4.0}
    model.process_prompt("p", ["text"], ["image"], {}, **kwargs, negative_prompt="n" * 100)
    with pytest.raises(ValueError, match="'uncond' K/V"):
        model.process_prompt("p", ["text"], ["image"], {}, **kwargs, negative_prompt="n" * 400)


def test_more_references_than_it_takes_are_refused_before_decoding():
    from mstar.model.media_limits import check_file_counts

    with pytest.raises(ValueError, match=f"at most {MAX_REFERENCE_IMAGES}"):
        check_file_counts({"image": MAX_REFERENCE_IMAGES + 1}, _model().max_input_files())


def test_turbo_runs_only_its_fixed_grid():
    model = _model(sample_sigmas=(1.0, 0.9, 0.5), default_num_inference_steps=3)
    assert model.get_loop_caps() == {"denoise_loop": 3}
    with pytest.raises(ValueError, match="exactly 3"):
        model.process_prompt("p", ["text"], ["image"], {}, num_inference_steps=40)


def test_vae_pixels_are_the_image_processors_to_the_bit_and_layout():
    """The VAE's latents depend on its input's exact values and memory layout (cuDNN kernels)."""
    image_processor = pytest.importorskip("diffusers.image_processor")
    from PIL import Image

    from mstar.model.components.diffusion.image_io import unit_pixels, vae_pixels

    rgba = _image(96, 64)
    pil = Image.fromarray(rgba.permute(1, 2, 0).numpy(), mode="RGBA")
    ref = image_processor.VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64).preprocess(
        pil, height=64, width=96).unsqueeze(2)
    ours = vae_pixels(unit_pixels(rgba))
    assert torch.equal(ours, ref) and ours.stride() == ref.stride()


def test_the_cache_free_vae_calls_are_diffusers_encode_and_decode_to_the_bit():
    """The nodes call the VAE's encoder and decoder without the one-frame feature cache diffusers passes
    (written, never read, for one frame); a diffusers change that reads it would break parity here, as one
    to the causal conv's padding would break the padded convs `compile` swaps in."""
    vae_module = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_qwenimage21")
    from mstar.model.qwen_image21.submodules import (
        QwenImage21VaeDecoderSubmodule,
        _decode,
        _encode_mode,
        _padded_conv_class,
    )

    torch.manual_seed(0)
    vae = vae_module.AutoencoderKLQwenImage21(
        base_dim=8, decoder_base_dim=12, z_dim=4, dim_mult=[1, 2, 2, 2, 2], num_res_blocks=1,
        latents_mean=[0.0] * 4, latents_std=[1.0] * 4,
    ).eval()
    pixels = torch.rand(1, 4, 1, 48, 32) * 2 - 1
    with torch.no_grad():
        mode = vae.encode(pixels).latent_dist.mode()
        decoded = vae.decode(mode, return_dict=False)[0]
        assert torch.equal(_encode_mode(vae, pixels), mode)
        assert torch.equal(_decode(vae, mode), decoded)
        # under `compile` the decode path's convs pad inside the convolution: the same function. The encoder,
        # which shares the VAE, keeps the pipeline's convs and bits
        QwenImage21VaeDecoderSubmodule(vae, compile=True)
        assert not any(isinstance(m, _padded_conv_class()) for m in vae.encoder.modules())
        assert any(isinstance(m, _padded_conv_class()) for m in vae.decoder.modules())
        assert torch.equal(_encode_mode(vae, pixels), mode)
        torch.testing.assert_close(_decode(vae, mode), decoded)


def test_the_compiled_vae_decoder_serves_other_sizes_without_recompiling():
    """The decoder warmed on its ``compile_regions`` example (the engine compiles it with dynamic=True) must take
    a live request's latents as they come: a recompile there stalls that request for minutes."""
    vae_module = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_qwenimage21")
    from mstar.model.qwen_image21.submodules import QwenImage21VaeDecoderSubmodule

    torch.manual_seed(0)
    vae = vae_module.AutoencoderKLQwenImage21(  # the checkpoint's 64 latent channels and RGBA pixels, one level
        base_dim=4, decoder_base_dim=4, z_dim=64, in_channels=4, out_channels=4, dim_mult=[1, 1],
        temperal_downsample=[False], num_res_blocks=1, latents_mean=[0.0] * 64, latents_std=[1.0] * 64,
    ).eval()
    decoder = QwenImage21VaeDecoderSubmodule(vae, True)
    try:
        for path, examples in decoder.compile_regions().items():
            module = decoder.get_submodule(path)
            module.forward = torch.compile(module.forward, dynamic=True, backend="eager")
            with torch.no_grad():
                module.forward(*examples[0])
        with torch._dynamo.config.patch(error_on_recompile=True):
            for h, w in ((52, 78), (64, 64)):
                decoder.forward(None, None, latents=torch.randn(h * w, 64), target=(h, w))
    finally:
        # a failure here must not leave the compiled frames to the next test
        torch._dynamo.reset()


@pytest.mark.parametrize("layout", ["preprocessor_config", "processor_config"])
def test_config_reads_the_pixel_bounds_from_either_processor_layout(tmp_path, layout):
    """The base repo saves the image processor alone, the Turbo repo inside its whole processor."""
    import json

    def write(path, data):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(json.dumps(data))

    text = {"model_type": "qwen3_vl_text", "vocab_size": 8, "hidden_size": 8, "intermediate_size": 8,
            "num_hidden_layers": 2, "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 8,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True}}
    vision = {"depth": 1, "hidden_size": 8, "intermediate_size": 8, "num_heads": 1, "in_channels": 3, "patch_size": 16,
              "temporal_patch_size": 2, "spatial_merge_size": 2, "num_position_embeddings": 4, "out_hidden_size": 8}
    write("text_encoder/config.json", {"text_config": text, "vision_config": vision, "image_token_id": 5})
    write("transformer/config.json", {"num_layers": 1})
    write("model_index.json", {"sample_sigmas": [1.0, 0.5]})
    write("scheduler/scheduler_config.json", {"shift": 1.0, "use_dynamic_shifting": False})
    bounds = {"size": {"shortest_edge": 1234, "longest_edge": 99999}}
    if layout == "preprocessor_config":
        write("processor/preprocessor_config.json", bounds)
    else:
        write("processor/processor_config.json", {"image_processor": bounds})
    config = QwenImage21Config.from_snapshot(tmp_path)
    assert (config.min_pixels, config.max_pixels) == (1234, 99999) and config.sample_sigmas == (1.0, 0.5)


@pytest.mark.parametrize("accelerator_graph", [False, True])
def test_captured_denoise_steps_on_cfg_ranks_are_refused_at_load(accelerator_graph):
    """A cfg rank's guided step holds both labels' K/V while its capture is sized for its one branch."""
    from pathlib import Path

    import yaml

    from mstar.model.qwen_image21.qwen_image21_model import QwenImage21Model

    config = yaml.safe_load((Path(__file__).parents[4] / "configs" / "qwen_image21_cfg2.yaml").read_text())
    model = QwenImage21Model(skip_weight_loading=True, accelerator_graph=accelerator_graph)
    if accelerator_graph:
        with pytest.raises(ValueError, match="can't capture its denoise steps with the dit on cfg_size > 1"):
            model.validate_config_yaml(config, "cfg2.yaml")
    else:
        model.validate_config_yaml(config, "cfg2.yaml")


def test_a_guided_denoise_capture_holds_both_labels_kv():
    """A guided capture's template rows clone (as the engine clones them), and its bucket holds the K/V
    slots its step declares: the target on ``main`` and on ``uncond`` per row."""
    from types import SimpleNamespace

    from mstar.model.qwen_image21.submodules import DitKey, QwenImage21DenoiseSubmodule

    sub = QwenImage21DenoiseSubmodule(
        SimpleNamespace(dtype=torch.bfloat16), QwenImage21Config(), loop_name="denoise_loop",
        capture_buckets=[("image_gen", DitKey(8, 6))], capture_batch_sizes=[3],
    )
    configs = sub.get_accelerator_graph_configs(torch.device("cpu"))
    assert [c.additional_key_info.guided for c in configs] == [False, True]
    for config in configs:
        rows = config.get_node_inputs(3, 0)
        step = sub.declare_step("image_gen", ["a", "b", "c"], rows)
        assert config.get_total_tokens(3) == [sum(s.span for s in step.steps[KV].segments)]


@torch.no_grad()
def test_a_cache_hit_and_a_batch_land_where_a_request_alone_does():
    """The plan of a 2-reference edit under CFG through the dit node (prefix walks, the fork, the two-label
    tail and the two-label denoise). A second request whose first text span and first half of the first
    image come from the cache lands on the same latents, to the bit. Two requests at different seeds and
    scales denoised in one batch, whose rows the KV plan packs label-major, land where each lands alone."""
    from mstar.model.qwen_image21.components.transformer import QwenImage21Transformer
    from mstar.model.qwen_image21.tests.paged_stub import CFG, LAYOUT, denoise, dit_node, drive, info

    dtype = torch.bfloat16
    torch.manual_seed(0)
    dit = QwenImage21Transformer(CFG)
    for p in dit.parameters():
        p.data = (torch.randn_like(p) * 0.2).to(dtype)
    dit.set_prefix_modulation()
    sub, attend, plan, embeds, latents = dit_node(dit, dtype)
    alone = drive(sub, attend, "a", plan, embeds, latents)

    # the cache serves request b's first text span and the first 12 tokens of its first image: a's pages
    held = LAYOUT[0] + 12
    attend.cache[("b", "main")] = {layer: (k[:held], v[:held]) for layer, (k, v) in attend.cache[("a", "main")].items()}
    hit = drive(sub, attend, "b", plan, embeds, latents, cut={0: None, 1: 12})
    torch.testing.assert_close(hit, alone, rtol=0, atol=0)

    infos = lambda: {"a": info("a", plan[-1]), "b": info("b", plan[-1], seed=9, guidance_scale=2.0)}  # noqa: E731
    each = {rid: denoise(sub, attend, {rid: fwd})[rid] for rid, fwd in infos().items()}
    together = denoise(sub, attend, infos())
    assert not torch.equal(each["a"], each["b"])
    for rid in ("a", "b"):
        torch.testing.assert_close(together[rid], each[rid], rtol=0, atol=0)


class _Gathers:
    """One GPU's cfg group: records what each guided step gathers (both branches, every row)."""

    world_size, rank = 1, 0

    def __init__(self):
        self.gathered: list[torch.Tensor] = []

    def all_gather(self, x, dim):
        self.gathered.append(x)
        return x


class _CfgRank:
    """Rank ``rank`` of a two-rank cfg group, run alone: the peer's half of each gather is one GPU's."""

    world_size = 2

    def __init__(self, rank: int, gathered: list[torch.Tensor]):
        self.rank, self.gathered = rank, list(gathered)

    def all_gather(self, x, dim):
        full = self.gathered.pop(0)
        assert dim == 1 and torch.equal(x, full[:, self.rank:self.rank + 1]), f"rank {self.rank}'s branch moved"
        return full


@pytest.mark.parametrize("rank", [0, 1])
@torch.no_grad()
def test_a_cfg_rank_runs_its_branch_as_one_gpu_does(rank):
    """Under cfg_size 2 a rank runs every prefix walk and, in each denoise step, its own branch alone over its
    own label: that branch's prediction is one GPU's to the bit, and so are the latents after combining the
    gathered pair. Its KV step still holds both labels, as one GPU's does."""
    from mstar.model.qwen_image21.components.transformer import QwenImage21Transformer
    from mstar.model.qwen_image21.submodules import QwenImage21DenoiseSubmodule
    from mstar.model.qwen_image21.tests.paged_stub import CFG, DIT_CONFIG, PagedAttendStub, dit_node, drive

    dtype = torch.bfloat16
    torch.manual_seed(0)
    dit = QwenImage21Transformer(CFG)
    for p in dit.parameters():
        p.data = (torch.randn_like(p) * 0.2).to(dtype)
    dit.set_prefix_modulation()
    sub, attend, plan, embeds, latents = dit_node(dit, dtype)
    one_gpu = _Gathers()
    sub.cfg_group = one_gpu
    alone = drive(sub, attend, "a", plan, embeds, latents)

    ranked = QwenImage21DenoiseSubmodule(dit, DIT_CONFIG, loop_name="denoise_loop",
                                         cfg_group=_CfgRank(rank, one_gpu.gathered))
    stub = PagedAttendStub()
    ranked.attend = lambda *_: stub
    declared = []
    begin = stub.begin
    stub.begin = lambda step: (declared.append(step), begin(step))
    got = drive(ranked, stub, "a", plan, embeds, latents)

    torch.testing.assert_close(got, alone, rtol=0, atol=0)
    denoise = declared[-1]
    assert {s.label for s in denoise.steps[KV].segments} == {MAIN, UNCOND}
    assert {s.label for s in denoise.steps[ranked.target_attn_key()].segments} == {(MAIN, UNCOND)[rank]}
