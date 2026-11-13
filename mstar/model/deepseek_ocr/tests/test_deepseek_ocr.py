"""DeepSeek-OCR on CPU, no weights: image views and their placeholder layout,
prompt rendering, the decoder's weight loading, and the graph and resources."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import Model
from mstar.model.deepseek_ocr.components.deepencoder import DeepEncoder
from mstar.model.deepseek_ocr.components.language_model import DeepseekOCRLanguageModel
from mstar.model.deepseek_ocr.config import MODES, DeepseekOCRTextConfig, ImageMode
from mstar.model.deepseek_ocr.deepseek_ocr_model import DeepseekOCRModel
from mstar.model.deepseek_ocr.preprocessing import image_views, num_image_tokens

GUNDAM = MODES["gundam"]

# DeepSeek-OCR's config.json, less the fields the port doesn't read
FIXTURE = Path(__file__).parent / "fixtures" / "config.json"


def _checkpoint_dir(tmp_path, **language):
    config = json.loads(FIXTURE.read_text())
    config["language_config"].update(language)
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


class _CharTokenizer:
    """One id per character, enough for the prompt's encode; ``<image>`` is 7."""

    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        ids = []
        for i, piece in enumerate(text.split("<image>")):
            ids += [7] * (i > 0) + [ord(c) % 1000 + 10 for c in piece]
        return ids

    def convert_tokens_to_ids(self, token):
        return 7


@pytest.fixture(autouse=True)
def _no_tokenizer_files(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "from_pretrained",
                        classmethod(lambda cls, path: _CharTokenizer()))


@pytest.mark.parametrize("size,max_crops,grid", [
    ((1596, 2240), 9, (2, 3)),   # portrait page
    ((2240, 1596), 9, (3, 2)),   # landscape page
    ((3000, 300), 9, (9, 1)),    # a strip takes every tile it may
    ((3000, 300), 6, (6, 1)),
])
def test_gundam_tiles_follow_the_aspect_ratio(size, max_crops, grid):
    mode = ImageMode(max_crops=max_crops)
    views = image_views(Image.new("RGB", size, (255, 255, 255)), mode)
    assert (views.cols, views.rows) == grid
    assert views.global_view.shape == (3, 1024, 1024) and views.global_view.dtype == torch.bfloat16
    assert views.crops.shape == (grid[0] * grid[1], 3, 640, 640)


def test_small_image_gets_only_the_global_view():
    views = image_views(Image.new("RGB", (640, 480), (0, 0, 0)), GUNDAM)
    assert views.crops is None and (views.cols, views.rows) == (1, 1)
    # padded with the normalization mean (127 of 255), letterboxed top and bottom
    pad = torch.tensor(127 / 255 * 2 - 1).to(torch.bfloat16)
    assert (views.global_view[:, 0, 0] == pad).all()
    assert views.global_view[:, 512, 512].float().allclose(torch.tensor(-1.0))


@pytest.mark.parametrize("name,expected", [
    # 16 rows of 16 + a newline, the separator, then 30 rows of 20 + a newline
    ("gundam", 16 * 17 + 1 + 30 * 21),
    ("tiny", 8 * 9 + 1),
    ("small", 10 * 11 + 1),
    ("base", 16 * 17 + 1),
    ("large", 20 * 21 + 1),
])
def test_placeholders_per_image(name, expected):
    assert num_image_tokens(MODES[name], 2, 3) == expected


def test_encoder_rows_are_local_then_global_then_separator():
    """Tile (r, c)'s token (i, j) sits at row r*10+i, column c*10+j of the local
    grid, each grid row ends in a newline, and the global view follows."""
    dim, cols, rows, tile = 4, 2, 3, 10
    with torch.device("meta"):
        encoder = DeepEncoder(2048, dim)
    encoder.image_newline = torch.nn.Parameter(torch.full((dim,), -1.0))
    encoder.view_seperator = torch.nn.Parameter(torch.full((dim,), -2.0))
    local = torch.arange(rows * cols * tile * tile, dtype=torch.float).view(rows * cols, tile * tile, 1)
    local = local.expand(-1, -1, dim).contiguous()
    global_tokens = torch.full((16 * 16, dim), 7.0)
    out = encoder.assemble(global_tokens, local, cols, rows)
    assert out.shape == (num_image_tokens(GUNDAM, cols, rows), dim)
    width = cols * tile + 1
    r, c, i, j = 2, 1, 3, 4
    assert out[(r * tile + i) * width + c * tile + j, 0] == local[r * cols + c, i * tile + j, 0]
    assert (out[width - 1 : rows * tile * width : width] == -1).all()
    assert (out[rows * tile * width : -1].view(16, 17, dim)[:, :16] == 7).all()
    assert (out[-1] == -2).all()


def test_encoder_captures_one_region_per_view_size():
    """Gundam cuts 1024 px global views and 640 px tiles; each size is its own
    captured region, its static views shaped like the views it replays."""
    from mstar.engine.accelerator_graph_config import PiecewiseCaptureShape
    from mstar.model.deepseek_ocr.submodules import DeepseekOCRVisionSubmodule

    with torch.device("meta"):
        encoder = DeepEncoder(2048, 1280)
    vision = DeepseekOCRVisionSubmodule(encoder, "vision_embeds", view_sizes=(GUNDAM.global_size, GUNDAM.image_size))
    configs = vision.get_piecewise_accelerator_graph_configs(torch.device("meta"), torch.bfloat16)
    assert set(configs) == {"views_640", "views_1024"}
    shape = PiecewiseCaptureShape(bs=3, seq_lens=[100] * 3, total_tokens=300)
    assert configs["views_640"].make_static_inputs(shape)["views"].shape == (3, 3, 640, 640)
    assert configs["views_640"].seq_len == GUNDAM.side_tokens(640) ** 2


def test_prompt_renders_images_before_the_text():
    from mstar.model.multimodal import PromptPart

    render = DeepseekOCRModel.render
    image, text = PromptPart("image"), PromptPart("text", text="Free OCR. ")
    # a chat may send the instruction first; the model reads the page first
    assert render([image, text], 1) == render([text, image], 1) == "<image>\nFree OCR."
    placed = PromptPart("text", text="Read <image> and <image>")
    assert render([image, image, placed], 2) == "Read <image> and <image>"


def _tiny_text_config(**overrides) -> DeepseekOCRTextConfig:
    values = dict(
        vocab_size=32, hidden_size=16, intermediate_size=24, moe_intermediate_size=8, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, n_routed_experts=4, n_shared_experts=2,
        num_experts_per_tok=2, first_k_dense_replace=1, moe_layer_freq=1, norm_topk_prob=False,
        routed_scaling_factor=1.0, max_position_embeddings=64, rope_theta=1e4, rms_norm_eps=1e-6,
        bos_token_id=0, eos_token_id=1, sliding_window=None,
    )
    return DeepseekOCRTextConfig(**{**values, **overrides})


def _hf_weights(config: DeepseekOCRTextConfig, skip: str | None = None):
    h, inter, moe = config.hidden_size, config.intermediate_size, config.moe_intermediate_size
    shapes = {"embed_tokens.weight": (config.vocab_size, h), "norm.weight": (h,),
              "lm_head.weight": (config.vocab_size, h)}
    for layer in range(config.num_hidden_layers):
        p = f"layers.{layer}."
        shapes |= {p + "input_layernorm.weight": (h,), p + "post_attention_layernorm.weight": (h,)}
        shapes |= {p + f"self_attn.{n}_proj.weight": (h, h) for n in "qkvo"}
        if not config.is_moe_layer(layer):
            shapes |= {p + "mlp.gate_proj.weight": (inter, h), p + "mlp.up_proj.weight": (inter, h),
                       p + "mlp.down_proj.weight": (h, inter)}
            continue
        shared = config.shared_intermediate_size
        shapes |= {p + "mlp.gate.weight": (config.n_routed_experts, h),
                   p + "mlp.shared_experts.gate_proj.weight": (shared, h),
                   p + "mlp.shared_experts.up_proj.weight": (shared, h),
                   p + "mlp.shared_experts.down_proj.weight": (h, shared)}
        for e in range(config.n_routed_experts):
            expert = p + f"mlp.experts.{e}."
            shapes |= {expert + "gate_proj.weight": (moe, h), expert + "up_proj.weight": (moe, h),
                       expert + "down_proj.weight": (h, moe)}
    gen = torch.Generator().manual_seed(0)
    return [(name, torch.randn(shape, generator=gen)) for name, shape in shapes.items() if name != skip]


def test_decoder_loads_checkpoint_experts_into_fused_tensors():
    config = _tiny_text_config()
    model = DeepseekOCRLanguageModel(config)
    weights = _hf_weights(config)
    model.load_weights(weights)
    source = dict(weights)
    fused = model.layers[1].mlp.experts
    moe = config.moe_intermediate_size
    assert torch.equal(fused.gate_up_proj[3, :moe], source["layers.1.mlp.experts.3.gate_proj.weight"])
    assert torch.equal(fused.gate_up_proj[3, moe:], source["layers.1.mlp.experts.3.up_proj.weight"])
    assert torch.equal(fused.down_proj[2], source["layers.1.mlp.experts.2.down_proj.weight"])


def test_decoder_refuses_a_checkpoint_missing_one_expert():
    config = _tiny_text_config()
    with pytest.raises(RuntimeError, match="unloaded"):
        DeepseekOCRLanguageModel(config).load_weights(_hf_weights(config, skip="layers.1.mlp.experts.2.up_proj.weight"))


def test_graph_and_resources(tmp_path):
    model = DeepseekOCRModel(str(_checkpoint_dir(tmp_path)))
    assert isinstance(model, Model)
    assert set(model.get_graph_walk_graphs()) == {"prefill_text", "prefill_vision", "decode"}
    resources = {spec.resource_key: spec for spec in model.get_node_resources()}
    assert set(resources) == {"kv_cache", "attn", "rope", "sampler"}
    assert all(spec.nodes == {"LLM"} for spec in resources.values())
    kv = resources["kv_cache"].config
    assert (kv.num_layers, kv.num_kv_heads, kv.head_dim, kv.max_seq_len) == (12, 10, 128, 8192)
    assert model.get_loop_caps(max_output_tokens=100) == {"decode_loop": 99}
    # decode's one signal loops back and reaches the client off the host copy
    # (DeviceLoopback); a second signal would send every row's token through Python
    decode = model.get_graph_walk_graphs()["decode"].section
    assert {(e.next_node, e.name) for e in decode.outputs} == {("LLM", "text_inputs"), (EMIT_TO_CLIENT, "text_inputs")}


def test_image_views_reach_the_encoder_with_their_grids(tmp_path):
    """A page (tiled 2x3) and a small image (its global view only): the
    prefill walk carries both views to the encoder, which reads each image's
    grid off the walk's media layout, and the prompt holds one pad per row
    the encoder will send."""
    from types import SimpleNamespace

    from mstar.model.deepseek_ocr.submodules import DeepseekOCRVisionSubmodule
    from mstar.model.multimodal import MEDIA_LAYOUT

    model = DeepseekOCRModel(str(_checkpoint_dir(tmp_path)), max_crops=6)
    page = torch.full((3, 2240, 1596), 255, dtype=torch.uint8)
    small = torch.zeros((3, 300, 400), dtype=torch.uint8)
    modalities = ["image", "image", "text"]
    out = model.process_prompt("Free OCR.", modalities, ["text"], tensors={"image_inputs": [page, small]})
    signals = {k: v for k, v in out.new_input_tensors.items() if v}
    assert set(signals) == {"text_inputs", "global_views", "tiles"}
    ids = signals["text_inputs"][0]
    assert int((ids == 7).sum()) == num_image_tokens(GUNDAM, 2, 3) + num_image_tokens(GUNDAM, 1, 1)

    prefill, decode = model.walk_plan("default", modalities, ["text"], signals, out.metadata)
    assert (prefill.walk, decode.walk) == ("prefill_vision", "decode")
    encoder = model.encoder("vision_encoder")
    assert set(encoder.inputs) == {"global_views", "tiles"}
    with torch.device("meta"):
        vision = DeepseekOCRVisionSubmodule(DeepEncoder(2048, 1280), encoder.output, view_sizes=(1024, 640))
    fwd = SimpleNamespace(step_metadata={MEDIA_LAYOUT: prefill.step_metadata[MEDIA_LAYOUT]})
    node = vision.prepare_inputs("prefill_vision", fwd, {name: signals[name] for name in encoder.inputs})
    assert node.kwargs["grid"] == [(2, 3), (1, 1)]
    assert node.tensor_inputs["global_views"].shape == (2, 3, 1024, 1024)
    assert node.tensor_inputs["tiles"].shape == (6, 3, 640, 640)

    text_only, _ = model.walk_plan("default", ["text"], ["text"], {"text_inputs": [ids]}, {})
    assert text_only.walk == "prefill_text"


@pytest.mark.parametrize("image_mode,max_crops", [
    ("base", 6),     # a one-view mode cuts no tiles to cap
    ("gundam", 6.0),  # range() over a float fails on every request, not at load
])
def test_max_crops_refused_at_load_where_it_cannot_apply(tmp_path, image_mode, max_crops):
    with pytest.raises(ValueError, match="max_crops"):
        DeepseekOCRModel(str(_checkpoint_dir(tmp_path)), image_mode=image_mode, max_crops=max_crops)


@pytest.mark.parametrize("prompt", ["", "  \n"])
def test_request_with_neither_image_nor_text_is_refused(tmp_path, prompt):
    model = DeepseekOCRModel(str(_checkpoint_dir(tmp_path)))
    with pytest.raises(ValueError, match="needs an image or text"):
        model.process_prompt(prompt, ["text"], ["text"])


def test_min_p_reaches_the_samplers_refusal(tmp_path):
    # Qwen3_5Adapter documents min_p as refused; dropped here, a request would run without it
    model = DeepseekOCRModel(str(_checkpoint_dir(tmp_path)))
    (spec,) = [s for s in model.get_node_resources() if s.resource_key == "sampler"]
    config = model.get_request_resource_configs(["text"], ["text"], {"min_p": 0.1})["sampler"]
    with pytest.raises(ValueError, match="min_p"):
        config.validate(spec)


def test_unlimited_ocr_window_is_refused(tmp_path):
    """R-SWA keeps the prompt visible beside the window, which no attention resource expresses yet."""
    with pytest.raises(NotImplementedError, match="sliding-window"):
        DeepseekOCRModel(str(_checkpoint_dir(tmp_path, sliding_window_size=128)))


@pytest.mark.parametrize("max_crops", [6, 9])
def test_a_step_of_packed_page_prefills_fits_a_captured_graph(tmp_path, max_crops):
    """As many pages as a prefill_image step packs fit the top capture bucket,
    however the pages tile, so the step replays rather than running eagerly."""
    from mstar.model.deepseek_ocr.submodules import DeepseekOCRLLMSubmodule

    model = DeepseekOCRModel(str(_checkpoint_dir(tmp_path)), max_crops=max_crops)
    config = _tiny_text_config()
    with torch.device("meta"):
        llm = DeepseekOCRLLMSubmodule(
            DeepseekOCRLanguageModel(config), config, model.encoders, max_prompt_tokens=model.max_prompt_tokens(),
        )
    configs = llm.get_accelerator_graph_configs(torch.device("meta"))
    (packed,) = [c for c in configs if c.capture_graph_walk == "prefill_vision"]
    largest = max(num_image_tokens(model.image_mode, cols, rows)
                  for cols in range(1, max_crops + 1) for rows in range(1, max_crops + 1)
                  if 2 <= cols * rows <= max_crops)
    pages = llm.max_batch_size("prefill_vision")
    # "<image>\nFree OCR." and the BOS are under 16 tokens
    assert 1 < pages <= max(packed.capture_batch_sizes)
    assert pages * (largest + 16) <= max(packed.get_total_tokens(pages))


def test_concurrent_prompts_match_one_at_a_time(tmp_path):
    """The model declares PREPROCESS_PARALLEL, so the API server prepares
    pages on a thread pool: each request's ids and views are what a serial
    call gives."""
    from concurrent.futures import ThreadPoolExecutor

    model = DeepseekOCRModel(str(_checkpoint_dir(tmp_path)), max_crops=6)
    assert model.PREPROCESS_PARALLEL
    gen = torch.Generator().manual_seed(0)
    images = [torch.randint(0, 256, (3, h, w), dtype=torch.uint8, generator=gen)
              for h, w in ((900, 700), (700, 1300), (500, 400), (1500, 1000))] * 2

    def prepare(image):
        return model.process_prompt("Free OCR.", ["image", "text"], ["text"], tensors={"image_inputs": [image]})

    serial = [prepare(image) for image in images]
    with ThreadPoolExecutor(4) as pool:
        parallel = list(pool.map(prepare, images))
    for (a, meta_a), (b, meta_b) in zip(serial, parallel, strict=True):
        assert torch.equal(a["text_inputs"][0], b["text_inputs"][0])
        for name in ("global_views", "tiles"):
            assert len(a.get(name, [])) == len(b.get(name, []))
            assert all(torch.equal(x, y) for x, y in zip(a.get(name, []), b.get(name, []), strict=True))
        assert meta_a == meta_b
