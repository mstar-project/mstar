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
    vision = DeepseekOCRVisionSubmodule(encoder, view_sizes=(GUNDAM.global_size, GUNDAM.image_size))
    configs = vision.get_piecewise_accelerator_graph_configs(torch.device("meta"), torch.bfloat16)
    assert set(configs) == {"views_640", "views_1024"}
    shape = PiecewiseCaptureShape(bs=3, seq_lens=[100] * 3, total_tokens=300)
    assert configs["views_640"].make_static_inputs(shape)["views"].shape == (3, 3, 640, 640)
    assert configs["views_640"].seq_len == GUNDAM.side_tokens(640) ** 2


def test_prompt_renders_images_before_the_text():
    from mstar.model.multimodal import PromptPart

    render = DeepseekOCRModel._render
    image, text = PromptPart("image"), PromptPart("text", text="Free OCR.")
    # a chat may send the instruction first; the model reads the page first
    assert render(None, [image, text], 1) == render(None, [text, image], 1) == "<image>\nFree OCR."
    assert render("Free OCR. ", None, 1) == "<image>\nFree OCR."
    assert render("Read <image> and <image>", None, 2) == "Read <image> and <image>"


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
    assert set(model.get_graph_walk_graphs()) == {"prefill", "prefill_image", "decode"}
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


def test_image_prefill_sends_views_to_the_encoder_and_grids_as_step_metadata(tmp_path):
    model = DeepseekOCRModel(str(_checkpoint_dir(tmp_path)))
    signals = {"text_inputs": ["ids"], "image_views": ["global", "tiles"]}
    args = model.get_initial_forward_pass_args(
        "default", ["image", "text"], ["text"], signals, model_kwargs={"image_grids": [[2, 3]]},
    )
    assert args.full_metadata.graph_walk == "prefill_image"
    assert {(e.next_node, e.name) for e in args.inputs} == {("LLM", "text_inputs"), ("vision_encoder", "image_views")}
    assert args.step_metadata == {"step_image_grids": [[2, 3]]}
    text_only = model.get_initial_forward_pass_args("default", ["text"], ["text"], {"text_inputs": ["ids"]})
    assert text_only.full_metadata.graph_walk == "prefill" and not text_only.step_metadata


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
    config = model.get_request_resource_configs({}, {"min_p": 0.1})["sampler"]
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
            DeepseekOCRLanguageModel(config), config, image_token_id=5, stop_ids=frozenset({1}),
            max_prompt_tokens=model.max_prompt_tokens(),
        )
    (packed,) = [c for c in llm.get_accelerator_graph_configs(torch.device("meta"))
                 if c.capture_graph_walk == "prefill_image"]
    largest = max(num_image_tokens(model.image_mode, cols, rows)
                  for cols in range(1, max_crops + 1) for rows in range(1, max_crops + 1)
                  if 2 <= cols * rows <= max_crops)
    pages = llm.max_batch_size("prefill_image")
    # "<image>\nFree OCR." and the BOS are under 16 tokens
    assert 1 < pages <= max(packed.capture_batch_sizes)
    assert pages * (largest + 16) <= max(packed.get_total_tokens(pages))


class _CharTokenizer:
    """One id per character, enough for process_prompt's encode."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 1000 + 10 for c in text]

    def convert_tokens_to_ids(self, token):
        return 7


def test_concurrent_prompts_match_one_at_a_time(tmp_path):
    """The model declares PREPROCESS_PARALLEL, so the API server prepares
    pages on a thread pool: each request's ids and views are what a serial
    call gives."""
    from concurrent.futures import ThreadPoolExecutor

    model = DeepseekOCRModel(str(_checkpoint_dir(tmp_path)), max_crops=6)
    assert model.PREPROCESS_PARALLEL
    model._tokenizer = _CharTokenizer()
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
        assert len(a["image_views"]) == len(b["image_views"])
        assert all(torch.equal(x, y) for x, y in zip(a["image_views"], b["image_views"], strict=True))
        assert meta_a == meta_b


def test_concurrent_first_prompts_load_one_tokenizer(tmp_path, monkeypatch):
    """The prepare pool's first requests reach the lazy tokenizer together."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import transformers

    loads = []

    def slow_load(path):
        loads.append(threading.get_ident())
        time.sleep(0.05)
        return _CharTokenizer()

    monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "from_pretrained", staticmethod(slow_load))
    model = DeepseekOCRModel(str(_checkpoint_dir(tmp_path)))
    with ThreadPoolExecutor(8) as pool:
        tokenizers = list(pool.map(lambda _: model.tokenizer, range(8)))
    assert len(loads) == 1
    assert all(t is tokenizers[0] for t in tokenizers)
