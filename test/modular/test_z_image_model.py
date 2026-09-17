"""Dummy-mode tests for the Z-Image-Turbo model declaration (no weights, GPU or network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, ".")

from mstar.graph.base import GraphNode, Loop, Sequential, TensorPointerInfo  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT  # noqa: E402
from mstar.model.z_image.config import DENOISE_LOOP, DIT_ATTN, ZImageConfig  # noqa: E402
from mstar.model.z_image.submodules import ZShape, shape_from_metadata  # noqa: E402
from mstar.model.z_image.z_image_model import ENCODE_TEXT_WALK, IMAGE_GEN_WALK, ZImageModel  # noqa: E402

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs" / "z_image_turbo.yaml")


class _StubTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        assert enable_thinking and add_generation_prompt and not tokenize
        return f"<u>{messages[0]['content']}<a>"

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512, **kwargs):
        ids = list(text.encode("utf-8"))[:max_length]
        return {"input_ids": torch.tensor([ids])}


def _make_model(**kwargs) -> ZImageModel:
    model = ZImageModel(model_path_hf="test/z_image", skip_weight_loading=True, **kwargs)
    model.set_config(ZImageConfig())
    model.tokenizer = _StubTokenizer()
    return model


def _info(name, dims):
    return TensorPointerInfo(dims=dims, dtype="torch.int64", nbytes=8, address=0, stride=[1] * len(dims),
                             uuid=f"uuid-{name}", source_session_id="t:0", source_entity="t")


def test_graph_structure():
    model = _make_model()
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {ENCODE_TEXT_WALK, IMAGE_GEN_WALK}
    assert model.nodes == ["dit", "text_encoder", "vae_decoder"]
    text = walks[ENCODE_TEXT_WALK]
    assert isinstance(text, GraphNode) and text.input_names == {"text_inputs"} and text.outputs[0].persist
    gen = walks[IMAGE_GEN_WALK]
    assert isinstance(gen, Sequential)
    loop, decoder = gen.sections
    assert isinstance(loop, Loop) and loop.name == DENOISE_LOOP and loop.max_iters == model.config.max_denoise_steps
    assert loop.section.input_names == {"text_embeds", "latents"}
    assert [(e.name, e.next_node) for e in loop.section.outputs] == [("latents", "dit")]
    assert decoder.outputs[0].next_node == EMIT_TO_CLIENT and decoder.outputs[0].output_modality == "image"
    walks_seen = set()
    for wg in model.get_worker_graphs(CONFIG_PATH):
        walks_seen |= wg.graph_walks
    assert walks_seen == {ENCODE_TEXT_WALK, IMAGE_GEN_WALK}


def test_resources_and_dummy_mode():
    assert [s.resource_key for s in _make_model().get_node_resources()] == [DIT_ATTN]
    assert _make_model(attention_backend="sdpa").get_node_resources() == []
    model = _make_model()
    assert all(model.get_submodule(n) is None for n in model.nodes)
    assert model.get_autocast_dtype() is None


def test_process_prompt_and_validation():
    model = _make_model()
    ids = model.process_prompt("a cat", ["text"], ["image"])["text_inputs"][0]
    assert bytes(ids.tolist()) == b"<u>a cat<a>" and ids.dtype == torch.long
    with pytest.raises(ValueError, match="text prompt"):
        model.process_prompt(None, ["text"], ["image"])
    with pytest.raises(ValueError, match="text-to-image only"):
        model.process_prompt("x", ["image", "text"], ["image"])
    with pytest.raises(ValueError, match="multiple of 16"):
        model.process_prompt("x", ["text"], ["image"], height=1000)
    with pytest.raises(ValueError, match="num_inference_steps"):
        model.process_prompt("x", ["text"], ["image"], num_inference_steps=0)


def test_schedule_and_caption_bucket_from_token_count():
    model = _make_model()
    args = model.get_initial_forward_pass_args("default", ["text"], ["image"], {"text_inputs": [_info("t", [45])]},
                                               model_kwargs={"width": 768, "num_inference_steps": 8})
    md = args.full_metadata
    assert md.graph_walk == ENCODE_TEXT_WALK and md.kwargs["walk_schedule"] == [ENCODE_TEXT_WALK, IMAGE_GEN_WALK]
    assert args.step_metadata["text_len"] == 45 and args.step_metadata["cap_len"] == 64
    assert (args.step_metadata["height"], args.step_metadata["width"]) == (1024, 768)
    shape = shape_from_metadata(model.config, args.step_metadata)
    assert shape == ZShape(grid=(64, 48), cap_len=64) and shape.image_tokens_padded == 3072
    nxt = model.get_partition_forward_pass_args("default", md, {"text_embeds": [_info("e", [1, 64, 2560])]})
    assert nxt.full_metadata.graph_walk == IMAGE_GEN_WALK and not nxt.request_done
    assert {e.name for e in nxt.inputs} == {"text_embeds", "latents"}
    assert model.get_partition_forward_pass_args("default", nxt.full_metadata, {}).request_done


def test_capture_shapes_cover_sizes_times_caption_buckets():
    model = _make_model(capture_sizes=[[1024, 1024]], capture_caption_lengths=[32, 64])
    assert [s.cap_len for _, s in model.capture_shapes()] == [32, 64]
    assert all(s.grid == (64, 64) for _, s in model.capture_shapes())
    assert _make_model(cuda_graph=False).capture_shapes() == []


def test_postprocess_png_and_adapter():
    model = _make_model()
    assert model.postprocess(torch.zeros(1, 3, 16, 16, dtype=torch.uint8), "image")[:4] == b"\x89PNG"
    pytest.importorskip("pydantic")
    from mstar.api_server.openai import adapters

    assert isinstance(adapters.get_adapter("z_image_turbo"), adapters.DiffusionImageAdapter)
