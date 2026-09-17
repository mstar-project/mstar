"""Dummy-mode tests for the FLUX.2 [klein] model declaration.

Structural pieces only — no weights, no GPU, no network: the graph walks and their
edges, resource declaration per attention backend, the schedule-driven walk
transitions for text-to-image and editing, request validation at the 400 seam, the
shape key / capture-bucket plumbing, and the OpenAI adapter mapping. The model is
built with a pinned config and a stub tokenizer, so nothing touches the HF cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, ".")

from mstar.graph.base import GraphNode, Loop, Sequential, TensorPointerInfo  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION  # noqa: E402
from mstar.model.flux2_klein.config import DENOISE_LOOP, DIT_ATTN, Flux2KleinConfig  # noqa: E402
from mstar.model.flux2_klein.flux2_klein_model import (  # noqa: E402
    ENCODE_IMAGE_WALK,
    ENCODE_TEXT_WALK,
    IMAGE_EDIT_WALK,
    IMAGE_GEN_WALK,
    Flux2KleinModel,
)
from mstar.model.flux2_klein.submodules import KleinShape, shape_from_metadata  # noqa: E402

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs" / "flux2_klein.yaml")


class _StubTokenizer:
    """Byte-level stand-in with the chat-template + max_length padding contract."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return f"<u>{messages[0]['content']}<a>"

    def __call__(self, text, return_tensors="pt", padding="max_length", truncation=True, max_length=512):
        ids = list(text.encode("utf-8"))[:max_length]
        pad = max_length - len(ids)
        return {
            "input_ids": torch.tensor([ids + [0] * pad]),
            "attention_mask": torch.tensor([[1] * len(ids) + [0] * pad]),
        }


def _make_model(**kwargs) -> Flux2KleinModel:
    model = Flux2KleinModel(model_path_hf="test/flux2_klein", skip_weight_loading=True, **kwargs)
    model.set_config(Flux2KleinConfig())
    model.tokenizer = _StubTokenizer()
    return model


def _tensor_info(name: str, dims: list[int]) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=dims, dtype="torch.float32", nbytes=4, address=0, stride=[1] * len(dims),
        uuid=f"uuid-{name}", source_session_id="test:0", source_entity="test",
    )


# ----------------------------------------------------------------------
# Graph structure and resources
# ----------------------------------------------------------------------


def test_graph_walks_and_nodes():
    model = _make_model()
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {ENCODE_TEXT_WALK, ENCODE_IMAGE_WALK, IMAGE_GEN_WALK, IMAGE_EDIT_WALK}
    assert model.nodes == ["dit", "text_encoder", "vae_decoder", "vae_encoder"]

    text = walks[ENCODE_TEXT_WALK]
    assert isinstance(text, GraphNode) and text.input_names == {"text_inputs", "text_mask"}
    (embeds,) = text.outputs
    assert embeds.name == "text_embeds" and embeds.persist and embeds.next_node == EMPTY_DESTINATION

    image = walks[ENCODE_IMAGE_WALK]
    (refs,) = image.outputs
    assert image.input_names == {"image_inputs"} and refs.name == "ref_latents" and refs.persist


@pytest.mark.parametrize("walk,edit", [(IMAGE_GEN_WALK, False), (IMAGE_EDIT_WALK, True)])
def test_denoise_walk_structure(walk, edit):
    model = _make_model()
    section = model.get_graph_walk_graphs()[walk]
    assert isinstance(section, Sequential) and len(section.sections) == 2
    loop, decoder = section.sections
    assert isinstance(loop, Loop) and loop.name == DENOISE_LOOP
    assert loop.max_iters == model.config.max_denoise_steps
    dit = loop.section
    assert isinstance(dit, GraphNode) and dit.name == "dit"
    expected = {"text_embeds", "latents"} | ({"ref_latents"} if edit else set())
    assert dit.input_names == expected
    # latents is the only loop-back edge; the step index is the loop counter
    assert [(e.name, e.next_node) for e in dit.outputs] == [("latents", "dit")]
    assert [(e.name, e.next_node) for e in loop.outputs] == [("latents", "vae_decoder")]
    (emit,) = decoder.outputs
    assert emit.next_node == EMIT_TO_CLIENT and emit.name == "image_output" and emit.output_modality == "image"


def test_resources_follow_the_attention_backend():
    flashinfer = _make_model(attention_backend="flashinfer").get_node_resources()
    assert len(flashinfer) == 1 and flashinfer[0].resource_key == DIT_ATTN and flashinfer[0].nodes == {"dit"}
    cfg = flashinfer[0].config
    assert cfg.num_qo_heads == 24 and cfg.head_dim == 128 and cfg.max_segments_per_request == 1
    assert _make_model(attention_backend="sdpa").get_node_resources() == []
    with pytest.raises(ValueError):
        _make_model(attention_backend="flash")


def test_worker_graphs_from_yaml():
    model = _make_model()
    walks = set()
    for wg in model.get_worker_graphs(CONFIG_PATH):
        walks |= wg.graph_walks
    assert walks == {ENCODE_TEXT_WALK, ENCODE_IMAGE_WALK, IMAGE_GEN_WALK, IMAGE_EDIT_WALK}


def test_dummy_mode_returns_no_submodules():
    model = _make_model()
    assert all(model.get_submodule(node) is None for node in model.nodes)


# ----------------------------------------------------------------------
# process_prompt: tokenization + validation
# ----------------------------------------------------------------------


def test_process_prompt_tokenizes_with_chat_template_to_fixed_length():
    model = _make_model()
    out = model.process_prompt("a cat", ["text"], ["image"])
    ids, mask = out["text_inputs"][0], out["text_mask"][0]
    assert ids.shape == (512,) and mask.shape == (512,) and ids.dtype == torch.long
    n = int(mask.sum())
    assert bytes(ids[:n].tolist()) == b"<u>a cat<a>"
    assert int(mask[n:].sum()) == 0


@pytest.mark.parametrize("kwargs,message", [
    ({"height": 1000}, "multiple of 16"),
    ({"width": 0}, "multiple of 16"),
    ({"num_inference_steps": 0}, "num_inference_steps"),
])
def test_process_prompt_rejects_bad_geometry(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _make_model().process_prompt("a cat", ["text"], ["image"], **kwargs)


def test_process_prompt_rejects_missing_prompt_or_wrong_modality():
    model = _make_model()
    with pytest.raises(ValueError, match="text prompt"):
        model.process_prompt(None, ["text"], ["image"])
    with pytest.raises(ValueError, match="only generates images"):
        model.process_prompt("x", ["text"], ["text"])
    with pytest.raises(ValueError, match="reference image"):
        model.process_prompt("x", ["image", "text"], ["image"], tensors={"image_inputs": []})
    too_many = {"image_inputs": [torch.zeros(3, 64, 64)] * (model.config.max_ref_images + 1)}
    with pytest.raises(ValueError, match="at most"):
        model.process_prompt("x", ["image", "text"], ["image"], tensors=too_many)


def test_steps_are_clamped_to_the_loop_ceiling():
    model = _make_model()
    args = model.get_initial_forward_pass_args(
        "default", ["text"], ["image"], _text_signals(), model_kwargs={"num_inference_steps": 10_000},
    )
    assert args.step_metadata["num_inference_steps"] == model.config.max_denoise_steps


# ----------------------------------------------------------------------
# Walk schedule (state machine)
# ----------------------------------------------------------------------


def _text_signals():
    return {"text_inputs": [_tensor_info("t", [512])], "text_mask": [_tensor_info("m", [512])]}


def _step(model, metadata, persist):
    return model.get_partition_forward_pass_args("default", metadata, persist)


def test_t2i_schedule_encode_text_then_image_gen_then_done():
    model = _make_model()
    args = model.get_initial_forward_pass_args("default", ["text"], ["image"], _text_signals(),
                                               model_kwargs={"height": 768, "width": 1024, "num_inference_steps": 4})
    md = args.full_metadata
    assert md.graph_walk == ENCODE_TEXT_WALK and md.is_prefill
    assert md.kwargs["walk_schedule"] == [ENCODE_TEXT_WALK, IMAGE_GEN_WALK]
    assert args.step_metadata["height"] == 768 and args.step_metadata["width"] == 1024
    assert args.step_metadata["ref_grids"] == []
    targets = {(e.next_node, e.name) for e in args.inputs}
    assert targets == {("text_encoder", "text_inputs"), ("text_encoder", "text_mask")}

    persist = {"text_embeds": [_tensor_info("e", [1, 512, 7680])]}
    nxt = _step(model, md, persist)
    assert nxt.full_metadata.graph_walk == IMAGE_GEN_WALK and not nxt.full_metadata.is_prefill
    assert not nxt.request_done
    by_name = {e.name: e for e in nxt.inputs}
    assert set(by_name) == {"text_embeds", "latents"}
    assert by_name["text_embeds"].tensor_info == persist["text_embeds"]
    assert by_name["latents"].tensor_info == []  # seeded by the dit at iteration 0
    assert persist["text_embeds"][0].uuid in {i.uuid for i in nxt.unpersist_tensors}

    done = _step(model, nxt.full_metadata, {})
    assert done.request_done and done.inputs == []


def test_edit_schedule_inserts_encode_image_and_defaults_size_to_the_reference():
    model = _make_model()
    refs = [_tensor_info("i0", [3, 512, 768]), _tensor_info("i1", [3, 256, 256])]
    signals = {**_text_signals(), "image_inputs": refs}
    args = model.get_initial_forward_pass_args("default", ["image", "image", "text"], ["image"], signals)
    md = args.full_metadata
    assert md.kwargs["walk_schedule"] == [ENCODE_TEXT_WALK, ENCODE_IMAGE_WALK, IMAGE_EDIT_WALK]
    # output size defaults to the first reference image; ref grids follow every image
    assert (args.step_metadata["height"], args.step_metadata["width"]) == (512, 768)
    assert args.step_metadata["ref_grids"] == [[32, 48], [16, 16]]

    nxt = _step(model, md, {"image_inputs": signals["image_inputs"]})
    assert nxt.full_metadata.graph_walk == ENCODE_IMAGE_WALK and nxt.full_metadata.is_prefill
    assert [(e.next_node, e.name) for e in nxt.inputs] == [("vae_encoder", "image_inputs")]

    persist = {"text_embeds": [_tensor_info("e", [1, 512, 7680])],
               "ref_latents": [_tensor_info("r", [1, 1792, 128])]}
    nxt = _step(model, nxt.full_metadata, persist)
    assert nxt.full_metadata.graph_walk == IMAGE_EDIT_WALK
    assert {e.name for e in nxt.inputs} == {"text_embeds", "ref_latents", "latents"}
    assert _step(model, nxt.full_metadata, {}).request_done


def test_shape_key_from_step_metadata_and_capture_buckets():
    model = _make_model(capture_sizes=[[1024, 1024], [768, 1024]])
    shape = shape_from_metadata(model.config, {"height": 512, "width": 768, "ref_grids": [[32, 48]]})
    assert shape == KleinShape(grid=(32, 48), text_len=512, ref_grids=((32, 48),))
    assert shape.total_tokens == 512 + 32 * 48 + 32 * 48
    captured = model.capture_shapes()
    assert [(w, s.grid) for w, s in captured] == [(IMAGE_GEN_WALK, (64, 64)), (IMAGE_GEN_WALK, (48, 64))]
    assert _make_model(cuda_graph=False).capture_shapes() == []


def test_postprocess_encodes_png():
    png = _make_model().postprocess(torch.zeros(1, 3, 32, 48, dtype=torch.uint8), "image")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError):
        _make_model().postprocess(torch.zeros(1), "text")


def test_autocast_is_off_so_numerics_follow_the_checkpoint():
    assert _make_model().get_autocast_dtype() is None


# ----------------------------------------------------------------------
# OpenAI adapter
# ----------------------------------------------------------------------


def test_diffusion_image_adapter_maps_size_seed_and_edits(tmp_path):
    pytest.importorskip("pydantic")
    from mstar.api_server.openai import adapters
    from mstar.api_server.openai.protocol import ImageGenerationRequest

    adapter = adapters.get_adapter("flux2_klein")
    assert isinstance(adapter, adapters.DiffusionImageAdapter) and adapter.supports_images
    req = ImageGenerationRequest(model="flux2_klein", prompt="a cat", size="768x1024", seed=3,
                                 num_inference_steps=8)
    sa = adapter.image_to_request(req, tmp_path)
    assert sa.output_modalities == ["image"] and sa.input_modalities == ["text"] and sa.text == "a cat"
    assert sa.model_kwargs == {"num_inference_steps": 8, "width": 768, "height": 1024, "seed": 3}
    with pytest.raises(ValueError, match="WxH"):
        adapter.image_to_request(ImageGenerationRequest(prompt="x", size="big"), tmp_path)

    edit = adapter.image_edit_to_request("make it neon", "/tmp/in.png", {"seed": 5, "size": "1024x1024"})
    assert edit.file_paths == {"image": ["/tmp/in.png"]}
    assert edit.input_modalities == ["image", "text"] and edit.output_modalities == ["image"]
    assert edit.model_kwargs == {"seed": 5, "width": 1024, "height": 1024}


# ----------------------------------------------------------------------
# Real tokenizer (skipped without the checkpoint's tokenizer in the HF cache)
# ----------------------------------------------------------------------


def _cached_snapshot():
    import os

    for env in ("HF_HUB_CACHE", "HF_HOME"):
        root = os.environ.get(env)
        if not root:
            continue
        base = Path(root) if env == "HF_HUB_CACHE" else Path(root) / "hub"
        snaps = sorted((base / "models--black-forest-labs--FLUX.2-klein-4B" / "snapshots").glob("*"))
        if snaps and (snaps[0] / "tokenizer" / "tokenizer_config.json").exists():
            return snaps[0]
    return None


def test_tokenize_matches_the_pipeline_recipe_with_the_real_tokenizer():
    snapshot = _cached_snapshot()
    if snapshot is None:
        pytest.skip("FLUX.2-klein-4B tokenizer not in the local HF cache")
    transformers = pytest.importorskip("transformers")
    model = Flux2KleinModel(model_path_hf=str(snapshot), skip_weight_loading=True)
    model.set_config(Flux2KleinConfig())
    model.tokenizer = None  # load the real one from the snapshot
    prompt = "A cat holding a sign that says hello world"
    ids, mask = model.tokenize(prompt)
    tok = transformers.AutoTokenizer.from_pretrained(str(snapshot / "tokenizer"))
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    expected = ("<|im_start|>user\nA cat holding a sign that says hello world<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert text == expected
    ref = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    assert torch.equal(ids, ref["input_ids"][0]) and torch.equal(mask, ref["attention_mask"][0])
    assert int(mask.sum()) == 21 and ids[int(mask.sum()):].unique().tolist() == [151643]
