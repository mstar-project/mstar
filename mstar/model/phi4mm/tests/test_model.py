"""Phi4MMModel on the CPU: the prompt as the reference processor tokenizes it, the
adapter and LongRoPE ceiling each prompt runs under, the walks, the resources, and
the checkpoint's PEFT adapters against the LoRA spec. Needs the checkpoint's
metadata in the Hugging Face cache; skips without it."""

from __future__ import annotations

import pytest
import torch

from mstar.model.phi4mm.tests._paths import snapshot

SNAPSHOT = snapshot()
pytestmark = pytest.mark.skipif(SNAPSHOT is None, reason="Phi-4-multimodal checkpoint not cached")


@pytest.fixture(scope="module")
def model():
    from mstar.model.phi4mm.phi4mm_model import Phi4MMModel

    return Phi4MMModel(SNAPSHOT)


def parts(*items):
    from mstar.model.multimodal import PromptPart

    seen: dict[str, int] = {}
    out = []
    for modality, text in items:
        out.append(PromptPart(modality, text=text, index=seen.get(modality, 0)))
        seen[modality] = seen.get(modality, 0) + 1
    return out


def test_the_prompt_is_the_processors(model):
    """The reference's ``<|user|><|image_1|>{text}<|end|><|assistant|>``, the image a
    single pad (expanded to its rows at intake), tokenized as the processor does."""
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(SNAPSHOT)
    text = "Describe this image in two sentences."
    ids = model._prompt_ids(parts(("image", None), ("text", text)), {})
    assert ids.tolist() == tok(f"<|user|><|endoftext10|>{text}<|end|><|assistant|>").input_ids
    ids = model._prompt_ids(parts(("text", "hi")), {})
    assert ids.tolist() == tok("<|user|>hi<|end|><|assistant|>").input_ids


def test_stop_ids_union_generation_config_and_tokenizer(model):
    # <|endoftext|> and <|end|>, which a chat turn ends on
    assert set(model.config.stop_token_ids) == {199999, 200020}


@pytest.mark.parametrize(
    "media, adapter",
    [({}, "keep"), ({"image_inputs": [1]}, "vision"), ({"audio_inputs": [1]}, "speech"),
     ({"image_inputs": [1], "audio_inputs": [1]}, "vision")],
)
def test_the_adapter_follows_the_media(model, media, adapter):
    """The reference's ``input_mode``: vision whenever there is an image, speech for audio
    alone; a text prompt keeps the client's choice."""
    from mstar.model.base import KEEP_ADAPTER

    chosen = model._adapter_for(media)
    assert chosen == (KEEP_ADAPTER if adapter == "keep" else adapter)
    assert model.prompt_lora_adapters() == {"vision", "speech"}


def test_the_prompt_length_picks_the_longrope_table(model):
    """``process_prompt`` sets the ceiling the position spec builds the request's
    table and prefix variant from."""
    from mstar.engine.resources.position.config import MAX_POSITION_KWARG, PositionSpec

    out = model.process_prompt("hello", ["text"], ["text"], {"text_inputs": []}, parts(("text", "hello")))
    n = out.new_input_tensors["text_inputs"][0].numel()
    assert out.metadata[MAX_POSITION_KWARG] == n
    (spec,) = [s for s in model.get_node_resources() if isinstance(s, PositionSpec)]
    assert spec.prefix_variant(out.metadata) is None
    assert spec.prefix_variant({MAX_POSITION_KWARG: 5000}) == b"longrope-long"
    assert spec.config.rotary_dim == 96 and spec.config.rope_table_dtype is torch.bfloat16


def test_walks_and_resources(model):
    walks = set(model.get_graph_walk_graphs())
    assert {"prefill_text", "prefill_vision", "prefill_audio", "prefill_vision_audio", "decode"} <= walks
    keys = {s.resource_key for s in model.get_node_resources()}
    assert {"kv_cache", "attn", "rope", "sampler", "vision_attn", "lora"} <= keys


def test_the_checkpoints_adapters_fit_the_lora_spec(model):
    """Both PEFT adapters place on the fused qkv / gate_up and the o / down projections
    of every layer, at their ranks, as the API server checks them at startup."""
    from mstar.engine.resources.lora.store import check_adapter_file

    spec = model._lora_spec()
    spec.apply_yaml_overrides(max_rank=320, max_slots=2)
    for name, path in spec.adapters.items():
        check_adapter_file(spec, name, path)


@pytest.mark.parametrize(("recorded", "served"), [(8_000, 8_000), (11_025, 8_000), (16_000, 16_000), (44_100, 16_000)])
def test_audio_reaches_the_intake_at_the_references_rate(model, tmp_path, recorded, served):
    """The reference's feature extractor takes 8-16 kHz uploads through its 8 kHz front
    end and faster ones through its 16 kHz one; below 8 kHz it refuses. (It reaches
    those rates by decimating, which this resamples properly instead.)"""
    import numpy as np
    import soundfile as sf

    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(recorded // 2, dtype=np.float32), recorded)
    loaded = model.load_audio(str(path), "cpu")
    assert loaded.metadata["sample_rate"] == served
    assert loaded.data.numel() == served // 2
    sf.write(path, np.zeros(3_000, dtype=np.float32), 6_000)
    with pytest.raises(ValueError, match="8 kHz or more"):
        model.load_audio(str(path), "cpu")
