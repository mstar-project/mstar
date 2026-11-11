"""Outer Model contracts with a synthetic local tokenizer, no weights or runtime.

Real-checkpoint tokenizer parity is tested separately in test_ministral3_text.
These tests run intake, walk planning and output processing, not GPU generation.
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.request_info import FINAL_PREFILL
from mstar.engine.resources import resolve_spec_dependencies
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.keys import KV_CACHE, MIXED, PREFILL_TEXT, ROPE, SAMPLER, VISION_ATTN
from mstar.model.ministral3.config import ATTN, LLM_NODE
from mstar.model.ministral3.image import IMAGE_GRIDS
from mstar.model.ministral3.ministral3_model import Ministral3Model
from mstar.model.ministral3.submodules import Ministral3Decoder
from mstar.model.ministral3.vision import VISION_OUTPUT, Ministral3VisionEncoder
from mstar.model.multimodal import MEDIA_LAYOUT, PromptPart
from mstar.model.submodule_base import END_PARTITION, HostRows, ModelInputsFromEngine

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ministral3_metadata")
    special = ["<unk>", "<s>", "</s>", "<pad>", "[IMG]", "[IMG_BREAK]", "[IMG_END]"]
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: i for i, token in enumerate([*special, *alphabet])}
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token=special[0], bos_token=special[1],
        eos_token=special[2], pad_token=special[3], clean_up_tokenization_spaces=False,
        additional_special_tokens=special[4:],
        chat_template=(
            "{{ bos_token }}{% for message in messages %}{{ message['role'] }}:"
            "{% for part in message['content'] %}{% if part['type'] == 'image' %}[IMG]"
            "{% else %}{{ part['text'] }}{% endif %}{% endfor %}\n"
            "{% endfor %}assistant:"
        ),
    )
    tokenizer.save_pretrained(directory)
    raw = json.loads((Path(__file__).parent / "fixtures/ministral3/config.json").read_text())
    raw["text_config"].update(hidden_size=96, num_hidden_layers=2, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=32, intermediate_size=192, vocab_size=len(tokenizer))
    raw["image_token_index"] = 4
    raw["vision_config"].update(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                num_attention_heads=2, head_dim=16, patch_size=2, image_size=32)
    (directory / "config.json").write_text(json.dumps(raw))
    processor = json.loads((Path(__file__).parent / "fixtures/ministral3/processor_config.json").read_text())
    processor["patch_size"] = processor["image_processor"]["patch_size"] = 2
    processor["image_processor"]["size"]["longest_edge"] = 32
    (directory / "processor_config.json").write_text(json.dumps(processor))
    (directory / "generation_config.json").write_text(json.dumps({
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 3,
        "temperature": 0.7, "top_p": 0.9, "min_p": 0.05,
    }))
    return directory


@pytest.fixture
def model(checkpoint):
    return Ministral3Model(str(checkpoint))


def test_intake_walks_and_kv_layout_agree(model):
    parts = [PromptPart("text", "Be concise.", role="system"), PromptPart("text", "Hello", role="user")]
    result = model.process_prompt("Be concise.\nHello", ["text", "text"], ["text"], prompt_parts=parts)
    ids, = result.new_input_tensors["text_inputs"]
    assert torch.equal(ids, model.text_processor.prompt_ids(parts))
    assert set(result.new_input_tensors) == {"text_inputs"}
    layout = result.metadata["kv_layout"][KV_CACHE]["main"]
    span, = layout.spans
    assert (span.kind, span.walk, span.length, span.advance, span.source) == (
        "ids", PREFILL_TEXT, ids.numel(), ids.numel(), "text_inputs",
    )
    assert layout.growth == (("decode_loop", 1),)
    assert layout.keyed_count() == 1
    labels = model.kv_labels()[KV_CACHE]["main"]
    assert labels.keys.tensor == span.source and labels.keys.walks == (span.walk,)
    assert labels.growth == (("decode", "decode_loop"),)
    plan = model.walk_plan("main", ["text"], ["text"], {"text_inputs": [object()]}, result.metadata)
    assert [step.walk for step in plan] == [PREFILL_TEXT, "decode"]
    assert plan[0].step_metadata[FINAL_PREFILL] is True
    assert plan[1].bind == {"text_inputs": "new_token"}
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {PREFILL_TEXT, "decode", "prefill_vision"}
    assert all(set(walks[name].get_nodes()) == {LLM_NODE} for name in (PREFILL_TEXT, "decode"))
    assert set(walks["prefill_vision"].get_nodes()) == {LLM_NODE, "vision_encoder"}
    assert walks["decode"].name == "decode_loop"
    assert walks["decode"].max_iters == model.config.max_position_embeddings
    assert model.get_combined_graph_walks() == {LLM_NODE: {MIXED: {PREFILL_TEXT, "decode"}}}
    first = walks[PREFILL_TEXT].outputs
    assert any(e.name == "new_token" and e.next_node == EMIT_TO_CLIENT and e.persist for e in first)
    later = walks["decode"].section.outputs
    assert {e.next_node for e in later if e.name == "text_inputs"} == {LLM_NODE, EMIT_TO_CLIENT}


def test_sampling_defaults_and_resources_agree(model, monkeypatch):
    specs = resolve_spec_dependencies(model.get_node_resources())
    assert all(spec.nodes == {LLM_NODE} for key, spec in specs.items() if key != VISION_ATTN)
    assert specs[VISION_ATTN].nodes == {"vision_encoder"}
    assert specs[SAMPLER].enable_min_p
    config = model.get_request_resource_configs(["text"], ["text"], {"temperature": 0, "ignore_eos": True})[SAMPLER]
    assert (config.temperature, config.top_p, config.min_p, config.ignore_eos) == (0, 0.9, 0.05, True)
    assert model.sampling_defaults["temperature"] == 0.7
    monkeypatch.setattr(model, "sampling_defaults", {})
    assert not resolve_spec_dependencies(model.get_node_resources())[SAMPLER].enable_min_p
    assert "enable_thinking" not in model.request_kwargs()


@pytest.mark.parametrize("budget,cap", [(1, 1), (2, 1), (5, 4)])
def test_reply_budget_counts_prefill_sample(model, budget, cap):
    assert model.get_loop_caps(max_output_tokens=budget) == {"decode_loop": cap}
    plan = model.walk_plan("main", ["text"], ["text"], {"text_inputs": [object()]}, {})
    with torch.device("meta"):
        decoder = Ministral3Decoder(model.config)
    info = SimpleNamespace(graph_walk=plan[0].walk, max_tokens=budget,
                           step_metadata=plan[0].step_metadata,
                           resource_configs=model.get_request_resource_configs(["text"], ["text"], {}))
    rows = HostRows(request_ids=(17,), buffers={"new_token": torch.tensor([[42]])})
    stopped = decoder.check_stop_batched([17], {17: info}, rows)
    assert stopped == ({17: {END_PARTITION}} if budget == 1 else {})


def test_tensor_and_inline_text_outputs_match(model):
    text = "café 中文 👋🏽\n"
    ids = model.tokenizer.encode(text, add_special_tokens=False)
    eos = model.tokenizer.eos_token_id
    tokens = [*ids, eos]
    whole = model.postprocess(torch.tensor(tokens), "text")
    chunks = model.postprocess_inline(tokens, "text", [{} for _ in tokens])
    assert b"".join(chunks) == whole == text.encode("utf-8")
    assert chunks[-1] == b""
    assert model.postprocess_inline([eos, eos], "text", [{}, {"ignore_eos": True}]) == [b"", b"</s>"]
    assert model.postprocess(torch.tensor([eos]), "text", {"ignore_eos": True}) == b"</s>"


@pytest.mark.parametrize("changes", [
    {"input_modalities": ["image"]}, {"input_modalities": ["audio"]}, {"input_modalities": ["video"]},
    {"prompt_parts": [PromptPart("image")]}, {"tensors": {"image_inputs": [torch.zeros(1)]}},
    {"tensors": {"text_inputs": [torch.tensor([1])]}}, {"output_modalities": ["audio"]},
    {"open_streams": frozenset({"text_inputs"})},
])
def test_unsupported_intake_fails_without_poisoning_next_request(model, changes):
    kwargs = dict(prompt="Hello", input_modalities=["text"], output_modalities=["text"])
    with pytest.raises(ValueError):
        model.process_prompt(**{**kwargs, **changes})
    assert model.process_prompt(**kwargs).new_input_tensors["text_inputs"][0].numel() > 0
    assert model.max_input_files() == {"audio": 0, "video": 0}


def test_remote_initialization_only_requests_metadata(checkpoint, monkeypatch):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(checkpoint)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    model = Ministral3Model("mistralai/Ministral-3-3B-Instruct-2512")
    assert calls[0]["allow_patterns"] == [*model.METADATA_FILES, "generation_config.json"]
    assert all("*" not in name and not name.endswith("safetensors") for name in calls[0]["allow_patterns"])
    # Metadata initialization must not fetch either weight representation.
    assert len(calls) == 1
    assert model._submodule_cache == {}
    with pytest.raises(RuntimeError, match="requires NVIDIA"):
        model.get_submodule(LLM_NODE, device="cpu")
    assert len(calls) == 1  # unsupported execution fails before any weight fetch
    with pytest.raises(ValueError, match="no node"):
        model.get_submodule("audio_encoder")


def test_text_intake_discards_client_media_layout(model):
    result = model.process_prompt("Hello", ["text"], ["text"], **{MEDIA_LAYOUT: {"output": ["vision"]}})
    assert "vision" not in result.metadata[MEDIA_LAYOUT].get("output", [])


def image_prompt(model, value=0):
    pytest.importorskip("torchvision")
    images = [torch.full((3, 5, 7), value, dtype=torch.uint8),
              torch.full((3, 3, 11), 255 - value, dtype=torch.uint8)]
    parts = [PromptPart("image"), PromptPart("text", "Compare"), PromptPart("image", index=1)]
    return model.process_prompt("Compare", [p.modality for p in parts], ["text"],
                                tensors={"image_inputs": images}, prompt_parts=parts)


@torch.no_grad()
def test_image_layout_graph_and_decoder_splice(model, monkeypatch):
    result = image_prompt(model)
    ids = result.new_input_tensors["text_inputs"][0]
    layout = result.metadata[MEDIA_LAYOUT]
    assert layout["tokens"] == [2, 2, 3]
    assert layout[IMAGE_GRIDS] == [[1, 4, 4], [1, 2, 6]]
    assert result.new_input_tensors["image_inputs"] == []  # raw images released after preparation
    plan = model.walk_plan("main", ["image", "text", "image"], ["text"],
                           {"text_inputs": [object()], "pixel_values": [object()]}, result.metadata)
    assert [step.walk for step in plan] == ["prefill_vision", "decode"]
    nodes = model.get_graph_walk_graphs()["prefill_vision"].get_nodes()
    assert nodes["vision_encoder"].input_names == {"pixel_values"}
    assert nodes["vision_encoder"].outputs[0].next_node == LLM_NODE
    assert nodes["vision_encoder"].outputs[0].name == VISION_OUTPUT
    assert nodes[LLM_NODE].input_names == {"text_inputs", VISION_OUTPUT}

    info = SimpleNamespace(step_metadata=plan[0].step_metadata, rid_handle=17)
    vision = Ministral3VisionEncoder(model.vision_config)
    vision_inputs = vision.prepare_inputs("prefill_vision", info,
        {"pixel_values": result.new_input_tensors["pixel_values"]})
    step = vision.declare_step("prefill_vision", [17], [vision_inputs])
    assert [segment.span for segment in step.segments] == [16, 12]  # whole images, not token rows

    decoder = Ministral3Decoder(model.config, encoders=model.encoders)
    decoder.node_resources[ROPE] = SimpleNamespace(position=lambda **kwargs: 0)
    # Distinct synthetic features reveal dropped/reordered image embeddings.
    features = torch.arange(7 * model.config.hidden_size, dtype=torch.float32).reshape(7, -1)
    incoming = {"text_inputs": [ids], VISION_OUTPUT: [features]}
    inp = decoder.prepare_inputs("prefill_vision", info, incoming)
    assert inp.custom_pos_ids is None and inp.resource_step_info is None  # ordinary sequential text positions
    assert inp.input_seq_len == len(ids)
    observed = []

    def record_embeddings(embeds, **kwargs):
        observed.append(embeds.clone())
        return embeds

    monkeypatch.setattr(decoder.model, "forward", record_embeddings)
    engine = ModelInputsFromEngine(request_ids=[17], per_request_info={}, resources={
        SAMPLER: SimpleNamespace(sample=lambda rids, logits: logits.argmax(-1)),
        ATTN: SimpleNamespace(select_last_hidden=lambda hidden: hidden[-1:]),
    })
    decoder.forward("prefill_vision", engine, **decoder.preprocess("prefill_vision", engine, [inp]))
    is_image = ids == model.image_processor.image_id
    torch.testing.assert_close(observed[0][is_image], features, atol=0, rtol=0)
    torch.testing.assert_close(observed[0][~is_image], decoder.model.embed(ids)[~is_image], atol=0, rtol=0)
    for bad in ({"text_inputs": [ids]}, {"text_inputs": [ids], VISION_OUTPUT: [features[:-1]]}):
        with pytest.raises(ValueError, match="rows for the prompt"):
            decoder.prepare_inputs("prefill_vision", info, bad)
    with pytest.raises(ValueError, match="unexpected inputs"):
        decoder.prepare_inputs("decode", info, incoming)


def test_image_prefixes_are_unkeyed_even_when_placeholder_ids_match(model):
    first, second = image_prompt(model, 0), image_prompt(model, 128)
    torch.testing.assert_close(first.new_input_tensors["text_inputs"][0], second.new_input_tensors["text_inputs"][0])
    assert not torch.equal(first.new_input_tensors["pixel_values"][0], second.new_input_tensors["pixel_values"][0])
    for prompt in (first, second):
        layout = prompt.metadata["kv_layout"][KV_CACHE]["main"]
        assert layout.keyed_count() == 0
        assert all(span.walk == "prefill_vision" for span in layout.spans)
        assert sum(span.length for span in layout.spans) == prompt.new_input_tensors["text_inputs"][0].numel()
        assert all(span.length == span.advance for span in layout.spans)
        assert [span.length for span in layout.spans if span.kind == "rows"] == [2, 2, 3]
        # Row delimiters are text spans and only the first span opens the prefill step.
        assert layout.spans[0].opens_step
        assert not any(span.opens_step for span in layout.spans[1:])
    text = model.process_prompt("Hello", ["text"], ["text"])
    assert text.metadata["kv_layout"][KV_CACHE]["main"].keyed_count() == 1


def test_context_limit_includes_image_row_delimiters(model, monkeypatch):
    result = image_prompt(model)
    length = result.new_input_tensors["text_inputs"][0].numel()
    monkeypatch.setattr(model, "config", replace(model.config, max_position_embeddings=length - 1))
    with pytest.raises(ValueError, match="after image expansion"):
        image_prompt(model)


@pytest.mark.parametrize("token", ["[IMG]", "[IMG_BREAK]", "[IMG_END]"])
def test_literal_image_control_tokens_are_rejected(model, token):
    with pytest.raises(ValueError, match="control tokens"):
        model.process_prompt("Describe " + token, ["text"], ["text"])
    assert model.process_prompt("Hello", ["text"], ["text"]).new_input_tensors["text_inputs"]
