"""Combined native vision -> decoder parity against a tiny Transformers model.

The real intake, node forwards, YaRN tables and weight-copy rules run on CPU.
Dense test doubles replace ragged/paged attention, rotation execution and
sampling. Position counters are advanced by this harness, not an engine runner:
these tests establish tensor and cache parity, not scheduling or GPU correctness.
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.position.manager import RopeManager
from mstar.model.keys import KV_CACHE, ROPE, SAMPLER, VISION_ATTN
from mstar.model.loader.base import LLAMA_STACKED_PARAMS, load_weights_into
from mstar.model.ministral3.config import ATTN
from mstar.model.ministral3.ministral3_model import Ministral3Model
from mstar.model.ministral3.resources import text_node_resources
from mstar.model.ministral3.submodules import Ministral3Decoder
from mstar.model.ministral3.vision import VISION_OUTPUT, Ministral3VisionEncoder
from mstar.model.multimodal import PromptPart
from mstar.model.submodule_base import ModelInputsFromEngine

pytest.importorskip("torchvision")
pytest.importorskip("transformers.models.mistral3.modeling_mistral3")

# Reuse the component tests' CPU resource doubles and local metadata fixture.
from test_ministral3_lm import DenseCacheAttention, GreedySampler, cpu_table_rotation  # noqa: E402
from test_ministral3_model import checkpoint as checkpoint_fixture  # noqa: E402
from test_ministral3_vision import DenseAttention  # noqa: E402
from transformers import AutoModelForImageTextToText  # noqa: E402
from transformers.modeling_outputs import BaseModelOutputWithPooling  # noqa: E402
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config  # noqa: E402
from transformers.models.pixtral.image_processing_pixtral import PixtralImageProcessor  # noqa: E402
from transformers.models.pixtral.processing_pixtral import PixtralProcessor  # noqa: E402

checkpoint = checkpoint_fixture


@pytest.fixture
def models(checkpoint, monkeypatch):
    monkeypatch.setenv("MSTAR_DEVICE_LOOPBACK", "1")
    monkeypatch.setattr(torch.ops.mstar, "rope_apply_qk_inplace", cpu_table_rotation)
    model = Ministral3Model(str(checkpoint))
    model.config = replace(model.config, max_position_embeddings=33000)
    root = json.loads((Path(checkpoint) / "config.json").read_text())
    config = Mistral3Config(**root)
    # Choose after creating the composite config, which resets nested backends.
    for cfg in (config, config.text_config, config.vision_config):
        cfg._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(521)
        hf = AutoModelForImageTextToText.from_config(config).float().eval()
    assert hf.model.language_model.config._attn_implementation == "eager"
    assert hf.model.vision_tower.config._attn_implementation == "eager"
    assert hf.lm_head.weight is hf.model.language_model.embed_tokens.weight
    metadata = json.loads((Path(checkpoint) / "processor_config.json").read_text())
    image = PixtralImageProcessor(**{k: v for k, v in metadata["image_processor"].items() if v is not None})
    processor = PixtralProcessor(image, model.tokenizer, patch_size=metadata["patch_size"],
                                 spatial_merge_size=metadata["spatial_merge_size"])
    return model, hf, processor


def request(model, processor, shapes):
    """Independent HF processing from the same raw images and chat content."""
    generator = torch.Generator().manual_seed(73)
    images = [torch.randint(0, 256, (3, h, w), dtype=torch.uint8, generator=generator) for h, w in shapes]
    parts = [PromptPart("text", "Look: ")]
    content = [{"type": "text", "text": "Look: "}]
    for index in range(len(images)):
        parts.extend([PromptPart("image", index=index), PromptPart("text", " then ")])
        content.extend([{"type": "image"}, {"type": "text", "text": " then "}])
    parts.append(PromptPart("text", "Describe."))
    content.append({"type": "text", "text": "Describe."})
    native = model.process_prompt("", [p.modality for p in parts], ["text"],
                                  tensors={"image_inputs": images}, prompt_parts=parts)
    rendered = model.tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                    tokenize=False, add_generation_prompt=True)
    oracle = processor(text=rendered, images=images or None, return_tensors="pt",
                       add_special_tokens=False, input_data_format="channels_first")
    torch.testing.assert_close(native.new_input_tensors["text_inputs"][0], oracle["input_ids"][0], atol=0, rtol=0)
    return native, dict(oracle)


class NativePipeline:
    def __init__(self, model, hf):
        self.model = model
        self.vision = Ministral3VisionEncoder(model.vision_config).eval()
        self.decoder = Ministral3Decoder(model.config, encoders=model.encoders).eval()
        self.vision.model.load_state_dict({k: v for k, v in hf.model.state_dict().items()
                                          if not k.startswith("language_model.")}, strict=True)
        loaded = load_weights_into(self.decoder.model, list(hf.model.language_model.state_dict().items()),
                                   stacked_params=LLAMA_STACKED_PARAMS, strict=True)
        assert loaded == set(dict(self.decoder.model.named_parameters()))
        assert self.decoder.model.lm_head.weight is self.decoder.model.embed_tokens.weight
        spec = next(spec for spec in text_node_resources(model.config) if spec.resource_key == ROPE)
        self.pos = RopeManager(replace(spec.config, rope_dtype=torch.float32, rope_table_dtype=torch.float32),
                               torch.device("cpu"), max_seq_len=model.config.max_position_embeddings)
        self.cache, self.sampler = DenseCacheAttention(), GreedySampler()
        self.resources = {KV_CACHE: self.cache, ATTN: self.cache, ROPE: self.pos, SAMPLER: self.sampler}
        self.decoder.bind_node_resources(self.resources)
        self.vision_attention = DenseAttention([])
        self.vision.bind_node_resources({VISION_ATTN: self.vision_attention})
        self.vision_calls = 0
        self.features = {}
        self.hidden = None
        self.decoder.model.register_forward_hook(self._record_hidden)

    def _record_hidden(self, module, inputs, output):
        self.hidden = output.clone()

    def prefill(self, requests, starts, corrupt=None):
        decoder_inputs, infos, walks = [], {}, []
        for rid, prompt in requests:
            signals = {name: [object()] for name, tensors in prompt.new_input_tensors.items() if tensors}
            step = self.model.walk_plan("main", [], ["text"], signals, prompt.metadata)[0]
            walks.append(step.walk)
            info = SimpleNamespace(step_metadata=step.step_metadata, rid_handle=rid)
            infos[rid] = info
            incoming = {"text_inputs": prompt.new_input_tensors["text_inputs"]}
            self.pos._counters[rid] = {"main": starts[rid]}
            if step.walk == "prefill_vision":
                prepared = self.vision.prepare_inputs(step.walk, info,
                    {"pixel_values": prompt.new_input_tensors["pixel_values"]})
                declared = self.vision.declare_step(step.walk, [rid], [prepared])
                self.vision_attention.lengths = [segment.span for segment in declared.segments]
                engine = ModelInputsFromEngine(request_ids=[rid], per_request_info={rid: info})
                output = self.vision.forward(step.walk, engine, **self.vision.preprocess(step.walk, engine, [prepared]))
                self.vision_calls += 1
                features = output[VISION_OUTPUT][0]
                self.features[rid] = features.clone()
                if corrupt == "reverse_features":
                    features = features.flip(0)
                elif corrupt == "zero_features":
                    features = torch.zeros_like(features)
                incoming[VISION_OUTPUT] = [features]
            decoder_inputs.append(self.decoder.prepare_inputs(step.walk, info, incoming))
        assert len(set(walks)) == 1, "this harness batches one prefill walk at a time"
        return self._step(walks[0], [rid for rid, _ in requests], decoder_inputs, infos)

    def decode(self, rids):
        inputs = [self.decoder.prepare_inputs("decode", None, {}, has_loopback=True) for _ in rids]
        return self._step("decode", rids, inputs, {})

    def _step(self, walk, rids, inputs, infos):
        declared = self.decoder.declare_step(walk, rids, inputs)
        self.cache.rows = [(segment.request_id, segment.span) for segment in declared.segments]
        self.pos._current_pos_ids["main"] = torch.cat([
            torch.arange(self.pos.position(rid, "main"), self.pos.position(rid, "main") + count)
            for rid, count in self.cache.rows
        ])
        engine = ModelInputsFromEngine(request_ids=rids, per_request_info=infos, resources=self.resources)
        output = self.decoder.forward_batched(walk, engine, **self.decoder.preprocess(walk, engine, inputs))
        assert torch.equal(output.row_outputs["new_token"], self.sampler.logits.argmax(-1))
        all_logits = self.decoder.model.logits(self.hidden).split([count for _, count in self.cache.rows])
        for rid, count in self.cache.rows:
            self.pos._counters[rid]["main"] += count
        return dict(zip(rids, all_logits, strict=True))


def compare(actual, expected, label):
    error = (actual - expected).abs().max().item()
    print(f"{label}: max_abs_error={error:.9g}")
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("shapes,start", [([], 0), ([(5, 7)], 0), ([(5, 7), (9, 3)], 16380)])
@torch.no_grad()
def test_combined_prefill_and_cached_decode(models, shapes, start):
    model, hf, processor = models
    prompt, reference = request(model, processor, shapes)
    native = NativePipeline(model, hf)
    ids = reference["input_ids"]
    reference["position_ids"] = torch.arange(start, start + ids.shape[1])[None]
    expected = hf(**reference, use_cache=True)
    actual = native.prefill([(17, prompt)], {17: start})[17]
    compare(actual, expected.logits[0], f"prefill images={len(shapes)} start={start}")
    assert native.sampler.tokens[17] == expected.logits[0, -1].argmax()
    if shapes:
        compare(native.features[17], expected.image_hidden_states, "projected image features")
        # Arithmetic control, separate from the independent end-to-end oracle:
        # matching the vision features must remove the decoder logit difference.
        aligned_inputs = {k: v for k, v in reference.items() if k not in ("pixel_values", "image_sizes")}
        aligned = hf(**aligned_inputs, mm_encoder_outputs={
            "image": BaseModelOutputWithPooling(pooler_output=(native.features[17],)),
        }, use_cache=False)
        torch.testing.assert_close(actual, aligned.logits[0], atol=0, rtol=0)
    for step in range(3):
        token = expected.logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat((ids, token), dim=1)
        positions = torch.tensor([[start + ids.shape[1] - 1]])
        expected = hf(input_ids=token, position_ids=positions, past_key_values=expected.past_key_values, use_cache=True)
        actual = native.decode([17])[17]
        compare(actual, expected.logits[0], f"cached decode step={step}")
        assert native.sampler.tokens[17] == expected.logits[0, -1].argmax()
        # Independent full recomputation includes the original images every time.
        full_inputs = {**reference, "input_ids": ids, "attention_mask": torch.ones_like(ids),
                       "position_ids": torch.arange(start, start + ids.shape[1])[None]}
        full = hf(**full_inputs, use_cache=False)
        compare(actual, full.logits[0, -1:], f"full recomputation step={step}")
    assert native.vision_calls == bool(shapes)  # images are not encoded again during decode
    assert native.pos.position(17, "main") == start + ids.shape[1]
    assert all(k.shape[0] == ids.shape[1] for k, _ in native.cache.cache.values())


@torch.no_grad()
def test_packed_image_requests_and_reordered_decode(models):
    model, hf, processor = models
    cases = {17: request(model, processor, [(5, 7)]), 4: request(model, processor, [(3, 11), (9, 5)])}
    starts = {17: 0, 4: 16380}
    expected = {}
    for rid, (_, reference) in cases.items():
        positions = torch.arange(starts[rid], starts[rid] + reference["input_ids"].shape[1])[None]
        expected[rid] = hf(**reference, position_ids=positions, use_cache=True)
    native = NativePipeline(model, hf)
    actual = native.prefill([(rid, prompt) for rid, (prompt, _) in cases.items()], starts)
    for rid in cases:
        compare(actual[rid], expected[rid].logits[0], f"packed prefill rid={rid}")
        assert native.sampler.tokens[rid] == expected[rid].logits[0, -1].argmax()
    for step, rids in enumerate(([4, 17], [17, 4], [4, 17])):
        for rid in rids:
            expected[rid] = hf(input_ids=expected[rid].logits[:, -1].argmax(-1, keepdim=True),
                position_ids=torch.tensor([[native.pos.position(rid, "main")]]),
                past_key_values=expected[rid].past_key_values, use_cache=True)
        actual = native.decode(rids)
        for rid in rids:
            compare(actual[rid], expected[rid].logits[0], f"reordered decode step={step} rid={rid}")
            assert native.sampler.tokens[rid] == expected[rid].logits[0, -1].argmax()
    assert native.vision_calls == 2


@pytest.mark.parametrize("corruption", ["reverse_features", "zero_features"])
@torch.no_grad()
def test_combined_parity_rejects_broken_image_connection(models, corruption):
    model, hf, processor = models
    prompt, reference = request(model, processor, [(5, 7), (9, 3)])
    expected = hf(**reference, use_cache=False).logits[0]
    good = NativePipeline(model, hf).prefill([(17, prompt)], {17: 0})[17]
    compare(good, expected, "negative-control baseline")
    bad = NativePipeline(model, hf).prefill([(17, prompt)], {17: 0}, corrupt=corruption)[17]
    # The final TEXT token must notice the images, not only the image positions.
    print(f"{corruption}: final_text_logit_error={(bad[-1] - expected[-1]).abs().max().item():.9g}")
    with pytest.raises(AssertionError):
        torch.testing.assert_close(bad[-1], expected[-1], atol=2e-6, rtol=1e-5)
