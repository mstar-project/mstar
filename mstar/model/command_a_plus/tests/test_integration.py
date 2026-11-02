"""Offline graph/node integration checks; attention and sampling use CPU doubles."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file
from torch import nn

from mstar.conductor.request_info import DEFAULT_PARTITION, CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup
from mstar.engine.resources import AttentionStep, SamplingReqConfig
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import MAX_OUTPUT_TOKENS
from mstar.model.command_a_plus.command_a_plus_model import CommandAPlusModel
from mstar.model.command_a_plus.config import (
    GLOBAL_ATTN,
    KV_CACHE,
    LOCAL_ATTN,
    ROPE,
    SAMPLER,
)
from mstar.model.command_a_plus.submodules import CommandAPlusLLMSubmodule, CommandAPlusVisionSubmodule
from mstar.model.command_a_plus.tests.test_backbone import bind_test_resources
from mstar.model.command_a_plus.tests.test_weight_loading import checkpoint
from mstar.model.multimodal import MEDIA_LAYOUT, PromptPart
from mstar.model.submodule_base import END_PARTITION, ModelInputsFromEngine

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_config.json"


def info(walk="prefill", max_tokens=6, ignore_eos=False, index=0, layout=None):
    return CurrentForwardPassInfo(
        request_id="a", graph_walk=walk, fwd_index=0, random_seed=0, max_tokens=max_tokens,
        resource_configs={SAMPLER: SamplingReqConfig(ignore_eos=ignore_eos)},
        dynamic_loop_iter_counts={"decode_loop": index} if walk == "decode" else {},
        step_metadata={} if layout is None else {MEDIA_LAYOUT: layout},
    )


class ToyLM(nn.Module):
    """The decoder contract over one-hot embeddings: a row's hidden state is its embedding."""

    def __init__(self, vocab):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(torch.eye(vocab))

    def forward(self, x, *, label):
        assert label == "main"
        return x

    def logits(self, x):
        return x


def toy_node(model):
    return CommandAPlusLLMSubmodule(ToyLM(model.config.vocab_size), model.config, (model.image_encoder,))


class NodeTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        shutil.copyfile(FIXTURE, Path(directory) / "config.json")
        self.model = CommandAPlusModel(directory)
        self.config = self.model.config
        self.node = toy_node(self.model)

    def test_packed_prefill_selects_each_requests_last_token_and_decode_routes_rows(self):
        sampler = SimpleNamespace(sample=Mock(side_effect=lambda rids, logits: logits.argmax(-1)))
        attention = SimpleNamespace(select_last_hidden=Mock(side_effect=lambda x: x[[2, 4]]))
        engine = ModelInputsFromEngine(["a", "b"], {}, {LOCAL_ATTN: attention, SAMPLER: sampler})
        for walk, rows, expected in (
            ("prefill", [[2, 5, 7], [11, 13]], [7, 13]),
            ("decode", [[17], [19]], [17, 19]),
        ):
            inputs = [self.node.prepare_inputs(walk, info(walk), {"text_inputs": [torch.tensor(row)]})
                      for row in rows]
            step = self.node.declare_step(walk, engine.request_ids, inputs)
            self.assertEqual([s.span for s in step.segments], [len(row) for row in rows])
            self.assertEqual(set(step.steps), {KV_CACHE, LOCAL_ATTN, GLOBAL_ATTN, ROPE, SAMPLER})
            for key in (LOCAL_ATTN, GLOBAL_ATTN):
                self.assertIsInstance(step.steps[key], AttentionStep)
                self.assertTrue(step.steps[key].causal)
            tracked = step.steps[SAMPLER].prefill_tracked_tokens
            self.assertEqual(set(tracked), {"a", "b"} if walk == "prefill" else set())
            output = self.node.forward_batched(walk, engine, **self.node.preprocess(walk, engine, inputs))
            self.assertEqual(output.row_outputs["new_token"].tolist(), expected)
            self.assertEqual(sampler.sample.call_args.args[0], engine.request_ids)
        attention.select_last_hidden.assert_called_once()

    def test_its_eos_ends_either_prefill_and_the_decode_loop(self):
        """A prefill row ends the partition on EOS or a one-token budget; decode
        stops on EOS alone, since the loop cap is its budget."""
        eos, other = self.config.eos_token_id, self.config.eos_token_id + 1
        for walk, stop in (("prefill", END_PARTITION), (self.model.image_walk, END_PARTITION),
                           ("decode", "decode_loop")):
            for token, max_tokens, ignore_eos, stops in (
                (eos, 6, False, True), (eos, 6, True, False), (other, 6, False, False),
                (other, 1, False, walk != "decode"),
            ):
                with self.subTest(walk=walk, token=token, max_tokens=max_tokens, ignore_eos=ignore_eos):
                    request = info(walk, max_tokens=max_tokens, ignore_eos=ignore_eos)
                    output = {"new_token": [torch.tensor([token])]}
                    self.assertEqual(self.node.check_stop("a", request, output), {stop} if stops else set())

    def test_only_decode_captures_and_it_runs_uncompiled(self):
        """Prefill CUDA graphs are unmeasured on this model, and torch.compile is off."""
        (config,) = self.node.get_accelerator_graph_configs(torch.device("cpu"))
        self.assertEqual((config.capture_graph_walk, config.compile), ("decode", False))


class ModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        shutil.copyfile(FIXTURE, Path(self.directory) / "config.json")
        self.model = CommandAPlusModel(self.directory)

    def test_graph_and_resources(self):
        model = self.model
        resources = {r.resource_key: r for r in model.get_node_resources()}
        self.assertEqual(resources[LOCAL_ATTN].config.sliding_window, 4)
        self.assertIsNone(resources[GLOBAL_ATTN].config.sliding_window)
        for key in (LOCAL_ATTN, GLOBAL_ATTN, ROPE):
            self.assertEqual(resources[key].depends_on(), {KV_CACHE})
        self.assertEqual(resources[KV_CACHE].config.max_seq_len, 64)
        self.assertTrue(resources[ROPE].config.interleave)
        self.assertEqual(resources[ROPE].config.rotary_dim, 8)
        walks = model.get_graph_walk_graphs()
        self.assertEqual(model.nodes, ["LLM", "vision_encoder"])
        self.assertEqual(len(model.get_partitions()), 1)
        (output,) = walks["prefill"].outputs
        self.assertEqual((output.name, output.next_node, output.conductor_new_token, output.persist),
                         ("new_token", EMIT_TO_CLIENT, True, True))
        self.assertEqual(walks["decode"].name, "decode_loop")
        self.assertEqual({e.name for e in walks["decode"].section.outputs}, {"new_token", "text_inputs"})
        plan = model.walk_plan(DEFAULT_PARTITION, ["text"], ["text"], {"text_inputs": [object()]}, {})
        self.assertEqual([(step.walk, dict(step.bind)) for step in plan],
                         [("prefill", {}), ("decode", {"text_inputs": "new_token"})])
        cfg = model.get_request_resource_configs([], [], {"temperature": 0})[SAMPLER]
        self.assertEqual((cfg.temperature, cfg.top_p, cfg.repetition_penalty), (0, 0.95, 1.04))

    def test_local_checkpoint_node_runs_prefill_and_four_decode_steps(self):
        save_file(checkpoint(self.model.config), Path(self.directory) / "model.safetensors")
        node = self.model.get_submodule("LLM", autocast_dtype=torch.float32)
        kv, pos, local, global_ = bind_test_resources(node.model.model, self.model.config)
        local.select_last_hidden = lambda hidden: hidden[-1:]
        sampler = SimpleNamespace(sample=lambda request_ids, logits: logits.argmax(-1))
        resources = {KV_CACHE: kv, ROPE: pos, LOCAL_ATTN: local, GLOBAL_ATTN: global_, SAMPLER: sampler}
        node.bind_node_resources(resources)
        engine = ModelInputsFromEngine(["a"], {}, resources)
        tokens = torch.tensor([2, 9, 4, 7, 11])
        offset = 0
        with torch.inference_mode():
            for step in range(5):
                walk = "prefill" if step == 0 else "decode"
                request = info(walk, max_tokens=20, ignore_eos=True, index=max(0, step - 1))
                inputs = [node.prepare_inputs(walk, request, {"text_inputs": [tokens]})]
                declaration = node.declare_step(walk, ["a"], inputs)
                self.assertEqual(declaration.segments[0].span, len(tokens))
                pos.positions["main"] = torch.arange(offset, offset + len(tokens))
                offset += len(tokens)
                result = node.forward_batched(walk, engine, **node.preprocess(walk, engine, inputs))
                tokens = result.row_outputs["new_token"]
                self.assertEqual(tokens.shape, (1,))
        self.assertTrue(all(len(k) == 9 for k, v in kv.cache.values()))
        self.assertEqual(len(kv.writes), 20)

    def test_image_requests_take_the_image_walk(self):
        layout = {"offset": [3], "tokens": [4], "output": ["image_embeds"], "grid": [[]]}
        plan = self.model.walk_plan(
            DEFAULT_PARTITION, ["text", "image"], ["text"], {"text_inputs": [object()], "pixel_values": [object()]},
            {MEDIA_LAYOUT: layout},
        )
        self.assertEqual([step.walk for step in plan], ["prefill_image", "decode"])
        # what the LLM checks the encoder's rows against
        self.assertEqual(plan[0].step_metadata[MEDIA_LAYOUT], layout)

    def test_vision_node_loads_from_the_checkpoint_and_encodes_tiles(self):
        from mstar.model.command_a_plus.tests.test_vision import checkpoint_name, tiny_encoder

        torch.manual_seed(0)
        vision = {checkpoint_name(n): torch.randn_like(p) for n, p in tiny_encoder().named_parameters()}
        save_file({**checkpoint(self.model.config), **vision}, Path(self.directory) / "model.safetensors")
        node = self.model.get_submodule("vision_encoder", autocast_dtype=torch.float32)
        tiles = [torch.rand(2, 3, 32, 32), torch.rand(1, 3, 32, 32)]
        inputs = node.prepare_inputs("prefill_image", info("prefill_image"), {"pixel_values": tiles})
        with torch.inference_mode():
            outputs = node.forward("prefill_image", None, **node.preprocess("prefill_image", None, [inputs]))
        self.assertEqual(tuple(outputs["image_embeds"][0].shape), (3 * 4, self.model.config.hidden_size))

    def test_vision_node_batches_requests_and_hands_each_its_own_embeddings(self):
        from mstar.model.command_a_plus.tests.test_vision import tiny_encoder

        torch.manual_seed(0)
        node = CommandAPlusVisionSubmodule(tiny_encoder()).eval()
        node.TILES_PER_CALL = 2  # a chunk straddles the two requests
        tiles = {"b": torch.rand(3, 3, 32, 32), "a": torch.rand(1, 3, 32, 32)}
        inputs = [node.prepare_inputs("prefill_image", info("prefill_image"), {"pixel_values": [t]})
                  for t in tiles.values()]
        engine = ModelInputsFromEngine(list(tiles), {})
        self.assertTrue(node.can_batch(None, inputs))
        with torch.inference_mode():
            batched = node.forward_batched("prefill_image", engine, **node.preprocess("prefill_image", engine, inputs))
            alone = [node.forward("prefill_image", None, **node.preprocess("prefill_image", None, [inp]))
                     for inp in inputs]
        self.assertEqual(list(batched), ["b", "a"])
        for rid, single in zip(tiles, alone, strict=True):
            self.assertEqual(len(batched[rid]["image_embeds"][0]), len(tiles[rid]) * 4)
            torch.testing.assert_close(batched[rid]["image_embeds"][0], single["image_embeds"][0])

    def test_load_dtype_and_reject_sequence_parallelism(self):
        save_file(checkpoint(self.model.config), Path(self.directory) / "model.safetensors")
        node = self.model.get_submodule("LLM", autocast_dtype=torch.bfloat16)
        self.assertTrue(all(p.dtype == torch.bfloat16 and not p.is_meta for p in node.parameters()))
        self.assertFalse(node.training)
        with self.assertRaisesRegex(ValueError, "sequence parallelism"):
            self.model.get_submodule("LLM", sp_group=CommGroup(0, 0, [0, 1]))
        with self.assertRaisesRegex(ValueError, "local checkpoint"):
            CommandAPlusModel("unavailable/repository")

    def test_two_walks_construct_one_tp8_node_group(self):
        import yaml

        path = Path(self.directory) / "deployment.yaml"
        path.write_text(yaml.safe_dump({"node_groups": [
            {"node_names": ["vision_encoder"], "ranks": [0]},
            {"node_names": ["LLM"], "ranks": list(range(8)), "tp_size": 8,
             "graph_walks": ["prefill", "prefill_image", "decode"]},
        ]}))
        graphs = self.model.get_worker_graphs(str(path))
        self.assertEqual({next(iter(g.graph_walks)) for g in graphs}, {"prefill", "prefill_image", "decode"})
        llm = [g for g in graphs if "LLM" in g.section.get_nodes()]
        self.assertTrue(llm and all(g.tp_size == 8 and g.sp_size == 1 for g in llm))
        self.assertEqual(len(self.model.get_sharding_config(str(path)).groups), 1)

    def test_prompt_adapter_and_byte_stream_preserve_unicode_and_markers(self):
        # Two tokens split the UTF-8 bytes for é. Ġ is the byte-level space.
        vocab = {10: "Ã", 11: "©", 12: "Ġhi", 13: "<|START_THINKING|>"}
        def convert(ids):
            return vocab[ids] if isinstance(ids, int) else [vocab[i] for i in ids]
        tokenizer = SimpleNamespace(
            all_special_ids=[0, 2, 3, 13], convert_ids_to_tokens=convert,
            apply_chat_template=Mock(return_value=[2, 10, 11]),
            backend_tokenizer=SimpleNamespace(decoder=SimpleNamespace(
                __getstate__=lambda: json.dumps({"type": "ByteLevel"}).encode(),
            )),
        )
        self.model._tokenizer = tokenizer
        # Room for the default output cap, which counts toward max_seq_len.
        self.model.max_seq_len = MAX_OUTPUT_TOKENS + 8
        tensors, metadata = self.model.process_prompt("hello", ["text"], ["text"])
        self.assertEqual(tensors["text_inputs"][0].tolist(), [2, 10, 11])
        layout = metadata["kv_layout"]["kv_cache"]["main"]
        self.assertEqual([span.length for span in layout.spans], [3])
        self.assertEqual(layout.growth, (("decode_loop", 1),))
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "hello"}], tokenize=True,
            add_generation_prompt=True, return_dict=False,
        )
        chunks = [self.model.postprocess(torch.tensor([i]), "text") for i in (10, 11, 12, 13, 3)]
        self.assertEqual(b"".join(chunks).decode("utf-8"), "é hi<|START_THINKING|>")
        turns = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"},
                 {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "again"}]
        tokenizer.apply_chat_template.reset_mock()
        self.model.process_prompt("again", ["text"], ["text"], messages=turns)
        tokenizer.apply_chat_template.assert_called_once_with(
            turns, tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        self.model.process_prompt("hi", ["text"], ["text"], max_output_tokens=self.model.max_seq_len - 3)
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            self.model.process_prompt("hi", ["text"], ["text"], max_output_tokens=self.model.max_seq_len - 2)
        # Without max_output_tokens the default cap still counts.
        with patch.object(self.model, "max_seq_len", MAX_OUTPUT_TOKENS + 2), \
                self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            self.model.process_prompt("hi", ["text"], ["text"])
        for bad in ([], [{"role": "tool", "content": "x"}], [{"role": "user"}], "hi"):
            with self.assertRaisesRegex(ValueError, "messages"):
                self.model.process_prompt("hi", ["text"], ["text"], messages=bad)
        with self.assertRaisesRegex(ValueError, "text and image input"):
            self.model.process_prompt("hi", ["audio"], ["text"])
        for ids in (torch.tensor([]), torch.tensor([128]), torch.tensor([-1]), torch.tensor([2.0])):
            with self.assertRaises(ValueError):
                self.model.process_prompt(None, ["text"], ["text"], tensors={"text_inputs": [ids]})


    def test_images_widen_their_placeholders_in_the_order_sent(self):
        image, start, end, line_break = 120, 117, 118, 119
        tokenizer = SimpleNamespace(
            apply_chat_template=Mock(return_value=[2, image, 11, image]),
            convert_tokens_to_ids=Mock(return_value=[start, end, line_break]),
        )
        tiles = [torch.rand(2, 3, 32, 32), torch.rand(1, 3, 32, 32)]
        processor = Mock(side_effect=lambda images, return_tensors: {"pixel_values": tiles[processor.call_count - 1]})
        self.model._tokenizer, self.model._image_processor = tokenizer, processor
        self.model.max_seq_len = MAX_OUTPUT_TOKENS + 64
        photos = [torch.rand(3, 40, 60), torch.rand(3, 20, 20)]
        parts = [PromptPart(modality="image", index=0), PromptPart(modality="text", text="hi"),
                 PromptPart(modality="image", index=1)]
        result, metadata = self.model.process_prompt(
            None, ["image", "text", "image"], ["text"], tensors={"image_inputs": photos}, prompt_parts=parts,
        )
        self.assertEqual(tokenizer.apply_chat_template.call_args.args[0], [{"role": "user", "content": [
            {"type": "image", "image": ""}, {"type": "text", "text": "hi"}, {"type": "image", "image": ""},
        ]}])
        tile = [image] * 4 + [line_break]
        self.assertEqual(result["text_inputs"][0].tolist(), [2, start, *tile * 2, end, 11, start, *tile, end])
        torch.testing.assert_close(result["pixel_values"], [t.to(torch.bfloat16) for t in tiles])
        (span,) = metadata["kv_layout"]["kv_cache"]["main"].spans
        self.assertEqual((span.kind, span.walk, span.length), ("rows", "prefill_image", len(result["text_inputs"][0])))
        # one run of pads a tile, and the node splices the tower's rows into exactly those
        layout = metadata[MEDIA_LAYOUT]
        self.assertEqual((layout["offset"], layout["tokens"]), ([2, 7, 15], [4, 4, 4]))
        node = toy_node(self.model)
        ids, rows = result["text_inputs"][0], torch.randn(12, self.model.config.vocab_size)
        engine = ModelInputsFromEngine(["a"], {}, {
            LOCAL_ATTN: SimpleNamespace(select_last_hidden=lambda x: x),
            SAMPLER: SimpleNamespace(sample=lambda rids, logits: logits),
        })
        inputs = node.prepare_inputs(
            "prefill_image", info("prefill_image", layout=layout), {"text_inputs": [ids], "image_embeds": [rows]},
        )
        hidden = node.forward_batched(
            "prefill_image", engine, **node.preprocess("prefill_image", engine, [inputs]),
        ).row_outputs["new_token"]
        pads = ids == image
        torch.testing.assert_close(hidden[pads], rows)
        torch.testing.assert_close(hidden[~pads], torch.eye(self.model.config.vocab_size)[ids[~pads]])
        torch.testing.assert_close(processor.call_args_list[0].kwargs["images"][0],
                                   (photos[0] * 255).round().to(torch.uint8))
        with self.assertRaisesRegex(ValueError, "2 image placeholders for 1 images"):
            self.model.process_prompt(None, ["image"], ["text"], tensors={"image_inputs": photos[:1]},
                                      messages=[{"role": "user", "content": [{"type": "image"}] * 2}])

    def test_overlong_image_prompt_stops_before_the_next_image(self):
        self.model._tokenizer = SimpleNamespace(
            apply_chat_template=Mock(return_value=[2, 120, 11, 120]),
            convert_tokens_to_ids=Mock(return_value=[117, 118, 119]),
        )
        processor = Mock(return_value={"pixel_values": torch.rand(2, 3, 32, 32)})
        self.model._image_processor = processor
        # 2 text tokens plus the first image's 12 already overflow
        self.model.max_seq_len = MAX_OUTPUT_TOKENS + 13
        parts = [PromptPart(modality="image", index=0), PromptPart(modality="image", index=1)]
        with self.assertRaisesRegex(ValueError, r"at least 14 tokens.*exceeds max_seq_len"):
            self.model.process_prompt(None, ["image", "image"], ["text"],
                                      tensors={"image_inputs": [torch.rand(3, 8, 8)] * 2}, prompt_parts=parts)
        self.assertEqual(processor.call_count, 1)

if __name__ == "__main__":
    unittest.main()
