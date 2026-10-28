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
from mstar.graph.base import GraphEdge
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import MAX_OUTPUT_TOKENS
from mstar.model.command_a_plus.command_a_plus_model import CommandAPlusModel
from mstar.model.command_a_plus.config import (
    GLOBAL_ATTN,
    KV_CACHE,
    LOCAL_ATTN,
    ROPE,
    SAMPLER,
    CommandAPlusConfig,
)
from mstar.model.command_a_plus.submodules import CommandAPlusLLMSubmodule, CommandAPlusVisionSubmodule
from mstar.model.command_a_plus.tests.test_backbone import bind_test_resources
from mstar.model.command_a_plus.tests.test_weight_loading import checkpoint
from mstar.model.multimodal import PromptPart
from mstar.model.submodule_base import ModelInputsFromEngine

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_config.json"


def info(walk="prefill", max_tokens=6, ignore_eos=False, index=0):
    return CurrentForwardPassInfo(
        request_id="a", graph_walk=walk, fwd_index=0, random_seed=0, max_tokens=max_tokens,
        resource_configs={SAMPLER: SamplingReqConfig(ignore_eos=ignore_eos)},
        dynamic_loop_iter_counts={"decode_loop": index} if walk == "decode" else {},
    )


class ToyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding.from_pretrained(torch.eye(vocab))
        self.model.embed = self.model.embed_tokens

    def forward(self, x, *, label):
        assert label == "main"
        return x

    def compute_logits(self, x):
        return x


class NodeTests(unittest.TestCase):
    def setUp(self):
        config = CommandAPlusConfig.from_json(FIXTURE)
        self.config = config.text_config
        self.node = CommandAPlusLLMSubmodule(
            ToyLM(self.config.vocab_size), self.config, config.vision_config.image_token_id,
        )

    def test_packed_prefill_selects_each_requests_last_token_and_decode_routes_rows(self):
        sampler = SimpleNamespace(sample=Mock(side_effect=lambda rids, logits: logits.argmax(-1)))
        attention = SimpleNamespace(select_last_hidden=Mock(side_effect=lambda x, label: x[[2, 4]]))
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
            self.assertEqual([output[rid]["new_token"][0].item() for rid in engine.request_ids], expected)
            self.assertEqual(sampler.sample.call_args.args[0], engine.request_ids)
        attention.select_last_hidden.assert_called_once()

    def test_prefill_publishes_a_decode_seed_without_reading_the_token(self):
        for maximum, continues in ((1, False), (2, True)):
            output = {"new_token": [torch.tensor([3])]}
            self.node.postprocess("a", info(max_tokens=maximum), output)
            self.assertEqual("decode_input" in output, continues)
            self.assertIn("new_token", output)

    def test_eos_from_prefill_ends_the_reply_at_the_first_decode_step(self):
        for ignore_eos, ends in ((False, True), (True, False)):
            with self.subTest(ignore_eos=ignore_eos):
                prefill = {"new_token": [torch.tensor([3])]}
                self.assertEqual(self.node.check_stop("a", info(ignore_eos=ignore_eos), prefill), set())
                request = info("decode", ignore_eos=ignore_eos)
                decode = {"new_token": [torch.tensor([7])]}
                self.node.postprocess("a", request, decode)
                self.assertEqual("new_token" not in decode, ends)
                self.assertEqual(self.node.check_stop("a", request, decode), {"decode_loop"} if ends else set())
        self.assertEqual(self.node._ended_at_prefill, set())
        self.node.check_stop("a", info(), {"new_token": [torch.tensor([3])]})
        self.node.cleanup_request("a")
        self.assertEqual(self.node._ended_at_prefill, set())

    def test_decode_stop_counts_prefill_token_and_respects_ignore_eos(self):
        for token, maximum, index, ignore_eos, stops in (
            (3, 5, 0, False, True), (3, 5, 0, True, False),
            (7, 2, 0, False, True), (7, 3, 0, False, False),
            (7, 3, 1, False, True),
        ):
            with self.subTest(token=token, maximum=maximum, index=index, ignore_eos=ignore_eos):
                request = info("decode", maximum, ignore_eos, index)
                output = {"new_token": [torch.tensor([token])]}
                self.node.postprocess("a", request, output)
                self.assertIs(output["text_inputs"], output["new_token"])
                self.assertEqual(self.node.check_stop("a", request, output), {"decode_loop"} if stops else set())
        self.node.request_state("a").kwargs["example"] = True
        self.node.cleanup_request("a")
        self.assertNotIn("a", self.node.request_states)

    def test_image_prefill_places_encoder_output_at_image_tokens(self):
        image = self.node.image_token_id
        ids = torch.tensor([2, image, image, 5])
        image_embeds = torch.stack([torch.arange(128.0), -torch.arange(128.0)]) / 128
        inputs = self.node.prepare_inputs(
            "prefill_image", info("prefill_image"), {"text_inputs": [ids], "image_embeds": [image_embeds]},
        )
        eye = torch.eye(self.config.vocab_size)
        torch.testing.assert_close(inputs.input_embeds[[0, 3]], eye[[2, 5]])
        torch.testing.assert_close(inputs.input_embeds[[1, 2]], image_embeds)
        # a chunked prefill cuts the embeddings with the ids
        cut = self.node.split_inputs("prefill_image", info("prefill_image"), inputs, 1, 3)
        torch.testing.assert_close(cut.input_embeds, image_embeds)
        self.assertEqual(cut.input_ids.tolist(), [image, image])

        sampler = SimpleNamespace(sample=Mock(side_effect=lambda rids, logits: logits.argmax(-1)))
        attention = SimpleNamespace(select_last_hidden=lambda x, label: x[-1:])
        engine = ModelInputsFromEngine(["a"], {}, {LOCAL_ATTN: attention, SAMPLER: sampler})
        step = self.node.declare_step("prefill_image", ["a"], [inputs])
        self.assertEqual(step.steps[SAMPLER].prefill_tracked_tokens["a"].tolist(), ids.tolist())
        output = self.node.forward_batched(
            "prefill_image", engine, **self.node.preprocess("prefill_image", engine, [inputs]),
        )
        self.assertEqual(output["a"]["new_token"][0].item(), 5)

    def test_invalid_inputs_fail_before_cache_planning(self):
        for walk, tensor in (
            ("other", torch.tensor([2])), ("decode", torch.tensor([2, 3])),
            ("prefill", torch.tensor([], dtype=torch.long)),
            ("prefill", torch.tensor([[2, 3]])), ("prefill", torch.tensor([2.0])),
        ):
            with self.subTest(walk=walk, shape=tensor.shape):
                with self.assertRaises(ValueError):
                    self.node.prepare_inputs(walk, info(walk), {"text_inputs": [tensor]})


class ModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        shutil.copyfile(FIXTURE, Path(self.directory) / "config.json")
        self.model = CommandAPlusModel(self.directory)

    def test_graph_resources_and_prefill_decode_transition(self):
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
        output_edges = walks["prefill"].outputs
        self.assertTrue(any(e.name == "new_token" and e.next_node == EMIT_TO_CLIENT and e.conductor_new_token
                            for e in output_edges))
        self.assertTrue(any(e.name == "decode_input" and e.next_node == EMPTY_DESTINATION and e.persist
                            for e in output_edges))
        self.assertEqual(walks["decode"].name, "decode_loop")
        seed = object()  # Metadata only; no tensor transfer in this test.
        first = model.get_initial_forward_pass_args(DEFAULT_PARTITION, ["text"], ["text"], {"text_inputs": [seed]})
        self.assertEqual(first.full_metadata.graph_walk, "prefill")
        next_ = model.get_partition_forward_pass_args(DEFAULT_PARTITION, first.full_metadata, {"decode_input": [seed]})
        self.assertEqual(next_.full_metadata.graph_walk, "decode")
        self.assertEqual(next_.inputs[0].tensor_info, [seed])
        self.assertEqual(next_.unpersist_tensors, [seed])
        self.assertTrue(model.get_partition_forward_pass_args(DEFAULT_PARTITION, next_.full_metadata, {}).request_done)
        first = model.get_initial_forward_pass_args(DEFAULT_PARTITION, ["text"], ["text"], {"text_inputs": [seed]})
        self.assertTrue(model.get_partition_forward_pass_args(DEFAULT_PARTITION, first.full_metadata, {}).request_done)
        cfg = model.get_request_resource_configs({}, {"temperature": 0})[SAMPLER]
        self.assertEqual((cfg.temperature, cfg.top_p, cfg.repetition_penalty), (0, 0.95, 1.04))

    def test_local_checkpoint_node_runs_prefill_and_four_decode_steps(self):
        save_file(checkpoint(self.model.config), Path(self.directory) / "model.safetensors")
        node = self.model.get_submodule("LLM", autocast_dtype=torch.float32)
        kv, pos, local, global_ = bind_test_resources(node.language_model.model, self.model.config)
        local.select_last_hidden = lambda hidden, label: hidden[-1:]
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
                result = node.forward_batched(walk, engine, **node.preprocess(walk, engine, inputs))["a"]
                node.postprocess("a", request, result)
                tokens = result["decode_input" if step == 0 else "text_inputs"][0]
                self.assertEqual(tokens.shape, (1,))
        self.assertTrue(all(len(k) == 9 for k, v in kv.cache.values()))
        self.assertEqual(len(kv.writes), 20)

    def test_image_requests_take_the_image_walk(self):
        text, pixels = object(), object()
        first = self.model.get_initial_forward_pass_args(
            DEFAULT_PARTITION, ["text", "image"], ["text"], {"text_inputs": [text], "pixel_values": [pixels]},
        )
        self.assertEqual(first.full_metadata.graph_walk, "prefill_image")
        self.assertEqual({(e.next_node, e.name) for e in first.inputs},
                         {("LLM", "text_inputs"), ("vision_encoder", "pixel_values")})
        self.assertEqual(first.unpersist_tensors, [text, pixels])
        next_ = self.model.get_partition_forward_pass_args(
            DEFAULT_PARTITION, first.full_metadata, {"decode_input": [text]},
        )
        self.assertEqual(next_.full_metadata.graph_walk, "decode")

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

    def test_real_graph_loop_stops_after_total_token_budget(self):
        graph = WorkerGraphIO(self.model.get_graph_walk_graphs()["decode"])
        node = CommandAPlusLLMSubmodule(
            ToyLM(self.model.config.vocab_size), self.model.config, self.model.vision_config.image_token_id,
        )
        graph.ingest_input(GraphEdge(name="text_inputs", next_node="LLM"))
        iterations = 0
        while not graph.wg_state_registry.is_done and iterations < 10:
            self.assertIn("LLM", graph.ready_node_names)
            graph.ready_node_names.discard("LLM")
            request = info("decode", max_tokens=3, index=graph.get_loop_indices()["decode_loop"])
            for loop in node.check_stop("a", request, {"new_token": [torch.tensor([7])]}):
                graph.register_loop_finish_signal(loop)
            completion = graph.mark_node_complete("LLM")
            for edge in completion.output_edges:
                if edge.next_node == "LLM" and (edge.name, edge.next_node) not in completion.filtered_signals:
                    graph.ingest_input(edge)
            iterations += 1
        self.assertTrue(graph.wg_state_registry.is_done)
        self.assertEqual(iterations, 2)  # One prefill token + two decode tokens.

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
