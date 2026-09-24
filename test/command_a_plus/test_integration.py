"""Offline graph/node integration checks; attention and sampling use CPU doubles."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from safetensors.torch import save_file
from torch import nn

from mstar.conductor.request_info import DEFAULT_PARTITION, CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup
from mstar.engine.resources import AttentionStep, SamplingReqConfig
from mstar.graph.base import GraphEdge
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.command_a_plus.command_a_plus_model import CommandAPlusModel
from mstar.model.command_a_plus.config import (
    GLOBAL_ATTN,
    KV_CACHE,
    LOCAL_ATTN,
    ROPE,
    SAMPLER,
    CommandAPlusConfig,
)
from mstar.model.command_a_plus.submodules import CommandAPlusLLMSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine
from test.command_a_plus.test_backbone import bind_test_resources
from test.command_a_plus.test_weight_loading import checkpoint

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

    def forward(self, x, *, label):
        assert label == "main"
        return x

    def compute_logits(self, x):
        return x


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.config = CommandAPlusConfig.from_json(FIXTURE).text_config
        self.node = CommandAPlusLLMSubmodule(ToyLM(self.config.vocab_size), self.config)

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

    def test_prefill_eos_and_one_token_budget_do_not_publish_decode_seed(self):
        for token, maximum, ignore_eos, continues in (
            (3, 5, False, False), (3, 5, True, True),
            (7, 1, False, False), (7, 2, False, True),
        ):
            with self.subTest(token=token, maximum=maximum, ignore_eos=ignore_eos):
                output = {"new_token": [torch.tensor([token])]}
                self.node.postprocess("a", info(max_tokens=maximum, ignore_eos=ignore_eos), output)
                self.assertEqual("decode_input" in output, continues)
                self.assertIn("new_token", output)

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
        self.assertEqual(model.nodes, ["LLM"])
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
        node = CommandAPlusLLMSubmodule(ToyLM(self.model.config.vocab_size), self.model.config)
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
        path.write_text(yaml.safe_dump({"node_groups": [{
            "node_names": ["LLM"], "ranks": list(range(8)), "tp_size": 8,
            "graph_walks": ["prefill", "decode"],
        }]}))
        graphs = self.model.get_worker_graphs(str(path))
        self.assertEqual(len(graphs), 2)
        self.assertEqual({next(iter(g.graph_walks)) for g in graphs}, {"prefill", "decode"})
        self.assertTrue(all(g.tp_size == 8 and g.sp_size == 1 for g in graphs))
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
        result = self.model.process_prompt("hello", ["text"], ["text"])
        self.assertEqual(result["text_inputs"][0].tolist(), [2, 10, 11])
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "hello"}], tokenize=True,
            add_generation_prompt=True, return_dict=False,
        )
        chunks = [self.model.postprocess(torch.tensor([i]), "text") for i in (10, 11, 12, 13, 3)]
        self.assertEqual(b"".join(chunks).decode("utf-8"), "é hi<|START_THINKING|>")
        with self.assertRaisesRegex(ValueError, "text input"):
            self.model.process_prompt("hi", ["image"], ["text"])
        for ids in (torch.tensor([]), torch.tensor([128]), torch.tensor([-1]), torch.tensor([2.0])):
            with self.assertRaises(ValueError):
                self.model.process_prompt(None, ["text"], ["text"], tensors={"text_inputs": [ids]})


if __name__ == "__main__":
    unittest.main()
