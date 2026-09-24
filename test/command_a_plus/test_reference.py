"""Opt-in checks against Transformers 5.17 and actual GPU engine resources.

MSTAR_TEST_HF=1 enables the CPU reference; MSTAR_TEST_CUDA=1 also enables
the tiny GPU model. No checkpoint weights or metadata are downloaded here.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from mstar.model.command_a_plus.command_a_plus_model import CommandAPlusModel
from mstar.model.command_a_plus.components.language_model import CommandAPlusForCausalLM
from mstar.model.command_a_plus.config import KV_CACHE, SAMPLER, CommandAPlusConfig
from mstar.model.command_a_plus.submodules import CommandAPlusLLMSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine
from test.command_a_plus.test_backbone import bind_test_resources
from test.command_a_plus.test_integration import FIXTURE, info


def reference_weights(reference):
    """Expand HF's runtime fused experts to the published checkpoint layout."""
    for name, tensor in reference.state_dict().items():
        if name == "lm_head.weight":
            continue
        name = name.replace("model.", "model.language_model.", 1)
        if name.endswith("experts.gate_up_proj"):
            prefix = name.removesuffix("gate_up_proj")
            for index, expert in enumerate(tensor):
                gate, up = expert.chunk(2, dim=0)
                yield f"{prefix}{index}.gate_proj.weight", gate
                yield f"{prefix}{index}.up_proj.weight", up
        elif name.endswith("experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            for index, expert in enumerate(tensor):
                yield f"{prefix}{index}.down_proj.weight", expert
        else:
            yield name, tensor


def make_pair(gpu=False, comm_group=None):
    from transformers import Cohere2MoeConfig, Cohere2MoeForCausalLM

    raw = json.loads(FIXTURE.read_text())
    if gpu:
        # FlashInfer supports head_dim=128; keep parameters tiny independently
        # of Q width. Real Command A+ also has hidden_size != Q projection width.
        raw["text_config"].update(hidden_size=64, head_dim=128, intermediate_size=64)
    config = CommandAPlusConfig.from_dict(raw).text_config
    hf_config = Cohere2MoeConfig(**raw["text_config"])
    hf_config._attn_implementation = "eager"
    torch.manual_seed(123)
    dtype, device = (torch.bfloat16, "cuda") if gpu else (torch.float32, "cpu")
    reference = Cohere2MoeForCausalLM(hf_config).to(device=device, dtype=dtype).eval()
    actual = CommandAPlusForCausalLM(config, comm_group).to(device=device, dtype=dtype).eval()
    actual.load_weights(reference_weights(reference))
    return raw, reference, actual


def observe(model):
    seen, handles = {}, []
    def hook(key):
        def capture(module, args, output):
            seen[key] = output.detach().reshape(-1, output.shape[-1]).clone()
        return capture
    for name, module in [("embedding", model.model.embed_tokens),
                         *[(f"layer_{i}", layer) for i, layer in enumerate(model.model.layers)],
                         ("norm", model.model.norm)]:
        handles.append(module.register_forward_hook(hook(name)))
    return seen, handles


@unittest.skipUnless(os.environ.get("MSTAR_TEST_HF") == "1", "set MSTAR_TEST_HF=1")
class ReferenceTests(unittest.TestCase):
    def check_routers(self, actual, reference):
        # Isolate routing from upstream BF16 rounding: identical router inputs
        # must select the same expert IDs and produce the same weights.
        generator = torch.Generator().manual_seed(902)
        x = torch.randn(23, actual.config.hidden_size, generator=generator).to(
            device=actual.model.embed_tokens.weight.device,
            dtype=actual.model.embed_tokens.weight.dtype,
        )
        for ours, theirs in zip(actual.model.layers, reference.model.layers, strict=True):
            weights, indices, _ = ours.mlp.gate(x)
            _, expected_weights, expected_indices = theirs.mlp.gate(x)
            torch.testing.assert_close(indices, expected_indices, atol=0, rtol=0)
            torch.testing.assert_close(weights, expected_weights, atol=0, rtol=0)

    def compare(self, actual, expected, errors, tolerance):
        self.assertEqual(actual.keys(), expected.keys())
        for name in actual:
            delta = (actual[name].float() - expected[name].float()).abs().max().item()
            errors[name] = max(errors.get(name, 0), delta)
            torch.testing.assert_close(actual[name], expected[name], atol=tolerance, rtol=tolerance,
                                       msg=lambda msg, name=name: f"{name}: {msg}")

    @torch.inference_mode()
    def test_fp32_prefill_cached_decode_layer_and_logit_parity(self):
        _, reference, actual = make_pair()
        self.check_routers(actual, reference)
        _, pos, _, _ = bind_test_resources(actual.model, actual.config)
        observed, handles = observe(actual)
        expected, ref_handles = observe(reference)
        self.addCleanup(lambda: [h.remove() for h in handles + ref_handles])
        tokens = torch.tensor([2, 17, 8, 91, 6, 11, 57, 22, 35])
        cache, errors = None, {}
        for start, stop in [(0, 5), (5, 6), (6, 7), (7, 8), (8, 9)]:
            pos.positions["main"] = torch.arange(start, stop)
            hidden = actual(actual.model.embed_tokens(tokens[start:stop]), label="main")
            result = reference(tokens[None, start:stop], past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            observed["logits"] = actual.compute_logits(hidden)
            expected["logits"] = result.logits.squeeze(0)
            self.compare(observed, expected, errors, 2e-5)
            torch.testing.assert_close(observed["logits"].argmax(-1), expected["logits"].argmax(-1))
        print("HF FP32 maximum absolute errors:", json.dumps(errors, sort_keys=True), flush=True)

    @unittest.skipUnless(os.environ.get("MSTAR_TEST_CUDA") == "1", "set MSTAR_TEST_CUDA=1")
    @torch.inference_mode()
    def test_bf16_paged_gpu_batched_prefill_decode_and_cleanup(self):
        from mstar.communication.tensors import LocalTransferEngine
        from mstar.distributed.communication import CommGroup, JointGroups
        from mstar.engine.resources import SamplingReqConfig
        from mstar.engine.resources.base import EngineResourceInfo, build_resource
        from mstar.engine.resources.kv.transfer import TransferEngineInfo
        from mstar.engine.resources.runner import StepRunner
        from mstar.engine.resources.step import StepContext

        group = None
        if os.environ.get("MSTAR_TEST_TP") == "1":
            import torch.distributed as dist
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group("nccl")
            self.addCleanup(dist.destroy_process_group)
            group = CommGroup(dist.get_rank(), dist.get_rank(), list(range(dist.get_world_size())))
            group.device_group = dist.group.WORLD
        raw, reference, actual = make_pair(gpu=True, comm_group=group)
        self.check_routers(actual, reference)
        directory = self.enterContext(tempfile.TemporaryDirectory())
        (Path(directory) / "config.json").write_text(json.dumps(raw))
        model = CommandAPlusModel(directory)
        specs = {s.resource_key: s for s in model.get_node_resources()}
        specs[KV_CACHE].config.page_size = 16
        specs[KV_CACHE].config.max_num_pages = 16
        transfer = TransferEngineInfo("reference", "reference", LocalTransferEngine("localhost"))
        resources = {key: build_resource(spec, EngineResourceInfo(
            device=torch.device("cuda"), kv_dtype=torch.bfloat16,
            transfer_engine_info=transfer,
            joint_comm_group=JointGroups(group, CommGroup.trivial()) if group else None,
            dependencies={dependency: specs[dependency] for dependency in spec.depends_on()},
        )) for key, spec in specs.items()}
        runner = StepRunner(resources)
        self.addCleanup(lambda: [resource.cleanup() for resource in resources.values()])
        node = CommandAPlusLLMSubmodule(actual, actual.config).eval()
        node.bind_node_resources(resources)
        observed, handles = observe(actual)
        expected, ref_handles = observe(reference)
        self.addCleanup(lambda: [h.remove() for h in handles + ref_handles])
        errors, first_logits = {}, None
        near_ties = 0
        initial_free_pages = resources[KV_CACHE]._arena.num_free
        # Two independent lifetimes exercise stale cache/position/sampler state.
        for _lifetime in range(2):
            rids = ["long", "short"]
            for rid in rids:
                runner.ingest_request(rid, {SAMPLER: SamplingReqConfig(
                    temperature=0, repetition_penalty=1, top_p=1, ignore_eos=True,
                )})
            try:
                rows = [torch.arange(2, 21, device="cuda"), torch.arange(31, 40, device="cuda")]
                caches = [None, None]
                engine = ModelInputsFromEngine(rids, {}, resources)
                for iteration in range(5):
                    walk = "prefill" if iteration == 0 else "decode"
                    inputs = [node.prepare_inputs(walk, info(walk, max_tokens=20, ignore_eos=True),
                              {"text_inputs": [row]}) for row in rows]
                    step = node.declare_step(walk, rids, inputs)
                    step.set_ctx(StepContext(rids, walk, slot=iteration % 2, capture=False))
                    admitted = runner.admit(step)
                    self.assertTrue(admitted.ok and admitted.ready, admitted)
                    runner.plan(step)
                    output = node.forward_batched(walk, engine, **node.preprocess(walk, engine, inputs))
                    runner.commit(step)
                    per_request, ref_logits = [], []
                    for index, row in enumerate(rows):
                        result = reference(row[None], past_key_values=caches[index], use_cache=True)
                        caches[index] = result.past_key_values
                        per_request.append(dict(expected))
                        ref_logits.append(result.logits[0, -1])
                    combined = {key: torch.cat([item[key] for item in per_request]) for key in expected}
                    self.compare(observed, combined, errors, 0.03)
                    ends = torch.tensor([len(rows[0])-1, sum(map(len, rows))-1], device="cuda")
                    logits = actual.compute_logits(observed["norm"][ends])
                    ref_logits = torch.stack(ref_logits)
                    self.compare({"logits": logits}, {"logits": ref_logits}, errors, 0.01)
                    predicted = torch.cat([output[rid]["new_token"][0] for rid in rids])
                    torch.testing.assert_close(predicted.long(), logits.argmax(-1))
                    for row, token in enumerate(predicted.long()):
                        if token != ref_logits[row].argmax():
                            near_ties += 1
                            # Argmax is discontinuous: a winner can flip when
                            # the gap is within twice the measured logit error.
                            # Always enforce logit parity above, and allow no
                            # disagreements outside this numerical bound.
                            error = (logits[row].float() - ref_logits[row].float()).abs().max()
                            gap = ref_logits[row].max().float() - ref_logits[row, token].float()
                            self.assertLessEqual(gap.item(), (2 * error).item())
                    if iteration == 0:
                        if first_logits is None:
                            first_logits = logits.clone()
                        else:
                            torch.testing.assert_close(logits, first_logits, atol=0, rtol=0)
                    rows = [output[rid]["new_token"][0] for rid in rids]
                    torch.cuda.synchronize()
            finally:
                for rid in rids:
                    runner.remove_request(rid)
                    node.cleanup_request(rid)
            self.assertEqual(resources[KV_CACHE]._arena.num_free, initial_free_pages)
        print("HF BF16 GPU maximum absolute errors:", json.dumps(errors, sort_keys=True), flush=True)
        print("Greedy disagreements within measured rounding bound:", near_ties, "of 20", flush=True)


@unittest.skipUnless(os.environ.get("COMMAND_A_METADATA_DIR"), "set COMMAND_A_METADATA_DIR to local metadata")
class OfficialTokenizerTests(unittest.TestCase):
    def test_official_chat_template_and_utf8_streaming(self):
        model = CommandAPlusModel(os.environ["COMMAND_A_METADATA_DIR"])
        tokenizer = model._get_tokenizer()
        for prompt in ("Hello", "é 🧠 你好 مرحبا", "<|START_THINKING|>test<|END_THINKING|>"):
            with self.subTest(prompt=prompt):
                expected = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=True,
                    add_generation_prompt=True, return_dict=False,
                )
                self.assertEqual(model.process_prompt(prompt, ["text"], ["text"])["text_inputs"][0].tolist(), expected)
                tokens = tokenizer.encode(prompt, add_special_tokens=False)
                streamed = b"".join(model.postprocess(torch.tensor([i]), "text") for i in tokens)
                self.assertEqual(streamed.decode("utf-8"), prompt)


if __name__ == "__main__":
    unittest.main()
