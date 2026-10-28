"""The generic text LM on the GPU against HF's ``Qwen3MoeForCausalLM``: a tiny random checkpoint
written to disk, loaded through ``TextLMModel``, and stepped over the node's real resources
(paged KV, FlashInfer attention, RoPE positions, the sampler) the way the engine steps it."""
import json

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.accelerator_graph_runner import autocast_scope
from mstar.engine.resources import StepContext, StepRunner, apply_yaml_overrides, resolve_spec_dependencies
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.text_lm.config import ATTN, DECODE_WALK, KV_CACHE, LLM_NODE, PREFILL_WALK

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (FlashInfer paged attention)")
transformers = pytest.importorskip("transformers")

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
# Qwen3-Coder-30B-A3B's attention geometry (GQA 32/4, head 128, theta 1e7) at 2 layers and
# a narrow MoE: the kernels see the real head layout, the test stays small.
TINY = dict(
    hidden_size=256, num_hidden_layers=2, num_attention_heads=32, num_key_value_heads=4, head_dim=128,
    intermediate_size=512, moe_intermediate_size=128, num_experts=16, num_experts_per_tok=4,
    norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[], vocab_size=512, rms_norm_eps=1e-6,
    rope_theta=1e7, max_position_embeddings=4096, tie_word_embeddings=False,
)


def _checkpoint(tmp_path):
    """HF's model at random weights, its lm_head scaled so greedy choices are not near-ties,
    saved under the checkpoint's own key names (one tensor per expert)."""
    from safetensors.torch import save_file
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    hf = Qwen3MoeForCausalLM(Qwen3MoeConfig(**TINY, attn_implementation="eager")).eval()
    hf.lm_head.weight.data *= 8
    state = {}
    for key, value in hf.state_dict().items():
        if key.endswith(".experts.gate_up_proj"):  # transformers 5 stacks the experts
            gate, up = value.chunk(2, dim=1)
            for e in range(value.shape[0]):
                state[key.replace("gate_up_proj", f"{e}.gate_proj.weight")] = gate[e]
                state[key.replace("gate_up_proj", f"{e}.up_proj.weight")] = up[e]
        elif key.endswith(".experts.down_proj"):
            for e in range(value.shape[0]):
                state[key.replace("down_proj", f"{e}.down_proj.weight")] = value[e]
        else:
            state[key] = value
    save_file({k: v.detach().to(DTYPE).contiguous() for k, v in state.items()}, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({**TINY, "model_type": "qwen3_moe"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"do_sample": False, "eos_token_id": [7]}))
    # the reference runs in fp32 from the same bf16-rounded weights
    return hf.to(DTYPE).float().to(DEVICE)


class _Node:
    """The engine's work around the submodule for one step at a time: admit, plan, forward,
    commit over the node's real resources."""

    def __init__(self, model, submodule, rids):
        specs = model.get_node_resources()
        apply_yaml_overrides(specs, {"resources": {KV_CACHE: {"max_num_pages": 16, "page_size": 16}}})
        by_key = resolve_spec_dependencies(specs)
        groups = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial())
        transfer = TransferEngineInfo(
            my_entity_id="text_lm_gpu", my_session_id="text_lm_gpu", transfer_engine=LocalTransferEngine("localhost"),
        )
        self.resources = {
            spec.resource_key: build_resource(spec, EngineResourceInfo(
                device=DEVICE, joint_comm_group=groups, transfer_engine_info=transfer, kv_dtype=DTYPE,
                dependencies={key: by_key[key] for key in spec.depends_on()},
            ))
            for spec in specs
        }
        self.runner = StepRunner(self.resources, node_resources={LLM_NODE: list(self.resources)})
        self.sub = submodule
        submodule.requires_grad_(False)
        submodule.bind_node_resources(self.resources)
        self.configs = model.get_request_resource_configs({}, {"temperature": 0.0, "ignore_eos": True})
        self.rids = rids
        for rid in rids:
            self.runner.ingest_request(rid, self.configs)

    def step(self, walk, ids_per_request):
        """One step over every request; each request's last-row logits, and the sampled tokens."""
        rids = self.rids
        infos = {rid: CurrentForwardPassInfo(request_id=rid, graph_walk=walk, fwd_index=0, random_seed=0,
                                             max_tokens=64, resource_configs=self.configs) for rid in rids}
        inputs = [self.sub.prepare_inputs(walk, infos[rid], {"text_inputs": [ids]})
                  for rid, ids in zip(rids, ids_per_request, strict=True)]
        step = self.sub.declare_step(walk, rids, inputs)
        with torch.no_grad(), autocast_scope(DTYPE, device_type=DEVICE.type):
            step.set_ctx(StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False))
            assert self.runner.admit(step).ok
            self.runner.plan(step)
            engine_inputs = ModelInputsFromEngine(request_ids=rids, per_request_info=infos,
                                                  resources=self.resources, step=step)
            kw = self.sub.preprocess(walk, engine_inputs, inputs)
            lm = self.sub.lm
            hidden = lm(lm.embed(kw["input_ids"]), label="main")
            if walk == PREFILL_WALK:
                hidden = self.resources[ATTN].select_last_hidden(hidden, label="main")
            logits = lm.logits(hidden).float()
            tokens = self.resources["sampler"].sample(list(rids), logits=logits)
            self.runner.commit(step)
        return logits, tokens

    def close(self):
        for rid in self.rids:
            self.runner.remove_request(rid)
            self.sub.cleanup_request(rid)
        for resource in self.resources.values():
            resource.cleanup()


def test_packed_prefill_and_batched_decode_match_hf(tmp_path):
    from mstar.model.text_lm.text_lm_model import TextLMModel

    hf = _checkpoint(tmp_path)
    model = TextLMModel(model_path_hf=str(tmp_path), max_seq_len=1024, accelerator_graph=False)
    sub = model.get_submodule(LLM_NODE, device="cuda", autocast_dtype=DTYPE)
    assert model.stop_token_ids() == frozenset({7})  # no tokenizer files: the configs alone
    torch.manual_seed(1)
    # two prompts of different lengths, one spanning several 16-token pages
    prompts = [torch.randint(0, TINY["vocab_size"], (n,), device=DEVICE) for n in (37, 5)]
    node = _Node(model, sub, ["a", "b"])
    seqs = [p.clone() for p in prompts]
    try:
        logits, tokens = node.step(PREFILL_WALK, prompts)
        worst = 0.0
        for step in range(6):
            with torch.no_grad():
                for i, seq in enumerate(seqs):
                    ref = hf(seq[None]).logits[0, -1]
                    err = (logits[i] - ref).abs().max().item() / ref.abs().max().item()
                    worst = max(worst, err)
                    assert int(tokens[i]) == int(ref.argmax()), (step, i)
            seqs = [torch.cat([s, t.view(1)]) for s, t in zip(seqs, tokens, strict=True)]
            logits, tokens = node.step(DECODE_WALK, [t.view(1) for t in tokens])
        # bf16 weights and activations against an fp32 reference from the same weights
        assert worst < 0.03, worst
    finally:
        node.close()
