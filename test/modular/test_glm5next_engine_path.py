"""Scratch smoke: reduced glm5_next through real v1 resources on CPU."""
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch


def _stub_flashinfer():
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(
        rmsnorm=lambda x, w, eps=1e-6: (
            x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        ).to(x.dtype) * w.to(x.dtype)
    )
    sys.modules.setdefault("flashinfer", fi)


@torch.no_grad()
def test_prefill_then_decode():
    _stub_flashinfer()
    from mstar.engine.resources import StepRunner
    from mstar.engine.resources.base import EngineResourceInfo, build_resource
    from mstar.engine.resources.kv import manager as kv_mod
    from mstar.engine.resources.spec import resolve_spec_dependencies
    from mstar.engine.resources.step import StepContext
    from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
    from mstar.model.glm5_next.config import SAMPLER, Glm5NextModelConfig
    from mstar.model.glm5_next.glm5_next_model import Glm5NextModel, process_weights_after_loading
    from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule
    from mstar.model.submodule_base import ModelInputsFromEngine

    class _StubTransfer:
        def __init__(self, *a, **k): pass
        def get_kv_transfer_info(self): return None
        def cleanup(self): pass
    kv_mod.KVTransferManager = _StubTransfer

    torch.manual_seed(0)
    cfg = Glm5NextModelConfig.reduced()
    model = Glm5NextModel("x", config_variant="reduced", kda_conv_dtype=torch.float32)
    lm = Glm5NextForCausalLM(cfg)
    for p in lm.parameters():
        if p.dtype.is_floating_point:
            torch.nn.init.normal_(p, std=0.02)
    process_weights_after_loading(lm, torch.device("cpu"))
    lm.eval()
    lm.requires_grad_(False)
    sub = Glm5NextLLMSubmodule(lm, cfg)

    specs = model.get_node_resources()
    by_key = resolve_spec_dependencies(specs)
    resources = {}
    for spec in specs:
        info = EngineResourceInfo(
            device=torch.device("cpu"), kv_dtype=torch.float32,
            dependencies={k: by_key[k] for k in spec.depends_on()},
        )
        resources[spec.resource_key] = build_resource(spec, info)
    runner = StepRunner(resources, node_resources={"LLM": list(resources)})
    sub.bind_node_resources(resources)

    # a fake greedy sampler (the real one needs triton/flashinfer kernels)
    class _Greedy:
        def sample(self, rids, logits, **kw):
            return logits.argmax(-1)
    resources[SAMPLER] = _Greedy()

    rid = "r0"
    runner.ingest_request(rid, {})
    prompt = torch.tensor([5, 9, 13, 7, 21], dtype=torch.long)

    def step(walk, ids):
        fwd_info = SimpleNamespace(request_id=rid)
        node_in = sub.prepare_inputs(walk, fwd_info, {"text_inputs": [ids]})
        ctx = StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False)
        st = sub.declare_step(walk, [rid], [node_in])
        st.set_ctx(ctx)
        # the sampler is faked, so drive the other three by hand
        st.steps.pop(SAMPLER)
        assert runner.admit(st).ok
        runner.plan(st)
        ei = ModelInputsFromEngine(request_ids=[rid], per_request_info={}, resources=resources)
        pre = sub.preprocess(walk, ei, [node_in])
        out = sub.forward_batched(walk, ei, **pre)
        runner.commit(st)
        return out[rid]["new_token"][0]

    t1 = step("prefill", prompt)
    t2 = step("decode", t1)
    t3 = step("decode", t2)
    kda = resources["kda_state"]
    assert kda.committed(rid) == 7
    assert resources["kv_cache"]._streams[rid]["main"].stored_len == 7
    print("tokens", t1.item(), t2.item(), t3.item())

    # reference: whole sequence prefilled in one go on a fresh request must
    # yield the same third token (chunk-vs-step parity of the engine path)
    rid2 = "r1"
    runner.ingest_request(rid2, {})
    rid = rid2
    full = torch.cat([prompt, t1, t2])
    t3_ref = step("prefill", full)
    assert t3_ref.item() == t3.item(), (t3_ref.item(), t3.item())
    runner.remove_request("r0")
    runner.remove_request("r1")
    assert kda.num_free == kda.max_slots
