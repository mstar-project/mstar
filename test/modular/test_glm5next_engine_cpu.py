"""glm5_next through the real ``Engine`` on CPU: load_model -> add_request ->
prepare_inputs -> exec_and_postprocess -> check_stop -> remove_request, for a
prefill and a batch of decodes, with two requests interleaved.

The one substitution is the sampler: its kernels are Triton/FlashInfer, so a
greedy stand-in takes its resource key after load. Everything else — the
MLA-layout KV cache, the MLA attention backend (fp32 SDPA fallback), the
slot-state resource, the runner, the engine's step protocol — is the real
thing. What this pins that the resource-level tests cannot: the engine
binds the resources into the layers, drives declare/admit/plan/commit around
the forward, pads nothing eagerly, and the model's output for a sequence
generated stepwise equals the output of prefilling that sequence whole.
"""
from __future__ import annotations

import sys
import types

sys.path.insert(0, ".")

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import SamplingReqConfig
from mstar.engine.resources.base import Resource
from mstar.engine.resources.kv import manager as kv_mod
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import StepContext
from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
from mstar.model.glm5_next.config import KDA_STATE, KV_CACHE, SAMPLER, Glm5NextModelConfig
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel, process_weights_after_loading
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule


class _GreedySampler(Resource):
    """Argmax in place of the Triton sampler; the Resource hooks are no-ops."""

    @classmethod
    def build(cls, spec, info):
        return cls()

    def sample(self, request_ids, logits, **kwargs):
        return logits.argmax(-1)


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self):
        return None

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(
        rmsnorm=lambda x, w, eps=1e-6: (
            x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        ).to(x.dtype) * w.to(x.dtype)
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    monkeypatch.setattr(kv_mod, "KVTransferManager", _StubTransfer)


def _build_engine(seed: int = 0):
    torch.manual_seed(seed)
    cfg = Glm5NextModelConfig.reduced()
    model = Glm5NextModel("x", config_variant="reduced", kda_conv_dtype=torch.float32, kda_max_requests=4)
    lm = Glm5NextForCausalLM(cfg)
    for p in lm.parameters():
        if p.dtype.is_floating_point:
            torch.nn.init.normal_(p, std=0.02)
    process_weights_after_loading(lm, torch.device("cpu"))
    lm.eval()
    sub = Glm5NextLLMSubmodule(lm, cfg)

    engine = Engine(autocast_dtype=None)
    engine.load_model(
        {"LLM": sub}, model.get_node_resources(),
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=torch.device("cpu"),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="e", my_session_id="s",
            transfer_engine=LocalTransferEngine("localhost"),
        ),
        kv_cache_type=torch.float32,
    )
    # swap the kernel sampler for the greedy stand-in everywhere the engine
    # handed it out
    greedy = _GreedySampler()
    engine._resources[SAMPLER] = greedy
    engine._submodules["LLM"].resources[SAMPLER] = greedy
    sub.node_resources[SAMPLER] = greedy
    engine.warmup()  # no CUDA: captures nothing, compiles nothing (disable_torch_compile)
    return engine, sub, cfg


def _info(rid: str, walk: str, max_tokens: int = 64) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=walk, fwd_index=0, random_seed=0,
        max_tokens=max_tokens,
        resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0)},
    )


def _run(engine: Engine, walk: str, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    rids = list(inputs)
    batch = ExecutingBatch(
        node_name="LLM",
        per_request_info={rid: _info(rid, walk) for rid in rids},
        per_request_input_tensors={rid: {"text_inputs": [ids]} for rid, ids in inputs.items()},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
    )
    engine.prepare_inputs(batch)
    assert not batch.failed_requests, batch.failed_requests
    outputs = engine.exec_and_postprocess(batch)
    assert batch.admit_error is None, batch.admit_error
    engine.finalize_batch(batch)
    return {rid: outputs[rid]["new_token"][0] for rid in rids}


def _add(engine: Engine, rid: str) -> None:
    engine.add_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0)})


class TestEngineCpu:
    def test_two_requests_interleaved_match_whole_prefill(self):
        engine, sub, cfg = _build_engine()
        kv = engine._resources[KV_CACHE]
        kda = engine._resources[KDA_STATE]
        a = torch.tensor([5, 9, 13, 7, 21, 2], dtype=torch.long)
        b = torch.tensor([44, 3, 8], dtype=torch.long)
        _add(engine, "a")
        _add(engine, "b")
        # prefill both in one batch, then three joint decode steps
        first = _run(engine, "prefill", {"a": a, "b": b})
        seq = {"a": [first["a"]], "b": [first["b"]]}
        for _ in range(3):
            step = _run(engine, "decode", {rid: seq[rid][-1] for rid in seq})
            for rid, tokens in seq.items():
                tokens.append(step[rid])
        assert kda.committed("a") == len(a) + 3 and kda.committed("b") == len(b) + 3
        assert kv._streams["a"]["main"].stored_len == len(a) + 3
        assert kda.num_free == kda.max_slots - 2

        # a fresh request prefilled with the whole generated prefix must
        # produce the same next token as the stepwise path did
        for rid, prompt in (("a", a), ("b", b)):
            full = torch.cat([prompt, *seq[rid][:-1]])
            _add(engine, f"{rid}_ref")
            ref = _run(engine, "prefill", {f"{rid}_ref": full})[f"{rid}_ref"]
            assert ref.item() == seq[rid][-1].item(), (rid, ref.item(), seq[rid][-1].item())

        for rid in ("a", "b", "a_ref", "b_ref"):
            engine.remove_request(rid)
        assert kda.num_free == kda.max_slots
        assert not kv._streams

    def test_check_stop_and_context_cap(self):
        engine, sub, cfg = _build_engine(seed=1)
        _add(engine, "r")
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        out = _run(engine, "prefill", {"r": prompt})
        batch = ExecutingBatch(
            node_name="LLM",
            per_request_info={"r": _info("r", "decode", max_tokens=2)},
            step_context=StepContext(request_ids=("r",), graph_walk="decode", slot=0, capture=False),
        )
        # generated so far = 1 prefill token + (0 decode iters + 1) = 2 >= max_tokens
        stops = engine.check_stop_for_batch(batch, {"r": {"new_token": [out["r"]]}})
        assert stops == {"r": {"decode_loop"}}

        # the index_topk cap: a prompt that would cross it fails only its rid
        engine.remove_request("r")
        _add(engine, "long")
        _add(engine, "short")
        too_long = torch.zeros(cfg.index_topk + 1, dtype=torch.long)
        rids = ["long", "short"]
        b = ExecutingBatch(
            node_name="LLM",
            per_request_info={rid: _info(rid, "prefill") for rid in rids},
            per_request_input_tensors={
                "long": {"text_inputs": [too_long]},
                "short": {"text_inputs": [prompt]},
            },
            step_context=StepContext(request_ids=tuple(rids), graph_walk="prefill", slot=0, capture=False),
        )
        engine.prepare_inputs(b)
        assert "long" in b.failed_requests and "index_topk" in str(b.failed_requests["long"])
        assert list(b.request_ids) == ["short"]
        outputs = engine.exec_and_postprocess(b)
        assert set(outputs) == {"short"}
        engine.remove_request("long")
        engine.remove_request("short")

    def test_max_batch_size_is_the_slot_pool(self):
        engine, sub, cfg = _build_engine()
        assert engine.get_max_batch_size("LLM", "decode") == 4
        assert engine.get_max_batch_size("LLM", "prefill") == 4
