"""glm5_next through the real ``Engine`` on CPU: load_model -> add_request ->
prepare_inputs -> exec_and_postprocess -> check_stop -> remove_request, for a
prefill and a batch of decodes, with two requests interleaved.
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
from mstar.model.submodule_base import InputMetadata


def _batch(**kwargs) -> ExecutingBatch:
    """An ExecutingBatch as the worker builds it: the engine reads each request's
    loop counts from its input metadata (main #351), not from its forward-pass info."""
    kwargs.setdefault("per_request_input_metadata", {
        rid: InputMetadata(dynamic_loop_iter_counts=dict(info.dynamic_loop_iter_counts))
        for rid, info in kwargs["per_request_info"].items()
    })
    return ExecutingBatch(**kwargs)


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

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del transfer_info, kwargs
        return False

    def remove_request(self, request_id):
        del request_id


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
    # the engine here is on CPU; on a GPU box warmup would still try to capture
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _build_engine(seed: int = 0, autocast_dtype: torch.dtype | None = None):
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

    engine = Engine(None, autocast_dtype=autocast_dtype)  # no offload, so no runtime
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
        request_id=f"wire-{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0, random_seed=0,
        max_tokens=max_tokens,
        resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0)},
    )


def _run(engine: Engine, walk: str, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    rids = list(inputs)
    batch = _batch(
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
    return {rid: outputs.per_rid_outputs[rid]["new_token"][0] for rid in rids}


def _context(sub: Glm5NextLLMSubmodule, rid: str) -> int:
    return sub.request_state(rid).get("context", 0)


def _add(engine: Engine, rid: str) -> None:
    engine.add_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0)})


class TestEngineCpu:
    def test_two_requests_interleaved_match_whole_prefill(self):
        engine, sub, cfg = _build_engine()
        kv = engine._resources[KV_CACHE]
        pool = engine._resources[KDA_STATE]
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
        assert _context(sub, "a") == len(a) + 3 and _context(sub, "b") == len(b) + 3
        assert kv._streams["a"]["main"].stored_len == len(a) + 3
        assert pool.num_free_slots == pool.config.usable_slots - 2

        # a fresh request prefilled with the whole generated prefix must
        # produce the same next token as the stepwise path did
        for rid, prompt in (("a", a), ("b", b)):
            full = torch.cat([prompt, *seq[rid][:-1]])
            _add(engine, f"{rid}_ref")
            ref = _run(engine, "prefill", {f"{rid}_ref": full})[f"{rid}_ref"]
            assert ref.item() == seq[rid][-1].item(), (rid, ref.item(), seq[rid][-1].item())

        for rid in ("a", "b", "a_ref", "b_ref"):
            engine.remove_request(rid)
        assert pool.num_free_slots == pool.config.usable_slots
        assert not kv._streams

    def test_check_stop_and_context_cap(self):
        engine, sub, cfg = _build_engine(seed=1)
        _add(engine, "r")
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        out = _run(engine, "prefill", {"r": prompt})
        batch = _batch(
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
        b = _batch(
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
        assert set(outputs.per_rid_outputs) == {"short"}
        engine.remove_request("long")
        engine.remove_request("short")

    def test_decode_stops_inside_the_context_cap(self):
        engine, sub, cfg = _build_engine(seed=2)
        limit = cfg.index_topk
        _add(engine, "r")
        prompt = torch.arange(limit - 5, dtype=torch.long)
        token = _run(engine, "prefill", {"r": prompt})["r"]
        for i in range(limit):
            token = _run(engine, "decode", {"r": token})["r"]
            info = _info("r", "decode", max_tokens=10 * limit)
            info.dynamic_loop_iter_counts["decode_loop"] = i
            batch = _batch(
                node_name="LLM", per_request_info={"r": info},
                step_context=StepContext(request_ids=("r",), graph_walk="decode", slot=0, capture=False),
            )
            if engine.check_stop_for_batch(batch, {"r": {"new_token": [token]}}).get("r"):
                break
        # 1 prefill token + 4 decode steps: the context is one short of the cap
        assert i == 3 and _context(sub, "r") == limit - 1
        # the step already scheduled when the stop lands still fits
        _run(engine, "decode", {"r": token})
        assert _context(sub, "r") == limit
        engine.remove_request("r")

    def test_forward_runs_outside_the_engine_autocast(self):
        engine, sub, cfg = _build_engine(autocast_dtype=torch.bfloat16)
        seen = []
        sub.language_model.model.layers[0].register_forward_pre_hook(
            lambda module, args: seen.append(torch.is_autocast_enabled("cpu")))
        _add(engine, "r")
        _run(engine, "prefill", {"r": torch.tensor([1, 2, 3], dtype=torch.long)})
        assert seen == [False]
        engine.remove_request("r")

    def test_max_batch_size_is_the_slot_pool(self):
        engine, sub, cfg = _build_engine()
        assert engine.get_max_batch_size("LLM", "decode") == 4
        assert engine.get_max_batch_size("LLM", "prefill") == 4

    def test_prefill_batch_is_capped_at_free_slots(self):
        engine, sub, cfg = _build_engine()
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        for rid in ("a", "b", "c"):
            _add(engine, rid)
        _run(engine, "prefill", {"a": prompt})
        assert engine.get_max_batch_size("LLM", "prefill") == 3
        _run(engine, "prefill", {"b": prompt, "c": prompt})
        assert engine.get_max_batch_size("LLM", "prefill") == 1
        assert engine.get_max_batch_size("LLM", "decode") == 4
        _add(engine, "d")
        _run(engine, "prefill", {"d": prompt})
        # nothing free: prefill waits rather than being scheduled to be refused
        assert engine.get_max_batch_size("LLM", "prefill") == 0
        assert engine.get_max_batch_size("LLM", "decode") == 4
        engine.remove_request("a")
        assert engine.get_max_batch_size("LLM", "prefill") == 1

    def test_a_follower_out_of_slots_rank_0_had_raises(self):
        """Rank 0 schedules a prefill only into free slots. A follower whose pool
        is full for it has diverged, and must not quietly re-queue a step rank 0
        is already running."""
        engine, sub, cfg = _build_engine()
        engine._tp_follower_nodes = {"LLM"}
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        for rid in "abcde":
            _add(engine, rid)
        _run(engine, "prefill", {rid: prompt for rid in "abcd"})
        with pytest.raises(RuntimeError, match="diverged"):
            _run(engine, "prefill", {"e": prompt})
        # the refusal was unwound: the four held slots, none for e
        pool = engine._resources[KDA_STATE]
        assert pool.num_free_slots == 0 and not pool._slots["e"]


def test_prompt_past_the_cap_is_refused_up_front():
    model = Glm5NextModel("x", config_variant="reduced", tokenizer_mode="byte")
    limit = model.config.index_topk
    out = model.process_prompt("a" * (limit - 2), ["text"], ["text"])
    assert out["text_inputs"][0].numel() == limit - 2
    # a ValueError in preprocessing is the client's 400
    with pytest.raises(ValueError, match="at most"):
        model.process_prompt("a" * (limit - 1), ["text"], ["text"])
