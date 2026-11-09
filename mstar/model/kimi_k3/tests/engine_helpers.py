"""The real ``Engine`` on CPU around a tiny Kimi K3 (``KimiK3ModelConfig.reduced``) loaded from a
random checkpoint in the release's names: admit -> plan -> forward -> commit for prefill and
decode steps, as the worker drives them (after ``test/modular/test_glm5next_engine_cpu.py``)."""
from __future__ import annotations

import sys
import types

import torch

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import SamplingReqConfig
from mstar.engine.resources.base import Resource
from mstar.engine.resources.kv import manager as kv_mod
from mstar.engine.resources.kv.config import KVReqConfig
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import StepContext
from mstar.model.base import LabelLayout, Span, keyed_labels
from mstar.model.kimi_k3.components.causal_lm import KimiK3ForCausalLM
from mstar.model.kimi_k3.config import KDA_STATE, KV_CACHE, LABEL, SAMPLER
from mstar.model.kimi_k3.kimi_k3_model import KimiK3Model
from mstar.model.kimi_k3.submodules import KimiK3LLMSubmodule
from mstar.model.kimi_k3.tests.reference import random_checkpoint
from mstar.model.submodule_base import InputMetadata
from mstar.model.utils import check_loaded, process_weights_after_loading


class GreedySampler(Resource):
    """Argmax in place of the kernel sampler; it keeps the step's logits for the parity checks."""

    def __init__(self):
        self.logits: list[torch.Tensor] = []

    @classmethod
    def build(cls, spec, info):
        return cls()

    def sample(self, request_ids, logits, **kwargs):
        self.logits.append(logits.detach().clone())
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
        return False

    def remove_request(self, request_id):
        del request_id


def install_cpu_stubs(monkeypatch) -> None:
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(
        rmsnorm=lambda x, w, eps=1e-6: (
            x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        ).to(x.dtype) * w.to(x.dtype))
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    monkeypatch.setattr(kv_mod, "KVTransferManager", _StubTransfer)
    # the engine runs on CPU; on a GPU box warmup would still try to capture
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def load_tiny(model: KimiK3Model, sd: dict[str, torch.Tensor]) -> KimiK3ForCausalLM:
    """``KimiK3ForCausalLM`` at the model's config, filled through the real loader."""
    lm = KimiK3ForCausalLM(model.config)
    stream = (("language_model." + k, v) for k, v in sd.items())
    check_loaded(lm, lm.load_weights(stream), "random checkpoint")
    process_weights_after_loading(lm, torch.device("cpu"))
    return lm.eval()


def build_engine(variant: str = "reduced", seed: int = 0, config=None, prefix_slots: int = 0,
                 page_size: int = 64, **model_kwargs):
    """``(engine, submodule, model, state_dict)`` over a random checkpoint; ``config`` replaces
    the variant's. ``prefix_slots``: the prefix cache open, over ``page_size``-token pages (a
    multiple of KDA's 64-token chunk), with that many KDA snapshots, as a serving config sets
    it (a salt stands in for the checkpoint)."""
    torch.manual_seed(seed)
    model = KimiK3Model("x", config_variant=variant, kda_conv_dtype=torch.float32,
                        kda_max_requests=4, tokenizer_mode="byte", **model_kwargs)
    if config is not None:
        model.config = config
    sd = random_checkpoint(model.config, seed=seed, a_log_pad=3)
    lm = load_tiny(model, sd)
    sub = KimiK3LLMSubmodule(lm, model.config)
    specs = model.get_node_resources()
    if prefix_slots:
        for spec in specs:
            if spec.resource_key == KV_CACHE:
                spec.apply_yaml_overrides(page_size=page_size, prefix_cache_salt="a salt")
            elif spec.resource_key == KDA_STATE:
                spec.apply_yaml_overrides(prefix_slots=prefix_slots)
    if prefix_slots:
        # the random checkpoint is on no disk: the salt roots the cache
        model.checkpoint_path = lambda: None
    engine = Engine(None)
    engine.load_model(
        {"LLM": sub}, specs, model=model if prefix_slots else None,
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=torch.device("cpu"),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="e", my_session_id="s", transfer_engine=LocalTransferEngine("localhost")),
        kv_cache_type=torch.float32,
    )
    greedy = GreedySampler()
    engine._resources[SAMPLER] = greedy
    engine._submodules["LLM"].resources[SAMPLER] = greedy
    sub.node_resources[SAMPLER] = greedy
    engine.warmup()  # no CUDA: captures nothing, compiles nothing
    return engine, sub, model, sd


def _info(rid: str, walk: str, max_tokens: int = 64) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=f"wire-{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0, random_seed=0,
        max_tokens=max_tokens, resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0)})


def add_request(engine: Engine, rid: str) -> None:
    engine.add_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0)})


def add_keyed_request(engine: Engine, model: KimiK3Model, rid: str, prompt: torch.Tensor, page_size: int = 64) -> None:
    """``add_request`` with the KV keys the data worker sends for ``prompt``, as Kimi lays it out
    (`KimiK3Model.process_prompt`), so the engine probes the prefix cache for it."""
    n = prompt.numel()
    layouts = {KV_CACHE: {LABEL: LabelLayout(spans=(Span("ids", "prefill", n, n, "text_inputs"),),
                                             growth=(("decode_loop", 1),))}}
    worker = types.SimpleNamespace(
        _keyed_labels=keyed_labels(model), _prefix_page_sizes={KV_CACHE: page_size}, model=None,
        _page_layout=lambda *args: PreprocessWorkerThread._page_layout(None, *args),
    )
    rows = {KV_CACHE: {LABEL: layouts[KV_CACHE][LABEL].row()}}
    sent = PreprocessWorkerThread._prefix_keys(worker, {"text_inputs": [prompt]}, layouts, rows, None, {})
    kv = KVReqConfig()
    kv.apply_conductor_config(kv_layout=rows[KV_CACHE], loop_caps={"decode_loop": 64},
                              **{name: by[KV_CACHE] for name, by in sent.items()})
    engine.add_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0), KV_CACHE: kv})


def _batch(walk: str, inputs: dict[str, torch.Tensor]) -> ExecutingBatch:
    rids = list(inputs)
    infos = {rid: _info(rid, walk) for rid in rids}
    return ExecutingBatch(
        node_name="LLM",
        per_request_info=infos,
        per_request_input_tensors={rid: {"text_inputs": [ids]} for rid, ids in inputs.items()},
        per_request_input_metadata={
            rid: InputMetadata(dynamic_loop_iter_counts=dict(info.dynamic_loop_iter_counts))
            for rid, info in infos.items()},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
    )


def failed_requests(engine: Engine, walk: str, inputs: dict[str, torch.Tensor]) -> set[str]:
    """The requests ``prepare_inputs`` fails for one step (the engine logs why)."""
    batch = _batch(walk, inputs)
    engine.prepare_inputs(batch)
    return set(batch.failed_requests)


def run_step(engine: Engine, walk: str, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """One step over ``rid -> input ids``; returns each request's sampled token."""
    rids = list(inputs)
    batch = _batch(walk, inputs)
    engine.prepare_inputs(batch)
    assert not batch.failed_requests, batch.failed_requests
    outputs = engine.exec_and_postprocess(batch)
    assert batch.admit_error is None, batch.admit_error
    engine.finalize_batch(batch)
    return {rid: outputs.per_rid_outputs[rid]["new_token"][0] for rid in rids}
