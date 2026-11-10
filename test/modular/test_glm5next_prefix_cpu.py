"""GLM-5.3 prefix reuse through the real ``Engine`` on CPU: a prompt that shares an earlier
prompt's pages starts from the KV pages and the KDA state cached at their end, and decodes the
greedy tokens a cold engine does.

``reduced_long_context`` serves the DSA path, and a 96-token prefill window cuts the first
prompt inside a KDA chunk.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import NamedTuple

sys.path.insert(0, ".")

import pytest
import torch
from test_glm5next_engine_cpu import _batch, _GreedySampler, _stubs  # noqa: F401 (the fixture)

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine
from mstar.engine.resources import SamplingReqConfig
from mstar.engine.resources.kv.config import KVReqConfig
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import StepContext
from mstar.model.base import keyed_labels
from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
from mstar.model.glm5_next.config import KDA_STATE, KV_CACHE, SAMPLER
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel, process_weights_after_loading
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule
from mstar.utils.device import DeviceCapability
from mstar.worker.host_copies import HostCopies

DECODES, MAX_TOKENS = 4, 64


def _model(k: int) -> Glm5NextModel:
    model = Glm5NextModel(
        "x", config_variant="reduced_long_context", tokenizer_mode="byte", kda_conv_dtype=torch.float32,
        kda_max_requests=4, dsa_long_context=True, prefill_window_tokens=96, mtp_num_draft_tokens=k,
    )
    # no weights on disk: the deployment's salt roots the cache instead
    model.checkpoint_path = lambda: None
    return model


def _engine(model: Glm5NextModel, prefix_slots: int) -> tuple[Engine, Glm5NextLLMSubmodule]:
    torch.manual_seed(0)
    lm = Glm5NextForCausalLM(model.config)
    for p in lm.parameters():
        if p.dtype.is_floating_point:
            # large enough that the greedy token depends on its context, not the last token alone
            torch.nn.init.normal_(p, std=0.1)
    process_weights_after_loading(lm, torch.device("cpu"))
    lm.eval()
    sub = Glm5NextLLMSubmodule(lm, model.config)
    specs = model.get_node_resources()
    for spec in specs:
        if spec.resource_key == KV_CACHE:
            spec.config.prefix_cache_salt = "a salt"
        if spec.resource_key == KDA_STATE:
            spec.config.prefix_slots = prefix_slots
    engine = Engine(None)
    engine.load_model(
        {"LLM": sub}, specs,
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=torch.device("cpu"),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="e", my_session_id="s", transfer_engine=LocalTransferEngine("localhost"),
        ),
        kv_cache_type=torch.float32, model=model,
    )
    greedy = _GreedySampler()
    engine._resources[SAMPLER] = greedy
    engine._submodules["LLM"].resources[SAMPLER] = greedy
    sub.node_resources[SAMPLER] = greedy
    engine.warmup()
    return engine, sub


def _request(model: Glm5NextModel, engine: Engine, rid: str, text: str) -> torch.Tensor:
    """Tokenize and key ``text`` as the preprocess worker and conductor do, and add it."""
    out = model.process_prompt(text, ["text"], ["text"])
    tensors, layouts = dict(out.new_input_tensors), out.metadata["kv_layout"]
    rows = {key: {label: layout.row() for label, layout in by.items()} for key, by in layouts.items()}
    page = engine._resources[KV_CACHE].config.tokens_per_page
    worker = SimpleNamespace(
        _keyed_labels=keyed_labels(model), _prefix_page_sizes={KV_CACHE: page}, model=model,
        _page_layout=lambda *args: PreprocessWorkerThread._page_layout(None, *args),
    )
    sent = PreprocessWorkerThread._prefix_keys(worker, tensors, layouts, rows, None, {})
    kv = KVReqConfig()
    kv.apply_conductor_config(
        kv_layout=rows[KV_CACHE], loop_caps={"decode_loop": DECODES + 4},
        **{name: by[KV_CACHE] for name, by in sent.items()},
    )
    engine.add_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0), KV_CACHE: kv})
    return tensors["text_inputs"][0]


def _info(rid: str, walk: str, decode_iter: int) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=f"wire-{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0, random_seed=0,
        max_tokens=MAX_TOKENS, resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0)},
        dynamic_loop_iter_counts={"decode_loop": decode_iter},
    )


class Run(NamedTuple):
    # each request's emitted tokens
    tokens: dict[str, list[int]]
    # the prompt tokens its prefill ran; the rest of its prompt came from the cache
    prefilled: dict[str, int]


def _step(engine: Engine, walk: str, inputs: dict[str, torch.Tensor], decode_iter: int = 0,
          ran: dict[str, int] | None = None) -> dict:
    """One step, as the worker runs it; ``ran`` receives the tokens it ran of each request."""
    rids = list(inputs)
    batch = _batch(
        node_name="LLM", per_request_info={rid: _info(rid, walk, decode_iter) for rid in rids},
        per_request_input_tensors={rid: {"text_inputs": [ids]} for rid, ids in inputs.items()},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
    )
    engine.prepare_inputs(batch)
    assert not batch.failed_requests, batch.failed_requests
    if ran is not None:
        ran.update({rid: inp.input_seq_len for rid, inp in zip(batch.prepared_rids, batch.inputs, strict=True)})
    outputs = engine.exec_and_postprocess(batch)
    assert batch.admit_error is None, batch.admit_error
    engine.finalize_batch(batch)
    # the worker's postprocess: the stop check's host rows key what the step sampled
    cpu = torch.device("cpu")
    host, host_rows = HostCopies(cpu, DeviceCapability.of(cpu)).prematerialize_for_check_stop(
        outputs, None, request_ids=batch.request_ids)
    engine.step_finished(batch, host, host_rows=host_rows)
    engine.check_stop_for_batch(batch, host, host_rows=host_rows)
    return outputs.per_rid_outputs


def _generate(engine: Engine, inputs: dict[str, torch.Tensor], decodes: int = DECODES) -> Run:
    """Prefill ``inputs`` in one step, then decode them together."""
    prefilled: dict[str, int] = {}
    out = _step(engine, "prefill", inputs, ran=prefilled)
    seqs = {rid: out[rid]["new_token"][0].reshape(-1).tolist() for rid in inputs}
    for i in range(decodes):
        out = _step(engine, "decode", {rid: out[rid]["text_inputs"][0].reshape(-1) for rid in inputs}, i)
        for rid in inputs:
            seqs[rid] += out[rid]["new_token"][0].reshape(-1).tolist()
    return Run(seqs, prefilled)


SHARED = "".join(f"line {i}: the quick brown fox jumps over the lazy dog. " for i in range(6))[:270]


@pytest.mark.parametrize("k", [0, 3], ids=["plain", "mtp"])
def test_a_prompt_past_a_cached_prefix_decodes_as_a_cold_run(k):
    model = _model(k)
    warm, warm_sub = _engine(model, prefix_slots=4)
    a = _request(model, warm, "a", SHARED + " and then the first request ends here.")
    _generate(warm, {"a": a})
    warm.remove_request("a")

    b_text = SHARED + " but the second one asks something else."
    b = _request(model, warm, "b", b_text)
    c = _request(model, warm, "c", "an unrelated prompt that shares nothing with the others.")
    page = warm._resources[KV_CACHE].config.tokens_per_page
    got = _generate(warm, {"b": b, "c": c})
    # what ran, not the KV cache's own hit count: the KDA pool cuts that to its snapshot
    assert got.prefilled == {"b": len(b) - len(SHARED) // page * page, "c": len(c)}
    assert warm_sub.request_state("b").get("context", 0) >= len(b)

    cold_model = _model(k)
    cold, _ = _engine(cold_model, prefix_slots=0)
    ref = _generate(cold, {
        "b": _request(cold_model, cold, "b", b_text),
        "c": _request(cold_model, cold, "c", "an unrelated prompt that shares nothing with the others."),
    })
    assert got.tokens == ref.tokens, "a prefix hit changed the greedy output"
