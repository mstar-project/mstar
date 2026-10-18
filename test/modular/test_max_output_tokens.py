"""A decode loop stops at the request's max_output_tokens, counting the token
prefill emitted.

Decode step k reads iteration k in its stop check, so with prefill's token
already out it has emitted k + 2 tokens. The loop runs through the real graph
runtime here, so each step sees the iteration count the worker would give it.
"""
from collections.abc import Callable
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RouteInput
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import WorkerGraph
from mstar.model.submodule_base import HostRows
from mstar.utils.containers import ParallelList

EOS = 7
TOKEN = 3
NODE = "LLM"
PARTITION = "default"


class Case(NamedTuple):
    walk: str
    loop: str
    sampler: str
    # (info, token) -> whether the model stops its decode loop
    stop: Callable[[SimpleNamespace, int], bool]


def _qwen3_5(batched: bool) -> Case:
    from mstar.model.qwen3_5.config import SAMPLER
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    sub = SimpleNamespace(config=SimpleNamespace(stop_token_ids=frozenset({EOS})))
    if batched:
        def stop(info, token):
            rows = HostRows(("r",), {"new_token": torch.tensor([token])})
            return bool(LLMSubmodule.check_stop_batched(sub, ["r"], {"r": info}, rows))
    else:
        def stop(info, token):
            return bool(LLMSubmodule.check_stop(
                sub, "r", info, {"new_token": [torch.tensor([token])]},
            ))
    return Case("decode", "decode_loop", SAMPLER, stop)


def _per_request(cls, config, walk, loop, sampler) -> Case:
    sub = SimpleNamespace(config=config)

    def stop(info, token):
        return bool(cls.check_stop(sub, "r", info, {"new_token": [torch.tensor([token])]}))
    return Case(walk, loop, sampler, stop)


def _qwen3_omni() -> Case:
    from mstar.model.qwen3_omni.config import THINKER_SAMPLER
    from mstar.model.qwen3_omni.submodules import ThinkerSubmodule

    return _per_request(
        ThinkerSubmodule, SimpleNamespace(im_end_token_id=EOS),
        "thinker_decode", "thinker_decode_loop", THINKER_SAMPLER,
    )


def _whisper() -> Case:
    from mstar.model.whisper.config import SAMPLER
    from mstar.model.whisper.submodules import WhisperDecoderSubmodule

    return _per_request(
        WhisperDecoderSubmodule, SimpleNamespace(eos_token_id=EOS),
        "decode", "decode_loop", SAMPLER,
    )


def _higgs_audio() -> Case:
    from mstar.model.higgs_audio.config import SAMPLER
    from mstar.model.higgs_audio.submodules import HiggsAudioLLMSubmodule

    return _per_request(
        HiggsAudioLLMSubmodule, SimpleNamespace(stop_token_ids=frozenset({EOS})),
        "decode", "decode_loop", SAMPLER,
    )


def _orpheus() -> Case:
    from mstar.model.orpheus.config import SAMPLER
    from mstar.model.orpheus.submodules import OrpheusLLMSubmodule

    return _per_request(
        OrpheusLLMSubmodule, SimpleNamespace(stop_token_id=EOS),
        "decode", "decode_loop", SAMPLER,
    )


CASES = {
    "qwen3_5": lambda: _qwen3_5(batched=False),
    "qwen3_5_batched": lambda: _qwen3_5(batched=True),
    "qwen3_omni": _qwen3_omni,
    "whisper": _whisper,
    "higgs_audio": _higgs_audio,
    "orpheus": _orpheus,
}


def _runtime(case: Case, tmp_path):
    """A one-worker runtime holding one request in the case's decode walk."""
    section = Loop(
        name=case.loop,
        section=GraphNode(
            name=NODE, input_names=["text_inputs"],
            outputs=[
                GraphEdge(next_node=NODE, name="text_inputs"),
                GraphEdge(next_node=EMIT_TO_CLIENT, name="text_inputs", output_modality="text"),
            ],
        ),
        max_iters=1000,
        outputs=[],
    )
    tm = SharedMemoryCommunicationManager(
        my_entity_id="worker_0", hostname="localhost", device="cpu",
        communicator=SimpleNamespace(send=lambda *a, **k: None),
        shm_dir=str(tmp_path),
    )
    runtime = PythonGraphRuntime(
        my_worker_id="worker_0",
        my_worker_graphs=[WorkerGraph(
            section=section, graph_walks={case.walk}, ranks=[0], worker_graph_id=0,
        )],
        all_wg_ids_to_graph_walks={0: {case.walk}},
        all_wg_ids_to_dyn_loops={0: {case.loop}},
        all_wg_ids_to_nodes={0: {NODE}},
        node_to_partition={NODE: PARTITION},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        tensor_manager=tm,
    )
    rid = runtime.add_request(
        request_id="r", partition=PARTITION, graph_walk=case.walk,
        partition_worker_graph_ids=[0],
        worker_graph_to_workers=ParallelList([0], [["worker_0"]]),
    )
    tm.register_request(rid, runtime.get_sharding_config(rid))
    return tm, runtime, rid


def _client_tokens(case: Case, max_tokens: int, tmp_path, eos_at: int | None = None):
    """Prefill's token, then one per decode step until the model stops the loop."""
    tm, runtime, rid = _runtime(case, tmp_path)
    block = ColumnarEdgeSpecs.empty()
    block.add_edge(rid, GraphEdge(name="text_inputs", next_node=NODE))
    runtime.ingest_inputs_batch(block, can_buffer=True)

    info = SimpleNamespace(
        graph_walk=case.walk, max_tokens=max_tokens, step_metadata={},
        resource_configs={case.sampler: SimpleNamespace(ignore_eos=False)},
        dynamic_loop_iter_counts={},
    )
    runtime.update_dynamic_loop_iters({rid: info}, PARTITION)
    tokens = [TOKEN]
    for _ in range(max_tokens + 2):
        if not runtime.get_ready_nodes(set()):
            break
        runtime.pop_rids(NODE, case.walk, [rid])
        token = EOS if len(tokens) + 1 == eos_at else TOKEN
        # the batch carries a copy of the counts taken when it was built
        step_info = SimpleNamespace(**{
            **vars(info), "dynamic_loop_iter_counts": dict(info.dynamic_loop_iter_counts),
        })
        if case.stop(step_info, token):
            runtime.stop_loops_batched(
                PARTITION, case.walk, NODE, ParallelList([rid], [[case.loop]]),
            )
        runtime.cleanup_consumed_inputs(NODE, [rid], [0])
        stored = tm.store_and_return_tensor_info_batch(
            [rid], {rid: {"text_inputs": [torch.tensor([token])]}}, ["text_inputs"],
        )
        tm.increment_ref_batch_uniform(stored.flat_uuids, 1)
        runtime.complete_and_route_batch(RouteInput(
            partition=PARTITION, graph_walk=case.walk, node_name=NODE,
            output_signals=["text_inputs"], wg_ids=ParallelList([rid], [0]),
            tensors=stored.flat_uuids, num_tensors=stored.num_tensors,
        ))
        runtime.update_dynamic_loop_iters({rid: info}, PARTITION)
        tokens.append(token)
    return tokens


@pytest.mark.parametrize("max_tokens", [2, 3, 64])
@pytest.mark.parametrize("model", list(CASES))
def test_stops_at_max_output_tokens(model, max_tokens, tmp_path):
    assert len(_client_tokens(CASES[model](), max_tokens, tmp_path)) == max_tokens


@pytest.mark.parametrize("model", list(CASES))
def test_eos_before_the_limit_still_ends_the_loop(model, tmp_path):
    tokens = _client_tokens(CASES[model](), 64, tmp_path, eos_at=10)
    assert len(tokens) == 10 and tokens[-1] == EOS
