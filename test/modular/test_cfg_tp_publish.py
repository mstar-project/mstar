from types import SimpleNamespace

from mstar.conductor.conductor import Conductor
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionState,
)
from mstar.engine.resources.kv.manager import KVSequenceInfo, PublishedKVInfo
from mstar.utils.ipc_format import WorkerGraphsDone


def _rank_publish(rank: int) -> PublishedKVInfo:
    return PublishedKVInfo.build_for_rank(
        rank=rank,
        world_size=2,
        seq_info={
            "cfg_text": KVSequenceInfo(
                seq_len=4 + rank,
                latest_kv_transfer_info=f"rank-{rank}",
                page_indices=[rank],
            )
        },
    )


def test_conductor_merges_kv_publish_info_from_every_tp_rank():
    partition = PartitionState(
        partition_name="default",
        metadata=CurrentForwardConductorMetadata(
            graph_walk="image_gen_cfg",
            is_prefill=False,
        ),
        current_worker_graph_ids={"wg"},
    )
    request = SimpleNamespace(
        partition_states={"default": partition},
        final_outputs={},
        persist_signals={},
        streaming_connections={},
        worker_graph_to_workers={"wg": ["worker_0", "worker_1"]},
    )
    conductor = Conductor.__new__(Conductor)
    conductor.enable_prof = False
    conductor.requests = {"request": request}

    for rank in (1, 0):
        done = conductor._process_worker_graphs_done(
            WorkerGraphsDone(
                request_id="request",
                worker_graph_ids=["wg"],
                is_first_tp_rank=rank == 0,
                resource_publish_info={"kv": _rank_publish(rank)},
            )
        )

    published = partition.resource_publish_info["kv"]
    assert published.get(0)["cfg_text"].latest_kv_transfer_info == "rank-0"
    assert published.get(1)["cfg_text"].latest_kv_transfer_info == "rank-1"
    assert done == ["default"]
