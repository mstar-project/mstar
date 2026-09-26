from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, ".")

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.conductor import Conductor
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionState,
    merge_publish_info,
)
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig, KVStep
from mstar.engine.resources.kv.manager import (
    KVManager,
    KVSequenceInfo,
    PublishedKVInfo,
)
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import Segment, StepContext
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


def _tp2_manager(rank: int, entity_id: str, shm_dir: str) -> KVManager:
    return KVManager(
        cfg=KVConfig(
            num_layers=1,
            num_kv_heads=2,
            num_qo_heads=2,
            head_dim=1,
            max_seq_len=16,
            max_num_pages=8,
            page_size=4,
        ),
        name="kv",
        joint_comm_group=SimpleNamespace(rank=rank, world_size=2),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id=entity_id,
            my_session_id="local",
            transfer_engine=LocalTransferEngine("local"),
            shm_dir=shm_dir,
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_tp2_kv_managers_transfer_each_published_rank_shard(tmp_path):
    """The rank aggregation must drive real rank-matched SHM retrievals."""
    request_id = "request"
    req_config = KVReqConfig(needed_labels=["cfg_text"])
    producers = [
        _tp2_manager(rank, f"producer-{rank}", str(tmp_path))
        for rank in range(2)
    ]
    consumers = [
        _tp2_manager(rank, f"consumer-{rank}", str(tmp_path))
        for rank in range(2)
    ]

    published: dict = {}
    try:
        for rank, producer in enumerate(producers):
            producer.ingest_request(request_id, req_config)
            step = KVStep(
                segments=(Segment(request_id, "cfg_text", 4),),
            )
            ctx = StepContext(
                request_ids=(request_id,),
                graph_walk="prefill_text",
                slot=0,
                capture=False,
            )
            assert producer.admit(step, ctx).ok
            producer.commit(step, ctx)
            page = producer._streams[request_id]["cfg_text"].page_indices[0]
            producer.kv_cache.tensor[:, page].fill_(rank + 1)
            merge_publish_info(published, {"kv": producer.publish(request_id)})

        rank_shards = published["kv"]
        assert set(rank_shards.info) == {0, 1}

        for rank, consumer in enumerate(consumers):
            consumer.ingest_request(request_id, req_config)
            outcome = consumer.admit_retrieve(
                request_id,
                node_name="LLM",
                graph_walk="image_gen_cfg",
                published=rank_shards,
            )
            assert outcome.ok and outcome.ready
            stream = consumer._streams[request_id]["cfg_text"]
            assert stream.stored_len == 4
            page = stream.page_indices[0]
            actual = consumer.kv_cache.tensor[:, page, :, :4]
            assert torch.all(actual == rank + 1)
    finally:
        for manager in consumers + producers:
            manager.remove_request(request_id)
            manager.cleanup()
