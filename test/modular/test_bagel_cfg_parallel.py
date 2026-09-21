import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig
from mstar.engine.resources.kv.manager import (
    KVManager,
    KVSequenceInfo,
    PublishedKVInfo,
)
from mstar.engine.resources.kv.transfer import (
    KVReadInfo,
    KVTransferManager,
    LocalOnlyKVTransferEngine,
    ShmKVTransferEngine,
    ShmKVTransferInfo,
    TransferEngineInfo,
    make_deployment_kv_shm_dir,
)
from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.bagel.submodules import CombineCFGSubmodule


def _bare_model() -> BagelModel:
    model = BagelModel.__new__(BagelModel)
    model.config = SimpleNamespace(
        num_timesteps=50,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        ignore_eos=False,
        vocab_size=32,
    )
    model._has_cfg_parallel = True
    model._has_llm_disaggregation = False
    return model


def test_cfg_graph_is_reused_for_tp_branches():
    model = _bare_model()
    walks = model.get_graph_walk_graphs()

    assert set(walks) == {
        "prefill_text", "prefill_vit", "prefill_vae", "decode",
        "image_gen", "image_gen_cfg",
    }
    assert model.get_default_sharding_config().tp_enabled_nodes == {
        "LLM", "LLM_cfg_text", "LLM_cfg_img",
    }
    assert set(walks["image_gen_cfg"].get_nodes()) == {
        "LLM", "LLM_cfg_text", "LLM_cfg_img", "combine_cfg", "vae_decoder",
    }


def test_xpu_cfg_replicas_match_main_tp_and_walk():
    root = Path(__file__).parents[2]
    with open(root / "configs/bagel_xpu_cfg_tp2.yaml") as f:
        groups = yaml.safe_load(f)["node_groups"]

    main = next(g for g in groups if "LLM" in g["node_names"])
    cfg_groups = [
        g for g in groups
        if {"LLM_cfg_text", "LLM_cfg_img"} & set(g["node_names"])
    ]

    assert len(cfg_groups) == 2
    assert all(group["tp_size"] == main["tp_size"] for group in cfg_groups)
    assert all(group.get("graph_walks") == ["image_gen_cfg"] for group in cfg_groups)


def test_pd_disaggregation_publishes_main_only_at_handoff_boundaries():
    model = _bare_model()
    model._has_llm_disaggregation = True
    args = {
        "default": SimpleNamespace(
            full_metadata=SimpleNamespace(kwargs={"requires_cfg": True})
        )
    }

    config = model.get_request_resource_configs(args)["kv"]

    assert config.publish_labels_per_node_walk == {
        ("LLM", "prefill_text"): ["main", "cfg_text", "cfg_img"],
        ("LLM", "prefill_vit"): ["main", "cfg_text"],
        ("LLM", "prefill_vae"): ["main", "cfg_text"],
    }
    assert config.final_publish_labels_per_node_walk == {
        ("LLM", "decode"): ["main", "cfg_img"],
    }
    assert config.get_publish_labels("LLM", "decode", ["main", "cfg_img"]) == []

    args["default"].full_metadata.kwargs["requires_cfg"] = False
    config = model.get_request_resource_configs(args)["kv"]
    assert {
        tuple(labels)
        for labels in config.publish_labels_per_node_walk.values()
    } == {("main",)}
    assert config.final_publish_labels_per_node_walk == {
        ("LLM", "decode"): ["main"],
    }


def test_bagel_configs_detect_only_graph_walk_split_llm_as_disaggregated():
    root = Path(__file__).parents[2]
    model = _bare_model()

    model.get_worker_graphs(str(root / "configs/bagel_cfg_parallel.yaml"))
    assert not model._has_llm_disaggregation

    model.get_worker_graphs(str(root / "configs/bagel_pd_disaggregated.yaml"))
    assert model._has_llm_disaggregation


def test_combine_cfg_is_parameterless():
    module = CombineCFGSubmodule(SimpleNamespace())
    assert list(module.parameters()) == []


def _kv_cache(tensor: torch.Tensor) -> KVCache:
    config = KVConfig(
        max_num_pages=tensor.shape[1],
        page_size=tensor.shape[3],
        num_layers=tensor.shape[0],
        num_kv_heads=tensor.shape[4],
        head_dim=tensor.shape[5],
        max_seq_len=tensor.shape[1] * tensor.shape[3],
    )
    cache = KVCache(config, torch.device("cpu"), tensor.dtype)
    cache.tensor.copy_(tensor)
    return cache


def test_shm_kv_transfer_copies_only_requested_page_ranges(tmp_path):
    source = torch.arange(
        2 * 4 * 2 * 4 * 1 * 2, dtype=torch.float32
    ).reshape(2, 4, 2, 4, 1, 2)
    destination = torch.zeros_like(source)
    source_cache = _kv_cache(source)
    destination_cache = _kv_cache(destination)
    producer = ShmKVTransferEngine(source_cache, "producer", str(tmp_path))
    consumer = ShmKVTransferEngine(destination_cache, "consumer", str(tmp_path))

    info = producer.get_kv_transfer_info(
        request_id="request", label="cfg_text", page_indices=[1, 3], seq_len=6,
    )
    reads = []
    for layer in range(2):
        reads.extend([
            KVReadInfo(layer, 0, 1, 0, 4),
            KVReadInfo(layer, 2, 3, 0, 2),
        ])
    consumer.read_batched_async(info, reads)

    torch.testing.assert_close(destination_cache.tensor[:, 0], source[:, 1])
    torch.testing.assert_close(
        destination_cache.tensor[:, 2, :, :2],
        source[:, 3, :, :2],
    )
    assert torch.count_nonzero(destination_cache.tensor[:, 1]) == 0


def test_shm_kv_transfer_requires_deployment_directory():
    cache = _kv_cache(torch.zeros((1, 1, 2, 4, 1, 1)))

    with pytest.raises(
        ValueError,
        match="shm_dir is required for shared-memory KV transfer",
    ):
        ShmKVTransferEngine(cache, "producer", None)  # type: ignore[arg-type]


def test_shm_publication_refreshes_when_seq_len_changes(tmp_path):
    source = torch.zeros((1, 1, 2, 4, 1, 1), dtype=torch.float32)
    source_cache = _kv_cache(source)
    producer = ShmKVTransferEngine(source_cache, "producer", str(tmp_path))

    info = producer.get_kv_transfer_info(
        request_id="request", label="main", page_indices=[0], seq_len=1,
    )
    source_cache.tensor.fill_(7)
    refreshed = producer.get_kv_transfer_info(
        request_id="request", label="main", page_indices=[0], seq_len=2,
    )

    assert refreshed.path == info.path
    torch.testing.assert_close(
        torch.load(refreshed.path, weights_only=True),
        source_cache.tensor,
    )
    producer.remove_request("request")
    assert not Path(refreshed.path).exists()


def test_shm_publications_are_namespaced_by_resource(tmp_path):
    source = torch.zeros((1, 1, 2, 4, 1, 1), dtype=torch.float32)
    source_cache = _kv_cache(source)
    first = ShmKVTransferEngine(
        source_cache,
        "producer",
        str(tmp_path),
        resource_key="thinker_kv",
    )
    second = ShmKVTransferEngine(
        source_cache,
        "producer",
        str(tmp_path),
        resource_key="talker_kv",
    )

    first_info = first.get_kv_transfer_info(
        request_id="request", label="main", page_indices=[0], seq_len=1,
    )
    second_info = second.get_kv_transfer_info(
        request_id="request", label="main", page_indices=[0], seq_len=1,
    )

    assert first_info.path != second_info.path
    assert first.owns_transfer_info(first_info, "request", "main")
    assert not first.owns_transfer_info(second_info, "request", "main")

    first.shutdown()
    second.shutdown()


def test_shm_read_failure_is_returned_on_a_future(tmp_path):
    cache = _kv_cache(torch.zeros((1, 1, 2, 4, 1, 1)))
    consumer = ShmKVTransferEngine(cache, "consumer", str(tmp_path))
    missing = ShmKVTransferInfo(
        path=str(tmp_path / "missing.pt"),
        page_indices=(0,),
        layout=cache.layout,
    )

    future = consumer.read_batched_async(
        missing, [KVReadInfo(0, 0, 0, 0, 1)],
    )

    assert future is not None and future.done()
    with pytest.raises(FileNotFoundError):
        future.result()


def test_shm_read_failure_is_latched_to_one_request(tmp_path):
    manager = KVManager(
        cfg=KVConfig(
            max_num_pages=4,
            page_size=4,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            max_seq_len=16,
        ),
        name="kv",
        joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="consumer",
            my_session_id="session",
            transfer_engine=LocalTransferEngine("consumer"),
            shm_dir=str(tmp_path),
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    manager.ingest_request("broken", KVReqConfig(needed_labels=["main"]))
    manager.ingest_request("healthy", KVReqConfig(needed_labels=["main"]))
    missing = ShmKVTransferInfo(
        path=str(tmp_path / "missing.pt"),
        page_indices=(0,),
        layout=manager.kv_cache.layout,
    )
    broken_publish = PublishedKVInfo.build_for_rank(
        rank=0,
        world_size=1,
        seq_info={
            "main": KVSequenceInfo(
                seq_len=1,
                latest_kv_transfer_info=missing,
                page_indices=[0],
            ),
        },
    )
    healthy_path = tmp_path / "healthy.pt"
    healthy_kv = torch.ones((1, 1, 2, 4, 1, 1), dtype=torch.float32)
    torch.save(healthy_kv, healthy_path)
    healthy_publish = PublishedKVInfo.build_for_rank(
        rank=0,
        world_size=1,
        seq_info={
            "main": KVSequenceInfo(
                seq_len=1,
                latest_kv_transfer_info=ShmKVTransferInfo(
                    path=str(healthy_path),
                    page_indices=(0,),
                    layout=manager.kv_cache.layout,
                ),
                page_indices=[0],
            ),
        },
    )

    broken = manager.admit_retrieve(
        "broken", "LLM_cfg_text", "image_gen_cfg", broken_publish,
    )
    broken_again = manager.admit_retrieve(
        "broken", "LLM_cfg_text", "image_gen_cfg", broken_publish,
    )
    healthy = manager.admit_retrieve(
        "healthy", "LLM_cfg_text", "image_gen_cfg", healthy_publish,
    )

    healthy_page = manager._streams["healthy"]["main"].page_indices[0]

    assert not broken.ok
    assert "FileNotFoundError" in broken.reason.message
    assert not broken_again.ok
    assert "FileNotFoundError" in broken_again.reason.message
    assert healthy.ok and healthy.ready
    torch.testing.assert_close(
        manager.kv_cache.tensor[:, healthy_page, :, :1],
        healthy_kv[:, 0, :, :1],
    )


def test_kv_shm_directory_is_private_to_the_deployment(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("MSTAR_KV_SHM_DIR", str(tmp_path))
    owner_pid = os.getpid()
    uid = os.getuid() if hasattr(os, "getuid") else 0
    stale = tmp_path / f"mstar_kv_{uid}_99999999_stale"
    stale.mkdir()
    (stale / "orphan.pt").touch()

    first = make_deployment_kv_shm_dir(
        "/tmp/server-a", "tcp://host:1234", owner_pid=owner_pid,
    )
    second = make_deployment_kv_shm_dir(
        "/tmp/server-b", "tcp://host:5678", owner_pid=owner_pid,
    )

    assert first != second
    assert not stale.exists()
    assert Path(first).stat().st_mode & 0o777 == 0o700
    assert Path(second).stat().st_mode & 0o777 == 0o700


def test_local_only_cpu_cache_does_not_publish_shm_snapshots():
    source = torch.zeros((1, 1, 2, 4, 1, 1), dtype=torch.float32)
    manager = KVTransferManager(
        TransferEngineInfo(
            my_entity_id="producer",
            my_session_id="session",
            transfer_engine=LocalTransferEngine("producer"),
        ),
        _kv_cache(source),
        resource_key="local_kv",
        needs_remote_transfer=False,
    )

    assert isinstance(
        manager._kv_transfer_engine, LocalOnlyKVTransferEngine
    )
    assert manager.get_kv_transfer_info(
        request_id="request",
        label="main",
        page_indices=[0],
        seq_len=1,
    ) is None
