import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import KVReqConfig, PagedKVConfig
from mstar.engine.resources.kv.manager import (
    CacheStream,
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
from mstar.model.bagel.components.modeling_utils import TimestepEmbedder
from mstar.model.bagel.submodules import CombineCFGSubmodule


def test_offload_reset_preserves_remote_content_epoch():
    stream = CacheStream(remote_reset_generation=7)
    stream.reset(content_reset=False)
    assert stream.remote_reset_generation == 7
    stream.reset()
    assert stream.remote_reset_generation is None


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
    model._image_gen_remote_handoff = False
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
    assert not model._image_gen_remote_handoff

    model.get_worker_graphs(str(root / "configs/bagel_pd_disaggregated.yaml"))
    assert model._has_llm_disaggregation
    assert not model._image_gen_remote_handoff

    model.get_worker_graphs(str(root / "configs/bagel_img_disaggregated.yaml"))
    assert model._image_gen_remote_handoff
    image_gen = model.get_graph_walk_graphs()["image_gen"]
    assert not image_gen.sections[0].section.enable_async_scheduling


def test_timestep_embedding_preserves_fp32_frequencies_after_bf16_cast():
    module = TimestepEmbedder(64, frequency_embedding_size=256).to(
        dtype=torch.bfloat16
    )
    expected = torch.exp(
        -torch.log(torch.tensor(10000.0))
        * torch.arange(128, dtype=torch.float32)
        / 128
    )

    assert module.timestep_freqs.dtype == torch.float32
    torch.testing.assert_close(module.timestep_freqs, expected, rtol=0, atol=0)


def test_timestep_embedding_survives_meta_to_empty_round_trip():
    module = TimestepEmbedder(64, frequency_embedding_size=256).to("meta")
    module.to_empty(device="cpu")
    expected = torch.exp(
        -torch.log(torch.tensor(10000.0))
        * torch.arange(128, dtype=torch.float32)
        / 128
    )
    torch.testing.assert_close(module.timestep_freqs, expected, rtol=0, atol=0)


def test_timestep_embedding_buffer_matches_reference_formula():
    module = TimestepEmbedder(64, frequency_embedding_size=256)
    timesteps = torch.tensor([0.0, 0.25, 0.5, 1.0])
    buffered = module(timesteps)
    reference = module.mlp(
        module.timestep_embedding(timesteps, module.frequency_embedding_size)
    )
    torch.testing.assert_close(buffered, reference, rtol=0, atol=0)

def test_combine_cfg_is_parameterless():
    module = CombineCFGSubmodule(SimpleNamespace())
    assert list(module.parameters()) == []


def _kv_cache(tensor: torch.Tensor) -> KVCache:
    config = PagedKVConfig(
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

    assert refreshed.path != info.path
    assert Path(info.path).exists()
    assert producer.get_kv_transfer_info(
        request_id="request", label="main", page_indices=[0], seq_len=2,
    ) == refreshed
    torch.testing.assert_close(
        torch.load(refreshed.path, weights_only=True),
        source_cache.tensor,
    )
    producer.remove_request("request")
    assert not Path(info.path).exists()
    assert not Path(refreshed.path).exists()


def test_shm_remove_waits_for_in_progress_publication(tmp_path, monkeypatch):
    source = torch.zeros((1, 1, 2, 4, 1, 1), dtype=torch.float32)
    producer = ShmKVTransferEngine(
        _kv_cache(source), "producer", str(tmp_path),
    )
    saving = threading.Event()
    continue_save = threading.Event()
    removed = threading.Event()
    errors = []
    save = torch.save

    def blocked_save(*args, **kwargs):
        saving.set()
        assert continue_save.wait(timeout=5)
        save(*args, **kwargs)

    monkeypatch.setattr(torch, "save", blocked_save)

    def publish():
        try:
            producer.get_kv_transfer_info(
                request_id="request", label="main",
                page_indices=[0], seq_len=1,
            )
        except BaseException as error:
            errors.append(error)

    def remove():
        try:
            producer.remove_request("request")
        except BaseException as error:
            errors.append(error)
        finally:
            removed.set()

    publish_thread = threading.Thread(target=publish)
    remove_thread = threading.Thread(target=remove)
    publish_thread.start()
    assert saving.wait(timeout=5)
    remove_thread.start()
    assert not removed.wait(timeout=0.05)
    continue_save.set()
    publish_thread.join(timeout=5)
    remove_thread.join(timeout=5)

    assert not publish_thread.is_alive()
    assert not remove_thread.is_alive()
    assert not errors
    assert not list(tmp_path.glob("*.pt"))


def test_shm_publication_refreshes_when_reset_generation_changes(tmp_path):
    source = torch.zeros((1, 1, 2, 4, 1, 1), dtype=torch.float32)
    source_cache = _kv_cache(source)
    producer = ShmKVTransferEngine(source_cache, "producer", str(tmp_path))

    info = producer.get_kv_transfer_info(
        request_id="request",
        label="main",
        page_indices=[0],
        seq_len=2,
        reset_generation=0,
    )
    source_cache.tensor.fill_(7)
    refreshed = producer.get_kv_transfer_info(
        request_id="request",
        label="main",
        page_indices=[0],
        seq_len=2,
        reset_generation=1,
    )

    assert refreshed.path != info.path
    assert producer.owns_transfer_info(info, "request", "main")
    assert producer.owns_transfer_info(refreshed, "request", "main")
    torch.testing.assert_close(
        torch.load(refreshed.path, weights_only=True),
        source_cache.tensor,
    )
    producer.remove_request("request")
    assert not Path(info.path).exists()
    assert not Path(refreshed.path).exists()


def test_shm_earlier_descriptor_keeps_its_snapshot_after_republish(tmp_path):
    source = torch.zeros((1, 2, 2, 4, 1, 1), dtype=torch.float32)
    producer_cache = _kv_cache(source)
    consumer_cache = _kv_cache(source)
    producer = ShmKVTransferEngine(
        producer_cache, "producer", str(tmp_path),
    )
    consumer = ShmKVTransferEngine(
        consumer_cache, "consumer", str(tmp_path),
    )

    producer_cache.tensor[:, 0].fill_(1)
    earlier = producer.get_kv_transfer_info(
        request_id="request", label="main", page_indices=[0],
        seq_len=4, reset_generation=0,
    )
    producer_cache.tensor[:, 1].fill_(2)
    later = producer.get_kv_transfer_info(
        request_id="request", label="main", page_indices=[1],
        seq_len=4, reset_generation=1,
    )

    assert earlier.path != later.path
    assert consumer.read_batched_async(
        earlier, [KVReadInfo(0, 0, 0, 0, 4)],
    ) is None
    assert torch.all(consumer_cache.tensor[:, 0] == 1)
    assert consumer.read_batched_async(
        later, [KVReadInfo(0, 0, 1, 0, 4)],
    ) is None
    assert torch.all(consumer_cache.tensor[:, 0] == 2)

    producer.shutdown()
    assert not Path(earlier.path).exists()
    assert not Path(later.path).exists()


def test_shm_repeated_prefill_stores_only_changed_tail_pages(tmp_path):
    source = torch.zeros((1, 3, 2, 4, 1, 1), dtype=torch.float32)
    producer_cache = _kv_cache(source)
    consumer_cache = _kv_cache(source)
    producer = ShmKVTransferEngine(
        producer_cache, "producer", str(tmp_path),
    )
    consumer = ShmKVTransferEngine(
        consumer_cache, "consumer", str(tmp_path),
    )

    publications = []
    for token in range(1, 13):
        page, offset = divmod(token - 1, 4)
        producer_cache.tensor[:, page, :, offset].fill_(token)
        publications.append(producer.get_kv_transfer_info(
            request_id="request",
            label="main",
            page_indices=list(range(page + 1)),
            seq_len=token,
            reset_generation=0,
        ))

    latest = publications[-1]
    assert len(latest.chunks) == 12
    assert all(len(chunk.page_indices) == 1 for chunk in latest.chunks)
    assert sum(
        torch.load(chunk.path, weights_only=True).shape[1]
        for chunk in latest.chunks
    ) == 12
    assert consumer.read_batched_async(
        latest, [KVReadInfo(0, page, page, 0, 4) for page in range(3)],
    ) is None
    torch.testing.assert_close(
        consumer_cache.tensor[:, :3],
        producer_cache.tensor[:, :3],
    )

    consumer_cache.tensor.zero_()
    assert consumer.read_batched_async(
        publications[5],
        [KVReadInfo(0, 0, 0, 0, 4), KVReadInfo(0, 1, 1, 0, 2)],
    ) is None
    torch.testing.assert_close(
        consumer_cache.tensor[:, 0],
        producer_cache.tensor[:, 0],
    )
    torch.testing.assert_close(
        consumer_cache.tensor[:, 1, :, :2],
        producer_cache.tensor[:, 1, :, :2],
    )
    assert torch.count_nonzero(consumer_cache.tensor[:, 1, :, 2:]) == 0

    producer.remove_request("request")
    assert all(not Path(chunk.path).exists() for chunk in latest.chunks)


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
        cfg=PagedKVConfig(
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
        needs_remote_transfer=True,
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


def _shm_consumer_manager(tmp_path) -> KVManager:
    return KVManager(
        cfg=PagedKVConfig(
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
        needs_remote_transfer=True,
    )


def _shm_publication(
    manager: KVManager,
    path: Path,
    tensor: torch.Tensor,
    seq_len: int,
    reset_generation: int,
) -> PublishedKVInfo:
    torch.save(tensor, path)
    return PublishedKVInfo.build_for_rank(
        rank=0,
        world_size=1,
        seq_info={
            "main": KVSequenceInfo(
                seq_len=seq_len,
                latest_kv_transfer_info=ShmKVTransferInfo(
                    path=str(path),
                    page_indices=(0,),
                    layout=manager.kv_cache.layout,
                ),
                page_indices=[0],
                reset_generation=reset_generation,
            )
        },
    )


def test_same_reset_generation_retrieves_only_appended_kv(tmp_path):
    manager = _shm_consumer_manager(tmp_path)
    manager.ingest_request("request", KVReqConfig(needed_labels=["main"]))
    path = tmp_path / "published.pt"

    first_source = torch.zeros((1, 1, 2, 4, 1, 1), dtype=torch.float32)
    first_source[:, 0, :, :2] = 1
    first = manager.admit_retrieve(
        "request",
        "consumer",
        "decode",
        _shm_publication(
            manager, path, first_source, seq_len=2, reset_generation=0,
        ),
    )
    page = manager._streams["request"]["main"].page_indices[0]
    assert first.ok and first.ready
    torch.testing.assert_close(
        manager.kv_cache.tensor[:, page, :, :2],
        first_source[:, 0, :, :2],
    )

    # Deliberately change the published prefix too: an incremental read must
    # leave the receiver's already-cached prefix untouched.
    appended_source = torch.full_like(first_source, 9)
    appended_source[:, 0, :, 2] = 2
    appended = manager.admit_retrieve(
        "request",
        "consumer",
        "decode",
        _shm_publication(
            manager, path, appended_source, seq_len=3, reset_generation=0,
        ),
    )
    assert appended.ok and appended.ready
    torch.testing.assert_close(
        manager.kv_cache.tensor[:, page, :, :2],
        first_source[:, 0, :, :2],
    )
    torch.testing.assert_close(
        manager.kv_cache.tensor[:, page, :, 2],
        appended_source[:, 0, :, 2],
    )


def test_remote_reset_generation_replaces_same_length_cached_kv(tmp_path):
    manager = _shm_consumer_manager(tmp_path)
    manager.ingest_request("request", KVReqConfig(needed_labels=["main"]))
    path = tmp_path / "published.pt"

    first_source = torch.ones((1, 1, 2, 4, 1, 1), dtype=torch.float32)
    first = manager.admit_retrieve(
        "request",
        "consumer",
        "decode",
        _shm_publication(
            manager, path, first_source, seq_len=2, reset_generation=0,
        ),
    )
    assert first.ok and first.ready

    second_source = torch.full_like(first_source, 2)
    second = manager.admit_retrieve(
        "request",
        "consumer",
        "decode",
        _shm_publication(
            manager, path, second_source, seq_len=2, reset_generation=1,
        ),
    )
    page = manager._streams["request"]["main"].page_indices[0]
    assert second.ok and second.ready
    torch.testing.assert_close(
        manager.kv_cache.tensor[:, page, :, :2],
        second_source[:, 0, :, :2],
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
    factory_calls = []
    manager = KVTransferManager(
        TransferEngineInfo(
            my_entity_id="producer",
            my_session_id="session",
            transfer_engine=LocalTransferEngine("producer"),
            shm_dir_factory=lambda: factory_calls.append(True) or "/unused",
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
    assert factory_calls == []
