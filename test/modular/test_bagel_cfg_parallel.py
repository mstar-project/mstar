from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import KVConfig
from mstar.engine.resources.kv.transfer import KVReadInfo, ShmKVTransferEngine
from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.bagel.submodules import CombineCFGSubmodule


def _bare_model() -> BagelModel:
    model = BagelModel.__new__(BagelModel)
    model.config = SimpleNamespace(num_timesteps=50)
    model._has_cfg_parallel = True
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


def test_xpu_config_only_changes_cfg_replica_placement():
    root = Path(__file__).parents[2]
    with open(root / "configs/bagel_cfg_parallel.yaml") as f:
        h100_groups = yaml.safe_load(f)["node_groups"]
    with open(root / "configs/bagel_xpu_cfg_tp2.yaml") as f:
        xpu_config = yaml.safe_load(f)
    xpu_groups = xpu_config["node_groups"]

    assert "kv_cache" not in xpu_config
    assert xpu_config["resources"]["kv"]["max_num_pages"] == 1024
    assert xpu_config["resources"]["attn"]["backend"] == "xpu_paged"

    def cfg_walks(groups, node):
        return next(g.get("graph_walks") for g in groups if node in g["node_names"])

    assert cfg_walks(h100_groups, "LLM_cfg_text") == ["image_gen_cfg"]
    assert cfg_walks(xpu_groups, "LLM_cfg_text") == ["image_gen_cfg"]
    assert cfg_walks(h100_groups, "LLM_cfg_img") == ["image_gen_cfg"]
    assert cfg_walks(xpu_groups, "LLM_cfg_img") == ["image_gen_cfg"]

    xpu_cfg_groups = [
        g for g in xpu_groups
        if any(n.startswith("LLM_cfg_") for n in g["node_names"])
    ]
    assert all(g["tp_size"] == 2 and len(g["ranks"]) == 2 for g in xpu_cfg_groups)


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
