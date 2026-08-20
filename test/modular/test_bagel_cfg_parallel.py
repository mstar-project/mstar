from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from mstar.engine.kv_store import KVReadInfo, ShmKVTransferEngine
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
        xpu_groups = yaml.safe_load(f)["node_groups"]

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


def test_shm_kv_transfer_copies_only_requested_page_ranges(tmp_path):
    source = torch.arange(
        2 * 4 * 2 * 4 * 1 * 2, dtype=torch.float32
    ).reshape(2, 4, 2, 4, 1, 2)
    destination = torch.zeros_like(source)
    producer = ShmKVTransferEngine(source, "producer", str(tmp_path))
    consumer = ShmKVTransferEngine(destination, "consumer", str(tmp_path))

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

    torch.testing.assert_close(destination[:, 0], source[:, 1])
    torch.testing.assert_close(destination[:, 2, :, :2], source[:, 3, :, :2])
    assert torch.count_nonzero(destination[:, 1]) == 0


def test_shm_publication_refreshes_when_seq_len_changes(tmp_path):
    source = torch.zeros((1, 1, 2, 4, 1, 1), dtype=torch.float32)
    producer = ShmKVTransferEngine(source, "producer", str(tmp_path))

    info = producer.get_kv_transfer_info(
        request_id="request", label="main", page_indices=[0], seq_len=1,
    )
    source.fill_(7)
    refreshed = producer.get_kv_transfer_info(
        request_id="request", label="main", page_indices=[0], seq_len=2,
    )

    assert refreshed.path == info.path
    torch.testing.assert_close(
        torch.load(refreshed.path, weights_only=True), source
    )
    producer.remove_request("request")
    assert not Path(refreshed.path).exists()
