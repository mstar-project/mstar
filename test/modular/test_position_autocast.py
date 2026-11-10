import torch

from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import RopeManager


def test_apply_qk_follows_the_node_autocast_else_the_input_dtype(monkeypatch):
    """Under a node whose policy autocasts, RoPE runs in its dtype; with autocast
    off, fp32 q/k stay fp32 (the kernel's portable path); rope_dtype overrides."""
    from mstar.engine.accelerator_graph_runner import autocast_scope

    kernel_dtypes = []
    monkeypatch.setattr(
        torch.ops.mstar, "rope_apply_qk_inplace",
        lambda q, k, *args, **kwargs: kernel_dtypes.append(q.dtype),
    )
    manager = RopeManager(PositionConfig(kv_cache="kv"), torch.device("cpu"), max_seq_len=8)
    manager._current_pos_ids["main"] = torch.arange(1)
    q = torch.ones((1, 1, 2), dtype=torch.float32)

    with autocast_scope(torch.bfloat16, "cpu"):
        manager.apply_qk(q, q.clone(), "main")
    with autocast_scope(None, "cpu"):
        result_q, _ = manager.apply_qk(q, q.clone(), "main")

    manager.apply_qk(q, q.clone(), "main", rope_dtype=torch.float16)

    assert kernel_dtypes == [torch.bfloat16, torch.float32, torch.float16]
    assert result_q.dtype == torch.float32
