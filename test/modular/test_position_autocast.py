import sys
import types

import torch

from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import RopeManager


def test_apply_qk_uses_autocast_for_the_tensor_device(monkeypatch):
    kernel_dtypes = []
    flashinfer = types.SimpleNamespace(
        rope=types.SimpleNamespace(
            apply_rope_pos_ids_inplace=lambda q, k, pos_ids, **kwargs: kernel_dtypes.append((q.dtype, k.dtype)),
        )
    )
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)

    manager = RopeManager(PositionConfig(kv_cache="kv"), torch.device("cpu"))
    manager._current_pos_ids["main"] = torch.arange(1)
    q = torch.ones((1, 1, 2), dtype=torch.float32)
    k = q.clone()

    with torch.amp.autocast("cpu", dtype=torch.float16):
        result_q, result_k = manager.apply_qk(q, k, "main")

    assert kernel_dtypes == [(torch.float16, torch.float16)]
    assert result_q.dtype == q.dtype
    assert result_k.dtype == k.dtype
