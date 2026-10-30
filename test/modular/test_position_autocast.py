import types
from contextlib import nullcontext
from unittest import mock

import pytest
import torch

from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import RopeManager


def test_apply_qk_uses_autocast_for_the_tensor_device(monkeypatch):
    kernel_dtypes = []

    def record_kernel_dtypes(q, k, *args, **kwargs):
        kernel_dtypes.append((q.dtype, k.dtype))

    monkeypatch.setattr(
        torch.ops.mstar,
        "rope_apply_qk_inplace",
        record_kernel_dtypes,
    )

    manager = RopeManager(
        PositionConfig(kv_cache="kv"), torch.device("cpu"), max_seq_len=8,
    )
    manager._current_pos_ids["main"] = torch.arange(1)
    q = torch.ones((1, 1, 2), dtype=torch.float32)
    k = q.clone()

    with torch.amp.autocast("cpu", dtype=torch.float16):
        result_q, result_k = manager.apply_qk(q, k, "main")

    assert kernel_dtypes == [(torch.float16, torch.float16)]
    assert result_q.dtype == q.dtype
    assert result_k.dtype == k.dtype


def test_apply_qk_without_autocast_downcasts_fp32_and_warns_once(monkeypatch, caplog):
    """With autocast off, fp32 RoPE still runs in bf16 (the kernel needs it),
    and one warning names the node; under autocast nothing warns."""
    from mstar.engine import node_dtype
    from mstar.engine.accelerator_graph_runner import autocast_scope

    kernel_dtypes = []
    monkeypatch.setattr(
        torch.ops.mstar, "rope_apply_qk_inplace",
        lambda q, k, *args, **kwargs: kernel_dtypes.append(q.dtype),
    )
    monkeypatch.setattr(node_dtype, "_warned", set())
    manager = RopeManager(PositionConfig(kv_cache="kv"), torch.device("cpu"), max_seq_len=8)
    manager._current_pos_ids["main"] = torch.arange(1)
    q = torch.ones((1, 1, 2), dtype=torch.float32)

    with caplog.at_level("WARNING", logger=node_dtype.__name__):
        with autocast_scope(torch.bfloat16, "cpu", node_name="thinker"):
            manager.apply_qk(q, q.clone(), "main")
        for _ in range(2):
            with autocast_scope(None, "cpu", node_name="code2wav"):
                result_q, _ = manager.apply_qk(q, q.clone(), "main")

    assert kernel_dtypes == [torch.bfloat16] * 3
    assert result_q.dtype == torch.float32
    warnings = [r.getMessage() for r in caplog.records if r.name == node_dtype.__name__]
    assert len(warnings) == 1
    assert "RoPE" in warnings[0] and "code2wav" in warnings[0]


@pytest.mark.parametrize("static_outputs", [False, True])
def test_downcast_during_capture_names_the_node(monkeypatch, caplog, static_outputs):
    """A capture runs under the node's scope too, so a downcast it records
    names the node rather than ``<unknown>``."""
    from mstar.engine import accelerator_graph_runner, node_dtype
    from mstar.utils.device import DeviceCapability

    backend = types.SimpleNamespace(
        device_type="cpu", current_stream=lambda: None, create_graph=lambda keep_graph=False: object(),
        capture=lambda graph, pool: nullcontext(), synchronize=lambda: None,
        recover_failed_capture=lambda pool, stream: None,
    )
    monkeypatch.setattr(DeviceCapability, "of", lambda device: types.SimpleNamespace(graph_backend=backend))
    monkeypatch.setattr(
        torch.ops.mstar, "rope_apply_qk_inplace", lambda q, k, *args, **kwargs: None,
    )
    monkeypatch.setattr(node_dtype, "_warned", set())
    manager = RopeManager(PositionConfig(kv_cache="kv"), torch.device("cpu"), max_seq_len=8)
    manager._current_pos_ids["main"] = torch.arange(1)
    q = torch.ones((1, 1, 2), dtype=torch.float32)

    def run():
        return {"q": manager.apply_qk(q, q.clone(), "main")[0]}

    with caplog.at_level("WARNING", logger=node_dtype.__name__):
        if static_outputs:
            accelerator_graph_runner.capture_with_static_outputs(
                run, {"q": q.clone()}, mock.Mock(), "cpu", None, node_name="code2wav",
            )
        else:
            accelerator_graph_runner.capture_into_graph(run, mock.Mock(), "cpu", None, node_name="code2wav")

    warnings = [r.getMessage() for r in caplog.records if r.name == node_dtype.__name__]
    assert len(warnings) == 1
    assert "code2wav" in warnings[0] and "<unknown>" not in warnings[0]
