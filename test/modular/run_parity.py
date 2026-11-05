"""A step that repeats, through the real runner, over a stand-in capture.

``replay_run`` runs ``forward`` once as the first replay a step stages, then
``AcceleratorGraphRunner.repeat`` for the rest: each replay reads its static
inputs (what the last replay carried in, its iteration inputs' column) and
writes its outputs in place, as a captured graph does. A model's test runs
the same forward N times by hand and compares.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from types import SimpleNamespace

import torch

from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner


def replay_run(
    forward: Callable[..., Mapping[str, torch.Tensor]], static: dict[str, torch.Tensor], carries, repeats: int,
    columns=(), iteration: Mapping[str, torch.Tensor] | None = None, stop_buffers=(),
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """``forward(**static) -> {name: [rows, ...]}`` replayed ``repeats``
    times; the stop buffers and the ``columns``' outputs, a column a replay."""
    first = {name: value.clone() for name, value in static.items()}
    for name, column in (iteration or {}).items():
        first[name] = column[:, 0].clone()
    static.update(first)
    outputs = {name: tensor.clone() for name, tensor in forward(**static).items()}

    class Graph:
        def replay(self):
            for name, tensor in forward(**static).items():
                outputs[name].copy_(tensor)

    slot = SimpleNamespace(graph=Graph(), static_inputs=static, static_outputs=SimpleNamespace(
        check_stop_buffers={name: outputs[name] for name in stop_buffers}, row_outputs=outputs, output_spans=None,
    ))
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner.slot_for = lambda lease: slot
    stops, rows, _ = runner.repeat(None, carries, repeats, lambda: None, columns, iteration)
    return stops, rows
