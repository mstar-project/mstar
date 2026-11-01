"""``NodeSubmodule.emit_from_host_rows`` through ``Engine.check_stop_for_batch``: what it cuts
from the host rows is routed with the step's outputs and seen by ``check_stop``; a raise
fails the batch's requests."""
from types import SimpleNamespace

import torch

from mstar.engine.engine import Engine
from mstar.model.submodule_base import BatchedModelOutput, HostRows, NodeSubmodule


class _Cutter(NodeSubmodule):
    """Emits the first ``n`` tokens of its row, ``n`` the row's last column."""

    def __init__(self, fail=False):
        super().__init__()
        self.fail, self.calls, self.seen = fail, [], {}

    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def emit_from_host_rows(self, request_ids, request_infos, host_rows, step_walk=None):
        self.calls.append((list(request_ids), step_walk))
        if self.fail:
            raise RuntimeError("bad rows")
        rows = dict(zip(host_rows.request_ids, host_rows.buffers["verdict"].tolist(), strict=False))
        return {rid: {"tokens": [torch.tensor(rows[rid][:rows[rid][-1]])]} for rid in request_ids}

    def check_stop(self, request_id, request_info, outputs):
        if "tokens" not in outputs:
            return set()
        self.seen[request_id] = outputs["tokens"][0].tolist()
        return {"loop"} if 0 in self.seen[request_id] else set()


def _check(sub, request_ids, host_rows):
    failures = {}
    view = torch.zeros(3, 1)
    outputs = BatchedModelOutput(
        per_rid_outputs={rid: {"next": [view[i:i + 1]]} for i, rid in enumerate(("a", "b", "c"))},
        row_views={"next": view.split(1)},
    )
    batch = SimpleNamespace(
        node_name="n", request_ids=request_ids, outputs=outputs,
        per_request_info_wrapped=dict.fromkeys(request_ids),
        register_failure=failures.setdefault,
        step_context=SimpleNamespace(graph_walk="decode", request_walks=None),
    )
    engine = SimpleNamespace(_submodules={"n": SimpleNamespace(submodule=sub)})
    stops = Engine.check_stop_for_batch(engine, batch, outputs.per_rid_outputs, host_rows=host_rows)
    return stops, outputs, failures


# rows a, b, c and a padding row; "c" stopped a step ago, so the batch no longer runs it
ROWS = HostRows(("a", "b", "c"), {"verdict": torch.tensor(
    [[5, 6, 7, 2], [0, 9, 9, 1], [4, 4, 4, 3], [0, 0, 0, 0]])})


def test_emitted_outputs_are_routed_and_checked():
    sub = _Cutter()
    stops, outputs, failures = _check(sub, ["a", "b"], ROWS)
    assert sub.calls == [(["a", "b"], "decode")] and not failures
    per_rid = outputs.per_rid_outputs
    assert {rid: per_rid[rid]["tokens"][0].tolist() for rid in ("a", "b")} == {"a": [5, 6], "b": [0]}
    assert "next" in per_rid["a"] and "tokens" not in per_rid["c"]
    # the merged entries are not views of one batch clone any more
    assert outputs.row_views is None
    assert sub.seen == {"a": [5, 6], "b": [0]} and stops == {"b": {"loop"}}


def test_a_raise_fails_every_request_of_the_batch():
    sub = _Cutter(fail=True)
    stops, outputs, failures = _check(sub, ["a", "b"], ROWS)
    assert stops == {} and set(failures) == {"a", "b"}
    assert all("tokens" not in out for out in outputs.per_rid_outputs.values())


def test_not_called_without_host_rows():
    sub = _Cutter()
    _check(sub, ["a"], None)
    assert sub.calls == []
