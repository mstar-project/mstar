"""A step whose submodule refuses to replay its prepared inputs (`NodeSubmodule.can_replay`)
leases no captured slot, so it runs eager; one it accepts leases as before."""
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.model.submodule_base import NodeSubmodule  # noqa: E402
from mstar.testing_utils.engine import make_engine, make_mgmt  # noqa: E402


class _Sub(NodeSubmodule):
    def __init__(self, replays: bool):
        super().__init__()
        self.replays = replays

    def prepare_inputs(self, *a, **kw):
        return None

    def forward(self, *a, **kw):
        return None

    def can_replay(self, graph_walk, inputs):
        self.asked = (graph_walk, [inp.input_seq_len for inp in inputs])
        return self.replays


class _Runner:
    def __init__(self):
        self.leases = []

    def lease_slot(self, **kwargs):
        self.leases.append(kwargs)


@pytest.mark.parametrize("replays", [False, True])
def test_a_step_its_submodule_wont_replay_takes_no_slot(replays):
    sub, runner = _Sub(replays), _Runner()
    mgmt = make_mgmt(submodule=sub, accelerator_graph_runner=runner)
    mgmt.lease_slot = lambda: 0
    eng = make_engine(_submodules={"llm": mgmt})
    batch = SimpleNamespace(
        node_name="llm", slot=0, request_ids=["a", "b"], per_request_info_wrapped={},
        per_request_input_metadata={}, cg_key_info=None,
        inputs=[SimpleNamespace(input_seq_len=100), SimpleNamespace(input_seq_len=60)],
        step_context=StepContext(request_ids=("a", "b"), graph_walk="prefill", slot=0, capture=False),
    )
    assert eng.reserve_replay_slot(batch) is None
    assert sub.asked == ("prefill", [100, 60])
    assert [lease["num_tokens"] for lease in runner.leases] == ([160] if replays else [])
