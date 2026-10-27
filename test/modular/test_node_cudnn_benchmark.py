"""A node's conv algorithm search (``NodeSubmodule.cudnn_benchmark``) is applied
by the engine around that node's forward, capture sizing, warm-up and capture,
and nowhere else.

These drive a real ``Engine`` on CPU and read ``torch.backends.cudnn.benchmark``
from inside the forward, as test_node_fp32_precision.py does for the float32
settings, so they fail on a build whose engine ignores the setting.
"""
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, ".")

import mstar.engine.engine as engine_module  # noqa: E402
from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.engine.cudnn_benchmark import cudnn_benchmark_scope, node_cudnn_benchmark  # noqa: E402
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources import StepContext  # noqa: E402
from mstar.model.submodule_base import NodeInputs, NodeSubmodule  # noqa: E402

WALK = "run"


def _searching() -> bool:
    return torch.backends.cudnn.benchmark


class _Recorder(NodeSubmodule):
    """Records the flag its forward runs under; ``fail`` raises from inside
    the forward."""

    disable_autocast = True

    def __init__(self, benchmark=None):
        super().__init__()
        if benchmark is not None:
            self.cudnn_benchmark = benchmark
        self.weight = torch.nn.Parameter(torch.eye(4))
        self.seen: list[bool] = []
        self.fail = False

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        return NodeInputs(tensor_inputs={"x": inputs["x"][0]}, input_seq_len=1)

    def forward(self, graph_walk, engine_inputs, x, **kwargs):
        self.seen.append(_searching())
        if self.fail:
            raise RuntimeError("forward failed")
        return {"y": [x @ self.weight]}


class _Rank:
    world_size = 1

    def barrier(self):
        pass


class _Groups:
    """A single-rank worker: no node is sharded."""

    def get_joint_group_for_node(self, node):
        return type("_Joint", (), {"world_size": 1, "tp_group": _Rank(), "sp_group": _Rank()})()

    def get_instance_world_size_for_node(self, node):
        return 1

    def get_instance_rank_for_node(self, node):
        return 0


def _engine(submodules: dict[str, NodeSubmodule]) -> Engine:
    engine = Engine(None, autocast_dtype=None)
    engine.load_model(
        submodules=submodules, specs=[], parallel_groups=_Groups(),
        device=torch.device("cpu"), transfer_engine_info=None,
    )
    return engine


def _step(engine: Engine, node: str, rid: int = 0):
    """What the worker's GPU thread runs for one step: prepare, then exec."""
    batch = ExecutingBatch(
        node_name=node,
        per_request_info={rid: CurrentForwardPassInfo(
            request_id=f"r{rid}", rid_handle=rid, graph_walk=WALK,
            fwd_index=0, random_seed=0, max_tokens=1,
        )},
        step_context=StepContext(request_ids=(rid,), graph_walk=WALK, slot=0, capture=False),
        per_request_input_tensors={rid: {"x": [torch.ones(2, 4)]}},
    )
    engine.prepare_inputs(batch)
    return engine.exec_and_postprocess(batch)


@pytest.fixture(autouse=True)
def _process_default():
    """Every test starts from, and must leave, torch's default (no search)."""
    torch.backends.cudnn.benchmark = False
    yield
    try:
        assert not _searching(), "a test left cudnn.benchmark on"
    finally:
        torch.backends.cudnn.benchmark = False


def test_two_nodes_on_one_worker_do_not_leak_into_each_other():
    """A vocoder that searches next to an encoder that doesn't, interleaved,
    with the vocoder raising once from inside its forward: the encoder never
    searches, and the default is back after every step."""
    vocoder, encoder = _Recorder(benchmark=True), _Recorder()
    engine = _engine({"vocoder": vocoder, "encoder": encoder})
    after = []
    for node in ("vocoder", "encoder", "vocoder", "encoder"):
        _step(engine, node)
        after.append(_searching())
    vocoder.fail = True
    with pytest.raises(RuntimeError, match="forward failed"):
        _step(engine, "vocoder")
    after.append(_searching())
    _step(engine, "encoder")
    after.append(_searching())
    assert vocoder.seen == [True] * 3
    assert encoder.seen == [False] * 3
    assert after == [False] * 6


def test_a_node_can_turn_the_search_off_under_a_process_that_has_it_on():
    node = _Recorder(benchmark=False)
    engine = _engine({"encoder": node})
    torch.backends.cudnn.benchmark = True
    try:
        _step(engine, "encoder")
        assert _searching()
    finally:
        torch.backends.cudnn.benchmark = False
    assert node.seen == [False]


def test_warm_up_and_capture_run_under_the_node_setting(monkeypatch):
    """The warm-up's search is what the capture records, so warm-up and capture
    (full-forward and piecewise) happen under the node's setting; the capture
    helper itself turns the search off inside the capture."""
    captured: list[tuple[str, bool]] = []

    def record(self, buckets=None):
        captured.append((self._submodule_name, _searching()))

    class _Region:
        any_graphs = False
        dropped_shapes = ()

        def __init__(self, node):
            self.node = node

        def prepare_for_capture(self):
            pass

        def warmup_and_capture(self, buckets=None):
            captured.append((f"{self.node}/region", _searching()))

    monkeypatch.setattr(engine_module.AcceleratorGraphRunner, "warmup_and_capture", record)
    monkeypatch.setattr(
        Engine, "_build_piecewise_runners",
        lambda self, node, mgmt: {"blocks": _Region(node)},
    )
    engine = _engine({"vocoder": _Recorder(benchmark=True), "encoder": _Recorder()})
    engine.warmup()
    assert captured == [
        ("vocoder", True), ("vocoder/region", True),
        ("encoder", False), ("encoder/region", False),
    ]


def test_capture_sizing_runs_under_the_node_setting(monkeypatch):
    """Sizing runs the forwards first, and a conv's first call caches its pick
    for that input size, so sizing searches too or the warm-up never would."""
    sized: list[tuple[str, bool]] = []

    class _Sizer:
        def __init__(self, label):
            self.label = label

        def size_captures(self):
            sized.append((self.label, _searching()))
            return {}

        def size_lent_captures(self, wanted):
            sized.append((self.label, _searching()))
            return {}

    engine = _engine({"vocoder": _Recorder(benchmark=True), "encoder": _Recorder()})
    monkeypatch.setattr(engine, "_device", torch.device("cuda"))
    monkeypatch.setattr(engine, "_gpu_memory_fraction", None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (1 << 30, 1 << 31))
    engine._plan_captures(
        {node: _Sizer(node) for node in ("vocoder", "encoder")},
        {node: {"blocks": _Sizer(f"{node}/region")} for node in ("vocoder", "encoder")},
    )
    assert sized == [
        ("vocoder", True), ("vocoder/region", True),
        ("encoder", False), ("encoder/region", False),
    ] * 2


@pytest.mark.parametrize("bad", [1, "true"])
def test_an_invalid_setting_fails_when_the_engine_loads(bad):
    node = _Recorder()
    node.cudnn_benchmark = bad
    with pytest.raises(ValueError, match="vocoder:"):
        _engine({"vocoder": node})


def test_no_setting_is_a_shared_no_op():
    assert node_cudnn_benchmark("encoder", _Recorder()) is None
    assert cudnn_benchmark_scope(None) is cudnn_benchmark_scope(None)
