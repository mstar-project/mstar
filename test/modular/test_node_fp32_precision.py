"""A node's float32 precision (``NodeSubmodule.float32_matmul_precision`` /
``cudnn_allow_tf32``) is applied by the engine around that node's forward, its
compile and its capture, and nowhere else.

The process default is TF32 (``apply_torch_config``). Before this setting a
node that needed full fp32 had to flip the process-wide flags inside its own
forward; these tests drive a real ``Engine`` on CPU and read the flags from
inside the forward, so they fail on a build whose engine ignores the setting.
"""
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, ".")

import mstar.engine.engine as engine_module  # noqa: E402
from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources import StepContext  # noqa: E402
from mstar.model.submodule_base import NodeInputs, NodeSubmodule  # noqa: E402

WALK = "run"


def _flags():
    return torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32


class _Recorder(NodeSubmodule):
    """Records the float32 flags its forward runs under; ``fail`` raises
    from inside the forward."""

    disable_autocast = True

    def __init__(self, matmul=None, cudnn=None):
        super().__init__()
        if matmul is not None:
            self.float32_matmul_precision = matmul
        if cudnn is not None:
            self.cudnn_allow_tf32 = cudnn
        self.weight = torch.nn.Parameter(torch.eye(4))
        self.seen: list[tuple[str, bool]] = []
        self.fail = False

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        return NodeInputs(tensor_inputs={"x": inputs["x"][0]}, input_seq_len=1)

    def forward(self, graph_walk, engine_inputs, x, **kwargs):
        self.seen.append(_flags())
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


def _engine(submodules: dict[str, NodeSubmodule], graph_runtime=None) -> Engine:
    engine = Engine(graph_runtime, autocast_dtype=None)
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
def _engine_default():
    """Every test starts from, and must leave, the engine's process default."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    yield
    try:
        assert _flags() == ("high", True), "a test left the float32 flags changed"
    finally:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True


def test_a_node_s_forward_runs_under_its_declared_precision():
    """The opt-out the models used to do themselves: full fp32 for one node,
    with nothing flipped in model code."""
    node = _Recorder(matmul="highest", cudnn=False)
    engine = _engine({"flow": node})
    out = _step(engine, "flow")
    assert node.seen == [("highest", False)]
    assert _flags() == ("high", True)
    assert torch.equal(out.get(0)["y"][0], torch.ones(2, 4))


def test_an_undeclared_node_keeps_the_process_default():
    node = _Recorder()
    _step(_engine({"llm": node}), "llm")
    assert node.seen == [("high", True)]


def test_two_nodes_on_one_worker_do_not_leak_into_each_other():
    """A at "highest", B at the default, interleaved, with A raising once from
    inside its forward: B never sees A's setting, and the default is back
    after every step."""
    a, b = _Recorder(matmul="highest"), _Recorder()
    engine = _engine({"a": a, "b": b})
    after = []
    for node in ("a", "b", "a", "b"):
        _step(engine, node)
        after.append(_flags())
    a.fail = True
    with pytest.raises(RuntimeError, match="forward failed"):
        _step(engine, "a")
    after.append(_flags())
    _step(engine, "b")
    after.append(_flags())
    assert a.seen == [("highest", True)] * 3
    assert b.seen == [("high", True)] * 3
    assert after == [("high", True)] * 6


def test_settings_apply_independently():
    """cuDNN alone leaves matmuls at the default, and the other way round."""
    conv_only, matmul_only = _Recorder(cudnn=False), _Recorder(matmul="medium")
    engine = _engine({"conv": conv_only, "mm": matmul_only})
    _step(engine, "conv")
    _step(engine, "mm")
    assert conv_only.seen == [("high", False)]
    assert matmul_only.seen == [("medium", True)]


def test_the_compiled_forward_is_traced_under_the_node_precision(monkeypatch):
    """Dynamo can't trace a flag change inside a compiled region, so the
    engine sets it around the compiled callable: the first call traces under
    the node's setting, and the other node's steps in between cause no
    recompile."""
    traced: dict[str, list[tuple[str, bool]]] = {}

    def backend_for(node):
        def backend(gm, example_inputs):
            traced.setdefault(node, []).append(_flags())
            return gm.forward
        return backend

    real_compile = torch.compile
    a, b = _Recorder(matmul="highest"), _Recorder()
    owner = {id(a): "a", id(b): "b"}

    def compile_with_recording_backend(fn, **kwargs):
        kwargs.pop("mode", None)
        return real_compile(fn, backend=backend_for(owner[id(fn.__self__)]), **kwargs)

    torch._dynamo.reset()
    monkeypatch.setattr(torch, "compile", compile_with_recording_backend)
    monkeypatch.setattr(
        engine_module, "with_inductor_overrides", lambda fn, *a, **k: fn, raising=False,
    )
    engine = _engine({"a": a, "b": b})
    with monkeypatch.context() as on_gpu:
        # the engine compiles only on an available accelerator; tracing itself stays on CPU
        on_gpu.setattr(engine, "_device_capability", SimpleNamespace(is_available=True))
        engine._compile_submodules()
    try:
        for node in ("a", "b", "a", "b"):
            _step(engine, node)
    finally:
        torch._dynamo.reset()
    assert traced == {"a": [("highest", True)], "b": [("high", True)]}


def test_capture_runs_under_the_node_precision(monkeypatch):
    """Capture fixes the kernels every replay runs, so warmup and capture
    (full-forward and piecewise) happen under the node's setting."""
    captured: list[tuple[str, tuple[str, bool]]] = []

    def record(self, buckets=None):
        captured.append((self._submodule_name, _flags()))

    class _Region:
        any_graphs = False
        dropped_shapes = ()

        def __init__(self, node):
            self.node = node

        def prepare_for_capture(self):
            pass

        def warmup_and_capture(self, buckets=None):
            captured.append((f"{self.node}/region", _flags()))

    monkeypatch.setattr(engine_module.AcceleratorGraphRunner, "warmup_and_capture", record)
    monkeypatch.setattr(
        Engine, "_build_piecewise_runners",
        lambda self, node, mgmt, capture_stream=None: {"blocks": _Region(node)},
    )
    engine = _engine({"a": _Recorder(matmul="highest", cudnn=False), "b": _Recorder()})
    engine.warmup()
    assert captured == [
        ("a", ("highest", False)), ("a/region", ("highest", False)),
        ("b", ("high", True)), ("b/region", ("high", True)),
    ]


def test_capture_sizing_runs_under_the_node_precision(monkeypatch):
    """Sizing a bucket runs its forward and a trial capture, so it compiles and
    picks kernels under the node's setting, as the capture itself does."""
    sized: list[tuple[str, tuple[str, bool]]] = []

    class _Sizer:
        def __init__(self, label):
            self.label = label

        def size_captures(self):
            sized.append((self.label, _flags()))
            return {}

        def size_lent_captures(self, wanted):
            sized.append((self.label, _flags()))
            return {}

    engine = _engine({"a": _Recorder(matmul="highest", cudnn=False), "b": _Recorder()})
    monkeypatch.setattr(engine, "_device", torch.device("cuda"))
    monkeypatch.setattr(engine, "_gpu_memory_fraction", None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (1 << 30, 1 << 31))
    engine._plan_captures(
        {node: _Sizer(node) for node in ("a", "b")},
        {node: {"blocks": _Sizer(f"{node}/region")} for node in ("a", "b")},
    )
    assert sized == [
        ("a", ("highest", False)), ("a/region", ("highest", False)),
        ("b", ("high", True)), ("b/region", ("high", True)),
    ] * 2


@pytest.mark.parametrize("bad", [
    {"matmul": "ultra"}, {"matmul": "HIGHEST"}, {"cudnn": 0},
])
def test_an_invalid_setting_fails_when_the_engine_loads(bad):
    with pytest.raises(ValueError, match="flow:"):
        _engine({"flow": _Recorder(**bad)})


def test_the_scope_restores_the_exact_state_it_found():
    """``matmul.allow_tf32 = False`` alone changes only the CUDA backend, which
    leaves torch's legacy getter raising. The scope still runs, and puts that
    state back as it was rather than normalizing it."""
    from mstar.engine.fp32_precision import Fp32Precision, fp32_precision_scope

    backends = torch.backends

    def state():
        return (
            backends.cuda.matmul.fp32_precision, backends.mkldnn.matmul.fp32_precision,
            backends.cudnn.conv.fp32_precision, backends.cudnn.rnn.fp32_precision,
        )

    backends.cuda.matmul.allow_tf32 = False
    try:
        before = state()
        with pytest.raises(RuntimeError):
            torch.get_float32_matmul_precision()
        with pytest.raises(KeyError):
            with fp32_precision_scope(Fp32Precision("high", False)):
                assert _flags() == ("high", False)
                raise KeyError("inside")
        assert state() == before
        assert backends.cuda.matmul.allow_tf32 is False
        with pytest.raises(RuntimeError):
            torch.get_float32_matmul_precision()
    finally:
        torch.set_float32_matmul_precision("high")


def test_cudnn_off_holds_when_the_cudnn_default_is_tf32():
    """``cudnn.allow_tf32 = False`` writes "none" to the conv and RNN keys,
    which inherit a "tf32" set on ``cudnn.fp32_precision``: the scope names
    "ieee" for them, and puts the inherited state back after."""
    from mstar.engine.fp32_precision import Fp32Precision, fp32_precision_scope

    cudnn = torch.backends.cudnn
    default = cudnn.fp32_precision
    cudnn.fp32_precision = "tf32"
    try:
        before = (cudnn.conv.fp32_precision, cudnn.rnn.fp32_precision)
        with fp32_precision_scope(Fp32Precision(None, False)):
            assert (cudnn.conv.fp32_precision, cudnn.rnn.fp32_precision) == ("ieee", "ieee")
            assert cudnn.allow_tf32 is False
        assert (cudnn.conv.fp32_precision, cudnn.rnn.fp32_precision) == before
        assert cudnn.fp32_precision == "tf32"
    finally:
        cudnn.fp32_precision = default
        cudnn.allow_tf32 = True


def test_a_thread_started_during_a_forward_keeps_the_node_setting():
    """The worker starts its executors lazily (the plan thread on the first
    speculative step), so one can start while a node's forward runs. Its
    initializer applies only the per-thread dynamo config."""
    from concurrent.futures import ThreadPoolExecutor

    from mstar.worker.worker import Worker

    worker = Worker.__new__(Worker)  # skips __init__, which opens sockets
    worker.device = torch.device("cpu")

    class _StartsAThread(_Recorder):
        def forward(self, graph_walk, engine_inputs, x, **kwargs):
            with ThreadPoolExecutor(1, initializer=worker._init_engine_thread) as ex:
                ex.submit(lambda: None).result()
            return super().forward(graph_walk, engine_inputs, x, **kwargs)

    node = _StartsAThread(matmul="highest")
    _step(_engine({"flow": node}), "flow")
    assert node.seen == [("highest", True)]


def test_no_setting_is_a_shared_no_op():
    from mstar.engine.fp32_precision import fp32_precision_scope, node_fp32_precision

    assert node_fp32_precision("llm", _Recorder()) is None
    assert fp32_precision_scope(None) is fp32_precision_scope(None)


def _runtime_kwargs(communicator=None, bookkeeping=None):
    """``_make_graph_runtime``'s arguments for a one-node worker, as in
    test_graph_runtime_factory.py."""
    from mstar.communication.tensor_store import PythonTensorBookkeeping, TensorStore
    from mstar.distributed.base import ShardingConfig
    from mstar.graph.base import GraphNode
    from mstar.model.base import WorkerGraph

    section = GraphNode(name="a", input_names={"x"}, outputs=[])
    store = TensorStore(bookkeeping=bookkeeping or PythonTensorBookkeeping())
    return dict(
        my_worker_id="worker0",
        my_worker_graphs=[WorkerGraph(
            section=section, graph_walks={WALK}, ranks=[0], worker_graph_id=0,
        )],
        all_wg_ids_to_graph_walks={0: {WALK}},
        all_wg_ids_to_dyn_loops={0: set()},
        all_wg_ids_to_nodes={0: {"a"}},
        node_to_partition={"a": "default"},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        tensor_manager=type("_TM", (), {"tensor_store": store})(),
        communicator=communicator,
    )


@pytest.mark.parametrize("runtime", ["python", "rust"])
def test_the_setting_holds_under_either_graph_runtime(runtime, monkeypatch):
    """The setting lives on the submodule and is applied by the engine, which
    the Python and Rust graph runtimes share; neither runtime sees it."""
    from mstar.worker.worker import _make_graph_runtime

    _kwargs = _runtime_kwargs
    monkeypatch.setenv("MSTAR_RUST_GRAPH", "1" if runtime == "rust" else "0")
    monkeypatch.delenv("MSTAR_RUST_ZMQ", raising=False)
    if runtime == "rust":
        pytest.importorskip("mstar_rust")
        from mstar.communication.codec import WireCodec
        from mstar.communication.rust_communicator import RustZMQCommunicator
        from mstar.communication.tensor_store import RustTensorBookkeeping
        from mstar.graph.runtime.rust import RustGraphRuntime

        comm = RustZMQCommunicator.__new__(RustZMQCommunicator)
        comm.codec = WireCodec
        graph_runtime = _make_graph_runtime(**_kwargs(
            communicator=comm, bookkeeping=RustTensorBookkeeping(),
        ))
        assert isinstance(graph_runtime, RustGraphRuntime)
    else:
        from mstar.graph.runtime.python import PythonGraphRuntime

        graph_runtime = _make_graph_runtime(**_kwargs())
        assert isinstance(graph_runtime, PythonGraphRuntime)

    a, b = _Recorder(matmul="highest", cudnn=False), _Recorder()
    engine = _engine({"a": a, "b": b}, graph_runtime=graph_runtime)
    for node in ("a", "b"):
        _step(engine, node)
    assert a.seen == [("highest", False)] and b.seen == [("high", True)]
    assert _flags() == ("high", True)
