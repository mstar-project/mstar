"""On a GPU: a node's float32 precision holds through the engine's real
capture and replay, compiled or not, and doesn't leak into the next node.

Three nodes share one engine and run the same fp32 matmul chain: "full" at
``float32_matmul_precision="highest"``, "tf32" at the engine default, and
"compiled" at "highest" with its capture compiled (max-autotune). Each
captures one batch-4 bucket, then they replay interleaved. Skips without CUDA.
"""
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.distributed.communication import CommGroup, JointGroups  # noqa: E402
from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig  # noqa: E402
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources import StepContext  # noqa: E402
from mstar.model.submodule_base import NodeInputs, NodeSubmodule  # noqa: E402

WALK = "run"
BS, ROWS, DIM = 4, 64, 1024
DEVICE = torch.device("cuda", 0)


class _MatmulChain(NodeSubmodule):
    """``x @ w1 @ w2`` in fp32: TF32 rounding shows at about 1e-3 relative."""

    disable_autocast = True

    def __init__(self, precision=None, compile_capture=False):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.w1 = torch.nn.Parameter(torch.randn(DIM, DIM, generator=generator) / DIM**0.5)
        self.w2 = torch.nn.Parameter(torch.randn(DIM, DIM, generator=generator) / DIM**0.5)
        if precision is not None:
            self.float32_matmul_precision = precision
        self._compile_capture = compile_capture

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        return NodeInputs(tensor_inputs={"x": inputs["x"][0]}, input_seq_len=1)

    def can_batch(self, batch, model_inputs):
        return True

    def preprocess(self, graph_walk, engine_inputs, inputs):
        return {"x": torch.stack([i.tensor_inputs["x"] for i in inputs]).to(DEVICE)}

    def chain(self, x):
        return x @ self.w1 @ self.w2

    def forward(self, graph_walk, engine_inputs, x, **kwargs):
        return {"y": [self.chain(x)[0]]}

    def forward_batched(self, graph_walk, engine_inputs, x, **kwargs):
        y = self.chain(x)
        return {rid: {"y": [y[i]]} for i, rid in enumerate(engine_inputs.request_ids)}

    def get_accelerator_graph_configs(self, device, tp_world_size=1):
        return [BatchedAcceleratorGraphConfig(
            capture_graph_walk=WALK,
            single_request_inputs=NodeInputs(
                tensor_inputs={"x": torch.zeros(ROWS, DIM, device=device)}, input_seq_len=1,
            ),
            compile=self._compile_capture,
            capture_batch_sizes=[BS],
        )]


class _Groups:
    def get_joint_group_for_node(self, node):
        return JointGroups(CommGroup.trivial(), CommGroup.trivial(), CommGroup.trivial())

    def get_instance_world_size_for_node(self, node):
        return 1

    def get_instance_rank_for_node(self, node):
        return 0


@pytest.fixture(scope="module")
def setup():
    torch.set_float32_matmul_precision("high")
    nodes = {
        "full": _MatmulChain("highest"),
        "tf32": _MatmulChain(),
        "compiled": _MatmulChain("highest", compile_capture=True),
    }
    for node in nodes.values():
        node.to(DEVICE)
    engine = Engine(None, autocast_dtype=None)
    engine.load_model(
        submodules=nodes, specs=[], parallel_groups=_Groups(),
        device=DEVICE, transfer_engine_info=None,
    )
    engine.warmup()
    for name in nodes:
        assert engine._submodules[name].accelerator_graph_runner is not None, f"{name} did not capture"
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(BS, ROWS, DIM, generator=generator)
    yield engine, nodes, x
    torch.set_float32_matmul_precision("high")


def _replay(engine, node, x):
    rids = list(range(BS))
    batch = ExecutingBatch(
        node_name=node,
        per_request_info={rid: CurrentForwardPassInfo(
            request_id=f"r{rid}", rid_handle=rid, graph_walk=WALK,
            fwd_index=0, random_seed=0, max_tokens=1,
        ) for rid in rids},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False),
        per_request_input_tensors={rid: {"x": [x[rid]]} for rid in rids},
    )
    engine.prepare_inputs(batch)
    out = engine.exec_and_postprocess(batch)
    assert batch.step_context.slot_lease is not None, f"{node} ran eager, not a replay"
    torch.cuda.synchronize()
    return torch.stack([out.get(rid)["y"][0] for rid in rids]).cpu()


def _eager(node, x, precision):
    torch.set_float32_matmul_precision(precision)
    try:
        with torch.no_grad():
            return node.chain(x.to(DEVICE)).cpu()
    finally:
        torch.set_float32_matmul_precision("high")


def _rel_err(y, x, node):
    exact = (x.double() @ node.w1.detach().cpu().double() @ node.w2.detach().cpu().double())
    return ((y.double() - exact).abs().max() / exact.abs().max()).item()


def test_tf32_is_live_on_this_gpu(setup):
    """Precondition: without it the other checks prove nothing."""
    _, nodes, x = setup
    node = nodes["full"]
    assert not torch.equal(_eager(node, x, "highest"), _eager(node, x, "high"))
    assert _rel_err(_eager(node, x, "high"), x, node) > 10 * _rel_err(_eager(node, x, "highest"), x, node)


def test_a_captured_highest_node_matches_its_eager_fp32_forward(setup):
    engine, nodes, x = setup
    assert torch.equal(_replay(engine, "full", x), _eager(nodes["full"], x, "highest"))


def test_a_captured_default_node_keeps_tf32(setup):
    engine, nodes, x = setup
    assert torch.equal(_replay(engine, "tf32", x), _eager(nodes["tf32"], x, "high"))


def test_a_compiled_capture_of_a_highest_node_is_full_fp32(setup):
    """Max-autotune may pick a Triton GEMM template over cuBLAS, so the error is
    neither bit-equal to eager nor within a fixed factor of it. Any fp32 kernel
    stays under 1e-5 on these shapes (cuBLAS 4.6e-7, Triton 1.5e-6 on an H200);
    TF32 lands near 3e-4."""
    engine, nodes, x = setup
    node = nodes["compiled"]
    err = _rel_err(_replay(engine, "compiled", x), x, node)
    assert err < 1e-5
    assert 10 * err < _rel_err(_eager(node, x, "high"), x, node)


def test_interleaved_replays_do_not_leak(setup):
    engine, nodes, x = setup
    expected = {
        "full": _eager(nodes["full"], x, "highest"),
        "tf32": _eager(nodes["tf32"], x, "high"),
    }
    compiled_first = _replay(engine, "compiled", x)
    for node in ("full", "tf32", "compiled", "tf32", "full", "compiled"):
        y = _replay(engine, node, x)
        if node == "compiled":
            assert torch.equal(y, compiled_first)
        else:
            assert torch.equal(y, expected[node]), node
        assert torch.get_float32_matmul_precision() == "high"
