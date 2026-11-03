"""Fused decode (``mstar/engine/fused_decode.py``): which buckets a submodule declares and the guard lets
through, the runner capturing the fused forward in a declared bucket's place and the graph everywhere
else, the fallback to the graph when the fused capture fails or its parity step disagrees, and the
kernels' counters coming back zeroed between replays.

The CPU tests drive the real runner and engine methods on the smallest objects they read. The GPU
tests capture a toy submodule whose fused forward is a small persistent Triton kernel that syncs
its CTAs the way the block kernels do (fp32 atomics into an accumulator, a counter the last CTA
resets), on a real ``AcceleratorGraphRunner``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.accelerator_graph_backend import AcceleratorGraphPool
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfigType,
    BatchedAcceleratorGraphConfig,
    FusedDecodeConfig,
)
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner, CaptureCost
from mstar.engine.engine import Engine
from mstar.engine.fused_decode import PARITY_REL_L2, FusedScratch, fused_parity, parity_failure
from mstar.engine.resources import BucketKey, CGSlotSpec
from mstar.engine.resources.lora.config import LoraBaseOnly
from mstar.kernels import megakernel as mk
from mstar.model.submodule_base import BatchedModelOutput, NodeInputs, NodeSubmodule
from mstar.utils import knobs

_CUDA = torch.device("cuda")
_CPU = torch.device("cpu")


# -- the declaration and the guard ------------------------------------------------------------


class _Plain(NodeSubmodule):
    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def test_a_submodule_declares_no_fused_decode_by_default():
    base = _Plain.__new__(_Plain)
    assert base.get_fused_decode_configs(_CUDA) == []
    assert "no fused forward" in base.prepare_fused_decode({})


def test_qwen35_declares_decode_at_one_and_two_rows_where_it_captures_them():
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    llm = LLMSubmodule.__new__(LLMSubmodule)
    assert llm.get_fused_decode_configs(_CUDA) == [FusedDecodeConfig(walk="decode", batch_sizes=(1, 2))]
    llm.DECODE_CAPTURE_BATCH_SIZES = [2, 4, 8]
    assert llm.get_fused_decode_configs(_CUDA) == [FusedDecodeConfig(walk="decode", batch_sizes=(2,))]
    llm.DECODE_CAPTURE_BATCH_SIZES = [4, 8]
    assert llm.get_fused_decode_configs(_CUDA) == []


def _platform(monkeypatch, capability):
    monkeypatch.setattr(mk, "current_platform", SimpleNamespace(cuda_capability=lambda device=None: capability))


# Qwen3.5-0.8B's sizes, the ones the kernels were built for
_SIZES = dict(
    hidden_size=1024, gdn_head_dims=(128, 128), attn_head_dim=256, kv_heads=2, page_size=128,
    state_dtype=torch.float32, conv_taps=4, gated_attention=True, adapters=False,
)


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_the_guard_passes_blackwell_at_the_sizes_the_kernels_take(monkeypatch, capability):
    _platform(monkeypatch, capability)
    assert mk.block_refusal(_CUDA, **_SIZES) is None
    assert mk.block_refusal(_CUDA, **{**_SIZES, "hidden_size": 2048}) is None


@pytest.mark.parametrize("change, says", [
    ({"page_size": 64}, "page size 64"),
    ({"gdn_head_dims": (128, 64)}, "GDN head dims"),
    ({"attn_head_dim": 128}, "attention head dim 128"),
    ({"kv_heads": 4}, "4 KV heads"),
    ({"state_dtype": torch.bfloat16}, "bfloat16"),
    ({"conv_taps": 3}, "3-tap conv"),
    ({"gated_attention": False}, "output gate"),
    ({"adapters": True}, "LoRA"),
    ({"hidden_size": 1536}, "hidden size 1536"),
    ({"hidden_size": 4096}, "hidden size 4096"),
])
def test_the_guard_names_what_it_refuses(monkeypatch, change, says):
    _platform(monkeypatch, (10, 3))
    assert says in mk.block_refusal(_CUDA, **{**_SIZES, **change})


@pytest.mark.parametrize("capability", [(9, 0), (12, 0), None])
def test_the_guard_refuses_other_gpus(monkeypatch, capability):
    _platform(monkeypatch, capability)
    assert "compute capability" in mk.block_refusal(_CUDA, **_SIZES)


def test_the_guard_refuses_the_cpu():
    assert "not a CUDA device" in mk.block_refusal(_CPU, **_SIZES)


def _stack(types):
    gdn_mixer = SimpleNamespace(head_k_dim=128, head_v_dim=128, conv1d=SimpleNamespace(weight=torch.zeros(8, 1, 4)))
    attn = SimpleNamespace(head_dim=256, num_kv_heads=2, output_gate=True)
    layers = [SimpleNamespace(self_attn=gdn_mixer if t == "linear_attention" else attn) for t in types]
    return SimpleNamespace(config=SimpleNamespace(layer_types=types, hidden_size=1024), layers=layers)


def test_the_stack_guard_takes_groups_of_three_delta_net_layers_then_attention(monkeypatch):
    from mstar.model.qwen3_5.fused_decode import stack_refusal

    _platform(monkeypatch, (10, 3))
    block = ["linear_attention"] * 3 + ["full_attention"]
    assert stack_refusal(_stack(block * 2), _CUDA, 128, torch.float32, adapters=False) is None
    for types in (block[:3], ["linear_attention", "full_attention"] * 4, ["full_attention"] + block[1:]):
        assert "don't group" in stack_refusal(_stack(types), _CUDA, 128, torch.float32, adapters=False)


def test_qwen35_refuses_where_decode_must_reproduce(monkeypatch):
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    monkeypatch.setenv("MSTAR_COMPILE_DETERMINISTIC", "1")
    llm = LLMSubmodule.__new__(LLMSubmodule)
    assert "compile_deterministic" in llm.prepare_fused_decode({})


def test_qwen35_refuses_a_pool_with_no_sink(monkeypatch):
    from mstar.model.qwen3_5.config import GDN_STATE, KV_CACHE
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    monkeypatch.delenv("MSTAR_COMPILE_DETERMINISTIC", raising=False)
    llm = LLMSubmodule.__new__(LLMSubmodule)
    pool = SimpleNamespace(config=SimpleNamespace(disable_sink_slot=True))
    assert "sink" in llm.prepare_fused_decode({KV_CACHE: object(), GDN_STATE: pool})


@pytest.mark.parametrize("value, expected", [(None, "auto"), ("auto", "auto"), ("off", "off"), (False, "off")])
def test_the_knob_is_auto_or_off(monkeypatch, value, expected):
    monkeypatch.delenv("MSTAR_FUSED_DECODE", raising=False)
    knobs.configure({} if value is None else {"fused_decode": value})
    try:
        assert knobs.fused_decode() == expected
        monkeypatch.setenv("MSTAR_FUSED_DECODE", "off")
        assert knobs.fused_decode() == "off"
    finally:
        knobs.configure({})
    with pytest.raises(ValueError, match="auto or off"):
        knobs.check({"fused_decode": "on"})


# -- the engine binding the declaration -------------------------------------------------------


class _Declares:
    def __init__(self, refusal=None):
        self.refusal = refusal
        self.prepared = 0

    def get_fused_decode_configs(self, device, tp_world_size=1):
        return [FusedDecodeConfig(walk="decode", batch_sizes=(1, 2))]

    def prepare_fused_decode(self, resources):
        self.prepared += 1
        return self.refusal


def _bind(submodule, world_size=1):
    engine = Engine.__new__(Engine)
    engine._device = _CUDA
    engine._submodules = {"LLM": SimpleNamespace(
        submodule=submodule, resources={}, joint_comm_group=SimpleNamespace(world_size=world_size),
    )}
    runner = SimpleNamespace(bound=[])
    runner.bind_fused_decode = runner.bound.append
    engine._bind_fused_decode("LLM", runner)
    return runner.bound


def test_the_engine_binds_what_the_guard_lets_through(monkeypatch):
    monkeypatch.delenv("MSTAR_FUSED_DECODE", raising=False)
    submodule = _Declares()
    assert _bind(submodule) == [submodule.get_fused_decode_configs(_CUDA)]
    assert submodule.prepared == 1


def test_a_refused_node_keeps_its_graphs_with_one_warning(monkeypatch, caplog):
    monkeypatch.delenv("MSTAR_FUSED_DECODE", raising=False)
    with caplog.at_level(logging.WARNING):
        assert _bind(_Declares(refusal="compute capability (9, 0)")) == []
    assert [r.getMessage() for r in caplog.records] == [
        "fused decode stays off on LLM: compute capability (9, 0)"
    ]


def test_a_lockstep_instance_keeps_its_graphs_and_prepares_nothing(monkeypatch, caplog):
    monkeypatch.delenv("MSTAR_FUSED_DECODE", raising=False)
    submodule = _Declares()
    with caplog.at_level(logging.WARNING):
        assert _bind(submodule, world_size=2) == []
    assert submodule.prepared == 0
    assert "lockstep instance of 2 ranks" in caplog.text


def test_the_knob_off_prepares_nothing(monkeypatch):
    monkeypatch.setenv("MSTAR_FUSED_DECODE", "off")
    submodule = _Declares()
    assert _bind(submodule) == []
    assert submodule.prepared == 0


# -- the runner choosing the fused capture ----------------------------------------------------


def _spec(bs, slot=0, walk="decode", kind=AcceleratorGraphConfigType.BASIC_BATCHED, key=None):
    config = SimpleNamespace(get_config_type=lambda: kind, additional_key_info=None)
    return CGSlotSpec(bucket=BucketKey(walk, bs, bs, key), slot=slot, config=config, config_idx=0)


def test_only_declared_batched_buckets_without_adapters_take_the_fused_forward():
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner._fused = {}
    config = FusedDecodeConfig(walk="decode", batch_sizes=(1, 2))
    assert runner._fused_config(_spec(1)) is None
    runner.bind_fused_decode([config])
    assert runner._fused_config(_spec(1)) is config
    assert runner._fused_config(_spec(2, slot=1)) is config
    assert runner._fused_config(_spec(4)) is None
    assert runner._fused_config(_spec(1, walk="prefill_text")) is None
    assert runner._fused_config(_spec(1, kind=AcceleratorGraphConfigType.FLASH_INFER_PACKED)) is None
    assert runner._fused_config(_spec(1, key=LoraBaseOnly(None))) is None


class _FlowRunner:
    """`warmup_and_capture` with every bucket planned; the graph and fused captures stubbed."""

    NUM_WARMUP = AcceleratorGraphRunner.NUM_WARMUP
    warmup_and_capture = AcceleratorGraphRunner.warmup_and_capture
    _capture_one = AcceleratorGraphRunner._capture_one
    fused_buckets = AcceleratorGraphRunner.fused_buckets
    _register_slot = AcceleratorGraphRunner._register_slot
    _buckets_captured_everywhere = AcceleratorGraphRunner._buckets_captured_everywhere
    _report_dropped = AcceleratorGraphRunner._report_dropped
    _fused_config = AcceleratorGraphRunner._fused_config
    bind_fused_decode = AcceleratorGraphRunner.bind_fused_decode

    def __init__(self, specs, fused_fails=()):
        self._device = _CPU
        self._graph_backend = SimpleNamespace(
            is_available=lambda: True, graph_pool_handle=lambda: AcceleratorGraphPool(object()),
            memory_allocated=lambda: 0,
        )
        self._submodule_name = "LLM"
        self._num_slots = 2
        self._specs = specs
        self._capture_costs = {spec.bucket: CaptureCost(0, 0, 0, 2) for spec in specs}
        self._buckets, self._executables, self._blocked_buckets = {}, {}, []
        self._memory_pool = None
        self._fused = {}
        self._fused_fails = set(fused_fails)
        group = CommGroup.trivial()
        self._comm_group = JointGroups(tp_group=group, sp_group=group)
        self._dummy_rows = SimpleNamespace(release_all=lambda: None)
        self.fused_calls = []

    def prepare_for_capture(self):
        return self._specs

    def _capture_graph(self, spec, forwards):
        return SimpleNamespace(name=f"graph:{spec.bucket.bs}:{spec.slot}", fused=False)

    def _capture_fused(self, spec, fused, graph_slot):
        self.fused_calls.append((spec.bucket.bs, spec.slot, graph_slot.name))
        if (spec.bucket.bs, spec.slot) in self._fused_fails:
            return graph_slot
        return SimpleNamespace(name=f"fused:{spec.bucket.bs}:{spec.slot}", fused=True)

    def _get_addtl_slot_specs(self, spec):
        return []

    def declare_inputs_for(self, lease):
        return []

    def _log_memory(self, before, after):
        pass


def test_declared_buckets_register_the_fused_capture_and_the_rest_their_graph():
    runner = _FlowRunner([_spec(bs, slot) for bs in (1, 2, 4) for slot in (0, 1)])
    runner.bind_fused_decode([FusedDecodeConfig(walk="decode", batch_sizes=(1, 2))])
    runner.warmup_and_capture(None)
    slots = {key.bs: [slot.name for slot in bucket.slots] for key, bucket in runner._buckets.items()}
    assert slots == {1: ["fused:1:0", "fused:1:1"], 2: ["fused:2:0", "fused:2:1"], 4: ["graph:4:0", "graph:4:1"]}
    assert {key.bs for key in runner.fused_buckets} == {1, 2}
    # the fused capture comes after the bucket's own graph, which it falls back to
    assert [call[2] for call in runner.fused_calls] == ["graph:1:0", "graph:1:1", "graph:2:0", "graph:2:1"]


def test_a_fused_capture_that_falls_back_leaves_its_graph_registered():
    runner = _FlowRunner([_spec(1, slot) for slot in (0, 1)], fused_fails={(1, 1)})
    runner.bind_fused_decode([FusedDecodeConfig(walk="decode", batch_sizes=(1,))])
    runner.warmup_and_capture(None)
    (bucket,) = runner._buckets.values()
    assert [slot.name for slot in bucket.slots] == ["fused:1:0", "graph:1:1"]
    # a bucket some of whose slots kept the graph isn't counted as fused
    assert runner.fused_buckets == set()


# -- the parity step --------------------------------------------------------------------------


class _Pool:
    """A resource whose rows' state is one tensor, and a submodule whose two paths write it."""

    def __init__(self):
        self.state = torch.arange(6, dtype=torch.float32).view(3, 2)

    def row_state(self, rids):
        return {"state": self.state[list(rids)].clone()}

    def load_row_state(self, rids, state):
        self.state[list(rids)] = state["state"]


class _Paths:
    def __init__(self, pool, hidden_skew=1.0, state_skew=1.0):
        self.pool, self.hidden_skew, self.state_skew = pool, hidden_skew, state_skew

    def decode_hidden(self, graph_walk, engine_inputs, fused, x):
        rows = [0, 2]
        scale = self.state_skew if fused else 1.0
        self.pool.state[rows] = self.pool.state[rows] * 0.5 + scale
        return (x * (self.hidden_skew if fused else 1.0)) + self.pool.state[rows].sum()


@pytest.mark.parametrize("hidden_skew, state_skew, fails", [
    (1.0, 1.0, None), (1.001, 1.001, None), (1.1, 1.0, "hidden"), (1.0, 1.5, "pool.state"),
])
def test_the_parity_step_compares_the_hidden_state_and_every_state_write(hidden_skew, state_skew, fails):
    pool = _Pool()
    before = pool.state.clone()
    errors = fused_parity(
        _Paths(pool, hidden_skew, state_skew), "decode", None, {"x": torch.full((2, 4), 100.0)}, {"pool": pool},
        [0, 2],
    )
    assert set(errors) == {"hidden", "pool.state"}
    failure = parity_failure(errors)
    assert (failure is None) == (fails is None)
    if fails:
        assert failure.startswith(fails) and errors[fails] >= PARITY_REL_L2
    # the rows hold what they held before the step
    assert torch.equal(pool.state, before)


def test_scratch_is_one_set_per_name_and_never_first_allocated_under_capture(monkeypatch):
    scratch = FusedScratch(_CPU)
    cnt = scratch.zeros("cnt", (8,), torch.int32)
    assert scratch.zeros("cnt", (8,), torch.int32) is cnt and not cnt.any()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    on_gpu = FusedScratch(_CUDA)
    with pytest.raises(RuntimeError, match="under capture"):
        on_gpu.zeros("acc", (4,), torch.float32)


# -- on a GPU: a real runner and a toy fused kernel ----------------------------------------------


def _needs_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("needs an sm_90+ GPU (the toy kernel's acquire-release counter)")


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _toy_gemv(x_ptr, w_ptr, y_ptr, acc_ptr, cnt_ptr, launches_ptr, B: tl.constexpr, K: tl.constexpr,
                  N: tl.constexpr, KB: tl.constexpr):
        """y = x @ w, K split across the grid into an fp32 accumulator; the last CTA in writes y and
        leaves the accumulator and the counter zero, as the block kernels leave theirs."""
        pid = tl.program_id(0)
        G = tl.num_programs(0)
        ks = pid * KB + tl.arange(0, KB)
        ns = tl.arange(0, N)
        w = tl.load(w_ptr + ks[:, None] * N + ns[None, :])
        for b in tl.static_range(B):
            x = tl.load(x_ptr + b * K + ks)
            tl.atomic_add(acc_ptr + b * N + ns, tl.sum(x[:, None] * w, axis=0), sem="relaxed")
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr, 1, sem="acq_rel", scope="gpu")
        if done == G - 1:
            for b in tl.static_range(B):
                acc = tl.load(acc_ptr + b * N + ns, cache_modifier=".cg")
                tl.store(y_ptr + b * N + ns, acc)
                tl.store(acc_ptr + b * N + ns, tl.zeros([N], dtype=tl.float32))
            tl.atomic_xchg(cnt_ptr, 0, sem="release", scope="gpu")
            tl.atomic_add(launches_ptr, 1)
except ImportError:  # pragma: no cover - CPU-only installs
    _toy_gemv = None


K = N = 64


class _Toy:
    """A one-config decode submodule with no resources: y = x @ w, the fused path on the toy kernel."""

    def __init__(self, fail=False, skew=1.0):
        gen = torch.Generator().manual_seed(0)
        self.w = torch.randn(K, N, generator=gen).to(_CUDA)
        self.template = torch.randn(1, K, generator=gen).to(_CUDA)
        self.fail, self.skew = fail, skew

    def get_accelerator_graph_configs(self, device, tp_world_size=1):
        return [BatchedAcceleratorGraphConfig(
            capture_graph_walk="decode", capture_batch_sizes=[1, 2, 4], compile=False,
            single_request_inputs=NodeInputs(tensor_inputs={"x": self.template}, input_seq_len=1),
        )]

    def declare_step(self, *args, **kwargs):
        return None

    def preprocess(self, graph_walk, engine_inputs, inputs):
        return {"x": torch.cat([inp.tensor_inputs["x"] for inp in inputs])}

    def forward_batched(self, graph_walk, engine_inputs, x):
        return BatchedModelOutput(row_outputs={"y": x @ self.w})

    def decode_hidden(self, graph_walk, engine_inputs, fused, x):
        if not fused:
            return x @ self.w
        if self.fail:
            raise RuntimeError("no kernel for this geometry")
        scratch = engine_inputs.fused_scratch
        y = torch.empty(x.shape[0], N, device=x.device)
        _toy_gemv[(4,)](x, self.w, y, scratch.zeros("acc", (x.shape[0], N), torch.float32),
                        scratch.zeros("cnt", (1,), torch.int32), scratch.zeros("launches", (1,), torch.int32),
                        B=x.shape[0], K=K, N=N, KB=K // 4)
        return y * self.skew

    def forward_fused(self, graph_walk, engine_inputs, x):
        return BatchedModelOutput(row_outputs={"y": self.decode_hidden(graph_walk, engine_inputs, True, x=x)})


def _runner(toy):
    device = torch.device("cuda", torch.cuda.current_device())
    group = CommGroup.trivial()
    step_runner = SimpleNamespace(ingest_capture_row=lambda rid: None,
                                  build_accelerator_graph_buffers=lambda *args, **kwargs: None)
    runner = AcceleratorGraphRunner(
        submodule_name="toy", submodule=toy, resources={}, step_runner=step_runner, device=device,
        autocast_dtype=None, joint_comm_group=JointGroups(tp_group=group, sp_group=group), num_slots=2,
    )
    runner.bind_fused_decode([FusedDecodeConfig(walk="decode", batch_sizes=(1, 2))])
    runner.warmup_and_capture(None)
    return runner


def _exact(x, w):
    return (x.double() @ w.double()).float()


# the graph's matmul may run in TF32; the toy kernel sums in fp32
_GRAPH_TOL = dict(rtol=2e-2, atol=2e-2)
_FUSED_TOL = dict(rtol=1e-4, atol=1e-4)


def _replay(runner, bs, slot, x):
    lease = runner.lease_slot("decode", bs, slot=slot, num_tokens=bs)
    out = runner.run_forward(lease, {"x": x})
    torch.cuda.synchronize()
    return out.row_outputs["y"][:bs].clone()


def test_a_declared_bucket_replays_its_fused_capture_and_its_counters_come_back_zero():
    _needs_gpu()
    toy = _Toy()
    runner = _runner(toy)
    assert {key.bs for key in runner.fused_buckets} == {1, 2}
    x = torch.randn(4, K, device=_CUDA)
    for bs in (1, 2):
        for slot in (0, 1):
            scratch = runner._fused_scratch[(BucketKey("decode", bs, bs), slot)].buffers()
            launches = int(scratch["launches"].item())
            for _ in range(3):
                torch.testing.assert_close(_replay(runner, bs, slot, x[:bs]), _exact(x[:bs], toy.w), **_FUSED_TOL)
                assert not scratch["cnt"].any() and not scratch["acc"].any()
            assert int(scratch["launches"].item()) == launches + 3, "the replay ran the fused kernel"
    # a bucket nothing declared replays its graph, with no scratch of its own
    torch.testing.assert_close(_replay(runner, 4, 0, x), _exact(x, toy.w), **_GRAPH_TOL)
    assert not any(key[0].bs == 4 for key in runner._fused_scratch)


def test_a_fused_forward_that_fails_at_capture_keeps_the_graph(caplog):
    _needs_gpu()
    with caplog.at_level(logging.WARNING):
        runner = _runner(_Toy(fail=True))
    assert runner.fused_buckets == set()
    assert caplog.text.count("failed to capture; it keeps its graph") == 4
    x = torch.randn(2, K, device=_CUDA)
    torch.testing.assert_close(_replay(runner, 2, 1, x), _exact(x, runner._submodule.w), **_GRAPH_TOL)


def test_a_fused_forward_that_disagrees_with_the_modules_keeps_the_graph(caplog):
    _needs_gpu()
    with caplog.at_level(logging.WARNING):
        runner = _runner(_Toy(skew=1.05))
    assert runner.fused_buckets == set()
    assert caplog.text.count("disagrees with the modules (hidden rel L2 0.0") == 4
    x = torch.randn(1, K, device=_CUDA)
    torch.testing.assert_close(_replay(runner, 1, 0, x), _exact(x, runner._submodule.w), **_GRAPH_TOL)
