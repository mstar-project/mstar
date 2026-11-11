"""A state op is a kernel that writes state itself, so nothing but its declaration says what it touches.

Its signature is checked at import, because a kernel that names the wrong parameter would otherwise fail on the first
request that reaches it. The format match decides between the fused kernel and the reference once, at bind, and a wrong
match either writes pages no read looks at or runs a kernel the storage can't take.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine import engine as engine_module
from mstar.engine.resources.kv.config import KVLayout
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Block,
    Causal,
    Entry,
    KVEntry,
    PagedNHD,
    PagedNHDTarget,
    Read,
    TensorType,
    Value,
    state_op,
)
from mstar.engine.state.adapters import PAGED, Resolved, StateSetup, _bindings, _resolve, build_handle
from mstar.engine.state.calls import ArrayHandle
from mstar.model.submodule_base import NodeSubmodule
from mstar.testing_utils.model import make_model
from mstar.utils.device import DeviceCapability

HEADS, DIM, PAGES, PAGE = 2, 4, 3, 8
state_op_module = sys.modules["mstar.engine.state.state_op"]


@pytest.fixture(autouse=True)
def _empty_registry(monkeypatch):
    monkeypatch.setattr(state_op_module, "STATE_OPS", {})


def _reference(kv, layer, q, k, v):
    q.add_(1)
    kv.add(layer, k=k + 1, v=v)


def _fused_op(**overrides):
    kwargs = {"adds": ("kv",), "mutates": ("q",), "formats": (PagedNHD(devices={"cpu"}),), "reference": _reference}

    @state_op(**{**kwargs, **overrides})
    def write_q(kv, layer, q, k, v, *, target: PagedNHDTarget):
        q.add_(1)
        target.cache[target.page_of_token[:target.count], 0, target.slot_of_token[:target.count]] = k + 1
        target.cache[target.page_of_token[:target.count], 1, target.slot_of_token[:target.count]] = v

    return write_q


class _StubKV:
    """Real pages, with one step's plan, in the shape `KVManager` hands a kernel."""

    def __init__(self, layout=KVLayout.NHD, device="cpu", dtype=torch.bfloat16, tokens=3):
        pages = torch.zeros(2, PAGES, 2, PAGE, HEADS, DIM, dtype=dtype)
        self.kv_cache = SimpleNamespace(layout=layout, dtype=dtype, device=torch.device(device), tensor=pages)
        self.tokens = tokens
        self.writes = []

    def layer_view(self, i):
        return self.kv_cache.tensor[i]

    def write_target(self, layer_idx=None, label=None):
        rows = torch.arange(self.tokens)
        return self.kv_cache.tensor[layer_idx], torch.zeros_like(rows), rows, self.tokens

    def write_kv(self, k, v, layer_idx=None, label=None):
        self.writes.append(layer_idx)
        target = self.write_target(layer_idx)
        self.kv_cache.tensor[layer_idx][0, 0, target[2]] = k
        self.kv_cache.tensor[layer_idx][0, 1, target[2]] = v

    def opens_label(self, rid, label):
        return True


class _StubAttn:
    """Reads from the pages it is handed, or takes K/V at its read when told to."""

    def __init__(self, requires_kv_write=True):
        self.requires_kv_write = requires_kv_write

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        return q


def _kv_array(**fields) -> Array:
    return Array("kv", nodes={"llm"}, layers={0, 1}, entry=KVEntry(heads=HEADS, head_dim=DIM), max_len=16,
                 reads={"attn": Read(mask=Causal(), op=Attend(q_heads=HEADS))}, **fields)


def _handle(kv=None, attn=None, array=None) -> ArrayHandle:
    kv = kv or _StubKV()
    array = array or _kv_array()
    return build_handle(array, {array.name: kv, "attn": attn or _StubAttn()}, _resolve([array], StateSetup()))


def _open(handle) -> SubmoduleStep:
    step = SubmoduleStep(segments=(Segment(1, "main", 3),), state={handle.name: ArrayStep()})
    step.set_ctx(StepContext(request_ids=[1], graph_walk="prefill", slot=0, capture=False))
    step.set_state_handles((handle,))
    step.begin_calls()
    return step


def _tensors():
    return tuple(torch.full((3, HEADS, DIM), fill, dtype=torch.bfloat16) for fill in (0.0, 1.0, 2.0))


# The checks at import


@pytest.mark.parametrize("overrides, cause", [
    ({"adds": ("nope",)}, r"parameters \['nope'\] aren't in its signature"),
    ({"mutates": ("nope",)}, r"parameters \['nope'\] aren't in its signature"),
    ({"adds": ()}, "a target addresses one array"),
    ({"adds": ("kv",), "writes": ("kv",)}, "a target addresses one array"),
    ({"formats": ("NHD",)}, "formats must be PagedNHD"),
    ({"reference": lambda kv, layer, q, k: None}, "must take its signature without target"),
    ({"reference": lambda kv, layer, q, k, x: None}, "must take its signature without target"),
    ({"reads": ("q",)}, "no layout yet: an op reads only the array it adds to"),
])
def test_a_bad_declaration_is_refused_at_import_naming_the_op_and_cause(overrides, cause):
    with pytest.raises(ValueError, match=f"state op 'write_q': .*{cause}"):
        _fused_op(**overrides)


def test_a_kernel_with_no_keyword_only_target_is_refused_at_import():
    with pytest.raises(ValueError, match="state op 'positional': its kernel needs a keyword-only parameter target"):
        @state_op(adds=("kv",), formats=(), reference=lambda kv, layer, target: None)
        def positional(kv, layer, target):
            pass


def test_a_kernel_with_no_layer_is_refused_at_import():
    with pytest.raises(ValueError, match=r"state op 'no_layer': parameters \['layer'\]"):
        @state_op(adds=("kv",), formats=(), reference=lambda kv: None)
        def no_layer(kv, *, target):
            pass


def test_two_ops_cant_share_a_name():
    _fused_op()
    with pytest.raises(ValueError, match="state op 'write_q': the name is taken"):
        _fused_op()


def test_the_decorator_records_each_op_so_a_conformance_test_can_find_it():
    call = _fused_op()
    assert state_op_module.STATE_OPS == {"write_q": call.op}


# The match at bind


@pytest.mark.parametrize("formats, kv, attn, fused", [
    ((PagedNHD(),), _StubKV(), _StubAttn(), True),
    ((PagedNHD(devices={"cpu"}),), _StubKV(), _StubAttn(), True),
    ((PagedNHD(devices={"cuda"}),), _StubKV(), _StubAttn(), False),
    ((PagedNHD(dtypes={torch.bfloat16}),), _StubKV(), _StubAttn(), True),
    ((PagedNHD(dtypes={torch.float32}),), _StubKV(), _StubAttn(), False),
    ((PagedNHD(),), _StubKV(layout=KVLayout.MLA), _StubAttn(), False),
    ((PagedNHD(),), _StubKV(), _StubAttn(requires_kv_write=False), False),
    ((), _StubKV(), _StubAttn(), False),
])
def test_the_match_reads_layout_dtype_device_and_where_the_read_takes_its_keys(formats, kv, attn, fused):
    _fused_op(formats=formats)
    handle = _handle(kv=kv, attn=attn)
    state_op_module.bind_state_ops({"kv": handle})
    assert ("write_q" in handle.fused_ops) is fused


def test_the_engine_binds_a_nodes_state_ops_with_its_handles(monkeypatch):
    """Without this bind every fused op runs its reference, which gives the same output and no error."""
    order = []
    monkeypatch.setattr(engine_module, "bind_state_ops", lambda handles: order.append(("ops", set(handles))))
    monkeypatch.setattr(engine_module, "node_handles", lambda *a, **k: {"kv": object()})

    class _Node(NodeSubmodule):
        forward = prepare_inputs = None

        def bind_node_state(self, handles):
            order.append(("submodule", set(handles)))

    engine = engine_module.Engine(graph_runtime=None, device_capability=DeviceCapability(torch.device("cpu")))
    engine.load_model(
        {"llm": _Node()}, [], WorkerParallelGroups(num_workers=1, global_rank=0), torch.device("cpu"),
        transfer_engine_info=None,
        model=make_model(checkpoint_path=lambda: None),
    )
    assert order == [("ops", {"kv"}), ("submodule", {"kv"})]


def test_binding_skips_what_has_no_pages():
    _fused_op(formats=(PagedNHD(),))
    state_op_module.bind_state_ops({"sampler": SimpleNamespace(), "name": "not a handle"})


# The call


def test_an_unlisted_format_runs_the_reference_which_reaches_the_cache_through_add():
    call = _fused_op()
    h = _handle()
    _open(h)
    q, k, v = _tensors()
    call(h, 0, q, k, v)
    assert h._kv.writes == [0], "the reference adds"
    assert (q == 1).all()


def test_a_listed_format_runs_the_fused_kernel_and_counts_as_the_layers_add():
    call = _fused_op()
    h = _handle()
    state_op_module.bind_state_ops({"kv": h})
    _open(h)
    q, k, v = _tensors()
    call(h, 0, q, k, v)
    assert h._kv.writes == [], "the kernel wrote the pages itself"
    assert (h._kv.kv_cache.tensor[0][0, 0, :3] == 2).all() and (h._kv.kv_cache.tensor[0][0, 1, :3] == 2).all()
    with pytest.raises(ValueError, match="an add on layer 0 after its kept add"):
        call(h, 0, q, k, v)
    h.read(0, q)


def test_fused_and_reference_write_the_same_and_the_layer_goes_through_the_normalizer():
    call = _fused_op()
    fused, reference = _handle(), _handle()
    state_op_module.bind_state_ops({"kv": fused})
    for h in (fused, reference):
        _open(h)
        q, k, v = _tensors()
        call(h, torch.tensor(1), q, k, v)
    assert torch.equal(fused._kv.kv_cache.tensor, reference._kv.kv_cache.tensor)


def test_a_layer_tensor_on_the_gpu_is_refused_by_name():
    call = _fused_op()
    h = _handle()
    state_op_module.bind_state_ops({"kv": h})
    _open(h)
    with pytest.raises(ValueError, match="Array 'kv': a layer tensor must be 0-d int64"):
        call(h, torch.tensor([0]), *_tensors())


def test_a_call_in_a_step_that_doesnt_list_the_array_is_refused():
    with pytest.raises(ValueError, match="Array 'kv': state op 'write_q' in a step that doesn't list the array"):
        _fused_op()(_handle(), 0, *_tensors())


def test_a_state_op_that_adds_to_a_rewritable_array_is_refused():
    value = Value("kv", nodes={"llm"}, layers={0, 1}, entry=Entry({"x": TensorType((2,), dtype=None)}))
    # a Value with no update, which resolves to no op
    h = build_handle(value, {"kv": SimpleNamespace(block=lambda field, layer: None)}, Resolved({}, {"kv": ()}, {}, {}))
    _open(h)
    with pytest.raises(ValueError, match="Array 'kv': no layout yet: add on a ValueHandle"):
        _fused_op()(h, 0, *_tensors())


def test_a_state_op_that_writes_an_append_only_array_is_refused():
    def reference(kv, layer):
        kv.write(layer, k=torch.zeros(1))

    @state_op(writes=("kv",), formats=(), reference=reference)
    def rewrite(kv, layer, *, target):
        pass

    h = _handle()
    _open(h)
    with pytest.raises(ValueError, match="Array 'kv': write on an append-only array"):
        rewrite(h, 0)


def test_a_state_op_that_writes_never_takes_the_paged_fused_path():
    @state_op(writes=("kv",), formats=(PagedNHD(devices={"cpu"}),), reference=lambda kv, layer: None)
    def rewrite(kv, layer, *, target):
        pass

    assert not _handle().takes_fused(rewrite.op), "a write's kernel would overwrite an append-only array's pages"


def test_no_fused_op_runs_on_an_array_where_one_layers_read_takes_k_and_v_at_its_kernel():
    """A fused op writes the pages, and that read takes its k and v from the add, so the layer's read finds no add."""
    _fused_op()
    array = Array("kv", nodes={"llm"}, layers={0, 1}, entry=KVEntry(heads=HEADS, head_dim=DIM), max_len=16, reads={
        "attn": Read(mask=Causal(), op=Attend(q_heads=HEADS), layers={0}),
        "held": Read(mask=Causal(), op=Attend(q_heads=HEADS), layers={1}),
    })
    resources = {"kv": _StubKV(), "attn": _StubAttn(), "held": _StubAttn(requires_kv_write=False)}
    h = build_handle(array, resources, _resolve([array], StateSetup()))
    state_op_module.bind_state_ops({"kv": h})
    assert h.fused_ops == frozenset(), "layer 1's read would refuse a read with no add before it"


def test_a_state_op_on_a_seq_sharded_array_is_refused():
    array = _kv_array(seq_sharded=True)
    # resolve refuses a seq_sharded paged array, so its bound read and that read's resource are given here
    resolved = Resolved({}, {array.name: _bindings(PAGED, array)}, {"attn": "attn"}, {})
    h = build_handle(array, {array.name: _StubKV(), "attn": _StubAttn()}, resolved)
    _open(h)
    with pytest.raises(ValueError, match="no layout yet: state op 'write_q' on a seq_sharded array"):
        _fused_op()(h, 0, *_tensors())


def test_a_compiled_caller_breaks_at_a_state_op(monkeypatch):
    seen = []

    @state_op(adds=("kv",), formats=(), reference=lambda kv, layer: seen.append(torch.compiler.is_compiling()))
    def traced(kv, layer, *, target):
        pass

    h = _handle()
    _open(h)

    def caller(x):
        traced(h, 0)
        return x + 1

    torch._dynamo.reset()
    assert torch.compile(caller)(torch.zeros(1)).item() == 1
    assert seen == [False], "a state call runs behind a break, as `add` does"


# An op that ends with its layer's read


def _reference_attend(kv, layer, q, k, v):
    _reference(kv, layer, q, k, v)
    return kv.read(layer, q)


def _attending_op(kernel_reads=True):
    @state_op(adds=("kv",), reads=("kv",), mutates=("q",), formats=(PagedNHD(devices={"cpu"}),),
              reference=_reference_attend)
    def attend_q(kv, layer, q, k, v, *, target: PagedNHDTarget):
        q.add_(1)
        target.cache[target.page_of_token[:target.count], 0, target.slot_of_token[:target.count]] = k + 1
        target.cache[target.page_of_token[:target.count], 1, target.slot_of_token[:target.count]] = v
        return kv.read(layer, q) if kernel_reads else None

    return attend_q


def test_an_op_that_reads_an_array_with_max_len_0_is_refused():
    array = Array("kv", nodes={"llm"}, layers={0, 1}, entry=KVEntry(heads=HEADS, head_dim=DIM), max_len=0,
                  reads={"attn": Read(mask=Block(), op=Attend(q_heads=HEADS))})
    h = _handle(array=array)
    _open(h)
    with pytest.raises(ValueError, match="state op 'attend_q' reads kv with max_len 0, whose read a compiled block"):
        _attending_op()(h, 0, *_tensors())


# The debug check


def _bad_op():
    @state_op(adds=("kv",), formats=(PagedNHD(),), reference=lambda kv, layer: None)
    def scribble(kv, layer, *, target):
        target.cache[PAGES - 1, 0, PAGE - 1] = 9

    return scribble


def test_the_debug_check_refuses_a_write_outside_the_steps_slots(monkeypatch):
    monkeypatch.setattr(state_op_module, "_DEBUG_ASSERTS", True)
    call = _bad_op()
    h = _handle()
    state_op_module.bind_state_ops({"kv": h})
    _open(h)
    with pytest.raises(ValueError, match="Array 'kv': state op 'scribble' changed .* outside this step's slots"):
        call(h, 0)


def test_the_debug_check_stays_out_of_a_capture(monkeypatch):
    """Its host sync can't run in a capture, which records launches rather than running them."""
    monkeypatch.setattr(state_op_module, "_DEBUG_ASSERTS", True)
    call = _bad_op()
    h = _handle()
    state_op_module.bind_state_ops({"kv": h})
    step = SubmoduleStep(segments=(Segment(1, "main", 3),), state={h.name: ArrayStep()})
    step.set_ctx(StepContext(request_ids=[1], graph_walk="prefill", slot=0, capture=True))
    step.set_state_handles((h,))
    step.begin_calls()
    call(h, 0)


def test_the_debug_check_passes_a_write_inside_them(monkeypatch):
    monkeypatch.setattr(state_op_module, "_DEBUG_ASSERTS", True)
    call = _fused_op()
    h = _handle()
    state_op_module.bind_state_ops({"kv": h})
    _open(h)
    call(h, 0, *_tensors())


def test_the_debug_check_is_off_by_default():
    assert not state_op_module._DEBUG_ASSERTS, "it copies the whole array around every call"
    call = _bad_op()
    h = _handle()
    state_op_module.bind_state_ops({"kv": h})
    _open(h)
    call(h, 0)


# The conformance harness


def _conformance(call, h):
    state_op_module.bind_state_ops({"kv": h})
    _open(h)
    q, k, v = _tensors()
    return state_op_module.conformance_problems(call, (h, 0, q, k, v), tolerance=1e-6)


def test_conformance_passes_a_kernel_that_matches_its_reference():
    assert _conformance(_fused_op(), _handle()) == []


def test_conformance_finds_a_kernel_that_mutates_differently():
    @state_op(adds=("kv",), mutates=("q",), formats=(PagedNHD(),), reference=_reference)
    def wrong_q(kv, layer, q, k, v, *, target):
        q.add_(2)
        target.cache[target.page_of_token[:target.count], 0, target.slot_of_token[:target.count]] = k + 1
        target.cache[target.page_of_token[:target.count], 1, target.slot_of_token[:target.count]] = v

    (problem,) = _conformance(wrong_q, _handle())
    assert problem.startswith("argument q")


def test_conformance_finds_a_kernel_that_writes_other_entries():
    @state_op(adds=("kv",), mutates=("q",), formats=(PagedNHD(),), reference=_reference)
    def wrong_k(kv, layer, q, k, v, *, target):
        q.add_(1)
        target.cache[target.page_of_token[:target.count], 0, target.slot_of_token[:target.count]] = k
        target.cache[target.page_of_token[:target.count], 1, target.slot_of_token[:target.count]] = v

    (problem,) = _conformance(wrong_k, _handle())
    assert problem.startswith("the step's slots")


def test_conformance_finds_a_kernel_that_writes_outside_the_steps_slots():
    @state_op(adds=("kv",), mutates=("q",), formats=(PagedNHD(),), reference=_reference)
    def spills(kv, layer, q, k, v, *, target):
        q.add_(1)
        target.cache[target.page_of_token[:target.count], 0, target.slot_of_token[:target.count]] = k + 1
        target.cache[target.page_of_token[:target.count], 1, target.slot_of_token[:target.count]] = v
        target.cache[PAGES - 1, 0, PAGE - 1] = 9

    (problem,) = _conformance(spills, _handle())
    assert problem == "the array outside the step's slots changed"


def test_conformance_finds_a_kernel_whose_return_differs():
    @state_op(adds=("kv",), reads=("kv",), mutates=("q",), formats=(PagedNHD(),), reference=_reference_attend)
    def wrong_out(kv, layer, q, k, v, *, target):
        q.add_(1)
        target.cache[target.page_of_token[:target.count], 0, target.slot_of_token[:target.count]] = k + 1
        target.cache[target.page_of_token[:target.count], 1, target.slot_of_token[:target.count]] = v
        return kv.read(layer, q) + 100

    (problem,) = _conformance(wrong_out, _handle())
    assert problem.startswith("the return")


def test_conformance_finds_a_kernel_that_skips_the_read_it_declares():
    problems = _conformance(_attending_op(kernel_reads=False), _handle())
    assert "the kernel doesn't read the layer, and the op declares reads" in problems
    assert "the return: the kernel returns None" in problems


def test_conformance_reads_the_layer_by_its_id_not_its_place_in_the_pool():
    array = Array("kv", nodes={"llm"}, layers={3, 7}, entry=KVEntry(heads=HEADS, head_dim=DIM), max_len=16,
                  reads={"attn": Read(mask=Causal(), op=Attend(q_heads=HEADS))})
    h = _handle(array=array)
    state_op_module.bind_state_ops({"kv": h})
    _open(h)
    q, k, v = _tensors()
    assert state_op_module.conformance_problems(_attending_op(), (h, 7, q, k, v), tolerance=1e-6) == []


def test_conformance_finds_an_op_that_reads_without_declaring_reads():
    @state_op(adds=("kv",), mutates=("q",), formats=(PagedNHD(),), reference=_reference_attend)
    def reads_quietly(kv, layer, q, k, v, *, target):
        q.add_(1)
        target.cache[target.page_of_token[:target.count], 0, target.slot_of_token[:target.count]] = k + 1
        target.cache[target.page_of_token[:target.count], 1, target.slot_of_token[:target.count]] = v
        return kv.read(layer, q)

    assert _conformance(reads_quietly, _handle()) == [
        "the reference reads the layer, and the op doesn't declare reads",
        "the kernel reads the layer, and the op doesn't declare reads",
    ]
