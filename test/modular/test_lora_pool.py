"""The per-request LoRA adapter pool on CPU: lifecycle, addressing, the host store,
numerics, build, and admission for a slot.

A two-layer stack with a fused column-parallel ``qkv`` target (GQA-sized k/v) and a
row-parallel ``o`` target. Other LoRA test modules import the helpers defined here.
"""

from __future__ import annotations

import gc
import threading
import time
import weakref

import pytest
import torch

pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

from mstar.engine.resources.base import EngineResourceInfo, Resource  # noqa: E402
from mstar.engine.resources.lora import LoraModuleRef, LoraReqConfig, LoraSpec, LoraStep, LoraTarget  # noqa: E402
from mstar.engine.resources.lora.pool import ZERO_SLOT, LoraAdapterPool  # noqa: E402
from mstar.engine.resources.lora.store import LoraAdapterStore  # noqa: E402
from mstar.engine.resources.runner import StepRunner  # noqa: E402
from mstar.engine.resources.step import (  # noqa: E402
    ASKS,
    AdmitRuntimeError,
    AllocationFailed,
    BucketKey,
    Segment,
    SlotLease,
    StepContext,
)
from mstar.model.components.lora.modules import ModuleLoraMap  # noqa: E402
from mstar.model.loader.base import CheckpointRules, StackedParamRule  # noqa: E402

HIDDEN, Q_OUT, KV_OUT, LAYERS = 16, 16, 8, 2
QKV = LoraTarget(
    in_features=HIDDEN, out_slices=(Q_OUT, KV_OUT, KV_OUT), parallel="column",
    slice_names=("q", "k", "v"), slice_units=(4, 2, 2),
)
O = LoraTarget(in_features=Q_OUT, out_slices=(HIDDEN,), parallel="row")
MODULE_MAP = ModuleLoraMap(
    {
        f"model.layers.{layer}.self_attn.{proj}_proj": LoraModuleRef(target, layer)
        for layer in range(LAYERS) for proj, target in (("qkv", "qkv"), ("o", "o"))
    },
    rules=CheckpointRules(stacked_params=tuple(
        StackedParamRule(".qkv_proj", f".{s}_proj", s) for s in ("q", "k", "v")
    )),
)
OUT_WIDTH = {"q_proj": Q_OUT, "k_proj": KV_OUT, "v_proj": KV_OUT, "o_proj": HIDDEN}
IN_WIDTH = {"q_proj": HIDDEN, "k_proj": HIDDEN, "v_proj": HIDDEN, "o_proj": Q_OUT}
CPU = torch.device("cpu")


def write_adapter(path, rank, projs=("q_proj", "k_proj", "v_proj", "o_proj"), alpha=None, seed=0,
                  extra=()):
    """A PEFT-layout adapter (``extra`` projections under ``mlp``); returns
    {(layer, proj): (a, b, alpha)}."""
    g = torch.Generator().manual_seed(seed)
    sd, ref = {}, {}
    for layer in range(LAYERS):
        for proj in (*projs, *extra):
            module = f"base_model.model.model.layers.{layer}.{'mlp' if proj in extra else 'self_attn'}.{proj}"
            a = torch.randn(rank, IN_WIDTH.get(proj, HIDDEN), generator=g)
            b = torch.randn(OUT_WIDTH.get(proj, HIDDEN), rank, generator=g)
            sd[f"{module}.lora_A.weight"], sd[f"{module}.lora_B.weight"] = a, b
            if alpha is not None:
                sd[f"{module}.alpha"] = torch.tensor(float(alpha))
            ref[(layer, proj)] = (a, b, float(rank if alpha is None else alpha))
    save_file(sd, str(path))
    return ref


@pytest.fixture
def adapters(tmp_path):
    """a1 on q/k/v/o, a2 on q and v (k stays zero), a3 on o alone."""
    refs, paths = {}, {}
    for name, rank, projs, alpha, seed in [
        ("a1", 2, ("q_proj", "k_proj", "v_proj", "o_proj"), 4.0, 1),
        ("a2", 4, ("q_proj", "v_proj"), None, 2),
        ("a3", 3, ("o_proj",), 1.5, 3),
    ]:
        paths[name] = str(tmp_path / f"{name}.safetensors")
        refs[name] = write_adapter(paths[name], rank, projs, alpha, seed)
    return refs, paths


def make_spec(paths, max_slots=2, max_rank=4, cpu_capacity=8, dtype=torch.float64, **kw) -> LoraSpec:
    return LoraSpec(
        resource_key="lora", nodes={"LLM"}, num_layers=LAYERS,
        targets={"qkv": QKV, "o": O}, module_map=MODULE_MAP,
        max_slots=max_slots, max_rank=max_rank, cpu_capacity=cpu_capacity,
        adapters=dict(paths), dtype=dtype, backend="torch", **kw,
    )


def make_pool(paths, **kw) -> LoraAdapterPool:
    return LoraAdapterPool(make_spec(paths, **kw), "cpu")


def ctx(rids, capture=False, lease=None, padded=None) -> StepContext:
    c = StepContext(request_ids=list(rids), graph_walk="decode", slot=0, capture=capture, slot_lease=lease)
    if padded is not None:
        c.set_padded_rids(padded)
    return c


def step(rids, spans=None) -> LoraStep:
    spans = spans or [1] * len(rids)
    return LoraStep(segments=tuple(Segment(r, "main", s) for r, s in zip(rids, spans, strict=True)))


def run_step(pool, rids, spans=None, commit=True, **ctx_kw):
    """Admit and plan a step over ``rids``; and commit it, unless a delta is to run in it."""
    s, c = step(rids, spans), ctx(rids, **ctx_kw)
    outcome = pool.admit(s, c)
    if outcome.ok:
        pool.plan(s, c)
        if commit:
            pool.commit(s, c)
    return outcome


def ingest(pool, rid, adapter):
    pool.ingest_request(rid, LoraReqConfig(adapter=adapter))


def token_slots_of(addr) -> list[int]:
    """Each token's slot, read back from the slot-grouped layout the kernels take; a
    token past every slot's range (a capture bucket's tail) reads as the zero slot."""
    order, indptr = addr.token_order.tolist(), addr.slot_indptr.tolist()
    slots = [ZERO_SLOT] * len(order)
    for slot in range(len(indptr) - 1):
        for position in range(indptr[slot], indptr[slot + 1]):
            slots[order[position]] = slot
    return slots


def lora_runner(*pools: Resource, keys=("lora",)) -> StepRunner:
    return StepRunner(dict(zip(keys, pools, strict=True)), {"LLM": list(keys), "vision": []})


def admitted(runner, rid, node="LLM") -> bool:
    outcome = runner.admit_retrieve(rid, node, "decode")
    assert outcome.ok
    return outcome.ready


def _nothing_published(rid, node):
    return None


def settle_reads(pool: LoraAdapterPool, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while pool._reads:
        assert time.monotonic() < deadline, f"reads never completed: {list(pool._reads)}"
        time.sleep(0.001)
    with pool._lock:  # a completion stores its copy under the lock
        pass


# --- lifecycle and addressing ------------------------------------------------------------------

def test_admit_refcount_lru_evict_and_the_zero_slot(adapters):
    _, paths = adapters
    pool = make_pool(paths, max_slots=2)
    for a, b in (pool.weights("qkv", 0), pool.weights("o", 1)):
        assert not a[ZERO_SLOT].any() and not b[ZERO_SLOT].any()
    assert pool.scalings[ZERO_SLOT] == 0
    for rid, name in [("base", None), ("r1", "a1"), ("r2", "a2"), ("r3", "a3"), ("r4", "a2"), ("r5", "a1"),
                      ("r1", "a1")]:
        ingest(pool, rid, name)  # a repeated ingest is a no-op
    with pytest.raises(ValueError):
        ingest(pool, "bad", "nope")
    assert run_step(pool, ["base", "r1", "r2"]).ok
    s1, s2 = pool.slot_of("r1"), pool.slot_of("r2")
    assert pool.slot_of("base") == ZERO_SLOT and {s1, s2} == {1, 2} and pool.slot_state(s1).refcount == 1
    # a second request on a resident adapter shares its slot
    assert run_step(pool, ["r1", "r2", "r4"]).ok
    assert pool.slot_of("r4") == s2 and pool.slot_state(s2).refcount == 2
    # full (slot 0 is never handed out): a3 has nowhere to go, and the refusal names it
    outcome = run_step(pool, ["r3"])
    assert isinstance(outcome.reason, AllocationFailed) and outcome.reason.request_id == "r3"
    pool.rollback_admit(step(["r3"]), ctx(["r3"]))
    assert pool.slot_of("r3") is None
    pool.remove_request("r2")
    pool.remove_request("r2")
    assert pool.slot_state(s2).refcount == 1  # r4 still holds a2
    # release a1 first, then a2: a3 evicts the slot released longest ago (a1's)
    pool.remove_request("r1")
    assert run_step(pool, ["r4"]).ok
    pool.remove_request("r4")
    assert run_step(pool, ["r3"]).ok
    assert pool.slot_of("r3") == s1 and pool.resident_slot("a1") is None
    assert pool.resident_slot("a2") == s2  # unheld, but still resident
    # the evicted slot takes the new adapter's per-layer ranks: a3 adapts o alone
    assert pool.layer_ranks("qkv", 0)[s1] == 0 and pool.layer_ranks("o", 1)[s1] == 3
    # a1 comes back into the slot a2 kept warm, the only unheld one
    assert run_step(pool, ["r5"]).ok and pool.slot_of("r5") == s2 and pool.resident_slot("a2") is None


def test_an_adapter_with_nothing_for_the_pools_targets_takes_no_slot(adapters, tmp_path):
    """Known from its header before any admit: its first request neither waits for the
    held slot nor runs the deltas."""
    _, paths = adapters
    path = str(tmp_path / "mlp.safetensors")
    write_adapter(path, rank=2, projs=(), extra=("up_proj",))
    pool = make_pool({**paths, "mlp": path}, max_slots=1, ignored_modules=lambda m: ".mlp." in m)
    runner = lora_runner(pool)
    for rid, name in [("r1", "a1"), ("e1", "mlp")]:
        ingest(pool, rid, name)
    assert admitted(runner, "r1") and run_step(pool, ["r1"]).ok
    assert admitted(runner, "e1") and not pool.runs_adapters(["e1"]) and pool.runs_adapters(["e1", "r1"])
    assert run_step(pool, ["r1", "e1"]).ok and pool.slot_of("e1") == ZERO_SLOT
    assert pool.slot_state(1).refcount == 1 and pool.resident_slot("mlp") is None and "mlp" not in pool.store


def test_rollback_undoes_only_what_this_admit_took(adapters):
    _, paths = adapters
    pool = make_pool(paths, max_slots=2)
    for rid, name in [("old", "a1"), ("new", "a2"), ("same", "a1")]:
        ingest(pool, rid, name)
    assert run_step(pool, ["old"]).ok
    held = pool.slot_of("old")
    s, c = step(["old", "new", "same"]), ctx(["old", "new", "same"])
    assert pool.admit(s, c).ok and pool.slot_state(held).refcount == 2
    # another resource refused the step: give back only the fresh takes
    pool.rollback_admit(s, c)
    assert pool.slot_of("old") == held and pool.slot_state(held).refcount == 1
    assert pool.slot_of("new") is None and pool.slot_of("same") is None
    new_slot = pool.resident_slot("a2")
    assert new_slot is not None and pool.slot_state(new_slot).refcount == 0
    # retried: a2 is still resident, so it is not reloaded elsewhere
    assert run_step(pool, ["old", "new", "same"]).ok
    assert pool.slot_of("new") == new_slot and pool.slot_state(held).refcount == 2


def test_a_miss_does_not_evict_a_resident_adapter_a_later_row_names(adapters):
    _, paths = adapters
    pool = make_pool(paths, max_slots=2)
    for rid, name in [("w1", "a1"), ("w2", "a2"), ("r3", "a3"), ("r1", "a1")]:
        ingest(pool, rid, name)
    assert run_step(pool, ["w1", "w2"]).ok
    a1_slot = pool.resident_slot("a1")
    pool.remove_request("w1")  # a1's slot is now the LRU victim
    assert run_step(pool, ["w2"]).ok
    pool.remove_request("w2")
    assert run_step(pool, ["r3", "r1"]).ok
    assert pool.slot_of("r1") == a1_slot and pool.slot_of("r3") == pool.resident_slot("a3") != a1_slot


def test_addressing_contents(adapters):
    _, paths = adapters
    pool = make_pool(paths, max_slots=2)
    for rid, name in [("r1", "a1"), ("b", None), ("r2", "a2")]:
        ingest(pool, rid, name)
    s, c = step(["r1", "b", "r2"], [2, 1, 3]), ctx(["r1", "b", "r2"])
    assert pool.admit(s, c).ok
    addr = pool.plan(s, c)
    assert addr.token_order.tolist() == [2, 0, 1, 3, 4, 5] and addr.slot_indptr.tolist() == [0, 1, 3, 6]
    assert token_slots_of(addr) == [1, 1, 0, 2, 2, 2] and pool.addressing() is addr
    # a later, smaller eager step reads views of its own width
    pool.commit(s, c)
    addr = pool.plan(step(["r2"]), ctx(["r2"]))
    assert addr.token_order.tolist() == [0] and addr.slot_indptr.tolist() == [0, 0, 0, 1]


def test_capture_and_padding_rows_take_nothing_and_buffers_are_per_slot_key(adapters):
    _, paths = adapters
    pool = make_pool(paths, max_slots=2)
    assert pool.admit(step(["d0", "d1"]), ctx(["d0", "d1"], capture=True)).ok
    for rid, name in [("r1", "a1"), ("r2", "a2")]:
        ingest(pool, rid, name)
    ptrs = {}
    for lease_slot in (0, 1):
        for real in (["r1", "r2"], ["r2"]):
            padded = [*real, *range(-1, -1 - (4 - len(real)), -1)]
            s, c = step(padded), ctx(real, lease=SlotLease(lease_slot, BucketKey("decode", 4, 4)), padded=padded)
            assert pool.admit(s, c).ok
            addr = pool.plan(s, c)
            pool.commit(s, c)
            assert addr.token_order.shape == (4,) and token_slots_of(addr)[len(real):] == [0] * (4 - len(real))
            ptrs.setdefault(lease_slot, set()).add(addr.token_order.data_ptr())
    s1, s2 = pool.slot_of("r1"), pool.slot_of("r2")
    assert pool.slot_state(s1).refcount == pool.slot_state(s2).refcount == 1
    assert len(ptrs[0]) == len(ptrs[1]) == 1 and ptrs[0] != ptrs[1] and pool.force_double_buffer

    # a repeated plan on a capture slot reuses its addressing; the other slot keeps its own
    def plan(rids, lease_slot, spans=None):
        return pool.plan(step(rids, spans), ctx(rids, lease=SlotLease(lease_slot, BucketKey("decode", 2, 2))))

    first, other = plan(["r1", "r2"], 0), plan(["r2", "r1"], 1)
    again = plan(["r1", "r2"], 0)
    assert again is first and token_slots_of(again) == [s1, s2] and token_slots_of(other) == [s2, s1]
    assert token_slots_of(plan(["r1", "r2"], 0, spans=[2, 0])) == [s1, s1]  # new spans: planned afresh


def test_store_shards_for_its_rank_and_keeps_an_lru(adapters):
    _, paths = adapters
    spec = make_spec(paths)
    whole = LoraAdapterStore(spec).load("a1", paths["a1"])
    for rank in range(2):
        part = LoraAdapterStore(spec, rank=rank, world_size=2).load("a1", paths["a1"])
        # qkv: B rows split per slice, A whole; o: A columns split, B whole
        torch.testing.assert_close(part.b["qkv"][:, :8], whole.b["qkv"][:, rank * 8:(rank + 1) * 8])
        torch.testing.assert_close(part.b["qkv"][:, 8:12], whole.b["qkv"][:, 16 + rank * 4:16 + (rank + 1) * 4])
        torch.testing.assert_close(part.a["qkv"], whole.a["qkv"])
        torch.testing.assert_close(part.a["o"], whole.a["o"][..., rank * 8:(rank + 1) * 8])
        torch.testing.assert_close(part.b["o"], whole.b["o"])
        assert part.digest == whole.digest
    store = LoraAdapterStore(make_spec(paths, cpu_capacity=2))
    for name in ("a1", "a2"):
        store.load(name, paths[name])
    assert not store.hold("a1")  # a1 now most recent
    store.load("a3", paths["a3"])
    assert "a1" in store and "a3" in store and "a2" not in store
    assert store.get("a3").digest != whole.digest == store.get("a1").digest


def test_a_runtime_load_keeps_no_copy_until_a_request_names_it(adapters):
    """Loads past ``cpu_capacity`` pin nothing outside the host tier's bound: a load's read
    checks the file and is dropped, and the first request on the adapter reads it again."""
    _, paths = adapters
    pool = make_pool({}, cpu_capacity=1)
    copies, read = [], pool.store.read

    def tracked(*args):
        adapter = read(*args)
        copies.append(weakref.ref(adapter))
        return adapter
    pool.store.read = tracked
    for name in ("a1", "a2", "a3"):
        pool.load_adapter(name, paths[name])
    gc.collect()
    assert len(copies) == 3 and not any(ref() for ref in copies)
    ingest(pool, "r", "a2")
    settle_reads(pool)
    assert "a2" in pool.store and len(copies) == 4


# --- numerics and compile ------------------------------------------------------------------------

def _naive_delta(x_tok, ref, layer, projs):
    """Per-token reference straight from the checkpoint, concatenated over ``projs``."""
    out = []
    for proj in projs:
        if (layer, proj) in ref:
            a, b, alpha = ref[(layer, proj)]
            out.append(alpha / a.shape[0] * (x_tok @ a.double().t()) @ b.double().t())
        else:
            out.append(torch.zeros(OUT_WIDTH[proj], dtype=torch.float64))
    return torch.cat(out)


@pytest.mark.parametrize("layer", [0, 1])
def test_callable_matches_per_token_reference(adapters, layer):
    refs, paths = adapters
    pool = make_pool(paths, max_slots=3)
    rows = [("r1", "a1", 2), ("b", None, 1), ("r2", "a2", 3), ("r3", "a3", 1), ("r4", "a1", 2)]
    for rid, name, _ in rows:
        ingest(pool, rid, name)
    spans = [s for _, _, s in rows]
    tokens = sum(spans)
    assert run_step(pool, [r for r, _, _ in rows], spans, commit=False).ok
    # a layer an adapter does not adapt is rank 0 there, which the kernels skip
    s1, s2, s3 = (pool.slot_of(r) for r in ("r1", "r2", "r3"))
    qkv, o = pool.layer_ranks("qkv", layer), pool.layer_ranks("o", layer)
    assert (qkv[s1], qkv[s2], qkv[s3], qkv[ZERO_SLOT]) == (2, 4, 0, 0)
    assert (o[s1], o[s2], o[s3], o[ZERO_SLOT]) == (2, 0, 3, 0)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(tokens, HIDDEN, generator=g, dtype=torch.float64)
    base = torch.randn(tokens, Q_OUT + 2 * KV_OUT, generator=g, dtype=torch.float64)
    y = base.clone()
    assert pool.layer_delta("qkv", layer)(x, y) is y
    xo = torch.randn(tokens, Q_OUT, generator=g, dtype=torch.float64)
    yo = pool.layer_delta("o", layer)(xo, torch.zeros(tokens, HIDDEN, dtype=torch.float64))
    for t, name in enumerate(name for _, name, span in rows for _ in range(span)):
        if name is None:
            torch.testing.assert_close(y[t], base[t])
            assert not yo[t].any()
            continue
        want = _naive_delta(x[t], refs[name], layer, ("q_proj", "k_proj", "v_proj"))
        torch.testing.assert_close(y[t] - base[t], want)
        torch.testing.assert_close(yo[t], _naive_delta(xo[t], refs[name], layer, ("o_proj",)))


def test_compiles_without_graph_breaks(adapters):
    _, paths = adapters
    pool = make_pool(paths)
    ingest(pool, "r1", "a1")
    ingest(pool, "b", None)
    assert run_step(pool, ["r1", "b"], [2, 1], commit=False).ok
    lora = pool.layer_delta("qkv", 0)

    def layer(x, y):
        return lora(x, y) * 2

    x, y = torch.randn(3, HIDDEN, dtype=torch.float64), torch.randn(3, Q_OUT + 2 * KV_OUT, dtype=torch.float64)
    want = layer(x, y.clone())
    torch.testing.assert_close(torch.compile(layer, backend="eager", fullgraph=True)(x, y.clone()), want)


def test_compiled_steps_of_varying_tokens_do_not_recompile(adapters):
    """As the engine compiles an eager forward (``dynamic=None``): one graph for the
    first shape, one dynamic graph from the second, one more for the first base-only
    step (the delta is skipped, guarded on ``pool.active``), and no recompile for the
    later steps, whose token counts, rows and adapters all differ."""
    from torch._dynamo.utils import counters

    _, paths = adapters
    pool = make_pool(paths, max_slots=2)
    lora = pool.layer_delta("qkv", 0)

    def layer(x, y):
        return lora(x, y) * 2

    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    torch._dynamo.reset()
    counters.clear()
    graphs = []
    steps = [(["a1"], [3]), (["a1", None], [2, 3]), (["a2", "a1", None], [4, 1, 4]),
             ([None], [2]), (["a2", "a2"], [1, 6]), (["a1", "a2", None, "a1"], [3, 3, 1, 2])]
    for i, (names, spans) in enumerate(steps):
        rids = [f"s{i}r{j}" for j in range(len(names))]
        for rid, name in zip(rids, names, strict=True):
            ingest(pool, rid, name)
        s, c = step(rids, spans), ctx(rids)
        assert pool.admit(s, c).ok
        pool.plan(s, c)
        tokens = sum(spans)
        x = torch.randn(tokens, HIDDEN, dtype=torch.float64)
        y = torch.randn(tokens, Q_OUT + 2 * KV_OUT, dtype=torch.float64)
        torch.testing.assert_close(compiled(x, y.clone()), layer(x, y.clone()))
        pool.commit(s, c)
        for rid in rids:
            pool.remove_request(rid)
        graphs.append(counters["stats"]["unique_graphs"])
    assert graphs[0] == 1 and graphs[1] <= 2 and graphs[2] == graphs[1], graphs
    assert graphs[3] == graphs[2] + 1 and graphs[-1] == graphs[3], graphs


# --- build: dtype, SP, served targets ------------------------------------------------------------

def test_slots_take_the_nodes_compute_dtype(adapters):
    """A spec without a dtype takes the one its nodes compute in, for slots and host copies."""
    _, paths = adapters
    spec = make_spec(paths, dtype=None)
    pool = LoraAdapterPool.build(spec, EngineResourceInfo(device=CPU, compute_dtype=torch.float16))
    a, b = pool.weights("qkv", 0)
    assert pool.dtype is torch.float16 and a.dtype is b.dtype is torch.float16
    ingest(pool, "r", "a1")
    assert run_step(pool, ["r"]).ok and pool.store.get("a1").a["qkv"].dtype is torch.float16
    # a spec's own dtype wins; with neither, the build says what is missing
    assert LoraAdapterPool(make_spec(paths), "cpu", dtype=torch.float16).dtype is torch.float64
    with pytest.raises(ValueError, match="no dtype"):
        LoraAdapterPool.build(spec, EngineResourceInfo(device=CPU))
    # the serving config's autocast_dtype, as the worker reads it
    from mstar.model.base import parse_autocast_dtype

    assert parse_autocast_dtype("float16") is torch.float16
    assert parse_autocast_dtype("torch.bfloat16") is torch.bfloat16


def test_a_lora_refusal_does_not_look_for_an_eviction_victim(caplog):
    """Evicting KV frees no slot: a LoRA refusal asks for no victim, nor warns."""
    import logging
    from types import SimpleNamespace

    from mstar.worker.worker import Worker

    w = Worker.__new__(Worker)
    asked = []
    engine = SimpleNamespace(
        evictable=lambda node, resources=None: resources is None or "kv" in resources,
        reclaimable=lambda *a, **k: asked.append(a) or 0, is_offloaded=lambda *a: False,
    )
    w.engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    w._last_active = {(1, "LLM"): 0.0}
    with caplog.at_level(logging.WARNING):
        assert w._try_offload_cold_request("LLM", {1}, affected_resources={"lora"}) is None
    assert asked == [] and "No eviction candidate" not in caplog.text
    # control: a KV refusal with nothing reclaimable looks for a victim and warns
    with caplog.at_level(logging.WARNING):
        assert w._try_offload_cold_request("LLM", {1}, affected_resources={"kv"}) is None
    assert asked and "No eviction candidate" in caplog.text


def test_a_node_running_sequence_parallelism_refuses_lora(adapters):
    """LoRA rows are per request, not per SP shard: the build refuses SP > 1 at startup,
    and takes TP with SP 1."""
    from mstar.distributed.communication import CommGroup, JointGroups

    _, paths = adapters

    def build(sp_group):
        groups = JointGroups(tp_group=CommGroup.trivial(), sp_group=sp_group)
        return LoraAdapterPool.build(make_spec(paths), EngineResourceInfo(device=CPU, joint_comm_group=groups))

    with pytest.raises(ValueError, match=r"sequence parallelism \(SP=2\)"):
        build(CommGroup(my_global_rank=0, my_group_rank=0, group_members=[0, 1]))
    assert build(CommGroup.trivial()).targets


def test_served_targets_narrow_to_what_the_adapters_touch(adapters):
    """The conductor keeps only the targets the configured adapters touch and every
    worker builds just those."""
    from types import SimpleNamespace

    from mstar.conductor.conductor import Conductor
    from mstar.model.base import serving_resources

    _, paths = adapters
    model = SimpleNamespace(get_node_resources=lambda: [], get_lora_specs=lambda: [make_spec({})])
    config = {"resources": {"lora": {"adapters": {"a3": paths["a3"]}, "max_slots": 2}}}
    conductor = Conductor.__new__(Conductor)
    conductor.model, conductor.model_config = model, config
    Conductor._settle_lora_adapters(conductor, {"a3": "00" * 32})
    block = config["resources"]["lora"]
    assert block["targets"] == ["o"] and block["adapter_digests"] == {"a3": "00" * 32}
    (llm,) = [s for s in serving_resources(model, config) if isinstance(s, LoraSpec)]
    assert set(llm.targets) == {"o"}
    # mechanism off: without the conductor's list a worker builds every target
    del block["targets"]
    (llm,) = [s for s in serving_resources(model, config) if isinstance(s, LoraSpec)]
    assert set(llm.targets) == {"qkv", "o"}


# --- admission: a full pool is backpressure, not a refused step ----------------------------------

class _Sim:
    """Each step runs every running row plus the waiting rows the runner admits, as the
    micro-scheduler's readiness scan does; running rows end after their length."""

    def __init__(self, pool):
        self.pool, self.runner, self.t = pool, lora_runner(pool), 0
        self.running: dict[str, int] = {}
        self.waiting: list[tuple[str, int]] = []
        self.served_at: dict[str, int] = {}
        self.refused: list[int] = []

    def arrive(self, rid, adapter, length):
        ingest(self.pool, rid, adapter)
        self.waiting.append((rid, length))

    def step(self):
        self.runner.admit_waiting(_nothing_published)
        fresh = [(r, n) for r, n in self.waiting if admitted(self.runner, r)]
        rows = list(self.running) + [r for r, _ in fresh]
        if rows and not run_step(self.pool, rows).ok:
            self.refused.append(self.t)  # every row in it lost the step
            self.pool.rollback_admit(step(rows), ctx(rows))
        elif rows:
            for r, n in fresh:
                self.waiting.remove((r, n))
                self.running[r], self.served_at[r] = n, self.t
            for r in list(self.running):
                self.running[r] -= 1
                if not self.running[r]:
                    del self.running[r]
                    self.runner.remove_request(r)
        self.t += 1


@pytest.mark.parametrize("override", [{"max_slots": 0}, {"max_rank": 8, "targets": "qkv"}])
def test_a_refused_override_leaves_the_spec_as_declared(override):
    """``validate_deployment`` reports the refusal, then splits the spec per node, which
    checks its fields again: a refused value left on it would be reported twice."""
    spec = make_spec({})
    with pytest.raises(ValueError):
        spec.apply_yaml_overrides(**override)
    assert spec == make_spec({})


@pytest.mark.parametrize("decides", [True, False])
def test_a_cold_adapter_waits_without_refusing_steps_and_is_served_once_the_slot_drains(adapters, decides):
    """A new request on a1 every step keeps the one slot held; one on a2 arrives at step 5.
    Requests after it wait behind it, so the slot drains for it. Control: a rank that does
    not decide (a follower's setting) lets a2's row ride in every step, whose admit is
    refused whole."""
    _, paths = adapters
    pool = LoraAdapterPool(make_spec(paths, max_slots=1), "cpu", decides=decides)
    sim = _Sim(pool)
    for t in range(40):
        sim.arrive(f"hot{t}", "a1", 3)
        if t == 5:
            sim.arrive("cold", "a2", 2)
        sim.step()
    if not decides:
        assert sim.refused == list(range(5, 40))
        return
    assert sim.refused == []
    # the hot rows admitted before it drain (3 steps)
    assert sim.served_at["cold"] - 5 <= 3 + 1
    assert max(sim.served_at.values()) > sim.served_at["cold"]


def test_admission_holds_back_only_rows_that_need_a_slot_and_only_on_the_leader(adapters):
    _, paths = adapters
    pool = make_pool(paths, max_slots=1)
    runner = lora_runner(pool)
    for rid, name in [("r1", "a1"), ("base", None), ("r2", "a2")]:
        ingest(pool, rid, name)
    assert admitted(runner, "r1") and run_step(pool, ["r1", "base"]).ok
    assert runner.admit_retrieve("r2", "LLM", "decode").queued
    assert admitted(runner, "r1") and admitted(runner, "base")
    assert admitted(runner, "r2", node="vision")
    # the admit refusal stays as a backstop for a row admission did not see
    assert isinstance(run_step(pool, ["r2"]).reason, AllocationFailed)
    runner.remove_request("r1")
    assert runner.admit_waiting(_nothing_published) == [("LLM", "r2")]
    # a lockstep follower does not decide and keeps no reservations
    follower = LoraAdapterPool(make_spec(paths, max_slots=1), "cpu", decides=False)
    for rid, name in [("r1", "a1"), ("r2", "a2")]:
        ingest(follower, rid, name)
    assert run_step(follower, ["r1"]).ok and admitted(lora_runner(follower), "r2")
    assert not follower._reserved


def test_a_request_asked_again_by_a_second_node_is_reserved_once(adapters):
    """A pool serving two nodes is asked on the second once the first has admitted the
    request: a second reservation would count a1 for good once the first is taken."""
    _, paths = adapters
    pool = make_pool(paths, max_slots=1)
    runner = StepRunner({"lora": pool}, {"LLM": ["lora"], "vision": ["lora"]})
    for rid, name in [("r1", "a1"), ("r2", "a2")]:
        ingest(pool, rid, name)
    assert admitted(runner, "r1") and admitted(runner, "r1", node="vision")
    assert run_step(pool, ["r1"]).ok
    runner.remove_request("r1")
    assert admitted(runner, "r2")


def test_a_request_on_a_resident_adapter_waits_for_its_host_copy(adapters):
    """a1 sits in an unheld slot but out of the host tier. A reservation pins no slot, so a
    miss admitted first could evict a1 and leave r's admit to read it: r is reserved only
    once its copy is in."""
    _, paths = adapters
    pool = make_pool(paths, max_slots=2, cpu_capacity=1)
    runner = lora_runner(pool)
    ingest(pool, "h", "a1")
    assert admitted(runner, "h") and run_step(pool, ["h"]).ok
    runner.remove_request("h")
    ingest(pool, "x", "a2")  # drops a1 from the tier
    settle_reads(pool)
    gate, read = threading.Event(), pool.store._read
    pool.store._read = lambda name, path: gate.wait(10) and read(name, path)
    ingest(pool, "r", "a1")
    assert pool.resident_slot("a1") is not None and "a1" not in pool.store
    assert runner.admit_retrieve("r", "LLM", "decode").queued
    gate.set()
    settle_reads(pool)
    assert runner.admit_waiting(_nothing_published) == [("LLM", "r")]


def test_a_request_whose_read_fails_fails_without_waiting_for_a_slot(adapters):
    """r2 queues for a2's copy on a full pool, and the read fails: r2 is let through at
    once, and its admit fails it instead of refusing the step until a1's slot drains."""
    _, paths = adapters
    pool = make_pool(paths, max_slots=1, cpu_capacity=1)
    runner = lora_runner(pool)
    gate, read = threading.Event(), pool.store._read

    def unreadable(name, path):
        if name == "a2":
            gate.wait(10)
            raise OSError("a2 is unreadable")
        return read(name, path)
    pool.store._read = unreadable
    for rid, name in [("r1", "a1"), ("r2", "a2")]:
        ingest(pool, rid, name)
    assert admitted(runner, "r1") and run_step(pool, ["r1"]).ok
    assert runner.admit_retrieve("r2", "LLM", "decode").queued
    gate.set()
    settle_reads(pool)
    assert runner.admit_waiting(_nothing_published) == [("LLM", "r2")]
    reason = run_step(pool, ["r1", "r2"]).reason
    assert isinstance(reason, AdmitRuntimeError) and reason.request_ids == {"r2"}


def test_a_request_let_through_on_a_failed_read_takes_no_slot_a_counted_one_needs(adapters):
    """rB is counted on the last slot before a2's read fails and r2 is let through, slots or
    not. In a step that lists r2 ahead of rB, r2 fails and rB is not refused for the slot."""
    _, paths = adapters
    pool = make_pool(paths, max_slots=2, cpu_capacity=1)
    runner = lora_runner(pool)
    gate, read = threading.Event(), pool.store._read

    def unreadable(name, path):
        if name == "a2":
            gate.wait(10)
            raise OSError("a2 is unreadable")
        return read(name, path)
    pool.store._read = unreadable
    for rid, name in [("r1", "a1"), ("rB", "a3")]:
        ingest(pool, rid, name)
    settle_reads(pool)
    assert admitted(runner, "r1") and run_step(pool, ["r1"]).ok
    assert admitted(runner, "rB")
    ingest(pool, "r2", "a2")
    assert runner.admit_retrieve("r2", "LLM", "decode").queued
    gate.set()
    settle_reads(pool)
    assert runner.admit_waiting(_nothing_published) == [("LLM", "r2")]
    reason = run_step(pool, ["r1", "r2", "rB"]).reason
    assert isinstance(reason, AdmitRuntimeError) and reason.request_ids == {"r2"}


def test_a_request_on_a_resident_adapter_whose_read_failed_still_counts_a_slot(adapters):
    """a1 sits in an unheld slot, out of the host tier, and its read fails: r may still run
    on that slot, so it is counted like any request, not let through, which would promise
    the one slot to y and r both."""
    _, paths = adapters
    pool = make_pool(paths, max_slots=1, cpu_capacity=1)
    runner = lora_runner(pool)
    ingest(pool, "h", "a1")
    assert admitted(runner, "h") and run_step(pool, ["h"]).ok
    runner.remove_request("h")
    ingest(pool, "x", "a2")  # drops a1 from the tier
    settle_reads(pool)
    read = pool.store._read

    def unreadable(name, path):
        if name == "a1":
            raise OSError("a1 is unreadable")
        return read(name, path)
    pool.store._read = unreadable
    for rid, name in [("r", "a1"), ("y", "a3")]:
        ingest(pool, rid, name)
    settle_reads(pool)
    assert pool.resident_slot("a1") is not None and "a1" not in pool.store
    assert admitted(runner, "y") and runner.admit_retrieve("r", "LLM", "decode").queued
    # y's miss evicts a1, so r is in no slot now: let through at once, to fail
    assert run_step(pool, ["y"]).ok and pool.resident_slot("a1") is None
    assert runner.admit_waiting(_nothing_published) == [("LLM", "r")]
    reason = run_step(pool, ["y", "r"]).reason
    assert isinstance(reason, AdmitRuntimeError) and reason.request_ids == {"r"}


def test_two_missing_adapters_cannot_both_take_the_last_slot(adapters):
    _, paths = adapters
    pool = make_pool(paths, max_slots=2)
    runner = lora_runner(pool)
    for rid, name in [("r1", "a1"), ("r2", "a2"), ("r3", "a3"), ("r4", "a2")]:
        ingest(pool, rid, name)
    assert admitted(runner, "r1") and run_step(pool, ["r1"]).ok
    # r2 reserves the last slot for a2; r3 waits, and r4 behind it
    assert {rid: admitted(runner, rid) for rid in ("r2", "r3", "r4")} == {"r2": True, "r3": False, "r4": False}
    assert run_step(pool, ["r1", "r2"]).ok
    runner.remove_request("r1")
    assert runner.admit_waiting(_nothing_published) == [("LLM", "r3"), ("LLM", "r4")]
    assert run_step(pool, ["r2", "r3", "r4"]).ok and pool.slot_of("r2") == pool.slot_of("r4")


class _Pages(Resource):
    """KV-like capacity: a unit per request, reserved at admission."""

    def __init__(self, units: int):
        self.free, self.held = units, set()

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def ask(self, rid, node_name, graph_walk):
        return None if rid in self.held else ASKS

    def reserve(self, rid, node_name, graph_walk, published):
        if not self.free:
            return self.free
        self.free -= 1
        self.held.add(rid)
        return None

    def unreserve(self, rid):
        self.free += 1
        self.held.discard(rid)

    def grew_since(self, mark):
        return self.free > mark

    def remove_request(self, rid):
        if rid in self.held:
            self.unreserve(rid)


def test_a_request_waiting_for_a_slot_holds_no_kv(adapters):
    """A request on a cold adapter waits in the same admission as KV, so it keeps no KV
    reservation while it waits; requests after it wait in arrival order, as behind any
    resource that asks."""
    _, paths = adapters
    pool, kv = make_pool(paths, max_slots=1), _Pages(2)
    runner = lora_runner(kv, pool, keys=("kv", "lora"))
    for rid, name in [("r1", "a1"), ("r2", "a2"), ("base", None)]:
        ingest(pool, rid, name)
    assert admitted(runner, "r1") and run_step(pool, ["r1"]).ok
    waits = runner.admit_retrieve("r2", "LLM", "decode")
    assert kv.held == {"r1"}, "a request waiting for a LoRA slot kept its KV reservation"
    assert waits.queued and runner.admit_retrieve("base", "LLM", "decode").queued
    runner.remove_request("r1")
    assert runner.admit_waiting(_nothing_published) == [("LLM", "r2"), ("LLM", "base")]
    assert kv.held == {"base", "r2"} and run_step(pool, ["r2", "base"]).ok
