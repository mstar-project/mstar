"""LoRA on each module type a model declares it from (``module_lora_spec``), on CPU.

Bound, every capable module type computes per row what it computes with that row's adapter
merged (``W + scaling * B @ A``), in a batch of two adapters and base rows, at each simulated
TP degree it shards at; unbound, it runs its pre-LoRA ops, graph and guards. Then what
binding, the module walk and the checkpoint reader refuse, by name.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytest.importorskip("safetensors")
from lora_module_helpers import DTYPE, compute, plan, pool, rand, run_ranks, serve, write_peft  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.distributed.utils import split_tensor_along_last_dim  # noqa: E402
from mstar.engine.resources.lora.checkpoint import normalize_lora_key, parse_lora_key, read_lora_file  # noqa: E402
from mstar.engine.resources.lora.config import describe_load_error  # noqa: E402
from mstar.engine.resources.lora.ops import lora_gather_shrink  # noqa: E402
from mstar.engine.resources.lora.store import inspect_adapter, inspect_keys  # noqa: E402
from mstar.model.components.distributed.embedding import VocabParallelEmbedding  # noqa: E402
from mstar.model.components.distributed.linear import (  # noqa: E402
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from mstar.model.components.distributed.linear_attn import _FusedBlockColumnParallelLinear  # noqa: E402
from mstar.model.components.linear import FusedColumnLinear  # noqa: E402
from mstar.model.components.linear_attn import GATE_ALIGN, SPLIT_BLOCK_NAMES, SPLIT_SHARD_BLOCKS, gate_pad  # noqa: E402
from mstar.model.components.lora.capable import make_lora_capable  # noqa: E402
from mstar.model.components.lora.modules import (  # noqa: E402
    PER_SEGMENT,
    _target_names,
    bind_module_lora,
    module_lora_spec,
)
from mstar.model.loader.base import CheckpointRules, StackedParamRule, load_weights_into  # noqa: E402

HIDDEN = 8
ROWS = [("r0", "a1", 3), ("r1", None, 2), ("r2", "a2", 4)]
TOKENS = sum(n for _, _, n in ROWS)


# --- the module types --------------------------------------------------------------------

@dataclass
class Case:
    """One capable module type: built at a rank (``make``), loaded from checkpoint modules
    ``ckpt`` (name -> output rows) through ``stacked`` rules. ``top``: one module outside
    the stack (a head, an embedding), else one per layer of a two-layer stack."""

    make: Callable[[CommGroup], nn.Module]
    attr: str
    ckpt: dict[str, int]
    worlds: tuple[int, ...]
    stacked: tuple[StackedParamRule, ...] = ()
    in_features: int = HIDDEN
    kind: str = "linear"  # "linear", "row", "gather", "embedding", "conv"
    bias: bool = False
    top: bool = False
    rows: dict | None = None  # module_lora_spec's `rows`
    # checkpoint modules the second adapter touches (the first touches all)
    partial: tuple[str, ...] = ()


QKV_RULES = tuple(StackedParamRule(".qkv_proj", f".{s}_proj", s) for s in "qkv")
GATE_UP_RULES = (StackedParamRule(".gate_up_proj", ".gate_proj", 0), StackedParamRule(".gate_up_proj", ".up_proj", 1))
FUSED_RULES = tuple(StackedParamRule(".qkv", f".to_{s}", s) for s in "qkv")
GDN_RULES = tuple(StackedParamRule(".in_proj_fused", f".in_proj_{s}", s) for s in ("qkv", "z", "a", "b"))
V_HEADS = 4


def _gdn(group: CommGroup) -> nn.Module:
    """GDN's fused in_proj: ``[q|k|v]`` one checkpoint tensor, a per-rank pad after ``a``."""
    tp = group.world_size
    return _FusedBlockColumnParallelLinear(
        comm_group=group, input_size=HIDDEN,
        output_sizes=[8, 8, 16, 16, V_HEADS, gate_pad(V_HEADS // tp) * tp, V_HEADS],
        bias=False, shard_map=SPLIT_SHARD_BLOCKS, block_names=SPLIT_BLOCK_NAMES, pad_align=GATE_ALIGN,
    )


CASES = {
    "column": Case(lambda g: ColumnParallelLinear(g, HIDDEN, 16, bias=True), "proj", {"proj": 16}, (1, 2, 4),
                   bias=True),
    "column_gather": Case(lambda g: ColumnParallelLinear(g, HIDDEN, 16, gather_output=True), "proj", {"proj": 16},
                          (2,), kind="gather"),
    "merged": Case(lambda g: MergedColumnParallelLinear(g, HIDDEN, [8, 12], bias=False), "gate_up_proj",
                   {"gate_proj": 8, "up_proj": 12}, (1, 2, 4), GATE_UP_RULES, partial=("up_proj",)),
    # 2 K/V heads: replicated across 4 ranks
    "qkv": Case(lambda g: QKVParallelLinear(g, HIDDEN, 4, total_num_heads=4, total_num_kv_heads=2, bias=True),
                "qkv_proj", {"q_proj": 16, "k_proj": 8, "v_proj": 8}, (1, 2, 4), QKV_RULES, bias=True,
                partial=("q_proj", "v_proj")),
    "row": Case(lambda g: RowParallelLinear(g, 16, HIDDEN, bias=True), "o_proj", {"o_proj": HIDDEN}, (1, 2, 4),
                in_features=16, kind="row", bias=True),
    "gdn_packs": Case(_gdn, "in_proj_fused", {"in_proj_qkv": 32, "in_proj_z": 16, "in_proj_a": V_HEADS,
                                              "in_proj_b": V_HEADS}, (1, 2, 4), GDN_RULES,
                      partial=("in_proj_z", "in_proj_b")),
    "fused_column": Case(lambda g: FusedColumnLinear(HIDDEN, {"q": 8, "k": 4, "v": 4}, bias=True), "qkv",
                         {"to_q": 8, "to_k": 4, "to_v": 4}, (1,), FUSED_RULES, bias=True, partial=("to_k",)),
    # one row per segment (its last token's), each rank its vocab columns
    "lm_head": Case(lambda g: ColumnParallelLinear(g, HIDDEN, 32), "lm_head", {"lm_head": 32}, (1, 2, 4),
                    top=True, rows={"lm_head": PER_SEGMENT}),
    "embedding": Case(lambda g: VocabParallelEmbedding(32, HIDDEN, comm_group=g), "embed_tokens",
                      {"embed_tokens": HIDDEN}, (1, 2), in_features=32, kind="embedding", top=True),
    "patch_conv": Case(lambda g: nn.Conv3d(2, 12, kernel_size=2, stride=2, bias=True), "proj", {"proj": 12}, (1,),
                       in_features=16, kind="conv", bias=True),
    # not split by a TP group: whole on every rank
    "linear": Case(lambda g: nn.Linear(HIDDEN, 12, bias=True), "fc", {"fc": 12}, (1, 2), bias=True),
}


class _Root(nn.Module):
    def __init__(self, case: Case, group: CommGroup):
        super().__init__()
        if case.top:
            setattr(self, case.attr, case.make(group))
        else:
            self.layers = nn.ModuleList(nn.Module() for _ in range(2))
            for layer in self.layers:
                setattr(layer, case.attr, case.make(group))

    def modules_run(self, case: Case) -> list[nn.Module]:
        if case.top:
            return [getattr(self, case.attr)]
        return [getattr(layer, case.attr) for layer in self.layers]


class _Node(nn.Module):  # a node submodule holding the root, as binding sees it
    def __init__(self, root: nn.Module):
        super().__init__()
        self.inner = root


def _prefixes(case: Case) -> list[str]:
    return [""] if case.top else ["layers.0.", "layers.1."]


def _weights(case: Case) -> dict[str, torch.Tensor]:
    out = {}
    for j, prefix in enumerate(_prefixes(case)):
        for i, (name, rows) in enumerate(case.ckpt.items()):
            shape = {"conv": (rows, 2, 2, 2, 2), "embedding": (case.in_features, rows)}.get(
                case.kind, (rows, case.in_features))
            out[f"{prefix}{name}.weight"] = rand(*shape, seed=10 * j + i, scale=0.3)
            if case.bias:
                out[f"{prefix}{name}.bias"] = rand(rows, seed=10 * j + 50 + i, scale=0.3)
    return out


def _factors(case: Case, rows: int, rank: int, seed: int):
    """Random factors, fp32-exact: the adapter file holds them in fp32."""
    if case.kind == "conv":
        a, b = rand(rank, 2, 2, 2, 2, seed=seed, scale=0.3), rand(rows, rank, 1, 1, 1, seed=seed + 1, scale=0.3)
    else:
        a, b = rand(rank, case.in_features, seed=seed, scale=0.3), rand(rows, rank, seed=seed + 1, scale=0.3)
    return a.float().double(), b.float().double()


def _adapters(case: Case, tmp_path):
    """``a1``: every checkpoint module, rank 4, alpha 8 (scaling 2). ``a2``: ``case.partial``
    (or all) of the last layer only, rank 2, no alpha (scaling 1)."""
    prefixes = _prefixes(case)
    emb = tuple(case.ckpt) if case.kind == "embedding" else ()
    a1 = {f"{p}{n}": _factors(case, rows, 4, 100 + 7 * i + 31 * j)
          for j, p in enumerate(prefixes) for i, (n, rows) in enumerate(case.ckpt.items())}
    a2 = {f"{prefixes[-1]}{n}": _factors(case, case.ckpt[n], 2, 300 + 7 * i)
          for i, n in enumerate(case.partial or tuple(case.ckpt))}
    refs = {"a1": {m: (a, b, 2.0) for m, (a, b) in a1.items()}, "a2": {m: (a, b, 1.0) for m, (a, b) in a2.items()}}
    paths = {
        "a1": write_peft(tmp_path / "a1.safetensors", a1, alpha=8.0, embedding=emb),
        "a2": write_peft(tmp_path / "a2.safetensors", a2, embedding=emb),
    }
    return paths, refs


def _row_adapters(case: Case) -> list[str | None]:
    """The adapter of each row the module sees: one row per segment for a head, else per token."""
    if case.rows:
        return [a for _, a, _ in ROWS]
    return [a for _, a, n in ROWS for _ in range(n)]


def _inputs(case: Case) -> torch.Tensor:
    n = len(_row_adapters(case))
    if case.kind == "embedding":
        return torch.randint(0, case.in_features, (n,), generator=torch.Generator().manual_seed(3))
    if case.kind == "conv":
        return rand(n, 2, 2, 2, 2, seed=4)
    return rand(n, case.in_features, seed=4)


def _ckpt_out(case: Case, x, w, b, lora) -> torch.Tensor:
    """One checkpoint module's full output, ``lora`` (per row ``(A, B, scaling)`` or None) merged."""
    y = w[x] if case.kind == "embedding" else x.reshape(x.shape[0], -1) @ w.reshape(w.shape[0], -1).t()
    if b is not None:
        y = y + b
    for t, entry in enumerate(lora):
        if entry is not None:
            a, bb, s = entry
            a, bb = a.reshape(a.shape[0], -1), bb.reshape(bb.shape[0], -1)
            h = a[:, x[t]] if case.kind == "embedding" else a @ x[t].reshape(-1)
            y[t] += s * (bb @ h)
    return y


def _expected(case: Case, spec, prefix: str, x, weights, refs, merged: bool, rank: int, world: int):
    """What the module at ``prefix`` outputs on ``rank``: each checkpoint module's full
    output split into the target's slices, then the rows of each this rank holds (the
    target's arithmetic), zeros in alignment padding."""
    target = spec.targets[spec.module_map(f"{prefix}{next(iter(case.ckpt))}").target]
    slices: dict[int, torch.Tensor] = {}
    for name in case.ckpt:
        lora = [refs[a].get(f"{prefix}{name}") if merged and a else None for a in _row_adapters(case)]
        y = _ckpt_out(case, x, weights[f"{prefix}{name}.weight"], weights.get(f"{prefix}{name}.bias"), lora)
        idx = target.module_slices(spec.module_map(f"{prefix}{name}").shard_id)
        for i, part in zip(idx, y.split([target.out_slices[i] for i in idx], dim=-1), strict=True):
            slices[i] = part
    if case.kind in ("row", "gather"):
        return torch.cat([slices[i] for i in range(target.num_slices)], dim=-1)
    parts = []
    for i in range(target.num_slices):
        start, size = target.slice_rows(i, rank, world)
        parts.append(F.pad(slices[i][:, start:start + size], (0, target.local_width(i, world) - size)))
    return torch.cat(parts, dim=-1)


def _local_input(case: Case, x, rank: int, world: int):
    size = case.in_features // world
    return x[:, rank * size:(rank + 1) * size] if case.kind == "row" else x


def _build(case: Case, group: CommGroup, weights) -> _Root:
    # `.to`, not the default dtype: ranks build on threads of their own
    root = _Root(case, group).to(DTYPE)
    for module in root.modules():
        if isinstance(module, _FusedBlockColumnParallelLinear):
            module.weight.data.zero_()  # the pad rows, never loaded
    loaded = load_weights_into(root, weights.items(), stacked_params=list(case.stacked))
    assert loaded == {n for n, _ in root.named_parameters()}
    return root.requires_grad_(False)


def _spec(case: Case):
    return module_lora_spec(
        lambda group: _Root(case, group), nodes={"N"}, root_path="inner",
        rules=CheckpointRules(stacked_params=case.stacked), embeddings=True, rows=case.rows,
    )


@pytest.mark.parametrize("name,world", [(n, w) for n, c in CASES.items() for w in c.worlds])
def test_mixed_batch_matches_each_rows_merged_adapter(name, world, tmp_path):
    case = CASES[name]
    paths, refs = _adapters(case, tmp_path)
    spec = serve(_spec(case), paths)
    weights, x = _weights(case), _inputs(case)

    def rank_run(rank, group):
        root = _build(case, group, weights)
        unbound = [m(_local_input(case, x, rank, world)) for m in root.modules_run(case)]
        lora_pool = pool(spec, rank, world)
        plan([lora_pool], ROWS)
        bind_module_lora(_Node(root), {"lora": lora_pool})
        off_vocab = None
        with compute(lora_pool):
            bound = [m(_local_input(case, x, rank, world)) for m in root.modules_run(case)]
            if case.kind == "embedding":
                ids = x.clone()
                ids[0] = -1  # an adapter row's id outside the vocabulary (an image position)
                off_vocab = root.embed_tokens.lora_delta(ids, torch.zeros(TOKENS, HIDDEN, dtype=DTYPE))
        return unbound, bound, off_vocab

    on = torch.tensor([a is not None for a in _row_adapters(case)])
    for rank, (unbound, bound, off_vocab) in enumerate(run_ranks(world, rank_run)):
        for prefix, base_out, lora_out in zip(_prefixes(case), unbound, bound, strict=True):
            rows = base_out.shape[0]
            base_out, lora_out = base_out.reshape(rows, -1), lora_out.reshape(rows, -1)
            base = _expected(case, spec, prefix, x, weights, refs, False, rank, world)
            merged = _expected(case, spec, prefix, x, weights, refs, True, rank, world)
            # the layout oracle against the loader: unadapted output, laid out by the target
            torch.testing.assert_close(base_out, base, rtol=0, atol=1e-12)
            torch.testing.assert_close(lora_out, merged, rtol=1e-10, atol=1e-10)
            # control, mechanism off: unbound, the module misses the merged rows by far more
            assert (merged[on] - base_out[on]).abs().max() > 1e-3, (name, prefix)
            torch.testing.assert_close(lora_out[~on], base[~on], rtol=0, atol=1e-12)
        if off_vocab is not None:
            assert torch.equal(off_vocab[0], torch.zeros(HIDDEN, dtype=DTYPE))
            torch.testing.assert_close(off_vocab[1:], (bound[0] - unbound[0])[1:], rtol=0, atol=1e-10)


# --- unbound: exactly the forward before LoRA ---------------------------------------------

def _old_column(self, input_):
    bias = self.bias if not self.skip_bias_add else None
    output_parallel = torch.nn.functional.linear(input_, self.weight, bias)
    if self.gather_output and self.tp_size > 1:
        return self.comm_group.all_gather(output_parallel, dim=-1)
    return output_parallel


def _old_row(self, input_):
    if self.input_is_parallel:
        input_parallel = input_
    else:
        input_parallel = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)[self.tp_rank].contiguous()
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
    output_parallel = torch.nn.functional.linear(input_parallel, self.weight, bias_)
    if self.reduce_results and self.tp_size > 1:
        return self.comm_group.all_reduce(output_parallel)
    return output_parallel


def _old_embedding(self, input_):
    if self.tp_size == 1:
        return F.embedding(input_, self.weight)
    mask = (input_ >= self.vocab_start_index) & (input_ < self.vocab_end_index)
    local_input = (input_ - self.vocab_start_index).clamp_(0, self.num_embeddings_per_partition - 1)
    local_output = F.embedding(local_input, self.weight)
    local_output = local_output * mask.unsqueeze(-1).to(local_output.dtype)
    return self.comm_group.all_reduce(local_output)


# the forwards before LoRA (44ab46571); a plain linear / conv keeps its class while unbound
OLD = {
    "column": _old_column, "column_gather": _old_column, "merged": _old_column, "qkv": _old_column,
    "gdn_packs": _old_column, "lm_head": _old_column, "row": _old_row,
    "fused_column": lambda self, x: F.linear(x, self.weight, self.bias), "embedding": _old_embedding,
    "linear": nn.Linear.forward, "patch_conv": nn.Conv3d.forward,
}


class _Ops(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


@contextmanager
def _dispatch_flags_restored():
    """Torch keeps "inside a dispatch mode" in module globals, not per thread, so a mode
    entered on several rank threads at once can leave them set; dynamo then skips every
    later frame in the process (this test's compile, and other tests on the worker)."""
    from torch.utils import _python_dispatch as dispatch
    names = (
        "_is_in_torch_dispatch_mode", "_is_in_non_infra_torch_dispatch_mode",
        "_is_in_any_mode_without_ignore_compile_internals",
    )
    saved = {n: getattr(dispatch, n) for n in names if hasattr(dispatch, n)}
    try:
        yield
    finally:
        for n, value in saved.items():
            setattr(dispatch, n, value)


def _guards(module, x) -> list[str]:
    torch._dynamo.reset()
    return [str(g) for g in torch._dynamo.explain(module)(x).out_guards]


@pytest.mark.parametrize("name", sorted(CASES))
def test_unbound_module_runs_the_pre_lora_forward(name):
    """Same aten ops and outputs (under TP its collectives too), dynamo graph, and no guard
    on LoRA: a deployment without a lora block traces as it did before."""
    case = CASES[name]
    weights, x = _weights(case), _inputs(case)
    world = max(w for w in case.worlds if w <= 2)

    def rank_run(rank, group):
        module = _build(case, group, weights).modules_run(case)[0]
        inp = _local_input(case, x, rank, world)
        with _Ops() as new_ops:
            new = module(inp)
        with _Ops() as old_ops:
            old = OLD[name](module, inp)
        return new_ops.ops, old_ops.ops, new, old

    with _dispatch_flags_restored():
        ranks = run_ranks(world, rank_run)
    for new_ops, old_ops, new, old in ranks:
        assert new_ops == old_ops
        assert torch.equal(new, old)

    module = _build(case, CommGroup.trivial(), weights).modules_run(case)[0]
    graphs = []

    def capture(gm, example_inputs):
        graphs.append([(n.op, str(n.target)) for n in gm.graph.nodes if n.op.startswith("call_")])
        return gm.forward

    torch._dynamo.reset()
    torch.compile(module, backend=capture, fullgraph=True)(x)
    torch.compile(lambda inp: OLD[name](module, inp), backend=capture, fullgraph=True)(x)
    assert len(graphs) == 2 and graphs[0] and graphs[0] == graphs[1], graphs
    assert [g for g in _guards(module, x) if "lora" in g] == []
    # control: swapped to its capable class, the delta slot is read and guarded
    assert any("lora_delta" in g for g in _guards(make_lora_capable(module), x))


# --- what binding refuses ------------------------------------------------------------------

def test_binding_refuses_a_second_delta_a_split_target_and_a_hand_spec(tmp_path):
    case = CASES["qkv"]
    paths, _ = _adapters(case, tmp_path)
    spec = serve(_spec(case), paths)
    weights = _weights(case)

    # double binding: a module already holding a delta from another resource of the node
    node = _Node(_build(case, CommGroup.trivial(), weights))
    bind_module_lora(node, {"lora": pool(spec)})
    with pytest.raises(ValueError, match="already holds a LoRA delta"):
        bind_module_lora(node, {"other": pool(spec)})

    root = _build(case, CommGroup.trivial(), weights)
    # a target the pool splits over 2 ranks, on a module built whole: wrong columns
    with pytest.raises(ValueError, match="whole on every rank"):
        bind_module_lora(_Node(root), {"lora": pool(spec, rank=0, world=2)})
    assert type(root.layers[0].qkv_proj) is QKVParallelLinear

    # a spec not built by `module_lora_spec`: nothing would bind it, so no rows or deltas
    with pytest.raises(ValueError, match="module_lora_spec"):
        bind_module_lora(_Node(root), {"lora": pool(replace(spec, module_map=lambda m: None, adapters={}))})
    assert type(root.layers[0].qkv_proj) is QKVParallelLinear


def test_target_names_widen_a_role_outside_every_stack():
    # a stacked role and a top-level module with the same role path
    names = _target_names([("layers", "attn.proj"), (None, "attn.proj")], {})
    assert sorted(names.values()) == ["layers_attn_proj", "root_attn_proj"]


# --- what the module walk refuses ----------------------------------------------------------

class _OwnForward(nn.Linear):
    def forward(self, x):
        return super().forward(x) * 2


def _with(module: nn.Module, change: Callable[[nn.Module], object]) -> nn.Module:
    change(module)
    return module


REFUSED = {
    "moe_experts": (lambda: _with(nn.Module(), lambda m: m.register_parameter(
        "gate_up_proj", nn.Parameter(torch.empty(4, 16, HIDDEN)))), "not a LoRA-capable linear"),
    "nn_embedding": (lambda: nn.Embedding(32, HIDDEN), "nn.Embedding"),
    "token_embedding_not_served": (lambda: VocabParallelEmbedding(32, HIDDEN), "a token embedding"),
    "subclass_forward": (lambda: _OwnForward(HIDDEN, HIDDEN), "a forward of its own"),
    "instance_forward": (lambda: _with(nn.Linear(HIDDEN, HIDDEN), lambda m: setattr(m, "forward", lambda x: x)),
                         "replaced on the instance"),
    "forward_hooks": (lambda: _with(nn.Linear(HIDDEN, HIDDEN), lambda m: m.register_forward_hook(lambda *a: None)),
                      "forward hooks"),
    "depthwise_conv": (lambda: nn.Conv1d(HIDDEN, HIDDEN, 3, groups=HIDDEN), "groups 8"),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_an_adapter_on_a_module_the_walk_refuses_is_refused_with_the_reason(name, tmp_path):
    make, reason = REFUSED[name]
    with torch.device("meta"):
        root = nn.Module()
        root.layers = nn.ModuleList([nn.Module()])
        root.layers[0].fc = nn.Linear(HIDDEN, HIDDEN)
        root.layers[0].bad = make()
    spec = module_lora_spec(lambda _: root, nodes={"N"}, root_path="")
    assert set(spec.targets) == {"fc"} and reason in spec.module_map.unservable["layers.0.bad"]
    path = write_peft(tmp_path / "a.safetensors", {
        "layers.0.fc": (rand(2, HIDDEN, seed=1), rand(HIDDEN, 2, seed=2)),
        "layers.0.bad": (rand(2, HIDDEN, seed=3), rand(HIDDEN, 2, seed=4)),
    })
    serve(spec, {"a": path})
    info = inspect_adapter(spec, path)
    assert info.unmapped == ("layers.0.bad",)
    with pytest.raises(ValueError, match=re.escape(reason)):
        spec.check_adapter("a", info)


def test_a_deployment_without_a_lora_block_never_builds_the_root():
    """Every deployment of a model declares its LoRA spec; the root is built and walked
    only once the spec is read, which only a deployment opting into LoRA does."""
    from types import SimpleNamespace

    from mstar.model.base import serving_resources

    built = []

    def build(group):
        built.append(group)
        root = nn.Module()
        root.layers = nn.ModuleList([nn.Module()])
        root.layers[0].fc = nn.Linear(HIDDEN, HIDDEN)
        return root

    model = SimpleNamespace(get_node_resources=lambda: [module_lora_spec(build, nodes={"N"}, root_path="")])
    assert serving_resources(model, {"resources": {}}) == [] and not built
    (spec,) = serving_resources(model, {"resources": {"lora": {}}})
    assert set(spec.targets) == {"fc"} and len(built) == 1


# --- the checkpoint: keys, scaling, header refusals ----------------------------------------

@pytest.mark.parametrize("key,want", [
    ("base_model.model.model.embed_tokens.lora_embedding_A", ("model.embed_tokens", "A")),
    ("base_model.model.layers.0.q_proj.lora_down.weight", ("layers.0.q_proj", "A")),
    ("base_model.model.layers.0.q_proj.alpha", ("layers.0.q_proj", "alpha")),
    ("base_model.model.layers.0.q_proj.lora_magnitude_vector", "DoRA"),
    ("lora_unet_double_blocks_0_img_attn_qkv.dora_scale", "DoRA"),
    ("base_model.model.lm_head.modules_to_save.weight", "modules_to_save"),
    ("base_model.model.embed_tokens.token_adapter.trainable_tokens_delta", "trainable_token"),
    ("base_model.model.layers.0.q_proj.lora_B.bias", "lora_bias"),
    ("base_model.model.score.weight", "not a lora_A"),
    ("base_model.model.layers.0.q_proj.oft_R.weight", "not a lora_A"),
])
def test_a_checkpoint_key_is_parsed_or_refused_by_what_it_is(key, want):
    if isinstance(want, str):
        with pytest.raises(ValueError, match=want):
            parse_lora_key(normalize_lora_key(key))
    else:
        assert parse_lora_key(normalize_lora_key(key)) == want


PATTERN_RANKS = {(0, "q_proj"): 4, (0, "k_proj"): 2, (0, "v_proj"): 4, (1, "v_proj"): 4}


@pytest.mark.parametrize("config,want", [
    # alpha / r per module: k at its own rank, v on layer 0 by its own alpha
    ({"r": 4, "lora_alpha": 8, "rank_pattern": {"k_proj": 2}, "alpha_pattern": {"layers.0.self_attn.v_proj": 32}},
     {("0", "q_proj"): 8, ("0", "k_proj"): 8, ("0", "v_proj"): 32, ("1", "v_proj"): 8}),
    # rslora: alpha / sqrt(r), stored as the alpha' whose alpha' / r equals it
    ({"r": 4, "lora_alpha": 8, "use_rslora": True, "alpha_pattern": {"v_proj": 16}},
     {("0", "q_proj"): 16, ("0", "k_proj"): 8 * math.sqrt(2), ("0", "v_proj"): 32, ("1", "v_proj"): 32}),
    # an alpha_pattern alone: the rest take alpha = r
    ({"alpha_pattern": {"q_proj": 12}},
     {("0", "q_proj"): 12, ("0", "k_proj"): 2, ("0", "v_proj"): 4, ("1", "v_proj"): 4}),
    # control, no patterns: lora_alpha everywhere (wrong on the modules patterns name)
    ({"r": 4, "lora_alpha": 8}, {("0", "q_proj"): 8, ("0", "k_proj"): 8, ("0", "v_proj"): 8, ("1", "v_proj"): 8}),
    # a rank_pattern the weights contradict
    ({"r": 4, "lora_alpha": 8, "rank_pattern": {"k_proj": 8}}, "rank_pattern"),
])
def test_alpha_and_rank_patterns_scale_each_module_as_peft_does(config, want, tmp_path):
    g = torch.Generator().manual_seed(0)
    tensors = {}
    for (layer, proj), r in PATTERN_RANKS.items():
        module = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
        tensors[f"{module}.lora_A.weight"] = torch.randn(r, 8, generator=g)
        tensors[f"{module}.lora_B.weight"] = torch.randn(8, r, generator=g)
    path = str(tmp_path / "adapter_model.safetensors")
    save_file(tensors, path)
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    if isinstance(want, str):
        with pytest.raises(ValueError, match=want):
            read_lora_file(path)
        return
    alphas = {tuple(m.split(".")[-3::2]): layer.alpha for m, layer in read_lora_file(path).layers.items()}
    assert alphas == pytest.approx(want)


def test_header_refusals_reach_the_client_as_themselves():
    """A misfit and a non-LoRA key are refusals a runtime load answers with as written."""
    with torch.device("meta"):
        root = nn.Module()
        root.layers = nn.ModuleList([nn.Module()])
        root.layers[0].fc = nn.Linear(HIDDEN, HIDDEN)
    spec = serve(module_lora_spec(lambda _: root, nodes={"N"}, root_path=""), {})
    key = "base_model.model.layers.0.fc.lora_{}.weight"
    bad = inspect_keys(spec, [(key.format("A"), (4, 2 * HIDDEN)), (key.format("B"), (HIDDEN, 4))])
    with pytest.raises(ValueError, match="does not fit") as misfit:
        spec.check_adapter("x", bad)
    with pytest.raises(ValueError, match="DoRA") as dora:
        inspect_keys(spec, [("base_model.model.layers.0.fc.lora_magnitude_vector", (HIDDEN,))])
    for exc in (misfit.value, dora.value):
        assert describe_load_error(exc) == str(exc)
    assert "/srv" not in describe_load_error(OSError("/srv/adapters/x: permission denied"))


def test_gather_shrink_op_matches_its_fake():
    torch.library.opcheck(
        lora_gather_shrink,
        # tokens on slots [1, 2, 0, 1], grouped by slot
        (torch.tensor([0, 5, -1, 31]), rand(3, 4, 32, seed=1), torch.tensor([2, 0, 3, 1], dtype=torch.int32),
         torch.tensor([0, 1, 3, 4], dtype=torch.int32), torch.tensor([0, 4, 2], dtype=torch.int32)),
    )
