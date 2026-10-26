"""Shared by the CPU tests of LoRA declared from a model's modules (``module_lora_spec``).

``ThreadGroup`` simulates a tensor-parallel group on one CPU (a thread per rank, the
collectives meeting at a barrier), so a model's own parallel layers run sharded,
unmodified. ``STACKS`` holds each enabled model's LLM node as the parity tests drive
it; ``build_stack`` / ``bind`` / ``run`` build Qwen3.5's text stack, kernels stubbed.
Import by module name (``from lora_module_helpers import ...``).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.lora import LoraReqConfig, LoraSpec
from mstar.engine.resources.lora.pool import LoraAdapterPool
from mstar.engine.resources.step import ResourceStep, Segment, StepContext
from mstar.model.qwen3_5.config import FULL_ATTENTION, LINEAR_ATTENTION, Qwen3_5Config
from mstar.model.submodule_base import ARNodeInputs, declared_step

DTYPE = torch.float64


class _Exchange:
    def __init__(self, world: int):
        self.barrier = threading.Barrier(world)
        self.slots: list = [None] * world

    def share(self, rank: int, value):
        self.slots[rank] = value
        self.barrier.wait()
        out = list(self.slots)
        self.barrier.wait()
        return out


class ThreadGroup(CommGroup):
    """One rank of a simulated group: collectives across the ranks' threads, summed in
    rank order (deterministic)."""

    def __init__(self, rank: int, world: int, exchange: _Exchange):
        super().__init__(my_global_rank=rank, my_group_rank=rank, group_members=list(range(world)))
        self._exchange = exchange

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        parts = self._exchange.share(self.rank, input_)
        out = parts[0].clone()
        for p in parts[1:]:
            out += p
        return out

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return torch.cat(self._exchange.share(self.rank, input_), dim=dim)


def run_ranks(world: int, fn: Callable[[int, CommGroup], object]) -> list:
    """``fn(rank, group)`` on every rank of a simulated group, concurrently; the results
    in rank order. A rank's exception aborts the others' collectives and is raised."""
    exchange = _Exchange(world)
    results: list = [None] * world
    errors: list = []

    def body(rank: int):
        try:
            results[rank] = fn(rank, ThreadGroup(rank, world, exchange))
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)
            exchange.barrier.abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


def rand(*shape, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=DTYPE) * scale


def write_peft(path, modules: dict[str, tuple[torch.Tensor, torch.Tensor]], alpha: float | None = None,
               embedding: tuple[str, ...] = (), prefix: str = "base_model.model.") -> str:
    """A PEFT-layout adapter: ``lora_A`` / ``lora_B`` per module (``lora_embedding_A`` /
    ``_B`` for ``embedding``), with an ``alpha`` key per module when given (else scaling 1)."""
    sd = {}
    for module, (a, b) in modules.items():
        if module in embedding:
            sd[f"{prefix}{module}.lora_embedding_A"] = a.float()
            sd[f"{prefix}{module}.lora_embedding_B"] = b.float()
        else:
            sd[f"{prefix}{module}.lora_A.weight"] = a.float()
            sd[f"{prefix}{module}.lora_B.weight"] = b.float()
        if alpha is not None:
            sd[f"{prefix}{module}.alpha"] = torch.tensor(float(alpha))
    save_file(sd, str(path))
    return str(path)


def serve(spec: LoraSpec, adapters: dict[str, str], max_rank: int = 4, **kw) -> LoraSpec:
    """``spec`` sized as a deployment's YAML would, on the CPU reference backend in fp64."""
    spec.apply_yaml_overrides(max_slots=2, max_rank=max_rank, adapters=dict(adapters), **kw)
    spec.dtype = DTYPE
    spec.backend = "torch"
    return spec


def pool(spec: LoraSpec, rank: int = 0, world: int = 1) -> LoraAdapterPool:
    return LoraAdapterPool(spec, "cpu", rank=rank, world_size=world)


@contextmanager
def compute(*pools):
    """The engine's compute scope around a planned step's preprocess and forward
    (``StepRunner.compute_scope``), on the given pools: deltas called outside one raise."""
    for p in pools:
        p.enter_compute()
    try:
        yield
    finally:
        for p in pools:
            p.exit_compute()


def _config(adapter: str | None) -> LoraReqConfig | None:
    """As the conductor opens a request: a base-model one carries no config."""
    return None if adapter is None else LoraReqConfig(adapter=adapter)


def _idle(rows) -> bool:
    """The step's ``StepContext.idle``, as the engine sets it: no row names an adapter."""
    return all(adapter is None for _, adapter, _ in rows)


def plan(pools, rows: list[tuple[str, str | None, int]]) -> None:
    """Ingest ``rows`` ((rid, adapter, span)) into each pool, then admit and plan one step
    over them, as the engine drives a step whose segments are one per request."""
    for p in pools:
        for rid, adapter, _ in rows:
            p.ingest_request(rid, _config(adapter))
        step = ResourceStep(segments=tuple(Segment(rid, "main", n) for rid, _, n in rows))
        ctx = StepContext(request_ids=[r for r, _, _ in rows], graph_walk="prefill", slot=0, capture=False,
                          idle=_idle(rows))
        assert p.admit(step, ctx).ok
        p.plan(step, ctx)


def admit_declared(lora_pool, step, rows, walk: str = "prefill") -> None:
    """Admit and plan a node's own declared ``step`` (its step under the pool's key)
    for ``rows`` ((rid, adapter, span))."""
    for rid, adapter, _ in rows:
        lora_pool.ingest_request(rid, _config(adapter))
    ctx = StepContext(request_ids=[r for r, _, _ in rows], graph_walk=walk, slot=0, capture=False,
                      idle=_idle(rows))
    lora_step = step.get(lora_pool.spec.resource_key)
    assert lora_step is not None, "the node's declare_step carries no step for the pool"
    assert lora_pool.admit(lora_step, ctx).ok
    lora_pool.plan(lora_step, ctx)


# -- adapters and merged weights --------------------------------------------------------------

def make_adapter(path, modules: dict[str, tuple[int, int]], rank: int, alpha: float | None, seed: int,
                 embedding: tuple[str, ...] = (), rename: Callable[[str], str] = lambda m: m):
    """A random PEFT adapter over ``modules`` (checkpoint module -> (out, in)); an
    ``embedding`` module's A is ``[r, vocab]`` (``in`` is the vocabulary). The file names
    each module ``rename(module)``; the returned reference ``{module: (A, B, scaling)}``
    names them as the checkpoint does. Factors are fp32-exact, as the file holds them."""
    tensors, ref = {}, {}
    scaling = 1.0 if alpha is None else alpha / rank
    for i, (module, (out, inp)) in enumerate(modules.items()):
        a = rand(rank, inp, seed=seed + 2 * i, scale=0.3).float().double()
        b = rand(out, rank, seed=seed + 2 * i + 1, scale=0.3).float().double()
        tensors[rename(module)] = (a, b)
        ref[module] = (a, b, scaling)
    renamed = tuple(rename(m) for m in embedding)
    return write_peft(path, tensors, alpha=alpha, embedding=renamed), ref


def merged(base: dict[str, torch.Tensor], ref, embedding: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
    """``base`` with the adapter folded in: ``W + scaling * B @ A`` (an embedding's
    ``[vocab, hidden]`` weight takes its transpose)."""
    out = dict(base)
    for module, (a, b, s) in ref.items():
        delta = s * (b @ a)
        out[module + ".weight"] = out[module + ".weight"] + (delta.t() if module in embedding else delta)
    return out


# -- stubs for the attention cores (they hold no adapter; both sides of a comparison run them) --

class _Rows:
    """A step's row boundaries, read by the stubbed cores when they run."""

    def __init__(self, lengths: list[int] | None = None):
        self.lengths = list(lengths or [])


class CausalAttend:
    """Causal GQA attention within each row, the cursor API the shared attention layers
    call. ``lengths``: the rows' token counts, or a ``_Rows`` read at call time."""

    label = "main"

    def __init__(self, lengths):
        self.rows = lengths if isinstance(lengths, _Rows) else _Rows(lengths)

    def bind_step(self, label):
        del label

    def set_layer_idx(self, idx):
        del idx

    def __call__(self, q, k, v):
        out, at = [], 0
        group = q.shape[1] // k.shape[1]
        for n in self.rows.lengths:
            qs = q[at:at + n].transpose(0, 1)
            ks = k[at:at + n].repeat_interleave(group, dim=1).transpose(0, 1)
            vs = v[at:at + n].repeat_interleave(group, dim=1).transpose(0, 1)
            out.append(F.scaled_dot_product_attention(qs, ks, vs, is_causal=True).transpose(0, 1))
            at += n
        return torch.cat(out)


def stub_attention(root: nn.Module, lengths: list[int]) -> None:
    """Every attention layer under ``root`` that calls the engine's attention cursor."""
    for module in root.modules():
        if hasattr(module, "_attn_key") and hasattr(module, "qkv_proj"):
            module.attend = CausalAttend(lengths)


class _DeltaRuleStub(CausalAttend):
    """Plain torch for what Qwen3.5's delta-net calls on the engine (``mix.conv``, then
    ``mix(...)``), per row: a causal depthwise conv over ``[q|k|v]``, then the gated delta
    rule. Not the kernels' numerics, but it reads every projection output, so a delta on
    the wrong columns shows."""

    def conv(self, x, weight, bias=None):
        out, at = [], 0
        for n in self.rows.lengths:
            xs = F.pad(x[at:at + n].t(), (weight.shape[1] - 1, 0))
            out.append(F.silu(F.conv1d(xs[None], weight[:, None, :], bias, groups=weight.shape[0])[0].t()))
            at += n
        return torch.cat(out)

    def __call__(self, q, k, v, a, b, A_log, dt_bias):
        rep = v.shape[1] // q.shape[1]
        q = F.normalize(q, dim=-1).repeat_interleave(rep, dim=1) * q.shape[-1] ** -0.5
        k = F.normalize(k, dim=-1).repeat_interleave(rep, dim=1)
        decay = torch.exp(-A_log.exp() * F.softplus(a + dt_bias))
        beta = torch.sigmoid(b)
        out, at = [], 0
        for n in self.rows.lengths:
            state = v.new_zeros(v.shape[1], k.shape[-1], v.shape[-1])
            for t in range(at, at + n):
                state = state * decay[t, :, None, None]
                err = v[t] - torch.einsum("hk,hkv->hv", k[t], state)
                state = state + beta[t, :, None, None] * k[t][:, :, None] * err[:, None, :]
                out.append(torch.einsum("hk,hkv->hv", q[t], state))
            at += n
        return torch.stack(out)


# -- Llama-shaped checkpoints -------------------------------------------------------------------

def llama_layers(c, prefix: str) -> dict[str, tuple[int, int]]:
    """A Llama stack's projections, checkpoint module -> ``(out, in)``, under ``prefix``."""
    q, kv, h, i = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim, \
        c.hidden_size, c.intermediate_size
    shapes = {
        "self_attn.q_proj": (q, h), "self_attn.k_proj": (kv, h), "self_attn.v_proj": (kv, h),
        "self_attn.o_proj": (h, q), "mlp.gate_proj": (i, h), "mlp.up_proj": (i, h), "mlp.down_proj": (h, i),
    }
    return {f"{prefix}layers.{n}.{p}": s for n in range(c.num_hidden_layers) for p, s in shapes.items()}


def llama_weights(c, prefix: str, head: str, qk_norm: bool = False, seed: int = 0) -> dict[str, torch.Tensor]:
    """A random Llama checkpoint in the HF layout: the stack under ``prefix``, the head
    at ``head``."""
    out, n = {}, 0

    def r(*shape, scale=0.2):
        nonlocal n
        n += 1
        return rand(*shape, seed=seed + n, scale=scale)

    out[f"{prefix}embed_tokens.weight"] = r(c.vocab_size, c.hidden_size, scale=1.0)
    out[f"{head}.weight"] = r(c.vocab_size, c.hidden_size)
    out[f"{prefix}norm.weight"] = 1 + r(c.hidden_size, scale=0.1)
    for module, shape in llama_layers(c, prefix).items():
        out[f"{module}.weight"] = r(*shape)
    for layer in range(c.num_hidden_layers):
        norms = ["input_layernorm", "post_attention_layernorm"]
        if qk_norm:
            norms += ["self_attn.q_norm", "self_attn.k_norm"]
        for norm in norms:
            dim = c.head_dim if "_norm" in norm and "self_attn" in norm else c.hidden_size
            out[f"{prefix}layers.{layer}.{norm}.weight"] = 1 + r(dim, scale=0.1)
    return out


def bind_under(module: nn.Module, prefix: str, lora_pool: LoraAdapterPool, check: bool = True) -> int:
    """Bind a standalone component as ``bind_module_lora`` binds it in its stack: every
    module the pool's map places under ``prefix`` (its path in the weights root), checked
    against the declaration unless ``check=False``. Returns how many it bound."""
    from mstar.model.components.lora.capable import make_lora_capable
    from mstar.model.components.lora.modules import _check

    spec, bound = lora_pool.spec, 0
    for path, ref in spec.module_map.bindings.items():
        if not path.startswith(prefix) or ref.target not in spec.targets:
            continue
        target = make_lora_capable(module.get_submodule(path[len(prefix):]))
        if check:
            _check(target, spec.targets[ref.target], lora_pool.targets[ref.target], path, spec.resource_key)
        target.lora_delta = lora_pool.layer_delta(ref.target, ref.layer)
        bound += 1
    return bound


# -- Qwen3.5's text stack ---------------------------------------------------------------------

HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM = 32, 48, 4, 2, 8
# 8 v-heads over 4 k-heads: the delta-net pads `a` at TP 1, 2 and 4
LIN_K_HEADS, LIN_V_HEADS, LIN_DK, LIN_DV, CONV = 4, 8, 8, 8, 4
KEY_DIM, VALUE_DIM = LIN_K_HEADS * LIN_DK, LIN_V_HEADS * LIN_DV
LAYER_TYPES = [LINEAR_ATTENTION, FULL_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION]
TEXT = "model.language_model."
PROJ_SHAPES = {  # (out, in) of each HF projection
    "self_attn.q_proj": (2 * HEADS * HEAD_DIM, HIDDEN),  # [q | gate] per head
    "self_attn.k_proj": (KV_HEADS * HEAD_DIM, HIDDEN),
    "self_attn.v_proj": (KV_HEADS * HEAD_DIM, HIDDEN),
    "self_attn.o_proj": (HIDDEN, HEADS * HEAD_DIM),
    "mlp.gate_proj": (INTER, HIDDEN),
    "mlp.up_proj": (INTER, HIDDEN),
    "mlp.down_proj": (HIDDEN, INTER),
    "linear_attn.in_proj_qkv": (2 * KEY_DIM + VALUE_DIM, HIDDEN),  # [q | k | v]
    "linear_attn.in_proj_z": (VALUE_DIM, HIDDEN),
    "linear_attn.in_proj_a": (LIN_V_HEADS, HIDDEN),
    "linear_attn.in_proj_b": (LIN_V_HEADS, HIDDEN),
    "linear_attn.out_proj": (HIDDEN, VALUE_DIM),
}


def tiny_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        num_hidden_layers=len(LAYER_TYPES), hidden_size=HIDDEN, intermediate_size=INTER,
        layer_types=list(LAYER_TYPES), num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
        linear_num_key_heads=LIN_K_HEADS, linear_num_value_heads=LIN_V_HEADS,
        linear_key_head_dim=LIN_DK, linear_value_head_dim=LIN_DV, linear_conv_kernel_dim=CONV,
        vocab_size=64, rms_norm_eps=1e-6, max_position_embeddings=4096, attn_output_gate=True,
    )


def _projs_on(layer: int) -> tuple[str, ...]:
    mixer = "self_attn." if LAYER_TYPES[layer] == FULL_ATTENTION else "linear_attn."
    return tuple(p for p in PROJ_SHAPES if p.startswith((mixer, "mlp.")))


def hf_text_weights(seed: int = 0) -> dict[str, torch.Tensor]:
    """The text tower of a released checkpoint."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape, scale=None):
        scale = scale if scale is not None else 1 / math.sqrt(shape[-1])
        return torch.randn(*shape, generator=g, dtype=DTYPE) * scale

    sd = {}
    for i in range(len(LAYER_TYPES)):
        prefix = f"{TEXT}layers.{i}."
        sd[prefix + "input_layernorm.weight"] = r(HIDDEN, scale=0.1)
        sd[prefix + "post_attention_layernorm.weight"] = r(HIDDEN, scale=0.1)
        for proj in _projs_on(i):
            sd[f"{prefix}{proj}.weight"] = r(*PROJ_SHAPES[proj])
        if LAYER_TYPES[i] == FULL_ATTENTION:
            sd[prefix + "self_attn.q_norm.weight"] = r(HEAD_DIM, scale=0.1)
            sd[prefix + "self_attn.k_norm.weight"] = r(HEAD_DIM, scale=0.1)
        else:
            sd[prefix + "linear_attn.conv1d.weight"] = r(2 * KEY_DIM + VALUE_DIM, 1, CONV, scale=0.5)
            sd[prefix + "linear_attn.A_log"] = r(LIN_V_HEADS, scale=0.5)
            sd[prefix + "linear_attn.dt_bias"] = r(LIN_V_HEADS, scale=0.5)
            sd[prefix + "linear_attn.norm.weight"] = 1 + r(LIN_DV, scale=0.1)
    sd[TEXT + "norm.weight"] = r(HIDDEN, scale=0.1)
    sd[TEXT + "embed_tokens.weight"] = r(64, HIDDEN, scale=1.0)
    sd["lm_head.weight"] = r(64, HIDDEN)
    return sd


def write_peft_adapter(directory, rank, projs, alpha=None, text_only=False, seed=1):
    """A PEFT adapter over ``projs`` on every layer that has them
    (``adapter_model.safetensors``, plus ``adapter_config.json`` when ``alpha`` is
    given). Returns ``(path, {hf module path: (a, b, scaling)})``."""
    import json

    directory.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    sd, ref = {}, {}
    for i in range(len(LAYER_TYPES)):
        for proj in (p for p in projs if p in _projs_on(i)):
            out, inp = PROJ_SHAPES[proj]
            module = (f"model.layers.{i}." if text_only else f"{TEXT}layers.{i}.") + proj
            a = (torch.randn(rank, inp, generator=g) / math.sqrt(inp)).float()
            b = (torch.randn(out, rank, generator=g) / math.sqrt(rank)).float()
            sd[f"base_model.model.{module}.lora_A.weight"] = a
            sd[f"base_model.model.{module}.lora_B.weight"] = b
            ref[f"{TEXT}layers.{i}.{proj}"] = (a.to(DTYPE), b.to(DTYPE), (alpha or rank) / rank)
    path = directory / "adapter_model.safetensors"
    save_file(sd, str(path))
    if alpha is not None:
        (directory / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": alpha}))
    return str(path), ref


def _declared(model) -> LoraSpec:
    """The model's LoRA spec, as it declares it beside its other resources."""
    return next(s for s in model.get_node_resources() if isinstance(s, LoraSpec))


def lora_spec(paths, **kw) -> LoraSpec:
    from mstar.model.qwen3_5.lora import qwen3_5_lora_spec

    return serve(qwen3_5_lora_spec(tiny_config()), paths, **kw)


class _GatedNorm(nn.Module):
    """``RMSNormGated``'s math in the input's dtype (it holds no adapter)."""

    def __init__(self, dim: int = LIN_DV, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=DTYPE))

    def forward(self, x, gate):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight.to(x.dtype) * x * F.silu(gate)


def build_stack(weights: dict[str, torch.Tensor], comm_group: CommGroup | None = None,
                config: Qwen3_5Config | None = None):
    """The real Qwen3.5 text stack (``comm_group``: one rank's shard of it), loaded by
    the model's own rules. Sets the process-wide default dtype while it builds."""
    from mstar.model.loader.base import load_weights_into
    from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
    from mstar.model.qwen3_5.weight_loader import _STACKED_PARAMS, qwen3_5_name_remapper

    torch.set_default_dtype(DTYPE)
    try:
        model = Qwen3_5ForCausalLM(config or tiny_config(), comm_group)
    finally:
        torch.set_default_dtype(torch.float32)
    for i, kind in enumerate(LAYER_TYPES):
        if kind == LINEAR_ATTENTION:
            model.model.layers[i].self_attn.norm = _GatedNorm()
    loaded = load_weights_into(
        model, weights.items(), stacked_params=_STACKED_PARAMS, name_remapper=qwen3_5_name_remapper,
    )
    assert loaded == {n for n, _ in model.named_parameters()}
    return model.requires_grad_(False).eval()


def _stub_cores(model, rows: _Rows) -> None:
    from mstar.model.qwen3_5.components.language_model import Qwen3_5Attention

    for layer in model.model.layers:
        if isinstance(layer.self_attn, Qwen3_5Attention):
            layer.self_attn.attend = CausalAttend(rows)
        else:
            layer.self_attn.mix = _DeltaRuleStub(rows)


def bind(model, rows: _Rows, resources: dict):
    """The LLM node over ``model``, ``resources`` bound and the kernels stubbed."""
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    llm = LLMSubmodule(model, model.config)
    llm.bind_node_resources(resources)
    _stub_cores(model, rows)
    return llm


def run(model, rows: _Rows, embeds_per_row: list[torch.Tensor]) -> torch.Tensor:
    rows.lengths = [e.shape[0] for e in embeds_per_row]
    with torch.no_grad():
        return model.model(torch.cat(embeds_per_row), label="main")


# -- every enabled model's LLM node, as the parity tests drive it -----------------------------

@dataclass
class Stack:
    config: Callable[[], object]
    lora_spec: Callable[[object], LoraSpec]                    # the model's declaration
    build: Callable[[object, CommGroup, dict], nn.Module]      # the node, by its own loader
    weights: Callable[[object], dict[str, torch.Tensor]]       # checkpoint names
    a1: Callable[[object], dict[str, tuple[int, int]]]         # every adapted module
    a2: Callable[[object], dict[str, tuple[int, int]]]         # a subset
    # the node's forward, per request: (its hidden rows, its head rows)
    run: Callable[[nn.Module, list[torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]
    declare: Callable[[nn.Module, list[str], list[torch.Tensor]], object]
    inputs: Callable[[object, int, int], torch.Tensor]         # (config, tokens, seed) -> a row
    # (the model at its released config, (config, TP group) -> the root its LoRA binds in)
    released: Callable[[], tuple[object, Callable]] | None = None
    worlds: tuple[int, ...] = (1, 2, 4)
    embedding: tuple[str, ...] = ()
    # the layouts the adapters' files name their modules in
    a1_rename: Callable[[str], str] = lambda m: m
    a2_rename: Callable[[str], str] = lambda m: m


def model_object(cls, config, **attrs):
    """A model class without its checkpoint download: its declarations read the config."""
    m = object.__new__(cls)
    m.config = config
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _last(hidden: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    return hidden[torch.tensor(lengths).cumsum(0) - 1]


def _per_row(hidden, head, lengths):
    starts = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    return [(hidden[starts[i]:starts[i + 1]], head[i:i + 1]) for i in range(len(lengths))]


def _tokens(vocab):
    return lambda c, n, seed: torch.randint(0, vocab(c), (n,), generator=torch.Generator().manual_seed(seed))


def _declare(walk):
    return lambda sub, rids, rows: declared_step(
        sub, walk, rids, [ARNodeInputs(input_ids=i, input_seq_len=len(i)) for i in rows],
    )


def _ids_run(stub_root, embed, stack, head):
    """A node run as embed -> stack -> head on each row's last token."""
    def run(sub, rows):
        lengths = [len(r) for r in rows]
        stub_attention(stub_root(sub), lengths)
        hidden = stack(sub, embed(sub, torch.cat(rows)))
        return _per_row(hidden, head(sub, _last(hidden, lengths)), lengths)
    return run


def _subset(modules, *ends):
    return {k: v for k, v in modules.items() if k.endswith(ends)}


def _loaded(model, weights, **rules):
    from mstar.model.loader import load_hf_weights

    assert load_hf_weights(model, weights.items(), **rules) == {n for n, _ in model.named_parameters()}
    return model.requires_grad_(False).eval()


def tiny():
    """Orpheus at test size."""
    from mstar.model.orpheus.config import OrpheusModelConfig

    return OrpheusModelConfig(
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, hidden_size=16,
        head_dim=4, intermediate_size=32, vocab_size=64, tie_word_embeddings=False, pad_token_id=0,
    )


def _orpheus():
    from mstar.model.orpheus.components.language_model import OrpheusForCausalLM
    from mstar.model.orpheus.config import OrpheusModelConfig
    from mstar.model.orpheus.orpheus_model import OrpheusModel
    from mstar.model.orpheus.submodules import OrpheusLLMSubmodule

    def build(c, group, weights):
        lm = OrpheusForCausalLM(c, comm_group=group).to(DTYPE)
        lm.load_weights(weights.items())
        return OrpheusLLMSubmodule(lm.requires_grad_(False).eval(), c)

    return Stack(
        config=tiny, lora_spec=lambda c: _declared(model_object(OrpheusModel, c)), build=build,
        weights=lambda c: llama_weights(c, "model.", "lm_head"),
        a1=lambda c: {**llama_layers(c, "model."), "lm_head": (c.vocab_size, c.hidden_size),
                      "model.embed_tokens": (c.hidden_size, c.vocab_size)},
        a2=lambda c: _subset(llama_layers(c, "model."), "0.self_attn.q_proj", "0.self_attn.v_proj",
                             "1.mlp.down_proj"),
        run=_ids_run(lambda s: s.language_model, lambda s, ids: s.embed_tokens(ids),
                     lambda s, x: s.language_model(x, label="main"), lambda s, x: s.lm_head(x)),
        declare=_declare("prefill"), inputs=_tokens(lambda c: c.vocab_size),
        released=lambda: (model_object(OrpheusModel, OrpheusModelConfig()),
                          lambda c, g: OrpheusForCausalLM(c, comm_group=g)),
        embedding=("model.embed_tokens",),
        a2_rename=lambda m: m.removeprefix("model."),  # the bare LlamaModel's naming
    )


def _higgs():
    from mstar.model.higgs_audio.components.llm import HiggsAudioLLM, higgs_llm_remap
    from mstar.model.higgs_audio.config import HiggsAudioModelConfig
    from mstar.model.higgs_audio.higgs_audio_model import HiggsAudioModel
    from mstar.model.higgs_audio.submodules import HiggsAudioLLMSubmodule
    from mstar.model.loader import LLAMA_STACKED_PARAMS

    head = "audio_decoder_proj.text_lm_head"
    return Stack(
        config=lambda: HiggsAudioModelConfig(
            hidden_size=16, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, intermediate_size=32, vocab_size=64,
        ),
        lora_spec=lambda c: _declared(model_object(HiggsAudioModel, c)),
        build=lambda c, group, w: HiggsAudioLLMSubmodule(_loaded(
            HiggsAudioLLM(c, comm_group=group).to(DTYPE), w, stacked_params=LLAMA_STACKED_PARAMS,
            name_remapper=higgs_llm_remap), c),
        weights=lambda c: llama_weights(c, "", head, qk_norm=True),
        a1=lambda c: {**llama_layers(c, ""), head: (c.vocab_size, c.hidden_size)},
        a2=lambda c: _subset(llama_layers(c, ""), "0.self_attn.k_proj", "1.mlp.up_proj", "1.self_attn.o_proj"),
        run=_ids_run(lambda s: s.model, lambda s, ids: s.model.embed_tokens(ids),
                     lambda s, x: s.model(input_embeds=x, label="main"), lambda s, x: s.model.lm_head(x)),
        declare=_declare("prefill_text"), inputs=_tokens(lambda c: c.vocab_size),
        released=lambda: (model_object(HiggsAudioModel, HiggsAudioModelConfig()),
                          lambda c, g: HiggsAudioLLM(c, comm_group=g)),
        a2_rename=lambda m: f"model.{m}",  # a Qwen3ForCausalLM-shaped adapter
    )


def _omni(dense: bool):
    """``dense``: every layer a dense MLP, so TP runs on a CPU (the sharded MoE dispatch
    is a Triton kernel); else layer 0 is MoE and takes no delta."""
    from mstar.model.qwen3_omni.components.thinker import Qwen3OmniThinkerModel
    from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig, ThinkerTextConfig
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
    from mstar.model.qwen3_omni.submodules import ThinkerSubmodule

    moe_layers = () if dense else (0,)

    def weights(c):
        tc = c.thinker_text
        out = llama_weights(tc, "model.", "lm_head", qk_norm=True)
        for layer in moe_layers:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                del out[f"model.layers.{layer}.mlp.{proj}.weight"]
            out[f"model.layers.{layer}.mlp.gate.weight"] = rand(tc.num_experts, tc.hidden_size, seed=900)
            for e in range(tc.num_experts):
                for proj, shape in (("gate_proj", (8, 16)), ("up_proj", (8, 16)), ("down_proj", (16, 8))):
                    out[f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"] = rand(*shape, seed=901 + e, scale=0.2)
        return out

    def adapted(c):
        return {m: s for m, s in llama_layers(c.thinker_text, "model.").items()
                if "self_attn" in m or int(m.split(".")[2]) not in moe_layers}

    # the released Qwen3-Omni-30B-A3B Thinker's text config
    full = Qwen3OmniModelConfig(thinker_text=ThinkerTextConfig(
        vocab_size=152064, hidden_size=2048, intermediate_size=768, num_hidden_layers=48,
        num_attention_heads=32, num_key_value_heads=4, head_dim=128, num_experts=128,
        num_experts_per_tok=8, moe_intermediate_size=768,
    ))
    return Stack(
        config=lambda: Qwen3OmniModelConfig(thinker_text=ThinkerTextConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4, num_experts=4,
            num_experts_per_tok=2, moe_intermediate_size=8,
            mlp_only_layers=[i for i in range(2) if i not in moe_layers],
        )),
        lora_spec=lambda c: _declared(model_object(Qwen3OmniModel, c)),
        build=lambda c, group, w: ThinkerSubmodule(_loaded(
            Qwen3OmniThinkerModel(c, comm_group=group).to(DTYPE), w,
            stacked_params=model_object(Qwen3OmniModel, c)._get_thinker_stacked_params(),
            name_remapper=Qwen3OmniModel._thinker_remap), c),
        weights=weights,
        a1=lambda c: {**adapted(c), "lm_head": (c.thinker_text.vocab_size, c.thinker_text.hidden_size)},
        a2=lambda c: _subset(adapted(c), "v_proj", "o_proj", "up_proj"),
        run=_ids_run(lambda s: s.model, lambda s, ids: s.model.model.embed_tokens(ids),
                     lambda s, x: s.model(input_embeds=x, label="main")[0], lambda s, x: s.model.lm_head(x)),
        declare=_declare("prefill_text"), inputs=_tokens(lambda c: c.thinker_text.vocab_size),
        released=None if dense else lambda: (model_object(Qwen3OmniModel, full),
                                             lambda c, g: Qwen3OmniThinkerModel(c, comm_group=g)),
        worlds=(1, 2, 4) if dense else (1,),
        a1_rename=lambda m: f"thinker.{m}",  # the full model's checkpoint naming
    )


def _qwen3_tts():
    from mstar.model.loader import LLAMA_STACKED_PARAMS
    from mstar.model.qwen3_tts.components.talker import Qwen3TTSCodePredictor, Qwen3TTSTalkerModel
    from mstar.model.qwen3_tts.config import (
        Qwen3TTSCodePredictorConfig,
        Qwen3TTSModelConfig,
        Qwen3TTSTalkerConfig,
    )
    from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel
    from mstar.model.qwen3_tts.submodules import TalkerSubmodule

    dims = dict(num_attention_heads=4, num_key_value_heads=2, hidden_size=16, intermediate_size=32,
                head_dim=4, vocab_size=64)

    def weights(c):
        tc = c.talker
        out = llama_weights(tc, "model.", "codec_head", qk_norm=True)
        out["model.codec_embedding.weight"] = out.pop("model.embed_tokens.weight")
        out["model.text_embedding.weight"] = rand(tc.text_vocab_size, tc.text_hidden_size, seed=800)
        for i, (n, shape) in enumerate((("linear_fc1", (8, 8)), ("linear_fc2", (16, 8)))):
            out[f"text_projection.{n}.weight"] = rand(*shape, seed=801 + i)
            out[f"text_projection.{n}.bias"] = rand(shape[0], seed=811 + i)
        return out

    return Stack(
        config=lambda: Qwen3TTSModelConfig(talker=Qwen3TTSTalkerConfig(
            num_hidden_layers=2, text_hidden_size=8, text_vocab_size=32, **dims,
            code_predictor=Qwen3TTSCodePredictorConfig(num_hidden_layers=1, **dims),
        )),
        lora_spec=lambda c: _declared(model_object(Qwen3TTSModel, c)),
        # the code predictor sits beside the Talker, with relative paths of the same shape
        build=lambda c, group, w: TalkerSubmodule(
            _loaded(Qwen3TTSTalkerModel(c, comm_group=group).to(DTYPE), w, stacked_params=LLAMA_STACKED_PARAMS),
            Qwen3TTSCodePredictor(c).to(DTYPE), c),
        weights=weights,
        a1=lambda c: {**llama_layers(c.talker, "model."), "codec_head": (c.talker.vocab_size, c.talker.hidden_size)},
        a2=lambda c: _subset(llama_layers(c.talker, "model."), "q_proj", "gate_proj"),
        run=_ids_run(lambda s: s.model, lambda s, ids: s.model.model.codec_embedding(ids),
                     lambda s, x: s.model(x, label="main"), lambda s, x: s.model.codec_head(x)),
        declare=_declare("talker_prefill"), inputs=_tokens(lambda c: c.talker.vocab_size),
        released=lambda: (model_object(Qwen3TTSModel, Qwen3TTSModelConfig()),
                          lambda c, g: Qwen3TTSTalkerModel(c, comm_group=g)),
        a1_rename=lambda m: f"talker.{m}",
    )


def _t3_config():
    from mstar.model.chatterbox.config import ChatterboxConfig, T3BackboneConfig, T3Config

    return ChatterboxConfig(t3=T3Config(
        backbone=T3BackboneConfig(
            hidden_size=16, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, intermediate_size=32,
        ),
        text_vocab_size=32, speech_vocab_size=40, speaker_embed_size=8, speech_cond_prompt_len=4,
        start_text_token=1, stop_text_token=0, start_speech_token=38, stop_speech_token=39,
        perceiver_num_queries=4, perceiver_num_heads=2, max_text_tokens=8, max_speech_tokens=8,
    ))


def _chatterbox_t3():
    """Under classifier-free guidance T3 runs every request twice, cond rows then uncond
    rows, and each copy takes the request's adapter. A row is ``[cond, uncond]`` embeddings."""
    from mstar.model.chatterbox.chatterbox_model import ChatterboxModel
    from mstar.model.chatterbox.components.t3 import T3Model
    from mstar.model.chatterbox.config import ChatterboxConfig
    from mstar.model.chatterbox.submodules import T3Submodule

    def weights(c):
        """The reference layout (``tfmr.*``, separate q/k/v and gate/up), from a model's
        own parameters so it covers every one."""
        bb = c.t3.backbone
        q, kv = bb.num_attention_heads * bb.head_dim, bb.num_key_value_heads * bb.head_dim
        out = {}
        for i, (name, p) in enumerate(T3Model(c.t3, comm_group=CommGroup.trivial()).named_parameters()):
            value = (1 + rand(*p.shape, seed=i, scale=0.1)) if p.dim() == 1 and "norm" in name \
                else rand(*p.shape, seed=i, scale=0.2)
            name = "tfmr." + name.removeprefix("backbone.") if name.startswith("backbone.") else name
            if ".qkv_proj." in name:
                for shard, part in zip("qkv", value.split([q, kv, kv]), strict=True):
                    out[name.replace(".qkv_proj.", f".{shard}_proj.")] = part
            elif ".gate_up_proj." in name:
                for shard, part in zip(("gate", "up"), value.chunk(2), strict=True):
                    out[name.replace(".gate_up_proj.", f".{shard}_proj.")] = part
            else:
                out[name] = value
        return out

    def build(c, group, w):
        model = T3Model(c.t3, comm_group=group).to(DTYPE)
        model.load_weights(w.items())
        return T3Submodule(model.requires_grad_(False).eval(), c, builtin_voice=None)

    def run(sub, rows):
        k, lengths = len(rows), [r.shape[1] for r in rows] * 2
        stub_attention(sub.model, lengths)
        hidden = sub.model.hidden(torch.cat([r[0] for r in rows] + [r[1] for r in rows]), label="cfg")
        per = _per_row(hidden, sub.model.logits(_last(hidden, lengths)), lengths)
        return [tuple(torch.cat(p) for p in zip(per[i], per[i + k], strict=True)) for i in range(k)]

    return Stack(
        config=_t3_config, lora_spec=lambda c: _declared(model_object(ChatterboxModel, c)),
        build=build, weights=weights,
        a1=lambda c: {**llama_layers(c.t3.backbone, "tfmr."), "speech_head": (40, 16)},
        a2=lambda c: _subset(llama_layers(c.t3.backbone, "tfmr."), "v_proj", "down_proj"),
        run=run, inputs=lambda c, n, seed: rand(2, n, 16, seed=40 + seed),
        declare=lambda sub, rids, rows: declared_step(sub, "prefill", rids, [
            ARNodeInputs(input_embeds=r[0], input_seq_len=r.shape[1], resource_step_info=True) for r in rows
        ]),
        released=lambda: (model_object(ChatterboxModel, ChatterboxConfig()),
                          lambda c, g: T3Model(c.t3, comm_group=g)),
    )


def _qwen3_5():
    """The gated q (the output gate inside each query head), the delta-net's
    ``[q|k|v|z|a|pad|b]`` in-projection (``a`` padded per rank), the LM head (one row
    per request) and the token embedding, whose delta the node adds in the step. Its
    released sizes, with the vision tower's spec, are the models test's own."""
    from mstar.model.qwen3_5.lora import qwen3_5_lora_spec
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    def a1(c):
        every = {f"{TEXT}layers.{i}.{p}": PROJ_SHAPES[p] for i in range(len(LAYER_TYPES)) for p in _projs_on(i)}
        return {**every, "lm_head": (c.vocab_size, HIDDEN), f"{TEXT}embed_tokens": (HIDDEN, c.vocab_size)}

    def run_(sub, rows):
        lengths, lm = [len(r) for r in rows], sub.model
        _stub_cores(lm, _Rows(lengths))
        hidden = lm.model(lm.model.embed(torch.cat(rows)), label="main")
        return _per_row(hidden, lm.lm_head(_last(hidden, lengths)), lengths)

    return Stack(
        config=tiny_config, lora_spec=qwen3_5_lora_spec,
        build=lambda c, group, w: LLMSubmodule(build_stack(w, group, c), c),
        weights=lambda c: hf_text_weights(), a1=a1,
        # leaves `in_proj_qkv` and `in_proj_a` out: slices of one fused target on and off it
        a2=lambda c: _subset(a1(c), "q_proj", "v_proj", "down_proj", "in_proj_z", "in_proj_b",
                             "linear_attn.out_proj"),
        run=run_, declare=_declare("prefill_text"), inputs=_tokens(lambda c: c.vocab_size),
        embedding=(f"{TEXT}embed_tokens",),
        a2_rename=lambda m: m.replace(TEXT, "model."),  # the text-only causal LM's naming
    )


STACKS = {
    "orpheus": _orpheus,
    "higgs_audio": _higgs,
    "qwen3_omni_thinker": lambda: _omni(dense=False),
    "qwen3_omni_thinker_dense": lambda: _omni(dense=True),
    "qwen3_tts_talker": _qwen3_tts,
    "chatterbox_t3": _chatterbox_t3,
    "qwen3_5": _qwen3_5,
}
