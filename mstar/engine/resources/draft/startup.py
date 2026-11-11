"""Startup checks for a node that drafts and verifies tokens, and the choice of
its mode.

The same checks run in `validate_deployment`, from config alone before any
worker spawns, so every problem is listed at once and the ranks agree, and again
at engine load against the built submodules. Nothing waits for a verify step to
fail.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from mstar.engine.resources.draft.config import DraftMode
from mstar.engine.resources.draft.drafter import ChunkBoundary, VerifyNode
from mstar.engine.resources.spec import NodeResourceSpec, ResourceCapabilities

if TYPE_CHECKING:
    from mstar.engine.resources.draft.node import DraftLag

logger = logging.getLogger(__name__)


def shape_problems(decl: VerifyNode, token_stop, specs: Mapping[str, NodeResourceSpec]) -> list[str]:
    """What about the node's shape a verify walk can't run on: each step must
    sample one token id per position through the engine's sampler and stop
    through a `TokenStop` without ``withhold``."""
    from mstar.engine.resources.sampler.config import SamplerSpec

    where = f"node {decl.node!r} (walk {decl.walk!r})"
    problems = []
    if token_stop is None:
        return [f"{where}: speculative decoding needs a TokenStop to cut a verify step's tokens at"]
    if token_stop.withhold:
        problems.append(f"{where}: its TokenStop withholds outputs, which a run of tokens a row can't")
    if not token_stop.stops_on(decl.walk) or decl.walk in token_stop.prefill_walks:
        problems.append(f"{where}: its TokenStop doesn't stop the decode loop on this walk")
    loopback = decl.device_loopback
    if loopback is None or loopback.walk != decl.walk or loopback.sampler != token_stop.sampler:
        problems.append(
            f"{where}: a verify walk declares DeviceLoopback on itself, with the TokenStop's sampler: "
            "its next step reads each row's token off the sampler's loop-back slot"
        )
    sampler = specs.get(token_stop.sampler)
    if not isinstance(sampler, SamplerSpec) or decl.node not in sampler.nodes:
        problems.append(f"{where}: no engine sampler {token_stop.sampler!r} to verify with")
    elif not sampler.verifies:
        # ingest refuses the options verify can't honour only on a sampler that says it verifies
        problems.append(
            f"{where}: sampler {token_stop.sampler!r} doesn't set SamplerSpec.verifies, which drafter_resources "
            "sets on the one sampler serving the node"
        )
    needs = decl.needs
    if needs.carry is not None and isinstance(sampler, SamplerSpec):
        from mstar.engine.resources.draft.drafter import DRAFT_CARRY

        state = (sampler.loopback_state or {}).get(DRAFT_CARRY)
        if state is None or (state.width, state.dtype) != (needs.carry.width, needs.carry.dtype):
            have = None if state is None else (state.width, state.dtype)
            problems.append(
                f"{where}: the drafter carries {needs.carry.width} values of {needs.carry.dtype} a request, which "
                f"the sampler keeps as loop-back state {DRAFT_CARRY!r}; it declares {have}"
            )
    if needs.kv_overhang:
        problems.append(f"{where}: the drafter writes {needs.kv_overhang} KV slots past the verify block")
    missing = decl.input_kinds - needs.input_kinds
    if missing:
        problems.append(f"{where}: the drafter takes no {sorted(missing)} input, which the model accepts")
    return problems


def submodule_problems(decl: VerifyNode, submodule) -> list[str]:
    """What the built submodule says of its drafter, beside the config's
    checks: it has one, the `Drafter` protocol it is, and one that resolves
    resources (``bind_resources``) is an ``nn.Module``, the only kind
    `bind_node_resources` reaches."""
    from torch import nn

    from mstar.engine.resources.draft.drafter import Drafter

    where = f"node {decl.node!r}"
    drafter = getattr(submodule, "drafter", None)
    if drafter is None:
        return [f"{where}: a verifying node's submodule must set `drafter`"]
    problems = []
    # by class, not structure: a drafter with the protocol's names but not its signatures (an old observe)
    # would pass a runtime_checkable isinstance and fail at its first verify trace
    if Drafter not in type(drafter).__mro__:
        problems.append(f"{where}: its drafter, a {type(drafter).__name__}, is not a Drafter: subclass it")
    if hasattr(drafter, "bind_resources") and not isinstance(drafter, nn.Module):
        problems.append(
            f"{where}: its drafter defines bind_resources but is not an nn.Module, so bind_node_resources "
            "never reaches it"
        )
    return problems


def geometry_problems(decl: VerifyNode, specs: Mapping[str, NodeResourceSpec]) -> list[str]:
    """Whether the drafter fits its target: the taps it reads are the node's
    `TapSpec` and inside the target's layers, the layer set it joins the
    target's KV pool with is there (`drafter_resources`; a set of another
    geometry has an attention binding that plans with it), its attention's query heads are the
    ones the pool's attention plans, its RoPE base is the node's positions',
    and the width it reads, its vocabulary and its dtype are the target's. A
    mismatched drafter would load and draft garbage: correct output, no speedup."""
    from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
    from mstar.engine.resources.position.config import PositionSpec
    from mstar.engine.resources.taps.config import TapSpec

    where = f"node {decl.node!r}"
    needs, target, drafter = decl.needs, decl.target, decl.needs.geometry
    on_node = [spec for spec in specs.values() if decl.node in spec.nodes]
    taps = [spec for spec in on_node if isinstance(spec, TapSpec)]
    problems = []
    if needs.taps:
        own = specs.get(needs.tap_key) if needs.tap_key is not None else None
        if not isinstance(own, TapSpec) or decl.node not in own.nodes:
            problems.append(f"{where}: the drafter reads target layers {tuple(needs.taps)} from TapSpec "
                            f"{needs.tap_key!r} (DrafterNeeds.tap_key), which the node doesn't declare")
        elif tuple(own.layers) != tuple(needs.taps):
            problems.append(f"{where}: the drafter reads target layers {tuple(needs.taps)}; TapSpec "
                            f"{needs.tap_key!r} declares {tuple(own.layers)}")
        others = sorted(spec.resource_key for spec in taps if spec is not own)
        if others:
            # a captured step would read another, never written, as zeros or stale rows
            problems.append(f"{where}: declares TapSpecs {others} besides the drafter's; run_layers writes one")
    elif taps:
        problems.append(f"{where}: declares taps its drafter doesn't read")
    if drafter is None and not needs.taps and needs.kv is None:
        return problems
    if target is None:
        return problems + [f"{where}: the drafter reads the target's layers, but the model declares no target "
                           "geometry (VerifyNode.target) to check it against"]
    if needs.taps and max(needs.taps) > target.num_layers:
        problems.append(f"{where}: the drafter reads target layers {tuple(needs.taps)}; the target has "
                        f"{target.num_layers} (layer {target.num_layers} is the stream after the last)")
    for spec in taps:
        if spec.hidden_size != target.hidden_size:
            problems.append(f"{where}: TapSpec {spec.resource_key!r} is {spec.hidden_size} wide; the target's "
                            f"residual stream is {target.hidden_size}")
    if needs.kv is not None:
        kv = needs.kv
        pool = specs.get(kv.kv_cache)
        if not (isinstance(pool, KVSpec) and isinstance(pool.config, PagedKVConfig) and decl.node in pool.nodes):
            problems.append(f"{where}: the drafter's layers join KV {kv.kv_cache!r}, which is not a paged pool "
                            "of the node")
        else:
            from mstar.engine.resources.attn.config import AttentionSpec

            # the attention over the pool that plans with the drafter's set, if one does
            binding = next((spec for spec in on_node if isinstance(spec, AttentionSpec)
                            and spec.config.kv_cache == kv.kv_cache and spec.config.layer_set == kv.name), None)
            slot = pool.config.layer_sets.get(kv.name)
            if slot is not None and slot.first + slot.num_layers > pool.config.total_layers:
                problems.append(f"{where}: layer set {kv.name!r} holds layers {slot.first}.."
                                f"{slot.first + slot.num_layers - 1} of KV {kv.kv_cache!r}, which has "
                                f"{pool.config.total_layers}")
            elif slot is not None and pool.config.layer_set_storage(kv.name) and binding is None:
                # an attention over the pool's own geometry would read its storage as the pool's
                problems.append(
                    f"{where}: the drafter's layer set {kv.name!r} has a KV geometry of its own "
                    f"({kv.num_kv_heads} KV heads of {kv.head_dim}; the pool's are {pool.config.total_kv_heads} of "
                    f"{pool.config.head_dim}), and no attention over KV {kv.kv_cache!r} plans with it: declare "
                    f"one with AttentionConfig(layer_set={kv.name!r})"
                )
            elif slot is None or slot.num_layers != kv.num_layers:
                have = {name: held.num_layers for name, held in pool.config.layer_sets.items()}
                problems.append(f"{where}: the drafter joins {kv.num_layers} layers to KV {kv.kv_cache!r} as layer "
                                f"set {kv.name!r}; the pool's layer sets hold {have}")
            # the checkpoint's counts on both sides: a built pool holds one rank's slice of its heads
            planned = pool.config.total_qo_heads
            if binding is not None and binding.config.num_qo_heads is not None:
                planned = binding.config.num_qo_heads
            if drafter is not None and drafter.num_qo_heads not in (None, planned):
                problems.append(
                    f"{where}: the drafter's attention has {drafter.num_qo_heads} query heads; the attention over "
                    f"KV {kv.kv_cache!r} plans {planned}. A drafter with query heads of its own attends through "
                    f"a binding that declares them: AttentionConfig(layer_set={kv.name!r}, num_qo_heads=...)"
                )
    if drafter is not None:
        if drafter.rope_theta is not None:
            pool_key = None if needs.kv is None else needs.kv.kv_cache
            for spec in on_node:
                # it rotates through the node's positions over the pool it joins
                if isinstance(spec, PositionSpec) and spec.config.kv_cache == pool_key \
                        and spec.config.rope_theta != drafter.rope_theta:
                    problems.append(f"{where}: the drafter's rope_theta is {drafter.rope_theta}; positions "
                                    f"{spec.resource_key!r} rotate with {spec.config.rope_theta}")
        # the residual width matters to a drafter that reads the target's (taps, its last hidden state);
        # a separate draft model reads only tokens
        reads_width = bool(needs.taps) or needs.target_hidden
        for field, what in (
            ("hidden_size", "residual width"), ("vocab_size", "vocabulary"), ("dtype", "dtype"),
        ):
            if field == "hidden_size" and not reads_width:
                continue
            mine, theirs = getattr(drafter, field), getattr(target, field)
            if mine != theirs:
                problems.append(f"{where}: the drafter's {what} is {mine}; the target's is {theirs}")
    return problems


def node_capabilities(decl: VerifyNode, specs: Mapping[str, NodeResourceSpec]) -> dict[str, "ResourceCapabilities"]:
    """What each resource of the verifying node says it can do, from config alone
    (`NodeResourceSpec.resource_capabilities`): for `validate_deployment` and every
    rank's startup alike."""
    from mstar.engine.resources.spec import ResourceCapabilitiesCtx

    node_specs = {key: spec for key, spec in specs.items() if decl.node in spec.nodes}
    ctx = ResourceCapabilitiesCtx(decl.node, node_specs)
    return {key: spec.resource_capabilities(ctx) for key, spec in node_specs.items()}


def draft_lag(decl: VerifyNode, specs: Mapping[str, NodeResourceSpec]):
    """Where a lagged drafter's lag goes on the node's verify steps (`drafter_step`), and the problems that
    leave it nowhere to go: ``(DraftLag | None, problems)``. A drafter that doesn't lag gets ``(None, [])``."""
    from mstar.engine.resources.attn.config import AttentionSpec
    from mstar.engine.resources.draft.node import DraftLag
    from mstar.engine.resources.position.config import PositionSpec

    needs, where = decl.needs, f"node {decl.node!r}"
    lag = needs.lag
    if lag is None:
        return None, []
    if lag.rows != 1:
        return None, [f"{where}: the drafter lags {lag.rows} rows; a draft model lags its target by one"]
    if sum(lag.passes) != decl.config.block or min(lag.passes, default=0) < 1:
        return None, [f"{where}: the drafter's passes {lag.passes} don't split the verify block of "
                      f"{decl.config.block} rows"]
    if needs.kv is None:
        return None, [f"{where}: the drafter lags the target but joins no layers to its KV pool"]
    pool, name = needs.kv.kv_cache, needs.kv.name
    on_node = [spec for spec in specs.values() if decl.node in spec.nodes]
    bindings = [spec.resource_key for spec in on_node if isinstance(spec, AttentionSpec)
                and spec.config.kv_cache == pool and spec.config.layer_set == name]
    positions = [spec.resource_key for spec in on_node
                 if isinstance(spec, PositionSpec) and spec.config.kv_cache == pool]
    problems = []
    if len(bindings) != 1:
        problems.append(f"{where}: the lagged drafter's attention plans its own extent: declare one attention over "
                        f"KV {pool!r} with AttentionConfig(layer_set={name!r}), not {len(bindings)}")
    if len(positions) != 1:
        problems.append(f"{where}: the lagged drafter rotates through the positions over KV {pool!r}; the node "
                        f"declares {len(positions)}")
    if problems:
        return None, problems
    return DraftLag(decl.walk, lag, pool, name, positions[0], bindings[0]), []


def resource_problems(decl: VerifyNode, specs: Mapping[str, NodeResourceSpec]) -> list[str]:
    """Every resource of the node that a verify walk can't use, with why."""
    return [
        f"node {decl.node!r}: resource {key!r}: {caps.verify_refusal}"
        for key, caps in node_capabilities(decl, specs).items() if caps.verify_refusal is not None
    ]


def resolve_mode(
    decl: VerifyNode, specs: Mapping[str, NodeResourceSpec], async_scheduling: bool, why_not_async: str = "",
    resident_planning: bool = True, blockers: Iterable[str] = (),
) -> tuple[DraftMode, str]:
    """The node's mode and why. Asynchronous needs async scheduling on the
    node, resident planning, and every resource correctable on the device;
    ``mode: auto`` falls back to synchronous, and an explicit ``asynchronous``
    that can't be met raises ``ValueError``. Deterministic over its arguments:
    config alone in `validate_deployment`, and in the engine one rank's view,
    which `Engine._agreed_mode` agrees across the lockstep instance."""
    wanted = decl.config.mode
    if wanted == DraftMode.SYNCHRONOUS:
        return DraftMode.SYNCHRONOUS, "configured"
    blockers = list(blockers)
    if not async_scheduling:
        blockers.append(why_not_async or "async scheduling is off for the node")
    if not resident_planning:
        blockers.append("resident planning is off (MSTAR_RESIDENT_PLANNING)")
    lag, _ = draft_lag(decl, specs)
    for key, caps in node_capabilities(decl, specs).items():
        refusal = caps.async_verify_refusal
        if refusal is not None and lag is not None and key == lag.attn_key:
            # config's refusal (the flashinfer pin) only: every pass of a lagged drafter has an extent
            # of its own, which only FA3 reads on the device
            refusal = ("a lagged drafter's passes are planned on the host on FlashInfer, and asynchronous "
                       "verify reads each pass's length on the device: verify_backend fa3")
        if refusal is not None:
            blockers.append(f"resource {key!r}: {refusal}")
    if not blockers:
        return DraftMode.ASYNCHRONOUS, "every resource corrects itself on the device"
    reason = "; ".join(blockers)
    if wanted == DraftMode.ASYNCHRONOUS:
        raise ValueError(f"node {decl.node!r}: asynchronous verify mode was asked for, but {reason}")
    return DraftMode.SYNCHRONOUS, reason


def node_chunks_prompts(model, node: str) -> bool:
    """Whether the model declares a chunkable walk on ``node``."""
    if model is None:
        return False
    policies = model.chunking_policy_of() or {}
    return any(policy.chunkable for (name, _), policy in policies.items() if name == node)


def combined_problem(decl: VerifyNode, combined_walk_of: Mapping[tuple[str, str], str]) -> str | None:
    """Mixed prompt-plus-verify steps are not supported yet: a combined walk
    holding the verify walk would commit its rows' blocks with no take-back."""
    combined = combined_walk_of.get((decl.node, decl.walk))
    if combined is None:
        return None
    return (
        f"node {decl.node!r}: walk {decl.walk!r} verifies drafts and is a member of combined walk "
        f"{combined!r}; mixed prompt-plus-verify steps are not supported yet, so declare none on a verifying node"
    )


def model_combined_walks(model) -> Mapping[tuple[str, str], str]:
    of = getattr(model, "combined_walk_of", None)
    return of() if of is not None else {}


def slack_problems(decls, specs, max_seq_len: int | None = None) -> list[str]:
    """A verifying node's KV pool holds ``2(k + 1)`` tokens a request past the
    serving limit, the reserved length of two committed steps whose results
    are unapplied, and declares them (``KVSpec.draft_slack``) so ingest holds
    a request to the serving limit and ``max_seq_len`` keeps its meaning."""
    from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig

    problems = []
    for decl in decls:
        slack = decl.config.kv_slack
        for spec in specs:
            if not (isinstance(spec, KVSpec) and isinstance(spec.config, PagedKVConfig) and decl.node in spec.nodes):
                continue
            if spec.draft_slack != slack:
                problems.append(
                    f"node {decl.node!r}: KV {spec.resource_key!r} declares draft_slack {spec.draft_slack}; "
                    f"verifying needs {slack} (DraftConfig.kv_slack), held back from the client's limit, "
                    "which drafter_resources sets"
                )
            elif max_seq_len is not None and spec.request_limit < max_seq_len:
                problems.append(
                    f"node {decl.node!r}: KV {spec.resource_key!r} holds a request to {spec.request_limit} "
                    f"positions, below the serving config's max_seq_len {max_seq_len}; set "
                    f"resources.{spec.resource_key}.max_seq_len to {max_seq_len} (the verify slack is added past it), "
                    f"or lower the serving max_seq_len to {spec.request_limit}"
                )
    return problems


def chunking_problem(decl: VerifyNode, chunks_prompts: bool) -> str | None:
    if chunks_prompts and decl.needs.chunk_boundary == ChunkBoundary.UNDECLARED:
        return f"node {decl.node!r}: its prompts prefill in chunks, and the drafter declares no chunk-boundary rule"
    return None


def uncorrected_problems(decl: VerifyNode, specs: Mapping[str, NodeResourceSpec]) -> list[str]:
    """Each recurrent pool of the node that no linear-attention resource reruns
    after a verify step (a `LinearAttnSpec` naming it as its ``recurrent_state``):
    it would keep the rejected tokens in its state. `LinearAttnSpec` refuses the
    variants that don't rerun, so naming the pool is enough."""
    from mstar.engine.resources.linear_attn.config import LinearAttnSpec
    from mstar.engine.resources.recurrent.config import RecurrentStateSpec

    on_node = [spec for spec in specs.values() if decl.node in spec.nodes]
    rerun = {spec.config.recurrent_state for spec in on_node if isinstance(spec, LinearAttnSpec)}
    return [
        f"node {decl.node!r}: recurrent pool {spec.resource_key!r}: no linear-attention resource reruns its state "
        "after a verify step (only GDN's does); its rejected tokens would stay in the state"
        for spec in on_node if isinstance(spec, RecurrentStateSpec) and spec.resource_key not in rerun
    ]


def node_problems(
    decl: VerifyNode, token_stop, specs: Mapping[str, NodeResourceSpec], chunks_prompts: bool,
    combined_walk_of: Mapping[tuple[str, str], str] | None = None, max_seq_len: int | None = None,
) -> tuple[list[str], DraftLag | None]:
    """Every problem with one verifying node that its declaration and its resources show, and its drafter's lag
    (`draft_lag`): the one list both `check_deployment` (before workers spawn, ``max_seq_len`` the serving
    config's) and the engine at load (with the built submodule's ``token_stop``) run, so the two can't disagree."""
    problems = shape_problems(decl, token_stop, specs) + geometry_problems(decl, specs)
    problems += resource_problems(decl, specs) + uncorrected_problems(decl, specs)
    lag, lag_problems = draft_lag(decl, specs)
    problems += lag_problems
    for problem in (chunking_problem(decl, chunks_prompts), combined_problem(decl, combined_walk_of or {})):
        if problem is not None:
            problems.append(problem)
    return problems + slack_problems([decl], specs.values(), max_seq_len), lag


def log_mode(decl: VerifyNode, mode: DraftMode, reason: str) -> None:
    logger.info(
        "speculative decoding on %s: %s, k=%d, %s mode (%s)",
        decl.node, decl.config.method, decl.config.num_draft_tokens, mode.value, reason,
    )


def check_deployment(
    decls: Iterable[VerifyNode], specs: Mapping[str, NodeResourceSpec],
    streaming_out: Mapping[str, bool], side_lane: Iterable[str], async_scheduling: Mapping[str, tuple[bool, str]],
    resident_planning: bool, chunked: Iterable[str] = (),
    combined_walk_of: Mapping[tuple[str, str], str] | None = None, max_seq_len: int | None = None,
) -> list[str]:
    """`validate_deployment`'s share: every problem with every verifying node,
    and each node's mode logged."""
    problems = []
    side = set(side_lane)
    for decl in decls:
        problems += node_problems(decl, decl.token_stop, specs, decl.node in set(chunked), combined_walk_of,
                                  max_seq_len)[0]
        if streaming_out.get(decl.node):
            problems.append(f"node {decl.node!r}: an outgoing streaming edge expects one item a step")
        if decl.node in side:
            problems.append(f"node {decl.node!r}: a side lane's steps are never pre-planned or lockstep")
        on, why = async_scheduling.get(decl.node, (True, ""))
        try:
            mode, reason = resolve_mode(decl, specs, on, why, resident_planning)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        log_mode(decl, mode, reason)
    return problems
