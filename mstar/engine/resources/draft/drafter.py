"""What a drafter declares and implements, and the one call a target's forward
makes for a verify step.

A verify step runs the target over ``k + 1`` rows a request: the last emitted
token, which has no KV yet, then ``k`` drafts. The engine fixes the order
(`draft_verify_step`): draft (from the row's carry, for a drafter that declares
one), agree on rank 0's drafts, run the target, verify in the sampler, let each
resource with state the block overwrote correct it (``correct_verify_state``), let a
stateful drafter observe and keep its new carry, and record what the step's
resources correct themselves from. A model holds the declaration, the target
function and the drafter's weights, and no state across steps.
"""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from mstar.engine.resources.kv.config import KVLayerSet
    from mstar.engine.resources.resident import CapturedResidentStep
    from mstar.engine.resources.spec import NodeResourceSpec
    from mstar.engine.resources.step import Lag, ResourceStep

# check_stop_buffers entries a verify step publishes, which reach the host in
# the stop check's copy: the block's tokens and the drafts each row accepted
VERIFY_TOKENS = "verify_tokens"
VERIFY_ACCEPTED = "verify_accepted"
# the host column the token cut adds to them: whether a row's emitted run ended the reply
VERIFY_STOPPED = "verify_stopped"
# The sampler loop-back state (``SamplerSpec.loopback_state``) a drafter's carry lives
# in: a row per request, held from admission to removal, which the engine reads before
# ``propose`` and writes from ``observe`` (and a prompt's last prefill writes first)
DRAFT_CARRY = "draft_carry"
# A lagged drafter's per-request flag (`DrafterNeeds.lag`), kept by the node's sampler: 1 when the
# target accepted every draft last step, so the drafter's row 0 (the token before the block, which
# its last block never reached) writes its KV; 0 sends that write to the sink. The engine sets it.
DRAFT_KEEP_FIRST = "draft_keep_first"


class ChunkBoundary(Enum):
    """How a drafter's state at a prompt chunk's last row is built."""
    # not said: a chunked prompt is refused on the node
    UNDECLARED = "undeclared"
    # its state at row ``p`` depends only on the target at ``p`` (DSpark)
    NONE = "none"


class CarrySpec(NamedTuple):
    """Per-request drafter state across steps (a drafter's hidden vector): the
    node's sampler keeps it as loop-back state `DRAFT_CARRY`, by request, from
    admission to removal."""
    width: int
    dtype: torch.dtype


class ModelGeometry(NamedTuple):
    """A model's shapes. A target's: startup checks a drafter's taps against its
    layers, and the residual width, vocabulary and weight dtype against the
    drafter's (`geometry_problems`). A drafter's: its own layers, KV heads and
    head size, which make the layer set it joins the target's KV pool with
    (`layer_set`; one of another KV geometry gets a storage of its own over the
    same pages, `PagedKVConfig.join`), the residual
    width it reads, the vocabulary it drafts in, its weight dtype, and the query
    heads and RoPE base of its attention, checked against the pool's attention
    and the node's positions."""
    num_layers: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    # as HF's config.json names it ("bfloat16")
    dtype: str
    # a drafter's: its attention plans over the pool it joins, and rotates through
    # the node's position resource; None checks nothing
    num_qo_heads: int | None = None
    rope_theta: float | None = None

    def layer_set(self, kv_cache: str, name: str) -> KVLayerSet:
        """The layer set of these layers in KV pool ``kv_cache``."""
        from mstar.engine.resources.kv.config import KVLayerSet

        return KVLayerSet(kv_cache, name, self.num_layers, self.num_kv_heads, self.head_dim)


class DrafterNeeds(NamedTuple):
    """What a drafter needs from the engine, checked at startup."""
    # target layers whose input (the residual stream entering them) it reads;
    # a layer equal to the target's layer count is the stream after the last layer
    taps: tuple[int, ...] = ()
    # the node's `TapSpec` that holds them, by resource key: the one source of the key
    tap_key: str | None = None
    # the layers it adds to the target's KV pool, which share its pages and length
    # (`drafter_resources` joins them)
    kv: KVLayerSet | None = None
    carry: CarrySpec | None = None
    # ``observe`` reads the target's last hidden state of every block row: the
    # target function sets ``TargetOutput.hidden``
    target_hidden: bool = False
    input_kinds: frozenset[str] = frozenset({"text"})
    chunk_boundary: ChunkBoundary = ChunkBoundary.UNDECLARED
    # KV slots it writes past the verify block, which no reservation covers
    kv_overhang: int = 0
    # its shapes, which must match the target's (`VerifyNode.target`); None for
    # a drafter that reads nothing of the target's but the tokens
    geometry: ModelGeometry | None = None
    # a separate draft model's (`Lag`): its verify block is the target's shifted back ``rows``,
    # ``[prev, bonus, d1 .. d(k-1)]``, run in ``passes`` (one-row passes after the first's two). The
    # engine copies it into the verify step's KV writes of ``kv``, positions and the drafter's attention
    # binding (`drafter_step`), and sinks row 0 unless `DRAFT_KEEP_FIRST` keeps it. No other walk lags.
    lag: "Lag | None" = None


class VerifyResult(NamedTuple):
    """A verify step's outcome, on the device and the same on every rank: a row
    emits ``tokens[:accepted + 1]``, its accepted drafts and then the bonus
    token, less what a stop or the budget cuts on the host."""
    tokens: torch.Tensor    # [n, k + 1] int64
    accepted: torch.Tensor  # [n] int64, 0..k


class DraftInputs(NamedTuple):
    """What a drafter's ``propose`` reads."""
    request_ids: Sequence[int]
    # each row's last emitted token, the first row of its verify block
    input_ids: torch.Tensor
    # the engine resources of the node, for taps and the drafter's own attention
    resources: dict[str, Any]
    num_draft_tokens: int
    # ``[n, width]``: each row's carry, as the last step's ``observe`` (or its
    # prompt's last prefill) left it; None for a drafter that declares none
    carry: torch.Tensor | None = None


class DrafterDecl(Protocol):
    """A drafter as its node declares it, from config alone: its needs and the
    resources it adds to the node's (`drafter_resources`). Subclass it."""
    needs: DrafterNeeds

    @abstractmethod
    def node_resources(self, nodes: set[str]) -> list[NodeResourceSpec]:
        ...


def drafter_resources(
    specs: list[NodeResourceSpec], decl: DrafterDecl, nodes: set[str], config,
) -> list[NodeResourceSpec]:
    """The verifying ``nodes``' resource specs with their drafter's: the
    resources it declares, its layers joined to the target's KV pool
    (``needs.kv``), and what follows from the draft ``config`` (a
    `DraftConfig`) alone. Each paged pool of the nodes holds the KV slack past
    the serving limit (`KVSpec.bind_draft_slack`), and the sampler serving the
    nodes, when one alone does, verifies and keeps no repetition penalty
    (verify refuses one per request anyway). A drafter's carry is kept by the
    node's sampler as loop-back state `DRAFT_CARRY`. Anything a deployment
    chose is left for startup to refuse."""
    from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
    from mstar.engine.resources.sampler.config import LoopbackState, SamplerSpec

    needs = decl.needs
    out = list(specs) + list(decl.node_resources(nodes))
    if needs.kv is not None:
        pools = [spec for spec in out if spec.resource_key == needs.kv.kv_cache]
        if len(pools) != 1 or not isinstance(pools[0], KVSpec):
            raise ValueError(f"the drafter joins KV {needs.kv.kv_cache!r}, which the node doesn't declare")
        if not isinstance(pools[0].config, PagedKVConfig):
            raise ValueError(f"the drafter joins KV {needs.kv.kv_cache!r}, which is not a paged pool")
        pools[0].config.join(needs.kv)
    kept = {}
    if needs.carry is not None:
        width, dtype = needs.carry
        kept[DRAFT_CARRY] = LoopbackState(width, dtype, (0.0,) * width)
    if needs.lag is not None:
        kept[DRAFT_KEEP_FIRST] = LoopbackState(1, torch.int64, (0,))
    if kept:
        samplers = [spec for spec in out if isinstance(spec, SamplerSpec) and spec.nodes & nodes]
        if len(samplers) != 1:
            raise ValueError(f"the drafter's carry is kept by the node's sampler; the node declares {len(samplers)}")
        samplers[0].loopback_state = {**samplers[0].loopback_state, **kept}
    for spec in out:
        if isinstance(spec, KVSpec) and isinstance(spec.config, PagedKVConfig) and not spec.nodes.isdisjoint(nodes):
            spec.bind_draft_slack(config.kv_slack)
    samplers = [spec for spec in out if isinstance(spec, SamplerSpec) and set(spec.nodes) == set(nodes)]
    if len(samplers) == 1:
        samplers[0].verifies = True
        samplers[0].enable_repetition_penalty = False
    return out


def carry_writes(needs: DrafterNeeds | None) -> frozenset[str]:
    """The sampler loop-back state a node's verify steps and prompt steps store
    for its drafter, which the engine adds to their SamplerStep's
    ``loopback_writes`` (`drafter_step`): `DRAFT_CARRY` for a drafter that
    carries state, nothing otherwise."""
    return frozenset({DRAFT_CARRY}) if needs is not None and needs.carry is not None else frozenset()


class TargetOutput(NamedTuple):
    """What a verify step's ``target`` returns for the block: its logits, and the target's last
    hidden states for a drafter that declares ``DrafterNeeds.target_hidden`` (None otherwise)."""
    logits: torch.Tensor
    hidden: torch.Tensor | None = None


class ObserveInputs(NamedTuple):
    """What a drafter's ``observe`` reads, once the step is verified."""
    # [n, k + 1]: each row's last emitted token and its drafts
    block: torch.Tensor
    result: VerifyResult
    # [n * (k + 1), hidden]: the target's last hidden states, for a drafter that declares
    # `DrafterNeeds.target_hidden`; None otherwise
    hidden: torch.Tensor | None = None


class PrefillInputs(NamedTuple):
    """What a drafter's ``prefill`` reads on a prompt step, once the target ran and sampled."""
    # the engine resources of the node
    resources: Mapping[str, Any]
    # [tokens]: the step's input ids, and [rows] the token it sampled for each
    input_ids: torch.Tensor
    sampled: torch.Tensor
    # [tokens, hidden]: the target's last hidden states and its input embeddings, every row
    hidden: torch.Tensor | None = None
    embeds: torch.Tensor | None = None
    # what the target's forward was given beside its input (Qwen3's taps), and preprocess's other tensors
    decoder_kwargs: Mapping[str, Any] = MappingProxyType({})
    tensors: Mapping[str, torch.Tensor] = MappingProxyType({})


def verify_writes(needs: DrafterNeeds | None) -> frozenset[str]:
    """The loop-back state a node's verify steps store for its drafter: `carry_writes`', and a lagged
    drafter's `DRAFT_KEEP_FIRST`, which only a verify step sets."""
    keep = frozenset({DRAFT_KEEP_FIRST}) if needs is not None and needs.lag is not None else frozenset()
    return carry_writes(needs) | keep


@runtime_checkable
class Drafter(Protocol):
    """A verifying node's drafter, built and loaded by the model. Subclass it
    (beside ``nn.Module`` for one with weights): ``propose`` is abstract, and
    ``observe`` and ``prefill`` have defaults."""
    needs: DrafterNeeds

    @abstractmethod
    def propose(self, inputs: DraftInputs) -> torch.Tensor:
        """``[n, k]`` int64 drafts, capturable."""
        ...

    def observe(self, inputs: ObserveInputs) -> torch.Tensor | None:
        """After the verify, in the same forward: what the drafter keeps of the
        step, returned as the rows' new carry for one that declares a carry
        (`DrafterNeeds.carry`). Nothing by default."""
        return None

    def prefill(self, inputs: PrefillInputs) -> torch.Tensor | None:
        """After a prompt step samples: the drafter's pass over the step's prompt
        rows, returning the rows' first carry for a drafter that declares one
        (`DrafterNeeds.carry`), which the scaffold stores. Nothing by default."""
        return None


@torch.compiler.disable
def _read_carry(sampler, request_ids: Sequence[int], n: int, name: str = DRAFT_CARRY) -> torch.Tensor:
    # the sampler's loop-back reads stay out of the compiled forward, as `sample` does
    return sampler.loopback_state(name, request_ids)[:n]


@torch.compiler.disable
def _sink_first_rows(kv, keep: torch.Tensor) -> None:
    kv.sink_lagged_rows(keep)


def draft_verify_step(
    sampler,
    request_ids: Sequence[int],
    input_ids: torch.Tensor,
    target: Callable[[torch.Tensor], TargetOutput],
    drafter: Drafter,
    resources: dict[str, Any],
    resident: CapturedResidentStep | None = None,
    correcting_keys: Sequence[str] = (),
) -> VerifyResult:
    """One verify step, in the engine's order. ``target`` maps the block's ids
    ``[n * (k + 1)]`` to a `TargetOutput`: logits ``[n * (k + 1), V]``, and with
    the drafter's ``needs.target_hidden`` the hidden states; ``sampler`` is the node's
    verifying `SamplerResource`, which also keeps a carrying drafter's
    `DRAFT_CARRY`; ``correcting_keys`` the resources that correct the block's state
    (`DrafterShare.correcting_keys`). Capturable: nothing here reads the device on the host."""
    k = sampler.verify_draft_tokens
    n = input_ids.shape[0]
    needs = drafter.needs
    carry = _read_carry(sampler, request_ids, n) if needs.carry is not None else None
    if needs.lag is not None:
        # the drafter's row 0 rewrites the token before the block, which its last block wrote unless
        # the target accepted every draft: in the graph, before any of the drafter's writes
        keep = _read_carry(sampler, request_ids, n, DRAFT_KEEP_FIRST).reshape(n) != 0
        _sink_first_rows(resources[needs.kv.kv_cache], keep)
    drafts = drafter.propose(DraftInputs(request_ids, input_ids, resources, k, carry))
    if drafts.shape != (n, k):
        raise ValueError(f"drafter proposed {tuple(drafts.shape)} drafts for {n} rows of {k}")
    # every rank writes KV for rank 0's drafts
    drafts = sampler.agree(drafts.long())
    block = torch.cat([input_ids.reshape(n, 1).long(), drafts], dim=1)
    out = target(block.reshape(-1))
    if not isinstance(out, TargetOutput):
        # a bare logits tensor would unpack along its rows; checked once, while the step is traced
        raise TypeError(f"a verify step's target returns a TargetOutput, not {type(out).__name__}")
    logits, hidden = out
    if needs.target_hidden and hidden is None:
        raise RuntimeError("the drafter reads the target's hidden states (DrafterNeeds.target_hidden), and the "
                           "target returned none")
    result = sampler.verify(request_ids, logits.view(n, k + 1, -1), drafts)
    for key in correcting_keys:
        # a resource whose state the block wrote in place (GDN's) corrects it here, in the graph
        resources[key].correct_verify_state(result)
    carry = drafter.observe(ObserveInputs(block, result, hidden))
    if needs.carry is not None:
        if carry is None:
            raise RuntimeError("a drafter that declares a carry returned none from observe")
        sampler.store_loopback_state(DRAFT_CARRY, request_ids, carry)
    if needs.lag is not None:
        sampler.store_loopback_state(DRAFT_KEEP_FIRST, request_ids, (result.accepted == k).long().view(n, 1))
    if resident is not None and resident.buffers.advance is not None:
        # the KV length cell advances by what the row keeps, in the graph
        resident.buffers.advance[:n].copy_(result.accepted + 1)
    return result


class VerifyNode(NamedTuple):
    """A model's declaration that ``node`` drafts and verifies on ``walk`` (its
    decode loop's walk), from the deployment's ``model_kwargs.draft``. Read by
    `validate_deployment` before workers spawn and by the engine at load."""
    node: str
    walk: str
    config: Any  # DraftConfig
    needs: DrafterNeeds
    # the node submodule's ``token_stop`` and ``device_loopback``, which validation reads
    # before any submodule exists: a verify walk's next step reads its row's token off the
    # sampler's loop-back slot, which the verify writes in the graph
    token_stop: Any = None
    device_loopback: Any = None
    # the input kinds the model's prompts on this node may carry
    input_kinds: frozenset[str] = frozenset({"text"})
    # the target's shapes, for a drafter that declares its own (`DrafterNeeds.geometry`)
    target: ModelGeometry | None = None
    # the drafter's own checkpoint directory, when it has one: prompt pages hold its KV
    # layers too, so the prefix cache is rooted in its weights as well as the target's
    checkpoint: str | None = None
    # the steps of the drafter's own resources on the node's prompt and verify steps, which the engine adds
    # to the model's (`drafter_step`)
    prefill_steps: Mapping[str, ResourceStep] = MappingProxyType({})
    verify_steps: Mapping[str, ResourceStep] = MappingProxyType({})
