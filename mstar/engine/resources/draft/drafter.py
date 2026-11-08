"""What a drafter declares and implements, and the one call a target's forward
makes for a verify step.

A verify step runs the target over ``k + 1`` rows a request: the last emitted
token, which has no KV yet, then ``k`` drafts. The engine fixes the order
(`draft_verify_step`): draft, agree on rank 0's drafts, run the target, verify
in the sampler, let a stateful drafter observe, and record what the step's
resources correct themselves from. A model holds the declaration, the target
function and the drafter's weights, and no state across steps.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from mstar.engine.resources.kv.config import KVLayerSet
    from mstar.engine.resources.resident import CapturedResidentStep
    from mstar.engine.resources.spec import NodeResourceSpec
    from mstar.engine.resources.step import ResourceStep

# check_stop_buffers entries a verify step publishes, which reach the host in
# the stop check's copy: the block's tokens and the drafts each row accepted
VERIFY_TOKENS = "verify_tokens"
VERIFY_ACCEPTED = "verify_accepted"
# the host column the token cut adds to them: whether a row's emitted run ended the reply
VERIFY_STOPPED = "verify_stopped"
# A drafter module's first KV pool layer until the engine binds it from the pool's layer set
# (`KVManager.layer_set`): the engine refuses a verifying node whose drafter is still unbound
# (`Drafter.kv_layers_unbound`).
UNBOUND_KV_LAYER = -1


class ChunkBoundary(Enum):
    """How a drafter's state at a prompt chunk's last row is built."""
    # not said: a chunked prompt is refused on the node
    UNDECLARED = "undeclared"
    # its state at row ``p`` depends only on the target at ``p`` (DSpark)
    NONE = "none"


class ModelGeometry(NamedTuple):
    """A model's shapes. A target's: startup checks a drafter's taps against its
    layers, and the residual width, vocabulary and weight dtype against the
    drafter's (`geometry_problems`). A drafter's: its own layers, KV heads and
    head size, which make the layer set it joins the target's KV pool with
    (`layer_set`; the join refuses another KV storage geometry), the residual
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
    # the layers it adds to the target's KV pool, which share its pages and length
    # (`drafter_resources` joins them)
    kv: KVLayerSet | None = None
    input_kinds: frozenset[str] = frozenset({"text"})
    chunk_boundary: ChunkBoundary = ChunkBoundary.UNDECLARED
    # KV slots it writes past the verify block, which no reservation covers
    kv_overhang: int = 0
    # its shapes, which must match the target's (`VerifyNode.target`); None for
    # a drafter that reads nothing of the target's but the tokens
    geometry: ModelGeometry | None = None


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


class DrafterDecl(Protocol):
    """A drafter as its node declares it, from config alone: its needs and the
    resources it adds to the node's (`drafter_resources`)."""
    needs: DrafterNeeds

    def node_resources(self, nodes: set[str]) -> list[NodeResourceSpec]:
        ...


def drafter_resources(specs: list[NodeResourceSpec], decl: DrafterDecl, nodes: set[str]) -> list[NodeResourceSpec]:
    """The node's resource specs with its drafter's: the resources it declares,
    its layers joined to the target's KV pool (``needs.kv``)."""
    from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig

    needs = decl.needs
    out = list(specs) + list(decl.node_resources(nodes))
    if needs.kv is not None:
        pools = [spec for spec in out if spec.resource_key == needs.kv.kv_cache]
        if len(pools) != 1 or not isinstance(pools[0], KVSpec):
            raise ValueError(f"the drafter joins KV {needs.kv.kv_cache!r}, which the node doesn't declare")
        if not isinstance(pools[0].config, PagedKVConfig):
            raise ValueError(f"the drafter joins KV {needs.kv.kv_cache!r}, which is not a paged pool")
        pools[0].config.join(needs.kv)
    return out


@runtime_checkable
class Drafter(Protocol):
    needs: DrafterNeeds
    # whether its KV layers are still at `UNBOUND_KV_LAYER`; False for a drafter with no KV layers
    kv_layers_unbound: bool

    def propose(self, inputs: DraftInputs) -> torch.Tensor:
        """``[n, k]`` int64 drafts, capturable."""
        ...


def _observe(drafter: Drafter, block: torch.Tensor, result: VerifyResult) -> None:
    observe = getattr(drafter, "observe", None)
    if observe is not None:
        observe(block, result)


def draft_verify_step(
    sampler,
    request_ids: Sequence[int],
    input_ids: torch.Tensor,
    target: Callable[[torch.Tensor], torch.Tensor],
    drafter: Drafter,
    resources: dict[str, Any],
    resident: CapturedResidentStep | None = None,
) -> VerifyResult:
    """One verify step, in the engine's order. ``target`` maps the block's ids
    ``[n * (k + 1)]`` to logits ``[n * (k + 1), V]``; ``sampler`` is the node's
    verifying `SamplerResource`. Capturable: nothing here reads the device on
    the host."""
    k = sampler.verify_draft_tokens
    n = input_ids.shape[0]
    drafts = drafter.propose(DraftInputs(request_ids, input_ids, resources, k))
    if drafts.shape != (n, k):
        raise ValueError(f"drafter proposed {tuple(drafts.shape)} drafts for {n} rows of {k}")
    # every rank writes KV for rank 0's drafts
    drafts = sampler.agree(drafts.long())
    block = torch.cat([input_ids.reshape(n, 1).long(), drafts], dim=1)
    logits = target(block.reshape(-1))
    result = sampler.verify(request_ids, logits.view(n, k + 1, -1), drafts)
    _observe(drafter, block, result)
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
