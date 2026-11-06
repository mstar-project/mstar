"""What a model declares about linear attention: variant, backend, spec, step.

Kept free of the managers so declaring a step does not import FlashInfer. The
state lives in a ``RecurrentStatePool`` (as `kv/` is to `attn/`), and head
geometry is read off its blocks rather than declared here.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal, get_args

from mstar.engine.resources.spec import CombinedSplit, NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource
    from mstar.engine.resources.recurrent.config import RecurrentStateConfig
    from mstar.engine.resources.sequence_shard import SeqShardSpec


class LinearAttnVariant(Enum):
    # gated delta rule: scalar decay per head. Qwen3.5, Qwen3-Next.
    GDN = "gdn"
    # Kimi delta attention: diagonal decay per K channel. Kimi Linear, GLM-5.3.
    KDA = "kda"
    # Mamba-2 / SSD: per-head scalar decay exp(dt * A), grouped B/C. Nemotron-H.
    MAMBA2 = "mamba2"


class LinearAttnBackend(Enum):
    FLASHINFER = "flashinfer"  # GDN: FlashInfer's kernels
    TRITON = "triton"  # KDA: the fused kernels in kda_triton.py


@dataclass
class LinearAttnConfig:
    recurrent_state: str  # name of the recurrent state pool
    variant: LinearAttnVariant
    backend: LinearAttnBackend = LinearAttnBackend.FLASHINFER

    # Defaults to head_k_dim ** -0.5 off the pool's geometry, as the kernels'
    # own default does.
    sm_scale: float | None = None

    # L2-normalise q and k for the delta rule: in the kernel where it can, by
    # `GDNManager.run` where it cannot (the SM90 chunked prefill). Qwen3.5 needs
    # it (the HF reference passes it) and skipping it diverges silently.
    qk_l2norm: bool = True

    # Mamba-2 only: clamp on dt after softplus, as HF's ``time_step_limit``
    # (Nemotron-H leaves it open: (0, inf)).
    time_step_limit: tuple[float, float] = (0.0, float("inf"))

    # GDN only: combined walks whose steps run decode rows on the decode kernel
    # and prompt rows on the chunked one (`GDNMixedWrapper`)
    splits: tuple[CombinedSplit, ...] = ()


# How a split step's prompt rows reach the recurrence (`linear_attn.base`): "ulysses" trades tokens
# for heads, "state_passing" hands each rank's state on to the next
LinearSpExchange = Literal["ulysses", "state_passing"]


@dataclass
class LinearAttnSpec(NodeResourceSpec):
    config: LinearAttnConfig
    # the sequence shard over its nodes, if any (`derive`)
    shard_key: str | None = field(default=None, init=False, repr=False)
    # a deployment's ``sp_exchange``, ulysses unset
    sp_exchanges = get_args(LinearSpExchange)

    def depends_on(self) -> set[str]:
        keys = {self.config.recurrent_state}
        if self.shard_key is not None:
            keys.add(self.shard_key)
        return keys

    def derive(self, specs: Mapping[str, NodeResourceSpec]) -> list[str]:
        from mstar.engine.resources.sequence_shard import take_shard_edge

        return take_shard_edge(self, specs)

    def resolve_exchange(self, specs: Mapping[str, NodeResourceSpec]) -> str:
        return self.own_exchange(specs)

    def split_step_refusals(
        self, node: str, shard: "SeqShardSpec", tp: int, sp: int, specs: Mapping[str, "NodeResourceSpec"],
    ) -> list[str]:
        # only a node whose shard splits gets here, so the choice is what its split steps take
        exchange = self.split_exchange
        pool = specs[self.config.recurrent_state].config
        name = f"linear attention {self.resource_key!r}"
        if self.config.variant is LinearAttnVariant.KDA:
            # KDAManager sets no supports_split_steps, which the engine refuses at build; a spec can't
            # see the built class (`resource_class` is LinearAttnManager for every variant), so this
            # names the variant. Both change together if KDA gains split steps
            return [
                f"{name} is Kimi delta attention, which takes no split steps, so node {node!r} can't split them: "
                "run it on TP ranks alone, or store its KV by position (context_parallel) with its sequence "
                "shard's exchange at auto, so its steps run copied"
            ]
        if shard.axis == 1 or shard.stripe_pool is not None:
            how = "each row (axis 1)" if shard.axis == 1 else f"by KV {shard.stripe_pool!r}'s stored positions"
            return [f"{name} splits a step's rows into contiguous pieces, and node {node!r}'s shard splits {how}"]
        heads = _unsharded_heads(pool, self.config.variant)
        if exchange == "ulysses":
            # Mamba-2's B/C groups may also be fewer a TP rank than SP ranks: each SP rank's heads
            # then read one group, repeated to it
            uneven = [
                f"{n} {what}" for what, n in heads.items()
                if (n // tp) % sp and not (what == "B/C groups" and sp % (n // tp) == 0)
            ]
            if uneven:
                return [
                    f"{name} trades tokens for heads over {sp} SP ranks (ulysses), and its {', '.join(uneven)} "
                    f"don't split {sp} ways per TP rank: set its sp_exchange to state_passing"
                ]
        if exchange == "state_passing" and sp >= 3 and self.config.variant is not LinearAttnVariant.MAMBA2:
            dk, dv = _unsharded_head_dims(pool)
            if dk != dv:
                return [
                    f"{name} hands state on over {sp} SP ranks (state_passing), whose middle pieces' transition "
                    f"needs equal key and value head dims; they are {dk} and {dv}"
                ]
        return []

    def combined_splits(self) -> tuple[CombinedSplit, ...]:
        return self.config.splits

    @property
    def writes_prefix_snapshots(self) -> bool:
        # only the GDN wrapper writes state checkpoints into the pool's snapshot slots
        return self.config.variant is LinearAttnVariant.GDN

    @property
    def restores_prefix(self) -> bool:
        # its state is the pool's
        return True

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.linear_attn.base import LinearAttnManager

        return LinearAttnManager

    def apply_yaml_overrides(self, backend: str | LinearAttnBackend | None = None):
        """Let the deployment pick the backend (and, as every spec, ``sp_exchanges``' choice);
        slot capacity is tuned on the pool."""
        if backend is not None:
            self.config.backend = LinearAttnBackend(backend)


def _unsharded_heads(pool: "RecurrentStateConfig", variant: LinearAttnVariant) -> dict[str, int]:
    """The pool's head counts as the model declared them, before any rank shards them."""
    if variant is LinearAttnVariant.MAMBA2:
        heads, head_dim, state_size = pool.blocks["ssm"]._unsharded_shape
        conv_dim = pool.blocks["conv"]._unsharded_shape[0]
        return {"heads": heads, "B/C groups": (conv_dim - heads * head_dim) // (2 * state_size)}
    v_heads, v_dim, k_dim = pool.blocks["state"]._unsharded_shape
    conv_dim = pool.blocks["conv"]._unsharded_shape[0]
    return {"key heads": (conv_dim - v_heads * v_dim) // (2 * k_dim), "value heads": v_heads}


def _unsharded_head_dims(pool: "RecurrentStateConfig") -> tuple[int, int]:
    _, v_dim, k_dim = pool.blocks["state"]._unsharded_shape
    return k_dim, v_dim


@dataclass(frozen=True)
class LinearAttnStep(ResourceStep):
    """One step's work for a linear-attention layer stack.

    Carries no state semantics: the segments give the rows and spans, and the
    pool's own step says what becomes of the slots. The walk is derived from
    the spans.

    ``speculative``: every row is a verify block of a speculating node, which
    first replays the accepted part of the row's last block (the pool's
    speculative blocks); the span is the block's length, the same for every row.
    """

    speculative: bool = False
