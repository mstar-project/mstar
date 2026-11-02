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
    FLASHINFER = "flashinfer"


@dataclass
class LinearAttnConfig:
    recurrent_state: str  # name of the recurrent state pool
    variant: LinearAttnVariant
    backend: LinearAttnBackend = LinearAttnBackend.FLASHINFER

    # Defaults to head_k_dim ** -0.5 off the pool's geometry, as the kernels'
    # own default does.
    sm_scale: float | None = None

    # GLM-5.3 sets this (`gate_lower_bound: -5.0`), selecting a different gate
    # formula in the KDA kernel. None keeps the softplus one.
    gate_lower_bound: float | None = None

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


# How a split step's prompt rows reach the recurrence: "ulysses" trades tokens for heads,
# "state_passing" hands each rank's state on to the next (`linear_attn.sequence_parallel`)
LinearSpExchange = Literal["none", "ulysses", "state_passing"]
LINEAR_SP_EXCHANGES: tuple[LinearSpExchange, ...] = get_args(LinearSpExchange)


@dataclass
class LinearAttnSpec(NodeResourceSpec):
    config: LinearAttnConfig
    # the sequence shard over its nodes, if any (`derive`)
    shard_key: str | None = field(default=None, init=False, repr=False)
    # how a split step reaches the recurrence: a deployment's choice, set only by its `resources:`
    # block, as `AttentionSpec.sp_exchange`; None follows the node (`resolve_exchange`)
    sp_exchange: LinearSpExchange | None = field(default=None, init=False, repr=False)
    # `sp_exchange` resolved (`derive`), and the pool's config, which the refusals read
    split_exchange: LinearSpExchange = field(default="none", init=False, repr=False)
    _pool: "RecurrentStateConfig | None" = field(default=None, init=False, repr=False, compare=False)

    def depends_on(self) -> set[str]:
        keys = {self.config.recurrent_state}
        if self.shard_key is not None:
            keys.add(self.shard_key)
        return keys

    def derive(self, specs: Mapping[str, NodeResourceSpec]) -> list[str]:
        from mstar.engine.resources.sequence_shard import take_shard_edge

        problems = take_shard_edge(self, specs)
        self.split_exchange = self.resolve_exchange(specs)
        self._pool = specs[self.config.recurrent_state].config
        return problems

    def resolve_exchange(self, specs: Mapping[str, NodeResourceSpec]) -> LinearSpExchange:
        """``sp_exchange``, or where unset the node's: ``ulysses`` where another resource on its
        nodes takes split steps, ``none`` otherwise, so a recurrent layer never makes a node split
        on its own. Another linear spec answers only with its own setting, so this doesn't recurse."""
        if self.sp_exchange is not None:
            return self.sp_exchange
        for spec in specs.values():
            if spec is self or not spec.nodes & self.nodes:
                continue
            exchange = spec.sp_exchange if isinstance(spec, LinearAttnSpec) else spec.resolve_exchange(specs)
            if exchange not in (None, "none"):
                return "ulysses"
        return "none"

    def captures_split_steps(self, specs: Mapping[str, NodeResourceSpec]) -> bool:
        # both exchanges plan their halo and hand-off per rank on the host, eager only
        return self.resolve_exchange(specs) == "none"

    def split_step_refusals(self, node: str, shard: "SeqShardSpec", tp: int, sp: int) -> list[str]:
        exchange = self.split_exchange
        name = f"linear attention {self.resource_key!r}"
        if exchange == "none":
            return [
                f"{name} takes no split steps (sp_exchange none), and node {node!r}'s other resources split its "
                "steps: set its sp_exchange to ulysses or state_passing, or none on the others"
            ]
        if shard.axis == 1 or shard.stripe_pool is not None:
            how = "each row (axis 1)" if shard.axis == 1 else f"by KV {shard.stripe_pool!r}'s stored positions"
            return [f"{name} splits a step's rows into contiguous pieces, and node {node!r}'s shard splits {how}"]
        heads = _unsharded_heads(self._pool, self.config.variant)
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
            dk, dv = _unsharded_head_dims(self._pool)
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

    def apply_yaml_overrides(
        self, backend: str | LinearAttnBackend | None = None,
        sp_exchange: LinearSpExchange | None = None,
    ):
        """Let the deployment pick the backend and how a split step reaches the recurrence
        (``sp_exchange``); slot capacity is tuned on the pool."""
        if sp_exchange is not None:
            if sp_exchange not in LINEAR_SP_EXCHANGES:
                raise ValueError(
                    f"sp_exchange must be one of {', '.join(LINEAR_SP_EXCHANGES)}; got {sp_exchange!r}"
                )
            self.sp_exchange = sp_exchange
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
    """
