"""What a model declares about per-request LoRA adapters.

Kept free of the pool and its kernels, so declaring a step imports neither.

Each declared target linear computes ``y += scaling[slot] * (x @ A[slot]^T) @ B[slot]^T`` next
to its frozen base GEMM, with the slot chosen per row: rows of one batch may run different
adapters, or none.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

import torch

from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig, instance_key

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource

# The resource key a model's ``LoraSpec`` is declared under, and so the serving YAML's
# ``resources:`` block that turns LoRA on (see ``mstar.engine.resources.lora.serving``).
LORA_RESOURCE_KEY = "lora"
# The request kwarg that names a request's adapter; the OpenAI ``model`` field maps onto it.
LORA_ADAPTER_KWARG = "lora_adapter"
# The digest of the adapter's weights when the API server selected it (``ServedModels``),
# set there, never by a client.
LORA_DIGEST_KWARG = "lora_digest"


def lora_enabled(model_config: Mapping[str, Any]) -> bool:
    """Whether a deployment's YAML turns LoRA on (a ``resources: lora:`` block), with or
    without adapters named in it: adapters may also be loaded at runtime. A ``resources:``
    value that is not a mapping turns nothing on: the conductor's ``validate_deployment``
    reports it with the deployment's other problems."""
    resources = model_config.get("resources")
    return isinstance(resources, Mapping) and LORA_RESOURCE_KEY in resources


def describe_load_error(exc: BaseException) -> str:
    """What a failed adapter read tells a client: a refusal (a ``ValueError``: rank,
    targets, shapes, weights changed on disk, a name already loaded) as itself, anything
    else (an I/O error naming the server's files, say) only that it could not be read.
    The caller logs ``exc`` in full."""
    if isinstance(exc, ValueError):
        return str(exc)
    return "the adapter could not be read (the server log has the details)"


# what a PEFT adapter directory holds its weights in
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"


def resolve_adapter_path(path: str) -> str:
    """The safetensors file an adapter path names: the path itself, or a PEFT directory's
    ``adapter_model.safetensors`` (what vLLM's ``lora_path`` names)."""
    if os.path.isdir(path):
        return os.path.join(path, ADAPTER_WEIGHTS_FILE)
    return path


@dataclass(frozen=True)
class LoraTarget:
    """One family of linears an adapter may touch, e.g. every layer's fused ``qkv_proj``.

    Shapes are the unsharded linear's. ``out_slices`` lists the output widths of a fused
    projection in order (``(q, k, v)``); each slice has its own A and B, named in a
    checkpoint by the stacked-shard id in ``slice_names``. A single linear has one slice.

    ``parallel`` says how the base linear is split across tensor-parallel ranks:

    * ``"column"``: output rows are split, so A is replicated and B is sharded by output rows
      of each slice. The delta lands on this rank's rows of the output.
    * ``"row"``: input columns are split, so A is sharded along the input and B replicated.
      ``x @ A_r^T`` is then a partial sum over this rank's input columns, and so is the
      expanded delta: add it to the partial base output *before* the layer's all-reduce,
      which sums both. LoRA adds no collective.
    * ``"none"``: every rank holds the whole adapter and adds the whole delta to an output
      that is already whole (after the base layer's own collective).

    ``slice_units`` counts the shardable units (heads) of each slice. A slice with fewer
    units than ranks is replicated across groups of ranks rather than split, as
    ``QKVParallelLinear`` does with GQA's K/V heads. None treats every row as a unit.

    ``slice_packs`` names checkpoint modules that pack several slices into one tensor: a
    gated delta-net's ``in_proj_qkv`` is ``[q | k | v]``, three outputs the base shards
    across ranks on their own. Such a module has one A, which each of its slices takes,
    and a B whose rows are those slices' in order.

    ``slice_align`` rounds each slice's rank-local width up to a multiple, for a base
    linear that pads its fused output per rank so the next slice starts aligned. The
    padding rows follow the slice's own, and B is zero there.

    ``layers`` names the stack layer indices this target exists on (a hybrid stack has
    attention on some layers only); None means all of the spec's ``num_layers``.

    ``node`` is the graph node the target's linears run on, which a spec over several
    nodes names for every target (each node's pool holds its own, ``LoraSpec.for_node``);
    None for a spec over one node.

    ``kind="embedding"`` is a token embedding: ``in_features`` is the vocabulary, a row's
    input is its token id, and the shrink gathers A's column for it (``x @ A^T`` of the
    token's one-hot), as PEFT's ``lora_embedding_A`` / ``_B`` define it.

    Which rows the linear sees, for the per-row adapter choice: one per token of the step
    (the default), one per segment (``per_segment``, e.g. an LM head over each row's last
    token), or one per ``token_fold`` consecutive tokens of a segment (a patch merger that
    folds 2x2 patches into one row).

    ``experts`` counts the groups of the weight: 1 for a linear, ``E`` for one stacked
    parameter of an MoE layer's routed experts (``[E, out, in]``), where each row runs
    its token's slot's adapter on the expert the router picked for it. Each expert has
    its own A and B per slice, named in a checkpoint by the stacked-shard id
    ``"<slice>:<expert>"`` (``"gate:3"``). ``expert_split`` is ``(first, count)`` of the
    experts one rank's module holds when expert parallelism splits them; None when it
    holds every expert.
    """

    in_features: int
    out_slices: tuple[int, ...]
    parallel: Literal["column", "row", "none"] = "column"
    slice_names: tuple[str | int, ...] = ()
    slice_units: tuple[int, ...] | None = None
    slice_packs: tuple[tuple[str | int, tuple[str | int, ...]], ...] | None = None
    slice_align: tuple[int, ...] | None = None
    layers: tuple[int, ...] | None = None
    node: str | None = None
    kind: Literal["linear", "embedding"] = "linear"
    per_segment: bool = False
    token_fold: int = 1
    experts: int = 1
    expert_split: tuple[int, int] | None = None

    def __post_init__(self):
        object.__setattr__(self, "out_slices", tuple(self.out_slices))
        object.__setattr__(self, "slice_names", tuple(self.slice_names))
        if self.parallel not in ("column", "row", "none"):
            raise ValueError(f"parallel must be 'column', 'row' or 'none', got {self.parallel!r}")
        if not self.out_slices:
            raise ValueError("a LoRA target needs at least one output slice")
        if self.slice_names and len(self.slice_names) != len(self.out_slices):
            raise ValueError(
                f"slice_names {self.slice_names} does not name each of the "
                f"{len(self.out_slices)} output slices"
            )
        if self.slice_units is not None and len(self.slice_units) != len(self.out_slices):
            raise ValueError(
                f"slice_units {self.slice_units} does not cover the {len(self.out_slices)} output slices"
            )
        if self.slice_packs is not None:
            packs = dict(self.slice_packs)
            object.__setattr__(
                self, "slice_packs", tuple((name, tuple(members)) for name, members in packs.items()),
            )
            for name, members in self.slice_packs:
                if name in self.slice_names or len(members) < 2 or any(
                    m not in self.slice_names for m in members
                ):
                    raise ValueError(
                        f"slice_packs {name!r} -> {members} must name two or more of the "
                        f"slices {self.slice_names}, under a name that is not one of them"
                    )
        if self.slice_align is not None and (
            len(self.slice_align) != len(self.out_slices) or min(self.slice_align) < 1
        ):
            raise ValueError(
                f"slice_align {self.slice_align} does not give each of the "
                f"{len(self.out_slices)} output slices a multiple >= 1"
            )
        if self.kind not in ("linear", "embedding"):
            raise ValueError(f"kind must be 'linear' or 'embedding', got {self.kind!r}")
        if self.kind == "embedding" and (self.num_slices != 1 or self.parallel != "none"):
            raise ValueError("an embedding target has one slice and parallel='none'")
        if self.token_fold < 1 or (self.per_segment and self.token_fold != 1):
            raise ValueError(
                f"token_fold must be >= 1, and 1 with per_segment; got {self.token_fold}"
            )
        if self.experts < 1:
            raise ValueError(f"experts must be >= 1; got {self.experts}")
        if self.experts > 1 and (
            self.kind != "linear" or self.row_grouping != 1 or self.slice_packs is not None
            or not self.slice_names
        ):
            raise ValueError(
                "an experts target is a linear over the step's tokens, with each slice named "
                "and no packs"
            )
        if self.expert_split is not None and self.experts == 1:
            raise ValueError("expert_split needs an experts target")

    @property
    def num_slices(self) -> int:
        return len(self.out_slices)

    @property
    def out_features(self) -> int:
        return sum(self.out_slices)

    @property
    def row_grouping(self) -> int:
        """Which addressing the linear reads: 1 per token, ``k`` per ``k`` folded tokens,
        0 per segment."""
        return 0 if self.per_segment else self.token_fold

    def expert_of(self, shard_id: str | int | None) -> int:
        """The expert a checkpoint module of an experts target holds, from its
        ``"<slice>:<expert>"`` shard id; 0 for any other target."""
        if self.experts == 1:
            return 0
        expert = _expert_part(shard_id)
        if expert is None or not 0 <= expert < self.experts:
            raise KeyError(f"shard id {shard_id!r} names no expert of {self.experts}")
        return expert

    def slice_index(self, shard_id: str | int | None) -> int:
        """Which slice a checkpoint module lands on, from its stacked-shard id."""
        if self.experts > 1:
            self.expert_of(shard_id)
            shard_id = shard_id.rsplit(":", 1)[0]
        if shard_id is None:
            if self.num_slices != 1:
                raise KeyError(
                    f"a fused target with slices {self.slice_names} needs a shard id"
                )
            return 0
        if shard_id not in self.slice_names:
            raise KeyError(f"unknown shard id {shard_id!r}; this target has {self.slice_names}")
        return self.slice_names.index(shard_id)

    def module_slices(self, shard_id: str | int | None) -> tuple[int, ...]:
        """The slices a checkpoint module lands on, in the order its B's rows hold them:
        several for a ``slice_packs`` name, else the one ``slice_index`` gives."""
        for name, members in self.slice_packs or ():
            if name == shard_id:
                return tuple(self.slice_names.index(m) for m in members)
        return (self.slice_index(shard_id),)

    def shard(self, world_size: int) -> "LoraTarget":
        """This target as one rank of ``world_size`` sees it: its slices the widths they
        have in that rank's base output, alignment padding included."""
        local = tuple(self.local_width(i, world_size) for i in range(self.num_slices))
        if world_size == 1 and local == self.out_slices:
            return self
        in_features = self.in_features
        if self.parallel == "row":
            in_features = _divide(in_features, world_size)
        return dataclasses.replace(
            self, in_features=in_features, out_slices=local,
            slice_units=None, slice_align=None,
        )

    def local_width(self, slice_idx: int, world_size: int) -> int:
        """Rows of slice ``slice_idx`` in one rank's base output, padding included."""
        rows = self.slice_rows(slice_idx, 0, world_size)[1]
        if self.slice_align is not None:
            align = self.slice_align[slice_idx]
            rows = -(-rows // align) * align
        return rows

    def slice_rows(self, slice_idx: int, rank: int, world_size: int) -> tuple[int, int]:
        """``(start, size)`` of ``rank``'s output rows within slice ``slice_idx``."""
        rows = self.out_slices[slice_idx]
        if self.parallel != "column" or world_size == 1:
            return 0, rows
        units = rows if self.slice_units is None else self.slice_units[slice_idx]
        if units < world_size:
            # replicated: each group of `world_size // units` ranks holds one unit
            replicas = _divide(world_size, units)
            size = rows // units
            return (rank // replicas) * size, size
        size = _divide(rows, world_size)
        return rank * size, size

    def input_cols(self, rank: int, world_size: int) -> tuple[int, int]:
        """``(start, size)`` of ``rank``'s input columns."""
        if self.parallel != "row" or world_size == 1:
            return 0, self.in_features
        size = _divide(self.in_features, world_size)
        return rank * size, size

    def shard_pair(
        self, slice_idx: int, a: torch.Tensor, b: torch.Tensor, rank: int, world_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``rank``'s share of one slice's ``a [r, in]`` and ``b [out_slice, r]``."""
        a = a.narrow(1, *self.input_cols(rank, world_size))
        return a, b.narrow(0, *self.slice_rows(slice_idx, rank, world_size))


def _expert_part(shard_id: str | int | None) -> int | None:
    if not isinstance(shard_id, str) or ":" not in shard_id:
        return None
    expert = shard_id.rsplit(":", 1)[1]
    return int(expert) if expert.isdigit() else None


def _divide(numerator: int, denominator: int) -> int:
    if numerator % denominator:
        raise ValueError(f"{numerator} is not divisible by {denominator}")
    return numerator // denominator


class LoraModuleRef(NamedTuple):
    """Where one checkpoint module's adapter lands: a target, a stack layer, and the
    stacked-shard id of its slice (None for a single-slice target). ``param`` names the
    parameter of the bound module the target adapts, for a module with a target per
    parameter (MoE experts); None for a module that is one target."""

    target: str
    layer: int
    shard_id: str | int | None = None
    param: str | None = None


# checkpoint module path -> where it lands, or None for a module the model does not know
LoraModuleResolver = Callable[[str], "LoraModuleRef | None"]


class AdapterInfo(NamedTuple):
    """What validation needs to know about an adapter checkpoint, from its modules' shapes.

    ``rank`` covers every LoRA module in the file, those left to another node's pool included.
    ``misfits`` names modules whose shapes do not fit their target (an adapter trained on
    another size of the model)."""

    rank: int
    unmapped: tuple[str, ...]
    misfits: tuple[str, ...] = ()


@dataclass
class LoraSpec(NodeResourceSpec):
    """Per-request LoRA adapters for the linears named in ``targets``.

    ``max_slots`` adapters are resident on the device at once, beside a zero adapter that
    base-model rows use, and up to ``cpu_capacity`` are kept pinned on the host.
    ``adapters`` maps the names requests ask for to safetensors paths. ``slot_wait_steps``
    bounds how long a request waits for a device slot while others keep slots busy: past
    it, requests on other adapters are held back until a slot drains (counted in the
    pool's steps). ``module_map`` places each checkpoint module on a target; a node's
    spec comes from ``module_lora_spec`` (``mstar.model.components.lora.modules``), whose
    ``LoraMap`` is what binds the node's modules, and a pool built with any other
    resolver is refused at bind. ``convert_keys`` rewrites a model-native checkpoint
    layout first. ``declare`` gives ``num_layers``, ``targets``, ``module_map`` and
    ``ignored_modules`` in place of passing them, on their first read: ``module_lora_spec``
    walks a node's modules there, which a deployment that never builds the spec skips.

    Opt-in (``NodeResourceSpec.opt_in``): built only for a deployment whose serving config
    has a ``resources: lora:`` block. One spec covers every node an adapter reaches (an
    LLM and its vision tower), and each node gets a pool of its own (``per_node``): the
    nodes may sit in different lockstep instances, whose ranks must make the same slot
    decisions.
    """

    opt_in: ClassVar[bool] = True
    per_node: ClassVar[bool] = True

    # what ``declare`` may give defaults by factory, which sets no class attribute that
    # would answer a read before ``__getattr__`` does
    num_layers: int = field(default_factory=int)
    targets: dict[str, LoraTarget] = field(default_factory=dict)
    module_map: LoraModuleResolver | None = field(default_factory=lambda: None)
    max_slots: int = 4
    max_rank: int = 16
    cpu_capacity: int = 16
    slot_wait_steps: int = 256
    adapters: dict[str, str] = field(default_factory=dict)
    # serve /v1/load_lora_adapter and /v1/unload_lora_adapter, which make the server read
    # client-named paths; off, they answer 404 as vLLM's do without its opt-in
    runtime_updates: bool = False
    # how long the conductor waits for every worker to answer a runtime load or unload
    # before it gives up on it (a load is then undone everywhere)
    runtime_update_timeout_s: float = 120.0
    # adapter name -> the digest (``identity.adapter_digest``) its weights were checked
    # and prefix-keyed under; every read must match it. The API server fills these in for
    # the serving config's adapters, and a runtime load carries its own.
    adapter_digests: dict[str, bytes] = field(default_factory=dict)
    # the slot weights' dtype; None takes the nodes' compute dtype (the serving config's
    # ``autocast_dtype``), which is what their base linears run in
    dtype: torch.dtype | None = None
    # "auto" picks Triton on CUDA and the torch reference elsewhere
    backend: Literal["auto", "torch", "triton"] = "auto"
    convert_keys: Callable | None = None
    # checkpoint modules this spec neither loads nor refuses: the deployment never runs
    # them, or (in a node's pool) another node's pool loads them
    ignored_modules: Callable[[str], bool] | None = field(default_factory=lambda: None)
    # declared targets not served: the experts targets unless the YAML's ``targets``
    # names them (``module_lora_spec`` leaves them out), and any that list leaves out. A
    # module bound to one gets no delta, and a later ``targets`` list may name it again
    dropped_targets: dict[str, LoraTarget] = field(default_factory=dict)
    declare: Callable[[], dict[str, Any]] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        super().__post_init__()
        _check_sizes({name: getattr(self, name) for name in _SIZES})
        if self.declare is not None:
            for name in _DECLARED:
                del self.__dict__[name]  # read through `__getattr__`
            return
        self._check_declaration()

    def __getattr__(self, name: str) -> Any:
        # reached only for an attribute not set: a field ``declare`` gives, read first
        declare = self.__dict__.get("declare")
        if declare is None or name not in _DECLARED:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        self.__dict__.update(declare())
        self.declare = None
        self._check_declaration()
        return self.__dict__[name]

    def _check_declaration(self) -> None:
        declared = {**self.targets, **self.dropped_targets}
        if not declared:
            raise ValueError(f"LoRA resource {self.resource_key!r} declares no targets")
        if len(self.nodes) > 1:
            astray = sorted(name for name, t in declared.items() if t.node not in self.nodes)
            if astray:
                raise ValueError(
                    f"LoRA resource {self.resource_key!r} spans nodes {sorted(self.nodes)}, so "
                    f"each target names one of them; {astray} do not"
                )

    @property
    def num_slots(self) -> int:
        """Device slots, the zero adapter's included."""
        return self.max_slots + 1

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.lora.pool import LoraAdapterPool

        return LoraAdapterPool

    def for_node(self, node: str) -> "LoraSpec | None":
        """``node``'s pool: the targets on it, keyed ``lora@<node>``; None when the
        YAML's ``targets`` left it none. Placement stays the whole spec's, so a pool reads
        an adapter file as the API server checked it, leaving the modules another node's
        targets place to that node's pool."""
        own = {name: t for name, t in self.targets.items() if t.node in (None, node)}
        if not own:
            return None
        elsewhere = frozenset(self.targets.keys() - own.keys())
        ignored = self.ignored_modules
        if elsewhere:
            ignored = _ignoring(ignored, self.module_map, elsewhere)
        dropped = {name: t for name, t in self.dropped_targets.items() if t.node in (None, node)}
        return dataclasses.replace(
            self, resource_key=instance_key(self.resource_key, node), nodes={node},
            targets=own, dropped_targets=dropped, ignored_modules=ignored,
        )

    def request_config(self, model_kwargs: dict | None) -> "LoraReqConfig | None":
        """The adapter the ``lora_adapter`` kwarg names, and the digest it was selected
        under; None for a base-model request, which leaves the pool idle
        (``Resource.idle_capture``)."""
        kwargs = model_kwargs or {}
        adapter = kwargs.get(LORA_ADAPTER_KWARG)
        return None if adapter is None else LoraReqConfig(adapter=adapter, digest=kwargs.get(LORA_DIGEST_KWARG))

    def apply_yaml_overrides(
        self,
        max_slots: int | None = None,
        max_rank: int | None = None,
        cpu_capacity: int | None = None,
        adapters: dict[str, str] | None = None,
        slot_wait_steps: int | None = None,
        adapter_digests: dict[str, str] | None = None,
        runtime_updates: bool | None = None,
        runtime_update_timeout_s: float | None = None,
        targets: list[str] | None = None,
    ):
        """How many adapters stay resident, the largest rank served, which adapters this
        deployment serves, and how long a request may wait for a slot before others are
        held back. ``targets`` serves just those of the model's declared targets, the
        experts targets included (served only when named, ``dropped_targets``): dropping
        the LM head and embedding saves their vocabulary-sized slots, a node left with
        none gets no pool, and an adapter that touches a dropped target is refused.
        Without ``targets`` the served set stays as it is.

        Every value is checked before any is set, so a refused override leaves the spec
        as declared."""
        sizes = {
            name: value
            for name, value in zip(_SIZES, (max_slots, max_rank, cpu_capacity, slot_wait_steps), strict=True)
            if value is not None
        }
        _check_sizes(sizes)
        if runtime_updates is not None and not isinstance(runtime_updates, bool):
            raise ValueError(f"runtime_updates must be true or false; got {runtime_updates!r}")
        if runtime_update_timeout_s is not None and (
            isinstance(runtime_update_timeout_s, bool)
            or not isinstance(runtime_update_timeout_s, int | float)
            or runtime_update_timeout_s <= 0
        ):
            raise ValueError(
                f"runtime_update_timeout_s must be a positive number; got {runtime_update_timeout_s!r}"
            )
        if adapters is not None:
            if not isinstance(adapters, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in adapters.items()
            ):
                raise ValueError(f"adapters must map names to paths; got {adapters!r}")
            # a PEFT directory names its adapter_model.safetensors, as at a runtime load
            adapters = {name: resolve_adapter_path(path) for name, path in adapters.items()}
        if adapter_digests is not None:
            served = self.adapters if adapters is None else adapters
            # hex, as the API server writes them into the config the conductor gets
            adapter_digests = {
                name: bytes.fromhex(digest) for name, digest in adapter_digests.items()
                if name in served
            }
        if targets is not None:
            if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
                raise ValueError(f"targets must be a list of target names; got {targets!r}")
            declared = {**self.targets, **self.dropped_targets}
            unknown = sorted(set(targets) - set(declared))
            if unknown:
                raise ValueError(
                    f"`resources: {self.resource_key}: targets:` names {unknown}, which this "
                    f"model does not declare; it declares {sorted(declared)}"
                )
            kept = {name: t for name, t in declared.items() if name in targets}
            if not kept:
                raise ValueError(
                    f"`resources: {self.resource_key}: targets:` keeps none of {sorted(declared)}"
                )
        check_expert_ids(
            self.resource_key, sizes.get("max_slots", self.max_slots), self.targets if targets is None else kept,
        )

        for name, value in sizes.items():
            setattr(self, name, value)
        if runtime_updates is not None:
            self.runtime_updates = runtime_updates
        if runtime_update_timeout_s is not None:
            self.runtime_update_timeout_s = float(runtime_update_timeout_s)
        if adapters is not None:
            self.adapters = adapters
            self.adapter_digests = {}
        if adapter_digests is not None:
            self.adapter_digests = adapter_digests
        if targets is not None:
            self.dropped_targets = {name: t for name, t in declared.items() if name not in kept}
            self.targets = kept

    def check_adapter(self, name: str, info: AdapterInfo) -> None:
        """Refuse an adapter this deployment cannot serve: no LoRA layers, a rank above
        ``max_rank``, or a module outside the declared targets."""
        if info.rank == 0:
            raise ValueError(f"LoRA adapter {name!r} holds no LoRA layers")
        if info.rank > self.max_rank:
            raise ValueError(
                f"LoRA adapter {name!r} has rank {info.rank}, above this "
                f"deployment's max_rank {self.max_rank}"
            )
        if info.unmapped:
            # a module map built from the model's modules says why one is not a target
            explain = getattr(self.module_map, "explain", None)
            why = [r for r in map(explain, info.unmapped) if r] if explain else []
            refs = (self.module_map(m) for m in info.unmapped) if self.dropped_targets else ()
            off = sorted({ref.target for ref in refs if ref is not None and ref.target in self.dropped_targets})
            if off:
                note = f"targets {off} are left out of this deployment's `resources: {self.resource_key}: targets:`"
                if any(self.dropped_targets[t].experts > 1 for t in off):
                    note += ", and an experts target is served only when that list names it"
                why.insert(0, note)
            raise ValueError(
                f"LoRA adapter {name!r} touches modules this deployment does not "
                f"serve LoRA on: {_some(info.unmapped)}. "
                f"Its LoRA targets are {sorted(self.targets)}; anything outside them is refused"
                + (f" ({_some(tuple(why), 3)})" if why else "")
            )
        if info.misfits:
            raise ValueError(
                f"LoRA adapter {name!r} does not fit this model: {_some(info.misfits)}; "
                "it was trained on another size of the model, or lays a module out "
                "differently from the target it names"
            )


# what ``LoraSpec.declare`` gives
_DECLARED = ("num_layers", "targets", "dropped_targets", "module_map", "ignored_modules")

# the fused-MoE align op (``moe_align_block_size.cu``) groups, in one thread block, fewer
# than this many ids once padded to its WARP_SIZE
_ALIGN_MAX_IDS = 1024
_ALIGN_WARP = 32


def check_expert_ids(resource_key: str, max_slots: int, targets: Mapping[str, LoraTarget]) -> None:
    """Refuse an experts target with more (slot, expert) ids, ``max_slots`` slots of its
    experts each, than the align op groups."""
    for name, target in targets.items():
        ids = max_slots * target.experts
        if target.experts == 1 or -(-ids // _ALIGN_WARP) * _ALIGN_WARP < _ALIGN_MAX_IDS:
            continue
        raise ValueError(
            f"LoRA resource {resource_key!r}: target {name!r} holds {target.experts} experts "
            f"per slot, so max_slots {max_slots} gives {ids} (slot, expert) ids, more than the "
            f"fused-MoE align op groups (fewer than {_ALIGN_MAX_IDS}, padded to {_ALIGN_WARP}); "
            f"serve at most {(_ALIGN_MAX_IDS - _ALIGN_WARP) // target.experts} slots "
            f"(`resources: {LORA_RESOURCE_KEY}: max_slots:`), or leave the experts targets "
            "out of its `targets:`"
        )


def _ignoring(
    never_run: Callable[[str], bool] | None, place: LoraModuleResolver, elsewhere: frozenset[str],
) -> Callable[[str], bool]:
    """``LoraSpec.ignored_modules`` of one node's pool: what the spec ignores, and the
    modules ``place`` puts on ``elsewhere``, the other nodes' targets."""

    def ignored(module: str) -> bool:
        if never_run is not None and never_run(module):
            return True
        ref = place(module)
        return ref is not None and ref.target in elsewhere

    return ignored


_SIZES = ("max_slots", "max_rank", "cpu_capacity", "slot_wait_steps")


def _check_sizes(sizes: Mapping[str, Any]) -> None:
    for name, value in sizes.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1; got {value!r}")


def _some(items: tuple[str, ...], n: int = 5) -> str:
    more = f" and {len(items) - n} more" if len(items) > n else ""
    return f"{list(items[:n])}{more}"


@dataclass
class LoraReqConfig(ResourceReqConfig):
    """The adapter a request runs under. A base-model request carries none.

    Which adapters are served is the API server's to check (``ServedModels``), where a
    request names one; a pool refuses at ingest a name it does not serve, or serves under
    other weights than ``digest``, which the request's prompt was keyed under: the name
    was unloaded and loaded again since."""

    adapter: str
    digest: bytes | None = None

    def validate(self, spec: "LoraSpec | None" = None) -> None:
        if not isinstance(self.adapter, str) or not self.adapter:
            raise ValueError(f"adapter must be a non-empty string; got {self.adapter!r}")
