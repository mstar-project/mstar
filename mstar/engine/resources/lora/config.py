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
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import torch

from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource

# The resource key a model's ``LoraSpec`` is declared under, and so the serving YAML's
# ``resources:`` block that turns LoRA on (see ``mstar.engine.resources.lora.serving``).
LORA_RESOURCE_KEY = "lora"
# The request kwarg that names a request's adapter; the OpenAI ``model`` field maps onto it.
LORA_ADAPTER_KWARG = "lora_adapter"


def lora_enabled(model_config: Mapping[str, Any]) -> bool:
    """Whether a deployment's YAML turns LoRA on (a ``resources: lora:`` block). A
    ``resources:`` value that is not a mapping turns nothing on: the conductor's
    ``validate_deployment`` reports it with the deployment's other problems."""
    resources = model_config.get("resources")
    return isinstance(resources, Mapping) and LORA_RESOURCE_KEY in resources


def served_lora_adapters(model_config: Mapping[str, Any]) -> tuple[str, ...]:
    """The adapter names a deployment's YAML serves; empty when it does not turn LoRA on."""
    block = model_config["resources"][LORA_RESOURCE_KEY] if lora_enabled(model_config) else None
    return tuple(block.get("adapters") or ()) if isinstance(block, Mapping) else ()


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

    ``kind="embedding"`` is a token embedding: ``in_features`` is the vocabulary, a row's
    input is its token id, and the shrink gathers A's column for it (``x @ A^T`` of the
    token's one-hot), as PEFT's ``lora_embedding_A`` / ``_B`` define it.

    Which rows the linear sees, for the per-row adapter choice: one per token of the step
    (the default), one per segment (``per_segment``, e.g. an LM head over each row's last
    token), or one per ``token_fold`` consecutive tokens of a segment (a patch merger that
    folds 2x2 patches into one row).
    """

    in_features: int
    out_slices: tuple[int, ...]
    parallel: Literal["column", "row", "none"] = "column"
    slice_names: tuple[str | int, ...] = ()
    slice_units: tuple[int, ...] | None = None
    slice_packs: tuple[tuple[str | int, tuple[str | int, ...]], ...] | None = None
    slice_align: tuple[int, ...] | None = None
    layers: tuple[int, ...] | None = None
    kind: Literal["linear", "embedding"] = "linear"
    per_segment: bool = False
    token_fold: int = 1

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

    def slice_index(self, shard_id: str | int | None) -> int:
        """Which slice a checkpoint module lands on, from its stacked-shard id."""
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


def _divide(numerator: int, denominator: int) -> int:
    if numerator % denominator:
        raise ValueError(f"{numerator} is not divisible by {denominator}")
    return numerator // denominator


class LoraModuleRef(NamedTuple):
    """Where one checkpoint module's adapter lands: a target, a stack layer, and the
    stacked-shard id of its slice (None for a single-slice target)."""

    target: str
    layer: int
    shard_id: str | int | None = None


# checkpoint module path -> where it lands, or None for a module the model does not know
LoraModuleResolver = Callable[[str], "LoraModuleRef | None"]


class AdapterInfo(NamedTuple):
    """What validation needs to know about an adapter checkpoint, read without its weights.

    ``rank`` covers every LoRA module in the file, those left to another spec included.
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
    ``adapters`` maps the names requests ask for to safetensors paths. ``module_map``
    places each checkpoint module on a target; a node's spec comes from
    ``module_lora_spec`` (``mstar.model.components.lora.modules``), whose
    ``ModuleLoraMap`` is what binds the node's modules (a spec with any other resolver
    is refused as the deployment's specs are declared, ``declared_lora_specs``). ``convert_keys`` rewrites a model-native checkpoint
    layout first.
    """

    num_layers: int
    targets: dict[str, LoraTarget]
    module_map: LoraModuleResolver
    max_slots: int = 4
    max_rank: int = 16
    cpu_capacity: int = 16
    adapters: dict[str, str] = field(default_factory=dict)
    # adapter name -> the digest (``identity.adapter_digest``) its weights were checked
    # and prefix-keyed under; every read must match it. The conductor fills these in for
    # the serving config's adapters, and a runtime load carries its own.
    adapter_digests: dict[str, bytes] = field(default_factory=dict)
    # the slot weights' dtype; None takes the nodes' compute dtype (the serving config's
    # ``autocast_dtype``), which is what their base linears run in
    dtype: torch.dtype | None = None
    # the kernels behind the custom ops: "auto" registers Triton's on CUDA and leaves the
    # torch reference elsewhere; "torch" registers none, so the reference runs unless
    # something else in the process registered Triton's (registration is process-wide)
    backend: Literal["auto", "torch", "triton"] = "auto"
    convert_keys: Callable | None = None
    # checkpoint modules this spec neither loads nor refuses: another node's LoRA spec of
    # the same model loads them (a vision tower's), or the deployment never runs them
    ignored_modules: Callable[[str], bool] | None = None
    # targets the YAML's ``targets`` list dropped: a layer binding one gets no delta,
    # while binding a name the spec never declared stays an error
    dropped_targets: frozenset[str] = field(default_factory=frozenset)
    _info: dict[str, AdapterInfo] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )

    def __post_init__(self):
        super().__post_init__()
        if not self.targets:
            raise ValueError(f"LoRA resource {self.resource_key!r} declares no targets")
        _check_sizes({name: getattr(self, name) for name in _SIZES})

    @property
    def num_slots(self) -> int:
        """Device slots, the zero adapter's included."""
        return self.max_slots + 1

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.lora.pool import LoraAdapterPool

        return LoraAdapterPool

    def apply_yaml_overrides(
        self,
        max_slots: int | None = None,
        max_rank: int | None = None,
        cpu_capacity: int | None = None,
        adapters: dict[str, str] | None = None,
        adapter_digests: dict[str, str] | None = None,
        targets: list[str] | None = None,
    ):
        """How many adapters stay resident, the largest rank served, and which adapters
        this deployment serves. ``targets`` keeps only those of the model's targets (one
        list for every LoRA spec of the model, so a name another spec declares is not an
        error here; ``serving_resources`` checks the list as a whole): dropping the LM
        head and embedding saves their vocabulary-sized slots, and an adapter that
        touches a dropped target is refused.

        Every value is checked before any is set, so a refused override leaves the spec
        as declared."""
        sizes = {
            name: value
            for name, value in zip(_SIZES, (max_slots, max_rank, cpu_capacity), strict=True)
            if value is not None
        }
        _check_sizes(sizes)
        if adapters is not None:
            if not isinstance(adapters, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in adapters.items()
            ):
                raise ValueError(f"adapters must map names to paths; got {adapters!r}")
            # a PEFT directory names its adapter_model.safetensors, as at a runtime load
            adapters = {name: resolve_adapter_path(path) for name, path in adapters.items()}
        if adapter_digests is not None:
            served = self.adapters if adapters is None else adapters
            # hex, as the conductor writes them into the config it spawns workers with
            adapter_digests = {
                name: bytes.fromhex(digest) for name, digest in adapter_digests.items()
                if name in served
            }
        if targets is not None and (
            not isinstance(targets, list) or not all(isinstance(t, str) for t in targets)
        ):
            raise ValueError(f"targets must be a list of target names; got {targets!r}")

        for name, value in sizes.items():
            setattr(self, name, value)
        if adapters is not None:
            self.adapters = adapters
            self.adapter_digests = {}
            self._info.clear()
        if adapter_digests is not None:
            self.adapter_digests = adapter_digests
        if targets is not None:
            self.dropped_targets = self.dropped_targets | (set(self.targets) - set(targets))
            self.targets = {name: t for name, t in self.targets.items() if name in targets}
            self._info.clear()

    def stop_ignoring(self, modules: Callable[[str], bool]) -> None:
        """Refuse, rather than ignore, the checkpoint modules ``modules`` matches: those
        of another node's spec the deployment does not build, which no spec loads now."""
        if self.ignored_modules is None:
            return
        ignored = self.ignored_modules
        self.ignored_modules = lambda m: ignored(m) and not modules(m)
        self._info.clear()

    def adapter_info(self, name: str) -> AdapterInfo:
        """Rank and unmapped modules of a configured adapter, read once and cached."""
        info = self._info.get(name)
        if info is None:
            from mstar.engine.resources.lora.store import inspect_adapter

            info = self._info[name] = inspect_adapter(self, self.adapters[name])
        return info

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

    def add_adapter(self, name: str, path: str, digest: bytes | None = None) -> None:
        """Serve ``name`` from ``path``, whose weights digest to ``digest``, from now on
        (a runtime load)."""
        self.adapters[name] = path
        if digest is not None:
            self.adapter_digests[name] = digest
        self._info.pop(name, None)

    def remove_adapter(self, name: str) -> None:
        """Stop serving ``name`` (a runtime unload)."""
        self.adapters.pop(name, None)
        self.adapter_digests.pop(name, None)
        self._info.pop(name, None)


_SIZES = ("max_slots", "max_rank", "cpu_capacity")


def _check_sizes(sizes: Mapping[str, Any]) -> None:
    for name, value in sizes.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1; got {value!r}")


def _some(items: tuple[str, ...], n: int = 5) -> str:
    more = f" and {len(items) - n} more" if len(items) > n else ""
    return f"{list(items[:n])}{more}"


@dataclass
class LoraReqConfig(ResourceReqConfig):
    """The adapter a request runs under; None runs the base model."""

    adapter: str | None = None

    def validate(self, spec: "LoraSpec | None" = None) -> None:
        if self.adapter is None:
            return
        if not isinstance(self.adapter, str) or not self.adapter:
            raise ValueError(f"adapter must be a non-empty string; got {self.adapter!r}")
        if spec is None:
            return
        if self.adapter not in spec.adapters:
            raise ValueError(
                f"unknown LoRA adapter {self.adapter!r}; this deployment serves "
                f"{sorted(spec.adapters)}"
            )
        spec.check_adapter(self.adapter, spec.adapter_info(self.adapter))


@dataclass(frozen=True)
class LoraStep(ResourceStep):
    """One step's rows: a segment per request, its ``span`` the tokens that request puts
    through the LoRA'd linears this step, in the order they are packed."""


@dataclass(frozen=True)
class LoraBaseOnly:
    """The capture key of a step none of whose rows names an adapter: the node's own
    ``cg_key_info`` (``key``), wrapped. A node serving adapters captures each bucket under
    both keys, and the base-only graph records no LoRA kernel: at decode width the
    launches of a delta that adds zero cost about what the work does."""

    key: Any = None

    def __str__(self) -> str:
        return "lora-base-only" if self.key is None else f"lora-base-only:{self.key}"


def node_capture_key(key: Any) -> Any:
    """The node's own ``cg_key_info`` in a capture key, without its ``LoraBaseOnly``."""
    return key.key if isinstance(key, LoraBaseOnly) else key
