"""The host tier of the adapter pool: adapters read, sharded for this rank, and pinned.

An adapter's tensors are laid out exactly like one device slot (rank padded to the pool's
``max_rank`` with zeros), so moving it onto the device is one copy per target.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Collection
from dataclasses import dataclass

import torch

from mstar.engine.resources.lora.checkpoint import normalize_lora_key, parse_lora_key, read_lora_file
from mstar.engine.resources.lora.config import (
    AdapterInfo,
    LoraSpec,
    LoraTarget,
)
from mstar.engine.resources.lora.identity import adapter_digest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedAdapter:
    """One adapter, ready to copy into a slot.

    ``a[target]`` is ``[target layers, slices * max_rank, in]`` and ``b[target]`` is
    ``[target layers, out, max_rank]``, both already this rank's shard; an experts target's
    hold a pair per expert, ``[target layers, experts, ...]``. A target the adapter does not
    touch is absent. ``layer_ranks[target]`` (int32 ``[target layers]``) is the slot's rank
    at each layer: ``rank``, or 0 where this rank's shard of the layer's A or B is all
    zero (it adds nothing), as the kernels read it. ``digest`` is ``adapter_digest`` of
    the unsharded weights, so every rank agrees on it, computed as the API server computes
    the one it roots the adapter's prefix-cache keys in.
    """

    name: str
    digest: bytes
    rank: int
    scaling: float
    a: dict[str, torch.Tensor]
    b: dict[str, torch.Tensor]
    layer_ranks: dict[str, torch.Tensor]


def _maps(spec: LoraSpec, module: str) -> bool:
    ref = spec.module_map(module)
    if ref is None or ref.target not in spec.targets:
        return False
    target = spec.targets[ref.target]
    if ref.layer not in _target_layers(spec, target):
        return False
    try:
        target.module_slices(ref.shard_id)
        target.expert_of(ref.shard_id)
    except KeyError:
        return False
    return True


def _ignored(spec: LoraSpec, module: str) -> bool:
    return spec.ignored_modules is not None and spec.ignored_modules(module)


def _target_layers(spec: LoraSpec, target: LoraTarget) -> tuple[int, ...]:
    return target.layers if target.layers is not None else tuple(range(spec.num_layers))


def as_linear(shape) -> tuple[int, ...]:
    """A LoRA factor's shape as a linear's: a conv adapter's ``A [r, c, *kernel]`` and
    ``B [out, r, 1, ...]`` flatten to ``[r, c * prod(kernel)]`` and ``[out, r]``, which is
    the delta on a conv whose kernel is its stride, over the flattened patch."""
    shape = tuple(shape)
    if len(shape) <= 2:
        return shape
    rest = 1
    for n in shape[1:]:
        rest *= n
    return shape[0], rest


def misfit(spec: LoraSpec, module: str, a_shape, b_shape) -> str | None:
    """Why a mapped module's ``A`` / ``B`` shapes do not fit its target, or None."""
    ref = spec.module_map(module)
    target = spec.targets[ref.target]
    rows = sum(target.out_slices[i] for i in target.module_slices(ref.shard_id))
    (r, k), (out, r_b) = as_linear(a_shape), as_linear(b_shape)
    if k == target.in_features and out == rows and r_b == r:
        return None
    return (
        f"{module} (A {tuple(a_shape)}, B {tuple(b_shape)}; target {ref.target!r} is "
        f"{target.in_features} -> {rows})"
    )


def adapter_targets(spec: LoraSpec, path: str) -> set[str]:
    """The targets of ``spec`` the checkpoint at ``path`` holds LoRA weights for."""
    if spec.convert_keys is None:
        from safetensors import safe_open

        with safe_open(path, "pt") as f:
            keys = list(f.keys())
        modules = set()
        for key in keys:
            modules.add(parse_lora_key(normalize_lora_key(key))[0])
    else:
        modules = set(read_lora_file(path, convert_keys=spec.convert_keys).layers)
    return {
        spec.module_map(module).target for module in modules
        if not _ignored(spec, module) and _maps(spec, module)
    }


def classify_adapter(spec: LoraSpec, shapes: dict[str, dict[str, tuple[int, ...]]]) -> AdapterInfo:
    """Rank, unmapped and misfitting modules of an adapter, from each module's A and B
    shapes (both present, as ``read_lora_file`` requires)."""
    unmapped, misfits = [], []
    for module, ab in shapes.items():
        if ab["A"][0] == 0:
            # its alpha / r has no value to fold into the slot's one scaling
            raise ValueError(f"LoRA module {module} has rank 0")
        if _ignored(spec, module):
            continue
        if not _maps(spec, module):
            unmapped.append(module)
        elif (error := misfit(spec, module, ab["A"], ab["B"])) is not None:
            misfits.append(error)
    return AdapterInfo(
        rank=max((ab["A"][0] for ab in shapes.values()), default=0),
        unmapped=tuple(sorted(unmapped)),
        misfits=tuple(sorted(misfits)),
    )


def check_adapter_file(spec: LoraSpec, name: str, path: str) -> bytes:
    """The adapter at ``path`` checked against ``spec`` and its digest, which a worker's
    read holds the file to; the digest is of the read that was checked. Raises
    ``ValueError`` for an adapter the deployment cannot serve."""
    raw = read_lora_file(path, convert_keys=spec.convert_keys)
    spec.check_adapter(name, classify_adapter(spec, {
        module: {"A": tuple(layer.a.shape), "B": tuple(layer.b.shape)}
        for module, layer in raw.layers.items()
    }))
    return adapter_digest(raw)


def check_adapters(spec: LoraSpec) -> dict[str, bytes]:
    """Every adapter ``spec`` serves, through ``check_adapter_file``; returns each one's
    digest.

    Raises naming every adapter that fails, so a bad adapter in the serving config stops
    the deployment at startup, the same whether or not a worker would have preloaded it.
    """
    digests: dict[str, bytes] = {}
    errors: list[str] = []
    for name, path in spec.adapters.items():
        try:
            digests[name] = check_adapter_file(spec, name, path)
        except Exception as exc:  # noqa: BLE001 - each bad adapter is named below
            errors.append(f"{name!r} at {path}: {exc}")
    if errors:
        raise ValueError(
            "the serving config's LoRA adapters cannot be served: " + "; ".join(errors)
        )
    return digests


class LoraAdapterStore:
    """Adapters held in pinned host memory, least recently used dropped past ``capacity``."""

    def __init__(
        self, spec: LoraSpec, rank: int = 0, world_size: int = 1, pin: bool | None = None,
        dtype: torch.dtype | None = None,
    ):
        self._spec = spec
        # the device slots' dtype, so a slot load is a plain copy
        self._dtype = dtype or spec.dtype or torch.float32
        self._rank = rank
        self._world_size = world_size
        self._pin = torch.accelerator.is_available() if pin is None else pin
        self._sharded = {name: t.shard(world_size) for name, t in spec.targets.items()}
        # stack layer -> row of the target's storage
        self._layer_rows = {
            name: {layer: row for row, layer in enumerate(_target_layers(spec, t))}
            for name, t in spec.targets.items()
        }
        self._adapters: OrderedDict[str, LoadedAdapter] = OrderedDict()

    def __contains__(self, name: str) -> bool:
        return name in self._adapters

    def __len__(self) -> int:
        return len(self._adapters)

    def get(self, name: str) -> LoadedAdapter | None:
        adapter = self._adapters.get(name)
        if adapter is not None:
            self._adapters.move_to_end(name)
        return adapter

    def load(self, name: str, path: str, digest: bytes) -> LoadedAdapter:
        """The adapter under ``name``, read from ``path`` unless it is already held."""
        adapter = self.get(name)
        if adapter is not None:
            return adapter
        return self.put(self.read(name, path, digest))

    def read(self, name: str, path: str, digest: bytes) -> LoadedAdapter:
        """``path`` read and sharded for this rank, without holding it: a caller can read
        outside whatever lock guards ``put``. ``digest`` is of the weights
        ``check_adapter_file`` checked against the spec, so a read that digests the same
        holds checked weights; one that digests otherwise is refused before anything is
        staged: the file changed since the adapter was checked and keyed, and serving it
        would put other weights under the same prefix-cache root, or different weights
        on different ranks."""
        spec = self._spec
        raw = read_lora_file(path, convert_keys=spec.convert_keys)
        read_digest = adapter_digest(raw)
        if read_digest != digest:
            raise ValueError(
                f"LoRA adapter {name!r} changed on disk since it was loaded (digest "
                f"{read_digest.hex()[:16]}, expected {digest.hex()[:16]}); load it again"
            )
        # modules another node's pool loads (or the deployment never runs) are not this one's
        modules = sorted(m for m in raw.layers if not _ignored(spec, m))
        rank = max((raw.layers[m].a.shape[0] for m in modules), default=0)
        # One scaling per slot; a module whose alpha / r differs has the ratio folded
        # into its B (PEFT's rank_pattern / alpha_pattern, Kohya's per-module alphas).
        # The slot's is the first nonzero one, so the fold never divides by zero (an
        # alpha of 0 zeroes its module's B). An adapter with nothing for this spec is
        # rank 0 here, which the kernels skip.
        scales = [raw.layers[m].alpha / raw.layers[m].a.shape[0] for m in modules]
        scaling = next((s for s in scales if s != 0), 0.0)

        R = spec.max_rank
        a_host: dict[str, torch.Tensor] = {}
        b_host: dict[str, torch.Tensor] = {}
        # [target layers] bool: whether this rank's shard of the layer's A / B is nonzero
        a_live: dict[str, torch.Tensor] = {}
        b_live: dict[str, torch.Tensor] = {}
        for module in modules:
            layer = raw.layers[module]
            ref = spec.module_map(module)
            target = spec.targets[ref.target]
            sharded = self._sharded[ref.target]
            if ref.target not in a_host:
                # in the slots' dtype and pinned from the start: an experts target's
                # tensors are GBs, which a staged copy would hold twice
                lead = (len(self._layer_rows[ref.target]),)
                if target.experts > 1:
                    lead += (target.experts,)
                a_host[ref.target] = torch.zeros(
                    *lead, sharded.num_slices * R, sharded.in_features, dtype=self._dtype, pin_memory=self._pin,
                )
                b_host[ref.target] = torch.zeros(
                    *lead, sharded.out_features, R, dtype=self._dtype, pin_memory=self._pin,
                )
                a_live[ref.target] = torch.zeros(lead[0], dtype=torch.bool)
                b_live[ref.target] = torch.zeros(lead[0], dtype=torch.bool)
            # the layer's row of the target's storage, and the expert's in it
            at = (self._layer_rows[ref.target][ref.layer],)
            if target.experts > 1:
                at += (target.expert_of(ref.shard_id),)
            a = layer.a.float().reshape(as_linear(layer.a.shape))
            b = layer.b.float().reshape(as_linear(layer.b.shape))
            r = a.shape[0]
            module_scaling = layer.alpha / r
            if module_scaling != scaling:
                b = b * (module_scaling / scaling)
            # a packed module's one A goes to each of its slices, its B rows to theirs.
            # The shrink then computes the same x @ A^T once per slice (3x for a
            # delta-net's in_proj_qkv); sharing it needs the kernels to map slices to
            # A rows, not done yet.
            slices = target.module_slices(ref.shard_id)
            widths = [target.out_slices[i] for i in slices]
            for i, b_i in zip(slices, b.split(widths), strict=True):
                a_r, b_r = target.shard_pair(i, a, b_i, self._rank, self._world_size)
                # past b_r's rows, a slice's alignment padding (if any) stays zero
                offset = sum(sharded.out_slices[:i])
                a_host[ref.target][(*at, slice(i * R, i * R + r))] = a_r
                b_host[ref.target][(*at, slice(offset, offset + b_r.shape[0]), slice(0, r))] = b_r
                a_live[ref.target][at[0]] |= bool(a_r.any())
                b_live[ref.target][at[0]] |= bool(b_r.any())

        logger.info(
            "LoRA store: read %s (rank %d, %d modules, scaling %.4g)", name, rank, len(modules), scaling,
        )
        return LoadedAdapter(
            name=name,
            digest=read_digest,
            rank=rank,
            scaling=scaling,
            a=a_host,
            b=b_host,
            layer_ranks={t: _pinned((a_live[t] & b_live[t]).to(torch.int32) * rank, self._pin) for t in a_host},
        )

    def put(self, adapter: LoadedAdapter, keep: Collection[str] = ()) -> LoadedAdapter:
        """Hold ``adapter``, dropping the least recently used past ``capacity``, but none
        named in ``keep`` (which may hold the tier past capacity for a while)."""
        self._adapters[adapter.name] = adapter
        self._adapters.move_to_end(adapter.name)
        excess = len(self._adapters) - self._spec.cpu_capacity
        if excess > 0:
            for dropped in [n for n in self._adapters if n not in keep and n != adapter.name][:excess]:
                del self._adapters[dropped]
                logger.info("LoRA store: dropped %s from the host tier", dropped)
        return adapter

    def drop(self, name: str) -> None:
        self._adapters.pop(name, None)


def _pinned(t: torch.Tensor, pin: bool) -> torch.Tensor:
    return t.pin_memory() if pin else t
