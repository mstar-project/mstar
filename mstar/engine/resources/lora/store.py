"""The host tier of the adapter pool: adapters read, sharded for this rank, and pinned.

An adapter's tensors are laid out exactly like one device slot (rank padded to the pool's
``max_rank`` with zeros), so moving it onto the device is one copy per target.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Collection, Sequence
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
    ``[target layers, out, max_rank]``, both already this rank's shard. A target the adapter
    does not touch is absent. ``digest`` is ``adapter_digest`` of the unsharded weights, so
    every rank agrees on it, computed as the API server computes the one it roots the
    adapter's prefix-cache keys in.
    """

    name: str
    digest: bytes
    rank: int
    scaling: float
    a: dict[str, torch.Tensor]
    b: dict[str, torch.Tensor]


def _maps(spec: LoraSpec, module: str) -> bool:
    ref = spec.module_map(module)
    if ref is None or ref.target not in spec.targets:
        return False
    target = spec.targets[ref.target]
    if ref.layer not in _target_layers(spec, target):
        return False
    try:
        target.module_slices(ref.shard_id)
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


def inspect_adapter(spec: LoraSpec, path: str) -> AdapterInfo:
    """Rank, unmapped and misfitting modules of the checkpoint at ``path``.

    Reads only the safetensors header unless the model converts a native layout, which
    needs the tensors.
    """
    if spec.convert_keys is None:
        from safetensors import safe_open

        with safe_open(path, "pt") as f:
            return inspect_keys(spec, ((key, f.get_slice(key).get_shape()) for key in f.keys()))
    adapter = read_lora_file(path, convert_keys=spec.convert_keys)
    return _classify(spec, {
        module: {"A": tuple(layer.a.shape), "B": tuple(layer.b.shape)}
        for module, layer in adapter.layers.items()
    })


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


def inspect_keys(spec: LoraSpec, items) -> AdapterInfo:
    """``inspect_adapter`` over ``(checkpoint key, shape)`` pairs, as a header lists them."""
    shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    for key, shape in items:
        # a key no plain LoRA has (DoRA, modules_to_save, ...) raises a refusal, named
        module, kind = parse_lora_key(normalize_lora_key(key))
        if kind != "alpha":
            shapes.setdefault(module, {})[kind] = tuple(shape)
    return _classify(spec, shapes)


def _classify(spec: LoraSpec, shapes: dict[str, dict[str, tuple[int, ...]]]) -> AdapterInfo:
    unmapped, misfits = [], []
    for module, ab in shapes.items():
        if _ignored(spec, module):
            continue
        if not _maps(spec, module):
            unmapped.append(module)
        elif "A" in ab and "B" in ab:
            error = misfit(spec, module, ab["A"], ab["B"])
            if error is not None:
                misfits.append(error)
    return AdapterInfo(
        rank=max((ab["A"][0] for ab in shapes.values() if "A" in ab), default=0),
        unmapped=tuple(sorted(unmapped)),
        misfits=tuple(sorted(misfits)),
    )


def check_adapter_file(
    spec: LoraSpec, name: str, path: str, others: Sequence[LoraSpec] = (),
) -> bytes:
    """The adapter at ``path`` checked against ``spec`` and the model's ``others`` (its
    LoRA specs for other nodes, a vision tower's say, which serve the same adapters), each
    from the checkpoint's header; returns its digest. The digest is read once per way the
    specs convert the checkpoint (``convert_keys``), so once for specs that read it alike.
    Raises ``ValueError`` for an adapter the deployment cannot serve."""
    digests: dict = {}  # convert_keys -> the digest of the file read that way
    for each in (spec, *others):
        each.check_adapter(name, inspect_adapter(each, path))
        if each.convert_keys not in digests:
            digests[each.convert_keys] = adapter_digest(read_lora_file(path, convert_keys=each.convert_keys))
    digest, *rest = digests.values()
    # one digest pins the adapter on every spec's pools (``adapter_digests``)
    if any(other != digest for other in rest):
        raise ValueError(
            f"LoRA adapter {name!r} digests differently under the model's LoRA specs; "
            "they must read checkpoints alike (convert_keys)"
        )
    return digest


def check_adapters(spec: LoraSpec, others: Sequence[LoraSpec] = ()) -> dict[str, bytes]:
    """Every adapter ``spec`` serves, through ``check_adapter_file`` (against ``others``
    too); returns each one's digest.

    Raises naming every adapter that fails, so a bad adapter in the serving config stops
    the deployment at startup, the same whether or not a worker would have preloaded it.
    """
    digests: dict[str, bytes] = {}
    errors: list[str] = []
    for name, path in spec.adapters.items():
        try:
            digests[name] = check_adapter_file(spec, name, path, others)
        except Exception as exc:  # noqa: BLE001 - each bad adapter is named below
            errors.append(f"{name!r} at {path}: {exc}")
    if errors:
        raise ValueError(
            "the serving config's LoRA adapters cannot be served: " + "; ".join(errors)
        )
    return digests


class LoraAdapterStore:
    """Adapters held in pinned host memory.

    Which adapters it holds follows from ``hold`` calls alone, in their order: each makes
    an adapter the most recent, dropping the least recent past ``capacity``. A held
    adapter's copy arrives later, by ``put``, once read. So when a read lands or fails
    never changes what is held, and ranks that hold in the same order hold the same set.
    """

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
        # held adapters, least recent first; None until a held adapter's copy is put
        self._adapters: OrderedDict[str, LoadedAdapter | None] = OrderedDict()

    def __contains__(self, name: str) -> bool:
        """Whether ``name``'s copy is in."""
        return self._adapters.get(name) is not None

    def get(self, name: str) -> LoadedAdapter | None:
        return self._adapters.get(name)

    def hold(self, name: str, keep: Collection[str] = ()) -> bool:
        """Make ``name`` the most recent adapter held, dropping the least recent past
        ``capacity``, but none named in ``keep`` (which may hold the tier past capacity for
        a while). True when it was not held, so its copy is the caller's to read in."""
        new = name not in self._adapters
        if new:
            self._adapters[name] = None
        self._adapters.move_to_end(name)
        excess = len(self._adapters) - self._spec.cpu_capacity
        if excess > 0:
            for dropped in [n for n in self._adapters if n not in keep and n != name][:excess]:
                del self._adapters[dropped]
                logger.info("LoRA store: dropped %s from the host tier", dropped)
        return new

    def put(self, adapter: LoadedAdapter) -> bool:
        """``adapter``'s copy, kept if it is still held."""
        if adapter.name not in self._adapters:
            return False
        self._adapters[adapter.name] = adapter
        return True

    def load(self, name: str, path: str, digest: bytes | None = None) -> LoadedAdapter:
        """Hold ``name`` and read it from ``path`` unless its copy is in."""
        self.hold(name)
        adapter = self.get(name)
        if adapter is None:
            adapter = self.read(name, path, digest)
            self.put(adapter)
        return adapter

    def read(self, name: str, path: str, digest: bytes | None = None) -> LoadedAdapter:
        """``path`` read, checked against the spec and sharded for this rank, without
        holding it: a caller can read outside whatever lock guards ``put``. With
        ``digest``, weights that digest otherwise are refused: the file changed since the
        adapter was checked and keyed, and serving it would put other weights under the
        same prefix-cache root, or different weights on different ranks."""
        adapter = self._read(name, path)
        if digest is not None and adapter.digest != digest:
            raise ValueError(
                f"LoRA adapter {name!r} changed on disk since it was loaded (digest "
                f"{adapter.digest.hex()[:16]}, expected {digest.hex()[:16]}); load it again"
            )
        return adapter

    def drop(self, name: str) -> None:
        self._adapters.pop(name, None)

    def _read(self, name: str, path: str) -> LoadedAdapter:
        spec = self._spec
        raw = read_lora_file(path, convert_keys=spec.convert_keys)
        if not raw.layers:
            raise ValueError(f"LoRA adapter {name!r} holds no LoRA layers")
        # modules another spec loads (or the deployment never runs) are not this one's
        modules = sorted(m for m in raw.layers if not _ignored(spec, m))
        unmapped = [m for m in modules if not _maps(spec, m)]
        if unmapped:
            raise ValueError(
                f"LoRA adapter {name!r} touches modules with no declared target: {unmapped[:5]}"
            )
        rank = max((raw.layers[m].a.shape[0] for m in modules), default=0)
        if rank > spec.max_rank:
            raise ValueError(
                f"LoRA adapter {name!r} has rank {rank}, above max_rank {spec.max_rank}"
            )
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
        for module in modules:
            layer = raw.layers[module]
            ref = spec.module_map(module)
            target = spec.targets[ref.target]
            sharded = self._sharded[ref.target]
            if ref.target not in a_host:
                rows = len(self._layer_rows[ref.target])
                a_host[ref.target] = torch.zeros(
                    rows, sharded.num_slices * R, sharded.in_features, dtype=torch.float32,
                )
                b_host[ref.target] = torch.zeros(
                    rows, sharded.out_features, R, dtype=torch.float32,
                )
            row = self._layer_rows[ref.target][ref.layer]
            error = misfit(spec, module, layer.a.shape, layer.b.shape)
            if error is not None:
                raise ValueError(f"LoRA adapter {name!r} does not fit this model: {error}")
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
                a_host[ref.target][row, i * R:i * R + r] = a_r
                b_host[ref.target][row, offset:offset + b_r.shape[0], :r] = b_r

        def host(t: torch.Tensor) -> torch.Tensor:
            t = t.to(self._dtype)
            return t.pin_memory() if self._pin else t

        logger.info(
            "LoRA store: read %s (rank %d, %d modules, scaling %.4g)", name, rank, len(modules), scaling,
        )
        return LoadedAdapter(
            name=name,
            digest=adapter_digest(raw),
            rank=rank,
            scaling=scaling,
            a={t: host(v) for t, v in a_host.items()},
            b={t: host(v) for t, v in b_host.items()},
        )
