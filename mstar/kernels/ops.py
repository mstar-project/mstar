"""Kernel ops: one declaration per kernel.

A ``KernelOp`` is one ``mstar::<name>`` operator, declared once with

- its schema and fake implementation, so a traced caller sees one opaque
  node whatever runs underneath;
- a reference implementation in plain torch: what runs for CPU tensors and on
  devices without an implementation, and what implementations are checked
  against (``mstar.kernels.numerics``);
- implementation factories per platform (``cuda``, ``rocm``, ``xpu``,
  ``default``) and per architecture (``rocm:gfx950``), picked once per
  process by a ``PlatformDispatch``. A factory does its own imports and JIT
  builds, so declaring an op loads no kernel library and touches no device;
- the shapes it is called with (``ShapePolicy``) and the bar its
  implementations meet (``Tolerance``).

Resources and layers keep deciding which op runs (invariants 1 and 8); the
registry guarantees an op exists on this platform, or says why not, before
anything is captured. ``python -m mstar.kernels`` lists every op.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch

from mstar.kernels import configs
from mstar.kernels.numerics import Tolerance
from mstar.platforms import current_platform
from mstar.platforms.dispatch import Factory, PlatformDispatch

logger = logging.getLogger(__name__)

NAMESPACE = "mstar"
# One fragment for every op declared here. Ops are defined through `Library`
# rather than `torch.library.custom_op`, whose Python wrapper costs more per
# eager call: 24 us against 16 for this path on an MI300X host, where calling
# the kernel directly takes 12.
_LIB = torch.library.Library(NAMESPACE, "FRAGMENT")

# Modules that declare ops, imported by `load_all` so a listing sees them all.
DECLARING_MODULES = (
    "mstar.engine.resources.rms_norm.rms_norm",
    "mstar.platforms.rocm_gemm",
)

_OPS: dict[str, KernelOp] = {}
_PLATFORM_KEYS = ("cuda", "rocm", "xpu", "default")


@dataclasses.dataclass(frozen=True, eq=False)
class ShapePolicy:
    """The shapes an op is called with.

    ``dims`` names a shape's dimensions and ``key`` reads them off a call's
    arguments. ``inputs(shape, dtype, device, generator)`` builds arguments for
    a shape, for warm-up, parity tests and the kernel harness. ``classes`` are
    the shape classes a tuned config is keyed by, (name, predicate) pairs of
    which the first match wins; ``warm`` the shapes an op is warmed and
    checked on when the caller names none.
    """
    dims: tuple[str, ...]
    key: Callable[..., dict[str, int]]
    inputs: Callable[[Mapping[str, int], torch.dtype, torch.device, torch.Generator | None], tuple]
    classes: tuple[tuple[str, Callable[[Mapping[str, int]], bool]], ...] = ()
    warm: tuple[Mapping[str, int], ...] = ()
    dtypes: tuple[torch.dtype, ...] = (torch.bfloat16,)

    def shape_class(self, shape: Mapping[str, int]) -> str | None:
        for name, matches in self.classes:
            if matches(shape):
                return name
        return None


class KernelOp:
    """One kernel as an ``mstar::<name>`` op; see the module docstring.

    ``impls`` maps ``"cuda"``, ``"rocm"``, ``"xpu"``, ``"default"`` or
    ``"<platform>:<arch>"`` to a factory returning a callable with the op's
    signature. With ``fallback_to_reference`` a platform without a working
    implementation runs the reference (logged once); without it, calling the
    op there raises, naming why.
    """

    def __init__(
        self,
        name: str,
        schema: str,
        *,
        fake: Callable[..., Any],
        reference: Callable[..., Any],
        impls: Mapping[str, Factory] | None = None,
        fallback_to_reference: bool = False,
        shapes: ShapePolicy | None = None,
        tolerance: Tolerance = Tolerance.rounded_once(),
        tags: tuple[torch.Tag, ...] = (),
    ):
        if name in _OPS:
            raise ValueError(f"{NAMESPACE}::{name} is declared twice")
        impls = dict(impls or {})
        unknown = [k for k in impls if ":" not in k and k not in _PLATFORM_KEYS]
        if unknown:
            raise ValueError(f"{NAMESPACE}::{name}: unknown implementation keys {unknown}")
        self.name = name
        self.qualname = f"{NAMESPACE}::{name}"
        self.schema = schema
        self.reference = reference
        self.fake = fake
        self.shapes = shapes
        self.tolerance = tolerance
        self.fallback_to_reference = fallback_to_reference
        self.impl_keys = tuple(impls)
        # key -> why its factory failed, for the last resolution
        self._failures: dict[str, str] = {}
        guarded = {key: self._guard(key, factory) for key, factory in impls.items()}
        default = guarded.pop("default", None)
        self._default_is_reference = default is None and fallback_to_reference
        if self._default_is_reference:
            default = self._reference_factory("no implementation for this platform")
        self._dispatch = PlatformDispatch(
            self.qualname,
            default=default,
            archs={k: f for k, f in guarded.items() if ":" in k},
            **{k: f for k, f in guarded.items() if ":" not in k},
        )

        # pt2-compliant, as torch.library.custom_op tags its ops
        _LIB.define(name + schema, tags=(torch.Tag.pt2_compliant_tag, *tags))
        dispatch = self._dispatch

        def accelerated(*args, **kwargs):
            return dispatch(*args, **kwargs)

        accelerated.__name__ = f"{name}_accelerated"
        # torch calls AMD GPUs "cuda" devices too: the dispatch picks by platform
        _LIB.impl(name, accelerated, "CUDA")
        _LIB.impl(name, accelerated, "XPU")
        _LIB.impl(name, reference, "CompositeExplicitAutograd")
        torch.library.register_fake(self.qualname, fake, lib=_LIB)
        _OPS[name] = self

    # --- calling ------------------------------------------------------------

    @property
    def op(self) -> torch._ops.OpOverloadPacket:
        return getattr(getattr(torch.ops, NAMESPACE), self.name)

    @property
    def dispatch(self) -> PlatformDispatch:
        """What accelerator tensors reach: the implementation for this process's
        platform and architecture, called with the op's arguments."""
        return self._dispatch

    def __call__(self, *args, **kwargs):
        return self.op(*args, **kwargs)

    # --- what runs here -----------------------------------------------------

    def _guard(self, key: str, factory: Factory) -> Factory:
        """``factory``, recording why it failed; a failure leaves the
        reference (with ``fallback_to_reference``) or a callable that raises."""

        def build():
            try:
                impl = factory()
            except Exception as exc:  # noqa: BLE001  any import or build error
                error = exc
            else:
                self._failures.pop(key, None)
                return impl
            reason = f"{type(error).__name__}: {error}"
            self._failures[key] = reason
            if self.fallback_to_reference:
                return self._reference_factory(reason)()

            def unavailable(*args, **kwargs):
                raise RuntimeError(f"{self.qualname} cannot run on {key}: {reason}") from error

            return unavailable

        return build

    def _reference_factory(self, reason: str) -> Factory:
        def build():
            logger.warning("%s runs its torch reference on %s: %s", self.qualname, current_platform.name, reason)
            return self.reference

        return build

    def implementation_key(self) -> str | None:
        """Which implementation this process runs: an impls key, ``"reference"``,
        or None when calling it on a GPU would raise. Runs no factory."""
        key = self._dispatch.selected_key()
        if key == "default" and self._default_is_reference:
            return "reference"
        return key

    def implementation(self) -> Callable[..., Any]:
        """The callable accelerator tensors reach, running its factory on first use."""
        return self._dispatch.resolve()

    def unavailable_reason(self) -> str | None:
        """None when an implementation (not the reference fallback) serves this
        platform; otherwise why not. Runs the factory, so call it at build time,
        never inside a capture."""
        key = self._dispatch.selected_key()
        if key is None:
            return f"no implementation for platform {current_platform.name}"
        if key == "default" and self._default_is_reference:
            return f"no implementation for platform {current_platform.name}; runs the reference"
        self._dispatch.resolve()
        return self._failures.get(key)

    def mutated_args(self) -> tuple[int, ...]:
        """Positions of the arguments the op writes in place."""
        schema = self.op.default._schema
        return tuple(
            i for i, arg in enumerate(schema.arguments)
            if arg.alias_info is not None and arg.alias_info.is_write
        )

    def warm(
        self,
        shapes: Iterable[Mapping[str, int]] | None = None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> list[Mapping[str, int]]:
        """Resolve the implementation and run the op once on synthetic inputs
        of each shape (the policy's ``warm`` shapes by default) on ``device``
        (the current GPU by default), so imports, JIT builds and autotuning
        happen now and not inside a captured graph. Returns the shapes run."""
        device = torch.device(device if device is not None else "cuda")
        if device.type != "cpu":
            self.implementation()
        if self.shapes is None:
            return []
        dtype = dtype or self.shapes.dtypes[0]
        ran = []
        for shape in self.shapes.warm if shapes is None else shapes:
            self.op(*self.shapes.inputs(shape, dtype, device, None))
            ran.append(shape)
        return ran

    def config(self, shape: Mapping[str, int] | None = None, device=None) -> Mapping[str, Any] | None:
        """This device's tuned config for ``shape``'s class (``mstar.kernels.configs``)."""
        facts = current_platform.device_facts(device)
        config = configs.load_config(self.name, facts.arch, facts.product)
        shape_class = self.shapes.shape_class(shape) if self.shapes is not None and shape is not None else None
        return configs.select(config, shape_class)

    def describe(self, probe: bool = False) -> dict[str, Any]:
        """What a listing shows. With ``probe``, also which implementation runs
        here and why one is unavailable (runs its factory)."""
        info = {
            "op": self.qualname,
            "schema": self.schema,
            "impls": list(self.impl_keys),
            "reference": f"{self.reference.__module__}.{self.reference.__qualname__}",
            "fallback_to_reference": self.fallback_to_reference,
            "tolerance": dataclasses.asdict(self.tolerance),
            "warm_shapes": [dict(s) for s in self.shapes.warm] if self.shapes else [],
            "shape_classes": [name for name, _ in self.shapes.classes] if self.shapes else [],
        }
        if probe:
            reason = self.unavailable_reason()
            key = self.implementation_key()
            info["platform"] = current_platform.name
            info["implementation"] = key
            info["unavailable_reason"] = reason
            info["runs"] = "reference" if key == "reference" or (reason and self.fallback_to_reference) else key
        return info

    def __repr__(self) -> str:
        return f"KernelOp({self.qualname!r})"


def get(name: str) -> KernelOp:
    """The op declared as ``name`` (``"rms_norm"`` or ``"mstar::rms_norm"``)."""
    return _OPS[name.removeprefix(f"{NAMESPACE}::")]


def ops() -> list[KernelOp]:
    """Every op declared so far, by name."""
    return [_OPS[name] for name in sorted(_OPS)]


def load_all() -> list[KernelOp]:
    """Import every declaring module, then list every op."""
    for module in DECLARING_MODULES:
        importlib.import_module(module)
    return ops()
