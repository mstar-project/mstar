"""Compiling a diffusion model's transformer or VAE decode: which modules stay
eager, and the inductor rounding the compiled region is traced under.

Shared by the flow models (FLUX.2 klein, Z-Image and LTX-2.5 compile their
transformer forward, LingBot-World its blocks) and by the VAE decode side (the
compile wrappers in ``decode_utils`` and the fused GroupNorm in ``group_norm``,
which asks ``kept_exact``); nothing here is model-specific. Lives beside the
scaffold rather than in a model package so a model never imports a sibling model.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterable, Sequence

import torch
from torch import nn

from mstar.model.components.diffusion.exact_norm import exact_rms_norm

logger = logging.getLogger(__name__)

# Modules whose eager CUDA kernels differ in the last bit from inductor's decompositions, by class:
# reductions (LayerNorm / RMSNorm / GroupNorm mean and variance; a model's own norm opts in with a
# ``compile_exact_op = True`` class attribute) and transcendental activations (SiLU).
EXACT_OP_CLASSES: dict[str, tuple[type, ...]] = {
    "norms": (nn.LayerNorm, nn.RMSNorm, nn.GroupNorm),
    "activations": (nn.SiLU,),
}
EXACT_OP_TYPES = tuple(t for types in EXACT_OP_CLASSES.values() for t in types)
ExactOps = bool | Sequence[str]


def exact_op_types(spec: ExactOps) -> tuple[type, ...] | None:
    """``True`` -> every class, ``False`` -> None, a list of class names -> those classes' types."""
    if spec is True:
        return EXACT_OP_TYPES
    if not spec:
        return None
    unknown = [name for name in spec if name not in EXACT_OP_CLASSES]
    if unknown:
        raise ValueError(f"compile_exact_ops: unknown op class(es) {unknown}; choose from {sorted(EXACT_OP_CLASSES)}")
    return tuple(t for name in spec for t in EXACT_OP_CLASSES[name])


def exclude_from_compile(transformer: nn.Module, types: Sequence[type] = EXACT_OP_TYPES) -> int:
    """Keep the transformer's norms and/or activations on the eager result inside a compiled
    forward. A module is kept by type (``types``) or, when ``types`` include the norms, with a
    ``compile_exact_op = True`` class attribute. A 1-D ``torch.nn.RMSNorm`` (not a subclass) runs
    :func:`exact_rms_norm`, an op the compiled graph keeps whole (on NVIDIA, several times faster
    than torch's kernel on the per-head q/k norms); every other kept module's forward becomes a
    dynamo graph break, so inductor only fuses the pointwise chains around it. A compiled path
    that would replace a module, such as a fused kernel, asks :func:`kept_exact` first. Returns
    the number of modules kept."""
    types = tuple(types)
    with_marked = any(t in EXACT_OP_CLASSES["norms"] for t in types)
    count = 0
    for module in transformer.modules():
        if isinstance(module, types) or (with_marked and getattr(module, "compile_exact_op", False)):
            if not kept_exact(module):
                exact_rms = type(module) is nn.RMSNorm and len(module.normalized_shape) == 1
                module.__class__ = _kept_class(type(module), exact_rms)
            count += 1
    return count


def kept_exact(module: nn.Module | None) -> bool:
    """Whether :func:`exclude_from_compile` keeps ``module`` on the eager result."""
    return getattr(module, "compile_kept_exact", False)


@functools.cache
def _kept_class(cls: type[nn.Module], exact_rms: bool) -> type[nn.Module]:
    """The subclass a kept module of class ``cls`` becomes, made once, so every kept module of a
    class shares one ``forward``. Dynamo guards a non-function ``forward`` set on an instance (a
    ``functools.partial``) by identity, so a frame the blocks share would recompile once per block;
    with ``forward`` on the shared class, every block passes the same guards."""
    forward = _exact_rms_norm_forward if exact_rms else torch.compiler.disable(cls.forward)
    return type(cls.__name__, (cls,), {"forward": forward, "compile_kept_exact": True})


def _exact_rms_norm_forward(module: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    return exact_rms_norm(x, module.weight, module.eps)


def compile_transformer_forward(
    transformer: nn.Module, eager_rounding: bool = True, exact_ops: ExactOps = False,
) -> None:
    """Compile ``transformer.forward`` in place (one static graph per shape; the graph runner
    captures those kernels). With ``eager_rounding`` inductor rounds every intermediate to the
    tensor dtype exactly where eager PyTorch does (``emulate_precision_casts``): without it, fused
    bf16 chains keep fp32 intermediates and the served images drift to 35-39 dB from the bit-exact
    eager path on a 4-step distilled sampler (measured). Even with it the compiled transformer
    lands at a median 37.6 dB over 100 prompts (klein-4B), because inductor's own reductions and
    activation decompositions round differently: ``exact_ops`` keeps those modules on the eager
    result (``exclude_from_compile``) and compiles the rest — ``True`` for all of them, or a list
    of the op classes to keep (``["norms"]``, ``["activations"]``), since every kept module but a
    1-D ``torch.nn.RMSNorm`` is a graph break that costs fusion (klein-4B paid 8% at B=1 for all of
    them while its RMSNorms were breaks too, Z-Image 48%)."""
    types = exact_op_types(exact_ops)
    if types:
        kept = exclude_from_compile(transformer, types)
        logger.info("compiled transformer keeps %d modules (%s) on the eager result", kept,
                    "all classes" if exact_ops is True else ", ".join(exact_ops))
    transformer.forward = with_inductor_rounding(
        torch.compile(transformer.forward, fullgraph=False, dynamic=False),
        eager_rounding=bool(eager_rounding),
    )


def with_inductor_rounding(compiled, *, eager_rounding: bool, deterministic: bool = False):
    """Wrap a compiled callable so ``emulate_precision_casts`` is in effect for the
    traces it builds, and only for those. This works with torch.compile's lazy
    compilation, and post-torch 2.12 per-thread config behavior, without introducing
    a global config change. ``deterministic`` also builds them with the engine's
    deterministic compile option (``accelerator_graph_runner.
    deterministic_inductor_options``): no on-device benchmarking that changes
    numerics, so every process compiles the same kernels.
    """
    import torch._inductor.config as inductor_config

    options = {"emulate_precision_casts": eager_rounding}
    if deterministic:
        from mstar.engine.accelerator_graph_runner import deterministic_inductor_options

        options.update(deterministic_inductor_options())

    def forward(*args, **kwargs):
        with inductor_config.patch(**options):
            return compiled(*args, **kwargs)

    return forward


def compile_blocks(blocks: Iterable[nn.Module], *, autotune: bool = False, **compile_kwargs) -> None:
    """Compile each block's forward in place, rounding every intermediate where
    eager does (``emulate_precision_casts``): by default under Inductor's
    defaults in its deterministic mode, so every process compiles the same
    kernels; with ``autotune`` under the -O level's options, whose kernel picks
    are per process. ``compile_kwargs`` go to ``torch.compile`` (static shapes
    and no full graph unless overridden). For a transformer whose blocks keep
    an eager call in the middle (a cache write, a ragged attention), so each
    block traces into graphs around it once.
    """
    if autotune:
        from mstar.engine.accelerator_graph_runner import compile_kwargs as level_kwargs

        compile_kwargs = {**level_kwargs(None), **compile_kwargs}
    compile_kwargs = {"fullgraph": False, "dynamic": False, **compile_kwargs}
    for block in blocks:
        block.forward = with_inductor_rounding(
            torch.compile(block.forward, **compile_kwargs), eager_rounding=True, deterministic=not autotune,
        )
