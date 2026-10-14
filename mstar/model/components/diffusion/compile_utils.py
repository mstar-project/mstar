"""Compiling a flow transformer: which modules stay eager, and the inductor
rounding the compiled region is traced under.

Shared by every model on the DiT scaffold (FLUX.2 klein, Z-Image); nothing here
is model-specific. Lives beside the scaffold rather than in a model package so a
model never imports a sibling model.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import nn

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
    """Keep the transformer's norms and/or activations on the eager kernels inside a compiled
    forward: each excluded module's forward becomes a dynamo graph break, so inductor only
    fuses the pointwise chains around the GEMMs and attention. A module is excluded by type
    (``types``) or, when ``types`` include the norms, with a ``compile_exact_op = True`` class
    attribute. Returns the number of modules excluded."""
    types = tuple(types)
    with_marked = any(t in EXACT_OP_CLASSES["norms"] for t in types)
    count = 0
    for module in transformer.modules():
        if isinstance(module, types) or (with_marked and getattr(module, "compile_exact_op", False)):
            module.forward = torch._dynamo.disable(module.forward)
            count += 1
    return count


def compile_transformer_forward(
    transformer: nn.Module, eager_rounding: bool = True, exact_ops: ExactOps = False,
) -> None:
    """Compile ``transformer.forward`` in place (one static graph per shape; the graph runner
    captures those kernels). With ``eager_rounding`` inductor rounds every intermediate to the
    tensor dtype exactly where eager PyTorch does (``emulate_precision_casts``): without it, fused
    bf16 chains keep fp32 intermediates and the served images drift to 35-39 dB from the bit-exact
    eager path on a 4-step distilled sampler (measured). Even with it the compiled transformer
    lands at a median 37.6 dB over 100 prompts (klein-4B), because inductor's own reductions and
    activation decompositions round differently: ``exact_ops`` keeps those modules eager
    (``exclude_from_compile``) and compiles the rest — ``True`` for all of them, or a list of the
    op classes to keep eager (``["norms"]``, ``["activations"]``), since each excluded module is a
    graph break that costs fusion (klein-4B pays 8% at B=1 for all of them, Z-Image 48%)."""
    types = exact_op_types(exact_ops)
    if types:
        excluded = exclude_from_compile(transformer, types)
        logger.info("compiled transformer keeps %d modules (%s) on the eager kernels", excluded,
                    "all classes" if exact_ops is True else ", ".join(exact_ops))
    transformer.forward = with_inductor_rounding(
        torch.compile(transformer.forward, fullgraph=False, dynamic=False),
        eager_rounding=bool(eager_rounding),
    )


def with_inductor_rounding(compiled, *, eager_rounding: bool):
    """Wrap a compiled callable so ``emulate_precision_casts`` is in effect for the
    traces it builds, and only for those. This works with torch.compile's lazy
    compilation, and post-torch 2.12 per-thread config behavior, without introducing
    a global config change.
    """
    import torch._inductor.config as inductor_config

    def forward(*args, **kwargs):
        with inductor_config.patch(emulate_precision_casts=eager_rounding):
            return compiled(*args, **kwargs)

    return forward
