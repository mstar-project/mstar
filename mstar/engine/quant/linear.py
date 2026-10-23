"""The quant pass over a node's linears, and the forward a quantized one runs.

A quantized linear keeps its class (and so its TP sharding and collectives):
its ``weight`` becomes the format's values beside a ``weight_scale``, and its
``linear_quant`` names the kernel its matmul runs. A plain ``nn.Linear`` is
re-classed to ``QuantizedLinear`` for the same effect.
"""
from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.quant.config import QuantConfig
from mstar.engine.quant.formats import FORMATS, QuantFormat
from mstar.engine.quant.kernels import KERNELS, QuantKernel
from mstar.utils.device import DeviceCapability

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class LinearQuant:
    config: QuantConfig
    format: QuantFormat
    kernel: QuantKernel


def run_linear(module: nn.Module, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """``F.linear(x, module.weight, bias)``, through the quant kernel if it has one."""
    quant = module.linear_quant
    if quant is None:
        return F.linear(x, module.weight, bias)
    return quant.kernel.run(module, x, bias)


class QuantizedLinear(nn.Linear):
    """An ``nn.Linear`` the quant pass converted; set by re-classing, never built."""
    linear_quant: LinearQuant | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_linear(self, x, self.bias)


def is_quantizable(module: nn.Module) -> bool:
    """A plain ``nn.Linear``, or a linear that routes its matmul through ``run_linear``."""
    return type(module) is nn.Linear or hasattr(type(module), "linear_quant")


def choose_kernel(config: QuantConfig, device: DeviceCapability, dtype: torch.dtype) -> QuantKernel:
    """The pinned kernel, else the format's first default one the device runs."""
    fmt = FORMATS[config.weights]
    names = (config.kernel,) if config.kernel else fmt.kernels
    reasons = []
    for name in names:
        kernel = KERNELS[name]
        if name not in fmt.kernels:
            reasons.append(f"{name}: does not read {fmt.name}")
        elif config.activations not in kernel.activations:
            reasons.append(f"{name}: does not run activations={config.activations!r}")
        elif not kernel.default and not config.kernel:
            continue
        elif (why := kernel.supports(device, dtype)) is not None:
            reasons.append(f"{name}: {why}")
        else:
            return kernel
    raise ValueError(
        f"no quant kernel for weights={fmt.name}, activations={config.activations!r} on "
        f"{device.device} in {dtype}: " + "; ".join(reasons or ["no candidates"])
    )


def _matched(names: list[str], globs: tuple[str, ...], what: str) -> set[str]:
    matched: set[str] = set()
    for glob in globs:
        hits = fnmatch.filter(names, glob)
        if not hits:
            raise ValueError(f"{what} glob {glob!r} matches no module")
        matched.update(hits)
    return matched


def _under(name: str, tops: set[str]) -> bool:
    return any(name == top or name.startswith(top + ".") for top in tops)


def select_linears(
    module: nn.Module, quant_skip: tuple[str, ...], config: QuantConfig, islands: set[str],
) -> tuple[dict[str, nn.Module], list[str]]:
    """The linears to quantize, by name, and the other linears' names; a linear
    under a skip glob or an fp32 island is never quantized."""
    names = [name for name, _ in module.named_modules() if name]
    skipped = _matched(names, quant_skip, "quant_skip") | _matched(names, config.skip, "quant.skip")
    chosen, left = {}, []
    for name, child in module.named_modules():
        if not name or not is_quantizable(child):
            continue
        if _under(name, skipped) or _under(name, islands) or _under(f"{name}.weight", islands):
            left.append(name)
        else:
            chosen[name] = child
    return chosen, left


def quant_mismatches(
    module: nn.Module, quant_skip: tuple[str, ...], config: QuantConfig | None, islands: set[str],
) -> list[str]:
    """Quantized linears ``module``'s policy would not quantize, or would quantize
    otherwise; after every node loads, these are linears shared with another node."""
    quantized = {n: c for n, c in module.named_modules() if getattr(c, "linear_quant", None) is not None}
    if not quantized:
        return []
    chosen = select_linears(module, quant_skip, config, islands)[0] if config is not None else {}
    return [
        f"{name} (quantized as {child.linear_quant.config}, policy {config if name in chosen else None})"
        for name, child in quantized.items()
        if name not in chosen or child.linear_quant.config != config
    ]


def _tied(module: nn.Module, chosen: dict[str, nn.Module]) -> list[str]:
    """Chosen weights that another param name of ``module`` also holds."""
    owners: dict[int, list[str]] = {}
    for name, param in module.named_parameters(remove_duplicate=False):
        owners.setdefault(id(param), []).append(name)
    return [
        f"{name}.weight = {[n for n in owners[id(child.weight)] if n != f'{name}.weight']}"
        for name, child in chosen.items()
        if len(owners.get(id(child.weight), ())) > 1
    ]


def _convert(child: nn.Module, quant: LinearQuant) -> float:
    """Quantize ``child`` in place; returns the weight's relative L2 error."""
    weight = child.weight.detach()
    values, scale = quant.format.quantize(weight, quant.config.weight_scale)
    w = weight.float()
    error = float((quant.format.dequantize(values, scale) - w).norm() / w.norm().clamp_min(1e-30))
    del w
    child.weight = nn.Parameter(values, requires_grad=False)
    child.register_parameter("weight_scale", nn.Parameter(scale, requires_grad=False))
    if type(child) is nn.Linear:
        child.__class__ = QuantizedLinear
    child.linear_quant = quant
    quant.kernel.prepare(child)
    return error


def quantize_linears(
    module: nn.Module,
    node_name: str,
    quant_skip: tuple[str, ...],
    config: QuantConfig,
    islands: set[str],
    device: DeviceCapability,
    param_dtype: torch.dtype | None,
) -> None:
    """Quantize ``module``'s selected linears in place, from their loaded weights.

    Runs after load, so the peak is the node's unquantized size; each bf16
    weight is freed as its layer converts.
    """
    chosen, left = select_linears(module, quant_skip, config, islands)
    fresh = {n: c for n, c in chosen.items() if getattr(c, "linear_quant", None) is None}
    tied = _tied(module, fresh)
    if tied:
        raise ValueError(
            f"node {node_name!r}: quantized weights may not be shared with another param; "
            f"add them to quant.skip: {tied[:5]}"
        )
    fmt = FORMATS[config.weights]
    for name, child in chosen.items():
        if name not in fresh and child.linear_quant.config != config:
            raise ValueError(
                f"node {node_name!r}: linear {name!r} is shared with a node that quantized "
                f"it as {child.linear_quant.config}, not {config}"
            )
    if not fresh:
        logger.info("node %s: quant %s: no linears to quantize (%d left as is)", node_name, fmt.name, len(left))
        return
    dtype = param_dtype or next(iter(fresh.values())).weight.dtype
    kernel = choose_kernel(config, device, dtype)
    quant = LinearQuant(config, fmt, kernel)
    before = after = 0
    errors: list[tuple[float, str]] = []
    unsupported = []
    for name, child in fresh.items():
        n, k = child.weight.shape
        if not kernel.shape_ok(n, k):
            unsupported.append(f"{name} {n}x{k}")
            continue
        before += child.weight.numel() * child.weight.element_size()
        errors.append((_convert(child, quant), name))
        after += child.weight.numel() * child.weight.element_size()
        after += child.weight_scale.numel() * child.weight_scale.element_size()
        logger.debug("node %s: %s quantized, rel error %.4g", node_name, name, errors[-1][0])
    if device.device.type == "cuda":
        torch.cuda.empty_cache()
    worst = max(errors, default=(0.0, "-"))
    mean = sum(e for e, _ in errors) / max(len(errors), 1)
    logger.info(
        "node %s: quant %s (%s, weights by %s, activations %s): %d linears quantized, "
        "%d left as is, weight bytes %.1f MiB -> %.1f MiB, rel error mean %.4f max %.4f (%s); "
        "quantized after load, so the load peak was the unquantized size",
        node_name, fmt.name, kernel.name, config.weight_scale, config.activations,
        len(errors), len(left), before / 2**20, after / 2**20, mean, worst[0], worst[1],
    )
    if unsupported:
        logger.warning(
            "node %s: %d linears left unquantized, a shape %s does not take: %s",
            node_name, len(unsupported), kernel.name, unsupported[:5],
        )
