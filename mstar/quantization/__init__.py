"""Quantization, any model at any precision, down to the layer (see ``docs`` and ``spec.py``).

The serving config's ``quantization:`` block (``spec``) picks a format per declared layer; the model declares
its weights roots (``resolve.QuantizationRoot`` via ``Model.get_quantization_roots``); the checkpoint says how
each layer is stored (``checkpoint``); ``resolve`` turns the three into a per-layer plan, refused at boot
(``validate``) with every problem; ``apply`` carries it out around the model's own loader, which the engine
wires through ``scope``.
"""

from mstar.quantization.checkpoint import CheckpointQuant, StoredBase, scan_checkpoint
from mstar.quantization.formats import AUTO, BF16, FORMATS, FormatInfo
from mstar.quantization.resolve import Action, LayerPlan, QuantizationRoot, QuantPlan, resolve_plan
from mstar.quantization.spec import NodeQuantSpec, QuantizationSpec, QuantizationSpecError, Rule, parse_quantization

__all__ = [
    "AUTO", "BF16", "FORMATS", "Action", "CheckpointQuant", "FormatInfo", "LayerPlan", "NodeQuantSpec",
    "QuantPlan", "QuantizationRoot", "QuantizationSpec", "QuantizationSpecError", "Rule", "StoredBase",
    "parse_quantization", "resolve_plan", "scan_checkpoint",
]
