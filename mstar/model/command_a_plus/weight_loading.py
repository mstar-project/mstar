"""Stream Command A+ text weights into fused, tensor-parallel parameters."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from mstar.model.command_a_plus.config import CommandAPlusTextConfig
from mstar.model.loader import default_weight_loader


def remap_text_weight_name(name: str) -> str | None:
    """Map text checkpoint names into the causal-LM wrapper; skip other weights."""
    prefix = "model.language_model."
    if not name.startswith(prefix):
        return None
    return "model." + name[len(prefix):]


@dataclass(frozen=True)
class _WeightTarget:
    name: str
    shape: tuple[int, ...]  # Full checkpoint shape, before TP slicing.
    shard_id: str | int | None = None


def _checkpoint_layout(config: CommandAPlusTextConfig) -> dict[str, _WeightTarget]:
    """Map each required source tensor to its destination and fused slice.

    Expert IDs are explicit integers, so checkpoint iteration order (including
    lexicographic expert 10 before expert 2) cannot change expert placement.
    """
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    shared = config.shared_intermediate_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim
    layout = {
        "model.embed_tokens.weight": _WeightTarget(
            "model.embed_tokens.weight", (config.vocab_size, hidden),
        ),
        "model.norm.weight": _WeightTarget("model.norm.weight", (hidden,)),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        for source, target, shape, shard in (
            ("input_layernorm.weight", "input_layernorm.weight", (hidden,), None),
            ("self_attn.q_proj.weight", "self_attn.qkv_proj.weight", (q_width, hidden), "q"),
            ("self_attn.k_proj.weight", "self_attn.qkv_proj.weight", (kv_width, hidden), "k"),
            ("self_attn.v_proj.weight", "self_attn.qkv_proj.weight", (kv_width, hidden), "v"),
            ("self_attn.o_proj.weight", "self_attn.o_proj.weight", (hidden, q_width), None),
            ("mlp.gate.weight", "mlp.gate.weight", (config.num_experts, hidden), None),
            ("mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.gate_up_proj.weight",
             (shared, hidden), 0),
            ("mlp.shared_experts.up_proj.weight", "mlp.shared_experts.gate_up_proj.weight",
             (shared, hidden), 1),
            ("mlp.shared_experts.down_proj.weight", "mlp.shared_experts.down_proj.weight",
             (hidden, shared), None),
        ):
            layout[prefix + source] = _WeightTarget(prefix + target, shape, shard)
        for expert in range(config.num_experts):
            for projection in ("gate", "up", "down"):
                source = f"{prefix}mlp.experts.{expert}.{projection}_proj.weight"
                if projection == "down":
                    target, shape = "down_proj", (hidden, intermediate)
                else:
                    target, shape = "gate_up_proj", (intermediate, hidden)
                layout[source] = _WeightTarget(
                    f"{prefix}mlp.experts.{target}", shape, f"{projection}:{expert}",
                )
    return layout


@torch.no_grad()
def load_command_a_plus_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    config: CommandAPlusTextConfig,
) -> set[str]:
    """Load the official, unfused text checkpoint; return loaded parameter names.

    Reuse the existing parameters' loaders for fusion offsets and TP slicing.
    Track individual source tensors, not just destination parameters: loading
    Q alone must not count as loading the entire QKV parameter. Non-text weights
    are ignored; unexpected, duplicate, misshaped or missing text weights fail.

    Loading is incremental, not transactional. On failure the model is partly
    loaded and must not be used until a complete load succeeds.
    """
    layout = _checkpoint_layout(config)
    parameters = dict(module.named_parameters())
    targets = {spec.name for spec in layout.values()}
    if targets != parameters.keys():
        raise ValueError(
            "Command A+ parameter layout mismatch: "
            f"missing={sorted(targets - parameters.keys())}, "
            f"unexpected={sorted(parameters.keys() - targets)}"
        )
    if any(parameter.is_meta for parameter in parameters.values()):
        raise ValueError("Allocate Command A+ parameters with to_empty before loading weights")

    seen: set[str] = set()
    for checkpoint_name, tensor in weights:
        name = remap_text_weight_name(checkpoint_name)
        if name is None:
            continue
        spec = layout.get(name)
        if spec is None:
            raise ValueError(f"Unexpected Command A+ text weight: {checkpoint_name}")
        if name in seen:
            raise ValueError(f"Duplicate Command A+ text weight: {checkpoint_name}")
        if tuple(tensor.shape) != spec.shape:
            raise ValueError(
                f"Shape mismatch for {checkpoint_name}: "
                f"expected {spec.shape}, got {tuple(tensor.shape)}"
            )
        if tensor.is_meta:
            raise ValueError(f"Checkpoint tensor has no data: {checkpoint_name}")
        parameter = parameters[spec.name]
        loader = getattr(parameter, "weight_loader", default_weight_loader)
        loader(parameter, tensor, spec.shard_id)
        seen.add(name)

    missing = layout.keys() - seen
    if missing:
        examples = ", ".join(sorted(missing)[:8])
        raise ValueError(f"Missing {len(missing)} Command A+ text weights: {examples}")
    return targets
