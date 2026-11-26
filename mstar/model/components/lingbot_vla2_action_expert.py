"""Action expert decoder and 10-step flow-matching loop for LingBot-VLA 2.0.

Wraps the 36-layer MoE-FFN + AdaRMSNorm action expert (``qwen_expert``) and
shared suffix input/output projections into an M*-compatible module supporting
both M* KV-cache state-contract execution (``add_read`` with
``PrefixSuffixGroups(first_group_len=1)``) and standalone eager attention hooks
for CPU parity tests.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.state.calls import add_read
from mstar.model.components.lingbot_vla2_attention import (
    LingBotVLAActionAttentionConfig,
    LingBotVLAActionProjections,
)
from mstar.model.components.lingbot_vla2_film import LingBotVLAAdaRMSNorm
from mstar.model.components.lingbot_vla2_moe import (
    LingBotVLAActionMoe,
    LingBotVLAActionMoeConfig,
)
from mstar.model.components.lingbot_vla2_suffix import (
    LingBotVLASuffixConfig,
    LingBotVLASuffixInputs,
)
from mstar.model.components.mrope import (
    apply_partial_mrope,
    compute_3d_cos_sin,
    compute_inv_freq,
)
from mstar.model.components.norm import RMSNorm
from mstar.model.loader import LLAMA_STACKED_PARAMS
from mstar.model.loader.base import default_weight_loader, load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards

_EXPERT_PREFIX = "model.qwenvl_with_expert.qwen_expert.model."
_SUFFIX_STEMS = (
    "model.state_proj.",
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.action_time_mlp_in.",
    "model.action_time_mlp_out.",
)


@dataclass(frozen=True)
class LingBotVLAActionExpertConfig:
    """Architecture hyperparameters for the LingBot-VLA 2.0 action expert."""

    num_hidden_layers: int = 36
    hidden_size: int = 768
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    num_experts: int = 32
    top_k: int = 4
    routed_intermediate_size: int = 512
    shared_intermediate_size: int = 704
    routed_scaling_factor: float = 4.0
    max_state_dim: int = 55
    max_action_dim: int = 55
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    final_norm_adanorm: bool = False
    action_fp32: bool = False
    fp32_attention: bool = False
    kv_key: str = "LLM_KV"

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        qwen3_vl_dir: str | Path,
        *,
        kv_key: str = "LLM_KV",
        fp32_attention: bool = False,
    ) -> "LingBotVLAActionExpertConfig":
        """Read expert configuration from VLA ``config.json`` and base Qwen3-VL ``config.json``."""
        vla = json.loads((Path(checkpoint_dir) / "config.json").read_text())
        qwen = json.loads((Path(qwen3_vl_dir) / "config.json").read_text())
        text = qwen.get("text_config", {})
        rope_params = text.get("rope_parameters") or text.get("rope_scaling") or {}
        rope_theta = float(rope_params.get("rope_theta", text.get("rope_theta", 5_000_000.0)))
        section = tuple(int(x) for x in rope_params.get("mrope_section", (24, 20, 20)))
        if len(section) != 3:
            raise ValueError(f"Expected 3D mrope_section, got {section}")

        return cls(
            num_hidden_layers=int(text["num_hidden_layers"]),
            hidden_size=int(vla.get("expert_hidden_size", 768)),
            num_attention_heads=int(vla.get("action_num_attention_heads", 32)),
            num_key_value_heads=int(vla.get("action_num_key_value_heads", 8)),
            head_dim=int(vla.get("action_head_dim", 128)),
            num_experts=int(vla.get("token_num_experts", 32)),
            top_k=int(vla.get("token_top_k", 4)),
            routed_intermediate_size=int(vla.get("token_moe_intermediate_size", 512)),
            shared_intermediate_size=int(vla.get("token_shared_intermediate_size", 704)),
            routed_scaling_factor=float(vla.get("routed_scaling_factor", 4.0)),
            max_state_dim=int(vla.get("max_state_dim", 55)),
            max_action_dim=int(vla.get("max_action_dim", 55)),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=rope_theta,
            mrope_section=(section[0], section[1], section[2]),
            final_norm_adanorm=bool(vla.get("final_norm_adanorm", False)),
            action_fp32=bool(vla.get("action_fp32", False)),
            fp32_attention=fp32_attention,
            kv_key=kv_key,
        )


class LingBotVLAActionLayer(nn.Module):
    """Single decoder layer of the LingBot-VLA 2.0 action expert."""

    def __init__(
        self,
        config: LingBotVLAActionExpertConfig,
        layer_idx: int,
        *,
        kv_key: str,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self._kv_key = kv_key
        self.state: Any = None

        self.input_layernorm = LingBotVLAAdaRMSNorm(
            config.hidden_size, config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = LingBotVLAActionProjections(
            LingBotVLAActionAttentionConfig(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
            ),
            comm_group=comm_group,
        )
        self.post_attention_layernorm = LingBotVLAAdaRMSNorm(
            config.hidden_size, config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = LingBotVLAActionMoe(
            LingBotVLAActionMoeConfig(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                top_k=config.top_k,
                routed_intermediate_size=config.routed_intermediate_size,
                shared_intermediate_size=config.shared_intermediate_size,
                routed_scaling_factor=config.routed_scaling_factor,
            )
        )

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.state = handles.get(self._kv_key)

    def forward(
        self,
        hidden_states: torch.Tensor,
        adarms_cond: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        attn_fn: Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        bs, seq_len, _ = hidden_states.shape
        residual = hidden_states
        normed = self.input_layernorm(hidden_states, adarms_cond)
        q, k, v = self.self_attn.project_qkv(normed)

        # LingBot-VLA 2.0 applies 3D MRoPE on suffix Q/K in float32.
        q_flat = q.reshape(bs * seq_len, q.shape[2], q.shape[3])
        k_flat = k.reshape(bs * seq_len, k.shape[2], k.shape[3])
        v_flat = v.reshape(bs * seq_len, v.shape[2], v.shape[3])
        q_rot, k_rot = apply_partial_mrope(
            q_flat.to(torch.float32),
            k_flat.to(torch.float32),
            cos.to(torch.float32),
            sin.to(torch.float32),
        )
        if not self.config.fp32_attention:
            q_rot = q_rot.to(hidden_states.dtype)
            k_rot = k_rot.to(hidden_states.dtype)
            v_in = v_flat.to(hidden_states.dtype)
        else:
            v_in = v_flat.to(torch.float32)

        if attn_fn is not None:
            attn_out = attn_fn(
                self.layer_idx,
                q_rot.reshape(bs, seq_len, q.shape[2], q.shape[3]),
                k_rot.reshape(bs, seq_len, k.shape[2], k.shape[3]),
                v_in.reshape(bs, seq_len, v.shape[2], v.shape[3]),
            )
        elif self.state is not None:
            attn_out = add_read(self.state, self.layer_idx, q_rot, k=k_rot, v=v_in)
        else:
            raise RuntimeError(
                "LingBotVLAActionLayer requires bound KV state (`bind_state`) or an explicit `attn_fn`"
            )

        attn_out = attn_out.reshape(bs, seq_len, self.config.num_attention_heads, self.config.head_dim)
        attn_out = attn_out.to(self.self_attn.self_attn.o_proj.weight.dtype)
        hidden_states = residual + self.self_attn.project_output(attn_out)

        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states, adarms_cond)
        return residual + self.mlp(normed)


class LingBotVLAActionExpert(nn.Module):
    """36-layer MoE + FiLM AdaRMSNorm action expert with 10-step Euler flow matching."""

    def __init__(
        self,
        config: LingBotVLAActionExpertConfig,
        *,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.suffix_inputs = LingBotVLASuffixInputs(
            LingBotVLASuffixConfig(
                max_state_dim=config.max_state_dim,
                max_action_dim=config.max_action_dim,
                expert_hidden_size=config.hidden_size,
                action_fp32=config.action_fp32,
            )
        )
        self.layers = nn.ModuleList(
            LingBotVLAActionLayer(config, i, kv_key=config.kv_key, comm_group=comm_group)
            for i in range(config.num_hidden_layers)
        )
        self.norm: nn.Module = (
            LingBotVLAAdaRMSNorm(config.hidden_size, config.hidden_size, eps=config.rms_norm_eps)
            if config.final_norm_adanorm
            else RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        )
        self.register_buffer(
            "mrope_inv_freq",
            compute_inv_freq(config.head_dim, config.rope_theta),
            persistent=False,
        )

    def reset_non_persistent_buffers(self, device: torch.device | None = None) -> None:
        """Recompute non-persistent rotary frequencies after meta-device loading."""
        target = self.mrope_inv_freq.device if device is None else device
        self.mrope_inv_freq = compute_inv_freq(
            self.config.head_dim, self.config.rope_theta, device=target
        )

    def _compute_suffix_rope(
        self, suffix_position_ids_3d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_3d_cos_sin(
            suffix_position_ids_3d,
            self.mrope_inv_freq,
            list(self.config.mrope_section),
            target_dtype=torch.float32,
        )

    def forward_step(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        suffix_position_ids_3d: torch.Tensor,
        *,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_fn: Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run one flow-matching velocity step over ``[state, noisy_actions]``."""
        adarms_cond, suffix_embeds, _, _ = self.suffix_inputs(state, noisy_actions, timestep)
        layer_dtype = self.layers[0].self_attn.self_attn.o_proj.weight.dtype
        hidden = suffix_embeds.to(layer_dtype)
        cond = adarms_cond.to(layer_dtype)

        if cos_sin is None:
            cos, sin = self._compute_suffix_rope(suffix_position_ids_3d)
        else:
            cos, sin = cos_sin

        for layer in self.layers:
            hidden = layer(hidden, cond, cos, sin, attn_fn=attn_fn)

        if isinstance(self.norm, LingBotVLAAdaRMSNorm):
            hidden = self.norm(hidden, cond)
        else:
            hidden = self.norm(hidden.to(self.norm.weight.dtype))

        # Slice out only the action tokens (index 0 is the state token).
        action_hidden = hidden[:, 1:, :]
        return self.suffix_inputs.project_actions(action_hidden)

    def denoise_actions(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        prefix_end_positions: torch.Tensor,
        *,
        num_steps: int = 10,
        attn_fn: Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Integrate ``dx/dt = v_theta(x_t, t)`` from ``t=1.0`` down to ``t=0.0`` via Euler steps."""
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        bs, chunk_size, _ = x_t.shape
        suffix_len = 1 + chunk_size
        device = x_t.device
        offsets = torch.arange(suffix_len, device=device, dtype=torch.long)
        pos_list = [
            (prefix_end_positions[b].to(device=device, dtype=torch.long) + offsets).unsqueeze(0).expand(3, -1)
            for b in range(bs)
        ]
        suffix_pos_3d = torch.cat(pos_list, dim=1)
        cos_sin = self._compute_suffix_rope(suffix_pos_3d)

        x = x_t.to(torch.float32)
        dt = -1.0 / float(num_steps)
        for step in range(num_steps):
            t_val = 1.0 + step * dt
            timestep = torch.full((bs,), t_val, device=device, dtype=torch.float32)
            v_t = self.forward_step(
                state,
                x,
                timestep,
                suffix_pos_3d,
                cos_sin=cos_sin,
                attn_fn=attn_fn,
            )
            x = x + dt * v_t.to(torch.float32)
        return x


def load_action_expert_weights(
    expert: LingBotVLAActionExpert,
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[set[str], set[str]]:
    """Load suffix projections, all action layers, and final norm in a single shard pass.

    Works on both sharded indexed checkpoints (``model.safetensors.index.json``)
    and single-file safetensors checkpoints. Raises if required action-expert
    parameters are missing.
    """
    checkpoint_dir = Path(checkpoint_dir)
    n_layers = expert.config.num_hidden_layers

    # Collect per-layer / per-module weight pairs and apply as soon as complete.
    suffix_weights: list[tuple[str, torch.Tensor]] = []
    attn_buckets: dict[int, list[tuple[str, torch.Tensor]]] = {i: [] for i in range(n_layers)}
    film_buckets: dict[tuple[int, str], list[tuple[str, torch.Tensor]]] = {}
    moe_pending: dict[int, dict[str, torch.Tensor]] = {i: {} for i in range(n_layers)}
    final_norm_weights: list[tuple[str, torch.Tensor]] = []

    loaded_keys: set[str] = set()
    layer_prefix = _EXPERT_PREFIX + "layers."
    norm_prefix = _EXPERT_PREFIX + "norm."

    for name, tensor in iter_safetensors_shards(checkpoint_dir, device=device):
        if name.startswith(_SUFFIX_STEMS):
            suffix_weights.append((name.removeprefix("model."), tensor))
            loaded_keys.add(name)
            continue
        if name.startswith(norm_prefix):
            final_norm_weights.append((name.removeprefix(norm_prefix), tensor))
            loaded_keys.add(name)
            continue
        if not name.startswith(layer_prefix):
            continue

        rest = name.removeprefix(layer_prefix)
        dot = rest.find(".")
        if dot <= 0 or not rest[:dot].isdigit():
            continue
        layer_idx = int(rest[:dot])
        if layer_idx >= n_layers:
            continue
        sub = rest[dot + 1 :]
        loaded_keys.add(name)

        if sub.startswith("input_layernorm."):
            film_buckets.setdefault((layer_idx, "input_layernorm"), []).append(
                (sub.removeprefix("input_layernorm."), tensor)
            )
        elif sub.startswith("post_attention_layernorm."):
            film_buckets.setdefault((layer_idx, "post_attention_layernorm"), []).append(
                (sub.removeprefix("post_attention_layernorm."), tensor)
            )
        elif sub.startswith("self_attn."):
            attn_buckets[layer_idx].append((sub, tensor))
        elif sub.startswith("mlp."):
            mlp_key = sub.removeprefix("mlp.")
            bucket = moe_pending[layer_idx]
            moe = expert.layers[layer_idx].mlp
            with torch.no_grad():
                if mlp_key == "gate.weight":
                    default_weight_loader(moe.gate.weight, tensor)
                elif mlp_key == "e_score_correction_bias":
                    default_weight_loader(moe.gate.e_score_correction_bias, tensor)
                elif mlp_key == "experts.down_proj":
                    default_weight_loader(moe.experts.down_proj, tensor)
                elif mlp_key == "shared_expert.down_proj.weight":
                    default_weight_loader(moe.shared_expert.down_proj.weight, tensor)
                elif mlp_key in ("experts.gate_proj", "experts.up_proj"):
                    bucket[mlp_key] = tensor
                    if "experts.gate_proj" in bucket and "experts.up_proj" in bucket:
                        default_weight_loader(
                            moe.experts.gate_up_proj,
                            torch.cat(
                                (bucket.pop("experts.gate_proj"), bucket.pop("experts.up_proj")),
                                dim=1,
                            ),
                        )
                elif mlp_key in ("shared_expert.gate_proj.weight", "shared_expert.up_proj.weight"):
                    bucket[mlp_key] = tensor
                    if (
                        "shared_expert.gate_proj.weight" in bucket
                        and "shared_expert.up_proj.weight" in bucket
                    ):
                        default_weight_loader(
                            moe.shared_expert.gate_up_proj.weight,
                            torch.cat(
                                (
                                    bucket.pop("shared_expert.gate_proj.weight"),
                                    bucket.pop("shared_expert.up_proj.weight"),
                                ),
                                dim=0,
                            ),
                        )
                else:
                    raise ValueError(f"Unexpected MoE parameter key in action expert: {name}")

    # Apply collected module weights strictly per component.
    missing: set[str] = set()
    suffix_loaded = load_weights_into(expert.suffix_inputs, suffix_weights, strict=False)
    for p_name in dict(expert.suffix_inputs.named_parameters()):
        if p_name not in suffix_loaded:
            missing.add(f"suffix_inputs.{p_name}")

    for i, layer in enumerate(expert.layers):
        attn_loaded = load_weights_into(
            layer.self_attn,
            attn_buckets[i],
            stacked_params=LLAMA_STACKED_PARAMS[:3],
            strict=False,
        )
        expected_attn_raw = {
            f"self_attn.{part}_proj.{kind}" for part in "qkv" for kind in ("weight", "bias")
        } | {"self_attn.o_proj.weight"}
        seen_attn_raw = {k for k, _ in attn_buckets[i]}
        for raw_k in sorted(expected_attn_raw - seen_attn_raw):
            missing.add(f"layers.{i}.{raw_k}")
        for p_name in dict(layer.self_attn.named_parameters()):
            if p_name not in attn_loaded:
                missing.add(f"layers.{i}.{p_name}")

        for kind in ("input_layernorm", "post_attention_layernorm"):
            norm_mod = getattr(layer, kind)
            norm_loaded = load_weights_into(norm_mod, film_buckets.get((i, kind), []), strict=False)
            for p_name in dict(norm_mod.named_parameters()):
                if p_name not in norm_loaded:
                    missing.add(f"layers.{i}.{kind}.{p_name}")

        if moe_pending[i]:
            missing.update(f"layers.{i}.mlp.{k}" for k in moe_pending[i])
        # Verify MoE tensors were actually populated (all 8 expected keys per layer).
        expected_moe_keys = {
            f"{layer_prefix}{i}.mlp.{k}"
            for k in (
                "e_score_correction_bias",
                "experts.down_proj",
                "experts.gate_proj",
                "experts.up_proj",
                "gate.weight",
                "shared_expert.down_proj.weight",
                "shared_expert.gate_proj.weight",
                "shared_expert.up_proj.weight",
            )
        }
        for ek in expected_moe_keys:
            if ek not in loaded_keys:
                missing.add(ek)

    norm_loaded = load_weights_into(expert.norm, final_norm_weights, strict=False)
    for p_name in dict(expert.norm.named_parameters()):
        if p_name not in norm_loaded:
            missing.add(f"norm.{p_name}")

    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"LingBot-VLA action expert checkpoint left {len(missing)} key(s) unloaded, e.g. {sample}"
        )
    return loaded_keys, missing
