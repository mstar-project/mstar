"""DeepSeek V4.1 MoE on the existing clamped expert and shared-MLP paths."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from mstar.model.components.moe import NoAuxTcRouter
from mstar.model.glm5_next.components.moe import Glm5NextSparseMoeBlock
from mstar.utils.deepseek_v41_expert_weights import convert_expert_fp4_to_fp8
from mstar.utils.fp8_block import Fp8BlockQuantConfig, dequantize_fp8_block_weight

_GATE = re.compile(r"(?:model\.)?layers\.(\d+)\.ffn\.gate\.(weight|bias|bias_vl)$")
_PROJECTION = re.compile(r"(?:model\.)?layers\.(\d+)\.ffn\.(experts\.(\d+)|shared_experts)\.(w[123])$")
_SHARD = {"w1": "gate", "w3": "up", "w2": "down"}


@dataclass(frozen=True)
class DeepSeekV41MoeConfig:
    """The published text config's MoE fields, with FP4 experts upcast to FP8."""

    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    swiglu_limit: float
    norm_topk_prob: bool = True
    n_shared_experts: int = 1
    hidden_act: str = "silu"
    gate_temp: float = 1.0
    vision_enabled: bool = False
    quantization_config: Fp8BlockQuantConfig = field(
        default_factory=lambda: Fp8BlockQuantConfig(weight_block_size=(32, 32))
    )
    moe_fp8_resident: bool = True
    moe_quant_kernel: str = "reference"

    def __post_init__(self) -> None:
        if (
            self.n_shared_experts != 1
            or self.hidden_act != "silu"
            or self.moe_quant_kernel != "reference"
            or not self.moe_fp8_resident
            or self.quantization_config.weight_block_size != (32, 32)
        ):
            raise ValueError("V4.1 MoE needs one shared clamped expert and the 32x32 FP8 reference path")
        if (
            self.hidden_size % 32
            or self.moe_intermediate_size % 32
            or self.n_routed_experts < 1
            or not 1 <= self.num_experts_per_tok <= self.n_routed_experts
        ):
            raise ValueError("V4.1 MoE dimensions need 32-wide weights and a valid top-k")

    @classmethod
    def from_hf(cls, config: dict) -> DeepSeekV41MoeConfig:
        """Read the official nested text_config; refuse a quant or router format we cannot load."""
        text = config["text_config"]
        quant = config["quantization_config"]
        if (
            text.get("scoring_func") != "sqrtsoftplus"
            or text.get("topk_method") != "noaux_tc"
            or quant.get("quant_method") != "fp8"
            or quant.get("expert_dtype") != "fp4"
            or quant.get("scale_fmt") != "ue8m0"
            or tuple(quant.get("weight_block_size", ())) != (32, 32)
        ):
            raise ValueError("unsupported DeepSeek V4.1 router or expert quantization format")
        return cls(
            hidden_size=int(text["hidden_size"]),
            moe_intermediate_size=int(text["moe_intermediate_size"]),
            n_routed_experts=int(text["n_routed_experts"]),
            num_experts_per_tok=int(text["num_experts_per_tok"]),
            routed_scaling_factor=float(text["routed_scaling_factor"]),
            swiglu_limit=float(text["swiglu_limit"]),
            norm_topk_prob=bool(text["norm_topk_prob"]),
            n_shared_experts=int(text["n_shared_experts"]),
            hidden_act=text["hidden_act"],
            gate_temp=float(text.get("gate_temp", 1.0)),
            vision_enabled=bool(config.get("vision_config")),
        )


class DeepSeekV41MoE(Glm5NextSparseMoeBlock):
    """Reference dispatch for V4.1, reusing GLM's clamped expert and shared MLP."""

    def __init__(self, layer_id: int, config: DeepSeekV41MoeConfig, comm_group=None) -> None:
        super().__init__(config, comm_group=comm_group)
        self.layer_id = layer_id
        self.gate = NoAuxTcRouter(
            config.hidden_size,
            config.n_routed_experts,
            config.num_experts_per_tok,
            config.routed_scaling_factor,
            config.norm_topk_prob,
            score_func="sqrtsoftplus",
            gate_temp=config.gate_temp,
            image_bias=config.vision_enabled,
        )
        self._needed = {"gate.weight", "gate.bias", "shared_experts.w1", "shared_experts.w2", "shared_experts.w3"}
        if config.vision_enabled:
            self._needed.add("gate.bias_vl")
        self._needed.update(f"experts.{i}.{proj}" for i in range(config.n_routed_experts) for proj in _SHARD)
        self._loaded: set[str] = set()

    def _mark_loaded(self, key: str) -> None:
        if key in self._loaded:
            raise ValueError(f"duplicate V4.1 MoE weight {key}")
        self._loaded.add(key)

    def load_gate(self, checkpoint_name: str, tensor: torch.Tensor) -> None:
        match = _GATE.fullmatch(checkpoint_name)
        if match is None or int(match[1]) != self.layer_id:
            raise ValueError(f"wrong V4.1 MoE gate name {checkpoint_name!r}")
        key = f"gate.{match[2]}"
        if key not in self._needed or key in self._loaded:
            raise ValueError(f"unexpected or duplicate V4.1 MoE weight {key}")
        target = {"weight": self.gate.weight, "bias": self.gate.e_score_correction_bias, "bias_vl": self.gate.bias_vl}[
            match[2]
        ]
        if target is None or target.shape != tensor.shape:
            raise ValueError(f"V4.1 MoE {key} shape {tuple(tensor.shape)} does not match target")
        target.data.copy_(tensor)
        self._mark_loaded(key)

    def load_projection_pair(
        self,
        checkpoint_base: str,
        weight: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        """Load one published ``.weight``/``.scale`` pair; no full-layer weight map is held."""
        match = _PROJECTION.fullmatch(checkpoint_base)
        if match is None or int(match[1]) != self.layer_id:
            raise ValueError(f"wrong V4.1 MoE projection name {checkpoint_base!r}")
        owner, expert, proj = match[2], match[3], match[4]
        key = f"{owner}.{proj}"
        if key not in self._needed or key in self._loaded:
            raise ValueError(f"unexpected or duplicate V4.1 MoE weight {key}")
        shard = _SHARD[proj]
        if expert is not None:
            if weight.dtype == torch.float4_e2m1fn_x2:
                weight = weight.view(torch.int8)
            fp8, block_scales = convert_expert_fp4_to_fp8(weight, scale)
            param, scales = (
                (self.experts.down_proj_fp8, self.experts.down_proj_scale_inv)
                if shard == "down"
                else (self.experts.gate_up_proj_fp8, self.experts.gate_up_proj_scale_inv)
            )
            shard_id = f"{shard}:{expert}"
            param.weight_loader(param, fp8, shard_id)
            scales.weight_loader(scales, block_scales, shard_id)
        else:
            if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e8m0fnu:
                raise ValueError("V4.1 shared expert needs E4M3 weights and E8M0 scales")
            module = self.shared_expert.gate_up_proj if shard != "down" else self.shared_expert.down_proj
            target = module.weight
            dense = dequantize_fp8_block_weight(weight, scale, (32, 32), target.dtype)
            shard_id = {"gate": 0, "up": 1, "down": None}[shard]
            target.weight_loader(target, dense, shard_id)
        self._mark_loaded(key)

    def process_weights_after_loading(self, device) -> None:
        missing = self._needed - self._loaded
        if missing:
            raise ValueError(
                f"V4.1 MoE layer {self.layer_id} has {len(missing)} missing weights: {', '.join(sorted(missing)[:3])}"
            )
        super().process_weights_after_loading(device)

    def forward(self, hidden_states: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size).contiguous()
        weights, ids, _ = self.gate(flat, image_mask=image_mask)
        routed = self._dispatch_clamped(flat, weights, ids)
        out = routed + self.shared_expert(flat)
        if self.tp_size > 1:
            out = self.comm_group.all_reduce(out)
        return out.view(shape)
