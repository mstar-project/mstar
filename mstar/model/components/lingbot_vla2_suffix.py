"""LingBot-VLA 2.0's state, noisy-action and timestep input projections.

The released V2 policy inherits ``embed_suffix`` from its V1 flow-matching
model. This component keeps those checkpoint parameter names and assembles the
expert input; the decoder, action output projection and graph are separate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.loader.base import load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards
from mstar.model.pi05.components.flow_matching import sincos_timestep_embedding


@dataclass(frozen=True)
class LingBotVLASuffixConfig:
    max_state_dim: int
    max_action_dim: int
    expert_hidden_size: int
    action_fp32: bool = False


class LingBotVLASuffixInputs(nn.Module):
    """Build ``[state, noisy actions]`` rows and the reference's two masks."""

    def __init__(self, config: LingBotVLASuffixConfig):
        super().__init__()
        if min(config.max_state_dim, config.max_action_dim, config.expert_hidden_size) <= 0:
            raise ValueError("LingBot-VLA suffix dimensions must be positive")
        if config.expert_hidden_size % 2:
            raise ValueError("LingBot-VLA timestep width must be even")
        self.config = config
        width = config.expert_hidden_size
        self.state_proj = nn.Linear(config.max_state_dim, width)
        self.action_in_proj = nn.Linear(config.max_action_dim, width)
        self.action_time_mlp_in = nn.Linear(width * 2, width)
        self.action_time_mlp_out = nn.Linear(width, width)

    def _project(self, layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        if self.config.action_fp32:
            return F.linear(
                x.float(), layer.weight.float(),
                None if layer.bias is None else layer.bias.float(),
            )
        return layer(x)

    def forward(
        self, state: torch.Tensor, noisy_actions: torch.Tensor, timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return time embedding, expert rows, padding mask and attention mask.

        ``state`` is ``[batch, state_dim]``, actions are
        ``[batch, chunk, action_dim]`` and the flow timestep is ``[batch]``.
        The attention mask is the reference's segment marker; the attention
        resource will turn it into the actual 2D visibility rule.
        """
        if state.ndim != 2 or state.shape[1] != self.config.max_state_dim:
            raise ValueError("LingBot-VLA state must be [batch, max_state_dim]")
        if noisy_actions.ndim != 3 or noisy_actions.shape[:1] != state.shape[:1] or (
            noisy_actions.shape[2] != self.config.max_action_dim
        ):
            raise ValueError("LingBot-VLA actions must be [batch, chunk, max_action_dim]")
        if timestep.ndim != 1 or timestep.shape[0] != state.shape[0]:
            raise ValueError("LingBot-VLA timestep must be [batch]")
        if noisy_actions.device != state.device or timestep.device != state.device:
            raise ValueError("LingBot-VLA suffix inputs must be on the same device")

        width = self.config.expert_hidden_size
        # The reference constructs this ladder in fp32 even when the model's
        # parameters are bf16. Recompute it after dtype/device moves.
        fraction = torch.linspace(0, 1, width // 2, dtype=torch.float32, device=state.device)
        time_fp32 = sincos_timestep_embedding(
            timestep.float(), width, fraction,
            torch.empty(state.shape[0], width, dtype=torch.float32, device=state.device),
        )
        time_emb = time_fp32.to(state.dtype)
        state_emb = self._project(self.state_proj, state)
        action_emb = self._project(self.action_in_proj, noisy_actions)
        time_rows = time_emb[:, None, :].expand(-1, action_emb.shape[1], -1)
        action_time = torch.cat((action_emb, time_rows), dim=-1)
        action_time = F.silu(self._project(self.action_time_mlp_in, action_time))
        action_time = self._project(self.action_time_mlp_out, action_time)

        embeddings = torch.cat((state_emb[:, None], action_time), dim=1)
        pad_mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=state.device)
        attention_mask = torch.zeros_like(pad_mask)
        attention_mask[:, :2] = True
        return time_emb, embeddings, pad_mask, attention_mask


def load_suffix_input_weights(
    suffix: LingBotVLASuffixInputs, checkpoint_dir: str | Path,
    *, device: str | torch.device = "cpu",
) -> set[str]:
    """Read only the eight indexed input-projection tensors from the VLA checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"LingBot-VLA suffix requires {index_path}")
    expected = {f"model.{name}" for name in dict(suffix.named_parameters())}
    index_keys = set(json.loads(index_path.read_text())["weight_map"])
    stems = ("model.state_proj.", "model.action_in_proj.",
             "model.action_time_mlp_in.", "model.action_time_mlp_out.")
    actual = {name for name in index_keys if name.startswith(stems)}
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"LingBot-VLA suffix checkpoint keys differ: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(extra)} extra (e.g. {extra[:3]})"
        )
    weights = iter_safetensors_shards(checkpoint_dir, device=device, keys=expected)
    return load_weights_into(
        suffix, ((name.removeprefix("model."), value) for name, value in weights), strict=True,
    )
