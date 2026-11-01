"""Per-embodiment MLPs: one set of weights per robot, picked by its id.

GR00T serves 32 embodiments from one checkpoint by giving its state encoder,
action encoder and action decoder a leading category axis, indexed per
request. Class and parameter names are the reference's, so the checkpoint's
keys map straight on.
"""
from __future__ import annotations

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """``[B, T]`` timesteps to ``[B, T, embedding_dim]``, the action encoder's
    own encoding of the denoising time.

    **Not** the DT's ``TimestepEncoder`` convention, though both live in one
    model: this is sin then cos with no flip, spaced by ``log(10000) /
    half_dim`` where the usual form divides by ``half_dim - 1``. It has no
    parameters, so the checkpoint cannot reveal a wrong convention; only the
    reference's own class can.

    The exponent is fp32 throughout, as the reference builds it; fp64 then a
    cast diverges. Spelled out rather than delegated to diffusers'
    ``Timesteps(flip_sin_to_cos=False, downscale_freq_shift=0)``, the same
    maths by another route, because at the checkpoint's width (``half_dim``
    768, not a power of two) the two part by more than rounding.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Rank 2 only, as the reference. A `[B]` tensor meant for the DT's
        # `TimestepEncoder` would otherwise pass silently under the wrong
        # sinusoid.
        if timesteps.dim() != 2:
            raise ValueError(
                f"the action encoder's timesteps are [B, T], got "
                f"{list(timesteps.shape)}; the DT's TimestepEncoder is the "
                "one that takes [B]"
            )
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(
            half_dim, dtype=torch.float, device=timesteps.device,
        ) * (torch.log(torch.tensor(10000.0)) / half_dim)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class CategorySpecificLinear(nn.Module):
    """``x @ W[id] + b[id]``, with ``W`` held per embodiment."""

    def __init__(self, num_categories: int, input_dim: int, output_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_categories, input_dim, output_dim))
        self.b = nn.Parameter(torch.empty(num_categories, output_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, T, input_dim]`` and ``cat_ids`` is ``[B]``."""
        return torch.bmm(x, self.W[cat_ids]) + self.b[cat_ids].unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two per-embodiment linears with a ReLU between."""

    def __init__(
        self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int,
    ):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        return self.layer2(torch.relu(self.layer1(x, cat_ids)), cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    """Noisy actions plus the denoising time, per embodiment, into DT tokens.

    ``W1`` embeds the action; the time is encoded by this file's own sinusoid;
    the two are **concatenated** (which is why the checkpoint's ``W2`` is
    ``2 * hidden`` wide) and mixed by ``W2`` under a swish, then ``W3``.
    """

    def __init__(self, num_categories: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.W1 = CategorySpecificLinear(num_categories, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_categories, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_categories, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        cat_ids: torch.Tensor,
    ) -> torch.Tensor:
        """``[B, T, action_dim]`` and one timestep per request to ``[B, T, hidden]``."""
        batch, horizon, _ = actions.shape
        # one time per request, spread over the horizon, as the reference does
        if timesteps.dim() != 1 or timesteps.shape[0] != batch:
            raise ValueError(
                f"one timestep per request, so [{batch}], got {list(timesteps.shape)}"
            )
        timesteps = timesteps.unsqueeze(1).expand(-1, horizon)

        action = self.W1(actions, cat_ids)
        tau = self.pos_encoding(timesteps).to(dtype=action.dtype)
        mixed = torch.cat([action, tau], dim=-1)
        # `x * sigmoid(x)`, as the reference writes it: `F.silu` is not
        # bit-identical to it
        mixed = self.W2(mixed, cat_ids)
        return self.W3(mixed * torch.sigmoid(mixed), cat_ids)


def build_action_encoder(config) -> MultiEmbodimentActionEncoder:
    """The ``denoise_step`` node's action encoder."""
    return MultiEmbodimentActionEncoder(
        num_categories=config.max_num_embodiments,
        action_dim=config.max_action_dim,
        hidden_size=config.input_embedding_dim,
    )


def build_action_decoder(config) -> CategorySpecificMLP:
    """The DT's output rows back to an action velocity, per embodiment.

    Its input width is the DT's ``output_dim``, not the DT's inner width.
    """
    return CategorySpecificMLP(
        num_categories=config.max_num_embodiments,
        input_dim=config.dit.output_dim,
        hidden_dim=config.hidden_size,
        output_dim=config.max_action_dim,
    )


def build_state_encoder(config) -> CategorySpecificMLP:
    """Raw state in, one conditioning token out (``[B, 1, input_embedding_dim]``).

    Its node names it ``state_encoder``, the reference's attribute name under
    ``action_head.`` (``weight_loader.NODE_PREFIXES``).
    """
    return CategorySpecificMLP(
        num_categories=config.max_num_embodiments,
        # the reference folds a state history into the one token's width
        input_dim=config.max_state_dim * config.state_history_length,
        hidden_dim=config.hidden_size,
        output_dim=config.input_embedding_dim,
    )
