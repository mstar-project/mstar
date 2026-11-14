"""Settled action latents to command rows, in the rig's own units.

The expert emits packed tokens: one token per noised latent frame, each
carrying ``frames_per_token`` robot rows end to end. Turning those back into
rows the arm can execute is the policy's last step, and it is where a wrong
convention shows up as plausible motion rather than as an error, so the three
pieces below follow the reference exactly.

The pieces are:

* the unpack, whose only subtlety is that the first chunk of a window owns
  the anchor latent, so its token 0 contributes one row instead of four,
* the clip, which puts the sampler's output back inside the range training
  normalized into, and
* the inverse q99 map, which is applied only to the dims that were
  normalized. The rot6d dims pass through untouched: they were never
  normalized, and scaling them would bend the rotation.

The forward and inverse q99 maps are deliberately not symmetric. The forward
divides by ``q99 - q01 + 1e-8`` to stay finite on a dim whose quantiles
coincide; the inverse multiplies by ``q99 - q01`` with no epsilon. Adding one
here would move every denormalized value by a hair, so it is left off.

Everything is torch rather than the reference's numpy: these run inside a
node forward, where tensor in and tensor out is the rule.
"""

from __future__ import annotations

import torch

# Robot rows carried by one action token.
FRAMES_PER_TOKEN = 4

# The range a gripper command may take.
GRIP_RANGE = (0.0, 1.0)

# The dims of a row that were q99-normalized, per rig. The rest of a row is
# rot6d, which is already a direction and is left alone.
Q99_DIMS = {
    "single_arm": (0, 1, 2, 9),
    "bimanual": (0, 1, 2, 9, 10, 11, 12, 19),
}

# The dims holding a gripper.
GRIP_DIMS = {"single_arm": (9,), "bimanual": (9, 19)}


def unpack_action_tokens(
    tokens: torch.Tensor,
    depth: int,
    action_dim: int,
    frames_per_token: int = FRAMES_PER_TOKEN,
) -> torch.Tensor:
    """Unpack ``(1, L, F * D)`` or ``(L, F * D)`` tokens to ``(rows, action_dim)``.

    At ``depth == 0`` the chunk owns the window's anchor latent: token 0
    repeats row 0, so only its first sub-frame is a real row and the chunk
    yields ``1 + (L - 1) * F`` rows. At ``depth >= 1`` every token yields
    ``F`` rows.

    ``packed // frames_per_token`` may exceed ``action_dim``: a row is stored
    in a slot wide enough for the rig, and the tail of the slot is padding,
    so each sub-frame is trimmed to ``action_dim``.
    """
    a = tokens.detach().to(torch.float32)
    if a.dim() == 3:
        a = a[0]
    if a.dim() != 2:
        raise ValueError(f"tokens must be (L, F*D) or (1, L, F*D), got {tuple(tokens.shape)}")
    n_lat, packed = a.shape
    if packed % frames_per_token:
        raise ValueError(
            f"a {packed}-wide token does not divide into {frames_per_token} sub-frames"
        )
    per = packed // frames_per_token
    if per < action_dim:
        raise ValueError(f"a sub-frame holds {per} values, below the rig's {action_dim}")
    sub = a.reshape(n_lat, frames_per_token, per)[..., :action_dim]
    if depth != 0:
        return sub.reshape(n_lat * frames_per_token, action_dim)
    # Token 0 is the anchor: keep its first sub-frame only.
    return torch.cat([sub[0, :1], sub[1:].reshape(-1, action_dim)], dim=0)


def pack_action_rows(
    rows: torch.Tensor, first_chunk: bool, frames_per_token: int = FRAMES_PER_TOKEN
) -> torch.Tensor:
    """The inverse of :func:`unpack_action_tokens`, for tests and for priming.

    Chunk 0 repeats row 0 across token 0 and then takes rows ``4j-3 .. 4j``;
    a later chunk takes four consecutive rows per token.
    """
    if rows.dim() != 2:
        raise ValueError(f"rows must be (T, D), got {tuple(rows.shape)}")
    t, d = rows.shape
    rows = rows.to(torch.float32)
    if not first_chunk:
        if t % frames_per_token:
            raise ValueError(f"{t} rows do not divide into {frames_per_token}-row tokens")
        return rows.reshape(t // frames_per_token, frames_per_token * d).contiguous()
    n_lat = (t - 1) // frames_per_token + 1
    out = torch.empty((n_lat, frames_per_token * d), dtype=torch.float32, device=rows.device)
    out[0] = rows[0].repeat(frames_per_token)
    for k in range(1, n_lat):
        out[k] = rows[frames_per_token * k - 3 : frames_per_token * k + 1].reshape(-1)
    return out


def clip_normalized(rows: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    """``rows`` with ``dims`` clipped to [-1, 1].

    Training clipped the normalized targets to this range; the sampler can
    leave it. A value already inside does not move.
    """
    out = rows.to(torch.float32).clone()
    out[..., dims] = out[..., dims].clamp(-1.0, 1.0)
    return out


def clip_grip(rows: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    """``rows`` with the gripper dims clipped to :data:`GRIP_RANGE`."""
    out = rows.to(torch.float32).clone()
    out[..., dims] = out[..., dims].clamp(*GRIP_RANGE)
    return out


def q99_denormalize(
    normalized: torch.Tensor,
    q01: torch.Tensor,
    q99: torch.Tensor,
    dims: torch.Tensor,
) -> torch.Tensor:
    """``(n + 1) / 2 * (q99 - q01) + q01`` on ``dims``, the inverse q99 map.

    No epsilon on the span here: see the module docstring.
    """
    out = normalized.to(torch.float32).clone()
    n = out[..., dims]
    lo = q01.to(torch.float32)[dims]
    hi = q99.to(torch.float32)[dims]
    out[..., dims] = (n + 1.0) / 2.0 * (hi - lo) + lo
    return out


def q99_normalize(
    actions: torch.Tensor,
    q01: torch.Tensor,
    q99: torch.Tensor,
    dims: torch.Tensor,
) -> torch.Tensor:
    """``clip(2 (a - q01) / (q99 - q01 + 1e-8) - 1, -1, 1)`` on ``dims``."""
    out = actions.to(torch.float32).clone()
    a = out[..., dims]
    lo = q01.to(torch.float32)[dims]
    hi = q99.to(torch.float32)[dims]
    out[..., dims] = (2.0 * (a - lo) / (hi - lo + 1e-8) - 1.0).clamp(-1.0, 1.0)
    return out


def denormalize_rows(
    rows: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor, dims: torch.Tensor
) -> torch.Tensor:
    """Clip the normalized dims to [-1, 1], then apply the inverse q99 map.

    The result stays within ``[q01, q99]`` on ``dims``.
    """
    return q99_denormalize(clip_normalized(rows, dims), q01, q99, dims)


def proprio_normalize(
    proprio: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """``(p - mean) / max(std, 1e-6)``.

    The floor on the standard deviation is what keeps a dim that never moved
    in the training data from dividing by zero.
    """
    s = std.to(torch.float32).clamp_min(1e-6)
    return (proprio.to(torch.float32) - mean.to(torch.float32)) / s
