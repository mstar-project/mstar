"""OrthoRoPE: Waypoint's orthogonal (x, y, t) rotary position embedding.

``d_xy`` and ``d_t`` count rotation *pairs*, not dims: at ``d_head == 64`` x
owns dims 0..15, y 16..31 and t 32..63, and nothing is left unrotated. The
three bands are disjoint, so the axes' phases add independently.

Both classes are fp32 islands, made so by running the arithmetic in fp32 rather
than by pinning dtypes: they hold no parameter and no buffer, so neither
appears in ``layers.FP32_MODULE_PATHS``, and their frequency tables live in a
``DeviceTableCache`` outside the module tree.

The cache stores post-RoPE keys, so replayed history is never re-rotated.
"""

import torch
from torch import nn

from mstar.model.waypoint.components.layers import DeviceTableCache, bf16_roundtrip
from mstar.model.waypoint.config import WaypointConfig


class OrthoRoPEAngles(nn.Module):
    """Builds the shared ``(cos, sin)`` angle tables for one forward, once
    under the DiT rather than once per block."""

    def __init__(self, config: WaypointConfig):
        super().__init__()
        self.config = config

        d_head = config.d_head
        if d_head % 8:
            raise ValueError(f"OrthoRoPE needs d_head divisible by 8 (x/y/t band split); got {d_head}.")
        d_xy, d_t = d_head // 8, d_head // 4

        # device="cpu" is load-bearing: __init__ runs under
        # `with torch.device("meta")`, and meta tensors carry no data.
        # The Nyquist factor holds the top spatial frequency under one cycle
        # per two cells on the *shorter* grid axis.
        max_freq = min(config.height, config.width) * float(config.rope_nyquist_frac)
        n = (d_xy + 1) // 2
        xy = torch.linspace(1.0, max_freq / 2, n, dtype=torch.float32, device="cpu") * torch.pi
        xy = xy.repeat_interleave(2)[:d_xy]  # [d_xy]

        theta = float(config.rope_theta)
        inv_t = 1.0 / (theta ** (torch.arange(0, d_t, 2, dtype=torch.float32, device="cpu") / d_t))
        inv_t = inv_t.repeat_interleave(2)  # [d_t]

        if config.reference_compat:
            # The reference serves these bf16-quantized; see the config field.
            xy, inv_t = bf16_roundtrip(xy), bf16_roundtrip(inv_t)

        self._tables = DeviceTableCache(xy, inv_t)

    def materialize(self, device: torch.device | str) -> None:
        """Copy derived frequency tables before compilation/capture warmup."""
        self._tables.get(torch.device(device))

    def forward(
        self, x_pos: torch.Tensor, y_pos: torch.Tensor, t_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, T]`` integer position grids -> ``(cos, sin)``, each
        ``[B, 1, T, d_head // 2]`` fp32 with a broadcast head axis.

        ``t_pos`` is the RoPE clock, not the ring clock ``f_pos``; they are
        equal for this checkpoint (``ts_mult == 1``) and diverge at any other
        serving fps.
        """
        xy, inv_t = self._tables.get(x_pos.device)

        if not torch.compiler.is_compiling():
            # Out-of-range positions wrap the phase instead of raising.
            torch._assert(
                (y_pos.max() < self.config.height) & (x_pos.max() < self.config.width),
                f"pos_ids out of bounds, {self.config.height}, {self.config.width}",
            )

        with torch.amp.autocast("cuda", enabled=False):
            # Cell centers in [-1, 1); t stays a raw (unnormalized) frame count.
            x = (2.0 * x_pos.float() + 1.0) / self.config.width - 1.0
            y = (2.0 * y_pos.float() + 1.0) / self.config.height - 1.0
            t = t_pos.float()

            # x and y share `xy`; the disjoint slices make the axes orthogonal.
            freqs = torch.cat(
                (x.unsqueeze(-1) * xy, y.unsqueeze(-1) * xy, t.unsqueeze(-1) * inv_t),
                dim=-1,  # [B, T, d_head // 2]
            )
            return freqs.cos()[:, None], freqs.sin()[:, None]


def apply_ortho_rope(x: torch.Tensor, rope_angles: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Rotate ``x`` ``[B, H, T, d_head]`` by the ``(cos, sin)`` tables.

    The output is a permutation of the interleaved layout (``unfold`` in,
    ``cat`` out); it cancels inside ``q @ k.T``, and is kept so that stored K/Q
    compare bit-for-bit with the reference. fp32 island: ``x`` is upcast for
    the rotation and cast back after.
    """
    cos, sin = rope_angles
    with torch.amp.autocast("cuda", enabled=False):
        x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).type_as(x)


class OrthoRoPE(nn.Module):
    """Stateless module wrapper around :func:`apply_ortho_rope`, constructed
    per attention layer to match the reference's call site. ``config`` is
    stored for that parity and otherwise unused."""

    def __init__(self, config: WaypointConfig | None = None):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor, rope_angles: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return apply_ortho_rope(x, rope_angles)
