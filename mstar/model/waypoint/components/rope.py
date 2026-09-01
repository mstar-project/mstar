"""OrthoRoPE: Waypoint's orthogonal (x, y, t) rotary position embedding.

Port of ``OrthoRoPEAngles`` and ``OrthoRoPE`` from
``world_engine/src/model/attn.py``. "Ortho" refers to the frequency layout: the
head dim's rotation pairs are partitioned into three disjoint bands, so the x,
y and t coordinates never share a rotation plane and their phases add
independently.

For ``d_head == 64`` the split is ``d_xy = d_head // 8 = 8`` rotation pairs each
for x and y and ``d_t = d_head // 4 = 16`` pairs for t — **32 pairs, which is
every one of the 64 dims**. Nothing is left unrotated: x owns dims 0..15, y
dims 16..31, t dims 32..63. (Pairs, not dims, is the trap. Reading ``d_xy``/
``d_t`` as dim counts gives "32 rotated dims, the top 32 untouched", which is
wrong — the reference builds a ``[..., d_head // 2]`` angle table and applies
it to all ``d_head // 2`` pairs. ``CONTRACTS.md`` section 4.3 and ``config.py``
both stated the wrong reading until this port corrected them.)

Numerics contract (``docs/waypoint/CONTRACTS.md`` section 4.1):

  * Both classes are fp32 islands — the reference marks them ``NoCastModule``,
    i.e. they refuse ``.to(dtype)``. The port does not reproduce that mechanism
    (it warns, and it fights ``to_empty``). Instead the arithmetic is
    unconditionally fp32: the tables are fp32, the bodies run under
    ``autocast(enabled=False)``, and ``OrthoRoPE`` calls ``.float()`` on its
    input before rotating and ``.type_as`` on the way out.
  * Consequently neither class appears in ``layers.FP32_MODULE_PATHS``: they
    hold no parameters and no buffers, so ``.to(torch.bfloat16)`` has nothing to
    corrupt and ``cast_serving_dtypes()`` has nothing to pin back. That is not
    an accident — the frequency tables are held in a ``DeviceTableCache``
    outside the module tree precisely so that no global cast, and no
    ``to_empty(device)``, can reach them.
  * The cos/sin tables are DERIVED state, not checkpoint state. As non-
    persistent buffers they would survive mstar's meta build as uninitialized
    garbage (``to_empty`` allocates, it does not fill, and the loader's
    completeness check covers parameters, not buffers). They are built on CPU
    at init and copied per device on first use, matching
    ``wan22.components.dit.Wan22RoPE3D``.

The cache stores post-RoPE keys (section 4.2), so replayed history is never
re-rotated; the angles a frame is rotated with are the angles it keeps forever.
"""

import torch
from torch import nn

from mstar.model.waypoint.components.layers import DeviceTableCache
from mstar.model.waypoint.config import WaypointConfig


class OrthoRoPEAngles(nn.Module):
    """Builds the shared ``(cos, sin)`` angle tables for one forward.

    Lives once under the DiT (not once per block): every block rotates with the
    same angles, so the tables are computed once per forward and threaded into
    each ``Attn``.

    Frequency layout, verbatim from the reference:

      * spatial — ``(linspace(1.0, max_freq / 2, (d_xy + 1) // 2) * pi)``
        ``.repeat_interleave(2)[:d_xy]``, with
        ``max_freq = min(height, width) * rope_nyquist_frac``. The Nyquist cap
        is what keeps the highest spatial frequency below one cycle per two
        cells on the *shorter* grid axis, so nothing aliases. The
        ``repeat_interleave(2)`` makes adjacent rotation pairs share a
        frequency; x and y share one table.
      * temporal — ``(1 / theta ** (arange(0, d_t, 2) / d_t))``
        ``.repeat_interleave(2)``, the standard NTK-style geometric ladder.

    Positions: x/y are normalized to ``[-1, 1)`` **cell centers**
    (``(2 * p + 1) / extent - 1``, so the grid is symmetric about 0 and
    resolution-independent), while t stays a raw frame count so that history
    beyond the ring is still phase-distinguishable.
    """

    def __init__(self, config: WaypointConfig):
        super().__init__()
        self.config = config

        d_head = config.d_head
        if d_head % 8:
            raise ValueError(f"OrthoRoPE needs d_head divisible by 8 (x/y/t band split); got {d_head}.")
        d_xy, d_t = d_head // 8, d_head // 4

        # device="cpu" everywhere below is load-bearing: __init__ runs inside
        # `with torch.device("meta")` in mstar's build path, and meta tensors
        # carry no data to compute frequencies from.
        max_freq = min(config.height, config.width) * float(config.rope_nyquist_frac)
        n = (d_xy + 1) // 2
        xy = torch.linspace(1.0, max_freq / 2, n, dtype=torch.float32, device="cpu") * torch.pi
        xy = xy.repeat_interleave(2)[:d_xy]  # [d_xy]

        theta = float(config.rope_theta)
        inv_t = 1.0 / (theta ** (torch.arange(0, d_t, 2, dtype=torch.float32, device="cpu") / d_t))
        inv_t = inv_t.repeat_interleave(2)  # [d_t]

        self._tables = DeviceTableCache(xy, inv_t)

    def forward(
        self, x_pos: torch.Tensor, y_pos: torch.Tensor, t_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x_pos``/``y_pos``/``t_pos`` are ``[B, T]`` integer position grids;
        returns ``(cos, sin)``, each ``[B, 1, T, d_head // 2]`` fp32 (the head
        axis is a broadcast singleton).

        ``t_pos`` is the RoPE clock and is NOT the ring clock ``f_pos``; they
        happen to be equal for this checkpoint (``ts_mult == 1``) and diverge
        for any other serving fps. Section 4.4.
        """
        xy, inv_t = self._tables.get(x_pos.device)

        if not torch.compiler.is_compiling():
            # Out-of-range positions produce wrapped phases rather than an
            # index error, so this is checked rather than trusted.
            torch._assert(
                (y_pos.max() < self.config.height) & (x_pos.max() < self.config.width),
                f"pos_ids out of bounds, {self.config.height}, {self.config.width}",
            )

        with torch.amp.autocast("cuda", enabled=False):
            # Cell centers in [-1, 1); t stays a raw (unnormalized) frame count.
            x = (2.0 * x_pos.float() + 1.0) / self.config.width - 1.0
            y = (2.0 * y_pos.float() + 1.0) / self.config.height - 1.0
            t = t_pos.float()

            # x and y share the `xy` table; the bands are disjoint slices of the
            # angle vector, which is what makes the three axes orthogonal.
            freqs = torch.cat(
                (x.unsqueeze(-1) * xy, y.unsqueeze(-1) * xy, t.unsqueeze(-1) * inv_t),
                dim=-1,  # [B, T, d_head // 2]
            )
            return freqs.cos()[:, None], freqs.sin()[:, None]


def apply_ortho_rope(x: torch.Tensor, rope_angles: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Rotate ``x`` ``[B, H, T, d_head]`` by the ``(cos, sin)`` tables.

    **This is the interleaved-pair form with a concatenated output, and the
    asymmetry is deliberate.** Pairs are read interleaved — ``unfold(-1, 2, 2)``
    splits the head into ``(x[..., 0::2], x[..., 1::2])`` — but the two rotated
    halves are written back with ``cat``, so result ``i`` of the even stream
    lands at output dim ``i`` and result ``i`` of the odd stream at
    ``i + d_head // 2``. The output is therefore a *permutation* of the
    conventional interleaved layout, not the interleaved layout itself.

    Rewriting this as an in-place interleave (``out[..., 0::2] = ...``) is the
    natural "fix". Do not do it — but the reason is narrower than it looks, and
    an earlier version of this docstring overstated it as "a scrambled head".

    Measured: the interleave rewrite changes the cached K by **5.6** and the
    model output by **7.2e-07**, i.e. the noise floor. It is invisible at the
    output because q and k receive the *same* head-dim permutation and V is
    never rotated, so the permutation cancels inside ``q @ k.T``. What it is
    NOT invisible to is anything that reads K or Q directly: a parity harness
    diffing cache contents, a quantizer with per-channel scales, a tensor-
    parallel head split, or any future consumer of the stored keys. The port
    keeps the reference's layout so that stored state is comparable
    bit-for-bit, not because attention would break. See CONTRACTS section 4.3.

    fp32 island: ``x`` is upcast before the rotation and cast back at the end,
    so the trig math never runs in bf16 regardless of the ambient autocast.
    """
    cos, sin = rope_angles
    with torch.amp.autocast("cuda", enabled=False):
        x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).type_as(x)


class OrthoRoPE(nn.Module):
    """Module wrapper around :func:`apply_ortho_rope`.

    Stateless — no parameters, no buffers — but kept as an ``nn.Module`` and
    constructed per attention layer (``self.rope = OrthoRoPE(config)``) because
    that is the reference's shape and the attention port calls it as
    ``self.rope(q, rope_angles)``. ``config`` is accepted and stored for that
    call-site parity; the reference ignores it here too.
    """

    def __init__(self, config: WaypointConfig | None = None):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor, rope_angles: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return apply_ortho_rope(x, rope_angles)
