"""Bidirectional LSTM over padded batches without packing.

``nn.utils.rnn.pack_padded_sequence`` needs the lengths on the host and drives
a data-dependent kernel schedule, which is why the reference runs one request
at a time. Running the backward direction over the *reversed valid prefix* of
each row instead gives the same result as packing: each direction starts from
a zero state at its own end of the valid span, and the padded tail only ever
produces outputs that are masked away.
"""

from __future__ import annotations

import torch
from torch import nn

from mstar.model.kokoro.components.masking import length_mask, mask_features, reverse_padded


class MaskedBiLSTM(nn.Module):
    """Single-layer bidirectional LSTM, ``[B, T, in] -> [B, T, 2 * hidden]``.

    ``fwd`` and ``bwd`` hold the checkpoint's ``*_l0`` and ``*_l0_reverse``
    parameters respectively (see the weight loader).
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.fwd = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.bwd = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Set by ``fuse``; kept out of the state dict, which stays the checkpoint's.
        self._fused: list[nn.LSTM] = []

    @torch.no_grad()
    def fuse(self) -> None:
        """Run both directions as one LSTM of twice the width.

        Its weights are block-diagonal: each gate's first ``hidden`` rows read
        the forward input and state, the second the reversed input and the
        backward state, so the two directions stay independent while their
        time steps advance together. The recurrence is a chain of tiny,
        latency-bound kernels per step, so this halves the steps' cost.
        """
        h, i = self.hidden_size, self.fwd.input_size
        ref = self.fwd.weight_ih_l0
        fused = nn.LSTM(2 * i, 2 * h, batch_first=True).to(device=ref.device, dtype=ref.dtype)
        for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            dst = getattr(fused, name)
            dst.zero_()
            for direction, (rows, cols) in enumerate(((0, 0), (h, i if "ih" in name else h))):
                src = getattr(self.bwd if direction else self.fwd, name)
                for gate in range(4):
                    block = src[gate * h : (gate + 1) * h]
                    r = gate * 2 * h + rows
                    if block.dim() == 1:
                        dst[r : r + h] = block
                    else:
                        dst[r : r + h, cols : cols + block.shape[1]] = block
        fused.flatten_parameters()
        self._fused = [fused]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        reversed_x = reverse_padded(x, lengths)
        if self._fused:
            out, _ = self._fused[0](torch.cat([x, reversed_x], dim=-1))
            forward_out, backward_out = out.split(self.hidden_size, dim=-1)
        else:
            forward_out, _ = self.fwd(x)
            backward_out, _ = self.bwd(reversed_x)
        out = torch.cat([forward_out, reverse_padded(backward_out, lengths)], dim=-1)
        return mask_features(out, length_mask(lengths, x.shape[1]))
