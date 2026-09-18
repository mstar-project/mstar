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
        self.fwd = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.bwd = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        forward_out, _ = self.fwd(x)
        backward_out, _ = self.bwd(reverse_padded(x, lengths))
        out = torch.cat([forward_out, reverse_padded(backward_out, lengths)], dim=-1)
        return mask_features(out, length_mask(lengths, x.shape[1]))
