"""Small helpers shared by the Nemotron-VoiceChat component ports.

These stages (Conformer STT, RNN-T, EarTTS talker, RVQ codec) are ported to
**exact checkpoint parity** so the single composite ``model.safetensors`` loads
with no missing / unexpected / shape-mismatched tensors. The forward passes for
the audio stages are Phase 4/5 (they need the NeMo / EarTTS reference to verify
numerically); the module *trees* here are authoritative and load-verified.
"""
from __future__ import annotations

import torch
from torch import nn

_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16, torch.float64)


def param(*shape: int, dtype: torch.dtype = torch.float32) -> nn.Parameter:
    """A parameter of an exact shape/dtype. Int tensors (checkpoint buffers such
    as flag tables / control codes) are registered as non-trainable parameters
    so the standard M* loader — which fills ``named_parameters()`` only — loads
    them too."""
    return nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=dtype in _FLOAT_DTYPES)


class RawWeight(nn.Module):
    """Module holding a single bare ``weight`` parameter of an exact shape.

    Used where the checkpoint stores ``<layer>.weight`` directly on a module
    (e.g. the codec's strided resample convs, whose Conv1d-vs-ConvTranspose1d
    orientation isn't needed for loading)."""

    def __init__(self, *shape: int):
        super().__init__()
        self.weight = param(*shape)
