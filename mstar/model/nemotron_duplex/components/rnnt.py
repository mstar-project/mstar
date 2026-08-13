"""RNN-T transcription head (``stt_model.rnnt_decoder.*`` / ``rnnt_joint.*``).

The streaming user-transcription branch: a 2-layer LSTM prediction network and a
joint network combining encoder + prediction features. Ported to exact
checkpoint parity; the LSTM params are declared by hand (matching ``nn.LSTM``'s
``weight_ih_l{n}`` / ``weight_hh_l{n}`` / ``bias_*`` names) to avoid meta-device
init issues. Forward is Phase 4.
"""
from __future__ import annotations

from torch import nn

from mstar.model.nemotron_duplex.components._util import param

PRED_HIDDEN = 640
ENC_DIM = 1024
VOCAB = 1025      # RNN-T vocab incl. blank


class _LSTM(nn.Module):
    """Hand-rolled 2-layer LSTM parameter set matching ``nn.LSTM`` names."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        gate = 4 * hidden_size
        for layer in range(num_layers):
            in_dim = input_size if layer == 0 else hidden_size
            setattr(self, f"weight_ih_l{layer}", param(gate, in_dim))
            setattr(self, f"weight_hh_l{layer}", param(gate, hidden_size))
            setattr(self, f"bias_ih_l{layer}", param(gate))
            setattr(self, f"bias_hh_l{layer}", param(gate))


class _DecRnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = _LSTM(PRED_HIDDEN, PRED_HIDDEN, num_layers=2)


class _Prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, PRED_HIDDEN)
        self.dec_rnn = _DecRnn()


class RnntDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.prediction = _Prediction()

    @staticmethod
    def remap(name: str) -> str | None:
        prefix = "stt_model.rnnt_decoder."
        return name[len(prefix):] if name.startswith(prefix) else None

    def load_weights(self, weights):
        from mstar.model.loader import load_hf_weights

        return load_hf_weights(self, weights, stacked_params=[], name_remapper=self.remap)


class RnntJoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(ENC_DIM, PRED_HIDDEN)
        self.pred = nn.Linear(PRED_HIDDEN, PRED_HIDDEN)
        # joint_net: [activation, dropout, Linear] — only index 2 carries params.
        self.joint_net = nn.Sequential(nn.ReLU(), nn.Dropout(0.0), nn.Linear(PRED_HIDDEN, VOCAB))

    @staticmethod
    def remap(name: str) -> str | None:
        prefix = "stt_model.rnnt_joint."
        return name[len(prefix):] if name.startswith(prefix) else None

    def load_weights(self, weights):
        from mstar.model.loader import load_hf_weights

        return load_hf_weights(self, weights, stacked_params=[], name_remapper=self.remap)
