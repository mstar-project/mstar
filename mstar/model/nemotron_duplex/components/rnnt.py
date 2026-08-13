"""RNN-T transcription head (``stt_model.rnnt_decoder.*`` / ``rnnt_joint.*``).

The streaming user-transcription branch: a 2-layer LSTM prediction network and a
joint network combining encoder + prediction features. Ported to exact
checkpoint parity; the LSTM params are declared by hand (matching ``nn.LSTM``'s
``weight_ih_l{n}`` / ``weight_hh_l{n}`` / ``bias_*`` names) to avoid meta-device
init issues.

Forward math mirrors NeMo ``RNNTDecoder.predict`` / ``RNNTJoint.joint``
(branch ``nemotron-labs-voicechat``, ``nemo/collections/asr/modules/rnnt.py``):

  * predict: ``embed(y)`` -> optional zero SOS prepend -> ``nn.LSTM`` (time-first,
    ``batch_first=False``, plain LSTM since ``normalization_mode is None``) ->
    prediction features ``g`` [B, U(+1), H] plus LSTM ``(h, c)`` state.
  * joint: ``enc(f)`` [B, T, H] and ``pred(g)`` [B, U, H] broadcast-added as
    ``enc(f)[:, :, None, :] + pred(g)[:, None, :, :]`` -> ``joint_net`` (ReLU ->
    Dropout(id in eval) -> Linear 640->1025) -> raw logits [B, T, U, 1025].
    ``log_softmax`` is ``None`` in the checkpoint config, so on CUDA no
    log-softmax is applied (raw logits are returned); argmax-equivalent either way.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.model.nemotron_duplex.components._util import param
from mstar.model.nemotron_duplex.config import RnntConfig


class _LSTM(nn.Module):
    """Hand-rolled LSTM parameter set matching ``nn.LSTM`` names."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        gate = 4 * hidden_size
        for layer in range(num_layers):
            in_dim = input_size if layer == 0 else hidden_size
            setattr(self, f"weight_ih_l{layer}", param(gate, in_dim))
            setattr(self, f"weight_hh_l{layer}", param(gate, hidden_size))
            setattr(self, f"bias_ih_l{layer}", param(gate))
            setattr(self, f"bias_hh_l{layer}", param(gate))


class _DecRnn(nn.Module):
    def __init__(self, pred_hidden: int, num_layers: int):
        super().__init__()
        self.lstm = _LSTM(pred_hidden, pred_hidden, num_layers)


class _Prediction(nn.Module):
    def __init__(self, num_classes: int, pred_hidden: int, num_layers: int):
        super().__init__()
        self.embed = nn.Embedding(num_classes, pred_hidden)
        self.dec_rnn = _DecRnn(pred_hidden, num_layers)


class RnntDecoder(nn.Module):
    def __init__(self, config: RnntConfig | None = None):
        super().__init__()
        self.config = config or RnntConfig()
        self.prediction = _Prediction(
            self.config.num_classes, self.config.pred_hidden, self.config.pred_rnn_layers,
        )
        self._lstm_cache: nn.LSTM | None = None

    @staticmethod
    def remap(name: str) -> str | None:
        prefix = "stt_model.rnnt_decoder."
        return name[len(prefix):] if name.startswith(prefix) else None

    def load_weights(self, weights):
        from mstar.model.loader import load_hf_weights

        return load_hf_weights(self, weights, stacked_params=[], name_remapper=self.remap)

    def _lstm(self) -> nn.LSTM:
        """Build (and cache) a real ``nn.LSTM`` from the hand-rolled params.

        Weights are fixed after loading, so the module is built once. Runs in
        eval mode: the checkpoint's inter-layer dropout (0.2) is inactive, so the
        forward is deterministic and matches the NeMo reference at inference.
        """
        hand = self.prediction.dec_rnn.lstm
        ref = hand.weight_ih_l0
        h, n = self.config.pred_hidden, self.config.pred_rnn_layers
        cache = self._lstm_cache
        if cache is None or cache.weight_ih_l0.device != ref.device or cache.weight_ih_l0.dtype != ref.dtype:
            lstm = nn.LSTM(h, h, num_layers=n, batch_first=False)
            lstm = lstm.to(device=ref.device, dtype=ref.dtype)
            with torch.no_grad():
                for layer in range(n):
                    for tag in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                        name = f"{tag}_l{layer}"
                        getattr(lstm, name).copy_(getattr(hand, name))
            lstm.eval()
            self._lstm_cache = lstm
            cache = lstm
        return cache

    def predict(self, y, state=None, add_sos: bool = True, batch_size: int | None = None):
        """Prediction-network forward (NeMo ``RNNTDecoder.predict``).

        Args:
            y: token ids ``[B, U]`` (long) or ``None`` (zero pad-token embedding).
            state: optional ``(h, c)`` each ``[L, B, H]``; ``None`` -> zeros.
            add_sos: prepend a zero "start of sequence" step -> output ``[B, U+1, H]``.
            batch_size: batch size when ``y`` and ``state`` are both ``None``.

        Returns:
            ``(g, (h, c))`` with ``g`` shape ``[B, U(+1), H]`` and state ``[L, B, H]``.
        """
        embed = self.prediction.embed
        device = embed.weight.device
        dtype = embed.weight.dtype

        if y is not None:
            if y.device != device:
                y = y.to(device)
            g = embed(y)  # [B, U, H]
        else:
            if batch_size is None:
                b = 1 if state is None else state[0].size(1)
            else:
                b = batch_size
            g = torch.zeros((b, 1, self.config.pred_hidden), device=device, dtype=dtype)

        if add_sos:
            b, _, h = g.shape
            start = torch.zeros((b, 1, h), device=g.device, dtype=g.dtype)
            g = torch.cat([start, g], dim=1).contiguous()  # [B, U+1, H]

        g = g.transpose(0, 1)  # [U(+1), B, H] time-first
        g, hid = self._lstm()(g, state)
        g = g.transpose(0, 1)  # [B, U(+1), H]
        return g, hid

    forward = predict


_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class RnntJoint(nn.Module):
    def __init__(self, config: RnntConfig | None = None):
        super().__init__()
        self.config = config or RnntConfig()
        c = self.config
        self.enc = nn.Linear(c.encoder_hidden, c.joint_hidden)
        self.pred = nn.Linear(c.pred_hidden, c.joint_hidden)
        # joint_net: [activation, dropout, Linear] — only index 2 carries params.
        act = _ACT.get(c.activation, nn.ReLU)()
        self.joint_net = nn.Sequential(act, nn.Dropout(0.0), nn.Linear(c.joint_hidden, c.num_classes))

    @staticmethod
    def remap(name: str) -> str | None:
        prefix = "stt_model.rnnt_joint."
        return name[len(prefix):] if name.startswith(prefix) else None

    def load_weights(self, weights):
        from mstar.model.loader import load_hf_weights

        return load_hf_weights(self, weights, stacked_params=[], name_remapper=self.remap)

    def project_encoder(self, f: torch.Tensor) -> torch.Tensor:
        """Project encoder features ``[B, T, 1024]`` -> ``[B, T, 640]``."""
        return self.enc(f)

    def project_prednet(self, g: torch.Tensor) -> torch.Tensor:
        """Project prediction features ``[B, U, 640]`` -> ``[B, U, 640]``."""
        return self.pred(g)

    def joint_after_projection(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Combine projected enc ``f`` ``[B, T, H]`` and pred ``g`` ``[B, U, H]``.

        ``f[:, :, None, :] + g[:, None, :, :]`` -> ``joint_net`` -> ``[B, T, U, 1025]``.
        Returns raw logits (``log_softmax is None`` -> not applied on CUDA).
        """
        inp = f.unsqueeze(dim=2) + g.unsqueeze(dim=1)  # [B, T, U, H]
        return self.joint_net(inp)  # [B, T, U, V+1]

    def joint(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Full joint step: project encoder ``[B, T, 1024]`` + prednet ``[B, U, 640]``."""
        return self.joint_after_projection(self.project_encoder(f), self.project_prednet(g))

    def forward(self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor) -> torch.Tensor:
        """NeMo ``RNNTJoint.forward`` (non-fused logits path).

        ``encoder_outputs`` ``[B, D=1024, T]`` and ``decoder_outputs`` ``[B, D=640, U]``
        (time-last, as produced by the conformer / decoder); transposed to
        ``[B, T, D]`` / ``[B, U, D]`` then jointed -> logits ``[B, T, U, 1025]``.
        """
        f = encoder_outputs.transpose(1, 2)  # [B, T, 1024]
        g = decoder_outputs.transpose(1, 2)  # [B, U, 640]
        return self.joint(f, g)
