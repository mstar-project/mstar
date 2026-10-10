"""Token2wav's voice prompt: a reference clip -> speech tokens, speaker x-vector and 24 kHz mel.

Ported from Step-Audio2's ``Token2wav._prepare_prompt`` (``minicpmo-utils``, Apache-2.0). The
three encoders are Chatterbox's native ports of the same CosyVoice-family networks, loaded
from the ONNX files MiniCPM-o ships (``speech_tokenizer_v2_25hz.onnx``, ``campplus.onnx``):

- tokens: ``S3Tokenizer`` with the ``s3tokenizer`` package's current quantizer, which clamps
  the tanh output before rounding (Chatterbox's ``FSQuantizer`` scales it by 0.999 instead),
  and its attention LayerNorm eps of 1e-6 (the ONNX graph and Chatterbox use 1e-5);
- x-vector: ``CAMPPlus`` on the mean-normalised Kaldi fbank. The ONNX export folded some
  batch norms into the preceding conv; those load as the conv weight plus a batch norm that
  only adds the folded bias;
- mel: ``MelSpectrogram24k`` of the clip resampled 16 -> 24 kHz, replicate-padded (or cut) to
  two frames per prompt token, which is what the flow's prompt condition must be.

Reading the ONNX files needs the ``onnx`` package (an optional dependency).
"""
from __future__ import annotations

import re
from typing import NamedTuple

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from mstar.model.chatterbox.components.audio import MelSpectrogram24k, kaldi_fbank_80
from mstar.model.chatterbox.components.s3_tokenizer import S3Tokenizer
from mstar.model.chatterbox.components.s3gen_xvector import CAMPPlus
from mstar.model.chatterbox.config import S3GenMelConfig, S3GenXVectorConfig, S3TokenizerConfig

PROMPT_SR = 16000
MEL_SR = 24000
FRAMES_PER_TOKEN = 2


class VoicePrompt(NamedTuple):
    """One voice's conditioning, computed once per reference clip."""

    tokens: torch.Tensor  # [1, P] int32 s3 tokens
    spk_emb: torch.Tensor  # [1, 192] raw CAMPPlus x-vector
    mel: torch.Tensor  # [1, 2P, 80] 24 kHz log-mel


class ClampedFSQuantizer(nn.Module):
    """``s3tokenizer``'s FSQ: tanh, clamp to ``(-1, 1)``, round to ``{0, 1, 2}``, base-3 code."""

    def __init__(self, dim: int, fsq_dim: int = 8, levels: int = 3):
        super().__init__()
        self.project_down = nn.Linear(dim, fsq_dim)
        self.levels = levels
        self.register_buffer("powers", levels ** torch.arange(fsq_dim, dtype=torch.int64), persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        h = torch.tanh(self.project_down(hidden).float())
        h = torch.clamp(h, -1 + eps, 1 - eps)
        h = ((h + 1.0) * (self.levels - 1) / 2.0).round().to(torch.int64)
        return (h * self.powers).sum(dim=-1)


class PromptTokenizer(S3Tokenizer):
    """Chatterbox's S3TokenizerV2 with the quantizer and LayerNorm eps the reference runs."""

    def __init__(self, config: S3TokenizerConfig | None = None):
        super().__init__(config)
        self.quantizer = ClampedFSQuantizer(self.config.n_state, self.config.fsq_dim, self.config.fsq_levels)
        for block in self.encoder.blocks:
            block.attn_ln.eps = 1e-6


class VoicePromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = PromptTokenizer()
        self.speaker = CAMPPlus(S3GenXVectorConfig())
        self.mel = MelSpectrogram24k(S3GenMelConfig())
        self.resample = torchaudio.transforms.Resample(PROMPT_SR, MEL_SR)

    @torch.inference_mode()
    def forward(self, wav16: torch.Tensor) -> VoicePrompt:
        """``wav16 [T]`` float32 at 16 kHz. Each part runs on its module's device."""
        tok_dev = self.tokenizer.window.device
        log_mel = self.tokenizer.log_mel(wav16.to(tok_dev))
        tokens, _ = self.tokenizer.encode_mel(log_mel, torch.tensor([log_mel.shape[-1]], device=tok_dev))
        tokens = tokens.int()

        spk_dev = next(self.speaker.parameters()).device
        spk_emb = self.speaker(kaldi_fbank_80(wav16.to(spk_dev))[None])

        mel_dev = self.mel.window.device
        mel = self.mel(self.resample(wav16.to(mel_dev)[None])).transpose(1, 2)
        mel = F.pad(mel, (0, 0, 0, tokens.shape[1] * FRAMES_PER_TOKEN - mel.shape[1]), mode="replicate")
        return VoicePrompt(tokens=tokens, spk_emb=spk_emb, mel=mel)


# ---------------------------------------------------------------------------
# ONNX initializers -> torch parameters
# ---------------------------------------------------------------------------


def _module_path(node_name: str) -> str:
    """``/head/layer1/layer1.0/conv1/Conv`` -> ``head.layer1.0.conv1``: the exporter's scope
    names repeat a container's name in its children's (``layer1/layer1.0``)."""
    parts = node_name.strip("/").split("/")[:-1]
    out: list[str] = []
    for i, part in enumerate(parts):
        if i and part.startswith(parts[i - 1] + "."):
            part = part[len(parts[i - 1]) + 1:]
        out.append(part)
    return ".".join(out)


_PARAM_SLOTS = {
    "Conv": {1: "weight", 2: "bias"},
    "MatMul": {1: "weight"},
    "Add": {0: "bias", 1: "bias"},
    "LayerNormalization": {1: "weight", 2: "bias"},
    "BatchNormalization": {1: "weight", 2: "bias", 3: "running_mean", 4: "running_var"},
}


def read_onnx_parameters(path: str) -> dict[str, torch.Tensor]:
    """Every initializer an ONNX graph feeds to a parameterised op, named by the module scope
    of the node that reads it; MatMul weights are transposed to ``nn.Linear``'s layout."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("reading token2wav's voice-prompt encoders needs `pip install onnx`") from e
    model = onnx.load(path)
    inits = {i.name: i for i in model.graph.initializer}
    out: dict[str, torch.Tensor] = {}
    for node in model.graph.node:
        slots = _PARAM_SLOTS.get(node.op_type)
        if slots is None:
            continue
        for idx, name in enumerate(node.input):
            if name not in inits or idx not in slots:
                continue
            tensor = torch.from_numpy(numpy_helper.to_array(inits[name]).copy())
            if node.op_type == "MatMul":
                tensor = tensor.t()
            key = f"{_module_path(node.name)}.{slots[idx]}"
            if key in out:
                raise ValueError(f"{path}: two initializers map to {key}")
            out[key] = tensor
    return out


def tokenizer_state(onnx_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in onnx_params.items():
        out[k.replace("quantizer.project_in.", "quantizer.project_down.") if k.startswith("quantizer.")
            else f"encoder.{k}"] = v
    return out


# Convolutions whose following batch norm the ONNX export folded in.
_FOLDED_BN = [
    (re.compile(r"^(head(?:\.layer\d\.\d)?)\.conv(\d)$"), r"\1.bn\2"),
    (re.compile(r"^(head\.layer\d\.\d\.shortcut)\.0$"), r"\1.1"),
    (re.compile(r"^xvector\.tdnn\.linear$"), "xvector.tdnn.nonlinear.batchnorm"),
    (re.compile(r"^(xvector\.block\d\.tdnnd\d+)\.linear1$"), r"\1.nonlinear2.batchnorm"),
    (re.compile(r"^xvector\.transit3\.linear$"), "xvector.out_nonlinear.batchnorm"),
]


def campplus_state(onnx_params: dict[str, torch.Tensor], model: CAMPPlus) -> dict[str, torch.Tensor]:
    """Map the ONNX initializers onto ``CAMPPlus``. A folded conv's bias goes into its batch
    norm, set to ``y = x + bias``: unit scale, zero mean, and a variance that makes
    ``var + eps`` exactly 1 in float32."""
    out = dict(onnx_params)
    for key in [k for k in onnx_params if k.endswith(".bias")]:
        conv = key[: -len(".bias")]
        if model.get_submodule(conv).bias is not None:
            continue
        bn = next((pat.sub(rep, conv) for pat, rep in _FOLDED_BN if pat.match(conv)), None)
        if bn is None:
            raise ValueError(f"campplus.onnx: {conv} has a folded bias but no known batch norm after it")
        bias = out.pop(key)
        norm = model.get_submodule(bn)
        one = torch.ones_like(bias)
        out.update({
            f"{bn}.weight": one,
            f"{bn}.bias": bias,
            f"{bn}.running_mean": torch.zeros_like(bias),
            f"{bn}.running_var": one - torch.tensor(norm.eps, dtype=bias.dtype),
        })
    for name, buf in model.named_buffers():
        if name.endswith("num_batches_tracked"):
            out[name] = torch.zeros_like(buf)
    return out
