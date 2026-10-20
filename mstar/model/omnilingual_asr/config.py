"""Omnilingual ASR CTC: architecture per size, and where each checkpoint lives.

Every size shares one code path; only the encoder's width and depth and the
vocabulary differ (fairseq2 ``wav2vec2_asr`` archs ``300m``/``1b``/``3b``/``7b``
and their ``_v2`` twins, in ``omnilingual_asr/models/wav2vec2_asr/config.py``).

Weights come in two releases, both Apache-2.0:

* ``v2`` (default): Meta's December 2025 checkpoints, lower CER per Meta, with
  a "written" vocabulary that adds capitals and punctuation. Published only on Meta's
  CDN (``dl.fbaipublicfiles.com/mms/``), with no checksums. The sha256 values
  below were computed by us from the downloads (October 2026).
* ``v1``: the November 2025 checkpoints, on the Hugging Face Hub as
  ``facebook/omniASR-CTC-*``, pinned to a revision; the sha256 values are the
  Hub's LFS hashes.

Every file fetched by name is checked against its hash before it is loaded,
whether downloaded now or found in the cache, so a replaced file fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

SAMPLE_RATE = 16_000
# The reference pipeline rejects longer clips (``MAX_ALLOWED_AUDIO_SEC``); the
# models were trained on clips of at most 30 s.
MAX_AUDIO_SECONDS = 40.0
# One encoder frame needs the first conv's receptive field: 400 samples (25 ms).
MIN_AUDIO_SAMPLES = 400

ENCODER_NODE = "encoder"
TRANSCRIBE_WALK = "transcribe"
WAVEFORM = "waveform"
TOKEN_IDS = "token_ids"

SIZES = ("300m", "1b", "3b", "7b")
VERSIONS = ("v1", "v2")


@dataclass
class OmniASRCTCConfig:
    model_dim: int
    num_layers: int
    ffn_inner_dim: int
    vocab_size: int
    num_heads: int = 16
    feature_dim: int = 512
    feature_extractor_layer_descs: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
    )
    pos_conv_kernel_size: int = 128
    num_pos_conv_groups: int = 16
    layer_norm_eps: float = 1e-5

    @property
    def hop_samples(self) -> int:
        """Samples per encoder frame (320: 50 frames/s)."""
        hop = 1
        for _, _, stride in self.feature_extractor_layer_descs:
            hop *= stride
        return hop


# fairseq2 ``large_lv60k`` (300M) and Omnilingual's ``1b``/``3b``/``7b`` SSL encoders.
_ENCODERS = {
    "300m": dict(model_dim=1024, num_layers=24, ffn_inner_dim=4096),
    "1b": dict(model_dim=1280, num_layers=48, ffn_inner_dim=5120),
    "3b": dict(model_dim=2048, num_layers=60, ffn_inner_dim=8192),
    "7b": dict(model_dim=2048, num_layers=128, ffn_inner_dim=8192),
}
_VOCAB_SIZES = {"v1": 9812, "v2": 10288}


def ctc_config(size: str, version: str) -> OmniASRCTCConfig:
    if size not in _ENCODERS:
        raise ValueError(f"Unknown Omnilingual ASR size {size!r}; expected one of {list(SIZES)}")
    if version not in _VOCAB_SIZES:
        raise ValueError(f"Unknown Omnilingual ASR weights version {version!r}; expected one of {list(VERSIONS)}")
    return OmniASRCTCConfig(vocab_size=_VOCAB_SIZES[version], **_ENCODERS[size])


@dataclass(frozen=True)
class Artifact:
    url: str
    sha256: str
    filename: str


_CDN = "https://dl.fbaipublicfiles.com/mms"


@dataclass(frozen=True)
class HubCheckpoint:
    repo_id: str
    revision: str
    filename: str
    sha256: str


_V1_TOKENIZER_SHA256 = "b954cc166b0c9e0271b953fa226fa27ca706a25b7029e84579fe2c60a2b451fe"
# v1 on the Hub: one ``.pt`` and ``omniASR_tokenizer.model`` per repo.
HUB_CHECKPOINTS: dict[str, HubCheckpoint] = {
    "300m": HubCheckpoint(
        "facebook/omniASR-CTC-300M",
        "8e35f0cc28fa6099e0c14d56db85ce0423baa691",
        "omniASR-CTC-300M.pt",
        "4d31c6a7725a03e90cf26080c67361928755a9f43d322dd574145fbe91c76aec",
    ),
    "1b": HubCheckpoint(
        "facebook/omniASR-CTC-1B",
        "8c22e3ffdaa4aab6431b128b84b991a7d9c2515c",
        "omniASR-CTC-1B.pt",
        "e8564fa59dab7caedbcdb54ab7fb9bd6c96989f4d19add2ad81ddd969716952c",
    ),
    "3b": HubCheckpoint(
        "facebook/omniASR-CTC-3B",
        "3f12f4cbf4a22e96be2274e6af61e05ab3652f89",
        "omniASR-CTC-3B.pt",
        "3fbd640786af80e4b13c0f9c79040ef77120d7ccf4bf18d1031b42c2ccde6dcd",
    ),
    "7b": HubCheckpoint(
        "facebook/omniASR-CTC-7B",
        "b25c5a800c2fce25cf8e460fca2a28c41217b5ac",
        "omniASR-CTC-7B.pt",
        "6b9a70fae0c19d940689d53bbe45963a818703786b8ad78c43fb5e8ccc941c25",
    ),
}
HUB_TOKENIZER = ("omniASR_tokenizer.model", _V1_TOKENIZER_SHA256)

# v2 on the CDN: size -> checkpoint.
CDN_CHECKPOINTS: dict[str, Artifact] = {
    "300m": Artifact(
        f"{_CDN}/omniASR-CTC-300M-v2.pt",
        "8ce340ada22435d189908a8af67e3bb04899b6f2f5329d036248b9c3e38d2b50",
        "omniASR-CTC-300M-v2.pt",
    ),
    "1b": Artifact(
        f"{_CDN}/omniASR-CTC-1B-v2.pt",
        "354f981756aa8f41591ea363e45b9c4eba1ec5144c2273af82e747efbb08919c",
        "omniASR-CTC-1B-v2.pt",
    ),
    "3b": Artifact(
        f"{_CDN}/omniASR-CTC-3B-v2.pt",
        "fa7f662c326842bb80561db97631ae3c48d911aec579654a1e8414c26caf9089",
        "omniASR-CTC-3B-v2.pt",
    ),
    "7b": Artifact(
        f"{_CDN}/omniASR-CTC-7B-v2.pt",
        "8fcb2c2ec475ace73313fe69128b9c130db9c4ea57c62ec5fb87e3ae781d0357",
        "omniASR-CTC-7B-v2.pt",
    ),
}
CDN_TOKENIZER = Artifact(
    f"{_CDN}/omniASR_tokenizer_written_v2.model",
    "8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e",
    "omniASR_tokenizer_written_v2.model",
)


def parse_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
        }[dtype]
    except KeyError:
        raise ValueError(f"dtype must be bfloat16, float16 or float32; got {dtype!r}") from None
