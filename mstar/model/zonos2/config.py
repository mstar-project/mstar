"""Configuration for the Zonos2 multi-codebook TTS transformer.

Ported from the reference ``zonos2.models.config.ModelConfig`` (see
``../ZONOS2/python/zonos2/models/config.py``) but flattened into a plain
dataclass with sensible defaults, mirroring the style of the other
``mstar`` model configs (e.g. :class:`OrpheusModelConfig`).

The defaults describe a small, representative Zonos2 network; swap in the
real checkpoint's values (``dim`` -> ``hidden_size``, ``n_layers`` ->
``num_layers``, etc.) when loading actual weights.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Zonos2Config:
    # ---- Transformer backbone --------------------------------------
    num_layers: int = 16
    hidden_size: int = 1024
    num_qo_heads: int = 16
    num_kv_heads: int = 4  # GQA (== num_qo_heads for full MHA)
    head_dim: int = 64
    intermediate_size: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    max_position_embeddings: int = 8192

    # ---- Multi-codebook audio / text token format ------------------
    n_codebooks: int = 9
    codebook_size: int = 1024
    # Text column vocabulary (UTF-8 byte tokens + conditioning tokens).
    # ``None`` disables the text embedding column entirely.
    text_vocab: int | None = 512
    eoa_id: int = 1024  # end-of-audio token
    audio_pad_id: int = 1025  # audio padding token
    loss_softcap: float = 15.0  # tanh logit soft-capping (0 disables)
    # End generation only when the *leading* (undelayed) codebook 0 emits eoa.
    # Under the inter-codebook delay, cb0 leads the end-of-audio signal, so a
    # delayed codebook (1..C-1) emitting eoa on its own is spurious and would
    # truncate the utterance early. True = robust to that (recommended);
    # False = legacy "any codebook" behaviour (matches the reference).
    eos_require_leading_codebook: bool = True

    # ---- Mixture-of-Experts ----------------------------------------
    # MoE is enabled on layers ``[moe_start_from_layer, num_layers -
    # moe_end_from_layer)``; the rest use the dense SwiGLU feed-forward.
    moe_n_experts: int = 8
    num_experts_per_tok: int = 2  # top-k routing
    moe_router_dim: int = 256  # router bottleneck width
    moe_intermediate_size: int = 0  # 0 -> reuse ``intermediate_size``
    moe_start_from_layer: int = 2
    moe_end_from_layer: int = 2
    norm_topk_prob: bool = False  # Zonos2 does NOT renormalize top-k weights
    # "quantile" (subtract balancing bias) or "legacy" (add it).
    moe_balancing_strategy: str = "quantile"

    # ---- Optional speaker conditioning (voice cloning) -------------
    # Not modeled here; kept for config parity / future use.
    speaker_enabled: bool = False
    speaker_embedding_dim: int = 128
    speaker_lda_dim: int | None = None

    # ---- Serving / vocoder settings --------------------------------
    sample_rate: int = 44100       # DAC output sample rate
    dac_model_type: str = "44khz"  # descript-audio-codec model tag
    dac_chunk_frames: int = 16     # streaming decode chunk (frames per DAC call)
    dac_hop_length: int = 512      # DAC audio samples per codebook frame (44khz)
    # Frames of already-decoded left context re-decoded and crossfaded at each
    # streaming chunk boundary so the convolutional decoder warms up on real
    # signal (0 disables the crossfade — restores the click at chunk edges).
    dac_overlap_frames: int = 4

    @property
    def audio_vocab(self) -> int:
        """Per-codebook output vocabulary (codes + eoa + pad)."""
        return self.codebook_size + 2

    @property
    def moe_inter(self) -> int:
        """Expert intermediate size, falling back to the dense value."""
        return self.moe_intermediate_size or self.intermediate_size

    def is_moe_layer(self, layer_id: int) -> bool:
        """Whether layer ``layer_id`` uses the MoE feed-forward.

        Matches ``TransformerBlock._is_moe_layer`` in the reference: MoE is
        active only for the middle band of layers.
        """
        if self.moe_n_experts <= 1:
            return False
        if layer_id < self.moe_start_from_layer:
            return False
        if (self.num_layers - layer_id) <= self.moe_end_from_layer:
            return False
        return True


# Aliases for the reference MoE balancing strategy (see
# ``zonos2.models.config.normalize_moe_balancing_strategy``): "quantile"
# subtracts the balancing bias, "legacy" adds it.
_MOE_BALANCING_ALIASES = {
    "current": "quantile", "quantile": "quantile", "qbalancing": "quantile",
    "old": "legacy", "legacy": "legacy", "aux": "legacy", "aux_loss": "legacy",
}


def _normalize_moe_balancing_strategy(strategy: str) -> str:
    return _MOE_BALANCING_ALIASES.get(str(strategy).strip().lower().replace("-", "_"), "quantile")


def load_zonos2_config(params: dict, **overrides) -> Zonos2Config:
    """Build a :class:`Zonos2Config` from a reference ``params.json`` dict.

    Maps the reference training-format field names (``dim``, ``n_layers``,
    ``ffn_dim_multiplier`` + ``multiple_of``, ``moe_router_topk`` ...) to the
    inference dims here, mirroring ``ModelConfig.from_zonos2_config`` in
    ``../ZONOS2/python/zonos2/models/config.py``. ``params`` may be flat or
    nested under a ``"model"`` key. ``overrides`` (e.g. from a serving YAML's
    ``model_kwargs``) win over the checkpoint values.
    """
    p = params.get("model", params) if isinstance(params, dict) else params

    def g(key, default=None):
        return p.get(key, default)

    dim = int(g("dim", 512))
    head_dim = int(g("head_dim", 128))
    n_heads = int(g("n_heads") or (dim // head_dim))
    n_kv_heads = int(g("n_kv_heads") or n_heads)

    # intermediate_size = round_up(ffn_dim_multiplier * dim, multiple_of)
    multiple_of = int(g("multiple_of", 256))
    ffn_dim = int(float(g("ffn_dim_multiplier", 4.0)) * dim)
    intermediate_size = multiple_of * ((ffn_dim + multiple_of - 1) // multiple_of)

    moe_n_experts = int(g("moe_n_experts", 1))

    cfg = Zonos2Config(
        num_layers=int(g("n_layers", 8)),
        hidden_size=dim,
        num_qo_heads=n_heads,
        num_kv_heads=n_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        rms_norm_eps=float(g("norm_eps", 1e-5)),
        rope_theta=float(g("rope_theta", 10000.0)),
        max_position_embeddings=int(g("max_seqlen", 2048)),
        n_codebooks=int(g("n_codebooks", 9)),
        codebook_size=int(g("codebook_size", 1024)),
        text_vocab=g("text_vocab"),
        eoa_id=int(g("eoa_id", 1024)),
        audio_pad_id=int(g("audio_pad_id", 1025)),
        loss_softcap=float(g("loss_softcap", 15.0)),
        moe_n_experts=moe_n_experts,
        num_experts_per_tok=int(g("moe_router_topk", 1)),
        moe_router_dim=int(g("moe_router_dim", 128)),
        moe_intermediate_size=0,  # reuse intermediate_size (matches reference)
        moe_start_from_layer=int(g("moe_start_from_layer", 0)),
        moe_end_from_layer=int(g("moe_end_from_layer", 0)),
        norm_topk_prob=False,
        moe_balancing_strategy=_normalize_moe_balancing_strategy(
            g("moe_balancing_strategy", "legacy")
        ),
        speaker_enabled=bool(g("speaker_enabled", False)),
        speaker_embedding_dim=int(g("speaker_embedding_dim", 128)),
        speaker_lda_dim=g("speaker_lda_dim"),
    )

    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
