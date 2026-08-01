"""Configuration for the Zonos2 multi-codebook TTS transformer.

This is a flattened port of the reference ``zonos2.models.config.ModelConfig``.
See ``../ZONOS2/python/zonos2/models/config.py``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Zonos2Config:
    """Zonos2 model and serving configuration.

    The fields fall into two groups, and their defaults differ in importance:

    * **Architecture** (backbone, token format, MoE, speaker). These fields
      describe the trained transformer, and they change with each checkpoint.
      :func:`load_zonos2_config` reads every value from ``params.json`` and
      passes it explicitly, so a checkpoint load never uses the defaults below.
      The defaults are a small placeholder network for direct construction in
      tests. They are not the dimensions of the released model.
      ``moe_balancing_strategy`` is the exception. See its note below.

    * **Serving and vocoder**. ``params.json`` does not contain these fields.
      The DAC codec is a separate pretrained model, and the streaming values are
      deployment policy. The loader does not touch them, so the defaults below
      apply on every run.
    """

    # ---- Transformer backbone (the loader reads params.json; these are placeholders) ----
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
    # The text column vocabulary holds the UTF-8 byte tokens and the
    # conditioning tokens. ``None`` disables the text embedding column.
    text_vocab: int | None = 512
    eoa_id: int = 1024  # end-of-audio token
    audio_pad_id: int = 1025  # audio padding token
    loss_softcap: float = 15.0  # tanh logit soft-cap (0 disables it)

    # ---- Mixture-of-Experts ----------------------------------------
    # MoE runs on layers ``[moe_start_from_layer, num_layers -
    # moe_end_from_layer)``. The other layers use the dense SwiGLU feed-forward.
    moe_n_experts: int = 8
    num_experts_per_tok: int = 2  # default top-k routing; see special_topk_layers
    # Per-layer top-k overrides. For example, ``{26: 2}`` routes layer 26 to
    # top-2. Every other MoE layer uses ``num_experts_per_tok``.
    special_topk_layers: dict[int, int] | None = None
    moe_router_dim: int = 256  # router bottleneck width
    moe_intermediate_size: int = 0  # 0 -> reuse ``intermediate_size``
    moe_start_from_layer: int = 2
    moe_end_from_layer: int = 2
    norm_topk_prob: bool = False  # Zonos2 does NOT renormalize the top-k weights
    # "legacy" adds the balancing bias before the top-k. "quantile" subtracts
    # it. The reference dataclass and the released checkpoint both use "legacy",
    # and ``params.json`` omits the field. Keep this default, so that a
    # ``Zonos2Config()`` built without a checkpoint stays correct. The released
    # ``balancing_biases`` are not zero, so the other value changes the routing.
    moe_balancing_strategy: str = "legacy"

    # ---- Optional speaker conditioning (voice cloning) -------------
    # The model projects the raw speaker embedding, optionally through an LDA
    # reduction. It then injects the result at the speaker token position. See
    # ``Zonos2ForCausalLM``. This applies only when ``speaker_enabled`` is set.
    speaker_enabled: bool = False
    speaker_embedding_dim: int = 128
    speaker_lda_dim: int | None = None
    # The Zonos2 checkpoint does not contain the speaker encoder. ``model.pth``
    # holds only the LDA and projection layers downstream of it. The reference
    # loads an external HF model whose architecture lives in remote code. See
    # ``../ZONOS2/python/zonos2/models/speaker_cloning.py``. The repo id says
    # "1.7B", but that remote class is an ECAPA-TDNN of about 12M parameters.
    # These two fields are deployment settings. ``params.json`` does not contain
    # them, so the defaults apply on every run.
    speaker_encoder_model_id: str = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
    speaker_encoder_sample_rate: int = 24_000

    # ---- Text-column conditioning tokens ---------------------------
    # The conditioning tokens occupy the tail of the text vocabulary in this
    # order: the speaking-rate buckets, the quality buckets (one block for each
    # feature), the clean/noisy background pair, then the accurate-mode marker.
    # ``text_vocab`` is the text padding id. Each count shifts the ids after it,
    # so all four counts are necessary to emit even one marker. See
    # ``prompt._conditioning_base_text_vocab``.
    speaker_background_token_enabled: bool = False
    accurate_mode_token_enabled: bool = False
    speaking_rate_num_buckets: int = 0
    speaking_rate_buckets: tuple[str, ...] = ()
    quality_num_buckets: int = 0
    quality_features: tuple[str, ...] = ()
    quality_buckets: dict[str, tuple[str, ...]] | None = None

    # ---- Serving and vocoder (absent from params.json; these defaults apply) ----
    sample_rate: int = 44100       # DAC output sample rate
    dac_model_type: str = "44khz"  # descript-audio-codec model tag
    dac_chunk_frames: int = 16     # streaming decode chunk: frames for each DAC call
    dac_hop_length: int = 512      # DAC audio samples for each codebook frame (44khz)
    dac_overlap_frames: int = 4

    @property
    def audio_vocab(self) -> int:
        """The output vocabulary of one codebook: the codes, eoa, and pad."""
        return self.codebook_size + 2

    @property
    def moe_inter(self) -> int:
        """The expert intermediate size. It falls back to the dense value."""
        return self.moe_intermediate_size or self.intermediate_size

    @property
    def speaker_background_num_buckets(self) -> int:
        """The text-vocab slots reserved for the clean/noisy background pair."""
        return 2 if self.speaker_background_token_enabled else 0

    @property
    def accurate_mode_num_buckets(self) -> int:
        """The text-vocab slots reserved for the accurate-mode marker."""
        return 1 if self.accurate_mode_token_enabled else 0

    @property
    def quality_bucket_counts(self) -> tuple[int, ...]:
        """The quality bucket count of each feature, in ``quality_features`` order.

        The order is important. The sum of the counts of all preceding features
        offsets the token id of a feature.
        """
        buckets = self.quality_buckets or {}
        return tuple(len(buckets.get(feature, ())) for feature in self.quality_features)

    def get_num_experts_per_tok(self, layer_id: int) -> int:
        """Return the number of experts that ``layer_id`` routes to.

        The value defaults to ``num_experts_per_tok``, with a minimum of 1.
        ``special_topk_layers`` overrides it for a given layer.
        """
        default_topk = self.num_experts_per_tok if self.num_experts_per_tok > 0 else 1
        special = self.special_topk_layers
        if special:
            topk = special.get(layer_id, special.get(str(layer_id), default_topk))
        else:
            topk = default_topk
        topk = int(topk)
        if topk < 1:
            raise ValueError(f"top-k for layer {layer_id} must be >= 1, got {topk}")
        return topk

    def is_moe_layer(self, layer_id: int) -> bool:
        """Return True if layer ``layer_id`` uses the MoE feed-forward.

        Only the middle band of layers uses MoE.
        """
        if self.moe_n_experts <= 1:
            return False
        if layer_id < self.moe_start_from_layer:
            return False
        if (self.num_layers - layer_id) <= self.moe_end_from_layer:
            return False
        return True


# Aliases for the MoE balancing strategy of the reference. See
# ``zonos2.models.config.normalize_moe_balancing_strategy``. "quantile"
# subtracts the balancing bias, and "legacy" adds it.
_MOE_BALANCING_ALIASES = {
    "current": "quantile", "quantile": "quantile", "qbalancing": "quantile",
    "old": "legacy", "legacy": "legacy", "aux": "legacy", "aux_loss": "legacy",
}


def _normalize_moe_balancing_strategy(strategy: str) -> str:
    normalized = str(strategy).strip().lower().replace("-", "_")
    try:
        return _MOE_BALANCING_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported moe_balancing_strategy={strategy!r}; expected one of "
            f"{sorted(set(_MOE_BALANCING_ALIASES))}."
        ) from exc


def load_zonos2_config(params: dict, **overrides) -> Zonos2Config:
    """Build a :class:`Zonos2Config` from a reference ``params.json`` dict.

    The function maps the training-format field names of the reference
    (``dim``, ``n_layers``, ``ffn_dim_multiplier`` with ``multiple_of``,
    ``moe_router_topk``, and others) to the inference dimensions here.
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

    # Per-layer top-k overrides. JSON stores the keys as strings, so normalize
    # them to int->int and validate them. This agrees with the reference
    # function ``normalize_special_topk_layers``.
    raw_special = g("special_topk_layers")
    special_topk_layers: dict[int, int] | None = None
    if raw_special:
        special_topk_layers = {}
        for k, v in raw_special.items():
            k, v = int(k), int(v)
            if v < 1:
                raise ValueError(f"special_topk_layers[{k}] must be >= 1, got {v}")
            special_topk_layers[k] = v

    # Quality conditioning. ``quality_features`` sets the feature order that the
    # token-id math uses. If the explicit list is absent, the reference uses the
    # key order of ``quality_buckets``. See its ``_model_quality_features``.
    raw_quality_buckets = g("quality_buckets") or {}
    quality_features = tuple(str(f) for f in (g("quality_features") or raw_quality_buckets))
    quality_buckets = {
        str(feature): tuple(str(item) for item in (buckets or ()))
        for feature, buckets in raw_quality_buckets.items()
    } or None

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
        special_topk_layers=special_topk_layers,
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
        speaker_background_token_enabled=bool(g("speaker_background_token_enabled", False)),
        accurate_mode_token_enabled=bool(g("accurate_mode_token_enabled", False)),
        speaking_rate_num_buckets=int(g("speaking_rate_num_buckets", 0) or 0),
        speaking_rate_buckets=tuple(str(b) for b in (g("speaking_rate_buckets") or ())),
        quality_num_buckets=int(g("quality_num_buckets", 0) or 0),
        quality_features=quality_features,
        quality_buckets=quality_buckets,
    )

    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
