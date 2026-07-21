"""Configuration for the Zonos2 multi-codebook TTS transformer.

This is ported from the reference ``zonos2.models.config.ModelConfig`` (see
``../ZONOS2/python/zonos2/models/config.py``). It is flattened into a plain
dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Zonos2Config:
    """Zonos2 model and serving configuration.

    The fields split into two groups at the checkpoint boundary. The two groups
    treat their defaults very differently:

    * **Architecture** (backbone, token format, MoE, speaker) describes the
      trained transformer. It varies with each checkpoint. On the real serving
      path :func:`load_zonos2_config` reads each value from ``params.json`` and
      passes it explicitly. The defaults below are therefore *never used* when
      the code loads a checkpoint. They are only a small, representative
      placeholder network for direct construction (tests, no-checkpoint boot).
      They are NOT the released model's dimensions. (``moe_balancing_strategy``
      is the subtle exception. A hand-built config that skips the loader still
      applies its default against the checkpoint's balancing biases. See its
      note below.)

    * **Serving and vocoder** settings are absent from ``params.json``. The DAC
      codec is a separate pretrained model. The streaming knobs are deployment
      policy. So the loader leaves them alone, and the defaults below are
      load-bearing on every run.
    """

    # ---- Transformer backbone (the loader reads params.json; defaults are placeholder) ----
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
    # Text column vocabulary (UTF-8 byte tokens plus conditioning tokens).
    # ``None`` disables the text embedding column entirely.
    text_vocab: int | None = 512
    eoa_id: int = 1024  # end-of-audio token
    audio_pad_id: int = 1025  # audio padding token
    loss_softcap: float = 15.0  # tanh logit soft-capping (0 disables it)

    # ---- Mixture-of-Experts ----------------------------------------
    # MoE runs on layers ``[moe_start_from_layer, num_layers -
    # moe_end_from_layer)``. The rest use the dense SwiGLU feed-forward.
    moe_n_experts: int = 8
    num_experts_per_tok: int = 2  # top-k routing (the default; see special_topk_layers)
    # Per-layer top-k overrides. For example, ``{26: 2}`` routes layer 26 to
    # top-2. Every other MoE layer uses ``num_experts_per_tok``.
    special_topk_layers: dict[int, int] | None = None
    moe_router_dim: int = 256  # router bottleneck width
    moe_intermediate_size: int = 0  # 0 -> reuse ``intermediate_size``
    moe_start_from_layer: int = 2
    moe_end_from_layer: int = 2
    norm_topk_prob: bool = False  # Zonos2 does NOT renormalize top-k weights
    # "legacy" adds the balancing bias before top-k. "quantile" subtracts it.
    # The reference dataclass and the released checkpoint both resolve to
    # "legacy". The checkpoint omits the field in params.json. This default
    # matches that and keeps a checkpoint-less ``Zonos2Config()`` faithful. The
    # released ``balancing_biases`` are nonzero, so the wrong value would flip
    # expert routing.
    moe_balancing_strategy: str = "legacy"

    # ---- Optional speaker conditioning (voice cloning) -------------
    # The code projects raw speaker embeddings, optionally through an LDA
    # reduction. It then injects them at the speaker token position in the LM
    # (see ``Zonos2ForCausalLM``). This runs only when enabled.
    # TODO: Implement this :)
    speaker_enabled: bool = False
    speaker_embedding_dim: int = 128
    speaker_lda_dim: int | None = None

    # ---- Serving and vocoder settings (not in params.json; these defaults are live) ----
    sample_rate: int = 44100       # DAC output sample rate
    dac_model_type: str = "44khz"  # descript-audio-codec model tag
    dac_chunk_frames: int = 16     # streaming decode chunk (frames for each DAC call)
    dac_hop_length: int = 512      # DAC audio samples for each codebook frame (44khz)
    dac_overlap_frames: int = 4

    @property
    def audio_vocab(self) -> int:
        """Per-codebook output vocabulary (codes + eoa + pad)."""
        return self.codebook_size + 2

    @property
    def moe_inter(self) -> int:
        """Expert intermediate size, falling back to the dense value."""
        return self.moe_intermediate_size or self.intermediate_size

    def get_num_experts_per_tok(self, layer_id: int) -> int:
        """Top-k experts routed for ``layer_id``.

        This defaults to ``num_experts_per_tok`` (>=1). ``special_topk_layers``
        overrides it for each layer.
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
        """Whether layer ``layer_id`` uses the MoE feed-forward.

        MoE is active only for the middle band of layers.
        """
        if self.moe_n_experts <= 1:
            return False
        if layer_id < self.moe_start_from_layer:
            return False
        if (self.num_layers - layer_id) <= self.moe_end_from_layer:
            return False
        return True


# Aliases for the reference MoE balancing strategy (see
# ``zonos2.models.config.normalize_moe_balancing_strategy``). "quantile"
# subtracts the balancing bias. "legacy" adds it.
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

    This maps the reference training-format field names (``dim``, ``n_layers``,
    ``ffn_dim_multiplier`` plus ``multiple_of``, ``moe_router_topk`` ...) to the
    inference dims here.
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

    # Per-layer top-k overrides. JSON stores keys as strings. Normalize them to
    # int->int and validate them. This mirrors the reference
    # ``normalize_special_topk_layers``.
    raw_special = g("special_topk_layers")
    special_topk_layers: dict[int, int] | None = None
    if raw_special:
        special_topk_layers = {}
        for k, v in raw_special.items():
            k, v = int(k), int(v)
            if v < 1:
                raise ValueError(f"special_topk_layers[{k}] must be >= 1, got {v}")
            special_topk_layers[k] = v

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
    )

    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
