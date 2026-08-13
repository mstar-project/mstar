"""Config dataclasses for NVIDIA NemotronLabs VoiceChat-11B (duplex).

The base checkpoint (``nvidia/NVIDIA-NemotronLabs-VoiceChat-11B``) is a single
``model.safetensors`` composite of three stages, whose per-stage architecture
dims come from the Spark repo's per-folder ``config.json`` files:

    * ``nano/``    — Nemotron-H hybrid Mamba-2 / attention / MLP LLM (9B)
    * ``eartts/``  — Gemma3-text "talker" transformer + RVQ codec (audio out)
    * (stt)        — Fast-Conformer streaming encoder + RNN-T head (audio in)

Only the fields M* actually needs are translated here. Numbers are taken
verbatim from the published ``nano/config.json`` / ``eartts/config.json``;
see the module docstrings for the source keys. Anything still unverified
against the NeMo reference (``nemotron-labs-voicechat`` branch of
NVIDIA-NeMo/Speech) is flagged with ``TODO(verify)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NanoConfig:
    """Nemotron-H backbone (``nano/config.json``, ``model_type='nemotron_h'``).

    ``hybrid_override_pattern`` is the per-layer type string: ``M`` = Mamba-2
    mixer, ``*`` = self-attention (NoPE), ``-`` = MLP (squared-ReLU). Its
    length equals ``num_hidden_layers``.
    """

    # --- backbone dims ---
    num_hidden_layers: int = 56
    hidden_size: int = 4480
    intermediate_size: int = 15680
    vocab_size: int = 131072
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    hybrid_override_pattern: str = (
        "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"
    )

    # --- attention (the ``*`` layers) — NoPE, no bias, no QK-norm ---
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    attention_bias: bool = False
    max_position_embeddings: int = 131072

    # --- Mamba-2 mixer (the ``M`` layers) ---
    mamba_num_heads: int = 128
    mamba_head_dim: int = 80
    ssm_state_size: int = 128          # d_state
    conv_kernel: int = 4              # d_conv
    n_groups: int = 8
    mamba_expand: int = 2             # d_inner = expand * hidden_size
    mamba_chunk_size: int = 1         # voicechat runs chunk_size=1 (streaming)
    time_step_rank: int = 256
    use_conv_bias: bool = True
    mamba_proj_bias: bool = False
    mamba_hidden_act: str = "silu"

    # --- MLP (the ``-`` layers) ---
    mlp_bias: bool = False
    mlp_hidden_act: str = "relu2"     # squared ReLU

    # --- VoiceChat fusion / heads ---
    # The voicechat path feeds a pre-fused audio+text embedding ("combined_embeds",
    # dim == hidden_size) instead of bare token ids; a function-calling head emits
    # (function_tokens, function_logits). See ``custom_input_specs`` /
    # ``custom_outputs`` in nano/config.json.
    combined_embed_dim: int = 4480
    has_function_head: bool = True

    # --- special tokens ---
    bos_token_id: int = 1
    eos_token_id: int = 12
    pad_token_id: int = 0
    # VoiceChat "pad-pair" turn-taking token (config: voicechat_pad_pair_token_id).
    pad_pair_token_id: int = 12

    @property
    def d_inner(self) -> int:
        # Mamba-2 inner width is heads*head_dim (10240 here). NOTE: this is
        # NOT expand*hidden (8960) — ``mamba_expand`` is nominal in Nemotron-H;
        # the SSM path is sized by mamba_num_heads*mamba_head_dim.
        # TODO(verify) against the HF NemotronH mixer construction.
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def conv_dim(self) -> int:
        # in_proj produces [z, x, B, C, dt]; the conv1d spans x + B + C.
        return self.d_inner + 2 * self.n_groups * self.ssm_state_size

    @property
    def layer_types(self) -> list[str]:
        """Per-layer kind: ``"mamba"`` | ``"attention"`` | ``"mlp"``."""
        assert len(self.hybrid_override_pattern) == self.num_hidden_layers, (
            f"hybrid_override_pattern len {len(self.hybrid_override_pattern)} "
            f"!= num_hidden_layers {self.num_hidden_layers}"
        )
        kind = {"M": "mamba", "*": "attention", "-": "mlp"}
        return [kind[c] for c in self.hybrid_override_pattern]

    @property
    def num_attention_layers(self) -> int:
        return self.hybrid_override_pattern.count("*")


@dataclass
class EarTTSConfig:
    """Audio-output stage (``eartts/config.json``).

    A Gemma3-text "talker" transformer autoregressively emits RVQ codec codes
    (``num_quantizers`` codebooks of ``codebook_size``), which the codec decoder
    turns into a 22.05 kHz waveform.
    """

    # talker transformer (backbone_type == "gemma3_text")
    hidden_size: int = 1152
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72

    # RVQ codec
    num_quantizers: int = 31          # codebooks emitted per audio frame
    codebook_size: int = 1024
    codec_hidden_size: int = 384
    codec_latent_size: int = 512

    sample_rate: int = 22050

    # TODO(verify): talker rope/eps, exact code-frame layout, and how the nano
    # LLM hidden state conditions the talker — confirm against the NeMo ref.


@dataclass
class ConformerSTTConfig:
    """Audio-input stage: Fast-Conformer streaming encoder + RNN-T head.

    From Nemotron-Speech-Streaming-En-0.6b. Encodes 16 kHz speech into
    LLM-space embeddings and emits a streaming user transcription (RNN-T,
    ``rnnt_vocab_size`` tokens; tokenizer in the base repo's ``rnnt_tokenizer/``).
    """

    d_model: int = 1024
    num_heads: int = 8
    num_layers: int = 24
    feat_in: int = 128               # log-mel bins
    input_sample_rate: int = 16000
    rnnt_vocab_size: int = 1024

    # 1500-position sliding window (model card). TODO(verify): subsampling
    # factor, conv kernel sizes, and projection into the nano hidden space.
    sliding_window: int = 1500


@dataclass
class NemotronDuplexConfig:
    """Top-level config bundling the three stages plus generation defaults."""

    nano: NanoConfig = field(default_factory=NanoConfig)
    eartts: EarTTSConfig = field(default_factory=EarTTSConfig)
    stt: ConformerSTTConfig = field(default_factory=ConformerSTTConfig)

    # generation defaults (text path). TODO(verify) against the model's
    # recommended sampling for the voicechat contract.
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    max_new_tokens: int = 2048

    # convenience passthroughs used by the Model glue
    @property
    def vocab_size(self) -> int:
        return self.nano.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self.nano.eos_token_id
