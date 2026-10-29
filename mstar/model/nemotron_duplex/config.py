"""Config dataclasses for NVIDIA NemotronLabs VoiceChat-11B (duplex).

The base checkpoint (``nvidia/NVIDIA-NemotronLabs-VoiceChat-11B``) is a single
``model.safetensors`` composite of three stages, whose per-stage architecture
dims come from the Spark repo's per-folder ``config.json`` files:

    * ``nano/``    — Nemotron-H hybrid Mamba-2 / attention / MLP LLM (9B)
    * ``eartts/``  — Gemma3-text "talker" transformer + RVQ codec (audio out)
    * (stt)        — Fast-Conformer streaming encoder (audio in)

Only the fields M* actually needs are translated here. Numbers are taken
verbatim from the published ``nano/config.json`` / ``eartts/config.json``;
see the module docstrings for the source keys. Anything still unverified
against the NeMo reference (``nemotron-labs-voicechat`` branch of
NVIDIA-NeMo/Speech) is flagged with ``TODO(verify)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mstar.model.components.audio_features import check_nemo_preprocessor
from mstar.model.components.fastconformer import FastConformerConfig

# Resource keys shared by the nano submodule, its layers and the model's
# ``get_node_resources`` (a deployment YAML tunes them under ``resources:``).
NANO_KV = "nano_kv"                 # paged KV cache of the 4 attention layers
NANO_ATTN = "nano_attn"             # attention planned over NANO_KV (NoPE: no position resource)
NANO_SAMPLER = "nano_sampler"       # agent-text channel sampler
MAMBA_STATE = "mamba_state"         # recurrent-state pool: conv + SSM state of the 27 Mamba-2 layers
MAMBA = "mamba"                     # the Mamba-2 (SSD) resource planned against MAMBA_STATE
TALKER_KV = "talker_kv"             # paged KV of the talker's 28 Gemma3 layers (labels: main, uncond)
TALKER_ATTN = "talker_attn"         # attention planned over TALKER_KV
TALKER_POS = "talker_pos"           # positions / RoPE for the talker (per-layer theta at call time)
TALKER_CFG_LABEL = "_cfg"           # the two talker streams combined into one plan (cond rows, then uncond)
AUDIO_IN = "audio_in"               # the client stream carrying the user's audio into the Encoder
ENCODER_STATE = "encoder_state"     # recurrent-state pool: each call's streaming-encoder conv tails
ENCODER_FRONT = "encoder_front"     # recurrent-state pool: each call's streaming-encoder audio and mel tails
CODEC_STATE = "codec_state"         # recurrent-state pool: each call's streaming codec decoder state
ENCODER_KV = "encoder_kv"           # bounded KV: each call's streaming-encoder attention window, per layer

# Every node's CUDA-graph batch sizes: a row per live session, up to the sessions
# a deployment serves by default (``NemotronDuplexModel.DEFAULT_SESSIONS``).
SESSION_BATCH_SIZES = [1, 2, 4, 8, 16, 32]


def session_too_long(max_session_s: float) -> ValueError:
    """The one error for a call whose audio outgrows ``max_session_s``, whichever
    stage sees it first."""
    return ValueError(f"the call's audio passed {max_session_s:g} s, the longest session this server takes")


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
    time_step_limit: tuple[float, float] = (0.0, float("inf"))  # HF ``time_step_limit`` (open for Nemotron-H)
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
    def num_mamba_layers(self) -> int:
        return sum(1 for k in self.layer_types if k == "mamba")

    @property
    def num_attention_layers(self) -> int:
        return self.hybrid_override_pattern.count("*")


@dataclass
class EarTTSConfig:
    """Audio-output stage (``config.json`` -> ``model.speech_generation.model.tts_config``).

    A Gemma3-text "talker" transformer autoregressively emits RVQ codec codes
    (``num_quantizers`` codebooks of ``codebook_size``), then a mixture-of-
    gaussians (MoG) head over the ``code_dim`` RVQ latent space.
    """

    # --- Gemma3-text backbone (``tts_config.backbone_config``) ---
    hidden_size: int = 1152
    intermediate_size: int = 4608
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    # head dim of the talker's paged KV in the engine: FlashInfer's paged
    # attention is exact for 64/128/256 and returns wrong values for 72 (probed
    # 2026-09-20 against a torch reference), so q/k/v are zero-padded to this
    kv_head_dim: int = 128
    sliding_window: int = 7500
    rms_norm_eps: float = 1e-6
    # Gemma3TextConfig defaults (not in the checkpoint's backbone_config subset).
    query_pre_attn_scalar: float = 256.0
    rope_theta_full: float = 1_000_000.0     # full-attention layers
    rope_theta_local: float = 10_000.0       # sliding-attention layers
    sliding_window_pattern: int = 6          # layer i is full-attn iff (i+1) % 6 == 0

    # --- RVQ code space (``tts_config``) ---
    num_quantizers: int = 31          # codebooks emitted per audio frame
    codebook_size: int = 1024
    code_dim: int = 512               # RVQ latent dim (``latent_size``)

    # --- MoG head (``tts_config.mog_head_config``) ---
    mog_num_predictions: int = 1024   # mixture components
    mog_low_rank: int = 64
    mog_min_log_std: float = -4.0
    mog_num_layers: int = 3
    mog_intermediate_size: int = 4608
    mog_eps: float = 1e-6

    # --- MaskGIT / MoG inference sampling (reference ``_get_generation_config``) ---
    # Real speech needs classifier-free guidance + injected noise + top-p mixture
    # sampling; without them the MoG head collapses to (near-)silent codes.
    mog_exponent: float = 3.0                 # masking-schedule exponent
    inference_num_iter: int = 8               # MaskGIT unmasking iterations
    inference_guidance_scale: float = 0.5     # classifier-free guidance weight
    inference_noise_scale: float = 0.8        # MoG latent noise (z = mu + exp(logs)*eps*ns)
    inference_top_p: float = 0.8              # nucleus filter on the mixture logits
    # The MoG/MaskGIT sampler is stochastic (noise + Gumbel). Draw it from a
    # dedicated seeded generator (the reference does ``torch.manual_seed(0)`` before
    # generation) so audio is reproducible AND does not land on rare collapse-to-
    # constant-code RNG states that an uncontrolled global RNG occasionally hits.
    inference_seed: int = 0

    sample_rate: int = 22050


def _voicechat_encoder() -> FastConformerConfig:
    # causal, each frame over the 70 before it ([70, 0]), no linear biases
    return FastConformerConfig(causal=True, left_context=70, attention_bias=False, convolution_bias=False)


@dataclass
class PerceptionConfig:
    """Audio-input stage (``config.json`` -> ``model.stt.model.perception``):
    the streaming FastConformer encoder, its featurizer's settings, and the
    projection of its frames into the nano's hidden size (``output_dim``).
    The defaults are the checkpoint's, so the resources can be declared
    before it is read."""

    encoder: FastConformerConfig = field(default_factory=_voicechat_encoder)
    output_dim: int = 4480
    # Frames ``DuplexStream`` holds back until more audio arrives: the newest
    # encoder frame reads mel frames that later audio still changes (the
    # subsampling's small future receptive field), so it is not final yet.
    streaming_lookahead_frames: int = 1

    # --- featurizer (``perception.preprocessor``) ---
    sample_rate: int = 16000
    n_fft: int = 512
    window_size: float = 0.025               # seconds -> win_length = round(sr*window_size)
    window_stride: float = 0.01              # seconds -> hop_length = round(sr*window_stride)
    preemph: float = 0.97
    # the reference (NVIDIA-NeMo/Speech ``nemotron-labs-voicechat``) predates
    # NeMo 2.5's featurizer: reflected edges, ``samples // hop + 1`` frames
    reflect_edges: bool = True

    @property
    def win_length(self) -> int:
        return round(self.sample_rate * self.window_size)

    @property
    def hop_length(self) -> int:
        return round(self.sample_rate * self.window_stride)

    @property
    def log_mel_settings(self) -> dict:
        """``NemoLogMel``'s settings for this featurizer; its bins are the encoder's
        (``load_nemo_perception`` passes them)."""
        return dict(
            sampling_rate=self.sample_rate, n_fft=self.n_fft, win_length=self.win_length,
            hop_length=self.hop_length, preemphasis=self.preemph, normalize=False,
            reflect_edges=self.reflect_edges,
        )


@dataclass
class CodecConfig:
    """RVQ audio codec (``config.json`` -> ``model.speech_generation.model.codec_config``).

    The encoder/decoder conv+ConvNeXt-block stacks are *derived* from these
    fields (channel schedule from ``base_hidden_size`` * ``channel_mult``,
    strides from ``rates``, ``num_blocks`` blocks per stage) rather than
    hardcoded — see ``encoder_layers`` / ``decoder_layers``.
    """

    base_hidden_size: int = 384
    channel_mult: tuple[int, ...] = (1, 2, 4)
    codebook_size: int = 1024
    groups: int = 1
    hop_length: int = 4
    kernel_size: int = 7
    latent_size: int = 512
    n_fft: int = 16
    num_blocks: int = 3
    num_quantizers: int = 31
    rates: tuple[int, ...] = (7, 7, 9)
    wav_to_token_ratio: int = 1764
    # Fixed NeMo codec constant (magnitude ceiling); not present in config.json.
    max_magnitude: float = 100.0

    @property
    def spec_channels(self) -> int:
        """Encoder input channels = (n_fft/2 + 1) real + imag = 2*(n_fft/2+1)."""
        return 2 * (self.n_fft // 2 + 1)

    @property
    def channels(self) -> list[int]:
        return [self.base_hidden_size * m for m in self.channel_mult]

    def encoder_layers(self) -> list[tuple]:
        """(kind, weight_shape, stride, transpose) per encoder layer."""
        ch = self.channels
        spec: list[tuple] = [("conv", (ch[0], self.spec_channels, 1), 1, False)]
        for i in range(len(ch)):
            spec += [("block", ch[i])] * self.num_blocks
            out = ch[i + 1] if i + 1 < len(ch) else self.latent_size
            spec.append(("conv", (out, ch[i], self.rates[i]), self.rates[i], False))
        return spec

    def decoder_layers(self) -> list[tuple]:
        ch = self.channels
        spec: list[tuple] = []
        prev = self.latent_size
        for i in reversed(range(len(ch))):
            spec.append(("conv", (prev, ch[i], self.rates[i]), self.rates[i], True))
            spec += [("block", ch[i])] * self.num_blocks
            prev = ch[i]
        spec.append(("conv", (self.spec_channels, ch[0], 1), 1, False))
        return spec


@dataclass
class NemotronDuplexConfig:
    """Top-level config bundling the stages plus generation defaults.

    Use :meth:`from_pretrained` to populate every sub-config from the
    checkpoint's ``config.json``; the dataclass defaults mirror that file so
    dummy-mode / weightless construction still works.
    """

    nano: NanoConfig = field(default_factory=NanoConfig)
    eartts: EarTTSConfig = field(default_factory=EarTTSConfig)
    stt: PerceptionConfig = field(default_factory=PerceptionConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)

    # generation defaults (text path); keep in sync with create_stream, offline_inference and realtime_api.
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    max_new_tokens: int = 2048

    # --- duplex fusion (AddFusion, parameter-free) + special tokens ---
    # Per-channel weights (config.json -> model.stt.model.duplex_*_channel_weight).
    # fuse_method is absent -> AddFusion: agent_text*w + user_audio*w + function*w.
    agent_text_weight: float = 1.0
    user_audio_weight: float = 1.0
    function_weight: float = 2.0
    use_function_head: bool = True
    predict_user_text: bool = False
    # Text-channel special tokens, per the checkpoint config.json
    # (model.speech_generation.model): bos_token=<s> (1), eos_token=</s> (2),
    # pad_token=<SPECIAL_12> (12). The pad token is the frequent between-word /
    # turn-taking "pad-pair" frame; ``inference_force_speech_silence_on_eos``
    # must silence ONLY the true EOS (2), NOT the pad frames (12) — else the
    # talker is forced silent between words and the spoken reply truncates after
    # a few words (verified via ASR).
    text_pad_id: int = 12
    text_bos_id: int = 1
    text_eos_id: int = 2

    # convenience passthroughs used by the Model glue
    @property
    def vocab_size(self) -> int:
        return self.nano.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self.nano.eos_token_id

    @classmethod
    def from_pretrained(cls, ckpt_dir: str) -> NemotronDuplexConfig:
        """Build from the checkpoint's ``config.json`` (NeMo VoiceChat layout).

        Only the stages whose components read from config are populated here;
        the rest fall back to dataclass defaults. Extend as modules migrate to
        config-driven construction.
        """
        import json
        import os

        with open(os.path.join(ckpt_dir, "config.json")) as f:
            raw = json.load(f)
        cfg = cls()

        perception = raw["model"]["stt"]["model"]["perception"]
        pre = perception["preprocessor"]
        encoder = FastConformerConfig.from_nemo(perception["encoder"])
        # no per-clip normalization: a stream can't
        check_nemo_preprocessor(pre, features=encoder.num_mel_bins, normalize="NA", pad_to=0)
        cfg.stt = PerceptionConfig(
            encoder=encoder,
            output_dim=perception["output_dim"],
            sample_rate=pre["sample_rate"],
            n_fft=pre["n_fft"],
            window_size=pre["window_size"],
            window_stride=pre["window_stride"],
            preemph=pre.get("preemph", 0.97),
        )

        cc = raw["model"]["speech_generation"]["model"]["codec_config"]
        cfg.codec = CodecConfig(
            base_hidden_size=cc["base_hidden_size"],
            channel_mult=tuple(cc["channel_mult"]),
            codebook_size=cc["codebook_size"],
            groups=cc["groups"],
            hop_length=cc["hop_length"],
            kernel_size=cc["kernel_size"],
            latent_size=cc["latent_size"],
            n_fft=cc["n_fft"],
            num_blocks=cc["num_blocks"],
            num_quantizers=cc["num_quantizers"],
            rates=tuple(cc["rates"]),
            wav_to_token_ratio=cc["wav_to_token_ratio"],
        )

        sg = raw["model"]["speech_generation"]["model"]
        tts = sg["tts_config"]
        bb, mog = tts["backbone_config"], tts["mog_head_config"]
        cfg.eartts = EarTTSConfig(
            hidden_size=bb["hidden_size"],
            intermediate_size=bb["intermediate_size"],
            num_hidden_layers=bb["num_hidden_layers"],
            num_attention_heads=bb["num_attention_heads"],
            num_key_value_heads=bb["num_key_value_heads"],
            head_dim=bb["head_dim"],
            sliding_window=bb["sliding_window"],
            num_quantizers=tts["num_quantizers"],
            codebook_size=tts["codebook_size"],
            code_dim=tts["latent_size"],
            mog_num_predictions=mog["num_predictions"],
            mog_low_rank=mog["low_rank"],
            mog_min_log_std=mog["min_log_std"],
            mog_num_layers=mog["num_layers"],
            mog_intermediate_size=mog["intermediate_size"],
            mog_eps=mog["eps"],
            mog_exponent=mog.get("exponent", 3.0),
            inference_num_iter=tts.get("inference_num_iter", 8),
            inference_guidance_scale=sg.get("inference_guidance_scale", 0.5),
            inference_noise_scale=sg.get("inference_noise_scale", 0.8),
            inference_top_p=sg.get("inference_top_p_or_k", 0.8),
        )

        sm = raw["model"]["stt"]["model"]
        cfg.agent_text_weight = sm.get("duplex_text_channel_weight", cfg.agent_text_weight)
        cfg.user_audio_weight = sm.get("duplex_user_channel_weight", cfg.user_audio_weight)
        cfg.function_weight = sm.get("duplex_function_channel_weight", cfg.function_weight)
        cfg.use_function_head = sm.get("use_function_head", cfg.use_function_head)
        cfg.predict_user_text = sm.get("predict_user_text", cfg.predict_user_text)
        return cfg
