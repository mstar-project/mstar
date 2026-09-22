"""Configuration for OmniVoice (``k2-fsa/OmniVoice``).

OmniVoice is a masked-diffusion TTS model, not an autoregressive one.  A
Qwen3-0.6B body runs **bidirectionally** over a fixed-length canvas laid out as

    [ style tokens | text tokens | (ref audio tokens) | target MASK tokens ]

where every position carries ``num_audio_codebook`` rows.  Text positions repeat
the same token id across all rows; audio positions hold one codebook entry per
row.  Generation unmasks the target region over ``num_step`` iterations and the
Higgs-Audio-v2 codec turns the finished token canvas into a waveform.

The architecture values are facts of the checkpoint's ``config.json``, hardcoded
so constructing the model never touches the network.
``OmniVoiceModel._refresh_checkpoint_defaults`` re-reads the ones that would
silently break parity if the checkpoint drifted.
"""

from dataclasses import dataclass, field

# The audio codec ships inside the OmniVoice repo under this subfolder; the
# upstream standalone mirror is the fallback the reference uses when the
# subfolder is absent.
CODEC_SUBFOLDER = "audio_tokenizer"
CODEC_FALLBACK_HF = "eustlb/higgs-audio-v2-tokenizer"


@dataclass
class OmniVoiceBackboneConfig:
    """The Qwen3 body, read bidirectionally.

    Identical in shape to ``Qwen3ForCausalLM`` at 0.6B, with two differences that
    matter at serving time: attention is **non-causal** (every position sees the
    whole document, prefix included), and there is no KV cache — the target
    region changes every diffusion step, and because the prefix attends *into*
    that region its hidden states change too, so nothing is reusable across
    steps.  ``use_cache`` in the checkpoint config is vestigial.
    """

    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    vocab_size: int = 151676
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = True


@dataclass
class OmniVoiceGenerationDefaults:
    """Per-request decoding knobs and their defaults.

    Every one of these is per-request: they ride ``step_metadata`` and are
    resolved once in ``OmniVoiceModel.get_initial_forward_pass_args``.  This is
    the reason to serve OmniVoice on a graph runtime at all — a fixed-pipeline
    port has to pick one value for the whole server.
    """

    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = True

    # Ceiling for the unmask Loop's ``max_iters``.  A request's own num_step
    # stops the loop early via check_stop; this only bounds the graph.
    max_num_step: int = 64


@dataclass
class OmniVoiceConfig:
    """OmniVoice model configuration."""

    backbone: OmniVoiceBackboneConfig = field(default_factory=OmniVoiceBackboneConfig)
    generation: OmniVoiceGenerationDefaults = field(
        default_factory=OmniVoiceGenerationDefaults
    )

    # Audio token space (config.json top level).
    num_audio_codebook: int = 8
    audio_vocab_size: int = 1025
    audio_mask_id: int = 1024

    # Codec (audio_tokenizer/config.json).  frame_rate is re-read from the
    # loaded codec rather than trusted from here: target lengths are computed
    # from it, and a drift would desynchronise duration from audio.
    sample_rate: int = 24000
    frame_rate: float = 25.0

    # v1 limits.  The reference splits text whose estimated audio exceeds
    # audio_chunk_threshold into audio_chunk_duration chunks and crossfades
    # them; that is a second graph walk and is deliberately not in v1, so a
    # request past the threshold is rejected at the request seam instead of
    # being silently truncated.
    max_target_seconds: float = 30.0

    # Requests per step. Packing is ragged so length skew costs nothing; this
    # bounds the Python-side per-item work and the number of reveal loops.
    max_batch_size: int = 16

    # The cap that actually matters: a step's cost is sum(doc_lens) tokens
    # through bidirectional attention, plus a head GEMM and a float32 logits
    # tensor over 2 * sum(target_len). Checked in the backbone's can_batch.
    max_packed_tokens: int = 24576

    # The reference's packed path plans flashinfer with float16 and its own CLI
    # loads the checkpoint that way, so fp16 is the tuned path -- the plan dtype
    # and the q/k dtype have to agree.
    load_dtype: str = "float16"

    # Set by _refresh_checkpoint_defaults; None until weights load.
    checkpoint_dtype: str | None = None

    @property
    def audio_embedding_rows(self) -> int:
        """Rows in ``audio_embeddings`` — one codebook's vocab per row block."""
        return self.num_audio_codebook * self.audio_vocab_size

    def target_tokens_for_seconds(self, seconds: float) -> int:
        return int(round(seconds * self.frame_rate))

    def seconds_for_target_tokens(self, tokens: int) -> float:
        return tokens / self.frame_rate
