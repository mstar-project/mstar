# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) Kyutai, all rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Adapted for M* on 2026-09-28 from NVIDIA/personaplex at the pin below:
# source-cited constants and the context-budget check.
"""Constants verified against NVIDIA/personaplex at the pin below.

[PP] paths are relative to ``moshi/moshi/`` at that commit. Derived values
state their formula; the checkpoint's ``config.json`` is a 56-byte stub, so
nothing is read from it.
"""

from dataclasses import dataclass

REFERENCE_SHA ="3428dfd95309a7f3c84fd93259ded0f810d1ff91"  # [PP] NVIDIA/personaplex commit pin

AUDIO_CODEBOOKS_PER_STREAM = 8  # [PP] models/lm.py:54
NUM_AUDIO_CODEBOOKS = 16  # [PP] models/loaders.py:96
NUM_CODEBOOKS = NUM_AUDIO_CODEBOOKS + 1  # [PP] models/lm.py:398-399
DEP_Q = 16  # [PP] models/loaders.py:187 (overrides the 8 at line 97)
DELAYS = (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)  # [PP] models/loaders.py:121
MAX_DELAY = max(DELAYS)  # [PP] models/lm.py:690-692
CACHE_SIZE = MAX_DELAY + 3  # [PP] models/lm.py:705-716

AUDIO_CARD = 2048  # [PP] models/loaders.py:71,98
TEXT_CARD = 32000  # [PP] models/loaders.py:94
AUDIO_INITIAL_TOKEN_ID = AUDIO_CARD  # [PP] models/lm.py:355-358
TEXT_INITIAL_TOKEN_ID = TEXT_CARD  # [PP] models/lm.py:360-363
TEXT_PAD_TOKEN_ID = 3  # [PP] models/loaders.py:95; models/lm.py:365-371
TEXT_EPAD_TOKEN_ID = 0  # [PP] models/lm.py:373-376
TEXT_BOS_TOKEN_ID = 1  # [PP] offline.py:286-295
TEXT_EOS_TOKEN_ID = 2  # [PP] offline.py:286-295
ZERO_TOKEN_ID = -1  # [PP] models/lm.py:378-382 (zero embedding, not an extra row)
UNGENERATED_TOKEN_ID = -2  # [PP] models/lm.py:384-390
SILENCE_TOKENS = (948, 243, 1178, 546, 1736, 1030, 1978, 2008)  # [PP] models/lm.py:56
SINE_TOKENS = (430, 1268, 381, 1611, 1095, 1495, 56, 472)  # [PP] models/lm.py:57

SAMPLE_RATE = 24000  # [PP] models/loaders.py:39
FRAME_RATE = 12.5  # [PP] models/loaders.py:40
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # [PP] offline.py:211
SILENCE_FRAMES = int(0.5 * FRAME_RATE)  # [PP] offline.py:214
CONTEXT_FRAMES = 3000  # [PP] models/loaders.py:104
# Once its ring is full, RingKVCache.complete labels the slot at end_offset %
# capacity as position end_offset, which masks it ([PP] modules/transformer.py:
# 263-299). Dense causal attention therefore matches only up to C - 1 forwards.
MAX_TEMPORAL_FORWARDS = CONTEXT_FRAMES - 1
# Upper bound on decoded user audio: every forward a user frame, no conditioning.
# The exact per-request limit (after conditioning) is validate_context_budget's.
MAX_USER_AUDIO_SAMPLES = MAX_TEMPORAL_FORWARDS * FRAME_SIZE
# Requests the LM runs in one step (the max_batch_size model kwarg). The served
# YAML's max_concurrent_requests is the tighter bound in practice; 1 runs every
# request alone, which makes outputs independent of what else is being served.
DEFAULT_MAX_BATCH_SIZE = 8

TEMPORAL_DIM = 4096  # [PP] models/loaders.py:93
TEMPORAL_LAYERS = 32  # [PP] models/loaders.py:100
TEMPORAL_HEADS = 32  # [PP] models/loaders.py:99
TEMPORAL_HEAD_DIM = TEMPORAL_DIM // TEMPORAL_HEADS  # [PP] modules/transformer.py:383
TEMPORAL_FFN_DIM = int(4.125 * TEMPORAL_DIM)  # [PP] models/loaders.py:101; models/lm.py:304
TEMPORAL_FFN_HIDDEN = 2 * TEMPORAL_FFN_DIM // 3  # [PP] modules/gating.py:64
NORM = "rms_norm_f32"  # [PP] models/loaders.py:107
NORM_EPS = 1e-8  # [PP] modules/transformer.py:141-143 (create_norm_fn's rms_norm_f32 override)
GATING = "silu"  # [PP] models/loaders.py:106,118
ROPE_THETA = 10000  # [PP] models/loaders.py:105
DEPFORMER_DIM = 1024  # [PP] models/loaders.py:109
DEPFORMER_LAYERS = 6  # [PP] models/loaders.py:112
DEPFORMER_HEADS = 16  # [PP] models/loaders.py:111
DEPFORMER_HEAD_DIM = DEPFORMER_DIM // DEPFORMER_HEADS  # [PP] modules/transformer.py:383
DEPFORMER_FFN_DIM = int(4.125 * DEPFORMER_DIM)  # [PP] models/loaders.py:110
DEPFORMER_FFN_HIDDEN = 2 * DEPFORMER_FFN_DIM // 3  # [PP] modules/gating.py:64
DEPFORMER_CONTEXT = None  # [PP] models/lm.py:321 (overrides loaders.py:116)
DEPFORMER_POS_EMB = "none"  # [PP] models/loaders.py:119
DEPFORMER_WEIGHTS_PER_STEP = DEP_Q  # [PP] models/lm.py:322-323



def gating_hidden(dim: int, dim_feedforward: int) -> int:
    """Hidden width of ActivationGating ([PP] modules/gating.py:56-66)."""
    return (21 * dim) // 8 if dim_feedforward == 4 * dim else (2 * dim_feedforward) // 3


@dataclass(frozen=True)
class LMArchConfig:
    """Shapes of the temporal transformer and depformer.

    Defaults are the PersonaPlex values above; tests shrink the transformer
    widths and depths but keep the codebook layout, vocabularies and 16
    per-step depth slots.
    """

    dim: int = TEMPORAL_DIM
    num_heads: int = TEMPORAL_HEADS
    num_layers: int = TEMPORAL_LAYERS
    ffn_hidden: int = TEMPORAL_FFN_HIDDEN
    depformer_dim: int = DEPFORMER_DIM
    depformer_heads: int = DEPFORMER_HEADS
    depformer_layers: int = DEPFORMER_LAYERS
    depformer_ffn_hidden: int = DEPFORMER_FFN_HIDDEN
    card: int = AUDIO_CARD
    text_card: int = TEXT_CARD
    norm_eps: float = NORM_EPS
    rope_theta: float = ROPE_THETA

    def __post_init__(self):
        if self.dim % self.num_heads or self.depformer_dim % self.depformer_heads:
            raise ValueError("model widths must be divisible by their head counts")
        if (self.dim // self.num_heads) % 2:
            raise ValueError("RoPE needs an even temporal head dimension")

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @property
    def depformer_head_dim(self) -> int:
        return self.depformer_dim // self.depformer_heads

    @classmethod
    def from_reference_kwargs(cls, kwargs: dict) -> "LMArchConfig":
        """Build from an [PP] ``LMModel`` kwargs dict (models/loaders.py:92-122)."""
        if (kwargs["n_q"], kwargs["dep_q"]) != (NUM_AUDIO_CODEBOOKS, DEP_Q):
            raise ValueError("PersonaPlex needs n_q=16 and dep_q=16")
        if tuple(kwargs["delays"]) != DELAYS or kwargs.get("existing_text_padding_id") != TEXT_PAD_TOKEN_ID:
            raise ValueError("delays and text padding must match PersonaPlex")
        if (kwargs["gating"], kwargs["depformer_gating"], kwargs["norm"]) != (GATING, GATING, NORM):
            raise ValueError("only SiLU gating with rms_norm_f32 is supported")
        if not (kwargs["depformer_multi_linear"] and kwargs["depformer_weights_per_step"]):
            raise ValueError("the depformer needs per-codebook input projections and per-step weights")
        if kwargs.get("depformer_pos_emb") != DEPFORMER_POS_EMB or kwargs.get("positional_embedding") != "rope":
            raise ValueError("expected RoPE on the temporal transformer and no depformer positions")
        if kwargs.get("layer_scale") is not None or kwargs.get("depformer_layer_scale") is not None:
            raise ValueError("layer scale is not used by PersonaPlex")
        dim, ddim = kwargs["dim"], kwargs["depformer_dim"]
        dep_ffn = kwargs.get("depformer_dim_feedforward") or int(kwargs["hidden_scale"] * ddim)
        return cls(
            dim=dim, num_heads=kwargs["num_heads"], num_layers=kwargs["num_layers"],
            ffn_hidden=gating_hidden(dim, int(kwargs["hidden_scale"] * dim)),
            depformer_dim=ddim, depformer_heads=kwargs["depformer_num_heads"],
            depformer_layers=kwargs["depformer_num_layers"], depformer_ffn_hidden=gating_hidden(ddim, dep_ffn),
            card=kwargs["card"], text_card=kwargs["text_card"], rope_theta=kwargs["max_period"],
        )


DEFAULT_TEXT_PROMPT = (  # [PP] offline.py:335
    "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
)
SYSTEM_TAG = "<system>"  # [PP] offline.py:83-90 (wrap_with_system_tags)

# M* graph and resource names (not from the reference).
LM_NODE, MIMI_ENCODER_NODE, MIMI_DECODER_NODE = "LM", "MimiEncoder", "MimiDecoder"
ENCODE_USER_AUDIO_WALK, CONDITIONING_WALK, GENERATE_WALK = "encode_user_audio", "conditioning", "generate"
GENERATE_AUDIO_WALK = "generate_audio"  # generate without emitting text
CONDITIONING_LOOP, DECODE_LOOP = "cond_loop", "decode_loop"
KV_LABEL = "main"  # the LM's one paged KV label
LM_KV, LM_ATTN, LM_POS = "lm_kv", "lm_attn", "lm_pos"
TEXT_SAMPLER, DEPTH_SAMPLER = "text_sampler", "depth_sampler"

TEXT_TEMPERATURE = 0.7  # [PP] models/lm.py:653; offline.py:363-374
AUDIO_TEMPERATURE = 0.8  # [PP] models/lm.py:652; offline.py:363-374
TEXT_TOP_K = 25  # [PP] models/lm.py:655; offline.py:363-374
AUDIO_TOP_K = 250  # [PP] models/lm.py:654; offline.py:363-374


def validate_context_budget(*, voice_embeddings: int, text_tokens: int, user_frames: int) -> int:
    """Count temporal forwards, excluding the no-forward priming step.

    Voice replay runs once per embedding ([PP] models/lm.py:1031-1038),
    followed by two silence blocks, text and user frames (offline.py:241-276).
    Requests are capped at MAX_TEMPORAL_FORWARDS, the longest span over which the
    reference's ring KV equals M*'s dense causal attention.
    """
    counts = (voice_embeddings, text_tokens, user_frames)
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("voice_embeddings, text_tokens and user_frames must be nonnegative integers")
    total = voice_embeddings + 2 * SILENCE_FRAMES + text_tokens + user_frames
    if total > MAX_TEMPORAL_FORWARDS:
        raise ValueError(
            f"PersonaPlex context budget exceeded: voice_embeddings={voice_embeddings} + "
            f"silence_frames={2 * SILENCE_FRAMES} + text_tokens={text_tokens} + "
            f"user_frames={user_frames} = {total} > {MAX_TEMPORAL_FORWARDS}"
        )
    return total
