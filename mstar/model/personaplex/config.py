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
# source-cited constants and the PR1 context-budget check.
"""Constants verified against NVIDIA/personaplex at the pin below.

[PP] paths are relative to ``moshi/moshi/`` at that commit. Derived values
state their formula; the public HF config is only a stub (§2 R21).
"""

REFERENCE_SHA = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"  # [PP] NVIDIA/personaplex commit pin

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

TEXT_TEMPERATURE = 0.7  # [PP] models/lm.py:653; offline.py:363-374
AUDIO_TEMPERATURE = 0.8  # [PP] models/lm.py:652; offline.py:363-374
TEXT_TOP_K = 25  # [PP] models/lm.py:655; offline.py:363-374
AUDIO_TOP_K = 250  # [PP] models/lm.py:654; offline.py:363-374


def validate_context_budget(*, voice_embeddings: int, text_tokens: int, user_frames: int) -> int:
    """Count temporal forwards (§4.5), excluding the no-forward priming step.

    Voice replay runs once per embedding ([PP] models/lm.py:1031-1038),
    followed by two silence blocks, text and user frames (offline.py:241-276).
    PR1 caps forwards at the reference's sliding-window size until M* has
    sliding-window attention.
    """
    counts = (voice_embeddings, text_tokens, user_frames)
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("voice_embeddings, text_tokens and user_frames must be nonnegative integers")
    total = voice_embeddings + 2 * SILENCE_FRAMES + text_tokens + user_frames
    if total > CONTEXT_FRAMES:
        raise ValueError(
            f"PersonaPlex context budget exceeded: voice_embeddings={voice_embeddings} + "
            f"silence_frames={2 * SILENCE_FRAMES} + text_tokens={text_tokens} + "
            f"user_frames={user_frames} = {total} > {CONTEXT_FRAMES}"
        )
    return total
