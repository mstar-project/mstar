"""
Qwen3OmniModel: 3-partition streaming model for Qwen3-Omni-Moe.

Qwen3-Omni is a dual-AR multimodal model with a Thinker (30B-A3B MoE)
that reasons over text/audio/vision inputs and a Talker (3B-A0.3B MoE)
that converts Thinker hidden states into streaming codec tokens.  A
Code2Wav vocoder converts codec tokens to 24 kHz PCM audio.

Architecture (3 async partitions):
    Thinker  — multimodal encoder + MoE LLM (text, audio, vision prefill -> decode)
    Talker   — smaller MoE LLM that predicts codec tokens from Thinker hidden states
    Code2Wav — vocoder that converts codec tokens to audio waveform

Streaming topology:
    Thinker --[thinker_states, FixedChunkPolicy(1)]--> Talker
    Talker  --[codec_tokens,  FixedChunkPolicy(25)]--> Code2Wav

Conductor-triggered pipelined prefill (Approach C):
    After each Thinker walk completes (prefill_text, prefill_audio,
    prefill_vision, thinker_decode), the conductor sends a
    ``talker_trigger`` to the Talker partition.  During prefill each
    trigger extends the Talker KV cache with the new Thinker hidden
    states.  The final trigger (when thinker_decode starts) tells the
    Talker to sample its first codec token and transition to decode.

Text-only mode:
    When output_modalities does not include "audio", only the Thinker
    partition runs.  Talker and Code2Wav are idle.
"""

import logging
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    StreamingConnectionState,
)
from mstar.engine.base import EngineType
from mstar.engine.kv_store import KVCacheConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import MAX_OUTPUT_TOKENS, ForwardPassArgs, Model, TensorAndMetadata
from mstar.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import Operation, WeightConverter
from mstar.streaming.chunk_policy import FixedChunkPolicy, LeftContextChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge
from mstar.utils.sampling import SamplingConfig

logger = logging.getLogger(__name__)

# Performance defaults (ON), validated vs HF to cos>=0.9999. Opt out for the
# HF-identical baseline: MSTAR_GPU_MEL=0 / MSTAR_GPU_IMAGE_PREPROCESS=0 /
# MSTAR_VLLM_PROMPT_LAYOUT=0 / MSTAR_QWEN3_NATIVE_{AUDIO,VISION}_ENCODER=0.
# (MSTAR_VLLM_AUDIO_SENTINELS, MSTAR_BATCH_VISION_PREFILL stay opt-in.)
# GPU log-mel: mel spectrograms on GPU instead of HF's CPU WhisperFeatureExtractor.
_GPU_MEL = os.environ.get("MSTAR_GPU_MEL", "1") in ("1", "true", "True")


def gpu_log_mel(waveform, mel_filters, window, n_fft, hop):
    """GPU log-mel matching HF ``WhisperFeatureExtractor._np_extract_fbank_features``.

    ``waveform`` (any 1-D-reshapable tensor) -> ``(n_mel, T)`` float32 on the input
    device, ``T = floor(len/hop)`` (== HF's valid, un-padded frame count). Same hann
    window (periodic), center+reflect STFT, power spectrogram, drop-last-frame, log10,
    per-clip max-8 clamp, and (x+4)/4 normalization. Module-level so the parity test
    (test_qwen3_omni_gpu_mel_parity.py) guards the exact production transform.
    """
    wav = waveform.reshape(-1)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window,
                      center=True, pad_mode="reflect", return_complex=True)  # (n_freq, T+1)
    mag = stft[..., :-1].abs().pow(2)                 # drop last frame -> (n_freq, T)
    mel = mel_filters.T @ mag                         # (n_mel, T)
    log = torch.clamp(mel, min=1e-10).log10()
    log = torch.maximum(log, log.max() - 8.0)
    log = (log + 4.0) / 4.0
    return log.to(torch.float32)


def _envflag(name: str, default: bool = False) -> bool:
    """Read a boolean env flag. Accepts 1/true/yes/on; returns ``default`` if unset."""
    import os as _os

    raw = _os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def vllm_prompt_layout_enabled() -> bool:
    """Replicate vLLM-Omni's prompt layout: position the audio block INSIDE the
    user turn BEFORE the instruction text, so the effective Thinker sequence is
    system-turn, then user-turn = [audio][instruction], then assistant. ON by
    default (the benchmarked config; also gives audio M-RoPE h/w parity vs vLLM).
    Set MSTAR_VLLM_PROMPT_LAYOUT=0 for the legacy bare-block layout.
    """
    return _envflag("MSTAR_VLLM_PROMPT_LAYOUT", default=True)


def vllm_audio_sentinels_enabled() -> bool:
    """When ON, wrap the audio span with the real Qwen3-Omni audio marker token
    IDs (151669 ``<|audio_start|>`` / 151670 ``<|audio_end|>``, what vLLM uses)
    instead of the legacy 151647/151648 (mislabeled ``<|audio_bos|>``/
    ``<|audio_eos|>``). Default OFF. Independent of MSTAR_VLLM_PROMPT_LAYOUT so
    both can be tested separately."""
    return _envflag("MSTAR_VLLM_AUDIO_SENTINELS")


def chunked_prefill_v2_enabled() -> bool:
    """Chunked Thinker prefill. When ON, a long Thinker prefill walk is
    split into <=C-token chunks run as separate normal prefill steps, so the
    worker round-robin interleaves them with decode steps instead of stalling
    every decoder for one ~27.5ms mega-step. Default OFF -> flag-off is
    byte-identical (no schedule split, no chunk metadata emitted).

    Gated to ``audio_output=False`` requests (i2t/t2t): when the Talker is
    conditioned the per-walk ``thinker_states`` accounting forbids splitting a
    walk.
    """
    return _envflag("MSTAR_CHUNKED_PREFILL_V2")


def chunked_prefill_v2_vision_enabled() -> bool:
    """Secondary gate (within V2) for chunking the **vision** (and audio)
    Thinker prefill, which requires the encoder-split walk + staged
    embeds/pos_ids/deepstack window slicing. Default OFF so text chunking can
    ship independently while the deepstack/mm_mask window alignment gets GPU
    validation. No-op unless ``MSTAR_CHUNKED_PREFILL_V2`` is also ON.
    """
    return _envflag("MSTAR_CHUNKED_PREFILL_V2_VISION")


def chunked_prefill_v2_assert() -> bool:
    """DEBUG validation for chunked prefill: after the last chunk of a walk,
    assert the conductor's summed chunk tokens equal the unchunked walk span.
    Cheap conductor-side check; default OFF.
    """
    return _envflag("MSTAR_CHUNKED_PREFILL_V2_ASSERT")


def mixed_batch_enabled() -> bool:
    """Mixed prefill+decode CAPTURED batch. When ON, the worker scheduler
    may assemble ONE fully-captured forward covering N ``thinker_decode`` rows
    (1 token each) plus ONE prefill-chunk row (C tokens), eliminating the
    decode/prefill-chunk alternation that chunked prefill alone still leaves.

    Default OFF -> flag-off is byte-identical: the scheduler never emits a
    ``thinker_mixed`` batch, so assembly / preprocess / dispatch changes are
    all unreachable. Requires ``MSTAR_CHUNKED_PREFILL_V2`` to produce the
    prefill chunks that a mixed step consumes; with V2 off there are no chunk
    rows to mix, so a mixed batch never assembles and the path falls back to
    normal decode.
    """
    return _envflag("MSTAR_MIXED_BATCH")


def mixed_batch_spec_enabled() -> bool:
    """Spec-chain integration: fold a mixed step INTO the running decode
    speculation chain instead of breaking the chain to run mixed on the
    non-speculative path. When ON, and a decode spec chain is live and
    a mixable prefill chunk becomes ready, the NEXT speculative batch is
    assembled as MIXED (the decode rids continue exactly as the normal
    continuation + the chunk row injected), submitted through the normal spec
    pipeline (reserve slot, submit-before-postprocess-of-N, overlap preserved),
    and the chain CONTINUES uninterrupted afterwards (the following spec step
    threads decode tokens out of the mixed step's outputs).

    Default OFF, and a strict extension of ``MSTAR_MIXED_BATCH``: with this flag
    off the worker keeps the non-folded behavior (break the chain → non-spec
    mixed step → chain re-warm), byte-identical. Without folding, each mixed
    step trades ~9ms of stall savings for ~15-25ms of lost speculation overlap
    (a net loss); riding the chain recovers that. Implies
    ``MSTAR_MIXED_BATCH`` (there is nothing to fold in without mixed steps).
    """
    return _envflag("MSTAR_MIXED_SPEC") and mixed_batch_enabled()


def mixed_batch_preplan_enabled() -> bool:
    """Pre-plan a chain-folded ``thinker_mixed`` step's packed
    FlashInfer attention on the plan_executor thread, exactly like decode
    pre-plan, so the GPU thread's mixed submission finds the prefill wrapper
    already planned and skips its inline plan.

    Without this, a folded mixed batch reserves NO slot and gets NO pre-plan
    (``reserve_replay_slot`` / ``pre_plan_for_batch`` both key on the
    BASIC_BATCHED-only ``_get_basic_batched_key_for``, which returns None for
    FLASH_INFER_PACKED). So the packed prefill-wrapper plan (~0.75-1.5ms for a
    32-row bucket: qo_indptr + paged indices + CUB scan) plus the packed input
    prep run INLINE on the GPU thread between replay(N) and the mixed replay —
    a serial bubble the uniform decode chain otherwise hides via plan_executor
    overlap.

    When ON, the mixed fold reserves a packed slot at fold time and dispatches
    the packed plan on plan_executor gated on N's advance_event, mirroring
    decode pre-plan. Default OFF -> flag-off is byte-identical (the packed
    reserve / pre-plan / reset surfaces all short-circuit). Implies
    ``MSTAR_MIXED_SPEC`` (there is no chain-folded mixed step to pre-plan
    without it).
    """
    return _envflag("MSTAR_MIXED_PREPLAN") and mixed_batch_spec_enabled()


def mixed_batch_vision_enabled() -> bool:
    """Allow a VISION prefill chunk (not just ``prefill_text``) to
    ride a captured ``thinker_mixed`` step. When ON, the scheduler may pick a
    ``prefill_vision`` chunk row as the mixed batch's single chunk row, and the
    Thinker's mixed capture carries per-layer ``deepstack_<i>`` statics + the
    per-request MRoPE ``mrope_pos_advance`` side-channel so the vision chunk's
    deepstack splice and 3D-grid position advance run inside the captured graph.

    Default OFF, and a strict extension of ``MSTAR_MIXED_BATCH``: with this
    flag off the mixed capture stays text-signature only and the scheduler
    gates the chunk row to ``prefill_text`` (the text-only behavior, byte-identical).
    Only meaningful when BOTH ``MSTAR_MIXED_BATCH`` (produces mixed steps) and
    ``MSTAR_CHUNKED_PREFILL_V2_VISION`` (produces vision chunk rows via the
    encoder-split walk + staged embeds/pos_ids/deepstack) are also ON; with
    either off there are no vision chunk rows to mix, so this is a no-op.
    """
    return (
        _envflag("MSTAR_MIXED_BATCH_VISION")
        and mixed_batch_enabled()
        and chunked_prefill_v2_vision_enabled()
    )


# --- Capture-provisioning latch (illegal-memory-access safety) --------------
# ``mixed_batch_vision_enabled()`` reads the env LIVE, and the env can be
# mutated mid-run (e.g. to A/B a flag on a single running server). So a
# boot-OFF -> runtime-ON flip of MSTAR_MIXED_BATCH_VISION could make the
# scheduler start routing a ``prefill_vision`` chunk into a ``thinker_mixed``
# capture that was built WITHOUT the per-layer ``deepstack_<i>`` static
# buffers. Those buffers are baked into the graph's ``static_input_keys`` at
# capture time (submodules.get_cuda_graph_configs / _build_prefill_vision_packed)
# and CANNOT be added afterwards; replaying a vision chunk there gives
# ``preprocess`` no static buffer to copy deepstack into — an illegal memory
# access on an uncaptured bucket.
#
# Routing therefore must gate on what the capture ACTUALLY provisioned, not on the
# live flag. ``get_cuda_graph_configs`` records the truth once at boot via
# ``mark_mixed_vision_provisioned``; the scheduler consults
# ``mixed_vision_capture_provisioned`` (see MicroScheduler._mixed_chunk_walks).
# This mirrors CudaGraphRunner._split_attn_env_snapshot, which process-statics the
# same class of capture-baked flag against the same live-env desync hazard.
_MIXED_VISION_PROVISIONED: bool | None = None


def mark_mixed_vision_provisioned(provisioned: bool) -> None:
    """Record, at capture time, whether the ``thinker_mixed`` CUDA-graph capture
    was built with per-layer deepstack static buffers (i.e. is able to replay a
    ``prefill_vision`` chunk row). Called once from ``get_cuda_graph_configs``.

    Immutable-by-convention after boot: the value reflects the ONE capture that
    ran and must NOT track later env changes — that immutability is the
    whole point of the latch."""
    global _MIXED_VISION_PROVISIONED
    _MIXED_VISION_PROVISIONED = bool(provisioned)


def mixed_vision_capture_provisioned() -> bool:
    """True iff the captured ``thinker_mixed`` step actually carries deepstack
    static buffers — the IMA-safe precondition for routing a ``prefill_vision``
    chunk into it (see ``_MIXED_VISION_PROVISIONED``).

    After boot the recorded capture truth is authoritative, so a runtime
    MSTAR_MIXED_BATCH_VISION flip can never route to an unprovisioned graph.
    Before any capture has run (value ``None`` — pure-CPU unit tests, or the
    pre-capture boot window that never overlaps real scheduling) it falls back to
    the live flag intent so flag-driven tests and boot ordering behave as written;
    production routing only ever happens post-capture, where the record wins."""
    if _MIXED_VISION_PROVISIONED is None:
        return mixed_batch_vision_enabled()
    return _MIXED_VISION_PROVISIONED


def mixed_split_attn_enabled() -> bool:
    """Split attention (MSTAR_MIXED_SPLIT_ATTN): captured thinker_mixed
    steps plan their decode rows on a tensor-core DECODE wrapper and the chunk
    row on its own prefill wrapper instead of one BatchPrefill plan over the
    whole mixed shape. In-graph microbenchmarking showed the single
    mixed-shape PLAN is what's slow, not the kernel (the decode wrapper is
    tensor-core = prefill kernel inside; plain decode kernel rejects GQA
    group 7). Requires the fixed-region row layout in _run_flashinfer_packed
    (real decode rows, then qo=1 dummies, chunk row last). Default OFF.
    """
    return _envflag("MSTAR_MIXED_SPLIT_ATTN")


def mixed_single_chunk_enabled() -> bool:
    """Fold-rate: let a prefill span that FITS in one chunk still take the
    chunked path as a single chunk, so it carries ``prefill_chunk_len``
    metadata and becomes ELIGIBLE to fold into a ``thinker_mixed`` step.

    Motivation: only a minority of prefill steps folded; the mixable gate
    excludes any step without chunk metadata,
    which is every short span (span <= MSTAR_PREFILL_CHUNK_TOKENS) — e.g. the
    question-text walk of an i2t prompt. A one-chunk chunked prefill runs the
    same math over [0, span) and pads to the same capture bucket when it does
    NOT fold, so the standalone cost is unchanged; when it does fold, the
    whole walk rides a decode step instead of serializing the batch.

    Default OFF (opt-in A/B). Only meaningful with ``MSTAR_MIXED_BATCH`` on;
    callers AND it with the matching mixed flag per walk kind.
    """
    return _envflag("MSTAR_MIXED_SINGLE_CHUNK")


def mixed_budget_tokens() -> int:
    """V2 budgeted chunked-prefill admission (``MSTAR_MIXED_BUDGET_TOKENS``).

    vLLM-style every-step chunk admission for M*'s captured-mixed machinery.
    Today a ready prefill chunk folds into the running decode spec chain ONLY at
    a fairness *yield boundary* (``must_yield_away`` — ~8% of steps); under
    continuous arrivals a mixable chunk then sits idle for several steps before
    it rides a decode step. With this budget > 0 the worker probes on EVERY spec
    chain step (like ``MSTAR_MIXED_SINGLE_CHUNK`` did) and folds a ready chunk
    NOW, capping the mixed step at ``budget`` total tokens (``n_decode`` 1-token
    rows + the ``C``-token chunk) so raised fold volume never assembles an
    oversized step. 0 (the default) disables it — flag-off is byte-identical.

    CRITICAL distinction from the deprecated ``MSTAR_MIXED_SINGLE_CHUNK``
    (net-negative): that flag ALSO routed short standalone prefills through the
    chunk planner (``allow_single_chunk``) so every short span became foldable —
    which capped admission throughput and starved decode occupancy ~10%. This
    budget does NOT touch ``allow_single_chunk``; the mixable gate
    (``_chunk_entry_passes_gates`` still requires ``prefill_chunk_len``) is
    unchanged, so ONLY genuinely-chunked long prefills — the chunks that already
    exist in the pipeline — are accelerated. Standalone/unchunked prefill
    admission is identical to today. The expected win is TTFT / admission
    latency at B2-B8 and arrival-heavy patterns; a fold is roughly compute-
    neutral (mixed ~30-36ms vs prefill+decode ~29ms replaced), so steady B32
    closed-loop is expected ~neutral (must not regress).

    Only meaningful with ``MSTAR_MIXED_SPEC`` (there is no spec-chain fold to
    accelerate without it, and thus ``MSTAR_MIXED_BATCH``); the worker ANDs it
    with ``mixed_batch_spec_enabled()``. Unlike the split-attn / single-chunk
    flags this does NOT bake anything into the CUDA-graph capture (it only
    changes WHEN an existing chunk folds), so it is safe to flip at runtime
    mid-run.
    """
    import os as _os
    raw = _os.environ.get("MSTAR_MIXED_BUDGET_TOKENS")
    if raw is None:
        return 0
    try:
        v = int(raw.strip())
    except ValueError:
        return 0
    return v if v > 0 else 0


def eager_fold_enabled() -> bool:
    """EAGER FOLD FALLBACK: co-admit a brand-new request's FIRST
    prefill chunk into the running decode step even when the chunk is LARGER than
    the largest captured mixed bucket (``MicroScheduler._MIXED_MAX_CHUNK_TOKENS``
    = 512), by running that ONE step EAGER — a single uncaptured varlen forward
    over the decode rows + the full chunk row (vLLM's exact unified-step shape).

    Motivation: the captured mixed fold (MSTAR_MIXED_BATCH / _SPEC / COADMIT) is
    hard-capped at C <= 512 because its only chunk buckets are {256, 512} (a
    larger chunk would route to an uncaptured graph = the UNCAP IMA hazard). On
    the ship config (MSTAR_PREFILL_CHUNK_TOKENS=2048) an i2t text prefill chunk
    is up to 2048 tokens, so it can NEVER fold today and instead runs standalone,
    FREEZING the concurrent decodes for that step. vLLM avoids exactly this by
    folding a full prefill into the decode step — because vLLM's prefill runs
    EAGER (dynamic per-step shape). This flag gives M* the same escape hatch: for
    a chunk in (512, MSTAR_EAGER_FOLD_MAX_CHUNK], assemble a ``thinker_mixed``
    batch and let it run eager (no captured graph matches, so
    ``KVCacheEngine.execute_forward`` falls to ``_execute_batched`` — the existing
    eager varlen path that already handles the ``thinker_mixed`` graph_walk).

    Byte-identical when off: the scheduler never relaxes the 512 chunk cap and the
    worker never breaks the decode chain for an eager fold, so every captured /
    spec / coadmit path is unchanged. When on it is a strict EXTENSION of
    ``MSTAR_MIXED_BATCH`` (the assembly machinery it reuses), so it ANDs with
    ``mixed_batch_enabled()``. The eager step is slower per-step than a replay, so
    the worker gates it to brand-new requests (fwd_index==0) and caps its
    frequency (MSTAR_EAGER_FOLD_MIN_GAP). Text chunks only unless the boot-time
    ``MSTAR_MIXED_BATCH_VISION`` flag is on (a vision chunk needs deepstack, which
    ``preprocess`` assembles for a mixed step only under that flag).
    """
    return _envflag("MSTAR_EAGER_FOLD") and mixed_batch_enabled()


def eager_fold_max_chunk() -> int:
    """Largest chunk C (tokens) the EAGER FOLD path (MSTAR_EAGER_FOLD) will
    co-admit into an eager mixed step. A chunk above this stays on today's
    standalone path (so an arbitrarily large prefill never builds one pathological
    eager step — bounded like vLLM's practical image/prompt sizes, and keeping the
    FlashInfer prefill workspace within the same envelope the standalone prefill
    already uses). Default 2048 = the largest ``PREFILL_TOKEN_BUCKETS`` entry, so
    it covers every chunk the default planner emits (ship
    MSTAR_PREFILL_CHUNK_TOKENS=2048). Raise it only alongside a larger prefill
    chunk cap. Read via MSTAR_EAGER_FOLD_MAX_CHUNK.
    """
    import os as _os

    raw = _os.environ.get("MSTAR_EAGER_FOLD_MAX_CHUNK")
    if raw is None:
        return 2048
    try:
        v = int(raw.strip())
    except ValueError:
        return 2048
    return v if v > 0 else 2048


def prefill_chunk_tokens() -> int:
    """Cap on chunk size C for chunked prefill. The planner picks the largest
    ``ThinkerSubmodule.PREFILL_TOKEN_BUCKETS`` entry <= min(remaining, this cap),
    floored at 128. Default 512.
    """
    import os as _os

    raw = _os.environ.get("MSTAR_PREFILL_CHUNK_TOKENS")
    if raw is None:
        return 512
    try:
        v = int(raw.strip())
    except ValueError:
        return 512
    return v if v > 0 else 512


# Chunk-size buckets: mirror ThinkerSubmodule.PREFILL_TOKEN_BUCKETS so a chunk
# step lands on an already-captured prefill CUDA-graph config. Kept as a
# module-level copy so the pure planner below does not import the submodule.
_PREFILL_CHUNK_BUCKETS = [128, 256, 512, 1024, 2048]


def plan_prefill_chunk(
    span: int, offset: int, cap: int,
    allow_single_chunk: bool = False,
) -> tuple[int, bool] | None:
    """Pure planner for one chunk of a resumable chunked Thinker prefill.

    ``span`` is the full token count the Thinker sees for this walk (already
    including any sentinel tokens). ``offset`` is how many tokens of that span
    prior chunks already consumed. ``cap`` is ``MSTAR_PREFILL_CHUNK_TOKENS``.

    Returns ``(chunk_len, walk_done)`` for the chunk starting at ``offset``, or
    ``None`` when the walk should NOT be chunked (span fits in a single chunk),
    so callers fall back to the byte-identical single-shot path.

    ``allow_single_chunk``: when True, a span that fits in one chunk returns
    ``(span, True)`` instead of ``None`` — a one-chunk chunked prefill. Same
    math as the single-shot path (one step covering [0, span)), but it carries
    ``prefill_chunk_len`` metadata, which is what makes the step ELIGIBLE to
    fold into a thinker_mixed batch (the scheduler's mixable gate requires
    chunk metadata). Callers pass mixed_batch_enabled() here: without mixed
    batching the metadata buys nothing, so short spans keep the byte-identical
    single-shot path.

    Chunk size = largest ``_PREFILL_CHUNK_BUCKETS`` entry <= min(remaining, cap),
    floored at 128, but never past the end of the span. ``walk_done`` is True
    when this chunk reaches the end of the span.
    """
    if span <= 0:
        return None
    offset = max(0, int(offset))
    remaining = span - offset
    if remaining <= 0:
        return None
    # First-chunk decision: if the whole span fits in one capped bucket-sized
    # chunk, don't chunk at all (single-shot, byte-identical to flag-off) —
    # unless allow_single_chunk wants the chunk metadata for mixability.
    if offset == 0 and span <= max(128, min(cap, _PREFILL_CHUNK_BUCKETS[-1])):
        # Only skip chunking when a single chunk actually covers the span,
        # i.e. span <= cap and span <= the largest bucket. Otherwise fall
        # through and chunk.
        if span <= cap:
            if allow_single_chunk:
                return span, True
            return None
    limit = min(cap, remaining)
    # Largest bucket <= limit, floored at 128.
    chunk_len = 128
    for b in _PREFILL_CHUNK_BUCKETS:
        if b <= limit:
            chunk_len = b
        else:
            break
    # Never exceed the remaining span.
    chunk_len = min(chunk_len, remaining)
    # Tail-merge: if the leftover after this chunk is tiny (<= 32 tokens,
    # e.g. the +2 sentinels of a 258-token vision span after a 256 chunk),
    # absorb it into this chunk instead of scheduling a separate micro-step.
    # Live evidence: ~40% of assembled mixed steps were C=2 tails before
    # this. The merged chunk still pads to the same capture bucket (buckets
    # have headroom over the nominal C), so no new bucket is required.
    leftover = span - (offset + chunk_len)
    if 0 < leftover <= 32:
        chunk_len += leftover
    walk_done = (offset + chunk_len) >= span
    return chunk_len, walk_done


def batch_vision_prefill_enabled() -> bool:
    """When ON, allow the Thinker ``prefill_vision`` walk to batch more than
    one request per step (like ``prefill_audio`` / ``prefill_text`` already
    do): the vision tower runs once over the concatenated multi-image batch
    and all requests are prefilled together, cutting Image-to-Text TTFT.

    Default OFF -> ``preprocess`` keeps the single-request assert and the
    eager single-request side-channels, so behavior is byte-identical to the
    pre-batching path. See ``ThinkerSubmodule.preprocess`` /
    ``get_cuda_graph_configs`` in ``submodules.py``.
    """
    return _envflag("MSTAR_BATCH_VISION_PREFILL")


def merged_prefill_enabled() -> bool:
    """Merged multimodal prefill (old plan rows B1/B5). When ON, an i2t
    admission whose Thinker prefill schedule is exactly one ``prefill_text`` +
    one ``prefill_vision`` (either order) collapses into a SINGLE
    ``prefill_multimodal`` walk that runs both spans in one Thinker forward,
    dropping the conductor round-trip between the two walks.

    The merged walk is a ``Sequential[vision_encoder → Thinker]`` (the encoder
    still runs first). Its post-preprocess tensor signature is IDENTICAL to
    ``prefill_vision`` (``input_embeds`` + ``cos_3d`` + ``sin_3d`` +
    per-layer ``deepstack_<i>``; ``mrope_pos_advance`` via the ``_PlanState``
    side-channel), so it REUSES the ``prefill_vision`` CUDA-graph capture — no
    new capture, no extra warmup.

    Default OFF -> flag-off is byte-identical: the walk is never registered, the
    schedule is never collapsed, so no i2t admission can route to it. Merge is
    gated to text output (no Talker involvement), non-chunked vision
    (``MSTAR_CHUNKED_PREFILL_V2_VISION`` off), and the exact one-text+one-vision
    schedule; anything else keeps the unmerged multi-walk path unchanged.
    """
    return _envflag("MSTAR_MERGED_PREFILL")


def merged_prefill_audio_enabled() -> bool:
    """Merged multimodal prefill, AUDIO twin (attacks s2t B2/B4). When ON, an
    admission whose Thinker prefill schedule is exactly one ``prefill_text`` +
    one ``prefill_audio`` (either order) collapses into a SINGLE
    ``prefill_multimodal_audio`` walk that runs both spans in one Thinker
    forward, dropping the conductor round-trip between the two walks.

    The merged walk is a ``Sequential[audio_encoder → Thinker]`` (the encoder
    still runs first). Audio carries NO deepstack and NO 3D-grid MRoPE jump
    (positions increment one per token, so the walk's MRoPE advance == its
    ``seq_len``), so the merged span's post-preprocess signature is IDENTICAL to
    ``prefill_text`` / ``prefill_audio`` (``input_embeds`` + ``cos_3d`` +
    ``sin_3d`` + ``masks_for_talker``; no side-channel). It therefore REUSES the
    ``prefill_text`` CUDA-graph capture (NOT the vision capture) — no new
    capture, no extra warmup.

    Independent of ``MSTAR_MERGED_PREFILL`` (the vision flag): this gate alone
    controls whether the audio walk is registered and the audio schedule is
    collapsed, so s2t (which has no vision) can A/B the audio merge without
    touching vision-merge behavior. Default OFF -> flag-off is byte-identical:
    the walk is never registered, the schedule is never collapsed. Merge is
    gated to text output (no Talker output involvement, so s2t is eligible /
    s2s+i2s are not) and the exact one-text+one-audio schedule; anything else
    keeps the unmerged multi-walk path unchanged. There is no chunked-audio
    walk (unlike vision's encode_vision split), so there is no
    chunked-prefill interaction to disable — see the interaction note in the
    branch report.
    """
    return _envflag("MSTAR_MERGED_PREFILL_AUDIO")


def merged_prefill_audio_max_bs() -> int:
    """Live-occupancy ceiling for the audio merge (``MSTAR_MERGED_PREFILL_AUDIO_MAX_BS``,
    default 24). ``MSTAR_MERGED_PREFILL_AUDIO`` registers the merged walk; this gate
    decides PER ADMISSION whether to actually use it, based on the live active-request
    count. Measured: the merge wins big at low concurrency (s2t B<=16 tok/s +19-28%)
    because it collapses the two decode-blocking prefill steps into one, but at B32 the
    heavier merged prefill stalls the dense decode wave and regresses ~26%. Merging only
    when occupancy <= this ceiling keeps the low-batch win without the high-batch loss,
    so one config wins at every batch. Set very high => always merge (old behavior);
    very low => never merge. Both merged and unmerged walks are captured in the same
    boot, so the per-admission choice never triggers recapture (parity-safe either way).
    """
    import os as _os
    raw = _os.environ.get("MSTAR_MERGED_PREFILL_AUDIO_MAX_BS")
    if raw is None:
        return 24
    try:
        v = int(raw)
    except ValueError:
        return 24
    return v


def _hf_encoder_attn_impl() -> str:
    """Pick the attention implementation for the HF-wrapper encoder fallback.

    The HF ``Qwen3OmniMoe{Audio,Vision}Encoder`` classes hard-fail at init if
    asked for ``flash_attention_2`` without the ``flash_attn`` package present
    (transformers raises ImportError rather than degrading). The native encoder
    path already degrades gracefully to torch SDPA when flash_attn is missing
    (see bagel vit_encoder), so the HF path must do the same to keep both
    variants benchmarkable on the *same* hardware footing. We only request FA2
    when flash_attn actually imports.
    """
    import importlib.util

    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    logger.warning(
        "flash_attn is not available; HF-wrapper encoders will fall back to "
        "torch SDPA (slower than FA2 varlen, but matches the native path's "
        "fallback so the M*-old vs M*-new comparison stays on equal footing)."
    )
    return "sdpa"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_local_hf_snapshot(repo_id: str, cache_dir: str | None = None) -> str:
    """Download (or locate) a HuggingFace snapshot and return the local path."""
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=False,
        )
    except Exception as e:
        logger.warning("Error downloading from HuggingFace: %s", str(e))
        return repo_id
    return str(Path(local_dir))


# GPU image preprocessing: the CPU round-trip to HF's Qwen2VLImageProcessor is the
# biggest I2T TTFT cost (~175 ms). MSTAR_GPU_IMAGE_PREPROCESS=1 runs the identical
# algorithm on-GPU (torchvision bicubic resize, same kernel HF calls); grid_thw is
# bit-exact, pixel_values cos>0.9999. =0 restores the byte-identical HF CPU path.


def _gpu_image_preprocess_enabled() -> bool:
    import os

    # ON by default (benchmarked canonical config); MSTAR_GPU_IMAGE_PREPROCESS=0
    # falls back to the byte-identical HF CPU image processor.
    return os.environ.get("MSTAR_GPU_IMAGE_PREPROCESS", "1") != "0"


def _smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Port of HF ``smart_resize`` (Qwen2VLImageProcessor).  Pure python ints."""
    import math

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _gpu_image_preprocess(
    img: "torch.Tensor",
    *,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean,
    image_std,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Resize + rescale + normalize + patchify a single image on its device.

    ``img`` is a (C, H, W) tensor on the GPU, float in [0, 1] (as produced by
    data_worker) or uint8 in [0, 255].  Returns ``(pixel_values, grid_thw)``
    matching HF's ``Qwen2VLImageProcessor`` output for one image:
    ``pixel_values`` is 2-D ``(grid_h*grid_w, C*temporal*patch*patch)`` and
    ``grid_thw`` is ``(1, 3)`` long ``[[1, grid_h, grid_w]]``.
    """
    from torchvision.transforms.v2.functional import InterpolationMode
    from torchvision.transforms.v2.functional import resize as tv_resize

    # Normalise layout to (C, H, W) and dtype to uint8 in [0, 255], exactly as
    # the CPU path does before handing the array to HF (which then casts to
    # uint8 -> tvF.resize).
    if img.dim() == 3 and img.shape[-1] in (1, 3) and img.shape[0] not in (1, 3):
        img = img.permute(2, 0, 1)  # HWC -> CHW
    if img.dtype.is_floating_point:
        img_u8 = (img * 255.0).clamp(0, 255).to(torch.uint8)
    else:
        img_u8 = img.to(torch.uint8)
    img_u8 = img_u8.contiguous()

    C, H, W = img_u8.shape
    factor = patch_size * merge_size
    h_bar, w_bar = _smart_resize(H, W, factor, min_pixels, max_pixels)

    # Resize on-device with the same torchvision kernel HF's fast backend uses
    # (bicubic + antialias on uint8).
    resized = tv_resize(
        img_u8,
        [h_bar, w_bar],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )

    # Fused rescale (1/255) + normalize, matching HF's
    # ``_fuse_mean_std_and_rescale_factor``: mean *= 255, std *= 255, then
    # (x - mean) / std on the float32 image.
    dev = resized.device
    mean_t = torch.as_tensor(image_mean, device=dev, dtype=torch.float32) * 255.0
    std_t = torch.as_tensor(image_std, device=dev, dtype=torch.float32) * 255.0
    patches = resized.to(torch.float32)
    patches = (patches - mean_t[:, None, None]) / std_t[:, None, None]
    patches = patches.unsqueeze(0)  # (1, C, h_bar, w_bar)

    grid_h, grid_w = h_bar // patch_size, w_bar // patch_size
    patches = patches.reshape(
        1,
        C,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    # [batch, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    flatten_patches = (
        patches.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
        .reshape(
            grid_h * grid_w,
            C * temporal_patch_size * patch_size * patch_size,
        )
    )
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    return flatten_patches, grid_thw


# ---------------------------------------------------------------------------
# Qwen3OmniModel
# ---------------------------------------------------------------------------

class Qwen3OmniModel(Model):
    """Qwen3-Omni: Thinker + Talker + Code2Wav 3-partition streaming model."""

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.model_path_hf = model_path_hf

        self.CONVERTER = [
            WeightConverter(
                source_patterns=[
                    "mlp.experts.*.gate_proj.weight",
                    "mlp.experts.*.up_proj.weight",
                ],
                target_patterns="mlp.experts.gate_up_proj",
                operations=[
                    Operation("MergeModulelist",  dim=0),
                    Operation("Concatenate", dim=1)
                ]
            ),
            WeightConverter(
                source_patterns=["mlp.experts.*.down_proj.weight"],
                target_patterns="mlp.experts.down_proj",
                operations=[Operation("MergeModulelist",  dim=0)],
            ),
        ]

        # Load config from pretrained checkpoint
        from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig

        local_dir = _resolve_local_hf_snapshot(model_path_hf, cache_dir=cache_dir)
        self.config = Qwen3OmniModelConfig.from_pretrained(local_dir)
        self.local_dir = local_dir

        # Allow yaml model_kwargs / constructor kwargs to toggle native encoders
        # (e.g. model_kwargs: {native_audio_encoder: true, native_vision_encoder: true}).
        for _flag in ("native_audio_encoder", "native_vision_encoder"):
            if _flag in kwargs:
                setattr(self.config, _flag, bool(kwargs[_flag]))

        # Tokenizer (Thinker uses a Qwen-family tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir, cache_dir=cache_dir, trust_remote_code=True,
        )

        # Full multimodal processor: combines tokenizer + image_processor +
        # video_processor + audio feature_extractor + chat template support.
        # Used by process_prompt to build the full ChatML prompt with the
        # correct image_pad / audio_pad / video_pad expansion.
        try:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                local_dir, cache_dir=cache_dir, trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(
                "Could not load Qwen3-Omni AutoProcessor (%s); "
                "process_prompt will fall back to raw tokenizer.encode.",
                e,
            )
            self._processor = None

        # Lazy submodule cache -- each worker only loads what it needs
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

        # GPU log-mel state (MSTAR_GPU_MEL=1): cached filterbank + window per
        # device, built lazily on first audio request. Default OFF -> HF path.
        self._gpu_mel_state: dict | None = None

    # -----------------------------------------------------------------------
    # Model ABC: KV cache config
    # -----------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        """Return separate KV cache configs for Thinker and Talker."""
        thinker_cfg = KVCacheConfig(
            num_layers=self.config.thinker_text.num_hidden_layers,
            num_kv_heads=self.config.thinker_text.num_key_value_heads,
            head_dim=self.config.thinker_head_dim,
            max_seq_len=self.config.thinker_text.max_position_embeddings,
            num_qo_heads=self.config.thinker_text.num_attention_heads,
            nodes=["Thinker"]
        )
        talker_cfg = KVCacheConfig(
            num_layers=self.config.talker_text.num_hidden_layers,
            num_kv_heads=self.config.talker_text.num_key_value_heads,
            head_dim=self.config.talker_head_dim,
            max_seq_len=self.config.thinker_text.max_position_embeddings,
            num_qo_heads=self.config.talker_text.num_attention_heads,
            nodes=["Talker"]
        )
        return [thinker_cfg, talker_cfg]

    # -----------------------------------------------------------------------
    # Model ABC: node engine types
    # -----------------------------------------------------------------------

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            "audio_encoder": EngineType.STATELESS,
            "vision_encoder": EngineType.STATELESS,
            "Thinker": EngineType.KV_CACHE,
            "Talker": EngineType.KV_CACHE,
            "Code2Wav": EngineType.STATELESS,
        }

    def get_max_talker_output_tokens(self, **model_kwargs):
        return model_kwargs.get("talker_max_output_tokens", MAX_OUTPUT_TOKENS)

    # -----------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -----------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphNode | Sequential]:
        """Define all graph walks for the 3-partition architecture.

        Thinker walks:
            prefill_text   - text token embedding + Thinker prefill
            prefill_audio  - audio feature encoding + Thinker prefill
            prefill_vision - vision feature encoding + Thinker prefill
            thinker_decode - autoregressive text token generation

        Talker walks:
            talker_prefill - prefill Talker KV cache from Thinker states
            talker_decode  - autoregressive codec token generation

        Code2Wav walks:
            code2wav_chunk - vocoder streaming decode
        """
        # -- Thinker prefill walks: process inputs and stream hidden states
        #    to the Talker partition via StreamingGraphEdge --
        prefill_text = GraphNode(
            name="Thinker",
            input_names=["text_inputs"],
            outputs=[
                GraphEdge( # last prefill samples a token
                    next_node=EMIT_TO_CLIENT,
                    name="new_token",
                    output_modality="text",
                    persist=True,
                ),
                StreamingGraphEdge(
                    next_node="Talker",
                    name="thinker_states",
                    target_partition="Talker",
                ),
                # The thinker_mask tensor includes two masks: one for multimodal inputs,
                # and one for text inputs (allowing us to cut out the system prompt and
                # assistant history from the talker input)
                StreamingGraphEdge(
                    next_node="Talker",
                    name="thinker_mask",
                    target_partition="Talker",
                ),
            ],
        )

        prefill_audio = Sequential([
            GraphNode(
                name="audio_encoder",
                # audio_seqlens carries the original (pre-padding) length of
                # each audio clip, used by the encoder to compute attention
                # masks and output position IDs.
                input_names=["audio_features", "audio_seqlens"],
                outputs=[GraphEdge(next_node="Thinker", name="audio_embeds")],
            ),
            GraphNode(
                name="Thinker",
                input_names=["audio_embeds"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        persist=True,
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_states",
                        target_partition="Talker",
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_mask",
                        target_partition="Talker",
                    ),
                ],
            ),
        ])

        prefill_vision = Sequential([
            GraphNode(
                name="vision_encoder",
                # image_grid_thw / video_grid_thw carries the (T, H, W) grid
                # dimensions per image/video, used by the encoder to compute
                # spatial position IDs and patch counts.
                input_names=["pixel_values", "image_grid_thw"],
                outputs=[
                    GraphEdge(next_node="Thinker", name="vision_embeds"),
                    GraphEdge(next_node="Thinker", name="deepstack")
                ],
            ),
            GraphNode(
                name="Thinker",
                input_names=["vision_embeds", "deepstack", "video_second_per_grid", "image_grid_thw"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        persist=True,
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_states",
                        target_partition="Talker",
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_mask",
                        target_partition="Talker",
                    ),
                ],
            ),
        ])

        # Chunked-vision variant (MSTAR_CHUNKED_PREFILL_V2_VISION): split the
        # Sequential above into a standalone encoder walk (encode_vision, which
        # persists vision_embeds + deepstack so the conductor can read the
        # vision token count from vision_embeds.dims and re-emit the Thinker
        # walk in chunks) followed by a Thinker-only prefill_vision walk that
        # consumes them from persist (image_grid_thw rides on the Thinker
        # schedule entry). Mirrors the audio encoder-split. Only
        # registered when the flag is ON, so flag-off keeps the Sequential above
        # byte-identical.
        encode_vision = GraphNode(
            name="vision_encoder",
            input_names=["pixel_values", "image_grid_thw"],
            outputs=[
                GraphEdge(
                    next_node=EMPTY_DESTINATION, name="vision_embeds", persist=True,
                ),
                GraphEdge(
                    next_node=EMPTY_DESTINATION, name="deepstack", persist=True,
                ),
            ],
        )
        prefill_vision_chunked = GraphNode(
            name="Thinker",
            input_names=["vision_embeds", "deepstack", "video_second_per_grid", "image_grid_thw"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="new_token",
                    output_modality="text",
                    persist=True,
                ),
                StreamingGraphEdge(
                    next_node="Talker",
                    name="thinker_states",
                    target_partition="Talker",
                ),
                StreamingGraphEdge(
                    next_node="Talker",
                    name="thinker_mask",
                    target_partition="Talker",
                ),
            ],
        )

        # Merged multimodal prefill (MSTAR_MERGED_PREFILL): one walk that runs
        # the text span AND the vision span in a single Thinker forward, dropping
        # the conductor round-trip between prefill_text and prefill_vision.
        # Structurally identical to the prefill_vision Sequential (encoder must
        # still run first), but the Thinker node ALSO declares text_inputs; its
        # prepare_inputs concatenates the per-span embeds/pos_ids/deepstack in
        # modality order (see submodules.py). Registered only when the flag is on
        # so flag-off keeps every walk above byte-identical.
        prefill_multimodal = Sequential([
            GraphNode(
                name="vision_encoder",
                input_names=["pixel_values", "image_grid_thw"],
                outputs=[
                    GraphEdge(next_node="Thinker", name="vision_embeds"),
                    GraphEdge(next_node="Thinker", name="deepstack"),
                ],
            ),
            GraphNode(
                name="Thinker",
                input_names=[
                    "text_inputs", "vision_embeds", "deepstack",
                    "video_second_per_grid", "image_grid_thw",
                ],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        persist=True,
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_states",
                        target_partition="Talker",
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_mask",
                        target_partition="Talker",
                    ),
                ],
            ),
        ])

        # Merged multimodal prefill, AUDIO twin (MSTAR_MERGED_PREFILL_AUDIO): one
        # walk that runs the text span AND the audio span in a single Thinker
        # forward, dropping the conductor round-trip between prefill_text and
        # prefill_audio. Structurally identical to the prefill_audio Sequential
        # (audio encoder must still run first), but the Thinker node ALSO declares
        # text_inputs; its prepare_inputs concatenates the per-span
        # embeds/pos_ids in modality order (see submodules.py). No deepstack, no
        # mrope_pos_advance side-channel (audio positions are +1/token). Registered
        # only when the flag is on so flag-off keeps every walk above
        # byte-identical.
        prefill_multimodal_audio = Sequential([
            GraphNode(
                name="audio_encoder",
                input_names=["audio_features", "audio_seqlens"],
                outputs=[GraphEdge(next_node="Thinker", name="audio_embeds")],
            ),
            GraphNode(
                name="Thinker",
                # text_inputs_suffix is only supplied by the interleaved
                # vLLM-layout s2t merge ([prefix, audio, suffix]); the 2-entry
                # layout emits it with an empty payload (declared => must arrive).
                input_names=["text_inputs", "text_inputs_suffix", "audio_embeds"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        persist=True,
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_states",
                        target_partition="Talker",
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_mask",
                        target_partition="Talker",
                    ),
                ],
            ),
        ])

        # -- Thinker decode: produces new_token (persist) + thinker_states
        #    (streaming to Talker) --
        thinker_decode = Loop(
            name="thinker_decode_loop",
            section=GraphNode(
                name="Thinker",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                    ),
                    GraphEdge(
                        next_node="Thinker",
                        name="text_inputs",
                        output_modality="text",
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_states",
                        target_partition="Talker",
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_mask",
                        target_partition="Talker",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        # -- Talker prefill: receives thinker_states + talker_trigger --
        # Dual-input gating: both thinker_states from streaming and
        # talker_trigger from conductor cross-partition trigger must be
        # present for a prefill step.
        talker_prefill = GraphNode(
            name="Talker",
            input_names=["thinker_states", "thinker_mask", "talker_trigger"],
            outputs=[],
        )

        talker_last_prefill = Sequential(
            sections=[
                GraphNode(
                    name="Talker",
                    input_names=["thinker_states", "thinker_mask", "talker_trigger"],
                    outputs=[
                        GraphEdge(
                            next_node=EMPTY_DESTINATION,
                            name="talker_input_embeds",
                            persist=True
                        ),
                        StreamingGraphEdge(
                            next_node="Code2Wav",
                            name="codec_tokens",
                            target_partition="Code2Wav",
                        ),
                    ]
                )
            ]
        )

        # -- Talker decode: autoregressive codec token generation --
        talker_decode = Loop(
            name="talker_decode_loop",
            section=Sequential(
                sections=[
                    GraphNode(
                        name="Talker",
                        input_names=["thinker_states", "thinker_mask", "talker_input_embeds"],
                        outputs=[
                            GraphEdge(
                                next_node="Talker",
                                name="talker_input_embeds",
                            ),
                            StreamingGraphEdge(
                                next_node="Code2Wav",
                                name="codec_tokens",
                                target_partition="Code2Wav",
                            ),
                        ],
                    )
                ]
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        # -- Code2Wav chunk: vocoder streaming decode --
        code2wav_chunk = GraphNode(
            name="Code2Wav",
            input_names=["codec_tokens"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="audio_chunk",
                    output_modality="audio",
                ),
            ],
        )

        walks = {
            "prefill_text": prefill_text,
            "prefill_audio": prefill_audio,
            "prefill_vision": prefill_vision,
            "thinker_decode": thinker_decode,
            "talker_prefill": talker_prefill,
            "talker_last_prefill": talker_last_prefill,
            "talker_decode": talker_decode,
            "code2wav_chunk": code2wav_chunk,
        }
        if chunked_prefill_v2_vision_enabled():
            # Replace the Sequential prefill_vision with the encoder-split pair
            # so the conductor can chunk the Thinker portion. flag-off keeps the
            # Sequential registered above.
            walks["encode_vision"] = encode_vision
            walks["prefill_vision"] = prefill_vision_chunked
        if merged_prefill_enabled():
            # Merged text+vision walk. Only registered under the flag so flag-off
            # leaves the walk table byte-identical.
            walks["prefill_multimodal"] = prefill_multimodal
        if merged_prefill_audio_enabled():
            # Merged text+audio walk. Only registered under the flag so flag-off
            # leaves the walk table byte-identical.
            walks["prefill_multimodal_audio"] = prefill_multimodal_audio
        return walks

    # -----------------------------------------------------------------------
    # Partition API: 3-partition streaming topology
    # -----------------------------------------------------------------------

    def get_partitions(self) -> list[PartitionDefinition]:
        return [
            PartitionDefinition(
                name="Thinker",
                graph_walks={
                    "prefill_text", "prefill_audio",
                    "prefill_vision", "thinker_decode",
                    # encode_vision is registered as a Thinker-partition walk
                    # only when chunked vision is on; harmless to always list.
                    *(("encode_vision",) if chunked_prefill_v2_vision_enabled() else ()),
                    # prefill_multimodal only exists under MSTAR_MERGED_PREFILL.
                    *(("prefill_multimodal",) if merged_prefill_enabled() else ()),
                    # prefill_multimodal_audio only exists under
                    # MSTAR_MERGED_PREFILL_AUDIO.
                    *(("prefill_multimodal_audio",)
                      if merged_prefill_audio_enabled() else ()),
                },
                initial_walk="prefill_text",
                producer_partitions=[],
            ),
            PartitionDefinition(
                name="Talker",
                graph_walks={"talker_prefill", "talker_last_prefill", "talker_decode"},
                initial_walk="talker_prefill",
                producer_partitions=["Thinker"],
            ),
            PartitionDefinition(
                name="Code2Wav",
                graph_walks={"code2wav_chunk"},
                initial_walk="code2wav_chunk",
                producer_partitions=["Talker"],
            ),
        ]

    def get_partition_topology(self) -> PartitionTopology:
        return PartitionTopology(
            partitions=["Thinker", "Talker", "Code2Wav"],
            connections=[
                Connection(
                    from_partition="Thinker",
                    to_partition="Talker",
                    edge_name="thinker_states",
                    chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1, continue_after_done=True),
                ),
                Connection(
                    from_partition="Thinker",
                    to_partition="Talker",
                    edge_name="thinker_mask",
                    chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1, continue_after_done=True),
                ),
                Connection(
                    from_partition="Talker",
                    to_partition="Code2Wav",
                    edge_name="codec_tokens",
                    chunk_policy_factory=lambda: LeftContextChunkPolicy(
                        chunk=self.config.code2wav.codec_chunk_frames,
                        left_context=self.config.code2wav.codec_left_context_frames,
                    ),
                ),
            ],
        )

    # -----------------------------------------------------------------------
    # Model ABC: sampling config
    # -----------------------------------------------------------------------
    def get_sampling_config(
        self, node_name: str,
        model_kwargs: dict | None = None,
    )  -> SamplingConfig | None:
        if model_kwargs is None:
            model_kwargs = {}

        if node_name == "Thinker":
            temperature = model_kwargs.get("thinker_temperature", 0.7)
            top_p = model_kwargs.get("thinker_top_p", 0.9)
            # only apply ignore_eos to the thinker
            ignore_eos = model_kwargs.get("ignore_eos", False)
            return SamplingConfig(
                vocab_size=self.config.thinker_text.vocab_size,
                temperature=temperature, top_p=top_p,
                ignore_eos=ignore_eos
            )
        if node_name == "Talker":
            temperature = model_kwargs.get("talker_temperature", 0.9)
            top_k = model_kwargs.get("talker_top_k", 50)
            top_p = model_kwargs.get("talker_top_p", 1.0)
            repetition_penalty = model_kwargs.get("talker_repetition_penalty", 1.05)
            return SamplingConfig(
                vocab_size=self.config.talker_text.vocab_size,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty
            )
        # fallback to default config
        return SamplingConfig()

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        # Qwen3-Omni's Code2Wav vocoder emits speech at 24 kHz.
        return 24000

    # -----------------------------------------------------------------------
    # Model ABC: initial forward pass args
    # -----------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        audio_output = "audio" in output_modalities

        if model_kwargs is None:
            model_kwargs = {}

        if partition_name == "Thinker":
            return self._get_thinker_initial_args(
                input_modalities, output_modalities,
                input_signals, model_kwargs or {},
            )
        elif partition_name == "Talker":
            # Talker starts in prefill mode
            full_metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="talker_prefill",
                is_prefill=True,
                kwargs={
                    "audio_output": audio_output,
                    "talker_prefill_done": False,
                    # The Talker consumes one thinker_states chunk per Thinker
                    # prefill walk, so this MUST equal the actual number of
                    # Thinker prefill walks -- not len(input_modalities).  The
                    # vLLM-layout path (MSTAR_VLLM_PROMPT_LAYOUT=1) splits text
                    # into prefix+suffix around the audio, producing an extra
                    # prefill_text walk; deriving the count from the real
                    # schedule keeps the Talker's last-prefill detection aligned.
                    # Count only walks that reach the Thinker (produce
                    # thinker_states). encode_vision (chunked-vision encoder
                    # split) is encoder-only and must be excluded so the
                    # Talker's last-prefill detection stays aligned.
                    "num_thinker_prefill_steps": sum(
                        1 for walk, _ in self._build_thinker_prefill_schedule(
                            input_modalities, input_signals,
                        ) if walk != "encode_vision"
                    ),
                    "prefill_chunks_processed": 0,
                    "voice": model_kwargs.get("voice", "Ethan"),
                    "talker_max_tokens": self.get_max_talker_output_tokens(**model_kwargs),
                },
            )
            return ForwardPassArgs(
                full_metadata=full_metadata,
                inputs=[GraphEdge(next_node="Talker", name="talker_trigger")] if audio_output else [],
                unpersist_tensors=[],
                request_done="audio" not in output_modalities,
                step_metadata={
                    "voice": model_kwargs.get("voice", "Ethan"),
                    "talker_max_tokens": full_metadata.kwargs.get("talker_max_tokens")
                }
            )
        elif partition_name == "Code2Wav":
            # Code2Wav starts with code2wav_chunk walk but no inputs --
            # it self-triggers via StreamBuffer when codec tokens arrive.
            full_metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="code2wav_chunk",
                is_prefill=False,
            )
            return ForwardPassArgs(
                full_metadata=full_metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done="audio" not in output_modalities,
            )
        raise ValueError(f"Unknown partition: {partition_name!r}")

    def _get_thinker_initial_args(
        self,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> ForwardPassArgs:
        """Build initial ForwardPassArgs for the Thinker partition.

        Constructs a prefill schedule from the input modalities, then
        begins the first walk in that schedule (always prefill_text).
        """
        audio_output = "audio" in output_modalities

        # Build prefill schedule: list of (graph_walk_name, tensor_info)
        schedule = self._build_thinker_prefill_schedule(
            input_modalities, input_signals,
        )

        # Merged multimodal prefill (MSTAR_MERGED_PREFILL): collapse an exact
        # [prefill_text, prefill_vision] schedule (either order) into ONE
        # prefill_multimodal walk. merged_vision_first records the span order so
        # the submodule concatenates in the right order. Returns the schedule
        # unchanged (merged_vision_first=None) when the merge does not apply, so
        # every non-eligible request keeps the byte-identical multi-walk path.
        schedule, merged_vision_first, merged_audio_order = (
            self._maybe_merge_prefill_schedule(
                schedule, audio_output, model_kwargs.get("_live_occupancy")
            )
        )

        first_walk = schedule[0][0] if schedule else "thinker_decode"
        is_last_prefill = bool(schedule and len(schedule) == 1)

        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=first_walk,
            is_prefill=bool(schedule),
            kwargs={
                "prefill_schedule": schedule,
                "prefill_step": 0,
                "audio_output": audio_output,
                # Per-walk committed-token counter for resumable chunked prefill
                # (MSTAR_CHUNKED_PREFILL_V2). 0 = start of the walk's span.
                "prefill_chunk_offset": 0,
            },
        )

        # First walk inputs
        inputs = self._get_thinker_prefill_inputs(full_metadata, input_signals)

        # Resumable chunked prefill: if the first walk is a long prefill_text
        # span and chunking applies, emit chunk 0 here and keep the input
        # tensor alive until the walk's final chunk. Absent metadata (flag off
        # or span short enough) => single full-span step, byte-identical.
        chunk_step_metadata: dict = {}
        walk_done = True
        if schedule:
            bounds = self._chunk_bounds(full_metadata, schedule, 0, input_signals)
            if bounds is not None:
                offset, chunk_len, walk_done = bounds
                chunk_step_metadata = {
                    "prefill_chunk_offset": offset,
                    "prefill_chunk_len": chunk_len,
                }
                # Only the final chunk of the final walk samples the first token.
                is_last_prefill = is_last_prefill and walk_done
                self._log_first_chunk(full_metadata, first_walk, offset, chunk_len)

        if walk_done:
            unpersist_tensors = sum(
                [inp.tensor_info for inp in inputs], start=[]
            )
        else:
            # Hold the prefill input tensor alive across chunks: release it only
            # on the chunk that finishes the walk.
            unpersist_tensors = []

        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={
                "is_prefill": True,
                # Tell the Thinker whether to emit thinker_states.  Text only
                # requests skip it to save cross-partition bandwidth.
                "audio_output": audio_output,
                "is_last_prefill": is_last_prefill,
                # Span order for a merged prefill_multimodal walk (None otherwise;
                # the submodule only reads it for that walk).
                "merged_vision_first": merged_vision_first,
                # Span order for a merged prefill_multimodal_audio walk: one of
                # "audio_first" / "text_first" / "interleaved" (None otherwise;
                # the submodule only reads it for that walk).
                "merged_audio_order": merged_audio_order,
                # prefill_chunk_offset / prefill_chunk_len for the submodule to
                # slice this chunk (absent => full span, byte-identical).
                **chunk_step_metadata,
            },
        )

    def _maybe_merge_prefill_schedule(
        self,
        schedule: list[tuple[str, dict[str, TensorPointerInfo]]],
        audio_output: bool,
        live_occupancy: int | None = None,
    ) -> tuple[
        list[tuple[str, dict[str, TensorPointerInfo]]], bool | None, str | None
    ]:
        """Collapse an exact one-text + one-{vision,audio} schedule (or the
        interleaved text/audio/text s2t schedule) into a single merged-prefill
        entry when the corresponding flag applies.

        Returns ``(schedule, merged_vision_first, merged_audio_order)``.
        ``merged_vision_first`` is ``True``/``False`` when the vision merge fired
        (span order) else ``None``. ``merged_audio_order`` is
        ``"audio_first"`` / ``"text_first"`` (2-entry) or ``"interleaved"``
        (3-entry text/audio/text) when the audio merge fired, else ``None``. When
        neither merge applies the schedule is returned UNCHANGED so the request
        keeps the byte-identical multi-walk path.

        Vision eligibility (all required): MSTAR_MERGED_PREFILL on; text output
        (no Talker); non-chunked vision (``MSTAR_CHUNKED_PREFILL_V2_VISION`` off,
        else the schedule holds ``encode_vision`` + a chunkable Thinker walk, a
        different span model); exactly two entries, one ``prefill_text`` and one
        ``prefill_vision``.

        Audio eligibility (all required): MSTAR_MERGED_PREFILL_AUDIO on; text
        output (so s2t is eligible, s2s/i2s are not); AND either
          * exactly two entries, one ``prefill_text`` + one ``prefill_audio``
            (legacy layout, MSTAR_VLLM_PROMPT_LAYOUT off), OR
          * exactly three entries ``[prefill_text, prefill_audio, prefill_text]``
            in that order — the vLLM-layout s2t schedule (default, audio inside
            the user turn) where process_prompt split the text into prefix+suffix
            (see the schedule builder). The merged entry then carries the prefix
            under ``text_inputs`` and the suffix under ``text_inputs_suffix`` (the
            two prefill_text entries would otherwise collide on ``text_inputs``).
        There is no chunked-audio walk, so no chunked-prefill guard is needed.
        """
        if (
            merged_prefill_enabled()
            and not audio_output
            and not chunked_prefill_v2_vision_enabled()
            and len(schedule) == 2
        ):
            walks = [w for w, _ in schedule]
            if sorted(walks) == ["prefill_text", "prefill_vision"]:
                vision_first = walks[0] == "prefill_vision"
                # Union the two entries' tensor dicts (their keys are disjoint:
                # text_inputs vs pixel_values/image_grid_thw/video_second_per_grid).
                merged_entry: dict[str, TensorPointerInfo] = {}
                for _, tensor_dict in schedule:
                    merged_entry.update(tensor_dict)
                return [("prefill_multimodal", merged_entry)], vision_first, None

        if (
            merged_prefill_audio_enabled()
            and not audio_output
            and (
                live_occupancy is None
                or live_occupancy <= merged_prefill_audio_max_bs()
            )
        ):
            # Occupancy gate: merge only at low concurrency (big s2t B<=16 win);
            # above the ceiling the heavier merged prefill would stall the dense
            # decode wave (B32 regression). live_occupancy is None => merge
            # (preserves always-merge when occupancy isn't plumbed) — parity-safe:
            # both merged and unmerged walks are captured, so either choice is exact.
            walks = [w for w, _ in schedule]
            if len(schedule) == 2 and sorted(walks) == [
                "prefill_audio", "prefill_text"
            ]:
                order = "audio_first" if walks[0] == "prefill_audio" else "text_first"
                # Union the two entries' tensor dicts (their keys are disjoint:
                # text_inputs vs audio_features/audio_seqlens).
                merged_entry = {}
                for _, tensor_dict in schedule:
                    merged_entry.update(tensor_dict)
                return [("prefill_multimodal_audio", merged_entry)], None, order
            if len(schedule) == 3 and walks == [
                "prefill_text", "prefill_audio", "prefill_text"
            ]:
                # vLLM-layout s2t: [prefix-text, audio, suffix-text]. Rename the
                # suffix text so it does not collide with the prefix's
                # ``text_inputs`` key; audio keys are disjoint.
                merged_entry = dict(schedule[1][1])          # audio_features/seqlens
                merged_entry.update(schedule[0][1])          # text_inputs (prefix)
                merged_entry["text_inputs_suffix"] = schedule[2][1]["text_inputs"]
                return (
                    [("prefill_multimodal_audio", merged_entry)], None, "interleaved",
                )

        return schedule, None, None

    def _append_vision_schedule(self, schedule: list, entry: dict) -> None:
        """Append the vision prefill schedule entries.

        Flag off: one ``prefill_vision`` walk (the encoder+Thinker Sequential).
        Flag on (MSTAR_CHUNKED_PREFILL_V2_VISION): split into ``encode_vision``
        (encoder, persists vision_embeds+deepstack) then a Thinker-only
        ``prefill_vision`` that the conductor can chunk. The encoder entry keeps
        the feature tensors; the Thinker entry carries image_grid_thw (for
        get_rope_index_vision) and video_second_per_grid (vision_embeds /
        deepstack come from persist).
        """
        if not chunked_prefill_v2_vision_enabled():
            schedule.append(("prefill_vision", entry))
            return
        schedule.append(("encode_vision", entry))
        # The Thinker chunk walk keeps image_grid_thw (needed by
        # get_rope_index_vision) and video_second_per_grid; vision_embeds /
        # deepstack come from the encoder's persist output.
        thinker_entry: dict = {}
        if "image_grid_thw" in entry:
            thinker_entry["image_grid_thw"] = entry["image_grid_thw"]
        if "video_second_per_grid" in entry:
            thinker_entry["video_second_per_grid"] = entry["video_second_per_grid"]
        schedule.append(("prefill_vision", thinker_entry))

    def _build_thinker_prefill_schedule(
        self,
        input_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[tuple[str, dict[str, TensorPointerInfo]]]:
        """Build the sequential prefill schedule for the Thinker.

        Order: [prefill_text] + [prefill_audio if audio inputs] + [prefill_vision if vision inputs]

        Each schedule entry is ``(walk_name, {input_name: tensor_info})``,
        capturing all tensors needed by that step's first node.  For audio
        and vision walks, this includes auxiliary tensors like
        ``audio_seqlens`` and ``image_grid_thw`` that the encoder nodes
        require alongside the primary feature tensor.
        """
        schedule: list[tuple[str, dict[str, TensorPointerInfo]]] = []

        texts = input_signals.get("text_inputs", [])
        audio_features = input_signals.get("audio_features", [])
        audio_seqlens = input_signals.get("audio_seqlens", [])
        pixel_values = input_signals.get("pixel_values", [])
        image_grid_thws = input_signals.get("image_grid_thw", [])
        # video uses pixel_values_videos in HF; we accept both keys here
        pixel_values_videos = input_signals.get("pixel_values_videos", [])
        video_grid_thws = input_signals.get("video_grid_thw", [])
        video_second_per_grid = input_signals.get("video_second_per_grid", [])

        # --- vLLM prompt-layout schedule -----------------------------------
        # process_prompt split text_inputs into [prefix, suffix] and wants the
        # audio interleaved: prefill_text(prefix) -> prefill_audio ->
        # prefill_text(suffix).  This puts the audio block INSIDE the user turn
        # before the instruction, matching vLLM.  Only triggers when the flag
        # is on AND there are exactly the expected two text spans + audio.
        if (
            vllm_prompt_layout_enabled()
            and len(texts) >= 2
            and len(audio_features) >= 1
        ):
            audio_entry: dict[str, TensorPointerInfo] = {
                "audio_features": audio_features[0],
            }
            if len(audio_seqlens) >= 1:
                audio_entry["audio_seqlens"] = audio_seqlens[0]
            schedule.append(("prefill_text", {"text_inputs": texts[0]}))
            schedule.append(("prefill_audio", audio_entry))
            schedule.append(("prefill_text", {"text_inputs": texts[1]}))
            return schedule

        text_idx = audio_idx = vision_idx = video_idx = 0
        for mod in input_modalities:
            if mod == "text":
                if text_idx < len(texts):
                    schedule.append((
                        "prefill_text",
                        {"text_inputs": texts[text_idx]},
                    ))
                    text_idx += 1
            elif mod == "audio":
                if audio_idx < len(audio_features):
                    entry: dict[str, TensorPointerInfo] = {
                        "audio_features": audio_features[audio_idx],
                    }
                    if audio_idx < len(audio_seqlens):
                        entry["audio_seqlens"] = audio_seqlens[audio_idx]
                    schedule.append(("prefill_audio", entry))
                    audio_idx += 1
            elif mod == "image":
                if vision_idx < len(pixel_values):
                    entry = {"pixel_values": pixel_values[vision_idx]}
                    if vision_idx < len(image_grid_thws):
                        entry["image_grid_thw"] = image_grid_thws[vision_idx]
                    self._append_vision_schedule(schedule, entry)
                    vision_idx += 1
            elif mod == "video":
                # Video uses pixel_values_videos + video_grid_thw, but the
                # graph node still consumes them under the "pixel_values" /
                # "image_grid_thw" input names (the vision encoder is shared).
                if video_idx < len(pixel_values_videos):
                    entry = {"pixel_values": pixel_values_videos[video_idx]}
                    if video_idx < len(video_grid_thws):
                        entry["image_grid_thw"] = video_grid_thws[video_idx]
                    if video_idx < len(video_second_per_grid):
                        entry["video_second_per_grid"] = video_second_per_grid[video_idx]
                    self._append_vision_schedule(schedule, entry)
                    video_idx += 1

        # Robustness guard: an empty user prompt (e.g. greedy T2S/I2S/A2S with no
        # caption) arrives with templated text_inputs present but "text" absent from
        # input_modalities, so the loop above schedules no prefill_text -> empty
        # schedule (IndexError) and the Talker expects zero thinker_states. Prefill
        # every templated text_inputs regardless; a no-op for normal requests.
        while text_idx < len(texts):
            schedule.append(("prefill_text", {"text_inputs": texts[text_idx]}))
            text_idx += 1

        return schedule

    def _get_thinker_prefill_inputs(
        self,
        metadata: CurrentForwardConductorMetadata,
        input_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[GraphEdge]:
        """Construct input GraphEdges for the current Thinker prefill step.

        Each schedule entry maps an ``(walk_name, {input_name: tensor_info})``.
        We emit one GraphEdge per input so that auxiliary tensors like
        ``audio_seqlens`` and ``image_grid_thw`` reach the encoder node
        alongside the primary feature tensor.
        """
        schedule = metadata.kwargs["prefill_schedule"]
        step = metadata.kwargs["prefill_step"]
        walk_name, tensor_dict = schedule[step]

        # Chunked-vision (MSTAR_CHUNKED_PREFILL_V2_VISION): the encoder-split
        # replaces the prefill_vision Sequential with encode_vision (encoder,
        # inputs from tensor_dict) + a Thinker-only prefill_vision that reads
        # vision_embeds / deepstack / image_grid_thw from persist_signals.
        if chunked_prefill_v2_vision_enabled() and walk_name == "prefill_vision":
            edges: list[GraphEdge] = []
            # vision_embeds / deepstack come from the encode_vision persist.
            for name in ("vision_embeds", "deepstack"):
                infos = input_signals.get(name, [])
                if not infos:
                    continue
                edge = GraphEdge(next_node="Thinker", name=name)
                edge.tensor_info = list(infos)
                edges.append(edge)
            # image_grid_thw / video_second_per_grid ride on the Thinker entry's
            # tensor_dict (conductor-known inputs, not encoder outputs). ALWAYS
            # emit the edge — with empty tensor_info when absent (image
            # requests have no video_second_per_grid) — because the node
            # declares the name in input_names and readiness requires every
            # declared name to arrive (empty payload still marks it ready;
            # this mirrors the original Sequential walk's behavior).
            for key in ("image_grid_thw", "video_second_per_grid"):
                edge = GraphEdge(next_node="Thinker", name=key)
                edge.tensor_info = (
                    [tensor_dict[key]] if key in tensor_dict else []
                )
                edges.append(edge)
            return edges

        # Merged multimodal walk (MSTAR_MERGED_PREFILL): the encoder takes
        # pixel_values + image_grid_thw; the Thinker additionally takes the FULL
        # text_inputs span (plus image_grid_thw for get_rope_index_vision and
        # video_second_per_grid). vision_embeds / deepstack reach the Thinker as
        # the encoder's Sequential outputs (graph edges), not conductor inputs.
        if walk_name == "prefill_multimodal":
            edges = []
            for name in ("pixel_values", "image_grid_thw"):
                if name in tensor_dict:
                    edge = GraphEdge(next_node="vision_encoder", name=name)
                    edge.tensor_info = [tensor_dict[name]]
                    edges.append(edge)
            # ALWAYS emit each declared Thinker input (empty payload still marks
            # the name ready — image requests carry no video_second_per_grid),
            # mirroring the chunked-vision readiness contract above.
            for name in ("text_inputs", "image_grid_thw", "video_second_per_grid"):
                edge = GraphEdge(next_node="Thinker", name=name)
                edge.tensor_info = [tensor_dict[name]] if name in tensor_dict else []
                edges.append(edge)
            return edges

        # Merged multimodal AUDIO walk (MSTAR_MERGED_PREFILL_AUDIO): the encoder
        # takes audio_features + audio_seqlens; the Thinker additionally takes the
        # FULL text_inputs span. audio_embeds reaches the Thinker as the encoder's
        # Sequential output (a graph edge), not a conductor input.
        if walk_name == "prefill_multimodal_audio":
            edges = []
            for name in ("audio_features", "audio_seqlens"):
                if name in tensor_dict:
                    edge = GraphEdge(next_node="audio_encoder", name=name)
                    edge.tensor_info = [tensor_dict[name]]
                    edges.append(edge)
            # text_inputs (prefix, or the whole text span for the 2-entry layout)
            # + text_inputs_suffix (only the interleaved vLLM-layout carries it).
            # ALWAYS emit each declared Thinker input — an empty payload still
            # marks the declared name ready (the 2-entry layout has no suffix),
            # mirroring the vision-merge readiness contract.
            for name in ("text_inputs", "text_inputs_suffix"):
                edge = GraphEdge(next_node="Thinker", name=name)
                edge.tensor_info = (
                    [tensor_dict[name]] if name in tensor_dict else []
                )
                edges.append(edge)
            return edges

        # Determine the target node — for audio/vision, the first node in
        # the Sequential walk is the encoder (not the Thinker). encode_vision
        # (chunked path) is encoder-only, same routing as the Sequential head.
        if walk_name == "prefill_text":
            target_node = "Thinker"
        elif walk_name == "prefill_audio":
            target_node = "audio_encoder"
        elif walk_name in ("prefill_vision", "encode_vision"):
            target_node = "vision_encoder"
        else:
            raise ValueError(f"Unrecognized prefill walk: {walk_name}")

        edges = []
        for input_name, tensor_info in tensor_dict.items():
            if input_name == "video_second_per_grid":
                continue # goes directly to Thinker
            edge = GraphEdge(next_node=target_node, name=input_name)
            edge.tensor_info = [tensor_info]
            edges.append(edge)

        if walk_name == "encode_vision":
            # Encoder-only step: image_grid_thw already routed to the encoder
            # above (it's in tensor_dict and not video_second_per_grid); the
            # encoder persists it for the following Thinker chunk walk. No
            # Thinker edges from this step.
            return edges

        if walk_name == "prefill_vision":
            for key in ["image_grid_thw", "video_second_per_grid"]:
                edge = GraphEdge(next_node="Thinker", name=key)
                if key in tensor_dict:
                    edge.tensor_info = [tensor_dict[key]]
                edges.append(edge)
        return edges

    # -----------------------------------------------------------------------
    # Resumable chunked prefill (MSTAR_CHUNKED_PREFILL_V2)
    # -----------------------------------------------------------------------

    def _text_chunk_bounds(
        self,
        metadata: "CurrentForwardConductorMetadata",
        schedule: list,
        step: int,
    ) -> tuple[int, int, bool] | None:
        """Resolve this step's chunk window for resumable chunked TEXT prefill.

        Returns ``(offset, chunk_len, walk_done)`` when the current prefill walk
        is a ``prefill_text`` span that should be streamed in chunks, or ``None``
        to fall back to the single-shot (byte-identical) path.

        Chunking applies ONLY when:
          * ``MSTAR_CHUNKED_PREFILL_V2`` is ON, and
          * the walk is ``prefill_text`` (its token count is known up-front from
            the input tensor's ``dims``; vision length is only known after the
            encoder runs and is handled by ``_vision_chunk_bounds`` via the
            encoder-split walk, gated under ``MSTAR_CHUNKED_PREFILL_V2_VISION``), and
          * ``audio_output`` is False, i.e. the Talker is NOT conditioned. When
            the Talker runs it consumes one ``thinker_states`` chunk per WALK;
            chunking a walk would emit one per token-chunk and drift the Talker's
            ``num_thinker_prefill_steps`` accounting. Text-output requests
            (S2T / I2T / T2T) skip thinker_states entirely, so chunking them is
            safe and exact.

        ``offset`` is the per-walk committed-token counter carried across steps
        in ``metadata.kwargs['prefill_chunk_offset']``.
        """
        if not chunked_prefill_v2_enabled():
            return None
        if metadata.kwargs.get("audio_output", True):
            return None
        walk, tensor_dict = schedule[step]
        if walk != "prefill_text":
            return None
        ti = tensor_dict.get("text_inputs")
        dims = getattr(ti, "dims", None)
        if not dims:
            return None
        span = int(dims[0])
        offset = int(metadata.kwargs.get("prefill_chunk_offset", 0))
        plan = plan_prefill_chunk(
            span, offset, prefill_chunk_tokens(),
            allow_single_chunk=(
                mixed_single_chunk_enabled() and mixed_batch_enabled()
            ),
        )
        if plan is None:
            return None
        chunk_len, walk_done = plan
        return offset, chunk_len, walk_done

    def _vision_chunk_bounds(
        self,
        metadata: "CurrentForwardConductorMetadata",
        schedule: list,
        step: int,
        persist_signals: dict[str, list["TensorPointerInfo"]] | None,
    ) -> tuple[int, int, bool] | None:
        """Resolve this step's chunk window for chunked VISION Thinker prefill.

        Requires the encoder-split (MSTAR_CHUNKED_PREFILL_V2_VISION): the walk
        is the Thinker-only ``prefill_vision`` and the vision token count comes
        from the persisted ``vision_embeds`` (the encode_vision output). The
        Thinker span is ``vision_len + 2`` (vision_bos / vision_eos sentinels,
        matching ThinkerSubmodule vision wrap). Same audio_output=False gate as
        text.
        """
        if not chunked_prefill_v2_vision_enabled():
            return None
        if metadata.kwargs.get("audio_output", True):
            return None
        walk = schedule[step][0]
        if walk != "prefill_vision":
            return None
        if persist_signals is None:
            return None
        infos = persist_signals.get("vision_embeds", [])
        if not infos:
            return None
        dims = getattr(infos[0], "dims", None)
        if not dims:
            return None
        span = int(dims[0]) + 2  # + vision_bos / vision_eos sentinels
        offset = int(metadata.kwargs.get("prefill_chunk_offset", 0))
        plan = plan_prefill_chunk(
            span, offset, prefill_chunk_tokens(),
            allow_single_chunk=(
                mixed_single_chunk_enabled() and mixed_batch_vision_enabled()
            ),
        )
        if plan is None:
            return None
        chunk_len, walk_done = plan
        return offset, chunk_len, walk_done

    def _chunk_bounds(
        self,
        metadata: "CurrentForwardConductorMetadata",
        schedule: list,
        step: int,
        persist_signals: dict[str, list["TensorPointerInfo"]] | None = None,
    ) -> tuple[int, int, bool] | None:
        """Unified chunk-bounds resolver: text first, then vision (encoder-split,
        gated MSTAR_CHUNKED_PREFILL_V2_VISION). Returns ``(offset, chunk_len,
        walk_done)`` or ``None``.
        """
        bounds = self._text_chunk_bounds(metadata, schedule, step)
        if bounds is not None:
            return bounds
        return self._vision_chunk_bounds(metadata, schedule, step, persist_signals)

    def _walk_span_tokens(
        self, schedule: list, step: int,
    ) -> int | None:
        """The full token count the Thinker sees for the walk at ``step``, or
        None if not statically known (encoder-output walks). Text-only walks
        are the only ones chunked prefill splits."""
        walk, tensor_dict = schedule[step]
        if walk != "prefill_text":
            return None
        ti = tensor_dict.get("text_inputs")
        dims = getattr(ti, "dims", None)
        if not dims:
            return None
        return int(dims[0])

    def _log_first_chunk(
        self, metadata, walk: str, offset: int, chunk_len: int,
    ) -> None:
        """DEBUG line on the first chunk of a walk: chunks planned, C, span."""
        if offset != 0 or not logger.isEnabledFor(logging.DEBUG):
            return
        schedule = metadata.kwargs["prefill_schedule"]
        step = metadata.kwargs["prefill_step"]
        span = self._walk_span_tokens(schedule, step)
        if span is None:
            return
        cap = prefill_chunk_tokens()
        # Number of chunks the planner will emit for this walk span.
        n, off = 0, 0
        while True:
            plan = plan_prefill_chunk(span, off, cap)
            if plan is None:
                n = 1
                break
            cl, done = plan
            n += 1
            off += cl
            if done:
                break
        logger.debug(
            "chunked_prefill_v2: walk=%s span=%d C<=%d chunks=%d (first=%d)",
            walk, span, cap, n, chunk_len,
        )

    def _assert_walk_span(
        self, metadata, schedule: list, step: int,
    ) -> None:
        """MSTAR_CHUNKED_PREFILL_V2_ASSERT: on leaving a chunked walk, verify the
        summed committed chunk tokens equal the walk's full span. The committed
        offset lives in ``prefill_chunk_offset`` and was advanced by exactly
        ``chunk_len`` per chunk; the final chunk (walk_done) did NOT advance it,
        so at walk exit offset == span - last_chunk_len. Recompute the expected
        total and compare. Cheap conductor-side check; no GPU state read."""
        if not chunked_prefill_v2_assert():
            return
        span = self._walk_span_tokens(schedule, step)
        if span is None:
            return
        # Replay the planner to get the committed-before-final-chunk offset.
        cap = prefill_chunk_tokens()
        off = 0
        while True:
            plan = plan_prefill_chunk(span, off, cap)
            if plan is None:
                # Not chunked: single full-span step.
                committed_before_final = 0
                break
            cl, done = plan
            if done:
                committed_before_final = off
                break
            off += cl
        actual_off = int(metadata.kwargs.get("prefill_chunk_offset", 0))
        assert actual_off == committed_before_final, (
            f"chunked_prefill_v2 span mismatch: walk step={step} span={span} "
            f"expected committed offset {committed_before_final} before final "
            f"chunk, got {actual_off}"
        )

    # -----------------------------------------------------------------------
    # Model ABC: partition forward pass args (STATE MACHINE)
    # -----------------------------------------------------------------------

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        if partition_name == "Thinker":
            return self._get_thinker_forward(
                partition_metadata, persist_signals,
            )
        elif partition_name == "Talker":
            return self._get_talker_forward(
                partition_metadata, persist_signals,
                incoming_connections,
            )
        elif partition_name == "Code2Wav":
            conn = incoming_connections[0] if incoming_connections else None
            return self._get_code2wav_forward(
                partition_metadata, conn,
            )
        raise ValueError(f"Unknown partition: {partition_name!r}")

    # -- Thinker state machine ---------------------------------------------

    def _get_thinker_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
    ) -> ForwardPassArgs:
        """Thinker partition state machine.

        1. Build prefill schedule: [prefill_text] + [prefill_audio] + [prefill_vision]
        2. Pop walks from schedule until done
        3. Transition to thinker_decode
        4. Each decode step: check new_token for EOS (im_end_token_id)
        5. On EOS: request_done=True for Thinker
        """

        if metadata.is_prefill:
            schedule = metadata.kwargs["prefill_schedule"]
            cur_step = metadata.kwargs["prefill_step"]

            # Resumable chunked prefill: if the chunk that just completed did
            # NOT consume the final token of the current walk's span, re-emit
            # the SAME walk next step with an advanced offset (do not advance
            # the schedule). Only once the walk's span is fully consumed do we
            # fall through to the normal schedule advance.
            rechunk = False
            bounds = self._chunk_bounds(metadata, schedule, cur_step, persist_signals)
            if bounds is not None:
                offset, chunk_len, walk_done = bounds
                if not walk_done:
                    metadata.kwargs["prefill_chunk_offset"] = offset + chunk_len
                    metadata.graph_walk = schedule[cur_step][0]
                    rechunk = True

            if not rechunk:
                # Leaving the current walk -> reset the per-walk chunk counter,
                # run the assert hook (summed chunk tokens == walk span), and
                # advance the schedule.
                self._assert_walk_span(metadata, schedule, cur_step)
                metadata.kwargs["prefill_chunk_offset"] = 0
                step = cur_step + 1
                if step < len(schedule):
                    # More prefill steps remaining
                    metadata.kwargs["prefill_step"] = step
                    metadata.graph_walk = schedule[step][0]
                else:
                    # All prefill done -- transition to thinker_decode
                    metadata.is_prefill = False
                    metadata.graph_walk = "thinker_decode"

        elif metadata.graph_walk == "thinker_decode":
            # if the decode loop returns to conductor, the thinker is fully done
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )


        if metadata.is_prefill:
            # Still in prefill -- delegate to _get_thinker_prefill_inputs
            # which handles the (walk_name, {input_name: tensor_info}) schedule
            # entry format and emits one GraphEdge per input (so auxiliary
            # tensors like image_grid_thw / audio_seqlens reach the encoder
            # alongside the primary feature tensor).
            schedule = metadata.kwargs["prefill_schedule"]
            step = metadata.kwargs["prefill_step"]
            is_last_prefill = (step == len(schedule) - 1)
            walk_done = True
            chunk_step_metadata: dict = {}
            inputs = self._get_thinker_prefill_inputs(metadata, persist_signals)

            # Resumable chunked prefill window for this step.
            bounds = self._chunk_bounds(metadata, schedule, step, persist_signals)
            if bounds is not None:
                offset, chunk_len, walk_done = bounds
                chunk_step_metadata = {
                    "prefill_chunk_offset": offset,
                    "prefill_chunk_len": chunk_len,
                }
                # Only the FINAL chunk of the FINAL walk is the true last
                # prefill (samples first-token logits + returns new_token once).
                is_last_prefill = is_last_prefill and walk_done
                if offset == 0:
                    self._log_first_chunk(
                        metadata, schedule[step][0], offset, chunk_len,
                    )
        else:
            # Decode: previous token feeds back as text_inputs
            is_last_prefill = False
            walk_done = True
            chunk_step_metadata = {}
            edge = GraphEdge(next_node="Thinker", name="text_inputs")
            edge.tensor_info = persist_signals.get("new_token", [])
            inputs = [edge]

        # Hold the prefill input tensor alive across chunks: release it only on
        # the chunk that finishes the walk (``walk_done``) or on non-chunked /
        # decode steps. Re-sending the same persisted pointer each chunk
        # accumulates the conductor ref-count, drained by the single unpersist
        # on the final chunk. ``walk_done`` is True for every non-chunked step.
        if not walk_done:
            unpersist_tensors = []
        else:
            unpersist_tensors = sum(
                [inp.tensor_info for inp in inputs], start=[]
            )

        step_metadata = {
            "is_prefill": metadata.is_prefill,
            "is_last_prefill": is_last_prefill,
            # Persist the audio_output flag across every Thinker step so
            # the submodule can gate thinker_states emission.  Default True
            # for backwards compatibility with callers that never set it.
            "audio_output": metadata.kwargs.get("audio_output", True),
            # prefill_chunk_offset / prefill_chunk_len for the submodule to
            # slice this chunk (absent => full span, byte-identical).
            **chunk_step_metadata,
        }

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=step_metadata,
        )

    # -- Talker state machine ----------------------------------------------

    def _get_talker_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Talker partition state machine.

        1. While prefill: return empty inputs (wait for cross-partition trigger)
           - When trigger arrives with is_last_prefill=False:
             extend KV cache only, no outputs
           - When trigger arrives with is_last_prefill=True:
             sample first codec token, produce all_codes
        2. After last prefill produces all_codes: transition to talker_decode
           - Set graph_walk="talker_decode", is_prefill=False
           - Return all_codes as input edge (conductor-driven)
        3. Each decode step: check all_codes for codec_eos
           - If codec_eos: request_done=True for Talker
           - Else: return all_codes as input again (loop)
        """
        if metadata.graph_walk == "talker_prefill":
            metadata.kwargs["prefill_chunks_processed"] += 1
            is_last_prefill = metadata.kwargs["num_thinker_prefill_steps"] == \
                 metadata.kwargs["prefill_chunks_processed"]
            metadata.graph_walk = "talker_last_prefill" if is_last_prefill else "talker_prefill"
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[GraphEdge(next_node="Talker", name="talker_trigger")],
                unpersist_tensors=[],
                step_metadata={
                    "is_prefill": True,
                    # voice is used for the last prefill
                    "voice": metadata.kwargs.get("voice", "Ethan"),
                    "talker_max_tokens": metadata.kwargs.get("talker_max_tokens")
                },
            )
        elif metadata.graph_walk == "talker_last_prefill":
            metadata.is_prefill = False
            metadata.graph_walk = "talker_decode"
            metadata.kwargs["talker_prefill_done"] = True

            # Feed talker_input_embeds back as input for first decode step
            edge = GraphEdge(next_node="Talker", name="talker_input_embeds")
            edge.tensor_info = persist_signals["talker_input_embeds"]
            inputs = [edge]
            unpersist_tensors = sum(
                [inp.tensor_info for inp in inputs], start=[]
            )

            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=inputs,
                unpersist_tensors=unpersist_tensors,
                step_metadata={
                    "is_prefill": False,
                    "talker_max_tokens": metadata.kwargs.get("talker_max_tokens")
                },
            )

        elif metadata.graph_walk == "talker_decode":
            # If the decode dynamic loop reaches the conductor, we can end the request.
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        raise ValueError(
            f"Talker in unexpected state: walk={metadata.graph_walk!r}, "
            f"is_prefill={metadata.is_prefill}"
        )

    # -- Code2Wav state machine --------------------------------------------

    def _get_code2wav_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        conn: StreamingConnectionState | None,
    ) -> ForwardPassArgs:
        """Code2Wav partition: streaming vocoder, self-triggered by StreamBuffer.

        Same pattern as Orpheus SNAC -- the conductor just tracks whether
        there are more codec tokens to process.
        """
        chunk_size = self.config.code2wav.codec_chunk_frames
        metadata.graph_walk = "code2wav_chunk"
        step_metadata = {"consumed_tokens": chunk_size}

        # Don't predict the last chunk from token counts: LeftContextChunkPolicy
        # emits an extra flush pass for the retained overlap, so a count-based
        # guess completes the request before that final chunk is emitted. The
        # `available <= 0` check above fires once consumption actually catches up.
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=[],
            unpersist_tensors=[],
            step_metadata=step_metadata,
            request_done=False,
        )

    # -----------------------------------------------------------------------
    # Model ABC: prompt processing
    # -----------------------------------------------------------------------

    def load_video(
        self, filepath: str, device: str
    ) -> TensorAndMetadata:
        # TODO: support audio in video
        from qwen_omni_utils.v2_5.vision_process import fetch_video
        video_input, video_sample_fps = fetch_video(
            {"video": filepath},
            return_video_sample_fps=True,
            image_patch_size=14,
            return_video_metadata=False
        )
        return TensorAndMetadata(
            data=video_input.to(device),
            metadata=dict(
                video_sample_fps=video_sample_fps
            )
        )

    def _user_turn_audio_split_index(
        self, input_ids: torch.Tensor
    ) -> int | None:
        """Index in ``input_ids`` right after ``<|im_start|>user\\n`` where the
        audio block must be inserted to match vLLM's layout.

        The Qwen ChatML user turn tokenizes as
        ``[<|im_start|>(151644), user(872), \\n(198), <prompt...>]``.  We locate
        the ``[im_start, user]`` pair and return the index just past the newline
        that follows it.  Returns None if no user turn is found.
        """
        im_start = self.config.im_start_token_id
        user_tok = self.config.user_token_id
        ids = input_ids.tolist()
        for i in range(len(ids) - 1):
            if ids[i] == im_start and ids[i + 1] == user_tok:
                j = i + 2
                # Skip the single newline token that the template emits after
                # the role name (id 198 for "\n"); guard against absence.
                if j < len(ids) and ids[j] == 198:
                    j += 1
                return j
        return None

    def _audio_mel_gpu(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU log-mel matching HF ``WhisperFeatureExtractor`` (MSTAR_GPU_MEL=1).

        The HF feature_extractor runs the STFT + mel filterbank + log on the CPU
        (numpy); for a 30 s clip that is ~tens of ms on the TTFT critical path and
        is amplified under host-CPU contention (the same poll-loop sensitivity that
        inflates M*'s TTFT). This computes the identical transform on the GPU.

        Returns ``(input_features (n_mel, T) float32 CPU, audio_seqlen (1,) long)``
        — byte-compatible with the HF path's per-audio output so everything
        downstream is unchanged. Numerically matches HF to cos>=0.9999 / max-abs
        ~1e-5 (test_qwen3_omni_gpu_mel_parity.py): same hann window (periodic),
        center+reflect STFT, power spectrogram, drop-last-frame, log10, max-8 clamp,
        (x+4)/4. ``T = floor(len/hop)`` == HF's valid (un-padded) frame count.
        """
        fe = self._processor.feature_extractor
        dev = waveform.device if waveform.is_cuda else torch.device("cuda")
        st = self._gpu_mel_state
        if st is None or st["dev"] != dev:
            import numpy as np
            st = {
                "dev": dev,
                "filters": torch.tensor(np.asarray(fe.mel_filters),
                                        dtype=torch.float32, device=dev),  # (n_freq, n_mel)
                "window": torch.hann_window(fe.n_fft, periodic=True, device=dev),
                "n_fft": fe.n_fft, "hop": fe.hop_length,
            }
            self._gpu_mel_state = st
        wav = waveform.to(dev, torch.float32)
        log = gpu_log_mel(wav, st["filters"], st["window"], st["n_fft"], st["hop"])
        feat = log.cpu()                                  # CPU float32 == HF contract
        seqlen = torch.tensor([feat.shape[1]], dtype=torch.long)
        return feat, seqlen

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        input_metadata: dict[str, dict] = {},
        **kwargs,
    ) -> NameToTensorList:
        """Build the full ChatML prompt + derived multimodal tensors.

        Uses HF's full ``AutoProcessor`` (combines tokenizer + image_processor
        + video_processor + feature_extractor + chat template) to:

        1. Build a ChatML-formatted prompt from ``prompt`` and any
           multimodal inputs in ``tensors``.
        2. Apply ``add_generation_prompt=True`` so the model receives the
           ``<|im_start|>assistant\\n`` suffix and knows to start the
           assistant response.
        3. Run the image_processor / feature_extractor on the raw modality
           tensors to produce ``pixel_values`` / ``image_grid_thw`` /
           ``audio_features`` / ``audio_seqlens``.
        4. Expand the single ``<|image_pad|>`` / ``<|audio_pad|>`` /
           ``<|video_pad|>`` placeholder in the tokenized text to N copies
           where N = number of patches after spatial merge (this is what
           ``Qwen3OmniMoeProcessor.replace_multimodal_special_tokens`` does
           internally).

        The result has ``text_inputs`` containing the FULL templated +
        expanded token IDs, plus the per-modality tensor outputs needed by
        the Thinker's prefill walks.
        """
        result: NameToTensorList = {}

        if tensors is None:
            tensors = {}

        # ----- Convert raw modality tensors to PIL/numpy form for HF -----
        raw_image_inputs = tensors.get("image_inputs", [])
        raw_audio_inputs = tensors.get("audio_inputs", [])
        raw_video_inputs = tensors.get("video_inputs", [])

        # When GPU image preprocessing is enabled we keep the raw GPU tensors
        # and never round-trip through CPU/numpy (see _gpu_image_preprocess).
        gpu_img_preprocess = _gpu_image_preprocess_enabled()

        pil_images: list = []
        if not gpu_img_preprocess:
            for img in raw_image_inputs:
                # data_worker.py provides images as (C, H, W) float32 in [0, 1]
                # on the GPU.  HF processors expect PIL/numpy uint8 (H, W, C)
                # in [0, 255] -- otherwise the default do_rescale=True double-
                # rescales and the model sees a near-zero (essentially black)
                # tensor regardless of the actual image content.
                if img.dtype.is_floating_point:
                    img_u8 = (img * 255.0).clamp(0, 255).to(torch.uint8)
                else:
                    img_u8 = img
                if img_u8.dim() == 3 and img_u8.shape[0] in (1, 3):
                    img_u8 = img_u8.permute(1, 2, 0)  # CHW -> HWC
                pil_images.append(img_u8.cpu().contiguous().numpy())

        # GPU log-mel is opt-in (MSTAR_GPU_MEL=1) and only when CUDA is present in
        # this worker; otherwise the raw audio is converted to numpy for the HF
        # (CPU) feature_extractor exactly as before. Default OFF = byte-identical.
        _use_gpu_mel = (
            _GPU_MEL and self._processor is not None and torch.cuda.is_available()
        )
        np_audios: list = []
        if not _use_gpu_mel:
            for waveform in raw_audio_inputs:
                np_audios.append(waveform.cpu().numpy())

        # ----- Preferred path: text-only chat template + separate modality processors -----
        #
        # We deliberately DO NOT include image/audio/video content blocks in
        # the messages list passed to apply_chat_template.  HF's chat template
        # would otherwise insert ``<|vision_start|><|image_pad|>...<|vision_end|>``
        # placeholders into text_inputs, which we don't want because:
        #
        #   1. Our prefill_vision / prefill_audio walks already wrap the
        #      modality content in their own start/end tokens before pushing
        #      it into the Thinker's KV cache.  Having the same wrapping in
        #      text_inputs would make the model see each modality twice
        #      (once as actual encoder embeddings via the modality walks,
        #      once as generic token embeddings via prefill_text), which is
        #      noise.
        #
        #   2. Unlike HF's single-shot prefill (which masked-scatter's the
        #      vision embeds INTO the placeholder positions in input_embeds),
        #      our multi-walk prefill builds up the same final KV cache via
        #      sequential walks.  The modality placeholders in text_inputs
        #      would never be replaced by real content in our flow.
        #
        # Functionally, both approaches end up with the same set of
        # embeddings in the KV cache (text + modality content).  Stripping
        # the placeholders avoids noise from the unfilled embeddings.
        system_text = (
            "You are Qwen, a virtual human developed by the "
            "Qwen team, Alibaba Group, capable of perceiving "
            "auditory and visual inputs, as well as generating "
            "text and speech."
        )
        messages = [
            {"role": "system", "content": system_text},
        ]
        # vLLM token parity: flatten_messages folds the system text into the prompt
        # blob, which we re-wrap in our own system turn -> double-counted system text.
        # Under MSTAR_VLLM_PROMPT_LAYOUT, strip the leading system-text copy so the
        # user turn is instruction-only and tokens match vLLM. OFF path untouched.
        user_prompt = prompt
        if vllm_prompt_layout_enabled() and prompt is not None:
            for sep in ("\n", ""):
                dup = system_text + sep
                if prompt.startswith(dup):
                    user_prompt = prompt[len(dup):]
                    break
        if user_prompt is not None:
            messages.append(
                {"role": "user", "content": user_prompt},
            )

        # apply_chat_template with TEXT-ONLY content -> no modality
        # placeholders are inserted.  add_generation_prompt=True
        # appends the trailing ``<|im_start|>assistant\n`` so the
        # model knows to start the assistant response.
        if self._processor is None:
            # __init__ sets _processor=None and warns if AutoProcessor fails to
            # load. Fail fast with a clear message instead of a cryptic
            # AttributeError on the first request (the old commented-out guard
            # promised a tokenizer fallback that was never wired up).
            raise RuntimeError(
                "Qwen3-Omni processor failed to load at init; cannot build the "
                "chat-template prompt. Check the checkpoint/processor files."
            )
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer(
            text, return_tensors="pt"
        )["input_ids"][0]

        # --- vLLM prompt-layout: audio INSIDE the user turn, BEFORE instr ---
        #
        # Legacy M* layout prefills audio as a separate bare block (schedule
        # [prefill_audio, prefill_text]) so the audio sits OUTSIDE any turn and
        # the instruction governs -> the model TRANSCRIBES.  vLLM-Omni applies
        # the stock HF chat template which puts the audio inside the user turn
        # before the instruction -> the trained "spoken-query -> reply" layout
        # -> the model ANSWERS.
        #
        # To replicate that without retokenizing across the boundary (which can
        # shift BPE merges), we slice the ALREADY-tokenized full sequence right
        # after ``<|im_start|>user\n`` into a prefix (system turn + user-turn
        # opener) and a suffix (instruction + ``<|im_end|>`` + assistant
        # prompt).  The schedule builder then runs
        # [prefill_text(prefix), prefill_audio, prefill_text(suffix)], so the
        # audio walk's BOS/AUDIO/EOS embeddings land between them -> exactly the
        # vLLM token layout (modulo sentinel IDs + M-RoPE, tracked separately).
        if (
            vllm_prompt_layout_enabled()
            and len(np_audios) > 0
            and prompt is not None
        ):
            split = self._user_turn_audio_split_index(input_ids)
            if split is not None:
                prefix_ids = input_ids[:split]
                suffix_ids = input_ids[split:]
                result["text_inputs"] = [prefix_ids, suffix_ids]
            else:
                logger.warning(
                    "MSTAR_VLLM_PROMPT_LAYOUT=1 but could not locate the user "
                    "turn in the tokenized prompt; falling back to legacy "
                    "layout for this request."
                )
                result["text_inputs"] = [input_ids]
        else:
            result["text_inputs"] = [input_ids]

        result["pixel_values"] = []
        result["image_grid_thw"] = []
        result["audio_seqlens"] = []
        result["audio_features"] = []
        result["video_second_per_grid"] = []
        result["video_grid_thw"] = []
        result["pixel_values_videos"] = []

        # Run image_processor / feature_extractor SEPARATELY for the
        # modality outputs.  These don't touch text_inputs.
        if gpu_img_preprocess:
            # GPU path: process each image fully on-device (no CPU round-trip).
            img_proc = self._processor.image_processor
            for img in raw_image_inputs:
                pv, grid_thw = _gpu_image_preprocess(
                    img,
                    patch_size=img_proc.patch_size,
                    temporal_patch_size=img_proc.temporal_patch_size,
                    merge_size=img_proc.merge_size,
                    min_pixels=img_proc.size["shortest_edge"],
                    max_pixels=img_proc.size["longest_edge"],
                    image_mean=img_proc.image_mean,
                    image_std=img_proc.image_std,
                )
                result["pixel_values"].append(pv)
                result["image_grid_thw"] += list(grid_thw)
        else:
            for img in pil_images:
                img_proc = self._processor.image_processor
                img_out = img_proc(images=[img], return_tensors="pt")
                result["pixel_values"].append(img_out["pixel_values"])
                result["image_grid_thw"] += img_out["image_grid_thw"]

        if _use_gpu_mel:
            for waveform in raw_audio_inputs:
                feat, seqlen = self._audio_mel_gpu(waveform)   # (n_mel, T), (1,)
                result["audio_seqlens"].append(seqlen)
                result["audio_features"].append(feat)
        else:
            for audio in np_audios:
                feat_extractor = self._processor.feature_extractor
                sr = getattr(feat_extractor, "sampling_rate", 16000)
                aud_out = feat_extractor(
                    audio, sampling_rate=sr,
                    padding=True,
                    truncation=False,
                    return_attention_mask=True,
                    return_tensors="pt"
                )
                aud_out["input_features"] = (
                    aud_out["input_features"]
                    .permute(0, 2, 1)[aud_out["attention_mask"].bool()]
                    .permute(1, 0)
                )
                result["audio_seqlens"].append(
                    aud_out["attention_mask"].sum(-1).to(torch.long)
                )
                result["audio_features"].append(
                    aud_out["input_features"]
                )

        # Video uses the video_processor; left as TODO since our
        # prefill_vision walk doesn't yet handle video frame stacks.
        for video, meta in zip(raw_video_inputs, input_metadata.get("video_inputs", []), strict=True):
            fps = meta.get(
                "video_sample_fps", 2.0
            )
            vid_out = self._processor.video_processor(
                videos=video,
                size={
                    "shortest_edge": 128 * 32 * 32,
                    "longest_edge": 768 * 32 * 32,
                }
            )
            result["video_second_per_grid"].append(
                torch.tensor([self._processor.video_processor.temporal_patch_size / fps])
            )
            result["video_grid_thw"] += vid_out["video_grid_thw"]
            result["pixel_values_videos"].append(vid_out["pixel_values_videos"])

        return result

    # -----------------------------------------------------------------------
    # Model ABC: postprocess
    # -----------------------------------------------------------------------

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
        request_kwargs: dict | None = None,
    ) -> bytes:
        if modality == "text":
            detok = self.tokenizer.decode(output)
            return detok.encode("utf-8")
        elif modality == "audio":
            if output.numel() == 0:
                return b""
            return output.cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for Qwen3-Omni: {modality!r}")

    # -----------------------------------------------------------------------
    # Model ABC: sharding
    # -----------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        # Talker LLM (attention + MoE-with-shared-expert) is TP-capable
        # via the same ``ParallelAttention`` / ``ParallelSparseMoeBlock*``
        # parts as the Thinker. The internal CodePredictor is intentionally
        # left at TP=1 (replicated weights, deterministic sampler) — see
        # ``_create_talker_submodule``. ``shard_dim`` stays empty because
        # every cross-edge signal (``thinker_states``, ``thinker_mask``,
        # ``codec_tokens``, ``talker_input_embeds``, ``new_token``) is
        # already replicated by the upstream all-reduce or sampler
        # broadcast before it leaves its producing node.
        return ShardingConfig(
            groups=[], tp_enabled_nodes={"Thinker", "Talker"}, shard_dim={},
        )

    # -----------------------------------------------------------------------
    # Model ABC: submodule loading
    # -----------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(
            node_name, device, tp_group=tp_group, autocast_dtype=autocast_dtype,
        )
        logger.info("Successfully loaded Qwen3-Omni submodule for %s", node_name)
        self._submodule_cache[node_name] = submodule

        # W3: If the Thinker was just loaded and the Talker already exists
        # (but TTS embeds were not initialized because Thinker wasn't
        # available at Talker creation time), initialize them now.
        if node_name == "Thinker":
            talker_sub = self._submodule_cache.get("Talker")
            if (
                talker_sub is not None
                and hasattr(talker_sub, '_tts_pad_embed_cached')
                and talker_sub._tts_pad_embed_cached is None
                and hasattr(submodule, 'model')
            ):
                try:
                    talker_sub.init_tts_embeds(submodule.model.embed_tokens)
                except Exception as e:
                    logger.warning(
                        "Deferred TTS embed init failed: %s", e,
                    )

        return submodule

    def _create_submodule(
        self, node_name: str, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name == "Thinker":
            return self._create_thinker_submodule(
                device, tp_group=tp_group, autocast_dtype=autocast_dtype,
            )
        elif node_name == "Talker":
            return self._create_talker_submodule(
                device, tp_group=tp_group, autocast_dtype=autocast_dtype,
            )
        elif node_name == "Code2Wav":
            return self._create_code2wav_submodule(device)
        elif node_name == "audio_encoder":
            return self._create_audio_encoder_submodule(device)
        elif node_name == "vision_encoder":
            return self._create_vision_encoder_submodule(device)
        return None

    @staticmethod
    def _thinker_remap(name: str) -> str | None:
        """Map HF checkpoint keys (after ``thinker.`` prefix strip) to model param paths.

        Handles the ``block_sparse_moe`` → ``mlp`` rename and the per-expert
        weight fusion: ``experts.{N}.{gate,up,down}_proj.weight`` becomes a
        shard_id-carrying key that the MoE weight_loaders consume via
        ``StackedParamRule``.
        """
        import re

        if "rotary_emb" in name:
            return None
        name = name.replace("block_sparse_moe.", "mlp.")
        # Per-expert weights: experts.N.{gate,up,down}_proj.weight
        # → experts.{gate_up_proj,down_proj} with shard_id handled by stacked rules.
        # We rewrite the name so StackedParamRule suffix matching works:
        # "experts.N.gate_proj.weight" → "experts.gate_proj.__N__.weight"
        # The weight_loader on the fused param extracts the expert index from shard_id.
        m = re.match(r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", name)
        if m:
            prefix, expert_idx, proj = m.groups()
            return f"{prefix}.experts.{proj}.__expert{expert_idx}__.weight"
        return name

    # MoE stacked param rules: route per-expert projections into fused params.
    # The shard_id encodes both the projection type AND the expert index.
    # The __expertN__ marker is injected by _thinker_remap; weight_loaders
    # parse it to determine the expert slot.
    _THINKER_STACKED_PARAMS: list = []  # populated lazily below

    def _get_thinker_stacked_params(self):
        from mstar.model.loader.base import StackedParamRule

        if self._THINKER_STACKED_PARAMS:
            return self._THINKER_STACKED_PARAMS
        E = self.config.thinker_text.num_experts
        # MoE expert rules MUST come before dense MLP rules because
        # the dense ".gate_proj" suffix would also match the remapped
        # MoE key "experts.gate_proj.__expertN__.weight". _apply_stacked
        # returns on first match, so longer/more-specific rules go first.
        rules = []
        for i in range(E):
            # source_suffix includes ".weight" so the replacement strips it —
            # the target params (experts.gate_up_proj, experts.down_proj) are
            # bare nn.Parameters, not Linear submodules, so they have no
            # ".weight" suffix in named_parameters().
            rules.append(StackedParamRule(
                target_suffix=".experts.gate_up_proj",
                source_suffix=f".experts.gate_proj.__expert{i}__.weight",
                shard_id=f"gate:{i}",
            ))
            rules.append(StackedParamRule(
                target_suffix=".experts.gate_up_proj",
                source_suffix=f".experts.up_proj.__expert{i}__.weight",
                shard_id=f"up:{i}",
            ))
            rules.append(StackedParamRule(
                target_suffix=".experts.down_proj",
                source_suffix=f".experts.down_proj.__expert{i}__.weight",
                shard_id=f"down:{i}",
            ))
        # Dense MLP gate/up fusion and attention qkv fusion.
        rules.append(StackedParamRule(".gate_up_proj", ".gate_proj", 0))
        rules.append(StackedParamRule(".gate_up_proj", ".up_proj", 1))
        rules.append(StackedParamRule(".qkv_proj", ".q_proj", "q"))
        rules.append(StackedParamRule(".qkv_proj", ".k_proj", "k"))
        rules.append(StackedParamRule(".qkv_proj", ".v_proj", "v"))
        self._THINKER_STACKED_PARAMS = rules
        return rules

    def _create_thinker_submodule(
        self, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.loader import load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_omni.components.thinker import Qwen3OmniThinkerModel

        with torch.device("meta"):
            thinker_model = Qwen3OmniThinkerModel(self.config, comm_group=tp_group)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            thinker_model = thinker_model.to(autocast_dtype)
        thinker_model.to_empty(device=device)

        weights = iter_safetensors_shards(
            self.local_dir, device=device,
            prefix="thinker."
        )
        # Strip the "thinker." prefix from checkpoint keys.
        weights = ((k.removeprefix("thinker."), v) for k, v in weights)

        load_hf_weights(
            thinker_model, weights,
            stacked_params=self._get_thinker_stacked_params(),
            name_remapper=self._thinker_remap,
        )
        thinker_model.eval()


        from mstar.model.qwen3_omni.submodules import ThinkerSubmodule
        return ThinkerSubmodule(
            thinker_model=thinker_model,
            config=self.config,
        )

    def _create_talker_submodule(
        self, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.loader import load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_omni.components.talker import (
            Qwen3OmniTalkerModel,
        )

        with torch.device("meta"):
            # ``tp_group`` shards the Talker LLM's attention + MoE. The
            # CodePredictor stays TP=1 (separate construction below) —
            # its compute is small and the deterministic FlashInfer sampler
            # produces bit-equal codes on every rank, so per-rank
            # replication is cheaper than 150+ NCCL all-reduces per
            # decode step (5 layers x 15 unrolled iterations).
            talker_model = Qwen3OmniTalkerModel(self.config, comm_group=tp_group)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            talker_model = talker_model.to(autocast_dtype)
        talker_model.to_empty(device=device)

        # Talker and CodePredictor share the "talker." prefix. We stream
        # once and split: code_predictor keys go to the CodePredictor,
        # everything else goes to the TalkerModel.
        talker_weights = []
        code_pred_weights = []
        _CP_PREFIX = "talker.code_predictor."
        _TALKER_PREFIX = "talker."

        initialized_thinker_embed_tokens = False
        text_config = self.config.thinker_text
        embed_tokens = torch.nn.Embedding(
            text_config.vocab_size, text_config.hidden_size,
            device=device
        ).eval()
        for k, v in iter_safetensors_shards(self.local_dir, device=device, prefix=_TALKER_PREFIX):
            if k.startswith(_CP_PREFIX):
                code_pred_weights.append((k.removeprefix(_CP_PREFIX), v))
            else:
                talker_weights.append((k.removeprefix(_TALKER_PREFIX), v))
        for _k, v in iter_safetensors_shards(
            self.local_dir, device=device, prefix="thinker.model.embed_tokens"
        ):
            initialized_thinker_embed_tokens = True
            with torch.no_grad():
                embed_tokens.weight.copy_(v)

        assert initialized_thinker_embed_tokens, \
            "thinker.model.embed_tokens not found to initialize talker TTS embeds"

        stacked = self._get_thinker_stacked_params()
        load_hf_weights(
            talker_model, iter(talker_weights),
            stacked_params=stacked,
            name_remapper=self._thinker_remap,
        )
        talker_model.eval()

        with torch.device("meta"):
            code_predictor = Qwen3OmniCodePredictor(self.config)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            code_predictor = code_predictor.to(autocast_dtype)
        code_predictor.to_empty(device=device)
        load_hf_weights(
            code_predictor, iter(code_pred_weights),
            stacked_params=stacked,
            name_remapper=self._thinker_remap,
        )
        code_predictor.consolidate_stacked_weights()
        code_predictor.eval()

        from mstar.model.qwen3_omni.submodules import TalkerSubmodule
        talker_sub = TalkerSubmodule(
            talker_model=talker_model,
            code_predictor=code_predictor,
            config=self.config,
        )
        talker_sub.init_tts_embeds(embed_tokens)
        del embed_tokens

        return talker_sub

    def _create_code2wav_submodule(self, device: str) -> NodeSubmodule:
        # Code2Wav is the vocoder that converts codec tokens to audio waveform.
        # The actual model class will be defined in components.
        from mstar.model.qwen3_omni.components.code2wav import Qwen3OmniMoeCode2Wav
        from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        # The vocoder is dominated by Conv1d/ConvTranspose1d at small channel
        # counts where cuDNN's default heuristic picks a sub-optimal algo.
        # benchmark=True autotunes per shape on the warm-up call, before
        # CUDA-graph capture, so the chosen algo is baked into the graph.
        torch.backends.cudnn.benchmark = True

        code2wav_model = Qwen3OmniMoeCode2Wav(self.config.code2wav)
        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[
                ModuleAndPrefix(code2wav_model, prefix="code2wav"),
            ],
            device=device,
        )
        code2wav_model.eval()
        code2wav_model.consolidate()

        from mstar.model.qwen3_omni.submodules import Code2WavSubmodule
        return Code2WavSubmodule(
            code2wav_model=code2wav_model,
            config=self.config,
        )

    def _create_audio_encoder_submodule(self, device: str) -> NodeSubmodule:
        """Load the audio encoder (AuT) from HF weights.

        Two paths, selected by ``config.native_audio_encoder``:
          * native (default on): batched, transformers-decoupled mstar module
            (``NativeAudioEncoderSubmodule``). Numerically matches HF (fp32
            exact; bf16 within the parity bar). Throughput gain over the HF
            wrapper is modest — ~1.2-1.7x in the SDPA microbenchmark, peaking at
            batch 4-8 (see benchmark/artifacts/README_qwen3_omni_encoders.md);
            the win is cross-request batching, not a faster attention kernel.
          * HF wrapper (fallback/reference, kept for one release).
        """
        from transformers import AutoConfig

        from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        config = AutoConfig.from_pretrained(self.local_dir, trust_remote_code=True)
        audio_config = config.thinker_config.audio_config

        if getattr(self.config, "native_audio_encoder", False):
            from mstar.model.qwen3_omni.components.audio_encoder import (
                NativeQwen3OmniAudioEncoder,
            )
            from mstar.model.qwen3_omni.submodules import NativeAudioEncoderSubmodule
            audio_encoder = NativeQwen3OmniAudioEncoder(audio_config).to(device)
            load_weights_from_hf_shards(
                repo_dir=self.local_dir,
                modules=[ModuleAndPrefix(audio_encoder, prefix="thinker.audio_tower")],
                device=device,
            )
            audio_encoder.eval()
            return NativeAudioEncoderSubmodule(audio_encoder=audio_encoder, config=self.config)

        # ---- HF-wrapper fallback path ----
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
        )

        # Build the audio encoder from config.
        # IMPORTANT: pass attn_implementation="flash_attention_2" so the
        # encoder uses the cu_seqlens FA2 path. With the HF default
        # (which resolves to "sdpa"), Qwen3OmniMoeAudioAttention runs
        # SDPA on the full packed sequence (no per-segment fusion),
        # which is significantly slower than FA2's varlen path.
        audio_encoder = Qwen3OmniMoeAudioEncoder._from_config(
            audio_config, attn_implementation=_hf_encoder_attn_impl()
        )

        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[ModuleAndPrefix(audio_encoder, prefix="thinker.audio_tower")],
            device=device,
        )
        audio_encoder.eval()

        from mstar.model.qwen3_omni.submodules import AudioEncoderSubmodule
        return AudioEncoderSubmodule(audio_encoder=audio_encoder, config=self.config)

    def _create_vision_encoder_submodule(self, device: str) -> NodeSubmodule:
        """Load the vision encoder (SigLIP2 ViT) from HF weights.

        Two paths, selected by ``config.native_vision_encoder``:
          * native (default on): batched, varlen mstar module
            (``NativeVisionEncoderSubmodule``). Numerically matches HF (fp32
            exact; bf16 within bar) for the pooler output and every DeepStack
            level. The large per-image speedup comes almost entirely from
            computing the patch embed as an ``F.linear`` instead of HF's bf16
            ``Conv3d`` (kernel==stride), which hits a cuDNN low-precision cliff
            (~3.3 s/image on H100) — the same swap could in principle be applied
            to the HF path. Attention is the same ``flash_attn_varlen_func``
            primitive HF uses, not a shape-specialized kernel.
          * HF wrapper (fallback/reference, kept for one release).
        """
        from transformers import AutoConfig

        from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        config = AutoConfig.from_pretrained(self.local_dir, trust_remote_code=True)
        vision_config = config.thinker_config.vision_config

        if getattr(self.config, "native_vision_encoder", False):
            from mstar.model.qwen3_omni.components.vision_encoder import (
                NativeQwen3OmniVisionEncoder,
            )
            from mstar.model.qwen3_omni.submodules import NativeVisionEncoderSubmodule
            vision_encoder = NativeQwen3OmniVisionEncoder(vision_config).to(device)
            load_weights_from_hf_shards(
                repo_dir=self.local_dir,
                modules=[ModuleAndPrefix(vision_encoder, prefix="thinker.visual")],
                device=device,
            )
            vision_encoder.eval()
            return NativeVisionEncoderSubmodule(vision_encoder=vision_encoder, config=self.config)

        # ---- HF-wrapper fallback path ----
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeVisionEncoder,
        )

        # Build the vision encoder.
        # CRITICAL: pass attn_implementation="flash_attention_2". Without
        # this, vision_config._attn_implementation defaults to None and is
        # resolved to "sdpa" at runtime (modeling_utils.py:1889). With
        # "sdpa", Qwen3OmniMoeVisionAttention.forward falls into the
        # per-segment Python loop (modeling_qwen3_omni_moe.py:892-913),
        # which issues N sequential attention calls per layer for an
        # N-frame video. This causes the 10× V2T/V2S TTFT regression vs
        # vllm-omni. With "flash_attention_2", a single varlen FA2 call
        # per layer handles all frames at once via cu_seqlens.
        vision_encoder = Qwen3OmniMoeVisionEncoder._from_config(
            vision_config, attn_implementation=_hf_encoder_attn_impl()
        )

        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[ModuleAndPrefix(vision_encoder, prefix="thinker.visual")],
            device=device,
        )
        vision_encoder.eval()

        from mstar.model.qwen3_omni.submodules import VisionEncoderSubmodule
        return VisionEncoderSubmodule(vision_encoder=vision_encoder, config=self.config)
