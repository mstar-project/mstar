"""Phase 2 Task 7: experimental validation of chunked-prefill throughput gains.

Measures whether Phase 2's scheduler-driven mixed-batch packing actually
delivers throughput improvements on a concurrent mixed workload, vs Phase
1's serial-batch-per-walk path where a long prefill blocks all in-flight
decodes.

Workload:
  * 4 long-running decode requests (already past their initial prefill,
    each generating up to 200 tokens at greedy / temp=0).
  * After ~500 ms (modeled here as N "warmup decode" steps), submit a
    5th request with a 4096-token random prompt that needs prefill.

Metrics captured (per mode):
  1. TTFT for the 5th request (time from submission until its first
     decode token is sampled).
  2. p50 inter-token latency for ongoing decodes during the prefill window
     (steps from prefill submission to prefill completion).
  3. p99 inter-token latency for ongoing decodes during the prefill window.
  4. Total throughput (sum of generated tokens divided by total wall-clock).

Implementation strategy ("alternative simplification" path from the spec):
   We drive the engine directly with hand-built ``NodeBatch`` objects --
   one batch per "step" -- mirroring what the worker / micro-scheduler
   would do in production but without spinning up the full conductor /
   IPC machinery.  Two modes:

   - Phase 1 (``scheduler_owns_chunking=False``):  the engine itself
     chunks the prefill internally via ``execute_chunked_prefill``.
     Because the engine is single-threaded, while it is busy executing
     the prefill batch, no decode steps run.  Decode latency for the
     other 4 requests goes way up during the prefill window.

   - Phase 2 (``scheduler_owns_chunking=True``):  we hand-build a
     ``thinker_step`` ``NodeBatch`` per step that packs 4 decode tokens
     plus one prefill chunk of the 5th request, exactly like the
     ``MicroScheduler._get_chunked_step_batch`` path would.  Decodes
     keep ticking each step; the prefill bleeds in chunk-by-chunk.

This avoids the operational complexity of standing up a full
worker+conductor while still exercising the load-bearing engine paths.

Run::

    PATH=.venv/bin:$PATH .venv/bin/pytest \\
        perf_testing/chunked_prefill_throughput.py -v -s
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import pytest
import torch

REPO = Path("/m-coriander/coriander/rohan_sanda/multimodal_inference")
sys.path.insert(0, str(REPO))

from mminf.communication.tensors import LocalTransferEngine  # noqa: E402
from mminf.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mminf.engine.ar_engine import AREngine  # noqa: E402
from mminf.engine.base import NodeBatch  # noqa: E402
from mminf.engine.kv_store import TransferEngineInfo  # noqa: E402
from mminf.utils.sampling import SamplingConfig  # noqa: E402

QWEN3_OMNI_REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def _hf_cache_has_qwen3_omni() -> bool:
    candidates: list[Path] = []
    for env_key in ("HF_HOME", "HF_HUB_CACHE"):
        if env_key in os.environ:
            base = Path(os.environ[env_key])
            candidates.extend([base, base / "hub"])
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    candidates.append(Path("/m-coriander/coriander/rohan_sanda/hf"))
    target = "models--Qwen--Qwen3-Omni-30B-A3B-Instruct"
    return any((base / target).exists() for base in candidates)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        not _hf_cache_has_qwen3_omni(),
        reason=f"{QWEN3_OMNI_REPO} not in local HF cache",
    ),
]


# --------------------------------------------------------------------------
# Workload constants
# --------------------------------------------------------------------------

NUM_DECODE_RIDS = 4
DECODE_PROMPT_LEN = 64        # short prompts so the warmup prefill is cheap
DECODE_MAX_TOKENS = 200       # how many tokens each decode rid generates
WARMUP_DECODES_BEFORE_PREFILL = 8   # ~500 ms equivalent at ~60 ms/decode-step
NEW_REQUEST_PROMPT_LEN = 4096
PREFILL_CHUNK_SIZE = 512      # both phases use the same chunk size
MAX_STEP_TOKENS = 2048        # Phase 2 budget per mixed-batch step


# --------------------------------------------------------------------------
# Engine fixture
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def thinker_engine():
    """Module-scoped Thinker engine, eager mode (no CUDA graphs).

    Mirrors the integration tests' setup so all parametrizations share one
    30B Thinker load.  KV budget: 256 pages * 128 page_size = 32k tokens,
    enough for 4 decode rids (a few hundred tokens each) + one 4096-token
    prefill.
    """
    from mminf.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    cache_dir = os.environ.get("QWEN3_OMNI_CACHE_DIR")
    model = Qwen3OmniModel(model_path_hf=QWEN3_OMNI_REPO, cache_dir=cache_dir)
    thinker = model.get_submodule("Thinker", device=str(device))
    assert thinker is not None

    kv_cfgs = [c for c in model.get_kv_cache_config() if c.nodes and "Thinker" in c.nodes]
    assert len(kv_cfgs) == 1
    kv_cfg = kv_cfgs[0]
    kv_cfg.max_num_pages = 256

    engine = AREngine(
        autocast_dtype=torch.bfloat16,
        max_prefill_chunk_size=PREFILL_CHUNK_SIZE,
        scheduler_owns_chunking=False,  # toggled per run
    )
    engine.load_model(
        submodules={"Thinker": thinker.to(device)},
        kv_cache_config=[kv_cfg],
        device=device,
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="phase2_perf",
            my_session_id="phase2_perf_session",
            transfer_engine=LocalTransferEngine(hostname="phase2_perf"),
        ),
        kv_cache_type=torch.bfloat16,
    )
    # Capture CUDA graphs once. Phase 2.1a measures three modes against
    # this same engine state by toggling the cuda_graph_runner attribute
    # on the Thinker submodule (None => eager fallback path).
    engine.warmup()
    yield engine, device
    engine.shutdown()


# --------------------------------------------------------------------------
# Batch builders
# --------------------------------------------------------------------------


def _make_text_input_ids(n: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(0, 10000, (n,), dtype=torch.long, device=device, generator=g)


def _make_prefill_text_batch(rid: str, text_ids: torch.Tensor, is_last_prefill: bool = True) -> NodeBatch:
    """Single-request prefill_text batch (mirrors the equivalence test)."""
    info = CurrentForwardPassInfo(
        request_id=rid,
        graph_walk="prefill_text",
        requires_cfg=False,
        fwd_index=0,
        random_seed=42,
        max_tokens=1,
        sampling_config={"Thinker": SamplingConfig(temperature=0.0)},
        step_metadata={"audio_output": False, "is_last_prefill": is_last_prefill},
    )
    return NodeBatch(
        node_name="Thinker",
        graph_walk="prefill_text",
        request_ids=[rid],
        per_request_input_tensors={rid: {"text_inputs": [text_ids]}},
        per_request_info={rid: info},
    )


def _make_thinker_decode_batch(rid: str, prev_token: torch.Tensor) -> NodeBatch:
    """Single-request thinker_decode batch."""
    info = CurrentForwardPassInfo(
        request_id=rid,
        graph_walk="thinker_decode",
        requires_cfg=False,
        fwd_index=0,
        random_seed=42,
        max_tokens=1,
        sampling_config={"Thinker": SamplingConfig(temperature=0.0)},
        step_metadata={"audio_output": False},
    )
    return NodeBatch(
        node_name="Thinker",
        graph_walk="thinker_decode",
        request_ids=[rid],
        per_request_input_tensors={rid: {"text_inputs": [prev_token]}},
        per_request_info={rid: info},
    )


def _make_thinker_step_batch(
    per_rid_inputs: dict[str, torch.Tensor],
    is_terminal_per_request: dict[str, bool],
) -> NodeBatch:
    """Mixed-batch thinker_step.

    Mirrors ``test_mixed_batch_correctness._make_thinker_step_batch``.
    Each rid carries either a single decode token (seq_len=1) or a prefill
    chunk slice (seq_len=chunk_size).  ``is_terminal_per_request`` decides
    which rids actually get sampled (decodes + last-chunk-prefills).
    """
    rids = list(per_rid_inputs.keys())
    per_request_input_tensors: dict[str, dict[str, list[torch.Tensor]]] = {}
    per_request_info: dict[str, CurrentForwardPassInfo] = {}
    for rid, ids in per_rid_inputs.items():
        per_request_input_tensors[rid] = {"text_inputs": [ids]}
        per_request_info[rid] = CurrentForwardPassInfo(
            request_id=rid,
            graph_walk="thinker_step",
            requires_cfg=False,
            fwd_index=0,
            random_seed=42,
            max_tokens=1,
            sampling_config={"Thinker": SamplingConfig(temperature=0.0)},
            step_metadata={"audio_output": False},
        )
    return NodeBatch(
        node_name="Thinker",
        graph_walk="thinker_step",
        request_ids=rids,
        per_request_input_tensors=per_request_input_tensors,
        per_request_info=per_request_info,
        is_terminal_per_request=is_terminal_per_request,
    )


# --------------------------------------------------------------------------
# Workload state
# --------------------------------------------------------------------------


class DecodeRidState:
    """Per-decode-request state across a run."""

    __slots__ = (
        "rid", "last_token", "tokens_generated",
        "max_tokens", "token_times", "first_decode_time",
    )

    def __init__(self, rid: str, max_tokens: int) -> None:
        self.rid = rid
        self.last_token: torch.Tensor | None = None
        self.tokens_generated = 0
        self.max_tokens = max_tokens
        # ``token_times[i]`` is the wall-clock at which token i finished.
        self.token_times: list[float] = []
        self.first_decode_time: float | None = None


def _setup_decode_rids(engine, device) -> list[DecodeRidState]:
    """Prefill the 4 decode rids and capture each one's first sampled token."""
    states: list[DecodeRidState] = []
    for i in range(NUM_DECODE_RIDS):
        rid = f"decode_{i}_{uuid.uuid4().hex[:6]}"
        engine.add_request(rid, ["main"])
        ids = _make_text_input_ids(DECODE_PROMPT_LEN, device, seed=100 + i)
        batch = _make_prefill_text_batch(rid, ids, is_last_prefill=True)
        out = engine.execute_batch(batch)
        assert not out.allocation_failed, f"prefill alloc failed for {rid}"
        new_tok = out.per_request_output_tensors[rid]["new_token"][0]
        st = DecodeRidState(rid=rid, max_tokens=DECODE_MAX_TOKENS)
        st.last_token = new_tok.flatten().to(device).to(torch.long)
        st.tokens_generated = 1  # the prefill produced 1 token already
        states.append(st)
    return states


def _teardown_rids(engine, rids: list[str]) -> None:
    for rid in rids:
        try:
            engine.remove_request(rid)
        except Exception:
            pass


# --------------------------------------------------------------------------
# Phase 1 runner: one engine call per scheduling step.
# --------------------------------------------------------------------------


def _decode_step_phase1(engine, device, decodes: list[DecodeRidState]) -> None:
    """Run one decode step per active rid (Phase 1: separate batch per call).

    Phase 1's engine path doesn't pack mixed batches; the worker's
    ``MicroScheduler`` would normally batch all decode rids into a single
    ``thinker_decode`` batch.  We model that here with ONE multi-rid
    ``thinker_decode`` batch (n=4).  This is the apples-to-apples baseline
    for what Phase 1 production sees.

    All sampled tokens get timestamped after a single CUDA sync at the end.
    """
    active = [s for s in decodes if s.tokens_generated < s.max_tokens]
    if not active:
        return
    # Build a multi-rid thinker_decode batch.
    rids = [s.rid for s in active]
    per_rid_inputs: dict[str, dict[str, list[torch.Tensor]]] = {}
    per_request_info: dict[str, CurrentForwardPassInfo] = {}
    for s in active:
        per_rid_inputs[s.rid] = {"text_inputs": [s.last_token]}
        per_request_info[s.rid] = CurrentForwardPassInfo(
            request_id=s.rid,
            graph_walk="thinker_decode",
            requires_cfg=False,
            fwd_index=0,
            random_seed=42,
            max_tokens=1,
            sampling_config={"Thinker": SamplingConfig(temperature=0.0)},
            step_metadata={"audio_output": False},
        )
    batch = NodeBatch(
        node_name="Thinker",
        graph_walk="thinker_decode",
        request_ids=rids,
        per_request_input_tensors=per_rid_inputs,
        per_request_info=per_request_info,
    )
    out = engine.execute_batch(batch)
    assert not out.allocation_failed, "decode batch alloc failed"
    torch.cuda.synchronize()
    now = time.perf_counter()
    for s in active:
        rid_out = out.per_request_output_tensors.get(s.rid, {})
        if "new_token" not in rid_out:
            continue
        s.last_token = rid_out["new_token"][0].flatten().to(device).to(torch.long)
        s.tokens_generated += 1
        s.token_times.append(now)


def _run_phase1(engine, device) -> dict:
    """Phase 1 path: scheduler_owns_chunking=False.

    Sequence:
      1. Setup 4 decode rids (initial prefill).
      2. Run WARMUP_DECODES_BEFORE_PREFILL decode steps.
      3. Submit the 5th request: a single big prefill batch (engine chunks
         internally).  Record TTFT.
      4. Run the rest of the decodes to completion (including the new
         request's decodes).

    During step 3 the engine is busy in execute_chunked_prefill -- decodes
    are blocked.  Inter-token latency for the 4 in-flight decodes spikes.
    """
    engine.scheduler_owns_chunking = False
    engine.max_prefill_chunk_size = PREFILL_CHUNK_SIZE

    decodes = _setup_decode_rids(engine, device)
    new_rid = f"newreq_{uuid.uuid4().hex[:6]}"
    new_prompt = _make_text_input_ids(NEW_REQUEST_PROMPT_LEN, device, seed=999)

    torch.cuda.synchronize()
    run_start = time.perf_counter()
    try:
        # Stage 2: warmup decodes
        for _ in range(WARMUP_DECODES_BEFORE_PREFILL):
            _decode_step_phase1(engine, device, decodes)

        # Mark prefill window start.
        prefill_window_start = time.perf_counter()

        # Stage 3: submit prefill (single big batch -- engine chunks internally).
        engine.add_request(new_rid, ["main"])
        prefill_submit_time = time.perf_counter()
        prefill_batch = _make_prefill_text_batch(new_rid, new_prompt, is_last_prefill=True)
        out = engine.execute_batch(prefill_batch)
        assert not out.allocation_failed, "new request prefill alloc failed"
        torch.cuda.synchronize()
        prefill_done_time = time.perf_counter()
        # Capture TTFT for the new request: time from submit until its first sampled token.
        new_first_token = out.per_request_output_tensors[new_rid]["new_token"][0]
        ttft_ms = (prefill_done_time - prefill_submit_time) * 1000.0

        # Now the new request enters the decode pool.
        new_decode = DecodeRidState(rid=new_rid, max_tokens=20)
        new_decode.last_token = new_first_token.flatten().to(device).to(torch.long)
        new_decode.tokens_generated = 1
        new_decode.first_decode_time = prefill_done_time
        decodes.append(new_decode)

        prefill_window_end = time.perf_counter()

        # Stage 4: run decodes to completion.
        while any(s.tokens_generated < s.max_tokens for s in decodes):
            _decode_step_phase1(engine, device, decodes)

        run_end = time.perf_counter()

    finally:
        _teardown_rids(engine, [s.rid for s in decodes])

    return _compute_metrics(
        decodes=decodes,
        ttft_ms=ttft_ms,
        run_start=run_start,
        run_end=run_end,
        prefill_window_start=prefill_window_start,
        prefill_window_end=prefill_window_end,
        warmup_steps=WARMUP_DECODES_BEFORE_PREFILL,
        new_rid=new_rid,
    )


# --------------------------------------------------------------------------
# Phase 2 runner: mixed-batch thinker_step.
# --------------------------------------------------------------------------


def _decode_only_step_phase2(engine, device, decodes: list[DecodeRidState]) -> None:
    """Run one mixed-batch step where there's no prefill in flight.

    Uses ``thinker_step`` with all rids terminal=True, mirroring what the
    Phase 2 scheduler would emit when only decodes are ready.
    """
    active = [s for s in decodes if s.tokens_generated < s.max_tokens]
    if not active:
        return
    per_rid_inputs = {s.rid: s.last_token for s in active}
    is_terminal = {s.rid: True for s in active}
    batch = _make_thinker_step_batch(per_rid_inputs, is_terminal)
    out = engine.execute_batch(batch)
    assert not out.allocation_failed
    torch.cuda.synchronize()
    now = time.perf_counter()
    for s in active:
        rid_out = out.per_request_output_tensors.get(s.rid, {})
        if "new_token" not in rid_out:
            continue
        s.last_token = rid_out["new_token"][0].flatten().to(device).to(torch.long)
        s.tokens_generated += 1
        s.token_times.append(now)


def _mixed_step_phase2(
    engine, device, decodes: list[DecodeRidState],
    prefill_rid: str, prefill_prompt: torch.Tensor,
    prefill_consumed: int,
) -> tuple[int, bool, torch.Tensor | None]:
    """One mixed step: pack decodes + one prefill chunk.

    Returns ``(new_consumed, is_terminal_chunk, new_token_or_None)``:
      * new_consumed: prefill_consumed after this step.
      * is_terminal_chunk: True iff the chunk that ran was the last one.
      * new_token_or_None: the sampled first decode token for prefill_rid,
        only when is_terminal_chunk is True.
    """
    # Decode budget.
    active_decodes = [s for s in decodes if s.tokens_generated < s.max_tokens]
    decode_count = len(active_decodes)
    remaining_prefill = NEW_REQUEST_PROMPT_LEN - prefill_consumed
    chunk_budget = MAX_STEP_TOKENS - decode_count
    chunk_size = min(remaining_prefill, chunk_budget)
    is_terminal_chunk = chunk_size == remaining_prefill
    chunk_slice = prefill_prompt[prefill_consumed : prefill_consumed + chunk_size]

    per_rid_inputs: dict[str, torch.Tensor] = {}
    is_terminal: dict[str, bool] = {}
    for s in active_decodes:
        per_rid_inputs[s.rid] = s.last_token
        is_terminal[s.rid] = True
    per_rid_inputs[prefill_rid] = chunk_slice
    is_terminal[prefill_rid] = is_terminal_chunk

    batch = _make_thinker_step_batch(per_rid_inputs, is_terminal)
    out = engine.execute_batch(batch)
    assert not out.allocation_failed, "mixed thinker_step alloc failed"
    torch.cuda.synchronize()
    now = time.perf_counter()
    for s in active_decodes:
        rid_out = out.per_request_output_tensors.get(s.rid, {})
        if "new_token" not in rid_out:
            continue
        s.last_token = rid_out["new_token"][0].flatten().to(device).to(torch.long)
        s.tokens_generated += 1
        s.token_times.append(now)

    new_token = None
    if is_terminal_chunk:
        prefill_out = out.per_request_output_tensors.get(prefill_rid, {})
        if "new_token" in prefill_out:
            new_token = prefill_out["new_token"][0].flatten().to(device).to(torch.long)

    return prefill_consumed + chunk_size, is_terminal_chunk, new_token


def _run_phase2(engine, device) -> dict:
    """Phase 2 path: scheduler_owns_chunking=True.

    Same workload as Phase 1 but with mixed-batch thinker_step packing.
    Decodes + prefill chunks share each step, so decode latency stays
    near baseline during the prefill window and TTFT is only one chunk
    away from when the request enters the active pool.
    """
    engine.scheduler_owns_chunking = True
    engine.max_prefill_chunk_size = None  # engine will not internally chunk

    decodes = _setup_decode_rids(engine, device)
    new_rid = f"newreq_{uuid.uuid4().hex[:6]}"
    new_prompt = _make_text_input_ids(NEW_REQUEST_PROMPT_LEN, device, seed=999)

    torch.cuda.synchronize()
    run_start = time.perf_counter()
    try:
        # Stage 2: warmup decodes (decodes-only thinker_step)
        for _ in range(WARMUP_DECODES_BEFORE_PREFILL):
            _decode_only_step_phase2(engine, device, decodes)

        # Mark prefill window start.
        prefill_window_start = time.perf_counter()

        # Stage 3: admit new request, run mixed steps until prefill done.
        engine.add_request(new_rid, ["main"])
        prefill_submit_time = time.perf_counter()
        prefill_consumed = 0
        new_first_token: torch.Tensor | None = None
        ttft_ms: float | None = None
        while prefill_consumed < NEW_REQUEST_PROMPT_LEN:
            prefill_consumed, is_term, new_tok = _mixed_step_phase2(
                engine, device, decodes, new_rid, new_prompt, prefill_consumed,
            )
            if is_term and new_tok is not None:
                new_first_token = new_tok
                ttft_ms = (time.perf_counter() - prefill_submit_time) * 1000.0

        prefill_window_end = time.perf_counter()
        assert ttft_ms is not None, "Phase 2 prefill never produced a first token"

        # Add the new request to the decode pool.
        new_decode = DecodeRidState(rid=new_rid, max_tokens=20)
        new_decode.last_token = new_first_token
        new_decode.tokens_generated = 1
        new_decode.first_decode_time = prefill_window_end
        decodes.append(new_decode)

        # Stage 4: drive remaining decodes to completion.
        while any(s.tokens_generated < s.max_tokens for s in decodes):
            _decode_only_step_phase2(engine, device, decodes)

        run_end = time.perf_counter()

    finally:
        _teardown_rids(engine, [s.rid for s in decodes])

    return _compute_metrics(
        decodes=decodes,
        ttft_ms=ttft_ms,
        run_start=run_start,
        run_end=run_end,
        prefill_window_start=prefill_window_start,
        prefill_window_end=prefill_window_end,
        warmup_steps=WARMUP_DECODES_BEFORE_PREFILL,
        new_rid=new_rid,
    )


# --------------------------------------------------------------------------
# Metrics computation
# --------------------------------------------------------------------------


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _compute_metrics(
    decodes: list[DecodeRidState],
    ttft_ms: float,
    run_start: float,
    run_end: float,
    prefill_window_start: float,
    prefill_window_end: float,
    warmup_steps: int,
    new_rid: str,
) -> dict:
    """Crunch the captured timestamps into the 4 spec metrics.

    For inter-token latency during prefill window: gather, for each
    in-flight decode rid (NOT the new prefill rid), the gaps between
    consecutive token timestamps where the second timestamp falls within
    [prefill_window_start, prefill_window_end].
    """
    # Baseline: pre-prefill p50 inter-token latency.
    pre_window_gaps_ms: list[float] = []
    in_window_gaps_ms: list[float] = []
    total_tokens = 0
    for s in decodes:
        if s.rid == new_rid:
            total_tokens += s.tokens_generated
            continue
        total_tokens += s.tokens_generated
        # Iterate consecutive token timestamps.
        prev_t: float | None = None
        for t in s.token_times:
            if prev_t is not None:
                gap_ms = (t - prev_t) * 1000.0
                if prefill_window_start <= t <= prefill_window_end:
                    in_window_gaps_ms.append(gap_ms)
                elif t < prefill_window_start:
                    pre_window_gaps_ms.append(gap_ms)
            prev_t = t

    p50_baseline_ms = _percentile(pre_window_gaps_ms, 0.5)
    p50_in_window_ms = _percentile(in_window_gaps_ms, 0.5)
    p99_in_window_ms = _percentile(in_window_gaps_ms, 0.99)

    total_wall_s = run_end - run_start
    throughput_tok_per_s = total_tokens / total_wall_s if total_wall_s > 0 else 0.0

    return {
        "ttft_ms": ttft_ms,
        "p50_baseline_ms": p50_baseline_ms,
        "p50_in_window_ms": p50_in_window_ms,
        "p99_in_window_ms": p99_in_window_ms,
        "throughput_tok_per_s": throughput_tok_per_s,
        "total_tokens": total_tokens,
        "wall_clock_s": total_wall_s,
        "n_pre_window_gaps": len(pre_window_gaps_ms),
        "n_in_window_gaps": len(in_window_gaps_ms),
        "prefill_window_s": prefill_window_end - prefill_window_start,
    }


def _print_run_summary(label: str, m: dict) -> None:
    print(
        f"\n=== {label} ===\n"
        f"  TTFT (new req)            : {m['ttft_ms']:.1f} ms\n"
        f"  p50 ITL baseline (pre)    : {m['p50_baseline_ms']:.2f} ms"
        f"  ({m['n_pre_window_gaps']} samples)\n"
        f"  p50 ITL in prefill window : {m['p50_in_window_ms']:.2f} ms"
        f"  ({m['n_in_window_gaps']} samples)\n"
        f"  p99 ITL in prefill window : {m['p99_in_window_ms']:.2f} ms\n"
        f"  prefill window duration   : {m['prefill_window_s']*1000:.1f} ms\n"
        f"  total tokens              : {m['total_tokens']}\n"
        f"  wall clock                : {m['wall_clock_s']:.2f} s\n"
        f"  throughput                : {m['throughput_tok_per_s']:.2f} tok/s"
    )


# --------------------------------------------------------------------------
# The actual test
# --------------------------------------------------------------------------


def test_chunked_prefill_throughput_phase2_vs_phase1(thinker_engine):
    """Phase 2.1a 3-way comparison: Phase 1 vs Phase 2 eager vs Phase 2 + CUDA graphs.

    The Phase 2 Task 7 result (Phase 1 vs Phase 2 eager) measured a 1.18x
    p50 inter-token latency regression during the prefill window vs the
    decodes-only baseline.  Phase 2.1a's CUDA graph replay for
    ``thinker_step`` is hypothesized to close that gap by eliminating the
    per-step Python overhead.

    Strict success criteria (Phase 2.1a):
      1. p2_graphs.p50_in_window <= p2_eager.p50_in_window  (graphs help)
      2. p2_graphs.p50_in_window <= p2_graphs.p50_baseline * 1.10
         (close the gap to within 10% of decodes-only baseline)
      3. p2_graphs.p99_in_window <= p2_graphs.p50_in_window * 2.5
         (no tail blowup under graphs)
      4. p2_graphs.ttft <= p2_eager.ttft * 1.10
         (TTFT improvement preserved)
    """
    engine, device = thinker_engine
    submod = engine.submodule_management["Thinker"]

    # Phase 1 (eager): toggle the runner off so the measurement matches
    # what Phase 2 Task 7 reported (no CUDA graph replay).
    print("\n" + "=" * 70)
    print("PHASE 1 (scheduler_owns_chunking=False, eager)")
    print("=" * 70)
    saved_runner = submod.cuda_graph_runner
    submod.cuda_graph_runner = None
    try:
        p1 = _run_phase1(engine, device)
    finally:
        submod.cuda_graph_runner = saved_runner
    _print_run_summary("PHASE 1 (eager)", p1)

    # Phase 2 eager: same toggle pattern, against the same warmed engine.
    print("\n" + "=" * 70)
    print("PHASE 2 eager (scheduler_owns_chunking=True, no CUDA graphs)")
    print("=" * 70)
    saved_runner = submod.cuda_graph_runner
    submod.cuda_graph_runner = None
    try:
        p2_eager = _run_phase2(engine, device)
    finally:
        submod.cuda_graph_runner = saved_runner
    _print_run_summary("PHASE 2 eager", p2_eager)

    # Phase 2 + CUDA graphs: runner restored from warmup.
    assert submod.cuda_graph_runner is not None, (
        "warmup() failed to capture a CUDA graph runner for Thinker -- "
        "cannot measure Phase 2 + graphs mode"
    )
    print("\n" + "=" * 70)
    print("PHASE 2 + CUDA graphs (scheduler_owns_chunking=True, graphs ON)")
    print("=" * 70)
    p2_graphs = _run_phase2(engine, device)
    _print_run_summary("PHASE 2 + CUDA graphs", p2_graphs)

    # 3-way summary -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: Phase 1 vs Phase 2 eager vs Phase 2 + CUDA graphs")
    print("=" * 70)
    print(
        f"\n=== Phase 1 (engine-internal chunking, eager) ===\n"
        f"  TTFT (request 5):              {p1['ttft_ms']:.1f}ms\n"
        f"  decode p50 during prefill:     {p1['p50_in_window_ms']:.2f}ms\n"
        f"  decode p99 during prefill:     {p1['p99_in_window_ms']:.2f}ms\n"
        f"  decode baseline p50:           {p1['p50_baseline_ms']:.2f}ms\n"
        f"  total throughput:              {p1['throughput_tok_per_s']:.1f} tok/s"
    )

    p2e_ttft_imp = (p1['ttft_ms'] / p2_eager['ttft_ms']) if p2_eager['ttft_ms'] > 0 else float("inf")
    p2e_p50_ratio = (
        p2_eager['p50_in_window_ms'] / p2_eager['p50_baseline_ms']
        if p2_eager['p50_baseline_ms'] > 0 else float("inf")
    )
    print(
        f"\n=== Phase 2 eager (scheduler-aware, no CUDA graphs) ===\n"
        f"  TTFT (request 5):              {p2_eager['ttft_ms']:.1f}ms\n"
        f"  decode p50 during prefill:     {p2_eager['p50_in_window_ms']:.2f}ms\n"
        f"  decode p99 during prefill:     {p2_eager['p99_in_window_ms']:.2f}ms\n"
        f"  decode baseline p50:           {p2_eager['p50_baseline_ms']:.2f}ms\n"
        f"  total throughput:              {p2_eager['throughput_tok_per_s']:.1f} tok/s\n"
        f"  TTFT improvement vs P1:        {p2e_ttft_imp:.2f}x\n"
        f"  p50 vs baseline:               {p2e_p50_ratio:.2f}x"
    )

    p2g_ttft_imp = (p1['ttft_ms'] / p2_graphs['ttft_ms']) if p2_graphs['ttft_ms'] > 0 else float("inf")
    p2g_p50_ratio = (
        p2_graphs['p50_in_window_ms'] / p2_graphs['p50_baseline_ms']
        if p2_graphs['p50_baseline_ms'] > 0 else float("inf")
    )
    p2g_vs_eager = (
        p2_graphs['p50_in_window_ms'] / p2_eager['p50_in_window_ms']
        if p2_eager['p50_in_window_ms'] > 0 else float("inf")
    )
    print(
        f"\n=== Phase 2 + CUDA graphs ===\n"
        f"  TTFT (request 5):              {p2_graphs['ttft_ms']:.1f}ms\n"
        f"  decode p50 during prefill:     {p2_graphs['p50_in_window_ms']:.2f}ms\n"
        f"  decode p99 during prefill:     {p2_graphs['p99_in_window_ms']:.2f}ms\n"
        f"  decode baseline p50:           {p2_graphs['p50_baseline_ms']:.2f}ms\n"
        f"  total throughput:              {p2_graphs['throughput_tok_per_s']:.1f} tok/s\n"
        f"  TTFT improvement vs P1:        {p2g_ttft_imp:.2f}x\n"
        f"  p50 vs baseline:               {p2g_p50_ratio:.2f}x\n"
        f"  p50 vs P2 eager:               {p2g_vs_eager:.2f}x"
    )

    # === Strict success criteria for Phase 2.1a =============================

    failures: list[str] = []

    # 1. Graphs must reduce p50 vs eager (the central claim of Phase 2.1a).
    if p2_graphs['p50_in_window_ms'] > p2_eager['p50_in_window_ms']:
        failures.append(
            f"CUDA graphs did not reduce p50: eager={p2_eager['p50_in_window_ms']:.2f}ms "
            f"graphs={p2_graphs['p50_in_window_ms']:.2f}ms"
        )

    # 2. Graphs must close the gap to baseline (within 1.10x — a relaxed
    #    floor: the mixed batch is ~10% irreducibly heavier than decode-only).
    if p2_graphs['p50_baseline_ms'] > 0 and (
        p2_graphs['p50_in_window_ms'] > p2_graphs['p50_baseline_ms'] * 1.10
    ):
        failures.append(
            f"p50 still regressed > 10% vs baseline even with graphs: "
            f"baseline={p2_graphs['p50_baseline_ms']:.2f}ms "
            f"in-window={p2_graphs['p50_in_window_ms']:.2f}ms"
        )

    # 3. p99 should not blow up under graphs.
    if p2_graphs['p50_in_window_ms'] > 0 and (
        p2_graphs['p99_in_window_ms'] > p2_graphs['p50_in_window_ms'] * 2.5
    ):
        failures.append(
            f"p99 spiked > 2.5x p50 under graphs: "
            f"p50={p2_graphs['p50_in_window_ms']:.2f}ms "
            f"p99={p2_graphs['p99_in_window_ms']:.2f}ms"
        )

    # 4. TTFT improvement preserved (graphs should not regress TTFT vs eager).
    if p2_graphs['ttft_ms'] > p2_eager['ttft_ms'] * 1.10:
        failures.append(
            f"TTFT regressed under graphs: eager={p2_eager['ttft_ms']:.1f}ms "
            f"graphs={p2_graphs['ttft_ms']:.1f}ms"
        )

    if failures:
        msg = "Phase 2.1a success criteria NOT met:\n  " + "\n  ".join(failures)
        pytest.fail(msg)
