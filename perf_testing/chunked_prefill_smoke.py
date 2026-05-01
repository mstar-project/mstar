"""Catastrophic-regression smoke check for chunked prefill TTFT.

Single-request chunked prefill is FUNDAMENTALLY N× slower than unchunked
when the workload is memory-bandwidth-bound (which is the case at 30B
params and batch=1 — each forward pass takes ~60ms regardless of token
count, dominated by HBM weight loads). For prompt_len=4096, chunk_size=512,
N=8 chunks → expected ~8× slowdown vs unchunked.

This smoke check exists to catch CATASTROPHIC regressions (e.g., 50×+
slower from a bug like accidental sync, double-tokenization, deadlocks),
not to flag the expected N× single-request inherent cost. The throughput
benefit of chunked prefill comes from Phase 2's mixed-batch scheduling
(interleaving prefill chunks with decodes from other requests), not from
single-request latency.

Run:
    PATH=.venv/bin:$PATH .venv/bin/pytest perf_testing/chunked_prefill_smoke.py -v -s
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

from test.integration.test_chunked_prefill_equivalence import (  # noqa: E402
    _make_prefill_text_batch,
    _make_text_input_ids,
)


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
        reason="Qwen3-Omni weights not in local HF cache",
    ),
]


@pytest.fixture(scope="module")
def thinker_engine_for_perf():
    """Reuse the integration test's engine setup pattern.

    Module-scoped: loading qwen3_omni Thinker takes ~30s; share one engine
    across all checks here.
    """
    from mminf.communication.tensors import LocalTransferEngine
    from mminf.engine.ar_engine import AREngine
    from mminf.engine.kv_store import TransferEngineInfo
    from mminf.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    cache_dir = os.environ.get("QWEN3_OMNI_CACHE_DIR")
    model = Qwen3OmniModel(model_path_hf="Qwen/Qwen3-Omni-30B-A3B-Instruct", cache_dir=cache_dir)
    thinker = model.get_submodule("Thinker", device=str(device))
    kv_cfgs = [c for c in model.get_kv_cache_config() if c.nodes and "Thinker" in c.nodes]
    assert len(kv_cfgs) == 1
    kv_cfg = kv_cfgs[0]
    kv_cfg.max_num_pages = 256

    engine = AREngine(autocast_dtype=torch.bfloat16, max_prefill_chunk_size=None)
    engine.load_model(
        submodules={"Thinker": thinker.to(device)},
        kv_cache_config=[kv_cfg],
        device=device,
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="perf_smoke",
            my_session_id="perf_smoke_session",
            transfer_engine=LocalTransferEngine(hostname="perf_smoke"),
        ),
        kv_cache_type=torch.bfloat16,
    )

    yield engine, device

    engine.shutdown()


def _run_prefill_text(engine, device, prompt_len: int, rid: str) -> None:
    """Single-shot prefill_text invocation for perf timing.

    Generates a fresh prompt (per-rid seed for variety), registers the request,
    runs ``execute_batch``, then frees the KV state. The caller times around
    this whole call — JIT/build work has already been amortized by an earlier
    warmup invocation.
    """
    text_ids = _make_text_input_ids(prompt_len, device, seed=hash(rid) & 0xFFFF)
    engine.add_request(rid, ["main"])
    try:
        batch = _make_prefill_text_batch(rid, text_ids)
        out = engine.execute_batch(batch)
        assert not out.allocation_failed, f"allocation failed for rid={rid}"
    finally:
        engine.remove_request(rid)


def test_chunked_prefill_no_catastrophic_regression(thinker_engine_for_perf):
    """Catastrophic-regression guard. Chunked single-request will be ~N× slower
    than unchunked because the workload is HBM-bandwidth-bound; this test
    accepts that inherent cost but catches anything dramatically worse.
    """
    engine, device = thinker_engine_for_perf

    prompt_len = 4096
    chunk_size = 512
    n_chunks = (prompt_len + chunk_size - 1) // chunk_size  # 8

    # Warm up both paths so first-call JIT doesn't pollute timing.
    engine.max_prefill_chunk_size = None
    _run_prefill_text(engine, device, prompt_len, f"warm_u_{uuid.uuid4().hex[:8]}")
    engine.max_prefill_chunk_size = chunk_size
    _run_prefill_text(engine, device, prompt_len, f"warm_c_{uuid.uuid4().hex[:8]}")
    torch.cuda.synchronize()

    def time_one(chunk_setting, label):
        engine.max_prefill_chunk_size = chunk_setting
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run_prefill_text(engine, device, prompt_len, f"{label}_{uuid.uuid4().hex[:8]}")
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    n = 3
    t_unchunked = sum(time_one(None, f"u{i}") for i in range(n)) / n
    t_chunked = sum(time_one(chunk_size, f"c{i}") for i in range(n)) / n

    ratio = t_chunked / t_unchunked
    # Generous physics-aware threshold: allow 2× the inherent N× cost plus
    # 200ms of fixed Python overhead. Catches anything dramatically worse.
    threshold_s = n_chunks * 2.0 * t_unchunked + 0.2

    print(
        f"\nprompt_len={prompt_len} chunk_size={chunk_size} n_chunks={n_chunks}\n"
        f"  unchunked: {t_unchunked*1000:.1f}ms  chunked: {t_chunked*1000:.1f}ms\n"
        f"  ratio: {ratio:.2f}×  expected ~{n_chunks}× (memory-bandwidth-bound)\n"
        f"  threshold: {threshold_s*1000:.1f}ms"
    )

    assert t_chunked < threshold_s, (
        f"chunked TTFT exceeded catastrophic-regression threshold: "
        f"unchunked={t_unchunked*1000:.1f}ms chunked={t_chunked*1000:.1f}ms "
        f"ratio={ratio:.2f}× threshold={threshold_s*1000:.1f}ms (n_chunks×2 + 200ms)"
    )
