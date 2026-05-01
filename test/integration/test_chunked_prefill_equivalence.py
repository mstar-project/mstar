"""Numerical equivalence: chunked prefill must match non-chunked prefill.

Builds one ``AREngine`` with the qwen3_omni Thinker submodule, no CUDA
graphs.  For each ``(prompt_len, chunk_size)`` pair, runs ``prefill_text``
twice — once with ``engine.max_prefill_chunk_size = None`` (unchunked
baseline) and once with ``engine.max_prefill_chunk_size = chunk_size``
(chunked) — using a fresh request_id each call.  Compares logits /
sampled token / populated KV cache contents within bf16 tolerance.

Why one engine + toggle (vs. ``build_pair`` from the plan):  loading the
30B Thinker takes ~30 s and ~30 GB of GPU memory; running it twice is
wasteful when a single engine can be reconfigured between calls by
flipping ``engine.max_prefill_chunk_size`` and using a fresh ``request_id``
(which gives each run its own KV cache state).

Why no CUDA graph capture:  ``_can_use_cuda_graph`` returns False when
``submod_mgmt.cuda_graph_runner is None``, so both the chunked and
unchunked paths fall through to the same eager ``_execute_sequential``
dispatch (``ThinkerSubmodule.can_batch`` returns False for prefill walks).
This makes the comparison apples-to-apples: identical kernels, only the
chunked orchestration differs.

Requires qwen3_omni weights in the HF cache::

    huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mminf.communication.tensors import LocalTransferEngine  # noqa: E402
from mminf.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mminf.engine.ar_engine import AREngine  # noqa: E402
from mminf.engine.base import NodeBatch  # noqa: E402
from mminf.engine.kv_store import TransferEngineInfo  # noqa: E402
from mminf.utils.sampling import SamplingConfig  # noqa: E402

QWEN3_OMNI_REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def _hf_cache_has_qwen3_omni() -> bool:
    """Return True if Qwen3-Omni snapshots are already on local disk.

    Same logic as ``test_prefill_cuda_graph._hf_cache_has_qwen3_omni`` plus a
    machine-specific fallback for the lab path used in ``CLAUDE.md``.
    """
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
        reason=f"{QWEN3_OMNI_REPO} not in local HF cache; run "
               f"`huggingface-cli download {QWEN3_OMNI_REPO}`",
    ),
]


def _make_transfer_info() -> TransferEngineInfo:
    """Build a single-node ``TransferEngineInfo`` backed by ``LocalTransferEngine``.

    The engine's ``PagedAllocationManager`` accepts only ``MooncakeTransferEngine``
    or ``LocalTransferEngine``; arbitrary stubs raise ``ValueError``.  Local is
    a no-op shim — no remote reads happen because this test never hands a
    request to another worker.
    """
    return TransferEngineInfo(
        my_entity_id="chunked_prefill_test",
        my_session_id="chunked_prefill_session",
        transfer_engine=LocalTransferEngine(hostname="chunked_prefill_test"),
    )


@pytest.fixture(scope="module")
def thinker_engine():
    """One ``AREngine`` with the qwen3_omni Thinker, NO CUDA graphs.

    Module-scoped because loading the 30B Thinker takes ~30 s and ~30 GB.
    All parametrized test cases share this one engine and use distinct
    request_ids so their KV state never overlaps.

    Deliberately skips ``warmup`` / CUDA-graph capture.  With
    ``submod_mgmt.cuda_graph_runner = None`` the engine's
    ``_can_use_cuda_graph`` returns False, so both the chunked and
    unchunked paths run through the same eager ``_execute_sequential``
    dispatch — the only difference between runs is whether the chunked
    orchestrator slices the prompt or hands it to the model whole.
    """
    from mminf.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    # CudaGraphRunner asserts an explicit cuda:N (no bare "cuda"); even
    # though we don't capture graphs, mirror the same idiom in case any
    # downstream code path checks it.
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    cache_dir = os.environ.get("QWEN3_OMNI_CACHE_DIR")  # optional override

    model = Qwen3OmniModel(model_path_hf=QWEN3_OMNI_REPO, cache_dir=cache_dir)
    thinker = model.get_submodule("Thinker", device=str(device))
    assert thinker is not None, "Thinker submodule failed to load"

    kv_cfgs = [c for c in model.get_kv_cache_config() if c.nodes and "Thinker" in c.nodes]
    assert len(kv_cfgs) == 1, f"expected 1 Thinker KV config, got {len(kv_cfgs)}"
    kv_cfg = kv_cfgs[0]
    # 256 pages × 128 page_size = 32768 tokens. Each parametrized case
    # holds 2 active rids of up to 2048 tokens (16 pages each); we free
    # them between cases via remove_request, so 256 is comfortable.
    kv_cfg.max_num_pages = 256

    # max_prefill_chunk_size starts at None; the test toggles per call.
    engine = AREngine(autocast_dtype=torch.bfloat16, max_prefill_chunk_size=None)
    transfer_info = _make_transfer_info()
    engine.load_model(
        submodules={"Thinker": thinker.to(device)},
        kv_cache_config=[kv_cfg],
        device=device,
        transfer_engine_info=transfer_info,
        kv_cache_type=torch.bfloat16,
    )
    # Deliberately skip engine.warmup() — we want
    # submod_mgmt.cuda_graph_runner == None for apples-to-apples eager
    # comparison between chunked and unchunked paths.
    assert engine.submodule_management["Thinker"].cuda_graph_runner is None

    yield engine, device

    engine.shutdown()


def _make_text_input_ids(prompt_len: int, device: torch.device, seed: int) -> torch.Tensor:
    """Generate ``prompt_len`` random token IDs in a "safe" vocab range.

    Mirrors ``_make_inputs`` in ``test_prefill_cuda_graph.py``: clamps to
    ``[0, 10000)`` to avoid Qwen's special tokens (``im_start``, ``audio_*``,
    ``vision_*``, etc.) which sit at high IDs and would change downstream
    branching (talker text mask, BOS/EOS sentinel handling).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(
        0, 10000, (prompt_len,),
        dtype=torch.long, device=device, generator=g,
    )


def _make_prefill_text_batch(
    rid: str,
    text_ids: torch.Tensor,
) -> NodeBatch:
    """Build a single-request ``prefill_text`` ``NodeBatch``.

    Models the input shape that ``ThinkerSubmodule.prepare_inputs`` reads
    when ``graph_walk == "prefill_text"``: it pulls ``inputs["text_inputs"][0]``
    from ``batch.per_request_input_tensors[rid]``.  ``per_label_seq_info`` is
    left empty so ``execute_batch``'s sync_retrieve loop is a no-op (no
    pre-existing remote KV state to import for a fresh rid).
    """
    info = CurrentForwardPassInfo(
        request_id=rid,
        graph_walk="prefill_text",
        requires_cfg=False,
        fwd_index=0,
        random_seed=42,
        max_tokens=1,
        # temperature=0 → greedy argmax, so the ``new_token`` comparison
        # below is deterministic across the chunked / unchunked runs (any
        # bf16 jitter on the leading logits would otherwise flip the
        # sampled token between the two paths).
        sampling_config={"Thinker": SamplingConfig(temperature=0.0)},
        # ``is_last_prefill=True`` makes ``ThinkerSubmodule.forward`` emit
        # ``logits`` for the final token (so we have something to sample +
        # compare).  ``audio_output=True`` keeps ``thinker_states`` flowing
        # so the output shape matches a real production prefill.
        step_metadata={"audio_output": True, "is_last_prefill": True},
    )
    return NodeBatch(
        node_name="Thinker",
        graph_walk="prefill_text",
        request_ids=[rid],
        per_request_input_tensors={rid: {"text_inputs": [text_ids]}},
        per_request_info={rid: info},
    )


def _extract_request_kv(engine: AREngine, rid: str) -> torch.Tensor:
    """Pull populated KV pages for a request and return a single tensor.

    KV cache layout (from ``AREngine.load_model``):
        ``[num_layers, max_num_pages, 2, page_size, num_kv_heads, head_dim]``
    where dim 2 is K/V split.  For a request with ``seq_len`` tokens spread
    across N ``page_indices`` (each holding ``page_size`` tokens), gather the
    N pages and slice out the populated prefix.

    Returns shape ``[num_layers, 2, seq_len, num_kv_heads, head_dim]``.
    """
    submod_mgmt = engine.submodule_management["Thinker"]
    kv_mgmt = submod_mgmt.kv_management
    kv_cache = kv_mgmt.kv_cache
    page_size = kv_mgmt.kv_cache_config.page_size

    state = kv_mgmt.alloc_manager.get_state(rid, "main")
    seq_len = state.seq_len
    page_indices = state.page_indices
    assert seq_len > 0, f"request {rid} has empty KV state"
    assert len(page_indices) >= (seq_len + page_size - 1) // page_size

    # Gather pages: shape [num_layers, num_pages, 2, page_size, kv_heads, head_dim].
    pages = kv_cache[:, page_indices, :, :, :, :]
    # Concatenate along token axis (dim 3): [num_layers, 2, num_pages*page_size, kv_heads, head_dim].
    flat = pages.permute(0, 2, 1, 3, 4, 5).contiguous()
    flat = flat.reshape(
        flat.shape[0], flat.shape[1],
        flat.shape[2] * flat.shape[3],
        flat.shape[4], flat.shape[5],
    )
    return flat[:, :, :seq_len, :, :].contiguous()


class _LogitCaptureSampler:
    """Wraps the engine's ``Sampler`` to record the last logits passed in.

    The engine deletes ``logits`` from the per-rid output dict after sampling
    (see ``AREngine._sample_decode_outputs``), so by the time
    ``execute_batch`` returns, raw logits are gone.  Patching ``sampler.sample``
    to clone the input logits captures them without otherwise altering
    behavior.  Restored to the original after each test.
    """

    def __init__(self, sampler):
        self._sampler = sampler
        self._orig_sample = sampler.sample
        self.last_logits: torch.Tensor | None = None

        def _patched(request_ids, logits, *args, **kwargs):
            # Logits passed in is the last-position logits for each rid.
            self.last_logits = logits.detach().clone()
            return self._orig_sample(request_ids, logits, *args, **kwargs)

        sampler.sample = _patched

    def restore(self):
        self._sampler.sample = self._orig_sample


@pytest.mark.parametrize("prompt_len", [600, 1024, 2048])
@pytest.mark.parametrize("chunk_size", [256, 512])
def test_chunked_prefill_matches_unchunked(thinker_engine, prompt_len: int, chunk_size: int):
    """Chunked prefill must produce the same final-position logits, sampled
    token, and KV cache contents as a single-pass unchunked prefill.
    """
    engine, device = thinker_engine

    text_ids = _make_text_input_ids(prompt_len, device, seed=0)

    rid_unchunked = f"unchunked_{uuid.uuid4().hex[:8]}"
    rid_chunked = f"chunked_{uuid.uuid4().hex[:8]}"

    sampler = engine.submodule_management["Thinker"].sampler
    capture = _LogitCaptureSampler(sampler)
    try:
        # ---- Unchunked baseline ----
        engine.max_prefill_chunk_size = None
        engine.add_request(rid_unchunked, ["main"])
        try:
            batch_a = _make_prefill_text_batch(rid_unchunked, text_ids)
            out_a = engine.execute_batch(batch_a)
            assert not out_a.allocation_failed
            assert capture.last_logits is not None, (
                "sampler.sample never invoked — is_last_prefill flag dropped?"
            )
            logits_a = capture.last_logits.flatten().clone()
            tok_a = out_a.per_request_output_tensors[rid_unchunked]["new_token"][0].flatten()[0].clone()
            kv_a = _extract_request_kv(engine, rid_unchunked).clone()

            # ---- Chunked ----
            capture.last_logits = None
            engine.max_prefill_chunk_size = chunk_size
            engine.add_request(rid_chunked, ["main"])
            try:
                batch_b = _make_prefill_text_batch(rid_chunked, text_ids)
                out_b = engine.execute_batch(batch_b)
                assert not out_b.allocation_failed
                assert capture.last_logits is not None, (
                    "sampler.sample not invoked on chunked path"
                )
                logits_b = capture.last_logits.flatten().clone()
                tok_b = out_b.per_request_output_tensors[rid_chunked]["new_token"][0].flatten()[0].clone()
                kv_b = _extract_request_kv(engine, rid_chunked).clone()

                # ---- Asserts ----
                # KV state should match: both runs wrote the same prompt.
                assert kv_a.shape == kv_b.shape, (
                    f"KV shape mismatch: unchunked {tuple(kv_a.shape)} "
                    f"vs chunked {tuple(kv_b.shape)}"
                )
                kv_max_abs = (kv_a - kv_b).abs().max().item()
                kv_a_scale = max(kv_a.abs().max().item(), 1e-6)
                kv_rel = kv_max_abs / kv_a_scale

                # Logits: final-position logits should be ~identical.
                assert logits_a.shape == logits_b.shape, (
                    f"logits shape mismatch: {tuple(logits_a.shape)} vs "
                    f"{tuple(logits_b.shape)}"
                )
                logits_max_abs = (logits_a - logits_b).abs().max().item()
                logits_a_scale = max(logits_a.abs().max().item(), 1e-6)
                logits_rel = logits_max_abs / logits_a_scale

                print(
                    f"\nprompt_len={prompt_len} chunk_size={chunk_size}: "
                    f"logits max_abs={logits_max_abs:.4e} rel={logits_rel:.4e}; "
                    f"KV max_abs={kv_max_abs:.4e} rel={kv_rel:.4e}; "
                    f"tok unchunked={tok_a.item()} chunked={tok_b.item()}"
                )

                torch.testing.assert_close(
                    logits_a, logits_b, atol=1e-2, rtol=1e-2,
                )
                assert torch.equal(tok_a, tok_b), (
                    f"greedy token differs: unchunked={tok_a.item()} "
                    f"vs chunked={tok_b.item()}"
                )
                torch.testing.assert_close(
                    kv_a, kv_b, atol=1e-2, rtol=1e-2,
                )
            finally:
                engine.remove_request(rid_chunked)
        finally:
            engine.remove_request(rid_unchunked)
    finally:
        capture.restore()
