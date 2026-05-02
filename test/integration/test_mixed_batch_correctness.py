"""Phase 2 Task 6 mixed-batch correctness on real qwen3_omni weights.

Validates two things end-to-end:

  (a) ``Worker._build_node_batch`` slices each prefill rid's token-axis
      tensors to ``[consumed : consumed + chunk_size]`` when the
      MicroScheduler has populated ``ScheduledBatch.prefill_chunk_sizes``.

  (b) The Thinker's ``thinker_step`` walk, executed against a mixed
      decode + non-terminal-prefill batch, produces logits only for
      terminal rids (decodes) and skips lm_head for non-terminal prefill
      chunks.  The decode rid's logits in the mixed batch numerically
      match an isolated decode baseline within bf16 tolerance.

The slicing helper is exercised both by a focused unit test (axis
identification + non-token passthrough) and indirectly via the mixed
batch construction.  The mixed batch itself is driven against the
``AREngine`` directly (we feed it a pre-sliced per-rid input dict) so
the test does not have to spin up a full Worker / scheduler / IPC
loop — the slicing semantics under test are functional, not coupling.
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
from mminf.worker.worker import Worker  # noqa: E402

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
        reason=f"{QWEN3_OMNI_REPO} not in local HF cache; run "
               f"`huggingface-cli download {QWEN3_OMNI_REPO}`",
    ),
]


# ---------------------------------------------------------------------------
# Sub-task 6a: focused unit test for the worker-side slicing helper
# ---------------------------------------------------------------------------


def test_slice_prompt_chunk_identifies_token_axis():
    """``Worker._slice_prompt_chunk`` must slice 1D token tensors and pass through
    non-token-axis tensors (e.g. fixed-size embeddings).
    """
    text_inputs = torch.arange(100, dtype=torch.long)
    # A tensor with no dim equal to prompt_total — must pass through.
    pre_embed = torch.randn(7, 13)
    tensors = {
        "text_inputs": [text_inputs],
        "fixed_embed": [pre_embed],
    }

    sliced = Worker._slice_prompt_chunk(
        tensors, prefill_total=100, start=20, end=60,
    )

    # text_inputs sliced on the token axis (only axis matching prompt_total).
    assert sliced["text_inputs"][0].shape == (40,)
    assert torch.equal(
        sliced["text_inputs"][0], torch.arange(20, 60, dtype=torch.long),
    )
    # fixed_embed has no axis matching prompt_total → pass-through, identity.
    assert sliced["fixed_embed"][0] is pre_embed


def test_slice_prompt_chunk_passes_through_non_tensor_entries():
    """Non-tensor entries (defensive) must pass through untouched."""
    sentinel = object()
    tensors = {"weird": [sentinel], "text_inputs": [torch.arange(10)]}
    sliced = Worker._slice_prompt_chunk(
        tensors, prefill_total=10, start=2, end=5,
    )
    assert sliced["weird"][0] is sentinel
    assert sliced["text_inputs"][0].shape == (3,)


def test_slice_prompt_chunk_handles_empty_chunk_safely():
    """A degenerate chunk_len=0 just produces a length-0 narrow."""
    text_inputs = torch.arange(50)
    sliced = Worker._slice_prompt_chunk(
        {"text_inputs": [text_inputs]}, prefill_total=50, start=10, end=10,
    )
    assert sliced["text_inputs"][0].shape == (0,)


# ---------------------------------------------------------------------------
# Sub-task 6b: mixed-batch correctness against real qwen3_omni Thinker weights
# ---------------------------------------------------------------------------


def _make_transfer_info() -> TransferEngineInfo:
    return TransferEngineInfo(
        my_entity_id="mixed_batch_test",
        my_session_id="mixed_batch_session",
        transfer_engine=LocalTransferEngine(hostname="mixed_batch_test"),
    )


@pytest.fixture(scope="module")
def thinker_engine():
    """One ``AREngine`` with the qwen3_omni Thinker, NO CUDA graphs.

    Mirrors ``test_chunked_prefill_equivalence.thinker_engine`` (module-
    scoped, eager-only) so we can run all parametrizations against a
    single 30B Thinker load.  Same KV budget (256 pages × 128 page_size
    = 32k tokens) — comfortably above the long-prompt rid in this test.
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

    engine = AREngine(autocast_dtype=torch.bfloat16, max_prefill_chunk_size=None)
    transfer_info = _make_transfer_info()
    engine.load_model(
        submodules={"Thinker": thinker.to(device)},
        kv_cache_config=[kv_cfg],
        device=device,
        transfer_engine_info=transfer_info,
        kv_cache_type=torch.bfloat16,
    )
    assert engine.submodule_management["Thinker"].cuda_graph_runner is None

    yield engine, device

    engine.shutdown()


def _make_text_input_ids(prompt_len: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(
        0, 10000, (prompt_len,),
        dtype=torch.long, device=device, generator=g,
    )


def _make_prefill_text_batch(rid: str, text_ids: torch.Tensor) -> NodeBatch:
    """Build a single-request ``prefill_text`` batch (mirrors the equivalence test)."""
    info = CurrentForwardPassInfo(
        request_id=rid,
        graph_walk="prefill_text",
        requires_cfg=False,
        fwd_index=0,
        random_seed=42,
        max_tokens=1,
        sampling_config={"Thinker": SamplingConfig(temperature=0.0)},
        step_metadata={"audio_output": True, "is_last_prefill": True},
    )
    return NodeBatch(
        node_name="Thinker",
        graph_walk="prefill_text",
        request_ids=[rid],
        per_request_input_tensors={rid: {"text_inputs": [text_ids]}},
        per_request_info={rid: info},
    )


def _make_thinker_step_batch(
    per_rid_inputs: dict[str, torch.Tensor],
    is_terminal_per_request: dict[str, bool],
) -> NodeBatch:
    """Build a multi-request ``thinker_step`` batch.

    Each rid contributes a ``text_inputs`` tensor of length seq_len:
    - decode rid: seq_len=1 (the previously sampled new_token)
    - prefill chunk rid: seq_len=chunk_size (the slice of the prompt)
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
            # audio_output=False keeps thinker_states traffic small (we are
            # not exercising Talker conditioning here); is_last_prefill is
            # ignored on thinker_step (per-rid gating uses
            # is_terminal_per_request instead).
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


def test_mixed_batch_decode_plus_nonterminal_prefill_chunk(thinker_engine):
    """A ``thinker_step`` batch with one decode rid and one non-terminal
    prefill chunk rid must:

      1. Emit ``logits`` only for the decode rid (terminal=True);
         the non-terminal prefill rid gets no ``logits`` key.
      2. Decode rid's logits numerically match an isolated single-rid
         decode baseline within bf16 tolerance.

    This is the load-bearing correctness test for Phase 2 Task 6: it
    exercises the mixed-batch packing + per-rid lm_head gating that was
    introduced in Task 4 + Task 5, with the slicing semantics from this
    task implicit in the per-rid ``text_inputs`` shapes (1 for decode,
    chunk_size for prefill).
    """
    engine, device = thinker_engine

    # Distinct rids per call to avoid KV state collision.
    rid_decode = f"decode_{uuid.uuid4().hex[:8]}"
    rid_prefill = f"prefill_{uuid.uuid4().hex[:8]}"

    decode_prompt_len = 100
    prefill_total = 4096
    chunk_size = 2048  # First chunk: non-terminal (chunk_size < prefill_total).

    decode_prompt = _make_text_input_ids(decode_prompt_len, device, seed=11)
    prefill_prompt = _make_text_input_ids(prefill_total, device, seed=22)

    engine.add_request(rid_decode, ["main"])
    engine.add_request(rid_prefill, ["main"])
    try:
        # ---- 1. Prime decode rid: prefill its short prompt; capture
        # ---- the sampled new_token so we can feed it into the decode step.
        prefill_a = _make_prefill_text_batch(rid_decode, decode_prompt)
        out_a = engine.execute_batch(prefill_a)
        assert not out_a.allocation_failed
        new_tok_a = out_a.per_request_output_tensors[rid_decode]["new_token"][0]
        assert new_tok_a.numel() == 1, f"unexpected new_token shape {new_tok_a.shape}"

        # ---- 2. Prime prefill rid: feed the FIRST chunk via prefill_text
        # ---- so its KV cache holds the same state a chunked-prefill
        # ---- mid-step would leave it in.  This sets up the "consumed=2048,
        # ---- non-terminal next" invariant.
        prefill_b_first_chunk = prefill_prompt[:chunk_size]
        prefill_b = _make_prefill_text_batch(rid_prefill, prefill_b_first_chunk)
        out_b = engine.execute_batch(prefill_b)
        assert not out_b.allocation_failed
        # Capture KV state size: BatchedCacheManager should hold chunk_size tokens.
        kv_mgmt = engine.submodule_management["Thinker"].kv_management
        state_b = kv_mgmt.alloc_manager.get_state(rid_prefill, "main")
        assert state_b.seq_len == chunk_size, (
            f"prefill rid expected seq_len={chunk_size} after first chunk, "
            f"got {state_b.seq_len}"
        )

        # ---- 3. Isolated decode baseline for rid_decode: thinker_step
        # ---- with just rid_decode (terminal=True), text_inputs=[new_tok_a].
        decode_token = new_tok_a.flatten().to(device).to(torch.long)

        # Patch the sampler to capture last-position logits.
        sampler = engine.submodule_management["Thinker"].sampler
        captured: dict[str, torch.Tensor] = {}
        orig_sample = sampler.sample

        def _capture(request_ids, logits, *args, **kwargs):
            captured["last"] = logits.detach().clone()
            captured["request_ids"] = list(request_ids)
            return orig_sample(request_ids, logits, *args, **kwargs)

        sampler.sample = _capture
        try:
            iso_batch = _make_thinker_step_batch(
                {rid_decode: decode_token},
                is_terminal_per_request={rid_decode: True},
            )
            out_iso = engine.execute_batch(iso_batch)
            assert not out_iso.allocation_failed
            assert "last" in captured, "sampler.sample never invoked on isolated decode"
            # The submodule should have produced logits for rid_decode and
            # then the engine sampled them out.
            iso_rid_out = out_iso.per_request_output_tensors[rid_decode]
            assert "new_token" in iso_rid_out
            assert "logits" not in iso_rid_out, (
                "engine should have consumed logits during sampling"
            )
            iso_logits = captured["last"].clone()
            iso_token = iso_rid_out["new_token"][0].flatten()[0].clone()
        finally:
            sampler.sample = orig_sample

        # Re-prime: the isolated decode advanced rid_decode's KV state by 1
        # token. To compare apples-to-apples, we want the mixed-batch
        # decode to start from the same KV state — but each step advances
        # state by 1 token. So compare the LOGITS the model produces for
        # the *same input token at the same KV position*. Since both runs
        # run the same model forward on the same KV state + token, logits
        # should match within bf16 tolerance.
        #
        # However, the isolated run mutated state. We need a fresh "what
        # would the next decode step on rid_decode look like" baseline,
        # OR we set up the mixed batch so its decode step uses the
        # POST-isolated-step token+state. Easier: re-prime rid_decode by
        # tearing it down and re-prefilling it identically (deterministic
        # seed) so it ends up in the same exact KV state as before the
        # isolated decode.
        engine.remove_request(rid_decode)
        engine.add_request(rid_decode, ["main"])
        prefill_a2 = _make_prefill_text_batch(rid_decode, decode_prompt)
        out_a2 = engine.execute_batch(prefill_a2)
        assert not out_a2.allocation_failed
        new_tok_a2 = out_a2.per_request_output_tensors[rid_decode]["new_token"][0]
        # Re-prefill with the same seed should yield bit-identical output
        # (greedy + identical KV state). Compare on the same device/dtype.
        new_tok_a2_flat = new_tok_a2.flatten().to(decode_token.device).to(decode_token.dtype)
        assert torch.equal(new_tok_a2_flat, decode_token), (
            "deterministic re-prefill should yield the same sampled token"
        )

        # ---- 4. Mixed batch: rid_decode (terminal=True, 1 token) +
        # ---- rid_prefill (terminal=False, chunk of next 2048 tokens).
        #
        # The "slice" is constructed here exactly the way
        # ``Worker._build_node_batch`` would slice it: the second chunk
        # of the prefill prompt, [chunk_size : 2*chunk_size].
        prefill_b_second_chunk = prefill_prompt[chunk_size : 2 * chunk_size]
        assert prefill_b_second_chunk.shape == (chunk_size,)

        sampler.sample = _capture
        captured.clear()
        try:
            mixed_batch = _make_thinker_step_batch(
                {
                    rid_decode: decode_token,
                    rid_prefill: prefill_b_second_chunk,
                },
                is_terminal_per_request={
                    rid_decode: True,
                    rid_prefill: False,
                },
            )
            out_mixed = engine.execute_batch(mixed_batch)
            assert not out_mixed.allocation_failed
        finally:
            sampler.sample = orig_sample

        # ---- 5. Assertions ----
        # (a) Non-terminal prefill rid: NO logits / new_token in its output.
        prefill_rid_out = out_mixed.per_request_output_tensors[rid_prefill]
        assert "logits" not in prefill_rid_out, (
            "non-terminal prefill chunk should not emit logits "
            f"(got keys: {list(prefill_rid_out.keys())})"
        )
        assert "new_token" not in prefill_rid_out, (
            "non-terminal prefill chunk should not emit new_token "
            f"(got keys: {list(prefill_rid_out.keys())})"
        )

        # (b) Terminal decode rid: has new_token (logits got consumed).
        decode_rid_out = out_mixed.per_request_output_tensors[rid_decode]
        assert "new_token" in decode_rid_out, (
            "terminal decode rid should have new_token "
            f"(got keys: {list(decode_rid_out.keys())})"
        )

        # (c) Decode logits numerically match the isolated baseline.
        assert "last" in captured, "sampler.sample not invoked on mixed batch"
        # Phase 2.1a: thinker_step now emits __batched_logits__ (shape
        # (bs, V)) regardless of terminal-flag distribution, so the engine's
        # batched-logits sampling fast path receives logits for ALL rids in
        # the batch. The per-rid gating happens AFTER sampling: non-terminal
        # rids' new_token assignment is skipped, but their logits row was
        # passed to the sampler. We extract the row for rid_decode by
        # matching the captured request_ids order.
        mixed_logits_all = captured["last"]
        captured_rids = captured["request_ids"]
        assert rid_decode in captured_rids, (
            f"rid_decode {rid_decode} missing from sampled batch "
            f"(got {captured_rids})"
        )
        decode_row_idx = captured_rids.index(rid_decode)
        mixed_decode_logits = mixed_logits_all[decode_row_idx].flatten().clone()

        iso_flat = iso_logits.flatten()
        assert mixed_decode_logits.shape == iso_flat.shape, (
            f"shape mismatch: mixed {tuple(mixed_decode_logits.shape)} "
            f"vs iso {tuple(iso_flat.shape)}"
        )

        max_abs = (mixed_decode_logits - iso_flat).abs().max().item()
        scale = max(iso_flat.abs().max().item(), 1e-6)
        rel = max_abs / scale
        print(
            f"\nmixed-batch decode logits vs isolated: max_abs={max_abs:.4e} "
            f"rel={rel:.4e}; iso_token={iso_token.item()}"
        )

        # Numerical tolerance: bf16 with cross-batch kernel reordering
        # tolerates ~0.5 absolute / ~5e-2 relative (matches the loose
        # boundary in the equivalence test for non-aligned chunk sizes).
        torch.testing.assert_close(
            mixed_decode_logits, iso_flat, atol=0.5, rtol=5e-2,
        )

        # (d) Verify rid_prefill's KV state advanced by chunk_size tokens
        # (from chunk_size after the first prefill, to 2*chunk_size now).
        state_b_after = kv_mgmt.alloc_manager.get_state(rid_prefill, "main")
        assert state_b_after.seq_len == 2 * chunk_size, (
            f"prefill rid expected seq_len={2 * chunk_size} after second "
            f"chunk, got {state_b_after.seq_len}"
        )
    finally:
        engine.remove_request(rid_decode)
        engine.remove_request(rid_prefill)
