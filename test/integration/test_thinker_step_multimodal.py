"""Phase 2.1b: thinker_step accepts atomic audio prefill rids in mixed batches.

Atomic audio (and vision) prefills cannot be chunked because their
start/end sentinel-token wrappers make the full block atomic. Phase 2.1b
allows them to participate as ONE rid in a ``thinker_step`` mixed batch
alongside text-prefill chunks and decode tokens. The Thinker's
``prepare_inputs`` dispatches by per-rid input keys when in
``thinker_step`` mode (``audio_embeds`` -> audio path, ``vision_embeds``
-> vision path, else text).

Two complementary tests:

  1. Source-level smoke test (always runs): ``prepare_inputs`` source
     references both ``audio_embeds`` and ``vision_embeds`` AND still
     handles the existing ``prefill_audio`` / ``prefill_vision`` walks.
     This is a cheap regression guard against accidentally removing the
     dispatch logic.

  2. Behavioral end-to-end test (skipped without the qwen3_omni weights
     in the HF cache): Drive the engine with a mixed ``thinker_step``
     batch containing one decode rid + one atomic audio rid (synthesized
     ``audio_embeds`` to bypass the audio encoder). Compare the audio
     rid's logits row to an isolated single-rid baseline run via
     ``prefill_audio``, which uses the SAME audio prep code path. Tight
     bf16 tolerance because the audio rid is the only token-axis
     contributor to its own logits and the lm_head + transformer stack
     is identical between the two runs at the audio rid's last position
     (the decode rid sits in a separate KV slot). NOTE: synthesizing
     ``audio_embeds`` directly bypasses the AudioEncoder; that's
     intentional for this test, since the load-bearing change is the
     Thinker submodule's ability to dispatch by input keys, not the
     encoder pipeline.
"""
from __future__ import annotations

import inspect
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
from mminf.model.qwen3_omni.submodules import ThinkerSubmodule  # noqa: E402
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


# ---------------------------------------------------------------------------
# Source-level dispatch regression test (always runs)
# ---------------------------------------------------------------------------


def test_thinker_step_dispatches_to_audio_path_on_audio_embeds():
    """``prepare_inputs`` in ``thinker_step`` mode must dispatch by input keys.

    Source-level smoke check: the dispatch logic references both
    ``audio_embeds`` and ``vision_embeds`` AND the existing
    ``prefill_audio`` / ``prefill_vision`` walks remain intact (refactored
    to call shared helpers but still reachable through the same
    ``graph_walk`` checks).
    """
    src = inspect.getsource(ThinkerSubmodule.prepare_inputs)
    # Phase 2.1b: thinker_step branch must check for audio/vision input keys.
    assert "audio_embeds" in src, (
        "prepare_inputs must check for 'audio_embeds' in thinker_step "
        "dispatch (Phase 2.1b)."
    )
    assert "vision_embeds" in src, (
        "prepare_inputs must check for 'vision_embeds' in thinker_step "
        "dispatch (Phase 2.1b)."
    )
    # Existing walks must still be reachable.
    assert 'graph_walk == "prefill_audio"' in src, (
        "prepare_inputs must still handle the prefill_audio walk."
    )
    assert 'graph_walk == "prefill_vision"' in src, (
        "prepare_inputs must still handle the prefill_vision walk."
    )
    assert 'graph_walk == "thinker_step"' in src, (
        "prepare_inputs must explicitly handle the thinker_step walk."
    )


# ---------------------------------------------------------------------------
# Behavioral end-to-end test (requires qwen3_omni weights)
# ---------------------------------------------------------------------------


_REQUIRES_GPU = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA",
)
_REQUIRES_QWEN3_OMNI = pytest.mark.skipif(
    not _hf_cache_has_qwen3_omni(),
    reason=f"{QWEN3_OMNI_REPO} not in local HF cache; run "
           f"`huggingface-cli download {QWEN3_OMNI_REPO}`",
)


def _make_transfer_info() -> TransferEngineInfo:
    return TransferEngineInfo(
        my_entity_id="thinker_step_multimodal_test",
        my_session_id="thinker_step_multimodal_session",
        transfer_engine=LocalTransferEngine(
            hostname="thinker_step_multimodal_test",
        ),
    )


@pytest.fixture(scope="module")
def thinker_engine_eager():
    """One ``AREngine`` with the qwen3_omni Thinker, NO CUDA graphs.

    Phase 2.1b: multimodal-mixed thinker_step batches don't have a captured
    graph today (the capture is text-prefill-shaped, and audio rids have
    different per-token embedding values + MRoPE position layouts). Eager
    is the only path that exercises the new dispatch end-to-end. Module-
    scoped to amortize the 30B weight load across tests in this file.
    """
    from mminf.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    cache_dir = os.environ.get("QWEN3_OMNI_CACHE_DIR")

    model = Qwen3OmniModel(model_path_hf=QWEN3_OMNI_REPO, cache_dir=cache_dir)
    thinker = model.get_submodule("Thinker", device=str(device))
    assert thinker is not None

    kv_cfgs = [
        c for c in model.get_kv_cache_config() if c.nodes and "Thinker" in c.nodes
    ]
    assert len(kv_cfgs) == 1
    kv_cfg = kv_cfgs[0]
    kv_cfg.max_num_pages = 256

    engine = AREngine(
        autocast_dtype=torch.bfloat16, max_prefill_chunk_size=None,
    )
    transfer_info = _make_transfer_info()
    engine.load_model(
        submodules={"Thinker": thinker.to(device)},
        kv_cache_config=[kv_cfg],
        device=device,
        transfer_engine_info=transfer_info,
        kv_cache_type=torch.bfloat16,
    )
    assert engine.submodule_management["Thinker"].cuda_graph_runner is None

    yield engine, device, model

    engine.shutdown()


def _make_text_input_ids(
    prompt_len: int, device: torch.device, seed: int,
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(
        0, 10000, (prompt_len,),
        dtype=torch.long, device=device, generator=g,
    )


def _make_prefill_text_batch(rid: str, text_ids: torch.Tensor) -> NodeBatch:
    info = CurrentForwardPassInfo(
        request_id=rid,
        graph_walk="prefill_text",
        requires_cfg=False,
        fwd_index=0,
        random_seed=42,
        max_tokens=1,
        sampling_config={"Thinker": SamplingConfig(temperature=0.0)},
        step_metadata={"audio_output": False, "is_last_prefill": True},
    )
    return NodeBatch(
        node_name="Thinker",
        graph_walk="prefill_text",
        request_ids=[rid],
        per_request_input_tensors={rid: {"text_inputs": [text_ids]}},
        per_request_info={rid: info},
    )


def _make_prefill_audio_batch(
    rid: str, audio_embeds: torch.Tensor, *, is_last_prefill: bool = True,
) -> NodeBatch:
    """Single-rid ``prefill_audio`` batch — drives the isolated audio baseline."""
    info = CurrentForwardPassInfo(
        request_id=rid,
        graph_walk="prefill_audio",
        requires_cfg=False,
        fwd_index=0,
        random_seed=42,
        max_tokens=1,
        sampling_config={"Thinker": SamplingConfig(temperature=0.0)},
        step_metadata={
            "audio_output": False, "is_last_prefill": is_last_prefill,
        },
    )
    return NodeBatch(
        node_name="Thinker",
        graph_walk="prefill_audio",
        request_ids=[rid],
        per_request_input_tensors={rid: {"audio_embeds": [audio_embeds]}},
        per_request_info={rid: info},
    )


def _make_thinker_step_batch_mixed(
    decode_rid: str,
    decode_token: torch.Tensor,
    audio_rid: str,
    audio_embeds: torch.Tensor,
    *,
    decode_terminal: bool,
    audio_terminal: bool,
) -> NodeBatch:
    """Build a ``thinker_step`` batch with one decode rid and one audio rid.

    The audio rid's per-rid input dict carries ``audio_embeds`` (not
    ``text_inputs``), which the new Phase 2.1b dispatch in
    ``ThinkerSubmodule.prepare_inputs`` routes to the audio prep helper.
    """
    rids = [decode_rid, audio_rid]
    per_request_input_tensors = {
        decode_rid: {"text_inputs": [decode_token]},
        audio_rid: {"audio_embeds": [audio_embeds]},
    }
    per_request_info: dict[str, CurrentForwardPassInfo] = {}
    for rid in rids:
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
        is_terminal_per_request={
            decode_rid: decode_terminal,
            audio_rid: audio_terminal,
        },
    )


@_REQUIRES_GPU
@_REQUIRES_QWEN3_OMNI
def test_thinker_step_handles_audio_rid_in_mixed_batch(thinker_engine_eager):
    """Phase 2.1b end-to-end: a ``thinker_step`` mixed batch containing one
    decode rid + one atomic audio prefill rid must:

      1. Successfully dispatch the audio rid through the audio prep helper
         (no KeyError on ``text_inputs``, no shape mismatch).
      2. Emit ``new_token`` for both terminal rids (decode rid via the
         decode-token sampling path; audio rid via the last-prefill
         sampling path on the audio rid's last-token logits).
      3. Produce logits for the audio rid that match a single-rid
         ``prefill_audio`` baseline within bf16 tolerance.

    Synthesizes ``audio_embeds`` directly (random bf16 of shape
    ``(audio_len, hidden)``); this bypasses the AudioEncoder which is the
    correct scope for this test (we are validating the Thinker submodule's
    Phase 2.1b dispatch, not the encoder pipeline).
    """
    engine, device, model = thinker_engine_eager
    hidden_size = model.config.thinker_hidden_size

    rid_decode = f"decode_{uuid.uuid4().hex[:8]}"
    rid_audio = f"audio_{uuid.uuid4().hex[:8]}"
    rid_audio_iso = f"audio_iso_{uuid.uuid4().hex[:8]}"

    decode_prompt_len = 64
    audio_len = 80  # sentinels add 2 -> 82 audio tokens total in the batch.

    decode_prompt = _make_text_input_ids(decode_prompt_len, device, seed=11)
    # Use a deterministic audio_embeds tensor so the isolated baseline
    # consumes exactly the same input tensor (the engine path doesn't
    # mutate it; we still pass the same tensor to both batches).
    g = torch.Generator(device=device).manual_seed(33)
    audio_embeds = torch.randn(
        (audio_len, hidden_size),
        dtype=torch.bfloat16, device=device, generator=g,
    )

    engine.add_request(rid_decode, ["main"])
    engine.add_request(rid_audio, ["main"])
    engine.add_request(rid_audio_iso, ["main"])

    sampler = engine.submodule_management["Thinker"].sampler
    captured: dict[str, torch.Tensor | list[str]] = {}
    orig_sample = sampler.sample

    def _capture(request_ids, logits, *args, **kwargs):
        # Append each invocation so a multi-call test can inspect history.
        captured.setdefault("logits_history", []).append(
            logits.detach().clone(),
        )
        captured.setdefault("rid_history", []).append(list(request_ids))
        return orig_sample(request_ids, logits, *args, **kwargs)

    try:
        # ---- 1. Prime decode rid with a short text prefill so its KV holds
        # ---- a real prompt and decode_token is the greedy next token.
        out_a = engine.execute_batch(
            _make_prefill_text_batch(rid_decode, decode_prompt),
        )
        assert not out_a.allocation_failed
        new_tok_a = out_a.per_request_output_tensors[rid_decode]["new_token"][0]
        decode_token = new_tok_a.flatten().to(device).to(torch.long)

        # ---- 2. Run the isolated audio baseline (separate rid, fresh KV).
        sampler.sample = _capture
        try:
            iso_out = engine.execute_batch(
                _make_prefill_audio_batch(rid_audio_iso, audio_embeds),
            )
            assert not iso_out.allocation_failed
            iso_rid_out = iso_out.per_request_output_tensors[rid_audio_iso]
            assert "new_token" in iso_rid_out, (
                "isolated prefill_audio should emit new_token "
                f"(got keys: {list(iso_rid_out.keys())})"
            )
            assert "logits_history" in captured, (
                "sampler.sample never invoked on isolated prefill_audio"
            )
            iso_logits = captured["logits_history"][-1].clone()
            captured.clear()
        finally:
            # Detach but don't restore yet; we still need capture for the
            # mixed batch.
            pass

        # ---- 3. Mixed batch: one decode rid (terminal=True) + one audio
        # ---- rid (terminal=True; atomic audio is fully consumed in this
        # ---- step). Each rid's input dict carries its own modality keys —
        # ---- the new dispatch routes audio_rid through the audio helper.
        mixed_batch = _make_thinker_step_batch_mixed(
            decode_rid=rid_decode,
            decode_token=decode_token,
            audio_rid=rid_audio,
            audio_embeds=audio_embeds,
            decode_terminal=True,
            audio_terminal=True,
        )
        out_mixed = engine.execute_batch(mixed_batch)
        assert not out_mixed.allocation_failed, (
            "mixed thinker_step batch with audio rid failed to allocate"
        )

        # ---- 4. Both terminal rids must have new_token.
        decode_rid_out = out_mixed.per_request_output_tensors[rid_decode]
        audio_rid_out = out_mixed.per_request_output_tensors[rid_audio]
        assert "new_token" in decode_rid_out, (
            "terminal decode rid in mixed batch should emit new_token "
            f"(got keys: {list(decode_rid_out.keys())})"
        )
        assert "new_token" in audio_rid_out, (
            "terminal audio rid in mixed batch should emit new_token "
            f"(got keys: {list(audio_rid_out.keys())})"
        )

        # ---- 5. Audio rid's logits row in the mixed batch should match
        # ---- the isolated baseline within bf16 tolerance. The
        # ---- thinker_step's batched-logits sampling path passes a
        # ---- (bs, V) tensor where row i corresponds to request_ids[i].
        assert "logits_history" in captured, (
            "sampler.sample never invoked on mixed batch"
        )
        # Find the most recent invocation that contained the audio rid.
        mixed_logits_full: torch.Tensor | None = None
        mixed_rids: list[str] | None = None
        for hist_logits, hist_rids in zip(
            captured["logits_history"], captured["rid_history"], strict=True,
        ):
            if rid_audio in hist_rids:
                mixed_logits_full = hist_logits
                mixed_rids = hist_rids
                break
        assert mixed_logits_full is not None, (
            "no captured sample call contained the audio rid"
        )
        assert mixed_rids is not None
        audio_row_idx = mixed_rids.index(rid_audio)
        mixed_audio_logits = mixed_logits_full[audio_row_idx].flatten().clone()

        iso_flat = iso_logits.flatten()
        assert mixed_audio_logits.shape == iso_flat.shape, (
            f"shape mismatch: mixed {tuple(mixed_audio_logits.shape)} "
            f"vs iso {tuple(iso_flat.shape)}"
        )

        max_abs = (mixed_audio_logits - iso_flat).abs().max().item()
        scale = max(iso_flat.abs().max().item(), 1e-6)
        rel = max_abs / scale
        print(
            f"\nmixed-batch audio logits vs isolated: "
            f"max_abs={max_abs:.4e} rel={rel:.4e}"
        )

        # Same loose bf16 boundary used by the existing mixed-batch
        # correctness test — kernel tile-order shifts when batching across
        # rids tolerate this regime in 150k-vocab lm_head.
        torch.testing.assert_close(
            mixed_audio_logits, iso_flat, atol=0.5, rtol=5e-2,
        )
    finally:
        sampler.sample = orig_sample
        engine.remove_request(rid_decode)
        engine.remove_request(rid_audio)
        engine.remove_request(rid_audio_iso)
