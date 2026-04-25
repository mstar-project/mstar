"""Parity test: Qwen3-Omni Thinker prefill_text via CUDA graph vs eager.

For each (bs, num_tokens) bucket captured by the Thinker prefill_text graph,
synthesize identical batched inputs, run them through both the eager
``forward_batched`` path and the CUDA-graph ``runner.run`` path, and assert
that the raw ``__batched_logits__`` + ``__batched_thinker_states__`` agree
within bf16 numerical tolerance (≤ 1e-2 relative per plan §6.2).

Why bypass ``engine.warmup()``: it calls ``_compile_submodules`` *after*
graph capture, which would leave the captured graph using uncompiled
``forward_batched`` while subsequent direct eager calls use the compiled
version — not apples-to-apples. Instead we construct ``CudaGraphRunner``
manually so both paths run through identical kernels and the only
difference being measured is graph capture/replay vs direct call.

Why read ``static_outputs`` directly: ``runner.run`` returns the post-
``_sample_and_remap`` per-rid dict (sampled tokens + sliced thinker_states),
which is the production shape but loses the raw logits we need for parity.
The static buffers hold the raw post-replay tensors and are not overwritten
between calls, so we can re-read them after ``runner.run`` returns.

Run locally::

    huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct
    pytest test/integration/test_prefill_cuda_graph.py -v -s
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mminf.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mminf.engine.ar_engine import AREngine  # noqa: E402
from mminf.engine.cuda_graph_runner import CudaGraphKey, CudaGraphRunner  # noqa: E402
from mminf.engine.kv_store import TransferEngineInfo  # noqa: E402
from mminf.model.submodule_base import ARNodeInputs, ModelInputsFromEngine  # noqa: E402

QWEN3_OMNI_REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def _hf_cache_has_qwen3_omni() -> bool:
    """Return True if Qwen3-Omni snapshots are already on local disk.

    Checks the standard HF cache locations (HF_HOME, HF_HUB_CACHE, plus the
    ~/.cache fallback) so the test can self-skip on machines without the
    ~60 GB Qwen3-Omni download.
    """
    candidates: list[Path] = []
    for env_key in ("HF_HOME", "HF_HUB_CACHE"):
        if env_key in os.environ:
            base = Path(os.environ[env_key])
            candidates.extend([base, base / "hub"])
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
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


class _StubTransferEngine:
    """Minimal stand-in for the Mooncake TransferEngine.

    The real engine handles RDMA registration and inter-worker reads. For a
    single-process parity test we never trigger any cross-worker transfers,
    so a stub that records ``register_memory`` calls and returns ``None``
    from ``get_async_reader`` (single-node SHM-style path) is enough.
    """

    def __init__(self):
        self.registered: list[tuple[int, int]] = []

    def register_memory(self, ptr: int, nbytes: int) -> int:
        self.registered.append((ptr, nbytes))
        return 0

    def unregister_memory(self, ptr: int) -> int:  # noqa: ARG002
        return 0

    def get_async_reader(self, device):  # noqa: ARG002
        return None

    def batch_transfer_sync_read(self, *args, **kwargs):
        raise RuntimeError("stub: no transfers expected in this test")


@pytest.fixture(scope="session")
def thinker_engine_with_runner():
    """Bring up the Thinker submodule on GPU and capture its CUDA graphs.

    Session-scoped because the warmup capture (~50 s on H100 across the 20
    Thinker captures) dominates wall time. All tests share one engine.

    Manually constructs the CudaGraphRunner instead of calling
    ``engine.warmup()`` to avoid the post-capture ``_compile_submodules``
    step, which would create a compile-vs-uncompile divergence between the
    captured graph and subsequent direct eager calls.
    """
    from mminf.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    # The runner calls torch.cuda.set_device(self.device) inside the capture
    # path, which refuses a bare torch.device("cuda") without an index.
    # Production workers always pass cuda:N explicitly; mirror that here.
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    cache_dir = os.environ.get("QWEN3_OMNI_CACHE_DIR")  # optional override

    model = Qwen3OmniModel(model_path_hf=QWEN3_OMNI_REPO, cache_dir=cache_dir)
    thinker = model.get_submodule("Thinker", device=str(device))
    assert thinker is not None, "Thinker submodule failed to load"

    # Pull the Thinker KV config out of the full list (model returns 3:
    # Thinker, Talker, code_predictor).
    kv_cfgs = [c for c in model.get_kv_cache_config() if c.nodes and "Thinker" in c.nodes]
    assert len(kv_cfgs) == 1, f"expected 1 Thinker KV config, got {len(kv_cfgs)}"
    kv_cfg = kv_cfgs[0]
    # Bound the KV cache to what this test needs: capture allocates pages
    # for padded_bs (4) × max_num_tokens (2048) = 8192 tokens = 64 pages,
    # plus eager+graph each need the same again at replay time. 256 pages
    # × 128 page_size = 32768 tokens leaves comfortable headroom.
    kv_cfg.max_num_pages = 256

    engine = AREngine(autocast_dtype=torch.bfloat16)
    transfer_info = TransferEngineInfo(
        my_entity_id="parity_test",
        my_session_id="parity_session",
        transfer_engine=_StubTransferEngine(),
    )
    engine.load_model(
        submodules={"Thinker": thinker.to(device)},
        kv_cache_config=[kv_cfg],
        device=device,
        transfer_engine_info=transfer_info,
        kv_cache_type=torch.bfloat16,
    )

    submod_mgmt = engine.submodule_management["Thinker"]
    kv_mgmt = submod_mgmt.kv_management
    runner = CudaGraphRunner(
        submodule_name="Thinker",
        submodule=submod_mgmt.submodule,
        kv_cache_config=kv_mgmt.kv_cache_config,
        alloc_manager=kv_mgmt.alloc_manager,
        sampler=submod_mgmt.sampler,
        buffer_manager=kv_mgmt.buffer_manager,
        device=device,
        autocast_dtype=torch.bfloat16,
    )
    runner.warmup_and_capture()
    assert runner.graphs, "warmup_and_capture produced no captured graphs"
    submod_mgmt.cuda_graph_runner = runner

    yield engine, runner, submod_mgmt.submodule

    # Best-effort teardown — release the KV cache + remove dangling rids
    # before the session ends so memory is freed for the next pytest run.
    engine.shutdown()


def _make_inputs(
    bs: int,
    total_tokens: int,
    hidden_size: int,
    device: torch.device,
    seed: int,
) -> tuple[list[str], list[ARNodeInputs]]:
    """Build bs ARNodeInputs whose seq_lens sum to total_tokens.

    CudaGraphKey.num_tokens is the TOTAL across the batch — set by
    FlashInferPackedCudaGraphConfig.get_total_tokens which returns the
    keys of packed_seq_len_to_inputs, the captured token-bucket dict.
    Splitting total_tokens evenly across bs requests (with remainder on
    the last) keeps the test inputs lined up with what was captured.
    Random bf16 embeds via a seeded generator so eager and graph paths
    rebuild the same inputs independently.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    request_ids = [f"req_{uuid.uuid4().hex[:8]}" for _ in range(bs)]
    base = total_tokens // bs
    seq_lens = [base] * bs
    seq_lens[-1] += total_tokens - sum(seq_lens)
    inputs: list[ARNodeInputs] = []
    for sl in seq_lens:
        embeds = torch.randn(
            (sl, hidden_size),
            dtype=torch.bfloat16, device=device, generator=g,
        )
        # 3-row position grid (temporal/h/w) — same shape ThinkerSubmodule.
        # prepare_inputs builds via get_rope_index_text on a text prefill.
        pos_ids = torch.arange(
            sl, dtype=torch.float, device=device,
        ).unsqueeze(0).expand(3, -1).contiguous()
        # masks_for_talker: simple text-only (multimodal row=0, text row=1).
        masks = torch.stack([
            torch.zeros(sl, dtype=torch.bool, device=device),
            torch.ones(sl, dtype=torch.bool, device=device),
        ])
        inputs.append(ARNodeInputs(
            input_seq_len=sl,
            input_embeds=embeds,
            custom_pos_ids=pos_ids,
            tensor_inputs={"masks_for_talker": masks},
        ))
    return request_ids, inputs


def _make_per_request_info(request_ids: list[str]) -> dict[str, CurrentForwardPassInfo]:
    return {
        rid: CurrentForwardPassInfo(
            request_id=rid,
            graph_walk="prefill_text",
            requires_cfg=False,
            fwd_index=0,
            random_seed=42,
            max_tokens=1,
            sampling_config={},
            step_metadata={"audio_output": True, "is_last_prefill": True},
        )
        for rid in request_ids
    }


def _run_eager_per_rid(
    engine: AREngine,
    submodule,
    request_ids: list[str],
    inputs: list[ARNodeInputs],
    per_request_info: dict[str, CurrentForwardPassInfo],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-rid sequential eager prefill — the production eager path.

    AREngine._execute_sequential calls submodule.forward in a per-rid loop
    for prefill_text (can_batch returns False for that walk). We can't use
    forward_batched here: it asserts cache_manager.get_qo_indptr_buf("main")
    is non-None, which only holds for the runner's static cache manager.
    A fresh BatchedCacheManager from _create_cache_manager has no CUDA-graph
    wrapper attached, so the assert fires before any model code runs.

    Returns (concatenated last-token logits (bs, V), concatenated thinker_states
    (total_tokens, 2*hidden)) — same shapes the graph path emits via the
    __batched_logits__ + __batched_thinker_states__ sentinels.
    """
    logits_chunks: list[torch.Tensor] = []
    states_chunks: list[torch.Tensor] = []
    for rid, inp in zip(request_ids, inputs, strict=True):
        cache_mgr = engine._create_cache_manager([rid], "Thinker")
        engine_inputs = ModelInputsFromEngine(
            request_ids=[rid],
            per_request_info={rid: per_request_info[rid]},
            cache_manager=cache_mgr,
        )
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                preprocessed = submodule.preprocess(
                    graph_walk="prefill_text",
                    engine_inputs=engine_inputs,
                    inputs=[inp],
                )
                out = submodule.forward(
                    graph_walk="prefill_text",
                    engine_inputs=engine_inputs,
                    **preprocessed,
                )
        logits_chunks.append(out["logits"][0])           # (1, V)
        states_chunks.append(out["thinker_states"][0])   # (seq_len, 2*hidden)
    return (
        torch.cat(logits_chunks, dim=0),                 # (bs, V)
        torch.cat(states_chunks, dim=0),                 # (total_tokens, 2*hidden)
    )


def _rel_err(actual: torch.Tensor, ref: torch.Tensor) -> float:
    """Max element-wise relative error vs reference (eps-cushioned)."""
    return ((actual - ref).abs() / (ref.abs() + 1e-6)).max().item()


@pytest.mark.parametrize("total_tokens", [128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("bs", [1, 2, 4])
def test_thinker_prefill_text_graph_matches_eager(
    thinker_engine_with_runner, bs: int, total_tokens: int,
):
    """Per-rid sequential eager prefill and CUDA-graph replay should produce
    matching last-token logits and thinker_states within bf16 tolerance
    (≤ 1e-2 relative on logits per plan §6.2).

    ``total_tokens`` is the sum across the batch (matches CudaGraphKey.num_tokens
    semantics — see _make_inputs). For each captured (bs, total_tokens) bucket
    the test distributes total_tokens evenly across bs requests.
    """
    engine, runner, submodule = thinker_engine_with_runner
    device = engine.device
    hidden_size = submodule.config.thinker_hidden_size

    key = CudaGraphKey(
        graph_walk="prefill_text",
        requires_cfg=False,
        bs=bs,
        num_tokens=total_tokens,
    )
    assert key in runner.graphs, f"capture missing for {key}; available: {list(runner.graphs)}"

    eager_rids, eager_inputs = _make_inputs(bs, total_tokens, hidden_size, device, seed=0)
    graph_rids, graph_inputs = _make_inputs(bs, total_tokens, hidden_size, device, seed=0)
    for ei, gi in zip(eager_inputs, graph_inputs, strict=True):
        assert torch.equal(ei.input_embeds, gi.input_embeds)

    for rid in eager_rids + graph_rids:
        engine.add_request(rid, ["main"])
    try:
        eager_per_info = _make_per_request_info(eager_rids)
        eager_logits, eager_states = _run_eager_per_rid(
            engine, submodule, eager_rids, eager_inputs, eager_per_info,
        )

        graph_per_info = _make_per_request_info(graph_rids)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                runner.run(
                    graph_walk="prefill_text",
                    requires_cfg=False,
                    request_ids=graph_rids,
                    inputs=graph_inputs,
                    per_request_info=graph_per_info,
                    submodule=submodule,
                )
        # Re-read the static buffers post-replay. Sampler.sample reads from
        # them but doesn't overwrite, so they hold the raw replay outputs.
        graph_data = runner.graphs[key]
        graph_logits = graph_data.static_outputs["__batched_logits__"][:bs]
        graph_states = graph_data.static_outputs["__batched_thinker_states__"][:total_tokens]

        assert eager_logits.shape == graph_logits.shape, (
            f"logits shape mismatch: eager {tuple(eager_logits.shape)} "
            f"vs graph {tuple(graph_logits.shape)}"
        )
        assert eager_states.shape == graph_states.shape, (
            f"thinker_states shape mismatch: eager {tuple(eager_states.shape)} "
            f"vs graph {tuple(graph_states.shape)}"
        )

        logits_max_abs = (eager_logits - graph_logits).abs().max().item()
        logits_rel = _rel_err(graph_logits, eager_logits)
        states_max_abs = (eager_states - graph_states).abs().max().item()
        states_rel = _rel_err(graph_states, eager_states)
        print(
            f"\nbs={bs} total_tokens={total_tokens}: "
            f"logits max_abs={logits_max_abs:.4e} rel={logits_rel:.4e}; "
            f"thinker_states max_abs={states_max_abs:.4e} rel={states_rel:.4e}"
        )

        assert logits_rel < 1e-2, (
            f"logits relative error {logits_rel:.4e} exceeds bf16 tolerance"
        )
        assert states_rel < 1e-2, (
            f"thinker_states relative error {states_rel:.4e} exceeds bf16 tolerance"
        )
    finally:
        for rid in eager_rids + graph_rids:
            engine.remove_request(rid)


@pytest.mark.parametrize("total_tokens", [128, 1024])
@pytest.mark.parametrize("bs", [1, 4])
def test_thinker_prefill_text_graph_replay_is_deterministic(
    thinker_engine_with_runner, bs: int, total_tokens: int,
):
    """Three replays of the same captured graph with identical inputs should
    produce bit-identical outputs. Sanity check that the runner's state-swap
    logic doesn't introduce drift across calls.

    ``total_tokens`` is the sum across the batch — same semantics as the
    parity test above, matching CudaGraphKey.num_tokens.
    """
    engine, runner, submodule = thinker_engine_with_runner
    device = engine.device
    hidden_size = submodule.config.thinker_hidden_size
    key = CudaGraphKey(
        graph_walk="prefill_text",
        requires_cfg=False,
        bs=bs,
        num_tokens=total_tokens,
    )
    assert key in runner.graphs

    snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(3):
        rids, inputs = _make_inputs(bs, total_tokens, hidden_size, device, seed=0)
        per_info = _make_per_request_info(rids)
        for rid in rids:
            engine.add_request(rid, ["main"])
        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    runner.run(
                        graph_walk="prefill_text",
                        requires_cfg=False,
                        request_ids=rids,
                        inputs=inputs,
                        per_request_info=per_info,
                        submodule=submodule,
                    )
            graph_data = runner.graphs[key]
            snapshots.append((
                graph_data.static_outputs["__batched_logits__"][:bs].clone(),
                graph_data.static_outputs["__batched_thinker_states__"][:total_tokens].clone(),
            ))
        finally:
            for rid in rids:
                engine.remove_request(rid)

    for i in range(1, 3):
        assert torch.equal(snapshots[0][0], snapshots[i][0]), (
            f"trial {i} logits differ from trial 0 — replay non-deterministic"
        )
        assert torch.equal(snapshots[0][1], snapshots[i][1]), (
            f"trial {i} thinker_states differ from trial 0"
        )
