"""Parity test: Qwen3-Omni Talker talker_prefill via CUDA graph vs eager.

For each (bs, total_tokens) bucket captured by the Talker talker_prefill
graph, synthesize identical batched inputs, run them through both the
eager per-rid path and the CUDA-graph ``runner.run`` path, and assert
the post-LLM hidden states agree within bf16 numerical tolerance.

Why bypass ``engine.warmup()``: same reason as test_prefill_cuda_graph.py
— it calls ``_compile_submodules`` *after* graph capture, which would
leave the captured graph using uncompiled ``forward_batched`` while
subsequent direct eager calls use the compiled version, mixing two
deltas (graph-vs-direct + compiled-vs-eager) into one comparison.

Why a custom eager loop: production ``_execute_sequential`` for
talker_prefill calls ``submodule.forward`` → ``_forward_prefill`` which
runs the model and returns ``{}`` (the talker_prefill walk exists only
to populate the KV cache for the subsequent talker_last_prefill +
talker_decode_loop; no logits are needed). To get hidden states for
parity comparison we inline a ``submodule.model(...)`` call that
mirrors what ``_forward_prefill`` does internally but exposes the
post-LLM hidden state.

Why read ``static_outputs`` directly: ``runner.run`` returns the post-
``_sample_and_remap`` dict — for talker_prefill that's ``{rid: {} for
rid in request_ids}`` since no ``__batched_logits__`` is emitted and no
per-rid sub-dicts exist. The captured forward writes the hidden state
into the ``__batched_talker_prefill_hidden__`` sentinel buffer; we
re-read it post-replay.

Run locally::

    huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct
    pytest test/integration/test_talker_prefill_cuda_graph.py -v -s
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
def talker_engine_with_runner():
    """Bring up the Talker_LLM submodule on GPU and capture its CUDA graphs.

    Session-scoped because the warmup capture (~30 s on H100 across the
    talker_decode + talker_prefill captures) dominates wall time.

    Manually constructs the CudaGraphRunner instead of calling
    ``engine.warmup()`` to avoid the post-capture ``_compile_submodules``
    step, which would create a compile-vs-uncompile divergence between
    the captured graph and subsequent direct eager calls.

    Note: ``Qwen3OmniModel._create_talker_submodule`` will load
    Thinker.embed_tokens temporarily (~620 MB) to compute the cached TTS
    pad/bos/eos embeds — those are only used by talker_decode and
    talker_last_prefill, so talker_prefill works without them, but the
    init runs unconditionally on submodule construction.
    """
    from mminf.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    # The runner calls torch.cuda.set_device(self.device) inside the
    # capture path, which refuses a bare torch.device("cuda") without an
    # index. Production workers always pass cuda:N explicitly.
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    cache_dir = os.environ.get("QWEN3_OMNI_CACHE_DIR")

    model = Qwen3OmniModel(model_path_hf=QWEN3_OMNI_REPO, cache_dir=cache_dir)
    talker = model.get_submodule("Talker_LLM", device=str(device))
    assert talker is not None, "Talker_LLM submodule failed to load"

    # Pull the Talker KV config out of the full list (model returns 3:
    # Thinker, Talker_LLM, code_predictor).
    kv_cfgs = [c for c in model.get_kv_cache_config() if c.nodes and "Talker_LLM" in c.nodes]
    assert len(kv_cfgs) == 1, f"expected 1 Talker_LLM KV config, got {len(kv_cfgs)}"
    kv_cfg = kv_cfgs[0]
    # Bound the KV cache: capture allocates pages for padded_bs (4) ×
    # max_num_tokens (1024) = 4096 tokens. With page_size=128 that's
    # 32 pages; eager+graph each need the same again at replay. 256
    # pages leaves comfortable headroom.
    kv_cfg.max_num_pages = 256

    engine = AREngine(autocast_dtype=torch.bfloat16)
    transfer_info = TransferEngineInfo(
        my_entity_id="parity_test",
        my_session_id="parity_session",
        transfer_engine=_StubTransferEngine(),
    )
    engine.load_model(
        submodules={"Talker_LLM": talker.to(device)},
        kv_cache_config=[kv_cfg],
        device=device,
        transfer_engine_info=transfer_info,
        kv_cache_type=torch.bfloat16,
    )

    submod_mgmt = engine.submodule_management["Talker_LLM"]
    kv_mgmt = submod_mgmt.kv_management
    runner = CudaGraphRunner(
        submodule_name="Talker_LLM",
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

    engine.shutdown()


def _make_inputs(
    bs: int,
    total_tokens: int,
    talker_hidden_size: int,
    device: torch.device,
    seed: int,
) -> tuple[list[str], list[ARNodeInputs]]:
    """Build bs ARNodeInputs whose seq_lens sum to total_tokens.

    CudaGraphKey.num_tokens is the TOTAL across the batch — set by
    FlashInferPackedCudaGraphConfig.get_total_tokens which returns the
    keys of packed_seq_len_to_inputs. Splitting total_tokens evenly
    across bs requests keeps test inputs aligned with capture.

    Talker has no embed_tokens layer (input_embeds come from upstream
    Thinker via text_projection / hidden_projection), so we synthesize
    small-magnitude bf16 random embeds. Magnitude 0.1 keeps activations
    in a bf16-stable range across the 32 Talker layers without needing
    to reproduce the real text_projection chain.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    request_ids = [f"req_{uuid.uuid4().hex[:8]}" for _ in range(bs)]
    base = total_tokens // bs
    seq_lens = [base] * bs
    seq_lens[-1] += total_tokens - sum(seq_lens)

    inputs: list[ARNodeInputs] = []
    for sl in seq_lens:
        embeds = (torch.randn(
            (sl, talker_hidden_size),
            dtype=torch.float32, device=device, generator=g,
        ) * 0.1).to(torch.bfloat16)
        inputs.append(ARNodeInputs(
            input_seq_len=sl,
            input_embeds=embeds,
        ))
    return request_ids, inputs


def _make_per_request_info(request_ids: list[str]) -> dict[str, CurrentForwardPassInfo]:
    return {
        rid: CurrentForwardPassInfo(
            request_id=rid,
            graph_walk="talker_prefill",
            requires_cfg=False,
            fwd_index=0,
            random_seed=42,
            max_tokens=1,
            sampling_config={},
        )
        for rid in request_ids
    }


def _run_eager_per_rid(
    engine: AREngine,
    submodule,
    request_ids: list[str],
    inputs: list[ARNodeInputs],
    per_request_info: dict[str, CurrentForwardPassInfo],
) -> torch.Tensor:
    """Per-rid sequential eager talker_prefill — production path with hidden state captured.

    Production ``_execute_sequential`` calls ``submodule.forward`` →
    ``_forward_prefill`` which runs ``self.model(...)`` and returns
    ``{}`` (the hidden state is computed but discarded — talker_prefill
    only needs to populate the KV cache). For parity testing we need
    the hidden state, so we inline the same call sequence:

      1. ``submodule.preprocess`` plans attention+rope on the cache_mgr
         and returns ``{"input_embeds": packed_tensor}``.
      2. ``submodule.model(input_embeds=..., cache_handle=cache_mgr)``
         runs the LLM and returns the post-norm hidden state — exactly
         what ``_forward_prefill`` calls internally before discarding.

    We can't use ``forward_batched`` here for the same reason as the
    Thinker test: the prefill code paths in forward_batched assume the
    runner's static cache manager (FlashInfer wrappers planned outside
    the graph). A fresh BatchedCacheManager from _create_cache_manager
    has no such wrappers attached.

    Returns concatenated hidden states (total_tokens, talker_hidden) —
    same shape the graph path emits via the
    __batched_talker_prefill_hidden__ sentinel.
    """
    hidden_chunks: list[torch.Tensor] = []
    for rid, inp in zip(request_ids, inputs, strict=True):
        cache_mgr = engine._create_cache_manager([rid], "Talker_LLM")
        engine_inputs = ModelInputsFromEngine(
            request_ids=[rid],
            per_request_info={rid: per_request_info[rid]},
            cache_manager=cache_mgr,
        )
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                preprocessed = submodule.preprocess(
                    graph_walk="talker_prefill",
                    engine_inputs=engine_inputs,
                    inputs=[inp],
                )
                hidden = submodule.model(
                    input_embeds=preprocessed["input_embeds"],
                    cache_handle=cache_mgr,
                )
        hidden_chunks.append(hidden)
    return torch.cat(hidden_chunks, dim=0)


def _rel_err(actual: torch.Tensor, ref: torch.Tensor) -> float:
    """Max-abs error normalized by reference's abs-max scale.

    See test_prefill_cuda_graph.py for rationale (element-wise relative
    blows up near zero; scale-normalized matches intuition).
    """
    ref_scale = max(ref.abs().max().item(), 1e-6)
    return (actual - ref).abs().max().item() / ref_scale


@pytest.mark.parametrize("total_tokens", [128, 256, 512, 1024])
@pytest.mark.parametrize("bs", [1, 2, 4])
def test_talker_prefill_graph_matches_eager(
    talker_engine_with_runner, bs: int, total_tokens: int,
):
    """Numerical agreement between per-rid sequential eager and CUDA-graph replay.

    For talker_prefill there are no logits to compare (the walk emits no
    logits in either path), so the assertion is purely on the post-LLM
    hidden state agreement under the same scale-based relative tolerance
    used by the Thinker thinker_states check (≤ 5e-3 rel against the
    reference's abs-max scale).

    Caveat: same as the Thinker prefill_text test — eager runs bs=1
    single-request FlashInfer prefill kernels per rid; graph runs the
    bs=N packed prefill kernel. These are different kernel dispatches
    even *without* CUDA graphs (no eager-packed prefill path exists in
    this codebase), so this test measures graph-replay AND
    kernel-dispatch deltas together.
    """
    engine, runner, submodule = talker_engine_with_runner
    device = engine.device
    talker_hidden_size = submodule.config.talker_hidden_size

    key = CudaGraphKey(
        graph_walk="talker_prefill",
        requires_cfg=False,
        bs=bs,
        num_tokens=total_tokens,
    )
    assert key in runner.graphs, f"capture missing for {key}; available: {list(runner.graphs)}"

    eager_rids, eager_inputs = _make_inputs(
        bs, total_tokens, talker_hidden_size, device, seed=0,
    )
    graph_rids, graph_inputs = _make_inputs(
        bs, total_tokens, talker_hidden_size, device, seed=0,
    )
    for ei, gi in zip(eager_inputs, graph_inputs, strict=True):
        assert torch.equal(ei.input_embeds, gi.input_embeds)

    for rid in eager_rids + graph_rids:
        engine.add_request(rid, ["main"])
    try:
        eager_per_info = _make_per_request_info(eager_rids)
        eager_hidden = _run_eager_per_rid(
            engine, submodule, eager_rids, eager_inputs, eager_per_info,
        )

        graph_per_info = _make_per_request_info(graph_rids)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                runner.run(
                    graph_walk="talker_prefill",
                    requires_cfg=False,
                    request_ids=graph_rids,
                    inputs=graph_inputs,
                    per_request_info=graph_per_info,
                    submodule=submodule,
                )
        graph_data = runner.graphs[key]
        graph_hidden = graph_data.static_outputs[
            "__batched_talker_prefill_hidden__"
        ][:total_tokens]

        assert eager_hidden.shape == graph_hidden.shape, (
            f"hidden shape mismatch: eager {tuple(eager_hidden.shape)} "
            f"vs graph {tuple(graph_hidden.shape)}"
        )

        hidden_max_abs = (eager_hidden - graph_hidden).abs().max().item()
        hidden_rel = _rel_err(graph_hidden, eager_hidden)
        print(
            f"\nbs={bs} total_tokens={total_tokens}: "
            f"talker_prefill_hidden max_abs={hidden_max_abs:.4e} rel={hidden_rel:.4e}"
        )

        assert hidden_rel < 5e-3, (
            f"talker_prefill_hidden relative error {hidden_rel:.4e} "
            "exceeds 5e-3 tolerance"
        )
    finally:
        for rid in eager_rids + graph_rids:
            engine.remove_request(rid)


@pytest.mark.parametrize("total_tokens", [128, 1024])
@pytest.mark.parametrize("bs", [1, 4])
def test_talker_prefill_graph_replay_is_deterministic(
    talker_engine_with_runner, bs: int, total_tokens: int,
):
    """Three replays of the same captured graph with identical inputs should
    produce bit-identical hidden states.

    Sanity check that the runner's state-swap logic doesn't introduce
    drift across calls. ``total_tokens`` is the sum across the batch.
    """
    engine, runner, submodule = talker_engine_with_runner
    device = engine.device
    talker_hidden_size = submodule.config.talker_hidden_size
    key = CudaGraphKey(
        graph_walk="talker_prefill",
        requires_cfg=False,
        bs=bs,
        num_tokens=total_tokens,
    )
    assert key in runner.graphs

    snapshots: list[torch.Tensor] = []
    for _ in range(3):
        rids, inputs = _make_inputs(
            bs, total_tokens, talker_hidden_size, device, seed=0,
        )
        per_info = _make_per_request_info(rids)
        for rid in rids:
            engine.add_request(rid, ["main"])
        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    runner.run(
                        graph_walk="talker_prefill",
                        requires_cfg=False,
                        request_ids=rids,
                        inputs=inputs,
                        per_request_info=per_info,
                        submodule=submodule,
                    )
            graph_data = runner.graphs[key]
            snapshots.append(
                graph_data.static_outputs[
                    "__batched_talker_prefill_hidden__"
                ][:total_tokens].clone()
            )
        finally:
            for rid in rids:
                engine.remove_request(rid)

    for i in range(1, 3):
        assert torch.equal(snapshots[0], snapshots[i]), (
            f"trial {i} hidden differs from trial 0 — replay non-deterministic"
        )
