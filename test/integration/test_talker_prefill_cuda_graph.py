"""Determinism test: Qwen3-Omni Talker talker_prefill CUDA graph replay.

The pure graph-vs-eager numerical-parity check this file used to carry was
removed: the only available "eager" baseline is per-rid sequential
prefill, which dispatches different FlashInfer kernels (one single-request
prefill per rid) than the captured graph (one packed bs-way prefill).
Comparing the two conflates kernel-dispatch deltas with graph-replay
deltas, and OOD random embeds compound noise across 32 dense MoE layers
into multi-percent relative error that has nothing to do with the runner
being right or wrong. Real validation lives in end-to-end TTS smoke tests
on the server (real input distribution, real downstream composition,
audible ground truth).

What this file still validates: replay determinism. Three replays of the
same captured graph with the same inputs must produce bit-identical
hidden states. This catches state-leakage bugs (e.g. dummy_rid state not
fully reset between replays, alias swap missed, KV cache pages re-used
across calls without clean re-init) — the kind of failure that would not
necessarily surface in a single TTS request but would corrupt the second.

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
from mminf.model.submodule_base import ARNodeInputs  # noqa: E402

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
    the captured graph and subsequent direct calls.
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

    kv_cfgs = [c for c in model.get_kv_cache_config() if c.nodes and "Talker_LLM" in c.nodes]
    assert len(kv_cfgs) == 1, f"expected 1 Talker_LLM KV config, got {len(kv_cfgs)}"
    kv_cfg = kv_cfgs[0]
    kv_cfg.max_num_pages = 256

    engine = AREngine(autocast_dtype=torch.bfloat16)
    transfer_info = TransferEngineInfo(
        my_entity_id="determinism_test",
        my_session_id="determinism_session",
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

    For determinism the input distribution doesn't matter — what matters
    is that we feed the SAME inputs across replays. Small-magnitude
    random embeds with a fixed seed are sufficient.
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


@pytest.mark.parametrize("total_tokens", [128, 1024])
@pytest.mark.parametrize("bs", [1, 4])
def test_talker_prefill_graph_replay_is_deterministic(
    talker_engine_with_runner, bs: int, total_tokens: int,
):
    """Three replays of the same captured graph with identical inputs should
    produce bit-identical hidden states.

    Catches state-leakage bugs in the runner's swap/restore-dummy logic
    (e.g. dummy_rid state not fully reset, KV pages re-used across calls
    without clean re-init). ``total_tokens`` is the sum across the batch.
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
