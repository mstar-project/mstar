"""THE MISSING RUNG: the MTP piecewise graphs against the REAL runner, on a GPU, in seconds.

Before this file the GLM-5.2 test ladder went

    CPU tests with stub runners  ->  (nothing)  ->  753B on 8 GPUs for 27 minutes

and every bug that escaped lived in the gap. The CPU seam stubs
(``test_glm52_mtp_loop.py``) plan and run EAGERLY on the real cache handle:
they never replay a captured graph, never go through a ``static_cm`` over
aliased dummy request ids, never exercise ``can_run``/``_resolve_key`` or
batch padding, and never touch the shared FlashInfer workspace. They prove
the caller's arithmetic and nothing about the replay machinery.

Two regressions measured on 2026-08-10 lived exactly there, and both were
silent — greedy verify rejects a bad draft, so the only symptom is lower
acceptance, which is indistinguishable from a modelling problem:

- ``mtp_trunk`` and ``mtp_sync`` are both ``(bs, k+1)``, so they took the
  same FlashInfer workspace buffer; ``plan()`` writes scheduling state there
  that the captured replay reads back, so one graph's plan corrupted the
  other's replay.
- the same shape collision gave them identical dummy request ids, and
  ``add_request`` overwrites, so one runner's setup discarded the other's
  dummy state.

Everything here runs on ONE GPU at reduced dims in a few seconds. The
property under test is the one the whole feature rests on: **routing a
decode step through the captured graphs must emit the byte-identical token
stream that the eager path emits.** Plus the check no benchmark makes for
you — that the graphs were actually captured rather than silently serving
eager, which is a 13x regression wearing a correctness costume.

Not covered here (stated so nobody reads a green run as more than it is):
TP>1 collectives, the MLA-absorb kernel path (the reduced config uses the
naive backend, so ``FlashInferMLAWrapper`` and its null-page scatter tail
are NOT exercised), and real-checkpoint numerics.
"""
import os

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cache_manager import WorkspaceBufferManager, create_cache_manager
from mstar.engine.cuda_graph_runner import build_piecewise_runners
from mstar.engine.kv_store import (
    KVCacheConfig,
    PagedAllocationManager,
    TransferEngineInfo,
)
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.submodules import (
    MTP_DRAFT_LABEL,
    MTP_SYNC_LABEL,
    MTP_TRUNK_LABEL,
    Glm52LLMSubmodule,
)
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.utils.sampling import Sampler, SamplingConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="piecewise capture needs a real GPU (CUDA graphs + FlashInfer)",
)

# Indexed on purpose: PiecewiseCudaGraphRunner.warmup_and_capture calls
# torch.cuda.set_device(self.device), which rejects a bare "cuda".
DEVICE = torch.device("cuda:0")
K = 2  # rows per MTP step = k+1 = 3


def _cfg() -> Glm52ModelConfig:
    cfg = Glm52ModelConfig.reduced()
    # 4 puts the MTP layer at a FULL indexer position under the IndexShare
    # formula, which Glm52MTPModule asserts at construction.
    cfg.num_hidden_layers = 4
    cfg.mtp_num_draft_tokens = K
    cfg.dsa_long_context = False
    return cfg


def _build(cfg) -> Glm52ForCausalLM:
    """Reduced model with finite weights. Values are arbitrary — the property
    is eager-vs-replay equality on the SAME weights, not quality."""
    torch.manual_seed(0)
    model = Glm52ForCausalLM(cfg).to(DEVICE).to(torch.bfloat16)
    for name, p in model.named_parameters():
        if p.dtype.is_floating_point:
            if "norm" in name:
                p.data.fill_(1.0)
            else:
                p.data.normal_(0, 0.02)
    for name, p in model.named_parameters():
        # Router bias stays fp32 in production (restore_fp32_params); the
        # submodule's to() refuses dtype casts for exactly this reason.
        if "e_score_correction_bias" in name:
            p.data = p.data.float()
    return model


def _kv(cfg, bs, page_size=128, max_num_pages=64):
    """KV pool with ONE EXTRA layer plane — the MTP module writes at layer
    index num_hidden_layers, sharing the trunk's page table and counter."""
    num_layers = cfg.num_hidden_layers + 1
    kv_cache = torch.zeros(
        num_layers, max_num_pages, 2, page_size,
        cfg.num_attention_heads, cfg.padded_head_dim,
        dtype=torch.bfloat16, device=DEVICE,
    ).contiguous()
    kv_cfg = KVCacheConfig(
        num_layers=num_layers, num_kv_heads=cfg.num_attention_heads,
        head_dim=cfg.padded_head_dim, max_seq_len=page_size * max_num_pages,
        max_num_pages=max_num_pages, page_size=page_size,
        num_qo_heads=cfg.num_attention_heads,
    )
    alloc = PagedAllocationManager(
        config=kv_cfg, kv_cache=kv_cache,
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="mtp_pcgr", my_session_id="mtp_pcgr",
            transfer_engine=LocalTransferEngine("localhost"),
        ),
    )
    rids = [f"r{i}" for i in range(bs)]
    for rid in rids:
        alloc.add_request(rid, ["main"])
    buffers = WorkspaceBufferManager(64 * 1024 * 1024, device=DEVICE)
    cm = create_cache_manager(
        request_ids=rids, active_labels_per_request={r: "main" for r in rids},
        kv_cache=kv_cache, alloc_manager=alloc, buffer_manager=buffers,
        kv_cache_config=kv_cfg, device=DEVICE,
    )
    return cm, alloc, buffers, kv_cfg, rids


def _sampler(cfg, rids):
    s = Sampler(device=DEVICE)
    for rid in rids:
        s.add_request(rid)
        s.set_config(rid, vocab_size=cfg.vocab_size, temperature=0.0,
                     top_k=0, top_p=1.0, repetition_penalty=1.0)
    return s


def _fwd_info(cfg, rid, max_tokens):
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk="decode", requires_cfg=False, fwd_index=0,
        random_seed=0, max_tokens=max_tokens,
        sampling_config={"LLM": SamplingConfig(
            vocab_size=cfg.vocab_size, ignore_eos=True, temperature=0.0,
            repetition_penalty=1.0)},
        dynamic_loop_iter_counts={},
    )


def _drive(cfg, model, prompts, steps, use_graphs):
    """Prefill + `steps` MTP decode steps. Returns per-request token lists.

    ``use_graphs`` picks the ONLY difference between the two arms: whether
    the engine hands the submodule warmed-up piecewise runners.
    """
    bs = len(prompts)
    sub = Glm52LLMSubmodule(model, cfg)
    cm, alloc, buffers, kv_cfg, rids = _kv(cfg, bs)
    sampler = _sampler(cfg, rids)
    runners = {}
    try:
        if use_graphs:
            runners = build_piecewise_runners(
                sub, DEVICE, torch.bfloat16, tp_world_size=1,
                kv_cache_config=kv_cfg, alloc_manager=alloc,
                buffer_manager=buffers,
            )
        ei = ModelInputsFromEngine(
            request_ids=rids, per_request_info={}, cache_manager=cm,
            sampler=sampler, piecewise_runners=runners,
        )
        infos = {rid: _fwd_info(cfg, rid, 4096) for rid in rids}
        out_tokens = {rid: [] for rid in rids}

        ars = [sub.prepare_inputs("prefill", infos[rid], {"text_inputs": [p]})
               for rid, p in zip(rids, prompts, strict=True)]
        packed = sub.preprocess("prefill", ei, ars)
        with torch.no_grad():
            res = sub.forward_batched("prefill", ei, **packed)
        nxt = {}
        for rid in rids:
            out_tokens[rid] += res[rid]["new_token"][0].tolist()
            sub.postprocess(rid, infos[rid], res[rid])
            nxt[rid] = res[rid]["text_inputs"][0]

        for _ in range(steps):
            ars = [sub.prepare_inputs("decode", infos[rid],
                                      {"text_inputs": [nxt[rid]]})
                   for rid in rids]
            packed = sub.preprocess("decode", ei, ars)
            with torch.no_grad():
                res = sub.forward_batched("decode", ei, **packed)
            for rid in rids:
                out_tokens[rid] += res[rid]["new_token"][0].tolist()
                sub.postprocess(rid, infos[rid], res[rid])
                nxt[rid] = res[rid]["text_inputs"][0]
        return out_tokens, runners
    finally:
        alloc.cleanup()
        for rid in rids:
            sampler.remove_request(rid)


def _prompts(bs, n=6):
    return [torch.arange(n, dtype=torch.long, device=DEVICE) + 3 + 5 * i
            for i in range(bs)]


def test_captured_mtp_step_matches_eager_bit_identically():
    """THE property. Same weights, same prompt, two arms: eager, and routed
    through the real captured trunk + draft-chain graphs. Greedy verify makes
    the emitted stream a function of the model alone, so ANY divergence is a
    plumbing bug in the replay path — which is precisely the bug class the
    CPU stubs structurally cannot see."""
    cfg = _cfg()
    model = _build(cfg)
    prompts = _prompts(1)

    # Sync capture is env-default-ON as of 2026-08-11; pin it OFF here so this
    # test isolates the trunk + draft-chain graphs (and covers the
    # CAPTURE_SYNC=0 escape-hatch path). test_sync_capture_matches_eager and
    # test_batch_padding_replay cover the sync-on path (bs=1 and bs=3).
    prev = os.environ.get("MSTAR_GLM52_MTP_CAPTURE_SYNC")
    os.environ["MSTAR_GLM52_MTP_CAPTURE_SYNC"] = "0"
    try:
        eager, _ = _drive(cfg, model, prompts, steps=6, use_graphs=False)
        replay, runners = _drive(cfg, model, prompts, steps=6, use_graphs=True)
    finally:
        if prev is None:
            os.environ.pop("MSTAR_GLM52_MTP_CAPTURE_SYNC", None)
        else:
            os.environ["MSTAR_GLM52_MTP_CAPTURE_SYNC"] = prev

    assert MTP_TRUNK_LABEL in runners, (
        "the trunk graph did not capture — a benchmark would have reported "
        "this as 'MTP is slow' instead of as a failure")
    assert MTP_DRAFT_LABEL in runners, "the draft-chain graph did not capture"
    assert eager["r0"] == replay["r0"], (
        f"captured replay diverged from eager:\n eager  {eager['r0']}\n "
        f"replay {replay['r0']}")


def test_sync_capture_matches_eager_bit_identically():
    """The padded sync pass, opted in. It regressed acceptance 0.76 -> 0.18 on
    the box on 2026-08-10 while its CPU seam test passed, because the stub
    never replayed a real graph and never shared a workspace with the trunk.
    This is the check that would have caught it in seconds."""
    cfg = _cfg()
    model = _build(cfg)
    prompts = _prompts(1)

    eager, _ = _drive(cfg, model, prompts, steps=6, use_graphs=False)
    prev = os.environ.get("MSTAR_GLM52_MTP_CAPTURE_SYNC")
    os.environ["MSTAR_GLM52_MTP_CAPTURE_SYNC"] = "1"
    try:
        replay, runners = _drive(cfg, model, prompts, steps=6, use_graphs=True)
    finally:
        if prev is None:
            os.environ.pop("MSTAR_GLM52_MTP_CAPTURE_SYNC", None)
        else:
            os.environ["MSTAR_GLM52_MTP_CAPTURE_SYNC"] = prev

    assert MTP_SYNC_LABEL in runners, "the sync graph did not capture"
    assert eager["r0"] == replay["r0"], (
        f"padded sync replay diverged from eager:\n eager  {eager['r0']}\n "
        f"replay {replay['r0']}")


def test_piecewise_graphs_do_not_share_dummy_request_slots():
    """Two graphs of the SAME shape must not share dummy request slots.

    mtp_trunk and mtp_sync are both (bs, k+1), and dummy rids were derived
    from the shape alone — so both runners addressed one slot in the shared
    allocator, and ``add_request`` OVERWRITES ``request_states[rid]``,
    discarding the other's dummy state and orphaning the pages capture had
    allocated to it. Asserted structurally because the symptom is silent.

    Workspace sharing is deliberately NOT asserted here: these runners share
    one FlashInfer workspace per (cache label, shape) on purpose, and that is
    safe because plan and replay are paired inside a single run() on one
    stream with the runners called strictly in order. Splitting it would cost
    +2.5 GiB/rank. See _build_persistent_wrappers."""
    cfg = _cfg()
    model = _build(cfg)
    sub = Glm52LLMSubmodule(model, cfg)
    # Set the attribute rather than the env: the flag is read in __init__,
    # so an env write after construction is a no-op (it silently produced a
    # KeyError'd 'mtp_sync' the first time this test ran).
    sub._mtp_capture_sync = True
    cm, alloc, buffers, kv_cfg, _ = _kv(cfg, 1)
    runners = build_piecewise_runners(
        sub, DEVICE, torch.bfloat16, tp_world_size=1,
        kv_cache_config=kv_cfg, alloc_manager=alloc,
        buffer_manager=buffers,
    )
    try:
        assert MTP_SYNC_LABEL in runners, "sync graph did not capture"
        trunk, sync = runners[MTP_TRUNK_LABEL], runners[MTP_SYNC_LABEL]
        # Same shape — the precondition that made the collision possible.
        assert set(trunk.graphs) & set(sync.graphs), (
            "test is not exercising the collision: no shared (bs, tokens) key")

        t_ids = {r for gd in trunk.graphs.values() for r in gd.dummy_rids}
        s_ids = {r for gd in sync.graphs.values() for r in gd.dummy_rids}
        assert t_ids and s_ids
        assert not (t_ids & s_ids), f"dummy rids collide: {t_ids & s_ids}"
    finally:
        alloc.cleanup()


def test_batch_padding_replay_matches_eager():
    """bs>1, and deliberately NOT a captured batch size: capture sizes are
    [1,2,4,8,16], so bs=3 pads to 4 and the replay carries a zero-length
    padding row. That padded regime is where the captured KV scatter writes
    its tail — the null-page reservation exists for it — and until now
    nothing in this repo ever measured or tested bs>1 at all."""
    cfg = _cfg()
    model = _build(cfg)
    prompts = _prompts(3)

    eager, _ = _drive(cfg, model, prompts, steps=5, use_graphs=False)
    replay, runners = _drive(cfg, model, prompts, steps=5, use_graphs=True)

    assert MTP_TRUNK_LABEL in runners
    for rid in eager:
        assert eager[rid] == replay[rid], (
            f"{rid}: padded (bs=3 -> 4) replay diverged from eager:\n "
            f"eager  {eager[rid]}\n replay {replay[rid]}")
