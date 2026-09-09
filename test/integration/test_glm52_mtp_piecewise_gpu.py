"""THE MISSING RUNG: the MTP piecewise graphs against the REAL runner, one GPU, seconds.

The GLM-5.2 ladder is

    test/modular/test_glm52_engine_cycle.py   CPU, real resources, regions run EAGER
    this file                                 one GPU, real resources, regions REPLAYED
    the 753B checkpoint on 8 GPUs             minutes per boot

The CPU rung drives the engine's per-step contract over the real ``KVManager``
(MLA latent layout), the real MLA attention resource on its SDPA fallback and
the real sampler, but every captured region runs its Python each call
(``EagerPiecewiseRunner``). It proves the caller's arithmetic and nothing about
replay: static buffers padded to a bucket, the k attention sub-plans landing at
the addresses the graph recorded, padding rows over the runner's dummy streams,
the sink-page scatter tail, ``can_run``/``_resolve`` bucket choice. Every bug in
that list is silent — greedy verify rejects a bad draft, so the only symptom is
lower acceptance, which reads as a modelling problem.

This rung is the CPU rung plus real capture: the same reduced config
(``Glm52ModelConfig.reduced()`` at 4 layers so the MTP plane lands on a FULL
indexer position, ``mla_absorb=True``), the resources built exactly the way
``Engine.load_model`` builds them (``get_node_resources`` -> YAML overrides ->
``build_resource``), the submodule bound to them, and ``PiecewiseCudaGraphRunner``s
built like ``Engine._build_piecewise_runners`` and captured. The reduced latent
dims (ckv 32, kpe 8) cannot use the FlashInfer MLA kernel, so the attention
resource takes its SDPA fallback — shape-static by design, so it captures and
replays exactly like the kernel path would.

The property: **at temperature 0 a decode step routed through the captured
regions emits the byte-identical token stream that the eager path emits**, and
the graphs were actually replayed rather than silently serving eager (a 13x
regression wearing a correctness costume).

Box notes (H200, 2026-09-08): the runner needs an INDEXED device
(``torch.cuda.set_device`` rejects a bare ``"cuda"``); grads must be off before
capture or Inductor tries to backward through ``mstar::flashinfer_rmsnorm`` and
every bucket fails; every parameter is randomized explicitly because a freshly
allocated reduced model can carry an all-zero ``lm_head`` and emit token 0
forever, which makes bit-identity vacuous.

Not covered here: TP>1 collectives, the FlashInfer MLA kernel path
(``test_mla_kernel_vs_fallback_gpu.py`` covers kernel-vs-fallback at real dims),
real-checkpoint numerics, and the full-forward ``CudaGraphRunner`` (MTP registers
none).
"""
from __future__ import annotations

import gc
import warnings

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.cuda_graph_runner import PiecewiseCudaGraphRunner, autocast_scope
from mstar.engine.resources import (
    SamplingReqConfig,
    StepContext,
    StepRunner,
    apply_yaml_overrides,
    resolve_spec_dependencies,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
from mstar.model.glm52.config import (
    KV_RESOURCE,
    SAMPLER_RESOURCE,
    Glm52ModelConfig,
)
from mstar.model.glm52.quantization import process_weights_after_loading
from mstar.model.glm52.submodules import (
    MTP_DRAFT_BUNDLE,
    MTP_DRAFT_LABEL,
    MTP_DRAFT_PHASE_LABEL,
    MTP_PREFILL_LABEL,
    MTP_SYNC_LABEL,
    MTP_TRUNK_LABEL,
    Glm52LLMSubmodule,
)
from mstar.model.submodule_base import ModelInputsFromEngine

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="piecewise capture needs a real GPU (CUDA graphs + the engine's resources)",
)

# Indexed on purpose: PiecewiseCudaGraphRunner.warmup_and_capture calls
# torch.cuda.set_device(self._device), which rejects a bare "cuda".
DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16  # the serving autocast dtype; the KV cache takes it too
NODE = "LLM"
# The deployment's ``resources: kv:`` block, applied to the model's declared
# specs exactly as the worker applies a YAML (page_size 8 / 64 pages: what the
# H200 rung ran with; a page per ~one MTP step keeps the padded plans small).
KV_YAML = {"resources": {KV_RESOURCE: {"max_num_pages": 64, "page_size": 8}}}
PROMPT_A = 5  # arange(5) + 3, as the CPU rung
PROMPT_B = 3  # arange(3) + 9


@pytest.fixture(autouse=True)
def _gpu_test_env(monkeypatch):
    # The SDPA fallback never touches the FlashInfer workspace; 512 MB per
    # (label, slot) x every driver built here is just VRAM churn.
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "64")
    yield
    gc.collect()
    torch.cuda.empty_cache()


def _cfg(k: int) -> Glm52ModelConfig:
    cfg = Glm52ModelConfig.reduced()
    cfg.num_hidden_layers = 4  # the MTP position lands FULL (4 = offset-1 + freq)
    cfg.mla_absorb = True  # the MLA resource on its (shape-static) SDPA fallback
    cfg.mtp_num_draft_tokens = k
    cfg.dsa_long_context = False
    # one packed prefill bucket; the prompts here are 3-5 tokens
    cfg.prefill_token_buckets = [64]
    cfg.prefill_capture_batch_sizes = [1]
    return cfg


def _model(cfg: Glm52ModelConfig, seed: int = 0) -> Glm52ForCausalLM:
    """Reduced model with finite, explicitly randomized weights. Values are
    arbitrary — the property is eager-vs-replay equality on the SAME weights,
    not quality — but every parameter is written: a freshly allocated
    ``lm_head`` can be all zeros on a GPU, and an all-zero head emits token 0
    forever, which any bit-identity check passes vacuously."""
    torch.manual_seed(seed)
    model = Glm52ForCausalLM(cfg).to(DEVICE, DTYPE)
    for name, p in model.named_parameters():
        if not p.dtype.is_floating_point:
            continue
        if "norm" in name:
            p.data.normal_(1.0, 0.05)
        else:
            p.data.normal_(0, 0.05)
    for name, p in model.named_parameters():
        # Router bias stays fp32 in production (restore_fp32_params); the
        # submodule's to() refuses dtype casts for exactly this reason.
        if "e_score_correction_bias" in name:
            p.data = p.data.float()
    # Engine.load_model does this before warmup: with grads on, the compiled
    # region's Inductor pass tries to backward through mstar::flashinfer_rmsnorm
    # and every bucket fails to capture.
    model.requires_grad_(False)
    process_weights_after_loading(model, DEVICE)
    return model.eval()


def _greedy() -> SamplingReqConfig:
    # ignore_eos: the reduced EOS ids (250-252) sit inside the 256-id vocab, so
    # greedy decode on random weights can hit one and end the stream early
    return SamplingReqConfig(
        temperature=0.0, top_k=0, top_p=1.0, ignore_eos=True, repetition_penalty=1.0,
    )


def _fwd_info(rid: str, max_tokens: int) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk="prefill", fwd_index=0, random_seed=0,
        max_tokens=max_tokens, resource_configs={SAMPLER_RESOURCE: _greedy()},
    )


def _build_resources(cfg: Glm52ModelConfig) -> tuple[dict, StepRunner]:
    """The node's resources the way ``Engine.load_model`` builds them: from the
    model's own declaration, through the YAML overrides, ``build_resource`` on
    the serving dtype, and a ``StepRunner`` scoped to the node."""
    # inside: mstar.model.base pulls the sampler's Triton kernels, which keeps
    # collection clean on a CUDA-less machine (test_fused_moe_fp8.py precedent)
    from mstar.model.glm52.glm52_model import Glm52Model

    model = Glm52Model(model_path_hf="", config_variant="reduced")
    model.config = cfg  # the declaration reads mla_absorb / mtp_num_draft_tokens
    specs = model.get_node_resources()
    apply_yaml_overrides(specs, KV_YAML)
    by_key = resolve_spec_dependencies(specs)
    groups = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial())
    transfer = TransferEngineInfo(
        my_entity_id="glm52_pw_gpu", my_session_id="glm52_pw_gpu",
        transfer_engine=LocalTransferEngine("localhost"),
    )
    resources = {
        spec.resource_key: build_resource(
            spec,
            EngineResourceInfo(
                device=DEVICE, joint_comm_group=groups, transfer_engine_info=transfer,
                kv_dtype=DTYPE,
                dependencies={key: by_key[key] for key in spec.depends_on()},
            ),
        )
        for spec in specs
    }
    runner = StepRunner(resources, node_resources={NODE: list(resources)})
    return resources, runner


def _capture_runners(
    sub: Glm52LLMSubmodule, resources: dict, runner: StepRunner,
) -> tuple[dict[str, PiecewiseCudaGraphRunner], dict]:
    """``Engine._build_piecewise_runners`` + the capture half of
    ``Engine.warmup``: every runner claims its static buffers before any of
    them captures, then each captures, and a region whose capture failed is
    dropped so the forward takes its eager path for that label.

    The regions compile as the config says (``compile_mode="default"``,
    MSTAR_GLM52_GRAPH_COMPILE=1 unless the environment says otherwise) — the
    production capture, not a test-only eager one."""
    # PORT-CHECK: if an eager-vs-replay assertion fails ONLY with the compile
    # flag on, re-run with MSTAR_GLM52_GRAPH_COMPILE=0 — an Inductor fusion
    # that moved a rounding is a numerics finding, not a replay-plumbing bug.
    configs = sub.get_piecewise_cuda_graph_configs(DEVICE, DTYPE, tp_world_size=1)
    runners = {
        label: PiecewiseCudaGraphRunner(
            label=f"{NODE}_{label}", config=config, resources=resources,
            step_runner=runner, device=DEVICE, autocast_dtype=DTYPE,
            joint_comm_group=None, node_name=NODE,
        )
        for label, config in configs.items()
    }
    for pw in runners.values():
        pw.prepare_for_capture()
    captured: dict[str, PiecewiseCudaGraphRunner] = {}
    with torch.no_grad():
        for label, pw in runners.items():
            pw.warmup_and_capture()
            if pw.any_graphs:
                captured[label] = pw
    return captured, configs


class _Driver:
    """The engine's per-step cycle for one node, over real resources — the
    CPU rung's ``_Driver`` with the real ``PiecewiseCudaGraphRunner``s in
    ``piecewise_runners`` (``regions=True``) instead of eager stand-ins.

    Order matches the worker: bind resources, capture (the runners' padding
    rows are ingested during capture), then open the real requests."""

    def __init__(
        self, sub: Glm52LLMSubmodule, cfg: Glm52ModelConfig, rids: list[str],
        regions: bool = False,
    ):
        self.sub = sub
        self.rids = list(rids)
        self.resources, self.runner = _build_resources(cfg)
        sub.bind_node_resources(self.resources)
        self.piecewise: dict[str, PiecewiseCudaGraphRunner] = {}
        self.configs: dict = {}
        if regions:
            self.piecewise, self.configs = _capture_runners(sub, self.resources, self.runner)
        for rid in rids:
            self.runner.ingest_request(rid, {SAMPLER_RESOURCE: _greedy()})

    def step(self, walk: str, batch: dict[str, tuple[CurrentForwardPassInfo, torch.Tensor]]):
        rids = list(batch)
        inputs = [
            self.sub.prepare_inputs(walk, info, {"text_inputs": [text]})
            for info, text in batch.values()
        ]
        step = self.sub.declare_step(walk, rids, inputs)
        ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False)
        # inference-only, under the same scope Engine.exec runs the step in
        with torch.no_grad(), autocast_scope(DTYPE, device_type=DEVICE.type):
            if step is not None:
                step.set_ctx(ctx)
                outcome = self.runner.admit(step)
                assert outcome.ok, outcome
                self.runner.plan(step)
            engine_inputs = ModelInputsFromEngine(
                request_ids=rids,
                per_request_info={rid: info for rid, (info, _) in batch.items()},
                resources=self.resources,
                piecewise_runners=self.piecewise,
                step=step,
            )
            kw = self.sub.preprocess(walk, engine_inputs, inputs)
            outs = self.sub.forward_batched(walk, engine_inputs, **kw)
            if step is not None:
                self.runner.commit(step)
        for rid, (info, _) in batch.items():
            self.sub.postprocess(rid, info, outs[rid])
        return outs

    def ran_eager(self) -> dict[str, bool]:
        """Which MTP regions fell back to eager at least once: the submodule
        warns once per region when no captured bucket served a step, and a
        run that passed on a fallback proves nothing about replay."""
        return {
            "trunk": self.sub._mtp_trunk_eager_warned,
            "sync": self.sub._mtp_sync_eager_warned,
            "draft": self.sub._mtp_draft_eager_warned,
        }

    def close(self):
        for rid in self.rids:
            self.runner.remove_request(rid)
            self.sub.cleanup_request(rid)
        for resource in self.resources.values():
            resource.cleanup()


def _drive(
    driver: _Driver, prompt: torch.Tensor, info: CurrentForwardPassInfo,
    max_steps: int = 64, carry_prefill_drafts: bool = True,
) -> list[int]:
    """Prefill + decode until ``check_stop``; the emitted stream as a list.

    ``carry_prefill_drafts`` feeds the prefill's ``[emitted, k drafts]`` bundle
    into the first decode step, as the persisted ``MTP_DRAFT_BUNDLE`` edge does
    in production (MSTAR_GLM52_MTP_PREFILL_DRAFTS=1, the default); off, the
    first decode is a 1-row step on a (bs, k+1) bucket."""
    rid = info.request_id
    emitted = []
    walk, text, decode_step = "prefill", prompt, 0
    for _ in range(max_steps):
        out = driver.step(walk, {rid: (info, text)})[rid]
        emitted.append(out["new_token"][0])
        if walk == "decode":
            info.dynamic_loop_iter_counts["decode_loop"] = decode_step
            decode_step += 1
        if driver.sub.check_stop(rid, info, out):
            break
        nxt = out["text_inputs"][0]
        if carry_prefill_drafts and MTP_DRAFT_BUNDLE in out:
            nxt = out[MTP_DRAFT_BUNDLE][0]
        walk, text = "decode", nxt
    return torch.cat(emitted).tolist()


def _prompt(n: int, offset: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.long, device=DEVICE) + offset


def _stream(
    model: Glm52ForCausalLM, cfg: Glm52ModelConfig, mode_k: int, regions: bool,
    prompt: torch.Tensor, max_tokens: int, rid: str = "r0",
    carry_prefill_drafts: bool = True,
) -> tuple[list[int], dict[str, bool], set[str]]:
    """One arm: the stream, which regions fell back to eager, which captured.

    ``mode_k=0`` is plain decode (MTP off) on the same weights — the MTP plane
    exists on the model but nothing reads it — the baseline greedy verify must
    reproduce by construction."""
    k = cfg.mtp_num_draft_tokens
    cfg.mtp_num_draft_tokens = mode_k
    try:
        sub = Glm52LLMSubmodule(model, cfg)
        driver = _Driver(sub, cfg, [rid], regions=regions)
        try:
            stream = _drive(
                driver, prompt, _fwd_info(rid, max_tokens),
                carry_prefill_drafts=carry_prefill_drafts,
            )
            return stream, driver.ran_eager(), set(driver.piecewise)
        finally:
            driver.close()
    finally:
        cfg.mtp_num_draft_tokens = k


def _expected_labels(k: int, draft_phase: bool = True) -> set[str]:
    labels = {MTP_TRUNK_LABEL, MTP_SYNC_LABEL, MTP_PREFILL_LABEL}
    if k >= 2:
        labels.add(MTP_DRAFT_LABEL)
    if draft_phase:
        labels.add(MTP_DRAFT_PHASE_LABEL)
    return labels


# ── (a) capture ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("k", [1, 2, 3])
def test_every_region_captures_every_bucket(monkeypatch, k):
    """Every MTP region captures at every reduced bucket, and same-shape
    regions keep their padding rows apart. Capture failure is logged, not
    raised (the engine serves eager on a failed region), so a benchmark would
    report it as 'MTP is slow'; this is the check that makes it a failure."""
    monkeypatch.setattr(Glm52LLMSubmodule, "MTP_CAPTURE_BATCH_SIZES", [1, 2])
    cfg = _cfg(k)
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    driver = _Driver(sub, cfg, [], regions=True)
    try:
        assert set(driver.configs) == _expected_labels(k)
        assert set(driver.piecewise) == _expected_labels(k), (
            "a region failed to capture — the runner's warning (with traceback) "
            "is in the captured log")
        for label, config in driver.configs.items():
            pw = driver.piecewise[label]
            for shape in config.get_capture_shapes(sorted(config.capture_batch_sizes)):
                assert pw.can_run(shape.bs, shape.total_tokens), (label, shape)
        # mtp_trunk and mtp_sync are both (bs, k+1): the shape collision that
        # once made two runners address one padding slot (dummy ids were
        # derived from the shape alone). The pool keys them by runner label
        # now; asserted structurally because the symptom is silent.
        trunk, sync = driver.piecewise[MTP_TRUNK_LABEL], driver.piecewise[MTP_SYNC_LABEL]
        assert set(trunk._graphs) & set(sync._graphs), "no shared (bs, tokens) bucket"
        t_ids = {rid for g in trunk._graphs.values() for rid in g.dummy_rids}
        s_ids = {rid for g in sync._graphs.values() for rid in g.dummy_rids}
        assert t_ids and s_ids
        assert not (t_ids & s_ids), f"padding rows collide: {t_ids & s_ids}"
    finally:
        driver.close()


# ── (b) replayed regions == eager == plain decode ────────────────────────


@pytest.mark.parametrize("k", [1, 2, 3])
def test_replayed_regions_match_eager_and_plain_decode_bitwise(monkeypatch, k):
    """THE property, per k. Same weights, same prompt, three arms:

    - plain decode (MTP off): the stream greedy verify reproduces by
      construction;
    - MTP on, every region eager (the CPU rung's regime, on the GPU);
    - MTP on, every region REPLAYED from its captured graph.

    Replay vs eager is the plumbing claim — at bs=1 the captured trunk is
    shape-identical to the eager one, so ANY divergence is the replay path.
    Replay vs plain decode is the design guarantee the feature rests on.
    """
    monkeypatch.setattr(Glm52LLMSubmodule, "MTP_CAPTURE_BATCH_SIZES", [1, 2])
    cfg = _cfg(k)
    model = _model(cfg)
    prompt = _prompt(PROMPT_A, 3)
    plain, _, _ = _stream(model, cfg, 0, regions=False, prompt=prompt, max_tokens=24)
    eager, _, _ = _stream(model, cfg, k, regions=False, prompt=prompt, max_tokens=24)
    replayed, fell_back, captured = _stream(
        model, cfg, k, regions=True, prompt=prompt, max_tokens=24)

    assert captured == _expected_labels(k), captured
    assert not any(fell_back.values()), (
        f"a region ran eager at least once ({fell_back}); the comparison below "
        "would prove nothing about replay")
    assert len(replayed) == 24
    assert replayed == eager, (
        f"captured replay diverged from the eager regions:\n eager    {eager}\n"
        f" replayed {replayed}")
    # PORT-CHECK: plain decode runs the trunk at M=1 rows, the MTP trunk at
    # M=k+1; cuBLAS may pick a different kernel per M and round a bf16 logit
    # differently. If ONLY this assertion fails on the box, find the first
    # divergent step and check whether the top-2 logits there sit within one
    # bf16 ulp (a near-tie, numerics) before treating it as a plumbing bug.
    assert replayed == plain, (
        f"captured replay diverged from plain decode:\n plain    {plain}\n"
        f" replayed {replayed}")


def test_first_decode_without_the_prefill_bundle_matches(monkeypatch):
    """MSTAR_GLM52_MTP_PREFILL_DRAFTS=0's shape: the first decode step is one
    row on the (bs, k+1) trunk bucket — the graph computes k+1 rows, the plan
    covers one, the scatter tail aims at the sink page."""
    monkeypatch.setattr(Glm52LLMSubmodule, "MTP_CAPTURE_BATCH_SIZES", [1, 2])
    cfg = _cfg(2)
    model = _model(cfg)
    prompt = _prompt(PROMPT_A, 3)
    plain, _, _ = _stream(model, cfg, 0, regions=False, prompt=prompt, max_tokens=16)
    replayed, fell_back, _ = _stream(
        model, cfg, 2, regions=True, prompt=prompt, max_tokens=16,
        carry_prefill_drafts=False)
    assert not any(fell_back.values()), fell_back
    assert replayed == plain, f"\n plain    {plain}\n replayed {replayed}"


# ── (c) padded batch ─────────────────────────────────────────────────────


def _drive_batch(driver: _Driver, prompts: dict[str, torch.Tensor], max_tokens: int):
    """Prefill each request alone, then decode them as one batch; the CPU
    rung's batch loop."""
    infos = {rid: _fwd_info(rid, max_tokens) for rid in prompts}
    emitted = {rid: [] for rid in prompts}
    texts = {}
    for rid, prompt in prompts.items():
        out = driver.step("prefill", {rid: (infos[rid], prompt)})[rid]
        emitted[rid].append(out["new_token"][0])
        texts[rid] = out[MTP_DRAFT_BUNDLE][0] if MTP_DRAFT_BUNDLE in out else out["text_inputs"][0]
    live = set(prompts)
    for _ in range(32):
        if not live:
            break
        outs = driver.step("decode", {rid: (infos[rid], texts[rid]) for rid in sorted(live)})
        for rid in sorted(live):
            emitted[rid].append(outs[rid]["new_token"][0])
            texts[rid] = outs[rid]["text_inputs"][0]
            if driver.sub.check_stop(rid, infos[rid], outs[rid]):
                live.discard(rid)
    return {rid: torch.cat(emitted[rid]).tolist() for rid in prompts}


@pytest.mark.parametrize("k", [2, 3])
def test_batch_of_two_padded_to_four_matches_single_streams(monkeypatch, k):
    """Two requests (different prompts, different accepted counts per step)
    through regions captured ONLY at bs=4: two padding rows per replay —
    zero-length plan rows, sink-page scatter tails, static buffers zeroed
    past the real rows. Each must emit what it emits alone.

    k=3 with num < bs is the layout the flat chain-position buffer once got
    wrong (blocks of ``num`` written, blocks of bucket ``bs`` read): the
    per-iteration ``chain_pos_{it}`` inputs are what makes padding safe, and
    only two chain iterations with padding rows exercise it.
    """
    monkeypatch.setattr(Glm52LLMSubmodule, "MTP_CAPTURE_BATCH_SIZES", [4])
    cfg = _cfg(k)
    model = _model(cfg)
    prompts = {"a": _prompt(PROMPT_A, 3), "b": _prompt(PROMPT_B, 9)}
    # alone, on the same bs=4 buckets (shape-identical to the batched replay)
    singles = {
        rid: _stream(model, cfg, k, regions=True, prompt=p, max_tokens=14, rid=rid)[0]
        for rid, p in prompts.items()
    }
    # alone, eager (the CPU rung's reference)
    singles_eager = {
        rid: _stream(model, cfg, k, regions=False, prompt=p, max_tokens=14, rid=rid)[0]
        for rid, p in prompts.items()
    }

    sub = Glm52LLMSubmodule(model, cfg)
    driver = _Driver(sub, cfg, list(prompts), regions=True)
    try:
        assert set(driver.piecewise) == _expected_labels(k)
        batched = _drive_batch(driver, prompts, max_tokens=14)
        assert not any(driver.ran_eager().values()), driver.ran_eager()
    finally:
        driver.close()
    for rid in prompts:
        assert batched[rid] == singles[rid], (
            f"{rid}: padded (bs=2 -> 4) replay diverged from the request alone:\n"
            f" alone   {singles[rid]}\n batched {batched[rid]}")
        # PORT-CHECK: eager runs the trunk at M=k+1 rows per request, the bs=4
        # bucket at M=4(k+1); same near-tie caveat as the plain-decode arm.
        assert batched[rid] == singles_eager[rid], (
            f"{rid}: padded replay diverged from the eager regions:\n"
            f" eager   {singles_eager[rid]}\n batched {batched[rid]}")


# ── (d) the three-graph fallback ─────────────────────────────────────────


def test_three_graph_fallback_matches_plain_decode(monkeypatch):
    """MSTAR_GLM52_MTP_DRAFT_PHASE_GRAPH=0: no draft-phase graph; the decode
    draft phase is the padded sync graph, the eager draft-1 gather, and k-1
    replays of the one-row chain graph, with the runner committing each and
    the submodule rewinding after."""
    monkeypatch.setenv("MSTAR_GLM52_MTP_DRAFT_PHASE_GRAPH", "0")
    monkeypatch.setattr(Glm52LLMSubmodule, "MTP_CAPTURE_BATCH_SIZES", [1, 2])
    cfg = _cfg(3)
    model = _model(cfg)
    prompt = _prompt(PROMPT_A, 3)
    plain, _, _ = _stream(model, cfg, 0, regions=False, prompt=prompt, max_tokens=18)
    eager, _, _ = _stream(model, cfg, 3, regions=False, prompt=prompt, max_tokens=18)
    replayed, fell_back, captured = _stream(
        model, cfg, 3, regions=True, prompt=prompt, max_tokens=18)
    assert captured == _expected_labels(3, draft_phase=False), captured
    assert not any(fell_back.values()), fell_back
    assert replayed == eager, f"\n eager    {eager}\n replayed {replayed}"
    assert replayed == plain, f"\n plain    {plain}\n replayed {replayed}"


# ── (e) host syncs per captured decode step ──────────────────────────────


def test_captured_decode_step_syncs_exactly_once(monkeypatch):
    """The sync discipline behind the draft-chain speedup.

    A captured MTP decode step must touch the host exactly ONCE — the batched
    ``.tolist()`` in greedy verify (a true data dependency: e decides the
    rewind and the plans). Everything else — the trunk and draft-phase plans,
    RoPE positions, the scatter maps, the per-iteration chain positions —
    reaches the device through pinned staging with non_blocking copies, so
    the CPU queues the whole draft phase without waiting on the GPU.

    ``torch.cuda.set_sync_debug_mode("warn")`` reports every synchronizing op
    PyTorch can see (``.item()``, ``.tolist()``, ``nonzero``, a blocking
    ``copy_`` from host memory, ``torch.tensor(..., device=cuda)``). It cannot
    see a ``non_blocking`` copy from *pageable* memory (CUDA drains the stream
    for that too) — ``pinned()`` is what closes that hole, asserted first.
    """
    from mstar.utils.pinned_staging import pinned

    assert pinned([1, 2, 3], torch.long).is_pinned(), (
        "pinned() must return pinned memory when CUDA is available")

    monkeypatch.setattr(Glm52LLMSubmodule, "MTP_CAPTURE_BATCH_SIZES", [1, 2])
    cfg = _cfg(2)
    model = _model(cfg)
    sub = Glm52LLMSubmodule(model, cfg)
    driver = _Driver(sub, cfg, ["r0"], regions=True)
    try:
        assert set(driver.piecewise) == _expected_labels(2)
        info = _fwd_info("r0", 4096)
        out = driver.step("prefill", {"r0": (info, _prompt(PROMPT_A, 3))})["r0"]
        text = out[MTP_DRAFT_BUNDLE][0]
        # one warm decode step (first-use allocations, lazily built wrappers),
        # then the measured one
        caught: list[warnings.WarningMessage] = []
        for measured in (False, True):
            torch.cuda.synchronize()
            if measured:
                torch.cuda.set_sync_debug_mode("warn")
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    out = driver.step("decode", {"r0": (info, text)})["r0"]
            finally:
                if measured:
                    torch.cuda.set_sync_debug_mode("default")
            text = out["text_inputs"][0]
        assert not any(driver.ran_eager().values()), driver.ran_eager()
        syncs = [str(w.message) for w in caught if "synchroniz" in str(w.message).lower()]
        assert len(syncs) == 1, (
            f"expected exactly one host sync per captured decode step (the verify "
            f".tolist()), saw {len(syncs)}:\n" + "\n".join(syncs))
    finally:
        torch.cuda.set_sync_debug_mode("default")
        driver.close()
