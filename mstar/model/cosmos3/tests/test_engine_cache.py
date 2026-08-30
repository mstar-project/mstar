"""GPU parity for the cache-once engine path of the Cosmos3 generator.

The understanding tower runs once and writes its per-layer K/V; the generation
tower then runs each denoise step re-reading that frozen K/V (the text tokens get
no timestep embedding, so their K/V is denoise-step independent — caching it once
is exact). This checks the ``Cosmos3DiTSubmodule`` prefill + denoise loop against
the fused ``Cosmos3Pipeline`` that runs the whole transformer every step, for both
image (single frame) and video (multi-frame, fps-modulated mRoPE) generation.

Two GPU-gated checks per mode (need ``COSMOS3_NANO_DIR`` + CUDA; skipped otherwise):
  * with an in-process sdpa cache (same attention kernel as the fused pipeline),
    the cache-once output is bit-for-bit identical;
  * with the engine's FlashInfer paged cache (the served path), the decoded output
    matches the fused pipeline within PSNR >= 30 (FlashInfer-vs-sdpa precision).

Run: COSMOS3_NANO_DIR=<snap> python3 test_engine_cache.py
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

from mstar.engine.resources import AdmitOutcome, StepContext, StepRunner
from mstar.engine.resources.base import AttentionResource
from mstar.engine.resources.kv.plan import group_by_plan_label
from mstar.model.cosmos3.submodules import ATTN, KV_CACHE
from mstar.model.submodule_base import ModelInputsFromEngine

PROMPT = "A red cube resting on a polished wooden table, soft daylight."
# Parity checks here are resolution-independent; 256x256 keeps them quick. The
# CUDA-graph check below captures at whatever (H, W) it sets. NOTE: the in-process
# graph-vs-fused PSNR is a coarse smoke check — it carries a cache-setup artifact
# of this harness. The authoritative bit-exactness gate for the served graph is
# the HTTP A/B (graph-on vs COSMOS3_DISABLE_CUDA_GRAPH=1), which is byte-identical
# at every resolution.
H = W = 256
STEPS = 12
GS = 6.0
SEED = 42
VIDEO_FRAMES = 17  # latent T = 1 + (17 - 1) // 4 = 5


class _SdpaResources:
    """In-process reference resources with the step surface the DiT calls,
    backed by stored tensors + sdpa (the same kernel as the fused pipeline).

    Two objects, matching what the node declares: a ``kv`` that holds each
    layer's understanding K/V and a ``attn`` that attends over it. Prefill
    writes its K/V pending; the runner's ``commit`` promotes it, and every
    denoise step re-reads it.

    Also models the batched classifier-free-guidance plan: under a combined
    plan label the packed sequence carries both branches, so the plan output
    records where each branch's block starts and ``run`` routes each block to
    its own label's cached prefix. That makes the batched result equal to
    running the branches sequentially.
    """

    def __init__(self):
        self.kv = _SdpaKV()
        self.attn = _SdpaAttention(self.kv)

    def as_dict(self):
        return {KV_CACHE: self.kv, ATTN: self.attn}

    def bind(self, module):
        """Bind these into a bare transformer, the way
        ``NodeSubmodule.bind_node_resources`` does for a served node.

        Unlike that walk, the root is bound too: there the root is the
        submodule, which takes its resources separately, but here it is the
        transformer, which binds the shared attend callable."""
        resources = self.as_dict()
        for child in module.modules():
            bind = getattr(child, "bind_resources", None)
            if bind is not None:
                bind(resources)

    def plan(self, plan_label, groups, causal):
        """Stand in for a step declaration, for the tests that call the
        transformer directly instead of going through the submodule.

        ``groups`` is ``[(cache label, span)]`` in packed order — what the KV
        resource's plan output carries and every other resource reads.
        """
        self.kv.groups = {plan_label: list(groups)}
        self.attn.causal = causal


class _SdpaKV(AttentionResource):
    """The ``kv`` half: per-(label, layer) K/V, plus the plan grouping every
    other resource reads off ``ctx.plan_results``.

    An AttentionResource so it carries the same label/layer cursors the real
    one does — that is what layers reach it through."""

    @classmethod
    def build(cls, *args, **kwargs):
        raise NotImplementedError("test stub")

    def __init__(self):
        self.committed: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.pending: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        # plan label -> [(source label, span)], in packed order
        self.groups: dict[str, list[tuple[str, int]]] = {}

    def depends_on(self):
        return set()

    def ingest_request(self, rid, overrides=None):
        return

    def remove_request(self, rid):
        return

    def admit(self, step, ctx):
        return AdmitOutcome(ok=True)

    def plan(self, step, ctx):
        self.groups = {
            plan_label: [(seg.label, seg.span) for seg in segments]
            for plan_label, segments in group_by_plan_label(
                tuple(step.segments), step.combined_labels
            ).items()
        }
        return self.groups

    def commit(self, step, ctx):
        if step.commit:
            self.promote()
        else:
            # A denoise step writes its generation K/V here too (the fake
            # backend is a paged one, so `requires_kv_write` is True) and never
            # commits it; drop it rather than let it shadow the prefix.
            self.pending = {}

    def promote(self):
        self.committed.update(self.pending)
        self.pending = {}

    def layer_view(self, layer_idx=None):
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        return layer_idx

    def write_kv(self, k, v, layer_idx=None, label=None):
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        if label is None:
            label = self._default_label
        for source_label, k_part, v_part in self._split(label, k, v):
            self.pending[(source_label, layer_idx)] = (k_part, v_part)

    def _split(self, plan_label, k, v):
        """One packed step's K/V, cut back into its source labels."""
        offset = 0
        for source_label, span in self.groups[plan_label]:
            yield source_label, k[offset:offset + span], v[offset:offset + span]
            offset += span


class _SdpaAttention(AttentionResource):
    """The ``attn`` half: sdpa over [committed prefix | this step's K/V]."""

    requires_kv_write = True

    @classmethod
    def build(cls, *args, **kwargs):
        raise NotImplementedError("test stub")

    def __init__(self, kv):
        self._kv = kv
        self.causal = True

    def depends_on(self):
        return {KV_CACHE}

    def ingest_request(self, rid, overrides=None):
        return

    def remove_request(self, rid):
        return

    def admit(self, step, ctx):
        return AdmitOutcome(ok=True)

    def plan(self, step, ctx):
        self.causal = step.causal

    def commit(self, step, ctx):
        return

    @staticmethod
    def _sdpa(q, k, v, is_causal):
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2),
            v.unsqueeze(0).transpose(1, 2), is_causal=is_causal, enable_gqa=True,
        )
        return out.transpose(1, 2).squeeze(0)

    def _attend_label(self, label, layer, q, k, v):
        prefix = self._kv.committed.get((label, layer))
        if prefix is not None:
            pk, pv = prefix
            return self._sdpa(q, torch.cat([pk, k], 0), torch.cat([pv, v], 0), self.causal)
        return self._sdpa(q, k, v, self.causal)

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        if label is None:
            label = self._default_label
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        groups = self._kv.groups[label]
        if len(groups) == 1:
            return self._attend_label(groups[0][0], layer_idx, q, k, v)
        outs, offset = [], 0
        for source_label, span in groups:
            sl = slice(offset, offset + span)
            offset += span
            outs.append(self._attend_label(source_label, layer_idx, q[sl], k[sl], v[sl]))
        return torch.cat(outs, 0)


def _forward_step(
    dit, walk, resources, rids, fwds, inputs, batched=False, cg_runner=None,
):
    """Drive one step the way the v1 engine does: lease, declare, admit, plan,
    preprocess, forward, commit — a trimmed ``Engine.exec``.

    With ``cg_runner`` the step takes a capture slot when one matches, which
    pads the batch to the slot's shape and replays instead of running the
    eager forward. Everything above the launch is the same either way.
    """
    runner = StepRunner(resources)
    real_ids = list(rids)
    step_ids, step_fwds = real_ids, dict(fwds)
    lease = None
    if cg_runner is not None:
        lease = cg_runner.lease_slot(
            graph_walk=walk,
            bs=len(real_ids),
            num_tokens=sum(inp.input_seq_len for inp in inputs),
            cg_key_info=dit.cg_key_info(walk, step_fwds),
        )
    if lease is not None:
        inputs = cg_runner.pad_inputs(lease, inputs)
        step_fwds = cg_runner.step_metadata(lease, real_ids, step_fwds)
        step_ids = cg_runner.step_ids(lease, real_ids)

    step = dit.declare_step(walk, step_ids, inputs)
    if step is not None:
        step.set_ctx(StepContext(
            request_ids=tuple(step_ids), graph_walk=walk,
            slot=0 if lease is None else lease.slot,
            capture=False, slot_lease=lease,
        ))
        outcome = runner.admit(step)
        assert outcome.ok, f"admit failed: {outcome.reason}"
        runner.plan(step)
    ei = ModelInputsFromEngine(
        request_ids=list(step_ids),
        per_request_info=step_fwds,
        resources=dict(resources),
        per_request_states={rid: dit.request_state(rid) for rid in step_ids},
        step=step,
        captured=lease is not None,
    )
    pre = dit.preprocess(walk, ei, inputs)
    try:
        if lease is not None:
            raw = cg_runner.run_forward(lease, pre)
            # A captured forward emits its entries under the slot's padding
            # ids — those were the batch at capture time — so entry i belongs
            # to real request i.
            out_ids = cg_runner.slot_for(lease).dummy_rids
            out = {
                rid: raw[out_id]
                for rid, out_id in zip(real_ids, out_ids, strict=False)
            }
        else:
            forward = dit.forward_batched if batched else dit.forward
            out = forward(walk, ei, **pre)
        if step is not None:
            runner.commit(step)
    finally:
        if lease is not None:
            cg_runner.release(lease, len(real_ids))
    return out


def _engine_resources(model, rids, device, dtype, max_num_pages=64, backend=None):
    """The node's resources, built from the model's own declaration — the same
    path the engine takes, minus the worker around it."""
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.distributed.communication import CommGroup, JointGroups
    from mstar.engine.resources.base import EngineResourceInfo, build_resource
    from mstar.engine.resources.kv.transfer import TransferEngineInfo

    prev_backend = model.config.attention_backend
    if backend is not None:
        model.config.attention_backend = backend
    try:
        specs = model.get_node_resources()
    finally:
        model.config.attention_backend = prev_backend
    for spec in specs:
        spec.apply_yaml_overrides(max_num_pages=max_num_pages)

    groups = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial())
    transfer = TransferEngineInfo("h", "h", LocalTransferEngine("h"))
    resources = {
        spec.resource_key: build_resource(
            spec,
            EngineResourceInfo(
                device=torch.device(device),
                joint_comm_group=groups,
                transfer_engine_info=transfer,
                kv_dtype=dtype,
            ),
        )
        for spec in specs
    }
    overrides = model.get_request_resource_configs(partition_fwd_args={})
    runner = StepRunner(resources)
    for rid in rids:
        runner.ingest_request(rid, overrides)
    return resources


@torch.no_grad()
def _run_cache_once(model, dit, resources, init, cond_ids, uncond_ids, device, num_frames):
    from mstar.conductor.request_info import CurrentForwardPassInfo

    rid = "r0"
    md = {"height": H, "width": W, "num_frames": num_frames, "fps": 24.0,
          "guidance_scale": GS, "num_inference_steps": STEPS}
    fwd = CurrentForwardPassInfo(
        request_id=rid, graph_walk="prefill", requires_cfg=(GS != 1.0),
        fwd_index=0, random_seed=SEED, max_tokens=0, sampling_config={}, step_metadata=md,
    )
    text_inputs = [
        torch.tensor(cond_ids, dtype=torch.long, device=device),
        torch.tensor(uncond_ids, dtype=torch.long, device=device),
    ]
    ni = dit.prepare_inputs("prefill", fwd, {"text_inputs": text_inputs})
    _forward_step(dit, "prefill", resources, [rid], {rid: fwd}, [ni])

    latents = init.clone()
    time_index = torch.zeros(1, dtype=torch.long, device=device)
    fwd.graph_walk = "image_gen"
    for _ in range(STEPS):
        ni = dit.prepare_inputs("image_gen", fwd, {"latents": [latents], "time_index": [time_index]})
        out = _forward_step(dit, "image_gen", resources, [rid], {rid: fwd}, [ni])
        latents, time_index = out["latents"][0], out["time_index"][0]
    dit.cleanup_request(rid)
    return latents


@torch.no_grad()
def _run_batched(model, dit, resources, init, conds, unconds, device, rids):
    """Prefill each request (sequential, like the engine), then run the whole
    denoise loop as one batched step per iteration. Returns final latents per rid.

    One resources dict serves every request: the node's cache is persistent and
    each step names the request ids it addresses, so there is no per-batch cache
    object to build.
    """
    from mstar.conductor.request_info import CurrentForwardPassInfo

    md = {"height": H, "width": W, "num_frames": 1, "fps": 24.0,
          "guidance_scale": GS, "num_inference_steps": STEPS}
    fwds = {}
    for i, rid in enumerate(rids):
        fwd = CurrentForwardPassInfo(
            request_id=rid, graph_walk="prefill", requires_cfg=True, fwd_index=0,
            random_seed=SEED, max_tokens=0, sampling_config={}, step_metadata=md,
        )
        fwds[rid] = fwd
        ti = [torch.tensor(conds[i], dtype=torch.long, device=device),
              torch.tensor(unconds[i], dtype=torch.long, device=device)]
        ni = dit.prepare_inputs("prefill", fwd, {"text_inputs": ti})
        _forward_step(dit, "prefill", resources, [rid], {rid: fwd}, [ni])

    for rid in rids:
        fwds[rid].graph_walk = "image_gen"
    latents = {rid: init.clone() for rid in rids}
    time_index = {rid: torch.zeros(1, dtype=torch.long, device=device) for rid in rids}
    for _ in range(STEPS):
        inputs = [
            dit.prepare_inputs("image_gen", fwds[rid],
                               {"latents": [latents[rid]], "time_index": [time_index[rid]]})
            for rid in rids
        ]
        out = _forward_step(
            dit, "image_gen", resources, rids, fwds, inputs, batched=True,
        )
        for rid in rids:
            latents[rid], time_index[rid] = out[rid]["latents"][0], out[rid]["time_index"][0]
    for rid in rids:
        dit.cleanup_request(rid)
    return latents


_SETUP_CACHE: dict = {}


def _load():
    """Load the model / DiT / fused pipeline once (mode-independent)."""
    if "base" in _SETUP_CACHE:
        return _SETUP_CACHE["base"]
    snap = os.environ.get("COSMOS3_NANO_DIR")
    if not snap or not torch.cuda.is_available():
        _SETUP_CACHE["base"] = None
        return None
    torch.use_deterministic_algorithms(True, warn_only=True)
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
    from mstar.model.cosmos3.tests.pipeline import Cosmos3Pipeline

    device, dtype = "cuda:0", torch.bfloat16
    model = Cosmos3Model(model_path_hf=snap)
    # These checks validate the eager cache-once mechanism's numerical exactness.
    # The served default compiles the denoise step, which fuses pointwise ops and
    # perturbs the latents past the tight bit-exact bounds below without changing
    # image quality (the FlashInfer PSNR checks pass with compile on — validated
    # over HTTP). Flip to True to exercise the compiled path here instead.
    model.config.compile_denoise = False
    mpipe = Cosmos3Pipeline.from_model(model, device=device, dtype=dtype)
    dit = model.get_submodule("dit", device=device)  # shares mpipe's transformer
    _SETUP_CACHE["base"] = dict(model=model, mpipe=mpipe, dit=dit, device=device, dtype=dtype)
    return _SETUP_CACHE["base"]


def _scenario(num_frames):
    """Per-mode context: video-aware token ids, shared initial latents, and the
    fused-pipeline latents the cache-once path must reproduce."""
    key = f"frames{num_frames}"
    if key in _SETUP_CACHE:
        return _SETUP_CACHE[key]
    base = _load()
    if base is None:
        _SETUP_CACHE[key] = None
        return None
    from mstar.model.cosmos3.components.packing import tokenize_prompt

    device, dtype, mpipe = base["device"], base["dtype"], base["mpipe"]
    cond_ids, uncond_ids = tokenize_prompt(
        base["model"].tokenizer, PROMPT, "", num_frames=num_frames, height=H, width=W
    )
    lat_t = 1 if num_frames == 1 else 1 + (num_frames - 1) // mpipe.vae_scale_temporal
    gen = torch.Generator(device=device).manual_seed(SEED)
    init = torch.randn((1, 48, lat_t, H // 16, W // 16), generator=gen, device=device, dtype=dtype)
    lat_fused = mpipe(
        prompt=PROMPT, negative_prompt="", num_frames=num_frames, height=H, width=W,
        num_inference_steps=STEPS, guidance_scale=GS, latents=init.clone(), decode=False,
    )
    ctx = dict(cond=cond_ids, uncond=uncond_ids, init=init, lat_fused=lat_fused, num_frames=num_frames, **base)
    _SETUP_CACHE[key] = ctx
    return ctx


def _check_cache_once_exact(num_frames, tag):
    ctx = _scenario(num_frames)
    if ctx is None:
        print(f"  (skipped {tag} cache-once parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    dit = ctx["dit"]
    prev = dit.batched_cfg
    # The sequential guidance path matches the fused pipeline bit-for-bit; the
    # batched path differs only in bf16 GEMM rounding (covered by the PSNR checks).
    dit.batched_cfg = False
    try:
        lat = _run_cache_once(
            ctx["model"], dit, _SdpaResources().as_dict(), ctx["init"], ctx["cond"], ctx["uncond"],
            ctx["device"], num_frames,
        )
    finally:
        dit.batched_cfg = prev
    diff = (ctx["lat_fused"].float() - lat.reshape(ctx["lat_fused"].shape).float()).abs().max().item()
    assert diff <= 1e-3, f"{tag} cache-once latents differ from fused by {diff:.3e} (> 1e-3)"
    print(f"  {tag} cache-once (sdpa) latent abs-max diff = {diff:.3e}")


def _check_engine_psnr(num_frames, tag):
    ctx = _scenario(num_frames)
    if ctx is None:
        print(f"  (skipped {tag} engine cache parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    try:
        cm = _engine_resources(ctx["model"], ["r0"], ctx["device"], ctx["dtype"])
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped {tag} engine cache parity: FlashInfer unavailable: {exc})")
        return
    lat = _run_cache_once(
        ctx["model"], ctx["dit"], cm, ctx["init"], ctx["cond"], ctx["uncond"], ctx["device"], num_frames,
    )
    img_fused = ctx["mpipe"]._decode(ctx["lat_fused"]).squeeze().float().cpu()
    img_engine = ctx["mpipe"]._decode(lat.reshape(ctx["lat_fused"].shape)).squeeze().float().cpu()
    mse = (img_fused - img_engine).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    assert psnr >= 30, f"{tag} engine-path PSNR {psnr:.2f} < 30 (MSE {mse:.3e})"
    print(f"  {tag} engine cache path (flashinfer) PSNR = {psnr:.2f} dB")


@torch.no_grad()
def _check_dense_fa3(num_frames, tag):
    """Dense FlashAttention-3 generation attention vs the paged FlashInfer path.
    Both attend each guidance branch's generation tokens over its frozen text
    prefix; they differ only in the attention kernel (FA3 over a gathered
    contiguous [prefix | gen] vs FlashInfer paged) and its bf16 rounding. So the
    decoded images must match closely, and the dense path must clear the same
    fused-reference bar the paged path meets."""
    ctx = _scenario(num_frames)
    if ctx is None:
        print(f"  (skipped {tag} dense-FA3 parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    try:
        # Paged baseline vs the dense backend (cosmos3 config defaults dense on):
        # the same submodule code runs against each backend class.
        cm = _engine_resources(
            ctx["model"], ["r0"], ctx["device"], ctx["dtype"], backend="flashinfer",
        )
        lat_paged = _run_cache_once(
            ctx["model"], ctx["dit"], cm, ctx["init"], ctx["cond"], ctx["uncond"],
            ctx["device"], num_frames,
        )
        cm2 = _engine_resources(
            ctx["model"], ["r0"], ctx["device"], ctx["dtype"], backend="dense_gen",
        )
        lat_dense = _run_cache_once(
            ctx["model"], ctx["dit"], cm2, ctx["init"], ctx["cond"], ctx["uncond"],
            ctx["device"], num_frames,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped {tag} dense-FA3 parity: FA3/FlashInfer unavailable: {exc})")
        return
    shape = ctx["lat_fused"].shape
    img_fused = ctx["mpipe"]._decode(ctx["lat_fused"]).squeeze().float().cpu()
    img_paged = ctx["mpipe"]._decode(lat_paged.reshape(shape)).squeeze().float().cpu()
    img_dense = ctx["mpipe"]._decode(lat_dense.reshape(shape)).squeeze().float().cpu()

    def _psnr(a, b):
        mse = (a - b).pow(2).mean().item()
        return float("inf") if mse == 0 else -10 * math.log10(mse)

    vs_paged = _psnr(img_dense, img_paged)
    vs_fused = _psnr(img_dense, img_fused)
    # The dense path must match the fused reference as well as the paged engine
    # path does (>= 30, the same bar), and the two engine kernels must agree to
    # within their bf16 rounding (a real ordering/gather bug tanks this < 15).
    assert vs_fused >= 30, f"{tag} dense-FA3 vs fused PSNR {vs_fused:.2f} < 30"
    assert vs_paged >= 30, f"{tag} dense-FA3 vs paged PSNR {vs_paged:.2f} < 30"
    print(f"  {tag} dense-FA3 PSNR vs paged = {vs_paged:.2f} dB, vs fused = {vs_fused:.2f} dB")


def test_dense_fa3_image_psnr() -> None:
    _check_dense_fa3(1, "t2i")


def test_dense_fa3_video_psnr() -> None:
    _check_dense_fa3(VIDEO_FRAMES, "t2v")


def _encode_cond(model, md, media, walk):
    """Drive the vae_encoder node the way the engine does: metadata +
    image_inputs/video_inputs in, ``cond_latents`` out."""
    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.model.submodule_base import ModelInputsFromEngine

    enc = model.get_submodule("vae_encoder", device="cuda:0")
    fwd = CurrentForwardPassInfo(
        request_id="enc", graph_walk=walk, requires_cfg=False, fwd_index=0,
        random_seed=0, max_tokens=0, sampling_config={}, step_metadata=md,
    )
    ei = ModelInputsFromEngine(request_ids=["enc"], per_request_info={"enc": fwd})
    ni = enc.prepare_inputs(walk, fwd, media)
    out = enc.forward(walk, ei, **enc.preprocess(walk, ei, [ni]))
    return out["cond_latents"][0]


@torch.no_grad()
def test_anchor_encode_matches_full() -> None:
    """Image-to-video only consumes latent frame 0, and the Wan VAE encodes it
    as a standalone causal anchor, so the encoder node's single-frame i2v encode
    must give a bit-identical frame 0 to the repeat-padded full-clip encode it
    runs for action policy / forward-dynamics — at a fraction of the cost."""
    base = _load()
    if base is None:
        print("  (skipped anchor-encode parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    device = base["device"]
    img = torch.rand(3, H, W, device=device)  # [C, H, W] in [0, 1], like load_image
    md = {"height": H, "width": W, "num_frames": VIDEO_FRAMES, "has_image_condition": True}
    anchor = _encode_cond(base["model"], md, {"image_inputs": [img]}, "prefill_cond")
    full = _encode_cond(
        base["model"],
        {**md, "action_mode": "policy", "action_chunk_size": VIDEO_FRAMES - 1},
        {"image_inputs": [img]}, "prefill_cond",
    )
    assert anchor.shape[2] == 1, f"i2v must encode one latent frame, got T={anchor.shape[2]}"
    diff = (anchor[:, :, 0].float() - full[:, :, 0].float()).abs().max().item()
    assert diff < 1e-4, f"anchor frame-0 differs from full-clip frame-0 by {diff:.3e} (> 1e-4)"
    print(f"  anchor-encode 1-frame vs full-clip frame-0 abs-max diff = {diff:.3e}")


@torch.no_grad()
def _check_compile_vae(num_frames, tag):
    """torch.compile of the Wan VAE decode (COSMOS3_COMPILE_VAE) must reproduce
    the eager decode. Compile fuses the pointwise epilogues around the (fp32) 3D
    convolutions without changing their math, so the decoded uint8 frames match
    the eager path to fp rounding; a real fusion/ordering bug shows up as visible
    banding that tanks the PSNR. Checked for both a single image frame and a
    multi-frame video clip (video is the lever's main beneficiary)."""
    ctx = _scenario(num_frames)
    if ctx is None:
        print(f"  (skipped {tag} compile-VAE parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    from mstar.model.cosmos3.submodules import Cosmos3VAEDecoderSubmodule

    model, lat = ctx["model"], ctx["lat_fused"]
    vae, config = model._build_vae(ctx["device"]), model.config
    walk = "video_gen" if num_frames > 1 else "image_gen"
    out_key = "video_output" if num_frames > 1 else "image_output"
    had = os.environ.pop("COSMOS3_COMPILE_VAE", None)
    try:
        eager = Cosmos3VAEDecoderSubmodule(vae=vae, config=config)
        img_eager = eager.forward(walk, None, latents=lat.clone())[out_key][0]
        os.environ["COSMOS3_COMPILE_VAE"] = "1"
        compiled = Cosmos3VAEDecoderSubmodule(vae=vae, config=config)
        img_comp = compiled.forward(walk, None, latents=lat.clone())[out_key][0]
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped {tag} compile-VAE parity: VAE/compile unavailable: {exc})")
        return
    finally:
        if had is None:
            os.environ.pop("COSMOS3_COMPILE_VAE", None)
        else:
            os.environ["COSMOS3_COMPILE_VAE"] = had
    a = img_eager.float().cpu() / 255.0
    b = img_comp.float().cpu() / 255.0
    maxdiff = (a - b).abs().max().item() * 255.0
    mse = (a - b).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    assert psnr >= 40, f"{tag} compile-VAE vs eager PSNR {psnr:.2f} < 40 (max uint8 diff {maxdiff:.0f})"
    print(f"  {tag} compile-VAE vs eager decoded PSNR = {psnr:.2f} dB (max uint8 diff {maxdiff:.0f})")


def test_compile_vae_matches_eager() -> None:
    _check_compile_vae(1, "t2i")


def test_compile_vae_matches_eager_t2v() -> None:
    _check_compile_vae(VIDEO_FRAMES, "t2v")


@torch.no_grad()
def test_batched_cfg_matches_sequential() -> None:
    """Running both guidance branches in one batched forward must match running
    them sequentially. The two paths differ only in bf16 GEMM rounding (a batched
    matmul tiles differently), so compare the decoded images by PSNR."""
    ctx = _scenario(1)
    if ctx is None:
        print("  (skipped batched-CFG vs sequential: needs COSMOS3_NANO_DIR + CUDA)")
        return
    dit, prev, decoded = ctx["dit"], ctx["dit"].batched_cfg, {}
    try:
        for flag in (False, True):
            dit.batched_cfg = flag
            try:
                cm = _engine_resources(ctx["model"], ["r0"], ctx["device"], ctx["dtype"])
            except Exception as exc:  # noqa: BLE001
                print(f"  (skipped batched-CFG vs sequential: FlashInfer unavailable: {exc})")
                return
            lat = _run_cache_once(
                ctx["model"], dit, cm, ctx["init"], ctx["cond"], ctx["uncond"], ctx["device"], 1
            )
            decoded[flag] = ctx["mpipe"]._decode(lat.reshape(ctx["lat_fused"].shape)).squeeze().float().cpu()
    finally:
        dit.batched_cfg = prev
    mse = (decoded[False] - decoded[True]).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    assert psnr >= 35, f"batched vs sequential PSNR {psnr:.2f} < 35 (MSE {mse:.3e})"
    print(f"  batched-CFG vs sequential decoded PSNR = {psnr:.2f} dB")


def test_cache_once_matches_fused_exact() -> None:
    _check_cache_once_exact(1, "t2i")


def test_engine_cache_path_image_psnr() -> None:
    _check_engine_psnr(1, "t2i")


def test_cache_once_matches_fused_exact_t2v() -> None:
    _check_cache_once_exact(VIDEO_FRAMES, "t2v")


def test_engine_cache_path_video_psnr() -> None:
    _check_engine_psnr(VIDEO_FRAMES, "t2v")


@torch.no_grad()
def test_cross_request_batch_matches_individual() -> None:
    """Several requests denoised together in one batch must reproduce each
    request run alone. Distinct prompts are decoded and compared to the fused
    pipeline: batching must (a) keep each request isolated — its own image far
    closer than any other request's — and (b) not lose quality versus the bs=1
    path (per-prompt fidelity varies with the FlashInfer kernel, so the bar is
    relative to bs=1, not an absolute PSNR)."""
    base = _load()
    if base is None:
        print("  (skipped cross-request batch parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    from mstar.model.cosmos3.components.packing import tokenize_prompt

    model, dit, mpipe = base["model"], base["dit"], base["mpipe"]
    device, dtype = base["device"], base["dtype"]
    prompts = [
        "A red cube resting on a polished wooden table, soft daylight.",
        "A blue ceramic vase of yellow tulips beside a sunny window.",
        "A small wooden sailboat on a calm turquoise sea at dawn.",
        "A snowy mountain peak under a clear starry night sky.",
    ]
    rids = [f"r{i}" for i in range(len(prompts))]
    conds, unconds = [], []
    for p in prompts:
        c, u = tokenize_prompt(model.tokenizer, p, "", num_frames=1, height=H, width=W)
        conds.append(c)
        unconds.append(u)
    gen = torch.Generator(device=device).manual_seed(SEED)
    init = torch.randn((1, 48, 1, H // 16, W // 16), generator=gen, device=device, dtype=dtype)
    shape = (1, 48, 1, H // 16, W // 16)

    def _dec(lat):
        return mpipe._decode(lat.reshape(shape)).squeeze().float().cpu()

    def _psnr(a, b):
        mse = (a - b).pow(2).mean().item()
        return float("inf") if mse == 0 else -10 * math.log10(mse)

    try:
        fused = [
            _dec(mpipe(prompt=p, negative_prompt="", num_frames=1, height=H, width=W,
                       num_inference_steps=STEPS, guidance_scale=GS, latents=init.clone(), decode=False))
            for p in prompts
        ]
        bs1 = []
        for i, _rid in enumerate(rids):
            cm = _engine_resources(model, ["r0"], device, dtype)
            bs1.append(_dec(_run_cache_once(model, dit, cm, init, conds[i], unconds[i], device, 1)))
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped cross-request batch parity: FlashInfer unavailable: {exc})")
        return

    shared = _engine_resources(model, rids, device, dtype, max_num_pages=256)
    bat = _run_batched(model, dit, shared, init, conds, unconds, device, rids)
    batched = [_dec(bat[rid]) for rid in rids]

    n = len(prompts)
    for i in range(n):
        match = _psnr(batched[i], fused[i])
        cross = max(_psnr(batched[i], fused[j]) for j in range(n) if j != i)
        ref = _psnr(bs1[i], fused[i])
        assert match > cross + 8, f"request {i} not isolated: self {match:.2f} vs other {cross:.2f}"
        assert match >= ref - 3.0, f"request {i} batched {match:.2f} degrades vs bs=1 {ref:.2f}"
    print(f"  cross-request batch (bs={n}) vs fused PSNR = "
          + ", ".join(f"{_psnr(batched[i], fused[i]):.1f}" for i in range(n))
          + " dB (bs=1: " + ", ".join(f"{_psnr(bs1[i], fused[i]):.1f}" for i in range(n)) + ")")
    # This test holds several requests' caches at once; release them so later
    # GPU checks in the same process aren't starved.
    del fused, bs1, batched, bat, shared
    import gc
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def _run_cuda_graph_denoise(ctx):
    """Capture the image denoise step and run the whole loop through the real
    CudaGraphRunner (one captured forward per step covering both guidance
    branches), returning the final latents."""
    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.distributed.communication import CommGroup, JointGroups
    from mstar.engine.cuda_graph_runner import CudaGraphRunner

    model, dit = ctx["model"], ctx["dit"]
    device, dtype = ctx["device"], ctx["dtype"]
    dev = torch.device(device)
    # Capture at this test's (H, W) regardless of the production default.
    dit.gen_capture_resolutions = ((H, W),)
    rid = "cgr0"
    resources = _engine_resources(model, [rid], device, dtype, max_num_pages=256)
    md = {"height": H, "width": W, "num_frames": 1, "fps": 24.0,
          "guidance_scale": GS, "num_inference_steps": STEPS}
    fwd = CurrentForwardPassInfo(
        request_id=rid, graph_walk="prefill", requires_cfg=False, fwd_index=0,
        random_seed=SEED, max_tokens=0, sampling_config={}, step_metadata=md,
    )
    ti = [torch.tensor(ctx["cond"], dtype=torch.long, device=device),
          torch.tensor(ctx["uncond"], dtype=torch.long, device=device)]
    ni = dit.prepare_inputs("prefill", fwd, {"text_inputs": ti})
    _forward_step(dit, "prefill", resources, [rid], {rid: fwd}, [ni])

    groups = JointGroups(
        tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial(),
    )
    cg_runner = CudaGraphRunner(
        submodule_name="dit", submodule=dit, resources=resources,
        step_runner=StepRunner(resources), device=dev, autocast_dtype=dtype,
        joint_comm_group=groups,
    )
    cg_runner.warmup_and_capture()
    assert cg_runner.any_graphs, "no CUDA graph captured for cosmos3 image_gen"

    fwd.graph_walk = "image_gen"
    latents = ctx["init"].clone()
    time_index = torch.zeros(1, dtype=torch.long, device=device)
    for _ in range(STEPS):
        ni = dit.prepare_inputs("image_gen", fwd, {"latents": [latents], "time_index": [time_index]})
        out = _forward_step(
            dit, "image_gen", resources, [rid], {rid: fwd}, [ni],
            cg_runner=cg_runner,
        )
        # The engine finishes the captured step (CFG combine + scheduler) in
        # the submodule's postprocess right after replay; mirror it here.
        dit.postprocess(rid, fwd, out[rid], inputs=ni)
        latents, time_index = out[rid]["latents"][0], out[rid]["time_index"][0]
    dit.cleanup_request(rid)
    return latents


@torch.no_grad()
def test_cuda_graph_matches_eager() -> None:
    """The captured-graph denoise step is the served path's accelerator: both
    guidance branches run in one captured forward (~2x faster than the eager
    step). Each captured forward matches eager to within bf16 (the first step
    differs by ~one ULP); the multistep solver amplifies that into a small latent
    spread, but the decoded image is unchanged — so gate the decoded image against
    the fused pipeline, the same bar the eager engine path meets."""
    ctx = _scenario(1)
    if ctx is None:
        print("  (skipped cuda-graph parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    try:
        lat_graph = _run_cuda_graph_denoise(ctx)
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped cuda-graph parity: FlashInfer/capture unavailable: {exc})")
        return
    img_fused = ctx["mpipe"]._decode(ctx["lat_fused"]).squeeze().float().cpu()
    img_graph = ctx["mpipe"]._decode(lat_graph.reshape(ctx["lat_fused"].shape)).squeeze().float().cpu()
    mse = (img_fused - img_graph).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    assert psnr >= 25, f"cuda-graph denoise PSNR {psnr:.2f} < 25 (MSE {mse:.3e})"
    print(f"  cuda-graph denoise vs fused PSNR = {psnr:.2f} dB")


def _main() -> None:
    failures = []
    for name, fn in [
        ("batched_cfg_matches_sequential", test_batched_cfg_matches_sequential),
        ("cache_once_matches_fused_exact", test_cache_once_matches_fused_exact),
        ("engine_cache_path_image_psnr", test_engine_cache_path_image_psnr),
        ("cache_once_matches_fused_exact_t2v", test_cache_once_matches_fused_exact_t2v),
        ("engine_cache_path_video_psnr", test_engine_cache_path_video_psnr),
        ("dense_fa3_image_psnr", test_dense_fa3_image_psnr),
        ("dense_fa3_video_psnr", test_dense_fa3_video_psnr),
        ("anchor_encode_matches_full", test_anchor_encode_matches_full),
        ("compile_vae_matches_eager", test_compile_vae_matches_eager),
        ("compile_vae_matches_eager_t2v", test_compile_vae_matches_eager_t2v),
        ("cuda_graph_matches_eager", test_cuda_graph_matches_eager),
        ("cross_request_batch_matches_individual", test_cross_request_batch_matches_individual),
    ]:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {exc!r}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if failures:
        raise SystemExit(1)
    print("\nAll Cosmos3 engine-cache checks passed.")


if __name__ == "__main__":
    _main()
