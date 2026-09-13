"""Port vs the Waypoint reference implementation: one forward, one frame, one rollout.

The reference is driven live in-process rather than read out of the oracle's
recorded activations. ``test/waypoint/record_oracle.py`` calls ``engine.model``
directly and so bypasses the engine's two ``torch.compile`` regions, which means its
``dit_out`` and ``committed_kv`` were produced by *eager* ``flex_attention`` -- and
eager flex ignores a ``BlockMask``'s block index lists, attending over the whole ring
including unwritten slots (``test_flex_attention_resource.py`` pins both halves of
that). The oracle stays authoritative for inputs (noise, controls, the seed latent)
and for kernel-independent ring bookkeeping (``written``, which buckets are live); its
activations are not the reference as served.

Both sides here run everything eager except one shared attention kernel:
``src.patch_model.flex_attention`` is rebound to the port's own
``flex_attention_masked``, so attention cannot be the variable under test.

**Every test is run on two ports**, built from the same checkpoint and differing only
in ``WaypointConfig.reference_compat``:

  * ``reference_compat=True`` reproduces the reference's lossy derived state -- the
    bf16 round-trip its ``NoCastModule._apply`` leaves in three non-persistent fp32
    tables, and the batch-5 sigma LUT of ``patch_cached_noise_conditioning``. The bar
    there is bit-exact, and no tolerance below is looser than zero.
  * The experimental exact-table port is deliberately more mathematically direct than
    the reference and so
    diverges from it. What the exact tests assert is the shape of that divergence:
    that it is present, that it is confined downstream of the three tables, and that
    nothing kernel-independent (the ring's ``written`` masks) moves with it.

``test_every_stage_is_bit_exact_under_reference_compat`` is what earns the zero
tolerances: with the flag on, all 30 stages plus the output agree exactly, at a frozen
and at a committing sigma. So a nonzero under the flag is a port bug, not accumulated
rounding.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
)
from mstar.engine.resources.attn.flex import flex_attention_masked
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.config import KVSpec, RingKVConfig, RingKVLayerConfig, RingKVStep
from mstar.engine.resources.kv.ring import RingKVManager
from mstar.engine.resources.step import StepContext
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.waypoint.components.layers import bf16_roundtrip
from mstar.model.waypoint.config import waypoint_1_5_1b_720p
from mstar.model.waypoint.weight_loader import build_waypoint_dit

_ROOT = Path("/mnt/storage/garv901/waypoint-1.5-1B")
REFERENCE_SRC = Path(os.environ.get("WAYPOINT_REFERENCE_SRC", _ROOT / "world_engine"))
CHECKPOINT = Path(os.environ.get("WAYPOINT_CHECKPOINT", _ROOT / "checkpoints/Waypoint-1.5-1B"))
ORACLE = Path(os.environ.get("WAYPOINT_ORACLE_DIR", _ROOT / "oracle"))

# Frames in the rollout layer. The local ring holds 16 frames, so this wraps it
# twice; the oracle recorded 41 (a seed plus 40 generated) and holds full ring
# snapshots at 0, 20 and 40.
ROLLOUT_FRAMES = int(os.environ.get("WAYPOINT_PARITY_FRAMES", "41"))
# The exact port diverges at frame 0 and compounds; a short rollout is enough to
# measure it, and the reference_compat run is the full-length bit-exact one.
DIVERGENCE_FRAMES = 6

# Explicit index, not bare "cuda": the reference's ``BaseModel.from_pretrained``
# asserts one so it can hand safetensors an ordinal to load onto.
DEVICE = torch.device("cuda", 0)
DTYPE = torch.bfloat16

# The two ports every test below is run against. ``ids`` are what a failure is
# reported under, so they say which numerics were expected, not which flag was set.
COMPAT_MODES = pytest.mark.parametrize("compat", [False, True], ids=["exact", "reference_compat"])

# Stages that read none of the three tables and so must agree with the reference in
# both modes. Their divergence would mean the port drifted somewhere new.
TABLE_INDEPENDENT_STAGES = ("patchify", "ctrl_emb")
# ``cond`` reads ``denoise_step_emb.freq``, ``rope`` reads ``rope_angles.xy``/
# ``inv_t``; with the exact tables both must differ from the reference.
TABLE_DEPENDENT_STAGES = ("cond", "rope")


def _reference_importable() -> bool:
    return (REFERENCE_SRC / "src" / "model" / "world_model.py").exists()


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(not _reference_importable(), reason=f"reference not at {REFERENCE_SRC}"),
    pytest.mark.skipif(not (CHECKPOINT / "model.safetensors").exists(), reason="checkpoint missing"),
    pytest.mark.skipif(not (ORACLE / "frames").is_dir(), reason=f"oracle not at {ORACLE}"),
]


def _import_reference():
    sys.path.insert(0, str(REFERENCE_SRC))
    import src as reference_pkg

    sys.modules.setdefault("world_engine", reference_pkg)
    from src import patch_model
    from src.model import StaticKVCache, WorldModel

    return WorldModel, StaticKVCache, patch_model


# ---------------------------------------------------------------------------
# Builds
# ---------------------------------------------------------------------------


class _DitNode(NodeSubmodule):
    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError("binding stand-in")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("binding stand-in")


@pytest.fixture(scope="module")
def reference():
    """The served reference: inference patches applied, flex pinned to the port's
    compiled kernel. The three fp32 island tables are captured *before* patching
    -- ``CachedDenoiseStepEmb`` keeps no handle back to the module it replaces.
    """
    WorldModel, StaticKVCache, patch_model = _import_reference()
    # The oracle recorded at 'high'; its own calibration measured high and medium
    # bit-identical for this model, and high vs highest differing (metadata.json).
    # It also decides the reference's sigma LUT: that table is a batch-5 fp32 GEMM,
    # which rounds through TF32 here and not at 'highest'.
    torch.set_float32_matmul_precision("high")

    cfg = WorldModel.load_config(str(CHECKPOINT))
    model = WorldModel.from_pretrained(str(CHECKPOINT), cfg=cfg, device=DEVICE, dtype=DTYPE).eval()
    islands = {
        "freq": model.denoise_step_emb.freq.clone(),
        "xy": model.transformer.rope_angles.xy.clone(),
        "inv_t": model.transformer.rope_angles.inv_t.clone(),
    }
    bare_conditioner = model.denoise_step_emb
    bare_cond_head = model.transformer.blocks[0].cond_head

    patch_model.apply_inference_patches(model)
    patch_model.flex_attention = flex_attention_masked
    cache = StaticKVCache(cfg, batch_size=1, dtype=DTYPE).to(device=DEVICE)
    yield {
        "cfg": cfg,
        "model": model,
        "kv": cache,
        "islands": islands,
        "bare_conditioner": bare_conditioner,
        "bare_cond_head": bare_cond_head,
    }


def _build_port(config, checkpoint: Path = CHECKPOINT):
    """Build a port from the checkpoint belonging to ``config``.

    The default preserves the original 720p harness. The explicit path is used
    by the 360p live-reference gate, whose weights are a distinct publication.
    """
    dit = build_waypoint_dit(config, str(checkpoint), device=DEVICE)
    spec = KVSpec(
        resource_key="kv",
        nodes={"dit"},
        config=RingKVConfig(
            num_layers=config.n_layers,
            num_kv_heads=config.n_kv_heads,
            head_dim=config.d_head,
            num_qo_heads=config.n_heads,
            tokens_per_frame=config.tokens_per_frame,
            num_worlds=1,
            layers=tuple(
                RingKVLayerConfig(
                    ring_frames=config.ring_frames(i),
                    ring_buckets=config.ring_buckets(i),
                    pinned_dilation=config.pinned_dilation(i),
                )
                for i in range(config.n_layers)
            ),
        ),
    )
    info = EngineResourceInfo(device=DEVICE, kv_dtype=DTYPE)
    kv = RingKVManager.build(spec, info)
    attn = AttentionManager.build(
        AttentionSpec(
            resource_key="attn",
            nodes={"dit"},
            config=AttentionConfig(kv_cache="kv", backend=AttnBackend.FLEX),
        ),
        EngineResourceInfo(device=DEVICE, kv_dtype=DTYPE, dependencies={"kv": spec}),
    )
    _DitNode(dit).bind_node_resources({"kv": kv, "attn": attn})
    assert not dit._regions_compiled
    return {
        "config": config,
        "dit": dit,
        "kv": kv,
        "attn": attn,
        "rid": None,
        "seq": 0,
    }


@pytest.fixture(scope="module")
def ports(reference):
    """Both ports, keyed by ``reference_compat``. Two full checkpoint loads, and
    two rings, so a test can hold one side's state while comparing the other.

    Depends on ``reference`` for the ordering, not the object: the compat port's
    sigma LUT is a fp32 GEMM whose result depends on
    ``float32_matmul_precision``, and the reference fixture is what sets it.
    """
    # This harness decomposes the reference's five passes in Python so it can
    # compare every intermediate. Keep the port's outer driver eager too;
    # Keep outer compilation off so this test isolates numerical compatibility;
    # compilation and CUDA graph selection have separate execution-mode gates.
    default = replace(waypoint_1_5_1b_720p(), compile_dit=False)
    exact = replace(default, reference_compat=False)
    return {False: _build_port(exact), True: _build_port(default)}


@pytest.fixture(scope="module")
def oracle():
    frames = sorted((ORACLE / "frames").glob("frame_*.pt"))
    return {"frames": frames, "rings": ORACLE / "ring"}


def _frame(oracle, index: int) -> dict:
    return torch.load(oracle["frames"][index], map_location="cpu", weights_only=False)


def _ctx(frame: dict) -> dict:
    return {k: v.to(DEVICE) for k, v in frame["ctx"].items()}


# ---------------------------------------------------------------------------
# Driving both sides
# ---------------------------------------------------------------------------


def _reset(port, reference):
    for layer in reference["kv"].layers:
        layer.reset()
    for layer in port["kv"].layers:
        layer.reset(0)


def _reference_forward(reference, x, sigma_value: float, ctx, *, commit: bool):
    reference["kv"].set_frozen(not commit)
    with torch.inference_mode():
        sigma = x.new_full((x.size(0), x.size(1)), sigma_value)
        return reference["model"](x, sigma, **ctx, kv_cache=reference["kv"]).clone()


def _port_forward(port, x, sigma_value: float, ctx, frame_pos: int, *, commit: bool):
    with torch.inference_mode():
        sigma = x.new_full((x.size(0), x.size(1)), sigma_value)
        return port["dit"](
            x,
            sigma,
            torch.tensor(frame_pos, dtype=torch.int64, device=DEVICE),
            mouse=ctx["mouse"],
            button=ctx["button"],
            scroll=ctx["scroll"],
            commit=commit,
        ).clone()


def _reference_frame(reference, noise, ctx, sigmas):
    """The reference's ``_denoise_pass`` + ``_cache_pass``, unrolled so every pass
    output is observable. ``zip(sigmas, sigmas.diff())`` is 4 steps over a 5-entry
    schedule; the fifth pass is the committing one at sigma 0.
    """
    cache = reference["kv"]
    outputs = []
    cache.set_frozen(True)
    x = noise
    sigma = x.new_empty((x.size(0), x.size(1)))
    with torch.inference_mode():
        for step_sigma, step_dsigma in zip(sigmas[:-1], sigmas.diff(), strict=True):
            v = reference["model"](x, sigma.fill_(step_sigma), **ctx, kv_cache=cache)
            outputs.append(v.clone())
            x = (x.float() + step_dsigma.float() * v.float()).type_as(x)
        x0 = x.clone()
        cache.set_frozen(False)
        outputs.append(reference["model"](x0, x0.new_zeros((1, 1)), **ctx, kv_cache=cache).clone())
    return x0, outputs


def _port_frame(port, noise, ctx, frame_pos: int):
    """``generate_frame`` with every pass output captured. Hooked rather than read
    from the return value: ``generate_frame`` hands back only the settled latent, and
    the committing pass's velocity -- the one that proves the fifth pass ran on the
    right input -- is discarded inside it."""
    outputs = []
    dit = port["dit"]
    handle = dit.register_forward_hook(lambda mod, inputs, output: outputs.append(output.clone()))
    _admit(port, frame_pos)
    try:
        with torch.inference_mode():
            x0 = dit.generate_frame(
                noise,
                torch.tensor(frame_pos, dtype=torch.int64, device=DEVICE),
                mouse=ctx["mouse"],
                button=ctx["button"],
                scroll=ctx["scroll"],
            )
    finally:
        handle.remove()
    _commit(port, frame_pos)
    return x0, outputs


def _new_request(port) -> None:
    """A fresh request id per test. The manager enforces ``frame == last + 1``, so a
    test that replays frame 0 or starts at frame 21 needs its own clock; the old
    request has to hand its world back first."""
    kv = port["kv"]
    if port["rid"] is not None:
        kv.reset_request(port["rid"], free=True)
        kv.remove_request(port["rid"])
    port["rid"] = f"parity{port['seq']}"
    port["seq"] += 1
    kv.ingest_request(port["rid"])


def _step_and_context(port, frame: int):
    rid = port["rid"]
    return (
        RingKVStep(frames=((rid, frame),)),
        StepContext(request_ids=(rid,), graph_walk="rollout", slot=0, capture=False),
    )


def _admit(port, frame: int) -> None:
    """Stage the fixed-address world index and local/global attention masks."""
    step, context = _step_and_context(port, frame)
    outcome = port["kv"].admit(step, context)
    assert outcome.ok, f"refused at frame {frame}: {outcome.reason}"
    context.plan_results["kv"] = port["kv"].plan(step, context)
    port["attn"].plan(AttentionStep(), context)
    assert not port["attn"].needs_token_visibility


def _commit(port, frame: int) -> None:
    port["kv"].commit(*_step_and_context(port, frame))


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def _deviation(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """``(max abs, max abs / peak of the reference)``."""
    left, right = a.float(), b.float()
    peak = left.abs().max().item()
    gap = (left - right).abs().max().item()
    return gap, gap / max(peak, 1e-30)


def _stage_modules(model, *, is_port: bool):
    blocks = model.blocks if is_port else model.transformer.blocks
    rope = model.rope_angles if is_port else model.transformer.rope_angles
    return (
        [("patchify", model.patchify), ("ctrl_emb", model.ctrl_emb), ("cond", model.denoise_step_emb), ("rope", rope)]
        + [(f"block{i}", block) for i, block in enumerate(blocks)]
        + [("out_norm", model.out_norm), ("unpatchify", model.unpatchify)]
    )


@contextlib.contextmanager
def _stage_capture(modules, store):
    """Keyed by name, not appended: the two sides run these modules in different
    orders (the port builds RoPE before the conditioner), and positional pairing
    would silently compare the wrong rows."""
    handles = []
    for name, module in modules:

        def hook(mod, inputs, output, name=name):
            tensor = output[0] if isinstance(output, tuple) else output
            store[name] = (tensor[0] if isinstance(tensor, tuple) else tensor).detach().float().cpu()

        handles.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _divergent_stages(reference_stages, port_stages, names):
    return [
        (name, *_deviation(reference_stages[name], port_stages[name]))
        for name in names
        if not torch.equal(reference_stages[name], port_stages[name])
    ]


def _comparable_ring(reference_layer, port_layer):
    """One world of ring state, in a shape the two sides share.

    The reference allocates 128 frame slots for a global layer but addresses only
    its 16 buckets, so its live region is ``[0, port ring_len)`` and its scratch is
    the tail ``[L, capacity)``. The port compacts the gap away. Callers assert the
    gap stays clear rather than trusting it.
    """
    ring_len = port_layer.ring_len
    reference_kv = torch.cat(
        (reference_layer.kv[:, :, :, :ring_len], reference_layer.kv[:, :, :, reference_layer.L :]),
        dim=3,
    )
    reference_written = torch.cat((reference_layer.written[:ring_len], reference_layer.written[reference_layer.L :]))
    return (reference_kv, reference_written), (port_layer.kv, port_layer.written)


def _ring_deviation(reference, port) -> list[tuple[int, float, float, bool]]:
    """Per layer: ``(index, max abs, relative, written masks equal)``."""
    rows = []
    for i, (reference_layer, port_layer) in enumerate(zip(reference["kv"].layers, port["kv"].layers, strict=True)):
        (ref_kv, ref_written), (port_kv, port_written) = _comparable_ring(reference_layer, port_layer)
        gap, relative = _deviation(ref_kv, port_kv)
        rows.append((i, gap, relative, torch.equal(ref_written, port_written)))
    return rows


def _assert_ring(ring, *, exact: bool):
    """``written`` is kernel- and table-independent, so it must match in both modes.
    The KV bytes must match only under ``reference_compat``."""
    assert all(row[3] for row in ring), "ring visibility masks differ"
    worst = max(ring, key=lambda row: row[1])
    if exact:
        assert worst[1] == 0.0, f"ring differs at layer {worst[0]}: maxabs={worst[1]:.4e}"
    return worst


# ---------------------------------------------------------------------------
# L0 -- the derived fp32 tables
# ---------------------------------------------------------------------------


def _island_tables(port, reference):
    dit = port["dit"]
    (port_freq,) = dit.denoise_step_emb._freq.get(DEVICE)
    port_xy, port_inv_t = dit.rope_angles._tables.get(DEVICE)
    return {
        "denoise_step_emb.freq": (reference["islands"]["freq"], port_freq),
        "rope_angles.xy": (reference["islands"]["xy"], port_xy),
        "rope_angles.inv_t": (reference["islands"]["inv_t"], port_inv_t),
    }


def test_the_fp32_island_tables_match_the_reference_under_reference_compat(ports, reference):
    """``reference_compat`` rebuilds the reference's lossy derived state, and these
    three tables are where it starts."""
    failures = []
    for name, (reference_table, port_table) in _island_tables(ports[True], reference).items():
        gap, relative = _deviation(reference_table, port_table)
        print(f"{name:>24}  maxabs={gap:.4e}  rel={relative:.4e}")
        if gap != 0.0:
            failures.append(f"{name}: maxabs={gap:.4e} rel={relative:.4e}")
    assert not failures, "compat fp32 tables differ from the reference's: " + "; ".join(failures)


def test_the_exact_tables_differ_from_the_reference_by_exactly_a_bf16_roundtrip(ports, reference):
    """The characterisation of the served divergence, at zero tolerance.

    The reference's ``NoCastModule._apply`` round-trips every tensor it holds through
    the requested dtype -- ``fn(t)`` casts fp32 to bf16, then the guard casts the
    *result* back. Parameters recover, because ``load_state_dict`` runs after the cast
    and refills them from the checkpoint. These three derived non-persistent buffers
    never get refilled, so the served reference runs on bf16-quantized frequencies and
    the exact port does not.

    Not a rounding detail. ``freq`` is multiplied by ``sigma * 1000`` and ``xy`` by a
    normalized coordinate, so a 1.8e-3 relative table error is a phase error of up to
    ~1.8 rad, and it is the dominant term in every layer below.
    """
    for name, (reference_table, port_table) in _island_tables(ports[False], reference).items():
        gap, relative = _deviation(reference_table, port_table)
        print(f"{name:>24}  maxabs={gap:.4e}  rel={relative:.4e}")
        assert gap != 0.0, f"{name}: the exact port already matches the reference's lossy table"
        assert torch.equal(bf16_roundtrip(port_table), reference_table), (
            f"{name}: the reference's table is not a bf16 round-trip of the exact one, so "
            "the divergence is no longer only NoCastModule's cast"
        )


# ---------------------------------------------------------------------------
# L1 -- one forward
# ---------------------------------------------------------------------------


def test_cached_noise_conditioning_is_not_a_numerical_no_op(reference):
    """``patch_cached_noise_conditioning`` is applied unconditionally by the reference,
    and the claim was that it is a numerical no-op. It is not, and the mechanism is the
    batch shape, not the LUT.

    Both halves are measured on the reference's own conditioner so the table difference
    above cannot leak in. ``CachedCondHead`` is exact, which is why ``reference_compat``
    covers only the embedding half. ``CachedDenoiseStepEmb`` is not: it builds its table
    by evaluating the fp32 MLP on all five sigmas at once, and at M=5 that dispatches to
    a TF32 tensor-core GEMM under ``float32_matmul_precision='high'``, while serving one
    sigma at a time is an M=1 GEMV that stays exact fp32.
    """
    from src.patch_model import CachedDenoiseStepEmb

    sigmas = list(reference["cfg"].scheduler_sigmas)
    conditioner = reference["bare_conditioner"]
    lut = CachedDenoiseStepEmb(conditioner, sigmas)

    embedding_gaps, head_gaps = [], []
    with torch.inference_mode():
        for value in sigmas:
            sigma = torch.tensor([[value]], device=DEVICE, dtype=DTYPE)
            cached, live = lut(sigma), conditioner(sigma)
            embedding_gaps.append(_deviation(cached, live))
            head_gaps.append(
                max(
                    _deviation(a, b)[0]
                    for a, b in zip(
                        reference["model"].transformer.blocks[0].cond_head(cached),
                        reference["bare_cond_head"](cached),
                        strict=True,
                    )
                )
            )
            print(
                f"sigma={value:<7.4f} embedding maxabs={embedding_gaps[-1][0]:.4e} "
                f"rel={embedding_gaps[-1][1]:.4e}  cond_head maxabs={head_gaps[-1]:.4e}"
            )

    assert max(head_gaps) == 0.0, "CachedCondHead was expected to be exact"
    assert max(gap for gap, _ in embedding_gaps) > 0.0, (
        "CachedDenoiseStepEmb agreed with the live conditioner; if the batch-5 LUT "
        "build has stopped reaching TF32, the port may drop this patch as a no-op"
    )


def test_the_compat_conditioner_reproduces_the_reference_sigma_lut(ports, reference):
    """The second half of ``reference_compat``: the port builds its own batch-5 table
    rather than borrowing the reference's, so this pins the two tables against each
    other row by row, and pins the exact port as the per-sigma path that differs."""
    compat, exact = ports[True]["dit"].denoise_step_emb, ports[False]["dit"].denoise_step_emb
    reference_lut = reference["model"].denoise_step_emb

    live_gaps = []
    with torch.inference_mode():
        for value in reference["cfg"].scheduler_sigmas:
            sigma = torch.tensor([[value]], device=DEVICE, dtype=DTYPE)
            gap, relative = _deviation(reference_lut(sigma), compat(sigma))
            live_gaps.append(_deviation(reference_lut(sigma), exact(sigma))[0])
            print(f"sigma={value:<7.4f} compat maxabs={gap:.4e} rel={relative:.4e} exact maxabs={live_gaps[-1]:.4e}")
            assert gap == 0.0, f"sigma={value}: compat LUT differs from the reference's"

    assert max(live_gaps) > 0.0, (
        "the exact conditioner already agrees with the reference's LUT, so the flag's conditioner half is a no-op"
    )


@COMPAT_MODES
def test_one_forward_matches_the_reference_on_an_empty_ring(ports, reference, oracle, compat):
    """The committing pass of the oracle's seed frame: weight loading, RoPE, adaLN
    and the conditioning head, with nothing in the ring behind it."""
    port = ports[compat]
    frame = _frame(oracle, 0)
    ctx = _ctx(frame)
    x = frame["latent"].to(DEVICE)

    _new_request(port)
    _reset(port, reference)
    reference_stages: dict[str, torch.Tensor] = {}
    with _stage_capture(_stage_modules(reference["model"], is_port=False), reference_stages):
        reference_out = _reference_forward(reference, x, 0.0, ctx, commit=True)
    port_stages: dict[str, torch.Tensor] = {}
    _admit(port, 0)
    with _stage_capture(_stage_modules(port["dit"], is_port=True), port_stages):
        port_out = _port_forward(port, x, 0.0, ctx, 0, commit=True)
    _commit(port, 0)

    names = [name for name, _ in _stage_modules(port["dit"], is_port=True)]
    divergent = _divergent_stages(reference_stages, port_stages, names)
    gap, relative = _deviation(reference_out, port_out)
    for name, stage_gap, stage_relative in divergent[:4]:
        print(f"stage {name:>10} maxabs={stage_gap:.4e} rel={stage_relative:.4e}")
    print(f"output maxabs={gap:.4e} rel={relative:.4e}")
    worst = _assert_ring(_ring_deviation(reference, port), exact=compat)
    print(f"ring worst layer={worst[0]} maxabs={worst[1]:.4e} rel={worst[2]:.4e}")

    if compat:
        assert gap == 0.0, (
            f"velocity differs: maxabs={gap:.4e} rel={relative:.4e}; "
            f"first divergent stage {divergent[0] if divergent else None}"
        )
        return

    # Exact port: the divergence has to reach the output, and it has to start at the
    # table consumers -- anything earlier is a new bug wearing this one's clothes.
    assert gap != 0.0, "the exact port matched the reference; reference_compat is a no-op"
    diverged = {name for name, _, _ in divergent}
    assert not diverged.intersection(TABLE_INDEPENDENT_STAGES), (
        f"stages that read no derived table diverged: {sorted(diverged.intersection(TABLE_INDEPENDENT_STAGES))}"
    )
    assert diverged.issuperset(TABLE_DEPENDENT_STAGES), (
        f"a table consumer agreed with the reference anyway: {sorted(set(TABLE_DEPENDENT_STAGES) - diverged)}"
    )


@COMPAT_MODES
def test_one_forward_matches_the_reference_on_a_wrapped_ring(ports, reference, oracle, compat):
    """The same forward with 20 frames of history behind it, so the ring has wrapped
    once and the dilated global layers hold more than one bucket."""
    port = ports[compat]
    frame = _frame(oracle, 21)
    ctx = _ctx(frame)
    x = frame["noise_bf16"].to(DEVICE)

    _load_snapshot(port, reference, oracle, 20)
    reference_out = _reference_forward(reference, x, 1.0, ctx, commit=False)
    _admit(port, 21)
    port_out = _port_forward(port, x, 1.0, ctx, 21, commit=False)

    gap, relative = _deviation(reference_out, port_out)
    print(f"output maxabs={gap:.4e} rel={relative:.4e}")
    if compat:
        assert gap == 0.0, f"velocity differs at frame 21: maxabs={gap:.4e} rel={relative:.4e}"
    else:
        assert gap != 0.0, "the exact port matched the reference; reference_compat is a no-op"


def test_every_stage_is_bit_exact_under_reference_compat(ports, reference, oracle):
    """The localization: with the flag on, nothing else differs. Both a frozen sigma
    and the committing one, so adaLN's two regimes and the ring write are both covered.

    This is what makes a failure elsewhere attributable. If this test goes red, the
    divergence is no longer only the tables and the LUT, and the layers below are
    measuring something new.
    """
    port = ports[True]
    frame = _frame(oracle, 0)
    ctx = _ctx(frame)
    x = frame["latent"].to(DEVICE)

    for sigma_value in (1.0, 0.0):
        commit = sigma_value == 0.0
        _new_request(port)
        _reset(port, reference)
        reference_stages: dict[str, torch.Tensor] = {}
        with _stage_capture(_stage_modules(reference["model"], is_port=False), reference_stages):
            reference_out = _reference_forward(reference, x, sigma_value, ctx, commit=commit)

        port_stages: dict[str, torch.Tensor] = {}
        _admit(port, 0)
        with _stage_capture(_stage_modules(port["dit"], is_port=True), port_stages):
            port_out = _port_forward(port, x, sigma_value, ctx, 0, commit=commit)

        names = [name for name, _ in _stage_modules(port["dit"], is_port=True)]
        divergent = _divergent_stages(reference_stages, port_stages, names)
        gap, relative = _deviation(reference_out, port_out)
        print(f"sigma={sigma_value}: {len(divergent)} divergent stages, output maxabs={gap:.4e}")
        assert not divergent, f"sigma={sigma_value} first divergence: {divergent[0]}"
        assert gap == 0.0, f"sigma={sigma_value} output maxabs={gap:.4e} rel={relative:.4e}"


# ---------------------------------------------------------------------------
# L2 -- one frame, all five passes
# ---------------------------------------------------------------------------


def _five_pass_deviations(port, reference, oracle):
    frame = _frame(oracle, 1)
    ctx = _ctx(frame)
    noise = frame["noise_bf16"].to(DEVICE)
    sigmas = torch.tensor(list(reference["cfg"].scheduler_sigmas), dtype=DTYPE, device=DEVICE)

    _load_snapshot(port, reference, oracle, 0)
    reference_x0, reference_outs = _reference_frame(reference, noise, ctx, sigmas)
    port_x0, port_outs = _port_frame(port, noise, ctx, 1)

    assert len(reference_outs) == len(port_outs) == 5
    rows = [(i, *_deviation(r, p)) for i, (r, p) in enumerate(zip(reference_outs, port_outs, strict=True))]
    return rows, _deviation(reference_x0, port_x0), _ring_deviation(reference, port)


@COMPAT_MODES
def test_all_five_pass_outputs_of_one_frame_match(ports, reference, oracle, compat):
    """The 4+1 driver: four frozen Euler steps then the committing pass. Every pass
    output is compared, not just the settled latent -- a wrong sigma or a wrong
    ``commit`` on an inner step is invisible in the last one alone."""
    port = ports[compat]
    passes, (x0_gap, x0_relative), ring = _five_pass_deviations(port, reference, oracle)
    for index, gap, relative in passes:
        print(f"pass {index} maxabs={gap:.4e} rel={relative:.4e}")
    print(f"latent maxabs={x0_gap:.4e} rel={x0_relative:.4e}")
    _assert_ring(ring, exact=compat)

    first = next((row for row in passes if row[1] != 0.0), None)
    if compat:
        assert first is None, f"first divergent pass: index={first[0]} maxabs={first[1]:.4e} rel={first[2]:.4e}"
        assert x0_gap == 0.0, f"settled latent maxabs={x0_gap:.4e} rel={x0_relative:.4e}"
        return

    # Exact port: pass 0 runs on an unwrapped, shared ring, so a divergence there is
    # the tables and nothing accumulated. All five passes and the latent carry it.
    assert first is not None and first[0] == 0, f"first divergent pass: {first}"
    assert all(gap != 0.0 for _, gap, _ in passes), f"a pass matched the reference: {passes}"
    assert x0_gap != 0.0, "the settled latent matched the reference"


# ---------------------------------------------------------------------------
# L3 -- rollout
# ---------------------------------------------------------------------------


def _load_snapshot(port, reference, oracle, frame_index: int) -> None:
    """Put both rings into the state the oracle recorded after ``frame_index``.

    The oracle's ring bytes came from the eager-flex recording, so they are not the
    reference as served -- but as a *shared* starting state for both sides they are
    exactly as good as any other, and they cost nothing to produce.
    """
    _new_request(port)
    _reset(port, reference)
    snapshot = torch.load(oracle["rings"] / f"ring_{frame_index:03d}.pt", map_location="cpu", weights_only=False)
    for layer, saved in zip(reference["kv"].layers, snapshot, strict=True):
        layer.kv.copy_(saved["kv"].to(DEVICE))
        layer.written.copy_(saved["written"].to(DEVICE))
    for reference_layer, port_layer in zip(reference["kv"].layers, port["kv"].layers, strict=True):
        (ref_kv, ref_written), _ = _comparable_ring(reference_layer, port_layer)
        port_layer.kv.copy_(ref_kv)
        port_layer.written.copy_(ref_written)


def _rollout(port, reference, oracle, frames: int):
    """Step both sides in lockstep from an empty ring, on the oracle's noise and
    controls. Yields ``(frame, latent deviation, ring rows)`` per frame."""
    _new_request(port)
    _reset(port, reference)
    sigmas = torch.tensor(list(reference["cfg"].scheduler_sigmas), dtype=DTYPE, device=DEVICE)
    for index in range(frames):
        frame = _frame(oracle, index)
        ctx = _ctx(frame)
        if frame["kind"] == "seed":
            # A real VAE-encoded frame: one committing pass, no ODE. Both sides
            # are handed the same latent, so the velocity is what is compared.
            latent = frame["latent"].to(DEVICE)
            left = _reference_forward(reference, latent, 0.0, ctx, commit=True)
            _admit(port, index)
            right = _port_forward(port, latent, 0.0, ctx, index, commit=True)
            _commit(port, index)
        else:
            noise = frame["noise_bf16"].to(DEVICE)
            left, _ = _reference_frame(reference, noise, ctx, sigmas)
            right, _ = _port_frame(port, noise, ctx, index)
        yield index, _deviation(left, right), _ring_deviation(reference, port)


def test_the_served_rollout_diverges_from_the_reference_at_every_frame(ports, reference, oracle):
    """The exact port as served, over enough frames for the ring to start carrying the
    divergence. Reported rather than gated on a magnitude: what is asserted is that it
    is there from frame 0 (so it is the tables, not accumulation), that it never
    accidentally comes back to zero, and that nothing kernel-independent moved.
    """
    port = ports[False]
    visited, clean = 0, []
    for index, (gap, relative), ring in _rollout(port, reference, oracle, DIVERGENCE_FRAMES):
        worst = _assert_ring(ring, exact=False)
        print(
            f"frame {index:3d} latent maxabs={gap:.4e} rel={relative:.4e}  "
            f"ring worst layer={worst[0]} maxabs={worst[1]:.4e}"
        )
        if gap == 0.0 or worst[1] == 0.0:
            clean.append((index, gap, worst[1]))
        visited += 1
    assert visited == DIVERGENCE_FRAMES, f"rollout stopped after {visited} frames"
    assert not clean, (
        f"frames where the exact port matched the reference: {clean}; reference_compat "
        "is not the only thing separating the two sides"
    )


def test_rollout_is_bit_exact_under_reference_compat(ports, reference, oracle):
    """The ring over a full rollout: slot addressing, the dilated write step, and the
    port's compaction of the global layers, all under the one condition that makes a
    nonzero attributable to the ring rather than to arithmetic.

    ``ROLLOUT_FRAMES`` wraps the 16-frame local ring twice.
    """
    port = ports[True]
    frames = min(ROLLOUT_FRAMES, len(oracle["frames"]))
    assert frames > 32, f"need more than two local ring wraps; oracle has {frames} frames"
    visited = 0
    for index, (gap, relative), ring in _rollout(port, reference, oracle, frames):
        assert gap == 0.0, f"frame {index}: latent maxabs={gap:.4e} rel={relative:.4e}"
        _assert_ring(ring, exact=True)
        visited += 1
    # A generator that stopped early would leave every assertion above unrun.
    assert visited == frames, f"rollout stopped after {visited} of {frames} frames"


def test_the_port_compacts_only_slots_the_reference_never_addresses(ports, oracle):
    """The compaction claim, against the oracle rather than against the port's own
    arithmetic: the reference allocates 128 frame slots for a global layer and writes
    16 of them, so the region the port drops is provably dead.

    ``written`` and which buckets hold energy are properties of the ring's addressing,
    not of the attention kernel, so the oracle is authoritative for them even though
    its activations are not.
    """
    port = ports[False]
    config = port["config"]
    checked = 0
    for index in sorted(int(p.stem.split("_")[1]) for p in oracle["rings"].glob("ring_*.pt")):
        snapshot = torch.load(oracle["rings"] / f"ring_{index:03d}.pt", map_location="cpu", weights_only=False)
        for layer_index, (saved, port_layer) in enumerate(zip(snapshot, port["kv"].layers, strict=True)):
            written, kv = saved["written"], saved["kv"]
            if kv.size(3) == port_layer.capacity:
                continue  # local layer: no compaction to check
            dead = slice(port_layer.ring_len, kv.size(3) - config.tokens_per_frame)
            assert not written[dead].any(), (
                f"frame {index} layer {layer_index}: the reference marked "
                f"{int(written[dead].sum())} slots written inside the region the port drops"
            )
            assert kv[:, :, :, dead].eq(0).all(), (
                f"frame {index} layer {layer_index}: nonzero KV inside the dropped region"
            )
            checked += 1
    assert checked, "no global-layer ring snapshots found; the compaction claim is untested"
