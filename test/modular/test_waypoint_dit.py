"""Contract tests for the Waypoint-1.5 DiT block and the 4+1 per-frame driver.

The bar is the reference implementation at ``world_engine/src/``. The failure
modes covered here all produce plausible video and raise nothing: a
denoise pass that commits to the ring, a value residual threaded the wrong way
round, a missing ``.clone()`` between the two passes of a frame, an fp32 island
that came back bf16, or a derived RoPE table left as whatever ``to_empty``
happened to allocate.

Sections: the 4+1 pass structure, the 720P structural facts, fp32 islands and
the meta build, value-residual ordering, and ``cond_proj`` tying across
``to_empty``.

The model owns no cache. Everything below drives the DiT against the two engine
resources it is bound to -- ``RingKVManager`` and ``FlexAttentionManager``, built
here directly rather than through an engine -- so the ring geometry, the
visibility rows and the attention numerics are the served ones and a test can
assert on what the model *did* without changing what it computed.

CPU-only and checkpoint-free. The real 720P config is used only for structural
assertions, which are cheap because the module is built on ``torch.device("meta")``
and never materialized -- a real 720P bf16 build is ~2.6 GB. Anything that runs
tensors through the model uses a reduced but structurally identical config
(4 layers, one global at stride 8, controller fusion on ``i % 3 == 0``, GQA live).
"""

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttnBackend
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.config import KVSpec, RingKVConfig, RingKVLayerConfig
from mstar.engine.resources.kv.ring import RingKVManager
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.waypoint.components.attention import WaypointAttention
from mstar.model.waypoint.components.dit import WaypointDiT
from mstar.model.waypoint.components.layers import FP32_MODULE_PATHS, rms_norm
from mstar.model.waypoint.components.rope import OrthoRoPEAngles, apply_ortho_rope
from mstar.model.waypoint.config import WaypointConfig, waypoint_1_5_1b_720p

TPF = 128  # tokens per frame in the reduced config


def reduced_config(**overrides) -> WaypointConfig:
    """4 layers / 128 tokens per frame, keeping every structural fact of 720P:
    one global layer (3) at stride 8, controller fusion on ``i % 3 == 0``,
    GQA at 2 query heads over 1 KV head, ``d_head`` divisible by 8."""
    base = {
        "n_layers": 4,
        "n_heads": 2,
        "n_kv_heads": 1,
        "d_model": 64,
        "mlp_ratio": 2,
        "channels": 4,
        "tokens_per_frame": TPF,
        "height": 8,
        "width": 16,
        "local_window": 4,
        "global_window": 32,
        "global_pinned_dilation": 8,
        "n_buttons": 8,
        # Most CPU unit inputs are fp32. Reference-compatible serving is bf16
        # and has its own numerical tests; keep these structural tests exact.
        "reference_compat": False,
    }
    return WaypointConfig(**{**base, **overrides})


def ring_kv_spec(config: WaypointConfig, *, num_worlds: int = 1) -> KVSpec:
    """The ``RingKVConfig`` a ``WaypointConfig``'s geometry implies.

    Hand-rolled here because the model does not declare its specs yet; when
    ``WaypointModel`` grows a ``get_node_resources``, this becomes a call to it
    and the duplication goes away. Every field is read off the config rather
    than restated, so a geometry change cannot leave the tests measuring a ring
    the model no longer asks for.
    """
    return KVSpec(
        resource_key="kv",
        nodes={"dit"},
        config=RingKVConfig(
            num_layers=config.n_layers,
            num_kv_heads=config.n_kv_heads,
            head_dim=config.d_head,
            num_qo_heads=config.n_heads,
            tokens_per_frame=config.tokens_per_frame,
            num_worlds=num_worlds,
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


def build_resources(config: WaypointConfig, dtype: torch.dtype = torch.float32):
    """The two resources the DiT calls, built the way the engine builds them.

    Through ``Resource.build(spec, info)`` and ``AttentionManager.build``'s
    factory, not the constructors: the spec is what picks ``RingKVManager`` over
    the paged one and what cross-checks the flex backend against a ring config,
    and a test that bypassed it would be driving a pairing the engine would
    refuse.
    """
    kv_spec = ring_kv_spec(config)
    cpu = torch.device("cpu")
    kv = RingKVManager.build(kv_spec, EngineResourceInfo(device=cpu, kv_dtype=dtype))
    attn = AttentionManager.build(
        AttentionSpec(
            resource_key="attn",
            nodes={"dit"},
            config=AttentionConfig(kv_cache="kv", backend=AttnBackend.FLEX),
        ),
        EngineResourceInfo(device=cpu, kv_dtype=dtype, dependencies={"kv": kv_spec}),
    )
    return kv, attn


class DitNode(NodeSubmodule):
    """Stands in for the node submodule that will own the DiT.

    Structural parity with phase 6, where the DiT is a child of the node rather
    than the node itself. ``bind_node_resources`` walks ``self.modules()`` and
    skips ``self``, so binding through the node is what the real submodule does;
    the 24 attention layers are the only consumers either way.
    """

    def __init__(self, dit: WaypointDiT):
        super().__init__()
        self.dit = dit

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        raise NotImplementedError("structural stand-in; nothing here runs a step")

    def forward(self, graph_walk, engine_inputs, **kwargs):
        raise NotImplementedError("structural stand-in; nothing here runs a step")


def bind_resources_on(dit: WaypointDiT, kv, attn) -> DitNode:
    """Bind through the engine's own walk rather than a copy of it: one call on
    the node is what has to reach the driver *and* every layer."""
    node = DitNode(dit)
    node.bind_node_resources({"kv": kv, "attn": attn})
    return node


class RecordingRingKV:
    """A real ``RingKVManager`` with every model-facing call logged.

    Delegation rather than a stub: the ring geometry, the visibility rows and
    (through the attention resource it is paired with) the numerics stay real,
    so a test can assert on what the model *did* without changing what it
    computed. Only the KV side is spied on -- the attention resource is passed
    through untouched, because nothing here needs to know what the kernel was
    handed, only what was committed to the world state.
    """

    def __init__(self, config: WaypointConfig, dtype: torch.dtype = torch.float32):
        self.config = config
        self.inner, self.attn = build_resources(config, dtype=dtype)
        self.upserts: list[dict] = []
        self.resets = 0
        self.states: list[str] = []

    # -- the model-facing surface, delegated ---------------------------------

    def upsert(
        self, k, v, layer_idx, frame_pos, *, commit, build_visibility=True,
    ):
        del build_visibility
        self.upserts.append(
            {
                "commit": commit,
                "layer": layer_idx,
                "frame_pos": int(frame_pos),
                "k": k.detach().clone(),
                "v": v.detach().clone(),
            }
        )
        return self.inner.upsert(k, v, layer_idx, frame_pos, commit=commit)

    def reset(self):
        self.resets += 1
        self.inner.reset()
        self.upserts.clear()

    def get_state(self, rid):
        self.states.append("get")
        return self.inner.get_state(rid)

    def load_state(self, rid, state):
        self.states.append("load")
        self.inner.load_state(rid, state)

    # -- introspection the tests read ----------------------------------------

    @property
    def layers(self):
        return self.inner.layers

    def capacity(self, layer_idx: int) -> int:
        return self.inner.capacity(layer_idx)

    def total_slots(self, layer_idx: int) -> int:
        return self.inner.total_slots(layer_idx)

    @property
    def tokens_per_frame(self) -> int:
        return self.inner.tokens_per_frame

    def passes(self) -> list[list[dict]]:
        """The upsert log regrouped into forwards: one entry per layer, in
        layer order, per pass. A misgrouping means the model did not visit every
        layer exactly once per forward, which the assertion below catches."""
        n = self.config.n_layers
        assert len(self.upserts) % n == 0, f"{len(self.upserts)} upserts is not a whole number of passes"
        grouped = [self.upserts[i : i + n] for i in range(0, len(self.upserts), n)]
        for group in grouped:
            assert [u["layer"] for u in group] == list(range(n))
        return grouped


def bound_dit(config: WaypointConfig, seed: int = 0, dtype: torch.dtype = torch.float32):
    """A reduced DiT wired to a recording ring and a real flex attention."""
    dit = build_reduced_dit(config, seed=seed)
    kv = RecordingRingKV(config, dtype=dtype)
    bind_resources_on(dit, kv, kv.attn)
    return dit, kv


def build_reduced_dit(config: WaypointConfig, seed: int = 0) -> WaypointDiT:
    torch.manual_seed(seed)
    dit = WaypointDiT(config).eval()
    for p in dit.parameters():
        torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        for block in dit.blocks:
            block.attn.v_lamb.fill_(0.25)
    dit.retie_cond_proj()
    return dit


def frame_inputs(config: WaypointConfig, seed: int = 3, dtype: torch.dtype = torch.float32):
    gen = torch.Generator().manual_seed(seed)
    C, H, W = config.latent_shape
    noise = torch.randn(1, 1, C, H, W, generator=gen, dtype=torch.float32).to(dtype)
    mouse = torch.tensor([[[0.25, -0.5]]], dtype=dtype)
    button = torch.zeros(1, 1, config.n_buttons, dtype=dtype)
    button[..., 2] = 1.0
    scroll = torch.tensor([[[1.0]]], dtype=dtype)
    return noise, mouse, button, scroll


# ---------------------------------------------------------------------------
# 6. Structural facts of the real 720P config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def meta_720p() -> WaypointDiT:
    """The real model, built on meta: full parameter tree, zero bytes of storage."""
    with torch.device("meta"):
        return WaypointDiT(waypoint_1_5_1b_720p())


def test_720p_layer_geometry(meta_720p):
    config = meta_720p.config
    assert config.n_layers == len(meta_720p.blocks) == 24
    assert sorted(config.global_layers) == [3, 7, 11, 15, 19, 23]
    assert len(config.global_layers) == 6 and config.n_layers - len(config.global_layers) == 18
    assert (config.global_attn_period, config.global_attn_offset) == (4, -1)


def test_720p_controller_conditioning_layers(meta_720p):
    """8 of 24 blocks fuse controller input, and the parameter tree itself
    records which: the other 16 have no ``ctrl_mlpfusion`` submodule at all."""
    config = meta_720p.config
    assert config.ctrl_conditioning_period == 3
    assert sorted(config.ctrl_layers) == [0, 3, 6, 9, 12, 15, 18, 21]

    fusing = [i for i, block in enumerate(meta_720p.blocks) if block.ctrl_mlpfusion is not None]
    assert fusing == [0, 3, 6, 9, 12, 15, 18, 21]
    assert config.d_ctrl_in == 259
    assert meta_720p.ctrl_emb.mlp.fc1.in_features == 259


def test_720p_gqa_is_live_and_the_fused_qkv_slabs_are_unequal(meta_720p):
    """32 query heads over 16 KV heads. The unequal slab widths are why a
    Q/K/V order mistake is a shape error for Q but *not* between K and V --
    swapping those two loads cleanly and produces wrong video."""
    config = meta_720p.config
    assert (config.n_heads, config.n_kv_heads, config.d_head) == (32, 16, 64)
    assert config.enable_gqa is True

    attn = meta_720p.blocks[0].attn
    assert (attn.q_out, attn.kv_out) == (2048, 1024)
    assert attn.qkv_proj.weight.shape == (2048 + 2 * 1024, 2048)
    assert attn.qkv_proj.bias is None and attn.out_proj.bias is None
    assert attn.enable_gqa is True


def test_720p_head_and_patchify_shapes(meta_720p):
    config = meta_720p.config
    ph, pw = config.patch
    assert (config.tokens_per_frame, config.height, config.width) == (512, 16, 32)
    assert config.latent_shape == (32, 32, 64)
    assert meta_720p.patchify.weight.shape == (2048, 32, ph, pw)
    assert meta_720p.patchify.bias is None
    # The checkpoint's [D, C, ph, pw] conv kernel becomes this Linear's
    # [C*ph*pw, D] (loader transforms 1-2).
    assert meta_720p.unpatchify.weight.shape == (32 * ph * pw, 2048)
    assert meta_720p.unpatchify.bias is not None
    assert meta_720p.out_norm.fc.weight.shape == (2 * 2048, 2048)


def test_720p_parameter_budget_and_cond_proj_tying(meta_720p):
    """1.86B stored / 1.28B resident: ``cond_proj`` has one physical set that
    all 24 blocks alias. ``named_parameters()`` deduplicates; ``state_dict()``
    does not, which is why the loader's completeness check runs against the
    former."""
    params = dict(meta_720p.named_parameters())
    assert sum(p.numel() for p in params.values()) == 1_281_958_040
    assert len(params) == 174
    assert len(meta_720p.state_dict()) == 174 + 23 * 6  # 23 aliased blocks x 6 matrices

    ref = meta_720p.blocks[0].cond_head.cond_proj
    for block in meta_720p.blocks[1:]:
        for j in range(6):
            assert block.cond_head.cond_proj[j].weight is ref[j].weight
        # bias_in is genuinely per-layer -- the checkpoint has 24 distinct values.
        assert block.cond_head.bias_in is not meta_720p.blocks[0].cond_head.bias_in
    assert len([n for n in params if n.endswith("cond_head.bias_in")]) == 24
    assert len([n for n in params if "cond_head.cond_proj" in n]) == 6

    # No buffers at all: nothing in the module tree for to_empty to leave
    # holding garbage.
    assert list(meta_720p.named_buffers()) == []


def test_config_rejects_unsupported_checkpoint_variants():
    with pytest.raises(ValueError, match="rope_impl"):
        WaypointConfig(rope_impl="llama")
    with pytest.raises(ValueError, match="noise_conditioning"):
        WaypointConfig(noise_conditioning="dit_air")
    with pytest.raises(ValueError, match="MoE"):
        WaypointConfig(moe=True)
    with pytest.raises(ValueError, match="prompt cross-attention"):
        WaypointConfig(prompt_conditioning="t5")
    with pytest.raises(ValueError, match="ungated attention path only"):
        WaypointAttention(WaypointConfig(gated_attn=True), 0)


# ---------------------------------------------------------------------------
# 6b. The resource binding
# ---------------------------------------------------------------------------


def test_binding_the_node_reaches_every_layer_and_the_driver_holds_nothing():
    """The attention layers are the *only* consumers of either resource.

    The driver holds neither: it decides which pass commits and passes that
    down as an argument, so there is no handle on it to leave unbound. That is
    what removes the half-bind trap this test used to guard -- binding on the
    DiT itself, which ``bind_node_resources`` cannot reach (it walks
    ``self.modules()`` and skips ``self``), used to leave ``dit.kv`` at None and
    raise nothing until four Euler steps into a frame. Asserted rather than
    assumed, because a future handle on the driver would silently bring it back.
    """
    config = reduced_config()
    with torch.device("meta"):
        dit = WaypointDiT(config)
    kv, attn = build_resources(config)
    layers = [block.attn for block in dit.blocks]
    assert all(la.kv is None and la.attn is None for la in layers)

    # The way phase 6's submodule does it: the DiT as a child.
    bind_resources_on(dit, kv, attn)

    assert all(la.kv is kv and la.attn is attn for la in layers)
    assert len(layers) == config.n_layers
    assert not hasattr(dit, "kv") and not hasattr(dit, "attn"), (
        "the driver took a resource handle back; `commit` is an argument and "
        "nothing else on the DiT root talks to the ring"
    )


def test_a_layer_binds_only_what_the_node_owns():
    """``.get``, not ``[]``: a layer may sit on a node that owns some of its
    resources and not others, and the missing one has to read as absent rather
    than raise at load."""
    config = reduced_config()
    with torch.device("meta"):
        dit = WaypointDiT(config)
    kv, _attn = build_resources(config)

    DitNode(dit).bind_node_resources({"kv": kv})

    assert dit.blocks[0].attn.kv is kv
    assert dit.blocks[0].attn.attn is None


# ---------------------------------------------------------------------------
# 7. The 4+1 pass structure
# ---------------------------------------------------------------------------


def test_generate_frame_is_four_frozen_denoise_passes_then_one_commit():
    """The single most important invariant. Five forwards: four
    Euler steps at sigma = 1.0, 0.9, 0.75, 0.3 that must not touch the ring,
    then one committing pass at sigma = 0 on the settled latent."""
    config = reduced_config()
    dit, kv = bound_dit(config)
    noise, mouse, button, scroll = frame_inputs(config)

    sigmas: list[torch.Tensor] = []
    handle = dit.denoise_step_emb.register_forward_pre_hook(
        lambda _m, args: sigmas.append(args[0].detach().clone())
    )
    try:
        with torch.no_grad():
            dit.generate_frame(
                noise, torch.tensor(0, dtype=torch.int64),
                mouse=mouse, button=button, scroll=scroll,
            )
    finally:
        handle.remove()

    # [B, N] per pass, one uniform sigma per frame.
    assert [tuple(s.shape) for s in sigmas] == [(1, 1)] * 5
    assert [s.item() for s in sigmas] == pytest.approx([1.0, 0.9, 0.75, 0.3, 0.0], abs=1e-6)
    assert list(config.scheduler_sigmas) == [1.0, 0.9, 0.75, 0.3, 0.0]
    assert config.num_denoise_steps == 4

    passes = kv.passes()
    assert len(passes) == 5, "a generated frame costs exactly five forwards"
    # Per pass, not per resource: `commit` arrives as an argument on every one
    # of the n_layers upserts, so a single-element set per pass is also the
    # assertion that no layer disagreed with the driver about which pass it was.
    commit_per_pass = [{u["commit"] for u in group} for group in passes]
    assert commit_per_pass == [{False}, {False}, {False}, {False}, {True}]
    # All five passes of a frame share one ring clock.
    assert {u["frame_pos"] for u in kv.upserts} == {0}
    # ...and nothing else on the resource. Dropping the world and snapshotting
    # it belong to the request lifecycle a level up; a driver that reset between
    # frames would be starting a new rollout every frame, silently.
    assert (kv.resets, kv.states) == (0, [])


def test_the_ring_only_moves_on_the_committing_pass():
    """The behavioural half of the same invariant, measured on the ring itself."""
    config = reduced_config()
    dit, kv = bound_dit(config)
    noise, mouse, button, scroll = frame_inputs(config)
    fp = torch.tensor(0, dtype=torch.int64)

    ring_lens = [layer.ring_len for layer in kv.layers]
    before = [layer.kv[:, :, :, :n].clone() for layer, n in zip(kv.layers, ring_lens, strict=True)]

    with torch.no_grad():
        sigma_table = dit._sigma_schedule(noise.device, noise.dtype)
        dit._denoise_pass(noise, fp, sigma_table, mouse=mouse, button=button, scroll=scroll)
    after_denoise = [layer.kv[:, :, :, :n] for layer, n in zip(kv.layers, ring_lens, strict=True)]
    assert all(torch.equal(a, b) for a, b in zip(before, after_denoise, strict=True))
    assert not any(layer.written[: layer.ring_len].any() for layer in kv.layers)

    with torch.no_grad():
        dit._cache_pass(noise, fp, mouse=mouse, button=button, scroll=scroll)
    assert all(layer.written[: layer.tokens_per_frame].all() for layer in kv.layers)


def test_generate_frame_clones_the_denoised_latent():
    """``x0 = self._denoise_pass(...).clone()`` -- the
    ``.clone()`` is load-bearing. Both passes run inside one CUDA-graph capture,
    so the cache pass allocates from the graph's private pool, and the denoise
    pass's output buffer is a block in that pool that nothing downstream holds:
    the cache pass's first allocation can land on it and stomp the latent it is
    supposed to be reading, with the address baked into the graph. The copy must
    land in caller-owned memory, i.e. outside the compiled region.
    """
    config = reduced_config()
    dit, _kv = bound_dit(config)
    noise, mouse, button, scroll = frame_inputs(config)

    produced: list[torch.Tensor] = []
    real_denoise = dit._denoise_pass

    def spy(*args, **kwargs):
        out = real_denoise(*args, **kwargs)
        produced.append(out)
        return out

    dit._denoise_pass = spy
    with torch.no_grad():
        x0 = dit.generate_frame(
            noise, torch.tensor(0, dtype=torch.int64),
            mouse=mouse, button=button, scroll=scroll,
        )

    assert len(produced) == 1
    assert x0 is not produced[0], "generate_frame returned the compiled region's own buffer"
    assert x0.data_ptr() != produced[0].data_ptr(), "generate_frame aliases the denoise output"
    assert torch.equal(x0, produced[0])
    assert x0.shape == noise.shape


def test_append_frame_is_the_committing_pass_alone():
    """Priming from a VAE-encoded real frame: no denoising, one forward, and the
    latent is already the settled x0 so there is nothing to clone."""
    config = reduced_config()
    dit, kv = bound_dit(config)
    latent, mouse, button, scroll = frame_inputs(config)

    sigmas: list[torch.Tensor] = []
    handle = dit.denoise_step_emb.register_forward_pre_hook(
        lambda _m, args: sigmas.append(args[0].detach().clone())
    )
    try:
        with torch.no_grad():
            out = dit.append_frame(
                latent, torch.tensor(0, dtype=torch.int64),
                mouse=mouse, button=button, scroll=scroll,
            )
    finally:
        handle.remove()

    assert [s.flatten().tolist() for s in sigmas] == [[0.0]]
    assert len(kv.passes()) == 1
    assert all(u["commit"] is True for u in kv.upserts), "priming must commit"
    assert out is latent


def test_sigma_schedule_is_built_in_the_latent_dtype():
    """``_sigma_schedule``: the reference takes ``.diff()`` in the
    serving dtype, so the Euler step sizes are bf16 differences of bf16 sigmas.
    Building the table in fp32 "for precision" changes the ODE."""
    config = reduced_config()
    dit = build_reduced_dit(config)
    cpu = torch.device("cpu")

    bf16 = dit._sigma_schedule(cpu, torch.bfloat16)
    fp32 = dit._sigma_schedule(cpu, torch.float32)
    assert bf16.dtype == torch.bfloat16 and fp32.dtype == torch.float32
    assert dit._sigma_schedule(cpu, torch.bfloat16) is bf16, "the schedule is memoized per (device, dtype)"

    bf16_diff = bf16.diff()
    fp32_diff_rounded = fp32.diff().to(torch.bfloat16)
    assert bf16_diff[0].item() == pytest.approx(-0.1015625, abs=0)
    assert fp32_diff_rounded[0].item() == pytest.approx(-0.10009765625, abs=0)
    assert not torch.equal(bf16_diff, fp32_diff_rounded)


# ---------------------------------------------------------------------------
# 8. The value residual
# ---------------------------------------------------------------------------


class CaptureKV:
    """Records the exact ``(k, v)`` handed to the ring and hands them straight
    back, so a test can inspect what would have been stored forever.

    Deliberately NOT a ``RingKVManager``, unlike ``RecordingRingKV`` above: what
    these tests read is the frame the layer *submitted*, and a real ring returns
    the whole capacity with that frame scattered into a slot the test would then
    have to find. Returning ``(k, v, None)`` keeps the KV the layer built and
    the KV the kernel sees the same tensor.
    """

    def __init__(self):
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def upsert(
        self, k, v, layer_idx, frame_pos, *, commit, build_visibility=True,
    ):
        del build_visibility
        self.calls.append((k.detach().clone(), v.detach().clone()))
        return k, v, None


class DenseAttn:
    """The attention half of the pair above: plain SDPA over whatever
    ``CaptureKV`` returned. ``visible`` is ``None`` and ignored -- there is no
    ring here to be visible into, and these tests are about what enters the
    cache, not about the mask."""

    requires_kv_write = False

    def attend(self, q, k, v, visible, *, enable_gqa, layer_idx=None):
        del layer_idx
        assert visible is None, "CaptureKV returns no visibility row"
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=enable_gqa)


def run_two_attention_layers(config: WaypointConfig, lamb0: float, lamb1: float):
    torch.manual_seed(21)
    layer0 = WaypointAttention(config, 0).eval()
    layer1 = WaypointAttention(config, 1).eval()
    capture, dense = CaptureKV(), DenseAttn()
    for layer, lamb in ((layer0, lamb0), (layer1, lamb1)):
        for p in layer.parameters():
            torch.nn.init.normal_(p, std=0.05)
        with torch.no_grad():
            layer.v_lamb.fill_(lamb)
        # Per-layer bind, directly: these two are loose layers, not a node's
        # module tree, and `bind_resources` is the seam a layer actually has.
        layer.bind_resources({"kv": capture, "attn": dense})

    idx = torch.arange(config.tokens_per_frame)
    angles = OrthoRoPEAngles(config)(
        x_pos=idx.remainder(config.width)[None],
        y_pos=idx.div(config.width, rounding_mode="floor")[None],
        t_pos=torch.zeros(1, config.tokens_per_frame, dtype=torch.long),
    )
    gen = torch.Generator().manual_seed(22)
    x = torch.randn(1, config.tokens_per_frame, config.d_model, generator=gen)
    fp = torch.tensor(0, dtype=torch.int64)
    with torch.no_grad():
        _, v1 = layer0(x, fp, angles, None, commit=True)
        _, v1_out = layer1(x, fp, angles, v1, commit=True)
    return layer0, layer1, x, angles, capture, v1, v1_out


def raw_qkv(layer: WaypointAttention, x: torch.Tensor):
    with torch.no_grad():
        q, k, v = layer.qkv_proj(x).split((layer.q_out, layer.kv_out, layer.kv_out), dim=-1)
    B, T = x.shape[:2]
    return (
        q.reshape(B, T, layer.n_heads, layer.d_head).transpose(1, 2),
        k.reshape(B, T, layer.n_kv_heads, layer.d_head).transpose(1, 2),
        v.reshape(B, T, layer.n_kv_heads, layer.d_head).transpose(1, 2),
    )


def test_v1_is_captured_pre_lerp_and_threads_through_unchanged():
    """Layer 0 returns its *pre*-lerp V, and every later layer
    passes that same tensor along untouched -- it does not substitute its own.
    Getting this backwards still produces plausible output, so the ordering is
    asserted directly rather than through the activations."""
    config = reduced_config()
    layer0, layer1, x, _angles, capture, v1, v1_out = run_two_attention_layers(config, 0.25, 0.5)

    _, _, v_raw0 = raw_qkv(layer0, x)
    _, _, v_raw1 = raw_qkv(layer1, x)

    assert torch.equal(v1, v_raw0), "v1 is not layer 0's raw, pre-lerp V"
    assert v1_out is v1, "a later layer replaced v1 instead of threading it through"
    assert not torch.equal(v_raw1, v1), "precondition: the two layers' raw V differ"

    # ...and the LERPED V is what enters the cache, at layer 1.
    cached_v1 = capture.calls[1][1]
    want = torch.lerp(v_raw1, v1, layer1.v_lamb)
    assert torch.equal(cached_v1, want)
    assert not torch.equal(cached_v1, v_raw1), "the cache stored the pre-lerp V"
    assert not torch.equal(cached_v1, v1)


def test_value_residual_lerp_direction():
    """``torch.lerp(v, v1, w)`` is ``v + w * (v1 - v)``: at ``v_lamb == 1`` the
    cached V *is* layer 0's V, at 0 it is the layer's own. Swapping the lerp
    operands is a silent sign flip on the residual."""
    config = reduced_config()
    _, layer1, x, _, capture_one, v1, _ = run_two_attention_layers(config, 0.25, 1.0)
    assert torch.equal(capture_one.calls[1][1], v1)

    _, layer1_zero, x0, _, capture_zero, v1_zero, _ = run_two_attention_layers(config, 0.25, 0.0)
    _, _, v_raw1 = raw_qkv(layer1_zero, x0)
    assert torch.equal(capture_zero.calls[1][1], v_raw1)


def test_q_and_k_are_normed_and_rotated_but_v_is_neither():
    """Q/K are RMS-normed then RoPE'd; V is neither. K enters the
    ring already rotated, so replayed history is never re-rotated."""
    config = reduced_config()
    layer0, _layer1, x, angles, capture, v1, _ = run_two_attention_layers(config, 0.25, 0.5)
    _, k_raw0, v_raw0 = raw_qkv(layer0, x)

    cached_k, cached_v = capture.calls[0]
    assert torch.equal(cached_k, apply_ortho_rope(rms_norm(k_raw0), angles))
    assert not torch.equal(cached_k, rms_norm(k_raw0)), "K reached the cache un-rotated"
    assert not torch.equal(cached_k, apply_ortho_rope(k_raw0, angles)), "K reached the cache un-normed"
    # Layer 0 lerps against itself, so its cached V is exactly its raw V.
    assert torch.equal(cached_v, v_raw0)
    assert torch.equal(v1, v_raw0)


def test_layer_zero_v_reaches_every_block_in_the_dit():
    """End to end: with every ``v_lamb`` at 1 the residual is total, so the V
    stored by all 4 blocks must be byte-identical to layer 0's. A block that
    re-captured ``v1`` from its own projection would drift here."""
    config = reduced_config()
    dit, kv = bound_dit(config)
    with torch.no_grad():
        for block in dit.blocks:
            block.attn.v_lamb.fill_(1.0)

    latent, mouse, button, scroll = frame_inputs(config)
    with torch.no_grad():
        dit.append_frame(
            latent, torch.tensor(0, dtype=torch.int64),
            mouse=mouse, button=button, scroll=scroll,
        )

    (single_pass,) = kv.passes()
    reference_v = single_pass[0]["v"]
    for record in single_pass[1:]:
        assert torch.equal(record["v"], reference_v), (
            f"block {record['layer']} did not thread layer 0's V into the cache"
        )
    # ...and K, which is not part of the residual, does differ per layer.
    assert not torch.equal(single_pass[1]["k"], single_pass[0]["k"])


# ---------------------------------------------------------------------------
# 9. fp32 islands and the meta-build path
# ---------------------------------------------------------------------------


def test_fp32_module_paths_is_exactly_the_noise_conditioner():
    """The reference marks three modules ``NoCastModule``; only one of them has
    parameters. ``OrthoRoPEAngles``/``OrthoRoPE`` hold none, so there is nothing
    for a dtype cast to corrupt and nothing to pin back."""
    assert FP32_MODULE_PATHS == ("denoise_step_emb",)


def test_cast_serving_dtypes_leaves_one_fp32_island_on_720p():
    """bf16 everywhere, then the islands back to fp32 -- run on
    the **meta** module so storage is later allocated directly in the serving
    dtype. Nothing is materialized here; a real 720P build is ~2.6 GB."""
    with torch.device("meta"):
        dit = WaypointDiT(waypoint_1_5_1b_720p())
    assert dit.cast_serving_dtypes() is dit

    by_dtype: dict[torch.dtype, list[str]] = {}
    for name, param in dit.named_parameters():
        by_dtype.setdefault(param.dtype, []).append(name)

    assert sorted(by_dtype[torch.float32]) == [
        "denoise_step_emb.mlp.fc1.weight",
        "denoise_step_emb.mlp.fc2.weight",
    ]
    assert set(by_dtype) == {torch.float32, torch.bfloat16}
    assert not [n for n in by_dtype[torch.bfloat16] if n.startswith("denoise_step_emb")]
    assert dit.dtype == torch.bfloat16
    # .to(dtype) on meta preserves the aliasing; to_empty is what breaks it.
    ref = dit.blocks[0].cond_head.cond_proj
    assert all(b.cond_head.cond_proj[j].weight is ref[j].weight for b in dit.blocks[1:] for j in range(6))


def test_to_empty_unties_cond_proj_and_retie_puts_it_back():
    """``Module._apply`` has no cross-module memo, so
    ``to_empty(device)`` silently gives 24 blocks 24 independent ``cond_proj``
    sets. Nothing raises; the symptoms are +0.6B resident parameters and 23
    blocks the loader never fills. Measured here on the reduced config, where
    ``to_empty`` is affordable -- the mechanism is size-independent."""
    config = reduced_config()
    with torch.device("meta"):
        dit = WaypointDiT(config)
    dit.cast_serving_dtypes()

    def tied() -> bool:
        ref = dit.blocks[0].cond_head.cond_proj
        return all(b.cond_head.cond_proj[j].weight is ref[j].weight for b in dit.blocks[1:] for j in range(6))

    n_tied = len(list(dit.named_parameters()))
    assert tied()

    dit.to_empty(device="cpu")
    assert not tied(), "to_empty preserved the tying; the retie contract may no longer be needed"
    untied_count = len(list(dit.named_parameters()))
    assert untied_count == n_tied + (config.n_layers - 1) * 6

    assert dit.retie_cond_proj() is dit
    assert tied()
    assert len(list(dit.named_parameters())) == n_tied


def test_derived_tables_survive_the_meta_build():
    """The RoPE frequency tables, the Fourier
    frequency table and the token grid are DERIVED state held in a
    ``DeviceTableCache`` *outside* the module tree. As non-persistent buffers
    they would come out of ``to_empty(device)`` as uninitialized garbage that no
    loader completeness check covers -- a silent wrong-numbers bug."""
    config = reduced_config()
    with torch.device("meta"):
        meta_dit = WaypointDiT(config)
    meta_dit.cast_serving_dtypes()
    meta_dit.to_empty(device="cpu")
    meta_dit.retie_cond_proj()

    torch.manual_seed(0)
    plain_dit = WaypointDiT(config)
    cpu = torch.device("cpu")

    tables = {
        "rope": lambda d: d.rope_angles._tables.get(cpu),
        "fourier": lambda d: d.denoise_step_emb._freq.get(cpu),
        "grid": lambda d: d._grid.get(cpu),
    }
    for name, getter in tables.items():
        from_meta, from_plain = getter(meta_dit), getter(plain_dit)
        for a, b in zip(from_meta, from_plain, strict=True):
            assert a.device.type == "cpu" and not a.is_meta, f"{name} table is still on meta"
            assert torch.equal(a, b), f"{name} table differs after the meta build"
            if a.is_floating_point():
                assert a.dtype == torch.float32 and bool(torch.isfinite(a).all())

    # None of them are module state: not buffers, not in state_dict, out of
    # reach of both to_empty and the bf16 cast.
    assert list(meta_dit.named_buffers()) == []
    assert not [k for k in meta_dit.state_dict() if "freq" in k or "grid" in k or "_tables" in k]


def test_meta_built_model_generates_a_frame_in_the_serving_dtypes():
    """The whole build order, end to end on CPU: meta build,
    cast, to_empty, retie, then a real 4+1 frame. Weights are random (there is
    no checkpoint here), so the bar is 'the serving dtypes and the derived
    tables are live and the output is finite', not a numeric one."""
    config = reduced_config()
    with torch.device("meta"):
        dit = WaypointDiT(config)
    dit.cast_serving_dtypes()
    dit.to_empty(device="cpu")
    dit.retie_cond_proj()
    torch.manual_seed(1)
    for p in dit.parameters():
        torch.nn.init.normal_(p, std=0.02)
    dit.eval()

    kv = RecordingRingKV(config, dtype=torch.bfloat16)
    bind_resources_on(dit, kv, kv.attn)
    noise, mouse, button, scroll = frame_inputs(config, dtype=torch.bfloat16)
    with torch.no_grad():
        x0 = dit.generate_frame(
            noise, torch.tensor(0, dtype=torch.int64),
            mouse=mouse, button=button, scroll=scroll,
        )

    assert x0.dtype == torch.bfloat16 and x0.shape == noise.shape
    assert bool(torch.isfinite(x0.float()).all())
    assert len(kv.passes()) == 5
    assert dit.denoise_step_emb.mlp.fc1.weight.dtype == torch.float32
    assert dit.patchify.weight.dtype == torch.bfloat16


def test_two_frames_advance_the_ring_clock_together():
    """The caller owns ``frame_pos`` and advances it by exactly one per
    committed frame; ``t_pos = f_pos * ts_mult`` is the RoPE clock and
    is threaded separately even though ``ts_mult == 1`` here."""
    config = reduced_config()
    assert config.ts_mult == 1 == waypoint_1_5_1b_720p().ts_mult

    dit, kv = bound_dit(config)
    noise, mouse, button, scroll = frame_inputs(config)
    with torch.no_grad():
        for f in range(2):
            dit.generate_frame(
                noise, torch.tensor(f, dtype=torch.int64),
                mouse=mouse, button=button, scroll=scroll,
            )

    passes = kv.passes()
    assert len(passes) == 10
    assert [next(iter({u["frame_pos"] for u in group})) for group in passes] == [0] * 5 + [1] * 5

    pos = dit._pos_ids(torch.tensor(3, dtype=torch.int64))
    assert pos.f_pos.item() == 3 and pos.f_pos.ndim == 0
    assert pos.t_pos.shape == (1, config.tokens_per_frame)
    assert bool((pos.t_pos == 3 * config.ts_mult).all())
    assert bool((pos.y_pos == torch.arange(TPF).div(config.width, rounding_mode="floor")).all())
    assert bool((pos.x_pos == torch.arange(TPF).remainder(config.width)).all())
