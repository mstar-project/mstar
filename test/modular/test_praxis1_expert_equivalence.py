"""Bit-exact equivalence of M*'s Praxis-1 action expert against the reference.

The companion of ``test_praxis1_reference_equivalence.py``, for the action
expert's layer rather than the trunk's pieces. Same ground rules: the
reference is the real package, imported, not a second transcription of our
reading of it, and every comparison is ``torch.equal``.

Run it with the reference checkout on ``PYTHONPATH``::

    PYTHONPATH=/path/to/praxis-1-oss pytest test/modular/test_praxis1_expert_equivalence.py

Each group carries a negative control that has to diverge. Two of them are
worth reading before trusting the group above them:

* ``test_control_rope_and_the_expert_norm_nearly_commute`` is a *rounding*
  control, not a mathematical one. The expert's q/k norm has no weight and
  the expert's rotary rotates every pair, so the rotation preserves each
  head's sum of squares exactly in real arithmetic and the two orders agree
  up to float rounding. That is unlike the trunk, where the norm's per
  channel weight lands after the sum and makes the order matter outright.
* ``test_control_head_plus_zero_only_shows_in_the_sign_bit``. ``+0.0`` moves
  ``-0.0`` to ``+0.0``, and ``torch.equal`` cannot see it because the two
  compare equal. The control reads ``signbit`` instead, which is the only
  way this flag is observable at all.
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch

pytest.importorskip(
    "praxis",
    reason="the Praxis-1 reference checkout is not on PYTHONPATH",
)

from praxis import action_expert as ref_expert  # noqa: E402
from praxis import checkpoint as ref_ckpt  # noqa: E402
from praxis import layers as ref_layers  # noqa: E402

from mstar.model.praxis1.components import action_expert as port_ae  # noqa: E402
from mstar.model.praxis1.components import norm_time as port_nt  # noqa: E402
from mstar.model.praxis1.components import rope as port_rope  # noqa: E402

# The checkpoint's proportions, shrunk: heads x head_dim is the model width,
# so there is no GQA and no fused-qkv width to get wrong. head_dim stays 64,
# the checkpoint's, because it is also the rotary width.
HEAD_DIM = 64
NUM_HEADS = 2
DIM = NUM_HEADS * HEAD_DIM
TRUNK_DIM = 24
ACTION_PATCH_DIM = 5
PROPRIO_PATCH_DIM = 3
FUTURE_PER_CHUNK = 2

# One window: 2 chunks of 2 latent frames, 6 tokens per frame.
CHUNK_SIZE = 2
NUM_CHUNKS = 2
TPF = 6
N_NOISED = CHUNK_SIZE * NUM_CHUNKS
N_CLEAN = CHUNK_SIZE * (NUM_CHUNKS - 1)
N_GLOBAL = 2 + TPF + 3  # registers, the anchor frame, text
Q_LEN = NUM_CHUNKS * (1 + FUTURE_PER_CHUNK) + N_NOISED
KV_LEN = N_GLOBAL + (N_CLEAN + N_NOISED) * TPF

EXPERT_ROPE = {
    "kind": "trunk",
    "dim": HEAD_DIM,
    "effective_dim": HEAD_DIM,
    "base": 26_000_000,
    "table_size": 1_000_000,
    "order": "before_norm",
}

MAX_INTRA, MAX_INTER = 360, 129_600


@pytest.fixture(autouse=True)
def _single_threaded():
    """Pin the intra-op threads: a bit-exact CPU reduction depends on them."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _kv_rope_ids() -> torch.Tensor:
    """Trunk rope ids of the expert KV, ``[global | clean | noised]``.

    The frames start at index 8 of the timeline on purpose: the positions
    then exceed the rotary's ``table_size`` and engage the high/low split,
    which a single early chunk would leave untested.
    """
    hw = torch.tensor([i * MAX_INTRA + j for i in range(2) for j in range(3)])
    frames = [1 + (8 + f) * MAX_INTER + hw for f in range(N_CLEAN + N_NOISED)]
    ids = torch.cat([torch.zeros(N_GLOBAL, dtype=torch.long), *frames])[None]
    assert int(ids.max()) > EXPERT_ROPE["table_size"]
    return ids


def _tables() -> tuple[torch.Tensor, torch.Tensor]:
    """``(q_table, k_table)`` from the port's rotary, at the expert's config."""
    rope = port_rope.RotaryEmbedding(
        dim=HEAD_DIM, effective_dim=HEAD_DIM, base=26_000_000, table_size=1_000_000
    )
    kv_ids = _kv_rope_ids()
    q_ids = ref_expert.trunk_query_rope_ids(
        kv_ids,
        num_global_tokens=N_GLOBAL,
        num_clean_frames=N_CLEAN,
        num_noised_frames=N_NOISED,
        tokens_per_frame=TPF,
        num_action_tokens=N_NOISED,
        num_future_per_chunk=FUTURE_PER_CHUNK,
        chunk_size=CHUNK_SIZE,
    )
    return rope(q_ids), rope(kv_ids)


def _masks() -> tuple[torch.Tensor, torch.Tensor]:
    """The reference's own dense ``(cross, self)`` masks for this window.

    The chunk-causal predicate belongs to the tower wiring, so it is taken
    from the reference rather than ported here; the block only has to behave
    identically under it.
    """
    cross, self_ = ref_expert.build_trunk_rope_masks(
        num_action_tokens=N_NOISED,
        num_future_per_chunk=FUTURE_PER_CHUNK,
        chunk_size=CHUNK_SIZE,
        num_clean_frames=N_CLEAN,
        num_noised_frames=N_NOISED,
        tokens_per_frame=TPF,
        window_size=N_CLEAN + N_NOISED,
        num_global_tokens=N_GLOBAL,
        device=torch.device("cpu"),
        block=False,
    )
    assert cross.shape == (Q_LEN, KV_LEN) and self_.shape == (Q_LEN, Q_LEN)
    return cross, self_


def _reference_expert() -> ref_expert.ActionExpert:
    """A small reference expert with non-degenerate weights everywhere.

    ``tap_norm_scale`` is deliberately randomized rather than left at its
    ones init: at ones the plain-weight and the ``1 + weight`` readings
    differ only by a factor of two uniformly, which is exactly the kind of
    difference a parity test can miss.
    """
    torch.manual_seed(0)
    expert = ref_expert.ActionExpert(
        action_patch_dim=ACTION_PATCH_DIM,
        proprio_patch_dim=PROPRIO_PATCH_DIM,
        trunk_dim=TRUNK_DIM,
        dim=DIM,
        num_layers=4,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_future_per_chunk=FUTURE_PER_CHUNK,
        max_chunks=4,
        max_action_tokens=8,
        rope=EXPERT_ROPE,
        head_plus_zero=True,
        layout_stamp=20260924,
    )
    for p in expert.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return expert.eval().requires_grad_(False)


# ---------------------------------------------------------------------------
# What the reference's contract is, read off the reference
# ---------------------------------------------------------------------------


def test_the_entry_point_is_keyword_only_and_there_is_no_forward():
    """The expert is not callable: the tower has to name the entry.

    ``ActionExpert`` defines no ``forward``, so ``expert(...)`` reaches
    ``nn.Module.forward`` and raises. Every argument of the real entry is
    keyword-only, which is what stops the tower wiring from passing the
    window's geometry positionally in the wrong order.
    """
    assert "forward" not in vars(ref_expert.ActionExpert)
    params = inspect.signature(ref_expert.ActionExpert.forward_trunk_rope).parameters
    assert [n for n, p in params.items() if p.kind is not p.KEYWORD_ONLY] == ["self"]


def test_the_shipped_rope_config_is_the_one_we_build():
    """dim 64 = head_dim, base 26e6, rotation before the q/k norm."""
    assert ref_ckpt.EXPERT_TRUNK_ROPE == EXPERT_ROPE


def test_the_layers_alternate_cross_then_self():
    """Even layers cross-attend a tap, odd layers self-attend. Hence even depth.

    The alternation is the tower's to drive, but it is the reason a block
    only carries ``is_cross``: there is no third kind, and a stack with an
    odd layer count has no pairing.
    """
    expert = _reference_expert()
    assert [b.is_cross for b in expert.blocks] == [True, False, True, False]
    assert expert.num_cross_layers == len(expert.blocks) // 2
    with pytest.raises(ValueError, match="must be even"):
        ref_expert.ActionExpert(
            action_patch_dim=ACTION_PATCH_DIM,
            proprio_patch_dim=PROPRIO_PATCH_DIM,
            trunk_dim=TRUNK_DIM,
            dim=DIM,
            num_layers=3,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            rope=EXPERT_ROPE,
        )


def test_the_expert_carries_no_classifier_free_guidance():
    """Guidance lives in the sampler above the expert, not in it.

    Asserted rather than assumed: an expert that took a guidance scale
    would need the tower to run it twice per step, which is a different
    serving shape, so the absence is part of the port's contract.
    """
    source = inspect.getsource(ref_expert)
    for word in ("guidance", "cfg_scale", "classifier"):
        assert word not in source


def test_expert_rotary_table_matches_and_rotates_every_pair():
    """``effective_dim == dim`` here, unlike the trunk's 64 of 128.

    A zero-frequency pad would leave half the head untouched and still
    produce a table of the right shape.
    """
    ref = ref_layers.RotaryEmbedding(dim=HEAD_DIM, effective_dim=HEAD_DIM)
    got = port_rope.RotaryEmbedding(dim=HEAD_DIM, effective_dim=HEAD_DIM)
    assert torch.equal(got.theta, ref.theta)
    assert (got.theta > 0).all()
    ids = _kv_rope_ids()
    assert torch.equal(got(ids), ref(ids))


def test_control_the_trunk_rope_config_pads_where_the_expert_does_not():
    """Control: the two rotary configs must not coincide.

    If the expert's table happened to equal the trunk's, the test above
    would say nothing about ``effective_dim``.
    """
    trunk = port_rope.RotaryEmbedding(dim=HEAD_DIM, effective_dim=HEAD_DIM // 2)
    assert (trunk.theta == 0).any()
    expert = port_rope.RotaryEmbedding(dim=HEAD_DIM, effective_dim=HEAD_DIM)
    assert not torch.equal(trunk.theta, expert.theta)


# ---------------------------------------------------------------------------
# The building blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_geglu_matches(dtype):
    torch.manual_seed(0)
    x = torch.randn(1, 7, 2 * DIM, dtype=dtype)
    assert torch.equal(port_ae.geglu(x), ref_expert.geglu(x))


def test_control_geglu_halves_are_not_interchangeable():
    """Control: gating on the first half instead of the second must differ."""
    torch.manual_seed(0)
    x = torch.randn(1, 7, 2 * DIM)
    value, gate = torch.chunk(x, 2, dim=-1)
    swapped = torch.nn.functional.gelu(gate, approximate="tanh") * value
    assert not torch.equal(port_ae.geglu(x), swapped)


def test_control_geglu_tanh_differs_from_exact_gelu():
    """Control: the approximation is part of the contract, not a detail."""
    torch.manual_seed(0)
    x = torch.randn(1, 7, 2 * DIM)
    value, gate = torch.chunk(x, 2, dim=-1)
    exact = torch.nn.functional.gelu(value) * gate
    assert not torch.equal(port_ae.geglu(x), exact)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_expert_timestep_embedder_matches(dtype):
    """Sinusoidal features then the MLP, at the checkpoint's frequency_dim."""
    torch.manual_seed(0)
    ref = ref_expert.TimestepEmbedder(DIM)
    got = port_ae.ExpertTimestepEmbedder(DIM)
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.2)
    got.load_state_dict(ref.state_dict())
    ref, got = ref.to(dtype).eval(), got.to(dtype).eval()
    # The raw flow time, in [0, 1]. No 1000x scaling anywhere.
    t = torch.rand(1, N_NOISED, dtype=dtype)
    with torch.no_grad():
        assert torch.equal(got(t), ref(t))


def test_timestep_frequency_dim_and_period_are_the_reference_defaults():
    got = port_ae.ExpertTimestepEmbedder(DIM)
    ref = ref_expert.TimestepEmbedder(DIM)
    assert got.frequency_dim == ref.frequency_dim == 256
    assert got.max_period == ref.max_period == 10_000


def test_control_cos_sin_order_matters():
    """Control: ``[sin, cos]`` is a column permutation of the first linear.

    It runs, it has the right shape, and it is a different function. Without
    this control the order in the port would be untested.
    """
    torch.manual_seed(0)
    emb = port_ae.ExpertTimestepEmbedder(DIM)
    for p in emb.parameters():
        torch.nn.init.normal_(p, std=0.2)
    t = torch.rand(1, N_NOISED)
    half = emb.frequency_dim // 2
    with torch.no_grad():
        # The same frequencies, the halves concatenated the other way round,
        # pushed through the same MLP.
        freqs = torch.exp(
            -math.log(emb.max_period) * torch.arange(0, half, dtype=torch.float32) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        swapped = emb.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=-1))
        assert not torch.equal(emb(t), swapped)


def test_control_the_timestep_is_not_scaled_by_1000():
    """Control: a 1000x scaling of ``t`` would be a different embedding.

    Several flow-matching references scale the timestep before the
    sinusoids. This one does not, and the two are distinguishable.
    """
    torch.manual_seed(0)
    emb = port_ae.ExpertTimestepEmbedder(DIM)
    for p in emb.parameters():
        torch.nn.init.normal_(p, std=0.2)
    t = torch.rand(1, N_NOISED)
    with torch.no_grad():
        assert not torch.equal(emb(t), emb(t * 1000))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adaln_matches(dtype):
    """``(1 + proj(silu(cond))) * expert_rms_norm(x)``, float32 in the middle."""
    torch.manual_seed(0)
    ref = ref_expert.AdaLN(DIM)
    got = port_ae.ExpertAdaLN(DIM)
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.2)
    got.load_state_dict(ref.state_dict())
    ref, got = ref.to(dtype).eval(), got.to(dtype).eval()
    x = torch.randn(1, Q_LEN, DIM, dtype=dtype)
    cond = torch.randn(1, Q_LEN, DIM, dtype=dtype)
    with torch.no_grad():
        assert torch.equal(got(x, cond), ref(x, cond))


def test_control_the_one_plus_on_the_adaln_projection_matters():
    """Control: dropping the ``1 +`` leaves a layer that still runs.

    The ``1`` is added to the projection's output, so without it the scale
    is ``proj(silu(cond))`` alone. That is a different number for any
    non-zero projection, and at the reference's zero init it would be a
    constant zero.
    """
    torch.manual_seed(0)
    ada = port_ae.ExpertAdaLN(DIM)
    for p in ada.parameters():
        torch.nn.init.normal_(p, std=0.2)
    x = torch.randn(1, Q_LEN, DIM)
    cond = torch.randn(1, Q_LEN, DIM)
    with torch.no_grad():
        scale = ada.proj(torch.nn.functional.silu(cond))
        without = (scale.float() * port_nt.expert_rms_norm(x)).to(x.dtype)
        assert not torch.equal(ada(x, cond), without)


def test_control_the_adaln_norm_is_the_expert_norm_not_the_trunks():
    """Control: the two norm families must not coincide.

    The expert's has no weight and eps ``1e-4``; the trunk's scales by
    ``1 + weight`` at eps ``1e-6``. Building the AdaLN on the trunk's norm
    is the port error this rules out.
    """
    torch.manual_seed(0)
    trunk = port_nt.Float32RMSNorm(DIM)
    with torch.no_grad():
        trunk.weight.copy_(torch.randn(DIM) * 0.1)
    x = torch.randn(1, Q_LEN, DIM)
    assert not torch.equal(trunk(x), port_nt.expert_rms_norm(x))
    # and specifically the epsilon differs, with the weight taken out
    with torch.no_grad():
        trunk.weight.zero_()
    assert not torch.equal(trunk(x * 1e-3), port_nt.expert_rms_norm(x * 1e-3))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_action_tau_encoder_matches(dtype):
    """The timestep fused into the action tokens, through its own embedder."""
    torch.manual_seed(0)
    ref = ref_expert.ActionTauEncoder(ACTION_PATCH_DIM, DIM)
    got = port_ae.ActionTauEncoder(ACTION_PATCH_DIM, DIM)
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.2)
    got.load_state_dict(ref.state_dict())
    ref, got = ref.to(dtype).eval(), got.to(dtype).eval()
    actions = torch.randn(1, N_NOISED, ACTION_PATCH_DIM, dtype=dtype)
    tau = torch.rand(1, N_NOISED, dtype=dtype)
    with torch.no_grad():
        assert torch.equal(got(actions, tau), ref(actions, tau))


def test_action_tau_encoder_rejects_a_mismatched_tau():
    """A tau that does not cover the actions must raise, not broadcast."""
    enc = port_ae.ActionTauEncoder(ACTION_PATCH_DIM, DIM)
    actions = torch.randn(1, N_NOISED, ACTION_PATCH_DIM)
    with pytest.raises(ValueError, match="does not cover"):
        enc(actions, torch.rand(1, N_NOISED + 1))


def test_control_the_two_timestep_embedders_are_separate_weights():
    """Control: the fused tau and the AdaLN condition are not one module.

    Same ``t``, two MLPs. If the port shared one embedder the test above
    would still pass, so the separation is asserted on the parameter names.
    """
    expert = _reference_expert()
    assert expert.action_encoder.tau is not expert.timestep_embedder
    fused = expert.action_encoder.tau.state_dict()
    cond = expert.timestep_embedder.state_dict()
    assert set(fused) == set(cond)
    assert not all(torch.equal(fused[k], cond[k]) for k in fused)


# ---------------------------------------------------------------------------
# The cross-attention KV source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kv_source_matches(dtype):
    """Tap norm, tap projection, and the globals concatenated in front."""
    torch.manual_seed(0)
    expert = _reference_expert()
    got = port_ae.ExpertKVSource(TRUNK_DIM, DIM)
    with torch.no_grad():
        got.tap_proj.weight.copy_(expert.tap_proj.weight)
        got.text_proj.weight.copy_(expert.text_proj.weight)
        got.tap_norm_scale.copy_(expert.tap_norm_scale)
    expert, got = expert.to(dtype), got.to(dtype).eval()
    tap = torch.randn(1, (N_CLEAN + N_NOISED) * TPF, TRUNK_DIM, dtype=dtype)
    globals_ = torch.randn(1, N_GLOBAL, TRUNK_DIM, dtype=dtype)
    with torch.no_grad():
        # The reference's own expression, from forward_trunk_rope.
        want_feat = expert.tap_proj(ref_expert._tap_norm(tap, expert.tap_norm_scale))
        want = torch.cat([expert.text_proj(globals_).to(want_feat.dtype), want_feat], dim=1)
        assert torch.equal(got(tap, got.global_kv(globals_)), want)
        assert torch.equal(got(tap), want_feat)


def test_control_tap_norm_scale_is_a_plain_weight_not_one_plus_w():
    """Control: the two readings of the same tensor must differ.

    The trunk's norms in this checkpoint store ``weight - 1``. The expert's
    tap scale does not, and reading it the trunk's way would shift every tap
    by one times the normed feature.
    """
    torch.manual_seed(0)
    tap = torch.randn(1, 12, TRUNK_DIM)
    scale = torch.randn(TRUNK_DIM) * 0.2
    plain = port_nt.expert_rms_norm(tap) * scale
    gemma = port_nt.expert_rms_norm(tap) * (1 + scale)
    assert not torch.equal(plain, gemma)


def test_control_the_globals_skip_the_tap_norm():
    """Control: the two KV halves do not go through the same path.

    ``text_proj`` has no norm in front of it. Pushing the globals through
    the tap's norm would be a plausible symmetry and a different KV.
    """
    torch.manual_seed(0)
    kv = port_ae.ExpertKVSource(TRUNK_DIM, DIM)
    with torch.no_grad():
        for p in kv.parameters():
            torch.nn.init.normal_(p, std=0.2)
        globals_ = torch.randn(1, N_GLOBAL, TRUNK_DIM)
        normed = kv.text_proj((port_nt.expert_rms_norm(globals_) * kv.tap_norm_scale))
        assert not torch.equal(kv.global_kv(globals_), normed)


# ---------------------------------------------------------------------------
# One full expert layer
# ---------------------------------------------------------------------------


def _blocks(is_cross: bool, dtype: torch.dtype):
    torch.manual_seed(0)
    ref = ref_expert.ActionExpertBlock(
        dim=DIM, num_heads=NUM_HEADS, head_dim=HEAD_DIM, is_cross=is_cross, kv_dim=DIM
    )
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.2)
    got = port_ae.ActionExpertBlock(
        dim=DIM, num_heads=NUM_HEADS, head_dim=HEAD_DIM, is_cross=is_cross, kv_dim=DIM
    )
    got.load_state_dict(ref.state_dict())
    return ref.to(dtype).eval().requires_grad_(False), got.to(dtype).eval().requires_grad_(False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cross_attention_layer_matches(dtype):
    """A full cross layer: AdaLN, rope per side, q/k norm, SDPA, GEGLU MLP."""
    ref, got = _blocks(is_cross=True, dtype=dtype)
    q_table, k_table = _tables()
    cross, _ = _masks()
    torch.manual_seed(1)
    x = torch.randn(1, Q_LEN, DIM, dtype=dtype)
    cond = torch.randn(1, Q_LEN, DIM, dtype=dtype)
    kv = torch.randn(1, KV_LEN, DIM, dtype=dtype)
    with torch.no_grad():
        want = ref.forward_trunk_rope(
            x, cond, kv_source=kv, mask=cross, q_table=q_table, k_table=k_table
        )
        mine = got(x, cond, kv_source=kv, mask=cross, q_table=q_table, k_table=k_table)
    assert mine.shape == want.shape == (1, Q_LEN, DIM)
    assert torch.equal(mine, want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_self_attention_layer_matches(dtype):
    """The odd layers: the same block, keys and values off its own tokens.

    The caller hands the query table twice, which is what the reference's
    expert does for a self layer.
    """
    ref, got = _blocks(is_cross=False, dtype=dtype)
    q_table, _ = _tables()
    _, self_mask = _masks()
    torch.manual_seed(2)
    x = torch.randn(1, Q_LEN, DIM, dtype=dtype)
    cond = torch.randn(1, Q_LEN, DIM, dtype=dtype)
    with torch.no_grad():
        want = ref.forward_trunk_rope(
            x, cond, kv_source=None, mask=self_mask, q_table=q_table, k_table=q_table
        )
        mine = got(x, cond, kv_source=None, mask=self_mask, q_table=q_table, k_table=q_table)
    assert torch.equal(mine, want)


def test_the_layer_itself_is_batch_agnostic():
    """Batch 1 is the entry's restriction, not the layer's.

    ``forward_trunk_rope`` on ``ActionExpert`` rejects ``b != 1`` because
    the window geometry and the masks are built for one row. The block has
    no such limit, and the port must not invent one: the tower will batch
    above it or not at all.
    """
    src = inspect.getsource(ref_expert.ActionExpert.forward_trunk_rope)
    assert "the trunk-rope layout runs one sample" in src
    ref, got = _blocks(is_cross=False, dtype=torch.float32)
    q_table, _ = _tables()
    _, self_mask = _masks()
    torch.manual_seed(3)
    x = torch.randn(2, Q_LEN, DIM)
    cond = torch.randn(2, Q_LEN, DIM)
    table = q_table.expand(2, -1, -1, -1)
    with torch.no_grad():
        want = ref.forward_trunk_rope(
            x, cond, kv_source=None, mask=self_mask, q_table=table, k_table=table
        )
        mine = got(x, cond, kv_source=None, mask=self_mask, q_table=table, k_table=table)
    assert torch.equal(mine, want)


def test_control_the_key_side_needs_its_own_rope_table():
    """Control: reusing the query table for the keys must change the answer.

    A cross layer's keys are trunk tokens; its queries are expert tokens.
    The tables have different lengths, so the wrong one usually raises, but
    with ``Q == KV`` lengths it would silently rotate the keys to the
    queries' positions. This drives that case directly.
    """
    _, got = _blocks(is_cross=True, dtype=torch.float32)
    q_table, k_table = _tables()
    torch.manual_seed(4)
    x = torch.randn(1, Q_LEN, DIM)
    cond = torch.randn(1, Q_LEN, DIM)
    kv = torch.randn(1, Q_LEN, DIM)
    mask = torch.ones(Q_LEN, Q_LEN, dtype=torch.bool)
    with torch.no_grad():
        right = got(
            x, cond, kv_source=kv, mask=mask, q_table=q_table, k_table=k_table[:, :Q_LEN]
        )
        wrong = got(x, cond, kv_source=kv, mask=mask, q_table=q_table, k_table=q_table)
    assert not torch.equal(right, wrong)


def test_control_a_self_layer_reads_its_normed_tokens_not_the_residual():
    """Control: the KV of a self layer comes from the AdaLN output.

    Feeding the residual stream instead is a one-word change that runs. The
    same weights, driven as a cross layer over ``x``, give a different
    answer, which is what makes the self path's choice testable.
    """
    _, self_block = _blocks(is_cross=False, dtype=torch.float32)
    _, cross_block = _blocks(is_cross=True, dtype=torch.float32)
    q_table, _ = _tables()
    _, self_mask = _masks()
    torch.manual_seed(5)
    x = torch.randn(1, Q_LEN, DIM)
    cond = torch.randn(1, Q_LEN, DIM)
    with torch.no_grad():
        normed_kv = self_block(
            x, cond, kv_source=None, mask=self_mask, q_table=q_table, k_table=q_table
        )
        residual_kv = cross_block(
            x, cond, kv_source=x, mask=self_mask, q_table=q_table, k_table=q_table
        )
    assert not torch.equal(normed_kv, residual_kv)


def test_control_rope_and_the_expert_norm_nearly_commute():
    """Control: the rotate/norm order is a rounding difference here.

    Stated so nobody reads the trunk's argument into the expert. The
    expert's q/k norm has no weight and every rotary pair rotates, so the
    rotation preserves each head's sum of squares in exact arithmetic and
    the two orders agree mathematically. They differ in the last bits, which
    is still a bit difference and still part of the contract, and the
    measured size of it is asserted so a regression that reorders them is
    visible as more than noise.
    """
    q_table, _ = _tables()
    torch.manual_seed(6)
    q = torch.randn(1, NUM_HEADS, Q_LEN, HEAD_DIM)
    rope_then_norm = port_nt.expert_rms_norm(port_rope.apply_rope(q, q, q_table)[0])
    norm_then_rope = port_rope.apply_rope(
        port_nt.expert_rms_norm(q), port_nt.expert_rms_norm(q), q_table
    )[0]
    assert not torch.equal(rope_then_norm, norm_then_rope)
    # Rounding, not a different function: well under the 1e-3 a real
    # reordering of a weighted norm would produce.
    assert (rope_then_norm - norm_then_rope).abs().max() < 1e-5


def test_control_the_expert_attention_has_no_gqa():
    """Control: the trunk's 24-over-8 GQA has no counterpart here.

    ``to_kv`` emits ``2 * num_heads * head_dim``, so k and v carry the full
    head count. A port that set ``enable_gqa`` or sized kv heads down would
    have to reshape, and this pins the widths it would have to change.
    """
    _, got = _blocks(is_cross=True, dtype=torch.float32)
    assert got.to_kv.out_features == 2 * NUM_HEADS * HEAD_DIM
    assert got.to_q.out_features == NUM_HEADS * HEAD_DIM == DIM
    # 4x expansion, gated: the projection is 8x the width, the hidden 4x.
    assert got.mlp_in.out_features == 8 * DIM
    assert got.mlp_out.in_features == 4 * DIM


def test_a_cross_block_without_a_kv_source_raises():
    _, got = _blocks(is_cross=True, dtype=torch.float32)
    q_table, _ = _tables()
    _, self_mask = _masks()
    x = torch.randn(1, Q_LEN, DIM)
    with pytest.raises(ValueError, match="needs a kv_source"):
        got(x, x, kv_source=None, mask=self_mask, q_table=q_table, k_table=q_table)


def test_a_self_block_given_a_kv_source_raises():
    """The reference would silently ignore it; here it is an error.

    A self layer that is handed a KV sequence is a wiring mistake, and the
    reference's ``kv_in = kv_source if self.is_cross else h`` drops it
    without a word.
    """
    _, got = _blocks(is_cross=False, dtype=torch.float32)
    q_table, _ = _tables()
    _, self_mask = _masks()
    x = torch.randn(1, Q_LEN, DIM)
    with pytest.raises(ValueError, match="drop the kv_source"):
        got(x, x, kv_source=x, mask=self_mask, q_table=q_table, k_table=q_table)


# ---------------------------------------------------------------------------
# The output head
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("plus_zero", [True, False])
def test_action_head_matches(dtype, plus_zero):
    """LayerNorm over the trailing action tokens, then the linear head."""
    expert = _reference_expert()
    got = port_ae.ActionHead(DIM, ACTION_PATCH_DIM, head_plus_zero=plus_zero)
    with torch.no_grad():
        got.head.weight.copy_(expert.head.weight)
        got.head.bias.copy_(expert.head.bias)
    expert, got = expert.to(dtype), got.to(dtype).eval()
    torch.manual_seed(7)
    x = torch.randn(1, Q_LEN, DIM, dtype=dtype)
    with torch.no_grad():
        want = expert.head(expert.norm_out(x[:, -N_NOISED:]))
        if plus_zero:
            want = want + 0.0
        mine = got(x, N_NOISED)
    assert mine.shape == (1, N_NOISED, ACTION_PATCH_DIM)
    assert torch.equal(mine, want)


def test_the_head_reads_only_the_action_tokens():
    """The proprio and future tokens condition the stack and are dropped."""
    head = port_ae.ActionHead(DIM, ACTION_PATCH_DIM)
    torch.manual_seed(8)
    x = torch.randn(1, Q_LEN, DIM)
    other = x.clone()
    other[:, : Q_LEN - N_NOISED] = torch.randn(1, Q_LEN - N_NOISED, DIM)
    with torch.no_grad():
        assert torch.equal(head(x, N_NOISED), head(other, N_NOISED))


def test_control_layer_norm_is_not_an_rms_norm():
    """Control: the output norm subtracts the mean, and that is visible.

    Both are weightless, both take ``(B, S, D)`` to the same shape, and
    swapping one for the other runs. On an input with a non-zero mean they
    disagree, which is the whole point of pinning which one it is.
    """
    torch.manual_seed(9)
    layer_norm = torch.nn.LayerNorm(DIM, elementwise_affine=False, eps=1e-6)
    x = torch.randn(1, Q_LEN, DIM) + 3.0  # a mean the centering has to remove
    with torch.no_grad():
        assert not torch.equal(layer_norm(x), port_nt.expert_rms_norm(x))
    # and the difference is not a rounding one
    with torch.no_grad():
        gap = (layer_norm(x) - port_nt.expert_rms_norm(x)).abs().max()
    assert gap > 0.1


def test_control_head_plus_zero_only_shows_in_the_sign_bit():
    """Control: ``+ 0.0`` is observable, but never through ``torch.equal``.

    The flag's whole effect is turning ``-0.0`` into ``+0.0``, and
    ``-0.0 == 0.0``, so the comparison every other test in this file uses is
    blind to it. ``test_action_head_matches`` still pins both settings
    against the reference's own expression; what cannot be shown there is
    that the two settings ever differ, and this is why.

    ``ActionHead`` itself cannot be driven to a negative zero: the linear's
    bias add turns ``-0.0 + 0.0`` back into ``+0.0``, so the mechanism is
    asserted on the values instead of through the module.
    """
    minus_zero = torch.full((N_NOISED, ACTION_PATCH_DIM), -0.0)
    assert minus_zero.signbit().all()
    assert not (minus_zero + 0.0).signbit().any()
    # The difference torch.equal cannot see, stated rather than assumed.
    assert torch.equal(minus_zero, minus_zero + 0.0)
