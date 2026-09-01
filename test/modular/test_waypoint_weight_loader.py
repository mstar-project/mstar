"""``mstar.model.waypoint.weight_loader`` against synthetic checkpoints.

The Waypoint-1.5-1B checkpoint is not on this machine and must not be
downloaded, so the bar here is not numerical parity — it is **every claim the
loader makes about keys, shapes, slices and counts, checked against a state dict
built from the reference's own key spellings** (``world_engine/src/model/
world_model.py::load_state_dict``, transcribed in ``docs/waypoint/PARAM_TREE.md``).
Everything runs on CPU with no checkpoint and no GPU.

Why that bar and not a looser one: almost every way this loader can be wrong is
*shape-legal*. ``PARAM_TREE.md`` section 8 lists sixteen failure modes and
labels nine of them silent — a q/k/v fusion built as ``cat([q, v, k])``, an
``fc1_x``/``fc1_c`` merge in the wrong column order, ``attn``/``mlp`` cond_proj
slots swapped, an ``unpatchify`` permute dropped. Each of those loads without an
exception and produces plausible video. So the synthetic tensors are
**distinguishable per key** (``_tensor`` seeds a generator off the key name):
every assertion here compares values, not shapes, and a swap is a failed
assertion rather than a clean load.

The four regression tests marked ``F1``-``F4`` pin defects that an audit found in
the shipped loader and that all four reproduced before the fix:

* **F1** — a pre-fused ``qkv_proj`` and the ``q/k/v_proj`` shards both loaded,
  silently, producing a tensor assembled from both sources in lexicographic key
  order (Q and K off the blob, V off ``v_proj``). Same hole for ``fc1``.
* **F2** — the ``cond_proj`` tie check probed ``[:, :64]``, so a divergence past
  column 64 loaded clean with ``verify_cond_proj_tie=True``.
* **F3** — ``attn_cond_head.bias_in`` was dropped unconditionally, so a file
  carrying only that spelling (``PARAM_TREE.md`` section 10.2 leaves which one
  the real file has unresolved) failed with 24 unloaded ``cond_head.bias_in``.
  The reference falls back to it (``world_model.py:386-389``).
* **F4** — shape validation ran before the drop filter, so a ``.cond_heads.`` key
  ending in ``.k_proj.weight`` raised a GQA error about a key T12 discards.

Nothing here needs the 2.6 GB of a real bf16 build: the parameter census is read
off the **meta** module (``numel()`` needs no storage) and every load test uses a
128-wide, 4-layer config. The one test that needs all 24 layers — the
``cond_proj`` tie lifecycle — keeps ``n_layers=24`` and shrinks ``d_model``, so
the "144 after ``to_empty``" number is the real one at a few MB.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import pytest
import torch

sys.path.insert(0, ".")

from mstar.model.loader.base import _apply_stacked, load_weights_into
from mstar.model.waypoint.components.dit import WaypointDiT
from mstar.model.waypoint.config import WaypointConfig
from mstar.model.waypoint.weight_loader import (
    COND_PROJ_SOURCE_BLOCK,
    WAYPOINT_STACKED_PARAMS,
    _adapt_checkpoint_stream,
    _attach_shard_loaders,
    build_waypoint_dit,
    parameter_census,
    remap_checkpoint_key,
)

pytest.importorskip("safetensors", reason="safetensors not installed")

from safetensors.torch import save_file  # noqa: E402

# ``NoiseConditioner``'s Fourier width is a constructor default, not a config
# field, and it is the in-dim of denoise_step_emb.mlp.fc1 (PARAM_TREE row 1).
FOURIER_DIM = 512

LEGACY = "legacy"
CANONICAL = "canonical"

# The three T4 spellings, in the reference's precedence order (last wins).
BIAS_IN_KEYS = {
    "attn": "attn_cond_head.bias_in",
    "mlp": "mlp_cond_head.bias_in",
    "canonical": "cond_head.bias_in",
}


def tiny_config() -> WaypointConfig:
    """A structurally faithful 4-layer Waypoint.

    Everything the loader reasons about is preserved: GQA with unequal q/kv rows
    (128 vs 64, so a q/k swap raises and a k/v swap does not — S11/S11b), a 2x2
    patch, ctrl layers at ``i % 3 == 0`` = {0, 3}, and ``d_model=128`` so the
    retired ``[:, :64]`` cond_proj probe covers only half a matrix and F2's
    regression test has somewhere to hide.
    """
    return WaypointConfig(
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_model=128,
        mlp_ratio=2,
        channels=4,
        tokens_per_frame=32,
        height=4,
        width=8,
        patch=(2, 2),
        n_buttons=8,
    )


def tie_config() -> WaypointConfig:
    """All 24 layers, 128 wide. The tie lifecycle counts *tensors*, not bytes."""
    return WaypointConfig(
        n_layers=24,
        n_heads=4,
        n_kv_heads=2,
        d_model=128,
        mlp_ratio=2,
        channels=4,
        tokens_per_frame=32,
        height=4,
        width=8,
        patch=(2, 2),
        n_buttons=8,
    )


def _tensor(key: str, shape: tuple[int, ...]) -> torch.Tensor:
    """A tensor whose contents are a function of its checkpoint key.

    This is the whole reason these tests can see a swap. Two same-shaped
    checkpoint tensors are never equal, so ``cat([q, v, k])`` instead of
    ``cat([q, k, v])``, or ``cond_proj`` slot 3 loaded into slot 0, fails an
    assertion instead of loading cleanly.
    """
    generator = torch.Generator().manual_seed(zlib.crc32(key.encode()) | 1)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def synthetic_checkpoint(
    config: WaypointConfig,
    *,
    spelling: str = LEGACY,
    bias_in: tuple[str, ...] = ("mlp",),
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """``(checkpoint state dict, expected named_parameters)``.

    ``spelling=LEGACY`` writes the keys the reference's transforms exist to
    rewrite; ``CANONICAL`` writes the already-post-transform names that the
    reference's ``pop``/``setdefault`` pairs also accept (PARAM_TREE section
    3.5) — including a pre-fused ``qkv_proj`` and ``ctrl_mlpfusion.mlp.fc1``,
    which exercise ``_SliceShardLoader``'s ``loaded_shard_id is None`` path.
    Which one the real file uses is section 10.1's open question, so both load.

    The expected dict is built from the reference's own formulas
    (``world_model.py:372-405`` and ``patch_model.py:110-112``), restated here
    rather than imported, so agreement means two independent transcriptions
    agree.
    """
    D, C = config.d_model, config.channels
    ph, pw = config.patch
    ffn = D * config.mlp_ratio
    q_rows = config.n_heads * config.d_head
    kv_rows = config.n_kv_heads * config.d_head
    legacy = spelling == LEGACY

    state: dict[str, torch.Tensor] = {}
    expected: dict[str, torch.Tensor] = {}

    def src(key: str, shape: tuple[int, ...]) -> torch.Tensor:
        state[key] = _tensor(key, shape)
        return state[key]

    # --- top level, identity ------------------------------------------------
    for leaf, shape in (
        ("denoise_step_emb.mlp.fc1.weight", (D * 4, FOURIER_DIM)),
        ("denoise_step_emb.mlp.fc2.weight", (D, D * 4)),
        ("ctrl_emb.mlp.fc1.weight", (ffn, config.d_ctrl_in)),
        ("ctrl_emb.mlp.fc2.weight", (D, ffn)),
        ("patchify.weight", (D, C, ph, pw)),
        ("out_norm.fc.weight", (2 * D, D)),
    ):
        expected[leaf] = src(leaf, shape)

    # T10: in the file, never in the port's tree.
    src("ctrl_cfg.null_emb", (1, 1, D))

    if legacy:
        # T1: [D, C, ph, pw] conv kernel -> [C*ph*pw, D] Linear weight.
        weight = src("unpatchify.weight", (D, C, ph, pw))
        expected["unpatchify.weight"] = weight.permute(1, 2, 3, 0).reshape(-1, D)
        # T2: one bias per latent channel, repeated across the patch.
        bias = src("unpatchify.bias", (C,))
        expected["unpatchify.bias"] = bias[:, None, None].expand(-1, ph, pw).reshape(-1)
    else:
        expected["unpatchify.weight"] = src("unpatchify.weight", (C * ph * pw, D))
        expected["unpatchify.bias"] = src("unpatchify.bias", (C * ph * pw,))

    # --- per block ----------------------------------------------------------
    prefix = "transformer.blocks." if legacy else "blocks."
    for i in range(config.n_layers):
        p, q = f"{prefix}{i}.", f"blocks.{i}."

        if legacy:
            # T11 (port-side): cat([q, k, v], dim=0), q first, along the rows.
            shards = [
                src(p + "attn.q_proj.weight", (q_rows, D)),
                src(p + "attn.k_proj.weight", (kv_rows, D)),
                src(p + "attn.v_proj.weight", (kv_rows, D)),
            ]
            expected[q + "attn.qkv_proj.weight"] = torch.cat(shards, dim=0)
        else:
            expected[q + "attn.qkv_proj.weight"] = src(
                p + "attn.qkv_proj.weight", (q_rows + 2 * kv_rows, D)
            )

        expected[q + "attn.out_proj.weight"] = src(p + "attn.out_proj.weight", (D, D))
        expected[q + "attn.v_lamb"] = src(p + "attn.v_lamb", ())  # rank 0, not [1] (S13)

        # T3: an explicit five-name allowlist in the reference; only fc1/fc2 exist.
        mlp_prefix = "dit_mlp." if legacy else "mlp."
        for leaf, shape in (("fc1.weight", (ffn, D)), ("fc2.weight", (D, ffn))):
            expected[q + "mlp." + leaf] = src(p + mlp_prefix + leaf, shape)

        # T4: up to three spellings, one target.
        for which in bias_in:
            src(p + BIAS_IN_KEYS[which], (D,))
        if bias_in:
            winner = max(bias_in, key=lambda w: list(BIAS_IN_KEYS).index(w))
            expected[q + "cond_head.bias_in"] = state[p + BIAS_IN_KEYS[winner]]

        # T5/T6: attn head -> slots 0..2 (attention branch), mlp head -> 3..5
        # (MLP branch). T9 keeps only COND_PROJ_SOURCE_BLOCK's set; the other
        # blocks' keys are in the file (the file is not deduplicated) and carry
        # the same values, which is what _CondProjTieCheck verifies.
        for j in range(3):
            if legacy:
                attn = src(p + f"attn_cond_head.cond_proj.{j}.weight", (D, D))
                mlp = src(p + f"mlp_cond_head.cond_proj.{j}.weight", (D, D))
            else:
                attn = src(p + f"cond_head.cond_proj.{j}.weight", (D, D))
                mlp = src(p + f"cond_head.cond_proj.{j + 3}.weight", (D, D))
            if i == COND_PROJ_SOURCE_BLOCK:
                expected[f"blocks.{i}.cond_head.cond_proj.{j}.weight"] = attn
                expected[f"blocks.{i}.cond_head.cond_proj.{j + 3}.weight"] = mlp

        if i not in config.ctrl_layers:
            continue
        if legacy:
            # T7: cat([fc1_x, fc1_c], dim=1) -- x first, along the INPUT axis.
            x = src(p + "ctrl_mlpfusion.fc1_x.weight", (D, D))
            c = src(p + "ctrl_mlpfusion.fc1_c.weight", (D, D))
            expected[q + "ctrl_mlpfusion.mlp.fc1.weight"] = torch.cat((x, c), dim=1)
            # T8: a plain rename, guarded separately from T7's both-halves guard.
            expected[q + "ctrl_mlpfusion.mlp.fc2.weight"] = src(
                p + "ctrl_mlpfusion.fc2.weight", (D, D)
            )
        else:
            expected[q + "ctrl_mlpfusion.mlp.fc1.weight"] = src(
                p + "ctrl_mlpfusion.mlp.fc1.weight", (D, 2 * D)
            )
            expected[q + "ctrl_mlpfusion.mlp.fc2.weight"] = src(
                p + "ctrl_mlpfusion.mlp.fc2.weight", (D, D)
            )

    # The tie: every block stores the same six matrices (PARAM_TREE section 5.2
    # proves the file is not deduplicated). Applied last so it overwrites the
    # per-block values generated above.
    for i in range(config.n_layers):
        if i == COND_PROJ_SOURCE_BLOCK:
            continue
        for j in range(3):
            if legacy:
                for head, slot in (("attn", j), ("mlp", j)):
                    key = f"{prefix}{i}.{head}_cond_head.cond_proj.{slot}.weight"
                    ref = f"{prefix}{COND_PROJ_SOURCE_BLOCK}.{head}_cond_head.cond_proj.{slot}.weight"
                    state[key] = state[ref]
            else:
                for slot in (j, j + 3):
                    key = f"{prefix}{i}.cond_head.cond_proj.{slot}.weight"
                    state[key] = state[f"{prefix}{COND_PROJ_SOURCE_BLOCK}.cond_head.cond_proj.{slot}.weight"]

    return state, expected


def write_checkpoint(directory: Path, state: dict[str, torch.Tensor]) -> Path:
    """Materialize a state dict as ``model.safetensors``. ``clone()`` because
    safetensors refuses to write tensors that share storage, and the tied
    ``cond_proj`` entries do."""
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().clone() for k, v in state.items()}, str(directory / "model.safetensors"))
    return directory


def build_from(tmp_path: Path, state: dict[str, torch.Tensor], config: WaypointConfig, **kwargs):
    name = f"ckpt{len(list(tmp_path.iterdir()))}"
    return build_waypoint_dit(config, write_checkpoint(tmp_path / name, state), **kwargs)


# ---------------------------------------------------------------------------
# 1. The parameter census on the real 720P config
# ---------------------------------------------------------------------------


def test_parameter_census_matches_the_720p_checkpoint():
    """174 tensors / 1,281,958,040 resident / 1,860,771,992 stored.

    These are PARAM_TREE section 5.2's numbers, less the 2,048 of
    ``ctrl_cfg.null_emb`` that T10 drops, and they are the arithmetic behind
    "1.28B resident, 1.86B stored, 3.72 GB on disk". A change in any of the
    three means the module tree stopped being the checkpoint's tree.

    Read off the meta module: ``numel()`` and ``state_dict()`` need no storage,
    so this costs nothing even though the same build with real memory is 2.6 GB.
    """
    with torch.device("meta"):
        dit = WaypointDiT(WaypointConfig())

    tensors, dedup_numel, raw_numel = parameter_census(dit)
    assert (tensors, dedup_numel, raw_numel) == (174, 1_281_958_040, 1_860_771_992)

    # The 312 - 174 = 138 gap IS the tie: 23 blocks x 6 aliased [2048, 2048].
    assert len(dit.state_dict()) == 312
    assert raw_numel - dedup_numel == 23 * 6 * 2048**2 == 578_813_952


# ---------------------------------------------------------------------------
# 2. The cond_proj tie lifecycle (CONTRACTS section 6.1)
# ---------------------------------------------------------------------------


def test_cond_proj_tie_lifecycle_across_to_empty():
    """6 -> 6 -> **144** -> 6. The 144 is the trap.

    ``Module._apply`` has no cross-module memo, so ``to_empty(device)`` silently
    un-aliases the six shared ``cond_proj`` matrices into 24 independent sets.
    ``.to(dtype)`` on meta does not. Nothing raises either way; the un-tied model
    is numerically correct and merely 0.6B parameters heavier, and the loader
    then leaves 23 blocks' ``cond_proj`` unfilled. This is the whole reason
    ``retie_cond_proj()`` is public and must follow ``to_empty``.
    """
    config = tie_config()

    def n_cond_proj(module) -> int:
        return sum(1 for name, _ in module.named_parameters() if ".cond_head.cond_proj." in name)

    with torch.device("meta"):
        dit = WaypointDiT(config)
    assert n_cond_proj(dit) == 6, "__init__ ties"

    dit.cast_serving_dtypes()
    assert n_cond_proj(dit) == 6, ".to(dtype) on meta preserves aliasing"

    dit.to_empty(device="cpu")
    assert n_cond_proj(dit) == config.n_layers * 6 == 144, "to_empty un-ties, silently"

    dit.retie_cond_proj()
    assert n_cond_proj(dit) == 6
    # The surviving names are the owner block's; T9 drops keys for every other
    # block on exactly that assumption.
    assert all(
        name.startswith(f"blocks.{COND_PROJ_SOURCE_BLOCK}.cond_head.cond_proj.")
        for name, _ in dit.named_parameters()
        if ".cond_head.cond_proj." in name
    )


def test_build_waypoint_dit_leaves_cond_proj_tied(tmp_path):
    """The same property, through the real entry point rather than by hand."""
    config = tiny_config()
    state, _ = synthetic_checkpoint(config)
    dit = build_from(tmp_path, state, config)

    tensors, dedup_numel, raw_numel = parameter_census(dit)
    assert tensors == len(dict(dit.named_parameters()))
    assert raw_numel - dedup_numel == (config.n_layers - 1) * 6 * config.d_model**2
    # Aliased, not merely equal: writing block 0's tensor must move block 3's.
    owner = dit.blocks[COND_PROJ_SOURCE_BLOCK].cond_head.cond_proj[2].weight
    assert dit.blocks[3].cond_head.cond_proj[2].weight is owner


# ---------------------------------------------------------------------------
# 3. T0-T12 round trip, both key spellings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", [LEGACY, CANONICAL])
def test_every_parameter_equals_its_checkpoint_source(tmp_path, spelling):
    """Every one of the loaded parameters, compared by value against an
    independent transcription of the reference's transforms.

    This is the round trip for T0-T12 at once: a name that does not appear in
    ``expected`` is a transform this test does not know about, and a value that
    differs is a transform applied wrongly. Both spellings run because
    PARAM_TREE section 10.1 could not establish which one the shipped file uses.
    """
    config = tiny_config()
    state, expected = synthetic_checkpoint(config, spelling=spelling)
    dit = build_from(tmp_path, state, config)

    loaded = dict(dit.named_parameters())
    assert set(loaded) == set(expected), (
        f"unmapped parameters {sorted(set(loaded) - set(expected))[:5]}, "
        f"expectations that reached no parameter {sorted(set(expected) - set(loaded))[:5]}"
    )

    wrong = [
        name
        for name, param in loaded.items()
        if not torch.equal(param.detach().cpu(), expected[name].to(param.dtype))
    ]
    assert not wrong, f"{len(wrong)} parameters differ from their checkpoint source: {wrong[:5]}"


def test_qkv_row_ranges(tmp_path):
    """q = rows [0:2048], k = [2048:3072], v = [3072:4096] (PARAM_TREE section 7).

    Explicit because this is where a swap is invisible: GQA makes q taller than
    k and v, so a q/k swap raises on the slice shape (S11b) but a **k/v swap
    does not** — both are ``[n_kv_heads*d_head, d_model]``, the load is clean,
    and attention output is meaningless but well-scaled (S11).
    """
    config = tiny_config()
    state, _ = synthetic_checkpoint(config, spelling=LEGACY)
    dit = build_from(tmp_path, state, config)

    q_rows = config.n_heads * config.d_head
    kv_rows = config.n_kv_heads * config.d_head
    assert q_rows != kv_rows, "the config must keep GQA, or a q/k swap becomes invisible too"

    for i in range(config.n_layers):
        fused = dit.blocks[i].attn.qkv_proj.weight.detach().cpu()
        p = f"transformer.blocks.{i}.attn."
        for shard, start, size in (
            ("q_proj", 0, q_rows),
            ("k_proj", q_rows, kv_rows),
            ("v_proj", q_rows + kv_rows, kv_rows),
        ):
            source = state[p + shard + ".weight"].to(fused.dtype)
            assert torch.equal(fused[start : start + size], source), (
                f"block {i}: rows [{start}:{start + size}] of qkv_proj are not {shard}"
            )


def test_fc1_column_halves(tmp_path):
    """``fc1_x`` = columns [0:D], ``fc1_c`` = [D:2D] (PARAM_TREE section 4.7).

    Explicit for the same reason as the qkv ranges, and worse: the merge is on
    **dim 1**, both halves are ``[D, D]``, so ``cat((c, x))`` keeps the shape
    exactly and applies controller conditioning to tokens and token content to
    the controller vector (S7). ``MLPFusion.forward``'s ``chunk(2, dim=1)`` takes
    the low columns as the token half, which is what fixes the order.
    """
    config = tiny_config()
    state, _ = synthetic_checkpoint(config, spelling=LEGACY)
    dit = build_from(tmp_path, state, config)
    D = config.d_model

    assert config.ctrl_layers, "the config must have ctrl layers for this to test anything"
    for i in sorted(config.ctrl_layers):
        fused = dit.blocks[i].ctrl_mlpfusion.mlp.fc1.weight.detach().cpu()
        assert fused.shape == (D, 2 * D)
        p = f"transformer.blocks.{i}.ctrl_mlpfusion."
        x_half, c_half = fused.chunk(2, dim=1)
        assert torch.equal(x_half, state[p + "fc1_x.weight"].to(fused.dtype)), f"block {i}: low columns are not fc1_x"
        assert torch.equal(c_half, state[p + "fc1_c.weight"].to(fused.dtype)), f"block {i}: high columns are not fc1_c"
    # The 16 non-ctrl layers own no such tensor at all -- the parameter tree
    # itself records which layers fuse.
    assert all(dit.blocks[i].ctrl_mlpfusion is None for i in range(config.n_layers) if i not in config.ctrl_layers)


def test_cond_proj_slots_follow_the_half_head_names(tmp_path):
    """attn head -> slots 0-2, mlp head -> 3-5, and not the reverse.

    ``CondHead.forward`` is unpacked as ``s0, b0, g0, s1, b1, g1``; 0-2 drive the
    attention sublayer and 3-5 the MLP. All six are ``[D, D]``, so swapping T5
    and T6 is mechanically invisible and numerically catastrophic (S5).
    """
    config = tiny_config()
    state, _ = synthetic_checkpoint(config, spelling=LEGACY)
    dit = build_from(tmp_path, state, config)

    p = f"transformer.blocks.{COND_PROJ_SOURCE_BLOCK}."
    for j in range(3):
        for head, slot in (("attn", j), ("mlp", j + 3)):
            got = dit.blocks[COND_PROJ_SOURCE_BLOCK].cond_head.cond_proj[slot].weight.detach().cpu()
            source = state[p + f"{head}_cond_head.cond_proj.{j}.weight"].to(got.dtype)
            assert torch.equal(got, source), f"slot {slot} did not come from {head}_cond_head.cond_proj.{j}"


@pytest.mark.parametrize(
    "key,expected",
    [
        # T0: both the checkpoint's two-level prefix and the collapsed one.
        ("transformer.blocks.5.attn.out_proj.weight", "blocks.5.attn.out_proj.weight"),
        ("blocks.5.attn.out_proj.weight", "blocks.5.attn.out_proj.weight"),
        # T3, five-name allowlist (only fc1/fc2 exist under moe=False).
        ("transformer.blocks.5.dit_mlp.fc1.weight", "blocks.5.mlp.fc1.weight"),
        ("transformer.blocks.5.dit_mlp.fc2.weight", "blocks.5.mlp.fc2.weight"),
        # T4: all three spellings share one target; precedence is settled by the
        # loader, not here (this function is deliberately not injective).
        ("transformer.blocks.0.attn_cond_head.bias_in", "blocks.0.cond_head.bias_in"),
        ("transformer.blocks.0.mlp_cond_head.bias_in", "blocks.0.cond_head.bias_in"),
        ("transformer.blocks.0.cond_head.bias_in", "blocks.0.cond_head.bias_in"),
        # T5 / T6: identity for attn, +3 for mlp.
        ("transformer.blocks.0.attn_cond_head.cond_proj.2.weight", "blocks.0.cond_head.cond_proj.2.weight"),
        ("transformer.blocks.0.mlp_cond_head.cond_proj.0.weight", "blocks.0.cond_head.cond_proj.3.weight"),
        ("transformer.blocks.0.mlp_cond_head.cond_proj.2.weight", "blocks.0.cond_head.cond_proj.5.weight"),
        # T8: guarded on fc2 alone, separately from T7.
        ("transformer.blocks.3.ctrl_mlpfusion.fc2.weight", "blocks.3.ctrl_mlpfusion.mlp.fc2.weight"),
        # T9: every block but the owner is dropped, both spellings.
        ("transformer.blocks.7.attn_cond_head.cond_proj.2.weight", None),
        ("transformer.blocks.7.cond_head.cond_proj.5.weight", None),
        # T10 / T12.
        ("ctrl_cfg.null_emb", None),
        ("transformer.blocks.0.cond_heads.0.weight", None),
        ("transformer.blocks.0.cond_heads.0.k_proj.weight", None),
        # T7 / T11 are fan-ins: the remapper leaves them for the stacked rules.
        ("transformer.blocks.2.attn.q_proj.weight", "blocks.2.attn.q_proj.weight"),
        ("transformer.blocks.0.ctrl_mlpfusion.fc1_x.weight", "blocks.0.ctrl_mlpfusion.fc1_x.weight"),
        # Top level is identity.
        ("patchify.weight", "patchify.weight"),
        ("out_norm.fc.weight", "out_norm.fc.weight"),
    ],
)
def test_remap_checkpoint_key(key, expected):
    assert remap_checkpoint_key(key) == expected


@pytest.mark.parametrize(
    "mapped,target,shard",
    [
        ("blocks.2.attn.q_proj.weight", "blocks.2.attn.qkv_proj.weight", "q"),
        ("blocks.2.attn.k_proj.weight", "blocks.2.attn.qkv_proj.weight", "k"),
        ("blocks.2.attn.v_proj.weight", "blocks.2.attn.qkv_proj.weight", "v"),
        ("blocks.0.ctrl_mlpfusion.fc1_x.weight", "blocks.0.ctrl_mlpfusion.mlp.fc1.weight", "x"),
        ("blocks.0.ctrl_mlpfusion.fc1_c.weight", "blocks.0.ctrl_mlpfusion.mlp.fc1.weight", "c"),
        # The leading dots keep .v_proj from matching inside qkv_proj, and
        # out_proj / cond_proj from matching anything.
        ("blocks.2.attn.qkv_proj.weight", "blocks.2.attn.qkv_proj.weight", None),
        ("blocks.2.attn.out_proj.weight", "blocks.2.attn.out_proj.weight", None),
        ("blocks.0.cond_head.cond_proj.0.weight", "blocks.0.cond_head.cond_proj.0.weight", None),
    ],
)
def test_stacked_rules_route_the_two_fusions(mapped, target, shard):
    assert _apply_stacked(mapped, WAYPOINT_STACKED_PARAMS) == (target, shard)


# ---------------------------------------------------------------------------
# 4. The per-shard tally, and why set(named_parameters()) - loaded is not enough
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shard_key", ["attn.q_proj", "attn.k_proj", "attn.v_proj"])
def test_missing_qkv_shard_raises(tmp_path, shard_key):
    config = tiny_config()
    state, _ = synthetic_checkpoint(config)
    for i in range(config.n_layers):
        del state[f"transformer.blocks.{i}.{shard_key}.weight"]

    with pytest.raises(RuntimeError, match=r"[1-9]\d* unloaded fused shards"):
        build_from(tmp_path, state, config)


@pytest.mark.parametrize("shard_key", ["ctrl_mlpfusion.fc1_x", "ctrl_mlpfusion.fc1_c"])
def test_missing_fc1_shard_raises(tmp_path, shard_key):
    config = tiny_config()
    state, _ = synthetic_checkpoint(config)
    for i in sorted(config.ctrl_layers):
        del state[f"transformer.blocks.{i}.{shard_key}.weight"]

    with pytest.raises(RuntimeError, match=r"[1-9]\d* unloaded fused shards"):
        build_from(tmp_path, state, config)


def test_named_parameters_check_alone_cannot_see_a_missing_shard(tmp_path):
    """Why the ``(target, shard_id)`` tally exists at all — S11d.

    ``load_weights_into`` returns *target* names, and q, k and v share one
    target, so a ``k_proj`` missing from every layer leaves
    ``set(named_parameters()) - loaded`` **empty**. That is the wan22-style
    completeness check passing on a transformer with a third of its attention
    projections uninitialized. This test records the hole rather than the fix:
    it asserts the set difference is empty, so if someone ever deletes the tally
    on the grounds that the set difference covers it, this is the counterexample.
    """
    config = tiny_config()
    state, _ = synthetic_checkpoint(config)
    for i in range(config.n_layers):
        del state[f"transformer.blocks.{i}.attn.k_proj.weight"]

    dit = build_waypoint_dit(config, skip_weight_loading=True)
    _attach_shard_loaders(dit, config)
    stream = _adapt_checkpoint_stream(iter(state.items()), config, None)
    loaded = load_weights_into(
        dit, stream, stacked_params=WAYPOINT_STACKED_PARAMS, name_remapper=remap_checkpoint_key
    )

    assert set(dict(dit.named_parameters())) - loaded == set(), (
        "the set difference is supposed to be empty here — that is the point"
    )
    # And the shipped loader, which keeps the tally, refuses the same checkpoint.
    with pytest.raises(RuntimeError, match=r"attn\.qkv_proj\.weight\[k\]"):
        build_from(tmp_path, state, config)


# ---------------------------------------------------------------------------
# 5. Drops, unexpected keys, and the transcribed config facts
# ---------------------------------------------------------------------------


def test_intended_drops_load_cleanly(tmp_path):
    """T10 and T12 are silent by design; everything else is loud.

    ``.cond_heads.`` (note the plural) is the reference's unconditional filter,
    and ``ctrl_cfg.null_emb`` is a training-time CFG tensor with no call site.
    Both must be dropped *explicitly* — leaving them unmatched would work by
    accident and put a hole in the unexpected-key accounting (S10).
    """
    config = tiny_config()
    state, expected = synthetic_checkpoint(config)
    assert "ctrl_cfg.null_emb" in state
    state["transformer.blocks.0.cond_heads.0.weight"] = torch.randn(7, 3)
    state["transformer.blocks.1.cond_heads.2.cond_proj.0.weight"] = torch.randn(5, 5)

    dit = build_from(tmp_path, state, config)
    assert set(dict(dit.named_parameters())) == set(expected)


@pytest.mark.parametrize(
    "key",
    [
        "transformer.blocks.0.attn.gate_proj.weight",  # gated_attn=False
        "transformer.blocks.0.dit_mlp.router.weight",  # moe=False; T3 renames it, nothing owns it
        "prompt_cfg.null_emb",  # prompt_conditioning=None
        "transformer.blocks.0.cond_head.who_knows",
        "some.entirely.new.key",
    ],
)
def test_unknown_key_raises(tmp_path, key):
    config = tiny_config()
    state, _ = synthetic_checkpoint(config)
    state[key] = torch.randn(4, 4)

    with pytest.raises(RuntimeError, match=r"[1-9]\d* unexpected checkpoint keys"):
        build_from(tmp_path, state, config)


def test_wrong_n_kv_heads_is_caught_by_the_shard_shape(tmp_path):
    """PARAM_TREE section 10.4: ``n_kv_heads`` was transcribed from a config.yaml
    nobody has read, and the reference's default is ``n_heads``. A wrong value
    reshapes GQA attention without erroring anywhere downstream, so the loader
    checks it against the tensors actually in the file."""
    config = tiny_config()
    state, _ = synthetic_checkpoint(config)
    wrong = tiny_config()
    wrong.n_kv_heads = config.n_heads

    with pytest.raises(RuntimeError, match="n_kv_heads"):
        build_from(tmp_path, state, wrong)


def test_wrong_patch_is_caught_by_the_conv_kernel(tmp_path):
    config = tiny_config()
    state, _ = synthetic_checkpoint(config)
    wrong = tiny_config()
    wrong.patch = (1, 1)

    with pytest.raises(RuntimeError, match="patch kernel"):
        build_from(tmp_path, state, wrong)


# ---------------------------------------------------------------------------
# 6. Regressions: F1, F2, F3, F4
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fusion", ["qkv", "fc1"])
def test_f1_prefused_tensor_and_split_shards_cannot_both_load(tmp_path, fusion):
    """A pre-fused key and its split shards in one file must raise.

    Before the fix this loaded silently. ``conflicts`` was keyed on
    ``(target, shard_id)``, so ``(target, None)`` — what a pre-fused
    ``attn.qkv_proj.weight`` claims — never collided with ``(target, "q")``. Both
    writers ran, and safetensors yields keys sorted, so ``k_proj``, ``q_proj``,
    the fused blob and then ``v_proj`` landed in that order: Q and K off the
    blob, V off ``v_proj``. A different shard layout gives a different tensor.
    Either spelling alone still loads (the ``CANONICAL`` round trip above).
    """
    config = tiny_config()
    state, _ = synthetic_checkpoint(config, spelling=LEGACY)
    D = config.d_model
    q_rows = config.n_heads * config.d_head
    kv_rows = config.n_kv_heads * config.d_head

    if fusion == "qkv":
        for i in range(config.n_layers):
            key = f"transformer.blocks.{i}.attn.qkv_proj.weight"
            state[key] = _tensor(key, (q_rows + 2 * kv_rows, D))
    else:
        for i in sorted(config.ctrl_layers):
            key = f"transformer.blocks.{i}.ctrl_mlpfusion.mlp.fc1.weight"
            state[key] = _tensor(key, (D, 2 * D))

    with pytest.raises(RuntimeError, match="pre-fused tensor and a split shard"):
        build_from(tmp_path, state, config)


def test_f2_cond_proj_tie_check_compares_in_full(tmp_path):
    """A divergence past column 64 must raise.

    The retired check compared a ``[:, :64]`` probe, so this exact checkpoint —
    block 2's slot 0 randomized from column 64 on — loaded clean with
    ``verify_cond_proj_tie=True``. The port keeps the owner block's copy and
    drops 23 sets; the reference loads all 24 into one tensor and keeps the last.
    They agree only while the stored copies agree, which is what this verifies.
    """
    config = tiny_config()
    assert config.d_model > 64, "the divergence has to live outside the retired probe"
    state, _ = synthetic_checkpoint(config)

    key = "transformer.blocks.2.attn_cond_head.cond_proj.0.weight"
    diverged = state[key].clone()
    diverged[:, 64:] = torch.randn_like(diverged[:, 64:])
    state[key] = diverged

    with pytest.raises(RuntimeError, match="divergent cond_proj copies"):
        build_from(tmp_path, state, config)

    # The escape hatch still loads it, and still keeps the owner block's copy.
    dit = build_from(tmp_path, state, config, verify_cond_proj_tie=False)
    owner = f"transformer.blocks.{COND_PROJ_SOURCE_BLOCK}.attn_cond_head.cond_proj.0.weight"
    got = dit.blocks[COND_PROJ_SOURCE_BLOCK].cond_head.cond_proj[0].weight.detach().cpu()
    assert torch.equal(got, state[owner].to(got.dtype))


@pytest.mark.parametrize(
    "present,winner",
    [
        (("mlp",), "mlp"),
        (("attn",), "attn"),  # F3: the fallback. Used to leave every block unloaded.
        (("attn", "mlp"), "mlp"),
        (("canonical",), "canonical"),
        (("attn", "canonical"), "canonical"),
        (("attn", "mlp", "canonical"), "canonical"),
    ],
)
def test_f3_bias_in_precedence(tmp_path, present, winner):
    """``mlp`` beats ``attn``, an already-canonical key beats both, and any one
    of the three alone is enough.

    This is ``world_model.py:386-389`` — ``mlp_bias if mlp_bias is not None else
    attn_bias``, under a ``setdefault`` — restated as a rank, because a streaming
    loader cannot see the whole dict and the resident weight must not depend on
    which shard a key happens to live in. Note the ordering is genuinely tested:
    safetensors yields keys sorted, so ``cond_head.bias_in`` arrives *before*
    ``mlp_cond_head.bias_in`` and a plain "last write wins" would pick the wrong
    one in the last two cases.
    """
    config = tiny_config()
    state, expected = synthetic_checkpoint(config, bias_in=present)
    dit = build_from(tmp_path, state, config)

    for i in range(config.n_layers):
        got = dit.blocks[i].cond_head.bias_in.detach().cpu()
        source = state[f"transformer.blocks.{i}.{BIAS_IN_KEYS[winner]}"].to(got.dtype)
        assert torch.equal(got, source), f"block {i}: bias_in did not come from {BIAS_IN_KEYS[winner]}"
        assert torch.equal(got, expected[f"blocks.{i}.cond_head.bias_in"].to(got.dtype))


def test_f3_no_bias_in_at_all_still_raises(tmp_path):
    """The fallback is a fallback, not a licence to skip the parameter. The
    reference loads ``strict=True`` and would report it missing too."""
    config = tiny_config()
    state, _ = synthetic_checkpoint(config, bias_in=())

    with pytest.raises(RuntimeError, match=r"[1-9]\d* unloaded parameters"):
        build_from(tmp_path, state, config)


def test_f4_dropped_keys_are_not_shape_validated(tmp_path):
    """T12 drops ``.cond_heads.`` unconditionally, including keys whose suffix
    the GQA/patch validators recognize.

    Validation used to run on the raw stream, before any drop, so these three
    keys raised about ``n_kv_heads`` and ``patch`` — a hard failure describing a
    config field, for keys the loader had already decided to throw away. Drops
    run first now; a dropped key's shape is not this model's business.
    """
    config = tiny_config()
    state, expected = synthetic_checkpoint(config)
    state["transformer.blocks.0.cond_heads.0.k_proj.weight"] = torch.randn(3, 5)
    state["transformer.blocks.0.cond_heads.0.q_proj.weight"] = torch.randn(9, 9)
    state["transformer.blocks.1.cond_heads.0.patchify.weight"] = torch.randn(2, 2)

    dit = build_from(tmp_path, state, config)
    assert set(dict(dit.named_parameters())) == set(expected)
