"""Weight loading for the native Waypoint-1.5-1B DiT (mstar loader pattern).

``build_waypoint_dit`` constructs ``WaypointDiT`` on meta, casts it to the serving
dtypes while still on meta (so ``to_empty`` allocates storage in the final
dtypes), moves it to the device, **re-ties ``cond_proj``**, then streams the
safetensors shards through ``load_weights_into``.

The order is fixed and ``retie_cond_proj()`` must follow ``to_empty``:
``Module._apply`` has no cross-module memo, so ``to_empty(device)`` silently
un-aliases the six ``cond_proj`` matrices that blocks 1..23 share with block 0.
Nothing raises; the symptoms are +0.6B resident parameters and 23 blocks of
``cond_proj`` the loader never fills (``CONTRACTS.md`` section 6.1).

Authoritative key map: ``docs/waypoint/PARAM_TREE.md``. Thirteen transforms sit
between the checkpoint's 369 keys and this module's 174 parameters:

===  ==============================================================  ===========
T0   ``transformer.blocks.{i}.`` -> ``blocks.{i}.``                  prefix
T1   ``unpatchify.weight`` ``[D,C,ph,pw]`` -> ``[C*ph*pw,D]``        reshape
T2   ``unpatchify.bias`` ``[C]`` -> ``[C*ph*pw]``                    reshape
T3   ``dit_mlp.{leaf}`` -> ``mlp.{leaf}``, 5-name allowlist          rename
T4   ``{attn,mlp}_cond_head.bias_in`` -> ``cond_head.bias_in``       merge
T5   ``attn_cond_head.cond_proj.{j}`` -> ``cond_head..{j}``          rename
T6   ``mlp_cond_head.cond_proj.{j}`` -> ``cond_head..{j+3}``         rename
T7   ``ctrl_mlpfusion.fc1_{x,c}`` -> ``ctrl_mlpfusion.mlp.fc1``      fuse dim 1
T8   ``ctrl_mlpfusion.fc2`` -> ``ctrl_mlpfusion.mlp.fc2``            rename
T9   ``cond_head.cond_proj.*`` for blocks 1..23                      drop
T10  ``ctrl_cfg.null_emb``                                           drop
T11  ``attn.{q,k,v}_proj`` -> ``attn.qkv_proj``                      fuse dim 0
T12  any ``.cond_heads.`` key (note the plural)                      drop
===  ==============================================================  ===========

T0 is not in ``PARAM_TREE.md``: it assumes the reference's two-level
``WorldModel``/``WorldDiT`` split, whereas ``components/dit.py`` collapses them
into one ``WaypointDiT`` whose blocks live at ``blocks.{i}``. Both spellings are
accepted here, as are the canonical post-transform spellings the reference's own
``pop``/``setdefault`` transforms tolerate (``PARAM_TREE.md`` section 3.5) —
which spelling the shipped file uses could not be established statically
(section 10.1), so guessing one was not an option.

Three things this file does that the mstar machinery does not give you:

  * **The two fusions need a fan-in, and ``name_remapper`` is ``str -> str|None``
    with none.** Both go through ``StackedParamRule``. Neither target module is a
    ``FusedColumnLinear`` — ``attention.py`` builds ``qkv_proj`` as a plain
    ``nn.Linear`` and ``layers.MLPFusion`` holds a merged ``[D,2D]`` ``mlp.fc1``
    — so neither parameter ships a ``weight_loader`` and ``default_weight_loader``
    would assert on the shard id. ``_attach_shard_loaders`` installs one, **after**
    ``to_empty`` (which drops attribute state along with the storage).
  * **Per-shard completeness.** ``load_weights_into`` returns *target* names, and
    q/k/v all share one target, so ``set(named_parameters()) - loaded`` is
    satisfied by any one of the three: a ``k_proj`` missing from every layer
    passes the wan22-style check silently (``PARAM_TREE.md`` S11d). The remapper
    therefore tallies ``(target, shard_id)`` pairs and the contract checks those
    too. Same hole, same fix, for ``fc1_x``/``fc1_c``.
  * **The reshaping transforms T1/T2 have no hook at all**, so they ride a thin
    adapter over the shard iterator, which is also where the config facts
    ``PARAM_TREE.md`` section 10.4 flags as transcribed-not-read (``n_kv_heads``,
    ``patch``) get validated against the tensor shapes actually on disk.

Completeness is a hard contract: a checkpoint key that reaches no parameter, a
parameter no key reached, a fused shard that never arrived, or two keys writing
the same slot, all raise. A silently skipped weight is a wrong-output bug, not a
warning. Explicitly dropped keys (T9/T10/T12) are expected and silent.

Two consequences of "two keys writing the same slot" that a naive
``(target, shard_id)`` tally does not cover, and that both fusions have:

  * A **pre-fused** key (``attn.qkv_proj.weight``, ``ctrl_mlpfusion.mlp.fc1``
    — canonical spellings ``PARAM_TREE.md`` section 3.5 says the reference
    tolerates) claims ``(target, None)``, which does not collide with
    ``(target, "q")``. Left alone, a file carrying both spellings assembles one
    parameter out of both sources in whatever order the shard iterator happens
    to yield — Q and K off the fused blob, V off ``v_proj``, no error. So
    ``(target, None)`` is made to conflict with every ``(target, shard)`` of a
    target that has a stacked rule.
  * ``bias_in`` is the opposite case: three spellings legitimately share one
    target (T4), so that collision is *arbitrated* by an explicit precedence
    rank rather than refused. See ``_BIAS_IN_SPELLINGS``.

Order inside the pipeline: the unconditional drops (T10/T12) run **before** the
shape validation in ``_adapt_checkpoint_stream``, because a ``.cond_heads.`` key
that happens to end in ``.k_proj.weight`` is a T12 drop and not a GQA violation.
T9's per-block ``cond_proj`` drop is *not* in that pre-filter: those keys are
exactly what ``_CondProjTieCheck`` exists to compare.

``load_hf_weights`` is deliberately not used, for the reason wan22 documents: its
``skip_predicate`` runs *before* the remapper and would drop keys outside the
unexpected-key accounting.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from torch import nn

# _apply_stacked is imported rather than reimplemented on purpose: the remapper
# has to resolve a key to the same target load_weights_into will, and a local
# copy of a three-line matcher is a thing that drifts.
from mstar.model.loader.base import StackedParamRule, _apply_stacked, load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards
from mstar.model.waypoint.components.dit import WaypointDiT
from mstar.model.waypoint.components.layers import CondHead
from mstar.model.waypoint.config import WaypointConfig

__all__ = [
    "build_waypoint_dit",
    "remap_checkpoint_key",
    "parameter_census",
    "WAYPOINT_STACKED_PARAMS",
    "COND_PROJ_SOURCE_BLOCK",
]


# Which block's cond_proj set is the physical one. The reference ties every
# block's cond_proj to block 0's in __init__ and then loads all 24 stored copies
# into that one tensor, so it effectively keeps block 23's (last write wins);
# the port keeps ONE copy and drops the rest, so it has to name a block.
#
# **Not a knob.** This records a fact of ``components/dit.py``, it does not
# choose one: ``WaypointDiT.retie_cond_proj`` hardcodes
# ``ref_proj = self.blocks[0].cond_head.cond_proj``, so block 0 is the only
# spelling ``named_parameters()`` reports after the retie and any other value
# here would turn all six kept keys into unexpected-key failures and leave the
# six real parameters unloaded. The constant exists to name the fact at its two
# use sites, and ``_assert_cond_proj_tied`` asserts the module tree still agrees
# with it — so if ``retie_cond_proj`` ever moves the owner, that fails with a
# sentence rather than with twelve confusing key errors. Changing this number
# without changing ``retie_cond_proj`` (which this file does not own) is not a
# supported edit.
#
# Block 0 is also what the reference's own __init__ ties to, what CONTRACTS
# section 6 and ``layers.CondHead``'s docstring prescribe, and what PARAM_TREE
# section 4.9's fill-forward loop uses as its reference.
#
# The choice only matters if the 24 stored copies disagree, which is exactly the
# silent divergence PARAM_TREE flags as S9 — so ``verify_cond_proj_tie`` checks
# that they agree instead of relying on the argument. When they agree, block 0
# and block 23 are the same tensor and the choice is moot; when they do not, the
# load raises rather than quietly disagreeing with the reference.
COND_PROJ_SOURCE_BLOCK = 0

# Slot ids for the two port-side fusions. Order defines the layout.
QKV_SHARD_IDS: tuple[str, ...] = ("q", "k", "v")
CTRL_FC1_SHARD_IDS: tuple[str, ...] = ("x", "c")

# Fused-shard routing. The leading dots matter: without them ``.v_proj`` would
# also match inside ``qkv_proj``. Note this is NOT ``LLAMA_STACKED_PARAMS``,
# whose extra ``gate_proj``/``up_proj`` rules would be live substring matchers
# for parameters Waypoint does not have.
#
# ``_apply_stacked`` rewrites by ``str.replace``, so
# ``blocks.0.attn.q_proj.weight``          -> ``blocks.0.attn.qkv_proj.weight``
# ``blocks.0.ctrl_mlpfusion.fc1_x.weight`` -> ``blocks.0.ctrl_mlpfusion.mlp.fc1.weight``
WAYPOINT_STACKED_PARAMS: list[StackedParamRule] = [
    StackedParamRule(".qkv_proj", ".q_proj", "q"),
    StackedParamRule(".qkv_proj", ".k_proj", "k"),
    StackedParamRule(".qkv_proj", ".v_proj", "v"),
    StackedParamRule(".mlp.fc1", ".fc1_x", "x"),
    StackedParamRule(".mlp.fc1", ".fc1_c", "c"),
]

# The port's block prefix (``components/dit.py`` collapses WorldModel/WorldDiT).
MODEL_BLOCK_PREFIX = "blocks."

# ``transformer.`` optional: T0. Accepts the reference's two-level spelling and
# the collapsed one.
_BLOCK_RE = re.compile(r"^(?:transformer\.)?blocks\.(\d+)\.(.+)$")
# Legacy half-heads. j is range(3) on both sides (PARAM_TREE section 4.5/4.6);
# a j >= 3 here is malformed and is left unmapped so it surfaces as unexpected.
_LEGACY_COND_PROJ_RE = re.compile(r"^(attn|mlp)_cond_head\.cond_proj\.(\d+)\.weight$")
_COND_PROJ_RE = re.compile(r"^cond_head\.cond_proj\.(\d+)\.weight$")

# T3 is an explicit five-name allowlist, not a ``dit_mlp.*`` wildcard: a wildcard
# is over-permissive and would silently absorb a future ``dit_mlp.*`` key instead
# of surfacing it. ``expert_*``/``router`` do not exist under moe=False; they are
# listed because the reference renames them, and the rename lands on a parameter
# this port does not have, which is a loud unexpected-key failure either way.
_DIT_MLP_LEAVES: tuple[str, ...] = (
    "fc1.weight",
    "fc2.weight",
    "expert_in",
    "expert_out",
    "router.weight",
)

# T10. ``CFG.forward`` is a training-time dropout with no call site in the
# reference's ``WorldModel.forward``; the port does not instantiate the tensor.
# Dropped explicitly rather than left unmatched so the unexpected-key accounting
# keeps no hole (PARAM_TREE S10).
_DROPPED_TOP_LEVEL_KEYS = frozenset({"ctrl_cfg.null_emb"})

# T12. Substring filter, unconditional, note the plural.
_COND_HEADS_FRAGMENT = ".cond_heads."

# Half the CondHead slots come from each legacy half-head.
_COND_PROJ_PER_HEAD = CondHead.n_cond // 2

# T4 precedence over the three spellings that share ``cond_head.bias_in``,
# lowest first, highest wins. This is the reference's pop/setdefault outcome
# (``world_model.py:386-389``) restated as a rank:
#
#     if attn_bias is not None or mlp_bias is not None:
#         state_dict.setdefault(p + "cond_head.bias_in",
#                               mlp_bias if mlp_bias is not None else attn_bias)
#
# so mlp beats attn, and `setdefault` means an already-canonical
# ``cond_head.bias_in`` beats both. Note what this is NOT: the reference does
# not *drop* ``attn_cond_head.bias_in``, it uses it as the fallback when the mlp
# spelling is absent — and PARAM_TREE section 10.2 leaves it unresolved which of
# the two the shipped file actually carries (the 49,152-parameter difference is
# five orders of magnitude below the precision of "1.86B"). Dropping the attn
# copy unconditionally is therefore a plausible day-one failure: 24 unloaded
# ``cond_head.bias_in`` on a file that only has the attn spelling.
#
# A rank is needed rather than "last write wins" because this loader streams: the
# reference arbitrates over a materialized dict, while here the three spellings
# can arrive in any order across shard files, and the resident weight must not
# depend on that order. Arbitration itself lives in ``build_waypoint_dit``, the
# only place that sees every key.
_BIAS_IN_SPELLINGS: tuple[str, ...] = (
    "attn_cond_head.bias_in",
    "mlp_cond_head.bias_in",
    "cond_head.bias_in",
)

# Reference init for the value-residual scalar, used by the no-checkpoint path.
_V_LAMB_INIT = 0.5
# Weight init std for the no-checkpoint path. Not a checkpoint fact; it only has
# to produce finite, sanely scaled activations for a structural smoke test.
_STRUCTURAL_INIT_STD = 0.02


# --------------------------------------------------------------------------
# Name remapping (T0, T3-T6, T8-T12)
# --------------------------------------------------------------------------


def _remap_block_suffix(suffix: str, layer_idx: int) -> str | None:
    """Map one per-block checkpoint suffix to its parameter suffix, or ``None``
    to drop it. ``suffix`` excludes the ``blocks.{i}.`` prefix."""
    # T4: both legacy spellings map to the one target, and the attn copy is a
    # FALLBACK, not a drop (see _BIAS_IN_SPELLINGS). Which of the three actually
    # gets written is decided by rank in build_waypoint_dit; a pure key->key
    # function cannot decide it, because it cannot see whether the winner is
    # elsewhere in the file.
    if suffix in ("attn_cond_head.bias_in", "mlp_cond_head.bias_in"):
        suffix = "cond_head.bias_in"

    # T5/T6: identity index map for the attn head (0,1,2 -> 0,1,2), +3 for the
    # mlp head (0,1,2 -> 3,4,5). Slots 0-2 drive the attention sublayer and 3-5
    # the MLP sublayer, which is what the source names say; swapping them is
    # PARAM_TREE S5, silent and numerically catastrophic.
    legacy = _LEGACY_COND_PROJ_RE.match(suffix)
    if legacy is not None:
        head, j = legacy.group(1), int(legacy.group(2))
        if j < _COND_PROJ_PER_HEAD:
            slot = j if head == "attn" else j + _COND_PROJ_PER_HEAD
            suffix = f"cond_head.cond_proj.{slot}.weight"

    # T3.
    for leaf in _DIT_MLP_LEAVES:
        if suffix == "dit_mlp." + leaf:
            suffix = "mlp." + leaf
            break

    # T8. Guarded on fc2 alone, separately from T7's both-halves guard; omitting
    # it leaves 8 layers' ctrl_mlpfusion.mlp.fc2 unloaded.
    if suffix == "ctrl_mlpfusion.fc2.weight":
        suffix = "ctrl_mlpfusion.mlp.fc2.weight"

    # T9. Runs last, on the post-T5/T6 name, so it catches both the legacy
    # half-head spellings and an already-canonical one.
    if _COND_PROJ_RE.match(suffix) is not None and layer_idx != COND_PROJ_SOURCE_BLOCK:
        return None

    # T7 (fc1_x/fc1_c) and T11 (q/k/v_proj) are deliberately left alone: they are
    # fan-ins, which a remapper cannot express, and WAYPOINT_STACKED_PARAMS
    # routes them.
    return suffix


def _is_unconditionally_dropped(name: str) -> bool:
    """T12 and T10 — the drops that depend on nothing but the key.

    Factored out because the shard adapter has to apply them *before* it
    validates tensor shapes: ``…cond_heads.0.k_proj.weight`` is a T12 drop, not
    a GQA shape violation, and validating first turns an expected drop into a
    hard failure. One predicate, two call sites, so the pre-filter and the
    remapper cannot drift apart on what counts as dropped.

    T9 (``cond_proj`` for blocks other than ``COND_PROJ_SOURCE_BLOCK``) is
    deliberately NOT here even though it is also a drop: those 138 keys are
    exactly what ``_CondProjTieCheck`` has to see, so they must survive the
    stream filter and be dropped later, in the remapper.
    """
    return _COND_HEADS_FRAGMENT in name or name in _DROPPED_TOP_LEVEL_KEYS


def _bias_in_rank(name: str) -> int | None:
    """T4 precedence rank of a per-block ``bias_in`` key, or ``None`` if the key
    is not one. Higher wins; see ``_BIAS_IN_SPELLINGS``."""
    block = _BLOCK_RE.match(name)
    if block is None:
        return None
    suffix = block.group(2)
    if suffix not in _BIAS_IN_SPELLINGS:
        return None
    return _BIAS_IN_SPELLINGS.index(suffix)


def remap_checkpoint_key(name: str) -> str | None:
    """Map one Waypoint checkpoint key to the native parameter path, or return
    ``None`` for a key that is intentionally dropped (T9/T10/T12).

    Pure function of the key — no model, no config — so it can be exercised
    directly. Keys it maps to a name that is not a parameter are the caller's
    problem: ``build_waypoint_dit`` treats those as unexpected and raises.

    **Not injective, by design, in exactly one place.** All three T4 ``bias_in``
    spellings map to ``blocks.{i}.cond_head.bias_in``; which one is allowed to
    write is a precedence question that needs the whole key set, so it is settled
    in ``build_waypoint_dit`` and not here (``_BIAS_IN_SPELLINGS``). Everywhere
    else a second key resolving to a slot that is already claimed is a hard
    error.
    """
    if _is_unconditionally_dropped(name):  # T10/T12
        return None

    block = _BLOCK_RE.match(name)
    if block is None:
        # Top-level keys are identity: denoise_step_emb.mlp.{fc1,fc2}.weight,
        # ctrl_emb.mlp.{fc1,fc2}.weight, patchify.weight, unpatchify.{weight,bias},
        # out_norm.fc.weight.
        return name

    layer_idx, suffix = int(block.group(1)), block.group(2)
    mapped = _remap_block_suffix(suffix, layer_idx)
    if mapped is None:
        return None
    # T0.
    return f"{MODEL_BLOCK_PREFIX}{layer_idx}.{mapped}"


# --------------------------------------------------------------------------
# Tensor transforms and shape validation (T1, T2) over the shard stream
# --------------------------------------------------------------------------


def _unpatchify_weight(tensor: torch.Tensor, config: WaypointConfig, key: str) -> torch.Tensor:
    """T1. ``[D, C, ph, pw]`` conv kernel -> ``[C*ph*pw, D]`` Linear weight.

    ``permute(1, 2, 3, 0)`` then ``reshape``: the Linear's output feature axis is
    ordered ``(c, ph, pw)`` with pw fastest, because ``WaypointDiT.forward``
    unpacks it as ``view(B, N, Hp, Wp, C, ph, pw)``. Dropping the permute, or
    transposing ph/pw inside it, keeps the shape and silently reprojects every
    output sub-pixel (PARAM_TREE S1/S1b). ``reshape``, not ``view`` — the
    permuted tensor is not contiguous.
    """
    ph, pw = config.patch
    if tensor.ndim == 4:
        d_model, channels, k_h, k_w = tensor.shape
        if (k_h, k_w) != (ph, pw):
            raise RuntimeError(
                f"{key} is a {k_h}x{k_w} patch kernel but WaypointConfig.patch is "
                f"{(ph, pw)}. PARAM_TREE section 10.4: patch was transcribed from the "
                "checkpoint's config.yaml, not read from it; the checkpoint wins."
            )
        if channels != config.channels or d_model != config.d_model:
            raise RuntimeError(
                f"{key} has shape {tuple(tensor.shape)}; expected "
                f"[d_model={config.d_model}, channels={config.channels}, {ph}, {pw}]."
            )
        return tensor.permute(1, 2, 3, 0).reshape(-1, d_model)

    # Already canonical (a checkpoint written post-transform). The reference's
    # own ndim == 4 guard makes T1 idempotent the same way.
    expected = (config.channels * ph * pw, config.d_model)
    if tuple(tensor.shape) != expected:
        raise RuntimeError(
            f"{key} has shape {tuple(tensor.shape)}; expected a 4-D "
            f"[{config.d_model}, {config.channels}, {ph}, {pw}] conv kernel or an "
            f"already-permuted {list(expected)} Linear weight."
        )
    return tensor


def _unpatchify_bias(tensor: torch.Tensor, config: WaypointConfig, key: str) -> torch.Tensor:
    """T2. One learned bias per latent channel, repeated across the patch.

    ``[C] -> [C,1,1] -> expand(-1, ph, pw) -> reshape(-1)``. The expand target is
    ``(C, ph, pw)`` so the flatten agrees with T1's row ordering; a
    ``repeat(ph*pw)`` produces the same ``[128]`` shape with the bias on the
    wrong sub-pixel, which integrates into a slow colour drift over an
    autoregressive rollout rather than failing (PARAM_TREE S2).
    """
    ph, pw = config.patch
    if tensor.numel() == config.channels:
        return tensor[:, None, None].expand(-1, ph, pw).reshape(-1)
    expected = config.channels * ph * pw
    if tensor.numel() != expected:
        raise RuntimeError(
            f"{key} has {tensor.numel()} elements; expected channels={config.channels} "
            f"(per-channel, to be expanded over the {ph}x{pw} patch) or the "
            f"already-expanded {expected}."
        )
    return tensor


def _check_patchify(tensor: torch.Tensor, config: WaypointConfig, key: str) -> None:
    """Validate ``config.patch`` and ``config.channels`` against the conv kernel.

    Unlike ``unpatchify``, this one is 4-D in both the file and the module, so it
    is validated rather than transformed.
    """
    ph, pw = config.patch
    if tensor.ndim != 4:
        raise RuntimeError(f"{key} should be a 4-D conv kernel; got {tuple(tensor.shape)}.")
    d_model, channels, k_h, k_w = tensor.shape
    if (k_h, k_w) != (ph, pw):
        raise RuntimeError(
            f"{key} is a {k_h}x{k_w} patch kernel but WaypointConfig.patch is "
            f"{(ph, pw)}. PARAM_TREE section 10.4: patch was transcribed rather than "
            "read; a wrong value silently rescales the token grid."
        )
    if (d_model, channels) != (config.d_model, config.channels):
        raise RuntimeError(
            f"{key} has shape {tuple(tensor.shape)}; expected "
            f"[d_model={config.d_model}, channels={config.channels}, {ph}, {pw}]."
        )


def _check_attn_proj(
    tensor: torch.Tensor, config: WaypointConfig, key: str, rows: int, what: str
) -> None:
    """Validate an unfused q/k/v projection against the config's head counts.

    This is the check that pins ``n_kv_heads``. It is worth doing explicitly even
    though ``_SliceShardLoader`` would also catch it: a wrong ``n_kv_heads``
    reshapes attention without erroring anywhere downstream, and "shard 'k' shape
    mismatch" does not tell a reader which config field to go look at.
    """
    if tensor.ndim != 2 or tuple(tensor.shape) != (rows, config.d_model):
        raise RuntimeError(
            f"{key} has shape {tuple(tensor.shape)}; expected "
            f"[{what} = {rows}, d_model = {config.d_model}] from n_heads="
            f"{config.n_heads}, n_kv_heads={config.n_kv_heads}, d_head={config.d_head}. "
            "PARAM_TREE section 10.4: n_kv_heads was transcribed from the checkpoint's "
            "config.yaml rather than read from it (the reference default is n_heads), "
            "and a wrong value reshapes GQA attention without raising."
        )


def _cond_proj_slot(key: str) -> tuple[int, int] | None:
    """``(block_idx, canonical slot 0..5)`` for a cond_proj key, else ``None``.

    Accepts the legacy half-head spellings and the canonical one; used only by
    the tie check, which runs on the raw stream before the remapper.
    """
    block = _BLOCK_RE.match(key)
    if block is None:
        return None
    layer_idx, suffix = int(block.group(1)), block.group(2)
    legacy = _LEGACY_COND_PROJ_RE.match(suffix)
    if legacy is not None:
        j = int(legacy.group(2))
        if j >= _COND_PROJ_PER_HEAD:
            return None
        return layer_idx, (j if legacy.group(1) == "attn" else j + _COND_PROJ_PER_HEAD)
    canonical = _COND_PROJ_RE.match(suffix)
    if canonical is not None:
        return layer_idx, int(canonical.group(1))
    return None


class _CondProjTieCheck:
    """Confirms the checkpoint's 24 stored ``cond_proj`` sets agree before 23 of
    them are dropped (PARAM_TREE S9).

    The port keeps ``COND_PROJ_SOURCE_BLOCK``'s copy; the reference keeps block
    23's, because it loads all 24 into one shared tensor and the last write wins.
    The two agree only if the stored copies are identical — which they should be,
    since the tie was in place at training time, but nothing in the file or in
    either loader enforces it. If a fine-tune ever broke the tie, the port and
    the reference would silently produce different video.

    Compares the matrices **in full**.

    An earlier revision compared a fixed ``[:, :64]`` column probe and justified
    it as "~3 MB of retained probes instead of 1.16 GB of streamed comparisons".
    That trade does not exist: the two figures measure different things. The
    ~1.13 GiB of stored ``cond_proj`` (24 blocks x 6 x ``[2048, 2048]`` bf16)
    streams past either way — this class sits on a stream the loader is already
    consuming and adds no reads at all. The only quantity the probe reduced is
    what is **retained**: one reference copy per slot, i.e. 6 x ``[2048, 2048]``
    fp32 = **96 MiB** held for the duration of the load, against ~3 MiB for the
    probe. 96 MiB once, at load, next to a 2.56 GB model, is not a cost worth an
    unsound check — and the probe was demonstrably unsound: randomizing block 2
    slot 0's columns 64 onward loaded clean with ``verify_cond_proj_tie=True``.

    ``verify_cond_proj_tie=False`` remains the escape hatch, and it is now the
    only thing between the check and a machine that cannot spare the 96 MiB.

    Whichever block arrives first for a slot becomes that slot's reference, so
    this does not depend on shard ordering, and equality across all 24 makes the
    block-0-vs-23 choice moot rather than merely defensible.
    """

    def __init__(self) -> None:
        self._reference: dict[int, tuple[int, torch.Tensor]] = {}
        self.divergent: list[str] = []

    def observe(self, key: str, tensor: torch.Tensor) -> None:
        slot_info = _cond_proj_slot(key)
        if slot_info is None or tensor.ndim != 2:
            return
        block_idx, slot = slot_info
        # fp32 on CPU: uniform, lossless from the checkpoint's bf16, and it keeps
        # the retained set off the serving device.
        seen = tensor.detach().to(device="cpu", dtype=torch.float32)
        known = self._reference.get(slot)
        if known is None:
            self._reference[slot] = (block_idx, seen.clone())
            return
        ref_block, ref_tensor = known
        if not torch.equal(ref_tensor, seen):
            self.divergent.append(f"slot {slot}: block {block_idx} != block {ref_block}")


def _adapt_checkpoint_stream(
    weights: Iterable[tuple[str, torch.Tensor]],
    config: WaypointConfig,
    tie_check: _CondProjTieCheck | None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Apply the two reshaping transforms and validate the transcribed config
    facts, in one streaming pass over the shards.

    T1/T2 live here rather than in the remapper because mstar has no reshape
    hook: ``name_remapper`` sees names only, and ``weight_loader`` is per-target
    (the two ``unpatchify`` parameters have no fused loader to hang it on).

    **Drops run before validation.** These checks fire on key *suffixes*, so a
    key that T12 drops unconditionally can still end in ``.k_proj.weight`` —
    ``…blocks.0.cond_heads.0.k_proj.weight`` did exactly that and raised a GQA
    shape error about a key the loader had already decided to throw away.
    Validating only what survives the drop filter is the fix; a dropped key's
    shape is not this model's business.
    """
    q_rows = config.n_heads * config.d_head
    kv_rows = config.n_kv_heads * config.d_head

    for key, tensor in weights:
        if _is_unconditionally_dropped(key):  # T10/T12, before anything reads a shape
            continue

        # Order matters: "unpatchify.weight".endswith("patchify.weight") is True,
        # so unpatchify must be tested first and the rest must be elif.
        if key.endswith("unpatchify.weight"):
            tensor = _unpatchify_weight(tensor, config, key)
        elif key.endswith("unpatchify.bias"):
            tensor = _unpatchify_bias(tensor, config, key)
        elif key.endswith("patchify.weight"):
            _check_patchify(tensor, config, key)
        elif key.endswith(".q_proj.weight"):
            _check_attn_proj(tensor, config, key, q_rows, "n_heads * d_head")
        elif key.endswith(".k_proj.weight") or key.endswith(".v_proj.weight"):
            _check_attn_proj(tensor, config, key, kv_rows, "n_kv_heads * d_head")
        elif tie_check is not None:
            tie_check.observe(key, tensor)

        yield key, tensor


# --------------------------------------------------------------------------
# Fused-parameter shard loaders (T7, T11)
# --------------------------------------------------------------------------


class _SliceShardLoader:
    """``param.weight_loader`` for a port-side fused parameter.

    Copies one checkpoint shard into its slice of the fused tensor. Exists
    because neither fusion target is a ``FusedColumnLinear``: ``qkv_proj`` is a
    plain ``nn.Linear`` in ``components/attention.py`` and ``mlp.fc1`` is a plain
    ``nn.Linear`` inside ``layers.MLPFusion``, so neither carries a loader and
    ``default_weight_loader`` asserts ``loaded_shard_id is None``. It also
    generalizes ``FusedColumnLinear`` in the one way ``ctrl_mlpfusion`` needs:
    that fusion concatenates along **dim 1** (the input-feature axis), not dim 0.
    """

    def __init__(self, param_name: str, dim: int, layout: dict[str, tuple[int, int]]):
        self.param_name = param_name
        self.dim = dim
        self.layout = layout

    def __call__(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | int | None = None,
    ) -> None:
        if loaded_shard_id is None:
            # A checkpoint already written in the fused spelling. Only reachable
            # for ctrl_mlpfusion.mlp.fc1 (PARAM_TREE section 3.5 lists it as a
            # canonical spelling the reference accepts); harmless and symmetric
            # for qkv_proj.
            if tuple(param.data.shape) != tuple(loaded_weight.shape):
                raise RuntimeError(
                    f"{self.param_name}: pre-fused checkpoint tensor has shape "
                    f"{tuple(loaded_weight.shape)}, parameter is {tuple(param.data.shape)}."
                )
            param.data.copy_(loaded_weight)
            return

        if loaded_shard_id not in self.layout:
            raise RuntimeError(
                f"{self.param_name}: unknown shard id {loaded_shard_id!r}; expected one "
                f"of {list(self.layout)}."
            )
        offset, size = self.layout[loaded_shard_id]
        dst = param.data.narrow(self.dim, offset, size)
        if tuple(dst.shape) != tuple(loaded_weight.shape):
            raise RuntimeError(
                f"{self.param_name}: shard {loaded_shard_id!r} is "
                f"{tuple(loaded_weight.shape)} but its slice of the fused parameter is "
                f"{tuple(dst.shape)} (dim {self.dim}, offset {offset})."
            )
        dst.copy_(loaded_weight)


def _attach_shard_loaders(
    dit: WaypointDiT, config: WaypointConfig
) -> dict[str, tuple[str, ...]]:
    """Install ``_SliceShardLoader`` on every fused parameter and return
    ``{param_name: required shard ids}``.

    MUST run after ``to_empty(device)``: that reallocates the Parameter objects
    and drops attached attributes along with the meta storage — the same reason
    ``FusedColumnLinear`` re-attaches its loaders from ``_apply``.
    """
    q_rows = config.n_heads * config.d_head
    kv_rows = config.n_kv_heads * config.d_head
    d_model = config.d_model

    fused: dict[str, tuple[str, ...]] = {}
    for name, param in dit.named_parameters():
        if name.endswith(".attn.qkv_proj.weight"):
            # cat([q, k, v], dim=0) — verified against the reference's own fusion
            # (patch_model.MergedQKVAttn's cat) and against the matching
            # split((q_out, kv_out, kv_out), dim=-1) in components/attention.py.
            # GQA makes the shards unequal, so a q/k swap raises but a k/v swap
            # loads cleanly and produces meaningless attention (PARAM_TREE S11).
            dim, layout = 0, {
                "q": (0, q_rows),
                "k": (q_rows, kv_rows),
                "v": (q_rows + kv_rows, kv_rows),
            }
            expected_shape = (q_rows + 2 * kv_rows, d_model)
        elif name.endswith(".ctrl_mlpfusion.mlp.fc1.weight"):
            # cat([fc1_x, fc1_c], dim=1) — x first. layers.MLPFusion splits it
            # straight back with chunk(2, dim=1), whose low columns are the token
            # half; the reverse order keeps the [2048, 4096] shape and applies
            # controller conditioning to tokens and vice versa (PARAM_TREE S7).
            dim, layout = 1, {"x": (0, d_model), "c": (d_model, d_model)}
            expected_shape = (d_model, 2 * d_model)
        else:
            continue

        if tuple(param.shape) != expected_shape:
            raise RuntimeError(
                f"{name} is {tuple(param.shape)} but this config implies "
                f"{expected_shape}; the module tree and WaypointConfig disagree "
                "before the checkpoint was even opened."
            )
        param.weight_loader = _SliceShardLoader(name, dim, layout)
        fused[name] = tuple(layout)

    return fused


# --------------------------------------------------------------------------
# Structural build helpers
# --------------------------------------------------------------------------


def parameter_census(dit: WaypointDiT) -> tuple[int, int, int]:
    """``(deduplicated tensors, deduplicated numel, raw state_dict numel)``.

    ``named_parameters()`` deduplicates aliased Parameters; ``state_dict()`` does
    not, so the gap between the two counts is exactly the tied ``cond_proj``. For
    the 720P checkpoint this is ``(174, 1_281_958_040, 1_860_771_992)`` — the
    "1.28B resident / 1.86B stored" figures.

    Those differ by 2,048 from ``PARAM_TREE.md`` section 5.2's 1,281,960,088 /
    1,860,774,040: that arithmetic carries ``ctrl_cfg.null_emb`` ``[1, 1, 2048]``
    in both totals, and the port drops it (T10).
    """
    params = dict(dit.named_parameters())
    return (
        len(params),
        sum(p.numel() for p in params.values()),
        sum(t.numel() for t in dit.state_dict().values()),
    )


def _assert_cond_proj_tied(dit: WaypointDiT, config: WaypointConfig) -> None:
    """Fail if ``retie_cond_proj()`` did not take.

    ``named_parameters()`` deduplicates aliased Parameters, so a correctly tied
    model reports 6 ``cond_proj`` tensors and an un-tied one reports
    ``6 * n_layers``. This is PARAM_TREE S9b, which is otherwise entirely silent
    — the un-tied model is numerically correct and merely 0.6B parameters
    heavier, so no output check catches it and the completeness contract would
    instead report 138 unloaded parameters with no hint as to why.

    Both the tensor count and the resident/stored numel gap are checked, because
    they fail differently: the count catches "never tied", while the numel gap
    catches a partial tie (some blocks aliased, some not) that still leaves the
    count wrong in a way a reader might not connect to memory. Both bounds are
    derived from ``config``, not hardcoded to the 720P variant, so they hold for
    the 360P sibling and for the reduced configs used in testing.
    """
    tied = [name for name, _ in dit.named_parameters() if ".cond_head.cond_proj." in name]
    if len(tied) != CondHead.n_cond:
        raise RuntimeError(
            f"cond_proj is not tied: named_parameters() reports {len(tied)} cond_proj "
            f"tensors, expected {CondHead.n_cond} (one physical set, aliased by all "
            f"{config.n_layers} blocks). to_empty(device) un-ties them and "
            "retie_cond_proj() must be called after it, not before (CONTRACTS "
            "section 6.1)."
        )
    # COND_PROJ_SOURCE_BLOCK is a record of which block retie_cond_proj aliases
    # the others onto, not a choice this file gets to make; assert it rather than
    # trust it. If the two ever disagree, T9 drops the six keys the module keeps
    # and keeps the six it drops, which surfaces as 6 unexpected keys plus 6
    # unloaded parameters and no explanation.
    owner_prefix = f"{MODEL_BLOCK_PREFIX}{COND_PROJ_SOURCE_BLOCK}.cond_head.cond_proj."
    if not all(name.startswith(owner_prefix) for name in tied):
        raise RuntimeError(
            f"cond_proj's surviving owner is not block {COND_PROJ_SOURCE_BLOCK}: "
            f"named_parameters() reports {sorted(tied)[:2]}. "
            "weight_loader.COND_PROJ_SOURCE_BLOCK and "
            "WaypointDiT.retie_cond_proj (which hardcodes blocks[0]) must name the "
            "same block; the constant records that fact and cannot change it."
        )
    _, dedup_numel, raw_numel = parameter_census(dit)
    # Every block past the first contributes 6 aliased [D, D] matrices that
    # state_dict() re-expands and named_parameters() does not. For 720P that is
    # 23 * 6 * 2048^2 = 578,813,952, i.e. the 1.86B - 1.28B gap.
    expected_gap = (config.n_layers - 1) * CondHead.n_cond * config.d_model**2
    if raw_numel - dedup_numel != expected_gap:
        raise RuntimeError(
            f"cond_proj tying is inconsistent: state_dict() holds {raw_numel} elements "
            f"and named_parameters() {dedup_numel}, a gap of {raw_numel - dedup_numel}; "
            f"a fully tied model must differ by exactly {expected_gap} "
            f"(({config.n_layers} - 1) blocks x {CondHead.n_cond} x {config.d_model}^2). "
            "Some blocks' cond_proj are aliased and some are not."
        )


def _assert_expected_layout(dit: WaypointDiT) -> None:
    """Fail early if the module tree is not the one the remapper targets."""
    params = dict(dit.named_parameters())
    if not any(name.startswith(f"{MODEL_BLOCK_PREFIX}0.") for name in params):
        raise RuntimeError(
            f"No parameter starts with {MODEL_BLOCK_PREFIX!r}; weight_loader's key map "
            "targets components/dit.py's collapsed tree (blocks.{i}, not "
            "transformer.blocks.{i}). Update MODEL_BLOCK_PREFIX to match the module."
        )


def _initialize_structurally(dit: WaypointDiT, seed: int = 0) -> None:
    """Fill a ``to_empty``-materialized model with finite values.

    ``to_empty`` allocates *uninitialized* storage, which routinely contains NaN
    and Inf bit patterns, so a "no checkpoint" model is not usable for a shape or
    plumbing smoke test until something writes to every parameter. Not an attempt
    to reproduce the reference's init — only the two parameters whose init is a
    documented fact are reproduced (``v_lamb`` = 0.5, and the two 1-D tensors
    ``cond_head.bias_in``/``unpatchify.bias``, both zeros in the reference).

    Iterating ``named_parameters()`` writes each tied ``cond_proj`` once, which is
    what makes the aliasing survive.
    """
    generators: dict[torch.device, torch.Generator] = {}
    with torch.no_grad():
        for _, param in dit.named_parameters():
            if param.ndim == 0:
                param.fill_(_V_LAMB_INIT)
            elif param.ndim == 1:
                param.zero_()
            else:
                generator = generators.get(param.device)
                if generator is None:
                    generator = torch.Generator(device=param.device)
                    generator.manual_seed(seed)
                    generators[param.device] = generator
                param.normal_(0.0, _STRUCTURAL_INIT_STD, generator=generator)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_waypoint_dit(
    config: WaypointConfig,
    checkpoint_dir: str | Path | None = None,
    device: torch.device | str = "cpu",
    *,
    skip_weight_loading: bool = False,
    verify_cond_proj_tie: bool = True,
) -> WaypointDiT:
    """Meta-build, materialize on ``device``, and load the checkpoint into a
    ready-to-serve (eval-mode) native Waypoint DiT.

    Args:
        config: the checkpoint's config. ``n_kv_heads`` and ``patch`` are
            validated against the tensor shapes actually in the file.
        checkpoint_dir: local directory holding ``model.safetensors`` (or an
            index plus shards). Required unless ``skip_weight_loading``. Never
            downloaded — the caller resolves the path.
        device: where to materialize.
        skip_weight_loading: build the structure only, with no checkpoint access
            at all. For shape/plumbing work before the weights exist; the result
            is randomly initialized and produces meaningless output.
        verify_cond_proj_tie: check that the checkpoint's 24 stored ``cond_proj``
            sets agree before 23 of them are dropped. See ``_CondProjTieCheck``.

    Raises:
        RuntimeError: on any completeness failure — an unexpected checkpoint key,
            an unloaded parameter, a fused shard that never arrived, two keys
            claiming one slot, or divergent ``cond_proj`` copies.
    """
    if not skip_weight_loading and checkpoint_dir is None:
        raise ValueError(
            "build_waypoint_dit needs a checkpoint_dir unless skip_weight_loading=True."
        )

    with torch.device("meta"):
        dit = WaypointDiT(config)
    dit.cast_serving_dtypes()
    dit.to_empty(device=device)
    # MUST follow to_empty, which un-aliases the shared cond_proj (CONTRACTS 6.1).
    dit.retie_cond_proj()
    _assert_cond_proj_tied(dit, config)

    if skip_weight_loading:
        _initialize_structurally(dit)
        return dit.eval()

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Waypoint checkpoint directory not found: {checkpoint_dir}")

    _assert_expected_layout(dit)
    fused_shards = _attach_shard_loaders(dit, config)
    params = dict(dit.named_parameters())

    unexpected: list[str] = []
    conflicts: list[str] = []
    # (target, shard_id) -> the checkpoint key that claimed it. This is the
    # per-shard tally: load_weights_into's returned set holds target names only,
    # so q, k and v all collapse to one entry and a k_proj missing from every
    # layer would satisfy `set(params) - loaded` (PARAM_TREE S11d). Recorded here
    # rather than in _SliceShardLoader because the remapper is the one place that
    # sees both the original key (for the error message) and the resolved target.
    arrivals: dict[tuple[str, str | int | None], str] = {}
    # T4's arbitrated collision: {target: (rank, winning checkpoint key)}.
    bias_in_claims: dict[str, tuple[int, str]] = {}

    def remap(name: str) -> str | None:
        mapped = remap_checkpoint_key(name)
        if mapped is None:
            return None  # T9/T10/T12: expected, silent, not an unexpected key.
        target, shard_id = _apply_stacked(mapped, WAYPOINT_STACKED_PARAMS)
        if target not in params:
            unexpected.append(name)
            return None

        # T4 is the one collision that is resolved rather than refused: three
        # spellings legitimately share ``cond_head.bias_in`` and the reference
        # picks between them (mlp > attn, canonical > both). Rank, not arrival
        # order — see _BIAS_IN_SPELLINGS.
        rank = _bias_in_rank(name)
        if rank is not None:
            held = bias_in_claims.get(target)
            if held is not None:
                if rank < held[0]:
                    return None  # a higher-precedence spelling already claimed it
                if rank == held[0]:
                    # The same spelling twice, i.e. a genuinely duplicated key.
                    conflicts.append(f"{held[1]} and {name} -> {target}")
            bias_in_claims[target] = (rank, name)
            arrivals[(target, shard_id)] = name
            return mapped

        claimed_by = arrivals.get((target, shard_id))
        if claimed_by is not None:
            # Two distinct checkpoint keys resolving to one slot. The reference
            # resolves this with setdefault (first wins); a streaming loader would
            # instead let the last one win, non-deterministically across shard
            # order, so it is refused rather than arbitrated.
            slot = target if shard_id is None else f"{target}[{shard_id}]"
            conflicts.append(f"{claimed_by} and {name} -> {slot}")

        # The same "one slot, two writers" failure across the fused/unfused
        # spellings, which the (target, shard_id) key above cannot see: a
        # pre-fused ``qkv_proj``/``mlp.fc1`` tensor claims (target, None) and a
        # split shard claims (target, "q"), and those never collide. Both then
        # write, and the surviving parameter is a mix of the two decided by
        # shard-iteration order — Q and K off the fused blob, V off ``v_proj``,
        # with nothing raised. Either spelling alone is fine; both together are
        # a checkpoint whose intent cannot be inferred, so it is refused.
        if target in fused_shards:
            rival_slots = (
                [(target, s) for s in fused_shards[target]]
                if shard_id is None
                else [(target, None)]
            )
            for slot_key in rival_slots:
                rival = arrivals.get(slot_key)
                if rival is not None:
                    conflicts.append(
                        f"{rival} and {name} -> {target} (a pre-fused tensor and a "
                        "split shard of the same fused parameter)"
                    )

        arrivals[(target, shard_id)] = name
        return mapped

    tie_check = _CondProjTieCheck() if verify_cond_proj_tie else None
    shards = _adapt_checkpoint_stream(
        iter_safetensors_shards(checkpoint_dir, device=device), config, tie_check
    )
    # load_weights_into directly, not load_hf_weights: the wrapper's skip
    # predicate runs before the remapper and would drop keys outside the
    # unexpected-key accounting below.
    loaded = load_weights_into(
        dit, shards, stacked_params=WAYPOINT_STACKED_PARAMS, name_remapper=remap
    )

    missing = sorted(set(params) - loaded)
    missing_shards = sorted(
        f"{name}[{shard_id}]"
        for name, shard_ids in fused_shards.items()
        # A pre-fused tensor satisfies every shard of its target at once. Safe
        # to short-circuit on now: `remap` refuses a file that carries both
        # spellings, so reaching here with (name, None) present means the
        # pre-fused tensor was the *only* writer.
        if (name, None) not in arrivals
        for shard_id in shard_ids
        if (name, shard_id) not in arrivals
    )
    if unexpected or missing or missing_shards or conflicts:
        raise RuntimeError(
            f"Waypoint DiT checkpoint mismatch at {checkpoint_dir}: "
            f"{len(unexpected)} unexpected checkpoint keys {unexpected[:5]}, "
            f"{len(missing)} unloaded parameters {missing[:5]}, "
            f"{len(missing_shards)} unloaded fused shards {missing_shards[:5]}, "
            f"{len(conflicts)} slots claimed twice {conflicts[:3]} — refusing to "
            "serve a partially loaded transformer."
        )
    if tie_check is not None and tie_check.divergent:
        raise RuntimeError(
            f"Waypoint DiT checkpoint at {checkpoint_dir} stores {len(tie_check.divergent)} "
            f"divergent cond_proj copies {tie_check.divergent[:5]}. The port keeps block "
            f"{COND_PROJ_SOURCE_BLOCK}'s and drops the other "
            f"{config.n_layers - 1} sets, which matches the reference (whose last write "
            "wins, i.e. block 23) only while all copies agree. Pass "
            "verify_cond_proj_tie=False to load block "
            f"{COND_PROJ_SOURCE_BLOCK}'s copy anyway."
        )
    return dit.eval()
