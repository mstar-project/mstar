"""Weight loading for the native Waypoint-1.5-1B DiT (mstar loader pattern).

``build_waypoint_dit`` constructs ``WaypointDiT`` on meta, casts it to the serving
dtypes while still on meta (so ``to_empty`` allocates storage in the final
dtypes), moves it to the device, **re-ties ``cond_proj``**, then streams the
safetensors shards through ``load_weights_into``.

``retie_cond_proj()`` must follow ``to_empty``; skipping it leaves 23 blocks of
``cond_proj`` this loader never fills.

The key map. Thirteen transforms sit between the checkpoint's 393 keys and this
module's 174 parameters:

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

T0's source spelling is the reference's two-level ``WorldModel``/``WorldDiT``
split, which ``components/dit.py`` collapses. Both spellings are accepted, as
are the canonical post-transform spellings the reference's own
``pop``/``setdefault`` transforms tolerate; which the shipped file uses could not
be established statically.

Three things mstar's machinery does not give you. The fusions need a fan-in and
``name_remapper`` is ``str -> str|None``, so both go through ``StackedParamRule``
with a ``_SliceShardLoader`` attached after ``to_empty``. ``load_weights_into``
returns *target* names and q/k/v share one target, so the remapper tallies
``(target, shard_id)`` pairs. T1/T2 have no reshape hook, so they ride an adapter
over the shard iterator, which is also where the transcribed config facts
(``n_kv_heads``, ``patch``) are checked against the shapes on disk.

Completeness is a hard contract: a key that reaches no parameter, a parameter no
key reached, a fused shard that never arrived, or two keys writing one slot all
raise. Explicitly dropped keys (T9/T10/T12) are expected and silent.

The unconditional drops (T10/T12) run **before** the shape validation, because a
``.cond_heads.`` key ending in ``.k_proj.weight`` is a T12 drop and not a GQA
violation. T9's per-block ``cond_proj`` drop is not in that pre-filter: those
keys are what ``_CondProjTieCheck`` compares.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from torch import nn

# _apply_stacked is imported rather than reimplemented: the remapper has to
# resolve a key to the same target load_weights_into will.
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


# Which block's cond_proj set is the physical one. Not a knob: it records that
# WaypointDiT.retie_cond_proj hardcodes self.blocks[0], so any other value turns
# the six kept keys into unexpected-key failures and leaves the six real
# parameters unloaded. _assert_cond_proj_tied checks the tree still agrees.
COND_PROJ_SOURCE_BLOCK = 0

# Slot ids for the two port-side fusions. Order defines the layout.
QKV_SHARD_IDS: tuple[str, ...] = ("q", "k", "v")
CTRL_FC1_SHARD_IDS: tuple[str, ...] = ("x", "c")

# Fused-shard routing. The leading dots matter: without them ".v_proj" would
# also match inside "qkv_proj". Not LLAMA_STACKED_PARAMS, whose extra
# gate_proj/up_proj rules would be live substring matchers for parameters
# Waypoint does not have.
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
# Legacy half-heads. j is range(3) on both sides; a j >= 3 is malformed and is
# left unmapped so it surfaces as unexpected.
_LEGACY_COND_PROJ_RE = re.compile(r"^(attn|mlp)_cond_head\.cond_proj\.(\d+)\.weight$")
_COND_PROJ_RE = re.compile(r"^cond_head\.cond_proj\.(\d+)\.weight$")

# T3 is an allowlist, not a dit_mlp.* wildcard, so a future dit_mlp.* key
# surfaces instead of being absorbed. expert_*/router do not exist under
# moe=False; they are listed because the reference renames them.
_DIT_MLP_LEAVES: tuple[str, ...] = (
    "fc1.weight",
    "fc2.weight",
    "expert_in",
    "expert_out",
    "router.weight",
)

# T10. CFG.forward is a training-time dropout with no call site in the
# reference's WorldModel.forward, so the port has no tensor for it. Dropped
# explicitly rather than left unmatched, so unexpected-key accounting keeps
# no hole.
_DROPPED_TOP_LEVEL_KEYS = frozenset({"ctrl_cfg.null_emb"})

# T12. Substring filter, unconditional, note the plural.
_COND_HEADS_FRAGMENT = ".cond_heads."

# Half the CondHead slots come from each legacy half-head.
_COND_PROJ_PER_HEAD = CondHead.n_cond // 2

# T4 precedence over the three spellings that share cond_head.bias_in, lowest
# first, highest wins — the reference's pop/setdefault outcome: mlp beats attn,
# canonical beats both. The attn spelling is a FALLBACK, not a drop; which of the
# two the shipped file carries is unestablished, and dropping it unconditionally
# would leave 24 unloaded bias_in on an attn-only file.
#
# A rank rather than last-write-wins because this loader streams: the three
# spellings can arrive in any shard order and the resident weight must not depend
# on it. Arbitration lives in build_waypoint_dit, which sees every key.
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
    # T4: both legacy spellings map to the one target; rank in build_waypoint_dit
    # decides which of the three writes.
    if suffix in ("attn_cond_head.bias_in", "mlp_cond_head.bias_in"):
        suffix = "cond_head.bias_in"

    # T5/T6: identity index map for the attn head, +3 for the mlp head. Slots
    # 0-2 drive the attention sublayer and 3-5 the MLP sublayer; swapping them
    # is silent and numerically catastrophic.
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

    # T8. Guarded on fc2 alone, separately from T7's both-halves guard.
    if suffix == "ctrl_mlpfusion.fc2.weight":
        suffix = "ctrl_mlpfusion.mlp.fc2.weight"

    # T9. Runs last, on the post-T5/T6 name, so it catches both the legacy
    # half-head spellings and an already-canonical one.
    if _COND_PROJ_RE.match(suffix) is not None and layer_idx != COND_PROJ_SOURCE_BLOCK:
        return None

    # T7 (fc1_x/fc1_c) and T11 (q/k/v_proj) are left alone: fan-ins, which a
    # remapper cannot express. WAYPOINT_STACKED_PARAMS routes them.
    return suffix


def _is_unconditionally_dropped(name: str) -> bool:
    """T12 and T10 — the drops that depend on nothing but the key.

    One predicate, two call sites, so the shard adapter's pre-filter and the
    remapper cannot drift apart. T9 is not here even though it is also a drop:
    those 138 keys are what ``_CondProjTieCheck`` has to see, so they survive the
    stream filter and are dropped later, in the remapper.
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

    Pure function of the key — no model, no config; a key mapped to a name that
    is not a parameter is the caller's problem.

    Not injective in exactly one place: all three T4 ``bias_in`` spellings map to
    ``blocks.{i}.cond_head.bias_in``, and picking a winner needs the whole key
    set, so ``build_waypoint_dit`` settles it. Everywhere else a second key
    resolving to a claimed slot is a hard error.
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
    output sub-pixel. ``reshape``, not ``view`` — the permuted tensor is not
    contiguous.
    """
    ph, pw = config.patch
    if tensor.ndim == 4:
        d_model, channels, k_h, k_w = tensor.shape
        if (k_h, k_w) != (ph, pw):
            raise RuntimeError(
                f"{key} is a {k_h}x{k_w} patch kernel but WaypointConfig.patch is "
                f"{(ph, pw)}. patch was transcribed from the checkpoint's config.yaml, "
                "not read from it; the checkpoint wins."
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
    wrong sub-pixel, which integrates into a slow colour drift over a rollout
    rather than failing.
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

    4-D in both the file and the module, so validated rather than transformed.
    """
    ph, pw = config.patch
    if tensor.ndim != 4:
        raise RuntimeError(f"{key} should be a 4-D conv kernel; got {tuple(tensor.shape)}.")
    d_model, channels, k_h, k_w = tensor.shape
    if (k_h, k_w) != (ph, pw):
        raise RuntimeError(
            f"{key} is a {k_h}x{k_w} patch kernel but WaypointConfig.patch is "
            f"{(ph, pw)}. patch was transcribed rather than read; a wrong value "
            "silently rescales the token grid."
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
            "n_kv_heads was transcribed from the checkpoint's config.yaml rather than "
            "read from it (the reference default is n_heads), and a wrong value "
            "reshapes GQA attention without raising."
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
    them are dropped.

    The port keeps ``COND_PROJ_SOURCE_BLOCK``'s copy; the reference keeps block
    23's, because it loads all 24 into one shared tensor and the last write wins.
    Nothing in the file or in either loader enforces that they match, so a
    fine-tune that broke the tie would silently make the two serve different
    video. Compares the matrices in full; whichever block arrives first for a
    slot becomes that slot's reference, so the result does not depend on shard
    ordering. Retains 6 x ``[2048, 2048]`` fp32 (96 MiB) for the load;
    ``verify_cond_proj_tie=False`` is the escape hatch.
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
    hook: ``name_remapper`` sees names only, and ``weight_loader`` is per-target.

    Drops run before validation: the checks fire on key *suffixes*, so a key
    T12 drops unconditionally can still end in ``.k_proj.weight``
    (``…blocks.0.cond_heads.0.k_proj.weight`` does) and a dropped key's shape is
    not this model's business.
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

    Copies one checkpoint shard into its slice of the fused tensor. Neither
    fusion target is a ``FusedColumnLinear`` — both are plain ``nn.Linear`` — so
    neither carries a loader, and ``default_weight_loader`` asserts
    ``loaded_shard_id is None``. ``ctrl_mlpfusion`` also concatenates along dim 1
    (the input-feature axis), which ``FusedColumnLinear`` does not do.
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
            # for ctrl_mlpfusion.mlp.fc1, whose fused form the reference also
            # accepts; harmless and symmetric for qkv_proj.
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
            # cat([q, k, v], dim=0), matching the split in components/attention.py.
            # GQA makes the shards unequal, so a q/k swap raises but a k/v swap
            # loads cleanly and produces meaningless attention.
            dim, layout = 0, {
                "q": (0, q_rows),
                "k": (q_rows, kv_rows),
                "v": (q_rows + kv_rows, kv_rows),
            }
            expected_shape = (q_rows + 2 * kv_rows, d_model)
        elif name.endswith(".ctrl_mlpfusion.mlp.fc1.weight"):
            # cat([fc1_x, fc1_c], dim=1) — x first. layers.MLPFusion splits it
            # back with chunk(2, dim=1), whose low columns are the token half;
            # the reverse order keeps the [2048, 4096] shape and applies
            # controller conditioning to tokens and vice versa.
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
    the 720P checkpoint this is ``(174, 1_281_958_040, 1_860_771_992)``; counting
    off the checkpoint gives 2,048 more in each, because that carries
    ``ctrl_cfg.null_emb`` ``[1, 1, 2048]``, which the port drops (T10).
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
    ``6 * n_layers``. An un-tied model is otherwise silent: numerically correct,
    0.6B parameters heavier, and visible only as 138 unloaded parameters. The
    count catches "never tied"; the resident/stored numel gap catches a partial
    tie. Both bounds come from ``config``, not the 720P numbers.
    """
    tied = [name for name, _ in dit.named_parameters() if ".cond_head.cond_proj." in name]
    if len(tied) != CondHead.n_cond:
        raise RuntimeError(
            f"cond_proj is not tied: named_parameters() reports {len(tied)} cond_proj "
            f"tensors, expected {CondHead.n_cond} (one physical set, aliased by all "
            f"{config.n_layers} blocks). to_empty(device) un-ties them and "
            "retie_cond_proj() must be called after it, not before."
        )
    # COND_PROJ_SOURCE_BLOCK records which block retie_cond_proj aliases the
    # others onto; if the two disagree, T9 drops the six keys the module keeps
    # and keeps the six it drops.
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
    # state_dict() re-expands and named_parameters() does not.
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
    and Inf bit patterns, so a "no checkpoint" model is unusable even for a shape
    smoke test until something writes every parameter. This is not the
    reference's init; only ``v_lamb`` = 0.5 and the zeroed 1-D tensors match it.

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

    ``config``'s transcribed ``n_kv_heads`` and ``patch`` are validated against
    the shapes in the file. ``checkpoint_dir`` is a local directory the caller
    has already resolved; nothing here downloads. ``skip_weight_loading`` builds
    a randomly initialized structure for shape and plumbing work.

    Raises ``RuntimeError`` on any completeness failure: an unexpected checkpoint
    key, an unloaded parameter, a fused shard that never arrived, two keys
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
    # MUST follow to_empty, which un-aliases the shared cond_proj.
    dit.retie_cond_proj()
    _assert_cond_proj_tied(dit, config)

    if skip_weight_loading:
        _initialize_structurally(dit)
        dit.eval().materialize_runtime_tables(device)
        if config.compile_dit:
            dit.compile_regions()
        return dit

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Waypoint checkpoint directory not found: {checkpoint_dir}")

    _assert_expected_layout(dit)
    fused_shards = _attach_shard_loaders(dit, config)
    params = dict(dit.named_parameters())

    unexpected: list[str] = []
    conflicts: list[str] = []
    # (target, shard_id) -> the checkpoint key that claimed it. load_weights_into's
    # returned set holds target names only, so q, k and v collapse to one entry
    # and a k_proj missing from every layer would still satisfy
    # `set(params) - loaded`.
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

        # T4 is the one collision resolved rather than refused: three spellings
        # legitimately share cond_head.bias_in and the reference picks between
        # them (mlp > attn, canonical > both). Rank, not arrival order.
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
            # Two distinct checkpoint keys resolving to one slot. Refused, not
            # arbitrated: the winner here would be whichever arrived last, which
            # is shard-order dependent.
            slot = target if shard_id is None else f"{target}[{shard_id}]"
            conflicts.append(f"{claimed_by} and {name} -> {slot}")

        # The same "one slot, two writers" failure across the fused/unfused
        # spellings, which (target, shard_id) cannot see: a pre-fused tensor
        # claims (target, None) and a split shard claims (target, "q"), so they
        # never collide, both write, and the survivor is a shard-order-dependent
        # mix. Either spelling alone is fine; both together are refused.
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
        # A pre-fused tensor satisfies every shard of its target at once, and
        # `remap` refuses a file carrying both spellings, so (name, None) here
        # means the pre-fused tensor was the only writer.
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
    dit.eval().materialize_runtime_tables(device)
    if config.compile_dit:
        dit.compile_regions()
    return dit
