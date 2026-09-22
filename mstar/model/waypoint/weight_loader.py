"""Weight loading for the native Waypoint DiT (mstar loader pattern).

``build_waypoint_dit`` builds ``WaypointDiT`` on meta, casts it to the
serving dtypes, materializes it on the device, re-ties ``cond_proj`` (must
follow ``to_empty``, which un-aliases it), then streams the safetensors
shards through ``load_weights_into`` via ``remap_checkpoint_key``. The
fused q/k/v and MLP-fusion projections route through
``WAYPOINT_STACKED_PARAMS`` instead, since a name remapper can't express a
fan-in. Loading is a completeness contract: any unexpected, missing, or
duplicate-claimed key raises.
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


# Which block's cond_proj is physical; must match WaypointDiT.retie_cond_proj's
# hardcoded blocks[0], or loading fails. _assert_cond_proj_tied checks this.
COND_PROJ_SOURCE_BLOCK = 0

# Slot ids for the two port-side fusions. Order defines the layout.
QKV_SHARD_IDS: tuple[str, ...] = ("q", "k", "v")
CTRL_FC1_SHARD_IDS: tuple[str, ...] = ("x", "c")

# Fused-shard routing; leading dots matter so ".v_proj" doesn't also match
# inside "qkv_proj".
WAYPOINT_STACKED_PARAMS: list[StackedParamRule] = [
    StackedParamRule(".qkv_proj", ".q_proj", "q"),
    StackedParamRule(".qkv_proj", ".k_proj", "k"),
    StackedParamRule(".qkv_proj", ".v_proj", "v"),
    StackedParamRule(".mlp.fc1", ".fc1_x", "x"),
    StackedParamRule(".mlp.fc1", ".fc1_c", "c"),
]

# The port's block prefix (``components/dit.py`` collapses WorldModel/WorldDiT).
MODEL_BLOCK_PREFIX = "blocks."

# ``transformer.`` prefix is optional: accepts both the reference's two-level
# spelling and the collapsed one.
_BLOCK_RE = re.compile(r"^(?:transformer\.)?blocks\.(\d+)\.(.+)$")
# Legacy half-heads. j is range(3) on both sides; a j >= 3 is malformed and is
# left unmapped so it surfaces as unexpected.
_LEGACY_COND_PROJ_RE = re.compile(r"^(attn|mlp)_cond_head\.cond_proj\.(\d+)\.weight$")
_COND_PROJ_RE = re.compile(r"^cond_head\.cond_proj\.(\d+)\.weight$")

# Allowlist, not a dit_mlp.* wildcard, so an unrecognized dit_mlp.* key
# surfaces as unexpected. expert_*/router don't exist under moe=False but
# are listed since the reference renames them.
_DIT_MLP_LEAVES: tuple[str, ...] = (
    "fc1.weight",
    "fc2.weight",
    "expert_in",
    "expert_out",
    "router.weight",
)

# CFG dropout is training-time only; the port has no tensor for it, so this
# key is dropped explicitly rather than left to surface as unexpected.
_DROPPED_TOP_LEVEL_KEYS = frozenset({"ctrl_cfg.null_emb"})

# Substring filter, unconditional; note the plural ("cond_heads").
_COND_HEADS_FRAGMENT = ".cond_heads."

# Half the CondHead slots come from each legacy half-head.
_COND_PROJ_PER_HEAD = CondHead.n_cond // 2

# Precedence order for the three spellings sharing cond_head.bias_in, lowest
# first: mlp beats attn, canonical beats both (matches the reference's
# pop/setdefault outcome). Rank rather than last-write-wins because shards can
# arrive in any order; arbitration lives in build_waypoint_dit.
_BIAS_IN_SPELLINGS: tuple[str, ...] = (
    "attn_cond_head.bias_in",
    "mlp_cond_head.bias_in",
    "cond_head.bias_in",
)

# Reference init for the value-residual scalar (no-checkpoint path only).
_V_LAMB_INIT = 0.5
# Weight init std for the no-checkpoint structural smoke-test path.
_STRUCTURAL_INIT_STD = 0.02


# --------------------------------------------------------------------------
# Name remapping
# --------------------------------------------------------------------------


def _remap_block_suffix(suffix: str, layer_idx: int) -> str | None:
    """Map one per-block checkpoint suffix to its parameter suffix, or ``None``
    to drop it. ``suffix`` excludes the ``blocks.{i}.`` prefix."""
    # Both legacy spellings map to the one target; rank in build_waypoint_dit
    # decides which of the three writes.
    if suffix in ("attn_cond_head.bias_in", "mlp_cond_head.bias_in"):
        suffix = "cond_head.bias_in"

    # Identity index map for the attn head, +3 for the mlp head. Slots 0-2
    # drive attention, 3-5 drive MLP; swapping them is silent and catastrophic.
    legacy = _LEGACY_COND_PROJ_RE.match(suffix)
    if legacy is not None:
        head, j = legacy.group(1), int(legacy.group(2))
        if j < _COND_PROJ_PER_HEAD:
            slot = j if head == "attn" else j + _COND_PROJ_PER_HEAD
            suffix = f"cond_head.cond_proj.{slot}.weight"

    for leaf in _DIT_MLP_LEAVES:
        if suffix == "dit_mlp." + leaf:
            suffix = "mlp." + leaf
            break

    # Guarded on fc2 alone, separately from the fc1 fusion's both-halves guard.
    if suffix == "ctrl_mlpfusion.fc2.weight":
        suffix = "ctrl_mlpfusion.mlp.fc2.weight"

    # Runs last, on the post-remap name, so it catches both the legacy
    # half-head spellings and an already-canonical one.
    if _COND_PROJ_RE.match(suffix) is not None and layer_idx != COND_PROJ_SOURCE_BLOCK:
        return None

    # fc1_x/fc1_c and q/k/v_proj are left alone: fan-ins a remapper can't
    # express. WAYPOINT_STACKED_PARAMS routes them instead.
    return suffix


def _is_unconditionally_dropped(name: str) -> bool:
    """Drops that depend on nothing but the key name.

    One predicate, two call sites, so the shard adapter's pre-filter and the
    remapper can't drift apart. The per-block cond_proj drop is not here:
    those keys must survive to reach ``_CondProjTieCheck`` and are dropped
    later, in the remapper.
    """
    return _COND_HEADS_FRAGMENT in name or name in _DROPPED_TOP_LEVEL_KEYS


def _bias_in_rank(name: str) -> int | None:
    """Precedence rank of a per-block ``bias_in`` key, or ``None`` if not one.
    Higher wins; see ``_BIAS_IN_SPELLINGS``."""
    block = _BLOCK_RE.match(name)
    if block is None:
        return None
    suffix = block.group(2)
    if suffix not in _BIAS_IN_SPELLINGS:
        return None
    return _BIAS_IN_SPELLINGS.index(suffix)


def remap_checkpoint_key(name: str) -> str | None:
    """Map one Waypoint checkpoint key to the native parameter path, or
    ``None`` for a key that's intentionally dropped.

    Pure function of the key — no model, no config. Not injective in one
    place: all three ``bias_in`` spellings map to
    ``blocks.{i}.cond_head.bias_in``, and ``build_waypoint_dit`` picks the
    winner since that needs the whole key set.
    """
    if _is_unconditionally_dropped(name):
        return None

    block = _BLOCK_RE.match(name)
    if block is None:
        # Top-level keys (patchify, unpatchify, denoise_step_emb, ctrl_emb,
        # out_norm) pass through unchanged.
        return name

    layer_idx, suffix = int(block.group(1)), block.group(2)
    mapped = _remap_block_suffix(suffix, layer_idx)
    if mapped is None:
        return None
    return f"{MODEL_BLOCK_PREFIX}{layer_idx}.{mapped}"


# --------------------------------------------------------------------------
# Tensor transforms and shape validation over the shard stream
# --------------------------------------------------------------------------


def _unpatchify_weight(tensor: torch.Tensor, config: WaypointConfig, key: str) -> torch.Tensor:
    """``[D, C, ph, pw]`` conv kernel -> ``[C*ph*pw, D]`` Linear weight.

    ``permute(1, 2, 3, 0)`` then ``reshape``: the output feature axis must be
    ordered ``(c, ph, pw)`` with pw fastest to match ``WaypointDiT.forward``'s
    ``view(B, N, Hp, Wp, C, ph, pw)``. Dropping the permute, or swapping
    ph/pw, keeps the shape but silently reprojects every output sub-pixel.
    ``reshape``, not ``view``: the permuted tensor isn't contiguous.
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
    # own ndim == 4 guard makes this idempotent the same way.
    expected = (config.channels * ph * pw, config.d_model)
    if tuple(tensor.shape) != expected:
        raise RuntimeError(
            f"{key} has shape {tuple(tensor.shape)}; expected a 4-D "
            f"[{config.d_model}, {config.channels}, {ph}, {pw}] conv kernel or an "
            f"already-permuted {list(expected)} Linear weight."
        )
    return tensor


def _unpatchify_bias(tensor: torch.Tensor, config: WaypointConfig, key: str) -> torch.Tensor:
    """One learned bias per latent channel, repeated across the patch.

    ``[C] -> [C,1,1] -> expand(-1, ph, pw) -> reshape(-1)``, matching the
    weight's row ordering. ``repeat(ph*pw)`` produces the same shape but puts
    the bias on the wrong sub-pixel — a silent numerical bug, not a crash.
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

    Pins ``n_kv_heads`` explicitly: a wrong value silently reshapes attention
    without erroring downstream, and a shard-shape-mismatch error wouldn't
    say which config field to check.
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
    """Confirms the checkpoint's per-block ``cond_proj`` copies agree before
    all but ``COND_PROJ_SOURCE_BLOCK``'s are dropped.

    Nothing else enforces this, so a fine-tune that broke the tie would
    silently make the port and reference serve different video. Compares
    matrices in full, independent of shard order; ``verify_cond_proj_tie=False``
    skips it.
    """

    def __init__(self) -> None:
        self._reference: dict[int, tuple[int, torch.Tensor]] = {}
        self.divergent: list[str] = []

    def observe(self, key: str, tensor: torch.Tensor) -> None:
        slot_info = _cond_proj_slot(key)
        if slot_info is None or tensor.ndim != 2:
            return
        block_idx, slot = slot_info
        # fp32 on CPU: uniform, lossless from bf16, and off the serving device.
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
    """Apply the reshape transforms and validate the transcribed config
    facts, in one streaming pass over the shards.

    Lives here rather than in the remapper because mstar has no reshape
    hook: ``name_remapper`` sees names only. Drops run before validation,
    since a dropped key's shape (e.g. ``…cond_heads.0.k_proj.weight``) is not
    this model's business.
    """
    q_rows = config.n_heads * config.d_head
    kv_rows = config.n_kv_heads * config.d_head

    for key, tensor in weights:
        if _is_unconditionally_dropped(key):  # before anything reads a shape
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
# Fused-parameter shard loaders
# --------------------------------------------------------------------------


class _SliceShardLoader:
    """``param.weight_loader`` for a port-side fused parameter.

    Copies one checkpoint shard into its slice of the fused tensor. Neither
    fusion target is a ``FusedColumnLinear`` (both are plain ``nn.Linear``),
    so neither carries a loader and ``default_weight_loader`` won't accept a
    shard id.
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
            # Checkpoint already in fused form; only reachable for
            # ctrl_mlpfusion.mlp.fc1, which the reference also accepts fused.
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

    Must run after ``to_empty(device)``, which reallocates the Parameter
    objects and drops attached attributes — the same reason
    ``FusedColumnLinear`` re-attaches its loaders from ``_apply``.
    """
    q_rows = config.n_heads * config.d_head
    kv_rows = config.n_kv_heads * config.d_head
    d_model = config.d_model

    fused: dict[str, tuple[str, ...]] = {}
    for name, param in dit.named_parameters():
        if name.endswith(".attn.qkv_proj.weight"):
            # cat([q, k, v], dim=0), matching components/attention.py's split.
            # GQA makes shards unequal: a q/k swap raises, but a k/v swap
            # loads cleanly and produces meaningless attention.
            dim, layout = 0, {
                "q": (0, q_rows),
                "k": (q_rows, kv_rows),
                "v": (q_rows + kv_rows, kv_rows),
            }
            expected_shape = (q_rows + 2 * kv_rows, d_model)
        elif name.endswith(".ctrl_mlpfusion.mlp.fc1.weight"):
            # cat([fc1_x, fc1_c], dim=1), x first — layers.MLPFusion splits it
            # back with chunk(2, dim=1); swapping the order swaps which half
            # gets token vs. controller conditioning.
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

    ``named_parameters()`` deduplicates aliased Parameters; ``state_dict()``
    does not, so the gap between the two counts is exactly the tied
    ``cond_proj``.
    """
    params = dict(dit.named_parameters())
    return (
        len(params),
        sum(p.numel() for p in params.values()),
        sum(t.numel() for t in dit.state_dict().values()),
    )


def _assert_cond_proj_tied(dit: WaypointDiT, config: WaypointConfig) -> None:
    """Fail if ``retie_cond_proj()`` did not take.

    An un-tied model is otherwise silent: numerically correct, just heavier
    and missing loaded parameters. The tensor count catches "never tied";
    the numel gap between ``state_dict()`` and ``named_parameters()`` catches
    a partial tie.
    """
    tied = [name for name, _ in dit.named_parameters() if ".cond_head.cond_proj." in name]
    if len(tied) != CondHead.n_cond:
        raise RuntimeError(
            f"cond_proj is not tied: named_parameters() reports {len(tied)} cond_proj "
            f"tensors, expected {CondHead.n_cond} (one physical set, aliased by all "
            f"{config.n_layers} blocks). to_empty(device) un-ties them and "
            "retie_cond_proj() must be called after it, not before."
        )
    # If COND_PROJ_SOURCE_BLOCK disagrees with which block retie_cond_proj
    # aliases onto, the loader drops the keys the module keeps and vice versa.
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

    ``to_empty`` allocates uninitialized storage (often NaN/Inf), so this
    isn't usable even for a shape smoke test until every parameter is
    written. Not the reference's init — only ``v_lamb`` and the zeroed 1-D
    tensors match it. Iterating ``named_parameters()`` (not ``state_dict()``)
    keeps the tied ``cond_proj`` aliased.
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

    ``checkpoint_dir`` is a local directory the caller has already resolved;
    nothing here downloads. ``skip_weight_loading`` builds a randomly
    initialized structure for shape/plumbing work instead.

    Raises ``RuntimeError`` on any completeness failure: an unexpected,
    missing, or duplicate-claimed key, or divergent ``cond_proj`` copies.
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
    # (target, shard_id) -> claiming checkpoint key. load_weights_into's
    # returned set holds target names only, so q/k/v collapse to one entry;
    # `set(params) - loaded` alone wouldn't catch a missing k_proj.
    arrivals: dict[tuple[str, str | int | None], str] = {}
    # Arbitrated bias_in collision: {target: (rank, winning checkpoint key)}.
    bias_in_claims: dict[str, tuple[int, str]] = {}

    def remap(name: str) -> str | None:
        mapped = remap_checkpoint_key(name)
        if mapped is None:
            return None  # Expected drop, not an unexpected key.
        target, shard_id = _apply_stacked(mapped, WAYPOINT_STACKED_PARAMS)
        if target not in params:
            unexpected.append(name)
            return None

        # The one collision resolved rather than refused: three spellings
        # legitimately share cond_head.bias_in; rank decides the winner, not
        # arrival order (see _BIAS_IN_SPELLINGS).
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

        # Same "two writers, one slot" failure across fused/unfused spellings,
        # invisible to (target, shard_id): a pre-fused tensor claims (target,
        # None) while a split shard claims (target, "q"), so they never
        # collide directly. Either spelling alone is fine; both together are
        # refused.
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
        # A pre-fused tensor satisfies every shard at once; `remap` refuses a
        # file with both spellings, so (name, None) means it was the only
        # writer.
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
