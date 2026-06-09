"""Weight loader for the Ling-2.0 thinker (TP-aware via load_hf_weights).

Step 3e refactor: instead of a custom per-shard loop, we now stream
the checkpoint through mminf's :func:`load_hf_weights` machinery.
Per-rank slicing happens inside the parameter-attached
``weight_loader`` callbacks of the TP-aware modules — same pattern as
Qwen3-Omni's loader at
``mminf/model/qwen3_omni/qwen3_omni_model.py:1242-1334``.

## What this loader handles

1. **Outer prefix strip**: ``model.X.Y`` → ``X.Y`` (the wrapper is
   ``BailingMM2NativeForConditionalGeneration.model``).
2. **Per-layer renames**: ``model.layers.{i}.attention.{query_key_value,
   dense,q_norm,k_norm}.weight`` → ``layers.{i}.self_attn.{qkv_proj,
   dense,q_norm,k_norm}.weight``; ``mlp.{gate,image_gate,audio_gate}.weight``
   → ``mlp.{...}.gate.weight`` (extra nesting for the router's inner
   nn.Linear); ``mlp.shared_experts.*`` → ``mlp.shared_expert.*``.
3. **Packed QKV split**: ``attention.query_key_value.weight`` is one
   `(Q+2K)*D x H` tensor in the checkpoint, but :class:`QKVParallelLinear`
   wants three calls (one each with shard_id ``"q"``/``"k"``/``"v"``).
   Done by ``_split_packed_qkv`` which intercepts QKV keys and emits
   three synthetic stream entries.
4. **Per-expert fusion**: 256 separate ``experts.N.gate_proj.weight``
   keys per layer → packed ``experts.gate_up_proj`` tensor.
   ``_remap_thinker_keys`` rewrites them to
   ``experts.{gate,up,down}_proj.__expertN__.weight`` so
   :class:`StackedParamRule.source_suffix` matching works; the per-rule
   ``shard_id="gate:N"`` / ``"up:N"`` / ``"down:N"`` strings drive
   mminf's per-rank ``_gate_up_weight_loader`` / ``_down_proj_weight_loader``
   to write into the right expert slot per rank.

Per-rank TP slicing happens automatically — every TP-aware module
(``QKVParallelLinear``, ``RowParallelLinear``, ``ParallelGatedMLP``,
``LingMoeBlock.experts``) attaches its own ``weight_loader`` callback
that knows its ``tp_rank``/``tp_size`` and slices the loaded tensor
accordingly.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

import torch

from mminf.model.loader.base import StackedParamRule, load_hf_weights
from mminf.model.loader.iterators import iter_safetensors_shards
from mminf.model.ming_omni_flash.components.model import LingMoeModel

logger = logging.getLogger(__name__)


# Outermost ckpt prefix — strip before everything else.
_CKPT_THINKER_PREFIX = "model."


# Per-key static rename rules (only the substring matters; expert
# fusion + QKV split are handled separately).
_SUBSTRING_RENAMES: list[tuple[str, str]] = [
    # Embed / norm / lm_head (after the outer model. strip).
    # `lm_head.weight` lands directly.
    # `model.word_embeddings.weight` → `embed_tokens.weight`
    # `model.norm.weight` → `norm.weight`
    # The substring matcher below handles `model.` → `` only when it's a prefix.

    # Attention rename (per-layer, applies to any layer index).
    # query_key_value isn't actually emitted past _split_packed_qkv (the
    # split produces synthetic q_proj/k_proj/v_proj keys instead), but
    # the rule's harmless and documents intent.
    ("attention.query_key_value", "self_attn.qkv_proj"),
    # Synthetic q/k/v keys emitted by _split_packed_qkv. Their StackedParamRule
    # routes them into the fused self_attn.qkv_proj via shard_id "q"/"k"/"v".
    ("attention.q_proj", "self_attn.q_proj"),
    ("attention.k_proj", "self_attn.k_proj"),
    ("attention.v_proj", "self_attn.v_proj"),
    ("attention.dense", "self_attn.dense"),
    ("attention.q_norm", "self_attn.q_norm"),
    ("attention.k_norm", "self_attn.k_norm"),
    # Router renames (per-layer, applies to gate / image_gate / audio_gate).
    # mlp.gate.weight → mlp.gate.gate.weight (nested through the router's nn.Linear)
    ("mlp.gate.weight", "mlp.gate.gate.weight"),
    ("mlp.image_gate.weight", "mlp.image_gate.gate.weight"),
    ("mlp.audio_gate.weight", "mlp.audio_gate.gate.weight"),
    # Shared expert (singular in mminf vs plural in ckpt).
    ("mlp.shared_experts.", "mlp.shared_expert."),
]


_EXPERT_KEY_RE = re.compile(
    r"^(.*)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def _strip_outer_model_prefix(key: str) -> str | None:
    """Strip the outermost ``model.`` (the wrapper). Returns None for
    keys we don't expect (audio.*, vision.*, etc. — these aren't part
    of the thinker text-only path)."""
    if not key.startswith(_CKPT_THINKER_PREFIX):
        return None
    stripped = key[len(_CKPT_THINKER_PREFIX):]
    # After the strip the LLM is rooted at "model.layers..." / "model.norm..." /
    # "model.word_embeddings..." (the inner HF wrapper). lm_head.weight is
    # directly here without an extra "model." prefix.
    return stripped


def _apply_substring_renames(key: str) -> str:
    for src, dst in _SUBSTRING_RENAMES:
        if src in key:
            key = key.replace(src, dst)
    # Embed / norm: strip the inner ``model.`` prefix where applicable.
    # `model.word_embeddings.weight` → `embed_tokens.weight`
    if key.startswith("model.word_embeddings"):
        key = key.replace("model.word_embeddings", "embed_tokens", 1)
    # `model.norm.weight` → `norm.weight`
    elif key.startswith("model.norm"):
        key = key.replace("model.norm", "norm", 1)
    # `model.layers.X` → `layers.X`
    elif key.startswith("model.layers."):
        key = key[len("model."):]
    return key


def _remap_thinker_keys(key: str) -> str | None:
    """Full name remapping for thinker keys.

    Returns the post-rename key, or None to drop the key entirely.
    """
    stripped = _strip_outer_model_prefix(key)
    if stripped is None:
        return None  # not a thinker key (audio.*, vision.*, etc.)

    # Per-expert fusion marker: rewrite so the StackedParamRule's
    # suffix-match picks them up.
    m = _EXPERT_KEY_RE.match(stripped)
    if m:
        prefix, expert_idx, proj = m.groups()
        # prefix looks like "model.layers.5"; strip the inner "model."
        if prefix.startswith("model.layers."):
            prefix = prefix[len("model."):]
        return f"{prefix}.mlp.experts.{proj}.__expert{expert_idx}__.weight"

    renamed = _apply_substring_renames(stripped)
    return renamed


def _build_thinker_stacked_params(num_experts: int) -> list[StackedParamRule]:
    """Build the per-expert + dense-MLP rules.

    Per-expert rules MUST come first because the dense-MLP ``.gate_proj``
    / ``.up_proj`` / ``.down_proj`` suffixes would also match the
    remapped MoE keys otherwise — :func:`_apply_stacked` returns on first
    match.
    """
    rules: list[StackedParamRule] = []
    for i in range(num_experts):
        rules.append(StackedParamRule(
            target_suffix=".experts.gate_up_proj",
            source_suffix=f".experts.gate_proj.__expert{i}__.weight",
            shard_id=f"gate:{i}",
        ))
        rules.append(StackedParamRule(
            target_suffix=".experts.gate_up_proj",
            source_suffix=f".experts.up_proj.__expert{i}__.weight",
            shard_id=f"up:{i}",
        ))
        rules.append(StackedParamRule(
            target_suffix=".experts.down_proj",
            source_suffix=f".experts.down_proj.__expert{i}__.weight",
            shard_id=f"down:{i}",
        ))
    # Dense layer-0 MLP fusion (ParallelGatedMLP holds gate_up_proj).
    rules.append(StackedParamRule(".gate_up_proj", ".gate_proj", 0))
    rules.append(StackedParamRule(".gate_up_proj", ".up_proj", 1))
    # Attention QKV fusion: synthetic q/k/v keys from _split_packed_qkv
    # route into the fused self_attn.qkv_proj.weight via shard_id strings.
    # QKVParallelLinear's weight_loader does per-rank head-axis slicing.
    rules.append(StackedParamRule(".qkv_proj", ".q_proj", "q"))
    rules.append(StackedParamRule(".qkv_proj", ".k_proj", "k"))
    rules.append(StackedParamRule(".qkv_proj", ".v_proj", "v"))
    return rules


def _split_packed_qkv(
    weights: Iterable[tuple[str, torch.Tensor]],
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Stream-transform: split each ``attention.query_key_value.weight``
    into 3 synthetic ``self_attn.{q,k,v}_proj.weight`` entries.

    ``QKVParallelLinear`` doesn't have a single ``query_key_value``
    weight_loader; it dispatches via shard_id ``"q"``/``"k"``/``"v"``
    on three separate keys. We emit those keys here so the stacked rules
    (``.qkv_proj``, ``.q_proj`` / ``.k_proj`` / ``.v_proj``) route them
    into the right slots.

    Packing in ckpt: weight is `(num_heads + 2*num_kv_heads)*head_dim x hidden`,
    rows ordered [Q rows, K rows, V rows].
    """
    q_size = num_attention_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv_total = q_size + 2 * kv_size

    pattern = re.compile(r"^(.*attention\.)query_key_value\.weight$")

    for raw_key, tensor in weights:
        m = pattern.match(raw_key)
        if m is None:
            yield raw_key, tensor
            continue
        if tensor.shape[0] != qkv_total:
            raise ValueError(
                f"{raw_key}: expected first dim {qkv_total} "
                f"(num_heads={num_attention_heads}, num_kv_heads={num_kv_heads},"
                f" head_dim={head_dim}); got {tensor.shape[0]}"
            )
        prefix = m.group(1)
        q_slice = tensor[0:q_size, :]
        k_slice = tensor[q_size:q_size + kv_size, :]
        v_slice = tensor[q_size + kv_size:qkv_total, :]
        yield f"{prefix}q_proj.weight", q_slice
        yield f"{prefix}k_proj.weight", k_slice
        yield f"{prefix}v_proj.weight", v_slice


def load_thinker_weights(
    model: LingMoeModel,
    local_dir: str,
    device: str = "cpu",
    strict: bool = True,
) -> None:
    """Stream the checkpoint into the TP-aware LingMoeModel.

    Sequencing:
      1. Iterate sharded safetensors via mminf's `iter_safetensors_shards`.
      2. Pre-split packed QKV keys into synthetic q/k/v keys.
      3. Pass through `load_hf_weights` with our `name_remapper` +
         per-expert StackedParamRules + dense-MLP rules. mminf's
         parameter-attached `weight_loader`s do per-rank slicing.

    Args:
        model: LingMoeModel constructed with the right comm_group; param
            tensors must already be on `device`.
        local_dir: path to the Ming snapshot.
        device: where to materialise loaded tensors (`"cpu"` /
            `"cuda"` / `"cuda:N"`).
        strict: if True, raise when any LingMoeModel parameter received
            no checkpoint tensor.
    """
    llm_cfg = None
    # Reach into the model to recover num_heads / num_kv_heads / head_dim
    # for the QKV split — we don't have the config here directly.
    first_attn = model.layers[0].self_attn
    num_heads = first_attn.total_num_heads
    num_kv = first_attn.total_num_kv_heads
    head_dim = first_attn.head_dim

    # Look up via the safetensors index: each layer's experts.{N} keys
    # might land in a different shard. iter_safetensors_shards yields
    # all matching keys across shards. We pre-strip to thinker-only keys
    # via the prefix arg so vision / audio shards (only present in 100B
    # model? not sure) don't get streamed.
    raw_weights = iter_safetensors_shards(
        local_dir, device=device, prefix=_CKPT_THINKER_PREFIX,
    )

    # Wrap with the QKV split + name remapper. load_hf_weights handles
    # the rest (stacked rules, weight_loader dispatch).
    split_weights = _split_packed_qkv(
        raw_weights,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv,
        head_dim=head_dim,
    )

    stacked = _build_thinker_stacked_params(
        num_experts=model.layers[-1].mlp.num_experts if model.layers[-1].is_moe
        else 0,  # if there's no MoE layer (e.g. tiny test model), skip
    )

    loaded = load_hf_weights(
        model, split_weights,
        stacked_params=stacked,
        name_remapper=_remap_thinker_keys,
    )

    if strict:
        target_keys = set(model.state_dict().keys())
        # Filter expert keys: each fused param gets loaded multiple times
        # (one per expert / shard); load_hf_weights returns the param
        # name once per first hit. That's fine — but it means we can't
        # check "every param was touched at least once". Instead, check
        # the simpler thing: every param that ISN'T a fused expert tensor
        # was touched.
        missing = []
        for k in target_keys:
            if k.endswith(".experts.gate_up_proj") or k.endswith(".experts.down_proj"):
                # Fused; load_hf_weights's `loaded` set has the target
                # name once per shard rule that matched, so if any one
                # rule matched we're OK. Just check it's in `loaded`.
                if k not in loaded:
                    missing.append(k)
            elif k not in loaded:
                missing.append(k)
        if missing:
            raise KeyError(
                f"Missing thinker parameters after load (strict=True). "
                f"Sample missing keys: {sorted(missing)[:10]} "
                f"(total {len(missing)})"
            )

    logger.info(
        "Loaded %d unique target params into LingMoeModel(num_hidden_layers=%d) "
        "from %s (rank %d/%d).",
        len(loaded), model.num_hidden_layers, local_dir,
        model.comm_group.rank, model.comm_group.world_size,
    )


# ===========================================================================
# Vision / audio encoder + projector loaders (step 4b)
# ===========================================================================
#
# These modules aren't TP-aware (run on a single rank in the typical
# topology — vision_encoder + audio_encoder colocate on rank 0 per
# configs/ming_flash_omni.yaml). Loading is a plain prefix-strip +
# load_state_dict path; no per-rank slicing or stacked-rule fusion.
#
# Released ckpt's relevant top-level prefixes:
#   vision.*              -> MingVisionEncoder (Qwen3MoeVisionTransformer)
#   audio.*               -> MingAudioEncoder  (Whisper)
#   linear_proj.*         -> MingVisionProjector (nn.Sequential under .proj)
#   linear_proj_audio.*   -> MingAudioProjector  (nn.Sequential under .proj)


def _load_prefixed_state_dict(
    module: torch.nn.Module,
    local_dir: str,
    prefix: str,
    inner_prefix: str = "",
    device: str = "cpu",
    strict: bool = True,
    allow_missing: set[str] | None = None,
) -> set[str]:
    """Common path for the encoder/projector loaders.

    Streams keys matching ``prefix`` from the safetensors shards, strips
    that outer prefix, optionally prepends ``inner_prefix``, then runs
    ``module.load_state_dict``.

    Args:
        module:        target nn.Module.
        local_dir:     snapshot dir with model.safetensors{,.index.json}.
        prefix:        outer ckpt prefix to filter shards by + strip.
        inner_prefix:  prepended to the stripped key before lookup. Used
                       by the projector loaders so ckpt's ``0.weight``
                       hits ``proj.0.weight`` on our module.
        device:        target device for loaded tensors.
        strict:        if True, raise on any key mismatch (missing or
                       unexpected) other than entries in ``allow_missing``.
        allow_missing: parameter / buffer names in the module's
                       state_dict that the ckpt is allowed to skip.
                       (E.g. Whisper's ``positional_embedding`` buffer is
                       regenerated locally — ckpt drops it.)

    Returns the set of keys actually loaded (post-rename).
    """
    raw_weights = iter_safetensors_shards(local_dir, device=device, prefix=prefix)
    state = {}
    for key, tensor in raw_weights:
        if not key.startswith(prefix):
            # Defensive: iter_safetensors_shards should already filter.
            continue
        sub_key = key[len(prefix):]
        if inner_prefix:
            sub_key = f"{inner_prefix}{sub_key}"
        state[sub_key] = tensor

    if not state:
        raise KeyError(
            f"No checkpoint keys matched prefix {prefix!r} under {local_dir}. "
            f"Snapshot may be a thinker-only / talker-only variant."
        )

    missing, unexpected = module.load_state_dict(state, strict=False)
    allow_missing = allow_missing or set()
    real_missing = [m for m in missing if m not in allow_missing]
    if strict and (real_missing or unexpected):
        raise KeyError(
            f"State-dict mismatch loading prefix {prefix!r}: "
            f"missing={real_missing[:10]} (total {len(real_missing)}); "
            f"unexpected={list(unexpected)[:10]} (total {len(unexpected)})."
        )

    logger.info(
        "Loaded %d params (prefix=%r) from %s (missing=%d, unexpected=%d).",
        len(state), prefix, local_dir, len(missing), len(unexpected),
    )
    return set(state.keys())


def load_vision_encoder_weights(
    encoder: torch.nn.Module,
    local_dir: str,
    device: str = "cpu",
    strict: bool = True,
) -> set[str]:
    """Load ``vision.*`` weights from the snapshot into a Ming vision encoder.

    Works with the module returned by ``build_vision_encoder``
    (``Qwen3MoeVisionTransformer`` from the staged Ming source). Key
    names after the ``vision.`` strip already match the module's
    state_dict — no further remapping needed.
    """
    return _load_prefixed_state_dict(
        encoder, local_dir, prefix="vision.", device=device, strict=strict,
    )


def load_audio_encoder_weights(
    encoder: torch.nn.Module,
    local_dir: str,
    device: str = "cpu",
    strict: bool = True,
) -> set[str]:
    """Load ``audio.*`` weights from the snapshot into MingAudioEncoder.

    The released ckpt ships its own (trained) ``positional_embedding``
    that overrides the sinusoidal init in :func:`_sinusoids` — load
    via ``load_state_dict``'s buffer support (no special-casing needed).
    """
    return _load_prefixed_state_dict(
        encoder, local_dir, prefix="audio.", device=device, strict=strict,
    )


def load_vision_projector_weights(
    projector: torch.nn.Module,
    local_dir: str,
    device: str = "cpu",
    strict: bool = True,
) -> set[str]:
    """Load ``linear_proj.*`` into MingVisionProjector.

    Ckpt key shape is ``linear_proj.{0,2}.{weight,bias}``; our module's
    state_dict shape is ``proj.{0,2}.{weight,bias}``, so we prepend
    ``proj.`` after stripping ``linear_proj.``.
    """
    return _load_prefixed_state_dict(
        projector, local_dir, prefix="linear_proj.",
        inner_prefix="proj.", device=device, strict=strict,
    )


def load_audio_projector_weights(
    projector: torch.nn.Module,
    local_dir: str,
    device: str = "cpu",
    strict: bool = True,
) -> set[str]:
    """Load ``linear_proj_audio.*`` into MingAudioProjector.

    Ckpt key shape is ``linear_proj_audio.{0,3}.{weight,bias}``; module
    has them under ``proj.{0,3}.{weight,bias}``.
    """
    return _load_prefixed_state_dict(
        projector, local_dir, prefix="linear_proj_audio.",
        inner_prefix="proj.", device=device, strict=strict,
    )
