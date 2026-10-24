
import json
import logging
import re
from collections.abc import Container
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open

logger = logging.getLogger(__name__)


def load_weights(
    state_dict: dict[str, Any],
    module: torch.nn.Module,
    prefix: str = None,
    enforce_missing_keys: bool = True,
):
    if prefix is not None:
        if not prefix.endswith("."):
            prefix += "."
        state_dict = {
            k.removeprefix(prefix): v for k, v in state_dict.items() \
                if k.startswith(prefix)
        }

    missing_keys, _ = module.load_state_dict(state_dict, strict=False)
    if enforce_missing_keys and missing_keys:
        raise KeyError(f"Missing keys when loading state_dict with prefix {prefix!r}: {missing_keys}")


@dataclass
class ModuleAndPrefix:
    module: torch.nn.Module
    prefix: str = None
    enforce_missing_keys: bool = True


def load_weights_from_file(
    safetensors_file: str,
    modules: list[ModuleAndPrefix],
    device: str = "cpu",
):
    # Precompute expected keys for each module
    module_key_maps = []
    for mod in modules:
        prefix = mod.prefix or ""
        if prefix and not prefix.endswith("."):
            prefix += "."

        key_map = {
            prefix + k: k
            for k in mod.module.state_dict().keys()
        }
        module_key_maps.append((mod, prefix, key_map))

    # Temporary per-module state dicts
    state_dicts = [dict() for _ in modules]

    # safetensors can't take cuda:0 etc
    st_device = "cuda" if str(device).startswith("cuda") else device

    with safe_open(safetensors_file, framework="pt", device=st_device) as f:
        for k in f.keys():
            for i, (_mod, _prefix, key_map) in enumerate(module_key_maps):
                if k in key_map:
                    tensor = f.get_tensor(k)
                    if device != st_device:
                        tensor = tensor.to(device, non_blocking=True)
                    state_dicts[i][key_map[k]] = tensor
                    break

    # Load modules
    for (mod, state_dict) in zip(modules, state_dicts, strict=True):
        missing_keys, _ = mod.module.load_state_dict(state_dict, strict=False, assign=True)
        if mod.enforce_missing_keys and missing_keys:
            raise KeyError(
                f"Missing keys when loading state_dict with prefix {mod.prefix!r}: {missing_keys}"
            )


@dataclass
class Operation:
    operation: str
    dim: int = 0

@dataclass
class WeightConverter:
    source_patterns: list[str]
    target_patterns: str
    operations: list[Operation]


@dataclass
class KeysAndConverter:
    keys: set[str] = field(default_factory=set)
    converter: WeightConverter | None = None

    def append_key(self, key):
        self.keys.add(key)

    def __post_init__(self):
        self.keys = set(self.keys)


def _apply_key_pattern(keys: list[str], conv: list[WeightConverter] | None=None):
    if conv is None:
        return {
            k: KeysAndConverter([k]) for k in keys
        }

    compiled_target_and_conv = []
    for c in conv:
        for pattern in c.source_patterns:
            compiled_target_and_conv.append((
                re.compile(pattern), c.target_patterns,
                c
            ))


    mod_key_to_hf_keys: dict[str, KeysAndConverter] = {}
    for key in keys:
        found = False
        for comp, target, co in compiled_target_and_conv:
            nk = comp.sub(target, key)
            if nk != key:
                found = True
                if nk not in mod_key_to_hf_keys:
                    mod_key_to_hf_keys[nk] = KeysAndConverter(converter=co)
                mod_key_to_hf_keys[nk].append_key(key)
                break
            key = nk
        if not found:
            mod_key_to_hf_keys[key] = KeysAndConverter(keys=[key])

    return mod_key_to_hf_keys


def _apply_operations(
    key_to_tensor: dict[str, torch.Tensor],
    converter: WeightConverter | None = None,
) -> torch.Tensor:
    """
    Apply WeightConverter operations to a group of keys.

    Returns the final tensor to assign to the target key.
    """
    if converter is None:
        # trivial case (no conversion)
        assert len(key_to_tensor) == 1
        return key_to_tensor[next(iter(key_to_tensor.keys()))]

    # --- 1. group tensors by source pattern ---
    pattern_to_tensors: list[list[tuple[int, torch.Tensor]]] = []

    for pattern in converter.source_patterns:
        regex = re.compile(f".*({pattern}).*")

        matched: list[tuple[int, torch.Tensor]] = []

        # Build a regex that captures the integer at the position of the "*"
        # in the source pattern.  This ensures we extract the EXPERT index
        # (e.g. ``42`` in ``...layers.5.mlp.experts.42.gate_proj.weight``),
        # not the layer index.  Without this anchor, ``re.search(r"\.(\d+)\.")``
        # returns the first integer in the key (the layer index), and all 128
        # experts in a given layer end up sharing the same index — they then
        # stack in arbitrary hash order, so expert N's weights land in some
        # other slot and the router selects the wrong expert at inference,
        # producing gibberish output.
        if "*" not in pattern:
            raise ValueError(
                f"Source pattern {pattern!r} has no '*' wildcard; cannot "
                "extract per-expert index for MergeModulelist."
            )
        # Replace the literal "*" with a numeric capture group; escape the rest.
        index_regex = re.compile(
            ".*" + re.escape(pattern).replace(r"\*", r"(\d+)") + ".*"
        )

        for k, tensor in key_to_tensor.items():
            m = regex.match(k)
            if not m:
                continue

            idx_match = index_regex.match(k)
            if idx_match is None:
                raise ValueError(
                    f"Could not extract expert index from key: {k} "
                    f"(pattern: {pattern})"
                )

            idx = int(idx_match.group(1))
            matched.append((idx, tensor))

        # sort by expert index
        matched.sort(key=lambda x: x[0])

        # Diagnostic: log how many keys this pattern matched, and the
        # range of indices.  Useful for confirming that the converter is
        # actually being applied to the checkpoint (rather than the
        # identity load path) and that experts are in 0..N-1 order.
        if matched:
            indices = [idx for idx, _ in matched]
            logger.debug(
                "WeightConverter pattern %r matched %d keys, "
                "indices range [%d..%d], expected sequential 0..N-1: %s",
                pattern, len(matched), indices[0], indices[-1],
                "OK" if indices == list(range(len(indices))) else "GAPS/UNORDERED",
            )

        # keep only tensors
        pattern_to_tensors.append([t for _, t in matched])

    # --- 2. apply operations in order ---
    current = pattern_to_tensors

    for op in converter.operations:
        if op.operation == "MergeModulelist":
            # stack each pattern independently
            # input: list[list[tensor]] → output: list[tensor]
            current = [
                torch.stack(tensors, dim=op.dim)
                for tensors in current
            ]

        elif op.operation == "Concatenate":
            # concatenate across patterns
            # input: list[tensor] → output: tensor
            current = torch.cat(current, dim=op.dim)

        else:
            raise ValueError(f"Unknown operation: {op.operation}")

    if isinstance(current, list):
        assert len(current) == 1
        current = current[0]
    return current


def load_weights_from_hf_shards(
    repo_dir: str | Path,
    modules: list[ModuleAndPrefix],
    device: str = "cpu",
    conv: list[WeightConverter] | None=None
):
    """Load weights from a HuggingFace safetensors checkpoint.

    Reads model.safetensors.index.json to find which shard each key lives in,
    then loads from each shard file. Checkpoints that ship a single unsharded
    ``model.safetensors`` (no index — e.g. whisper-large-v3) are handled by
    treating that one file as the whole weight map.
    """
    repo_dir = Path(repo_dir)
    index_path = repo_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        single = repo_dir / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(
                f"No model.safetensors.index.json or model.safetensors in {repo_dir}"
            )
        with safe_open(str(single), framework="pt", device="cpu") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}

    mod_key_to_hf_keys = _apply_key_pattern(weight_map.keys(), conv)
    mod_key_to_idx = {}

    # Precompute expected keys for each module
    module_keys = []
    all_module_keys = set()
    all_hf_keys = set()
    for i, mod in enumerate(modules):
        prefix = mod.prefix or ""
        if prefix and not prefix.endswith("."):
            prefix += "."
        for k in mod.module.state_dict().keys():
            if prefix + k in mod_key_to_hf_keys.keys():
                mod_key_to_hf_keys[k] = mod_key_to_hf_keys[prefix + k]
                all_hf_keys.update(mod_key_to_hf_keys[k].keys)
                if prefix:
                    del mod_key_to_hf_keys[prefix + k]
            mod_key_to_idx[k] = i
        module_keys.append(mod.module.state_dict().keys())
        all_module_keys.update(module_keys[-1])

    mod_key_to_hf_keys = {
        k: v for k, v in mod_key_to_hf_keys.items() if k in all_module_keys
    }

    shard_to_hf_keys: dict[str, list[str]] = {}
    for k in all_hf_keys:
        shard_to_hf_keys.setdefault(weight_map[k], []).append(k)

    hf_key_to_tensor = {}
    st_device = "cuda" if str(device).startswith("cuda") else device

    # load in tensors to get {mod_key}
    for shard_file, keys_in_shard in shard_to_hf_keys.items():
        shard_path = str(repo_dir / shard_file)
        keys_set = set(keys_in_shard)
        with safe_open(shard_path, framework="pt", device=st_device) as f:
            for k in f.keys():
                if k not in keys_set:
                    continue
                tensor = f.get_tensor(k)
                if device != st_device:
                    tensor = tensor.to(device, non_blocking=True)
                hf_key_to_tensor[k] = tensor

    # Temporary per-module state dicts
    state_dicts: list[dict[str, torch.Tensor]] = [dict() for _ in modules]
    for mod_key, hf_keys_and_op in mod_key_to_hf_keys.items():
        tensor = _apply_operations({
                k: hf_key_to_tensor[k] for k in hf_keys_and_op.keys
            }, hf_keys_and_op.converter
        )
        state_dicts[mod_key_to_idx[mod_key]][mod_key] = tensor

    # Load modules
    for mod, state_dict in zip(modules, state_dicts, strict=True):
        missing_keys, _ = mod.module.load_state_dict(state_dict, strict=False, assign=True)
        if mod.enforce_missing_keys and missing_keys:
            raise KeyError(f"Missing keys when loading state_dict with prefix {mod.prefix!r}: {missing_keys}")


def _byte_level_alphabet() -> dict[str, int]:
    """GPT-2's ``bytes_to_unicode``, inverted: the printable character a
    byte-level BPE vocab entry uses for each byte -> that byte."""
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    cs = list(bs)
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


class TokenBytes:
    """Each token id's raw bytes under a byte-level BPE tokenizer.

    ``tokenizer.decode([id])`` turns a token that ends inside a multi-byte
    character into U+FFFD, so decoding token by token garbles any character
    split across tokens. Raw bytes joined in order are the reply's UTF-8, and
    the API server decodes a request's text chunks only after joining them.
    Added tokens (``<think>``, ``<|im_end|>``) are their text, as in
    ``decode(skip_special_tokens=False)``. Filled lazily per id.
    """

    def __init__(self, tokenizer):
        decoder = getattr(getattr(tokenizer, "backend_tokenizer", None), "decoder", None)
        if type(decoder).__name__ != "ByteLevel":
            raise ValueError(f"TokenBytes needs a ByteLevel decoder; got {type(decoder).__name__}")
        self._tokenizer = tokenizer
        self._alphabet = _byte_level_alphabet()
        self._added = {i: t.content.encode("utf-8") for i, t in tokenizer.added_tokens_decoder.items()}
        self._cache: dict[int, bytes] = {}

    def __call__(self, token_id: int) -> bytes:
        out = self._cache.get(token_id)
        if out is None:
            out = self._added.get(token_id)
            if out is None:
                piece = self._tokenizer.convert_ids_to_tokens(token_id)
                # an id past the tokenizer's vocab (the model's padded rows)
                # has no piece; `decode` gives it no text either
                out = b"" if piece is None else bytes(self._alphabet[c] for c in piece)
            self._cache[token_id] = out
        return out

    def join(self, ids) -> bytes:
        return b"".join(map(self, ids))


class ByteLevelDetokenizer:
    """Byte-faithful per-token detokenization for byte-level BPE tokenizers
    (GPT-2 / Whisper / Qwen).

    Streaming models emit one token per step to the client, so a
    multi-byte UTF-8 character split across tokens must be emitted as raw
    bytes and reassembled client-side — per-token ``tokenizer.decode``
    would yield U+FFFD for each partial character.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_ids = set(tokenizer.all_special_ids)
        self.byte_decoder = _byte_level_alphabet()

    def to_bytes(self, token_ids: list[int]) -> bytes:
        raw = bytearray()
        tokens = self.tokenizer.convert_ids_to_tokens(
            [i for i in token_ids if i not in self.special_ids]
        )
        for token in tokens:
            raw.extend(self.byte_decoder[c] for c in token)
        return bytes(raw)


def drop_stop_tokens(
    output: torch.Tensor, stop_token_ids: Container[int], request_kwargs: dict | None,
) -> list[int]:
    """Drop the stop tokens that ended a text reply, unless the request sets ``ignore_eos``.

    A stop token ends the turn, not the reply: a client sends the reply back as
    an assistant message, whose template closes it. Under ``ignore_eos`` it ends
    nothing and stays, one chunk per token.
    """
    ids = output.reshape(-1).tolist()
    if (request_kwargs or {}).get("ignore_eos"):
        return ids
    return [i for i in ids if i not in stop_token_ids]


# generation_config.json key -> mstar knob
_KNOBS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
    "max_new_tokens": "max_output_tokens",
}


def load_generation_defaults(
    checkpoint_dir: str | Path | None, prefix: str = "", stage: str = "",
) -> dict:
    """The sampling defaults ``checkpoint_dir/generation_config.json`` declares.

    Keys come back under mstar names (``max_new_tokens`` -> ``max_output_tokens``).
    ``prefix`` selects one stage's keys in a multi-stage file (``"talker_"`` reads
    ``talker_temperature``) and ``stage`` prefixes the returned names
    (``"code_predictor_"``). ``do_sample: false`` is greedy: temperature 0.
    Returns ``{}`` when there is no file, so the caller's own defaults apply.
    """
    if checkpoint_dir is None:
        return {}
    path = Path(checkpoint_dir) / "generation_config.json"
    if not path.is_file():
        return {}
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        logger.warning("Unreadable %s; using the model's own defaults", path)
        return {}
    out = {}
    for hf_key, knob in _KNOBS.items():
        value = config.get(prefix + hf_key)
        if value is not None:
            out[stage + knob] = value
    # HF spells a stage's switch both ways: ``talker_do_sample``, ``subtalker_dosample``
    if False in (config.get(prefix + "do_sample"), config.get(prefix + "dosample")):
        out[stage + "temperature"] = 0.0
    return out
