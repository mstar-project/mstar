
import json
import logging
import os
import re
from collections.abc import Container, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

if TYPE_CHECKING:
    from mstar.engine.state.contract import Layout

# safetensors is imported where it is used: mstar.model.base imports this
# module, and safetensors is not a core dependency.

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
    from safetensors.torch import safe_open

    from mstar.model.loader.quantization import refuse_unserved_quantization

    refuse_unserved_quantization(safetensors_file)
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

    # straight to the node's device: safetensors' "cuda" is the current GPU, not cuda:N
    with safe_open(safetensors_file, framework="pt", device=str(device)) as f:
        for k in f.keys():
            for i, (_mod, _prefix, key_map) in enumerate(module_key_maps):
                if k in key_map:
                    state_dicts[i][key_map[k]] = f.get_tensor(k)
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
    from safetensors.torch import safe_open

    from mstar.model.loader.quantization import refuse_unserved_quantization

    repo_dir = Path(repo_dir)
    refuse_unserved_quantization(repo_dir)
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

    # load in tensors to get {mod_key}
    for shard_file, keys_in_shard in shard_to_hf_keys.items():
        shard_path = str(repo_dir / shard_file)
        keys_set = set(keys_in_shard)
        # straight to the node's device: safetensors' "cuda" is the current GPU, not cuda:N
        with safe_open(shard_path, framework="pt", device=str(device)) as f:
            for k in f.keys():
                if k not in keys_set:
                    continue
                hf_key_to_tensor[k] = f.get_tensor(k)

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


class ByteLevelDetokenizer:
    """Byte-faithful per-token detokenization for byte-level BPE tokenizers
    (GPT-2 / Whisper / Qwen).

    Streaming models emit one token per step to the client, so a
    multi-byte UTF-8 character split across tokens must be emitted as raw
    bytes and reassembled client-side — per-token ``tokenizer.decode``
    would yield U+FFFD for each partial character.

    Special tokens are dropped, like ``skip_special_tokens``, unless
    ``keep_special``: then the joined bytes of any id sequence decode to what
    ``tokenizer.decode(ids)`` gives.
    """

    def __init__(self, tokenizer, keep_special: bool = False):
        self.tokenizer = tokenizer
        special = frozenset(tokenizer.all_special_ids)
        # the ids ``to_bytes`` drops, and the ones it writes out as text (``keep_special``)
        self.dropped_ids = frozenset() if keep_special else special
        self.kept_special_ids = special if keep_special else frozenset()
        # GPT-2's bytes_to_unicode: printable unicode char <-> raw byte.
        printable = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("\xa1"), ord("\xac") + 1))
            + list(range(ord("\xae"), ord("\xff") + 1))
        )
        chars = printable[:]
        n = 0
        for b in range(256):
            if b not in printable:
                printable.append(b)
                chars.append(256 + n)
                n += 1
        self.byte_decoder = {chr(c): b for b, c in zip(printable, chars, strict=True)}
        # Added tokens this tokenizer's decode writes out as they are, not
        # through the byte map: transformers 4.x slow tokenizers do this for
        # non-special added tokens, so "«é»" would otherwise become U+FFFD.
        self.verbatim = {}
        for i, added in getattr(tokenizer, "added_tokens_decoder", {}).items():
            text = str(added)
            if i in self.dropped_ids or self._byte_map(text) == text.encode("utf-8"):
                continue
            try:
                decoded = tokenizer.decode([i])
            except (KeyError, TypeError, ValueError):  # 5.x slow decode raises on "<中>"
                continue
            if decoded == text:
                self.verbatim[i] = text.encode("utf-8")
        self._token_bytes: dict[int, bytes] = {}

    def token_bytes(self, token_id: int) -> bytes:
        """``to_bytes([token_id])``, memoized: a decode step's chunk is one token."""
        data = self._token_bytes.get(token_id)
        if data is None:
            data = self._token_bytes[token_id] = self.to_bytes([token_id])
        return data

    def _byte_map(self, token: str) -> bytes:
        try:
            return bytes(self.byte_decoder[c] for c in token)
        except KeyError:
            # an added token outside the byte alphabet ("<中>") is its own
            # UTF-8, as in the tokenizers library's ByteLevel decoder
            return token.encode("utf-8")

    def to_bytes(self, token_ids: list[int]) -> bytes:
        raw = bytearray()
        ids = [i for i in token_ids if i not in self.dropped_ids]
        for i, token in zip(ids, self.tokenizer.convert_ids_to_tokens(ids), strict=True):
            if token is None:  # an id past the vocabulary, which decode skips too
                continue
            verbatim = self.verbatim.get(i)
            if verbatim is not None:
                raw.extend(verbatim)
                continue
            raw.extend(self._byte_map(token))
        return bytes(raw)

    @staticmethod
    def is_byte_level(tokenizer) -> bool:
        """Whether ``to_bytes`` reproduces this tokenizer's ``decode``: a
        ByteLevel decoder (fast) or a GPT-2 byte map (slow), and no
        ``clean_up_tokenization_spaces`` rewrite."""
        if getattr(tokenizer, "clean_up_tokenization_spaces", False):
            return False
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            return type(backend.decoder).__name__ == "ByteLevel"
        return isinstance(getattr(tokenizer, "byte_decoder", None), dict)


class TextChunkDetokenizer:
    """One decode step's text chunk for a model that streams a token per step.

    With a byte-level tokenizer the chunk is the tokens' raw bytes, special
    tokens kept, so a character split across steps reassembles in the API
    server's incremental decoder and the joined reply equals
    ``tokenizer.decode`` of all its ids. Any other tokenizer keeps the
    per-step ``decode``.
    """

    __slots__ = ("tokenizer", "kept_special_ids", "_bytes", "_decoded")

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # the ids a chunk writes out as text (a step's ``decode`` keeps special tokens too)
        self.kept_special_ids = frozenset(tokenizer.all_special_ids)
        self._bytes = (
            ByteLevelDetokenizer(tokenizer, keep_special=True)
            if ByteLevelDetokenizer.is_byte_level(tokenizer) else None
        )
        self._decoded: dict[int, bytes] = {}

    def to_bytes(self, token_ids: list[int]) -> bytes:
        if self._bytes is None:
            return self.tokenizer.decode(token_ids).encode("utf-8")
        return self._bytes.to_bytes(token_ids)

    def token_bytes(self, token_id: int) -> bytes:
        """``to_bytes([token_id])``, memoized."""
        if self._bytes is not None:
            return self._bytes.token_bytes(token_id)
        data = self._decoded.get(token_id)
        if data is None:
            data = self._decoded[token_id] = self.to_bytes([token_id])
        return data


class ByteIdDetokenizer:
    """Token ids that are bytes, as a reduced model's byte tokenizer
    (``tokenizer_mode: byte``) makes them: an id is its low byte."""

    kept_special_ids: frozenset[int] = frozenset()

    @staticmethod
    def to_bytes(token_ids: list[int]) -> bytes:
        return bytes(t & 0xFF for t in token_ids)

    @staticmethod
    def token_bytes(token_id: int) -> bytes:
        return bytes((token_id & 0xFF,))


class SentencePieceDetokenizer:
    """Per-token detokenization for SentencePiece-style vocabularies (``▁``
    marks a word start, ``<0xNN>`` pieces are byte fallback), read from an HF
    ``tokenizer.json``. Same contract as :class:`ByteLevelDetokenizer`: bytes
    per token, so a character split across byte pieces reassembles client-side.
    """

    kept_special_ids: frozenset[int] = frozenset()

    def __init__(self, tokenizer_json: dict, extra_skip_ids: set[int] = frozenset()):
        vocab = tokenizer_json["model"]["vocab"]
        if isinstance(vocab, list):  # Unigram: [piece, score] pairs
            vocab = {piece: idx for idx, (piece, _) in enumerate(vocab)}
        vocab = dict(vocab)
        for added in tokenizer_json.get("added_tokens", []):
            vocab[added["content"]] = added["id"]
        self.pieces = {idx: piece for piece, idx in vocab.items()}
        self.skip_ids = {a["id"] for a in tokenizer_json.get("added_tokens", []) if a["special"]} | set(extra_skip_ids)

    def token_bytes(self, token_id: int) -> bytes:
        piece = self.pieces.get(token_id)
        if piece is None or token_id in self.skip_ids:
            return b""
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            return bytes([int(piece[3:5], 16)])
        return piece.replace("▁", " ").encode("utf-8")

    def to_bytes(self, token_ids: list[int]) -> bytes:
        return b"".join(self.token_bytes(i) for i in token_ids)


class TextOutput(NamedTuple):
    """How a model's text output, token ids, becomes bytes: ``Model.text_output``
    declares it, and ``Model.postprocess`` and ``Model.postprocess_inline``
    derive their text paths from it."""

    # every kind has ``kept_special_ids``: the special-token ids its bytes write out as text
    detokenizer: ByteLevelDetokenizer | TextChunkDetokenizer | SentencePieceDetokenizer | ByteIdDetokenizer
    # ids that end a reply, dropped unless the request sets ``ignore_eos``
    stop_token_ids: Container[int] = frozenset()
    # the text that ends the model's reasoning, None for a model that does not
    # reason: a reply's ``structured_outputs`` grammar starts after it on a turn
    # whose prompt leaves the reasoning open (`grammar.after_reasoning`)
    reasoning_end: str | None = None


class ByteLevelTextOutput:
    """``Model.text_output`` for a model over a byte-level BPE tokenizer: each
    token's raw bytes with special tokens dropped, ``config.eos_token_ids``
    ending a reply. Built on first use, as the host's ``tokenizer`` may load
    lazily; a reduced model's byte ids (``_tokenizer_mode`` of ``"byte"``)
    never touch it."""

    @cached_property
    def _text_output(self) -> TextOutput:
        detokenizer = ByteIdDetokenizer() if self._tokenizer_mode == "byte" else ByteLevelDetokenizer(self.tokenizer)
        return TextOutput(detokenizer, frozenset(self.config.eos_token_ids))

    def text_output(self) -> TextOutput:
        return self._text_output


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
    switch_only: bool = False,
) -> dict:
    """The sampling defaults ``checkpoint_dir/generation_config.json`` declares.

    Keys come back under mstar names (``max_new_tokens`` -> ``max_output_tokens``).
    ``prefix`` selects one stage's keys in a multi-stage file (``"talker_"`` reads
    ``talker_temperature``) and ``stage`` prefixes the returned names
    (``"code_predictor_"``). ``do_sample: false`` is greedy: temperature 0.
    ``switch_only`` reads ``do_sample`` alone, for a file that is the backbone's
    chat preset rather than the model's own recipe.
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
    for hf_key, knob in ({} if switch_only else _KNOBS).items():
        value = config.get(prefix + hf_key)
        if value is not None:
            out[stage + knob] = value
    # HF spells a stage's switch both ways: ``talker_do_sample``, ``subtalker_dosample``
    if False in (config.get(prefix + "do_sample"), config.get(prefix + "dosample")):
        out[stage + "temperature"] = 0.0
    return out


def process_weights_after_loading(root: torch.nn.Module, device: torch.device) -> None:
    """Finalize kernel layouts across a freshly loaded module tree: any submodule exposing
    ``process_weights_after_loading(device)`` gets it called (the attention layers build
    their absorbed w_kc/w_vc here).
    """
    for module in root.modules():
        hook = getattr(module, "process_weights_after_loading", None)
        if callable(hook):
            hook(device)


def resolve_checkpoint(model_path_hf: str | None, cache_dir: str | None = None) -> str | None:
    """A local checkpoint directory as given, a Hub id as its snapshot (fetched if not cached),
    None for no path."""
    if not model_path_hf:
        return None
    if Path(model_path_hf).exists():
        return str(model_path_hf)
    from mstar.utils.hf_snapshot import resolve_snapshot_dir

    return str(resolve_snapshot_dir(model_path_hf, cache_dir=cache_dir))


def read_fp8_block_quant_config(checkpoint_dir: str):
    """The ``Fp8BlockQuantConfig`` in ``checkpoint_dir/config.json``, or None: no file, none
    declared, or unreadable (the weights then load as they are, bf16)."""
    from mstar.utils.fp8_block import Fp8BlockQuantConfig

    path = Path(checkpoint_dir) / "config.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        logger.warning("could not read %s: %s", path, e)
        return None
    return Fp8BlockQuantConfig.from_hf_config_dict(raw.get("quantization_config"))


def load_snapshot_tokenizer(model_path_hf: str, cache_dir: str | None = None):
    """The checkpoint's tokenizer, through ``AutoTokenizer`` or, where transformers 4.x
    cannot build the backend class a transformers-5 ``tokenizer_config.json`` names,
    ``fast_tokenizer_from_snapshot``."""
    from transformers import AutoTokenizer

    from mstar.utils.hf_snapshot import resolve_snapshot_dir

    source = str(resolve_snapshot_dir(model_path_hf, cache_dir=cache_dir))
    try:
        return AutoTokenizer.from_pretrained(source, cache_dir=cache_dir)
    except ValueError:
        return fast_tokenizer_from_snapshot(source)


def fast_tokenizer_from_snapshot(source: str):
    """A fast tokenizer from ``tokenizer.json``, which is version-independent, with the
    snapshot's chat template and the special tokens ``tokenizer_config.json`` declares."""
    from transformers import PreTrainedTokenizerFast

    snap = Path(source)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(snap / "tokenizer.json"))
    template = snap / "chat_template.jinja"
    if template.is_file():
        tokenizer.chat_template = template.read_text()
    # built from tokenizer.json alone, the tokenizer has no special tokens, so the
    # detokenizer would print <|user|> and the other turn markers into replies
    config = snap / "tokenizer_config.json"
    if config.is_file():
        cfg = json.loads(config.read_text())
        special = {k: cfg[k] for k in ("eos_token", "pad_token") if cfg.get(k)}
        tokenizer.add_special_tokens(
            {**special, "additional_special_tokens": list(cfg.get("extra_special_tokens") or [])})
    return tokenizer


# The 25 European languages of NVIDIA's Granary-trained speech checkpoints
# (canary-1b-v2, parakeet-tdt-0.6b-v3).
GRANARY_LANGUAGES = (
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it",
    "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
)


def base_language(code: str) -> str:
    """``en-US``, ``en_us`` or ``EN`` -> ``en``."""
    if not isinstance(code, str):
        raise ValueError(f"a language is a code such as 'en'; got {code!r}")
    return code.replace("_", "-").split("-")[0].lower()


def capture_ladder(key: str, value) -> list[int]:
    """A capture ladder (token buckets or batch sizes) from model_kwargs: a 0, negative or
    repeated entry would go into the capture build, and an empty one would quietly mean the
    default."""
    if (not isinstance(value, (list, tuple)) or not value or len(set(value)) != len(value)
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in value)):
        raise ValueError(f"{key} must be a non-empty list of distinct positive integers, got {value!r}")
    return list(value)


def warn_retired_ladder_env(name: str, setting: str) -> None:
    """Warn that the environment variable ``name``, a capture ladder the serving config's
    ``prefill_token_buckets`` replaced, is set but no longer read; ``setting`` is the
    serving config entry that sets it."""
    if name in os.environ:
        logger.warning("%s is no longer read; set the ladder in the serving config: %s", name, setting)


def check_loaded(module: torch.nn.Module, loaded: set[str], what: str) -> None:
    """Raise unless ``loaded`` names every parameter of ``module``: one built on
    ``meta`` and filled by a loader would otherwise serve uninitialized memory."""
    missing = sorted({name for name, _ in module.named_parameters()} - loaded)
    if missing:
        raise KeyError(f"{what}: checkpoint is missing {len(missing)} parameters, e.g. {missing[:4]}")


def kv_pages(tokens: int, page_size: int, requests: int) -> int:
    """Pages for ``requests`` sequences of ``tokens`` each, plus the sink page
    every KV cache keeps out of circulation."""
    return 1 + requests * -(-tokens // page_size)


def check_kv_pools_fit(
    model, config: dict, config_path: str, log: logging.Logger,
    request_layouts: "Mapping[str, Mapping[str, Layout]] | None" = None, loop_caps: Mapping[str, int] | None = None,
) -> None:
    """Refuse a ``max_concurrent_requests`` a resource the model builds cannot hold, as its
    spec's `capacity_problem` answers: the first problem raises.

    A request keeps its pages from prefill to its last token. Past a pool, a
    prefill batch is refused. With host pages (``cpu_offload_pages``) the worker
    offloads decoding requests to admit it, which costs far more than queueing
    would. Without them it retries until requests time out. So the cap's requests
    at full length must fit each cache next to its sink page; the conductor
    queues the rest. Warnings go to the model's ``log``. ``request_layouts``
    gives the `Layout` per array and label of the request the deployment is
    sized for, with its loops at ``loop_caps``, and a pool it doesn't name
    holds one stream of its ``max_seq_len``.
    """
    from mstar.distributed.base import node_group_axis_sizes
    from mstar.engine.resources import KVReqConfig
    from mstar.engine.state.adapters import kv_layouts
    from mstar.model.base import serving_resources

    cap = config.get("max_concurrent_requests")
    if cap is None:
        log.warning(
            "%s sets no max_concurrent_requests: past the requests the KV caches "
            "hold, new ones aren't queued. A cache with cpu_offload_pages offloads "
            "decoding requests to admit them, and one without stalls until requests "
            "time out", config_path,
        )
        return
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise ValueError(f"{config_path}: max_concurrent_requests must be a positive int; got {cap!r}")
    try:
        # as a server builds them: the deployment's overrides, and what the pools derive
        specs = serving_resources(model, config)
    except ValueError:
        return  # a `resources:` block the validation's resource checks report
    # each pool's config as the conductor opens the request, from the rows the data worker sends
    requests: dict[str, KVReqConfig] = {}
    for key, by_label in (kv_layouts(model.declare_state(), request_layouts, {}) if request_layouts else {}).items():
        uncapped = {loop for layout in by_label.values() for loop, _ in layout.growth} - (loop_caps or {}).keys()
        if uncapped:
            raise ValueError(f"{config_path}: request_layouts[{key!r}] grows loops {sorted(uncapped)}, which loop_caps "
                             "doesn't cap")
        rows = {label: layout.row() for label, layout in by_label.items()}
        requests[key] = KVReqConfig(needed_labels=list(rows))
        requests[key].apply_conductor_config(kv_layout=rows, loop_caps=loop_caps)
    node_groups = config.get("node_groups") or ()
    for spec in specs:
        axis_sizes = [
            node_group_axis_sizes(group) for group in node_groups if spec.nodes & set(group.get("node_names", ()))
        ]
        problem = spec.capacity_problem(cap, requests.get(spec.resource_key), axis_sizes)
        if problem is not None:
            raise ValueError(f"{config_path}: {problem}")
