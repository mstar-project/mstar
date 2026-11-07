"""The serving config's ``quantization:`` block: which format each declared linear runs in.

Short form, one format for every linear the model declares a candidate::

    quantization: fp8_block

Long form::

    quantization:
      format: fp8_block          # default for the declared candidates; `auto` (the default) serves the
                                 # checkpoint as stored, `bf16` keeps every linear bf16
      layers:                    # ordered overrides, first match wins
        - {match: "model.layers.0-3.**", format: bf16}
        - {match: "re:model\\.layers\\.\\d+\\.mlp\\.experts", format: fp8_block}
      keep_bf16: [lm_head]       # sugar for `{match: ..., format: bf16}` rules, tried after `layers`
      nodes:                     # per-node blocks of the same keys; their rules go before the top level's
        vision_encoder: {format: bf16}

A pattern names module paths under the node's weights root, as ``named_modules()`` prints them: dotted
segments, each an ``fnmatch`` pattern for one segment, an inclusive index range or ``**`` for any number of
segments (``model.layers.0-3.**``), or ``re:<regex>`` matched against the whole path. Parsing checks only the
block's own shape and names; what a pattern selects is the resolver's job (``resolve.resolve_plan``), which
refuses a rule that selects nothing.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mstar.quantization.formats import AUTO, BF16, FORMATS, KV_CACHE_FORMATS

_NODE_KEYS = ("format", "layers", "keep_bf16", "kv_cache")
_TOP_KEYS = (*_NODE_KEYS, "nodes")
_RULE_KEYS = ("match", "format")
REGEX_PREFIX = "re:"


class QuantizationSpecError(ValueError):
    """Every problem found in a ``quantization:`` block, one per line."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass(frozen=True)
class Rule:
    """``format`` for the module paths ``pattern`` matches; ``origin`` says where the rule came from, for
    messages (``layers[2]``, ``nodes.LLM.keep_bf16[0]``)."""

    pattern: str
    format: str
    origin: str = ""

    def matches(self, path: str) -> bool:
        return pattern_matches(path, self.pattern)


@dataclass(frozen=True)
class NodeQuantSpec:
    """One node's effective block: its own rules, then the top level's, first match wins."""

    format: str = AUTO
    rules: tuple[Rule, ...] = ()
    kv_cache: str = AUTO

    @property
    def is_default(self) -> bool:
        return self.format == AUTO and not self.rules and self.kv_cache == AUTO

    def rule_for(self, path: str) -> Rule | None:
        return next((rule for rule in self.rules if rule.matches(path)), None)


@dataclass(frozen=True)
class QuantizationSpec:
    top: NodeQuantSpec = field(default_factory=NodeQuantSpec)
    nodes: Mapping[str, NodeQuantSpec] = field(default_factory=dict)
    # whether the YAML had a `quantization:` block at all
    given: bool = False

    def for_node(self, node: str) -> NodeQuantSpec:
        return self.nodes.get(node, self.top)


_RANGE = re.compile(r"(\d+)-(\d+)")


def _segment_matches(part: str, pat: str) -> bool:
    span = _RANGE.fullmatch(pat)
    if span is not None:
        return part.isdigit() and int(span.group(1)) <= int(part) <= int(span.group(2))
    return fnmatch.fnmatchcase(part, pat)


def _segments_match(parts: list[str], pats: list[str]) -> bool:
    if not pats:
        return not parts
    if pats[0] == "**":
        return any(_segments_match(parts[i:], pats[1:]) for i in range(len(parts) + 1))
    return bool(parts) and _segment_matches(parts[0], pats[0]) and _segments_match(parts[1:], pats[1:])


def pattern_matches(path: str, pattern: str) -> bool:
    """Whether module ``path`` matches ``pattern``: ``re:<regex>`` against the whole path, else segment by
    segment, each an ``fnmatch`` pattern for one segment (``*`` never spans a dot), an inclusive index range
    (``0-3``), or ``**`` for any number of segments (``model.layers.0-3.**``)."""
    if pattern.startswith(REGEX_PREFIX):
        return re.fullmatch(pattern[len(REGEX_PREFIX):], path) is not None
    return _segments_match(path.split("."), pattern.split("."))


def _check_format(value: Any, where: str, problems: list[str], allowed=FORMATS) -> str | None:
    if not isinstance(value, str) or (value not in allowed and value not in (AUTO, BF16)):
        names = sorted({AUTO, BF16, *allowed})
        problems.append(f"{where} must be one of {names}, got {value!r}")
        return None
    return value


def _check_pattern(value: Any, where: str, problems: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        problems.append(f"{where} must be a non-empty module-path pattern, got {value!r}")
        return None
    if value.startswith(REGEX_PREFIX):
        try:
            re.compile(value[len(REGEX_PREFIX):])
        except re.error as e:
            problems.append(f"{where}: {value!r} is not a valid regex: {e}")
            return None
    return value


def _parse_block(raw: Any, where: str, problems: list[str], keys=_NODE_KEYS) -> tuple[dict, list[Rule]] | None:
    if isinstance(raw, str):
        raw = {"format": raw}
    if not isinstance(raw, Mapping):
        problems.append(f"{where} must be a format name or a mapping, got {raw!r}")
        return None
    unknown = sorted(set(raw) - set(keys))
    if unknown:
        problems.append(f"{where} has unknown key(s) {unknown}; known: {list(keys)}")
    out: dict[str, Any] = {}
    if "format" in raw:
        out["format"] = _check_format(raw["format"], f"{where}.format", problems)
    if "kv_cache" in raw:
        out["kv_cache"] = _check_format(raw["kv_cache"], f"{where}.kv_cache", problems, KV_CACHE_FORMATS)
    rules: list[Rule] = []
    layers = raw.get("layers", [])
    if not isinstance(layers, list):
        problems.append(f"{where}.layers must be a list of {{match, format}} rules, got {layers!r}")
        layers = []
    for i, item in enumerate(layers):
        at = f"{where}.layers[{i}]"
        if not isinstance(item, Mapping) or set(item) != set(_RULE_KEYS):
            problems.append(f"{at} must be a mapping with exactly the keys {list(_RULE_KEYS)}, got {item!r}")
            continue
        pattern = _check_pattern(item["match"], f"{at}.match", problems)
        fmt = _check_format(item["format"], f"{at}.format", problems)
        if fmt == AUTO:
            problems.append(f"{at}.format: a rule names a format; `auto` is only the default")
            continue
        if pattern is not None and fmt is not None:
            rules.append(Rule(pattern, fmt, at))
    keep = raw.get("keep_bf16", [])
    # a bare string would iterate as one-character patterns
    if isinstance(keep, str) or not isinstance(keep, list):
        problems.append(f"{where}.keep_bf16 must be a list of module-path patterns, got {keep!r}")
        keep = []
    for i, item in enumerate(keep):
        pattern = _check_pattern(item, f"{where}.keep_bf16[{i}]", problems)
        if pattern is not None:
            rules.append(Rule(pattern, BF16, f"{where}.keep_bf16[{i}]"))
    return out, rules


def parse_quantization(raw: Any) -> QuantizationSpec:
    """The ``quantization:`` value of a serving config (None when absent), checked. Raises
    ``QuantizationSpecError`` listing every problem."""
    if raw is None:
        return QuantizationSpec()
    problems: list[str] = []
    parsed = _parse_block(raw, "quantization", problems, keys=_TOP_KEYS)
    if parsed is None:
        raise QuantizationSpecError(problems)
    top_values, top_rules = parsed
    top = NodeQuantSpec(
        format=top_values.get("format") or AUTO, rules=tuple(top_rules), kv_cache=top_values.get("kv_cache") or AUTO,
    )
    nodes: dict[str, NodeQuantSpec] = {}
    raw_nodes = raw.get("nodes", {}) if isinstance(raw, Mapping) else {}
    if not isinstance(raw_nodes, Mapping):
        problems.append(f"quantization.nodes must map node names to blocks, got {raw_nodes!r}")
        raw_nodes = {}
    for node, block in raw_nodes.items():
        parsed = _parse_block(block, f"quantization.nodes.{node}", problems)
        if parsed is None:
            continue
        values, rules = parsed
        nodes[str(node)] = NodeQuantSpec(
            format=values.get("format") or top.format,
            rules=(*rules, *top.rules),
            kv_cache=values.get("kv_cache") or top.kv_cache,
        )
    if problems:
        raise QuantizationSpecError(problems)
    return QuantizationSpec(top=top, nodes=nodes, given=True)


def spec_problems(config: Mapping[str, Any], placed_nodes: set[str] | None = None) -> list[str]:
    """The ``quantization:`` block's own problems, and node blocks naming nodes no node group places."""
    try:
        spec = parse_quantization(config.get("quantization"))
    except QuantizationSpecError as e:
        return list(e.problems)
    if placed_nodes is not None:
        unknown = sorted(set(spec.nodes) - placed_nodes)
        if unknown:
            return [f"quantization.nodes names {unknown}, which no node group places; placed: {sorted(placed_nodes)}"]
    return []
