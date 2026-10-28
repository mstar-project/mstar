"""``load_weights_into`` routes names through an indexed rule matcher; it
must pick exactly the rule the first-win linear scan (``_apply_stacked``)
picks, for every name."""
import itertools
import random

from mstar.model.loader.base import (
    LLAMA_STACKED_PARAMS,
    WHISPER_STACKED_PARAMS,
    StackedParamRule,
    _apply_stacked,
    _StackedMatcher,
)


def _expert_rules(n):
    rules = []
    for i in range(n):
        for proj, sid in (("gate_proj", f"gate:{i}"), ("up_proj", f"up:{i}")):
            rules.append(StackedParamRule(
                ".experts.gate_up_proj_scale_inv",
                f".experts.{proj}.__expert{i}__.weight_scale_inv", sid))
            rules.append(StackedParamRule(
                ".experts.gate_up_proj_fp8", f".experts.{proj}.__expert{i}__.weight", sid))
        rules.append(StackedParamRule(
            ".experts.down_proj", f".experts.down_proj.__expert{i}__.weight", f"down:{i}"))
    rules.append(StackedParamRule(".conv1d.weight", ".q_conv1d.weight", "q"))
    return rules + LLAMA_STACKED_PARAMS


def _names(n_experts):
    for layer, e, proj, suffix in itertools.product(
            (0, 11), range(n_experts), ("gate_proj", "up_proj", "down_proj"),
            ("weight", "weight_scale_inv")):
        yield f"model.layers.{layer}.mlp.experts.{proj}.__expert{e}__.{suffix}"
    for name in ("model.layers.0.self_attn.q_proj.weight",
                 "model.layers.0.self_attn.q_conv1d.weight",
                 "model.layers.0.mlp.shared_expert.gate_proj.weight",
                 "model.layers.0.mlp.up_proj.weight", "lm_head.weight",
                 "a..b", ".experts.gate_proj.__expert1__.weight", "x.q_proj_extra.w"):
        yield name


def test_matcher_agrees_with_linear_scan():
    for rules in (_expert_rules(12), LLAMA_STACKED_PARAMS, WHISPER_STACKED_PARAMS, []):
        match = _StackedMatcher(rules)
        for name in _names(12):
            assert match(name) == _apply_stacked(name, rules), name


def test_matcher_keeps_first_win_across_overlapping_rules():
    """Rules whose suffixes overlap, in random orders: the index must never
    let a later rule win over an earlier one that also matches."""
    pieces = ["a", "b", "ab", "", "x"]
    rng = random.Random(0)
    for _ in range(300):
        rules = [
            StackedParamRule(f"T{i}", "." + ".".join(rng.choices(pieces, k=rng.randint(1, 3))), i)
            for i in range(rng.randint(1, 8))
        ]
        match = _StackedMatcher(rules)
        for _ in range(20):
            name = ".".join(rng.choices(pieces, k=rng.randint(1, 5)))
            assert match(name) == _apply_stacked(name, rules), (name, rules)
