"""Speculative decoding: the ``model_kwargs.draft`` schema, the startup
refusals a deployment gets before any worker spawns, the mode each node runs
in, and the request options refused at ingest."""
from __future__ import annotations

import re
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, ".")

from mstar.engine.resources.draft.config import DRAFT_METHODS, DraftConfig, DraftMethod, DraftMode  # noqa: E402
from mstar.engine.resources.draft.drafter import ChunkBoundary, DrafterNeeds, VerifyNode  # noqa: E402
from mstar.engine.resources.draft.startup import (  # noqa: E402
    check_deployment,
    resolve_mode,
    resource_problems,
    shape_problems,
    slack_problems,
)
from mstar.engine.resources.kv.config import KVLayerSet, KVSpec, PagedKVConfig  # noqa: E402
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PosScheme  # noqa: E402
from mstar.engine.resources.sampler.config import SamplerSpec, SamplingReqConfig  # noqa: E402
from mstar.engine.resources.spec import (  # noqa: E402
    NodeResourceSpec,
    ResourceCapabilities,
    ResourceCapabilitiesCtx,
)
from mstar.model.submodule_base import DeviceLoopback, TokenStop  # noqa: E402

NODE = "LLM"
_CTX = ResourceCapabilitiesCtx(NODE, {})


# a drafter that loads no checkpoint, served by its own table, as a test's scripted drafter is
_SCRIPTED = {**DRAFT_METHODS, "scripted": DraftMethod("scripted", needs_model=False)}


# ---- the schema ----

def test_the_block_parses_with_defaults():
    cfg = DraftConfig.parse({"method": "dspark", "model": "org/drafter", "num_draft_tokens": 7})
    assert (cfg.method, cfg.model, cfg.num_draft_tokens) == ("dspark", "org/drafter", 7)
    assert cfg.mode == DraftMode.AUTO
    assert cfg.block == 8 and cfg.kv_slack == 16
    own = DraftConfig.parse({"method": "scripted", "num_draft_tokens": 3, "mode": "synchronous"}, _SCRIPTED)
    assert own.mode == DraftMode.SYNCHRONOUS


@pytest.mark.parametrize("block,pattern", [
    ({"method": "dspark", "model": "m", "num_draft_tokens": 7, "draft_tokens": 7}, "unknown key 'draft_tokens'"),
    ({"method": "eagle", "num_draft_tokens": 2}, "method must be one of"),
    ({"method": "dspark", "num_draft_tokens": 2}, "set model"),
    ({"method": "scripted", "model": "m", "num_draft_tokens": 2}, "drop model"),
    ({"method": "scripted", "num_draft_tokens": 0}, "positive integer"),
    ({"method": "scripted", "num_draft_tokens": True}, "positive integer"),
    ({"method": "scripted", "num_draft_tokens": 2, "mode": "fast"}, "mode must be one of"),
    (["dspark"], "must be a mapping"),
])
def test_a_bad_block_is_refused_naming_the_problem(block, pattern):
    with pytest.raises(ValueError, match=pattern):
        DraftConfig.parse(block, _SCRIPTED)


def test_several_problems_come_back_together():
    with pytest.raises(ValueError) as refusal:
        DraftConfig.parse({"method": "dspark", "num_draft_tokens": -1, "extra": 1})
    message = str(refusal.value)
    assert "unknown key 'extra'" in message and "set model" in message and "positive integer" in message


def test_no_block_leaves_speculative_decoding_off():
    assert DraftConfig.from_model_kwargs({"prefill_token_buckets": [8]}) is None
    assert DraftConfig.from_model_kwargs(None) is None


def test_a_model_may_serve_a_method_of_its_own():
    assert DraftConfig.parse({"method": "scripted", "num_draft_tokens": 2}, _SCRIPTED).method == "scripted"
    with pytest.raises(ValueError, match="method must be one of"):
        DraftConfig.parse({"method": "scripted", "num_draft_tokens": 2})


# ---- the node ----

def _stop(**kw):
    return TokenStop(**{
        "loop": "decode_loop", "buffer": "new_token", "stop_ids": frozenset({2}), "sampler": "sampler",
        "walk": "decode", "prefill_walks": frozenset({"prefill"}), **kw,
    })


_LOOPBACK = DeviceLoopback(walk="decode", signal="text_inputs", sampler="sampler", buffer="new_token")


_BLOCK = {"method": "dspark", "model": "m", "num_draft_tokens": 3}


def _decl(mode="auto", needs=DrafterNeeds(chunk_boundary=ChunkBoundary.NONE), stop=None, **kw):
    cfg = DraftConfig.parse({"method": "dspark", "model": "m", "num_draft_tokens": 3, "mode": mode})
    kw.setdefault("device_loopback", _LOOPBACK)
    return VerifyNode(NODE, "decode", cfg, needs, token_stop=_stop() if stop is None else stop, **kw)


def _specs(**extra):
    specs = {
        "kv_cache": KVSpec(
            "kv_cache", {NODE}, PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=64), draft_slack=8,
        ),
        "rope": PositionSpec("rope", {NODE}, PositionConfig(kv_cache="kv_cache")),
        "sampler": SamplerSpec("sampler", {NODE}, vocab_size=32, verifies=True),
    }
    specs.update(extra)
    return specs


def test_a_well_shaped_node_has_no_problems():
    decl = _decl()
    assert shape_problems(decl, decl.token_stop, _specs()) == []
    assert resource_problems(decl, _specs()) == []


@pytest.mark.parametrize("stop,needs,pattern", [
    (None, DrafterNeeds(), "needs a TokenStop"),
    (_stop(withhold=("audio",)), DrafterNeeds(), "withholds outputs"),
    (_stop(walk="other"), DrafterNeeds(), "doesn't stop the decode loop"),
    (_stop(sampler="nope"), DrafterNeeds(), "no engine sampler 'nope'"),
    (_stop(sampler="plain"), DrafterNeeds(), "SamplerSpec.verifies"),
    (_stop(), DrafterNeeds(kv_overhang=1), "past the verify block"),
    (_stop(), DrafterNeeds(input_kinds=frozenset({"image"})), r"takes no \['text'\]"),
])
def test_a_node_of_the_wrong_shape_is_refused(stop, needs, pattern):
    decl = _decl(needs=needs, stop=stop if stop is not None else _stop())
    plain = SamplerSpec("plain", {NODE}, vocab_size=32)
    problems = shape_problems(decl, stop, _specs(plain=plain))
    assert any(re.search(pattern, p) for p in problems), problems


def test_resources_that_cant_verify_are_named_and_the_base_refuses():
    specs = _specs(
        rope=PositionSpec("rope", {NODE}, PositionConfig(kv_cache="kv_cache", scheme=PosScheme.BLOCK)),
    )
    specs["bare"] = _Unchecked("bare", {NODE})
    problems = resource_problems(_decl(), specs)
    assert any("'bare'" in p and "declares no speculative decoding support" in p for p in problems), problems
    assert any("'rope'" in p and "doesn't follow the KV length" in p for p in problems), problems


class _Unchecked(NodeResourceSpec):
    """A resource with no stance on speculative decoding."""

    @property
    def resource_class(self):
        raise NotImplementedError


def test_block_causal_attention_is_refused_though_its_parent_can_verify():
    from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, BlockCausalAttentionSpec

    config = AttentionConfig(kv_cache="kv_cache")
    assert resource_problems(_decl(), _specs(attn=AttentionSpec("attn", {NODE}, config))) == []
    problems = resource_problems(_decl(), _specs(attn=BlockCausalAttentionSpec("attn", {NODE}, config, block_size=4)))
    assert any("'attn'" in p and "block-causal" in p for p in problems), problems


@pytest.mark.parametrize("walk,ok", [
    ("decode", True), (frozenset({"decode", "decode_audio"}), True), ("other", False),
    (frozenset({"other", "decode_audio"}), False),
], ids=["own walk", "several with it", "another", "several without it"])
def test_the_verify_walks_loopback_may_cover_several_walks(walk, ok):
    """A `DeviceLoopback` over several walks (main's `covers`) serves a verify walk it covers."""
    decl = _decl(device_loopback=_LOOPBACK._replace(walk=walk))
    problems = shape_problems(decl, decl.token_stop, _specs())
    assert (not any("DeviceLoopback on itself" in p for p in problems)) == ok, problems


def test_a_context_parallel_pool_is_refused():
    specs = _specs()
    specs["kv_cache"] = KVSpec(
        "kv_cache", {NODE}, PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=64,
                                          context_parallel=True), draft_slack=8,
    )
    problems = resource_problems(_decl(), specs)
    assert any("'kv_cache'" in p and "context-parallel" in p for p in problems), problems
    assert resource_problems(_decl(), _specs()) == []


def test_a_node_that_shards_its_sequence_is_refused():
    """`SeqShardSpec` keeps the base's refusal: a verify step of k + 1 rows a request would split
    over the node's SP ranks, which nothing on the verify path takes."""
    from mstar.engine.resources.sequence_shard import SeqShardSpec

    problems = resource_problems(_decl(), _specs(seq=SeqShardSpec("seq", {NODE}, labels=("main",), axis=0)))
    assert any("'seq'" in p and "declares no speculative decoding support" in p for p in problems), problems


def test_a_sampler_with_a_history_or_logprobs_is_refused():
    specs = _specs(sampler=SamplerSpec("sampler", {NODE}, vocab_size=32, max_repetition_window=8))
    assert any("history" in p for p in resource_problems(_decl(), specs))
    specs = _specs(sampler=SamplerSpec("sampler", {NODE}, vocab_size=32, enable_logprobs=True))
    assert any("log-probs" in p for p in resource_problems(_decl(), specs))


# ---- the mode ----

class _HostPlanned(NodeResourceSpec):
    """Verifies, but only from exact host lengths (FlashInfer's plan, say)."""

    def resource_capabilities(self, ctx):
        return ResourceCapabilities(verify_refusal=None, async_verify_refusal="plans on the host")

    @property
    def resource_class(self):
        raise NotImplementedError


def test_auto_is_asynchronous_when_everything_corrects_on_the_device():
    mode, reason = resolve_mode(_decl(), _specs(), async_scheduling=True)
    assert mode == DraftMode.ASYNCHRONOUS


@pytest.mark.parametrize("specs,on,resident,why", [
    (_specs(), False, True, "async scheduling is off"),
    (_specs(), True, False, "resident planning is off"),
    (_specs(attn=_HostPlanned("attn", {NODE})), True, True, "plans on the host"),
])
def test_auto_falls_back_to_synchronous_and_says_why(specs, on, resident, why):
    mode, reason = resolve_mode(_decl(), specs, async_scheduling=on, resident_planning=resident)
    assert mode == DraftMode.SYNCHRONOUS and why in reason
    # asked for explicitly, the same refuses to start
    with pytest.raises(ValueError, match=why):
        resolve_mode(_decl(mode="asynchronous"), specs, async_scheduling=on, resident_planning=resident)


def test_synchronous_when_asked():
    assert resolve_mode(_decl(mode="synchronous"), _specs(), async_scheduling=True)[0] == DraftMode.SYNCHRONOUS


def test_the_deployment_check_lists_every_problem():
    problems = check_deployment(
        [_decl(needs=DrafterNeeds())], _specs(), streaming_out={NODE: True}, side_lane=[NODE],
        async_scheduling={}, resident_planning=True, chunked=[NODE],
    )
    assert any("streaming edge" in p for p in problems)
    assert any("side lane" in p for p in problems)
    assert any("chunk-boundary rule" in p for p in problems)


# ---- the requests ----

@pytest.mark.parametrize("options,pattern", [
    ({"repetition_penalty": 1.1}, "repetition_penalty"),
    ({"repetition_window": 8}, "repetition_window"),
    ({"min_tokens": 4}, "min_tokens"),
    ({"return_logprobs": True}, "return_logprobs"),
    ({"penalize_prompt": False}, "penalize_prompt"),
])
def test_options_verify_cant_honour_are_refused_at_ingest(options, pattern):
    spec = SamplerSpec("sampler", {NODE}, vocab_size=32, verifies=True, enable_logprobs=True)
    with pytest.raises(ValueError, match=pattern):
        SamplingReqConfig(**options).validate(spec)
    # the same request on a node that doesn't verify is fine as far as verify goes
    try:
        SamplingReqConfig(**options).validate(SamplerSpec("sampler", {NODE}, vocab_size=32, enable_logprobs=True))
    except ValueError as exc:
        assert "speculative" not in str(exc)


def test_supported_options_pass():
    spec = SamplerSpec("sampler", {NODE}, vocab_size=32, verifies=True, enable_min_p=True)
    SamplingReqConfig(temperature=0.7, top_k=20, top_p=0.9, min_p=0.05, ignore_eos=True).validate(spec)


def test_a_check_deployment_model_hook_path():
    """`check_draft` from a model's declarations, as `validate_deployment` calls it."""
    from mstar.graph.base import GraphEdge, GraphNode
    from mstar.validation.draft import check_draft

    node = GraphNode(name=NODE, input_names={"x"}, outputs=[GraphEdge(next_node="tts", name="h", is_streaming=True)])
    model = SimpleNamespace(
        verify_nodes=lambda: [_decl()], get_graph_walk_graphs=lambda: {"decode": node}, chunking_policy_of=lambda: {},
    )
    config = {"node_groups": [{"node_names": [NODE], "ranks": [0]}], "model_kwargs": {"draft": _BLOCK}}
    problems = check_draft(model, config, list(_specs().values()))
    assert len(problems) == 1 and "streaming edge" in problems[0]
    assert check_draft(SimpleNamespace(verify_nodes=lambda: []), {}, None) == []


def test_a_kv_pool_without_the_slack_is_refused():
    from mstar.engine.resources.draft.startup import check_deployment, node_problems

    decl, specs = _decl(), _specs()
    deployment = lambda limit: check_deployment([decl], specs, {}, (), {}, True, max_seq_len=limit)  # noqa: E731
    assert deployment(56) == []
    problems = deployment(64)
    assert len(problems) == 1 and "set resources.kv_cache.max_seq_len to 64" in problems[0]
    assert "or lower the serving max_seq_len to 56" in problems[0]
    # the engine's load runs the same list (no serving limit there)
    assert node_problems(decl, decl.token_stop, specs, False, max_seq_len=64) == problems


def _fit(needs=None, target=None, tap_size=16, more_taps=False, tap_nodes=None, **drafter):
    """A verifying node that declares a drafter, its specs: a fit unless a test says otherwise."""
    from mstar.engine.resources.draft.drafter import ModelGeometry
    from mstar.engine.resources.kv.config import KVLayerSet
    from mstar.engine.resources.taps.config import TapSpec

    shape = dict(num_layers=1, num_kv_heads=2, head_dim=8, hidden_size=16, vocab_size=32, dtype="bfloat16")
    geometry = ModelGeometry(**{**shape, "num_qo_heads": 4, "rope_theta": 1e4, **drafter})
    pool = PagedKVConfig(num_layers=2, num_kv_heads=2, num_qo_heads=4, head_dim=8, max_seq_len=64)
    pool.join(KVLayerSet("kv_cache", "drafter", 1, 2, 8))
    specs = _specs(
        kv_cache=KVSpec("kv_cache", {NODE}, pool, draft_slack=8),
        rope=PositionSpec("rope", {NODE}, PositionConfig(kv_cache="kv_cache", rope_theta=1e4)),
        taps=TapSpec("taps", tap_nodes or {NODE}, layers=(1,), hidden_size=tap_size),
        **({"extra": TapSpec("extra", {NODE}, layers=(0,), hidden_size=tap_size)} if more_taps else {}),
    )
    kw = dict(taps=(1,), tap_key="taps", kv=KVLayerSet("kv_cache", "drafter", 1, 2, 8), geometry=geometry)
    decl = _decl(
        needs=DrafterNeeds(chunk_boundary=ChunkBoundary.NONE, **{**kw, **(needs or {})}),
        target=target if target != "default" else ModelGeometry(**{**shape, "num_layers": 2}),
    )
    return decl, specs


@pytest.mark.parametrize("fit, message", [
    (dict(needs=dict(taps=(0,))), "TapSpec 'taps' declares (1,)"),
    # the drafter's TapSpec is found by its key, not by being the node's only one
    (dict(needs=dict(tap_key="other")), "from TapSpec 'other' (DrafterNeeds.tap_key), which the node doesn't declare"),
    (dict(needs=dict(tap_key=None)), "from TapSpec None"),
    # declared, but on another node: the verifying node's steps would find no tap resource
    (dict(tap_nodes={"elsewhere"}), "from TapSpec 'taps' (DrafterNeeds.tap_key), which the node doesn't declare"),
    (dict(more_taps=True), "declares TapSpecs ['extra'] besides the drafter's; run_layers writes one"),
    (dict(needs=dict(taps=())), "declares taps its drafter doesn't read"),
    (dict(target=None), "no target geometry"),
    (dict(tap_size=8), "is 8 wide"),
    (dict(num_qo_heads=8), "8 query heads"),
    (dict(rope_theta=1e6), "rope_theta is 1000000.0"),
    (dict(hidden_size=32), "residual width is 32"),
    (dict(vocab_size=64), "vocabulary is 64"),
    (dict(dtype="float32"), "dtype is float32"),
    (dict(needs=dict(taps=(3,))), "the target has 2"),
    (dict(needs=dict(kv=KVLayerSet("other", "drafter", 1, 2, 8))), "is not a paged pool of the node"),
    (dict(needs=dict(kv=KVLayerSet("kv_cache", "other", 1, 2, 8))), "as layer set 'other'"),
    (dict(needs=dict(kv=KVLayerSet("kv_cache", "drafter", 3, 2, 8))), "joins 3 layers"),
])
def test_a_drafter_that_doesnt_fit_its_target_is_refused(fit, message):
    from mstar.engine.resources.draft.startup import geometry_problems

    decl, specs = _fit(**{"target": "default", **fit})
    assert any(message in p for p in geometry_problems(decl, specs)), geometry_problems(decl, specs)


def test_a_drafter_that_fits_has_no_geometry_problems_and_a_pool_short_of_its_layers_does():
    from mstar.engine.resources.draft.startup import geometry_problems

    decl, specs = _fit(target="default")
    assert geometry_problems(decl, specs) == []
    specs["kv_cache"].config.num_layers = 2
    assert any("holds layers 2..2" in p for p in geometry_problems(decl, specs))


def test_a_kv_pool_shared_with_another_node_is_refused():
    """A prefill/decode split, or any other node reading the pool, would see rejected drafts."""
    specs = _specs(kv_cache=KVSpec("kv_cache", {NODE, "prefill_node"}, PagedKVConfig(
        num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=64,
    )))
    problems = resource_problems(_decl(), specs)
    assert any("shared with nodes ['prefill_node']" in p for p in problems), problems


def test_a_verify_walk_inside_a_combined_walk_is_refused():
    """A mixed step would commit its verify rows' blocks with no take-back."""
    from mstar.engine.resources.draft.startup import combined_problem

    assert combined_problem(_decl(), {}) is None
    problem = combined_problem(_decl(), {(NODE, "decode"): "mixed"})
    assert problem is not None and "combined walk 'mixed'" in problem


def test_a_request_is_held_to_the_serving_limit_not_the_pools():
    """The pool's max_seq_len carries the draft slack; a request's limit doesn't."""
    from mstar.engine.resources.kv.config import KVReqConfig, PrefixSpan, StreamLayout

    spec = _specs()["kv_cache"]
    assert spec.request_limit == 56

    def request(positions):
        return KVReqConfig(kv_layout={"main": StreamLayout(spans=(PrefixSpan(positions, positions, "prefill"),))})

    request(56).validate(spec)
    with pytest.raises(ValueError, match="past the model's context of 56"):
        request(57).validate(spec)


def test_a_pool_that_declares_no_slack_is_refused():
    from mstar.engine.resources.draft.startup import slack_problems

    bare = KVSpec("kv_cache", {NODE}, PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=64))
    problems = slack_problems([_decl()], [bare], 56)
    assert len(problems) == 1 and "draft_slack 0" in problems[0]


def test_data_parallel_replicas_are_not_lockstep():
    """A DP group schedules each replica alone; only TP x SP is lockstep."""
    from mstar.validation.draft import async_scheduling_of

    model = SimpleNamespace(verify_nodes=lambda: [_decl()], get_graph_walk_graphs=lambda: {})
    dp = {"node_groups": [{"node_names": [NODE], "ranks": [0, 1]}], "tp_async_sched": False}
    assert async_scheduling_of(model, dp) == {}
    tp = {"node_groups": [{"node_names": [NODE], "ranks": [0, 1], "tp_size": 2}], "tp_async_sched": False}
    assert async_scheduling_of(model, tp)[NODE][0] is False


def test_verify_backend_is_validated_on_every_node():
    from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec

    with pytest.raises(ValueError, match="verify_backend"):
        AttentionConfig(kv_cache="kv", verify_backend="fa33")
    spec = AttentionSpec("attn", {NODE}, AttentionConfig(kv_cache="kv"))
    with pytest.raises(ValueError, match="verify_backend"):
        spec.apply_yaml_overrides(verify_backend="fa33")
    # a config fact only: whether FA3 runs is the workers' question
    assert spec.resource_capabilities(_CTX).async_verify_refusal is None


def test_verify_backend_is_one_enum_and_the_flashinfer_pin_is_reported_by_the_spec():
    from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, VerifyBackend

    assert AttentionConfig(kv_cache="kv").verify_backend is VerifyBackend.AUTO
    spec = AttentionSpec("attn", {NODE}, AttentionConfig(kv_cache="kv", verify_backend="fa3"))
    assert spec.config.verify_backend is VerifyBackend.FA3
    spec.apply_yaml_overrides(verify_backend="flashinfer")
    assert spec.config.verify_backend is VerifyBackend.FLASHINFER
    # the pin's one report: the spec's, which `resolve_mode` reads (the manager grants it)
    caps = spec.resource_capabilities(_CTX)
    assert "flashinfer" in caps.async_verify_refusal and caps.verify_refusal is None


@pytest.mark.parametrize("loopback", [None, _LOOPBACK._replace(walk="prefill"), _LOOPBACK._replace(sampler="other")])
def test_a_verify_walk_without_its_device_loopback_is_refused(loopback):
    """A verify walk's next step reads each row's token off the loop-back slot."""
    decl = _decl(device_loopback=loopback)
    assert any("DeviceLoopback" in p for p in shape_problems(decl, decl.token_stop, _specs()))


def test_a_draft_block_on_a_model_that_serves_none_is_refused():
    """A draft block is never silently ignored, and its keys are still checked."""
    from mstar.validation.draft import check_draft

    model = SimpleNamespace(verify_nodes=lambda: [])
    block = {"method": "dspark", "model": "m", "num_draft_tokens": 3}
    problems = check_draft(model, {"model_kwargs": {"draft": block}}, None)
    assert len(problems) == 1 and "drop model_kwargs.draft" in problems[0]
    with pytest.raises(ValueError, match="unknown key"):
        check_draft(model, {"model_kwargs": {"draft": {**block, "k": 3}}}, None)
    assert check_draft(model, {}, None) == []


def test_a_model_drafting_other_than_the_block_says_is_refused():
    """The model's VerifyNode config is the parsed model_kwargs.draft,
    and a verifying node with no block at all is refused."""
    from mstar.graph.base import GraphNode
    from mstar.validation.draft import check_draft

    node = GraphNode(name=NODE, input_names={"x"}, outputs=[])
    model = SimpleNamespace(
        verify_nodes=lambda: [_decl()], get_graph_walk_graphs=lambda: {"decode": node}, chunking_policy_of=lambda: {},
    )
    groups = {"node_groups": [{"node_names": [NODE], "ranks": [0]}]}
    specs = list(_specs().values())
    assert check_draft(model, {**groups, "model_kwargs": {"draft": _BLOCK}}, specs) == []
    drift = check_draft(model, {**groups, "model_kwargs": {"draft": {**_BLOCK, "num_draft_tokens": 4}}}, specs)
    assert len(drift) == 1 and "other than model_kwargs.draft" in drift[0]
    missing = check_draft(model, groups, specs)
    assert len(missing) == 1 and "without a model_kwargs.draft block" in missing[0]


def _declared(other_sampler_nodes=None):
    """A verifying node's specs as a model declares them: a paged pool and a sampler with
    none of the draft's fields set, and a pool and sampler of another node."""
    return [
        KVSpec("kv_cache", {NODE}, PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=64)),
        KVSpec("other_kv", {"other"}, PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=64)),
        SamplerSpec("sampler", {NODE}, vocab_size=32),
        SamplerSpec("other_sampler", other_sampler_nodes or {"other"}, vocab_size=32),
    ]


class _NoResources:
    needs = DrafterNeeds(chunk_boundary=ChunkBoundary.NONE)

    def node_resources(self, nodes):
        return []


def test_drafter_resources_derives_the_slack_and_the_verifying_sampler_from_the_draft_config():
    from mstar.engine.resources.draft.config import DraftConfig
    from mstar.engine.resources.draft.drafter import drafter_resources

    config = DraftConfig.parse({"method": "dspark", "model": "m", "num_draft_tokens": 3})
    out = {spec.resource_key: spec for spec in drafter_resources(_declared(), _NoResources(), {NODE}, config)}
    assert out["kv_cache"].draft_slack == config.kv_slack == 8
    assert out["kv_cache"].config.max_seq_len == 64 + 8 and out["kv_cache"].request_limit == 64
    assert out["sampler"].verifies and not out["sampler"].enable_repetition_penalty
    # another node's pool and sampler are left as declared
    assert out["other_kv"].draft_slack == 0 and out["other_kv"].config.max_seq_len == 64
    assert not out["other_sampler"].verifies and out["other_sampler"].enable_repetition_penalty
    # what startup checks then holds
    decl = _decl(needs=DrafterNeeds(chunk_boundary=ChunkBoundary.NONE))
    assert slack_problems([decl], list(out.values()), max_seq_len=64) == []


def test_a_slack_set_twice_is_bound_once_and_a_model_set_one_is_refused():
    pool = KVSpec("kv_cache", {NODE}, PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=64))
    pool.bind_draft_slack(8)
    pool.bind_draft_slack(8)
    assert pool.draft_slack == 8 and pool.config.max_seq_len == 72
    with pytest.raises(ValueError, match="a model leaves it 0"):
        pool.bind_draft_slack(4)


def test_a_sampler_another_node_shares_is_not_made_a_verifying_one():
    from mstar.engine.resources.draft.config import DraftConfig
    from mstar.engine.resources.draft.drafter import drafter_resources

    declared = _declared()
    declared[2].nodes = {NODE, "other"}
    config = DraftConfig.parse({"method": "dspark", "model": "m", "num_draft_tokens": 3})
    out = {spec.resource_key: spec for spec in drafter_resources(declared, _NoResources(), {NODE}, config)}
    # startup then refuses the node (shape_problems: the sampler doesn't verify)
    assert not out["sampler"].verifies and out["sampler"].enable_repetition_penalty


def _request(scratch=0, decode=100):
    """A request's KV config as the conductor builds it from a model's layout row."""
    from mstar.engine.resources.kv.config import KVReqConfig

    cfg = KVReqConfig()
    cfg.apply_conductor_config(
        kv_layout={"main": [[[10, 10, "prefill", 0, None, None, True, "ids"]], [["decode_loop", 1]], scratch, None, 0]},
        loop_caps={"decode_loop": decode},
    )
    return cfg


def test_a_verifying_pool_floors_each_layouts_scratch_at_the_slack():
    config = PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=64, page_size=4)
    pool = KVSpec("kv_cache", {NODE}, config)
    pool.bind_draft_slack(8)
    cfg = _request()
    assert cfg.kv_layout["main"].scratch == 0
    cfg.apply_spec(pool)
    assert cfg.kv_layout["main"].scratch == 8
    # the pages the request is sized at count the slack: 10 + 100 + 8 slots, 4 to a page
    assert cfg.kv_layout["main"].pages({"decode_loop": 100}, 4) == 30


def test_a_label_with_its_own_scratch_on_a_verifying_pool_is_refused_and_a_plain_pool_keeps_it():
    pool = KVSpec("kv_cache", {NODE}, PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=64))
    plain = _request(scratch=3)
    plain.apply_spec(pool)
    assert plain.kv_layout["main"].scratch == 3
    pool.bind_draft_slack(8)
    with pytest.raises(ValueError, match="not supported yet"):
        _request(scratch=3).apply_spec(pool)


def test_a_max_seq_len_override_on_a_verifying_pool_is_the_clients_limit():
    """The YAML override keeps its meaning: the pool holds the slack past it, and
    applying the overrides twice (validation, then the worker) changes nothing."""
    from mstar.engine.resources.spec import apply_yaml_overrides

    pool = KVSpec("kv_cache", {NODE}, PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=64))
    pool.bind_draft_slack(8)
    for _ in range(2):
        apply_yaml_overrides([pool], {"resources": {"kv_cache": {"max_seq_len": 48}}})
        assert pool.config.max_seq_len == 56 and pool.request_limit == 48
    decl = _decl(needs=DrafterNeeds(chunk_boundary=ChunkBoundary.NONE))
    assert slack_problems([decl], [pool], max_seq_len=48) == []
    # a serving limit past the pool's: the remedy is the override at that limit, the slack added past it
    (problem,) = slack_problems([decl], [pool], max_seq_len=60)
    assert "holds a request to 48 positions" in problem and "set resources.kv_cache.max_seq_len to 60" in problem
    # a plain pool takes the value as written
    plain = KVSpec("kv_cache", {NODE}, PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=64))
    apply_yaml_overrides([plain], {"resources": {"kv_cache": {"max_seq_len": 48}}})
    assert plain.config.max_seq_len == 48
