"""CosyVoice3 plans on the contract what it planned by hand, and carries a stream's state where it kept it.

The specs and steps below are the reference, copied from the model as it stood before the port, so the adapters can't
pass by agreeing with themselves. The differences are named where they appear: the LM's repetition window is a Value
where the node kept a list in ``request_state``, the flow's plan over both streams is named by its labels, ``flow_kv``
keeps the pool's ``prefix_cache`` default, and the two other Values leave ``per_layer`` at its default. The last tests
hold the parts that moved off ``request_state`` against what they replaced: the window against the shift register the
node kept, the EOS a prefill token forces against the steps it reaches, and token2wav's loop-back edges against what its
nodes return.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import yaml
from test_cosyvoice3_model import _info, _lm, _lm_request, _lm_run, _random_flow, _random_hift, _tiny_config, _voice
from test_startup_graph_validation import _FACTORIES, _bare
from test_state_ports import _assert_equal_per_key, _assert_steps_equal, _published, _stored

from mstar.engine.resources import BlockCausalAttentionSpec
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.engine.state.adapters import StateSetup, node_handles, state_steps
from mstar.engine.state.analysis import state_declaration_problems
from mstar.model.base import enabled_resources, keyed_labels, serving_resources
from mstar.model.cosyvoice3 import submodules
from mstar.model.cosyvoice3.components.flow import FlowCounts
from mstar.model.cosyvoice3.config import (
    FLOW_ATTN,
    FLOW_COND_LABEL,
    FLOW_KV,
    FLOW_KV_PAGE_FRAMES,
    FLOW_NODE,
    FLOW_TAILS,
    FLOW_UNCOND_LABEL,
    HIFT_NODE,
    HIFT_STATE,
    LM_ATTN,
    LM_KV,
    LM_LABEL,
    LM_NODE,
    RAS_HISTORY,
    CosyVoice3Config,
)
from mstar.model.cosyvoice3.cosyvoice3_model import DEFAULT_FLOW_REQUESTS, HIFT_STATE_REQUESTS, CosyVoice3Model
from mstar.model.cosyvoice3.submodules import DECODE_LOOP, MEL, PREV_TOKEN, FlowSubmodule, HiFTSubmodule
from mstar.model.submodule_base import ARNodeInputs, engine_step

# the position resource main named, which the contract now derives from ``lm_kv``
LM_POS = "lm_pos"

CONFIGS = {
    "default": CosyVoice3Config(),
    "no_cache_no_capture": CosyVoice3Config(flow_cache_frames=0, hift_capture_frame_buckets=()),
}


def _reference_specs(model: CosyVoice3Model) -> list:
    """`CosyVoice3Model.get_node_resources` before the port, with `_flow_resources` and `hift_state_specs`."""
    cfg = model.config
    lm = cfg.lm
    kv = PagedKVConfig(
        num_layers=lm.num_hidden_layers,
        num_kv_heads=lm.num_key_value_heads,
        head_dim=lm.head_dim,
        max_seq_len=model._max_seq_len(),
        num_qo_heads=lm.num_attention_heads,
    )
    specs = [
        KVSpec(resource_key=LM_KV, nodes={LM_NODE}, config=kv),
        AttentionSpec(resource_key=LM_ATTN, nodes={LM_NODE}, config=AttentionConfig(kv_cache=LM_KV, decode_fmha=True)),
        # Named: the Llama 3.1 factors `ParallelAttention` passed on every call, which leave every frequency as it is,
        # are on the spec
        PositionSpec(resource_key=LM_POS, nodes={LM_NODE}, config=PositionConfig(
            kv_cache=LM_KV, rope_theta=lm.rope_theta, low_freq_factor=1.0, high_freq_factor=1.0, old_context_len=8192,
        )),
        # Named: the repetition window, a list in `request_state`, is a Value of the deployment's width
        RecurrentStateSpec(resource_key=RAS_HISTORY, nodes={LM_NODE}, config=RecurrentStateConfig(
            num_layers=1, labels=("main",),
            blocks={"tokens": RecurrentBlockConfig(shape=(max(cfg.sampling.ras_window, 1),), dtype=torch.int64)},
        )),
    ]
    # Named: main set ``per_layer=False`` on the blocks of the two Values below, and the port leaves its default. On one
    # layer either holds one tensor per slot (test_each_value_pool_takes_mains_bytes).
    if cfg.flow_cache_frames:
        dit, steps = cfg.flow.dit, cfg.flow.cfm.n_timesteps
        specs += [
            KVSpec(resource_key=FLOW_KV, nodes={FLOW_NODE}, config=PagedKVConfig(
                num_layers=dit.depth * steps,
                num_kv_heads=dit.heads,
                num_qo_heads=dit.heads,
                head_dim=dit.dim_head,
                max_seq_len=cfg.flow_cache_frames,
                max_num_pages=model._flow_kv_pages(DEFAULT_FLOW_REQUESTS),
                page_size=FLOW_KV_PAGE_FRAMES,
                # Named: main set False, and the pool keeps its default. No label is keyed for prefix reuse, so
                # the engine never opens this cache (test_no_label_is_keyed_for_prefix_reuse).
                prefix_cache=True,
            )),
            BlockCausalAttentionSpec(
                resource_key=FLOW_ATTN, nodes={FLOW_NODE}, config=AttentionConfig(kv_cache=FLOW_KV),
                block_size=dit.static_chunk_size,
            ),
            RecurrentStateSpec(resource_key=FLOW_TAILS, nodes={FLOW_NODE}, config=RecurrentStateConfig(
                num_layers=1,
                blocks={
                    "tails": RecurrentBlockConfig(shape=(4 * steps, dit.conv_pos_kernel - 1, dit.dim), dtype=None),
                },
                max_slots=2 * DEFAULT_FLOW_REQUESTS,
                labels=("main",),
            )),
        ]
    if cfg.hift_captured:
        hift = cfg.hift
        specs.append(RecurrentStateSpec(resource_key=HIFT_STATE, nodes={HIFT_NODE}, config=RecurrentStateConfig(
            num_layers=1,
            blocks={
                "mel_tail": RecurrentBlockConfig(shape=(hift.in_channels, cfg.hift_tail_frames), dtype=torch.float32),
                "phase_carry": RecurrentBlockConfig(shape=(hift.nb_harmonics + 1,), dtype=torch.float32),
            },
            max_slots=HIFT_STATE_REQUESTS + 1,
            labels=("main",),
        )))
    return specs


def _assert_equal_to_main(specs, reference, kv_dtype: torch.dtype = torch.bfloat16) -> None:
    """Compare both sides' dtypes as the pools store them, since main left the dtype of ``flow_tails`` to the
    deployment."""
    _assert_equal_per_key(specs, [_stored(spec, kv_dtype) for spec in reference], kv_dtype)


@pytest.mark.parametrize("name", CONFIGS)
def test_specs_equal_the_hand_written_ones(name):
    model = _bare(CosyVoice3Model, config=CONFIGS[name])
    _assert_equal_to_main(enabled_resources(model, {}, StateSetup()), _reference_specs(model))
    assert state_declaration_problems(model) == []


def test_every_published_config_is_covered():
    stems = {path.stem for path in _published("CosyVoice3Model")}
    assert stems == {"cosyvoice3", "cosyvoice3_split"}, "a config here that the port never compares could change unseen"


@pytest.mark.parametrize("path", _published("CosyVoice3Model"), ids=lambda path: path.stem)
def test_specs_equal_the_hand_written_ones_on_every_published_config(path, tmp_path, monkeypatch):
    deployment = yaml.safe_load(path.read_text())
    model = _FACTORIES["CosyVoice3Model"](deployment["model"], deployment.get("model_kwargs"), tmp_path, monkeypatch)
    # the LM's dtype, which neither config sets
    kv_dtype = torch.bfloat16
    expected = _reference_specs(model)
    apply_yaml_overrides(expected, deployment)
    _assert_equal_to_main(serving_resources(model, deployment), expected, kv_dtype)


@pytest.mark.parametrize("key", [FLOW_TAILS, HIFT_STATE])
def test_each_value_pool_takes_mains_bytes(key):
    model = _bare(CosyVoice3Model, config=_tiny_config())
    (ported,) = [spec for spec in enabled_resources(model, {}, StateSetup()) if spec.resource_key == key]
    (spec,) = [spec for spec in _reference_specs(model) if spec.resource_key == key]
    blocks = {name: replace(block, per_layer=False) for name, block in spec.config.blocks.items()}
    main = replace(spec, config=replace(spec.config, blocks=blocks))
    info = EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.bfloat16)

    def nbytes(spec) -> int:
        return sum(block.nbytes for block in build_resource(spec, info)._blocks.values())

    assert nbytes(ported) == nbytes(main), f"{key}'s pool holds other bytes than main's"


def test_no_label_is_keyed_for_prefix_reuse():
    model = _bare(CosyVoice3Model, config=CosyVoice3Config())
    assert keyed_labels(model) == {}, "a keyed label asks for prefix reuse, which main gave no CosyVoice3 pool"


# Steps


class _StubPool:
    """Answers what a handle reads of its pool when it is bound, and holds nothing."""

    def layer_view(self, layer=None):
        return torch.empty(0)

    def block(self, field, layer):
        return torch.empty(0)


def _reference_lm_step(rids: list[int], spans: tuple[int, ...]) -> SubmoduleStep:
    """`LMSubmodule.declare_step` before the port."""
    return SubmoduleStep(
        segments=[Segment(request_id=rid, label=LM_LABEL, span=span) for rid, span in zip(rids, spans, strict=True)],
        steps={
            LM_KV: KVStep(), LM_ATTN: AttentionStep(causal=True), LM_POS: PositionStep(),
            # Named: each row steps its repetition window
            RAS_HISTORY: RecurrentStep(),
        },
    )


@pytest.mark.parametrize("walk, spans", [("prefill", (40, 37)), ("prefill_voice", (52, 9)), ("decode", (1, 1))])
def test_lm_steps_equal_the_hand_written_ones(walk, spans):
    sub, _ = _lm(_tiny_config())
    rows = [ARNodeInputs(input_seq_len=span) for span in spans]
    step = engine_step(sub, graph_walk=walk, request_ids=[1, 2], inputs=rows)
    _assert_steps_equal(step, _reference_lm_step([1, 2], spans))


# Named: the region's plan over both streams is named by its labels joined with ``+``, where main named it ``cfg``.
FLOW_PLAN = "cond+uncond"


def _reference_new_frames_step(request_ids: list[int], seq_lens: list[int]) -> SubmoduleStep:
    """`FlowSubmodule._new_frames_step` before the port."""
    spans = [n // 2 for n in seq_lens]
    labels = (FLOW_COND_LABEL, FLOW_UNCOND_LABEL)
    kv_segments = tuple(
        Segment(rid, label, span) for label in labels for rid, span in zip(request_ids, spans, strict=True)
    )
    return SubmoduleStep(
        combined_labels={labels: FLOW_PLAN},
        steps={
            FLOW_KV: KVStep(segments=kv_segments, commit=True),
            FLOW_ATTN: AttentionStep(segments=kv_segments, causal=False),
            FLOW_TAILS: RecurrentStep(
                segments=tuple(Segment(rid, "main", span) for rid, span in zip(request_ids, spans, strict=True))
            ),
        },
    )


def _reference_vocode_step(request_ids: list[int]) -> SubmoduleStep:
    """`HiFTSubmodule._vocode_step` before the port."""
    segments = tuple(Segment(rid, "main", 1) for rid in request_ids)
    return SubmoduleStep(steps={HIFT_STATE: RecurrentStep(segments=segments)})


def _region_step(node: str, step: SubmoduleStep) -> SubmoduleStep:
    """``step`` as a region's runner expands it, over handles bound to stub pools."""
    model = _bare(CosyVoice3Model, config=CosyVoice3Config())
    pools = {spec.resource_key: _StubPool() for spec in enabled_resources(model, {}, StateSetup())}
    return state_steps(step, node_handles(model, node, pools), graph_walk="token2wav")


def test_the_flow_region_step_equals_the_hand_written_one():
    # the second row pads the batch: its zero span reads the sink
    step = _region_step(FLOW_NODE, FlowSubmodule._new_frames_step([1, 2], [100, 0]))
    _assert_steps_equal(step, _reference_new_frames_step([1, 2], [100, 0]))


def test_the_vocode_region_step_equals_the_hand_written_one():
    step = _region_step(HIFT_NODE, HiFTSubmodule._vocode_step([1, 2], [32, 32]))
    _assert_steps_equal(step, _reference_vocode_step([1, 2]))


# What moved off request_state


def test_the_repetition_window_is_the_shift_register_the_node_kept(monkeypatch):
    """Each step samples against its row's last W tokens, oldest first and -1 before the first, its own input token
    included. The node kept that list and shifted it once a step; the ring takes slot ``i % W`` on decode iteration
    i, and wraps here."""
    cfg = _tiny_config()
    width = cfg.sampling.ras_window
    sub, res = _lm(cfg)
    seen, sample = [], submodules.ras_sample

    def spy(logits, **kwargs):
        seen.append(kwargs["history"][0].tolist())
        return sample(logits, **kwargs)

    monkeypatch.setattr(submodules, "ras_sample", spy)
    steps = width + 4
    with torch.no_grad():
        tokens = _lm_run(sub, res, {0: (_lm_request(limits=(0, steps)), 5)}, steps=steps, ignore_eos=True)[0]
    register, expected = [-1] * width, [[-1] * width]
    for token in tokens[:-1]:
        register = register[1:] + [token]
        expected.append(register)
    assert seen == expected


def test_only_the_first_decode_step_is_forced_to_eos_by_the_prefill_token():
    """Named: the node kept that the prefill token ended a request and forced every decode step after it. A step
    reads its iteration instead, so one prepared ahead of the first step's EOS samples, and the worker drops that
    step's outputs because the loop's stop is pending."""
    cfg = _tiny_config()
    sub, _ = _lm(cfg)
    info = _info(0, 0)
    control = {PREV_TOKEN: [torch.tensor([cfg.lm.speech_token_size + 2])]}
    forced = []
    for count in (0, 1):
        info.dynamic_loop_iter_counts[DECODE_LOOP] = count
        step = sub.prepare_inputs("decode", info, control)
        forced.append(sub.preprocess("decode", None, [step])["force_eos"].item())
    assert forced == [True, False]


def test_each_token2wav_node_reads_back_every_edge_it_sends_itself():
    model = _FACTORIES["CosyVoice3Model"]("cosyvoice3", {}, None, None)
    loop = model.get_graph_walk_graphs()["token2wav"]
    for node in loop.section.sections:
        back = {edge.name for edge in node.outputs if edge.next_node == node.name}
        assert back and back <= set(node.input_names), f"{node.name} sends itself an edge it doesn't read"
    # a chunk brings at least one new token, and the stream's end may come as a chunk of its own
    assert loop.max_iters == model._speech_limit + 2


def test_the_token2wav_nodes_return_the_edges_the_loop_declares():
    cfg = _tiny_config()
    model = _FACTORIES["CosyVoice3Model"]("cosyvoice3", {}, None, None)
    edges = {node.name: node.outputs for node in model.get_graph_walk_graphs()["token2wav"].section.sections}
    tokens = torch.randint(3, 40, (30,), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        (mel,) = FlowSubmodule(_random_flow(cfg), None, cfg)._run(
            [0], [tokens], [_voice(cfg, 10)], [tokens], [FlowCounts()], [True], [2],
        )
        (audio,) = HiFTSubmodule(_random_hift(cfg), cfg)._run([0], [mel[MEL][0].double()], [True], [(None, [])], {})
    for name, out in ((FLOW_NODE, mel), (HIFT_NODE, audio)):
        assert set(out) == {edge.name for edge in edges[name]}, f"{name} returns other outputs than its edges"
