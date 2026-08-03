"""Ragged (cacheless) attention wiring: the ``NodeSubmodule`` opt-in, the
engine-owned state, and the ``StatelessEngine`` plumbing that injects it into
``ModelInputsFromEngine``.

The plumbing tests run on CPU using the ``make_attn_state`` escape hatch, so they
exercise the wiring without touching FlashInfer. The CUDA-marked tests cover the
real default state.
"""

import pytest
import torch

from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.attention_state import (
    AttentionMode,
    AttentionState,
    RaggedAttention,
    RaggedAttentionConfig,
    build_ragged_attention_state,
)
from mstar.engine.base import NodeBatch
from mstar.engine.stateless_engine import StatelessEngine, make_enc_dec_config
from mstar.model.submodule_base import NodeInputs, NodeSubmodule

NUM_HEADS = 16
HEAD_DIM = 72


class _StubState:
    """Minimal AttentionState; records what it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.planned = []

    def plan(self, cu_seqlens):
        self.planned.append(cu_seqlens)

    def run(self, q, k, v):
        return q


def _stub_config(**overrides) -> RaggedAttentionConfig:
    kwargs = {
        "num_qo_heads": NUM_HEADS,
        "num_kv_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "make_attn_state": _StubState,
        # Keep the CPU-path workspace allocation trivial.
        "workspace_bytes": 1024,
    }
    kwargs.update(overrides)
    return RaggedAttentionConfig(**kwargs)


class _Submodule(NodeSubmodule):
    """Records the ``engine_inputs`` its forward was handed."""

    disable_torch_compile = True

    def __init__(self, ragged_config: RaggedAttentionConfig | None = None):
        super().__init__()
        self._ragged_config = ragged_config
        self.seen: list = []

    def get_ragged_attention_config(self, tp_world_size: int = 1):
        return self._ragged_config

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        return NodeInputs()

    def forward(self, graph_walk, engine_inputs, **kwargs):
        self.seen.append(engine_inputs)
        return {"out": [torch.zeros(1)]}


def _engine(submodules: dict[str, NodeSubmodule]) -> StatelessEngine:
    engine = StatelessEngine(make_enc_dec_config(torch.bfloat16))
    engine.load_model(
        submodules,
        WorkerParallelGroups(num_workers=1, global_rank=0),
        torch.device("cpu"),
    )
    return engine


def _batch(node_name: str, request_ids: list[str]) -> NodeBatch:
    return NodeBatch(
        node_name=node_name,
        graph_walk="encode",
        request_ids=list(request_ids),
        per_request_input_tensors={rid: {} for rid in request_ids},
        per_request_info={rid: None for rid in request_ids},
    )


# --------------------------------------------------------------------------- #
# submodule opt-in
# --------------------------------------------------------------------------- #


def test_submodules_declare_no_ragged_attention_by_default():
    class Plain(NodeSubmodule):
        def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
            return NodeInputs()

        def forward(self, graph_walk, engine_inputs, **kwargs):
            return {}

    assert Plain().get_ragged_attention_config() is None


def test_config_defaults():
    config = RaggedAttentionConfig(
        num_qo_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM
    )
    assert config.causal is False           # encoders are bidirectional
    assert config.sm_scale is None          # derived from the TRUE head dim
    assert config.make_attn_state is None   # default state
    # No per-request bounds → opts out of captured ragged attention.
    assert config.max_segments_for(4) is None
    assert config.max_tokens_for(4) is None


def test_capture_bounds_scale_with_batch_size():
    """Segments are spans, not requests — one clip window-chunks into several."""
    config = RaggedAttentionConfig(
        num_qo_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM,
        max_segments_per_request=3, max_tokens_per_request=256,
    )
    assert config.max_segments_for(1) == 3
    assert config.max_segments_for(8) == 24
    assert config.max_tokens_for(8) == 2048


@pytest.mark.parametrize(
    ("mode", "eager", "cuda_graph"),
    [
        (AttentionMode.BOTH, True, True),
        (AttentionMode.EAGER, True, False),
        (AttentionMode.CUDA_GRAPH, False, True),
    ],
    ids=["both", "eager_only", "cuda_graph_only"],
)
def test_enabled_for_selects_paths(mode, eager, cuda_graph):
    config = RaggedAttentionConfig(
        num_qo_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM,
        enabled_for=mode,
    )
    assert config.eager_enabled is eager
    assert config.cuda_graph_enabled is cuda_graph


def test_enabled_for_defaults_to_both():
    config = RaggedAttentionConfig(
        num_qo_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM
    )
    assert config.enabled_for is AttentionMode.BOTH


def test_cuda_graph_only_skips_the_engines_eager_state():
    """The node keeps another backend eagerly, so it shouldn't pay a workspace."""
    submodule = _Submodule(_stub_config(enabled_for=AttentionMode.CUDA_GRAPH))
    engine = _engine({"vit": submodule})
    engine.warmup()
    assert engine._ragged.get("vit") is None
    assert "vit" not in engine._ragged.workspaces

    engine.execute_batch(_batch("vit", ["r0"]))
    assert submodule.seen[0].ragged_attention_state is None


def test_eager_only_still_builds_the_engine_state():
    submodule = _Submodule(_stub_config(enabled_for=AttentionMode.EAGER))
    engine = _engine({"encoder": submodule})
    engine.warmup()
    assert engine._ragged.get("encoder") is not None


def test_capture_bounds_are_independent():
    """The piecewise runner needs only the segment bound; it buckets tokens itself."""
    config = RaggedAttentionConfig(
        num_qo_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM,
        max_segments_per_request=3,
    )
    assert config.max_segments_for(4) == 12
    assert config.max_tokens_for(4) is None


# --------------------------------------------------------------------------- #
# building states
# --------------------------------------------------------------------------- #


def test_make_attn_state_override_is_used():
    config = _stub_config()
    state = build_ragged_attention_state(
        config, torch.zeros(8, dtype=torch.uint8), torch.device("cpu")
    )
    assert isinstance(state, _StubState)
    assert state.kwargs["config"] is config
    assert state.kwargs["use_cuda_graph"] is False


def test_make_attn_state_receives_graph_bucket_bounds():
    state = build_ragged_attention_state(
        _stub_config(), torch.zeros(8, dtype=torch.uint8), torch.device("cpu"),
        use_cuda_graph=True, max_num_segments=4, max_total_tokens=512,
    )
    assert state.kwargs["use_cuda_graph"] is True
    assert state.kwargs["max_num_segments"] == 4
    assert state.kwargs["max_total_tokens"] == 512


def test_ragged_attention_keeps_workspace_alive():
    """Kernels read the workspace; it must outlive the state built against it."""
    ragged = RaggedAttention(device=torch.device("cpu"))
    ragged.build("audio_encoder", _stub_config())
    assert ragged.get("audio_encoder") is not None
    assert ragged.workspaces["audio_encoder"].numel() == 1024


def test_ragged_attention_get_is_none_for_unknown_node():
    ragged = RaggedAttention(device=torch.device("cpu"))
    assert ragged.get("nope") is None


# --------------------------------------------------------------------------- #
# StatelessEngine plumbing
# --------------------------------------------------------------------------- #


def test_warmup_builds_state_for_declaring_node():
    submodule = _Submodule(_stub_config())
    engine = _engine({"audio_encoder": submodule})
    engine.warmup()
    assert isinstance(engine._ragged.get("audio_encoder"), _StubState)


def test_warmup_skips_node_without_config():
    engine = _engine({"plain": _Submodule(None)})
    engine.warmup()
    assert engine._ragged.get("plain") is None


def test_state_is_injected_into_engine_inputs():
    submodule = _Submodule(_stub_config())
    engine = _engine({"audio_encoder": submodule})
    engine.warmup()
    engine.execute_batch(_batch("audio_encoder", ["r0"]))

    assert len(submodule.seen) == 1
    state = submodule.seen[0].ragged_attention_state
    assert state is engine._ragged.get("audio_encoder")


def test_engine_inputs_state_is_none_without_config():
    submodule = _Submodule(None)
    engine = _engine({"plain": submodule})
    engine.warmup()
    engine.execute_batch(_batch("plain", ["r0"]))

    assert submodule.seen[0].ragged_attention_state is None


def test_each_node_gets_its_own_state():
    audio, vision = _Submodule(_stub_config()), _Submodule(_stub_config())
    engine = _engine({"audio_encoder": audio, "vision_encoder": vision})
    engine.warmup()
    engine.execute_batch(_batch("audio_encoder", ["r0"]))
    engine.execute_batch(_batch("vision_encoder", ["r1"]))

    assert audio.seen[0].ragged_attention_state is not vision.seen[0].ragged_attention_state


def test_build_failure_leaves_node_without_state():
    """A broken factory must not take the whole engine down at warmup."""
    def boom(**kwargs):
        raise RuntimeError("no attention for you")

    engine = _engine({"audio_encoder": _Submodule(_stub_config(make_attn_state=boom))})
    engine.warmup()
    assert engine._ragged.get("audio_encoder") is None


def test_tp_world_size_is_passed_to_submodule():
    seen = {}

    class TPSubmodule(_Submodule):
        def get_ragged_attention_config(self, tp_world_size: int = 1):
            seen["tp_world_size"] = tp_world_size

    engine = _engine({"encoder": TPSubmodule(None)})
    engine.warmup()
    assert seen["tp_world_size"] == 1


# --------------------------------------------------------------------------- #
# default state (real FlashInfer)
# --------------------------------------------------------------------------- #

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="default ragged state requires CUDA"
)


@requires_cuda
def test_default_state_is_a_ragged_prefill_wrapper():
    from mstar.utils.flashinfer_utils import RaggedPrefillWrapper

    config = RaggedAttentionConfig(
        num_qo_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM
    )
    device = torch.device("cuda:0")
    state = build_ragged_attention_state(
        config,
        torch.empty(config.workspace_bytes, dtype=torch.uint8, device=device),
        device,
    )
    assert isinstance(state, RaggedPrefillWrapper)
    assert isinstance(state, AttentionState)   # satisfies the protocol
    assert state.num_qo_heads == NUM_HEADS
    assert state.head_dim == HEAD_DIM
    assert state.sm_scale == pytest.approx(HEAD_DIM ** -0.5)
    assert state.causal is False


@requires_cuda
def test_default_state_plans_and_runs():
    config = RaggedAttentionConfig(
        num_qo_heads=NUM_HEADS, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM
    )
    device = torch.device("cuda:0")
    ragged = RaggedAttention(device=device)
    state = ragged.build("audio_encoder", config)

    seg_lens = [128, 64, 32]
    cu = torch.tensor([0, 128, 192, 224], dtype=torch.int32)
    qkv = tuple(
        torch.randn(sum(seg_lens), NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        for _ in range(3)
    )
    state.plan(cu)
    out = state.run(*qkv)
    assert out.shape == qkv[0].shape
