"""LingBot World declares its state, and the engine builds the pool, the reads and the draws' state from it.

`main` kept LingBot's rolling K/V in a store of its own, attended both pairs through one ragged resource and held the
text K/V, the generator, the noise and the VAE caches in the submodule's per-request state. Now the dit declares a paged
array whose kept-window read reaches the sink and the latest window, a text array whose queries are the chunk's, and a
Value of the draws' state, and the VAE caches ride the rollout loop's edges. The hand-written specs and draws below are
`main`'s, copied from the model as it stood before the port.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, ".")

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import KVReqConfig, RaggedAttentionConfig, cross_label
from mstar.engine.resources.attn.kept_window import KeptWindowAttention
from mstar.engine.resources.attn.ragged.config import KeptWindowAttentionSpec, RaggedCrossAttentionSpec
from mstar.engine.resources.kv.config import DEFAULT_PAGE_SIZE, KVSpec, RetentionPolicy
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import StepContext
from mstar.engine.state import Bidirectional
from mstar.engine.state.adapters import StateSetup, build_specs, kv_layouts, node_handles, state_steps
from mstar.model.base import serving_resources
from mstar.model.components import wan_vae
from mstar.model.lingbot_world.components.dit import LingBotCausalDiT
from mstar.model.lingbot_world.config import LingBotWorldConfig
from mstar.model.lingbot_world.lingbot_world_model import REFERENCE_MODEL_KWARGS, LingBotWorldModel
from mstar.model.lingbot_world.submodules import (
    DIT_KV,
    PROGRESS,
    ROLLOUT_LOOP_NAME,
    ROLLOUT_WALK,
    SELF_ATTN,
    TEXT,
    TEXT_ATTN,
    EncoderStream,
    LingBotDitSubmodule,
    decoder_cache,
    to_edge,
)

# `main`'s ragged attention over both pairs, and its labels (`submodules.py` at the port's base)
MAIN_ATTN = RaggedAttentionConfig(num_qo_heads=12, num_kv_heads=12, head_dim=128, dtype=torch.bfloat16)
MAIN_CHUNK, MAIN_WINDOW, MAIN_TEXT = "chunk", "window", "text"


def _model(**kwargs) -> LingBotWorldModel:
    return LingBotWorldModel(skip_weight_loading=True, **kwargs)


def _tiny(**fields) -> LingBotWorldConfig:
    """Two blocks of two heads, a 2-frame sink and 10 frames attended: a chunk reaches past the window at chunk 2."""
    return LingBotWorldConfig(**{
        "dim": 32, "ffn_dim": 64, "num_heads": 2, "num_layers": 2, "freq_dim": 16, "text_dim": 8, "text_len": 8,
        "rope_max_frames": 64, "local_attn_size": 10, "sink_size": 2, "max_chunks": 8, **fields,
    })


def _main_attended_frames(cfg: LingBotWorldConfig, chunk: int) -> int:
    """`main`'s ``attended_frames``: the chunk's own frames and what its store kept."""
    frames = (chunk + 1) * cfg.chunk_size
    return min(frames, cfg.local_attn_size) if cfg.local_attn_size > -1 else frames


class _Sdpa:
    """A ragged kernel on the CPU: each call attends all of k and v, as a step of one row does, and records it."""

    force_double_buffer = False

    def __init__(self):
        self.calls, self.steps = [], []

    def plan(self, step, ctx):
        self.steps.append(step)

    # a kernel, which a compiled block calls behind a break
    @torch.compiler.disable
    def run(self, q, k, v, label=None):
        self.calls.append(SimpleNamespace(label=label, q=q.shape, k=k.shape, dtype=q.dtype, keys=k.clone()))
        return F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)).transpose(0, 1)


class _Rollout:
    """The dit node on the CPU over its declared state: a pool of 8-entry pages, the kept-window read and the text read
    on SDPA stand-ins for the ragged kernel, and the engine's host fields. A chunk runs as the engine steps it."""

    LAT = (4, 4)

    def __init__(self, cfg: LingBotWorldConfig, pages: int = 64):
        self.cfg = cfg
        model = _model()
        model.config = cfg
        self.model = model
        self.setup = StateSetup()
        kv_spec = next(spec for spec in build_specs(model.declare_state()) if spec.resource_key == DIT_KV)
        self.pool = KVManager(
            cfg=replace(kv_spec.config, page_size=8, max_num_pages=pages), name=DIT_KV, joint_comm_group=None,
            transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")), device=torch.device("cpu"),
            dtype=torch.float32,
        )
        # as the worker sets it from the node's writes, so a rollout's footprint counts its chunks
        self.pool.local_loops = {ROLLOUT_LOOP_NAME}
        self.window, self.text = _Sdpa(), _Sdpa()
        self.attention = KeptWindowAttention(self.window, DIT_KV)
        self.attention.bind_dependencies({DIT_KV: self.pool})
        # reference numerics build the pool alone
        self.planned = {} if cfg.reference_numerics else {SELF_ATTN: self.attention, TEXT_ATTN: self.text}
        self.handles = node_handles(model, "dit", {DIT_KV: self.pool, **self.planned}, setup=self.setup)
        torch.manual_seed(0)
        self.dit = LingBotCausalDiT(cfg).eval()
        self.sub = LingBotDitSubmodule(None, None, cfg)
        self.sub.dit = self.dit
        self.sub.bind_node_state(self.handles)
        self.chunks: dict[int, int] = {}

    def open(self, rid: int):
        """Take ``rid`` in, as intake and the engine's queue do: None once the pool allots it, else the mark it waits
        on."""
        (row,) = kv_layouts(self.model.declare_state(), self.model.state_layout(*self.LAT), {})[DIT_KV].values()
        request = KVReqConfig(needed_labels=["main"])
        request.apply_conductor_config(kv_layout={"main": row.row()},
                                       loop_caps={ROLLOUT_LOOP_NAME: self.cfg.max_chunks})
        # the worker finds no rebuild for the dit, so the pool guarantees a rollout its footprint
        request.rebuildable = False
        self.pool.ingest_request(rid, overrides=request)
        self.setup.host_fields.open(rid)
        self.chunks[rid] = 0
        self.pool.ask(rid, "dit", ROLLOUT_WALK)
        return self.pool.reserve(rid, "dit", ROLLOUT_WALK, None)

    def info(self, rid: int, seed: int = 7) -> CurrentForwardPassInfo:
        info = CurrentForwardPassInfo(
            request_id=str(rid), graph_walk=ROLLOUT_WALK, fwd_index=0, random_seed=seed, max_tokens=0,
            step_metadata={"num_latent_frames": self.cfg.max_chunks * self.cfg.chunk_size}, rid_handle=rid,
        )
        info.dynamic_loop_iter_counts[ROLLOUT_LOOP_NAME] = self.chunks[rid]
        return info

    def inputs(self, rid: int):
        lat_h, lat_w = self.LAT
        return self.sub.prepare_inputs(ROLLOUT_WALK, self.info(rid), {
            "image_inputs": [torch.zeros(3, 8 * lat_h, 8 * lat_w)], "text_embeds_pos": [torch.randn(3, 8)],
            "camera_poses": [torch.zeros(1)], "intrinsics": [torch.zeros(4)],
        })

    def admit(self, rid: int, inputs):
        """The chunk's step, admitted and planned, or None where the pool refused it."""
        step = state_steps(self.sub.declare_step(ROLLOUT_WALK, [rid], [inputs]), self.handles, graph_walk=ROLLOUT_WALK)
        ctx = StepContext(request_ids=(rid,), graph_walk=ROLLOUT_WALK, slot=0, capture=False)
        step.set_ctx(ctx)
        outcome = self.pool.admit(step.steps[DIT_KV], ctx)
        if not outcome.ok:
            return outcome
        ctx.plan_results = {DIT_KV: self.pool.plan(step.steps[DIT_KV], ctx)}
        for key, resource in self.planned.items():
            resource.plan(step.steps[key], ctx)
        return step

    def chunk(self, rid: int) -> list[torch.Tensor]:
        """One chunk of ``rid``: two denoise passes and the commit pass, then the commits. Returns each pass's flow."""
        inputs = self.inputs(rid)
        step = self.admit(rid, inputs)
        cs, (lat_h, lat_w) = self.cfg.chunk_size, self.LAT
        generator = torch.Generator().manual_seed(self.chunks[rid])
        step.begin_calls()
        with torch.inference_mode():
            text_kv = self.dit.text_kv(torch.randn(3, 8, generator=generator))
            conditioning = self.dit.chunk_conditioning(torch.zeros(1, 384, cs, lat_h, lat_w), self.chunks[rid] * cs)
            y = torch.randn(20, cs, lat_h, lat_w, generator=generator)
            latent, t = inputs.tensor_inputs["latent"], torch.tensor([500.0])
            flows = [self.dit(latent, t, text_kv, y, conditioning, keep=keep) for keep in (False, False, True)]
        step.check_calls()
        self.pool.commit(step.steps[DIT_KV], step.ctx)
        step.commit_calls()
        step.end_calls()
        self.chunks[rid] += 1
        return flows


# The declaration


def test_the_specs_are_main_s_resources_but_three():
    """`main` built a rolling store and one ragged resource for both pairs. The declaration builds, in their place, a
    paged pool under the same key, a kept-window resource over it for the chunk's pair, and a second ragged resource for
    the text pair, each with `main`'s attention geometry. Their dtype is the model's bf16, which `main` set per spec."""
    model = _model()
    kv, window, text = serving_resources(model, {"resources": {}})
    assert isinstance(kv, KVSpec) and (kv.resource_key, kv.nodes) == (DIT_KV, {"dit"})
    assert (kv.config.num_layers, kv.config.num_kv_heads, kv.config.head_dim) == (30, 12, 128)
    assert kv.config.max_num_pages == 509, "the pool should hold one rollout's window at 720x1280"
    assert isinstance(window, KeptWindowAttentionSpec) and window.kv_cache == DIT_KV
    assert isinstance(text, RaggedCrossAttentionSpec) and text.resource_key == TEXT_ATTN
    for spec in (window, text):
        assert spec.nodes == {"dit"}
        assert replace(spec.config, dtype=MAIN_ATTN.dtype) == MAIN_ATTN, f"{spec.resource_key}: main's geometry"
    assert model.get_autocast_dtype() is torch.bfloat16


def test_neither_node_runs_under_the_model_s_bf16(monkeypatch):
    """The model's dtype sizes the pool and the kernels. The DiT enters the reference's autocast contexts itself and
    the text encoder runs as it loads, so each node opts out and keeps `main`'s numerics."""
    import transformers

    from mstar.engine.node_dtype import resolve_node_dtype

    monkeypatch.setattr(transformers.UMT5EncoderModel, "from_pretrained", lambda *args, **kwargs: torch.nn.Module())
    model = LingBotWorldModel()
    text_encoder = model.get_submodule("text_encoder")
    for node in (text_encoder, LingBotDitSubmodule(None, None, model.config)):
        assert resolve_node_dtype(node, model.get_autocast_dtype()) is None, type(node).__name__


def test_the_reference_numerics_read_with_no_op_and_declare_no_text():
    """FA2 attends what the read reaches and the step's own text K/V, so the pool is the only resource."""
    model = _model(**REFERENCE_MODEL_KWARGS)
    kv, progress = model.declare_state()
    assert {a.name for a in model.declare_state()} == {DIT_KV, PROGRESS}
    (read,) = kv.reads.values()
    assert read.op is None and read.mask == Bidirectional(kept_window=True)
    assert [spec.resource_key for spec in serving_resources(model, {"resources": {}})] == [DIT_KV]


@pytest.mark.parametrize(("image_hw", "frame"), [((464, 832), 1508), ((720, 1280), 3600)])
def test_the_layout_keeps_the_sink_and_the_window(image_hw, frame):
    """A chunk's tokens an iteration of the rollout loop, of which the pool keeps a 6-frame sink and an 8-frame window
    exactly, and the read reaches the chunk's 4 frames past them: `main`'s 18 attended frames."""
    model = _model()
    lat_h, lat_w = image_hw[0] // 8, image_hw[1] // 8
    assert model.config.frame_tokens(lat_h, lat_w) == frame
    (layout,) = kv_layouts(model.declare_state(), model.state_layout(lat_h, lat_w), {})[DIT_KV].values()
    assert layout.growth == ((ROLLOUT_LOOP_NAME, 4 * frame),)
    assert layout.retention == RetentionPolicy(context_budget=8 * frame, protected_prefix=6 * frame, exact=True)


@pytest.mark.parametrize(("image_hw", "pages"), [((464, 832), 214), ((720, 1280), 508)])
def test_max_num_pages_is_admission_s_count(image_hw, pages):
    """The model restates the pool's count, which it can't import, so a rollout counted through intake's route, from
    its layout's row at the loop's cap, holds as many pages as the model sizes for, and the pool one more, its sink."""
    model = _model()
    cfg = model.config
    (layout,) = kv_layouts(model.declare_state(), model.state_layout(image_hw[0] // 8, image_hw[1] // 8), {})[
        DIT_KV].values()
    request = KVReqConfig(needed_labels=["main"])
    request.apply_conductor_config(kv_layout={"main": layout.row()}, loop_caps={ROLLOUT_LOOP_NAME: cfg.max_chunks})
    (kv, *_) = serving_resources(model, {"resources": {}})
    caps = {ROLLOUT_LOOP_NAME: cfg.max_chunks}
    assert request.pages(None, caps, kv.config) == pages
    assert cfg.window_pages(cfg.frame_tokens(image_hw[0] // 8, image_hw[1] // 8), DEFAULT_PAGE_SIZE) == pages
    if image_hw == (720, 1280):
        assert model.kernel_settings() == {DIT_KV: {"max_num_pages": pages + 1}}


def test_the_largest_area_bounds_a_frame_at_any_aspect():
    model = _model()
    for image_hw in [(720, 1280), (1280, 720), (960, 960), (408, 2256), (2256, 408), (480, 832)]:
        assert model.config.frame_tokens(*model.geometry(image_hw, 720, 1280)) <= model.config.max_frame_tokens


@pytest.mark.parametrize("image_hw", [(464, 832), (720, 1280)])
def test_a_rollout_past_the_pool_is_refused_at_intake(image_hw):
    """A deployment that serves the default size sets 215 pages, and a larger rollout is a 400 before any step."""
    model = _model()
    (layout,) = kv_layouts(model.declare_state(), model.state_layout(image_hw[0] // 8, image_hw[1] // 8), {})[
        DIT_KV].values()
    request = KVReqConfig(needed_labels=["main"])
    request.apply_conductor_config(kv_layout={"main": layout.row()},
                                   loop_caps={ROLLOUT_LOOP_NAME: model.config.max_chunks})
    (kv, *_) = serving_resources(model, {"resources": {DIT_KV: {"max_num_pages": 215}}})
    if image_hw == (464, 832):
        request.validate(kv)
    else:
        with pytest.raises(ValueError, match="pages"):
            request.validate(kv)


def test_a_rollout_that_fits_waits_for_the_pages_another_holds():
    """Two rollouts on a pool of one's window: the engine's queue holds the second until the first is removed."""
    rollout = _Rollout(_tiny(), pages=1 + _tiny().window_pages(4, 8))
    assert rollout.open(1) is None
    assert rollout.open(2) is not None, "the second rollout should wait for the first's pages"
    for _ in range(3):
        rollout.chunk(1)
    rollout.pool.remove_request(1)
    assert rollout.pool.reserve(2, "dit", ROLLOUT_WALK, None) is None
    rollout.chunk(2)


# The steps and the reads


def test_a_chunk_s_step_declares_its_tokens_its_text_and_its_draws():
    rollout = _Rollout(_tiny())
    rollout.open(1)
    inputs = rollout.inputs(1)
    declared = rollout.sub.declare_step(ROLLOUT_WALK, [1], [inputs])
    assert [(s.label, s.span) for s in declared.segments] == [("main", 4 * 4)]
    assert [(s.label, s.span) for s in declared.state[TEXT].segments] == [("text", 8)]
    assert set(declared.state[PROGRESS].host[(1, "main", 0)]) == {"rng"}
    step = state_steps(declared, rollout.handles, graph_walk=ROLLOUT_WALK)
    assert set(step.steps) == {DIT_KV, SELF_ATTN, TEXT_ATTN}
    assert step.steps[TEXT_ATTN].pairs == (("main", "text"),)
    reference = _Rollout(_tiny(reference_numerics=True))
    assert TEXT not in reference.sub.declare_step(ROLLOUT_WALK, [1], [inputs]).state


def test_the_reads_call_their_kernels_as_main_s_one_resource_did():
    """`main` ran both pairs on one ragged resource: the chunk's queries over its window, and over the text. Each read
    calls its own resource's kernel with `main`'s shapes, dtype and segment lengths, under labels of its own."""
    cfg = _tiny()
    rollout = _Rollout(cfg)
    rollout.open(1)
    frame, heads, dim = 4, cfg.num_heads, cfg.head_dim
    for _ in range(4):
        rollout.chunk(1)
    rollout.window.calls.clear()
    rollout.text.calls.clear()
    rollout.chunk(1)
    attended = _main_attended_frames(cfg, 4) * frame
    main_calls = {
        cross_label(MAIN_CHUNK, MAIN_WINDOW): ((4 * frame, heads, dim), (attended, heads, dim)),
        cross_label(MAIN_CHUNK, MAIN_TEXT): ((4 * frame, heads, dim), (cfg.text_len, heads, dim)),
    }
    port = {"window": rollout.window.calls, "text": rollout.text.calls}
    for (main_label, (q, k)), calls in zip(main_calls.items(), port.values(), strict=True):
        assert len(calls) == 3 * cfg.num_layers, f"each pass of each layer should call the kernel once ({main_label})"
        assert {(tuple(c.q), tuple(c.k), c.dtype) for c in calls} == {(q, k, torch.float32)}, main_label
    (window_plan, text_plan) = rollout.window.steps[-1], rollout.text.steps[-1]
    assert [s.span for s in window_plan.segments] == [4 * frame, attended]
    assert [s.span for s in text_plan.segments] == [4 * frame, cfg.text_len]


@pytest.mark.parametrize("local_attn_size", [10, -1])
def test_the_read_reaches_the_sink_and_the_latest_window(local_attn_size):
    """The frames each chunk's read reaches, in order: `main`'s store kept the sink and the latest frames up to
    ``local_attn_size``, and every frame without a window."""
    cfg = _tiny(local_attn_size=local_attn_size)
    rollout = _Rollout(cfg, pages=256)
    rollout.open(1)
    tags = []
    for chunk in range(6):
        rollout.window.calls.clear()
        rollout.chunk(1)
        tags.append(rollout.window.calls[-1].k[0] // 4)
        end = (chunk + 1) * cfg.chunk_size
        if local_attn_size == -1 or end <= local_attn_size:
            assert tags[-1] == end
        else:
            assert tags[-1] == local_attn_size
    if local_attn_size != -1:
        assert rollout.pool._streams[1]["main"].released, "the pool should release the pages behind the window"


def test_the_reference_numerics_attend_what_the_kept_window_read_attends(monkeypatch):
    """Under ``reference_numerics`` the read returns the entries it reaches, and FA2 attends them and the step's own
    text K/V. With SDPA standing in for FA2, each pass's flow equals the one the two reads give, past the window."""
    from mstar.model.lingbot_world.components import dit

    monkeypatch.setattr(dit, "_flash_attention", lambda q, k, v: F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2))
    fast, reference = _Rollout(_tiny(), pages=256), _Rollout(_tiny(reference_numerics=True), pages=256)
    reference.dit.load_state_dict(fast.dit.state_dict())
    for rollout in (fast, reference):
        rollout.open(1)
    for chunk in range(4):
        for i, (ours, theirs) in enumerate(zip(fast.chunk(1), reference.chunk(1), strict=True)):
            torch.testing.assert_close(theirs, ours, msg=f"chunk {chunk}, pass {i}")


# The draws and the caches


def _main_draws(cfg: LingBotWorldConfig, seed: int, lat_f: int, lat_hw, chunks: int):
    """`main`'s draws: the rollout's noise in one call from the seeded generator, then each chunk's re-noise draws
    from the same generator, in order."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    noise = torch.randn(cfg.out_dim, lat_f, *lat_hw, dtype=torch.float32, generator=generator)
    draws = []
    for chunk in range(chunks):
        latent = noise[:, chunk * cfg.chunk_size:(chunk + 1) * cfg.chunk_size]
        renoise = [torch.randn(latent.shape, generator=generator, dtype=torch.float32) for _ in range(3)]
        draws.append((latent, renoise))
    return draws


def test_the_draws_are_main_s_for_three_chunks():
    """Each chunk draws the rollout's noise again from the seed and keeps its own frames, and goes on from the
    generator state the last chunk wrote to ``PROGRESS``, so its draws are `main`'s bit for bit."""
    cfg = _tiny()
    rollout = _Rollout(cfg)
    rollout.open(1)
    lat_f = cfg.max_chunks * cfg.chunk_size
    for chunk, (latent, renoise) in enumerate(_main_draws(cfg, 7, lat_f, _Rollout.LAT, 3)):
        inputs = rollout.inputs(1)
        assert torch.equal(inputs.tensor_inputs["latent"], latent), f"chunk {chunk}'s initial noise"
        assert all(torch.equal(a, b) for a, b in zip(inputs.kwargs["renoise"], renoise, strict=True)), chunk
        rollout.chunk(1)


def test_a_chunk_prepared_again_draws_the_same_noise():
    """A step refused at admit is prepared again, and reads the state the last commit left."""
    rollout = _Rollout(_tiny())
    rollout.open(1)
    rollout.chunk(1)
    first, again = rollout.inputs(1), rollout.inputs(1)
    assert all(torch.equal(a, b) for a, b in zip(first.kwargs["renoise"], again.kwargs["renoise"], strict=True))
    assert first.kwargs["rng"] == again.kwargs["rng"]


def test_a_decoder_cache_comes_back_from_its_edge_a_feat_cache():
    """The compiled decode guards on the cache's type, and diffusers' markers come back as they went."""
    slots = wan_vae.FeatCache([None, "Rep", torch.arange(6.0).view(1, 2, 3)])
    back = decoder_cache(None, to_edge(slots, torch.device("cpu")))
    assert type(back) is wan_vae.FeatCache
    assert back[0] is None and back[1] == "Rep" and torch.equal(back[2], slots[2])


def test_the_encoder_stream_comes_back_from_its_edge():
    vae = SimpleNamespace(_cached_conv_counts={"encoder": 3})
    stream = EncoderStream(vae)
    stream.cache = [None, torch.ones(2), torch.zeros(1, 2)]
    back = EncoderStream(vae, stream.edge(torch.device("cpu")))
    assert back.cache[0] is None and torch.equal(back.cache[1], stream.cache[1]) and back.steady is None
    stream.cache, stream.steady = None, torch.full((1, 4), 2.0)
    settled = EncoderStream(vae, stream.edge(torch.device("cpu")))
    assert settled.cache is None and torch.equal(settled.steady, stream.steady)


# Compile


def _compiled_rollout_graphs(num_layers: int, chunks: int) -> int:
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    rollout = _Rollout(_tiny(num_layers=num_layers), pages=256)
    rollout.dit.compile_blocks(backend="eager", fullgraph=False, dynamic=False)
    rollout.open(1)
    for _ in range(chunks):
        rollout.chunk(1)
    return torch._dynamo.utils.counters["stats"]["unique_graphs"]


def test_compiled_blocks_trace_once_per_kind_of_pass():
    """The layer and the pool's plan stay out of the compiled blocks, so a rollout compiles them once for each value of
    ``keep``, not once per layer or per chunk: five chunks run past the window's fill."""
    one_chunk = _compiled_rollout_graphs(num_layers=2, chunks=1)
    assert one_chunk > 0
    assert _compiled_rollout_graphs(num_layers=4, chunks=5) == one_chunk


# The warm-up


def test_the_warmup_is_a_two_chunk_rollout_at_the_default_size(monkeypatch):
    """The shipped warm-up's image, the model repo's 1868x1040 teaser, comes out at 464x832, the size the decode warms
    at load, and its poses make two chunks."""
    from pathlib import Path

    import yaml

    from mstar.api_server.entrypoint import WarmupRequest

    model = _model()
    monkeypatch.setattr(model, "_ensure_tokenizer", lambda: None)
    monkeypatch.setattr(model, "tokenizer", lambda text, **kwargs: SimpleNamespace(input_ids=[0, 1]))
    configs = Path(__file__).parents[2] / "configs"
    for name in ("lingbot_world_1_3b.yaml", "lingbot_world_1_3b_fast.yaml"):
        (spec,) = yaml.safe_load((configs / name).read_text())["warmup_requests"]
        warmup = WarmupRequest.from_config(spec)
        ((media,),) = [warmup.files]
        assert media.modality == "image" and media.path.endswith("lingbot-world-v2-1.3b-causal-fast/assets/teaser.png")
        out = model.process_prompt(warmup.text, ["image", "text"], warmup.output_modalities,
                                   tensors={"image_inputs": [torch.zeros(3, 1040, 1868, dtype=torch.uint8)]},
                                   **warmup.model_kwargs)
        assert tuple(out.new_input_tensors["image_inputs"][0].shape[1:]) == (464, 832)
        assert model.get_loop_caps(**warmup.model_kwargs) == {ROLLOUT_LOOP_NAME: 2}
        assert np.allclose(np.asarray(warmup.model_kwargs["camera_poses"])[:, :, :3], np.eye(4)[:3, :3])
