"""Submodule checks for PersonaPlex (LM, MimiEncoder, MimiDecoder).

The LM's per-iteration logic is driven engine-free with the dense CPU attention
backend and greedy samplers, and through the engine hooks over a real recurrent
state pool; the paged-KV/FlashInfer path is only checked for its wiring here (fake
resources). Fixture-backed tests replay the NVIDIA fixtures of
test_personaplex_components.py and test_personaplex_mimi.py.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from personaplex_fixture_utils import load_binary_fixture
from torch.utils._python_dispatch import TorchDispatchMode

from mstar.engine.node_dtype import resolve_node_dtype
from mstar.engine.resources import Segment, SubmoduleStep
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent import RecurrentStep
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext
from mstar.model.personaplex import config as cfg
from mstar.model.personaplex.components.delayed_cache import CacheRows
from mstar.model.personaplex.components.temporal import DenseTemporalCache, apply_rope_adjacent
from mstar.model.personaplex.personaplex_model import PersonaPlexModel
from mstar.model.personaplex.submodules import (
    LM_WALKS,
    LMSubmodule,
    MimiDecoderSubmodule,
    MimiEncoderSubmodule,
    ResourceTemporalBackend,
    _sampled_token_ids,
)
from mstar.model.personaplex.weights import PersonaPlexLM, build_lm, reference_lm_key
from mstar.model.submodule_base import BatchedModelOutput, ModelInputsFromEngine

FIXTURES = Path(__file__).parent / "fixtures/personaplex"
TINY = cfg.LMArchConfig(
    dim=32, num_heads=4, num_layers=2, ffn_hidden=40, depformer_dim=16, depformer_heads=2,
    depformer_layers=2, depformer_ffn_hidden=24,
)
EIGHT = cfg.AUDIO_CODEBOOKS_PER_STREAM


def _greedy_text(logits):
    return logits.float().argmax(-1)


def _greedy_depth(logits, _step):
    return logits.float().argmax(-1)


def _meta(n_emb, n_text, n_user):
    return {"num_voice_embeddings": n_emb, "num_text_tokens": n_text, "num_user_frames": n_user,
            "num_samples": n_user * 1920, "conditioning_frames": n_emb + 12 + n_text}


def _fixture(env, name):
    configured = os.environ.get(env)
    path = Path(configured) if configured else FIXTURES / name
    if not path.is_file() and not configured:
        pytest.skip(f"needs {name} from the NVIDIA oracle; see fixtures/personaplex/README.rst or set {env}")
    return load_binary_fixture(path)


class _Run:
    """Drive one request's LM frames engine-free, the way the conditioning and generate walks do."""

    def __init__(self, sub, meta, voice, persona, user_codes):
        self.sub, self.meta, self.codes = sub, meta, user_codes
        self.backend = DenseTemporalCache(sub.lm.config.num_layers, sub.lm.config.rope_theta)
        self.static = {"voice_embeddings": [voice["embeddings"]], "voice_cache": [voice["cache"]],
                       "persona_tokens": [persona], "user_audio_codes": [user_codes]}
        self.progress = sub.start_conditioning(meta, self.static)
        self.rows, self.index = None, 0

    @property
    def conditioning(self):
        return self.index < self.meta["conditioning_frames"]

    @property
    def total(self):
        return self.meta["conditioning_frames"] + self.meta["num_user_frames"]

    def user(self):
        return self.codes[:, :, self.index - self.meta["conditioning_frames"]]

    def step(self):
        """One frame; returns the frame's emitted outputs (none while conditioning)."""
        if self.conditioning:
            frame = self.sub.conditioning_frame(self.progress, self.meta, self.static)
            self.sub.lm.temporal.forward_embeddings(frame.embeds, self.backend, text_logits=False)
            self.rows, out = frame.rows, {}
        else:
            frame = self.sub.generate_frame(self.rows, self.user(), self.backend, _greedy_text, _greedy_depth)
            self.rows = frame.rows
            out = {"agent_frame": [frame.agent_frame], "text_frame": [frame.text_frame]}
        self.backend.advance()
        self.index += 1
        return out

    def run_conditioning(self):
        while self.conditioning:
            self.step()
        return self

    def run(self):
        return [self.step() for _ in range(self.total - self.index)]


def _tiny_submodule(seed=0):
    torch.manual_seed(seed)
    return LMSubmodule(PersonaPlexLM(TINY).eval())


def _random_inputs(n_emb=3, n_text=4, n_user=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    voice = {"embeddings": torch.randn(n_emb, 1, 1, TINY.dim, generator=g),
             "cache": torch.randint(0, 2048, (1, 17, 4), generator=g)}
    persona = torch.randint(4, 32000, (n_text,), generator=g)
    codes = torch.randint(0, 2048, (1, 8, n_user), generator=g)
    return voice, persona, codes, _meta(n_emb, n_text, n_user)


def _assert_same_outputs(got, want, what):
    assert got.keys() == want.keys(), what
    for name in want:
        for value, reference in zip(got[name], want[name], strict=True):
            assert torch.equal(value, reference), (what, name)


class _RowBackends:
    """A batched temporal forward over per-request dense KV: row ``i`` uses ``backends[i]``."""

    def __init__(self, backends):
        self.backends = backends

    def begin(self):
        pass

    def rope(self, q, k):
        pairs = [b.rope(q[i : i + 1], k[i : i + 1]) for i, b in enumerate(self.backends)]
        return torch.cat([q for q, _ in pairs]), torch.cat([k for _, k in pairs])

    def attend(self, layer_idx, q, k, v):
        return torch.cat([b.attend(layer_idx, q[i : i + 1], k[i : i + 1], v[i : i + 1])
                          for i, b in enumerate(self.backends)])

    def advance(self):
        for backend in self.backends:
            backend.advance()


class _GreedySampler:
    def sample(self, request_ids, logits):
        return logits.float().argmax(-1)


def _cat_rows(rows):
    return CacheRows(*(torch.cat(parts) for parts in zip(*rows, strict=True)))


class _Engine:
    """The engine's path for the LM over a real state pool: prepare_inputs -> preprocess -> forward -> commit.

    Temporal attention stays per-request dense KV, since that resource needs a GPU.
    """

    def __init__(self, sub, max_slots=16):
        self.sub = sub
        spec = next(s for s in PersonaPlexModel(skip_weight_loading=True).get_node_resources()
                    if s.resource_key == cfg.LM_CACHE)
        spec.config.max_slots = max_slots
        self.pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
        self.runner = StepRunner({cfg.LM_CACHE: self.pool})
        self.added, self.iters, self.handles, self.backends = set(), {}, {}, {}
        self.dense = (TINY.num_layers, TINY.rope_theta)

    def add(self, rid):
        self.pool.ingest_request(rid)
        self.added.add(rid)
        self.iters[rid] = {}
        self.handles[rid] = len(self.handles)
        self.backends[rid] = DenseTemporalCache(*self.dense)

    def remove(self, rid):
        self.pool.remove_request(rid)
        self.added.remove(rid)

    def info(self, rid, meta):
        return SimpleNamespace(step_metadata=meta, dynamic_loop_iter_counts=self.iters[rid],
                               rid_handle=self.handles[rid])

    def prepare(self, walk, rid, meta, inputs):
        return self.sub.prepare_inputs(walk, self.info(rid, meta), inputs)

    def step(self, walk, rids, node_inputs, batched=None, bucket=None, text=None, depth=None):
        """Run one step. ``bucket`` pads it to a captured size with the capture config's template rows."""
        loop = cfg.CONDITIONING_LOOP if walk == cfg.CONDITIONING_WALK else cfg.DECODE_LOOP
        ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False)
        rows, inputs = list(rids), list(node_inputs)
        backends = [self.backends[rid] for rid in rids]
        if bucket is not None:
            (config,) = self.sub.get_accelerator_graph_configs(torch.device("cpu"))
            padding = config.get_node_inputs(bucket - len(rids), 0)
            rows += [f"pad{i}" for i in range(len(padding))]
            inputs += padding
            backends += [DenseTemporalCache(*self.dense) for _ in padding]
            ctx.slot_lease = SlotLease(slot=0, bucket=BucketKey(walk, bucket, bucket, 0))
            ctx.set_padded_rids(tuple(rows))
        step = SubmoduleStep(segments=[Segment(request_id=rid, label=cfg.KV_LABEL, span=1) for rid in rows],
                             steps={cfg.LM_CACHE: RecurrentStep()})
        step.set_ctx(ctx)
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        self.sub._backend = _RowBackends(backends)
        engine_inputs = ModelInputsFromEngine(
            request_ids=rows, per_request_info={}, step=step,
            resources={cfg.LM_CACHE: self.pool, cfg.TEXT_SAMPLER: text or _GreedySampler(),
                       cfg.DEPTH_SAMPLER: depth or _GreedySampler()},
        )
        staged = self.sub.preprocess(walk, engine_inputs, inputs)
        if batched if batched is not None else len(rows) > 1:
            raw = self.sub.forward_batched(walk, engine_inputs, **staged)
            assert isinstance(raw, BatchedModelOutput) and raw.check_stop_buffers is None
            produced = raw.clone_row_outputs(len(rids))
            outputs = {rid: {name: [tensor[i : i + 1]] for name, tensor in produced.items()}
                       for i, rid in enumerate(rids)}
        else:
            outputs = {rids[0]: self.sub.forward(walk, engine_inputs, **staged)}
        self.runner.commit(step)
        for rid in rids:
            self.iters[rid][loop] = self.iters[rid].get(loop, 0) + 1
            self.backends[rid].advance()
        return outputs


def _class_reference_frame(sub, cache, user, backend):
    """A generate frame through ``DelayedTokenCache`` with only the depth steps it needs: an independent control."""
    assert cache.prepare(user=user[:, :, None])
    dtype = sub.lm.temporal.text_emb.weight.dtype
    embeds = sub.lm.temporal.embed_codes(cache.model_input)[:, 0].to(dtype)
    target, forced = cache.target[:, :, 0].clone(), cache.target_provided[:, :, 0].clone()
    hidden, logits = sub.lm.temporal.forward_embeddings(embeds, backend)
    text = _greedy_text(logits)
    free = (~forced[0, 1:]).nonzero()
    steps = int(free.max()) + 1 if len(free) else 0
    audio = sub.lm.depth.run_frame(torch.where(forced[:, 0], target[:, 0], text), hidden, target[:, 1:], forced[:, 1:],
                                   _greedy_depth, steps=steps)
    output = cache.commit(text, audio)
    return {"agent_frame": [output[:, 1 : 1 + EIGHT]], "text_frame": [output[:, 0, 0]]}


# --------------------------------------------------------------------------- fixture-free


@torch.no_grad()
def test_shared_lm_interleaved_requests_match_isolated_runs():
    """Cache state stays per request, including different voice and persona lengths."""
    sub = _tiny_submodule(seed=31)
    cases = [_random_inputs(n_emb=1, n_text=2, n_user=3, seed=32),
             _random_inputs(n_emb=2, n_text=1, n_user=4, seed=33)]
    expected = [_Run(sub, meta, voice, persona, codes).run() for voice, persona, codes, meta in cases]
    runs = [_Run(sub, meta, voice, persona, codes) for voice, persona, codes, meta in cases]
    actual = [[], []]
    # Alternate requests; the shorter one finishes while the other continues.
    for index in range(max(run.total for run in runs)):
        for request, run in enumerate(runs):
            if index < run.total:
                actual[request].append(run.step())
    for request in range(2):
        for step, (got, want) in enumerate(zip(actual[request], expected[request], strict=True)):
            _assert_same_outputs(got, want, (request, step))


@torch.no_grad()
def test_batched_generate_frames_match_isolated_runs():
    """Rows at different offsets share a step and still match each request run alone."""
    sub = _tiny_submodule(seed=41)
    cases = [_random_inputs(n_emb=1, n_text=2, n_user=3, seed=42),
             _random_inputs(n_emb=2, n_text=1, n_user=4, seed=43),
             _random_inputs(n_emb=1, n_text=3, n_user=2, seed=44)]
    expected = [_Run(sub, c[3], *c[:3]).run_conditioning().run() for c in cases]
    runs = [_Run(sub, meta, voice, persona, codes).run_conditioning() for voice, persona, codes, meta in cases]
    assert len({int(run.rows.offset) for run in runs}) > 1
    for tick in range(max(len(want) for want in expected)):
        live = [i for i in range(3) if tick < len(expected[i])]
        rows = _cat_rows([runs[i].rows for i in live])
        user = torch.cat([runs[i].user() for i in live])
        backend = _RowBackends([runs[i].backend for i in live])
        frame = sub.generate_frame(rows, user, backend, _greedy_text, _greedy_depth)
        for row, i in enumerate(live):
            runs[i].backend.advance()
            runs[i].index += 1
            runs[i].rows = CacheRows(*(field[row : row + 1] for field in frame.rows))
            got = {"agent_frame": [frame.agent_frame[row : row + 1]], "text_frame": [frame.text_frame[row : row + 1]]}
            _assert_same_outputs(got, expected[i][tick], (tick, i))


@torch.no_grad()
@pytest.mark.parametrize("batch", [1, 3])
def test_engine_path_over_the_pool_matches_the_class_reference_over_whole_requests(batch):
    """prepare_inputs -> preprocess -> forward over the real pool, every frame, vs ``DelayedTokenCache``.

    Requests start at different frames, so batches hold rows at different offsets.
    """
    sub = _tiny_submodule(seed=71)
    cases = [_random_inputs(n_emb=1 + i % 2, n_text=2 + i, n_user=6, seed=171 + i) for i in range(batch)]
    engine = _Engine(sub)
    refs = [_Run(sub, meta, voice, persona, codes) for voice, persona, codes, meta in cases]
    rids = [f"r{i}" for i in range(batch)]
    for rid in rids:
        engine.add(rid)
    starts = [2 * i for i in range(batch)]
    for tick in range(max(start + run.total for start, run in zip(starts, refs, strict=True))):
        for walk in (cfg.CONDITIONING_WALK, cfg.GENERATE_WALK):
            conditioning = walk == cfg.CONDITIONING_WALK
            live = [i for i in range(batch) if 0 <= tick - starts[i] < refs[i].total
                    and (tick - starts[i] < refs[i].meta["conditioning_frames"]) == conditioning]
            if not live:
                continue
            inputs = [engine.prepare(walk, rids[i], refs[i].meta, refs[i].static) for i in live]
            got = engine.step(walk, [rids[i] for i in live], inputs)
            for i in live:
                run = refs[i]
                if conditioning:
                    run.step()
                    assert set(got[rids[i]]) == {cfg.CONDITIONING_TICK}
                else:
                    want = _class_reference_frame(sub, run.progress.cache, run.user(), run.backend)
                    run.backend.advance()
                    run.index += 1
                    _assert_same_outputs(got[rids[i]], want, f"tick {tick} request {i}")
    for i, rid in enumerate(rids):
        # An overshoot iteration is skipped without reading anything from the device.
        assert engine.prepare(cfg.GENERATE_WALK, rid, refs[i].meta, refs[i].static) is None


@torch.no_grad()
def test_a_recycled_slot_starts_from_the_new_request_alone():
    sub = _tiny_submodule(seed=74)
    first, second = _random_inputs(n_emb=1, n_text=1, n_user=3, seed=174), _random_inputs(n_emb=2, n_text=2, seed=175)
    engine = _Engine(sub, max_slots=2)

    def serve(rid, case):
        voice, persona, codes, meta = case
        run = _Run(sub, meta, voice, persona, codes)
        engine.add(rid)
        outputs = []
        for index in range(run.total):
            walk = cfg.CONDITIONING_WALK if index < meta["conditioning_frames"] else cfg.GENERATE_WALK
            inputs = engine.prepare(walk, rid, meta, run.static)
            outputs.append(engine.step(walk, [rid], [inputs], batched=False)[rid])
        engine.remove(rid)
        return outputs[meta["conditioning_frames"]:]

    serve("first", first)
    got = serve("second", second)
    voice, persona, codes, meta = second
    want = _Run(sub, meta, voice, persona, codes).run_conditioning().run()
    for step, (actual, expected) in enumerate(zip(got, want, strict=True)):
        _assert_same_outputs(actual, expected, step)


@torch.no_grad()
def test_forward_batched_gives_each_request_its_own_rows_and_samples_every_depth_step():
    class Sampler:
        def __init__(self):
            self.shapes, self.tokens = [], []

        def sample(self, request_ids, logits):
            assert request_ids == ["a", "b"]
            self.shapes.append(tuple(logits.shape))
            self.tokens.append(logits.argmax(-1))
            return self.tokens[-1]

    sub = _tiny_submodule(seed=31)
    cases = [_random_inputs(n_emb=1, n_text=1, n_user=2, seed=52), _random_inputs(n_emb=2, n_text=3, n_user=2, seed=53)]
    engine = _Engine(sub)
    runs = dict(zip("ab", (_Run(sub, c[3], *c[:3]) for c in cases), strict=True))
    want = {rid: _Run(sub, c[3], *c[:3]).run_conditioning().step()
            for rid, c in zip("ab", cases, strict=True)}
    for rid, run in runs.items():
        engine.add(rid)
        for _ in range(run.meta["conditioning_frames"]):
            inputs = engine.prepare(cfg.CONDITIONING_WALK, rid, run.meta, run.static)
            engine.step(cfg.CONDITIONING_WALK, [rid], [inputs], batched=False)
    inputs = [engine.prepare(cfg.GENERATE_WALK, rid, run.meta, run.static) for rid, run in runs.items()]
    text, depth = Sampler(), Sampler()
    got = engine.step(cfg.GENERATE_WALK, ["a", "b"], inputs, text=text, depth=depth)
    # The rows sample different text, so handing one row the other's tokens would show.
    assert text.tokens[0][0] != text.tokens[0][1]
    assert text.shapes == [(2, TINY.text_card)] and depth.shapes == [(2, TINY.card)] * EIGHT
    for rid in "ab":
        _assert_same_outputs(got[rid], want[rid], rid)
        assert got[rid]["agent_frame"][0].shape == (1, 8, 1) and got[rid]["text_frame"][0].shape == (1,)


@torch.no_grad()
def test_generated_text_frame_serializes_for_worker_transfer():
    """A B=1 text column must have byte-viewable strides for worker transfer."""
    from mstar.communication.tensors import _serialize_tensor

    sub = _tiny_submodule()
    voice, persona, codes, meta = _random_inputs(n_user=1)
    text = _Run(sub, meta, voice, persona, codes).run_conditioning().step()["text_frame"][0]
    assert text.shape == (1,) and text.dtype == torch.long
    assert _serialize_tensor(text) == torch.tensor([text.item()], dtype=torch.long).numpy().tobytes()


@torch.no_grad()
def test_conditioning_is_fully_forced_and_generate_reads_the_mask_each_frame(monkeypatch):
    sub = _tiny_submodule()
    voice, persona, codes, meta = _random_inputs()
    run = _Run(sub, meta, voice, persona, codes)
    commit, masks = run.progress.cache.commit, []
    original = sub.lm.depth.run_frame

    def checked_commit(*args):
        assert run.progress.cache.target_provided.all(), "conditioning commits only forced values"
        return commit(*args)

    def spy(text, hidden, target, provided, sample, **kw):
        masks.append(provided.clone())
        return original(text, hidden, target, provided, sample, **kw)

    monkeypatch.setattr(run.progress.cache, "commit", checked_commit)
    monkeypatch.setattr(sub.lm.depth, "run_frame", spy)
    run.run_conditioning()
    assert masks == []  # no depformer while conditioning
    run.run()
    assert len(masks) == meta["num_user_frames"]
    # First user step: agent acoustic (delay 1) still forced by the last conditioning frame,
    # agent semantic sampled. Later steps sample all agent channels.
    first = torch.zeros(16, dtype=torch.bool)
    first[1:] = True
    assert torch.equal(masks[0][0], first)
    for mask in masks[1:]:
        assert not mask[0, :8].any() and mask[0, 8:].all()


@torch.no_grad()
def test_generate_runs_eight_depth_steps_on_every_frame():
    sub = _tiny_submodule(seed=61)
    voice, persona, codes, meta = _random_inputs(n_emb=1, n_text=2, n_user=3, seed=161)
    run = _Run(sub, meta, voice, persona, codes).run_conditioning()
    steps, samples = [], []
    original = sub.lm.depth.step

    def step(index, *args, **kwargs):
        steps.append(index)
        return original(index, *args, **kwargs)

    def sample(logits, index):
        samples.append(index)
        return _greedy_depth(logits, index)

    sub.lm.depth.step = step
    for _ in range(meta["num_user_frames"]):
        frame = sub.generate_frame(run.rows, run.user(), run.backend, _greedy_text, sample)
        run.rows, run.index = frame.rows, run.index + 1
        run.backend.advance()
    assert steps == samples == list(range(EIGHT)) * meta["num_user_frames"]


@torch.no_grad()
def test_generate_frame_reads_nothing_back_from_tensors():
    """A host read in the step would break capture and serialize the async worker."""
    reads = []

    class Reads(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func.__name__.split(".")[0] in ("_local_scalar_dense", "item", "is_nonzero"):
                reads.append(func.__name__)
            return func(*args, **(kwargs or {}))

    sub = _tiny_submodule(seed=62)
    voice, persona, codes, meta = _random_inputs(n_emb=1, n_text=1, n_user=2, seed=162)
    run = _Run(sub, meta, voice, persona, codes).run_conditioning()

    class Silent:
        """Temporal attention without the test backend's own bookkeeping reads."""

        def rope(self, q, k):
            return q, k

        def attend(self, layer_idx, q, k, v):
            return v

    with Reads():
        sub.generate_frame(run.rows, run.user(), Silent(), _greedy_text, _greedy_depth)
    assert reads == []


def test_capture_config_covers_both_generate_walks_up_to_the_batch_cap():
    from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig

    for cap, sizes in ((1, [1]), (3, [1, 2, 3]), (8, [1, 2, 4, 8]), (12, [1, 2, 4, 8, 12]),
                       (32, [1, 2, 4, 8, 16, 32]), (128, [1, 2, 4, 8, 16, 32, 64, 128])):
        (config,) = LMSubmodule(PersonaPlexLM(TINY), max_batch_size=cap).get_accelerator_graph_configs(
            torch.device("cpu"))
        assert isinstance(config, BatchedAcceleratorGraphConfig) and config.capture_batch_sizes == sizes
        assert config.capture_graph_walk == cfg.GENERATE_WALK
        assert set(config.replay_graph_walks) == {cfg.GENERATE_WALK, cfg.GENERATE_AUDIO_WALK}
        assert config.input_seq_dims == {"user_frame": 0}


@torch.no_grad()
@pytest.mark.parametrize("real", [1, 3, 5])
def test_padding_rows_on_the_pool_sink_leave_real_rows_unchanged(real):
    """Pad real steps with the config's own template rows, as a replay does, for two frames in a row."""
    sub = _tiny_submodule(seed=73)
    (config,) = sub.get_accelerator_graph_configs(torch.device("cpu"))
    bucket = next(size for size in config.capture_batch_sizes if size >= real)
    cases = [_random_inputs(n_emb=1, n_text=1 + i, n_user=3, seed=173 + i) for i in range(real)]
    engine = _Engine(sub)
    rids = [f"r{i}" for i in range(real)]
    runs = [_Run(sub, c[3], *c[:3]) for c in cases]
    want = [_Run(sub, c[3], *c[:3]).run_conditioning().run() for c in cases]
    for rid, run in zip(rids, runs, strict=True):
        engine.add(rid)
        for _ in range(run.meta["conditioning_frames"]):
            engine.step(cfg.CONDITIONING_WALK, [rid],
                        [engine.prepare(cfg.CONDITIONING_WALK, rid, run.meta, run.static)], batched=False)
    for j in range(3):
        inputs = [engine.prepare(cfg.GENERATE_WALK, rid, run.meta, run.static)
                  for rid, run in zip(rids, runs, strict=True)]
        got = engine.step(cfg.GENERATE_WALK, rids, inputs, batched=True, bucket=bucket)
        for i, rid in enumerate(rids):
            _assert_same_outputs(got[rid], want[i][j], f"row {i} of {bucket}, frame {j}")


def test_sampler_adapter_rejects_malformed_tokens():
    expected = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError, match="text sampler must return shape \\(1,\\)"):
        _sampled_token_ids(torch.zeros(1, 1, dtype=torch.int32), expected, "text sampler")
    with pytest.raises(ValueError, match="depth sampler step 0 must return int32 or int64"):
        _sampled_token_ids(torch.zeros(1), expected, "depth sampler step 0")


@torch.no_grad()
def test_forward_accepts_eager_int32_sampler_tokens():
    """Eager FlashInfer returns int32 IDs; both PersonaPlex samplers feed an int64 cache."""

    class Int32Sampler:
        def __init__(self):
            self.calls = 0

        def sample(self, request_ids, logits):
            assert request_ids == ["request"] and logits.shape[0] == 1
            self.calls += 1
            return logits.argmax(-1).int()

    sub = _tiny_submodule()
    voice, persona, codes, meta = _random_inputs(n_emb=1, n_text=1, n_user=1)
    want = _Run(sub, meta, voice, persona, codes).run_conditioning().step()
    run = _Run(sub, meta, voice, persona, codes)
    engine = _Engine(sub)
    engine.add("request")
    for _ in range(meta["conditioning_frames"]):
        engine.step(cfg.CONDITIONING_WALK, ["request"],
                    [engine.prepare(cfg.CONDITIONING_WALK, "request", meta, run.static)], batched=False)
    text, depth = Int32Sampler(), Int32Sampler()
    got = engine.step(cfg.GENERATE_WALK, ["request"], [engine.prepare(cfg.GENERATE_WALK, "request", meta, run.static)],
                      batched=False, text=text, depth=depth)["request"]
    assert (text.calls, depth.calls) == (1, EIGHT)
    _assert_same_outputs(got, want, "int32")


def test_bf16_kv_hint_does_not_autocast_lm_or_mimi():
    model = PersonaPlexModel(skip_weight_loading=True)
    assert model.get_autocast_dtype() is torch.bfloat16
    assert LMSubmodule.disable_autocast and MimiEncoderSubmodule.disable_autocast
    assert MimiDecoderSubmodule.disable_autocast
    assert resolve_node_dtype(_tiny_submodule(), model.get_autocast_dtype()) is None


def test_declare_step_addresses_the_cache_on_every_walk_and_samplers_only_on_generate():
    sub = _tiny_submodule()
    inputs = [SimpleNamespace(input_seq_len=1)]
    cond = sub.declare_step(cfg.CONDITIONING_WALK, ["r"], inputs)
    assert set(cond.steps) == {cfg.LM_KV, cfg.LM_ATTN, cfg.LM_POS, cfg.LM_CACHE}
    assert cond.steps[cfg.LM_ATTN].causal is True
    for walk in (cfg.GENERATE_WALK, cfg.GENERATE_AUDIO_WALK):
        gen = sub.declare_step(walk, ["r"], inputs)
        assert set(gen.steps) == {cfg.LM_KV, cfg.LM_ATTN, cfg.LM_POS, cfg.LM_CACHE, cfg.TEXT_SAMPLER, cfg.DEPTH_SAMPLER}
        assert [(s.request_id, s.label, s.span) for s in gen.segments] == [("r", "main", 1)]


@torch.no_grad()
def test_prepare_inputs_validation_and_engine_hooks():
    sub = _tiny_submodule()
    voice, persona, codes, meta = _random_inputs()
    engine = _Engine(sub)
    engine.add("r")
    inputs = {"voice_embeddings": [voice["embeddings"]], "voice_cache": [voice["cache"]],
              "persona_tokens": [persona], "user_audio_codes": [codes]}
    with pytest.raises(ValueError, match="unknown"):
        engine.prepare("decode", "r", meta, inputs)
    with pytest.raises(ValueError, match="disagree"):
        engine.prepare(cfg.CONDITIONING_WALK, "r", {**meta, "num_text_tokens": 9}, inputs)
    node_inputs = engine.prepare(cfg.CONDITIONING_WALK, "r", meta, inputs)
    assert node_inputs.input_seq_len == 1 and node_inputs.input_embeds.shape == (1, TINY.dim)
    # The engine may prepare a frame again; the host cache must not advance twice.
    assert engine.prepare(cfg.CONDITIONING_WALK, "r", meta, inputs) is node_inputs
    staged = sub.preprocess(cfg.CONDITIONING_WALK, None, [node_inputs, node_inputs])
    assert set(staged) == {"input_embeds", "tokens", "provided", "offset"}
    assert staged["input_embeds"].shape == (2, TINY.dim) and staged["tokens"].shape == (2, 17, 4)
    with pytest.raises(RuntimeError, match="not bound"):
        sub.forward(cfg.CONDITIONING_WALK, None, **staged)
    engine.iters["r"][cfg.CONDITIONING_LOOP] = 2
    with pytest.raises(RuntimeError, match="prepared after"):
        engine.prepare(cfg.CONDITIONING_WALK, "r", meta, inputs)
    engine.iters["r"][cfg.CONDITIONING_LOOP] = meta["conditioning_frames"]
    assert engine.prepare(cfg.CONDITIONING_WALK, "r", meta, inputs) is None  # an overshoot is skipped
    state = sub.request_state(engine.handles["r"])
    assert sub.PROGRESS_KEY in state
    generate = engine.prepare(cfg.GENERATE_WALK, "r", meta, inputs)
    assert sub.PROGRESS_KEY not in state  # conditioning is over
    assert torch.equal(generate.tensor_inputs["user_frame"], codes[:, :, 0])
    assert set(sub.preprocess(cfg.GENERATE_WALK, None, [generate])) == {"user_frame"}
    with pytest.raises(ValueError, match="user_audio_codes"):
        engine.prepare(cfg.GENERATE_WALK, "r", {**meta, "num_user_frames": 9}, inputs)
    # One row takes the unbatched forward; the cap is the model kwarg.
    assert sub.can_batch(None, [node_inputs]) is False
    assert sub.can_batch(None, [node_inputs, node_inputs]) is True
    assert sub.max_batch_size(cfg.GENERATE_WALK) == cfg.DEFAULT_MAX_BATCH_SIZE
    assert LMSubmodule(sub.lm, max_batch_size=1).max_batch_size(cfg.CONDITIONING_WALK) == 1


class _FakeKV:
    def __init__(self):
        self.calls, self.keys, self.values = [], {}, {}
        self.default_label, self.layer = None, None

    def set_default_label(self, label):
        self.default_label = label

    def set_default_layer_idx(self, idx):
        self.layer = idx
        self.calls.append(("layer", idx))

    def write_kv(self, k, v):
        self.keys.setdefault(self.layer, []).append(k)
        self.values.setdefault(self.layer, []).append(v)

    def layer_view(self):
        return self.layer


class _FakeAttn:
    requires_kv_write = True

    def __init__(self, kv):
        self.kv = kv

    def set_default_label(self, label):
        pass

    def set_default_layer_idx(self, idx):
        pass

    def run(self, q, kv_cache_layer, k, v):
        keys = torch.stack(self.kv.keys[kv_cache_layer])  # [S, N, H, D]
        values = torch.stack(self.kv.values[kv_cache_layer])
        scores = torch.einsum("nhd,snhd->nhs", q, keys) / q.shape[-1] ** 0.5
        return torch.einsum("nhs,snhd->nhd", scores.softmax(-1), values)


class _FakePos:
    def __init__(self):
        self.calls, self.position = [], 0

    def apply_qk(self, q, k, label, **kwargs):
        self.calls.append((label, kwargs))
        q, k = apply_rope_adjacent(q[:, None], k[:, None], torch.tensor([self.position]), kwargs["rope_theta"])
        return q[:, 0], k[:, 0]


@torch.no_grad()
def test_resource_backend_wires_label_layers_and_interleaved_rope(monkeypatch):
    # The Mac conftest stub lacks a type used by an upstream kernel signature
    # imported through AttentionCallable. This test uses fake resources below;
    # provide only that annotation, without changing the shared stub or kernels.
    language = sys.modules.get("triton.language")
    if language is not None and not hasattr(language, "int64"):
        monkeypatch.setattr(language, "int64", int, raising=False)
    sub = _tiny_submodule()
    kv, pos = _FakeKV(), _FakePos()
    sub.bind_node_resources({cfg.LM_KV: kv, cfg.LM_ATTN: _FakeAttn(kv), cfg.LM_POS: pos})
    assert isinstance(sub._backend, ResourceTemporalBackend)
    dense = DenseTemporalCache(TINY.num_layers, TINY.rope_theta)
    x = torch.randn(3, 1, TINY.dim)
    for t in range(3):
        pos.position = t
        sub._backend.begin()
        got, _ = sub.lm.temporal.forward_embeddings(x[t], sub._backend)
        want, _ = sub.lm.temporal.forward_embeddings(x[t], dense)
        dense.advance()
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    assert kv.default_label == "main"
    assert [c for c in kv.calls if c[0] == "layer"] == [("layer", i) for _ in range(3) for i in range(2)]
    assert {(label, kw["interleave"], kw["rope_theta"]) for label, kw in pos.calls} == {("main", True, 10000.0)}


def test_stop_is_declared_on_the_lm_walks_only():
    sub = _tiny_submodule()
    assert all(sub.get_stop_policies(walk) == {} for walk in LM_WALKS)
    assert sub.get_stop_policies(cfg.CODEC_WALK) is None


def test_build_lm_loads_strictly_in_bf16():
    source = _tiny_submodule(3).lm
    weights = [(reference_lm_key(k), v) for k, v in source.state_dict().items()]
    lm = build_lm(TINY, weights, dtype=torch.bfloat16)
    assert {p.dtype for p in lm.parameters()} == {torch.bfloat16} and not lm.training
    for name, value in lm.state_dict().items():
        assert torch.equal(value, source.state_dict()[name].to(torch.bfloat16)), name
    with pytest.raises(ValueError, match="1 missing"):
        build_lm(TINY, weights[1:])


def test_model_get_submodule_rejects_unknown_nodes_without_downloading(monkeypatch):
    model = PersonaPlexModel()
    monkeypatch.setattr(model, "_download", lambda name: pytest.fail("unexpected download"))
    with pytest.raises(ValueError, match="no node"):
        model.get_submodule("Talker")


# --------------------------------------------------------------------------- R tier: NVIDIA fixtures


@torch.no_grad()
def test_lm_submodule_conditioning_and_generate_match_nvidia_lmgen():
    """NVIDIA's embedding-replay B=1 case: preset, 6 silence, 3 text, 6 silence, 20 user frames."""
    pytest.importorskip("safetensors", reason="mstar.model.loader imports safetensors")
    data = _fixture("PERSONAPLEX_REFERENCE_LM", "reference_lm.pt")
    from mstar.model.personaplex.weights import load_lm_weights

    lm = PersonaPlexLM(cfg.LMArchConfig.from_reference_kwargs(data["lm_kwargs"])).eval()
    load_lm_weights(lm, data["state_dict"].items())
    case = next(c for c in data["cases"] if c["batch_size"] == 1 and c["mode"] == "embedding_replay")
    records, preset = case["records"], case["preset"]
    n_user = data["user_frames"]
    user = torch.cat([(torch.arange(8).view(1, 8, 1) * 37 + 100 + (i + 4) * 11) % 2048 for i in range(n_user)], -1)
    meta = _meta(len(preset["embeddings"]), 3, n_user)
    run = _Run(LMSubmodule(lm), meta, {"embeddings": preset["embeddings"], "cache": preset["cache"]},
               torch.tensor([7, 11, 19]), user)
    seen = []
    original = lm.temporal.forward_embeddings

    def record_forward(x, backend, *, text_logits=True):
        hidden, logits = original(x, backend, text_logits=text_logits)
        seen.append((hidden, logits))
        return hidden, logits

    lm.temporal.forward_embeddings = record_forward
    gen = run.run_conditioning().run()
    cond_frames = meta["conditioning_frames"]
    assert cond_frames + len(gen) == len(records) == len(seen)
    for index, (record, (hidden, logits)) in enumerate(zip(records, seen, strict=True)):
        what = f"forward {index} {record['phase']}"
        torch.testing.assert_close(hidden, record["transformer_out"][:, 0], rtol=1e-5, atol=1e-5, msg=what)
        if record["phase"].startswith("user_"):
            torch.testing.assert_close(logits, record["text_logits"][:, 0, 0], rtol=1e-5, atol=1e-5, msg=what)
        else:
            assert logits is None, what  # conditioning skips the text head
    for out, record in zip(gen, records[cond_frames:], strict=True):
        assert torch.equal(out["agent_frame"][0], record["output"][:, 1:9]), record["phase"]
        assert torch.equal(out["text_frame"][0], record["output"][:, 0, 0]), record["phase"]


@contextmanager
def _without_triton_stub():
    stubs = {name: sys.modules.pop(name) for name in ("triton", "triton.language") if name in sys.modules}
    try:
        yield
    finally:
        sys.modules.update(stubs)


@torch.no_grad()
def test_mimi_submodules_match_nvidia_streaming_and_trim():
    with _without_triton_stub():
        pytest.importorskip("transformers", reason="the Mimi codec builds transformers.MimiModel")
    data = _fixture("PERSONAPLEX_REFERENCE_MIMI", "reference_mimi.pt")
    from mstar.model.personaplex.components import mimi as mimi_mod

    tiny = data["mimi_kwargs"]["tiny"]
    with _without_triton_stub():
        codec = mimi_mod.load_mimi_codec(
            data["state_dict"], mimi_mod.mimi_config_kwargs(tiny["seanet"], tiny["transformer"], tiny["quantizer"]))
    case = data["cases"][0]
    frames = case["frames"]
    encoder, decoder = MimiEncoderSubmodule(codec), MimiDecoderSubmodule(codec)
    meta = {"num_user_frames": frames, "num_samples": frames * 1920 - 700}
    info = SimpleNamespace(step_metadata=meta)
    audio = case["audio"].reshape(-1)
    prepared = encoder.prepare_inputs("encode_user_audio", info, {"audio_inputs": [audio]})
    assert torch.equal(prepared.tensor_inputs["pcm"], case["audio"])
    codes = encoder.forward("encode_user_audio", None, **prepared.tensor_inputs)["user_audio_codes"][0]
    assert torch.equal(codes, case["codes"])
    short = encoder.prepare_inputs("encode_user_audio", info, {"audio_inputs": [audio[:-700]]})
    assert torch.equal(short.tensor_inputs["pcm"][..., -700:], torch.zeros(1, 1, 700))
    with pytest.raises(ValueError, match="metadata expects"):
        encoder.prepare_inputs("encode_user_audio", info, {"audio_inputs": [audio[:1000]]})
    agent = [case["codes"][:, :, i : i + 1] for i in range(frames)]
    pcm = _stream_decode(decoder, agent, meta["num_samples"])
    assert pcm.shape == (1, 1, meta["num_samples"])
    torch.testing.assert_close(pcm, case["pcm_from_codes"][..., : meta["num_samples"]], rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- streamed Mimi decode

# A small random Mimi with the real geometry (1920 samples per 80 ms frame, 8 codebooks, a
# 250-step ring window); fast on CPU and needs no fixture.
TINY_MIMI = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
                 intermediate_size=64, num_filters=4, codebook_size=64, codebook_dim=16,
                 vector_quantization_hidden_dimension=16, num_quantizers=8, num_semantic_quantizers=1,
                 sliding_window=250, use_cache=False, upsample_groups=32)


def _tiny_codec(seed=0, layers=TINY_MIMI["num_hidden_layers"], dtype=torch.float32):
    """A random tiny Mimi whose codebook entries are distinct, so different codes decode differently
    (a fresh MimiModel's codebooks are all zeros, which would make every code stream sound the same).

    Float64 isolates cross-shape alignment checks from fp32 convolution-kernel rounding.
    The default and the NVIDIA fixture comparisons remain fp32.
    """
    from mstar.model.personaplex.components.mimi import MimiCodec, build_hf_mimi

    with _without_triton_stub():
        pytest.importorskip("transformers", reason="the Mimi codec builds transformers.MimiModel")
        torch.manual_seed(seed)
        model = build_hf_mimi({**TINY_MIMI, "num_hidden_layers": layers}).float().eval()
    with torch.no_grad():
        for rvq in (model.quantizer.semantic_residual_vector_quantizer,
                    model.quantizer.acoustic_residual_vector_quantizer):
            for layer in rvq.layers:
                layer.codebook.embed_sum.copy_(torch.randn_like(layer.codebook.embed_sum))
                layer.codebook._embed = None
    return MimiCodec(model.to(dtype=dtype))


def _agent_frames(n, seed=1, card=64):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, card, (1, 8, 1), generator=g) for _ in range(n)]


def _chunk_inputs(group):
    """What the stream hands the decoder: k stacked [1, 8, 1] frames, or one frame as it came."""
    return torch.stack(group) if len(group) > 1 else group[0]


def _driver(codec, max_slots=16):
    from benchmark.personaplex.decode import DecoderDriver

    return DecoderDriver(MimiDecoderSubmodule(codec), max_slots=max_slots)


def _decode_chunk(driver, rid, group, num_samples, final=False):
    """One chunk through the decoder node over the real pool; None when there is nothing to run."""
    node = driver.prepare(rid, num_samples, group, final=final)
    return None if node is None else driver.step([rid], [node])[rid]


def _stream_decode(codec_or_driver, frames, num_samples, rid="r", sizes=None):
    """Feed agent frames through the decoder node as the stream delivers them (the harnesses' own helper)."""
    from benchmark.personaplex.decode import DecoderDriver, stream_decode

    if isinstance(codec_or_driver, DecoderDriver):
        driver = codec_or_driver
    elif isinstance(codec_or_driver, MimiDecoderSubmodule):
        driver = DecoderDriver(codec_or_driver)
    else:
        driver = _driver(codec_or_driver)
    info = SimpleNamespace(step_metadata={"num_samples": num_samples})
    return stream_decode(driver.decoder, frames, info, rid=rid, sizes=sizes, driver=driver)


def _whole(codec, frames, num_samples):
    """The independent control: one decode over every frame, trimmed to the input's length."""
    return codec.decode(torch.cat(frames, dim=-1))[..., :num_samples]


def test_stream_codes_keep_frame_and_codebook_order():
    frames = [torch.arange(8).view(1, 8, 1) + 10 * i for i in range(3)]
    stacked = torch.stack(frames)  # [3, 1, 8, 1]
    want = torch.cat([f[0] for f in frames], dim=-1)  # [8, 3]: codebook rows, frame columns
    assert torch.equal(MimiDecoderSubmodule.stream_codes(stacked), want)
    assert torch.equal(MimiDecoderSubmodule.stream_codes(frames[1]), frames[1][0])
    assert not torch.equal(stacked.reshape(8, 3), want)  # a plain reshape would scramble them
    with pytest.raises(ValueError, match="agent frames"):
        MimiDecoderSubmodule.stream_codes(torch.zeros(2, 8, 1, dtype=torch.long))


def test_tiny_codec_decodes_different_codes_differently():
    """Guards the fixture itself: the decoder tests below mean nothing if all codes sound alike."""
    codec = _tiny_codec()
    a, b = torch.cat(_agent_frames(8, seed=2), dim=-1), torch.cat(_agent_frames(8, seed=3), dim=-1)
    with torch.no_grad():
        assert (codec.decode(a) - codec.decode(b)).abs().max().item() > 1e-3


@pytest.mark.parametrize("total", [3, 4, 12, 28, 30, 53, 78, 79])
def test_decoder_blocks_match_the_chunks_the_stream_releases(total):
    """The codec partition's StreamBuffer, driven with the model's own chunk policy, releases exactly
    the decoder's blocks (and an empty final chunk after a full one)."""
    from benchmark.personaplex.decode import schedule_sizes
    from mstar.model.personaplex.personaplex_model import PersonaPlexModel
    from mstar.streaming.stream_buffer import StreamBuffer

    connection = PersonaPlexModel.get_partition_topology(None).connections[0]
    buffer = StreamBuffer(request_id="r", edge_name="agent_frame", from_partition=cfg.LM_PARTITION,
                          policy=connection.chunk_policy_factory())
    released = []

    def poll():
        while buffer.has_chunk_ready():
            chunk = buffer.pop_chunk()
            released.append(chunk.num_items)

    for i in range(total):
        buffer.pre_read_register(f"t{i}")
        buffer.put(f"t{i}", torch.full((1, 8, 1), i))
        poll()
    buffer.signal_done()
    poll()
    assert released == schedule_sizes(total)


@torch.no_grad()
@pytest.mark.parametrize("frames", [28, 37, 53, 300])
def test_streamed_decode_matches_the_whole_decode(frames):
    """Full, partial and empty last chunks align with a whole decode, past the ring window too."""
    codec = _tiny_codec(dtype=torch.float64)
    agent = _agent_frames(frames)
    num_samples = frames * cfg.FRAME_SIZE - 700
    streamed = _stream_decode(codec, agent, num_samples)
    assert streamed.shape == (1, 1, num_samples)
    torch.testing.assert_close(streamed, _whole(codec, agent, num_samples), rtol=0, atol=1e-9)


@torch.no_grad()
def test_a_long_reply_is_the_whole_track_decode():
    """With the checkpoint's 8 stacked windowed layers the decoder reaches back further than one window;
    the stream's state carries all of it, so a long reply equals a whole-track decode (no finite context).

    In float64 on CPU the stream is within about 1e-14 of the whole decode here, while forgetting
    everything before the last 500 frames (the previous decoder's context) is off by about 2e-11, so the
    tolerance sits between the two and the negative control checks the fixture can tell them apart.
    """
    codec = _tiny_codec(layers=8, dtype=torch.float64)
    frames, tolerance = 700, 1e-12
    agent = _agent_frames(frames)
    num_samples = frames * cfg.FRAME_SIZE
    whole = _whole(codec, agent, num_samples)
    torch.testing.assert_close(_stream_decode(codec, agent, num_samples), whole, rtol=0, atol=tolerance)
    tail = 100 * cfg.FRAME_SIZE
    truncated = codec.decode(torch.cat(agent, dim=-1)[..., -600:])[..., -tail:]
    assert (truncated - whole[..., -tail:]).abs().max().item() > 10 * tolerance


@torch.no_grad()
@pytest.mark.parametrize("frames,short", [(12, 700), (13, 700), (12, 6 * 1920 + 5)])
def test_every_emission_respects_the_sample_budget(frames, short):
    """Each chunk emits its frames at once; none, final or not, emits past the input's length."""
    codec = _tiny_codec(dtype=torch.float64)
    agent = _agent_frames(frames)
    num_samples = frames * cfg.FRAME_SIZE - short
    driver = _driver(codec)
    driver.add("r")
    pieces, sent = [], 0
    for start in range(0, frames, 4):
        # The final flag can come with the last data chunk or only with an empty chunk after it.
        piece = _decode_chunk(driver, "r", agent[start : start + 4], num_samples, final=False)
        sent += piece.shape[-1]
        assert sent == min(num_samples, min(start + 4, frames) * cfg.FRAME_SIZE)
        pieces.append(piece)
    tail = _decode_chunk(driver, "r", [], num_samples, final=True)
    assert tail.shape[-1] == 0
    joined = torch.cat([*pieces, tail], dim=-1)
    assert joined.shape[-1] == num_samples
    torch.testing.assert_close(joined, _whole(codec, agent, num_samples), rtol=0, atol=1e-9)
    assert _decode_chunk(driver, "r", [], num_samples, final=False) is None  # no data, not final: nothing to run


@torch.no_grad()
@pytest.mark.parametrize("sizes", [[1] * 40, [3] * 13 + [1], [7, 7, 7, 7, 7, 5], [40], [6, 0, 34]])
def test_streamed_audio_does_not_depend_on_how_the_stream_groups_frames(sizes):
    codec = _tiny_codec(dtype=torch.float64)
    agent = _agent_frames(40)
    num_samples = 40 * cfg.FRAME_SIZE - 300
    want = _stream_decode(codec, agent, num_samples)
    got = _stream_decode(codec, agent, num_samples, sizes=sizes)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-9)


@torch.no_grad()
def test_an_empty_final_chunk_emits_nothing_and_keeps_the_state():
    codec = _tiny_codec(dtype=torch.float64)
    driver = _driver(codec)
    driver.add("r")
    agent = _agent_frames(8)
    _decode_chunk(driver, "r", agent[:4], 8 * cfg.FRAME_SIZE)
    assert _decode_chunk(driver, "r", [], 8 * cfg.FRAME_SIZE, final=True).shape[-1] == 0
    # the stream carries on exactly where it was
    rest = _decode_chunk(driver, "r", agent[4:], 8 * cfg.FRAME_SIZE)
    want = _whole(codec, agent, 8 * cfg.FRAME_SIZE)[..., 4 * cfg.FRAME_SIZE :]
    torch.testing.assert_close(rest, want, rtol=0, atol=1e-9)


@torch.no_grad()
def test_decoder_state_is_per_request_and_released():
    codec = _tiny_codec(dtype=torch.float64)
    driver = _driver(codec)
    a, b = _agent_frames(10, seed=2), _agent_frames(10, seed=3)
    num_samples = 10 * cfg.FRAME_SIZE
    interleaved = {"a": [], "b": []}
    driver.add("a")
    driver.add("b")
    for start in range(0, 10, 4):  # two requests' chunks interleaved on one decoder
        for rid, frames in (("a", a), ("b", b)):
            interleaved[rid].append(
                _decode_chunk(driver, rid, frames[start : start + 4], num_samples, final=start + 4 >= 10))
    for rid, frames in (("a", a), ("b", b)):
        torch.testing.assert_close(torch.cat(interleaved[rid], dim=-1), _whole(codec, frames, num_samples),
                                   rtol=0, atol=1e-9)
    driver.remove("a")
    assert "a" not in driver.decoder.request_states and "b" in driver.decoder.request_states
    # A request that reuses the id (and the freed slot) starts clean: b's frames under "a" give b's audio.
    again = _stream_decode(driver, b, num_samples, rid="a")
    torch.testing.assert_close(again, _whole(codec, b, num_samples), rtol=0, atol=1e-9)


@torch.no_grad()
def test_batched_rows_at_different_stream_offsets_each_get_their_own_audio():
    """Rows of different widths and histories share one decode; each matches its stream decoded alone."""
    codec = _tiny_codec(dtype=torch.float64)
    driver, solo = _driver(codec), _driver(codec)
    a, b = _agent_frames(12, seed=4), _agent_frames(9, seed=5)
    num_samples = 12 * cfg.FRAME_SIZE
    for d in (driver, solo):
        d.add("a")
        d.add("b")
        for start in (0, 4):  # request a is two chunks in, b is fresh
            _decode_chunk(d, "a", a[start : start + 4], num_samples)
    nodes = [driver.prepare("a", num_samples, a[8:12]), driver.prepare("b", num_samples, b)]
    got = driver.step(["a", "b"], nodes)
    want = {"a": _decode_chunk(solo, "a", a[8:12], num_samples), "b": _decode_chunk(solo, "b", b, num_samples)}
    for rid in ("a", "b"):
        torch.testing.assert_close(got[rid], want[rid], rtol=0, atol=1e-9)
    assert got["a"].shape[-1] == 4 * cfg.FRAME_SIZE and got["b"].shape[-1] == 9 * cfg.FRAME_SIZE
    head = min(want["a"].shape[-1], want["b"].shape[-1])
    assert (want["a"][..., :head] - want["b"][..., :head]).abs().max().item() > 1e-3


@torch.no_grad()
def test_preparing_the_next_chunk_first_changes_nothing():
    """Speculation prepares chunk N+1 before chunk N runs: prepare_inputs reads no stream state, and the
    forward reads it from the pool in step order, so the audio is unchanged."""
    codec = _tiny_codec(dtype=torch.float64)
    agent = _agent_frames(12)
    num_samples = 12 * cfg.FRAME_SIZE
    driver = _driver(codec)
    driver.add("r")
    first, second = driver.prepare("r", num_samples, agent[:4]), driver.prepare("r", num_samples, agent[4:])
    out = [driver.step(["r"], [node])["r"] for node in (first, second)]
    torch.testing.assert_close(torch.cat(out, dim=-1), _whole(codec, agent, num_samples), rtol=0, atol=1e-9)


def test_decoder_declares_its_stream_state_on_every_row():
    codec = _tiny_codec()
    step = MimiDecoderSubmodule(codec).declare_step(cfg.CODEC_WALK, ["a", "b"], [None, None])
    assert set(step.steps) == {cfg.MIMI_STATE}
    assert [(s.request_id, s.label, s.span) for s in step.segments] == [("a", "main", 1), ("b", "main", 1)]


def _code_margins(codec, pcm):
    """Best-vs-second codebook distance per (codebook, frame) along HF's residual path."""
    quantizer = codec.model.quantizer
    latent = codec.encode_latent(pcm)
    margins = []
    for rvq, count in ((quantizer.semantic_residual_vector_quantizer, 1),
                       (quantizer.acoustic_residual_vector_quantizer, 7)):
        residual = rvq.input_proj(latent) if rvq.input_proj is not None else latent
        for layer in rvq.layers[:count]:
            embed = layer.codebook.embed
            dist = torch.cdist(residual[0].T[None].float(), embed[None].float())[0]
            top = dist.topk(2, largest=False).values
            margins.append(top[:, 1] - top[:, 0])
            residual = residual - embed[dist.argmin(-1)].T[None]
    return torch.stack(margins)


@torch.no_grad()
def test_offline_components_match_nvidia_with_reference_user_codes():
    """Compare M* stages with NVIDIA's offline.py, substituting NVIDIA's encoder codes.

    Tiny random LM and Mimi, greedy: warmup + reset, .pt voice replay, persona,
    131 user frames from stereo audio (channel 0 used, last frame zero-padded,
    > 124 frames so Mimi's ring mask matters), per-frame decode, trim to input.
    """
    pytest.importorskip("safetensors", reason="mstar.model.loader imports safetensors")
    with _without_triton_stub():
        pytest.importorskip("transformers", reason="the Mimi codec builds transformers.MimiModel")
    data = _fixture("PERSONAPLEX_REFERENCE_OFFLINE", "reference_offline.pt")
    from mstar.model.personaplex.components import mimi as mimi_mod
    from mstar.model.personaplex.weights import load_lm_weights

    lm = PersonaPlexLM(cfg.LMArchConfig.from_reference_kwargs(data["lm_kwargs"])).eval()
    load_lm_weights(lm, data["lm_state_dict"].items())
    tiny = data["mimi_kwargs"]["tiny"]
    with _without_triton_stub():
        mimi_config = mimi_mod.mimi_config_kwargs(tiny["seanet"], tiny["transformer"], tiny["quantizer"])
        codec = mimi_mod.load_mimi_codec(data["mimi_state_dict"], mimi_config)
    audio = data["user_audio"][0]  # PersonaPlexModel.load_audio keeps channel 0
    num_samples = audio.shape[-1]
    n_user = -(-num_samples // 1920)
    preset, persona = data["voice_preset"], torch.tensor(data["persona_tokens"])
    meta = _meta(preset["embeddings"].shape[0], len(persona), n_user)
    meta["num_samples"] = num_samples
    info = SimpleNamespace(step_metadata=meta)

    encoder = MimiEncoderSubmodule(codec)
    prepared = encoder.prepare_inputs(cfg.ENCODE_USER_AUDIO_WALK, info, {"audio_inputs": [audio]})
    codes = encoder.forward(cfg.ENCODE_USER_AUDIO_WALK, None, **prepared.tensor_inputs)["user_audio_codes"][0]
    reference_codes = data["user_codes"]
    # Codes may differ only where fp32 makes the nearest codebook entry a tie; a
    # residual quantizer then changes that frame's later codebooks too.
    margins = _code_margins(codec, prepared.tensor_inputs["pcm"])
    mismatch = (codes != reference_codes)[0]
    for frame in mismatch.any(0).nonzero().view(-1).tolist():
        first = int(mismatch[:, frame].nonzero()[0])
        assert margins[first, frame] < 1e-6, (frame, first, float(margins[first, frame]))
    assert mismatch.any(0).float().mean() < 0.02

    # This is component parity, not uninterrupted M* encoder-to-LM parity: the
    # near-tie above changes a code, so downstream stages use NVIDIA's codes.
    generated = _Run(LMSubmodule(lm), meta, preset, persona, reference_codes).run_conditioning().run()
    assert [int(out["text_frame"][0]) for out in generated] == data["text_tokens"]

    decoder = MimiDecoderSubmodule(codec)
    pcm = _stream_decode(decoder, [o["agent_frame"][0] for o in generated], int(info.step_metadata["num_samples"]))
    assert pcm.shape == (1, 1, num_samples)
    torch.testing.assert_close(pcm[0, 0], data["pcm"], rtol=1e-5, atol=1e-5)


def test_decoder_takes_the_deployment_batch_cap():
    """max_batch_size 1 must run the decoder one request a step too, or concurrent audio is not exact."""
    codec = _tiny_codec()
    assert MimiDecoderSubmodule(codec).max_batch_size(cfg.CODEC_WALK) == cfg.DEFAULT_MAX_BATCH_SIZE
    assert MimiDecoderSubmodule(codec, max_batch_size=1).max_batch_size(cfg.CODEC_WALK) == 1
