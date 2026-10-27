"""Submodule checks for PersonaPlex (LM, MimiEncoder, MimiDecoder).

The LM's per-iteration logic is driven engine-free with the dense CPU attention
backend and greedy samplers; the paged-KV/FlashInfer path is only checked for
its wiring here (fake resources). Fixture-backed tests replay the NVIDIA fixtures of
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

from mstar.engine.engine import Engine
from mstar.model.personaplex import config as cfg
from mstar.model.personaplex.components.delayed_cache import DelayedTokenCache
from mstar.model.personaplex.components.temporal import DenseTemporalCache, apply_rope_adjacent
from mstar.model.personaplex.personaplex_model import PersonaPlexModel
from mstar.model.personaplex.submodules import (
    LMSubmodule,
    MimiDecoderSubmodule,
    MimiEncoderSubmodule,
    ResourceTemporalBackend,
    _sampled_token_ids,
)
from mstar.model.personaplex.weights import PersonaPlexLM, build_lm, reference_lm_key
from mstar.model.submodule_base import BatchedModelOutput

FIXTURES = Path(__file__).parent / "fixtures/personaplex"
TINY = cfg.LMArchConfig(
    dim=32, num_heads=4, num_layers=2, ffn_hidden=40, depformer_dim=16, depformer_heads=2,
    depformer_layers=2, depformer_ffn_hidden=24,
)


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
    """Drive LMSubmodule frames the way the conditioning and generate walks do."""

    def __init__(self, sub, meta, voice, persona, user_codes):
        self.sub, self.meta = sub, meta
        self.backend = DenseTemporalCache(sub.lm.config.num_layers, sub.lm.config.rope_theta)
        self.static = {"voice_embeddings": [voice["embeddings"]], "voice_cache": [voice["cache"]],
                       "persona_tokens": [persona], "user_audio_codes": [user_codes]}

    def walk(self, walk, state, on_frame=None):
        outputs = []
        loop = cfg.CONDITIONING_LOOP if walk == cfg.CONDITIONING_WALK else cfg.DECODE_LOOP
        while True:
            inputs = {**self.static, "lm_state": [] if state is None else [state]}
            frame = self.sub.prepare_frame(walk, self.meta, inputs)
            assert frame is not None
            if on_frame:
                on_frame(frame)
            out = self.sub.run_frame(frame, self.backend, _greedy_text, _greedy_depth)
            self.backend.advance()
            outputs.append(out)
            state = out["lm_state"][0]
            info = SimpleNamespace(graph_walk=walk, step_metadata=self.meta)
            if self.sub.check_stop("r", info, out) == {loop}:
                break
            assert self.sub.check_stop("r", info, out) == set()
        # One more iteration would be an overshoot: vetoed, not run.
        assert self.sub.prepare_frame(walk, self.meta, {**self.static, "lm_state": [state]}) is None
        return outputs, state


# --------------------------------------------------------------------------- fixture-free


def test_lm_state_round_trips_and_rejects_corruption():
    cache = DelayedTokenCache(1)
    cache.prepare_embedding()
    cache.commit(torch.tensor([5]), torch.arange(16).view(1, 16))
    state = cache.state_tensor(7)
    assert state.shape == (1, 2 * 17 * 4 + 2) and state.dtype == torch.long
    again, step = DelayedTokenCache.from_state_tensor(state)
    assert step == 7 and again.offset == cache.offset
    assert torch.equal(again.cache, cache.cache) and torch.equal(again.provided, cache.provided)
    bad = state.clone()
    bad[0, 68] = 2
    with pytest.raises(ValueError, match="0/1"):
        DelayedTokenCache.from_state_tensor(bad)
    with pytest.raises(ValueError, match="int64"):
        DelayedTokenCache.from_state_tensor(state.int())
    cache.prepare(user=torch.zeros(1, 8, 1, dtype=torch.long))
    with pytest.raises(RuntimeError, match="commit"):
        cache.state_tensor(8)


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


@torch.no_grad()
def test_shared_lm_interleaved_request_state_matches_isolated_runs():
    """Check model-local state isolation; dense per-request KV is supplied explicitly.

    This exercises real tiny LM/depth computation and packed cache transitions,
    including different voice/persona lengths. It does not exercise production
    engine admission, paged KV, cancellation or worker resource cleanup.
    """
    sub = _tiny_submodule(seed=31)
    cases = [_random_inputs(n_emb=1, n_text=2, n_user=3, seed=32),
             _random_inputs(n_emb=2, n_text=1, n_user=4, seed=33)]
    expected = []
    runs = []
    for voice, persona, codes, meta in cases:
        isolated = _Run(sub, meta, voice, persona, codes)
        conditioning, state = isolated.walk(cfg.CONDITIONING_WALK, None)
        generated, _ = isolated.walk(cfg.GENERATE_WALK, state)
        expected.append(conditioning + generated)
        runs.append(_Run(sub, meta, voice, persona, codes))

    states = [None, None]
    actual = [[], []]
    # Alternate requests; the shorter one finishes while the other continues.
    for index in range(max(len(outputs) for outputs in expected)):
        for request, run in enumerate(runs):
            if index >= len(expected[request]):
                continue
            walk = cfg.CONDITIONING_WALK if index < run.meta["conditioning_frames"] else cfg.GENERATE_WALK
            frame = sub.prepare_frame(walk, run.meta, {
                **run.static, "lm_state": [] if states[request] is None else [states[request]],
            })
            output = sub.run_frame(frame, run.backend, _greedy_text, _greedy_depth)
            run.backend.advance()
            states[request] = output["lm_state"][0]
            actual[request].append(output)

    for request in range(2):
        for step, (got, want) in enumerate(zip(actual[request], expected[request], strict=True)):
            assert got.keys() == want.keys(), (request, step)
            for name in want:
                for value, reference in zip(got[name], want[name], strict=True):
                    assert torch.equal(value, reference), (request, step, name)


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


def _engine_step(sub, walk, engine_inputs, node_inputs, batched=True):
    """preprocess -> forward(_batched) -> the engine's row collection -> postprocess, per request."""
    rids = engine_inputs.request_ids
    staged = sub.preprocess(walk, engine_inputs, node_inputs)
    if batched:
        raw = sub.forward_batched(walk, engine_inputs, **staged)
        # The stop check must read the lm_state postprocess adds, not a buffer of the forward's.
        assert isinstance(raw, BatchedModelOutput) and raw.check_stop_buffers is None
        rows = raw.clone_row_outputs(len(rids))
        outputs = {rid: {name: [tensor[i : i + 1]] for name, tensor in rows.items()} for i, rid in enumerate(rids)}
    else:
        outputs = {rids[0]: sub.forward(walk, engine_inputs, **staged)}
    for rid, inp in zip(rids, node_inputs, strict=True):
        sub.postprocess(rid, None, outputs[rid], inp)
    return outputs


def _needed(sub, frames):
    return sub.depth_steps(cfg.GENERATE_WALK, sub._target_rows(frames)[1])


def _assert_same_outputs(got, want, what):
    assert got.keys() == want.keys(), what
    for name in want:
        for value, reference in zip(got[name], want[name], strict=True):
            assert torch.equal(value, reference), (what, name)


@torch.no_grad()
def test_batched_frames_match_isolated_runs():
    """Rows at different frame offsets share a step and still match each request run alone.

    The second request starts two frames late, so the batches hold caches at
    different offsets, and for some steps the requests are in different walks
    and run separately. Three requests cover voice, persona and user lengths
    that differ.
    """
    sub = _tiny_submodule(seed=41)
    cases = [_random_inputs(n_emb=1, n_text=2, n_user=3, seed=42),
             _random_inputs(n_emb=2, n_text=1, n_user=4, seed=43),
             _random_inputs(n_emb=1, n_text=3, n_user=2, seed=44)]
    expected = []
    for voice, persona, codes, meta in cases:
        isolated = _Run(sub, meta, voice, persona, codes)
        conditioning, state = isolated.walk(cfg.CONDITIONING_WALK, None)
        generated, _ = isolated.walk(cfg.GENERATE_WALK, state)
        expected.append(conditioning + generated)

    runs = [_Run(sub, meta, voice, persona, codes) for voice, persona, codes, meta in cases]
    starts = [0, 2, 2]
    states, actual = [None] * len(runs), [[] for _ in runs]
    batch_sizes = []
    for tick in range(max(start + len(want) for start, want in zip(starts, expected, strict=True))):
        by_walk = {}
        for request, run in enumerate(runs):
            index = tick - starts[request]
            if not 0 <= index < len(expected[request]):
                continue
            walk = cfg.CONDITIONING_WALK if index < run.meta["conditioning_frames"] else cfg.GENERATE_WALK
            by_walk.setdefault(walk, []).append(request)
        for walk, requests in by_walk.items():
            frames = [sub.prepare_frame(walk, runs[r].meta, {
                **runs[r].static, "lm_state": [] if states[r] is None else [states[r]],
            }) for r in requests]
            backend = _RowBackends([runs[r].backend for r in requests])
            outputs = sub.run_frames(frames, backend, _greedy_text, _greedy_depth)
            batch_sizes.append(len(requests))
            for r, output in zip(requests, outputs, strict=True):
                runs[r].backend.advance()
                states[r] = output["lm_state"][0]
                actual[r].append(output)

    assert max(batch_sizes) == 3 and 1 in batch_sizes
    for request in range(len(runs)):
        assert len(actual[request]) == len(expected[request])
        for step, (got, want) in enumerate(zip(actual[request], expected[request], strict=True)):
            _assert_same_outputs(got, want, (request, step))


@torch.no_grad()
def test_forward_batched_returns_each_request_its_own_rows():
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
    runs, states, want = [], [], []
    for voice, persona, codes, meta in cases:
        run = _Run(sub, meta, voice, persona, codes)
        _, state = run.walk(cfg.CONDITIONING_WALK, None)
        reference = _Run(sub, meta, voice, persona, codes)
        _, reference_state = reference.walk(cfg.CONDITIONING_WALK, None)
        frame = sub.prepare_frame(cfg.GENERATE_WALK, meta, {**reference.static, "lm_state": [reference_state]})
        want.append(sub.run_frame(frame, reference.backend, _greedy_text, _greedy_depth))
        runs.append(run)
        states.append(state)
    node_inputs = [sub.prepare_inputs(cfg.GENERATE_WALK, SimpleNamespace(step_metadata=run.meta),
                                      {**run.static, "lm_state": [state]})
                   for run, state in zip(runs, states, strict=True)]
    sub._backend = _RowBackends([run.backend for run in runs])
    text_sampler, depth_sampler = Sampler(), Sampler()
    engine_inputs = SimpleNamespace(request_ids=["a", "b"], resources={
        cfg.TEXT_SAMPLER: text_sampler, cfg.DEPTH_SAMPLER: depth_sampler,
    })
    got = _engine_step(sub, cfg.GENERATE_WALK, engine_inputs, node_inputs)
    assert set(got) == {"a", "b"}
    # The rows sample different text, so handing one row the other's tokens would show.
    assert text_sampler.tokens[0][0] != text_sampler.tokens[0][1]
    # First generate frame: only depth step 0 samples (the agent acoustic codebooks are still forced).
    assert text_sampler.shapes == [(2, TINY.text_card)] and depth_sampler.shapes == [(2, TINY.card)]
    _assert_same_outputs(got["a"], want[0], "a")
    _assert_same_outputs(got["b"], want[1], "b")
    for output in got.values():
        assert output["agent_frame"][0].shape == (1, 8, 1) and output["text_frame"][0].shape == (1,)


@torch.no_grad()
def test_a_batch_must_hold_one_walk():
    sub = _tiny_submodule()
    voice, persona, codes, meta = _random_inputs(n_emb=1, n_text=1, n_user=1)
    run = _Run(sub, meta, voice, persona, codes)
    conditioning = sub.prepare_frame(cfg.CONDITIONING_WALK, meta, {**run.static, "lm_state": []})
    _, state = _Run(sub, meta, voice, persona, codes).walk(cfg.CONDITIONING_WALK, None)
    generate = sub.prepare_frame(cfg.GENERATE_WALK, meta, {**run.static, "lm_state": [state]})
    with pytest.raises(ValueError, match="one walk"):
        sub.run_frames([conditioning, generate], _RowBackends([run.backend, run.backend]),
                       _greedy_text, _greedy_depth)


@torch.no_grad()
def test_generated_text_frame_serializes_for_worker_transfer():
    """A B=1 text column must have byte-viewable strides for worker transfer."""
    from mstar.communication.tensors import _serialize_tensor

    sub = _tiny_submodule()
    voice, persona, codes, meta = _random_inputs(n_user=1)
    run = _Run(sub, meta, voice, persona, codes)
    _, state = run.walk(cfg.CONDITIONING_WALK, None)
    generated, _ = run.walk(cfg.GENERATE_WALK, state)
    text = generated[0]["text_frame"][0]
    assert text.shape == (1,) and text.dtype == torch.long
    assert _serialize_tensor(text) == torch.tensor([text.item()], dtype=torch.long).numpy().tobytes()


@torch.no_grad()
def test_conditioning_frames_are_fully_forced_and_generate_reads_the_mask_each_frame():
    sub = _tiny_submodule()
    voice, persona, codes, meta = _random_inputs()
    run = _Run(sub, meta, voice, persona, codes)
    masks = []
    original = sub.lm.depth.run_frame

    def spy(text, hidden, target, provided, sample, **kw):
        masks.append(provided.clone())
        return original(text, hidden, target, provided, sample, **kw)

    sub.lm.depth.run_frame = spy
    cond, state = run.walk(cfg.CONDITIONING_WALK, None)
    assert len(cond) == meta["conditioning_frames"] and masks == []  # no depformer while conditioning
    assert all(set(out) == {"lm_state"} for out in cond)
    gen, state = run.walk(cfg.GENERATE_WALK, state)
    assert len(gen) == meta["num_user_frames"] == len(masks)
    # First user step: agent acoustic (delay 1) still forced by the last conditioning frame,
    # agent semantic sampled. Later steps sample all agent channels.
    acoustic = torch.zeros(16, dtype=torch.bool)
    acoustic[1:8] = True
    acoustic[8:] = True  # user channels are always provided
    assert torch.equal(masks[0][0], acoustic)
    for mask in masks[1:]:
        assert not mask[0, :8].any() and mask[0, 8:].all()
    assert DelayedTokenCache.from_state_tensor(state)[1] == meta["conditioning_frames"] + meta["num_user_frames"]


class _CountingDepth:
    """Counts the depformer steps and depth-sampler calls a frame actually makes."""

    def __init__(self, sub):
        self.depth, self.steps, self.samples = sub.lm.depth, [], []
        self._step = self.depth.step

    def __enter__(self):
        def step(index, *args, **kwargs):
            self.steps.append(index)
            return self._step(index, *args, **kwargs)

        self.depth.step = step
        return self

    def __exit__(self, *exc):
        self.depth.step = self._step

    def sample(self, logits, step):
        self.samples.append(step)
        return _greedy_depth(logits, step)


def _conditioned_pair(sub, case):
    """Two identical runs after conditioning: one to step with every depth step, one with the shortened depformer."""
    voice, persona, codes, meta = case
    runs = [_Run(sub, meta, voice, persona, codes) for _ in range(2)]
    states = [run.walk(cfg.CONDITIONING_WALK, None)[1] for run in runs]
    return runs, states


@torch.no_grad()
@pytest.mark.parametrize("seed", [61, 62, 63])
def test_shortened_depformer_matches_every_depth_step_and_skips_the_forced_suffix(seed):
    sub = _tiny_submodule(seed=seed)
    case = _random_inputs(n_emb=2, n_text=3, n_user=12, seed=seed + 100)
    (full, short), (full_state, short_state) = _conditioned_pair(sub, case)
    meta = case[3]
    for j in range(meta["num_user_frames"]):
        frames = [sub.prepare_frame(cfg.GENERATE_WALK, meta, {**run.static, "lm_state": [state]})
                  for run, state in ((full, full_state), (short, short_state))]
        want = sub.run_frames([frames[0]], full.backend, _greedy_text, _greedy_depth, depth_steps=cfg.DEP_Q)[0]
        with _CountingDepth(sub) as counted:
            got = sub.run_frames([frames[1]], short.backend, _greedy_text, counted.sample)[0]
        full.backend.advance()
        short.backend.advance()
        # User codebooks (steps 8-15) are always forced; the first frame also forces the agent's acoustic codes.
        expected = 1 if j == 0 else cfg.AUDIO_CODEBOOKS_PER_STREAM
        assert counted.steps == list(range(expected)) and counted.samples == list(range(expected)), j
        _assert_same_outputs(got, want, f"generate frame {j}")
        full_state, short_state = want["lm_state"][0], got["lm_state"][0]


@torch.no_grad()
def test_stopping_one_depth_step_early_changes_the_committed_state():
    """The step count is tight: dropping the last sampled step loses a committed agent code."""
    sub = _tiny_submodule(seed=64)
    case = _random_inputs(n_emb=1, n_text=2, n_user=4, seed=164)
    (full, short), (full_state, short_state) = _conditioned_pair(sub, case)
    meta = case[3]
    changed = []
    for _ in range(meta["num_user_frames"]):
        frames = [sub.prepare_frame(cfg.GENERATE_WALK, meta, {**run.static, "lm_state": [state]})
                  for run, state in ((full, full_state), (short, short_state))]
        needed = _needed(sub, [frames[1]])
        want = sub.run_frames([frames[0]], full.backend, _greedy_text, _greedy_depth, depth_steps=cfg.DEP_Q)[0]
        got = sub.run_frames([frames[1]], short.backend, _greedy_text, _greedy_depth, depth_steps=needed - 1)[0]
        full.backend.advance()
        short.backend.advance()
        changed.append(not torch.equal(got["lm_state"][0], want["lm_state"][0]))
        # Continue both runs from the correct state, so each frame is checked on its own.
        full_state = short_state = want["lm_state"][0]
    assert all(changed), changed


def test_depth_steps_needed_runs_through_the_last_free_step_of_any_row():
    from mstar.model.personaplex.components.depformer import depth_steps_needed

    forced = torch.ones(2, cfg.DEP_Q, dtype=torch.bool)
    assert depth_steps_needed(forced) == 0
    holes = forced.clone()
    holes[0, 2] = holes[1, 5] = False  # forced steps between free ones still run
    assert depth_steps_needed(holes) == 6
    free_last = forced.clone()
    free_last[1, cfg.DEP_Q - 1] = False  # an unexpectedly free user codebook is not skipped
    assert depth_steps_needed(free_last) == cfg.DEP_Q


@torch.no_grad()
def test_batch_of_first_and_later_generate_frames_runs_the_longer_depformer():
    """A row on its first generate frame next to a later one: 8 steps for both, each row still its solo result."""
    sub = _tiny_submodule(seed=65)
    cases = [_random_inputs(n_emb=1, n_text=2, n_user=3, seed=165),
             _random_inputs(n_emb=2, n_text=1, n_user=3, seed=166)]
    rows, want = [], []
    for skip_frames, case in zip((0, 1), cases, strict=True):
        voice, persona, codes, meta = case
        solo, batched = _Run(sub, meta, voice, persona, codes), _Run(sub, meta, voice, persona, codes)
        states = [solo.walk(cfg.CONDITIONING_WALK, None)[1], batched.walk(cfg.CONDITIONING_WALK, None)[1]]
        for _ in range(skip_frames):  # advance both copies past the first generate frame
            for k, run in enumerate((solo, batched)):
                frame = sub.prepare_frame(cfg.GENERATE_WALK, meta, {**run.static, "lm_state": [states[k]]})
                states[k] = sub.run_frame(frame, run.backend, _greedy_text, _greedy_depth)["lm_state"][0]
                run.backend.advance()
        frame = sub.prepare_frame(cfg.GENERATE_WALK, meta, {**solo.static, "lm_state": [states[0]]})
        want.append(sub.run_frames([frame], solo.backend, _greedy_text, _greedy_depth, depth_steps=cfg.DEP_Q)[0])
        rows.append((batched, sub.prepare_frame(cfg.GENERATE_WALK, meta, {**batched.static, "lm_state": [states[1]]})))
    frames = [frame for _, frame in rows]
    assert [_needed(sub, [frame]) for frame in frames] == [1, cfg.AUDIO_CODEBOOKS_PER_STREAM]
    with _CountingDepth(sub) as counted:
        got = sub.run_frames(frames, _RowBackends([run.backend for run, _ in rows]), _greedy_text, counted.sample)
    assert counted.steps == list(range(cfg.AUDIO_CODEBOOKS_PER_STREAM))
    for row in range(2):
        _assert_same_outputs(got[row], want[row], f"row {row}")


@torch.no_grad()
def test_preprocess_passes_the_depth_step_count_to_forward():
    sub = _tiny_submodule(seed=66)
    voice, persona, codes, meta = _random_inputs(n_emb=1, n_text=1, n_user=2, seed=167)
    run = _Run(sub, meta, voice, persona, codes)
    _, state = run.walk(cfg.CONDITIONING_WALK, None)
    inputs = sub.prepare_inputs(cfg.GENERATE_WALK, SimpleNamespace(step_metadata=meta),
                                {**run.static, "lm_state": [state]})
    assert sub.preprocess(cfg.GENERATE_WALK, None, [inputs])["depth_steps"] == 1
    conditioning = sub.prepare_inputs(cfg.CONDITIONING_WALK, SimpleNamespace(step_metadata=meta),
                                      {**run.static, "lm_state": []})
    assert sub.preprocess(cfg.CONDITIONING_WALK, None, [conditioning])["depth_steps"] == 0


def _reference_run_frames(sub, frames, backend):
    """The LM frame step before it was split for capture (12c009973), kept as an independent control (greedy)."""
    from mstar.model.personaplex.components.depformer import depth_steps_needed

    walk = frames[0].walk
    conditioning = walk == cfg.CONDITIONING_WALK
    embeds = torch.cat([frame.embeds for frame in frames])
    hidden, text_logits = sub.lm.temporal.forward_embeddings(embeds, backend, text_logits=not conditioning)
    target = torch.cat([frame.cache.target[:, :, 0] for frame in frames])
    provided = torch.cat([frame.cache.target_provided[:, :, 0] for frame in frames])
    if conditioning:
        sampled_text, audio = target[:, 0], target[:, 1:]
    else:
        sampled_text = _greedy_text(text_logits)
        next_text = torch.where(provided[:, 0], target[:, 0], sampled_text)
        audio = sub.lm.depth.run_frame(next_text, hidden, target[:, 1:], provided[:, 1:], _greedy_depth,
                                       steps=depth_steps_needed(provided[:, 1:]))
    results = []
    for row, frame in enumerate(frames):
        cache = frame.cache
        output = cache.commit(sampled_text[row : row + 1], audio[row : row + 1])
        if frame.restore_cache is not None:
            cache.load(frame.restore_cache)
        outputs = {"lm_state": [cache.state_tensor(frame.frame + 1)]}
        if not conditioning:
            outputs["agent_frame"] = [output[:, 1 : 1 + cfg.AUDIO_CODEBOOKS_PER_STREAM].contiguous()]
            outputs["text_frame"] = [output[:, 0, 0].clone(memory_format=torch.contiguous_format)]
        results.append(outputs)
    return results


class _GreedySampler:
    def sample(self, request_ids, logits):
        return logits.float().argmax(-1)


@torch.no_grad()
@pytest.mark.parametrize("batch", [1, 3])
def test_engine_step_matches_the_pre_capture_frame_step_over_whole_requests(batch):
    """prepare_inputs -> preprocess -> forward -> row collection -> postprocess, every frame, vs the old step.

    Requests start at different frames, so batches mix walks' first and later frames.
    """
    sub = _tiny_submodule(seed=71)
    cases = [_random_inputs(n_emb=1 + i % 2, n_text=2 + i, n_user=6, seed=171 + i) for i in range(batch)]
    engine_runs = [_Run(sub, meta, voice, persona, codes) for voice, persona, codes, meta in cases]
    reference_runs = [_Run(sub, meta, voice, persona, codes) for voice, persona, codes, meta in cases]
    states = {"engine": [None] * batch, "reference": [None] * batch}
    starts = [2 * i for i in range(batch)]
    totals = [run.meta["conditioning_frames"] + run.meta["num_user_frames"] for run in engine_runs]
    for tick in range(max(start + total for start, total in zip(starts, totals, strict=True))):
        for walk in (cfg.CONDITIONING_WALK, cfg.GENERATE_WALK):
            conditioning = walk == cfg.CONDITIONING_WALK
            rows = [i for i in range(batch) if 0 <= tick - starts[i] < totals[i]
                    and (tick - starts[i] < engine_runs[i].meta["conditioning_frames"]) == conditioning]
            if not rows:
                continue
            node_inputs = [sub.prepare_inputs(walk, SimpleNamespace(step_metadata=engine_runs[i].meta), {
                **engine_runs[i].static, "lm_state": [] if states["engine"][i] is None else [states["engine"][i]]})
                for i in rows]
            sub._backend = _RowBackends([engine_runs[i].backend for i in rows])
            rids = [f"r{i}" for i in rows]
            engine_inputs = SimpleNamespace(request_ids=rids, resources={
                cfg.TEXT_SAMPLER: _GreedySampler(), cfg.DEPTH_SAMPLER: _GreedySampler()})
            got = _engine_step(sub, walk, engine_inputs, node_inputs, batched=len(rows) > 1)
            frames = [sub.prepare_frame(walk, reference_runs[i].meta, {
                **reference_runs[i].static,
                "lm_state": [] if states["reference"][i] is None else [states["reference"][i]]}) for i in rows]
            want = _reference_run_frames(sub, frames, _RowBackends([reference_runs[i].backend for i in rows]))
            for i in rows:
                engine_runs[i].backend.advance()
                reference_runs[i].backend.advance()
            for k, i in enumerate(rows):
                _assert_same_outputs(got[rids[k]], want[k], f"tick {tick} request {i} {walk}")
                states["engine"][i] = got[rids[k]]["lm_state"][0]
                states["reference"][i] = want[k]["lm_state"][0]


@torch.no_grad()
def test_captured_eight_step_depformer_matches_the_exact_count_on_first_and_later_frames():
    sub = _tiny_submodule(seed=72)
    case = _random_inputs(n_emb=1, n_text=2, n_user=4, seed=172)
    (exact, fixed), (exact_state, fixed_state) = _conditioned_pair(sub, case)
    meta = case[3]
    for j in range(meta["num_user_frames"]):
        frames = [sub.prepare_frame(cfg.GENERATE_WALK, meta, {**run.static, "lm_state": [state]})
                  for run, state in ((exact, exact_state), (fixed, fixed_state))]
        want = sub.run_frames([frames[0]], exact.backend, _greedy_text, _greedy_depth)[0]
        got = sub.run_frames([frames[1]], fixed.backend, _greedy_text, _greedy_depth,
                             depth_steps=cfg.AUDIO_CODEBOOKS_PER_STREAM)[0]
        exact.backend.advance()
        fixed.backend.advance()
        _assert_same_outputs(got, want, f"generate frame {j}")
        exact_state, fixed_state = want["lm_state"][0], got["lm_state"][0]


def test_capture_config_covers_both_generate_walks_up_to_the_batch_cap():
    from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig

    for cap, sizes in ((1, [1]), (3, [1, 2, 3]), (8, [1, 2, 4, 8]), (12, [1, 2, 4, 8, 12]),
                       (32, [1, 2, 4, 8, 16, 32])):
        (config,) = LMSubmodule(PersonaPlexLM(TINY), max_batch_size=cap).get_accelerator_graph_configs(
            torch.device("cpu"))
        assert isinstance(config, BatchedAcceleratorGraphConfig) and config.capture_batch_sizes == sizes
        assert config.capture_graph_walk == cfg.GENERATE_WALK
        assert set(config.replay_graph_walks) == {cfg.GENERATE_WALK, cfg.GENERATE_AUDIO_WALK}
        assert config.input_seq_dims == {"input_embeds": 0, "target": 0, "provided": 0}


@torch.no_grad()
@pytest.mark.parametrize("real", [1, 3, 5])
def test_padding_rows_from_the_capture_template_leave_real_rows_unchanged(real):
    """Pad a real batch with the config's own template rows (as the runner does) and run the 8-step step."""
    sub = _tiny_submodule(seed=73)
    (config,) = sub.get_accelerator_graph_configs(torch.device("cpu"))
    bucket = next(size for size in config.capture_batch_sizes if size >= real)
    cases = [_random_inputs(n_emb=1, n_text=1 + i, n_user=3, seed=173 + i) for i in range(real)]
    padded_runs, solo_runs, node_inputs, frames = [], [], [], []
    for voice, persona, codes, meta in cases:
        for runs in (padded_runs, solo_runs):
            run = _Run(sub, meta, voice, persona, codes)
            _, state = run.walk(cfg.CONDITIONING_WALK, None)
            runs.append((run, state))
        run, state = padded_runs[-1]
        node_inputs.append(sub.prepare_inputs(cfg.GENERATE_WALK, SimpleNamespace(step_metadata=meta),
                                              {**run.static, "lm_state": [state]}))
        run, state = solo_runs[-1]
        frames.append(sub.prepare_frame(cfg.GENERATE_WALK, meta, {**run.static, "lm_state": [state]}))
    padding = config.get_node_inputs(bucket - real, 0)
    assert all("frame" not in inp.kwargs for inp in padding)
    staged = sub.preprocess(cfg.GENERATE_WALK, None, [*node_inputs, *padding])
    # Every row is on its first generate frame (1 step); template rows need the 8 the capture records.
    assert staged["input_embeds"].shape[0] == bucket
    assert staged["depth_steps"] == (cfg.AUDIO_CODEBOOKS_PER_STREAM if padding else 1)
    staged["depth_steps"] = cfg.AUDIO_CODEBOOKS_PER_STREAM  # a replay runs the captured step
    sub._backend = _RowBackends([run.backend for run, _ in padded_runs]
                                + [DenseTemporalCache(TINY.num_layers, TINY.rope_theta) for _ in padding])
    rids = [f"r{i}" for i in range(bucket)]
    engine_inputs = SimpleNamespace(request_ids=rids, resources={
        cfg.TEXT_SAMPLER: _GreedySampler(), cfg.DEPTH_SAMPLER: _GreedySampler()})
    raw = sub.forward_batched(cfg.GENERATE_WALK, engine_inputs, **staged)
    rows = raw.clone_row_outputs(real)  # the engine keeps only the real rows
    assert all(tensor.shape[0] == real for tensor in rows.values())
    want = [sub.run_frame(frame, run.backend, _greedy_text, _greedy_depth)
            for frame, (run, _) in zip(frames, solo_runs, strict=True)]
    for i, inp in enumerate(node_inputs):
        outputs = {name: [tensor[i : i + 1]] for name, tensor in rows.items()}
        sub.postprocess(rids[i], None, outputs, inp)
        _assert_same_outputs(outputs, want[i], f"row {i} of {bucket}")


def test_a_generate_frame_with_a_free_user_codebook_is_refused():
    """The captured step runs 8 depth steps; a frame needing more must fail loudly, not replay short."""
    provided = torch.ones(2, cfg.NUM_CODEBOOKS, dtype=torch.bool)
    provided[:, 1 : 1 + cfg.AUDIO_CODEBOOKS_PER_STREAM] = False
    assert LMSubmodule.depth_steps(cfg.GENERATE_WALK, provided) == cfg.AUDIO_CODEBOOKS_PER_STREAM
    provided[1, cfg.NUM_CODEBOOKS - 1] = False
    with pytest.raises(RuntimeError, match="user codebook"):
        LMSubmodule.depth_steps(cfg.GENERATE_WALK, provided)
    assert LMSubmodule.depth_steps(cfg.CONDITIONING_WALK, provided) == 0


@torch.no_grad()
def test_forward_accepts_eager_int32_sampler_tokens():
    """Eager FlashInfer returns int32 IDs; both PersonaPlex samplers feed an int64 cache."""
    class DenseForwardBackend(DenseTemporalCache):
        def begin(self):
            pass

    class Int32Sampler:
        def __init__(self):
            self.calls = 0

        def sample(self, request_ids, logits):
            assert request_ids == ["request"] and logits.shape[0] == 1
            self.calls += 1
            return logits.argmax(-1).int()

    sub = _tiny_submodule()
    voice, persona, codes, meta = _random_inputs(n_emb=1, n_text=1, n_user=1)
    run = _Run(sub, meta, voice, persona, codes)
    run.backend = DenseForwardBackend(TINY.num_layers, TINY.rope_theta)
    _, state = run.walk(cfg.CONDITIONING_WALK, None)
    node_inputs = sub.prepare_inputs(cfg.GENERATE_WALK, SimpleNamespace(step_metadata=meta),
                                     {**run.static, "lm_state": [state]})
    sub._backend = run.backend
    text_sampler, depth_sampler = Int32Sampler(), Int32Sampler()
    engine_inputs = SimpleNamespace(request_ids=["request"], resources={
        cfg.TEXT_SAMPLER: text_sampler, cfg.DEPTH_SAMPLER: depth_sampler,
    })
    got = _engine_step(sub, cfg.GENERATE_WALK, engine_inputs, [node_inputs], batched=False)["request"]
    assert (text_sampler.calls, depth_sampler.calls) == (1, 1)  # first generate frame: depth step 0 only

    reference = _Run(sub, meta, voice, persona, codes)
    _, reference_state = reference.walk(cfg.CONDITIONING_WALK, None)
    reference_frame = sub.prepare_frame(cfg.GENERATE_WALK, meta, {
        **reference.static, "lm_state": [reference_state],
    })
    want = sub.run_frame(reference_frame, reference.backend, _greedy_text, _greedy_depth)
    assert got.keys() == want.keys()
    for name in got:
        assert all(torch.equal(actual, expected) for actual, expected in zip(got[name], want[name], strict=True))


def test_sampler_adapter_rejects_malformed_tokens():
    expected = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError, match="text sampler must return shape \\(1,\\)"):
        _sampled_token_ids(torch.zeros(1, 1, dtype=torch.int32), expected, "text sampler")
    with pytest.raises(ValueError, match="depth sampler step 0 must return int32 or int64"):
        _sampled_token_ids(torch.zeros(1), expected, "depth sampler step 0")


def test_bf16_kv_hint_does_not_autocast_lm_or_mimi():
    model = PersonaPlexModel(skip_weight_loading=True)
    # This checks autocast policy, independently of graph-runtime construction.
    engine = Engine.__new__(Engine)
    engine._autocast_dtype = model.get_autocast_dtype()
    assert model.get_autocast_dtype() is torch.bfloat16
    assert LMSubmodule.disable_autocast and MimiEncoderSubmodule.disable_autocast
    assert MimiDecoderSubmodule.disable_autocast
    assert engine._autocast_dtype_for(_tiny_submodule()) is None


def test_declare_step_declares_samplers_only_on_generate_walks():
    sub = _tiny_submodule()
    inputs = [SimpleNamespace(input_seq_len=1)]
    cond = sub.declare_step(cfg.CONDITIONING_WALK, ["r"], inputs)
    assert set(cond.steps) == {cfg.LM_KV, cfg.LM_ATTN, cfg.LM_POS}
    assert cond.steps[cfg.LM_ATTN].causal is True
    for walk in (cfg.GENERATE_WALK, cfg.GENERATE_AUDIO_WALK):
        gen = sub.declare_step(walk, ["r"], inputs)
        assert set(gen.steps) == {cfg.LM_KV, cfg.LM_ATTN, cfg.LM_POS, cfg.TEXT_SAMPLER, cfg.DEPTH_SAMPLER}
        assert [(s.request_id, s.label, s.span) for s in gen.segments] == [("r", "main", 1)]


@torch.no_grad()
def test_conditioning_rejects_an_unforced_target():
    sub = _tiny_submodule()
    voice, persona, codes, meta = _random_inputs()
    frame = sub.prepare_frame(cfg.CONDITIONING_WALK, meta, {
        "voice_embeddings": [voice["embeddings"]], "voice_cache": [voice["cache"]],
        "persona_tokens": [persona], "lm_state": []})
    frame.cache.provided[:, 3, frame.cache.offset % cfg.CACHE_SIZE] = False
    with pytest.raises(RuntimeError, match="unprovided"):
        sub.run_frame(frame, DenseTemporalCache(2, 10000), _greedy_text, _greedy_depth)


@torch.no_grad()
def test_lm_frame_validation_and_engine_hooks():
    sub = _tiny_submodule()
    voice, persona, codes, meta = _random_inputs()
    with pytest.raises(ValueError, match="needs the conditioning lm_state"):
        sub.prepare_frame(cfg.GENERATE_WALK, meta, {"user_audio_codes": [codes], "lm_state": []})
    early = DelayedTokenCache(1).state_tensor(2)
    with pytest.raises(ValueError, match="before this walk's first frame"):
        sub.prepare_frame(cfg.GENERATE_WALK, meta, {"user_audio_codes": [codes], "lm_state": [early]})
    with pytest.raises(ValueError, match="disagree"):
        sub.prepare_frame(cfg.CONDITIONING_WALK, {**meta, "num_text_tokens": 9}, {
            "voice_embeddings": [voice["embeddings"]], "voice_cache": [voice["cache"]],
            "persona_tokens": [persona], "lm_state": []})
    with pytest.raises(ValueError, match="unknown"):
        sub.prepare_frame("decode", meta, {})
    fwd_info = SimpleNamespace(step_metadata=meta)
    inputs = {"voice_embeddings": [voice["embeddings"]], "voice_cache": [voice["cache"]],
              "persona_tokens": [persona], "lm_state": []}
    node_inputs = sub.prepare_inputs(cfg.CONDITIONING_WALK, fwd_info, inputs)
    assert node_inputs.input_seq_len == 1 and node_inputs.input_embeds.shape == (1, TINY.dim)
    frame = node_inputs.kwargs["frame"]
    # preprocess hands the forward tensors only; the cache stays on the request's inputs for postprocess.
    assert node_inputs.tensor_inputs["target"].shape == node_inputs.tensor_inputs["provided"].shape == (1, 17)
    staged = sub.preprocess(cfg.CONDITIONING_WALK, None, [node_inputs, node_inputs])
    assert set(staged) == {"input_embeds", "target", "provided", "depth_steps"} and staged["depth_steps"] == 0
    assert staged["input_embeds"].shape == (2, TINY.dim) and staged["target"].shape == (2, 17)
    assert torch.equal(staged["target"][1], frame.cache.target[0, :, 0])
    with pytest.raises(RuntimeError, match="not bound"):
        sub.forward(cfg.CONDITIONING_WALK, None, **staged)
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


def test_check_stop_ignores_other_walks_and_missing_state():
    sub = _tiny_submodule()
    meta = _meta(3, 4, 5)
    state = DelayedTokenCache(1).state_tensor(meta["conditioning_frames"])
    info = SimpleNamespace(graph_walk=cfg.CONDITIONING_WALK, step_metadata=meta)
    assert sub.check_stop("r", info, {"lm_state": [state]}) == {cfg.CONDITIONING_LOOP}
    assert sub.check_stop("r", info, {}) == set()
    gen = SimpleNamespace(graph_walk=cfg.GENERATE_AUDIO_WALK, step_metadata=meta)
    assert sub.check_stop("r", gen, {"lm_state": [state]}) == set()
    end = DelayedTokenCache(1).state_tensor(meta["conditioning_frames"] + 5)
    assert sub.check_stop("r", gen, {"lm_state": [end]}) == {cfg.DECODE_LOOP}


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
    cond, state = run.walk(cfg.CONDITIONING_WALK, None)
    gen, state = run.walk(cfg.GENERATE_WALK, state)
    assert len(cond) + len(gen) == len(records) == len(seen)
    for index, (record, (hidden, logits)) in enumerate(zip(records, seen, strict=True)):
        what = f"forward {index} {record['phase']}"
        torch.testing.assert_close(hidden, record["transformer_out"][:, 0], rtol=1e-5, atol=1e-5, msg=what)
        if record["phase"].startswith("user_"):
            torch.testing.assert_close(logits, record["text_logits"][:, 0, 0], rtol=1e-5, atol=1e-5, msg=what)
        else:
            assert logits is None, what  # conditioning skips the text head
    for out, record in zip(gen, records[len(cond):], strict=True):
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


def _tiny_codec(seed=0):
    from mstar.model.personaplex.components.mimi import MimiCodec, build_hf_mimi

    with _without_triton_stub():
        pytest.importorskip("transformers", reason="the Mimi codec builds transformers.MimiModel")
        torch.manual_seed(seed)
        return MimiCodec(build_hf_mimi(TINY_MIMI).float().eval())


def _agent_frames(n, seed=1, card=64):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, card, (1, 8, 1), generator=g) for _ in range(n)]


def _chunk_inputs(group):
    """What the stream hands the decoder: k stacked [1, 8, 1] frames, or one frame as it came."""
    return torch.stack(group) if len(group) > 1 else group[0]


def _decode_chunk(decoder, rid, group, num_samples, final=False):
    info = SimpleNamespace(step_metadata={"num_samples": num_samples})
    inputs = {"agent_frame": [_chunk_inputs(group)]} if group else {}
    node = decoder.prepare_inputs(cfg.CODEC_WALK, info, inputs, is_final_stream_chunk=final)
    if node is None:
        return None
    engine = SimpleNamespace(request_ids=[rid])
    staged = decoder.preprocess(cfg.CODEC_WALK, engine, [node])
    return decoder.forward(cfg.CODEC_WALK, engine, **staged)["audio_output"][0]


def _stream_decode(decoder, frames, num_samples, rid="r", sizes=None):
    """Feed agent frames through the decoder node as the stream delivers them (the harnesses' own helper)."""
    from benchmark.personaplex.decode import stream_decode

    info = SimpleNamespace(step_metadata={"num_samples": num_samples})
    return stream_decode(decoder, frames, info, rid=rid, sizes=sizes)


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


@torch.no_grad()
@pytest.mark.parametrize("frames", [28, 37, 53, 300])
def test_streamed_decode_within_one_window_matches_the_whole_decode(frames):
    """Exact (to fp32 rounding) while the track fits the decoder's context: full, partial and empty last chunks."""
    codec = _tiny_codec()
    agent = _agent_frames(frames)
    num_samples = frames * cfg.FRAME_SIZE - 700
    streamed = _stream_decode(MimiDecoderSubmodule(codec), agent, num_samples)
    assert streamed.shape == (1, 1, num_samples)
    torch.testing.assert_close(streamed, _whole(codec, agent, num_samples), rtol=0, atol=1e-5)


@torch.no_grad()
def test_streamed_decode_past_the_window_differs_only_slightly_and_needs_its_context():
    """Past one window the stacked layers reach further back than the context (the Qwen3-TTS trade-off)."""
    codec = _tiny_codec()
    frames = cfg.STREAM_CONTEXT_FRAMES + 150
    agent = _agent_frames(frames)
    num_samples = frames * cfg.FRAME_SIZE
    whole = _whole(codec, agent, num_samples)
    streamed = _stream_decode(MimiDecoderSubmodule(codec), agent, num_samples)
    window = cfg.STREAM_CONTEXT_FRAMES * cfg.FRAME_SIZE
    torch.testing.assert_close(streamed[..., :window], whole[..., :window], rtol=0, atol=1e-5)
    error = (streamed - whole).abs().max().item()
    no_context = (_stream_decode(MimiDecoderSubmodule(codec, context_frames=0), agent, num_samples) - whole)
    assert error < 1e-3, error
    assert no_context.abs().max().item() > 100 * max(error, 1e-6)  # the context is what keeps chunks close


@torch.no_grad()
@pytest.mark.parametrize("frames,short", [(12, 700), (13, 700), (12, 6 * 1920 + 5)])
def test_every_emission_respects_the_sample_budget(frames, short):
    """No chunk, final or not, emits past the input's length; frames short of a block wait for the final chunk."""
    codec = _tiny_codec()
    agent = _agent_frames(frames)
    num_samples = frames * cfg.FRAME_SIZE - short
    decoder = MimiDecoderSubmodule(codec)
    pieces, sent = [], 0
    for start in range(0, frames, 4):
        # The final flag can come with the last data chunk or only with an empty chunk after it.
        piece = _decode_chunk(decoder, "r", agent[start : start + 4], num_samples, final=False)
        sent += piece.shape[-1]
        assert sent <= num_samples
        pieces.append(piece)
    held = frames - 12 if frames > 12 else 0  # blocks of 4 and 8 cover 12 frames
    assert sum(p.shape[-1] for p in pieces) == min(num_samples, (frames - held) * cfg.FRAME_SIZE)
    tail = _decode_chunk(decoder, "r", [], num_samples, final=True)
    assert tail.shape[-1] == max(num_samples - sent, 0)
    joined = torch.cat([*pieces, tail], dim=-1)
    assert joined.shape[-1] == num_samples
    torch.testing.assert_close(joined, _whole(codec, agent, num_samples), rtol=0, atol=1e-5)
    assert _decode_chunk(decoder, "r", [], num_samples, final=False) is None  # no data, not final: nothing to run


def test_blocks_follow_the_stream_schedule():
    sizes, done = [], 0
    while done < 120:
        sizes.append(MimiDecoderSubmodule.block_frames(done))
        done += sizes[-1]
    assert sizes[:6] == [*cfg.STREAM_CHUNK_SCHEDULE, *[cfg.STREAM_CHUNK_FRAMES] * 3]
    assert MimiDecoderSubmodule.block_frames(1) == 3  # mid-block: the frames left to the next boundary
    assert MimiDecoderSubmodule.block_frames(30) == cfg.STREAM_CHUNK_FRAMES - 2


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
@pytest.mark.parametrize("sizes", [[1] * 40, [3] * 13 + [1], [7, 7, 7, 7, 7, 5], [40], [6, 0, 34]])
def test_streamed_audio_does_not_depend_on_how_the_stream_groups_frames(sizes):
    """Bit-identical whatever the delivery: the decoder decodes in its own blocks and holds the rest."""
    codec = _tiny_codec()
    agent = _agent_frames(40)
    num_samples = 40 * cfg.FRAME_SIZE - 300
    want = _stream_decode(MimiDecoderSubmodule(codec), agent, num_samples)
    got = _stream_decode(MimiDecoderSubmodule(codec), agent, num_samples, sizes=sizes)
    assert torch.equal(got, want)


@torch.no_grad()
def test_decode_last_reruns_seanet_only_over_its_receptive_field():
    codec = _tiny_codec()
    codes = torch.cat(_agent_frames(60), dim=-1)
    whole = codec.decode(codes)
    for frames in (1, 4, 25):
        last = codec.decode_last(codes, frames, cfg.SEANET_MARGIN_FRAMES)
        torch.testing.assert_close(last, whole[..., -frames * cfg.FRAME_SIZE :], rtol=0, atol=1e-6)
    no_margin = codec.decode_last(codes, 4, 0) - whole[..., -4 * cfg.FRAME_SIZE :]
    assert no_margin.abs().max().item() > 1e-4  # the margin is what makes it exact
    with pytest.raises(ValueError):
        codec.decode_last(codes, 0, cfg.SEANET_MARGIN_FRAMES)


@torch.no_grad()
def test_decoder_state_is_per_request_and_released():
    codec = _tiny_codec()
    decoder = MimiDecoderSubmodule(codec)
    a, b = _agent_frames(10, seed=2), _agent_frames(10, seed=3)
    num_samples = 10 * cfg.FRAME_SIZE
    interleaved = {"a": [], "b": []}
    for start in range(0, 10, 4):  # two requests' chunks interleaved on one decoder
        interleaved["a"].append(_decode_chunk(decoder, "a", a[start : start + 4], num_samples, final=start + 4 >= 10))
        interleaved["b"].append(_decode_chunk(decoder, "b", b[start : start + 4], num_samples, final=start + 4 >= 10))
    for rid, frames in (("a", a), ("b", b)):
        torch.testing.assert_close(torch.cat(interleaved[rid], dim=-1), _whole(codec, frames, num_samples),
                                   rtol=0, atol=1e-5)
    decoder.cleanup_request("a")
    assert "a" not in decoder.request_states and "b" in decoder.request_states
    # A recycled handle starts clean: decoding request b's frames under handle "a" gives b's audio.
    again = _stream_decode(decoder, b, num_samples, rid="a")
    torch.testing.assert_close(again, _whole(codec, b, num_samples), rtol=0, atol=1e-5)


@torch.no_grad()
def test_batched_rows_at_different_stream_offsets_each_get_their_own_audio():
    codec = _tiny_codec()
    decoder, solo_decoder = MimiDecoderSubmodule(codec), MimiDecoderSubmodule(codec)
    a, b = _agent_frames(12, seed=4), _agent_frames(4, seed=5)
    num_samples = 12 * cfg.FRAME_SIZE
    for start in (0, 4):  # request a is two chunks in, so its window is longer than b's
        _decode_chunk(decoder, "a", a[start : start + 4], num_samples)
        _decode_chunk(solo_decoder, "a", a[start : start + 4], num_samples)
    info = SimpleNamespace(step_metadata={"num_samples": num_samples})
    nodes = [decoder.prepare_inputs(cfg.CODEC_WALK, info, {"agent_frame": [_chunk_inputs(a[8:12])]}),
             decoder.prepare_inputs(cfg.CODEC_WALK, info, {"agent_frame": [_chunk_inputs(b)]})]
    engine = SimpleNamespace(request_ids=["a", "b"])
    got = decoder.forward_batched(cfg.CODEC_WALK, engine, **decoder.preprocess(cfg.CODEC_WALK, engine, nodes))
    want_a = _decode_chunk(solo_decoder, "a", a[8:12], num_samples)
    want_b = _decode_chunk(solo_decoder, "b", b, num_samples)
    torch.testing.assert_close(got["a"]["audio_output"][0], want_a, rtol=0, atol=1e-5)
    torch.testing.assert_close(got["b"]["audio_output"][0], want_b, rtol=0, atol=1e-5)


@torch.no_grad()
def test_preparing_the_next_chunk_first_changes_nothing():
    """prepare_inputs reads no request state, so N+1's preparation before N's forward is harmless."""
    codec = _tiny_codec()
    agent = _agent_frames(12)
    num_samples = 12 * cfg.FRAME_SIZE
    decoder = MimiDecoderSubmodule(codec)
    info = SimpleNamespace(step_metadata={"num_samples": num_samples})
    engine = SimpleNamespace(request_ids=["r"])
    first = decoder.prepare_inputs(cfg.CODEC_WALK, info, {"agent_frame": [_chunk_inputs(agent[:4])]})
    second = decoder.prepare_inputs(cfg.CODEC_WALK, info, {"agent_frame": [_chunk_inputs(agent[4:])]})
    out = [decoder.forward(cfg.CODEC_WALK, engine, **decoder.preprocess(cfg.CODEC_WALK, engine, [node]))
           ["audio_output"][0] for node in (first, second)]
    torch.testing.assert_close(torch.cat(out, dim=-1), _whole(codec, agent, num_samples), rtol=0, atol=1e-5)


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
    run = _Run(LMSubmodule(lm), meta, preset, persona, reference_codes)
    _, state = run.walk(cfg.CONDITIONING_WALK, None)
    generated, _ = run.walk(cfg.GENERATE_WALK, state)
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
