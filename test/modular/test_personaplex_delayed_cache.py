"""Delayed token cache: fixture-free CPU tests, plus a replay of actual NVIDIA LMGen events."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from mstar.model.personaplex import config as cfg
from mstar.model.personaplex.components.delayed_cache import DelayedTokenCache


def _audio(batch, base):
    return torch.arange(batch * 8, dtype=torch.long).view(batch, 8, 1) + base


def _samples(batch, offset):
    text = torch.arange(batch, dtype=torch.long) + 1000 + offset
    audio = torch.arange(batch * 16, dtype=torch.long).view(batch, 16) + 500 + offset * 20
    return text, audio


def _frame(cache, *, user=None, agent=None, text=None):
    offset = cache.offset
    if not cache.prepare(user=user, agent=agent, text=text):
        return None
    return cache.commit(*_samples(cache.cache.shape[0], offset))


def test_source_constants():
    assert cfg.DELAYS == (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)
    assert (cfg.CACHE_SIZE, cfg.MAX_DELAY, cfg.NUM_CODEBOOKS, cfg.DEP_Q) == (4, 1, 17, 16)
    assert (cfg.SAMPLE_RATE, cfg.FRAME_RATE, cfg.FRAME_SIZE, cfg.SILENCE_FRAMES) == (24000, 12.5, 1920, 6)
    assert cfg.SILENCE_TOKENS == (948, 243, 1178, 546, 1736, 1030, 1978, 2008)
    assert cfg.SINE_TOKENS == (430, 1268, 381, 1611, 1095, 1495, 56, 472)
    assert (cfg.AUDIO_INITIAL_TOKEN_ID, cfg.TEXT_INITIAL_TOKEN_ID) == (2048, 32000)
    assert (cfg.TEXT_EPAD_TOKEN_ID, cfg.TEXT_BOS_TOKEN_ID, cfg.TEXT_EOS_TOKEN_ID, cfg.TEXT_PAD_TOKEN_ID) == (0, 1, 2, 3)
    assert (cfg.ZERO_TOKEN_ID, cfg.UNGENERATED_TOKEN_ID) == (-1, -2)
    assert (cfg.TEMPORAL_DIM, cfg.TEMPORAL_LAYERS, cfg.TEMPORAL_HEADS, cfg.TEMPORAL_HEAD_DIM) == (4096, 32, 32, 128)
    assert (cfg.TEMPORAL_FFN_DIM, cfg.TEMPORAL_FFN_HIDDEN) == (16896, 11264)
    assert (cfg.DEPFORMER_DIM, cfg.DEPFORMER_LAYERS, cfg.DEPFORMER_HEADS, cfg.DEPFORMER_HEAD_DIM) == (1024, 6, 16, 64)
    assert (cfg.DEPFORMER_FFN_DIM, cfg.DEPFORMER_FFN_HIDDEN) == (4224, 2816)
    assert (cfg.DEPFORMER_CONTEXT, cfg.DEPFORMER_POS_EMB, cfg.DEPFORMER_WEIGHTS_PER_STEP) == (None, "none", 16)
    assert (cfg.NORM, cfg.NORM_EPS, cfg.GATING, cfg.ROPE_THETA) == ("rms_norm_f32", 1e-8, "silu", 10000)
    assert (cfg.TEXT_TEMPERATURE, cfg.AUDIO_TEMPERATURE, cfg.TEXT_TOP_K, cfg.AUDIO_TOP_K) == (0.7, 0.8, 25, 250)


@pytest.mark.parametrize("batch", [1, 2])
def test_per_stream_priming_loss(batch):
    cache = DelayedTokenCache(batch)
    assert (cache.cache == cfg.UNGENERATED_TOKEN_ID).all()
    assert not cache.provided.any()
    agent0, user0 = _audio(batch, 100), _audio(batch, 200)
    assert not cache.prepare(user=user0, agent=agent0, text=torch.full((batch,), 10))
    assert cache.offset == 1
    assert torch.equal(cache.cache[:, :, 0:1], cache.initial)
    assert torch.equal(cache.cache[:, 2:9, 1], agent0[:, 1:, 0])
    assert torch.equal(cache.cache[:, 10:17, 1], user0[:, 1:, 0])
    assert (cache.cache[:, [0, 1, 9], 1] == cfg.UNGENERATED_TOKEN_ID).all()
    agent1, user1 = _audio(batch, 300), _audio(batch, 400)
    assert cache.prepare(user=user1, agent=agent1, text=torch.full((batch,), 11))
    assert torch.equal(cache.model_input, cache.initial)
    assert (cache.cache[:, 2:9, 1] == cfg.AUDIO_INITIAL_TOKEN_ID).all()
    assert (cache.cache[:, 10:17, 1] == cfg.AUDIO_INITIAL_TOKEN_ID).all()
    assert torch.equal(cache.cache[:, 2:9, 2], agent1[:, 1:, 0])
    assert cache.commit(*_samples(batch, 1)) is None
    assert cache.prepare(user=_audio(batch, 600), agent=_audio(batch, 700), text=torch.full((batch,), 12))
    assert (cache.model_input[:, 0] == 11).all()
    assert torch.equal(cache.model_input[:, 1, 0], agent1[:, 0, 0])
    assert torch.equal(cache.model_input[:, 9, 0], user1[:, 0, 0])
    assert (cache.model_input[:, 2:9] == cfg.AUDIO_INITIAL_TOKEN_ID).all()
    assert (cache.model_input[:, 10:17] == cfg.AUDIO_INITIAL_TOKEN_ID).all()
    assert cache.commit(*_samples(batch, 2)) is not None


@pytest.mark.parametrize("batch", [1, 2])
def test_conditioning_boundary_gather_and_final_step(batch):
    cache = DelayedTokenCache(batch)
    # .pt replay's first embedding includes dummy token priming internally.
    for index in range(3):
        cache.prepare_embedding()
        assert cache.offset == index + 1  # one no-forward dummy prime, then one prepare per embedding
        if index == 0:
            assert torch.equal(cache.model_input, cache.initial)
        assert (cache.target[:, 0] == cfg.TEXT_PAD_TOKEN_ID).all()
        assert (cache.target[:, 1:] == cfg.AUDIO_INITIAL_TOKEN_ID).all()
        assert cache.target_all_provided()
        output = cache.commit(*_samples(batch, cache.offset))
        assert (output is None) == (index == 0)
    assert cache.offset == 4
    restored = (torch.arange(batch * 17 * 4).reshape(batch, 17, 4) % cfg.AUDIO_CARD).long()
    old_mask, old_offset = cache.provided.clone(), cache.offset
    cache.load(restored)
    assert torch.equal(cache.cache, restored)
    assert torch.equal(cache.provided, old_mask) and cache.offset == old_offset
    restored.zero_()
    assert cache.cache.any()  # load copies rather than aliasing the preset
    silence = torch.tensor(cfg.SILENCE_TOKENS).view(1, 8, 1).expand(batch, -1, -1)
    sine = torch.tensor(cfg.SINE_TOKENS).view(1, 8, 1).expand(batch, -1, -1)
    for token in [3] * 6 + [7, 11, 19] + [3] * 6:
        assert cache.prepare(user=sine, agent=silence, text=torch.full((batch,), token))
        assert cache.target_all_provided()
        cache.commit(*_samples(batch, cache.offset))
    first_offset = cache.offset
    outputs, users = [], []
    for index in range(20):
        offset = cache.offset
        user = _audio(batch, index * 30)
        users.append(user)
        assert cache.prepare(user=user)
        mask = cache.target_provided[:, :, 0]
        assert not mask[:, 0:2].any()
        assert bool(mask[:, 2:9].all()) == (index == 0)
        assert mask[:, 9:].all()
        output = cache.commit(*_samples(batch, offset))
        outputs.append(output)
        if index == 0:
            expected = torch.cat((torch.full((batch, 1, 1), 3), silence, sine), dim=1)
        else:
            previous_text, previous_audio = _samples(batch, offset - 1)
            _, current_audio = _samples(batch, offset)
            expected = torch.cat((
                previous_text[:, None, None], previous_audio[:, :1, None],
                current_audio[:, 1:8, None], users[index - 1],
            ), dim=1)
        assert torch.equal(output, expected)
    assert len(outputs) == 20 and cache.offset == first_offset + 20
    last_text, last_audio = _samples(batch, cache.offset - 1)
    assert not torch.equal(outputs[-1][:, 0, 0], last_text)
    assert not torch.equal(outputs[-1][:, 1, 0], last_audio[:, 0])
    assert torch.equal(outputs[-1][:, 2:9, 0], last_audio[:, 1:8])


def test_forcing_is_per_batch_and_read_from_live_mask():
    cache = DelayedTokenCache(2)
    _frame(cache)
    _frame(cache)
    assert cache.prepare()
    cache.target[:, :, 0] = 123
    cache.target_provided[0, ::2, 0] = True
    cache.target_provided[1, 1::2, 0] = True
    mask = cache.target_provided[:, :, 0].clone()
    text, audio = _samples(2, cache.offset)
    samples = torch.cat((text[:, None], audio), dim=1)
    column = cache.offset % cfg.CACHE_SIZE
    consumed = (cache.offset - 1) % cfg.CACHE_SIZE
    cache.commit(text, audio)
    assert torch.equal(cache.cache[:, :, column], torch.where(mask, 123, samples))
    assert not cache.provided[:, :, consumed].any()


def test_cache_validation_and_step_lifecycle():
    with pytest.raises(ValueError, match="positive integer"):
        DelayedTokenCache(0)
    cache = DelayedTokenCache(2)
    with pytest.raises(ValueError, match="user must be int64"):
        cache.prepare(user=torch.zeros(1, 8, 1, dtype=torch.long))
    assert cache.offset == 0 and (cache.cache == -2).all()
    with pytest.raises(RuntimeError, match="prepare"):
        cache.commit(*_samples(2, 0))
    cache.prepare_embedding()
    with pytest.raises(RuntimeError, match="commit"):
        cache.prepare()
    with pytest.raises(ValueError, match="sampled_audio"):
        cache.commit(torch.zeros(2, dtype=torch.long), torch.zeros(2, 8, dtype=torch.long))
    cache.commit(*_samples(2, cache.offset))
    with pytest.raises(ValueError, match="voice cache"):
        cache.load(torch.zeros(2, 17, 3, dtype=torch.long))


def test_context_budget_boundary():
    assert (cfg.CONTEXT_FRAMES, cfg.MAX_TEMPORAL_FORWARDS) == (3000, 2999)
    assert cfg.validate_context_budget(voice_embeddings=5, text_tokens=7, user_frames=2975) == 2999
    with pytest.raises(ValueError) as exc:
        cfg.validate_context_budget(voice_embeddings=5, text_tokens=7, user_frames=2976)
    assert str(exc.value) == (
        "PersonaPlex context budget exceeded: voice_embeddings=5 + silence_frames=12 + "
        "text_tokens=7 + user_frames=2976 = 3000 > 2999"
    )


@pytest.mark.parametrize("invalid", [-1, 1.5, True])
def test_context_budget_invalid_counts(invalid):
    with pytest.raises(ValueError, match="nonnegative integers"):
        cfg.validate_context_budget(voice_embeddings=invalid, text_tokens=1, user_frames=1)


@pytest.fixture(scope="module")
def reference_trace():
    configured = os.environ.get("PERSONAPLEX_REFERENCE_TRACE")
    path = Path(configured) if configured else Path(__file__).parent / "fixtures/personaplex/reference_trace.json"
    if not path.is_file() and not configured:
        pytest.skip(
            "T3 needs NVIDIA LMGen trace; see fixtures/personaplex/README.rst or set PERSONAPLEX_REFERENCE_TRACE"
        )
    data = json.loads(path.read_text())
    assert data["schema_version"] == 2
    content = {key: value for key, value in data.items() if key not in ("argv", "content_sha256")}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == data["content_sha256"]
    assert data["source"] == "NVIDIA/personaplex" and data["source_sha"] == cfg.REFERENCE_SHA
    assert data["use_sampling"] is False and data["device"] == "cpu"
    assert data["user_frames"] >= 20
    assert data["lm_kwargs"]["delays"] == list(cfg.DELAYS) and data["lm_kwargs"]["dep_q"] == 16
    assert len(data["source_hashes"]["models/lm.py"]) == 64
    assert len(data["cases"]) == 4
    assert {(case["batch_size"], case["mode"]) for case in data["cases"]} == {
        (batch, mode) for batch in (1, 2) for mode in ("token_priming", "embedding_replay")
    }
    return data


def _tensor(value):
    return None if value is None else torch.tensor(value, dtype=torch.long)


def _assert_state(cache, expected, context):
    assert cache.offset == expected["offset"], context
    assert torch.equal(cache.cache, _tensor(expected["cache"])), context
    assert torch.equal(cache.provided, torch.tensor(expected["provided"], dtype=torch.bool)), context


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("mode", ["token_priming", "embedding_replay"])
def test_actual_lmgen_trace(reference_trace, batch, mode):
    case = next(c for c in reference_trace["cases"] if c["batch_size"] == batch and c["mode"] == mode)
    commits = [event for event in case["events"] if event["op"] == "commit"]
    for phase, count in (("silence_before_", 6), ("text_", 3), ("silence_after_", 6), ("user_", 20)):
        assert sum(event["phase"].startswith(phase) for event in commits) >= count
    assert commits[-1]["phase"] == f"user_{reference_trace['user_frames'] - 1}"
    assert all(event["output"] is not None for event in commits if event["phase"].startswith("user_"))
    if mode == "embedding_replay":
        assert sum(event["op"] == "load" for event in case["events"]) == 1
        assert sum(event["phase"].startswith("embedding_") for event in commits) == 3
    else:
        prepares = [e for e in case["events"] if e["op"] == "prepare"]
        assert [e["state"]["offset"] for e in prepares[:3]] == [1, 1, 2]
        assert [e["forward"] for e in prepares[:3]] == [False, True, True]
    cache = DelayedTokenCache(batch)
    for index, event in enumerate(case["events"]):
        context = f"B={batch} {mode} event={index} {event['op']} {event['phase']}"
        if event["op"] == "prepare":
            assert cache.prepare(
                user=_tensor(event["user"]), agent=_tensor(event["agent"]), text=_tensor(event["text"])
            ) == event["forward"], context
            if event["forward"]:
                assert torch.equal(cache.model_input, _tensor(event["input"])), context
                assert torch.equal(cache.target, _tensor(event["target"])), context
                assert torch.equal(cache.target_provided, torch.tensor(event["mask"], dtype=torch.bool)), context
        elif event["op"] == "commit":
            output = cache.commit(_tensor(event["sampled_text"]), _tensor(event["sampled_audio"]))
            expected = _tensor(event["output"])
            assert (output is None) == (expected is None), context
            if output is not None:
                assert torch.equal(output, expected), context
        elif event["op"] == "load":
            cache.load(_tensor(event["cache"]))
        else:
            assert event["op"] == "init" and index == 0, context
        _assert_state(cache, event["state"], context)


@pytest.mark.parametrize("batch", [1, 2])
def test_production_embedding_replay(reference_trace, batch, monkeypatch):
    """Drive the production helper; expected calls/states come only from LMGen.

    The observer calls the original prepare unchanged, so it checks the helper's
    internal priming calls as well as the following commits and voice restore.
    """
    case = next(c for c in reference_trace["cases"] if c["batch_size"] == batch and c["mode"] == "embedding_replay")
    events = case["events"]
    cache = DelayedTokenCache(batch)
    production_prepare = cache.prepare
    cursor = 0
    helper_calls = 0

    def observed_prepare(*, user=None, agent=None, text=None):
        nonlocal cursor
        event = events[cursor]
        context = f"B={batch} production embedding event={cursor} {event['phase']}"
        assert event["op"] == "prepare", context
        for name, actual in (("user", user), ("agent", agent), ("text", text)):
            expected = _tensor(event[name])
            assert (actual is None) == (expected is None), context
            if actual is not None:
                assert torch.equal(actual, expected), context
        ready = production_prepare(user=user, agent=agent, text=text)
        assert ready == event["forward"], context
        if ready:
            assert torch.equal(cache.model_input, _tensor(event["input"])), context
            assert torch.equal(cache.target, _tensor(event["target"])), context
            assert torch.equal(cache.target_provided, torch.tensor(event["mask"], dtype=torch.bool)), context
        _assert_state(cache, event["state"], context)
        cursor += 1
        return ready

    monkeypatch.setattr(cache, "prepare", observed_prepare)
    while cursor < len(events):
        event = events[cursor]
        context = f"B={batch} production embedding event={cursor} {event['op']} {event['phase']}"
        if event["op"] == "prepare":
            if event["phase"].startswith("embedding_"):
                cache.prepare_embedding()
                helper_calls += 1
                # The helper must consume precisely this embedding's prepares.
                assert events[cursor]["op"] == "commit" and events[cursor]["phase"] == event["phase"], context
            else:
                cache.prepare(user=_tensor(event["user"]), agent=_tensor(event["agent"]), text=_tensor(event["text"]))
            continue
        if event["op"] == "commit":
            output = cache.commit(_tensor(event["sampled_text"]), _tensor(event["sampled_audio"]))
            expected = _tensor(event["output"])
            assert (output is None) == (expected is None), context
            if output is not None:
                assert torch.equal(output, expected), context
        elif event["op"] == "load":
            cache.load(_tensor(event["cache"]))
        else:
            assert event["op"] == "init" and cursor == 0, context
        _assert_state(cache, event["state"], context)
        cursor += 1
    assert helper_calls == sum(e["op"] == "commit" and e["phase"].startswith("embedding_") for e in events)
