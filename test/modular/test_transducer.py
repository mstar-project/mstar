"""Batched greedy transducer decoding (TDT and RNN-T) against a plain
one-request reference loop, on tiny random heads."""

import pytest
import torch
import torch.nn.functional as F

from mstar.model.components.transducer import TransducerConfig, TransducerHead


@pytest.fixture(autouse=True)
def _one_thread():
    # token lists compare exactly, and CPU GEMMs can round differently with
    # the intra-op thread count
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _head(durations, num_prompts=0, seed=0) -> TransducerHead:
    torch.manual_seed(seed)
    config = TransducerConfig(
        encoder_hidden_size=24, hidden_size=16, num_layers=2, vocab_size=11, blank_token_id=10,
        start_token_id=10, durations=durations, max_symbols_per_step=3,
        num_prompts=num_prompts, prompt_intermediate_size=20 if num_prompts else 0,
    )
    head = TransducerHead(config).eval()
    with torch.no_grad():
        for p in head.parameters():
            p.normal_(0, 0.5)
        # make blank likely enough that sequences end with a mix of both
        head.joint.head.bias[config.blank_token_id] += 1.0
    return head


def _reference(head: TransducerHead, states: torch.Tensor, prompt_id=None) -> list[int]:
    """One request, written as the papers state it."""
    cfg = head.config
    frames = head.project_frames(states[None], None if prompt_id is None else torch.tensor([prompt_id]))[0]
    h = c = torch.zeros(1, cfg.num_layers, cfg.hidden_size)
    out, h, c = head.decoder(torch.tensor([cfg.start_token_id]), h, c)
    t, tokens, at_frame = 0, [], 0
    while t < frames.shape[0]:
        logits = head.joint.head(F.relu(frames[t] + out[0]))
        token = int(logits[:cfg.vocab_size].argmax())
        if cfg.durations:
            step = cfg.durations[int(logits[cfg.vocab_size:].argmax())]
            if token == cfg.blank_token_id and step == 0:
                step = 1
        else:
            step = 1 if token == cfg.blank_token_id else 0
        # NeMo: a frame takes max_symbols tokens, then the cursor moves on
        at_frame = at_frame + 1 if token != cfg.blank_token_id and step == 0 else 0
        if at_frame >= cfg.max_symbols_per_step:
            step, at_frame = 1, 0
        if token != cfg.blank_token_id:
            tokens.append(token)
            out, h, c = head.decoder(torch.tensor([token]), h, c)
        t += step
    return tokens


@pytest.mark.parametrize("durations", [[0, 1, 2, 3, 4], []], ids=["tdt", "rnnt"])
def test_batched_greedy_matches_reference(durations):
    head = _head(durations)
    torch.manual_seed(1)
    lens = [17, 5, 30, 1]
    states = torch.zeros(len(lens), max(lens), 24)
    for i, n in enumerate(lens):
        states[i, :n] = torch.randn(n, 24)
    with torch.no_grad():
        batched = head.greedy_decode(states, torch.tensor(lens))
        expected = [_reference(head, states[i, :n]) for i, n in enumerate(lens)]
    assert batched == expected
    assert any(batched), "the random head emitted nothing; the test would be vacuous"


def test_rnnt_caps_symbols_per_frame():
    head = _head([], seed=2)
    with torch.no_grad():
        # never blank: every frame emits exactly max_symbols tokens
        head.joint.head.bias[head.config.blank_token_id] = -1e4
        tokens = head.greedy_decode(torch.randn(1, 4, 24), torch.tensor([4]))
    assert len(tokens[0]) == 4 * head.config.max_symbols_per_step


def test_rnnt_finds_emissions_across_blank_runs():
    """An RNN-T step scores a window of frames: emissions on a window's first
    and last frame, after blank runs longer than a window, and at the clip's
    last frame are all found, in the reference's order."""
    head = _head([], seed=9)
    cfg = head.config
    with torch.no_grad():
        # blank wherever hidden unit 0 is quiet, token 4 where it is loud
        head.joint.head.bias[:cfg.vocab_size] = -1e4
        head.joint.head.bias[cfg.blank_token_id] = 0.0
        head.joint.head.weight[4, 0] = 1e3
        head.encoder_projector.weight.zero_()
        head.encoder_projector.bias.zero_()
        head.encoder_projector.weight[0, 0] = 1.0
        lens = [23, 23, 9]
        states = torch.zeros(len(lens), 23, 24)
        for row, frames in enumerate([[0, 3, 4, 11, 22], [7, 8, 9, 10], [8]]):
            states[row, frames, 0] = 100.0
        batched = head.greedy_decode(states, torch.tensor(lens))
        expected = [_reference(head, states[i, :n]) for i, n in enumerate(lens)]
    assert batched == expected
    assert [len(t) for t in batched] == [5 * cfg.max_symbols_per_step, 4 * cfg.max_symbols_per_step,
                                         cfg.max_symbols_per_step]


def test_prompt_changes_the_frames():
    head = _head([], num_prompts=6)
    torch.manual_seed(3)
    states = torch.randn(2, 12, 24)
    with torch.no_grad():
        a = head.project_frames(states, torch.tensor([0, 0]))
        b = head.project_frames(states, torch.tensor([0, 5]))
        batched = head.greedy_decode(states, torch.tensor([12, 12]), torch.tensor([1, 4]))
        expected = [_reference(head, states[i], prompt_id=p) for i, p in enumerate([1, 4])]
    assert torch.equal(a[0], b[0]) and not torch.allclose(a[1], b[1])
    assert batched == expected


def test_written_out_lstm_matches_nn_lstm():
    head = _head([], seed=4)
    torch.manual_seed(5)
    tokens = torch.randint(0, 10, (3,))
    h, c = torch.randn(3, 2, 16), torch.randn(3, 2, 16)
    with torch.no_grad():
        out, h1, c1 = head.decoder(tokens, h, c)
        lstm = torch.nn.LSTM(16, 16, num_layers=2, batch_first=True)
        lstm.load_state_dict(head.decoder.lstm.state_dict())
        ref, (h2, c2) = lstm(
            head.decoder.embedding(tokens)[:, None], (h.transpose(0, 1).contiguous(), c.transpose(0, 1).contiguous()),
        )
    assert torch.allclose(out, head.decoder.decoder_projector(ref[:, 0]), atol=1e-6)
    assert torch.allclose(h1, h2.transpose(0, 1), atol=1e-6) and torch.allclose(c1, c2.transpose(0, 1), atol=1e-6)


def test_tdt_moves_on_after_max_symbols_at_one_frame():
    """A head stuck on (token, duration 0) at a frame: as in NeMo, it emits
    max_symbols there and moves on, so later frames still decode. HF's loop
    has no such guard and would spend the clip's step budget at frame 1."""
    head = _head([0, 1, 2, 3, 4], seed=8)
    cfg = head.config
    with torch.no_grad():
        # blank with duration 1 wherever hidden unit 0 is quiet ...
        head.joint.head.bias[:cfg.vocab_size] = -1e4
        head.joint.head.bias[cfg.blank_token_id] = 0.0
        head.joint.head.bias[cfg.vocab_size + 1] = 50.0
        # ... and token 4 with duration 0 where it is loud
        head.joint.head.weight[4, 0] = head.joint.head.weight[cfg.vocab_size, 0] = 1e3
        head.encoder_projector.weight.zero_()
        head.encoder_projector.bias.zero_()
        head.encoder_projector.weight[0, 0] = 1.0
        states = torch.zeros(1, 5, 24)
        states[0, 1, 0] = states[0, 3, 0] = 100.0
        tokens = head.greedy_decode(states, torch.tensor([5]))
        expected = _reference(head, states[0])
    assert tokens[0] == expected == [4] * (2 * cfg.max_symbols_per_step)


def _captured_runners(node):
    import gc
    from unittest import mock

    from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner

    # an earlier test's graphs freed by the collector mid-capture would void it
    gc.collect()
    runners = {}
    for label, config in node.get_piecewise_accelerator_graph_configs(torch.device("cuda", 0), torch.float32).items():
        runner = PiecewiseAcceleratorGraphRunner(
            label=label, config=config, resources={}, step_runner=mock.Mock(), device=torch.device("cuda", 0),
            autocast_dtype=None, num_slots=1,
        )
        runner.warmup_and_capture(None)  # no plan: every bucket
        assert not runner.dropped_shapes
        runners[label] = runner
    return runners


class _Replays:
    """A runner that notes which regions replayed."""

    def __init__(self, runner, label, seen):
        self._runner, self._label, self._seen = runner, label, seen

    def can_run(self, *args):
        return self._runner.can_run(*args)

    def run(self, *args, **kwargs):
        self._seen.add(self._label)
        return self._runner.run(*args, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("durations", [[0, 1, 2, 3, 4], []], ids=["tdt", "rnnt"])
@pytest.mark.parametrize("lens,replayed", [
    ([17, 5, 30], {"greedy_T512"}),
    # each row to its own bucket: both captures replay, under one frames buffer
    ([600, 3, 41, 520, 9, 77], {"greedy_T512", "greedy_T8192"}),
    # more long rows than the long capture holds: it replays a part at a time
    ([600, 530, 700, 650, 520, 610], {"greedy_T8192"}),
], ids=["short", "mixed", "wide-long"])
def test_captured_chunks_match_eager(durations, lens, replayed):
    from mstar.model.parakeet.submodules import TransducerSubmodule

    node = TransducerSubmodule(_head(durations, seed=6).cuda())
    assert node._step is not node.head.step, "the captures should replay the compiled step"
    # as the engine does before warmup: the compiled step captures inference-only
    node.requires_grad_(False)
    seen = set()
    runners = {label: _Replays(runner, label, seen) for label, runner in _captured_runners(node).items()}
    torch.manual_seed(7)
    states = torch.zeros(len(lens), max(lens), 24, device="cuda")
    for i, n in enumerate(lens):
        states[i, :n] = torch.randn(n, 24, device="cuda")
    lengths = torch.tensor(lens, device="cuda")
    with torch.no_grad():
        captured = [t.tolist() for t in node._decode(states, lengths, None, lens, runners)]
        eager = [t.tolist() for t in node._decode(states, lengths, None, lens, {})]
    assert seen == replayed
    assert captured == eager
    assert sum(map(len, eager)) > len(lens), "the random head emitted nothing; the test would be vacuous"


def test_rows_past_the_working_captures_decode_uncaptured():
    """The engine keeps a region whose widest capture failed at warmup; the
    rows no remaining capture holds decode uncaptured, the rest replay."""
    from mstar.model.parakeet.submodules import TransducerSubmodule

    class Region:
        # 600-700 frames is the long bucket, whose widest capture is 4 rows;
        # here it failed and 1 and 2 remain
        def __init__(self):
            self.replayed = []

        def can_run(self, b):
            return b <= 2

        def run(self, static_inputs, real_bs):
            self.replayed.append(real_bs)
            if static_inputs:
                self.state = static_inputs
            return node.head.advance(self.state, node._step)

    node = TransducerSubmodule(_head([0, 1, 2, 3, 4], seed=6), compile_step=False)
    region = Region()
    lens = [600, 530, 700, 650, 520, 610]
    torch.manual_seed(7)
    states = torch.zeros(len(lens), max(lens), 24)
    for i, n in enumerate(lens):
        states[i, :n] = torch.randn(n, 24)
    with torch.no_grad():
        tokens = node._decode(states, torch.tensor(lens), None, lens, {"greedy_T8192": region})
        expected = node._decode(states, torch.tensor(lens), None, lens, {})
    # rows 0-3 are one 4-row part (uncaptured), rows 4-5 a 2-row replay
    assert set(region.replayed) == {2}
    assert [t.tolist() for t in tokens] == [t.tolist() for t in expected]


def test_greedy_runs_in_full_fp32(monkeypatch):
    """The engine runs fp32 GEMMs in TF32; decoding and the captured chunk
    lift that, and put it back."""
    from types import SimpleNamespace

    from mstar.model.parakeet.submodules import TransducerSubmodule

    head = _head([0, 1, 2, 3, 4])
    seen = []
    advance = head.advance

    def spy(state, step=None):
        seen.append(torch.get_float32_matmul_precision())
        return advance(state, step)

    monkeypatch.setattr(head, "advance", spy)
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        with torch.no_grad():
            head.greedy_decode(torch.randn(2, 6, 24), torch.tensor([6, 4]))
            frames = head.project_frames(torch.randn(2, 6, 24), None)
            TransducerSubmodule(head)._capture(SimpleNamespace(static_inputs=head.start(frames, torch.tensor([6, 4]))))
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)
    assert seen and set(seen) == {"highest"}
