"""Batched greedy transducer decoding (TDT and RNN-T) against a plain
one-request reference loop, on tiny random heads. The decode loop through the
engine's recurrent state pool is ``test_transducer_loop.py``."""

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


def _greedy(head: TransducerHead, states, lengths, prompt_ids=None) -> list[list[int]]:
    """Each request's tokens from ``advance`` alone: the batch's state, from
    the start state, stepped a chunk at a time until no row is active."""
    rows = states.shape[0]
    state = {name: start.expand(rows, *start.shape).clone() for name, start in head.initial_state().items()}
    chunks = []
    for _ in range(head.config.decode_iterations(states.shape[1])):
        tokens, active = head.advance(states, lengths, state, prompt_ids=prompt_ids)
        chunks.append(tokens)
        if not active.any():
            break
    return [[token for token in row if token >= 0] for row in torch.cat(chunks, dim=1).tolist()]


@pytest.mark.parametrize("durations", [[0, 1, 2, 3, 4], []], ids=["tdt", "rnnt"])
def test_batched_greedy_matches_reference(durations):
    head = _head(durations)
    torch.manual_seed(1)
    lens = [17, 5, 30, 1]
    states = torch.zeros(len(lens), max(lens), 24)
    for i, n in enumerate(lens):
        states[i, :n] = torch.randn(n, 24)
    with torch.no_grad():
        batched = _greedy(head, states, torch.tensor(lens))
        expected = [_reference(head, states[i, :n]) for i, n in enumerate(lens)]
    assert batched == expected
    assert any(batched), "the random head emitted nothing; the test would be vacuous"


def test_rnnt_caps_symbols_per_frame():
    head = _head([], seed=2)
    with torch.no_grad():
        # never blank: every frame emits exactly max_symbols tokens
        head.joint.head.bias[head.config.blank_token_id] = -1e4
        tokens = _greedy(head, torch.randn(1, 4, 24), torch.tensor([4]))
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
        batched = _greedy(head, states, torch.tensor(lens))
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
        batched = _greedy(head, states, torch.tensor([12, 12]), torch.tensor([1, 4]))
        expected = [_reference(head, states[i], prompt_id=p) for i, p in enumerate([1, 4])]
    assert torch.equal(a[0], b[0]) and not torch.allclose(a[1], b[1])
    assert batched == expected


def test_the_start_state_fills_the_pool_blocks():
    head = _head([0, 1, 2, 3, 4])
    blocks = head.config.state_blocks()
    start = head.initial_state()
    assert start.keys() == blocks.keys()
    for name, block in blocks.items():
        assert start[name].shape == block.shape and start[name].dtype == block.dtype, name
    zeros = torch.zeros(1, 2, 16)
    with torch.no_grad():
        out, h, c = head.decoder(torch.tensor([head.config.start_token_id]), zeros, zeros)
    assert torch.equal(start["dec_out"], out[0]) and torch.equal(start["h"], h[0]) and torch.equal(start["c"], c[0])
    assert int(start["cursor"]) == int(start["symbols"]) == 0


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
        tokens = _greedy(head, states, torch.tensor([5]))
        expected = _reference(head, states[0])
    assert tokens[0] == expected == [4] * (2 * cfg.max_symbols_per_step)


def test_the_node_decodes_in_full_fp32(monkeypatch):
    """The engine runs fp32 GEMMs in TF32. The transducer node asks it for full
    fp32 around its forward and capture, and computes its start state in full
    fp32 too, then puts the setting back."""
    from mstar.engine.fp32_precision import node_fp32_precision
    from mstar.model.parakeet.submodules import TransducerSubmodule

    head = _head([0, 1, 2, 3, 4])
    seen = []
    decoder = head.decoder.forward

    def spy(*args):
        seen.append(torch.get_float32_matmul_precision())
        return decoder(*args)

    monkeypatch.setattr(head.decoder, "forward", spy)
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        node = TransducerSubmodule(head, compile_step=False)
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)
    assert seen == ["highest"]
    assert node_fp32_precision("transducer", node).matmul == "highest"
