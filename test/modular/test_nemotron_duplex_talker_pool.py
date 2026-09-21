"""The talker on the engine's KV pool: the step declares both streams of every
session as segments of one combined plan, a session's first step prefills the
speaker warm-up with its first frame, the sampling noise is drawn per row ahead
of the (captured) forward and reproduces the generator-driven sampling, and the
MaskGIT schedule is computed on the host. Tiny random weights, CPU."""
from types import SimpleNamespace

import torch

from mstar.engine.resources import AttentionStep, KVStep, PositionStep
from mstar.model.nemotron_duplex.components.eartts_talker import _rand_rows, maskgit_schedule
from mstar.model.nemotron_duplex.config import (
    TALKER_ATTN,
    TALKER_CFG_LABEL,
    TALKER_KV,
    TALKER_POS,
    NemotronDuplexConfig,
)
from mstar.model.nemotron_duplex.submodules import EarTTSTalkerSubmodule
from test.modular.test_nemotron_duplex_talker_batch import CHAR_PAD, S2C, TINY, make_talker

P = 37   # speaker warm-up positions


def make_sub() -> EarTTSTalkerSubmodule:
    cfg = NemotronDuplexConfig()
    cfg.eartts = TINY
    return EarTTSTalkerSubmodule(make_talker(), cfg, subword_to_char=S2C, char_pad_idx=CHAR_PAD)


def test_maskgit_schedule_matches_the_reference_formula():
    """The reference forms ``ceil(masking_rate(linspace) * d)`` on the device and
    reads it back per iteration; the host schedule is the same arithmetic."""
    from mstar.model.nemotron_duplex.components.eartts_talker import _masking_rate

    for num_iter, d in ((8, 31), (4, TINY.num_quantizers), (8, 4)):
        ks = maskgit_schedule(num_iter, 3.0, d)
        rates = torch.linspace(0.0, 1.0, num_iter + 1)[:-1].unsqueeze(-1)
        num_maskings = torch.ceil(_masking_rate(rates, 3.0) * d).long()
        ref = (num_maskings - torch.nn.functional.pad(num_maskings[1:], [0, 0, 0, 1]))[:, 0].tolist()
        assert list(ks) == ref
        assert len(ks) == num_iter and sum(ks) == d and all(k >= 0 for k in ks)
    # the real schedule's first iteration fills nothing (ceil(30.98) == 31): skipped, as in the reference
    assert maskgit_schedule(8, 3.0, 31)[0] == 0 and any(k > 0 for k in maskgit_schedule(8, 3.0, 31))
    assert maskgit_schedule(8, 3.0, 31) is maskgit_schedule(8, 3.0, 31)   # cached per config


def test_warmup_and_frame_inputs_shapes():
    sub = make_sub()
    cond, uncond, prev = sub.warmup()
    assert cond.shape == (P, TINY.hidden_size) and uncond.shape == (P, TINY.hidden_size)
    assert prev.shape == (TINY.num_quantizers,) and torch.all(prev == TINY.codebook_size)  # speech-pad frame
    assert sub.warmup() is sub._warmup                 # computed once per speaker
    ids = torch.tensor([5, sub.config.text_eos_id])
    conds = sub.talker.text_conditioning_from_ids(ids.tolist(), S2C, CHAR_PAD)
    prev2 = torch.stack([prev, prev])
    fc, fu = sub.talker.frame_inputs(prev2, conds, ids, sub.config.text_eos_id)
    assert fc.shape == fu.shape == (2, TINY.hidden_size)


def test_pre_drawn_noise_reproduces_generator_sampling():
    """generate_step with noise drawn ahead per row == generate_step drawing
    from those rows' generators itself."""
    t = make_talker()
    n, e = 3, TINY
    kw = dict(num_iter=4, guidance_scale=0.5, noise_scale=0.8, top_p=0.8)
    hidden = torch.randn(n, 1, e.hidden_size)
    uncond = torch.randn(n, 1, e.hidden_size)
    gens_a = [torch.Generator().manual_seed(100 + i) for i in range(n)]
    gens_b = [torch.Generator().manual_seed(100 + i) for i in range(n)]
    noise = []
    for k in maskgit_schedule(4, e.mog_exponent, e.num_quantizers):
        if k == 0:
            noise.append((None, None))
            continue
        u = _rand_rows((n, 1, e.mog_num_predictions), torch.device("cpu"), torch.float32, gens_a, torch.rand)
        eps = _rand_rows((n, 1, e.code_dim), torch.device("cpu"), torch.float32, gens_a, torch.randn)
        noise.append((u, eps))
    with_noise = t.generate_step(hidden, hidden_uncond=uncond, noise=noise, **kw)
    with_gens = t.generate_step(hidden, hidden_uncond=uncond, generator=gens_b, **kw)
    torch.testing.assert_close(with_noise, with_gens)


def test_step_declares_both_streams_in_one_combined_plan():
    sub = make_sub()
    # first step of "a" (warm-up + frame), steady state for "b"
    first = sub.prepare_inputs("talker_decode", SimpleNamespace(request_id="a"), {"new_token": [torch.tensor([5])]})
    sub.request_state("b").add(sub.PREV_CODES_KEY, torch.zeros(TINY.num_quantizers, dtype=torch.long))
    later = sub.prepare_inputs("talker_decode", SimpleNamespace(request_id="b"), {"new_token": [torch.tensor([7])]})
    assert first.input_seq_len == P + 1 and first.kwargs["first"] is True
    assert later.input_seq_len == 1 and later.kwargs["first"] is False
    step = sub.declare_step("talker_decode", ["a", "b"], [first, later])
    assert set(step.keys()) == {TALKER_KV, TALKER_ATTN, TALKER_POS}
    kv = step.get(TALKER_KV)
    assert isinstance(kv, KVStep) and kv.combined_labels == {("main", "uncond"): TALKER_CFG_LABEL}
    assert isinstance(step.get(TALKER_ATTN), AttentionStep) and step.get(TALKER_ATTN).causal
    assert isinstance(step.get(TALKER_POS), PositionStep)
    # label-major: every request's cond segment, then every request's uncond segment
    assert [(s.request_id, s.label, s.span) for s in step.segments] == [
        ("a", "main", P + 1), ("b", "main", 1), ("a", "uncond", P + 1), ("b", "uncond", 1),
    ]


def test_preprocess_packs_rows_label_major_and_draws_noise_per_row():
    sub = make_sub()
    e = TINY
    sub.request_state("b").add(sub.PREV_CODES_KEY, torch.zeros(e.num_quantizers, dtype=torch.long))
    inputs = [
        sub.prepare_inputs("talker_decode", SimpleNamespace(request_id="a"), {"new_token": [torch.tensor([5])]}),
        sub.prepare_inputs("talker_decode", SimpleNamespace(request_id="b"), {"new_token": [torch.tensor([7])]}),
    ]
    eng = SimpleNamespace(request_ids=["a", "b"], resources={}, per_request_states=None)
    pre = sub.preprocess("talker_decode", eng, inputs)
    total = (P + 1) + 1
    assert pre["x"].shape == (2 * total, e.hidden_size)     # [a cond (38), b cond (1), a uncond (38), b uncond (1)]
    assert pre["spans"] == [P + 1, 1]
    assert pre["noise_u"].shape == (e.inference_num_iter, 2, 1, e.mog_num_predictions)
    assert pre["noise_eps"].shape == (e.inference_num_iter, 2, 1, e.code_dim)
    # a steady-state batch is exactly two rows per session
    inputs2 = [sub.prepare_inputs("talker_decode", SimpleNamespace(request_id="b"), {"new_token": [torch.tensor([7])]})]
    pre2 = sub.preprocess("talker_decode", SimpleNamespace(request_ids=["b"], resources={}, per_request_states=None),
                          inputs2)
    assert pre2["x"].shape == (2, e.hidden_size) and pre2["spans"] == [1]


def test_decode_capture_config_commits_two_labels_per_row():
    sub = make_sub()
    cfgs = sub.get_cuda_graph_configs(torch.device("cpu"))
    assert len(cfgs) == 1 and cfgs[0].capture_graph_walk == "talker_decode"
    assert cfgs[0].total_tokens_multiplier == 2 and cfgs[0].compile is False
    assert cfgs[0].single_request_inputs.input_seq_len == 1
    assert cfgs[0].single_request_inputs.kwargs["first"] is False


def test_postprocess_carries_the_codes_into_the_next_frame():
    sub = make_sub()
    codes = torch.arange(TINY.num_quantizers)
    sub.postprocess("a", None, {"codec_tokens": [codes]})
    torch.testing.assert_close(sub.request_state("a")[sub.PREV_CODES_KEY], codes)
