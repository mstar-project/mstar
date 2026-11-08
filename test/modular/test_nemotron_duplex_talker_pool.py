"""The talker on the engine's KV pool: the step declares both streams of every
session as segments of one combined plan, a session's first step prefills the
speaker warm-up with its first frame, the sampling noise is drawn per row ahead
of the (captured) forward and reproduces the generator-driven sampling, and the
MaskGIT schedule is computed on the host. Tiny random weights, CPU."""
from types import SimpleNamespace

import pytest
import torch
from test_nemotron_duplex_codec import tiny_codec
from test_nemotron_duplex_talker_batch import CHAR_PAD, S2C, TINY, make_talker

from mstar.engine.resources import AttentionStep, KVStep, PositionStep
from mstar.engine.resources.recurrent import RecurrentStep
from mstar.model.nemotron_duplex.components.eartts_talker import _rand_rows, maskgit_schedule
from mstar.model.nemotron_duplex.config import (
    CODEC_STATE,
    TALKER_ATTN,
    TALKER_CFG_LABEL,
    TALKER_KV,
    TALKER_POS,
    NemotronDuplexConfig,
)
from mstar.model.nemotron_duplex.submodules import EarTTSTalkerSubmodule

P = 37   # speaker warm-up positions


def fwd(handle: str, iteration: int = 0) -> SimpleNamespace:
    """The engine keys request state by the worker-local handle, not the wire id."""
    return SimpleNamespace(rid_handle=handle, request_id=f"wire-{handle}",
                           dynamic_loop_iter_counts={"talker_decode_loop": iteration})


def make_sub(mapping=S2C) -> EarTTSTalkerSubmodule:
    cfg = NemotronDuplexConfig()
    cfg.eartts = TINY
    return EarTTSTalkerSubmodule(make_talker(), cfg, codec=tiny_codec(), subword_to_char=mapping, char_pad_idx=CHAR_PAD)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA exact-width conditioning"))])
def test_token_conditioning_preserves_exact_physical_width_and_padding(dtype, device):
    from mstar.engine.accelerator_graph_backend import get_accelerator_graph_backend
    from mstar.engine.accelerator_graph_runner import capture_into_graph
    from mstar.engine.conditional_capture import ConditionalWarmup

    mapping = {**S2C, 12: tuple(range(12))}
    sub = make_sub(mapping).to(device=device, dtype=dtype).requires_grad_(False)
    backend = get_accelerator_graph_backend(sub._characters.device) if device == "cuda" else None
    captured = {}
    for ids in ([7], [5, 9], [7, 12, 12, 12], [99, 99], [5, 7, 9, 12]):
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            values = torch.tensor(ids, device=device)
            def run(values=values):
                return sub.talker.embed_subword.from_token_ids(
                    values, sub._char_lengths, sub._char_offsets, sub._characters, sub._char_widths,
                )
            if device == "cuda":
                if len(ids) not in captured:
                    with ConditionalWarmup(backend):
                        run()
                    graph, output = capture_into_graph(
                        run, backend.graph_pool_handle(), values.device, None, conditionals=True,
                    )
                    captured[len(ids)] = values, graph, output
                target, graph, actual = captured[len(ids)]
                target.copy_(values)
                graph.replay()
                backend.synchronize()
            else:
                actual = run()
            expected = sub.talker.text_conditioning_from_ids(ids, mapping, CHAR_PAD)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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


def test_empty_chunk_runs_a_pad_frame():
    """The token stream ends on an empty chunk when its last token was taken
    before its end arrived. The talker runs a PAD frame rather than raising or
    skipping: either drops the rid from its stream-ended loop, and the worker's
    stop then fails the whole batch."""
    sub = make_sub()
    for inputs in ({"new_token": []}, {}):
        out = sub.prepare_inputs("talker_decode", fwd("a"), inputs, is_final_stream_chunk=True)
        assert out.input_ids.tolist() == [sub.config.text_pad_id] and out.kwargs["first"] is True


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("final,token", [(True, False), (True, True), (False, False)])
def test_each_step_emits_its_frame_and_the_last_the_calls_right_edge(final, token, first):
    """A step emits its frame's final samples, a call's first frame less the
    overlap-add's trimmed left edge. The PAD frame of the empty final chunk
    only ends the loop (one audio frame per text token, as offline): it emits
    the call's right edge as it stood before that frame, nothing for a call
    with no frame. A final chunk carrying a token emits its frame and the edge
    after it."""
    sub = make_sub()
    edge = sub.codec.stream_edge
    inputs = {"new_token": [torch.tensor([5])] if token else [],
              "codec_tokens": [torch.zeros(TINY.num_quantizers, dtype=torch.long)]}
    ni = sub.prepare_inputs("talker_decode", fwd("a", 0 if first else 1), inputs, is_final_stream_chunk=final)
    frame = torch.arange(100, dtype=torch.int16).view(1, -1)
    end = torch.arange(-1, -2 * edge - 1, -1, dtype=torch.int16).view(1, -1)        # before | after
    out = {"codec_tokens": [torch.arange(TINY.num_quantizers)], "audio_chunk": [frame], "audio_end": [end]}
    sub.postprocess("a", None, out, inputs=ni)
    if final and not token:
        assert set(out) == ({"audio_chunk"} if not first else set())
        if not first:
            assert torch.equal(out["audio_chunk"][0], end[..., :edge])
        return
    want = frame[..., edge:] if first else frame
    if final:
        want = torch.cat([want, end[..., edge:]], dim=-1)
    assert torch.equal(out["audio_chunk"][0], want)
    if not (first or final):
        assert out["audio_chunk"][0] is frame                  # the engine's row view, untouched


def test_step_declares_both_streams_in_one_combined_plan():
    sub = make_sub()
    # first step of "a" (warm-up + frame), steady state for "b"
    first = sub.prepare_inputs("talker_decode", fwd("a"), {"new_token": [torch.tensor([5])]})
    later = sub.prepare_inputs("talker_decode", fwd("b", 1), {
        "new_token": [torch.tensor([7])], "codec_tokens": [torch.zeros(TINY.num_quantizers, dtype=torch.long)],
    })
    assert first.input_seq_len == P + 1 and first.kwargs["first"] is True
    assert later.input_seq_len == 1 and later.kwargs["first"] is False
    step = sub.declare_step("talker_decode", ["a", "b"], [first, later])
    assert set(step.keys()) == {TALKER_KV, TALKER_ATTN, TALKER_POS, CODEC_STATE}
    # one decoded frame per session, in its own codec stream slot
    codec = step.get(CODEC_STATE)
    assert isinstance(codec, RecurrentStep)
    assert [(s.request_id, s.label, s.span) for s in codec.segments] == [("a", "main", 1), ("b", "main", 1)]
    kv = step.get(TALKER_KV)
    assert isinstance(kv, KVStep) and kv.combined_labels == {("main", "uncond"): TALKER_CFG_LABEL}
    assert isinstance(step.get(TALKER_ATTN), AttentionStep) and step.get(TALKER_ATTN).causal
    assert isinstance(step.get(TALKER_POS), PositionStep)
    # label-major: every request's cond segment, then every request's uncond segment
    assert [(s.request_id, s.label, s.span) for s in step.segments] == [
        ("a", "main", P + 1), ("b", "main", 1), ("a", "uncond", P + 1), ("b", "uncond", 1),
    ]


def test_frame_fusion_reads_prepared_previous_codes_inside_forward(monkeypatch):
    sub = make_sub()
    e = TINY
    inputs = [
        sub.prepare_inputs("talker_decode", fwd("a"), {"new_token": [torch.tensor([5])]}),
        sub.prepare_inputs("talker_decode", fwd("b", 1), {
            "new_token": [torch.tensor([7])], "codec_tokens": [torch.zeros(e.num_quantizers, dtype=torch.long)],
        }),
    ]
    backbone = []
    sub.talker.backbone_pooled = lambda x, label: backbone.append(x.clone()) or x
    sub.talker.generate_step = lambda hidden, **kwargs: torch.zeros(len(hidden), 1, e.num_quantizers, dtype=torch.long)
    monkeypatch.setattr("mstar.model.nemotron_duplex.submodules.codec_step", lambda codec, pool, codes: SimpleNamespace(
        final=torch.zeros(len(codes), 1), end_before=torch.zeros(len(codes), 1), end_after=torch.zeros(len(codes), 1),
    ))
    for current in (inputs, inputs[1:]):
        n = len(current)
        eng = SimpleNamespace(request_ids=["a", "b"][:n], per_request_states=None, resources={
            CODEC_STATE: object(),
            TALKER_ATTN: SimpleNamespace(
                select_last_hidden=lambda hidden, label, n=n: torch.zeros(2 * n, e.hidden_size)),
        })
        pre = sub.preprocess("talker_decode", eng, current)
        assert pre["noise_u"].shape == (n, e.inference_num_iter, 1, e.mog_num_predictions)
        assert pre["noise_eps"].shape == (n, e.inference_num_iter, 1, e.code_dim)
        warm_c, warm_u, _ = sub.warmup()
        frame_c, frame_u = sub.talker.frame_inputs(
            pre["prev_codes"], sub.talker.text_conditioning_from_ids(pre["input_ids"].tolist(), S2C, CHAR_PAD),
            pre["input_ids"], sub.config.text_eos_id,
        )
        expected = torch.cat([
            torch.cat([warm, frame[i:i + 1]]) if inp.kwargs["first"] else frame[i:i + 1]
            for warm, frame in ((warm_c, frame_c), (warm_u, frame_u))
            for i, inp in enumerate(current)
        ])
        sub.forward_batched("talker_decode", eng, **pre)
        torch.testing.assert_close(backbone[-1], expected, rtol=0, atol=0)
        # A replay's new codes must affect the captured fusion, without another preprocess.
        pre["prev_codes"].fill_(1)
        sub.forward_batched("talker_decode", eng, **pre)
        assert not torch.equal(backbone[-1], expected)
        previous = backbone[-1].clone()
        pre["input_ids"].fill_(9)  # a later replay's longer token is conditioned inside the body
        sub.forward_batched("talker_decode", eng, **pre)
        assert not torch.equal(backbone[-1], previous)


def test_decode_capture_config_commits_two_labels_per_row():
    sub = make_sub()
    cfgs = sub.get_accelerator_graph_configs(torch.device("cpu"))
    assert len(cfgs) == 1 and cfgs[0].capture_graph_walk == "talker_decode"
    assert cfgs[0].total_tokens_multiplier == 2 and cfgs[0].compile is False
    assert cfgs[0].single_request_inputs.input_seq_len == 1
    assert cfgs[0].single_request_inputs.kwargs["first"] is False


def test_the_next_frame_reads_its_supplied_codes_before_previous_postprocess():
    sub = make_sub()
    codes = torch.arange(TINY.num_quantizers)
    later = sub.prepare_inputs("talker_decode", fwd("a", 1),
                               {"new_token": [torch.tensor([5])], "codec_tokens": [codes]})
    assert later.input_seq_len == 1 and later.kwargs["first"] is False
    torch.testing.assert_close(later.tensor_inputs["prev_codes"], codes)
    sub.postprocess("a", None, {"codec_tokens": [torch.zeros_like(codes)]})
    torch.testing.assert_close(later.tensor_inputs["prev_codes"], codes)
    with pytest.raises(ValueError, match="previous codec tokens"):
        sub.prepare_inputs("talker_decode", fwd("a", 1), {"new_token": [torch.tensor([5])]})
    # Capture and padding bypass prepare_inputs, so their declared tensor is
    # also a valid previous frame rather than a hidden request-state fallback.
    (cfg,) = sub.get_accelerator_graph_configs(torch.device("cpu"))
    template = cfg.single_request_inputs
    torch.testing.assert_close(template.tensor_inputs["prev_codes"], sub.warmup()[2])
    prepared = sub.preprocess("talker_decode", SimpleNamespace(request_ids=["a", "pad"]), [later, template])
    assert prepared["prev_codes"].shape == (2, TINY.num_quantizers) and prepared["spans"] == [1, 1]


class _DenseAttend:
    """Stands in for the engine's planned attention over this step's packed
    tokens: one causal sequence, FlashInfer's default scale (the declared head
    dim ** -0.5). Records what the layer handed it."""

    def __init__(self):
        self.seen = None

    def __call__(self, q, k, v):
        self.seen = (q, k, v)
        n, d = q.shape[0], q.shape[-1]
        s = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * d ** -0.5
        s = s.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
        return torch.einsum("hqk,khd->qhd", s.softmax(-1), v.float()).to(q.dtype)


@pytest.mark.parametrize("kv_head_dim", [128, TINY.head_dim])
def test_forward_pooled_pads_qkv_to_the_kv_head_dim_exactly(kv_head_dim):
    """The pool's head dim is FlashInfer's 128 (the talker's 72 zero-padded);
    the padded attention must equal the plain per-sequence attention."""
    import dataclasses

    from mstar.model.nemotron_duplex.components.eartts_talker import _Attn, _rope_cos_sin

    torch.manual_seed(0)
    cfg = dataclasses.replace(TINY, kv_head_dim=kv_head_dim)
    attn = _Attn(cfg, qk_norm=True)
    for p in attn.parameters():
        p.data.normal_(0, 0.3)
    attn.attend = _DenseAttend()
    n = 5
    x = torch.randn(n, cfg.hidden_size)
    cos, sin = _rope_cos_sin(torch.arange(n), cfg.rope_theta_local, cfg.head_dim)

    pooled = attn.forward_pooled(x, cos, sin)

    q, k, v = attn.attend.seen
    assert q.shape == k.shape == v.shape == (n, cfg.num_attention_heads, kv_head_dim)
    for t in (q, k, v):
        assert torch.equal(t[..., cfg.head_dim:], torch.zeros_like(t[..., cfg.head_dim:]))
    ref = attn.forward(x[None], cos, sin, causal=True)[0]
    torch.testing.assert_close(pooled, ref, atol=1e-4, rtol=1e-4)


def test_warmup_steps_run_eager_and_steady_state_steps_hit_the_decode_capture():
    """Capture selection and preparation use the same snapshotted iteration,
    even when the mutable request has advanced or not yet postprocessed."""
    from mstar.engine.engine import ForwardPassInfoWrapper
    from mstar.model.submodule_base import InputMetadata

    sub = make_sub()
    infos = {
        "a": ForwardPassInfoWrapper(fwd("a", 4), InputMetadata(dynamic_loop_iter_counts={"talker_decode_loop": 0})),
        "b": ForwardPassInfoWrapper(fwd("b", 0), InputMetadata(dynamic_loop_iter_counts={"talker_decode_loop": 1})),
    }
    first = sub.prepare_inputs("talker_decode", infos["a"], {"new_token": [torch.tensor([5])]})
    later = sub.prepare_inputs("talker_decode", infos["b"], {
        "new_token": [torch.tensor([7])], "codec_tokens": [torch.zeros(TINY.num_quantizers, dtype=torch.long)],
    })
    assert sub.cg_key_info("talker_decode", infos) is None
    assert sub.declare_step("talker_decode", ["a", "b"], [first, later]).cg_key_info is None
    assert first.input_seq_len == P + 1 and later.input_seq_len == 1
    assert sub.cg_key_info("talker_decode", {"b": infos["b"]}) == sub.DECODE_KEY
    assert sub.declare_step("talker_decode", ["b"], [later]).cg_key_info == sub.DECODE_KEY
    (cfg,) = sub.get_accelerator_graph_configs(torch.device("cpu"))
    assert cfg.additional_key_info == sub.DECODE_KEY
