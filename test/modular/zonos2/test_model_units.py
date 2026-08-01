"""Unit tests for Zonos2 model pieces that don't need a checkpoint or GPU.

Two of these guard fixes on this branch:
  * the MoE router selects ``get_num_experts_per_tok(layer)`` experts, so the
    ``special_topk_layers`` override actually reaches routing;
  * speaker conditioning is injected at exactly the speaker token position(s).

``Zonos2Router.forward`` and the model's ``out_norm`` route through mstar's
flashinfer RMSNorm (CUDA-only), so the router test checks ``top_k`` directly
(the resolved value *is* the routed count) and the speaker test swaps
``out_norm`` for ``Identity`` — irrelevant to the injection, and row-wise
anyway, so per-position isolation is preserved.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.zonos2.components.language_model import (
    Zonos2DecoderLayer,
    Zonos2ForCausalLM,
    Zonos2Router,
    build_zonos2_moe,
    softcap,
)
from mstar.model.zonos2.config import Zonos2Config


# -- softcap ----------------------------------------------------------------
def test_softcap_bounds_and_monotonic():
    cap = 15.0
    # Moderate range so tanh stays unsaturated (strictly monotonic in float32).
    x = torch.linspace(-30, 30, 301)
    y = softcap(x, cap)
    assert (y.abs() < cap).all()                       # strictly bounded
    assert (y.diff() > 0).all()                        # monotonic increasing
    assert torch.allclose(softcap(torch.zeros(1), cap), torch.zeros(1))
    # Extreme inputs saturate at (never beyond) the cap.
    assert softcap(torch.tensor(1e4), cap).item() <= cap
    assert softcap(torch.tensor(-1e4), cap).item() >= -cap


# -- MoE router top-k per layer --------------------------------------------
def _moe_cfg(**kw) -> Zonos2Config:
    base = dict(
        hidden_size=16, moe_n_experts=4, num_experts_per_tok=1,
        moe_router_dim=8, moe_start_from_layer=0,
    )
    base.update(kw)
    return Zonos2Config(**base)


def test_router_topk_honors_special_layers():
    # ``top_k`` is the count the router feeds to ``torch.topk`` when selecting
    # experts, so this pins the special-topk override to routing without
    # running forward (which routes through the CUDA-only RMSNorm).
    cfg = _moe_cfg(num_experts_per_tok=1, special_topk_layers={2: 3})
    assert Zonos2Router(cfg, layer_id=0).top_k == 1   # global default
    assert Zonos2Router(cfg, layer_id=2).top_k == 3   # per-layer override
    # String keys (checkpoint JSON form) resolve the same way.
    cfg_str = _moe_cfg(num_experts_per_tok=1, special_topk_layers={"2": 3})
    assert Zonos2Router(cfg_str, layer_id=2).top_k == 3


def test_router_topk_is_static_int():
    # CUDA-graph safety: ``top_k`` must be a plain int fixed at construction,
    # not a tensor / callable / per-token value. A future config change that
    # made routing count data-dependent would silently reintroduce dynamic
    # shapes into the (to-be-captured) MoE dispatch — this pins it.
    router = Zonos2Router(_moe_cfg(num_experts_per_tok=2), layer_id=0)
    assert isinstance(router.top_k, int)
    assert router.top_k == 2


# -- MoE block wiring + checkpoint name remap -------------------------------
def _moe_model_cfg() -> Zonos2Config:
    return Zonos2Config(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=128,
        moe_n_experts=8, num_experts_per_tok=2, moe_router_dim=32,
        moe_start_from_layer=1, moe_end_from_layer=0, num_layers=4,
        num_qo_heads=4, num_kv_heads=4, head_dim=16,
        n_codebooks=2, text_vocab=32,
    )


def test_moe_layer_is_sparse_moe_block_with_zonos2_router():
    # The MoE feed-forward is the shared block with Zonos2's router injected
    # (not a Zonos2-private copy), so Zonos2 rides the common expert dispatch.
    from mstar.model.components import SparseMoeBlock

    cfg = _moe_model_cfg()
    layer = Zonos2DecoderLayer(cfg, layer_id=2, comm_group=CommGroup.trivial())
    assert isinstance(layer.feed_forward, SparseMoeBlock)
    assert isinstance(layer.feed_forward.gate, Zonos2Router)


def test_moe_block_threads_router_states():
    # The block must return the router's next state when asked, and pass an
    # incoming state through to the router (the EDA chain depends on both).
    cfg = _moe_model_cfg()
    block = build_zonos2_moe(cfg, layer_id=2)

    seen = {}
    incoming = torch.randn(4, cfg.moe_router_dim)

    # Stub the router so this stays CPU-only (the real one uses CUDA RMSNorm)
    # and returns a recognizable next-state.
    def stub(x, router_states=None):
        seen["router_states"] = router_states
        return (
            torch.ones(x.shape[0], 2),
            torch.zeros(x.shape[0], 2, dtype=torch.int64),
            torch.full((x.shape[0], cfg.moe_router_dim), 7.0),
        )

    block.gate.forward = stub

    out, next_states = block(
        torch.randn(4, cfg.hidden_size), incoming, return_router_states=True,
    )
    assert out.shape == (4, cfg.hidden_size)
    assert seen["router_states"] is incoming              # state passed in
    assert torch.equal(next_states, torch.full((4, cfg.moe_router_dim), 7.0))

    # Default (no flag) stays a bare tensor, matching the block's contract.
    bare = block(torch.randn(4, cfg.hidden_size), incoming)
    assert isinstance(bare, torch.Tensor)


def test_load_weights_remaps_reference_router_prefix_to_gate():
    # The reference checkpoint names the router ``router``; SparseMoeBlock
    # holds it as ``gate``. If this remap regressed, every router tensor would
    # silently stay at its init value instead of erroring — hence this test.
    cfg = _moe_model_cfg()
    model = Zonos2ForCausalLM(cfg)
    moe_layers = [i for i in range(cfg.num_layers) if cfg.is_moe_layer(i)]
    assert moe_layers, "config must have MoE layers for this test to mean anything"

    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    moe_names = {
        n for n in tensors
        if any(n.startswith(f"layers.{i}.feed_forward") for i in moe_layers)
    }
    assert any(".gate." in n for n in moe_names)

    # Synthetic checkpoint in the *reference* naming.
    ckpt = {}
    for n in moe_names:
        ref = n.replace(".feed_forward.gate.", ".feed_forward.router.")
        if ref.endswith(".experts.gate_up_proj"):
            base = ref[: -len(".gate_up_proj")]
            for w in ("w1", "w3"):
                ckpt[f"{base}.{w}"] = torch.randn(
                    cfg.moe_n_experts, cfg.moe_inter, cfg.hidden_size,
                )
        elif ref.endswith(".experts.down_proj"):
            base = ref[: -len(".down_proj")]
            ckpt[f"{base}.w2"] = torch.randn(
                cfg.moe_n_experts, cfg.hidden_size, cfg.moe_inter,
            )
        else:
            ckpt[ref] = torch.randn_like(tensors[n].data.float()).to(tensors[n].dtype)

    loaded = model.load_weights(ckpt.items())
    assert not (moe_names - loaded), f"unloaded MoE tensors: {sorted(moe_names - loaded)}"

    # Values really landed (a no-op remap would leave the init values).
    last = moe_layers[-1]
    for suffix in ("down_proj.weight", "rmsnorm_eda.weight", "balancing_biases"):
        assert torch.equal(
            tensors[f"layers.{last}.feed_forward.gate.{suffix}"].data,
            ckpt[f"layers.{last}.feed_forward.router.{suffix}"],
        ), suffix
    # And the expert w1/w3 -> gate_up_proj fusion still works through the block.
    gate_up = tensors[f"layers.{last}.feed_forward.experts.gate_up_proj"].data
    assert torch.equal(
        gate_up[:, : cfg.moe_inter, :],
        ckpt[f"layers.{last}.feed_forward.experts.w1"],
    )
    assert torch.equal(
        gate_up[:, cfg.moe_inter :, :],
        ckpt[f"layers.{last}.feed_forward.experts.w3"],
    )


# -- speaker conditioning injection ----------------------------------------
def _speaker_model(**kw) -> Zonos2ForCausalLM:
    cfg = Zonos2Config(
        num_layers=0, hidden_size=16, n_codebooks=3, codebook_size=8,
        text_vocab=10, moe_n_experts=1, **kw,
    )
    model = Zonos2ForCausalLM(cfg).eval()
    model.out_norm = nn.Identity()  # CUDA-only RMSNorm; irrelevant to injection
    # mstar's parallel embedding/linear layers leave weights uninitialized
    # (populated only at checkpoint load), so give them finite, deterministic
    # values before exercising the forward.
    torch.manual_seed(0)
    for p in model.parameters():
        nn.init.normal_(p, std=0.02)
    return model


def _stub_cache() -> SimpleNamespace:
    return SimpleNamespace(advance_seq_lens=lambda: None, set_layer_idx=lambda i: None)


def _ids(model: Zonos2ForCausalLM, T: int = 4) -> torch.Tensor:
    return torch.zeros(T, model.n_codebooks + 1, dtype=torch.long)


def test_speaker_projection_shapes():
    with_lda = _speaker_model(speaker_enabled=True, speaker_embedding_dim=5, speaker_lda_dim=4)
    assert with_lda.speaker_lda_projection.in_features == 5
    assert with_lda.speaker_lda_projection.out_features == 4
    assert with_lda.speaker_projection.in_features == 4        # fed by the LDA output
    assert with_lda.speaker_projection.out_features == 16      # hidden_size

    no_lda = _speaker_model(speaker_enabled=True, speaker_embedding_dim=5, speaker_lda_dim=None)
    assert no_lda.speaker_lda_projection is None
    assert no_lda.speaker_projection.in_features == 5          # raw embedding


def test_speaker_injection_only_at_positions():
    model = _speaker_model(speaker_enabled=True, speaker_embedding_dim=5, speaker_lda_dim=4)
    ids, cache = _ids(model, T=4), _stub_cache()
    pos = 1
    with torch.no_grad():
        base = model(ids, _stub_cache())
        spk = model(
            ids, cache,
            speaker_emb_values=torch.randn(1, 5),
            speaker_token_positions=torch.tensor([pos]),
        )
    assert not torch.allclose(base[pos], spk[pos])            # injected row changed
    for i in range(base.shape[0]):
        if i != pos:
            assert torch.allclose(base[i], spk[i])            # every other row intact


def test_speaker_disabled_ignores_values():
    model = _speaker_model(speaker_enabled=False)
    assert model.speaker_projection is None
    ids = _ids(model)
    with torch.no_grad():
        base = model(ids, _stub_cache())
        # Supplying values is a harmless no-op when the model is speaker-disabled.
        out = model(
            ids, _stub_cache(),
            speaker_emb_values=torch.randn(1, 5),
            speaker_token_positions=torch.tensor([0]),
        )
    assert torch.allclose(base, out)


# -- conditioning token ids -------------------------------------------------
# Conditioning tokens sit in the tail of the text vocab, in a fixed order:
# rate, then quality (per feature), then the clean/noisy background pair, then
# accurate-mode. Every count shifts everything after it, so these pin the exact
# arithmetic rather than just relative ordering.
_TEXT_VOCAB = 469          # 448 byte vocab + 21 rate buckets, the shipped shape
_RATE_BUCKETS = 21
_QUALITY_COUNTS = (3, 4)   # two features, 3 and 4 buckets
_BG = 2
_ACC = 1
# base = 469 - 21 - 7 - 2 - 1 = 438
_BASE = _TEXT_VOCAB - _RATE_BUCKETS - sum(_QUALITY_COUNTS) - _BG - _ACC


def test_conditioning_token_ids_are_contiguous_blocks():
    from mstar.model.zonos2.prompt import (
        accurate_mode_token_id,
        quality_token_id,
        speaker_background_token_id,
        speaking_rate_token_id,
    )

    args = (_TEXT_VOCAB, _RATE_BUCKETS, _QUALITY_COUNTS, _BG, _ACC)

    # Rate occupies [base, base + 21).
    assert speaking_rate_token_id(_TEXT_VOCAB, _RATE_BUCKETS, 0, _QUALITY_COUNTS, _BG, _ACC) == _BASE
    assert speaking_rate_token_id(_TEXT_VOCAB, _RATE_BUCKETS, 20, _QUALITY_COUNTS, _BG, _ACC) == _BASE + 20

    # Quality feature 0 follows rate; feature 1 follows feature 0's 3 buckets.
    assert quality_token_id(_TEXT_VOCAB, _RATE_BUCKETS, _QUALITY_COUNTS, 0, 0, _BG, _ACC) == _BASE + 21
    assert quality_token_id(_TEXT_VOCAB, _RATE_BUCKETS, _QUALITY_COUNTS, 1, 0, _BG, _ACC) == _BASE + 21 + 3
    assert quality_token_id(_TEXT_VOCAB, _RATE_BUCKETS, _QUALITY_COUNTS, 1, 3, _BG, _ACC) == _BASE + 21 + 6

    # Background pair (clean first), then the single accurate-mode marker.
    assert speaker_background_token_id(*args[:3], True, _BG, _ACC) == _BASE + 28
    assert speaker_background_token_id(*args[:3], False, _BG, _ACC) == _BASE + 29
    assert accurate_mode_token_id(_TEXT_VOCAB, _RATE_BUCKETS, _QUALITY_COUNTS, _BG, _ACC) == _BASE + 30
    # The whole conditioning tail ends exactly at text_vocab (the padding id).
    assert _BASE + 31 == _TEXT_VOCAB


def test_conditioning_token_ids_reject_out_of_range():
    from mstar.model.zonos2.prompt import quality_token_id, speaking_rate_token_id

    with pytest.raises(ValueError):
        speaking_rate_token_id(_TEXT_VOCAB, _RATE_BUCKETS, _RATE_BUCKETS, _QUALITY_COUNTS, _BG, _ACC)
    with pytest.raises(ValueError):
        quality_token_id(_TEXT_VOCAB, _RATE_BUCKETS, _QUALITY_COUNTS, 0, 3, _BG, _ACC)
    with pytest.raises(ValueError):
        # text_vocab too small to hold the declared buckets.
        speaking_rate_token_id(10, _RATE_BUCKETS, 0, _QUALITY_COUNTS, _BG, _ACC)


# -- prompt frame layout ----------------------------------------------------
def _builder(**kw):
    from mstar.model.zonos2.prompt import TTSPromptBuilder

    base = dict(
        n_codebooks=9, audio_pad_id=1025, text_vocab=_TEXT_VOCAB,
        prepend_silence=False, speaking_rate_num_buckets=_RATE_BUCKETS,
        quality_bucket_counts=_QUALITY_COUNTS,
        speaker_background_num_buckets=_BG, accurate_mode_num_buckets=_ACC,
    )
    base.update(kw)
    return TTSPromptBuilder(**base)


def test_prompt_without_speaker_is_unchanged():
    from mstar.model.zonos2.prompt import text_to_byte_ids

    frames = _builder().build("hi")
    assert frames.shape == (len(text_to_byte_ids("hi")), 10)
    assert frames[:, -1].tolist() == text_to_byte_ids("hi")
    assert (frames[:, :9] == 1025).all()        # audio columns are all padding


def test_prompt_speaker_slot_and_marker_order():
    from mstar.model.zonos2.prompt import text_to_byte_ids

    frames = _builder().build(
        "hi", speaker=True, clean_speaker_background=True, accurate_mode=True,
    )
    # slot, background, accurate, then the byte tokens.
    assert frames.shape[0] == 3 + len(text_to_byte_ids("hi"))
    assert frames[0, -1].item() == _TEXT_VOCAB       # slot carries the padding id
    assert frames[1, -1].item() == _BASE + 28        # clean background
    assert frames[2, -1].item() == _BASE + 30        # accurate mode
    assert frames[3, -1].item() == text_to_byte_ids("hi")[0]
    # Audio columns stay padding throughout the conditioning prefix, so the
    # slot is indistinguishable from padding until the model overwrites it.
    assert (frames[:3, :9] == 1025).all()


def test_prompt_speaker_markers_respond_to_flags():
    clean = _builder().build("hi", speaker=True, clean_speaker_background=True)
    noisy = _builder().build("hi", speaker=True, clean_speaker_background=False)
    assert clean[1, -1].item() == _BASE + 28
    assert noisy[1, -1].item() == _BASE + 29
    # accurate_mode=False omits the marker entirely (absent == expressive).
    assert clean.shape[0] == noisy.shape[0]
    assert clean.shape[0] == 2 + 4                    # slot + background + BOS/h/i/EOS

    # A checkpoint without the background token emits the slot alone.
    bare = _builder(speaker_background_num_buckets=0, accurate_mode_num_buckets=0)
    frames = bare.build("hi", speaker=True, accurate_mode=True)
    assert frames.shape[0] == 1 + 4
    assert frames[0, -1].item() == _TEXT_VOCAB


def test_prompt_speaker_slot_matches_make_speaker_slot():
    builder = _builder()
    frames = builder.build("hi", speaker=True)
    assert torch.equal(frames[:1], builder.speaker_slot())


def test_prompt_rate_and_quality_follow_speaker_markers():
    frames = _builder().build(
        "hi", speaker=True, accurate_mode=True,
        speaking_rate_bucket=5, quality_buckets=[2, None],
    )
    # slot, background, accurate, rate, quality(feature 0) — feature 1 is None
    # so it emits nothing.
    assert frames[3, -1].item() == _BASE + 5
    assert frames[4, -1].item() == _BASE + 21 + 2
    assert frames.shape[0] == 5 + 4


# -- packed-batch speaker positions ----------------------------------------
def test_speaker_token_positions_are_absolute_in_packed_batch():
    """The model index_copy's over the concatenated batch, so a request's slot
    position is its exclusive prefix sum of sequence lengths, not 0."""
    from mstar.model.submodule_base import ARNodeInputs
    from mstar.model.zonos2.submodules import Zonos2LLMSubmodule

    sub = Zonos2LLMSubmodule(
        model=nn.Linear(1, 1), n_codebooks=9, text_vocab=_TEXT_VOCAB,
        eoa_id=1024, params=SimpleNamespace(),
    )
    seq_lens = [7, 5, 3]
    inputs = []
    for i, n in enumerate(seq_lens):
        inp = ARNodeInputs(input_ids=torch.zeros(n, 10, dtype=torch.long), input_seq_len=n)
        if i != 1:                                    # middle request has no clone
            inp.tensor_inputs["speaker_embedding"] = torch.full((1, 5), float(i))
        inputs.append(inp)

    values, positions = sub._collect_speaker_inputs(inputs, seq_lens)
    assert positions.tolist() == [0, 12]              # 0, then 7 + 5
    assert values.shape == (2, 5)
    assert values[:, 0].tolist() == [0.0, 2.0]        # order follows the batch

    # No cloning requests at all (every decode step) yields nothing to inject.
    plain = [ARNodeInputs(input_ids=torch.zeros(n, 10, dtype=torch.long), input_seq_len=n)
             for n in seq_lens]
    assert sub._collect_speaker_inputs(plain, seq_lens) == (None, None)


# -- speaker encoder output selection ---------------------------------------
def test_speaker_encoder_selects_and_pools_matching_output():
    """The backbone returns a (T, D) hidden-state sequence; the encoder must
    mean-pool it to one vector of exactly speaker_embedding_dim."""
    from mstar.model.zonos2.speaker_encoder import Qwen3SpeakerEncoder

    enc = Qwen3SpeakerEncoder.__new__(Qwen3SpeakerEncoder)   # skip the HF download
    enc.model_id = "test"
    enc.embedding_dim = 4

    seq = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    out = enc._select_embedding(seq)
    assert out.shape == (4,)
    assert torch.allclose(out, seq[0].mean(0))

    # Several candidates: the one with the expected width wins.
    wrong = torch.zeros(1, 3, 7)
    assert enc._select_embedding((wrong, seq)).shape == (4,)

    with pytest.raises(ValueError, match="expects a 4-dim speaker embedding"):
        enc._select_embedding(wrong)
