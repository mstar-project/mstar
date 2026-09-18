"""CPU tests for the Kokoro integration: tiny random weights, no checkpoint.

Covers the pieces that must hold for a padded batch to equal single requests
(masked BiLSTM, per-row STFT, the whole synthesizer), the checkpoint name
mapping and weight-norm folding, voice blends, sentence chunking, the model
contract (graph, state machine, prompt tensors) and the synthesis submodule's
loop iteration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardConductorMetadata, CurrentForwardPassInfo  # noqa: E402
from mstar.graph.base import GraphNode, Loop  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT  # noqa: E402
from mstar.model.kokoro import g2p  # noqa: E402
from mstar.model.kokoro.components import KokoroTTS  # noqa: E402
from mstar.model.kokoro.components.lstm import MaskedBiLSTM  # noqa: E402
from mstar.model.kokoro.components.stft import RealSTFT  # noqa: E402
from mstar.model.kokoro.config import (  # noqa: E402
    AUDIO_CHUNK,
    BOUNDARY_TOKEN_ID,
    CHUNK_LOOP,
    PHONEME_IDS,
    PHONEME_LENS,
    REF_STYLE,
    SPEED,
    SYNTH_NODE,
    SYNTH_WALK,
    KokoroBertConfig,
    KokoroISTFTNetConfig,
    KokoroModelConfig,
)
from mstar.model.kokoro.g2p import Chunk, G2PFrontend  # noqa: E402
from mstar.model.kokoro.kokoro_model import KokoroModel  # noqa: E402
from mstar.model.kokoro.submodules import KokoroSynthSubmodule  # noqa: E402
from mstar.model.kokoro.voices import VoiceRegistry  # noqa: E402
from mstar.model.kokoro.weight_loader import fold_weight_norm, remap_name  # noqa: E402
from mstar.model.registry import HF_MODELS, get_model_class  # noqa: E402
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "kokoro.yaml"
VOCAB = {c: i + 1 for i, c in enumerate("abcdefghij .,!?")}


def tiny_config() -> KokoroModelConfig:
    return KokoroModelConfig(
        n_token=len(VOCAB) + 1,
        hidden_dim=16,
        style_dim=8,
        n_layer=1,
        max_dur=6,
        vocab=VOCAB,
        plbert=KokoroBertConfig(
            hidden_size=16, num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
            num_hidden_layers=2, embedding_size=8,
        ),
        istftnet=KokoroISTFTNetConfig(
            upsample_kernel_sizes=[4, 8], upsample_rates=[2, 4], gen_istft_hop_size=5, gen_istft_n_fft=20,
            resblock_dilation_sizes=[[1, 3], [1, 3]], resblock_kernel_sizes=[3, 5], upsample_initial_channel=16,
        ),
        decoder_hidden=24,
        asr_res_dim=4,
        chunk_target_phonemes=12,
        max_chunks=8,
    )


def tiny_model(seed: int = 0) -> KokoroTTS:
    torch.manual_seed(seed)
    model = KokoroTTS(tiny_config()).eval()
    model.decoder.generator.m_source.deterministic = True
    return model


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


# --------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------


def test_masked_bilstm_matches_packed_bidirectional():
    torch.manual_seed(0)
    ref = torch.nn.LSTM(6, 5, batch_first=True, bidirectional=True)
    ours = MaskedBiLSTM(6, 5)
    state = ref.state_dict()
    ours.fwd.load_state_dict({k: v for k, v in state.items() if not k.endswith("_reverse")})
    ours.bwd.load_state_dict({k[: -len("_reverse")]: v for k, v in state.items() if k.endswith("_reverse")})
    x = torch.randn(3, 7, 6)
    lengths = torch.tensor([7, 4, 1])
    packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    expected, _ = torch.nn.utils.rnn.pad_packed_sequence(ref(packed)[0], batch_first=True, total_length=7)
    out = ours(x, lengths)
    assert torch.allclose(out, expected, atol=1e-6)
    assert out[1, 4:].abs().sum() == 0 and out[2, 1:].abs().sum() == 0


def test_real_stft_inverse_matches_torch_istft_per_row():
    stft = RealSTFT(20, 5)
    torch.manual_seed(0)
    frames = torch.tensor([41, 25])
    magnitude = torch.rand(2, 11, 41) * 0.2
    phase = torch.rand(2, 11, 41) * 6.2 - 3.1
    out = stft.inverse(magnitude, phase, frames)
    window = torch.hann_window(20, periodic=True)
    for row, n in enumerate(frames.tolist()):
        z = magnitude[row, :, :n] * torch.exp(1j * phase[row, :, :n])
        expected = torch.istft(z, 20, hop_length=5, win_length=20, window=window)
        assert torch.allclose(out[row, : expected.numel()], expected, atol=1e-6)


def test_real_stft_transform_matches_torch_stft_per_row():
    stft = RealSTFT(20, 5)
    torch.manual_seed(0)
    lengths = torch.tensor([300, 120])
    x = torch.randn(2, 300)
    x[1, 120:] = 0
    magnitude, phase, frame_lengths = stft.transform(x, lengths)
    window = torch.hann_window(20, periodic=True)
    for row, n in enumerate(lengths.tolist()):
        z = torch.stft(x[row, :n], 20, hop_length=5, win_length=20, window=window, return_complex=True)
        m = int(frame_lengths[row])
        assert m == z.shape[-1]
        # a batched FFT and a single-row FFT differ in the last bit, so compare
        # magnitudes with a tolerance and phases modulo a full turn
        assert torch.allclose(magnitude[row, :, :m], z.abs(), atol=1e-5)
        wrapped = (phase[row, :, :m] - z.angle() + torch.pi) % (2 * torch.pi) - torch.pi
        assert wrapped.abs().max() < 1e-4
        assert magnitude[row, :, m:].abs().sum() == 0


def test_synthesizer_batch_equals_single_requests():
    model = tiny_model()
    ids = [torch.tensor([0, 1, 5, 3, 9, 2, 0]), torch.tensor([0, 4, 4, 0]), torch.tensor([0, 7, 1, 2, 8, 6, 3, 5, 0])]
    styles = torch.randn(3, 16)
    speeds = torch.tensor([1.0, 0.8, 1.4])
    padded = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=BOUNDARY_TOKEN_ID)
    lengths = torch.tensor([len(x) for x in ids])
    with torch.no_grad():
        audio, frame_lengths, pred_dur = model(padded, lengths, styles, speeds)
        for row, x in enumerate(ids):
            single_audio, single_frames, single_dur = model(x[None], lengths[row : row + 1], styles[row : row + 1],
                                                            speeds[row : row + 1])
            assert torch.equal(pred_dur[row, : len(x)], single_dur[0])
            assert pred_dur[row, len(x) :].sum() == 0
            assert int(frame_lengths[row]) == int(single_frames[0]) == int(single_dur.sum())
            n = int(single_frames[0]) * model.config.samples_per_frame
            assert single_audio.shape[-1] == n
            assert rel_l2(audio[row, :n], single_audio[0]) < 5e-3
            assert audio[row, n:].abs().sum() == 0


def test_speed_scales_durations():
    model = tiny_model()
    ids = torch.tensor([[0, 1, 5, 3, 9, 2, 0]])
    lengths = torch.tensor([7])
    style = torch.randn(1, 16)
    with torch.no_grad():
        _, _, slow = model.encode_text(ids, lengths, style, torch.tensor([0.5]))
        _, _, fast = model.encode_text(ids, lengths, style, torch.tensor([2.0]))
    assert slow.sum() > fast.sum()


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------


def test_fold_weight_norm_matches_torch():
    torch.manual_seed(0)
    conv = torch.nn.utils.weight_norm(torch.nn.Conv1d(4, 6, 3), dim=0)
    folded = fold_weight_norm(conv.weight_g.data, conv.weight_v.data)
    assert torch.allclose(folded, conv.weight, atol=1e-7)


@pytest.mark.parametrize(
    ("checkpoint_key", "expected"),
    [
        (
            "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight",
            "bert.layer.attention.query.weight",
        ),
        ("bert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight", "bert.layer.ffn.linear_in.weight"),
        ("bert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias", "bert.layer.ffn.linear_out.bias"),
        ("bert.encoder.embedding_hidden_mapping_in.weight", "bert.mapping_in.weight"),
        ("bert.embeddings.LayerNorm.weight", "bert.embeddings.LayerNorm.weight"),
        ("bert.pooler.weight", None),
        ("predictor.text_encoder.lstms.0.weight_ih_l0_reverse", "predictor.text_encoder.lstms.0.bwd.weight_ih_l0"),
        ("predictor.shared.bias_hh_l0", "predictor.shared.fwd.bias_hh_l0"),
        ("predictor.duration_proj.linear_layer.weight", "predictor.duration_proj.weight"),
        ("text_encoder.cnn.1.1.gamma", "text_encoder.cnn.1.1.gamma"),
        ("decoder.generator.resblocks.2.alpha1.0", "decoder.generator.resblocks.2.alpha1.0"),
    ],
)
def test_remap_name(checkpoint_key, expected):
    assert remap_name(checkpoint_key) == expected


def test_remapped_checkpoint_names_cover_every_parameter():
    """Every parameter of the M* module is reachable from a checkpoint key
    written in the upstream naming, and vice versa (modulo the pooler)."""
    model = tiny_model()
    params = set(dict(model.named_parameters()))
    # Invert the mapping for the names that differ, then check round trip.
    inverse = {
        "bert.mapping_in.": "bert.encoder.embedding_hidden_mapping_in.",
        "bert.layer.ffn.linear_in.": "bert.encoder.albert_layer_groups.0.albert_layers.0.ffn.",
        "bert.layer.ffn.linear_out.": "bert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.",
        "bert.layer.": "bert.encoder.albert_layer_groups.0.albert_layers.0.",
        "predictor.duration_proj.": "predictor.duration_proj.linear_layer.",
    }
    for name in params:
        upstream = name
        for ours, theirs in inverse.items():
            if upstream.startswith(ours):
                upstream = theirs + upstream[len(ours) :]
                break
        upstream = upstream.replace(".fwd.", ".").replace(".bwd.", ".")
        if ".bwd." in name:
            upstream += "_reverse"
        if ".attention.qkv." in name:
            continue  # fused from query/key/value by the stacked rules
        assert remap_name(upstream) == name, (upstream, name)


# --------------------------------------------------------------------------
# voices
# --------------------------------------------------------------------------


@pytest.fixture
def voices_dir(tmp_path):
    torch.manual_seed(0)
    for name in ("af_heart", "af_sky", "am_adam"):
        torch.save(torch.randn(510, 1, 16), tmp_path / f"{name}.pt")
    return tmp_path


def test_voice_registry_blends(voices_dir):
    registry = VoiceRegistry(voices_dir, pack_rows=510, style_dim=8)
    assert registry.names == ["af_heart", "af_sky", "am_adam"]
    heart, sky, adam = (registry.pack(n) for n in registry.names)
    assert registry.parse_blend("af_heart(2)+af_sky") == [("af_heart", 2.0), ("af_sky", 1.0)]
    assert registry.parse_blend("af_heart-am_adam(0.5)") == [("af_heart", 1.0), ("am_adam", -0.5)]
    assert torch.allclose(registry.resolve("af_heart(2)+af_sky"), (2 * heart + sky) / 3, atol=1e-6)
    assert torch.allclose(registry.resolve("af_heart,af_sky"), (heart + sky) / 2, atol=1e-6)
    assert torch.allclose(registry.resolve("af_heart-am_adam(0.5)"), (heart - 0.5 * adam) / 1.5, atol=1e-6)
    assert torch.equal(registry.style("af_sky", 37), sky[36])
    assert torch.equal(registry.style("af_sky", 0), sky[0]) and torch.equal(registry.style("af_sky", 9999), sky[509])
    assert registry.language_of("am_adam+af_sky") == "a"
    with pytest.raises(ValueError, match="Unknown Kokoro voice"):
        registry.resolve("bf_nobody")
    with pytest.raises(ValueError):
        registry.resolve("")


# --------------------------------------------------------------------------
# G2P chunking
# --------------------------------------------------------------------------


def _tok(phonemes, text=None, whitespace=" "):
    return SimpleNamespace(phonemes=phonemes, text=text or phonemes, whitespace=whitespace)


def _sentence(words, end="."):
    tokens = [_tok(w) for w in words]
    tokens[-1].whitespace = ""
    tokens.append(_tok(end, whitespace=" "))
    return tokens


def test_split_sentences_and_pack():
    quoted = _sentence(["ef"], end="?")
    quoted[-1].whitespace = ""  # the question mark closes against the quote
    tokens = _sentence(["ab", "cd"]) + quoted + [_tok("”", whitespace=" ")] + _sentence(["gh", "ij"])
    sentences = g2p.split_sentences(tokens)
    assert [g2p._tokens_phonemes(s) for s in sentences] == ["ab cd.", "ef?”", "gh ij."]
    chunks = g2p.chunk_tokens(tokens, target=12, hard_max=510)
    assert [c.phonemes for c in chunks] == ["ab cd. ef?”", "gh ij."]
    assert chunks[0].text == "ab cd. ef?”"


def test_long_sentence_is_split_at_clauses_then_words():
    tokens = _sentence(["aaaa", "bbbb"], end=",") + _sentence(["cccc", "dddd"], end=";") + _sentence(["eeee", "ffff"])
    chunks = g2p.chunk_tokens(tokens, target=8, hard_max=16)
    assert all(len(c.phonemes) <= 16 for c in chunks)
    assert " ".join(c.phonemes for c in chunks) == "aaaa bbbb, cccc dddd; eeee ffff."
    # a single word longer than the window still gets cut rather than dropped
    huge = [_tok("x" * 40, whitespace="")]
    assert [len(c.phonemes) for c in g2p.chunk_tokens(huge, target=8, hard_max=16)] == [40]


def test_chunk_phoneme_strings_packs_sentences():
    pairs = [("A.", "aa."), ("B.", "bb."), ("C.", "cc."), ("D.", "dd.")]
    chunks = g2p.chunk_phoneme_strings(pairs, target=7, hard_max=510)
    assert [c.phonemes for c in chunks] == ["aa. bb.", "cc. dd."]
    chunks = g2p.chunk_phoneme_strings(pairs, target=11, hard_max=510, first_target=3)
    assert [c.phonemes for c in chunks] == ["aa.", "bb. cc. dd."]


def test_first_chunk_stays_short_for_time_to_first_audio():
    tokens = _sentence(["ab"]) + _sentence(["cd", "ef"]) + _sentence(["gh"]) + _sentence(["ij"])
    chunks = g2p.chunk_tokens(tokens, target=100, hard_max=510, first_target=5)
    assert [c.phonemes for c in chunks] == ["ab.", "cd ef. gh. ij."]
    # without a first-chunk target everything packs under the main target
    assert [c.phonemes for c in g2p.chunk_tokens(tokens, target=100, hard_max=510)] == ["ab. cd ef. gh. ij."]


def test_missing_spacy_model_gives_an_install_hint(monkeypatch):
    import types

    fake_en = types.SimpleNamespace(G2P=lambda **kwargs: (_ for _ in ()).throw(OSError("[E050] Can't find model")))
    monkeypatch.setitem(sys.modules, "misaki", types.SimpleNamespace(en=fake_en))
    monkeypatch.setitem(sys.modules, "misaki.en", fake_en)
    frontend = G2PFrontend(chunk_target=40, max_phonemes=510, espeak_fallback=False)
    with pytest.raises(ImportError, match="spacy download en_core_web_sm"):
        frontend.backend("a")


def test_misaki_english_g2p_if_available():
    pytest.importorskip("misaki")
    pytest.importorskip("spacy")
    frontend = G2PFrontend(chunk_target=40, max_phonemes=510, espeak_fallback=False)
    try:
        chunks = frontend.chunk("Hello world. This is a test! Another sentence here?", "a")
    except (ImportError, OSError) as exc:  # spaCy model not downloaded
        pytest.skip(f"misaki English G2P unavailable: {exc}")
    assert len(chunks) >= 2 and all(0 < len(c.phonemes) <= 510 for c in chunks)
    assert chunks[0].phonemes.startswith("h")


# --------------------------------------------------------------------------
# model contract
# --------------------------------------------------------------------------


class _StubG2P:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []
        self.backends = []

    def backend(self, lang):
        self.backends.append(lang)

    def chunk(self, text, lang):
        self.calls.append((text, lang))
        return self.chunks

    def chunk_phonemes(self, phonemes):
        return [Chunk("", phonemes)]


def make_model(voices_dir, chunks=None) -> KokoroModel:
    model = object.__new__(KokoroModel)
    model.config = tiny_config()
    model.default_lang = None
    model.voices = VoiceRegistry(voices_dir, pack_rows=510, style_dim=8)
    model.g2p = _StubG2P(chunks or [Chunk("Hello.", "ab cd."), Chunk("Bye!", "efg!")])
    model._submodule_cache = {}
    return model


def test_deployment_overrides_reach_the_config(voices_dir, tmp_path, monkeypatch):
    import json

    from mstar.model.kokoro import kokoro_model as km

    (tmp_path / "config.json").write_text(json.dumps({"vocab": VOCAB, "n_token": len(VOCAB) + 1}))
    (tmp_path / "voices").mkdir()
    for name in ("af_heart", "bf_alice"):
        torch.save(torch.randn(510, 1, 256), tmp_path / "voices" / f"{name}.pt")
    monkeypatch.setattr(km, "_resolve_snapshot", lambda repo, cache, patterns: str(tmp_path))
    model = KokoroModel(
        str(tmp_path), lang_code="b", frame_buckets=[64, 128], max_batch_frames=512, default_voice="bf_alice"
    )
    assert model.config.frame_buckets == [64, 128] and model.config.max_batch_frames == 512
    assert model.get_default_voice() == "bf_alice" and model.default_lang == "b"
    with pytest.raises(ValueError, match="Unknown Kokoro option"):
        KokoroModel(str(tmp_path), hidden_dim=3)


def test_registry_and_config(voices_dir):
    assert get_model_class("kokoro") is KokoroModel
    assert HF_MODELS["kokoro"]["model_path_hf"] == "hexgrad/Kokoro-82M"
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert config["model"] == "kokoro"
    model = make_model(voices_dir)
    assert set(config["node_groups"][0]["node_names"]) == set(model.nodes) == {SYNTH_NODE}
    assert config["node_groups"][0]["graph_walks"] == [SYNTH_WALK]


def test_graph_is_one_streaming_loop(voices_dir):
    model = make_model(voices_dir)
    walks = model.get_graph_walk_graphs()
    loop = walks[SYNTH_WALK]
    assert isinstance(loop, Loop) and loop.name == CHUNK_LOOP and loop.max_iters == model.config.max_chunks
    node = loop.section
    assert isinstance(node, GraphNode) and node.name == SYNTH_NODE
    assert node.input_names == {PHONEME_IDS, PHONEME_LENS, REF_STYLE, SPEED}
    assert node.enable_async_scheduling is False
    emits = [e for e in node.outputs if e.next_node == EMIT_TO_CLIENT]
    assert [(e.name, e.output_modality) for e in emits] == [(AUDIO_CHUNK, "audio")]
    assert model.get_node_resources() == []


def test_process_prompt_tensors(voices_dir):
    model = make_model(voices_dir)
    out = model.process_prompt("Hello. Bye!", ["text"], ["audio"], voice="af_sky(2)+af_heart", speed=1.25)
    ids, lens, style, speed = out[PHONEME_IDS][0], out[PHONEME_LENS][0], out[REF_STYLE][0], out[SPEED][0]
    assert ids.shape == (2, 8) and lens.tolist() == [8, 6]  # "ab cd." -> 6 ids + bos/eos; "efg!" -> 4 + 2
    assert ids[0, 0] == BOUNDARY_TOKEN_ID and ids[0, 7] == BOUNDARY_TOKEN_ID and ids[1, 6:].tolist() == [0, 0]
    assert ids[0, 1:7].tolist() == [VOCAB[c] for c in "ab cd."]
    assert style.shape == (2, 16)
    assert torch.equal(style[0], model.voices.style("af_sky(2)+af_heart", 6))
    assert torch.equal(style[1], model.voices.style("af_sky(2)+af_heart", 4))
    assert speed.tolist() == [1.25]
    assert model.g2p.calls == [("Hello. Bye!", "a")]


def test_process_prompt_validation(voices_dir):
    model = make_model(voices_dir)
    with pytest.raises(ValueError, match="speed"):
        model.process_prompt("Hi", ["text"], ["audio"], speed=9.0)
    with pytest.raises(ValueError, match="Unknown Kokoro voice"):
        model.process_prompt("Hi", ["text"], ["audio"], voice="zz_nobody")
    with pytest.raises(ValueError, match="non-empty"):
        model.process_prompt("   ", ["text"], ["audio"])
    with pytest.raises(ValueError, match="audio"):
        model.process_prompt("Hi", ["text"], ["text"])
    model.g2p = _StubG2P([])
    with pytest.raises(ValueError, match="speakable"):
        model.process_prompt("...", ["text"], ["audio"])
    model.g2p = _StubG2P([Chunk("", "a")] * 9)
    with pytest.raises(ValueError, match="chunks"):
        model.process_prompt("many", ["text"], ["audio"])


def test_warmup_builds_the_default_language_backend(voices_dir):
    model = make_model(voices_dir)
    model.warmup_preprocess()
    assert model.g2p.backends == ["a"]  # af_heart -> American English
    model.default_lang = "b"
    model.warmup_preprocess()
    assert model.g2p.backends[-1] == "b"


def test_process_prompt_language_and_phonemes(voices_dir):
    model = make_model(voices_dir)
    model.process_prompt("Hola", ["text"], ["audio"], voice="am_adam", lang_code="es")
    assert model.g2p.calls[-1] == ("Hola", "es")
    out = model.process_prompt(None, ["text"], ["audio"], phonemes="ab cd")
    assert out[PHONEME_IDS][0].shape == (1, 7) and model.g2p.calls[-1][0] == "Hola"


def test_state_machine_and_postprocess(voices_dir):
    model = make_model(voices_dir)
    signals = {name: [f"ptr_{name}"] for name in (PHONEME_IDS, PHONEME_LENS, REF_STYLE, SPEED)}
    first = model.get_initial_forward_pass_args("default", ["text"], ["audio"], signals)
    assert first.full_metadata.graph_walk == SYNTH_WALK and first.request_done is False
    assert [(e.next_node, e.name, e.tensor_info) for e in first.inputs] == [
        (SYNTH_NODE, name, [f"ptr_{name}"]) for name in (PHONEME_IDS, PHONEME_LENS, REF_STYLE, SPEED)
    ]
    metadata = CurrentForwardConductorMetadata(graph_walk=SYNTH_WALK, is_prefill=False)
    assert model.get_partition_forward_pass_args("default", metadata, {}).request_done is True
    assert model.get_initial_forward_pass_args("default", ["text"], ["text"], signals).request_done is True
    pcm = torch.tensor([1, -2, 32767], dtype=torch.int16)
    assert model.postprocess(pcm, "audio") == pcm.numpy().tobytes()
    expected = torch.tensor([16383, -32767], dtype=torch.int16).numpy().tobytes()
    assert model.postprocess(torch.tensor([0.5, -1.0]), "audio") == expected
    assert model.postprocess(torch.zeros(0), "audio") == b""
    assert model.get_output_sample_rate() == model.config.sample_rate == 24000
    assert model.get_voices() == ["af_heart", "af_sky", "am_adam"] and model.get_default_voice() == "af_heart"
    with pytest.raises(ValueError):
        model.postprocess(pcm, "text")


# --------------------------------------------------------------------------
# submodule
# --------------------------------------------------------------------------


def _fwd_info(rid: str, iteration: int) -> CurrentForwardPassInfo:
    info = CurrentForwardPassInfo(request_id=rid, graph_walk=SYNTH_WALK, fwd_index=0, random_seed=0, max_tokens=0)
    info.dynamic_loop_iter_counts[CHUNK_LOOP] = iteration
    return info


def test_submodule_iterates_chunks_and_stops():
    config = tiny_config()
    submodule = KokoroSynthSubmodule(tiny_model(), config)
    torch.manual_seed(0)
    requests = {
        "r1": {PHONEME_IDS: [torch.tensor([[0, 1, 2, 0, 0], [0, 3, 4, 5, 0]])], PHONEME_LENS: [torch.tensor([4, 5])],
               REF_STYLE: [torch.randn(2, 16)], SPEED: [torch.tensor([1.0])]},
        "r2": {PHONEME_IDS: [torch.tensor([[0, 7, 0]])], PHONEME_LENS: [torch.tensor([3])],
               REF_STYLE: [torch.randn(1, 16)], SPEED: [torch.tensor([1.5])]},
    }
    prepared = {
        rid: submodule.prepare_inputs(SYNTH_WALK, _fwd_info(rid, 0), tensors) for rid, tensors in requests.items()
    }
    assert prepared["r1"].input_seq_len == 4 and prepared["r1"].tensor_inputs["input_ids"].tolist() == [0, 1, 2, 0]
    assert prepared["r2"].input_seq_len == 3 and prepared["r2"].tensor_inputs["speed"].item() == 1.5
    # iteration 1 exists only for r1; iteration 2 for nobody
    assert submodule.prepare_inputs(SYNTH_WALK, _fwd_info("r1", 1), requests["r1"]).input_seq_len == 5
    assert submodule.prepare_inputs(SYNTH_WALK, _fwd_info("r2", 1), requests["r2"]) is None
    assert submodule.check_stop("r1", _fwd_info("r1", 0), {}) == set()
    assert submodule.check_stop("r1", _fwd_info("r1", 1), {}) == {CHUNK_LOOP}
    assert submodule.check_stop("r2", _fwd_info("r2", 0), {}) == {CHUNK_LOOP}

    engine_inputs = ModelInputsFromEngine(request_ids=["r1", "r2"], per_request_info={})
    batch = submodule.preprocess(SYNTH_WALK, engine_inputs, [prepared["r1"], prepared["r2"]])
    assert batch["input_ids"].shape == (2, 4) and batch["lengths"].tolist() == [4, 3]
    assert batch["input_ids"][1].tolist() == [0, 7, 0, BOUNDARY_TOKEN_ID]
    with torch.no_grad():
        out = submodule.forward_batched(SYNTH_WALK, engine_inputs, **batch)
        _, _, pred_dur = submodule.model(batch["input_ids"], batch["lengths"], batch["style"], batch["speed"])
    for row, rid in enumerate(["r1", "r2"]):
        pcm = out[rid][AUDIO_CHUNK][0]
        assert pcm.dtype == torch.int16
        assert pcm.numel() == int(pred_dur[row].sum()) * config.samples_per_frame
    batch_r2 = submodule.preprocess(SYNTH_WALK, engine_inputs, [prepared["r2"]])
    with torch.no_grad():
        single = submodule.forward(SYNTH_WALK, engine_inputs, **batch_r2)[AUDIO_CHUNK][0]
    # the same chunk alone and inside a padded batch agree to rounding
    assert single.shape == out["r2"][AUDIO_CHUNK][0].shape
    assert rel_l2(single.float(), out["r2"][AUDIO_CHUNK][0].float()) < 5e-3
    submodule.cleanup_request("r1")
    assert "r1" not in submodule.request_states


def test_openai_adapter_maps_speech_request():
    pytest.importorskip("pydantic")
    from mstar.api_server.openai.adapters import get_adapter
    from mstar.api_server.openai.protocol import SpeechRequest

    adapter = get_adapter("kokoro")
    assert adapter is not None and adapter.supports_speech
    req = SpeechRequest(input="Hello", voice="af_bella+af_sky", speed=1.2, lang_code="b", temperature=0.7)
    args = adapter.speech_to_request(req, Path("/tmp"))
    assert args.text == "Hello" and args.output_modalities == ["audio"] and args.input_modalities == ["text"]
    assert args.model_kwargs["voice"] == "af_bella+af_sky" and args.model_kwargs["speed"] == 1.2
    assert args.model_kwargs["lang_code"] == "b" and "temperature" not in args.model_kwargs


# --------------------------------------------------------------------------
# CUDA-graph buckets (the capture functions run eagerly on CPU here)
# --------------------------------------------------------------------------


def test_bucket_selection_and_grouping():
    from mstar.model.kokoro.submodules import group_by_bucket, pick_bucket

    buckets = [48, 64, 96, 128]
    assert pick_bucket(1, buckets) == 48 and pick_bucket(64, buckets) == 64 and pick_bucket(65, buckets) == 96
    assert pick_bucket(129, buckets) is None
    groups = group_by_bucket([100, 10, 64, 500, 90], buckets)
    assert groups == [(48, [1]), (64, [2]), (96, [4]), (128, [0]), (None, [3])]


def test_piecewise_regions_match_the_eager_halves():
    """Each captured half, run as the runner would run it (padded to its bucket,
    padding rows zeroed), reproduces the eager computation on the real rows."""
    from mstar.engine.cuda_graph_config import PiecewiseCallInputs, PiecewiseCaptureShape
    from mstar.model.kokoro.submodules import frame_region, text_region
    from mstar.model.submodule_base import ModelInputsFromEngine

    config = tiny_config()
    config.text_buckets = [8, 16]
    config.frame_buckets = [16, 32, 64]
    config.capture_batch_sizes = [1, 2, 4]
    config.max_batch_frames = 128
    submodule = KokoroSynthSubmodule(tiny_model(), config)
    regions = submodule.get_piecewise_cuda_graph_configs(torch.device("cpu"), torch.float32)
    assert set(regions) == {text_region(8), text_region(16)} | {frame_region(f) for f in (16, 32, 64)}
    assert regions[text_region(16)].seq_len == 16 and regions[text_region(16)].capture_batch_sizes == [1, 2, 4]
    # bs * frames <= max_batch_frames caps the decoder's batch per bucket
    assert regions[frame_region(64)].capture_batch_sizes == [1, 2]
    assert regions[frame_region(16)].capture_batch_sizes == [1, 2, 4]

    torch.manual_seed(1)
    ids = [torch.tensor([0, 1, 5, 3, 9, 2, 0]), torch.tensor([0, 4, 4, 0])]
    style = torch.randn(2, 16)
    speed = torch.tensor([1.0, 0.8])
    padded = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=BOUNDARY_TOKEN_ID)
    lengths = torch.tensor([7, 4])
    engine_inputs = ModelInputsFromEngine(request_ids=["a", "b", "pad", "pad"], per_request_info={})
    with torch.no_grad():
        d_ref, t_ref, dur_ref = submodule.model.encode_text(padded, lengths, style, speed)
        # text region: bucket T=8, batch padded to 4 rows of zeros
        shape = PiecewiseCaptureShape(bs=4, seq_lens=[8] * 4, total_tokens=32)
        static = regions[text_region(8)].make_static_inputs(shape)
        static["input_ids"][:2, :7] = padded
        static["lengths"][:2] = lengths
        static["style"][:2] = style
        static["speed"][:2] = speed
        out = regions[text_region(8)].capture_fn(PiecewiseCallInputs(static_inputs=static, engine_inputs=engine_inputs))
        assert out["d"].shape == (4, 8, 24) and out["t_en"].shape == (4, 16, 8) and out["pred_dur"].shape == (4, 8)
        assert torch.equal(out["pred_dur"][:2, :7], dur_ref)
        assert rel_l2(out["d"][:2, :7], d_ref) < 1e-4 and rel_l2(out["t_en"][:2, :, :7], t_ref) < 1e-4
        assert torch.isfinite(out["d"]).all()  # padding rows stay finite

        # frame region: bucket F=32, one real row
        num_frames = int(dur_ref[1].sum())
        assert num_frames <= 32
        en, asr, frame_lengths = submodule.model.align(d_ref[1:2], t_ref[1:2], dur_ref[1:2], 32)
        audio_ref, _ = submodule.model.synthesize_frames(d_ref[1:2], t_ref[1:2], dur_ref[1:2], style[1:2], num_frames)
        shape = PiecewiseCaptureShape(bs=2, seq_lens=[32, 32], total_tokens=64)
        static = regions[frame_region(32)].make_static_inputs(shape)
        static["en"][:1] = en
        static["asr"][:1] = asr
        static["frame_lengths"][:1] = frame_lengths
        static["style"][:1] = style[1:2]
        call = PiecewiseCallInputs(static_inputs=static, engine_inputs=engine_inputs)
        out = regions[frame_region(32)].capture_fn(call)
        n = num_frames * config.samples_per_frame
        assert out["audio"].shape == (2, 32 * config.samples_per_frame)
        assert rel_l2(out["audio"][0, :n], audio_ref[0]) < 5e-3 and out["audio"][0, n:].abs().sum() == 0


def test_synthesize_uses_runners_when_they_fit():
    """A fake runner stands in for a captured graph: the submodule routes
    through it per bucket and slices each row to its own length."""
    from mstar.model.kokoro.submodules import text_region

    config = tiny_config()
    config.text_buckets = [8]
    config.frame_buckets = [16, 64]
    submodule = KokoroSynthSubmodule(tiny_model(), config)
    regions = submodule.get_piecewise_cuda_graph_configs(torch.device("cpu"), torch.float32)

    class FakeRunner:
        def __init__(self, config):
            self.config = config
            self.calls = []

        def can_run(self, bs):
            return bs <= max(self.config.capture_batch_sizes)

        def run(self, static_inputs, real_bs):
            from mstar.engine.cuda_graph_config import PiecewiseCallInputs, PiecewiseCaptureShape

            bs = next(b for b in self.config.capture_batch_sizes if b >= real_bs)
            seq_len = self.config.seq_len
            shape = PiecewiseCaptureShape(bs=bs, seq_lens=[seq_len] * bs, total_tokens=bs * seq_len)
            static = self.config.make_static_inputs(shape)
            for k, v in static_inputs.items():
                static[k][:real_bs] = v
            self.calls.append((self.config.seq_len, real_bs))
            out = self.config.capture_fn(PiecewiseCallInputs(static_inputs=static, engine_inputs=None))
            return {k: v[:real_bs] for k, v in out.items()}

    runners = {name: FakeRunner(cfg) for name, cfg in regions.items()}
    torch.manual_seed(2)
    ids = [torch.tensor([0, 1, 5, 3, 9, 2, 0]), torch.tensor([0, 4, 4, 0]), torch.tensor([0, 7, 1, 8, 0])]
    padded = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=BOUNDARY_TOKEN_ID)
    lengths = torch.tensor([7, 4, 5])
    style = torch.randn(3, 16)
    speed = torch.tensor([1.0, 1.0, 1.0])
    with torch.no_grad():
        eager = submodule._synthesize(padded, lengths, style, speed, {})
        captured = submodule._synthesize(padded, lengths, style, speed, runners)
    assert runners[text_region(8)].calls == [(8, 3)]
    frame_calls = [c for name, r in runners.items() if name.startswith("frames") for c in r.calls]
    assert sum(bs for _, bs in frame_calls) == 3  # every row went through exactly one frame bucket
    for a, b in zip(eager, captured, strict=True):
        assert a.shape == b.shape and rel_l2(a.float(), b.float()) < 5e-3
