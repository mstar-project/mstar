"""Word-level timestamps: the teacher-forced cross-attention pass against HF's
decoder attentions, and the DTW / word grouping on synthetic alignments."""

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from mstar.model.loader import WHISPER_STACKED_PARAMS, load_hf_weights  # noqa: E402
from mstar.model.whisper.components import alignment  # noqa: E402
from mstar.model.whisper.components.decoder import WhisperDecoderModel  # noqa: E402
from mstar.model.whisper.config import WhisperModelConfig  # noqa: E402
from mstar.model.whisper.whisper_model import WhisperModel  # noqa: E402


def _tiny() -> WhisperModelConfig:
    return WhisperModelConfig(
        d_model=64, decoder_layers=2, decoder_attention_heads=4, decoder_ffn_dim=128,
        encoder_layers=1, encoder_attention_heads=4, encoder_ffn_dim=128, num_mel_bins=16,
        vocab_size=51866, max_target_positions=448, max_source_positions=50, chunk_length=1,
        alignment_heads=[[0, 1], [1, 2], [1, 3]],
    )


def _hf_and_ours(cfg: WhisperModelConfig):
    import transformers
    from transformers import WhisperForConditionalGeneration

    hf_cfg = transformers.WhisperConfig(
        d_model=cfg.d_model, encoder_layers=cfg.encoder_layers,
        encoder_attention_heads=cfg.encoder_attention_heads, encoder_ffn_dim=cfg.encoder_ffn_dim,
        decoder_layers=cfg.decoder_layers, decoder_attention_heads=cfg.decoder_attention_heads,
        decoder_ffn_dim=cfg.decoder_ffn_dim, num_mel_bins=cfg.num_mel_bins,
        max_source_positions=cfg.max_source_positions, vocab_size=cfg.vocab_size,
        max_target_positions=cfg.max_target_positions,
    )
    torch.manual_seed(0)
    hf = WhisperForConditionalGeneration._from_config(hf_cfg, attn_implementation="eager").eval()
    ours = WhisperDecoderModel(cfg)
    load_hf_weights(
        ours, list(hf.model.decoder.state_dict().items()),
        stacked_params=WHISPER_STACKED_PARAMS, name_remapper=WhisperModel._decoder_remap,
    )
    ours.zero_missing_biases()
    return hf, ours.eval()


def test_cross_attention_weights_match_hf_decoder_attentions():
    cfg = _tiny()
    hf, ours = _hf_and_ours(cfg)
    torch.manual_seed(1)
    enc = torch.randn(cfg.max_source_positions, cfg.d_model)
    tokens = torch.tensor([50258, 50259, 50360, 50364, 1234, 567, 89, 50257])
    with torch.no_grad():
        out = hf.model.decoder(
            input_ids=tokens[None], encoder_hidden_states=enc[None], output_attentions=True,
        )
        heads = [(int(layer), int(head)) for layer, head in cfg.alignment_heads]
        weights = ours.cross_attention_weights(tokens, enc, heads)
    assert weights.shape == (3, len(tokens), cfg.max_source_positions)
    for i, (layer, head) in enumerate(heads):
        expected = out.cross_attentions[layer][0, head]
        assert torch.allclose(weights[i], expected, atol=1e-4), (layer, head, (weights[i] - expected).abs().max())
    # every row is a distribution over the encoder positions
    assert torch.allclose(weights.sum(-1), torch.ones(3, len(tokens)), atol=1e-4)


def test_dtw_follows_a_diagonal_band_and_is_monotonic():
    rows, frames = 5, 40
    matrix = torch.zeros(rows, frames)
    for i in range(rows):
        matrix[i, 8 * i: 8 * i + 8] = 1.0
    r, f = alignment.dtw(-matrix.numpy())
    assert r[0] == 0 and f[0] == 0 and r[-1] == rows - 1 and f[-1] == frames - 1
    assert np.all(np.diff(r) >= 0) and np.all(np.diff(f) >= 0)
    assert np.all((np.diff(r) + np.diff(f)) >= 1)  # every step moves
    assert alignment.token_start_frames(matrix).tolist() == [0, 8, 16, 24, 32]



def test_warm_up_compiles_or_falls_back_and_the_dtw_still_agrees(monkeypatch):
    """warm_up compiles the DTW at load; a compile failure falls back to the
    plain-Python loop rather than failing the first word-timestamp request."""
    cost = np.random.default_rng(0).random((6, 20))
    expected = alignment._dtw_trace(np.ascontiguousarray(cost))
    alignment.warm_up()
    assert np.array_equal(alignment._dtw_trace_fast(np.ascontiguousarray(cost)), expected)

    def broken(_cost):
        raise RuntimeError("cannot compile")

    monkeypatch.setattr(alignment, "_dtw_trace_fast", broken)
    alignment.warm_up()
    assert alignment._dtw_trace_fast is alignment._dtw_trace

def test_median_filter_reflects_edges():
    x = torch.arange(10.0).view(1, 1, 10)
    assert alignment.median_filter(x, 3).flatten().tolist() == [1, 1, 2, 3, 4, 5, 6, 7, 8, 8]
    assert torch.equal(alignment.median_filter(x, 1), x)


def test_words_split_on_spaces_and_punctuation_merges():
    vocab = {1: " He", 2: " hop", 3: "ed", 4: ",", 5: " \"", 6: " there", 7: ".", 8: "\xe2", 9: "\x80\x9c"}

    def decode(ids):
        text = "".join(vocab[i] for i in ids)
        return text.encode("latin-1", "ignore").decode("utf-8", "replace") if any(i in (8, 9) for i in ids) else text

    assert alignment.split_words(decode, [1, 2, 3, 4, 6, 7]) == [
        (" He", [1]), (" hoped", [2, 3]), (",", [4]), (" there", [6]), (".", [7]),
    ]
    # a token without a leading space continues the word before it
    assert alignment.split_words(decode, [1, 3]) == [(" Heed", [1, 3])]
    # a multi-byte character split across tokens stays one piece
    assert alignment.split_words(decode, [8, 9])[0][1] == [8, 9]
    # a replacement character the transcript itself holds ends its piece, as in openai-whisper
    assert alignment.split_words(decode, [8, 1, 6]) == [("�", [8]), (" He", [1]), (" there", [6])]

    starts = [0.0, 0.2, 0.5, 0.7, 0.9, 1.2, 1.5]  # row starts: 6 tokens + end-of-text
    words = alignment.split_words(decode, [1, 2, 3, 4, 6, 7])
    bounds = np.concatenate([[0], np.cumsum([len(t) for _, t in words])])
    spans = zip(words, bounds[:-1], bounds[1:], strict=True)
    timed = [{"word": w, "start": starts[a], "end": starts[b]} for (w, _), a, b in spans]
    merged = alignment.merge_punctuation(timed)
    assert [w["word"] for w in merged] == [" He", " hoped,", " there."]
    # the word keeps its own timing; the comma's is dropped
    assert merged[1] == {"word": " hoped,", "start": 0.2, "end": 0.7}
    # opening marks first, right to left, as openai-whisper: the quote joins
    # "world" rather than the comma's word swallowing it
    quoted = [{"word": w, "start": float(k), "end": k + 1.0} for k, w in enumerate([" Hello", ",", ' "', "world"])]
    assert [w["word"] for w in alignment.merge_punctuation(quoted)] == [" Hello,", ' "world']


def test_languages_without_spaces_split_per_character():
    # openai-whisper's split_tokens_on_unicode for zh/ja/th/lo/my/yue: one word per
    # whole character, with a character split across two tokens kept together
    text = " 今天天气很好，我们去公园散步吧。"
    pieces = []
    for ch in text:
        b = ch.encode()
        pieces += [b[:2], b[2:]] if ch == "公" else [b]
    vocab = dict(enumerate(pieces))

    def decode(ids):
        return b"".join(vocab[i] for i in ids).decode("utf-8", "replace")

    words = alignment.split_words(decode, list(vocab), "zh")
    assert [w for w, _ in words] == list(text)
    assert dict(words)["公"] == [text.index("公"), text.index("公") + 1]
    assert len(alignment.split_words(decode, list(vocab), "en")) == 1

    # a transcript cut inside a character (at the position table) keeps every
    # token and does not read past the decoded text
    cut = {0: "今".encode(), 1: b"\xe5", 2: b"\xa4"}
    words = alignment.split_words(lambda ids: b"".join(cut[i] for i in ids).decode("utf-8", "replace"), [0, 1, 2], "zh")
    assert sum((toks for _, toks in words), []) == [0, 1, 2]


def test_word_timings_end_to_end_on_a_synthetic_alignment():
    vocab = {1: " He", 2: " hop", 3: "ed", 4: " there"}
    rows, frames = 5, 100  # 4 text tokens + end-of-text, 2 s of encoder positions
    matrix = torch.zeros(rows, frames)
    for i in range(rows):
        matrix[i, 20 * i: 20 * i + 20] = 1.0
    weights = matrix.unsqueeze(0).repeat(2, 1, 1)
    words = alignment.word_timings(weights, [1, 2, 3, 4], lambda ids: "".join(vocab[i] for i in ids), num_frames=200)
    assert words == [
        {"word": "He", "start": 0.0, "end": 0.4},
        {"word": "hoped", "start": 0.4, "end": 1.2},
        {"word": "there", "start": 1.2, "end": 1.6},
    ]
    assert alignment.word_timings(weights, [], lambda ids: "", num_frames=200) == []
