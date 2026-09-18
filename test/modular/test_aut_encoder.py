"""The shared AuT audio encoder against HF's ``Qwen3OmniMoeAudioEncoder``.

Tiny random weights on CPU: the token-count arithmetic, the windowed layout,
weight loading through the Whisper stacked rules, and the forward on one and
on several packed requests (HF encodes one audio at a time; the packed
output must match each of them).
"""

import pytest
import torch

pytest.importorskip("transformers")

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeAudioEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeAudioEncoderLayer,
)

from mstar.model.components.aut_encoder import AuTEncoder, AuTEncoderConfig  # noqa: E402

TINY = dict(
    d_model=64, encoder_layers=2, encoder_attention_heads=4, encoder_ffn_dim=128,
    num_mel_bins=16, n_window=50, n_window_infer=200, conv_chunksize=3,
    downsample_hidden_size=8, output_dim=48, max_source_positions=1500,
)


@pytest.fixture(autouse=True)
def _windowed_hf_attention(monkeypatch):
    """HF's SDPA/eager path never applies the window mask its encoder builds
    (``_prepare_attention_mask`` is only relevant to FA2, which reads
    ``cu_seqlens`` directly), so on CPU the reference attends across windows.
    The released checkpoints and every GPU engine (vLLM, FA2) are windowed;
    hand the layers the reference's own block-diagonal mask so the oracle is."""
    original = Qwen3OmniMoeAudioEncoderLayer.forward

    def windowed(self, hidden_states, cu_seqlens, attention_mask=None, **kwargs):
        if attention_mask is None:
            n = hidden_states.shape[0]
            attention_mask = torch.full((1, 1, n, n), torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype)
            bounds = cu_seqlens.tolist()
            for a, b in zip(bounds[:-1], bounds[1:], strict=True):
                attention_mask[..., a:b, a:b] = 0
        return original(self, hidden_states, cu_seqlens, attention_mask=attention_mask, **kwargs)

    monkeypatch.setattr(Qwen3OmniMoeAudioEncoderLayer, "forward", windowed)


def _pair(**overrides):
    values = dict(TINY, **overrides)
    hf_cfg = Qwen3OmniMoeAudioEncoderConfig(**values)
    torch.manual_seed(0)
    hf = Qwen3OmniMoeAudioEncoder._from_config(hf_cfg, attn_implementation="sdpa").eval()
    ours = AuTEncoder(AuTEncoderConfig.from_hf(values)).eval()
    ours.load_weights(list(hf.state_dict().items()))
    return hf, ours


def _hf_encode(hf, features: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return hf(features, feature_lens=torch.tensor([features.shape[-1]])).last_hidden_state


def test_token_arithmetic_matches_hf_formula():
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths

    cfg = AuTEncoderConfig()
    for n in [1, 37, 99, 100, 101, 230, 1000, 3000, 30017]:
        assert cfg.tokens_for_frames(n) == int(_get_feat_extract_output_lengths(torch.tensor(n)))
    assert cfg.chunk_frames == 100 and cfg.tokens_per_chunk == 13 and cfg.window_tokens == 104
    assert cfg.window_lengths(30) == [30]
    assert cfg.window_lengths(104 * 3 + 5) == [104, 104, 104, 5]
    assert cfg.conv_out_features == 480 * 16
    small = AuTEncoderConfig(**TINY)
    assert small.window_tokens == 26 and small.conv_out_features == 8 * 2


def test_from_hf_reads_the_checkpoint_audio_config():
    hf = {"d_model": 896, "encoder_layers": 18, "encoder_attention_heads": 14, "encoder_ffn_dim": 3584,
          "output_dim": 1024, "n_window": 50, "n_window_infer": 800, "num_mel_bins": 128,
          "conv_chunksize": 500, "downsample_hidden_size": 480, "model_type": "qwen3_asr_audio_encoder",
          "max_source_positions": 1500}
    cfg = AuTEncoderConfig.from_hf(hf)
    assert cfg.d_model == 896 and cfg.encoder_layers == 18 and cfg.output_dim == 1024
    assert cfg.head_dim == 64


def test_loads_completely_and_rejects_partial_checkpoints():
    hf, ours = _pair()
    assert set(dict(ours.named_parameters())) == set(ours.load_weights(list(hf.state_dict().items())))
    partial = [(k, v) for k, v in hf.state_dict().items() if not k.startswith("proj2")]
    with pytest.raises(RuntimeError, match="unloaded"):
        AuTEncoder(AuTEncoderConfig.from_hf(TINY)).load_weights(partial)


@pytest.mark.parametrize("num_frames", [230, 100, 350, 61])
def test_single_request_matches_hf(num_frames):
    hf, ours = _pair()
    torch.manual_seed(1)
    feats = torch.randn(TINY["num_mel_bins"], num_frames)
    expected = _hf_encode(hf, feats)
    with torch.no_grad():
        actual, layout = ours(feats.unsqueeze(0), [num_frames])
    assert layout.tokens_per_request == [expected.shape[0]]
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-4), (actual - expected).abs().max()


def test_packed_requests_match_hf_one_by_one():
    hf, ours = _pair()
    torch.manual_seed(2)
    lens = [230, 150, 305]
    feats = [torch.randn(TINY["num_mel_bins"], n) for n in lens]
    padded = torch.zeros(len(lens), TINY["num_mel_bins"], max(lens))
    for i, f in enumerate(feats):
        padded[i, :, : f.shape[-1]] = f
    with torch.no_grad():
        actual, layout = ours(padded, lens)
    cfg = ours.config
    assert layout.tokens_per_request == [cfg.tokens_for_frames(n) for n in lens]
    # every request's windows, back to back: 26-token windows plus remainders
    expected_windows = [w for n in layout.tokens_per_request for w in cfg.window_lengths(n)]
    assert layout.window_lengths == expected_windows
    assert layout.total_tokens == sum(layout.tokens_per_request) == actual.shape[0]
    offset = 0
    for f, n_tok in zip(feats, layout.tokens_per_request, strict=True):
        expected = _hf_encode(hf, f)
        assert torch.allclose(actual[offset:offset + n_tok], expected, atol=1e-4)
        offset += n_tok


def test_encode_runs_through_a_bound_ragged_resource():
    """With a resource bound, attention goes through ``run(q, k, v)`` and the
    layout must have been planned by the caller; the unbound path is SDPA."""
    hf, ours = _pair()
    calls = []

    class _Recorder:
        def run(self, q, k, v):
            calls.append(q.shape)
            # dense attention over the whole packed batch stands in for the
            # planned kernel; only the call is under test
            o = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
            )
            return o.transpose(0, 1)

    rec = _Recorder()
    for module in ours.modules():
        if hasattr(module, "bind_resources"):
            module.bind_resources({"aut_attn": rec})
    # attn_key is None on this instance, so nothing binds
    assert all(layer.self_attn.ragged is None for layer in ours.layers)
    keyed = AuTEncoder(AuTEncoderConfig.from_hf(TINY), attn_key="aut_attn")
    keyed.load_weights(list(hf.state_dict().items()))
    for module in keyed.modules():
        if hasattr(module, "bind_resources"):
            module.bind_resources({"aut_attn": rec})
    assert all(layer.self_attn.ragged is rec for layer in keyed.layers)
    feats = torch.randn(1, TINY["num_mel_bins"], 120)
    with torch.no_grad():
        out, layout = keyed(feats, [120])
    assert len(calls) == TINY["encoder_layers"] and calls[0] == (layout.total_tokens, 4, 16)
    assert out.shape == (layout.total_tokens, TINY["output_dim"])
