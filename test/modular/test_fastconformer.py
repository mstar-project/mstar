"""The shared FastConformer encoder (Canary, Parakeet, Nemotron ASR) on tiny
random weights: parity with HF's encoders (when the installed transformers has
them), batch invariance offline and causal, the causal variant's causality,
the BatchNorm fold and NeMo's weight names.
"""

import pytest
import torch

from mstar.model.components.fastconformer import (
    FastConformerConfig,
    FastConformerEncoder,
    chunked_attention_mask,
    nemo_to_hf_name,
)

TINY = dict(
    hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
    conv_kernel_size=9, num_mel_bins=16, subsampling_factor=8, subsampling_conv_channels=8,
    subsampling_conv_kernel_size=3, subsampling_conv_stride=2, attention_bias=True,
    convolution_bias=True, scale_input=False,
)


def _hf_encoder(**overrides):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "ParakeetEncoder"):
        pytest.skip("transformers has no ParakeetEncoder")
    config = transformers.ParakeetEncoderConfig(**{**TINY, **overrides})
    torch.manual_seed(0)
    hf = transformers.ParakeetEncoder(config).eval()
    if not overrides.get("convolution_bias", True) and hf.layers[0].conv.pointwise_conv1.bias is not None:
        pytest.skip("this transformers ignores convolution_bias")
    with torch.no_grad():
        # non-trivial BatchNorm statistics, so the fold is exercised
        for layer in hf.layers:
            layer.conv.norm.running_mean.normal_(0, 0.5)
            layer.conv.norm.running_var.uniform_(0.5, 2.0)
            layer.conv.norm.weight.normal_(1, 0.2)
            layer.conv.norm.bias.normal_(0, 0.2)
    return hf, config


def _ours_from(hf, config) -> FastConformerEncoder:
    ours = FastConformerEncoder(FastConformerConfig.from_hf(config.to_dict())).eval()
    loaded = ours.load_hf_weights(hf.state_dict().items())
    assert loaded == {name for name, _ in ours.named_parameters()}
    return ours


@pytest.mark.parametrize("bias", [True, False])
def test_matches_hf_parakeet_encoder(bias):
    hf, config = _hf_encoder(attention_bias=bias, convolution_bias=bias)
    ours = _ours_from(hf, config)
    torch.manual_seed(1)
    feats = torch.randn(1, 203, 16)
    with torch.no_grad():
        expected = hf(feats, attention_mask=torch.ones(1, 203, dtype=torch.long)).last_hidden_state
        out, lengths = ours(feats, torch.tensor([203]))
    assert lengths.tolist() == [26] == [expected.shape[1]]
    assert torch.allclose(out, expected, atol=1e-5), (out - expected).abs().max()


def test_padded_batch_matches_hf_and_single_requests():
    hf, config = _hf_encoder()
    ours = _ours_from(hf, config)
    torch.manual_seed(2)
    lens = [203, 97, 150]
    feats = torch.zeros(3, 203, 16)
    for i, n in enumerate(lens):
        feats[i, :n] = torch.randn(n, 16)
    mask = (torch.arange(203)[None] < torch.tensor(lens)[:, None]).long()
    with torch.no_grad():
        expected = hf(feats, attention_mask=mask).last_hidden_state
        out, out_lens = ours(feats, torch.tensor(lens))
        for i, n in enumerate(lens):
            single, _ = ours(feats[i:i + 1, :n], torch.tensor([n]))
            m = int(out_lens[i])
            assert single.shape[1] == m
            assert torch.allclose(out[i, :m], single[0], atol=1e-5)
            assert torch.allclose(out[i, :m], expected[i, :m], atol=1e-5)


def test_output_lengths_match_hf_formula():
    hf, config = _hf_encoder()
    ours = FastConformerConfig.from_hf(config.to_dict())
    lengths = torch.arange(2, 400)
    assert torch.equal(ours.output_lengths(lengths), hf._get_subsampling_output_length(lengths).long())
    assert ours.output_lengths(203) == 26


def _causal_config(lookahead: int) -> FastConformerConfig:
    return FastConformerConfig(
        hidden_size=32, num_layers=2, num_heads=4, intermediate_size=64, conv_kernel_size=9,
        num_mel_bins=16, subsampling_channels=8, attention_bias=False, convolution_bias=False,
        causal=True, left_context=8, lookahead=lookahead,
    )


@pytest.mark.parametrize("lookahead", [0, 3])
def test_causal_encoder_ignores_audio_past_the_lookahead(lookahead):
    """A frame's output depends only on its chunk and what came before."""
    torch.manual_seed(3)
    enc = FastConformerEncoder(_causal_config(lookahead)).eval()
    for p in enc.parameters():
        torch.nn.init.normal_(p, std=0.2)
    feats = torch.randn(1, 160, 16)
    with torch.no_grad():
        base, _ = enc(feats, torch.tensor([160]))
        changed = feats.clone()
        changed[:, 96:] = torch.randn(1, 64, 16)  # mel frame 96 = encoder frame 12
        other, _ = enc(changed, torch.tensor([160]))
    chunk = lookahead + 1
    last_untouched = (12 // chunk) * chunk  # frames of earlier chunks only
    assert torch.allclose(base[:, :last_untouched], other[:, :last_untouched], atol=1e-6)
    assert not torch.allclose(base[:, 12:], other[:, 12:])


@pytest.mark.parametrize("lookahead", [0, 13])
def test_causal_padded_batch_matches_single_requests(lookahead):
    """Nemotron serves padded, bucketed batches through the chunked path."""
    torch.manual_seed(7)
    enc = FastConformerEncoder(_causal_config(lookahead)).eval()
    for p in enc.parameters():
        torch.nn.init.normal_(p, std=0.2)
    # 26, 15 and 10 encoder frames: across and inside 14-frame chunks at lookahead 13
    lens = [200, 112, 70]
    feats = torch.zeros(3, 200, 16)
    for i, n in enumerate(lens):
        feats[i, :n] = torch.randn(n, 16)
    with torch.no_grad():
        out, out_lens = enc(feats, torch.tensor(lens))
        for i, n in enumerate(lens):
            single, _ = enc(feats[i:i + 1, :n], torch.tensor([n]))
            m = int(out_lens[i])
            assert single.shape[1] == m
            assert torch.allclose(out[i, :m], single[0], atol=1e-5), (out[i, :m] - single[0]).abs().max()


def test_causal_loader_takes_nemo_flat_subsampling_names():
    """NeMo saves a streaming encoder's subsampling flat (``pre_encode.conv.N``),
    like an offline one; only HF's streaming names are remapped."""
    torch.manual_seed(6)
    source = FastConformerEncoder(_causal_config(3))
    flat = {name: torch.randn_like(t) for name, t in source.state_dict().items() if name.startswith("subsampling.")}
    target = FastConformerEncoder(_causal_config(3))
    assert target.load_hf_weights(flat.items()) == set(flat)
    assert all(torch.equal(target.state_dict()[name], t) for name, t in flat.items())


def test_chunked_mask_shape():
    mask = chunked_attention_mask(8, left_context=2, lookahead=1, device="cpu")
    # chunks of 2 frames, one chunk of history
    assert mask[5].tolist() == [False, False, True, True, True, True, False, False]
    assert mask[0].tolist() == [True, True] + [False] * 6


@pytest.mark.parametrize("lookahead", [0, 3, 13])
def test_causal_matches_hf_streaming_encoder_offline(lookahead):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "NemotronAsrStreamingEncoder"):
        pytest.skip("transformers has no NemotronAsrStreamingEncoder")
    config = transformers.NemotronAsrStreamingEncoderConfig(
        **{**TINY, "attention_bias": False, "convolution_bias": False},
        sliding_window=9, default_num_lookahead_tokens=3,
    )
    torch.manual_seed(4)
    hf = transformers.NemotronAsrStreamingEncoder(config).eval()
    ours = FastConformerEncoder(FastConformerConfig.from_hf(config.to_dict(), lookahead=lookahead)).eval()
    loaded = ours.load_hf_weights(hf.state_dict().items())
    assert loaded == {name for name, _ in ours.named_parameters()}
    feats = torch.randn(1, 200, 16)
    with torch.no_grad():
        expected = hf(feats, attention_mask=torch.ones(1, 200, dtype=torch.long), num_lookahead_tokens=lookahead)
        out, lengths = ours(feats, torch.tensor([200]))
    assert lengths.tolist() == [expected.last_hidden_state.shape[1]] == [out.shape[1]]
    assert torch.allclose(out, expected.last_hidden_state, atol=1e-5)


# HF's names as NeMo's ConformerEncoder saves them (Canary-Qwen's checkpoint),
# written out apart from the loader's table so a slip in either shows up
_HF_TO_NEMO = (
    ("subsampling.layers.", "pre_encode.conv."), ("subsampling.linear.", "pre_encode.out."),
    (".self_attn.q_proj.", ".self_attn.linear_q."), (".self_attn.k_proj.", ".self_attn.linear_k."),
    (".self_attn.v_proj.", ".self_attn.linear_v."), (".self_attn.o_proj.", ".self_attn.linear_out."),
    (".self_attn.relative_k_proj.", ".self_attn.linear_pos."), (".self_attn.bias_u", ".self_attn.pos_bias_u"),
    (".self_attn.bias_v", ".self_attn.pos_bias_v"), (".conv.norm.", ".conv.batch_norm."),
)


def test_nemo_named_checkpoint_matches_hf():
    """HF's weights under NeMo's names load the same encoder: a swapped name
    (q/k, a BatchNorm statistic) would change the output."""
    hf, config = _hf_encoder()
    nemo = {}
    for name, tensor in hf.state_dict().items():
        for hf_part, nemo_part in _HF_TO_NEMO:
            name = name.replace(hf_part, nemo_part)
        nemo[name] = tensor
    ours = FastConformerEncoder(FastConformerConfig.from_hf(config.to_dict())).eval()
    loaded = ours.load_hf_weights((nemo_to_hf_name(name), tensor) for name, tensor in nemo.items())
    assert loaded == {name for name, _ in ours.named_parameters()}
    torch.manual_seed(5)
    feats = torch.randn(1, 203, 16)
    with torch.no_grad():
        expected = hf(feats, attention_mask=torch.ones(1, 203, dtype=torch.long)).last_hidden_state
        out, _ = ours(feats, torch.tensor([203]))
    assert torch.allclose(out, expected, atol=1e-5), (out - expected).abs().max()


class _PaddedReplay:
    """A CPU stand-in for one bucket's captures: pads the batch to the next
    captured size, with zeroed rows, and runs the captured function on it."""

    def __init__(self, node, bucket):
        from mstar.model.components.fastconformer_submodule import _batch_sizes

        self.node, self.sizes = node, _batch_sizes(bucket)

    def can_run(self, bs):
        return bs <= max(self.sizes)

    def run(self, static_inputs, real_bs):
        from types import SimpleNamespace

        bs = next(size for size in self.sizes if size >= real_bs)
        padded = {
            name: torch.cat([value, value.new_zeros(bs - real_bs, *value.shape[1:])])
            for name, value in static_inputs.items()
        }
        return {"states": self.node._capture(SimpleNamespace(static_inputs=padded))["states"][:real_bs]}


@pytest.mark.parametrize("causal", [False, True], ids=["offline", "causal"])
@pytest.mark.parametrize("captured", [False, True], ids=["eager", "captured"])
def test_encoder_node_batches_like_single_requests(causal, captured):
    """The node groups a batch into replays (``_plan``), pads each group to its
    bucket (or to the longest row when eager) and runs a clip past every bucket
    alone; every row gets what it would alone."""
    from mstar.model.components.fastconformer_submodule import (
        MEL_BUCKETS,
        FastConformerEncoderSubmodule,
        _region,
    )

    torch.manual_seed(11)
    config = _causal_config(3) if causal else FastConformerConfig.from_hf(TINY)
    enc = FastConformerEncoder(config).eval()
    for p in enc.parameters():
        torch.nn.init.normal_(p, std=0.2)
    node = FastConformerEncoderSubmodule(enc, normalize_features=False)
    # two rows in the smallest bucket, one each in two others, one past them all
    lens = [100, 300, 250, 600, MEL_BUCKETS[-1] + 104]
    feats = [torch.randn(n, 16) for n in lens]
    out_lengths = [int(config.output_lengths(n)) for n in lens]
    runners = {_region(b): _PaddedReplay(node, b) for b in MEL_BUCKETS} if captured else {}
    with torch.no_grad():
        states = node._encode(feats, out_lengths, runners)
        for i, n in enumerate(lens):
            single, _ = enc(feats[i][None], torch.tensor([n]))
            assert states[i].shape == single[0].shape
            assert torch.allclose(states[i], single[0], atol=1e-5), (i, (states[i] - single[0]).abs().max())



def test_encoder_node_features_pad_per_group_not_to_the_batch(monkeypatch):
    """``preprocess`` computes log-mel per replay group, so a clip past every
    bucket doesn't pad the batch's short clips to its length, and each row
    still gets the features it would alone."""
    from mstar.model.components.fastconformer_submodule import MEL_BUCKETS, FastConformerEncoderSubmodule
    from mstar.model.submodule_base import NodeInputs

    torch.manual_seed(5)
    node = FastConformerEncoderSubmodule(FastConformerEncoder(FastConformerConfig.from_hf(TINY)).eval())
    hop = node.log_mel.hop_length
    clips = [torch.randn(n * hop) for n in (100, 300, MEL_BUCKETS[-1] + 500, 120)]
    batch, padded_to = node.log_mel.batch, []

    def spy(waveforms):
        padded_to.append(max(w.numel() for w in waveforms))
        return batch(waveforms)

    monkeypatch.setattr(node.log_mel, "batch", spy)
    out = node.preprocess("prefill", None, [NodeInputs(tensor_inputs={"audio": c}, input_seq_len=1) for c in clips])
    assert padded_to.count(clips[2].numel()) == 1
    assert max(n for n in padded_to if n != clips[2].numel()) <= clips[1].numel()
    for clip, feats in zip(clips, out["features"], strict=True):
        torch.testing.assert_close(feats, batch([clip])[0].to(feats.dtype), rtol=1e-5, atol=1e-5)


def test_replay_plan_covers_every_row_and_merges_short_clips():
    """``_plan`` gives each row one group; short clips ride in a longer clip's
    replay when a replay of their own costs more than the padding, rows of one
    long bucket keep their own, and a clip past every bucket runs eagerly."""
    from mstar.model.components.fastconformer_submodule import MEL_BUCKETS, _part_sizes, _plan, _replay_cost

    def cost(groups):
        return sum(sum(_replay_cost(bs, b) for bs in _part_sizes(len(rows), b)) for b, rows in groups if b)

    lens = [200, 3000, 600, 300, 900, 1500]
    groups = _plan(lens)
    assert sorted(r for _, rows in groups for r in rows) == list(range(len(lens)))
    assert all(max(lens[r] for r in rows) <= b for b, rows in groups)
    per_bucket = {}
    for r, n in enumerate(lens):
        per_bucket.setdefault(next(b for b in MEL_BUCKETS if n <= b), []).append(r)
    assert len(groups) < len(per_bucket) and cost(groups) < cost(per_bucket.items())
    # eight full-length clips and four short ones: padding the short ones to 4096 costs more
    assert [(b, len(rows)) for b, rows in _plan([3000] * 8 + [200] * 4)] == [(256, 4), (4096, 8)]
    assert _plan([MEL_BUCKETS[-1] + 1, 300]) == [(512, [1]), (None, [0])]

def test_reproduces_the_duplex_encoder():
    """``nemotron_duplex`` keeps its own encoder: a causal, bias-free
    FastConformer under NeMo's names. Its weights load into this one through
    ``nemo_to_hf_name``, and on the same features both give the same frames,
    the ground a later merge of the two stands on. (They differ before the
    encoder: duplex's featurizer keeps one more mel frame per clip.)"""
    from mstar.model.nemotron_duplex.components.conformer import ConformerEncoder
    from mstar.model.nemotron_duplex.config import ConformerSTTConfig

    duplex_config = ConformerSTTConfig(
        d_model=32, num_heads=4, num_layers=2, feat_in=16, n_mels=16, conv_kernel_size=9,
        subsampling_conv_channels=8, att_context_size=(8, 0),
    )
    torch.manual_seed(12)
    duplex = ConformerEncoder(duplex_config).eval()
    with torch.no_grad():
        for p in duplex.parameters():
            p.normal_(0, 0.2)
    ours = FastConformerEncoder(FastConformerConfig(
        hidden_size=32, num_layers=2, num_heads=4, intermediate_size=128, conv_kernel_size=9,
        num_mel_bins=16, subsampling_channels=8, attention_bias=False, convolution_bias=False,
        causal=True, left_context=8, lookahead=0,
    )).eval()
    loaded = ours.load_hf_weights((nemo_to_hf_name(name), t) for name, t in duplex.state_dict().items())
    assert loaded == {name for name, _ in ours.named_parameters()}
    feats = torch.randn(1, 203, 16)
    with torch.no_grad():
        expected, expected_lens = duplex(feats.transpose(1, 2), torch.tensor([203]))
        out, out_lens = ours(feats, torch.tensor([203]))
    assert out_lens.tolist() == expected_lens.tolist()
    assert torch.allclose(out, expected.transpose(1, 2), atol=1e-5), (out - expected.transpose(1, 2)).abs().max()


_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton kernels need a GPU")


@_needs_gpu
@pytest.mark.parametrize("causal", [False, True], ids=["offline", "causal"])
def test_conv_module_kernel_matches_eager_ops(causal):
    """GLU, padding mask, depthwise conv, bias and SiLU in one kernel against
    the eager ops on bf16: within two rounding steps of them everywhere, and
    no further from the same function in fp32 than they are. Which elements
    differ by a step depends on the platform (``exp``'s last bit; ROCm's eager
    depthwise conv runs on MIOpen's naive kernel), so the test bounds the
    error rather than counting matches. Odd T and C exercise the tile edges."""
    from mstar.model.components.fastconformer_kernels import glu_depthwise_conv

    F = torch.nn.functional
    torch.manual_seed(14)
    b, t, c, k = 3, 77, 200, 9
    dev, dt = "cuda", torch.bfloat16
    x = torch.randn(b, t, 2 * c, device=dev, dtype=dt)
    weight = (torch.randn(c, 1, k, device=dev) * 0.3).to(dt)
    bias = torch.randn(c, device=dev).to(dt)
    valid = torch.arange(t, device=dev)[None] < torch.tensor([77, 40, 9], device=dev)[:, None]
    left_pad = k - 1 if causal else (k - 1) // 2

    def eager(x, weight, bias, silu):
        y = F.glu(x, dim=-1).masked_fill(~valid[:, :, None], 0.0).transpose(1, 2)
        y = F.conv1d(F.pad(y, (left_pad, k - 1 - left_pad)), weight, bias, groups=c).transpose(1, 2)
        return F.silu(y) if silu else y

    for silu in (True, False):
        fused = glu_depthwise_conv(x, valid, weight, bias, left_pad, silu=silu)
        ref = eager(x, weight, bias, silu)
        exact = eager(x.float(), weight.float(), bias.float(), silu)
        torch.testing.assert_close(fused, ref, rtol=2 ** -6, atol=1e-2)
        err_fused = (fused.float() - exact).abs().mean().item()
        err_eager = (ref.float() - exact).abs().mean().item()
        assert err_fused <= 1.1 * err_eager, (err_fused, err_eager)


@_needs_gpu
def test_subsampling_kernel_matches_eager_ops():
    """Bias, ReLU and the padding mask on a conv output, in place: the eager
    ops' rounding exactly."""
    from mstar.model.components.fastconformer_kernels import bias_relu_mask_

    torch.manual_seed(15)
    x = torch.randn(3, 24, 37, 11, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(24, device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([37, 20, 1], device="cuda")
    mask = (torch.arange(37, device="cuda")[None] < lengths[:, None])[:, None, :, None]
    ref = torch.relu((x + bias[None, :, None, None]) * mask)
    assert torch.equal(bias_relu_mask_(x.clone(), bias, lengths), ref)
    assert torch.equal(bias_relu_mask_(x.clone(), bias, None), torch.relu(x + bias[None, :, None, None]))


@_needs_gpu
@pytest.mark.parametrize("causal", [False, True], ids=["offline", "causal"])
def test_triton_encoder_matches_eager_and_single_requests(causal, monkeypatch, torch_flag):
    """The whole encoder on the GPU, fp32 without TF32: the Triton path
    agrees with the eager one, and a padded batch gives each row what it
    gets alone."""
    from mstar.model.components import fastconformer_kernels

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch_flag(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(16)
    config = _causal_config(3) if causal else FastConformerConfig.from_hf(TINY)
    enc = FastConformerEncoder(config).eval().cuda()
    for p in enc.parameters():
        torch.nn.init.normal_(p, std=0.2)
    lens = [203, 97, 150]
    feats = torch.zeros(3, 203, 16, device="cuda")
    for i, n in enumerate(lens):
        feats[i, :n] = torch.randn(n, 16)
    with torch.no_grad():
        fused, out_lens = enc(feats, torch.tensor(lens, device="cuda"))
        for i, n in enumerate(lens):
            single, _ = enc(feats[i:i + 1, :n], torch.tensor([n], device="cuda"))
            m = int(out_lens[i])
            torch.testing.assert_close(fused[i, :m], single[0], atol=1e-4, rtol=1e-4)
        monkeypatch.setattr(fastconformer_kernels, "_TRITON_AVAILABLE", False)
        eager, _ = enc(feats, torch.tensor(lens, device="cuda"))
    torch.testing.assert_close(fused, eager, atol=1e-4, rtol=1e-4)


@_needs_gpu
def test_add_layer_norm_kernel_matches_eager_ops():
    """The residual add (alpha 0.5 and 1) and the LayerNorm after it in one
    pass, on bf16: the sum exactly, the norm within a rounding step."""
    from mstar.model.components.fastconformer_kernels import add_layer_norm

    torch.manual_seed(17)
    norm = torch.nn.LayerNorm(200).cuda().to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.normal_(1, 0.2)
        norm.bias.normal_(0, 0.2)
        x = torch.randn(3, 77, 200, device="cuda", dtype=torch.bfloat16)
        delta = torch.randn_like(x)
        for alpha in (0.5, 1.0):
            s, y = add_layer_norm(x, delta, alpha, norm)
            ref = torch.add(x, delta, alpha=alpha)
            assert torch.equal(s, ref)
            torch.testing.assert_close(y, norm(ref), rtol=2 ** -7, atol=1e-2)
            assert (y != norm(ref)).float().mean() < 0.01
            assert add_layer_norm(x, delta, alpha, norm, keep_sum=False)[0] is None
