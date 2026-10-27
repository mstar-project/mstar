"""The shared FastConformer encoder (Canary, Parakeet, Nemotron ASR, Nemotron
VoiceChat) on tiny random weights: parity with HF's encoders (when the
installed transformers has them), batch invariance offline and causal, the
causal variant's causality, the BatchNorm fold and NeMo's weight names; and the
stream node, whose frames over any chunking are the offline ones.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.bounded import BoundedKVSpec
from mstar.engine.resources.kv.bounded.manager import BoundedKVManager
from mstar.engine.resources.recurrent import RecurrentStateConfig, RecurrentStateSpec
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import StepContext
from mstar.model.components.audio_features import NemoLogMel
from mstar.model.components.fastconformer import (
    FastConformerConfig,
    FastConformerEncoder,
    chunked_attention_mask,
    nemo_to_hf_name,
)
from mstar.model.components.fastconformer_stream import (
    FastConformerStreamSubmodule,
    encoder_kv_config,
    encoder_state_blocks,
    front_state_blocks,
)
from mstar.model.submodule_base import BatchedModelOutput, ModelInputsFromEngine

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


def test_from_nemo_refuses_a_causal_encoder_with_no_chunk():
    """A causal encoder's chunk is its right context plus one frame: an
    unlimited right context (-1) makes none, so it is refused at load."""
    enc = {
        "d_model": 32, "n_layers": 2, "n_heads": 4, "ff_expansion_factor": 2, "conv_kernel_size": 9,
        "feat_in": 16, "subsampling_factor": 8, "subsampling_conv_channels": 8, "subsampling": "dw_striding",
        "causal_downsampling": True, "conv_context_size": "causal", "conv_norm_type": "layer_norm",
        "att_context_style": "chunked_limited",
    }
    assert FastConformerConfig.from_nemo({**enc, "att_context_size": [6, 0]}).lookahead == 0
    with pytest.raises(ValueError, match="att_context_size"):
        FastConformerConfig.from_nemo({**enc, "att_context_size": [6, -1]})


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


def test_encoder_node_fails_only_the_clip_its_featurizer_cannot_take():
    """A clip of two frames that the reflecting featurizer can't featurize (no
    longer than n_fft/2 samples) fails its own request in ``prepare_inputs``,
    not the whole batch's features in ``preprocess``."""
    from mstar.model.components.fastconformer_submodule import FastConformerEncoderSubmodule

    log_mel = NemoLogMel(num_mel_bins=16, normalize=False, reflect_edges=True)
    node = FastConformerEncoderSubmodule(FastConformerEncoder(FastConformerConfig.from_hf(TINY)), log_mel=log_mel)
    assert log_mel.num_frames(200) == 2
    with pytest.raises(ValueError, match="200 samples"):
        node.prepare_inputs("prefill", None, {"audio": [torch.randn(200)]})


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

# -- the stream node ----------------------------------------------------------

KV, CONV, FRONT = "encoder_kv", "encoder_state", "encoder_front"


def _stream_node(n_fft=512, win_length=400, lookahead=0, normalize=False, reflect_edges=True):
    """The stream node over a shrunk causal [6, 0] encoder with random weights
    and a featurizer with random filters, NeMo's pre-2.5 convention."""
    torch.manual_seed(0)
    enc = FastConformerEncoder(FastConformerConfig(
        hidden_size=64, num_heads=4, num_layers=3, intermediate_size=256, num_mel_bins=32, subsampling_channels=8,
        attention_bias=False, convolution_bias=False, causal=True, left_context=6, lookahead=lookahead,
    ), kv_key=KV, conv_key=CONV)
    proj = nn.Linear(64, 48)
    with torch.no_grad():
        for t in [*enc.parameters(), *proj.parameters()]:
            t.copy_(torch.randn_like(t) * 0.2)
    log_mel = NemoLogMel(
        num_mel_bins=32, n_fft=n_fft, win_length=win_length, normalize=normalize, reflect_edges=reflect_edges,
        mel_filters=torch.rand(32, n_fft // 2 + 1),
    )
    return FastConformerStreamSubmodule(
        enc.eval(), log_mel, proj.eval(), input_edge="audio_in", output_edge="audio_frame",
        front_key=FRONT, batch_sizes=[1, 2, 4, 8, 16, 32],
    )


def _captured_on_cpu(config, label, resources=None, step_runner=None, bs=(1, 2, 4)):
    """The engine's piecewise runner for ``config``, captured on CPU: the real
    runner (bucket padding, its own static buffers per bucket, the region's
    admit / plan / commit), with a "graph" whose replay re-runs the region
    eagerly. ``calls`` counts its replays."""
    from unittest import mock

    from mstar.engine import accelerator_graph_runner as cgr
    from mstar.engine.accelerator_graph_backend import AcceleratorGraphPool

    class _Replay:
        def __init__(self, fn):
            self.fn = fn

        def replay(self):
            self.fn()

    config.capture_batch_sizes = list(bs)
    backend = SimpleNamespace(is_available=lambda: True, set_device=lambda: None, synchronize=lambda: None)
    runner = cgr.PiecewiseAcceleratorGraphRunner(
        label=label, config=config, resources=resources or {}, step_runner=step_runner or StepRunner({}),
        device=torch.device("cpu"), autocast_dtype=None, num_slots=1, node_name=label,
        memory_pool=AcceleratorGraphPool("cpu"), device_capability=SimpleNamespace(graph_backend=backend),
    )
    # a CPU device has no capture plan, so every bucket is warmed up and captured; the warm-up's
    # memory readings are the accelerator's, which a CPU device has none of
    no_memory_stats = dict(memory_allocated=lambda device=None: 0, max_memory_allocated=lambda device=None: 0,
                           reset_peak_memory_stats=lambda device=None: None, empty_cache=lambda: None)
    with mock.patch.object(cgr, "capture_into_graph", lambda fn, *a: (_Replay(fn), fn())), \
            mock.patch.multiple(torch.accelerator, **no_memory_stats):
        runner.prepare_for_capture()
        runner.warmup_and_capture(None)
    assert runner.any_graphs and not runner.dropped_shapes
    runner.calls = 0
    run = runner.run

    def counted(*args, **kwargs):
        runner.calls += 1
        return run(*args, **kwargs)

    runner.run = counted
    return runner


class _Encoder:
    """The stream node with its resources, driven as the engine drives an eager
    step: declare, admit and plan them, preprocess, forward, commit. The bounded
    KV runs its torch path on CPU, in fp32 (the engine's KV dtype here)."""

    def __init__(self, **featurizer):
        self.node = _stream_node(**featurizer)
        config, hop = self.node.encoder.config, self.node.log_mel.hop_length

        def pool(key, num_layers, blocks):
            spec = RecurrentStateSpec(key, {"conformer_encoder"},
                                      RecurrentStateConfig(num_layers=num_layers, blocks=blocks, max_slots=8))
            return RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cpu")))

        self.pool = pool(CONV, config.num_layers, encoder_state_blocks(config, torch.float32))
        self.front = pool(FRONT, 1, front_state_blocks(config, hop))
        self.kv = BoundedKVManager.build(
            BoundedKVSpec(KV, {"conformer_encoder"}, encoder_kv_config(config, max_slots=8)),
            EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.float32),
        )
        self.resources = {CONV: self.pool, FRONT: self.front, KV: self.kv}
        self.node.bind_node_resources(self.resources)
        self.runner = StepRunner(self.resources)
        self.regions = {}       # the piecewise runners, captured on CPU, by label, when a test sets them

    def prepare(self, handle, chunk, final=False):
        inputs = {"audio_in": [chunk]} if chunk is not None else {"audio_in": []}
        return self.node.prepare_inputs("encode", SimpleNamespace(rid_handle=handle), inputs,
                                        is_final_stream_chunk=final)

    def declare(self, rids, inputs):
        """The node's declaration, with the leases the engine takes for it first."""
        from mstar.engine.engine import Engine

        engine = Engine.__new__(Engine)
        engine._submodules = {"conformer_encoder": SimpleNamespace(piecewise_runners=self.regions)}
        ctx = StepContext(request_ids=tuple(rids), graph_walk="encode", slot=0, capture=False)
        engine._maybe_lease_piecewise_regions("conformer_encoder", ctx, inputs)
        return self.node.declare_step("encode", rids, inputs, piecewise_leases=ctx.piecewise_leases)

    def step(self, rids, inputs) -> BatchedModelOutput:
        """One engine step over ``rids``: batched when the node can batch them,
        else each stream's own step, as the engine's per-request path runs it."""
        if len(rids) > 1 and not self.node.can_batch(None, inputs):
            out = BatchedModelOutput(per_rid_outputs={})
            for rid, inp in zip(rids, inputs, strict=True):
                out.per_rid_outputs.update(self._step([rid], [inp]).per_rid_outputs)
            return out
        return self._step(rids, inputs)

    def _step(self, rids, inputs) -> BatchedModelOutput:
        for rid in rids:
            for resource in self.resources.values():
                resource.ingest_request(rid)
        step = self.declare(rids, inputs)
        if step is not None:
            step.set_ctx(StepContext(request_ids=tuple(rids), graph_walk="encode", slot=0, capture=False))
            assert self.runner.admit(step).ok
            self.runner.plan(step)
        engine_inputs = ModelInputsFromEngine(
            request_ids=list(rids), per_request_info={}, resources=dict(self.resources),
            piecewise_runners=dict(self.regions), step=step,
        )
        with torch.no_grad():
            out = self.node.forward_batched("encode", engine_inputs,
                                            **self.node.preprocess("encode", engine_inputs, inputs))
        if step is not None:
            self.runner.commit(step)
        return out

    def encode(self, handle, chunk, final=False):
        """One step for one stream: what the stream hands the node, then its frames."""
        return self.step([handle], [self.prepare(handle, chunk, final)]).per_rid_outputs[handle]["audio_frame"]

    def offline(self, wav):
        """The whole clip through the same featurizer, encoder and projection."""
        with torch.no_grad():
            return self.node.encode_clip(wav)


def _encoder_regions(enc: _Encoder, *labels) -> dict:
    """The node's piecewise regions as the engine runs them, captured on CPU."""
    configs = enc.node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)
    return {label: _captured_on_cpu(configs[label], label, enc.resources, enc.runner) for label in labels}


def _encode_batch(enc, steps):
    """One batched step, as the engine runs it: each ``(handle, chunk, final)``
    row through prepare_inputs, then one step over all of them. Returns each
    row's frames by handle."""
    out = enc.step([h for h, _c, _f in steps], [enc.prepare(h, c, f) for h, c, f in steps]).per_rid_outputs
    return {h: out[h]["audio_frame"] for h, _c, _f in steps}


def _stream(enc, wavs, sizes):
    """Stream each call in its own chunk size, every call with audio left in
    each step, then end them all; the frames each call got."""
    got = {h: [] for h in wavs}
    pos = dict.fromkeys(wavs, 0)
    while any(pos[h] < wavs[h].numel() for h in wavs):
        rids, inputs = [], []
        for h in wavs:
            if pos[h] < wavs[h].numel():
                inputs.append(enc.prepare(h, wavs[h][pos[h]:pos[h] + sizes[h]]))
                pos[h] += sizes[h]
                rids.append(h)
        out = enc.step(rids, inputs).per_rid_outputs
        for h in rids:
            got[h] += out[h]["audio_frame"]
    for h in wavs:
        got[h] += enc.encode(h, None, final=True)
    return got


STEP, LAYERS = FastConformerStreamSubmodule.STEP_REGION, FastConformerStreamSubmodule.LAYERS_REGION


@pytest.mark.parametrize("end_rides_last_chunk", [False, True])
# 32160 samples is 201 hops, 1 mod 8: the length where NeMo's two mel conventions
# disagree on the encoder frame count
@pytest.mark.parametrize("n", [16000 * 2 + 777, 16000 * 2, 16000 * 2 + 160])
def test_encoder_streams_the_offline_frames(n, end_rides_last_chunk):
    """Chunks of any size, smaller than a frame included, then the end of input,
    on an empty final chunk or on the last chunk of audio (as an upload's usually
    is): the frames emitted in order are the offline encoding of the call, and
    the mel frames the offline featurizer's, the reflected right edge included
    (a length on the hop grid is the one whose last frame reads it), the newest
    of which the front end holds."""
    enc = _Encoder()
    wav = torch.randn(n) * 0.1
    want = enc.offline(wav)
    mel = enc.node.log_mel(wav)
    for handle, size in enumerate((320, 1280, 2560, 9000, 64000)):
        got = []
        for at in range(0, wav.numel(), size):
            last = at + size >= wav.numel()
            got += enc.encode(handle, wav[at:at + size], final=last and end_rides_last_chunk)
        if not end_rides_last_chunk:
            got += enc.encode(handle, None, final=True)
        assert len(got) == want.shape[0]
        assert torch.allclose(torch.stack(got), want, atol=1e-5), size
        st = enc.node.request_state(handle).get(enc.node.STATE_KEY)
        assert st.mels == mel.shape[0]
        held = enc.front.block("mel", 0)[enc.front._slots[handle]["main"].index]   # the newest, right-aligned
        assert torch.allclose(held, mel[-held.shape[0]:], atol=1e-5), size


def test_encoder_batches_calls_at_different_points():
    """Calls in one step, each with its own chunking (one frame a step, several,
    or none), each get their own offline frames."""
    enc = _Encoder()
    wavs = {1: torch.randn(16000 * 2) * 0.1, 2: torch.randn(16000 * 3) * 0.1, 3: torch.randn(9000) * 0.1}
    got = _stream(enc, wavs, {1: 1280, 2: 4000, 3: 320})
    for h, wav in wavs.items():
        assert torch.allclose(torch.stack(got[h]), enc.offline(wav), atol=1e-5), h


def test_encoder_live_steps_replay_the_regions_with_padding_rows():
    """Calls one frame of audio further on take the whole-step region; the
    others in the same step (a call's first frames, one under a frame a step)
    the eager front end and the blocks region, both in buckets wider than the
    batch. Every call gets its offline frames: the padding rows, which address
    the pools' sinks, and the rows another path served leave them alone."""
    enc = _Encoder()
    enc.regions = _encoder_regions(enc, STEP, LAYERS)
    step, layers = enc.regions.values()
    wavs = {1: torch.randn(16000 * 2) * 0.1, 2: torch.randn(16000) * 0.1, 3: torch.randn(6000) * 0.1}
    got = _stream(enc, wavs, {1: 1280, 2: 1280, 3: 640})
    assert step.calls >= 20 and layers.calls >= 5
    for h, wav in wavs.items():
        assert torch.allclose(torch.stack(got[h]), enc.offline(wav), atol=1e-5), h


def test_encoder_layer_region_runs_only_the_calls_with_a_frame():
    """Calls one frame on take the whole-step region; the blocks region then
    runs only the other calls that got a frame, not every call of the step."""
    enc = _Encoder()
    enc.regions = _encoder_regions(enc, STEP, LAYERS)
    layers = enc.regions[LAYERS]
    rows, run, step = [], layers.run, enc.step
    layers.run = lambda *a, request_ids, **k: rows.append(len(request_ids)) or run(*a, request_ids=request_ids, **k)
    enc.step = lambda rids, inputs: rows.append(f"step of {len(rids)}") or step(rids, inputs)
    wavs = {1: torch.randn(16000) * 0.1, 2: torch.randn(16000) * 0.1, 3: torch.randn(16000) * 0.1}
    got = _stream(enc, wavs, {1: 1280, 2: 1280, 3: 640})
    # once calls 1 and 2 are one frame on, the blocks region runs call 3 alone
    assert rows[4:10] == ["step of 3", 1, "step of 3", "step of 3", 1, "step of 3"], rows
    for h, wav in wavs.items():
        assert torch.allclose(torch.stack(got[h]), enc.offline(wav), atol=1e-5), h


def test_encoder_paths_read_the_plan_current_when_they_use_it(monkeypatch):
    """The KV attends under a plan of exactly the rows it is given, padding
    included, with the eager path on this step's plan before a region plans its
    own (the pools' gather / scatter_ read the plan the pool holds when they
    run). A call with a chunk over a second runs in a step of its own."""
    enc = _Encoder()
    enc.regions = _encoder_regions(enc, STEP, LAYERS)
    attend = enc.kv.attend

    def attend_checked(layer_idx, q, *args, **kwargs):
        assert len(enc.kv.current.rows) == q.shape[0], "the KV attended under a plan for other rows"
        return attend(layer_idx, q, *args, **kwargs)

    monkeypatch.setattr(enc.kv, "attend", attend_checked)
    wavs = {1: torch.randn(16000 * 2) * 0.1, 2: torch.randn(16000) * 0.1, 3: torch.randn(6000) * 0.1,
            4: torch.randn(16000 * 3) * 0.1}
    got = _stream(enc, wavs, {1: 1280, 2: 1280, 3: 640, 4: 20000})
    assert all(region.calls for region in enc.regions.values())
    for h, wav in wavs.items():
        assert torch.allclose(torch.stack(got[h]), enc.offline(wav), atol=1e-5), h


def _kv_spans(step) -> list[int]:
    return [seg.span for seg in step.get(KV).segments]


def test_encoder_an_all_steady_step_leaves_its_resources_to_its_region():
    """When every call is one frame on and the whole-step region is captured
    for the batch, the region's own step owns the resources: the node declares
    nothing, so no second plan is built and thrown away. Otherwise it declares
    every call in each, and in the KV its new frames at its position."""
    enc = _Encoder()
    wavs = {1: torch.randn(1280 * 6) * 0.1, 2: torch.randn(1280 * 6) * 0.1}
    for at in (0, 1280):
        enc.step([1, 2], [enc.prepare(h, wavs[h][at:at + 1280]) for h in (1, 2)])
    nxt = [enc.prepare(h, wavs[h][2560:3840]) for h in (1, 2)]
    step = enc.declare([1, 2], nxt)                                           # no region captured
    assert set(step.keys()) == {FRONT, CONV, KV}
    assert _kv_spans(step) == [1, 1]
    states = [enc.node.request_state(h).get(enc.node.STATE_KEY) for h in (1, 2)]
    assert [step.get(KV).positions[h].written for h in (1, 2)] == [st.frames for st in states]
    # the call's own position, built when its frames last moved, not one per declaration
    assert all(step.get(KV).positions[h] is st.position for h, st in zip((1, 2), states, strict=True))
    enc.regions = _encoder_regions(enc, STEP)
    assert enc.declare([1, 2], nxt) is None
    for at in range(2560, 1280 * 6, 1280):
        out = enc.step([1, 2], [enc.prepare(h, wavs[h][at:at + 1280]) for h in (1, 2)]).per_rid_outputs
        assert all(len(out[h]["audio_frame"]) == 1 for h in (1, 2))
    assert enc.regions[STEP].calls == 4
    mixed = [enc.prepare(1, torch.randn(640) * 0.1), enc.prepare(2, torch.randn(1280) * 0.1)]
    step = enc.declare([1, 2], mixed)
    assert set(step.keys()) == {FRONT, CONV, KV}
    assert _kv_spans(step) == [mixed[0].kwargs["n"], 1]


def test_encoder_a_step_mixes_a_steady_call_with_eager_chunks_past_the_window():
    """One step holds a call one frame on, a call whose chunk adds more frames
    than the attention window (the eager blocks, a step of T > 1 where each
    frame sees only the window before it, which the call one frame on then
    joins) and a call under a frame a step: each gets its offline frames."""
    enc = _Encoder()
    enc.regions = _encoder_regions(enc, STEP, LAYERS)
    window = enc.node.encoder.config.left_context
    wavs = {1: torch.randn(16000 * 3) * 0.1, 2: torch.randn(16000 * 3) * 0.1, 3: torch.randn(16000) * 0.1}
    sizes = {1: 1280, 2: 16000, 3: 640}
    assert 16000 // 1280 > window
    widths, front = [], enc.node._front

    def spy_front(*args):
        x = front(*args)
        widths.append(0 if x is None else x.shape[1])
        return x

    enc.node._front = spy_front
    got = _stream(enc, wavs, sizes)
    assert enc.regions[STEP].calls > 0 and max(widths) > window
    for h, wav in wavs.items():
        assert torch.allclose(torch.stack(got[h]), enc.offline(wav), atol=1e-5), h


def test_encoder_a_steady_call_joins_the_eager_layers_of_its_step():
    """A call one frame on whose window has wrapped, in a step where another
    call's chunk makes several frames: this step's KV plan, which the eager
    blocks run on, writes both calls' frames, so the steady call can't also
    take the whole-step region (whose write of its frame would follow one from
    padding into the ring entry the region reads)."""
    enc = _Encoder()
    enc.regions = _encoder_regions(enc, STEP, LAYERS)
    window = enc.node.encoder.config.left_context
    at = 1280 * (window + 4)
    wavs = {1: torch.randn(at + 1280 * 3) * 0.1, 2: torch.randn(9000) * 0.1}
    got = {1: [], 2: []}
    for pos in range(0, at, 1280):
        got[1] += enc.encode(1, wavs[1][pos:pos + 1280])
    out = _encode_batch(enc, [(1, wavs[1][at:at + 1280], False), (2, wavs[2], False)])
    assert len(out[2]) > 1
    for h in (1, 2):
        got[h] += out[h]
    got[1] += enc.encode(1, wavs[1][at + 1280:], final=True)
    got[2] += enc.encode(2, None, final=True)
    for h in (1, 2):
        assert torch.allclose(torch.stack(got[h]), enc.offline(wavs[h]), atol=1e-5), h


def test_encoder_runs_a_steady_call_eagerly_when_its_region_has_no_graph():
    """The whole-step region leases the batch's bucket, but replays only the
    steady calls: when the runner dropped the capture for that sub-batch (here
    one call's), they take the eager path, which this step's plans cover."""
    enc = _Encoder()
    enc.regions = _encoder_regions(enc, STEP, LAYERS)
    step_region = enc.regions[STEP]
    step_region._graphs = {k: v for k, v in step_region._graphs.items() if k.bs != 1}
    wavs = {1: torch.randn(1280 * 6) * 0.1, 2: torch.randn(1280 * 6) * 0.1}
    got = {1: [], 2: []}
    for at in (0, 1280, 2560):
        for h, frames in _encode_batch(enc, [(h, wavs[h][at:at + 1280], False) for h in (1, 2)]).items():
            got[h] += frames
    # call 1 is one frame on (steady), call 2 appends half a frame
    out = _encode_batch(enc, [(1, wavs[1][3840:5120], False), (2, wavs[2][3840:4480], False)])
    for h in (1, 2):
        got[h] += out[h]
    for h, end in ((1, 5120), (2, 4480)):
        got[h] += enc.encode(h, wavs[h][end:], final=True)
        assert torch.allclose(torch.stack(got[h]), enc.offline(wavs[h]), atol=1e-5), h


@pytest.mark.parametrize("b_end_rides", [False, True])
@pytest.mark.parametrize("a_chunk", [1280, 9000])
@pytest.mark.parametrize("b_len", [16000 * 2, 16000 * 2 + 640, 16000 * 2 + 777])
def test_encoder_a_calls_last_frames_do_not_depend_on_its_batch(b_len, a_chunk, b_end_rides):
    """Call B ends in the same step as call A's longer chunk, A still mid-call:
    B's last frames, which read past its audio, are still its offline ones."""
    enc = _Encoder()
    starts = list(range(0, b_len, 1280))
    steps = len(starts) + (0 if b_end_rides else 1)
    a, b = torch.randn(a_chunk * steps) * 0.1, torch.randn(b_len) * 0.1
    got_b = []
    for k, pb in enumerate(starts):
        b_final = b_end_rides and k == len(starts) - 1
        rows = [(1, a[k * a_chunk:(k + 1) * a_chunk], False), (2, b[pb:pb + 1280], b_final)]
        got_b += _encode_batch(enc, rows)[2]
    if not b_end_rides:
        got_b += _encode_batch(enc, [(1, a[-a_chunk:], False), (2, None, True)])[2]
    want = enc.offline(b)
    assert len(got_b) == want.shape[0]
    assert torch.allclose(torch.stack(got_b), want, atol=1e-5)


def test_encoder_keeps_a_chunk_over_a_second_out_of_a_batch():
    """A pass pads every call to its longest chunk, so a step holding an upload
    or a large append is not batched: can_batch refuses it, by the chunk length
    prepare_inputs reports, and the engine runs each call's step on its own.
    Calls with a second or less share one pass. The frames are the offline
    ones either way."""
    enc = _Encoder()
    fronts, front = [], enc.node._front

    def spy_front(sts, chunks, final, *rest):
        fronts.append(sorted(c.numel() for c in chunks if c is not None))
        return front(sts, chunks, final, *rest)

    enc.node._front = spy_front
    wav = {1: torch.randn(16000 * 5) * 0.1, 2: torch.randn(1280) * 0.1, 3: torch.randn(16001) * 0.1,
           4: torch.randn(16000) * 0.1}
    prepared = {h: enc.prepare(h, w, final=True) for h, w in wav.items()}
    assert {h: inp.input_seq_len for h, inp in prepared.items()} == {h: w.numel() for h, w in wav.items()}
    assert not enc.node.can_batch(None, list(prepared.values()))
    assert not enc.node.can_batch(None, [prepared[2], prepared[3]])
    assert enc.node.can_batch(None, [prepared[2], prepared[4]])
    assert enc.node.can_batch(None, [prepared[1]])                       # alone, it is its own step

    out = _encode_batch(enc, [(h, w, True) for h, w in wav.items()])
    assert sorted(fronts) == [[1280], [16000], [16001], [16000 * 5]]    # every call on its own
    for h, w in wav.items():
        assert torch.allclose(torch.stack(out[h]), enc.offline(w), atol=1e-5)

    fronts.clear()
    short = {5: torch.randn(1280) * 0.1, 6: torch.randn(16000) * 0.1}
    out = _encode_batch(enc, [(h, w, True) for h, w in short.items()])
    assert fronts == [[1280, 16000]]                                      # one pass
    for h, w in short.items():
        assert torch.allclose(torch.stack(out[h]), enc.offline(w), atol=1e-5)


def test_encoder_takes_no_batch_wider_than_its_graphs():
    """Its graphs are piecewise regions, which the engine's batch cap does not
    see: a wider step would run every call eagerly, so the node caps its batch
    at its widest capture and the scheduler splits the rest."""
    from mstar.engine.engine import Engine

    node = _stream_node()
    engine = Engine.__new__(Engine)
    engine._submodules = {"conformer_encoder": SimpleNamespace(submodule=node, accelerator_graph_runner=None,
                                                               capture_runner=None)}
    assert engine.get_max_batch_size("conformer_encoder", "encode") == max(node.capture_batch_sizes)


@pytest.mark.parametrize("unsupported", [
    dict(lookahead=3), dict(normalize=True), dict(reflect_edges=False), dict(n_fft=2048),
])
def test_encoder_refuses_what_it_cannot_stream(unsupported):
    """A right context, a per-clip normalization, the 2.5 featurizer (whose
    zero-padded edges a stream's front end does not cut) or a window longer
    than the samples the front end keeps are refused when the node is built,
    not met as wrong frames mid-call."""
    with pytest.raises(ValueError, match=next(iter(unsupported))):
        _stream_node(**unsupported)


@pytest.mark.parametrize("n_fft", [512, 1024])
def test_encoder_steady_steps_match_offline_for_the_featurizer_they_get(n_fft):
    """The whole-step region's slices assume the featurizer's half window spans
    more than one hop and at most two (512 / 160); with any other the encoder
    serves every step through the general path, which takes any geometry."""
    enc = _Encoder(n_fft=n_fft)
    regions = _encoder_regions(enc, STEP) if n_fft == 512 else {}
    assert (STEP in enc.node.get_piecewise_accelerator_graph_configs(
        torch.device("cpu"), torch.float32)) == (n_fft == 512)
    enc.regions = regions
    wav = torch.randn(16000 * 2) * 0.1
    got = torch.stack(_stream(enc, {1: wav}, {1: 1280})[1])
    assert torch.allclose(got, enc.offline(wav), atol=1e-5)
    if regions:
        assert regions[STEP].calls > 0


@pytest.mark.parametrize("n_fft", [512, 256])
@pytest.mark.parametrize("n", [1580, 1728])
def test_encoder_ends_a_call_with_its_last_frame_when_the_end_adds_no_mel_frame(n_fft, n):
    """With a half window under a hop, the end of input can make a last encoder
    frame final without adding a mel frame: the call still gets it."""
    enc = _Encoder(n_fft=n_fft, win_length=n_fft)
    wav = torch.randn(n) * 0.1
    got = []
    for at in range(0, n, 320):
        got += enc.encode(1, wav[at:at + 320])
    got += enc.encode(1, None, final=True)
    want = enc.offline(wav)
    assert len(got) == want.shape[0]
    assert torch.allclose(torch.stack(got), want, atol=1e-5)


def test_encoder_decides_a_calls_steadiness_once_per_step(monkeypatch):
    """Whether a call takes the whole-step region is decided once, where its
    inputs are prepared; the declaration and the forward read that decision."""
    enc = _Encoder()
    enc.regions = _encoder_regions(enc, STEP)
    wavs = {h: torch.randn(1280 * 4) * 0.1 for h in (1, 2)}

    def decided_already(*args):
        raise AssertionError("steadiness decided again after prepare_inputs")

    for at in range(0, 1280 * 4, 1280):
        inputs = [enc.prepare(h, wavs[h][at:at + 1280]) for h in wavs]
        with monkeypatch.context() as m:
            m.setattr(enc.node, "is_steady", decided_already)
            enc.step(list(wavs), inputs)
    assert enc.regions[STEP].calls == 2


def test_encoder_moves_a_steps_chunks_to_the_device_in_one_copy():
    """Each call's chunk arrives on the host; one copy per call would each be a
    blocking transfer from pageable memory, so the step copies them as one."""
    from torch.profiler import ProfilerActivity, profile

    chunks = [torch.randn(1280), None, torch.randn(640), torch.randn(1280)]
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        moved = FastConformerStreamSubmodule._to_device(chunks, torch.device("meta"))
    copies = [e for e in prof.events() if e.name == "aten::_to_copy"]
    assert len(copies) == 1
    assert [None if c is None else (c.device.type, c.shape) for c in moved] == [
        ("meta", (1280,)), None, ("meta", (640,)), ("meta", (1280,))]


def test_encoder_step_region_replays_three_calls_with_a_padding_row():
    """Three calls one frame on replay the whole-step region in a bucket of
    four: the padding row addresses both pools' sinks and the calls get their
    offline frames."""
    enc = _Encoder()
    enc.regions = _encoder_regions(enc, STEP)
    region = enc.regions[STEP]
    rows, run = [], region.run
    region.run = lambda *a, request_ids, **k: rows.append(len(request_ids)) or run(*a, request_ids=request_ids, **k)
    wavs = {h: torch.randn(16000) * 0.1 for h in (1, 2, 3)}
    got = _stream(enc, wavs, dict.fromkeys(wavs, 1280))
    assert rows and set(rows) == {3}
    for h, wav in wavs.items():
        assert torch.allclose(torch.stack(got[h]), enc.offline(wav), atol=1e-5), h


def test_encoder_offers_nothing_to_the_stop_check():
    """The encoder has no check_stop and its stream ends with the client's: its
    frames are not copied to the host every step for one."""
    enc = _Encoder()
    out = enc.step([1], [enc.prepare(1, torch.randn(4000) * 0.1)])
    assert isinstance(out, BatchedModelOutput) and out.check_stop_buffers == {}
    assert len(out.per_rid_outputs[1]["audio_frame"]) > 0


def test_encoder_waits_for_enough_audio_to_featurize():
    enc = _Encoder()
    assert enc.encode(1, torch.zeros(10)) == []


@pytest.mark.parametrize("n", [257, 1000, 1279, 1281])
def test_encoder_a_short_call_gets_the_offline_frames(n):
    """Offline encodes any call longer than n_fft/2 samples, under one frame too."""
    enc = _Encoder()
    wav = torch.randn(n) * 0.1
    want = enc.offline(wav)
    for handle, size in enumerate((100, n)):
        got = []
        for at in range(0, n, size):
            got += enc.encode(handle, wav[at:at + size])
        got += enc.encode(handle, None, final=True)
        assert len(got) == want.shape[0]
        assert torch.allclose(torch.stack(got), want, atol=1e-5), size


@pytest.mark.parametrize("chunks", [[], [torch.zeros(256)], [torch.zeros(100), torch.zeros(156)]])
def test_encoder_fails_a_call_that_ends_without_enough_audio(chunks):
    """No audio, or no more than n_fft/2 samples (which offline can't featurize):
    the call fails rather than reply to nothing."""
    enc = _Encoder()
    for chunk in chunks:
        enc.encode(1, chunk)
    with pytest.raises(ValueError, match="ended with"):
        enc.encode(1, None, final=True)


def test_encoder_state_is_per_call_and_cleaned_up():
    enc = _Encoder()
    free = enc.pool.num_free_slots
    enc.encode(1, torch.randn(4000) * 0.1)
    enc.encode(2, torch.randn(4000) * 0.1)
    assert enc.pool.num_free_slots == enc.front.num_free_slots == free - 2
    assert set(enc.kv._requests) == {1, 2}
    enc.node.cleanup_request(1)
    for resource in enc.resources.values():
        resource.remove_request(1)
    assert 1 not in enc.node.request_states and 2 in enc.node.request_states
    assert enc.pool.num_free_slots == enc.front.num_free_slots == free - 1
    assert set(enc.kv._requests) == {2}


def _without_fft(config):
    """``config`` with its region raising on any FFT op: torch.stft and
    torch.fft reach the dispatcher as ``aten._fft_*``."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class _NoFFT(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert not func.__name__.startswith("_fft_"), f"{func} in a captured region"
            return func(*args, **(kwargs or {}))

    fn = config.capture_fn

    def region(call):
        with _NoFFT():
            return fn(call)

    config.capture_fn = region
    return config


def test_encoder_regions_run_no_fft():
    """A captured cuFFT plan lives in torch's plan cache, whose LRU eviction
    destroys it under the graph; so the regions take their STFT as a matmul.
    The eager path, which the guard leaves alone, keeps torch.stft."""
    enc = _Encoder()
    configs = enc.node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)
    enc.regions = {label: _captured_on_cpu(_without_fft(configs[label]), label, enc.resources, enc.runner)
                   for label in (STEP, LAYERS)}
    wavs = {1: torch.randn(16000) * 0.1, 2: torch.randn(16000 + 640) * 0.1}
    got = _stream(enc, wavs, {1: 1280, 2: 640})
    assert all(enc.regions[label].calls for label in (STEP, LAYERS))
    for h, wav in wavs.items():
        assert torch.allclose(torch.stack(got[h]), enc.offline(wav), atol=1e-5), h


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

    torch_flag(torch.backends.cudnn, "allow_tf32", False)
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
