"""CosyVoice3 flow solves that run only a hop's new frames over the cached K/V of the rest.

CPU tests cover which rows go incremental, the packed layout and the conv tails
(checked against a whole-utterance convolution). The CUDA tests capture the region
against the real KV, attention and tail resources and compare it with one whole
streaming solve, and with itself cut into other hops.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mstar.engine.accelerator_graph_backend import get_accelerator_graph_backend
from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.spec import resolve_spec_dependencies
from mstar.model.cosyvoice3.components.dit import DiT, PackedRows
from mstar.model.cosyvoice3.components.flow import CosyVoice3Flow, CosyVoice3FlowRow, ReferenceVoice
from mstar.model.cosyvoice3.config import (
    FLOW_ATTN,
    FLOW_COND_LABEL,
    FLOW_KV,
    FLOW_TAILS,
    FLOW_UNCOND_LABEL,
    CosyVoice3Config,
    DiTConfig,
    FlowConfig,
)
from mstar.model.cosyvoice3.cosyvoice3_model import CosyVoice3Model
from mstar.model.cosyvoice3.submodules import FLOW_NEW_FRAMES_REGION, FlowSubmodule, flow_region

VOCAB = 40
CHUNK = 50


def _config(**overrides) -> CosyVoice3Config:
    flow = FlowConfig(
        vocab_size=VOCAB,
        pre_lookahead_channels=32,
        spk_embed_dim=16,
        dit=DiTConfig(dim=128, depth=2, heads=2, dim_head=64, conv_pos_kernel=5, conv_pos_groups=4, time_freq_dim=16),
        noise_frames=2000,
    )
    flow.cfm.n_timesteps = 3
    cfg = CosyVoice3Config(
        flow=flow,
        silent_tokens=(1, 2),
        max_speech_tokens=512,
        flow_capture_frame_buckets=(128, 256),
        flow_capture_batch_sizes=(1, 2, 4),
        flow_capture_batch_frames=1024,
        flow_cache_frames=512,
        flow_new_frame_buckets=(128, 256, 512, 1024),
        flow_new_batch_sizes=(1, 2, 4),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _flow(cfg, dtype=torch.float32, device="cpu", seed=0) -> CosyVoice3Flow:
    torch.manual_seed(seed)
    flow = CosyVoice3Flow(cfg.flow)
    with torch.no_grad():
        for name, p in flow.named_parameters():
            if p.ndim >= 2:
                torch.nn.init.normal_(p, std=0.02 if "embedding" not in name else 0.5)
            else:
                torch.nn.init.zeros_(p)
        for block in flow.decoder.estimator.transformer_blocks:
            # AdaLN-zero's zeros would leave every block out of the solve
            block.attn_norm.linear.bias.fill_(0.1)
    flow = flow.to(device).eval()
    if dtype != torch.float32:
        flow.set_estimator_dtype(dtype)
    return flow


def _voice(cfg, prompt_tokens, seed, device="cpu") -> ReferenceVoice:
    gen = torch.Generator().manual_seed(seed)
    return ReferenceVoice(
        prompt_tokens=torch.randint(0, VOCAB, (prompt_tokens,), generator=gen).to(device),
        prompt_feat=torch.randn(2 * prompt_tokens, cfg.flow.output_size, generator=gen).to(device),
        embedding=torch.randn(cfg.flow.spk_embed_dim, generator=gen).to(device),
    )


def _tokens(n, seed, device="cpu") -> torch.Tensor:
    return torch.randint(0, VOCAB, (n,), generator=torch.Generator().manual_seed(seed)).to(device)


def _node(cfg, flow=None) -> FlowSubmodule:
    return FlowSubmodule(flow if flow is not None else _flow(cfg), None, cfg)


# -- which rows go incremental ----------------------------------------------------------


def _row(cfg, settle, prompt=5, final=False, streaming=True):
    tokens = _tokens(settle + 3, 1)
    return CosyVoice3FlowRow(_voice(cfg, prompt, 1), tokens, settle, final=final, streaming=streaming)


def test_a_row_runs_from_what_it_has_cached():
    cfg = _config()
    node = _node(cfg)
    state = node.request_state(1)
    item = node._new_frames_row(0, 1, _row(cfg, 20), 3, object())
    assert (item.first, item.end) == (0, 50)
    state.add("cached_frames", 50)
    item = node._new_frames_row(0, 1, _row(cfg, 45), 3, object())
    assert (item.first, item.end) == (50, 100)


def test_a_row_past_its_budget_resolves_for_good():
    cfg = _config(flow_cache_frames=100)
    node = _node(cfg)
    assert node._new_frames_row(0, 2, _row(cfg, 45), 3, object()) is not None
    assert node._new_frames_row(0, 2, _row(cfg, 70), 3, object()) is None
    assert node.request_state(2).get("resolving") is True
    # even a row short enough again, which no stream does, would miss the frames skipped
    assert node._new_frames_row(0, 2, _row(cfg, 45), 3, object()) is None


@pytest.mark.parametrize(
    "steps, streaming, runner", [(9, True, object()), (3, False, object()), (3, True, None)],
    ids=["other_step_count", "not_streaming", "no_capture"],
)
def test_what_the_captured_solve_cant_run_resolves(steps, streaming, runner):
    cfg = _config()
    node = _node(cfg)
    node.request_state(3)
    assert node._new_frames_row(0, 3, _row(cfg, 20, streaming=streaming), steps, runner) is None


def test_batches_stay_within_the_largest_capture():
    cfg = _config(flow_new_frame_buckets=(100, 300), flow_new_batch_sizes=(1, 2))
    node = _node(cfg)
    items = [SimpleNamespace(first=0, end=n) for n in (150, 100, 100, 250, 10, 10)]
    batches = node._new_frames_batches(items)
    assert [[i.end for i in b] for b in batches] == [[150, 100], [100], [250, 10], [10]]


def test_a_capture_left_out_for_memory_falls_back_to_resolving(monkeypatch):
    cfg = _config()
    node = _node(cfg)
    node.request_state(4)
    region = SimpleNamespace(can_run=lambda bs, tokens: False)
    called = []
    monkeypatch.setattr(node, "_solve", lambda rows, steps, runners: called.append(len(rows)) or [torch.zeros(80, 2)])
    # the first hop settles 45 tokens behind a 5-token prompt, and reads 3 more
    tokens = _tokens(60, 1)
    with torch.no_grad():
        node._run([4], [tokens], [_voice(cfg, 5, 1)], [False], [3], {FLOW_NEW_FRAMES_REGION: region})
    assert called == [1] and node.request_state(4).get("resolving") is True


def test_packing_lays_each_rows_tail_before_its_new_frames():
    cfg = _config()
    node = _node(cfg)
    rows = [_row(cfg, 20), _row(cfg, 45)]
    from mstar.model.cosyvoice3.submodules import _NewFramesRow

    batch = [_NewFramesRow(0, rows[0], 0, 50), _NewFramesRow(1, rows[1], 50, 100)]
    packed = node._pack_new_frames(batch).rows
    k = cfg.flow.dit.conv_pos_kernel - 1
    assert packed.positions.tolist() == [*range(50), *range(50, 100)]
    assert packed.tail_at[0].tolist() == list(range(1, 1 + k))
    assert packed.seq_at[0].item() == 1 + k
    assert packed.tail_at[1, 0].item() == 1 + k + 50
    assert packed.seq_at[50].item() == 1 + k + 50 + k
    # the row's tail after the step ends where its new frames end
    assert packed.last_at[0, -1].item() == packed.seq_at[49].item()
    assert packed.last_at[1, -1].item() == packed.seq_at[99].item()


# -- conv tails --------------------------------------------------------------------------


def test_conv_position_with_tails_equals_one_whole_convolution():
    """Two rows cut into hops of different lengths, with the buffer's padding at zero."""
    dit = DiT(_config().flow.dit).double().eval()
    k = dit.config.conv_pos_kernel - 1
    lengths = [60, 37]
    gen = torch.Generator().manual_seed(0)
    whole = [torch.randn(2, n, dit.config.dim, generator=gen, dtype=torch.float64) for n in lengths]
    want = [dit.input_embed.conv_pos_embed(w).detach() for w in whole]
    tails = torch.zeros(2, 2, 2, k, dit.config.dim, dtype=torch.float64)
    done, got = [0, 0], [[], []]
    with torch.no_grad():
        for news in ([30, 2], [10, 15], [20, 20]):
            total = sum(news)
            pad = 7
            positions, seq_at, tail_at, last_at, cursor = [], [], [], [], 1
            for n in news:
                tail_at.append(list(range(cursor, cursor + k)))
                seq_at += list(range(cursor + k, cursor + k + n))
                last_at.append(list(range(cursor + n, cursor + n + k)))
                positions += [0] * n
                cursor += k + n
            rows = PackedRows(
                positions=torch.tensor(positions + [0] * pad),
                seq_at=torch.tensor(seq_at + [0] * pad),
                tail_at=torch.tensor(tail_at),
                last_at=torch.tensor(last_at),
                real=torch.tensor([True] * total + [False] * pad),
            )
            h = torch.cat([w[:, d : d + n] for w, d, n in zip(whole, done, news, strict=True)], dim=1)
            h = F.pad(h, (0, 0, 0, pad))
            out, tails = dit._conv_position(h, tails, rows)
            at = 0
            for r, n in enumerate(news):
                got[r].append(out[:, at : at + n])
                at += n
                done[r] += n
    for r in range(2):
        torch.testing.assert_close(torch.cat(got[r], dim=1), want[r], rtol=0, atol=1e-12)


# -- the model's declaration -------------------------------------------------------------


def _model(tmp_path, monkeypatch, **kwargs) -> CosyVoice3Model:
    monkeypatch.setattr("mstar.model.cosyvoice3.cosyvoice3_model.resolve_snapshot", lambda *a: str(tmp_path))
    monkeypatch.setattr(CosyVoice3Model, "_build_text_tokenizer", lambda self: None)
    return CosyVoice3Model("x", **kwargs)


def test_the_flow_declares_its_cache_when_on(tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    keys = {spec.resource_key for spec in model.get_node_resources()}
    assert {FLOW_KV, FLOW_ATTN, FLOW_TAILS} <= keys
    assert set(model.kv_labels()[FLOW_KV]) == {FLOW_COND_LABEL, FLOW_UNCOND_LABEL}
    configs = model.get_request_resource_configs(["text"], ["audio"], {})
    assert configs[FLOW_KV].needed_labels == [FLOW_COND_LABEL, FLOW_UNCOND_LABEL]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"flow_cache_frames": 0}, {"stream_hop_tokens": 0}, {"flow_estimator_dtype": "float32"},
        {"lm_dtype": "float16"}, {"flow_new_frame_buckets": []}, {"stream_max_hop_tokens": 60},
    ],
    ids=["asked", "not_streaming", "float32_estimator", "other_kv_dtype", "no_captures", "hops_off_chunks"],
)
def test_the_cache_is_off_where_it_cant_run(tmp_path, monkeypatch, kwargs):
    model = _model(tmp_path, monkeypatch, **kwargs)
    assert model.config.flow_cache_frames == 0
    assert FLOW_KV not in {spec.resource_key for spec in model.get_node_resources()}
    assert FLOW_KV not in model.get_request_resource_configs(["text"], ["audio"], {})


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"flow_cache_frames": 1000}, "multiple of"),
        ({"flow_cache_frames": 4096, "flow_new_frame_buckets": [128]}, "past the largest"),
    ],
)
def test_a_budget_no_capture_or_page_holds_is_refused(tmp_path, monkeypatch, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _model(tmp_path, monkeypatch, **kwargs)


def test_a_pool_too_small_for_the_requests_is_refused(tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    pages = model._flow_kv_pages(4)
    config = {
        "model": "cosyvoice3",
        "max_concurrent_requests": 4,
        "resources": {FLOW_KV: {"max_num_pages": pages - 1}},
    }
    with pytest.raises(ValueError, match="two streams"):
        model.validate_config_yaml(config, "test.yaml")
    config["resources"][FLOW_KV]["max_num_pages"] = pages
    model.validate_config_yaml(config, "test.yaml")


def test_without_an_override_the_cap_is_checked_against_the_default_pool(tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    from mstar.model.cosyvoice3.cosyvoice3_model import DEFAULT_FLOW_REQUESTS

    config = {"model": "cosyvoice3", "max_concurrent_requests": DEFAULT_FLOW_REQUESTS}
    model.validate_config_yaml(config, "test.yaml")
    with pytest.raises(ValueError, match="two streams"):
        model.validate_config_yaml({**config, "max_concurrent_requests": DEFAULT_FLOW_REQUESTS + 1}, "test.yaml")


# -- the region against the real resources ------------------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer's kernels and capture need a GPU")


class _NoTransfer:
    """Nothing here sends KV to another worker."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture(autouse=True)
def _no_transfer(monkeypatch):
    monkeypatch.setattr("mstar.engine.resources.kv.manager.KVTransferManager", _NoTransfer)


class _Rig:
    """A flow node over real resources and captured regions, without a server."""

    def __init__(self, cfg, seed=0):
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.cfg = cfg
        self.flow = _flow(cfg, torch.bfloat16, self.device, seed)
        model = SimpleNamespace(config=cfg, _lm_dtype=torch.bfloat16)
        model._flow_kv_pages = lambda requests: CosyVoice3Model._flow_kv_pages(model, requests)
        specs = CosyVoice3Model._flow_resources(model)
        by_key = resolve_spec_dependencies(specs)
        self.resources = {}
        for spec in specs:
            if spec.resource_key == FLOW_KV:
                spec.config.max_num_pages = 512
            info = EngineResourceInfo(
                device=self.device, compute_dtype=torch.bfloat16,
                dependencies={key: by_key[key] for key in spec.depends_on()},
            )
            self.resources[spec.resource_key] = build_resource(spec, info)
        self.steps = StepRunner(self.resources, node_resources={"flow": list(self.resources)})
        self.node = FlowSubmodule(self.flow, None, cfg)
        self.node.bind_node_resources(self.resources)
        pool = get_accelerator_graph_backend(self.device).graph_pool_handle()
        self.runners = {}
        configs = self.node.get_piecewise_accelerator_graph_configs(self.device, torch.bfloat16)
        for label, config in configs.items():
            runner = PiecewiseAcceleratorGraphRunner(
                label=label, config=config, resources=self.resources, step_runner=self.steps, device=self.device,
                autocast_dtype=None, num_slots=1, memory_pool=pool,
            )
            runner.warmup_and_capture(None)
            assert runner.any_graphs and not runner.dropped_shapes
            self.runners[label] = runner

    def open(self, rid):
        self.node.request_state(rid)
        self.steps.ingest_request(rid, {FLOW_KV: self._kv_config()})

    def _kv_config(self):
        from mstar.engine.resources.kv.config import KVReqConfig

        return KVReqConfig(needed_labels=[FLOW_COND_LABEL, FLOW_UNCOND_LABEL])

    def hop(self, rids, tokens_list, voices, finals):
        """One node step; each row's newly settled mel."""
        n = self.cfg.flow.cfm.n_timesteps
        with torch.no_grad():
            out = self.node._run(rids, tokens_list, voices, finals, [n] * len(rids), self.runners)
        return [o["mel"][0] for o in out]


def _chain(rig, rid, voice, tokens, cuts):
    """A stream cut at ``cuts`` tokens delivered; its whole mel."""
    rig.open(rid)
    got = []
    for i, n in enumerate(cuts):
        final = i == len(cuts) - 1
        got.append(rig.hop([rid], [tokens[:n]], [voice], [final])[0])
    return torch.cat(got, dim=-1)


@cuda
def test_how_a_stream_is_cut_changes_no_frame():
    cfg = _config()
    rig = _Rig(cfg)
    voice = _voice(cfg, 25, 3, rig.device)
    tokens = _tokens(120, 4, rig.device)
    # the hop schedule settles on chunks, with look-ahead; the deliveries decide how many settle at once.
    # Polls that settle nothing change no bit, and one long solve only its kernels' rounding.
    a = _chain(rig, 1, voice, tokens, [28, 78, 120])
    b = _chain(rig, 2, voice, tokens, [120])
    c = _chain(rig, 3, voice, tokens, [28, 53, 78, 103, 120])
    assert a.shape == b.shape == c.shape == (cfg.flow.output_size, 240)
    assert torch.equal(a, c)
    assert (a - b).abs().mean() < 0.01 * a.abs().mean()


@cuda
def test_new_frames_match_a_whole_solve_to_the_estimators_precision():
    """Against the re-solving path, whose attention is SDPA: other kernels, same mask."""
    cfg = _config()
    rig = _Rig(cfg)
    voice = _voice(cfg, 25, 3, rig.device)
    tokens = _tokens(120, 4, rig.device)
    got = _chain(rig, 1, voice, tokens, [28, 78, 120])
    row = CosyVoice3FlowRow(voice, tokens, 120, final=True, streaming=True)
    with torch.no_grad():
        want = rig.flow.solve_rows([row], cfg.flow.cfm.n_timesteps)[0]
    assert got.shape == want.shape
    err = (got.float() - want.float()).abs()
    scale = want.abs().mean()
    assert err.mean() < 0.05 * scale, f"mean {err.mean():.4f} against a mel of mean size {scale:.3f}"


@cuda
def test_rows_batched_together_match_rows_alone():
    cfg = _config()
    rig = _Rig(cfg)
    voices = [_voice(cfg, 25, 10, rig.device), _voice(cfg, 15, 11, rig.device), _voice(cfg, 40, 12, rig.device)]
    tokens = [_tokens(90, 20, rig.device), _tokens(70, 21, rig.device), _tokens(60, 22, rig.device)]
    alone = [_chain(rig, 10 + i, voices[i], t, [28, 90 if i == 0 else t.numel()]) for i, t in enumerate(tokens)]
    for rid in (20, 21, 22):
        rig.open(rid)
    # all three in one step, then all three final
    first = rig.hop([20, 21, 22], [t[:28] for t in tokens], voices, [False] * 3)
    last = rig.hop([20, 21, 22], tokens, voices, [True] * 3)
    for i in range(3):
        got = torch.cat([first[i], last[i]], dim=-1)
        assert got.shape == alone[i].shape
        err = (got.float() - alone[i].float()).abs()
        assert err.mean() < 0.02 * alone[i].abs().mean(), f"row {i}: {err.mean():.4f}"


@cuda
def test_a_bucket_larger_than_the_hop_changes_nothing():
    """A hop's frames padded up to a larger bucket solve as they do in a bucket they fill:
    both CFG branches keep their K/V and attention rows. Attention is made to matter
    here, since the default test weights barely use it."""
    cfg = _config(flow_new_batch_sizes=(1,))
    outs, totals = [], []
    for buckets in (None, "exact"):
        if buckets == "exact":
            cfg = _config(flow_new_batch_sizes=(1,), flow_new_frame_buckets=tuple(sorted(set(totals))))
        rig = _Rig(cfg)
        with torch.no_grad():
            for block in rig.flow.decoder.estimator.transformer_blocks:
                block.attn_norm.linear.bias.fill_(1.0)
                block.attn.to_out[0].weight.mul_(20.0)
        if buckets is None:
            real = rig.runners[FLOW_NEW_FRAMES_REGION].run

            def spy(*args, _real=real, **kwargs):
                totals.append(sum(kwargs["seq_lens"]) // 2)
                return _real(*args, **kwargs)

            rig.runners[FLOW_NEW_FRAMES_REGION].run = spy
        voice = _voice(cfg, 25, 3, rig.device)
        tokens = _tokens(120, 4, rig.device)
        outs.append(_chain(rig, 1, voice, tokens, [28, 78, 120]))
        if buckets is None:
            assert any(t not in cfg.flow_new_frame_buckets for t in totals), totals
    padded, exact = (o.float() for o in outs)
    err = (padded - exact).abs().mean()
    assert err < 0.002 * exact.abs().mean(), f"mean {err:.5f} against a mel of mean size {exact.abs().mean():.3f}"


@cuda
def test_a_row_past_its_budget_carries_on_by_resolving():
    cfg = _config(flow_cache_frames=256)
    rig = _Rig(cfg)
    assert flow_region(256) in rig.runners and FLOW_NEW_FRAMES_REGION in rig.runners
    voice = _voice(cfg, 25, 3, rig.device)
    tokens = _tokens(120, 4, rig.device)
    got = _chain(rig, 1, voice, tokens, [28, 78, 120])  # 50 + 100 + 240 frames with the prompt
    assert rig.node.request_state(1).get("resolving") is True
    assert got.shape == (cfg.flow.output_size, 240)
    row = CosyVoice3FlowRow(voice, tokens, 120, final=True, streaming=True)
    with torch.no_grad():
        want = rig.flow.solve_rows([row], cfg.flow.cfm.n_timesteps)[0]
    assert (got.float() - want.float()).abs().mean() < 0.05 * want.abs().mean()
    assert np.isfinite(got.float().cpu().numpy()).all()


def close(got, want):
    # a few bfloat16 rounding steps, relative to each value
    assert ((got.float() - want).abs() <= 0.03 * want.abs().clamp_min(1.0)).all()


@cuda
def test_the_fused_kernels_match_the_plain_ops():
    """A block's residual norm and partial rotary, against the modules they stand in for."""
    from mstar.model.cosyvoice3.components.dit import (
        _modulated_norm,
        _residual_modulated_norm,
        _table,
        apply_partial_rotary,
    )
    from mstar.model.cosyvoice3.components.dit_kernels import partial_rotary_

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)
    # a width past a multiple of 128 too: the scaffold's kernel masks its columns
    for dim in (1024, 1000):
        x = torch.randn(37, dim, dtype=torch.bfloat16, device=device, generator=gen)
        y = torch.randn(37, dim, dtype=torch.bfloat16, device=device, generator=gen)
        gate, shift, scale = (torch.randn(1, dim, dtype=torch.bfloat16, device=device, generator=gen) for _ in range(3))

        def plain_norm(v, dim=dim, shift=shift, scale=scale):
            return F.layer_norm(v.float(), (dim,), eps=1e-6) * (1 + scale.float()) + shift.float()

        close(_modulated_norm(x, _table(shift), _table(scale), 1e-6), plain_norm(x))
        out, norm = _residual_modulated_norm(x, y, _table(gate), _table(shift), _table(scale), 1e-6)
        want = (x.float() + gate.float() * y.float()).to(torch.bfloat16).float()
        close(out, want)
        close(norm, plain_norm(want))

    q = torch.randn(300, 1024, dtype=torch.bfloat16, device=device, generator=gen)
    k = torch.randn(300, 1024, dtype=torch.bfloat16, device=device, generator=gen)
    freqs = torch.outer(torch.arange(300, device=device).float(), torch.rand(32, device=device, generator=gen))
    freqs = torch.stack((freqs, freqs), dim=-1).flatten(-2)
    want_q = apply_partial_rotary(q[None], freqs[None])[0].clone()
    want_k = apply_partial_rotary(k[None], freqs[None])[0].clone()
    partial_rotary_(q, k, freqs.cos(), freqs.sin())
    assert torch.equal(q[:, 64:], want_q[:, 64:])
    close(q, want_q.float())
    close(k, want_k.float())
