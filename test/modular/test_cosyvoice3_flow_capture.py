"""The CosyVoice3 flow solve as piecewise CUDA graphs per frame bucket.

CPU tests use tiny random float64 weights and drive the real
``PiecewiseAcceleratorGraphRunner`` (bucket and batch resolution, static-buffer
copies, padding rows, output copies), with ``capture_into_graph`` replaced by
one that "replays" by running the region eagerly. The CUDA test captures for
real and checks that a replay equals the eager solve of the same padded inputs.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from mstar.engine import accelerator_graph_runner as cgr
from mstar.engine.accelerator_graph_backend import get_accelerator_graph_backend
from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner
from mstar.model.cosyvoice3.components.flow import CosyVoice3Flow, CosyVoice3FlowRow, ReferenceVoice
from mstar.model.cosyvoice3.config import CosyVoice3Config, DiTConfig, FlowConfig
from mstar.model.cosyvoice3.submodules import MEL, FlowSubmodule, flow_region

VOCAB = 40


@pytest.fixture(autouse=True)
def _threads():
    prior = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(prior)


def _config(**overrides) -> CosyVoice3Config:
    flow = FlowConfig(
        vocab_size=VOCAB,
        pre_lookahead_channels=32,
        spk_embed_dim=16,
        dit=DiTConfig(dim=32, depth=2, heads=2, dim_head=16, conv_pos_kernel=5, conv_pos_groups=4, time_freq_dim=16),
        noise_frames=2000,
    )
    flow.cfm.n_timesteps = 3
    cfg = CosyVoice3Config(
        flow=flow,
        silent_tokens=(1, 2),
        max_speech_tokens=512,
        flow_capture_frame_buckets=(64, 96, 128),
        flow_capture_batch_sizes=(1, 2, 4),
        flow_capture_batch_frames=256,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _flow(cfg: CosyVoice3Config, dtype=torch.float64, seed: int = 0) -> CosyVoice3Flow:
    torch.manual_seed(seed)
    flow = CosyVoice3Flow(cfg.flow)
    with torch.no_grad():
        for param in flow.parameters():
            param.normal_(0, 0.2)
    return flow.to(dtype).eval()


def _voice(cfg: CosyVoice3Config, prompt_tokens: int, seed: int = 0, dtype=torch.float64) -> ReferenceVoice:
    g = torch.Generator().manual_seed(seed)
    return ReferenceVoice(
        prompt_tokens=torch.randint(0, VOCAB, (prompt_tokens,), generator=g),
        prompt_feat=torch.randn(2 * prompt_tokens, cfg.flow.output_size, generator=g, dtype=dtype),
        embedding=torch.randn(cfg.flow.spk_embed_dim, generator=g, dtype=dtype),
    )


def _tokens(n: int, seed: int) -> torch.Tensor:
    # speech ids above the silent ones, so no token is filtered
    return torch.randint(3, VOCAB, (n,), generator=torch.Generator().manual_seed(seed))


class _EagerReplay:
    """Stands in for a CUDA graph: a replay reruns the captured callable."""

    def __init__(self, run):
        self.replay = run


def _runners(
    node: FlowSubmodule, monkeypatch, device=torch.device("cpu"),
) -> dict[str, PiecewiseAcceleratorGraphRunner]:
    monkeypatch.setattr(
        cgr, "capture_into_graph", lambda run, pool, dev, dtype, keep_graph=False: (_EagerReplay(run), run())
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    # the capture's memory accounting has no CPU backend
    for name in ("memory_allocated", "max_memory_allocated", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.accelerator, name, lambda *args, **kwargs: 0)
    runners = {}
    for label, config in node.get_piecewise_accelerator_graph_configs(device, torch.float32).items():
        runner = PiecewiseAcceleratorGraphRunner(
            label=label, config=config, resources={}, step_runner=mock.Mock(), device=device,
            autocast_dtype=None, num_slots=1,
        )
        runner._graph_backend = SimpleNamespace(synchronize=lambda: None)
        for shape in runner.prepare_for_capture():
            runner._capture_one(shape, 0, forwards=1)
        runners[label] = runner
    return runners


def _replays(runners, monkeypatch) -> list[tuple[str, int]]:
    """(region, real batch size) of every replay, in order."""
    calls: list[tuple[str, int]] = []
    for label, runner in runners.items():
        run = runner.run

        def spy(*args, _label=label, _run=run, **kwargs):
            calls.append((_label, kwargs["real_bs"]))
            return _run(*args, **kwargs)

        monkeypatch.setattr(runner, "run", spy)
    return calls


def _solve(node: FlowSubmodule, cases, runners=None, n_timesteps=None) -> list[torch.Tensor]:
    """One node step over fresh requests: ``cases`` is (voice, tokens, final)."""
    steps = n_timesteps or node.config.flow.cfm.n_timesteps
    rids = list(range(len(cases)))
    out = node._run(
        rids, [t for _, t, _ in cases], [v for v, _, _ in cases], [f for _, _, f in cases], [steps] * len(cases),
        runners,
    )
    for rid in rids:
        node.cleanup_request(rid)
    return [o[MEL][0] for o in out]


# -- padding ---------------------------------------------------------------------------


def test_a_solve_padded_past_its_rows_gives_every_row_its_own_frames():
    cfg = _config()
    flow = _flow(cfg)
    rows = [
        CosyVoice3FlowRow(_voice(cfg, 10, seed=1), _tokens(60, 1), 40, final=False, streaming=True),
        CosyVoice3FlowRow(_voice(cfg, 23, seed=2), _tokens(31, 2), 31, final=True, streaming=True),
        CosyVoice3FlowRow(_voice(cfg, 5, seed=3), _tokens(12, 3), 12, final=True, streaming=False),
    ]
    alone = [flow.solve_rows([row], 3)[0] for row in rows]
    longest = max(flow.valid_frames(row) for row in rows)
    for frames in (longest, longest + 1, longest + 37, 2 * longest):
        inputs = flow.prepare_solve(rows, frames=frames)
        assert inputs.mu.shape[-1] == inputs.cond.shape[-1] == frames
        padded = flow.row_mels(rows, flow.solve_prepared(inputs, 3))
        for mel, ref in zip(padded, alone, strict=True):
            torch.testing.assert_close(mel, ref, rtol=0, atol=1e-10)


def test_a_bucket_shorter_than_a_row_is_refused():
    cfg = _config()
    flow = _flow(cfg)
    row = CosyVoice3FlowRow(_voice(cfg, 10), _tokens(20, 0), 20, final=True, streaming=True)
    with pytest.raises(ValueError, match="cannot hold a row of 60 frames"):
        flow.prepare_solve([row], frames=59)


# -- the node's regions ----------------------------------------------------------------


def test_regions_capture_batch_sizes_within_the_frame_budget():
    # the engine looks up only this name; a misnamed override inherits the base's {}
    assert "get_piecewise_accelerator_graph_configs" in FlowSubmodule.__dict__
    cfg = _config(flow_capture_frame_buckets=(64, 96, 128, 512))
    node = FlowSubmodule(_flow(cfg), None, cfg)
    regions = node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)
    # 512 frames fit no batch size within 256 frames, so it is not a region
    assert {label: r.capture_batch_sizes for label, r in regions.items()} == {
        flow_region(64): [1, 2, 4],
        flow_region(96): [1, 2],
        flow_region(128): [1, 2],
    }
    shape = regions[flow_region(96)].get_capture_shapes([2])[0]
    static = regions[flow_region(96)].make_static_inputs(shape)
    assert {k: (tuple(v.shape), v.dtype) for k, v in static.items()} == {
        "mu": ((2, 80, 96), torch.float64),
        "cond": ((2, 80, 96), torch.float64),
        "spks": ((2, 80), torch.float64),
        "valid": ((2,), torch.long),
        "streaming": ((2,), torch.bool),
    }


def test_batch_sizes_past_the_node_cap_are_not_captured():
    cfg = _config(flow_max_batch_size=2)
    node = FlowSubmodule(_flow(cfg), None, cfg)
    assert node.capture_batch_sizes(64) == [1, 2]


def test_a_bucket_past_the_flow_noise_is_not_a_region():
    cfg = _config(flow_capture_frame_buckets=(64, 96))
    cfg.flow.noise_frames = 80
    node = FlowSubmodule(_flow(cfg), None, cfg)
    assert list(node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)) == [flow_region(64)]


@pytest.mark.parametrize(
    ("prompt_and_lengths", "expected"),
    [
        # 10 + 20 tokens = 60 frames -> the 64 bucket, batch 1
        ([(10, 20)], [(flow_region(64), 1)]),
        # longest row 2 * (10 + 37) = 94 frames -> 96; batch 2
        ([(10, 37), (5, 8)], [(flow_region(96), 2)]),
        # three rows of at most 64 frames replay padded to the batch-4 graph
        ([(10, 20), (6, 9), (12, 18)], [(flow_region(64), 3)]),
        # three rows of 128 frames: the 128 bucket captures batch 1 and 2 only
        ([(10, 54), (10, 54), (10, 54)], []),
        # 2 * (10 + 60) = 140 frames, past every bucket
        ([(10, 60)], []),
    ],
)
def test_node_replays_the_smallest_bucket_that_holds_the_batch(monkeypatch, prompt_and_lengths, expected):
    cfg = _config()
    node = FlowSubmodule(_flow(cfg), None, cfg)
    cases = [
        (_voice(cfg, p, seed=i), _tokens(n, 10 + i), True) for i, (p, n) in enumerate(prompt_and_lengths)
    ]
    eager = _solve(node, cases)
    runners = _runners(node, monkeypatch)
    calls = _replays(runners, monkeypatch)
    replayed = _solve(node, cases, runners)
    assert calls == expected
    for got, ref in zip(replayed, eager, strict=True):
        assert got.shape == ref.shape
        torch.testing.assert_close(got, ref, rtol=0, atol=1e-10)


def test_another_step_count_runs_eagerly(monkeypatch):
    cfg = _config()
    node = FlowSubmodule(_flow(cfg), None, cfg)
    cases = [(_voice(cfg, 10), _tokens(20, 0), True)]
    runners = _runners(node, monkeypatch)
    calls = _replays(runners, monkeypatch)
    eager = _solve(node, cases, n_timesteps=2)
    assert len(_solve(node, cases, runners)) == 1 and calls == [(flow_region(64), 1)]
    torch.testing.assert_close(_solve(node, cases, runners, n_timesteps=2)[0], eager[0], rtol=0, atol=0)
    assert calls == [(flow_region(64), 1)]


def test_streamed_hops_through_the_buckets_equal_one_eager_stream(monkeypatch):
    """A stream settles 40 tokens (the first hop, stretched to end on a chunk),
    then the rest at the final chunk; both solves replay the 128 bucket."""
    cfg = _config()
    tokens, voice = _tokens(47, 5), _voice(cfg, 10)

    def stream(node, runners):
        hops = []
        for n, final in ((43, False), (47, True)):
            out = node._run([0], [tokens[:n]], [voice], [final], [3], runners)
            hops.append(out[0][MEL][0])
        node.cleanup_request(0)
        return torch.cat(hops, dim=-1)

    node = FlowSubmodule(_flow(cfg), None, cfg)
    eager = stream(node, None)
    runners = _runners(node, monkeypatch)
    calls = _replays(runners, monkeypatch)
    replayed = stream(node, runners)
    assert calls == [(flow_region(128), 1), (flow_region(128), 1)]
    assert replayed.shape == eager.shape == (80, 2 * 47)
    torch.testing.assert_close(replayed, eager, rtol=0, atol=1e-10)


def test_a_replays_mel_outlives_the_next_replay(monkeypatch):
    cfg = _config()
    node = FlowSubmodule(_flow(cfg), None, cfg)
    runners = _runners(node, monkeypatch)
    first = _solve(node, [(_voice(cfg, 10, seed=1), _tokens(20, 1), True)], runners)[0]
    kept = first.clone()
    _solve(node, [(_voice(cfg, 12, seed=2), _tokens(18, 2), True)], runners)
    assert torch.equal(first, kept)


# -- CUDA ------------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
@pytest.mark.parametrize("estimator_dtype", [torch.bfloat16, torch.float32])
def test_captured_solve_equals_the_eager_solve_of_the_same_padded_inputs(estimator_dtype):
    """Real capture and replay: the same kernels on the same shapes give the
    same bits; padding a row to its bucket stays within the estimator's
    precision of its exact-length solve."""
    device = torch.device("cuda", torch.cuda.current_device())
    # every bucket captures batch 1, 2 and 4
    cfg = _config(flow_capture_batch_sizes=(1, 2, 4), flow_capture_batch_frames=512)
    cfg.flow.cfm.n_timesteps = 10
    flow = _flow(cfg, dtype=torch.float32).to(device)
    if estimator_dtype != torch.float32:
        flow.set_estimator_dtype(estimator_dtype)
    node = FlowSubmodule(flow, None, cfg)
    pool = get_accelerator_graph_backend(device).graph_pool_handle()
    runners = {}
    for label, config in node.get_piecewise_accelerator_graph_configs(device, torch.float32).items():
        runner = PiecewiseAcceleratorGraphRunner(
            label=label, config=config, resources={}, step_runner=mock.Mock(), device=device,
            autocast_dtype=None, num_slots=1, memory_pool=pool,
        )
        runner.warmup_and_capture(None)  # no plan: every bucket
        assert runner.any_graphs and not runner.dropped_shapes
        runners[label] = runner

    rows = [
        CosyVoice3FlowRow(_voice(cfg, 10, seed=1, dtype=torch.float32), _tokens(60, 1).to(device), 40, final=False,
                streaming=True),
        CosyVoice3FlowRow(_voice(cfg, 7, seed=2, dtype=torch.float32), _tokens(31, 2).to(device), 31, final=True,
                streaming=True),
        CosyVoice3FlowRow(_voice(cfg, 5, seed=3, dtype=torch.float32), _tokens(12, 3).to(device), 12, final=True,
                streaming=False),
    ]
    # 100 frames (bucket 128, batch 1); 76 and 34 (bucket 96, batch 2); all
    # three (bucket 128, padded to batch 4)
    for batch in (rows[:1], rows[1:], rows):
        frames = max(flow.valid_frames(row) for row in batch)
        bucket = next(b for b in cfg.flow_capture_frame_buckets if b >= frames)
        inputs = flow.prepare_solve(batch, frames=bucket)
        runner = runners[flow_region(bucket)]
        assert runner.can_run(len(batch))
        # replays of other buckets share the pool and must not disturb this one
        for other in runners.values():
            if other is not runner:
                short = flow.prepare_solve(rows[2:], frames=other._config.seq_len)
                other.run(static_inputs=short._asdict(), real_bs=1)
        replayed = runner.run(static_inputs=inputs._asdict(), real_bs=len(batch))["mel"]
        with node_precision():
            eager = flow.solve_prepared(inputs, 10)
            exact = flow.solve_rows(batch, 10)
        if len(batch) == runner._padded_bs(len(batch)):
            assert torch.equal(replayed, eager)
        else:
            # a padding row changes the batch shape, so other kernels may run
            torch.testing.assert_close(replayed, eager, rtol=0, atol=_tolerance(estimator_dtype))
        for got, ref in zip(flow.row_mels(batch, replayed), exact, strict=True):
            torch.testing.assert_close(got, ref, rtol=0, atol=_tolerance(estimator_dtype))


def node_precision():
    from mstar.model.cosyvoice3.submodules import _without_tf32

    return _without_tf32()


def _tolerance(dtype: torch.dtype) -> float:
    # random weights at 0.2 make a loud estimator; another kernel's rounding
    # moves a frame by a few hundredths of its ~unit scale after ten guided
    # steps in bf16, and by a few thousandths in float32
    return 0.25 if dtype == torch.bfloat16 else 1e-2


# -- turning capture off per deployment --------------------------------------------------


def _model(tmp_path, monkeypatch, **kwargs):
    from mstar.model.cosyvoice3.cosyvoice3_model import CosyVoice3Model

    monkeypatch.setattr(CosyVoice3Model, "_build_text_tokenizer", lambda self: None)
    return CosyVoice3Model(model_path_hf=str(tmp_path), **kwargs)


def test_capture_fields_come_from_model_kwargs(tmp_path, monkeypatch):
    default = CosyVoice3Config()
    assert _model(tmp_path, monkeypatch).config.flow_capture_frame_buckets == default.flow_capture_frame_buckets
    cfg = _model(
        tmp_path, monkeypatch,
        flow_capture_frame_buckets=[300, 100, 100, 200.0], flow_capture_batch_sizes=[4, 1, 2],
        flow_capture_batch_frames=1024,
    ).config
    assert cfg.flow_capture_frame_buckets == (100, 200, 300)
    assert cfg.flow_capture_batch_sizes == (1, 2, 4) and cfg.flow_capture_batch_frames == 1024
    for bad in ({"flow_capture_frame_buckets": [0]}, {"flow_capture_frame_buckets": "100"},
                {"flow_capture_batch_sizes": [1.5]}, {"flow_capture_batch_frames": -1}):
        with pytest.raises(ValueError, match="flow_capture"):
            _model(tmp_path, monkeypatch, **bad)


@pytest.mark.parametrize("kwargs", [{"flow_capture_frame_buckets": []}, {"flow_capture_batch_frames": 0}],
                         ids=["no-buckets", "no-frame-budget"])
def test_a_deployment_can_turn_capture_off(tmp_path, monkeypatch, kwargs):
    """No region is captured, and a solve that the default would replay runs
    eagerly to the same mel."""
    model_cfg = _model(tmp_path, monkeypatch, **kwargs).config
    on_cfg = _config()
    off_cfg = _config(flow_capture_frame_buckets=model_cfg.flow_capture_frame_buckets,
                      flow_capture_batch_frames=model_cfg.flow_capture_batch_frames)
    on, off = FlowSubmodule(_flow(on_cfg), None, on_cfg), FlowSubmodule(_flow(off_cfg), None, off_cfg)
    cases = [(_voice(on_cfg, 10), _tokens(20, 0), True)]
    on_runners, off_runners = _runners(on, monkeypatch), _runners(off, monkeypatch)
    assert off_runners == {} and list(on_runners)
    calls = _replays(on_runners, monkeypatch)
    replayed = _solve(on, cases, on_runners)
    assert calls == [(flow_region(64), 1)]
    torch.testing.assert_close(_solve(off, cases, off_runners)[0], replayed[0], rtol=0, atol=1e-10)
