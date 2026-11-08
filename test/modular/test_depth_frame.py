"""The Qwen3-TTS 1.7B depth loop as one kernel (``DepthFrame``) against the loop it replaces.

The task lists put the two halves of a split O or DOWN tile on the two CTAs of one cluster, which
add their partial sums through each other's shared memory; a half dealt to any other CTA adds the
wrong rows or never arrives. On a Hopper GPU, the frame with given codes has to write the loop's
codes and embedding sums, and a KV cache and logits within twice what the loop's own choice of
tiles moves them. Its draws have to be ``fused_sample``'s.
"""
from collections import defaultdict

import pytest
import torch

from mstar.engine.state import Sampler
from mstar.engine.state.calls import SamplerHandle
from mstar.model.qwen3_tts.components.depth_frame import schedule
from mstar.model.qwen3_tts.components.depth_frame.schedule import NPAGES, TB

GRID = 132
requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="needs a Hopper GPU",
)


def _tasks():
    tasks, ntasks, _ = schedule.build(schedule.frame_phases(), GRID, NPAGES, TB)
    return [[[int(v) for v in tasks[c, i]] for i in range(ntasks[c])] for c in range(GRID)]


def test_a_split_tile_s_halves_run_on_the_two_ctas_of_one_cluster():
    lists = _tasks()
    halves = {}
    for c, tasks in enumerate(lists):
        for d in tasks:
            if d[0] in (schedule.T_O, schedule.T_DOWN):
                halves[(d[0], d[1], d[8], d[2])] = c
    assert halves, "the frame has no split phases"
    for (ty, layer, pos, tile2), cta in halves.items():
        assert cta % 2 == tile2 % 2, f"rank {tile2 % 2} of a split tile runs on CTA {cta}"
        assert halves[(ty, layer, pos, tile2 ^ 1)] == cta ^ 1, "a split tile's halves run on two clusters"


def test_each_phase_waits_for_every_signal_of_the_one_before():
    signals, waits = defaultdict(int), set()
    for tasks in _tasks():
        for d in tasks:
            signals[d[5]] += 1
            waits.add((d[3], d[4]))
    for counter, target in waits:
        if target:
            assert signals[counter] == target, f"counter {counter}: {signals[counter]} signals for a wait on {target}"


def test_a_grid_that_would_deal_a_cluster_two_split_tiles_is_refused():
    with pytest.raises(AssertionError, match="combine buffer"):
        schedule.build(schedule.frame_phases(), 114, NPAGES, TB)


def test_no_weight_tile_lands_on_pages_an_unfinished_task_still_holds():
    pages = {op.type: op.pages for op in (schedule.QKV, schedule.O, schedule.GATEUP, schedule.DOWN, schedule.INPROJ,
                                          schedule.HEAD)}
    for tasks in _tasks():
        # task j's tile is issued just before the wait of task issue_at[j]; d[7] counts the tasks issued there
        issue_at = [i for i, d in enumerate(tasks) for _ in range(d[7])]
        assert len(issue_at) == len(tasks), "a task's tile is never issued"
        weighted = [j for j, d in enumerate(tasks) if d[0] in pages]
        held = {j: {(tasks[j][6] + q) % NPAGES for q in range(pages[tasks[j][0]])} for j in weighted}
        for j in weighted:
            live = [k for k in weighted if k < j and held[k] & held[j] and k >= issue_at[j]]
            assert not live, f"task {j}'s tile overwrites pages tasks {live} still read"


class _StubSampler:
    """Hands out ``draw`` and records the offsets advanced instead of sampling."""

    def __init__(self, draw):
        self.draw, self.advanced = draw, []

    def fused_draw_args(self):
        return self.draw

    def advance_offsets(self, draws):
        self.advanced.append(draws)


class _StubFrame:
    """Records its launches instead of running the kernel."""

    max_rows = 64
    kv_cache = depth_positions = partials = None

    def __init__(self):
        self.runs = 0

    def run(self, *args):
        self.runs += 1


def _talker(frame):
    """The `TalkerSubmodule` state `_run_depth` reads, around ``frame``, its loop replaced by a marker."""
    from types import SimpleNamespace

    from mstar.model.qwen3_tts.submodules import TalkerSubmodule

    talker = TalkerSubmodule.__new__(TalkerSubmodule)
    talker._depth_frame = frame
    talker.code_predictor = SimpleNamespace(num_code_groups=16)
    talker.model = SimpleNamespace(model=SimpleNamespace(codec_embedding=SimpleNamespace(weight=torch.zeros(3072, 8))))
    talker._depth_loop = lambda *args: "loop"
    return talker


def _run_depth(talker, sampler, rows=3):
    talker.cp_sampler = SamplerHandle(Sampler("code_predictor", nodes={"Talker"}, vocab_size=2048, draws=15), sampler)
    talker.cp_sampler._open = True
    return talker._run_depth(torch.zeros(rows, 2048), torch.zeros(rows, dtype=torch.long))


def test_a_step_the_frame_can_draw_runs_the_frame_and_moves_the_offsets_past_its_draws():
    frame, sampler = _StubFrame(), _StubSampler(draw=object())
    all_codes, embed_sum = _run_depth(_talker(frame), sampler)
    assert frame.runs == 1 and all_codes.shape == (3, 16) and embed_sum.shape == (3, 8)
    assert sampler.advanced == [15], "the next step would draw from this step's random numbers"


def test_a_step_the_frame_cannot_draw_runs_the_loop():
    frame = _StubFrame()
    assert _run_depth(_talker(frame), _StubSampler(draw=None)) == "loop"
    assert frame.runs == 0, "the frame ran on a step whose draws it cannot reproduce"


def test_a_step_with_more_rows_than_the_frame_holds_runs_the_loop():
    frame = _StubFrame()
    assert _run_depth(_talker(frame), _StubSampler(draw=object()), rows=frame.max_rows + 1) == "loop"
    assert frame.runs == 0, "the frame ran past its row scratch"


def test_without_the_frame_the_loop_runs():
    assert _run_depth(_talker(None), _StubSampler(draw=object())) == "loop"


def _depth_loop(predictor, talker_table, hidden, codes):
    """`ResidualCodePredictor.depth_loop` with every group's code given. Returns its codes and embedding sum, the
    logits group 15 was drawn from, and its KV cache."""
    logits_seen = []

    def given(logits):
        logits_seen.append(logits.clone())
        return codes[:, len(logits_seen)].contiguous()

    all_codes, embed_sum = predictor.depth_loop(hidden, codes[:, 0].contiguous(), talker_table[codes[:, 0]], given)
    return all_codes, embed_sum, logits_seen[-1], predictor.depth.kv_cache[:, :codes.shape[0]]


def _predictor():
    from mstar.model.components.code_predictor_kernels import ResidualCodePredictor
    from mstar.model.qwen3_tts.config import Qwen3TTSModelConfig, Qwen3TTSTalkerConfig

    torch.manual_seed(0)
    config = Qwen3TTSModelConfig(talker=Qwen3TTSTalkerConfig(hidden_size=2048))
    predictor = ResidualCodePredictor(config.code_predictor, 2048, 128, embed_sum_dtype=None)
    predictor = predictor.to("cuda", torch.bfloat16).eval()
    for name, param in predictor.named_parameters():
        if name.endswith("norm.weight"):
            param.data.uniform_(0.5, 1.5)
        else:
            param.data.normal_(std=0.02)
    talker_table = torch.randn(config.talker.vocab_size, 2048, device="cuda", dtype=torch.bfloat16)
    predictor.consolidate_stacked_weights()
    predictor.depth.make(torch.bfloat16, "cuda")
    return predictor, talker_table


@pytest.fixture(scope="module")
def _frame():
    from mstar.model.qwen3_tts.components.depth_frame.frame import DepthFrame

    predictor, talker_table = _predictor()
    return predictor, talker_table, DepthFrame(predictor, talker_table)


@requires_hopper
@pytest.mark.parametrize("bs", [1, 13, 64])
def test_the_frame_with_given_codes_matches_the_depth_loop(_frame, bs):
    """The same frame at every size: a row count specialized at its first launch would break the later sizes."""
    predictor, talker_table, frame = _frame
    hidden = torch.randn(bs, 2048, device="cuda", dtype=torch.bfloat16)
    codes = torch.randint(0, 2048, (bs, 16), device="cuda")
    codes[:, 0] = torch.randint(0, talker_table.shape[0], (bs,), device="cuda")
    with torch.no_grad():
        codes_want, sum_want, logits_want, kv_want = _depth_loop(predictor, talker_table, hidden, codes)
    runs = []
    for _ in range(2):
        all_codes, embed_sum = torch.empty_like(codes), torch.empty_like(sum_want)
        frame.run(hidden, codes, frame.kv_cache, frame.depth_positions, frame.partials, all_codes, embed_sum, None)
        runs.append((all_codes, embed_sum, frame.kv_cache[:, :bs].clone(), frame.LOGITS[:bs].clone()))
    all_codes, embed_sum, kv, logits = runs[0]
    assert torch.equal(all_codes, codes_want) and torch.equal(embed_sum, sum_want), "given codes summed differently"
    # The loop itself moves by up to 1.1% of the largest KV value and 0.8% of the largest logit between a step of
    # these rows alone and a 64-row step, where `cp_linear` picks other tiles. The frame is held to under twice that.
    torch.testing.assert_close(kv, kv_want, rtol=0, atol=0.02 * kv_want.abs().max().item())
    torch.testing.assert_close(logits, logits_want, rtol=0, atol=0.015 * logits_want.abs().max().item())
    assert all(torch.equal(a, b) for a, b in zip(runs[0], runs[1], strict=True)), "the frame does not repeat itself"


def _draw(bs: int, offset: int):
    from mstar.engine.state import FusedDrawArgs

    return FusedDrawArgs(
        torch.full((bs,), 0.9, device="cuda"), torch.full((bs,), 50, device="cuda", dtype=torch.int32),
        torch.ones(bs, device="cuda"), torch.arange(bs, device="cuda"),
        torch.full((bs,), offset, device="cuda", dtype=torch.long),
    )


@requires_hopper
def test_a_sampled_frame_draws_its_last_group_from_its_own_logits_at_the_last_offset(_frame):
    """The sampled frame is the given-codes frame with `fused_sample`'s draws: fed back its own codes, the frame has
    to sum the same embeddings and leave the logits group 15 was drawn from."""
    from mstar.engine.resources.sampler.utils import fused_sample

    predictor, talker_table, frame = _frame
    bs = 13
    hidden = torch.randn(bs, 2048, device="cuda", dtype=torch.bfloat16)
    layer0 = torch.randint(0, talker_table.shape[0], (bs,), device="cuda")
    draw = _draw(bs, offset=5)
    kv, positions, partials = frame.kv_cache, frame.depth_positions, frame.partials
    codes, embed_sum = torch.empty(bs, 16, device="cuda", dtype=torch.long), torch.empty_like(hidden)
    frame.run(hidden, layer0, kv, positions, partials, codes, embed_sum, draw)
    given_sum = torch.empty_like(embed_sum)
    frame.run(hidden, codes, kv, positions, partials, torch.empty_like(codes), given_sum, None)
    assert torch.equal(given_sum, embed_sum), "the sampled frame embedded other codes than the ones it drew"
    last = fused_sample(frame.LOGITS[:bs], draw.temperature, draw.top_k, draw.top_p, draw.seed, draw.offset + 14)
    assert torch.equal(codes[:, 15], last), "group 15 was drawn from other logits or at another offset"


@requires_hopper
def test_a_captured_frame_replays_what_it_ran(_frame):
    """The engine captures the Talker step with the frame's cluster launch inside it."""
    predictor, talker_table, frame = _frame
    bs = 13
    hidden = torch.randn(bs, 2048, device="cuda", dtype=torch.bfloat16)
    layer0 = torch.randint(0, talker_table.shape[0], (bs,), device="cuda")
    positions, partials = frame.depth_positions, frame.partials
    draw = _draw(bs, offset=5)

    def outputs():
        return (torch.zeros(5, bs, 2, 16, 8, 128, device="cuda", dtype=torch.bfloat16),
                torch.empty(bs, 16, device="cuda", dtype=torch.long), torch.empty_like(hidden))

    ran, replayed = outputs(), outputs()
    frame.run(hidden, layer0, ran[0], positions, partials, ran[1], ran[2], draw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        frame.run(hidden, layer0, replayed[0], positions, partials, replayed[1], replayed[2], draw)
    graph.replay()
    assert all(torch.equal(a, b) for a, b in zip(ran, replayed, strict=True)), "the replay ran another frame"


@requires_hopper
def test_a_frame_draws_through_the_contract_as_the_sampler_would(_frame):
    """`fused_draw_args` hands the kernel a real sampler's buffers and `drew` moves its offsets, so the kernel's last
    group is `fused_sample`'s at the last offset, and the next step starts past all fifteen."""
    from mstar.engine.resources.sampler.resource import SamplerResource
    from mstar.engine.resources.sampler.utils import SamplerBuffers, SamplingConfig, fused_sample
    from mstar.engine.state import drew, fused_draw_args

    predictor, talker_table, frame = _frame
    rows = [f"r{i}" for i in range(13)]
    bufs = SamplerBuffers.allocate(64, torch.device("cuda"), cg_slots=1)
    for i, rid in enumerate(rows):
        config = SamplingConfig(temperature=0.9, top_k=50, top_p=1.0)
        config.set_seed(i)
        bufs.register_request(rid, config)
    bufs.gather_static(rows, 13, 0)
    bufs.gather_dynamic(rows, 13, 0)
    resource = SamplerResource(vocab_size=2048, enable_repetion_penalty=False, device=torch.device("cuda"))
    resource._cg_sampler = bufs.sampler_for(13, 0)
    handle = SamplerHandle(Sampler("code_predictor", nodes={"Talker"}, vocab_size=2048, draws=15), resource)
    handle._open = True

    hidden = torch.randn(13, 2048, device="cuda", dtype=torch.bfloat16)
    layer0 = torch.randint(0, talker_table.shape[0], (13,), device="cuda")
    codes, embed_sum = torch.empty(13, 16, device="cuda", dtype=torch.long), torch.empty_like(hidden)
    draw = fused_draw_args(handle)
    offsets = draw.offset.clone()
    frame.run(hidden, layer0, frame.kv_cache, frame.depth_positions, frame.partials, codes, embed_sum, draw)
    drew(handle, 15)
    last = fused_sample(frame.LOGITS[:13], draw.temperature, draw.top_k, draw.top_p, draw.seed, offsets + 14)
    assert torch.equal(codes[:, 15], last), "the kernel drew other tokens than the sampler's own draw"
    assert torch.equal(draw.offset, offsets + 15), "the next step would draw from this step's random numbers"
