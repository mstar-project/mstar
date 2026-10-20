"""GR00T N1.7's whole ``action_gen`` loop on the GPU, against a same-stack oracle.

``test_gr00t_n17_dit_gpu.py`` pins **one** DT forward against NVIDIA's ``AlternateVLDiT``.
This file pins the loop around it: four flow-matching iterations of
``Gr00tN17DenoiseStepSubmodule.denoise`` (``submodules.py:657``) -- action encoder, position
table, the 32-block DT, action decoder, Euler update -- from the seeded noise to the served
action chunk, and then through ``Gr00tN17Preprocessor.unapply`` to the joint values a robot
would actually be sent. Everything between the bridge and the gripper, in one number per
request.

**The same-stack oracle.** The answer is not an inline reference here but NVIDIA's own
``gr00t`` code *run on M\\*'s stack*: the same torch, the same CUDA, the same SDPA kernels,
recorded per request under ``~/gr00t-runs/oracle_fp32`` and ``~/gr00t-runs/oracle_bf16``. Two
runs of the same oracle, so the bf16 one against the fp32 one is this stage's **bf16 floor**,
measured here rather than written down, and the T3 bar is mean abs error within 1.2x of it.
``~/gr00t-runs/dump`` is a third recording, from NVIDIA's *own* venv, and only its inputs are
read: ``input_ids`` (which rows are the image), ``embodiment_id``, the embodiment tag and the
raw state the decode composes onto.

Five things are measured, and two are deliberately broken:

* **R11, the served chunk.** All 20 recorded DROID requests, ours in bf16 against the fp32
  oracle, at the T3 bar. This is the only bar in the port that reads the whole loop at once:
  a wrong position table, a dropped Euler term or an off-by-one in the schedule all land here.
* **R11, the decoded actions.** The same 20 chunks, ours and both oracles, each through
  ``unapply`` and compared **per joint group**. The 132-wide chunk is mostly padding, so a
  group-sized error is a few percent of the chunk and invisible in the bar above; after the
  decode it is the whole of one joint.
* **The bucket count.** The 20 requests occupy exactly **two** buckets by (text rows, image
  rows). A batch in this engine shares its span lengths, so the bucket count is what the
  served deployment pays for, and it is a stated C08 bar.
* **R4(b), batching.** Two same-shaped requests batched against each run alone.
* **R10, mixed loop iterations.** One shared forward with each row at its own iteration index,
  against each row run alone -- what the scaffold's loop does whenever two requests arrive a
  step apart.
* **The two controls.** Hand the text blocks the image rows (a span-assignment bug, which has
  to sit orders above the batching gap), and perturb one row's conditioning (after which the
  *other* row has to stay bit-identical).

Every span runs on **SDPA**. No attention resource is bound, so ``attends()`` returns nothing
and every block falls to ``_SDPA_ATTEND`` (``components/dit.py:334``) -- the sanctioned parity
backend, which ``test_gr00t_n17_dit_gpu.py`` showed matches FlashInfer's ragged kernels at the
floor. The ragged path is **not** exercised here; that file is where it is tested, and this one
would confound a kernel difference with a loop difference if it used them.

Three traps are worth naming, because each produces a test that passes and says nothing:

1. **The conditioning must come from the oracle, not the dump.** Both recordings carry
   ``vl_embeds``, and they differ by mean abs 8.4e-02 (median; 7.3e-02 to 1.1e-01 over the 20)
   because the backbone ran on two different stacks -- about the size of the oracle's own
   bf16 rounding of the same tensor. Conditioning ours on the dump's rows and scoring it
   against the oracle's answer still passes (median 0.98x, max 1.09x of the floor), which is
   the danger: the ratio stops being a reading of *our* loop and becomes a reading of the gap
   between two backbones, with a quarter of the headroom to the bar spent on it. So ours is handed
   the **bf16 oracle's** own ``vl_embeds`` and ``state_features`` -- exactly what the bf16
   oracle's action head was handed -- and only then is the ratio the head's arithmetic.
2. **The batch-invariance bars are ratios to a floor measured in this test.** A shared GEMM
   re-blocks with the row count, so the same request run at M=2 does not return the bits it
   returns at M=1, and ``torch.equal`` on a batched-vs-alone comparison would be flaky at the
   kernels' discretion. An absolute tolerance instead would rot in silence the first time
   cuBLAS picks a different tile. So the floor is measured by running the **same** request
   twice in one batch against it alone: both rows are identical, so what is left is purely
   the re-blocking, and the bar is a ratio to that.
3. **One copy of the weights.** The denoise step is 2.55 GiB in bf16 against 12 GiB of device
   memory, so it is built once for the whole session and every test shares it. Loading it per
   test would still fit, but the file would spend a minute of its runtime on the loader.

Running it needs CUDA and the recordings, and nothing else: it is SDPA-only, so unlike
``test_gr00t_n17_dit_gpu.py`` it does **not** need the CUDA shim for FlashInfer's JIT, and
plain ``pytest`` is enough. The invocation the suite is otherwise run with does no harm::

    CUDA_HOME=$HOME/.local/cuda-home \\
    PATH=$HOME/mstar/.venv/bin:$HOME/.local/cuda-home/bin:$PATH \\
    pytest test/modular/test_gr00t_n17_action_gen_gpu.py

Not a CPU-core test: it needs a GPU, the cached checkpoint and ~420 MB of recordings that are
not in the repo, and it is skipped without any of them.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from mstar.model.gr00t_n17.components.dit import ActionPositionEmbedding, DiT
from mstar.model.gr00t_n17.components.embodiment import build_action_decoder, build_action_encoder
from mstar.model.gr00t_n17.config import Gr00tN17Config
from mstar.model.gr00t_n17.preprocessing import Gr00tN17Preprocessor
from mstar.model.gr00t_n17.submodules import Gr00tN17DenoiseStepSubmodule
from mstar.model.gr00t_n17.weight_loader import load_gr00t_n17_weights

_HUB = Path.home() / ".cache/huggingface/hub"
_GR00T = _HUB / "models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495"
_BACKBONE = next(iter(glob.glob(str(_HUB / "models--nvidia--Cosmos-Reason2-2B/snapshots/*"))), None)

# The recordings. ~420 MB in all and outside the repo, so these tests gate on them the way
# `test_gr00t_n17_preprocessing.py`'s decode-parity test does.
_RUNS = Path.home() / "gr00t-runs"
_ORACLE_FP32, _ORACLE_BF16, _DUMP = _RUNS / "oracle_fp32", _RUNS / "oracle_bf16", _RUNS / "dump"
_REQUESTS = 20


def _recorded() -> bool:
    """Whether all three recordings are complete. A partial set would silently shrink the
    sample the bars are measured over, so the count is checked rather than the directory."""
    return all(
        d.is_dir() and len(list(d.glob("req_*.pt"))) == _REQUESTS
        for d in (_ORACLE_FP32, _ORACLE_BF16, _DUMP)
    )


pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and _GR00T.exists() and bool(_BACKBONE) and _recorded()),
    reason=(
        f"the full action_gen loop needs CUDA, the cached checkpoint and the {_REQUESTS} recorded "
        f"requests in {_RUNS}/{{oracle_fp32,oracle_bf16,dump}} (see docs-local/c08/qa/)"
    ),
)

_DEVICE = torch.device("cuda:0")
_BF16 = torch.bfloat16

# The T3 bar: a multiple of the floor measured in the test, never an absolute tolerance.
_T3 = 1.2
# What the 20 requests' spans are, and that there are only two of them (a C08 bar). Spelled
# out rather than counted into a variable: these are the numbers the test exists to catch
# changing. The 448 image rows are the four recorded frames (DROID's two cameras) at a
# 1 x 16 x 28 patch grid merged by 2, so 112 rows each; only the prompt length varies.
_BUCKETS = {(13, 448): 8, (21, 448): 12}
# The denoise step's own tensors: action encoder, position table, the 32 DT blocks, decoder.
# `load_gr00t_n17_weights` already raises on a parameter that got none; this says the node is
# the whole step and not the DT alone (456 of these are the DT's, which the CPU file counts).
_STEP_TENSORS = 467
# GR00T's four steps, read off `schedule_for` (`submodules.py:496`). Sigma *ascends* here --
# it is GR00T's time, not a noise level -- and the recordings were made against it.
_SIGMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
_TIMESTEPS = (0.0, 250.0, 500.0, 750.0)
# The rms of the oracle's own answer, which says the weights are the trained ones: measured
# 1.057 to 1.106 over the 20. The same guard as the DT file's `_OUTPUT_RMS`, and for the same
# reason -- inside a floor set by untrained weights every bar below passes on anything.
_ACTION_RMS = (0.9, 1.3)
# R10's two iteration indices: one row mid-loop and one row at the last step, so neither the
# timestep nor the sigma pair is shared and a row reading its neighbour's index is visible.
_MIXED_ITERS = (1, 3)
# How far above the batching floor the span-assignment control has to sit. Hand-chosen with
# headroom; measured 26x. A reviewer should ask whether 10x is the right floor to demand --
# the point of the number is only that a wiring bug and a re-blocking difference are not the
# same order, and the measurement says they are apart by more than one.
_CONTROL_MARGIN = 10


# ----------------------------------------------------------------------------
# One recorded request: its conditioning, its answer, and its floor
# ----------------------------------------------------------------------------


@dataclass
class _Request:
    """One recorded DROID request. The conditioning is on the device in bf16, because that is
    how the oracle's head received it; the two answers stay on the host in fp32, because the
    decode runs in numpy and the comparison is a handful of scalars."""

    index: int
    text: torch.Tensor  # [1, n_text, 2048], the bridge's non-image rows
    image: torch.Tensor  # [1, n_image, 2048]
    state: torch.Tensor  # [1, 1, 1536], the state encoder's one row
    embodiment_id: torch.Tensor  # [1]
    noise: torch.Tensor  # [1, 40, 132], the oracle's seeded trajectory
    noise_dump: torch.Tensor  # ... and NVIDIA's own venv's, which must be the same bits
    want: torch.Tensor  # [1, 40, 132] fp32: the oracle in fp32, the answer
    rounded: torch.Tensor  # ... and in bf16, which sets the floor
    tag: str  # the embodiment tag `unapply` keys its statistics by
    raw_state: dict[str, np.ndarray]  # the raw state the relative groups compose onto

    @property
    def bucket(self) -> tuple[int, int]:
        return (self.text.shape[1], self.image.shape[1])

    @property
    def floor(self) -> float:
        """This request's bf16 floor: the oracle in bf16 against itself in fp32."""
        return (self.rounded - self.want).abs().mean().item()


def _load(config: Gr00tN17Config, index: int) -> _Request:
    """One request, with its conditioning taken from the **bf16** oracle (trap 1 in the
    banner) and the image rows picked out by the dump's ``input_ids``.

    Which rows are the image is the reference's own predicate, ``input_ids == image_token_id``
    (``gr00t/model/modules/qwen3_backbone.py:364``), and ``test_gr00t_n17_bridge.py`` is where it is pinned against
    the reference. Here it is simply applied, to split the one recorded ``vl_embeds`` tensor
    into the two spans the DT's cross blocks read.
    """
    oracle32 = torch.load(_ORACLE_FP32 / f"req_{index:02d}.pt", map_location="cpu", weights_only=False)
    oracle16 = torch.load(_ORACLE_BF16 / f"req_{index:02d}.pt", map_location="cpu", weights_only=False)
    dump = torch.load(_DUMP / f"req_{index:02d}.pt", map_location="cpu", weights_only=False)

    is_image = dump["collated"]["inputs"]["input_ids"][0] == config.text.image_token_id
    embeds = oracle16["vl_embeds"].to(_DEVICE, _BF16)[0]
    return _Request(
        index=index,
        text=embeds[~is_image].unsqueeze(0),
        image=embeds[is_image].unsqueeze(0),
        state=oracle16["state_features"].to(_DEVICE, _BF16),
        embodiment_id=dump["collated"]["inputs"]["embodiment_id"].to(_DEVICE),
        noise=oracle16["noise"].to(_DEVICE, _BF16),
        noise_dump=dump["noise"],
        want=oracle32["action_pred"].float(),
        rounded=oracle16["action_pred"].float(),
        tag=dump["embodiment_tag"],
        raw_state=dump["raw_obs"]["state"],
    )


# ----------------------------------------------------------------------------
# The one build: the denoise step, its schedule, and the decode
# ----------------------------------------------------------------------------


@dataclass
class _Loop:
    config: Gr00tN17Config
    submodule: Gr00tN17DenoiseStepSubmodule
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    preprocessor: Gr00tN17Preprocessor


@pytest.fixture(scope="session")
def loop() -> _Loop:
    """The denoise step at the checkpoint's weights, in bf16 on the device, built once.

    Session-scoped for the reason in trap 3: 2.55 GiB on a 12 GiB card, and the loader is the
    slowest thing in the file. The schedule is asserted here rather than in a test of its own
    because every bar below is measured against it: a schedule that stopped being GR00T's
    would move every number in the file at once, and the recordings would no longer be the
    answer to what is being run.
    """
    config = Gr00tN17Config.from_hf(_GR00T, _BACKBONE)
    submodule = Gr00tN17DenoiseStepSubmodule(
        config,
        build_action_encoder(config),
        ActionPositionEmbedding(config),
        DiT(config),
        build_action_decoder(config),
        loop_name="denoise_loop",
    )
    loaded = load_gr00t_n17_weights(submodule, "denoise_step", _GR00T, device=_DEVICE)
    assert len(loaded) == _STEP_TENSORS, len(loaded)
    submodule.to(_DEVICE).to(_BF16).eval()
    torch.cuda.empty_cache()

    # `schedule_for` reads neither argument (`submodules.py:496`): the steps are the config's
    # and the bucket key does not change them.
    schedule = submodule.schedule_for(None, None)
    assert tuple(schedule.sigmas.tolist()) == _SIGMAS
    assert tuple(schedule.timesteps.tolist()) == _TIMESTEPS
    yield _Loop(
        config=config,
        submodule=submodule,
        sigmas=schedule.sigmas.to(_DEVICE),
        timesteps=schedule.timesteps.to(_DEVICE),
        # The decode needs only `statistics.json`, so no backbone path: without it the
        # text/image processor is not built, and `unapply` does not use one.
        preprocessor=Gr00tN17Preprocessor.from_pretrained(
            _GR00T, max_state_dim=config.max_state_dim, max_action_horizon=config.action_horizon,
        ),
    )
    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def requests(loop: _Loop) -> list[_Request]:
    """All 20 recordings, loaded once. The conditioning is ~2 MB of device memory per request
    against the step's 2.55 GiB, and re-reading the dumps per test would cost more than the
    forwards do."""
    return [_load(loop.config, index) for index in range(_REQUESTS)]


@pytest.fixture(scope="session")
def pair(requests: list[_Request]) -> tuple[_Request, _Request]:
    """Two **different** requests that share a bucket, which is the only kind of pair this
    engine batches: ``joint_attention`` packs one segment per request and every segment in a
    plan shares its length, so two span lengths are two buckets and two forwards (D07)."""
    by_bucket: dict[tuple[int, int], list[_Request]] = {}
    for request in requests:
        by_bucket.setdefault(request.bucket, []).append(request)
    shared = next(group for group in by_bucket.values() if len(group) >= 2)
    return shared[0], shared[1]


# ----------------------------------------------------------------------------
# Driving the loop
# ----------------------------------------------------------------------------


def _run(loop: _Loop, batch: list[_Request], iters: tuple[int, ...] | None = None) -> torch.Tensor:
    """``action_gen``'s loop over a batch of same-shaped requests: ``[B, 40, 132]`` in fp32.

    ``denoise`` is called with ``engine_inputs=None`` and ``bucket_key=None`` because it reads
    neither (``submodules.py:657``) -- it is the region a graph captures, so everything
    host-side was already spent in ``request_inputs``. What the scaffold does *around* the
    step (the iteration counter, the seeded noise, the buckets, the overshoot veto) is
    ``DenoiseLoopSubmodule``'s and is tested on the CPU; what is driven here is the step.

    ``iters=None`` is the real loop: four steps, each row at the same index. A tuple instead
    gives each row **its own** index in one shared forward, which is R10 -- the scaffold does
    that whenever two requests reach the node a step apart.
    """
    latents = torch.cat([request.noise for request in batch])
    text = torch.cat([request.text for request in batch])
    image = torch.cat([request.image for request in batch])
    state = torch.cat([request.state for request in batch])
    embodiment_id = torch.cat([request.embodiment_id for request in batch])
    size = len(batch)

    def step(latents: torch.Tensor, timestep, sigma, sigma_next) -> torch.Tensor:
        with torch.no_grad():
            return loop.submodule.denoise(
                None, None, latents, timestep, sigma, sigma_next, text, image, state, embodiment_id,
            )

    if iters is None:
        for k in range(loop.config.num_inference_timesteps):
            latents = step(
                latents, loop.timesteps[k].expand(size), loop.sigmas[k].expand(size),
                loop.sigmas[k + 1].expand(size),
            )
        return latents.float()
    ks = torch.tensor(iters, device=_DEVICE)
    return step(latents, loop.timesteps[ks], loop.sigmas[ks], loop.sigmas[ks + 1]).float()


def _reblocking_floor(
    loop: _Loop, request: _Request, iters: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """How far one request's own answer moves when a second row shares its forward.

    The second row is **the same request**, so both rows are identical and nothing about the
    conditioning changed: what is left is the shared GEMM re-blocking at M=2 against M=1. That
    is the floor the two batching bars are ratios to (trap 2), and this is also where the
    duplicate batch's two rows are checked to agree bitwise -- if they did not, the "floor"
    would be measuring something else as well.
    """
    duplicated = _run(loop, [request, request], iters=None if iters is None else (iters[0], iters[0]))
    alone = _run(loop, [request], iters=None if iters is None else iters[:1])
    assert torch.equal(duplicated[0], duplicated[1]), (
        f"req {request.index}: the two rows of a duplicate batch disagree, so the measured "
        "floor is not the re-blocking alone"
    )
    return (duplicated[0] - alone[0]).abs()


def _ratio_within_t3(gap: torch.Tensor, floor: torch.Tensor, what: str) -> None:
    """The T3 bar on a batching gap: mean and per-element max, both as ratios to the floor."""
    for name, reduce in (("mean", torch.mean), ("max", torch.amax)):
        measured, bar = reduce(gap).item(), reduce(floor).item()
        assert measured <= _T3 * bar, (
            f"{what}: {name} |batched - alone| {measured:.3e} past the T3 bar {_T3 * bar:.3e} "
            f"({_T3}x the re-blocking floor's {name} {bar:.3e}, measured in this test)"
        )


# ----------------------------------------------------------------------------
# R11: the loop against the oracle
# ----------------------------------------------------------------------------


def test_the_recordings_share_one_seeded_trajectory(requests: list[_Request]):
    """The oracle's ``noise`` is bit-for-bit NVIDIA's own venv's, on all 20 requests.

    The loop starts from a seeded trajectory, and the whole of R11 reads as parity only if ours
    starts from the same one the oracle did. The oracle was re-run on M\\*'s stack from the
    dump's recorded seed, so the two ``noise`` tensors *should* be identical -- but "should" is
    the word that hides a re-seeded run, and a trajectory drawn from a different seed is still
    a plausible-looking chunk at the right rms. If this fails, every ratio in this file is
    comparing two different integrations and the bars mean nothing.
    """
    for request in requests:
        assert torch.equal(request.noise.cpu(), request.noise_dump), (
            f"req {request.index}: the oracle's seeded noise is not the dump's"
        )


def test_the_four_step_loop_matches_the_oracle_on_every_recorded_request(
    loop: _Loop, requests: list[_Request],
):
    """R11: ours in bf16 against the fp32 oracle, all 20 requests, at the T3 bar.

    This is the port's one end-to-end number. Everything from the bridge's rows to the served
    chunk is inside it: the action encoder's timestep conditioning, the position table landing
    on the 40 action rows and not the state row (``submodules.py:679``), the DT's 32 blocks,
    the decoder's horizon slice, and four Euler updates carried in bf16 as the reference
    carries them (``submodules.py:699``). The CPU file pins each of those against an inline
    reference one at a time; nothing but this says they compose into the trained policy's
    actions. If it fails, a robot driven by this port goes somewhere else.

    Each request has to land at **its own** floor, not at the batch's average: the floor
    ranges 3.1e-03 to 9.0e-03 over the 20, so a single pooled bar would let the easy requests
    pay for a bad one.

    The oracle's action rms is asserted with it. The DT file's trap 1 applies unchanged here --
    a stack whose weights are not the trained ones lands at a magnitude where every floor is
    wide enough to pass anything -- and on this side of the comparison the answer comes off
    disk, so the guard is on the recording rather than on the forward.
    """
    for request in requests:
        rms = request.want.pow(2).mean().sqrt().item()
        assert _ACTION_RMS[0] < rms < _ACTION_RMS[1], (
            f"req {request.index}: oracle action rms {rms:.3g} outside {_ACTION_RMS}, so the "
            "recording is not the trained model's actions"
        )
        ours = _run(loop, [request]).cpu()
        error, floor = (ours - request.want).abs().mean().item(), request.floor
        assert error <= _T3 * floor, (
            f"req {request.index}: mean abs error {error:.3e} past the T3 bar {_T3 * floor:.3e} "
            f"({_T3}x its measured bf16 floor {floor:.3e})"
        )


def test_the_decoded_actions_match_the_oracle_in_every_joint_group(
    loop: _Loop, requests: list[_Request],
):
    """R11 after the decode: ours, the fp32 oracle and the bf16 oracle each through
    ``Gr00tN17Preprocessor.unapply``, compared **per joint group**.

    The bar above reads the 132-wide normalised chunk, most of which is padding this
    embodiment does not use, and it pools every group into one mean. The decode is where that
    stops being harmless: ``unapply`` cuts the width per group from the *absolute* statistics
    (``preprocessing.py:335``), un-normalises each group by its own percentile bounds and
    composes the relative groups onto the request's raw state. So a group whose bounds span a
    small range is scaled down and one with a wide range is scaled up, and a 7-dim group that
    is 5% of the chunk is 100% of one joint. Comparing per group is what keeps a wide-range
    group from hiding inside the chunk's mean.

    Both oracles are decoded here too, rather than reusing their recorded ``decoded_action``:
    the floor has to be the bf16/fp32 difference *through this decode*, so that the decode's
    own arithmetic is inside the floor and inside our error alike and the ratio reads the
    loop. ``unapply``'s own parity against the reference is
    ``test_gr00t_n17_preprocessing.py``'s, not this file's.

    The bar is the **worst** group against the worst group's floor -- measured median 1.013,
    max 1.066 over the 20. On 19 of the 20 it is one group's own ratio, because the group with
    the largest error is the group with the largest floor; the exception is req 07, where the
    error peaks on ``joint_position`` (9 dims) and the floor on ``eef_9d``. Scoring each group
    against its own floor instead is the stricter reading and it also passes, but only just:
    the worst is 1.194 against a 1.2 bar, on ``gripper_position``, whose mean is taken over a
    single joint x 40 steps and is therefore a noisy ratio of two small samples. That number
    is reported here rather than asserted, because a bar with 0.5% of margin fails on the next
    driver and says nothing when it does.

    If this fails while the bar above passes, the error is concentrated in one group -- which
    is the shape a decode-side bug has, and the shape a chunk-wide mean cannot see.
    """
    for request in requests:
        ours = _run(loop, [request]).cpu()

        def decode(chunk: torch.Tensor, request: _Request = request) -> dict[str, np.ndarray]:
            return loop.preprocessor.unapply(request.tag, chunk.numpy(), request.raw_state)

        got, want, rounded = decode(ours), decode(request.want), decode(request.rounded)
        assert got.keys() == want.keys() and got, got.keys()
        per_group = {
            group: (
                float(np.abs(got[group] - want[group]).mean()),
                float(np.abs(rounded[group] - want[group]).mean()),
            )
            for group in want
        }
        worst, floor = (max(side) for side in zip(*per_group.values(), strict=True))
        assert worst <= _T3 * floor, (
            f"req {request.index}: worst joint group's mean abs error {worst:.3e} past the T3 "
            f"bar {_T3 * floor:.3e} ({_T3}x the worst group's measured bf16 floor {floor:.3e}); "
            f"per group {({g: f'{e:.2e}/{f:.2e}' for g, (e, f) in per_group.items()})}"
        )


def test_the_recorded_requests_occupy_two_buckets(requests: list[_Request]):
    """The 20 requests are exactly two (text rows, image rows) buckets, 8 and 12 deep.

    The bucket key is what the engine batches and captures by: requests in one batch share
    their span lengths, and each distinct key is its own forward and its own CUDA graph
    (``bucket_key_for``, ``submodules.py:476``). So the count is not trivia -- it is what a
    DROID deployment pays per step, and C08 states it as a bar.

    The image rows are constant at 448 and only the text rows vary, which is also what makes
    the two batching tests below possible: a pair of *different* requests that share a bucket
    exists only because 12 of the 20 have the same prompt length. If this fails, either the
    prompts changed shape or the span counting did, and in the second case the ``pair`` fixture
    is quietly comparing requests that the engine would never batch together.
    """
    counts: dict[tuple[int, int], int] = {}
    for request in requests:
        counts[request.bucket] = counts.get(request.bucket, 0) + 1
    assert counts == _BUCKETS, counts


# ----------------------------------------------------------------------------
# R4(b): batching two requests of the same shape
# ----------------------------------------------------------------------------


def test_batching_two_requests_of_one_shape_costs_only_the_gemm_re_blocking(
    loop: _Loop, pair: tuple[_Request, _Request],
):
    """R4(b): two different same-shaped requests, batched against each run alone, at the T3 bar
    over a floor measured in this test.

    Batching is what the engine does with the node, so a request's served actions must not
    depend on who it shares a step with. The comparison is deliberately **not**
    ``torch.equal``: a shared GEMM re-blocks with the row count, so even running the *same*
    request at M=2 does not return its M=1 bits, and an exact test would fail at the kernels'
    discretion. ``_reblocking_floor`` measures exactly that cost and the bar is a ratio to it
    (trap 2 in the banner).

    The floor is measured **per row**, from each request's own duplicate batch, because the
    two rows do not share a floor: measured 1.15e-03 and 1.17e-03 mean, and 2.34e-02 against
    3.12e-02 at the per-element max, which is one bf16 ulp apart at those magnitudes. Against
    one row's floor applied to the whole batch -- how C08's QA first measured it -- the gap is
    mean 1.01x and max 1.33x; per row it is 1.00x and 1.00x, because a row's bits turn out not
    to depend on its neighbour's contents at all. That exactness is a kernel's choice today
    and is not what is asserted here; ``test_perturbing_one_rows_conditioning_...`` is where
    row isolation is pinned.

    If this fails, batching changes the policy and the engine has to serve this node one
    request at a time.
    """
    batched = _run(loop, list(pair))
    for row, request in enumerate(pair):
        alone = _run(loop, [request])
        _ratio_within_t3(
            (batched[row] - alone[0]).abs(),
            _reblocking_floor(loop, request),
            f"req {request.index} batched with req {pair[1 - row].index}",
        )


def test_giving_the_text_blocks_the_image_rows_dwarfs_the_batching_gap(
    loop: _Loop, pair: tuple[_Request, _Request],
):
    """What the batching bar above is small *compared to*, shown by breaking the wiring.

    A ratio bar is only meaningful if the thing it rejects is a different order from the thing
    it admits. So this hands the text blocks the image rows instead -- ``_span_for``
    (``components/dit.py:218``) says a cross block reads the text rows when ``idx % 4 == 0``
    and the image rows otherwise, and a swap leaves every shape, every length and every
    finite-ness check intact. The image rows are sliced to the text rows' count so that even
    the bucket key is unchanged; the only thing wrong is which rows a block read.

    Measured, that moves the chunk by mean 2.97e-02 against a re-blocking floor of 1.15e-03 --
    26x, where the bar admits 1.2x. Both runs are single-request and on SDPA, so the span
    assignment is the only difference between them.

    If this fails, the batching bar is not reading the conditioning at all, and a run that
    read the wrong rows could sit inside it.
    """
    request, _ = pair
    text_rows = request.text.shape[1]
    swapped = _Request(**{**vars(request), "text": request.image[:, :text_rows]})
    assert swapped.bucket == request.bucket, "the control must not change the bucket as well"

    gap = (_run(loop, [swapped])[0] - _run(loop, [request])[0]).abs()
    floor = _reblocking_floor(loop, request)
    assert gap.mean().item() > _CONTROL_MARGIN * floor.mean().item(), (
        f"wrong-span control: mean {gap.mean().item():.3e} is only "
        f"{gap.mean().item() / floor.mean().item():.1f}x the re-blocking floor "
        f"{floor.mean().item():.3e}, so the batching bar does not read the span assignment"
    )


# ----------------------------------------------------------------------------
# R10: rows at different loop iterations
# ----------------------------------------------------------------------------


def test_rows_at_different_loop_iterations_match_each_row_run_alone(
    loop: _Loop, pair: tuple[_Request, _Request],
):
    """R10: one shared forward with each row at its own iteration index, against each row run
    alone, at the T3 bar over the same in-test floor.

    The scaffold's loop does not restart per request: a request that arrives a step late rides
    in the same forward as one that is already three steps in, so one call carries rows at
    different timesteps, different sigmas and different sigma pairs. Everything the step does
    with those is per-row -- the action encoder's timestep embedding, the DT's ``ada_norm``
    conditioning, the Euler ``dt`` -- and a reduction or a broadcast that took the batch's
    first value instead would be invisible with every row at the same index, which is how the
    bar above runs.

    Indices 1 and 3 rather than 0 and 1: index 0's sigma is 0 and its ``dt`` is the same as
    every other step's, so a row reading its neighbour's sigma would still move the trajectory
    the same distance. Measured, the gap is mean 5.55e-04 and max 1.56e-02, which is 1.00x the
    per-row re-blocking floor at those indices.

    If this fails, the engine has to serialise the loop's iterations, and the batch's
    throughput becomes the slowest-arriving request's.
    """
    assert len(set(_MIXED_ITERS)) == len(_MIXED_ITERS), "the point is two different indices"
    mixed = _run(loop, list(pair), iters=_MIXED_ITERS)
    for row, (request, k) in enumerate(zip(pair, _MIXED_ITERS, strict=True)):
        alone = _run(loop, [request], iters=(k,))
        _ratio_within_t3(
            (mixed[row] - alone[0]).abs(),
            _reblocking_floor(loop, request, iters=(k,)),
            f"req {request.index} at iteration {k}, sharing a forward with iteration "
            f"{_MIXED_ITERS[1 - row]}",
        )


def test_perturbing_one_rows_conditioning_leaves_the_other_row_bit_identical(
    loop: _Loop, pair: tuple[_Request, _Request],
):
    """R10's control, and the one that actually bites.

    The bar above is a ratio to a floor, so it would admit a small amount of cross-row leakage
    as long as it stayed near the re-blocking cost. This asserts the property directly and
    exactly: blank row 0's conditioning -- its 448 image rows zeroed, the robot's cameras gone
    dark -- and row 1 must come back **bit-identical**. There is no tolerance to tune here, so
    nothing about it can rot.

    Row 0 is asserted to have *moved*, by the same margin the span-assignment control is held
    to, or the test would pass just as well on a perturbation that reached nothing. That check
    is why the image rows are what is perturbed and not the state row: ``state_features`` has
    rms 0.0335, and shifting all 1536 of its values by 1.0 -- a 30x change in the signal --
    moves the chunk by only 1.2x the re-blocking floor, because the state is 1 of the DT's 41
    rows and reaches the action rows through attention alone. A control that small would not
    distinguish "row 1 is isolated" from "nothing was perturbed". Zeroing the image rows moves
    row 0 by 24x the floor at one step and 33x over the loop.

    Both the mixed single step and the full four-step loop are checked, because they fail
    separately: one step catches a reduction over the batch dimension inside the step, and four
    catch a carried trajectory that is shared or overwritten between iterations rather than
    being per row.

    If this fails, one request's actions depend on another's conditioning, which is a data leak
    between two robots sharing a server and not a numerical issue.
    """
    first, second = pair
    perturbed = _Request(**{**vars(first), "image": torch.zeros_like(first.image)})
    for label, iters in (("one step", _MIXED_ITERS), ("the four-step loop", None)):
        before = _run(loop, [first, second], iters=iters)
        after = _run(loop, [perturbed, second], iters=iters)
        assert torch.equal(after[1], before[1]), (
            f"{label}: req {second.index} moved when req {first.index}'s image rows were "
            f"blanked, by {(after[1] - before[1]).abs().max().item():.3e}"
        )
        moved = (after[0] - before[0]).abs().mean().item()
        floor = _reblocking_floor(loop, first, iters=iters).mean().item()
        assert moved > _CONTROL_MARGIN * floor, (
            f"{label}: req {first.index}'s own row moved by only {moved:.3e} ({moved / floor:.1f}x "
            f"the re-blocking floor {floor:.3e}) with its image rows blanked, so the "
            "perturbation is not reaching the step and the bitwise check above is vacuous"
        )
