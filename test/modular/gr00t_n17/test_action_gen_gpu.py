"""GR00T N1.7's whole ``action_gen`` loop on the GPU, against NVIDIA's reference recordings.

``test_dit_gpu.py`` pins one DT forward. This file pins the four ``denoise`` iterations
(``Gr00tN17DenoiseStepSubmodule``) in plain bf16, outside the served autocast, from the seeded noise
to the action chunk, then ``unapply`` to joint values, and checks that batching two requests, or two
rows at different loop iterations, changes neither row.

The recordings (outside the repo) are NVIDIA's ``gr00t`` code run on M\\*'s stack, one per DROID
request: ``oracle_{fp32,bf16}`` from the recorded noise, ``oracle_{fp32,bf16}_seedN`` from the
noise M\\*'s server draws for seed N, and ``dump``, the requests' inputs. The bf16 run against the
fp32 one is the **bf16 floor**, per request; the bar is mean abs error within **1.2x** of it. Ours
is handed the bf16 run's own ``vl_embeds`` and ``state_features``, since the dump's come from
another stack's backbone and the ratio would read two backbones, not our loop.

Every span runs on SDPA (no attention resource is bound); the ragged kernels are
``test_dit_gpu.py``'s. Skips without CUDA, the cached checkpoint or the recordings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from mstar.model.gr00t_n17.components.dit import ActionPositionEmbedding, DiT
from mstar.model.gr00t_n17.components.embodiment import build_action_decoder, build_action_encoder
from mstar.model.gr00t_n17.config import Gr00tN17Config
from mstar.model.gr00t_n17.preprocessing import Gr00tN17Preprocessor
from mstar.model.gr00t_n17.submodules import Gr00tN17DenoiseStepSubmodule
from mstar.model.gr00t_n17.weight_loader import load_gr00t_n17_weights

from .helpers import BACKBONE, BAR, GR00T, RECORDINGS, REQUESTS, RUNS, complete

_ORACLE_FP32, _ORACLE_BF16, _DUMP = RUNS / "oracle_fp32", RUNS / "oracle_bf16", RUNS / "dump"
_SERVING = Path(__file__).resolve().parents[3] / "configs" / "gr00t_n17.yaml"
# Each recording set's (fp32, bf16) pair. `recorded` starts from the dump's own noise and is what
# the parity tests read.
_SETS = {
    "recorded": (_ORACLE_FP32, _ORACLE_BF16),
    **{
        f"seed{s}": (RUNS / f"oracle_fp32_seed{s}", RUNS / f"oracle_bf16_seed{s}")
        for s in (7, 8, 9, 10, 11)
    },
}
# The wrong-noise control: seed 7's conditioning and answer, seed 8's noise.
_RIGHT_SET, _WRONG_NOISE_SET = "seed7", "seed8"


pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and GR00T.exists() and bool(BACKBONE)
         and complete(_ORACLE_FP32, _ORACLE_BF16, _DUMP)),
    reason=(
        f"the full action_gen loop needs CUDA, the cached checkpoint and the {REQUESTS} recorded "
        f"requests in {RUNS}/{{oracle_fp32,oracle_bf16,dump}} ({RECORDINGS})"
    ),
)

_DEVICE = torch.device("cuda:0")
_BF16 = torch.bfloat16

# The fp32 answer's rms: says the weights are the trained ones, since untrained weights set a
# floor inside which every bar passes.
_ACTION_RMS = (0.9, 1.3)
# One row mid-loop and one at the last step, so neither the timestep nor the sigma pair is shared.
_MIXED_ITERS = (1, 3)
# How far past its floor (chunk, decoded worst group) our loop must land on the wrong seed's
# noise. Hand-chosen. The decoded bar is lower because a group's error is a few dims of a
# mostly-padding chunk; it still sits above the 1.2x parity bar.
_WRONG_NOISE_CHUNK, _WRONG_NOISE_DECODED = 50, 2.5


# One recorded request: its conditioning, its answer, and its floor


@dataclass
class _Request:
    """One recorded DROID request. The conditioning is on the device in bf16, as the reference
    head received it; the answers stay on the host in fp32 for the numpy decode."""

    index: int
    text: torch.Tensor  # [1, n_text, 2048], the bridge's non-image rows
    image: torch.Tensor  # [1, n_image, 2048]
    state: torch.Tensor  # [1, 1, 1536], the state encoder's one row
    embodiment_id: torch.Tensor  # [1]
    noise: torch.Tensor  # [1, 40, 132], the reference run's seeded trajectory
    noise_dump: torch.Tensor  # ... and the dump's, which must be the same bits
    want: torch.Tensor  # [1, 40, 132] fp32: the reference in fp32, the answer
    rounded: torch.Tensor  # ... and in bf16, which sets the floor
    tag: str  # the embodiment tag `unapply` keys its statistics by
    raw_state: dict[str, np.ndarray]  # the raw state the relative groups compose onto

    @property
    def bucket(self) -> tuple[int, int]:
        return (self.text.shape[1], self.image.shape[1])

    @property
    def floor(self) -> float:
        """This request's bf16 floor: the reference in bf16 against itself in fp32."""
        return (self.rounded - self.want).abs().mean().item()


def _load(
    config: Gr00tN17Config, index: int, fp32: Path = _ORACLE_FP32, bf16: Path = _ORACLE_BF16,
) -> _Request:
    """One request of one recording set, conditioned on the **bf16** run's tensors. Image rows
    use the reference's own predicate, ``input_ids == image_token_id`` (``qwen3_backbone.py``)."""
    oracle32 = torch.load(fp32 / f"req_{index:02d}.pt", map_location="cpu", weights_only=False)
    oracle16 = torch.load(bf16 / f"req_{index:02d}.pt", map_location="cpu", weights_only=False)
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


# The one build: the denoise step, its schedule, and the decode


@dataclass
class _Loop:
    config: Gr00tN17Config
    submodule: Gr00tN17DenoiseStepSubmodule
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    preprocessor: Gr00tN17Preprocessor


@pytest.fixture(scope="session")
def loop() -> _Loop:
    """The denoise step at the checkpoint's weights, bf16 on the device, built once (2.55 GiB,
    and the slowest load in the file)."""
    config = Gr00tN17Config.from_hf(GR00T, BACKBONE)
    submodule = Gr00tN17DenoiseStepSubmodule(
        config,
        build_action_encoder(config),
        ActionPositionEmbedding(config),
        DiT(config),
        build_action_decoder(config),
        loop_name="denoise_loop",
    )
    load_gr00t_n17_weights(submodule, "denoise_step", GR00T, device=_DEVICE)
    submodule.to(_DEVICE).to(_BF16).eval()
    torch.cuda.empty_cache()

    # `schedule_for` reads neither argument: the steps are the config's.
    schedule = submodule.schedule_for(None, None)
    yield _Loop(
        config=config,
        submodule=submodule,
        sigmas=schedule.sigmas.to(_DEVICE),
        timesteps=schedule.timesteps.to(_DEVICE),
        # The decode needs only `statistics.json`, so no backbone path.
        preprocessor=Gr00tN17Preprocessor.from_pretrained(
            GR00T, max_state_dim=config.max_state_dim, max_action_horizon=config.action_horizon,
        ),
    )
    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def recordings(loop: _Loop) -> Callable[[str], list[_Request]]:
    """A recording set's 20 requests by its ``_SETS`` name, each set loaded once; skips a set
    that is not complete."""
    loaded: dict[str, list[_Request]] = {}

    def get(name: str) -> list[_Request]:
        if name not in loaded:
            if not complete(*_SETS[name]):
                pytest.skip(f"recording set {name} is not {REQUESTS} requests in {_SETS[name]}")
            loaded[name] = [_load(loop.config, index, *_SETS[name]) for index in range(REQUESTS)]
        return loaded[name]

    return get


@pytest.fixture(scope="session")
def requests(recordings: Callable[[str], list[_Request]]) -> list[_Request]:
    """The ``recorded`` set. Its noise must be the dump's bit for bit: a re-seeded run still
    gives a plausible chunk, and every ratio would then compare two different integrations."""
    recorded = recordings("recorded")
    for request in recorded:
        assert torch.equal(request.noise.cpu(), request.noise_dump), (
            f"req {request.index}: the oracle's seeded noise is not the dump's"
        )
    return recorded


@pytest.fixture(scope="session")
def pair(requests: list[_Request]) -> tuple[_Request, _Request]:
    """Two **different** requests that share a bucket, the only kind of pair this engine batches:
    every segment in a plan shares its length."""
    by_bucket: dict[tuple[int, int], list[_Request]] = {}
    for request in requests:
        by_bucket.setdefault(request.bucket, []).append(request)
    shared = next(group for group in by_bucket.values() if len(group) >= 2)
    return shared[0], shared[1]


# Driving the loop


def _run(loop: _Loop, batch: list[_Request], iters: tuple[int, ...] | None = None) -> torch.Tensor:
    """``action_gen``'s loop over same-shaped requests: ``[B, 40, 132]`` in fp32.

    ``denoise`` reads neither ``engine_inputs`` nor ``bucket_key``, so both are None. ``iters=None``
    is the real four-step loop; a tuple gives each row **its own** iteration in one forward.
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
    """How far one request's answer moves when a second row shares its forward.

    A shared GEMM re-blocks with the row count, so M=2 does not return M=1's bits. The second row
    is **the same request**, so what is left is the re-blocking alone: the floor the batching bars
    are ratios to.
    """
    duplicated = _run(loop, [request, request], iters=None if iters is None else (iters[0], iters[0]))
    alone = _run(loop, [request], iters=None if iters is None else iters[:1])
    assert torch.equal(duplicated[0], duplicated[1]), (
        f"req {request.index}: the two rows of a duplicate batch disagree, so the measured "
        "floor is not the re-blocking alone"
    )
    return (duplicated[0] - alone[0]).abs()


def _ratio_within_bar(gap: torch.Tensor, floor: torch.Tensor, what: str) -> None:
    """The bar on a batching gap: mean and per-element max, both as ratios to the floor."""
    for name, reduce in (("mean", torch.mean), ("max", torch.amax)):
        measured, bar = reduce(gap).item(), reduce(floor).item()
        assert measured <= BAR * bar, (
            f"{what}: {name} |batched - alone| {measured:.3e} past the bar {BAR * bar:.3e} "
            f"({BAR}x the re-blocking floor's {name} {bar:.3e}, measured in this test)"
        )


def _chunk_error(ours: torch.Tensor, request: _Request) -> float:
    """Our mean abs error against the fp32 reference. Its floor is ``request.floor``."""
    return (ours - request.want).abs().mean().item()


def _decoded_error(
    loop: _Loop, ours: torch.Tensor, request: _Request,
) -> tuple[float, float, dict[str, tuple[float, float]]]:
    """After the decode: the worst joint group's mean abs error, the worst group's bf16 floor,
    and every group's (error, floor)."""

    def decode(chunk: torch.Tensor) -> dict[str, np.ndarray]:
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
    return worst, floor, per_group


# The loop against the reference


def test_the_four_step_loop_matches_the_oracle_on_every_recorded_request(
    loop: _Loop, requests: list[_Request],
):
    """Ours in bf16 against the fp32 reference, all 20 requests, each at **its own** floor.

    The floor varies about 3x across requests, so one pooled bar would let the easy requests pay
    for a bad one. The reference's rms is asserted too, so the answer off disk is the trained
    model's.
    """
    for request in requests:
        rms = request.want.pow(2).mean().sqrt().item()
        assert _ACTION_RMS[0] < rms < _ACTION_RMS[1], (
            f"req {request.index}: oracle action rms {rms:.3g} outside {_ACTION_RMS}, so the "
            "recording is not the trained model's actions"
        )
        ours = _run(loop, [request]).cpu()
        error, floor = _chunk_error(ours, request), request.floor
        assert error <= BAR * floor, (
            f"req {request.index}: mean abs error {error:.3e} past the bar {BAR * floor:.3e} "
            f"({BAR}x its measured bf16 floor {floor:.3e})"
        )


def test_the_decoded_actions_match_the_oracle_in_every_joint_group(
    loop: _Loop, requests: list[_Request],
):
    """Ours and both reference runs through ``unapply``, compared **per joint group**.

    The 132-wide chunk is mostly padding: a 7-dim group that is 5% of the chunk is 100% of one
    joint. Both reference runs are decoded so the decode's own arithmetic sits inside the floor;
    ``unapply``'s parity is ``test_preprocessing.py``'s. The bar is the worst group against the
    worst group's floor. Each group against its own floor passes only just (``gripper_position``
    is a mean over 40 values), so it is not asserted.
    """
    for request in requests:
        worst, floor, per_group = _decoded_error(loop, _run(loop, [request]).cpu(), request)
        assert worst <= BAR * floor, (
            f"req {request.index}: worst joint group's mean abs error {worst:.3e} past the "
            f"bar {BAR * floor:.3e} ({BAR}x the worst group's measured bf16 floor {floor:.3e}); "
            f"per group {({g: f'{e:.2e}/{f:.2e}' for g, (e, f) in per_group.items()})}"
        )


def test_the_parity_bar_rejects_our_loop_on_another_seeds_noise(
    loop: _Loop, recordings: Callable[[str], list[_Request]],
):
    """Negative control on the noise: seed 7's conditioning and answer, seed 8's noise.

    A dropped seed or a shared generator still returns a plausible chunk at the trained rms, so
    both readings must put this far past the floor on **every** request. A loop that ignores its
    conditioning still follows its noise, so the blanked-camera control is the other half.
    """
    right = recordings(_RIGHT_SET)
    wrong = recordings(_WRONG_NOISE_SET)
    for request, other in zip(right, wrong, strict=True):
        assert not torch.equal(request.noise, other.noise), (
            f"req {request.index}: {_RIGHT_SET} and {_WRONG_NOISE_SET} have the same noise"
        )
        ours = _run(loop, [_Request(**{**vars(request), "noise": other.noise})]).cpu()
        chunk = _chunk_error(ours, request) / request.floor
        worst, floor, _ = _decoded_error(loop, ours, request)
        assert chunk > _WRONG_NOISE_CHUNK and worst / floor > _WRONG_NOISE_DECODED, (
            f"req {request.index} on {_WRONG_NOISE_SET}'s noise: chunk {chunk:.1f}x its floor "
            f"(must exceed {_WRONG_NOISE_CHUNK}x), decoded worst group {worst / floor:.1f}x "
            f"(must exceed {_WRONG_NOISE_DECODED}x), so the parity bar does not read the noise"
        )


def test_the_parity_bar_rejects_our_loop_with_the_image_rows_blanked(
    loop: _Loop, requests: list[_Request],
):
    """Negative control on the conditioning: the right noise, the 448 image rows zeroed, and
    both readings must fail the 1.2x bar on every request. Zeroed, not scaled: an affine change
    is nearly invisible through the per-row ``nn.LayerNorm`` each block opens with
    (``AdaLayerNorm``)."""
    for request in requests:
        blind = _Request(**{**vars(request), "image": torch.zeros_like(request.image)})
        ours = _run(loop, [blind]).cpu()
        chunk = _chunk_error(ours, request) / request.floor
        worst, floor, _ = _decoded_error(loop, ours, request)
        assert chunk > BAR and worst / floor > BAR, (
            f"req {request.index} with its image rows zeroed: chunk {chunk:.2f}x its floor, "
            f"decoded worst group {worst / floor:.2f}x, where the parity bar is {BAR}x, so "
            "the bar does not read the conditioning"
        )


def test_every_recorded_bucket_is_one_the_shipped_yaml_captures(requests: list[_Request]):
    """Each bucket key is its own forward and CUDA graph (``bucket_key_for``), and a bucket with no
    graph runs eager without an error, so the yaml must capture every recorded one."""
    serving = yaml.safe_load(_SERVING.read_text(encoding="utf-8"))
    captured = {tuple(shape) for shape in serving["model_kwargs"]["capture_shapes"]}
    assert {request.bucket for request in requests} <= captured, captured


# Batching two requests of the same shape


def test_batching_two_requests_of_one_shape_costs_only_the_gemm_re_blocking(
    loop: _Loop, pair: tuple[_Request, _Request],
):
    """Two different same-shaped requests batched against each run alone, at the bar over each
    row's own re-blocking floor.

    The floor is per row, not one row's applied to both: a per-element max of bf16 differences
    moves in whole ulps, so two rows' maxima can differ by a third. Isolation is the next test's.
    """
    batched = _run(loop, list(pair))
    for row, request in enumerate(pair):
        alone = _run(loop, [request])
        _ratio_within_bar(
            (batched[row] - alone[0]).abs(),
            _reblocking_floor(loop, request),
            f"req {request.index} batched with req {pair[1 - row].index}",
        )


def test_a_rows_result_does_not_depend_on_who_shares_its_batch(
    loop: _Loop, requests: list[_Request],
):
    """Isolation, **bitwise**: one request fixed, its partner and its position varied.

    The duplicate-batch floor is blind to a cross-row leak (two identical rows make the leak a
    no-op). cuBLAS picks a kernel from shapes and dtypes, not values, so the fixed row's bits
    must not change.
    """
    for anchor in (0, 8):  # one per bucket; `requests` is ordered by index
        fixed = requests[anchor]
        partners = requests[anchor:anchor + 4]
        baseline = None
        for partner in partners:
            first = _run(loop, [fixed, partner])[0]
            if baseline is None:
                baseline = first
            assert torch.equal(first, baseline), (
                f"req {fixed.index} changed when batched with req {partner.index}"
            )
            second = _run(loop, [partner, fixed])[1]
            assert torch.equal(first, second), (
                f"req {fixed.index} changed when moved to row 1 beside req {partner.index}"
            )


# Rows at different loop iterations


def test_rows_at_different_loop_iterations_match_each_row_run_alone(
    loop: _Loop, pair: tuple[_Request, _Request],
):
    """One shared forward with each row at its own iteration, against each row alone, at the bar.

    A request that arrives a step late rides in the same forward as one three steps in, so the
    timestep embedding, ``ada_norm`` conditioning and Euler ``dt`` must be per row. Indices 1 and
    3, not 0: index 0's sigma is 0 and its ``dt`` equals every other step's.
    """
    mixed = _run(loop, list(pair), iters=_MIXED_ITERS)
    for row, (request, k) in enumerate(zip(pair, _MIXED_ITERS, strict=True)):
        alone = _run(loop, [request], iters=(k,))
        _ratio_within_bar(
            (mixed[row] - alone[0]).abs(),
            _reblocking_floor(loop, request, iters=(k,)),
            f"req {request.index} at iteration {k}, sharing a forward with iteration "
            f"{_MIXED_ITERS[1 - row]}",
        )
