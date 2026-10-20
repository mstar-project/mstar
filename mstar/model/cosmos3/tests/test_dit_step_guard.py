"""A Cosmos3 DiT denoise step is never dispatched onto a scheduler that has
already run all its steps.

UniPC counts its own steps; one past the last asserts ``this_order > 0`` inside
the batched forward and fails every request batched with that one. Seen at 8
concurrent kv rollouts: a finished request, re-queued after a refused admit,
came back with no loop-back latents, so ``prepare_inputs`` took it for a fresh
loop start (``time_index`` 0) over its last window's spent scheduler. The
worker no longer re-queues it (see ``test_admit_failure_handling``); this is
the DiT's own guard, which fails just that request.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers import UniPCMultistepScheduler  # noqa: E402

from mstar.engine.engine import Engine  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model  # noqa: E402
from mstar.model.cosmos3.submodules import (  # noqa: E402
    VIDEO_GEN_AR_WALK,
    Cosmos3DiTSubmodule,
)

# nvidia/Cosmos3-Edge scheduler/scheduler_config.json (the fields UniPC reads)
EDGE_SCHEDULER = {
    "num_train_timesteps": 1000, "solver_order": 2, "prediction_type": "flow_prediction",
    "predict_x0": True, "solver_type": "bh2", "lower_order_final": True,
    "use_flow_sigmas": True, "use_karras_sigmas": True, "flow_shift": 1.0,
    "final_sigmas_type": "zero", "timestep_spacing": "linspace",
    "sigma_min": 0.147, "sigma_max": 200.0,
}
STEPS, WINDOWS = 4, 2


def _dit() -> Cosmos3DiTSubmodule:
    cfg = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True).config
    cfg.use_native_flow_schedule = True
    dit = Cosmos3DiTSubmodule(
        transformer=None, config=cfg,
        scheduler=UniPCMultistepScheduler.from_config(EDGE_SCHEDULER),
    )
    dit.transformer = SimpleNamespace(proj_in=SimpleNamespace(weight=torch.zeros(1)))
    dit.get_device = lambda: torch.device("cpu")
    return dit


def _kv_rollout(dit, rid):
    """A kv-mode windowed request as its prefill leaves it."""
    st = dit.request_state(rid)
    st.add_all(
        scheduler=dit._new_scheduler(STEPS, "cpu", flow_shift=12.0),
        ar_schedule=SimpleNamespace(window=lambda i: SimpleNamespace(commit_start=0, commit_end=1)),
        ar_steps=STEPS, ar_iters_per_window=STEPS + 1,
        ar_total_iters=WINDOWS * (STEPS + 1), ar_kv_mode=True,
        ar_retention={}, ar_tokens_per_unit=4,
        cond={"num_vision_tokens": 4}, uncond=None, latent_shape=(1, 2, 1, 2, 2),
    )
    return st


def _prepare(dit, rid, inputs):
    return dit.prepare_inputs(
        VIDEO_GEN_AR_WALK,
        SimpleNamespace(request_id=f"req-{rid}", rid_handle=rid, random_seed=0), inputs,
    )


def _step(dit, st, inp):
    """What the forward does to the request's scheduler for one denoise step."""
    sched = st["scheduler"]
    lat = inp.tensor_inputs["latents"]
    local = int(inp.tensor_inputs["time_index"].item()) % st["ar_iters_per_window"]
    return dit._scheduler_step(st, torch.zeros_like(lat[0]), sched.timesteps[local], lat[0])


def test_rollout_steps_every_window_and_commit_unchanged():
    """The guard leaves a rollout's sequence alone: each window's denoise steps,
    the kv commit that runs on the spent scheduler, the next window's fresh
    scheduler, and the loop's one extra dispatched step is still vetoed."""
    dit = _dit()
    st = _kv_rollout(dit, 1)
    inp = _prepare(dit, 1, {})
    kinds = []
    for _ in range(WINDOWS * (STEPS + 1)):
        commit = inp.resource_step_info.commit
        kinds.append("commit" if commit else "gen")
        ti = inp.tensor_inputs["time_index"]
        if commit:
            assert st["scheduler"].step_index == STEPS
            st.add("scheduler", dit._new_scheduler(STEPS, "cpu", flow_shift=12.0))
            lat = inp.tensor_inputs["latents"]
        else:
            lat = _step(dit, st, inp).unsqueeze(0)
        inp = _prepare(dit, 1, {"latents": [lat], "time_index": [ti + 1]})
        if inp is None:
            break
    assert kinds == (["gen"] * STEPS + ["commit"]) * WINDOWS
    assert inp is None


def test_finished_request_re_queued_fails_alone_instead_of_asserting():
    """The B200 trace, on CPU: request 1 finished its last window; re-queued
    with no loop-back latents it prepares as time_index 0 over the spent
    scheduler. Request 2 is mid-window. Before, both prepared and the batched
    forward asserted, failing both; now request 1 fails alone."""
    dit = _dit()
    done, live = _kv_rollout(dit, 1), _kv_rollout(dit, 2)
    lat = _prepare(dit, 1, {}).tensor_inputs["latents"][0]
    for t in done["scheduler"].timesteps:
        lat = dit._scheduler_step(done, torch.zeros_like(lat), t, lat)
    with pytest.raises(AssertionError):
        # the crash being guarded: one step past the scheduler's last
        dit._scheduler_step(done, torch.zeros_like(lat), done["scheduler"].timesteps[-1], lat)
    first = _prepare(dit, 2, {})
    _step(dit, live, first)

    batch = SimpleNamespace(
        node_name="dit", request_ids=[1, 2], graph_walk=VIDEO_GEN_AR_WALK,
        step_context=StepContext(
            request_ids=[1, 2], graph_walk=VIDEO_GEN_AR_WALK, slot=0, capture=False,
        ),
        per_request_info_wrapped={
            r: SimpleNamespace(request_id=f"req-{r}", rid_handle=r, random_seed=0)
            for r in (1, 2)
        },
        per_request_input_tensors={
            1: {},
            2: {"latents": [first.tensor_inputs["latents"]], "time_index": [torch.tensor([1])]},
        },
        per_request_input_metadata={}, final_stream_rids=set(),
        skipped_rids=set(), failed_requests={}, inputs=None, running_batched=None,
    )
    batch.register_prepare_batch = lambda inputs: setattr(batch, "inputs", inputs)
    batch.drop_rids = lambda rids: setattr(
        batch, "request_ids", [r for r in batch.request_ids if r not in rids])
    batch.register_failure = lambda rid, err: batch.failed_requests.__setitem__(rid, str(err))
    eng = Engine.__new__(Engine)
    eng._submodules = {"dit": SimpleNamespace(submodule=dit, resources={})}
    eng._enable_nvtx = False
    eng._loopback_steps = {}

    eng.prepare_inputs(batch)

    assert batch.request_ids == [2], "the live request must still run"
    assert list(batch.failed_requests) == [1]
    assert "ran all 4 steps" in batch.failed_requests[1]
    # The error reaches the conductor and the client, so it names the
    # request by its id, not by this worker's recycled handle.
    assert "request req-1 " in batch.failed_requests[1]
