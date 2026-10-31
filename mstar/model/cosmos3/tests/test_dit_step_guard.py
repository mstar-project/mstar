"""A Cosmos3 DiT denoise loop is never restarted over a request that ran it.

Seen at 8 concurrent kv rollouts: a finished request, re-queued after a refused
admit, came back with no loop-back latents, so ``prepare_inputs`` took it for a
fresh loop start over its last window. The worker no longer re-queues it (see
``test_admit_failure_handling``); this is the DiT's own guard, which fails just
that request.
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
    VIDEO_GEN_AR_LOOP,
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


def _fwd(rid, k):
    return SimpleNamespace(
        request_id=f"req-{rid}", rid_handle=rid, random_seed=0, dynamic_loop_iter_counts={VIDEO_GEN_AR_LOOP: k},
    )


def _prepare(dit, rid, inputs, k=0):
    return dit.prepare_inputs(VIDEO_GEN_AR_WALK, _fwd(rid, k), inputs)


def _step(dit, st, inp, k):
    """What the forward does to a request's latents and solver state for one denoise step."""
    lat = inp.tensor_inputs["latents"]
    local = k % st["ar_iters_per_window"]
    solver = {name: inp.tensor_inputs[name] for name in dit._solver_state}
    return dit._scheduler_step(st, torch.zeros_like(lat), lat, local, solver)


def test_rollout_steps_every_window_and_commit_unchanged():
    """A whole two-window kv rollout: a step for each denoise iteration, the kv
    commit, the next window's fresh solver state, and the loop's one extra
    dispatched step is still vetoed."""
    dit = _dit()
    st = _kv_rollout(dit, 1)
    inp = _prepare(dit, 1, {})
    kinds = []
    for k in range(WINDOWS * (STEPS + 1)):
        commit = inp.resource_step_info.commit
        kinds.append("commit" if commit else "gen")
        if commit:
            st.add("scheduler", dit._new_scheduler(STEPS, "cpu", flow_shift=12.0))
            lat = inp.tensor_inputs["latents"]
            solver = dit._first_solver_state(st, lat)
        else:
            lat, solver = _step(dit, st, inp, k)
        inp = _prepare(dit, 1, {"latents": [lat], **{n: [t] for n, t in solver.items()}}, k + 1)
        if inp is None:
            break
    assert kinds == (["gen"] * STEPS + ["commit"]) * WINDOWS
    assert inp is None


def test_finished_request_re_queued_fails_alone():
    """The B200 trace, on CPU: request 1 finished its last window; re-queued with
    no loop-back latents past iteration 0 it must not restart. Request 2 is
    mid-window. It keeps running and request 1 fails alone."""
    dit = _dit()
    _kv_rollout(dit, 1)
    live = _kv_rollout(dit, 2)
    first = _prepare(dit, 2, {})
    new, solver = _step(dit, live, first, 0)

    batch = SimpleNamespace(
        node_name="dit", request_ids=[1, 2], graph_walk=VIDEO_GEN_AR_WALK,
        step_context=StepContext(
            request_ids=[1, 2], graph_walk=VIDEO_GEN_AR_WALK, slot=0, capture=False,
        ),
        per_request_info_wrapped={1: _fwd(1, STEPS), 2: _fwd(2, 1)},
        per_request_input_tensors={
            1: {},
            2: {"latents": [new], **{n: [t] for n, t in solver.items()}},
        },
        per_request_input_metadata={}, final_stream_rids=set(),
        skipped_rids=set(), failed_requests={}, inputs=None, running_batched=None,
    )
    batch.register_prepare_batch = lambda inputs: setattr(batch, "inputs", inputs)
    batch.drop_rids = lambda rids: setattr(
        batch, "request_ids", [r for r in batch.request_ids if r not in rids])
    batch.register_failure = lambda rid, err, walk=None: batch.failed_requests.__setitem__(rid, str(err))
    eng = Engine.__new__(Engine)
    eng._submodules = {"dit": SimpleNamespace(submodule=dit, resources={})}
    eng._enable_nvtx = False
    eng._loopback_steps = {}

    eng.prepare_inputs(batch)

    assert batch.request_ids == [2], "the live request must still run"
    assert list(batch.failed_requests) == [1]
    assert "no loop-back latents" in batch.failed_requests[1]
    # The error reaches the conductor and the client, so it names the
    # request by its id, not by this worker's recycled handle.
    assert "request req-1 " in batch.failed_requests[1]
