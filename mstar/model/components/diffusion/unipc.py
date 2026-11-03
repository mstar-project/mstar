"""UniPC (bh2, order 2, flow prediction) as a scaffold scheduler.

Port of diffusers ``UniPCMultistepScheduler`` for the config Wan2.2 and Cosmos3
ship: ``solver_type="bh2"``, ``solver_order=2``, ``predict_x0``,
``prediction_type="flow_prediction"``, ``lower_order_final``, no thresholding, no
disabled corrector. Any deviation from the reference is a bug, not a choice; the
port is asserted in lockstep against it per step.

The reference scheduler keeps state across ``step()`` calls, but a denoise loop is
stateless per iteration, so that state is split in two: per-request *tables* (the
sigmas, staged on the device once and read by the loop counter's step index) and
*loop-carried tensors* (``UniPCState``), routed through the graph's loop-back edges
(``DenoiseLoopSubmodule.SOLVER_STATE``, named by :data:`SOLVER_STATE`). All solver
math runs on the sample's device in the sample's dtype, as the reference does, and
none of it reads a device value on the host: a step queues behind the one still
running.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

SOLVER_ORDER = 2
NUM_TRAIN_TIMESTEPS = 1000

# The loop-back edges of a loop on this scheduler, beside the sample.
MODEL_OUTPUTS, LAST_SAMPLE = "unipc_model_outputs", "unipc_last_sample"
SOLVER_STATE = (MODEL_OUTPUTS, LAST_SAMPLE)

# Nudge sigmas[0] off exactly 1.0 so lambda stays finite at the first step.
_SIGMA_ONE_EPS = 1e-6


@dataclass
class UniPCState:
    """Loop-carried UniPC solver state for one request.

    ``model_outputs`` is the order-2 ring buffer of converted outputs (slot 1 is
    the previous step, slot 0 the one before). ``last_sample`` is the sample the
    previous predictor was given, consumed by this step's corrector. Both are zero
    before they are written, and the order ramp guarantees they are not read then.
    """

    model_outputs: torch.Tensor
    last_sample: torch.Tensor


class UniPCStep(NamedTuple):
    """One step's result: the next sample and the state its loop-back edges carry."""

    sample: torch.Tensor
    model_outputs: torch.Tensor
    last_sample: torch.Tensor


def initial_state(sample: torch.Tensor) -> dict[str, torch.Tensor]:
    """Iteration 0's solver state for ``sample``, by edge name: all zeros, which the
    order ramp never reads."""
    return {
        MODEL_OUTPUTS: torch.zeros((SOLVER_ORDER, *sample.shape), dtype=sample.dtype, device=sample.device),
        LAST_SAMPLE: torch.zeros_like(sample),
    }


def make_unipc_tables(num_inference_steps: int, flow_shift: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-request sigma and timestep tables for the plain flow grid (Wan2.2).

    Returns sigmas float32 ``[N+1]`` with the terminal zero appended, and timesteps
    int64 ``[N]``. Sigmas are computed float64 and cast at the end, and the
    timesteps are truncated by the int64 cast, both as the reference does: either
    change moves bits.
    """
    sigmas = np.linspace(1, 1 / NUM_TRAIN_TIMESTEPS, num_inference_steps + 1)[:-1]
    sigmas = flow_shift * sigmas / (1 + (flow_shift - 1) * sigmas)
    if np.fabs(sigmas[0] - 1) < _SIGMA_ONE_EPS:
        sigmas[0] -= _SIGMA_ONE_EPS
    timesteps = torch.from_numpy((sigmas * NUM_TRAIN_TIMESTEPS).copy()).to(torch.int64)
    sigmas = torch.from_numpy(np.concatenate([sigmas, [0.0]]).astype(np.float32))
    return sigmas, timesteps


def unipc_effective_order(step_index: int, num_inference_steps: int) -> int:
    """Predictor order at step k.

    Capped both by the steps remaining (so the last step is order 1) and by the
    warmup ramp. Step k's corrector reuses step k-1's value.
    """
    lower_order_nums = min(step_index, SOLVER_ORDER)
    this_order = min(SOLVER_ORDER, num_inference_steps - step_index)
    return min(this_order, lower_order_nums + 1)


def unipc_convert_model_output(
    model_output: torch.Tensor, sample: torch.Tensor, sigmas: torch.Tensor, step_index: int
) -> torch.Tensor:
    """Flow-prediction x0 conversion.

    ``sigma_t`` stays a CPU 0-dim tensor on purpose, as the reference leaves it. A
    CPU scalar takes the CUDA kernel's full-precision Scalar path; a device 0-dim
    tensor would instead be type-promoted to the bf16 operand's dtype before the
    multiply. That is a bitwise difference, and it breaks lockstep with the
    reference, and it is why a forward that holds this step cannot be compiled whole.
    """
    sigma_t = sigmas[step_index]
    return sample - sigma_t * model_output


def _lambda(sigma: torch.Tensor) -> torch.Tensor:
    """log(alpha) - log(sigma) under the flow map alpha = 1 - sigma.

    At the terminal sigma of 0 this is +inf; the predictor's expm1 arithmetic
    resolves that to the exact x0 output, as the reference does.
    """
    return torch.log(1 - sigma) - torch.log(sigma)


def _bh2_rhos(rks: list[torch.Tensor], hh: torch.Tensor, order: int, device: torch.device) -> torch.Tensor:
    """Solve the B(h) system for the corrector weights.

    ``rks`` are the ratios for the history points; the trailing 1.0 is appended
    here. Only the corrector reaches this solve at order 2 — the order-2 predictor
    short-cuts to [0.5] in the caller.
    """
    rks_t = torch.stack([*rks, torch.ones((), device=device)])
    h_phi_1 = torch.expm1(hh)
    h_phi_k = h_phi_1 / hh - 1
    b_h = torch.expm1(hh)  # solver_type == "bh2"

    rows, b = [], []
    factorial_i = 1
    for i in range(1, order + 1):
        rows.append(torch.pow(rks_t, i - 1))
        b.append(h_phi_k * factorial_i / b_h)
        factorial_i *= i + 1
        h_phi_k = h_phi_k / hh - 1 / factorial_i
    # solve_ex: the same factorization as solve, without solve's host-side
    # check of the LAPACK info (a device sync every step)
    return torch.linalg.solve_ex(torch.stack(rows), torch.stack(b)).result


def unipc_predictor_step(
    state: UniPCState,
    sample: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    order: int,
) -> torch.Tensor:
    """UniP update x_k -> x_{k+1}.

    ``state.model_outputs[1]`` must already hold step k's converted output: the
    ring buffer is shifted before the predictor runs.
    """
    device = sample.device
    sigma_t = sigmas[step_index + 1].to(device)
    sigma_s0 = sigmas[step_index].to(device)
    alpha_t = 1 - sigma_t
    m0 = state.model_outputs[1]

    h = _lambda(sigma_t) - _lambda(sigma_s0)
    hh = -h  # predict_x0
    h_phi_1 = torch.expm1(hh)
    b_h = torch.expm1(hh)  # bh2

    x_t = sigma_t / sigma_s0 * sample - alpha_t * h_phi_1 * m0
    if order == 2:
        sigma_s1 = sigmas[step_index - 1].to(device)
        rk = (_lambda(sigma_s1) - _lambda(sigma_s0)) / h
        d1s = torch.stack([(state.model_outputs[0] - m0) / rk], dim=1)
        # The order-2 predictor simplifies to rhos_p = 0.5. Keep the einsum: its
        # lowering is not bitwise-interchangeable with a scalar multiply, and this
        # port is held to bit-exactness.
        rhos_p = torch.ones(1, dtype=sample.dtype, device=device) * 0.5
        pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s)
        x_t = x_t - alpha_t * b_h * pred_res
    return x_t.to(sample.dtype)


def unipc_corrector_step(
    state: UniPCState,
    this_model_output: torch.Tensor,
    this_sample: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    order: int,
) -> torch.Tensor:
    """UniC correction of x_k using step k's converted output (source
    ``multistep_uni_c_bh_update``). Runs *before* the ring shift, so
    ``state.model_outputs[1]`` is step k-1's output and ``[0]`` step k-2's;
    ``state.last_sample`` is the sample step k-1's predictor started from.
    """
    device = this_sample.device
    sigma_t = sigmas[step_index].to(device)
    sigma_s0 = sigmas[step_index - 1].to(device)
    alpha_t = 1 - sigma_t
    m0 = state.model_outputs[1]

    h = _lambda(sigma_t) - _lambda(sigma_s0)
    hh = -h  # predict_x0
    h_phi_1 = torch.expm1(hh)
    b_h = torch.expm1(hh)  # bh2

    corr_res: torch.Tensor | float = 0
    if order == 1:
        rhos_c = torch.ones(1, dtype=this_sample.dtype, device=device) * 0.5
    else:
        sigma_s1 = sigmas[step_index - 2].to(device)
        rk = (_lambda(sigma_s1) - _lambda(sigma_s0)) / h
        rhos_c = _bh2_rhos([rk], hh, order, device).to(this_sample.dtype)
        d1s = torch.stack([(state.model_outputs[0] - m0) / rk], dim=1)
        # Keep the einsum, same reason as the predictor.
        corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s)

    d1_t = this_model_output - m0
    x_t_ = sigma_t / sigma_s0 * state.last_sample - alpha_t * h_phi_1 * m0
    x_t = x_t_ - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)
    return x_t.to(this_sample.dtype)


def unipc_step(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    state: UniPCState,
    host_sigmas: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    num_steps: int,
) -> UniPCStep:
    """One ``UniPCMultistepScheduler.step``: conversion, corrector, predictor.

    ``sample`` and ``model_output`` carry a leading batch dim of 1 and ``state`` the
    tensors of :func:`initial_state` over that sample. ``host_sigmas`` is the CPU
    table (see :func:`unipc_convert_model_output`), ``sigmas`` the same table staged
    on the sample's device.
    """
    converted = unipc_convert_model_output(model_output, sample, host_sigmas, step_index)
    if step_index > 0:
        sample = unipc_corrector_step(
            state, this_model_output=converted, this_sample=sample, sigmas=sigmas,
            step_index=step_index, order=unipc_effective_order(step_index - 1, num_steps),
        )
    model_outputs = torch.stack([state.model_outputs[1], converted])
    next_sample = unipc_predictor_step(
        UniPCState(model_outputs=model_outputs, last_sample=sample), sample=sample, sigmas=sigmas,
        step_index=step_index, order=unipc_effective_order(step_index, num_steps),
    )
    return UniPCStep(next_sample, model_outputs, sample)
