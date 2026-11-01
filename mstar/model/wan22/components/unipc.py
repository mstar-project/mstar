"""Inline UniPC (bh2, order 2, flow prediction) for the Wan2.2 denoise loop.

Exact port of diffusers 0.39.0 ``UniPCMultistepScheduler``, restricted to this
checkpoint's scheduler config. Any deviation from the reference is a bug, not a
choice; the port is asserted in lockstep against it per step.

The reference scheduler keeps state across ``step()`` calls, but the denoise loop
is stateless per iteration, so that state is split in two: per-request *tables*
(``make_unipc_tables``), derivable from the step count and recomputed each
iteration, and *loop-carried tensors* (``UniPCState``), routed through the graph's
loop-back edges. All solver math runs float32 on the sample's device, as the
reference does.

``unipc_coefficients`` / ``unipc_coefficient_step`` are the same update with the
per-step branches folded into weights computed up front, for the scaffold's
batched and captured loop; the step functions stay the bit-exact path.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

# The Wan2.2-TI2V-5B checkpoint's scheduler config (scheduler_config.json).
SOLVER_ORDER = 2
NUM_TRAIN_TIMESTEPS = 1000

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


def make_unipc_tables(num_inference_steps: int, flow_shift: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-request sigma and timestep tables.

    Returns sigmas float32 ``[N+1]`` with the terminal zero appended, and timesteps
    int64 ``[N]``. Sigmas are computed float64 and cast at the end, and the
    timesteps are truncated by the int64 cast — both as the reference does, and both
    load-bearing for bit-exactness.
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
    reference. (It is also why this node cannot be compiled — see
    ``Wan22DitSubmodule``.)
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
    return torch.linalg.solve(torch.stack(rows), torch.stack(b))


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


# ---------------------------------------------------------------------------
# Coefficient form, for a batched or captured loop
# ---------------------------------------------------------------------------


class UniPCCoefficients(NamedTuple):
    """The UniPC corrector and predictor of each step as weighted sums, ``[N]`` float32 each.

    The step functions above branch on the step index (no corrector at step 0, the
    order ramp) and solve for the corrector weights on the device. Every branch is
    known per request up front, so the same update can be written with per-step
    weights instead, which lets rows at different steps share one forward and a
    captured graph replay any step. Over step k's converted output ``m_k``, the
    previous two ``m_{k-1}``, ``m_{k-2}``, the sample ``x_k`` and the corrector's
    ``last_sample``:

        x_c     = corr_sample x_k + corr_last last_sample
                  + corr_prev m_{k-1} + corr_prev2 m_{k-2} + corr_this m_k
        x_{k+1} = pred_sample x_c + pred_this m_k + pred_prev m_{k-1}

    Same algebra as the step functions; the weights are computed in float64, so the
    result differs from them only in float32 rounding.
    """

    corr_sample: torch.Tensor
    corr_last: torch.Tensor
    corr_prev: torch.Tensor
    corr_prev2: torch.Tensor
    corr_this: torch.Tensor
    pred_sample: torch.Tensor
    pred_this: torch.Tensor
    pred_prev: torch.Tensor


def _bh2_rhos_f64(rk: float, hh: float, order: int) -> np.ndarray:
    """``_bh2_rhos`` in float64 for one history ratio ``rk``."""
    rks = np.array([rk, 1.0])
    h_phi_1 = np.expm1(hh)
    h_phi_k = h_phi_1 / hh - 1
    b_h = np.expm1(hh)
    rows, b = [], []
    factorial_i = 1
    for i in range(1, order + 1):
        rows.append(rks ** (i - 1))
        b.append(h_phi_k * factorial_i / b_h)
        factorial_i *= i + 1
        h_phi_k = h_phi_k / hh - 1 / factorial_i
    return np.linalg.solve(np.stack(rows), np.array(b))


def unipc_coefficients(sigmas: torch.Tensor, num_inference_steps: int) -> UniPCCoefficients:
    """Per-step weights of the coefficient form for ``make_unipc_tables``' sigmas."""
    s = sigmas.double().numpy()
    with np.errstate(divide="ignore"):
        # +inf at the terminal sigma of 0, which the expm1 below turns into -1, as in
        # the step functions
        lam = np.log(1 - s) - np.log(s)
    n = num_inference_steps
    out = {name: np.zeros(n) for name in UniPCCoefficients._fields}
    for k in range(n):
        if k == 0:
            out["corr_sample"][k] = 1.0  # no corrector before the first output
        else:
            alpha_t = 1 - s[k]
            h = lam[k] - lam[k - 1]
            h_phi_1 = b_h = np.expm1(-h)
            order = unipc_effective_order(k - 1, n)
            out["corr_last"][k] = s[k] / s[k - 1]
            if order == 1:
                rho_prev2, rho_this = 0.0, 0.5
            else:
                rk = (lam[k - 2] - lam[k - 1]) / h
                rho_d1, rho_this = _bh2_rhos_f64(rk, -h, order)
                rho_prev2 = rho_d1 / rk
            out["corr_prev"][k] = -alpha_t * h_phi_1 + alpha_t * b_h * (rho_prev2 + rho_this)
            out["corr_prev2"][k] = -alpha_t * b_h * rho_prev2
            out["corr_this"][k] = -alpha_t * b_h * rho_this
        alpha_t = 1 - s[k + 1]
        h = lam[k + 1] - lam[k]
        h_phi_1 = b_h = np.expm1(-h)
        out["pred_sample"][k] = s[k + 1] / s[k]
        out["pred_this"][k] = -alpha_t * h_phi_1
        if unipc_effective_order(k, n) == 2:
            rk = (lam[k - 1] - lam[k]) / h
            out["pred_this"][k] += alpha_t * b_h * 0.5 / rk
            out["pred_prev"][k] = -alpha_t * b_h * 0.5 / rk
    return UniPCCoefficients(**{name: torch.from_numpy(v).to(torch.float32) for name, v in out.items()})


def unipc_coefficient_step(
    sample: torch.Tensor,
    this_model_output: torch.Tensor,
    model_outputs: torch.Tensor,
    last_sample: torch.Tensor,
    coeffs: UniPCCoefficients,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One corrector + predictor update in the coefficient form, for a batch.

    ``sample``, ``this_model_output`` (``m_k``) and ``last_sample`` are ``[B, ...]``,
    ``model_outputs`` is the ``[B, 2, ...]`` ring (slot 1 the previous output), and
    each weight broadcasts over a row (``[B, 1, ...]``). Returns the next sample, the
    shifted ring and the corrected sample (the next step's ``last_sample``), float32.
    """
    prev, prev2 = model_outputs[:, 1], model_outputs[:, 0]
    corrected = (
        coeffs.corr_sample * sample + coeffs.corr_last * last_sample
        + coeffs.corr_prev * prev + coeffs.corr_prev2 * prev2 + coeffs.corr_this * this_model_output
    )
    new_sample = coeffs.pred_sample * corrected + coeffs.pred_this * this_model_output + coeffs.pred_prev * prev
    return new_sample, torch.stack([prev, this_model_output], dim=1), corrected
