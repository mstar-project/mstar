"""Precomputed-constant UniPC step for the LingBot denoise loop.

Bit-identical restructuring of ``mstar.model.wan22.components.unipc`` (which is
an exact port of diffusers' UniPC, bh2 / order 2 / flow prediction). Every
scalar in the solver update depends only on ``(sigmas, step_index, order)`` —
not on the sample — so this module hoists those computations out of the hot
loop into a per-(num_steps, shift) table built once, running *the same torch
ops in the same order on the same dtypes/devices* as the reference. The
per-step work is then just the sample-sized tensor expressions, evaluated in
the reference's exact association order.

Bitwise equivalence with the reference implementation is asserted by
``verify_against_reference`` (used by the perf driver and tests): identical
inputs must produce ``torch.equal`` outputs at every step.

The table build also runs the reference's ``torch.linalg.solve`` — once per
request instead of once per step, removing cuSOLVER's per-step D2H info-check
sync from the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mstar.model.wan22.components.unipc import (
    SOLVER_ORDER,
    _bh2_rhos,
    _lambda,
    unipc_effective_order,
)

__all__ = ["UniPCStepTable", "SOLVER_ORDER", "unipc_effective_order"]


@dataclass
class _StepConsts:
    # convert_model_output: sigma stays a CPU 0-dim tensor (Scalar kernel path,
    # matching the reference exactly).
    conv_sigma: torch.Tensor
    # predictor (UniP, x_k -> x_{k+1})
    p_ratio: torch.Tensor  # sigma_t / sigma_s0                (0-dim, device)
    p_a1: torch.Tensor  # alpha_t * h_phi_1                    (0-dim, device)
    p_order2: bool
    p_inv_rk: torch.Tensor | None  # 1 / rk                    (0-dim, device)
    p_ab: torch.Tensor | None  # alpha_t * b_h                 (0-dim, device)
    p_rhos: torch.Tensor | None  # rhos_p (= [0.5])            (1-elem, device)
    # corrector (UniC, correcting x_k with step k's output)
    c_ratio: torch.Tensor | None
    c_a1: torch.Tensor | None
    c_ab: torch.Tensor | None
    c_order1: bool
    c_inv_rk: torch.Tensor | None
    c_rhos: torch.Tensor | None  # full rhos_c (last entry used on d1_t)


class UniPCStepTable:
    """All k-dependent solver constants for one (num_steps, shift, device) run.

    Built with the reference op sequences (including the corrector's
    ``linalg.solve``) so every constant is bitwise what the reference computes
    per step.
    """

    def __init__(self, sigmas: torch.Tensor, num_steps: int, device: torch.device):
        self.num_steps = num_steps
        self.steps: list[_StepConsts] = []
        dev = torch.device(device)
        for k in range(num_steps):
            conv_sigma = sigmas[k]  # CPU 0-dim, as the reference keeps it

            # ---- predictor consts (reference: unipc_predictor_step) ----
            p_order = unipc_effective_order(k, num_steps)
            sigma_t = sigmas[k + 1].to(dev)
            sigma_s0 = sigmas[k].to(dev)
            alpha_t = 1 - sigma_t
            h = _lambda(sigma_t) - _lambda(sigma_s0)
            hh = -h
            h_phi_1 = torch.expm1(hh)
            b_h = torch.expm1(hh)
            p_ratio = sigma_t / sigma_s0
            p_a1 = alpha_t * h_phi_1
            p_order2 = p_order == 2
            p_inv_rk = p_ab = p_rhos = None
            if p_order2:
                sigma_s1 = sigmas[k - 1].to(dev)
                rk = (_lambda(sigma_s1) - _lambda(sigma_s0)) / h
                p_inv_rk = rk  # divide by rk per-step, matching `(m0_prev - m0) / rk`
                p_ab = alpha_t * b_h
                p_rhos = torch.ones(1, dtype=torch.float32, device=dev) * 0.5

            # ---- corrector consts (reference: unipc_corrector_step at step k,
            #      order = unipc_effective_order(k-1, num_steps); runs when k>0) ----
            c_ratio = c_a1 = c_ab = c_inv_rk = c_rhos = None
            c_order1 = False
            if k > 0:
                c_order = unipc_effective_order(k - 1, num_steps)
                sigma_t_c = sigmas[k].to(dev)
                sigma_s0_c = sigmas[k - 1].to(dev)
                alpha_t_c = 1 - sigma_t_c
                h_c = _lambda(sigma_t_c) - _lambda(sigma_s0_c)
                hh_c = -h_c
                h_phi_1_c = torch.expm1(hh_c)
                b_h_c = torch.expm1(hh_c)
                c_ratio = sigma_t_c / sigma_s0_c
                c_a1 = alpha_t_c * h_phi_1_c
                c_ab = alpha_t_c * b_h_c
                c_order1 = c_order == 1
                if not c_order1:
                    sigma_s1_c = sigmas[k - 2].to(dev)
                    rk_c = (_lambda(sigma_s1_c) - _lambda(sigma_s0_c)) / h_c
                    c_inv_rk = rk_c
                    c_rhos = _bh2_rhos([rk_c], hh_c, c_order, dev).to(torch.float32)
                else:
                    c_rhos = torch.ones(1, dtype=torch.float32, device=dev) * 0.5
            self.steps.append(
                _StepConsts(
                    conv_sigma=conv_sigma,
                    p_ratio=p_ratio,
                    p_a1=p_a1,
                    p_order2=p_order2,
                    p_inv_rk=p_inv_rk,
                    p_ab=p_ab,
                    p_rhos=p_rhos,
                    c_ratio=c_ratio,
                    c_a1=c_a1,
                    c_ab=c_ab,
                    c_order1=c_order1,
                    c_inv_rk=c_inv_rk,
                    c_rhos=c_rhos,
                )
            )

    def convert_model_output(self, model_output: torch.Tensor, sample: torch.Tensor, k: int) -> torch.Tensor:
        # reference: sample - sigma_t * model_output   (sigma_t CPU 0-dim scalar path)
        return sample - self.steps[k].conv_sigma * model_output

    def corrector_step(
        self,
        model_outputs: torch.Tensor,
        last_sample: torch.Tensor,
        this_model_output: torch.Tensor,
        this_sample: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        st = self.steps[k]
        m0 = model_outputs[1]
        if st.c_order1:
            corr_res: torch.Tensor | float = 0
            rhos_c = st.c_rhos
        else:
            d1s = torch.stack([(model_outputs[0] - m0) / st.c_inv_rk], dim=1)
            rhos_c = st.c_rhos
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s)
        d1_t = this_model_output - m0
        x_t_ = st.c_ratio * last_sample - st.c_a1 * m0
        x_t = x_t_ - st.c_ab * (corr_res + rhos_c[-1] * d1_t)
        return x_t.to(this_sample.dtype)

    def predictor_step(self, model_outputs: torch.Tensor, sample: torch.Tensor, k: int) -> torch.Tensor:
        st = self.steps[k]
        m0 = model_outputs[1]
        x_t = st.p_ratio * sample - st.p_a1 * m0
        if st.p_order2:
            d1s = torch.stack([(model_outputs[0] - m0) / st.p_inv_rk], dim=1)
            pred_res = torch.einsum("k,bkc...->bc...", st.p_rhos, d1s)
            x_t = x_t - st.p_ab * pred_res
        return x_t.to(sample.dtype)


def verify_against_reference(num_steps: int, shift: float, shape, device="cuda", seed=0) -> None:
    """Assert bitwise equality with the wan22 reference across a full loop."""
    from mstar.model.lingbot.submodules import make_flow_unipc_tables
    from mstar.model.wan22.components.unipc import (
        UniPCState,
        unipc_convert_model_output,
        unipc_corrector_step,
        unipc_predictor_step,
    )

    sigmas, _ = make_flow_unipc_tables(num_steps, shift)
    table = UniPCStepTable(sigmas, num_steps, torch.device(device))
    g = torch.Generator(device=device).manual_seed(seed)
    lat_a = torch.randn((1, *shape), generator=g, device=device, dtype=torch.float32)
    lat_b = lat_a.clone()
    ring_a = torch.zeros((SOLVER_ORDER, 1, *shape), device=device, dtype=torch.float32)
    ring_b = ring_a.clone()
    last_a = torch.zeros_like(lat_a)
    last_b = last_a.clone()
    for k in range(num_steps):
        noise = torch.randn_like(lat_a)  # stand-in for the DiT output
        # reference
        m_a = unipc_convert_model_output(noise, lat_a, sigmas, k)
        s_a = lat_a
        if k > 0:
            s_a = unipc_corrector_step(
                UniPCState(model_outputs=ring_a, last_sample=last_a),
                this_model_output=m_a,
                this_sample=s_a,
                sigmas=sigmas,
                step_index=k,
                order=unipc_effective_order(k - 1, num_steps),
            )
        ring_a = torch.stack([ring_a[1], m_a])
        nlat_a = unipc_predictor_step(
            UniPCState(model_outputs=ring_a, last_sample=s_a),
            sample=s_a,
            sigmas=sigmas,
            step_index=k,
            order=unipc_effective_order(k, num_steps),
        )
        # fast
        m_b = table.convert_model_output(noise, lat_b, k)
        s_b = lat_b
        if k > 0:
            s_b = table.corrector_step(ring_b, last_b, m_b, s_b, k)
        ring_b = torch.stack([ring_b[1], m_b])
        nlat_b = table.predictor_step(ring_b, s_b, k)
        assert torch.equal(m_a, m_b), f"convert mismatch at k={k}"
        assert torch.equal(s_a, s_b), f"corrector mismatch at k={k}"
        assert torch.equal(nlat_a, nlat_b), f"predictor mismatch at k={k}"
        last_a, lat_a = s_a, nlat_a
        last_b, lat_b = s_b, nlat_b
