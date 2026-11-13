"""The trunk's feed-forward layers: the dense SwiGLU and the mixture of experts.

A tower's leading blocks carry the dense SwiGLU and the rest carry the MoE;
which blocks those are is the checkpoint's ``dense_blocks``, not a constant.
The two are different enough that sharing an implementation would cost more
than it saves, so both live here.

The MoE is DeepSeek-V3's ``noaux_tc`` routing by shape. Six conventions here
each produce a plausible number when inverted, and four of them are places
:mod:`mstar.model.components.moe`'s ``NoAuxTcRouter`` makes the other choice
(it agrees on the first two):

* **the routing bias picks experts but does not weight them.** ``expert_bias``
  is added to the sigmoid scores only to run the top-k; the gate is then the
  *unbiased* score of each picked expert. Gathering from the biased scores
  instead renormalizes to one just the same, so it runs, and it leaks
  training's load-balancing control into the output,
* **the router is float32, always.** Both the rows and the router weight are
  widened before the GEMM, whatever the activation dtype. A bf16 router is
  not a slightly different router: near-tied experts flip, and the output of
  a flipped token moves by the distance between two experts, not by an
  epsilon,
* **the renormalization has no epsilon.** The DeepSeek, HF and M* routers
  divide by ``sum + 1e-20`` to guard an all-zero top-k. This one divides by
  the bare sum. Measured, the difference is *inert*: eight sigmoid scores sum
  to something of order one, where ``1e-20`` is far below the float32
  resolution, so the two expressions give the same number and only logits
  near -50 separate them. Recorded because it is the one convention here no
  parity test can protect,
* **the gate is cast to the activation dtype after the renormalization and
  before the experts.** So the number that scales an expert's output row is a
  bf16 number, and the division that produced it ran in float32. Renormalizing
  after the cast, or carrying float32 gates into the combine, both still sum
  to one,
* **``routed_scaling_factor`` scales the summed routed output, not the gates.**
  In exact arithmetic ``2.5 * sum(g_i y_i)`` and ``sum((2.5 g_i) y_i)`` are
  the same number. In bf16 the first is one rounding applied to a sum of
  eight, and the second is eight roundings fed into the sum, so the two
  disagree in the last bits of every channel,
* **the top-k is sorted.** ``torch.topk`` with ``sorted=True`` returns the
  picked experts in descending biased score; ``sorted=False``, which M*'s
  router passes, returns the same *set* in another permutation (measured, not
  assumed). That permutation fixes the order the eight gated rows are
  accumulated in and the order rows sit in inside an expert's group. In
  float32 it changes the combine. In bf16 eager it does not: the sum
  accumulates in float32 and the single rounding absorbs the reassociation,
  so the convention has to be kept for the GEMM tiling rather than for the
  sum.

The shared expert is always on and **ungated**: its output is added raw,
without the 2.5 and without a sigmoid gate on it. M*'s
:class:`~mstar.model.components.moe.ParallelSparseMoeBlock` requires a
``shared_expert_gate`` when a shared expert is present, which is the Qwen3-Omni
Talker's convention, not this one.

Checkpoint keys
---------------

The MoE's parameter names are the reference's, so a loader needs no remap:
``ffn.router.weight``, ``ffn.routed.gate_up_weight``, ``ffn.routed.down_weight``,
``ffn.shared.gate_up``, ``ffn.shared.down``, ``ffn.expert_bias``. The dense FFN
is the one place we rename: the reference builds it as an ``nn.Sequential``, so
its tensors are stored under the index names ``ffn.0.linear.a_weight``,
``ffn.0.linear.b_weight`` and ``ffn.1.weight``, which become ``ffn.linear.*``
and ``ffn.down.weight`` here.

``expert_bias`` is the only non-bf16 tensor in the checkpoint. It is a buffer
rather than a parameter, and :meth:`MoEFeedForward._apply` keeps it float32
through ``model.to(torch.bfloat16)``, because the scores it is added to are
float32 and rounding the bias to bf16 would flip the selections it exists to
make.

Not ported
----------

The reference's chunked dispatch (``backbone.moe.dispatch_chunk_size``) is a
*different* combine, not a memory-saving rearrangement of this one: it applies
``routed_scaling_factor`` in float32 and rounds the routed sum once at the end.
This module implements the unchunked regime only and takes no chunk size,
rather than offering a knob whose two settings give two different answers.

**This is a known gap, not a safe simplification.** ``null`` is the
*reference's default*, not what the checkpoints carry: all four configs in the
reference's ``tests/data`` (droid, droid-k, yam, base) set
``dispatch_chunk_size`` to 16384. So no published checkpoint selects the path
implemented here, and this module is bit-exact against a regime none of them
runs.

Nor is the chunk size a threshold a served batch stays under, which is the
tempting reading and is wrong. The reference picks the path on
``dispatch_chunk_size is not None`` alone, and the two paths differ in the
combine rather than in the splitting: chunked multiplies the gate by
``routed_scaling_factor`` in float32, sums the ``top_k`` rows in float32
through four interleaved accumulators, and rounds once at the end, where this
module gates in the row dtype, sums, and scales the sum. Measured against the
reference on CPU, with 16384 so large that a single chunk covers the batch and
no boundary is ever crossed, the two disagree at **every** row count tested,
down to a single row: bf16 differs by up to 0.25 absolute, 2.8e-3 relative in
RMS, on every row; float32 by about 4e-6 absolute, 7e-8 relative. That is the
last bits of bf16 per element rather than a wrong answer per element, but it
is not bit-exactness, it is systematic across every row and every MoE block,
and bit-exactness is this port's whole acceptance bar.

Until the chunked combine is implemented here as a second regime with its own
parity tests, :func:`mstar.model.praxis1.config._moe_fields` refuses any
checkpoint that sets the field, which today is all of them. The refusal is a
placeholder for the work and not a judgement that the field is exotic: serving
these weights on this path would run and quietly answer with a policy the
reference does not produce.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# The router GEMM runs over row chunks of this size, as the reference does.
# It is not a tuning knob: a different chunking lets the GEMM backend pick a
# different kernel, and a different summation order flips near-tied experts.
# No served batch comes close to one chunk, so the loop runs once in practice
# and exists to keep the shapes the reference hands the GEMM.
ROUTER_CHUNK_ROWS = 65536


class SwiGLUProjection(nn.Module):
    """The two bias-free input projections of a SwiGLU, kept apart.

    Two separate ``(hidden, dim)`` tensors rather than one ``(2 * hidden, dim)``
    matrix, because that is how the checkpoint stores them and how the
    reference multiplies them: two GEMMs, not one. Measured on CPU, packing
    them and splitting the result is bit-identical, so this is a layout and a
    loader decision rather than a numerical one. The MoE's two SwiGLUs make the
    opposite choice (:class:`SharedExpert` and :class:`RoutedExperts` are both
    packed), which is the reason not to quietly unify them.
    """

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.a_weight = nn.Parameter(torch.empty(hidden, dim))
        self.b_weight = nn.Parameter(torch.empty(hidden, dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return F.linear(x, self.a_weight), F.linear(x, self.b_weight)


class SwiGLUFeedForward(nn.Module):
    """The dense feed-forward of the blocks that have no experts.

    ``silu(x @ a.T) * (x @ b.T)``, then a bias-free projection back to ``dim``.
    The activation is on the ``a`` half: swapping the halves is a different
    function of the same weights, and the checkpoint decides which is which.

    Note the hidden width. The dense blocks are nine times wider than one
    routed expert (11520 against 1280), so this is not the MoE's expert body
    with the routing removed and the two cannot share an implementation.
    """

    def __init__(self, dim: int = 3072, hidden: int = 11520) -> None:
        super().__init__()
        self.linear = SwiGLUProjection(dim, hidden)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.linear(x)
        return self.down(F.silu(a) * b)


class RoutedExperts(nn.Module):
    """The routed expert weight banks, expert-first and K-major.

    ``gate_up_weight`` is ``(E, dim, 2 * hidden)`` and ``down_weight`` is
    ``(E, hidden, dim)``: for each expert the *contraction* dimension comes
    first. That is the layout a grouped GEMM wants for its ``b`` operand, so
    the banks are stored transposed relative to an ``nn.Linear`` weight. A
    loader that writes an ``nn.Linear``-shaped ``(2 * hidden, dim)`` slice here
    gets a shape error for ``dim != 2 * hidden``, and silence when they happen
    to match.

    The module holds no forward: the experts only ever run through
    :meth:`MoEFeedForward.routed_forward`, which needs the dispatch metadata.
    """

    def __init__(self, num_experts: int, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_up_weight = nn.Parameter(torch.empty(num_experts, dim, 2 * hidden))
        self.down_weight = nn.Parameter(torch.empty(num_experts, hidden, dim))


class SharedExpert(nn.Module):
    """The always-on expert: a dense SwiGLU with packed gate|up weights.

    Unlike :class:`SwiGLUProjection` this one *is* packed, into a single
    ``(dim, 2 * hidden)`` tensor, and it is stored K-major, so it is applied
    as ``x @ gate_up`` rather than ``F.linear(x, w)``. The pack is what the
    checkpoint has and a loader cannot get it wrong quietly: an ``nn.Linear``
    shaped slice does not fit.

    The right-multiply is the reference's spelling, and measured here it is
    *not* a numerical difference: on CPU ``x @ W`` and ``F.linear(x, W.t())``
    agree bit for bit (a mutation to the second form passes the whole parity
    suite). It is kept because the two spellings reach different GEMM
    backends, and on an accelerator an explicit transpose is a layout the
    kernel has to deal with rather than one it is handed.
    """

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_up = nn.Parameter(torch.empty(dim, 2 * hidden))
        self.down = nn.Parameter(torch.empty(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = x @ self.gate_up
        gate, up = gate_up.chunk(2, dim=-1)
        return (F.silu(gate) * up) @ self.down


def router_logits(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Float32 router logits of the flat rows ``x``, over fixed row chunks.

    Both operands are widened to float32 before the GEMM. Widening only the
    weight, or running the GEMM in the activation dtype and casting the result,
    both give float32 logits that disagree with these, and the disagreement
    shows up as a different expert for a token whose top two scores are close.
    """
    w = weight.to(torch.float32)
    out = x.new_empty(*x.shape[:-1], w.shape[0], dtype=torch.float32)
    for start in range(0, x.shape[0], ROUTER_CHUNK_ROWS):
        end = min(start + ROUTER_CHUNK_ROWS, x.shape[0])
        out[start:end] = F.linear(x[start:end].to(torch.float32), w)
    return out


def expert_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` of the routed experts' packed projection.

    The gate is the *first* half of the packed output. This is the one place
    where the reference's own CUDA path does not match its CPU path: on CUDA
    it runs a hand-written kernel whose sigmoid rounds differently from
    ``F.silu``. There is nothing to choose here, only something to know: a
    port that matches the eager reference bit for bit has matched the eager
    reference, and an accelerated path has to be checked against its own
    kernel.
    """
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


class MoEFeedForward(nn.Module):
    """One shared expert plus ``top_k`` of ``num_experts`` routed ones.

    See the module docstring for the six conventions this layer gets from the
    reference rather than from the DeepSeek-V3 family it resembles. The shape
    of a call is::

        logits (fp32) -> sigmoid -> + expert_bias -> top-k (sorted)
          -> gather UNBIASED scores -> renormalize (fp32, no eps) -> cast
          -> dispatch, two grouped GEMMs, SwiGLU, gate, undo sort, sum
          -> * routed_scaling_factor + shared(x)

    ``num_experts`` and ``top_k`` here count the routed experts only: 128 and
    8 for the published checkpoints, and the same naming the reference's own
    layer uses. The checkpoint's ``backbone.moe`` calls the same two
    quantities 129 and 9, because *its* ``num_experts`` and ``top_k`` count
    the shared expert, which is always on and so is not something the router
    picks; its routed pair is ``num_routed_experts`` and ``top_k_routed``.
    Build this layer from :meth:`mstar.model.praxis1.config.Praxis1Config.moe_kwargs`
    and not from fields chosen by name, because 129 and 9 do not fail here:
    they route to a 129th bank the checkpoint never filled and hand every
    token one expert too many.
    """

    expert_bias: torch.Tensor

    def __init__(
        self,
        dim: int = 3072,
        hidden: int = 1280,
        num_experts: int = 128,
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.routed = RoutedExperts(num_experts, dim, hidden)
        self.shared = SharedExpert(dim, hidden)
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def _apply(self, fn, recurse: bool = True) -> nn.Module:
        """Move the routing bias with the module but never change its dtype.

        ``model.to(torch.bfloat16)`` would otherwise round the bias, and the
        bias exists to break near-ties in float32 scores. A bf16 bias decides
        those ties by whichever way it happened to round.
        """
        bias = self.expert_bias
        super()._apply(fn, recurse)
        self.expert_bias = bias.to(device=self.expert_bias.device)
        return self

    def route(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The picked experts and their gates, in the dtype of ``x_flat``.

        Returns ``(top_idx, gate)``, both ``(tokens, top_k)``. ``top_idx`` is
        int64 and ordered by descending biased score; ``gate`` holds the
        renormalized unbiased scores in the same order.
        """
        logits = router_logits(self.router.weight, x_flat)
        scores = torch.sigmoid(logits)
        biased = scores + self.expert_bias.to(scores.dtype)
        # sorted=True (the default). See the module docstring: the returned
        # order sets the combine's summation order and the dispatch's row
        # order inside each expert group.
        _, top_idx = torch.topk(biased, self.top_k, dim=-1)
        # The weights come from the UNBIASED scores. This gather is the whole
        # difference between the bias steering selection and the bias steering
        # the output.
        picked = scores.gather(dim=-1, index=top_idx)
        # The bare sum in the denominator, and the division in float32.
        gate = picked / picked.sum(dim=-1, keepdim=True)
        return top_idx, gate.to(x_flat.dtype)

    def dispatch(
        self, x_flat: torch.Tensor, top_idx: torch.Tensor, gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sort the ``(token, expert)`` pairs by expert and gather their rows.

        Returns ``(rows, sorted_gate, inverse, offsets)``: the expert-major
        rows, their gates in the same order, the permutation that undoes the
        sort, and the int32 end offset of each expert's run, which is what a
        grouped GEMM takes as its group boundaries.

        The sort is **stable**. Row order inside an expert cannot change a
        row's own value, but it does change the tiles the GEMM reduces in, and
        a stable sort is also the only way the dispatch is the same from one
        run to the next.

        The rows are cast to the *expert bank's* dtype here, not at the block
        input. In the served configuration the two already agree, so the cast
        is a no-op; it is the boundary that matters. Where the banks are
        narrower than the activations, this gather is the only place the
        activation may round, because the router upstream of it has to see the
        unrounded rows.
        """
        k = top_idx.shape[1]
        tokens = x_flat.shape[0]
        experts = top_idx.reshape(-1)
        order = experts.argsort(stable=True)
        sorted_tokens = torch.arange(tokens, device=x_flat.device).repeat_interleave(k)
        rows = x_flat[sorted_tokens[order]].to(self.routed.gate_up_weight.dtype)
        sorted_gate = gate.reshape(-1)[order]
        offsets = torch.searchsorted(
            experts[order],
            torch.arange(1, self.num_experts + 1, device=x_flat.device),
            out_int32=True,
        )
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=x_flat.device)
        return rows, sorted_gate, inverse, offsets

    def routed_forward(
        self, x_flat: torch.Tensor, top_idx: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        """The routed experts' contribution, **unscaled**.

        Two grouped GEMMs over the expert-major rows with the SwiGLU between
        them, then each row times its gate, the sort undone, and the ``top_k``
        rows of a token summed. The 2.5 is not applied here: see
        :meth:`forward`.

        ``torch._grouped_mm`` is the reference's own op, which is why it is
        here rather than a loop over the experts that received rows. The loop
        is bit-equal to it on CPU, but it is a loop over single-expert GEMMs
        and nothing promises the accelerator reduces those the way it reduces
        one grouped call.
        """
        rows, sorted_gate, inverse, offsets = self.dispatch(x_flat, top_idx, gate)
        gate_up = torch._grouped_mm(rows, self.routed.gate_up_weight, offsets)
        out = torch._grouped_mm(
            expert_swiglu(gate_up), self.routed.down_weight, offsets
        )
        # Gate each expert row before the sum, so the products round to the
        # row dtype first and the sum adds eight already-rounded numbers.
        weighted = out * sorted_gate.unsqueeze(-1)
        channels = weighted.shape[-1]
        return weighted[inverse].reshape(x_flat.shape[0], self.top_k, channels).sum(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., dim)`` in, ``(..., dim)`` out; the leading dims are flat
        inside, since routing is per token and knows nothing of the sequence.
        """
        shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        top_idx, gate = self.route(x_flat)
        # The shared expert runs on the same rows, with no gate of its own and
        # outside the routed scaling.
        y_shared = self.shared(x_flat)
        y_routed = self.routed_forward(x_flat, top_idx, gate)
        # The scale lands on the summed routed output. Rounding happens here,
        # once per channel, and not eight times inside route().
        out = y_routed * self.routed_scaling_factor + y_shared
        return out.reshape(shape)
