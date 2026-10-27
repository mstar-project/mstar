"""Sampling for discrete diffusion: score every position, return log-probs too.

The contract is the one an autoregressive sampler cannot express. A diffusion
step hands in the logits for every unrevealed position, gets back a token for
each **and** the log-probability that won, and the caller decides how many of
them to commit. Which positions are still masked, how many to reveal this
step, and where the accepted tokens go are the model's business, because they
depend on a schedule and a ranking that differ per decoder.

Everything here runs batched over a packed ``[rows, positions, vocab]``
tensor. Guidance is a per-position vector, so a batch mixing guidance scales
is one kernel path. The token draw is not: when any request samples, each one
draws on its own slice from its own seeded stream (see ``_pick``).
"""

import math

import torch
import torch.nn.functional as F

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.diffusion_sampler.config import (
    DiffusionSamplerSpec,
    DiffusionSamplerStep,
    DiffusionSamplingReqConfig,
)
from mstar.engine.resources.step import StepContext


class DiffusionSamplerResource(Resource):
    """Scores a diffusion step's canvas positions for a batch of requests."""

    def __init__(
        self,
        vocab_size: int,
        num_rows: int,
        forbidden_class: int | None,
        device: torch.device,
    ):
        self._vocab_size = vocab_size
        self._num_rows = num_rows
        self._forbidden_class = forbidden_class
        self._device = device
        self._configs: dict[str, DiffusionSamplingReqConfig] = {}
        self._iteration: int = 0

    @classmethod
    def build(cls, spec: DiffusionSamplerSpec, info: EngineResourceInfo):
        return cls(
            vocab_size=spec.vocab_size,
            num_rows=spec.num_rows,
            forbidden_class=spec.forbidden_class,
            device=info.device,
        )

    # -- request lifecycle -------------------------------------------------

    def ingest_request(self, rid: str, overrides: DiffusionSamplingReqConfig | None = None):
        self._configs[rid] = overrides or DiffusionSamplingReqConfig()

    def remove_request(self, rid: str):
        self._configs.pop(rid, None)

    # -- step lifecycle ----------------------------------------------------

    def plan(self, step: DiffusionSamplerStep, ctx: StepContext):
        # Nothing to stage: every input arrives with the `sample` call, and
        # the step object carries no layout. Present so the runner has
        # something to call.
        del step, ctx

    # -- sampling ----------------------------------------------------------

    def sample(
        self,
        request_ids: list[str],
        c_logits: torch.Tensor,
        u_logits: torch.Tensor | None = None,
        seq_lens: list[int] | None = None,
        iterations: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score every position; return ``(tokens, logprobs)``.

        Args:
            request_ids: one id per entry of ``seq_lens``, in packed order.
            c_logits: ``[rows, positions, vocab]``, every request's positions
                concatenated along ``positions``.
            u_logits: the unconditional half, same shape, or ``None`` when no
                request in the batch asked for guidance.
            seq_lens: positions belonging to each request. ``None`` means one
                request owning the whole tensor.
            iterations: which diffusion step each request is on. Requests in
                one batch are generally at *different* iterations, since each
                runs its own step count, and the draw is seeded per request
                from ``(seed, iteration)`` so a replay matches. Defaults to
                zeros, which only makes sense for an unseeded caller.

        Returns:
            ``tokens`` and ``logprobs``, both ``[rows, positions]``.
            ``logprobs`` is each position's max log-probability, the
            confidence a ranked reveal sorts on, also when the drawn token is
            not the argmax.
        """
        if c_logits.dim() != 3:
            raise ValueError(
                f"expected [rows, positions, vocab] logits, got {tuple(c_logits.shape)}"
            )
        if c_logits.shape[0] != self._num_rows or c_logits.shape[2] != self._vocab_size:
            raise ValueError(
                f"logits are {tuple(c_logits.shape)}, but the spec declared "
                f"{self._num_rows} rows and a vocabulary of {self._vocab_size}"
            )
        if seq_lens is None:
            seq_lens = [c_logits.shape[1]]
        if len(seq_lens) != len(request_ids):
            raise ValueError(
                f"{len(request_ids)} requests but {len(seq_lens)} sequence lengths"
            )
        if sum(seq_lens) != c_logits.shape[1]:
            raise ValueError(
                f"sequence lengths sum to {sum(seq_lens)}, logits carry "
                f"{c_logits.shape[1]} positions"
            )
        if iterations is None:
            iterations = [0] * len(request_ids)
        elif len(iterations) != len(request_ids):
            raise ValueError(
                f"{len(request_ids)} requests but {len(iterations)} iterations"
            )

        scales = [self._config(rid).guidance_scale for rid in request_ids]
        if u_logits is None or not any(scales):
            log_probs = F.log_softmax(c_logits, dim=-1)
        else:
            log_probs = self._combine(
                c_logits, u_logits, self._guidance(scales, seq_lens), mixed=not all(scales)
            )

        if self._forbidden_class is not None:
            log_probs[..., self._forbidden_class] = -float("inf")

        tokens = self._pick(log_probs, request_ids, seq_lens, iterations)
        # The max, not the drawn token's log-prob: the reference ranks the
        # reveal on it whichever token the Gumbel draw landed on.
        return tokens, log_probs.max(dim=-1).values

    # -- internals ---------------------------------------------------------

    def _guidance(self, scales, seq_lens):
        """Each request's guidance scale across the positions it owns, as
        ``[1, positions, 1]``. Built on the host and copied without a sync."""
        g = torch.tensor(
            [s for s, n in zip(scales, seq_lens, strict=True) for _ in range(n)],
            dtype=torch.float32,
            pin_memory=self._device.type == "cuda",
        )
        return g.to(self._device, non_blocking=True).view(1, -1, 1)

    @staticmethod
    def _combine(c_logits, u_logits, guidance, mixed):
        """Classifier-free guidance, in the reference's operation order.

        The log_softmax really is applied twice on the guided path: the
        combine runs over log-probabilities, not logits, and the result is
        renormalised. Matching that matters more than tidying it, because the
        checkpoint was tuned behind it.

        A batch with no guidance anywhere never gets here: ``sample`` skips the
        second normalisation, since ``log_softmax(log_softmax(x))`` is not
        ``log_softmax(x)``. ``mixed`` is decided on the host for the same
        reason the vector is built there: a ``bool()`` on a device tensor
        would wait for the forward.
        """
        plain = F.log_softmax(c_logits, dim=-1)
        guided = torch.log_softmax(
            plain + guidance * (plain - F.log_softmax(u_logits, dim=-1)), dim=-1
        )
        if not mixed:
            return guided
        # Mixed batch: positions at guidance 0 keep the singly-normalised form.
        return torch.where(guidance != 0, guided, plain)

    def _pick(self, log_probs, request_ids, seq_lens, iterations):
        """Greedy where temperature is 0, Gumbel over the top slice elsewhere.

        The top slice is taken per request too, at that request's own
        ``top_ratio``, which is why this loops rather than picking one k for
        the step.

        The Gumbel noise is drawn per request, on that request's own slice,
        from a generator seeded by its ``(seed, iteration)``. Drawing once for
        the whole batch would be cheaper by one kernel launch and wrong: a
        request's output would then depend on who else happened to be in the
        step and in what order, which is exactly the property the packed
        parity tier exists to protect.
        """
        greedy = log_probs.argmax(dim=-1)
        if not any(self._config(rid).temperature > 0 for rid in request_ids):
            return greedy

        default = DiffusionSamplingReqConfig()
        sampled = torch.empty_like(greedy)
        start = 0
        for rid, n, k in zip(request_ids, seq_lens, iterations, strict=True):
            sl = slice(start, start + n)
            start += n
            cfg = self._configs.get(rid, default)
            if cfg.temperature <= 0:
                sampled[:, sl] = greedy[:, sl]
                continue
            block = log_probs[:, sl, :]
            top = max(1, math.ceil(cfg.top_ratio * block.shape[-1]))
            val, ind = block.topk(top, dim=-1)
            filtered = torch.full_like(block, -float("inf"))
            filtered.scatter_(-1, ind, val)
            u = torch.rand(
                filtered.shape, dtype=filtered.dtype, device=filtered.device,
                generator=self._generator(cfg.seed, k),
            )
            gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)
            sampled[:, sl] = (filtered / cfg.temperature + gumbel).argmax(dim=-1)
        return sampled

    def _config(self, rid: str) -> DiffusionSamplingReqConfig:
        return self._configs.get(rid) or DiffusionSamplingReqConfig()

    def _generator(self, seed: int | None, iteration: int) -> torch.Generator | None:
        """This request's stream for this iteration, or None when unseeded.

        Mixing the iteration in rather than advancing one long-lived stream:
        async scheduling does not promise that a request's steps run in order
        relative to anyone else's, so a stream that depended on call order
        would not replay.
        """
        if seed is None:
            return None
        generator = torch.Generator(device=self._device)
        generator.manual_seed((int(seed) * 1_000_003 + int(iteration)) & 0x7FFF_FFFF_FFFF_FFFF)
        return generator
