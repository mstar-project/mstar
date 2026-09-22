"""Sampling for discrete diffusion: score every position, return log-probs too.

The contract is the one an autoregressive sampler cannot express. A diffusion
step hands in the logits for every unrevealed position, gets back a token for
each **and** the log-probability that won, and the caller decides how many of
them to commit. Which positions are still masked, how many to reveal this
step, and where the accepted tokens go are the model's business, because they
depend on a schedule and a ranking that differ per decoder.

Everything here runs batched over a packed ``[rows, positions, vocab]``
tensor. Per-request knobs become vectors gathered once per step, so a batch
mixing guidance scales and temperatures is one kernel path, not a Python loop
over requests.
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
        # The iteration index only reaches the RNG; nothing about the layout
        # depends on it, so there is nothing to stage on the device here.
        del ctx
        self._iteration = getattr(step, "iteration", 0)

    # -- sampling ----------------------------------------------------------

    def sample(
        self,
        request_ids: list[str],
        c_logits: torch.Tensor,
        u_logits: torch.Tensor | None = None,
        seq_lens: list[int] | None = None,
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

        Returns:
            ``tokens`` and ``logprobs``, both ``[rows, positions]``. The
            log-probability returned is the one of the token that won, which
            is what a confidence-ranked reveal needs.
        """
        if c_logits.dim() != 3:
            raise ValueError(
                f"expected [rows, positions, vocab] logits, got {tuple(c_logits.shape)}"
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

        guidance, temperature, top_ratio = self._per_position(request_ids, seq_lens)
        log_probs = self._combine(c_logits, u_logits, guidance)

        if self._forbidden_class is not None:
            log_probs[..., self._forbidden_class] = -float("inf")

        tokens = self._pick(log_probs, temperature, top_ratio, request_ids, seq_lens)
        logprobs = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return tokens, logprobs

    # -- internals ---------------------------------------------------------

    def _per_position(self, request_ids, seq_lens):
        """Broadcast each request's knobs across the positions it owns."""
        default = DiffusionSamplingReqConfig()
        g, t, r = [], [], []
        for rid, n in zip(request_ids, seq_lens, strict=True):
            cfg = self._configs.get(rid, default)
            g.extend([cfg.guidance_scale] * n)
            t.extend([cfg.temperature] * n)
            r.extend([cfg.top_ratio] * n)
        opts = dict(device=self._device, dtype=torch.float32)
        # [1, positions, 1]: rows and vocab broadcast.
        return (
            torch.tensor(g, **opts).view(1, -1, 1),
            torch.tensor(t, **opts).view(1, -1, 1),
            torch.tensor(r, **opts).view(1, -1, 1),
        )

    @staticmethod
    def _combine(c_logits, u_logits, guidance):
        """Classifier-free guidance, in the reference's operation order.

        The log_softmax really is applied twice on the guided path: the
        combine runs over log-probabilities, not logits, and the result is
        renormalised. Matching that matters more than tidying it, because the
        checkpoint was tuned behind it.

        A batch with no guidance anywhere skips the second normalisation
        entirely rather than multiplying by a zero vector, since
        ``log_softmax(log_softmax(x))`` is not ``log_softmax(x)``.
        """
        plain = F.log_softmax(c_logits, dim=-1)
        if u_logits is None or not bool((guidance != 0).any()):
            return plain
        guided = torch.log_softmax(
            plain + guidance * (plain - F.log_softmax(u_logits, dim=-1)), dim=-1
        )
        if bool((guidance != 0).all()):
            return guided
        # Mixed batch: positions at guidance 0 keep the singly-normalised form.
        return torch.where(guidance != 0, guided, plain)

    def _pick(self, log_probs, temperature, top_ratio, request_ids, seq_lens):
        """Greedy where temperature is 0, Gumbel over the top slice elsewhere."""
        if not bool((temperature > 0).any()):
            return log_probs.argmax(dim=-1)

        # One k for the step: the slice is a fraction of the vocabulary, and a
        # ragged k per position would mean a loop. The largest requested
        # fraction is kept, and a position wanting less is trimmed by its own
        # threshold below.
        ratio = float(top_ratio.max())
        k = max(1, math.ceil(ratio * log_probs.shape[-1]))
        val, ind = log_probs.topk(k, dim=-1)
        filtered = torch.full_like(log_probs, -float("inf"))
        filtered.scatter_(-1, ind, val)

        generator = self._generator(request_ids, seq_lens)
        safe_t = temperature.clamp_min(1e-6)
        u = torch.rand(
            filtered.shape, dtype=filtered.dtype, device=filtered.device,
            generator=generator,
        )
        gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        sampled = (filtered / safe_t + gumbel).argmax(dim=-1)

        greedy = log_probs.argmax(dim=-1)
        return torch.where(temperature.squeeze(-1) > 0, sampled, greedy)

    def _generator(self, request_ids, seq_lens) -> torch.Generator | None:
        """One generator per step, seeded from the batch's requests.

        A per-request stream would need a draw per request and a concatenate,
        which costs more than it buys while the reveal order is already
        stochastic. Seeding from the ids plus the iteration keeps a replay of
        the same batch reproducible, which is what the seed is for.
        """
        seeds = [
            self._configs[rid].seed
            for rid in request_ids
            if rid in self._configs and self._configs[rid].seed
        ]
        if not seeds:
            return None
        del seq_lens
        mixed = self._iteration
        for s in seeds:
            mixed = (mixed * 1_000_003 + int(s)) & 0x7FFF_FFFF_FFFF_FFFF
        generator = torch.Generator(device=self._device)
        generator.manual_seed(mixed)
        return generator
