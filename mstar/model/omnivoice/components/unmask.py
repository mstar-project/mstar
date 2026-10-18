"""The reveal: which scored cells get committed, and how many per step.

Scoring itself is not here. It moved to ``DiffusionSamplerResource``, which
does the CFG combine and the token draw batched across a step's requests;
what is left is the part that belongs to this decoder rather than to a
sampler. The schedule arithmetic is an exact port of ``_generate_iterative``
(``omnivoice/models/omnivoice.py``) and the ranking of
``_predict_tokens_with_scoring``'s tail, so the operation order is kept
verbatim even where it could be tidied.

Nothing here touches the network or the engine; it is pure tensor math over
one request's canvas, so the parity test can drive it directly.
"""

import math
from functools import lru_cache

import torch


def get_time_steps(
    num_step: int,
    t_shift: float,
    t_start: float = 0.0,
    t_end: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """The shifted timestep ramp, ``num_step + 1`` points on [t_start, t_end].

    ``t_shift`` below 1 pushes points toward t_start, spending more of the
    budget at low SNR where the canvas is still mostly masked.
    """
    device = device or torch.device("cpu")
    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)
    return t_shift * timesteps / (1 + (t_shift - 1) * timesteps)


@lru_cache(maxsize=256)
def build_reveal_schedule(
    target_len: int,
    num_codebook: int,
    num_step: int,
    t_shift: float,
) -> tuple[int, ...]:
    """How many of the ``target_len * num_codebook`` cells to reveal per step.

    The per-step count follows the gap between consecutive timesteps, rounded
    up and clamped by what is left; the final step takes the entire remainder so
    the canvas is always fully revealed regardless of rounding drift.

    Cached, and returning a tuple so it stays immutable: the schedule depends
    only on these four numbers, but ``postprocess`` needs it on every step of
    every request, and recomputing it there means a linspace and a Python loop
    per step for a list that never changes within a request.
    """
    timesteps = get_time_steps(num_step=num_step, t_shift=t_shift).tolist()
    total = target_len * num_codebook
    remaining = total
    schedule: list[int] = []
    for step in range(num_step):
        if step == num_step - 1:
            count = remaining
        else:
            count = min(
                math.ceil(total * (timesteps[step + 1] - timesteps[step])),
                remaining,
            )
        count = int(count)
        schedule.append(count)
        remaining -= count
    return tuple(schedule)



def gumbel_sample(
    logits: torch.Tensor,
    temperature: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add Gumbel noise at ``temperature``; caller takes the argmax.

    ``generator`` carries the request's seed. Both draws in a step (token
    choice and reveal order) come from it, so a seeded request replays
    exactly; passing ``None`` keeps the global RNG and stays unreproducible.
    """
    scaled_logits = logits / temperature
    u = torch.rand(
        scaled_logits.shape,
        dtype=scaled_logits.dtype,
        device=scaled_logits.device,
        generator=generator,
    )
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise



def apply_reveal(
    tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    scores: torch.Tensor,
    reveal_count: int,
    audio_mask_id: int,
    layer_penalty_factor: float,
    position_temperature: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Reveal ``reveal_count`` cells of ``tokens`` in place and return it.

    Ranking is over the flattened ``(C, T)`` canvas, so a step may spend its
    whole budget on one codebook row.  Two biases shape the choice, both from
    the reference:

    - ``layer_penalty_factor`` subtracts ``layer_index * factor`` from the
      score, pushing the earliest codebooks to resolve first (they carry the
      coarse content the later residual rows refine).
    - ``position_temperature`` adds Gumbel noise to the *position* ranking, so
      the reveal order is stochastic even when token choice is greedy.

    Already-revealed cells are driven to -inf so a cell is only written once.

    Args:
        tokens: ``[1, C, T]``, the request's live canvas, modified in place.
        pred_tokens, scores: ``[1, C, T]`` from the diffusion sampler.
        generator: the request's seeded RNG, or ``None`` for the global one.
    """
    if reveal_count <= 0:
        return tokens

    num_codebook = tokens.shape[1]
    layer_ids = torch.arange(num_codebook, device=tokens.device).view(1, -1, 1)
    scores = scores - (layer_ids * layer_penalty_factor)

    if position_temperature > 0.0:
        scores = gumbel_sample(scores, position_temperature, generator)

    scores = scores.masked_fill(tokens != audio_mask_id, -float("inf"))

    _, topk_idx = torch.topk(scores.flatten(), reveal_count)
    flat_tokens = tokens.flatten()
    flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
    tokens.copy_(flat_tokens.view_as(tokens))
    return tokens
