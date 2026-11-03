"""The unmask step: CFG scoring, confidence ranking, and the reveal schedule.

An exact port of OmniVoice's ``_predict_tokens_with_scoring`` and the schedule
arithmetic inside ``_generate_iterative`` (``omnivoice/models/omnivoice.py``).
Parity against the reference is decided here, so the arithmetic is kept to the
bit even where it could be tidied — the log_softmax is applied twice on the CFG
path because the reference applies it twice, and the mask id is driven to -inf
after the combine rather than before.

Nothing here touches the network or the engine; it is pure tensor math over a
step's packed logits, so the parity test can drive it directly.
"""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F

from mstar.utils.h2d import to_device_nonblocking


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
    only on these four numbers, but ``forward_batched`` needs it on every step
    of every request, and recomputing it there means a linspace and a Python loop
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


def filter_top_k(logits: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    """Keep the top ``ratio`` of the vocabulary, -inf elsewhere."""
    k = math.ceil(ratio * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


def _cat(parts: list[torch.Tensor], dim: int) -> torch.Tensor:
    """``torch.cat`` without the copy for a single part."""
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=dim)


def _per_position(values: list, lens: list[int], device: torch.device) -> float | torch.Tensor:
    """A per-request knob over the packed positions: the shared value as a
    Python float when every request has the same one (the reference's
    arithmetic to the bit), else each position's value, ``[P]``."""
    if len(set(values)) == 1:
        return values[0]
    return to_device_nonblocking(
        torch.repeat_interleave(torch.tensor(values), torch.tensor(lens)), device
    )


def unmask_step(
    logits: torch.Tensor,
    canvases: list[torch.Tensor],
    uncond_rows: list[int],
    counts: list[int],
    generators: list[torch.Generator | None],
    guidance_scale: list[float],
    class_temperature: list[float],
    layer_penalty_factor: list[float],
    position_temperature: list[float],
    audio_mask_id: int,
) -> torch.Tensor:
    """One unmask iteration for every request of a step, batched.

    The reference's scoring (``_predict_tokens_with_scoring``) and reveal for
    each request, over the step's packed target positions at once. Each request
    draws from its own generator in the reference's order (token choice, then
    reveal order), and its result is the reference's arithmetic to the bit
    when the step's requests share their knobs. A temperature that differs
    between them is divided by per position instead of as one scalar, which
    can round differently (CUDA divides by a scalar as a multiply by its
    reciprocal), so a request batched with others can then flip a reveal at a
    near-tie.

    Args:
        logits: ``[1, C, R, V]``: every request's conditional target block in
            order, then the unconditional blocks (``PackedCanvas``'s gather order).
        canvases: each request's live canvas, ``[C, T_i]``; not written.
        uncond_rows: where each request's unconditional block starts in
            ``logits``, or -1 for a request with CFG off.
        counts: the cells each request reveals this iteration.
        generators: each request's RNG for this iteration (``None``: global).

    Returns:
        ``[C, P]``: the next canvases end to end, ``P = sum(T_i)``.
    """
    lens = [canvas.shape[-1] for canvas in canvases]
    num_codebook, device, total = canvases[0].shape[0], logits.device, sum(lens)

    log_probs = F.log_softmax(logits[:, :, :total].float(), dim=-1)
    cfg = [row >= 0 for row in uncond_rows]
    if any(cfg):
        if all(cfg):
            u_logits = logits[:, :, total:2 * total]
        else:
            # a request without CFG reads its conditional rows; its result is discarded below
            starts = [row if row >= 0 else sum(lens[:i]) for i, row in enumerate(uncond_rows)]
            index = torch.cat([torch.arange(at, at + n) for at, n in zip(starts, lens, strict=True)])
            u_logits = logits.index_select(2, to_device_nonblocking(index, device))
        u_log_probs = F.log_softmax(u_logits.float(), dim=-1)
        scale = _per_position(guidance_scale, lens, device)
        scale = scale.view(1, 1, -1, 1) if torch.is_tensor(scale) else scale
        # log_probs + scale * (log_probs - u_log_probs), to the bit, in u_log_probs' buffer: every
        # float32 [C, P, V] copy at once is what max_packed_tokens has to leave room for
        guided = torch.log_softmax(u_log_probs.neg_().add_(log_probs).mul_(scale).add_(log_probs), dim=-1)
        del u_logits, u_log_probs
        if all(cfg):
            log_probs = guided
        else:
            on = _per_position(cfg, lens, device)
            log_probs = torch.where(on.view(1, 1, -1, 1), guided, log_probs)
        del guided
    log_probs[..., audio_mask_id] = -float("inf")

    pred_tokens = log_probs.argmax(dim=-1)
    sampled = [t > 0.0 for t in class_temperature]
    if any(sampled):
        shape = (1, num_codebook, 0, log_probs.shape[-1])
        noise = _cat([
            torch.rand(shape[:2] + (n,) + shape[3:], dtype=log_probs.dtype, device=device, generator=gen)
            if hot else log_probs.new_zeros(shape[:2] + (n,) + shape[3:])
            for n, hot, gen in zip(lens, sampled, generators, strict=True)
        ], dim=2)
        temp = _per_position([t if hot else 1.0 for t, hot in zip(class_temperature, sampled, strict=True)],
                             lens, device)
        temp = temp.view(1, 1, -1, 1) if torch.is_tensor(temp) else temp
        # plus Gumbel noise, -log(-log(u)), built in the noise's buffer like the combine above
        log_log = noise.add_(1e-10).log_().neg_().add_(1e-10).log_()
        drawn = filter_top_k(log_probs, ratio=0.1).div_(temp).sub_(log_log).argmax(-1)
        if all(sampled):
            pred_tokens = drawn
        else:
            on = _per_position(sampled, lens, device)
            pred_tokens = torch.where(on.view(1, 1, -1), drawn, pred_tokens)
    scores = log_probs.max(dim=-1)[0]

    layer_ids = torch.arange(num_codebook, device=device).view(1, -1, 1)
    penalty = _per_position(layer_penalty_factor, lens, device)
    scores = scores - (layer_ids * (penalty.view(1, 1, -1) if torch.is_tensor(penalty) else penalty))
    shuffled = [t > 0.0 and c > 0 for t, c in zip(position_temperature, counts, strict=True)]
    if any(shuffled):
        noise = _cat([
            torch.rand((1, num_codebook, n), dtype=scores.dtype, device=device, generator=gen)
            if hot else scores.new_zeros((1, num_codebook, n))
            for n, hot, gen in zip(lens, shuffled, generators, strict=True)
        ], dim=2)
        temp = _per_position([t if hot else 1.0 for t, hot in zip(position_temperature, shuffled, strict=True)],
                             lens, device)
        temp = temp.view(1, 1, -1) if torch.is_tensor(temp) else temp
        noisy = scores / temp - torch.log(-torch.log(noise + 1e-10) + 1e-10)
        if all(shuffled):
            scores = noisy
        else:
            on = _per_position(shuffled, lens, device)
            scores = torch.where(on.view(1, 1, -1), noisy, scores)
    tokens = _cat(canvases, dim=-1).unsqueeze(0)
    scores = scores.masked_fill(tokens != audio_mask_id, -float("inf"))

    # Request i reveals its top counts[i] cells of its own (C, T_i) block: sort
    # every cell by score, then stably by request, and take the first counts[i]
    # of each request's run. Static shapes, nothing read back on the host.
    order = torch.sort(scores.reshape(-1), descending=True, stable=True).indices
    rank = torch.arange(num_codebook * total, device=device)
    if len(lens) == 1:
        take = rank < counts[0]
    else:
        runs = [num_codebook * n for n in lens]
        host = to_device_nonblocking(
            torch.tensor([lens, counts, [sum(runs[:i]) for i in range(len(runs))]]), device
        )
        owner = torch.repeat_interleave(torch.arange(len(lens), device=device), host[0], output_size=total)
        owner = owner.expand(num_codebook, total).reshape(-1)
        order = order[torch.sort(owner[order], stable=True).indices]
        by_owner = owner[order]
        take = rank - host[2][by_owner] < host[1][by_owner]
    reveal = torch.zeros_like(take).scatter_(0, order, take)
    return torch.where(reveal.view(1, num_codebook, total), pred_tokens, tokens)[0]
