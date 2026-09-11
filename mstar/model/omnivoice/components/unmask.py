"""The unmask step: CFG scoring, confidence ranking, and the reveal schedule.

An exact port of OmniVoice's ``_predict_tokens_with_scoring`` and the schedule
arithmetic inside ``_generate_iterative`` (``omnivoice/models/omnivoice.py``).
Parity against the reference is decided here, so the operation order is kept
verbatim even where it could be tidied — the log_softmax is applied twice on the
CFG path because the reference applies it twice, and the mask id is driven to
-inf after the combine rather than before.

Nothing here touches the network or the engine; it is pure tensor math over one
request's logits, so the parity test can drive it directly.
"""

import math

import torch
import torch.nn.functional as F


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


def build_reveal_schedule(
    target_len: int,
    num_codebook: int,
    num_step: int,
    t_shift: float,
) -> list[int]:
    """How many of the ``target_len * num_codebook`` cells to reveal per step.

    The per-step count follows the gap between consecutive timesteps, rounded
    up and clamped by what is left; the final step takes the entire remainder so
    the canvas is always fully revealed regardless of rounding drift.
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
    return schedule


def filter_top_k(logits: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    """Keep the top ``ratio`` of the vocabulary, -inf elsewhere."""
    k = math.ceil(ratio * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


def gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise at ``temperature``; caller takes the argmax."""
    scaled_logits = logits / temperature
    u = torch.rand_like(scaled_logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise


def predict_tokens_with_scoring(
    c_logits: torch.Tensor,
    u_logits: torch.Tensor,
    audio_mask_id: int,
    guidance_scale: float,
    class_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CFG-combine conditional and unconditional logits, then rank.

    Args:
        c_logits: conditional logits, ``[1, C, T, V]``.
        u_logits: unconditional logits, same shape.  Ignored when
            ``guidance_scale`` is 0.
        audio_mask_id: the MASK class, forced to -inf so it is never predicted.
        guidance_scale: 0 disables CFG entirely (and the caller may then skip
            the unconditional forward).
        class_temperature: 0 takes the argmax; above 0 samples from the top
            decile with Gumbel noise.

    Returns:
        ``(pred_tokens, confidence_scores)``, each ``[1, C, T]``.  The scores
        are max log-probabilities and drive which cells get revealed.
    """
    if guidance_scale != 0:
        c_log_probs = F.log_softmax(c_logits, dim=-1)
        u_log_probs = F.log_softmax(u_logits, dim=-1)
        log_probs = torch.log_softmax(
            c_log_probs + guidance_scale * (c_log_probs - u_log_probs),
            dim=-1,
        )
    else:
        log_probs = F.log_softmax(c_logits, dim=-1)

    log_probs[..., audio_mask_id] = -float("inf")

    if class_temperature > 0.0:
        filtered_probs = filter_top_k(log_probs, ratio=0.1)
        pred_tokens = gumbel_sample(filtered_probs, class_temperature).argmax(dim=-1)
    else:
        pred_tokens = log_probs.argmax(dim=-1)

    confidence_scores = log_probs.max(dim=-1)[0]
    return pred_tokens, confidence_scores


def apply_reveal(
    tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    scores: torch.Tensor,
    reveal_count: int,
    audio_mask_id: int,
    layer_penalty_factor: float,
    position_temperature: float,
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
        pred_tokens, scores: ``[1, C, T]`` from ``predict_tokens_with_scoring``.
    """
    if reveal_count <= 0:
        return tokens

    num_codebook = tokens.shape[1]
    layer_ids = torch.arange(num_codebook, device=tokens.device).view(1, -1, 1)
    scores = scores - (layer_ids * layer_penalty_factor)

    if position_temperature > 0.0:
        scores = gumbel_sample(scores, position_temperature)

    scores = scores.masked_fill(tokens != audio_mask_id, -float("inf"))

    _, topk_idx = torch.topk(scores.flatten(), reveal_count)
    flat_tokens = tokens.flatten()
    flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
    tokens.copy_(flat_tokens.view_as(tokens))
    return tokens
