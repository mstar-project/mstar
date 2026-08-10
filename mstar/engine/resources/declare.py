"""Step declarations for runner-driven models.

A migrated model does not sequence lifecycle stages. It declares what its
step is: which cache streams it touches, what spans they grow by, which
attention and rope plans back it, which streams fork, and what commits
when the step lands. The runner drives the declaration against the step
surface; the values below are immutable and valid for one step.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlanSpec:
    """One attention (and optional rope) plan of a declared step.

    A plain plan (``combined`` False) covers exactly one label. A combined
    plan batches every (label, request) pair label-major into one plan
    under ``combined_key``; it may carry a single label (a batched forward
    whose guidance branch is off still runs against the combined key).
    ``spans`` gives each label's per-request token counts in batch order.
    ``rope_pos_ids`` optionally supplies explicit position ids (a tensor
    for a plain plan, a per-label list dict for a combined one); None lets
    the embedder build them from the pool's counters. ``commit`` says
    whether the step's spans become part of the streams' history;
    ``pos_advance`` overrides the per-request position advance when it
    differs from the span.
    """
    labels: tuple[str, ...]
    spans: dict[str, tuple[int, ...]]
    is_causal: bool = True
    write_store: bool = True
    dense_gen: bool = False
    rope: bool = False
    rope_pos_ids: "dict[str, list[torch.Tensor]] | torch.Tensor | None" = None
    combined: bool = False
    combined_key: str = "_cfg_batched"
    commit: bool = True
    pos_advance: tuple[int, ...] | None = None


@dataclass(frozen=True)
class StepDeclaration:
    """A model's declaration of one step.

    ``plans`` run in order. ``pre_forks`` are (from_label, to_label)
    pairs applied before anything plans; ``post_forks`` are applied at
    commit, after the committed spans have landed (a fork copies the
    source stream as it stands, so a post fork sees the step's writes).
    """
    plans: tuple[PlanSpec, ...]
    pre_forks: tuple[tuple[str, str], ...] = ()
    post_forks: tuple[tuple[str, str], ...] = ()
