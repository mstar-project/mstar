"""Order-preserving multimodal prompt adapter.

Intake hands models an ordered list of :class:`PromptPart` — the user's text
and attachments as written — instead of a bag of per-modality lists.
:func:`prefill_plan` turns that into the ordered sequence of text spans and
attachments to prefill.

A model renders its prompt once with the tokenizer's own placeholders where
the attachments belong, tokenizes it once, and reads the placement back with
:func:`find_media_spans` / :func:`split_around_spans`. One tokenizer call, so
no BPE merge is cut at a span boundary, and a model says only how its prompt
renders — never how to take one apart.

Prompt building and schedule building both call :func:`prefill_plan`, so they
cannot disagree about the spans; :func:`check_plan` fails the request if a
rendered prompt ever does.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

TEXT = "text"


@dataclass(frozen=True)
class PromptPart:
    """One element of a prompt, in request order.

    ``index`` is the position within its own modality's list, so a media part
    addresses ``tensors[f"{modality}_inputs"][index]``.
    """

    modality: str
    text: str | None = None
    index: int = 0


@dataclass(frozen=True)
class MediaSpan:
    """A ``<|x_start|> pad* <|x_end|>`` run located in a tokenized prompt.

    Bounds are sentinel-inclusive: the modality walk emits those embeddings
    itself, alongside the encoder output.
    """

    modality: str
    index: int
    start: int
    stop: int


def parts_from_modalities(
    input_modalities: list[str], texts: Iterable[str] | str | None = None
) -> list[PromptPart]:
    """Rebuild the parts for a layout, filling the text slots from ``texts``.

    ``input_modalities`` is one entry per part, in order, and is the layout
    every consumer plans from. Rebuilding from it beats carrying a second copy
    of the ordering that could drift. Unfilled text slots carry ``None``, which
    is enough to plan with.
    """
    if isinstance(texts, str):
        texts = [texts]
    remaining = iter(texts or ())
    parts: list[PromptPart] = []
    seen: dict[str, int] = {}
    for modality in input_modalities:
        if modality == TEXT:
            parts.append(PromptPart(modality=TEXT, text=next(remaining, None)))
        else:
            index = seen.get(modality, 0)
            parts.append(PromptPart(modality=modality, index=index))
            seen[modality] = index + 1
    return parts


def prefill_plan(
    parts: list[PromptPart], *, leading_text: bool = True
) -> list[PromptPart]:
    """Order the prefill segments for ``parts``.

    Adjacent text collapses into one segment: the rendered prompt puts it in
    one contiguous run. A trailing segment always exists — the template's turn
    close — even with no user text there. ``leading_text`` says whether the
    template opens with one too; BAGEL's generation prompt does not.

    Text segments are numbered in order, so ``index`` addresses the model's own
    list of spans, and each carries its text (``None`` when only the template
    contributes it).
    """
    plan: list[PromptPart] = []
    n_text = 0
    pending = leading_text
    run: list[str] = []

    def flush() -> None:
        nonlocal n_text, pending, run
        plan.append(
            PromptPart(modality=TEXT, text="".join(run) or None, index=n_text)
        )
        n_text += 1
        pending = False
        run = []

    for part in parts:
        if part.modality == TEXT:
            pending = True
            if part.text:
                run.append(part.text)
            continue
        if pending:
            flush()
        plan.append(part)
    flush()
    return plan


def find_media_spans(
    input_ids: torch.Tensor, specs: dict[str, tuple[int, int, int]]
) -> list[MediaSpan]:
    """Locate every placeholder run in ``input_ids``, in token order.

    ``specs`` maps a modality to its ``(start_id, pad_id, end_id)`` triple.
    Indices run per modality, so the nth image span addresses the nth image.
    """
    by_pad = {pad: (modality, start, end) for modality, (start, pad, end) in specs.items()}
    ids = input_ids.tolist()
    spans: list[MediaSpan] = []
    seen: dict[str, int] = {}
    i, n = 0, len(ids)
    while i < n:
        entry = by_pad.get(ids[i])
        if entry is None:
            i += 1
            continue
        modality, start_id, end_id = entry
        j = i
        while j < n and ids[j] == ids[i]:
            j += 1
        if i == 0 or ids[i - 1] != start_id or j >= n or ids[j] != end_id:
            raise ValueError(
                f"{modality} placeholder run at {i}..{j} is not wrapped in its "
                "start/end sentinels"
            )
        index = seen.get(modality, 0)
        spans.append(MediaSpan(modality=modality, index=index, start=i - 1, stop=j + 1))
        seen[modality] = index + 1
        i = j + 1
    return spans


def split_around_spans(
    input_ids: torch.Tensor, spans: list[MediaSpan]
) -> list[torch.Tensor]:
    """Return the text between the spans, in order, dropping empty pieces.

    Two adjacent attachments leave nothing between them, and a zero-length walk
    has nothing to embed. :func:`prefill_plan` drops the same pieces.
    """
    segments: list[torch.Tensor] = []
    cursor = 0
    for span in spans:
        if span.start > cursor:
            segments.append(input_ids[cursor:span.start])
        cursor = span.stop
    if cursor < len(input_ids):
        segments.append(input_ids[cursor:])
    return segments


def check_attachments(parts: list[PromptPart], counts: dict[str, int]) -> None:
    """Fail loudly when the layout and the attachments that arrived disagree.

    :func:`check_plan` compares the plan against the prompt it rendered, but
    both sides come from the same parts — a layout claiming two images renders
    two placeholders and scans two spans whatever was uploaded. Only the caller
    knows how many attachments the request actually carried, so it checks here,
    at intake, where the failure can still be answered with a 400.
    """
    declared: dict[str, int] = {}
    for part in parts:
        if part.modality != TEXT:
            declared[part.modality] = declared.get(part.modality, 0) + 1
    for modality in sorted(set(declared) | set(counts)):
        want, got = declared.get(modality, 0), counts.get(modality, 0)
        if want != got:
            raise ValueError(
                f"multimodal input mismatch: the layout declares {want} "
                f"{modality} input(s), the request carries {got}"
            )


def check_plan(plan: list[PromptPart], spans: list[MediaSpan], n_text: int) -> None:
    """Fail loudly when the rendered prompt disagrees with the plan.

    A mismatch means an attachment was dropped or reordered between intake and
    tokenization — the failure this adapter exists to prevent.
    """
    planned = [(p.modality, p.index) for p in plan if p.modality != TEXT]
    scanned = [(s.modality, s.index) for s in spans]
    if planned != scanned:
        raise ValueError(
            f"multimodal placement mismatch: planned {planned}, prompt has {scanned}"
        )
    planned_text = sum(1 for p in plan if p.modality == TEXT)
    if planned_text != n_text:
        raise ValueError(
            f"multimodal placement mismatch: planned {planned_text} text spans, "
            f"prompt has {n_text}"
        )
