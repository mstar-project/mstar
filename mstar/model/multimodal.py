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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Parallel, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.keys import MIXED

if TYPE_CHECKING:
    from mstar.conductor.request_info import CurrentForwardConductorMetadata
    from mstar.graph.base import TensorPointerInfo
    from mstar.model.base import ForwardPassArgs

TEXT = "text"
VIDEO = "video"


@dataclass(frozen=True)
class PromptPart:
    """One element of a prompt, in request order.

    ``index`` is the position within its own modality's list, so a media part
    addresses ``tensors[f"{modality}_inputs"][index]``. ``role`` is the chat
    role of the message the part came from, ``None`` from entrypoints with no
    messages.
    """

    modality: str
    text: str | None = None
    index: int = 0
    role: str | None = None


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
    input_modalities: list[str],
    texts: Iterable[str] | str | None = None,
    roles: Iterable[str | None] | None = None,
) -> list[PromptPart]:
    """Rebuild the parts for a layout, filling the text slots from ``texts``.

    ``input_modalities`` is one entry per part, in order, and is the layout
    every consumer plans from. Rebuilding from it beats carrying a second copy
    of the ordering that could drift. Unfilled text slots carry ``None``, which
    is enough to plan with. ``roles`` is one per part, in the same order.
    """
    if isinstance(texts, str):
        texts = [texts]
    remaining = iter(texts or ())
    part_roles = iter(roles or ())
    parts: list[PromptPart] = []
    seen: dict[str, int] = {}
    for modality in input_modalities:
        role = next(part_roles, None)
        if modality == TEXT:
            parts.append(PromptPart(modality=TEXT, text=next(remaining, None), role=role))
        else:
            index = seen.get(modality, 0)
            parts.append(PromptPart(modality=modality, index=index, role=role))
            seen[modality] = index + 1
    return parts


class Turn(NamedTuple):
    """The parts of consecutive messages of one role."""

    role: str
    parts: list[PromptPart]


def chat_turns(parts: list[PromptPart]) -> list[Turn]:
    """Group ``parts`` into turns, starting a new one at every role change.

    A part with no role is the user's, so a request from an entrypoint with no
    messages renders as the one user turn it always did.
    """
    turns: list[Turn] = []
    for part in parts:
        role = part.role or "user"
        if not turns or turns[-1].role != role:
            turns.append(Turn(role, []))
        turns[-1].parts.append(part)
    return turns


def messages_from_parts(
    parts: list[PromptPart], default_system: str | None = None,
) -> list[dict]:
    """One OpenAI-style message per turn of ``parts``, for a chat template to take as is.

    An attachment is an empty ``{"type": modality, modality: ""}`` item: the
    template writes its placeholder there, in the turn it belongs to, and the
    tokenized prompt is scanned for it. ``default_system`` leads only when no
    part is a system message's, wherever that message sits: a template writes
    every system message it is given, so both would reach the model.
    """
    messages = [
        {"role": turn.role, "content": [
            {"type": TEXT, "text": part.text or ""} if part.modality == TEXT
            else {"type": part.modality, part.modality: ""}
            for part in turn.parts
        ]}
        for turn in chat_turns(parts)
    ]
    if default_system is not None and not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": default_system})
    return messages


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


class MediaPads(NamedTuple):
    """A prompt with each media placeholder expanded to its tokens, and where
    each expanded run sits. The lists run in prompt order, one entry a run."""

    ids: torch.Tensor
    # the run's pad id, and its index among that pad's runs
    pad: list[int]
    index: list[int]
    # the run's first pad position in ``ids``, and its length
    offset: list[int]
    tokens: list[int]


def expand_media_pads(
    ids: torch.Tensor,
    counts: Mapping[int, Sequence[int]],
    sentinels: Mapping[int, tuple[int, int]] | None = None,
) -> MediaPads:
    """Expand the nth occurrence of each pad id in ``ids`` to ``counts[pad][n]``
    pads, wrapped in ``sentinels[pad]`` when given (a template that writes a
    bare pad, as the processor's ``replace_*_token`` then wraps it).

    Raises ``ValueError`` when a pad occurs other than once per count: user
    text that spells a placeholder token, or a layout naming an attachment
    the request did not carry. ``ids`` are host ids, straight from the
    tokenizer.
    """
    sentinels = sentinels or {}
    # a loop over the placeholders, not the tokens: a text-only prompt has none
    hits = torch.isin(ids, ids.new_tensor(list(counts))).nonzero().flatten() if counts else ids.new_empty(0)
    seen = dict.fromkeys(counts, 0)
    pieces: list[torch.Tensor] = []
    layout = MediaPads(ids, [], [], [], [])
    cursor = size = 0
    for i, token in zip(hits.tolist(), ids[hits].tolist(), strict=True):
        n = seen[token]
        if n >= len(counts[token]):
            raise ValueError(
                f"the prompt holds more placeholders {token} than the "
                f"{len(counts[token])} attachments it has tokens for"
            )
        seen[token] = n + 1
        pieces.append(ids[cursor:i])
        size += i - cursor
        start, end = sentinels.get(token, (None, None))
        if start is not None:
            pieces.append(ids.new_tensor([start]))
            size += 1
        tokens = int(counts[token][n])
        layout.pad.append(token)
        layout.index.append(n)
        layout.offset.append(size)
        layout.tokens.append(tokens)
        pieces.append(ids.new_full((tokens,), token))
        size += tokens
        if end is not None:
            pieces.append(ids.new_tensor([end]))
            size += 1
        cursor = i + 1
    short = {pad: (seen[pad], len(c)) for pad, c in counts.items() if seen[pad] != len(c)}
    if short:
        raise ValueError(f"placeholder counts in the prompt (found, expected) per pad id: {short}")
    if not pieces:
        return layout
    return layout._replace(ids=torch.cat([*pieces, ids[cursor:]]))


# ---------------------------------------------------------------------------
# The VLM walk set: `prefill_text` for a text-only prompt; for a prompt with
# media, one walk that runs the encoders of the media it holds, then the LLM
# over the whole prompt as one row; then a `decode` loop. The LLM splices
# each encoder's output in at its pads inside its forward (see
# `SampledTokenDecoder`).
# ---------------------------------------------------------------------------

# the step metadata key of a media walk's layout (`media_layout`)
MEDIA_LAYOUT = "media_layout"


@dataclass(frozen=True)
class MediaEncoder:
    """An encoder node of the media walks, as data.

    ``name`` names the walks (``prefill_vision``, ``prefill_vision_audio``);
    ``node`` reads ``inputs`` and sends the LLM ``output``, one row per pad
    of ``pad_ids`` in the prompt, in prompt order.
    """

    name: str
    node: str
    inputs: tuple[str, ...]
    output: str
    pad_ids: tuple[int, ...]

    def __post_init__(self):
        if not self.pad_ids:
            # the splice would have no pad to fill
            raise ValueError(f"media encoder {self.node!r} names no pad ids")


def vision_encoder(pad_ids: tuple[int, ...]) -> MediaEncoder:
    """The ``vision_encoder`` node every VLM here declares."""
    return MediaEncoder("vision", "vision_encoder", ("pixel_values",), "vision_embeds", pad_ids)


def media_walk_name(encoders: Sequence[MediaEncoder]) -> str:
    return "prefill_" + "_".join(e.name for e in encoders)


def media_walk_names(encoders: Sequence[MediaEncoder]) -> list[str]:
    """One walk per non-empty subset of ``encoders``, in declaration order."""
    return [
        media_walk_name([e for i, e in enumerate(encoders) if mask >> i & 1])
        for mask in range(1, 1 << len(encoders))
    ]


def media_layout(
    pads: MediaPads,
    output_of: Mapping[int, str],
    grids: Mapping[int, Sequence[Sequence[int]]] | None = None,
) -> dict[str, list]:
    """A media walk's ``MEDIA_LAYOUT`` step metadata, columnar: each run's
    ``offset`` and ``tokens``, the encoder ``output`` it is read from, and
    its patch ``grid`` (``[]`` for media without one). Plain lists: it rides
    the step's control message, so every node of the walk reads it on the
    host (`media_grids`)."""
    grids = grids or {}
    return {
        "offset": list(pads.offset),
        "tokens": list(pads.tokens),
        "output": [output_of[p] for p in pads.pad],
        "grid": [
            [int(v) for v in grids[p][i]] if p in grids else []
            for p, i in zip(pads.pad, pads.index, strict=True)
        ],
    }


def media_grids(layout: Mapping, output: str) -> list[tuple[int, int, int]]:
    """The patch grid of each run a ``MEDIA_LAYOUT`` fills from encoder output
    ``output``, in prompt order."""
    return [
        tuple(grid) for grid, out in zip(layout.get("grid", ()), layout.get("output", ()), strict=True)
        if out == output
    ]


def media_encoders(encoders: Sequence[MediaEncoder], layout: Mapping) -> list[MediaEncoder]:
    """The encoders a prompt's ``MEDIA_LAYOUT`` runs, in declaration order:
    those whose output fills one of its runs. Both the prefill walk and
    the KV layout read this, so they name the same walk."""
    outputs = set(layout.get("output", ()))
    return [e for e in encoders if e.output in outputs]


def media_walks(
    max_iters: int, encoders: Sequence[MediaEncoder] = (),
) -> dict[str, GraphSection]:
    """``prefill_text``, one media walk per subset of ``encoders``, and
    ``decode``, over the ``LLM`` node and the encoders. ``max_iters`` bounds
    the decode loop; it is a safety bound, not the per-request budget."""

    def emit_first_token() -> GraphEdge:
        return GraphEdge(
            next_node=EMIT_TO_CLIENT, name="new_token", persist=True, output_modality="text",
        )

    walks: dict[str, GraphSection] = {
        "prefill_text": GraphNode(name="LLM", input_names=["text_inputs"], outputs=[emit_first_token()]),
    }
    for mask in range(1, 1 << len(encoders)):
        present = [e for i, e in enumerate(encoders) if mask >> i & 1]
        nodes = [
            GraphNode(
                name=e.node, input_names=list(e.inputs),
                outputs=[GraphEdge(next_node="LLM", name=e.output)],
            )
            for e in present
        ]
        walks[media_walk_name(present)] = Sequential([
            nodes[0] if len(nodes) == 1 else Parallel(nodes),
            GraphNode(
                name="LLM",
                input_names=["text_inputs", *(e.output for e in present)],
                outputs=[emit_first_token()],
            ),
        ])
    walks["decode"] = Loop(
        name="decode_loop",
        section=GraphNode(
            name="LLM",
            input_names=["text_inputs"],
            outputs=[
                GraphEdge(next_node="LLM", name="text_inputs"),
                GraphEdge(next_node=EMIT_TO_CLIENT, name="text_inputs", output_modality="text"),
            ],
        ),
        max_iters=max_iters,
        outputs=[],
    )
    return walks


def text_combined_walks() -> dict[str, dict[str, set[str]]]:
    """Text prefill and decode rows of the ``LLM`` node share a step."""
    return {"LLM": {MIXED: {"prefill_text", "decode"}}}


@dataclass(frozen=True)
class PrefillStep:
    walk: str
    # (node, input name) -> the tensors that input carries
    input_tensors: dict[tuple[str, str], list[TensorPointerInfo]]
    # merged into the walk's step metadata
    metadata: dict = field(default_factory=dict)


def media_prefill_step(
    signals: dict[str, list[TensorPointerInfo]],
    encoders: Sequence[MediaEncoder],
    layout: dict | None = None,
    *,
    model_name: str,
) -> PrefillStep:
    """The one prefill walk a request runs over its whole prompt
    (``text_inputs``, one tensor): with media in its ``layout``, the walk over
    the encoders it runs (`media_encoders`); without, ``prefill_text``."""
    texts = signals.get("text_inputs", [])
    if not texts:
        # the conductor answers this with a 400
        raise ValueError(
            f"{model_name} has nothing to prefill: the request carries no "
            f"{sorted(signals) or 'inputs'}. An empty prompt cannot be served."
        )
    if len(texts) != 1:
        # the model's intake renders one prompt; the conductor answers this with a 400
        raise ValueError(f"{model_name} got a prompt as {len(texts)} text tensors, not the one whole prompt")
    present = media_encoders(encoders, layout or {})
    if not present:
        sent = sorted(name for e in encoders for name in e.inputs if signals.get(name))
        if sent:
            # a text prefill would drop them
            raise ValueError(f"{model_name} got {sent} with no media layout to place them by")
        return PrefillStep("prefill_text", {("LLM", "text_inputs"): texts})
    inputs = {("LLM", "text_inputs"): texts}
    for e in present:
        for name in e.inputs:
            inputs[(e.node, name)] = signals.get(name, [])
    return PrefillStep(media_walk_name(present), inputs, {MEDIA_LAYOUT: layout or {}})


def walk_input_edges(step: PrefillStep) -> list[GraphEdge]:
    """One edge per (node, input) the step's walk reads."""
    edges = []
    for (node, name), tensors in step.input_tensors.items():
        edge = GraphEdge(next_node=node, name=name)
        edge.tensor_info = list(tensors)
        edges.append(edge)
    return edges


def _prefill_args(
    metadata: CurrentForwardConductorMetadata, inputs: list[GraphEdge], step_metadata: dict,
) -> ForwardPassArgs:
    from mstar.model.base import ForwardPassArgs

    return ForwardPassArgs(
        full_metadata=metadata,
        inputs=inputs,
        unpersist_tensors=sum((inp.tensor_info for inp in inputs), start=[]),
        step_metadata={"is_prefill": metadata.is_prefill, **step_metadata},
    )


def initial_prefill_args(
    step: PrefillStep, input_modalities: list[str], output_modalities: list[str],
) -> ForwardPassArgs:
    """The prefill walk; `next_prefill_args` takes it from there."""
    from mstar.conductor.request_info import CurrentForwardConductorMetadata

    metadata = CurrentForwardConductorMetadata(
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        graph_walk=step.walk,
        is_prefill=True,
    )
    return _prefill_args(metadata, walk_input_edges(step), step.metadata)


def next_prefill_args(
    metadata: CurrentForwardConductorMetadata,
    persist_signals: dict[str, list[TensorPointerInfo]],
) -> ForwardPassArgs:
    """After the prefill, loop in decode until the decoder's stop check ends it."""
    from mstar.model.base import ForwardPassArgs

    if metadata.is_prefill:
        metadata.is_prefill = False
        metadata.graph_walk = "decode"
        edge = GraphEdge(next_node="LLM", name="text_inputs")
        edge.tensor_info = persist_signals.get("new_token", [])
        return _prefill_args(metadata, [edge], {})
    # reaching here again after decode stopped means the request is finished
    metadata.kwargs["decode_finished"] = True
    return ForwardPassArgs(
        full_metadata=metadata, inputs=[], unpersist_tensors=[], request_done=True,
    )
