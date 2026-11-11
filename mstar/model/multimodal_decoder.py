"""`MultimodalDecoderModel`: the `Model` side of a VLM on the shared scaffold.

An ``LLM`` node (a `SampledTokenDecoder`) behind the model's media encoders,
on `mstar.model.multimodal`'s walks. The base owns what every such model
writes the same way: the walks and the mixed text step, the decode loop's
cap, media intake, the KV label and layout, the walk plan, the sampler
(its spec, and each request's config over the checkpoint's
generation_config defaults), sharding, the submodule cache, and where the
files come from. A model sets `NAME` and
`METADATA_FILES`, and writes its own ``__init__`` (calling this one, then
setting ``config``, with ``max_position_embeddings`` and ``stop_token_ids``,
``tokenizer`` and ``media``, a `MediaIntake` per attachment modality),
resources (the LLM's sampler from `_sampler_spec`), ``_prompt_ids`` (render
and tokenize) and ``_create_submodule``.

Not in ``multimodal.py``: ``mstar.model.base`` imports that module, so a
`Model` subclass there would be a circular import.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources.sampler.config import SamplerSpec, SamplingReqConfig
from mstar.engine.resources.spec import CombinedSplit, ResourceReqConfig
from mstar.graph.base import GraphSection, TensorPointerInfo
from mstar.model.base import KVLabel, LabelLayout, Model, PrefixKeys, ProcessPromptOutput, Span, WalkStep
from mstar.model.keys import KV_CACHE, MIXED, PREFILL_TEXT, SAMPLER
from mstar.model.multimodal import (
    MEDIA_LAYOUT,
    TEXT,
    VIDEO,
    MediaEncoder,
    MediaPads,
    MediaRun,
    PromptPart,
    check_attachments,
    expand_media_pads,
    media_encoders,
    media_layout,
    media_prefill_step,
    media_walk_name,
    media_walk_names,
    media_walks,
    parts_from_modalities,
    text_combined_walks,
)
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import TextOutput, load_generation_defaults

# the sampling knobs a request may set
SAMPLING_KEYS = (
    "temperature", "top_p", "top_k", "min_p",
    "repetition_penalty", "penalize_prompt", "ignore_eos",
    "return_logprobs", "logprobs_mode",
)

# A mixed step's prompt rows (as many as `max_rows_in_combined` lets in) take
# the prefill kernels, its decode rows the decode ones; for the ``splits`` of
# the LLM's attention resources.
MIXED_SPLIT = CombinedSplit(MIXED, prefill_walks=(PREFILL_TEXT,))


class PreparedMedia(NamedTuple):
    """One attachment, ready for its encoder: the encoder's input tensors,
    and the tokens of each run of pads it fills, in prompt order, with each
    run's grid for an encoder that reads one (a ViT's patch grid, an audio
    tower's mel frames per chunk) or a model that places media by grid
    (MRoPE). The grids ride the walk's ``MEDIA_LAYOUT``, host values every
    node reads without a device sync."""

    inputs: dict[str, torch.Tensor]
    tokens: list[int]
    grids: list[Sequence[int]] | None = None


@dataclass(frozen=True)
class MediaIntake:
    """How a prompt's attachments of one ``modality`` reach their encoder.

    Each attachment fills ``runs(attachment, metadata)`` placeholders
    ``pad`` in the rendered prompt (one when ``runs`` is None);
    ``prepare(attachment, metadata)`` gives its `PreparedMedia`, and each
    placeholder is expanded to its run's tokens, between ``sentinels`` when
    given. ``metadata`` is the attachment's loader metadata, ``{}`` when it
    came with none. Intakes may share a ``pad`` (and then its sentinels):
    the nth placeholder of a pad goes to the nth attachment of that pad in
    the order of the prompt's parts, which is where `_prompt_ids` renders
    each attachment.
    """

    modality: str
    encoder: MediaEncoder
    pad: int
    prepare: Callable[[torch.Tensor, dict], PreparedMedia]
    sentinels: tuple[int, int] | None = None
    runs: Callable[[torch.Tensor, dict], int] | None = None


class MediaPrompt(NamedTuple):
    """A prompt after intake: its ids with each placeholder expanded to its
    run (each run's encoder output and grid), and every encoder's inputs in
    prompt order."""

    pads: MediaPads
    inputs: NameToTensorList


class MultimodalDecoderModel(Model):
    DECODE_LOOP = "decode_loop"

    # Image preprocessing is a few small ops per request, and a thread per
    # core only slows them: each op waits on whichever thread a co-tenant
    # descheduled. At 64 threads c=1 TTFT p95 was 152 ms (some preprocesses
    # 35-130 ms); at 4, preprocessing held at 7-10 ms and p95 at 49 ms. One
    # thread is steadier but slower (13 ms).
    PREPROCESS_TORCH_THREADS = 4
    INTERNAL_KWARGS = (MEDIA_LAYOUT,)
    # in errors and logs
    NAME: str
    # what the API and conductor processes download: enough to build the
    # config and tokenize (generation_config.json, which `__init__` reads,
    # comes too); workers fetch the full snapshot (`_resolve_weights`)
    METADATA_FILES: tuple[str, ...]

    # one per attachment modality the model takes, set by its `__init__`
    media: tuple[MediaIntake, ...] = ()
    # False for a model every prompt of which carries media (an audio-only
    # STT checkpoint): no text-only prefill walk, no mixed step, no prefix keys
    # (only a text span is keyed), and its LLM node, given this as its
    # ``text_prompts``, captures neither. The walk plan refuses a text-only
    # request.
    TEXT_PROMPTS = True
    # `TextOutput.reasoning_end`
    REASONING_END: str | None = None
    # the checkpoint's sampling defaults, under whatever a request sets (set
    # by `__init__`; a model built without it reads none)
    sampling_defaults: Mapping[str, object] = MappingProxyType({})

    def __init__(self, model_path_hf: str, cache_dir: str | None = None):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.local_dir = self._resolve_model_metadata()
        # so weights are materialized at most once per node
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self.sampling_defaults = {
            k: v for k, v in load_generation_defaults(self.local_dir).items() if k in SAMPLING_KEYS
        }

    def _resolve_model_metadata(self) -> str:
        local_path = Path(self.model_path_hf)
        if local_path.is_dir():
            return str(local_path)
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=self.model_path_hf, cache_dir=self.cache_dir,
            allow_patterns=[*self.METADATA_FILES, "generation_config.json"],
        )

    def _resolve_weights(self) -> str:
        """The full snapshot; only workers call this."""
        local_path = Path(self.model_path_hf)
        if local_path.is_dir():
            return str(local_path)
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=self.model_path_hf, cache_dir=self.cache_dir)

    @property
    def encoders(self) -> tuple[MediaEncoder, ...]:
        """The media's encoders, each once (modalities may share one)."""
        return tuple(dict.fromkeys(m.encoder for m in self.media))

    def encoder(self, node_name: str) -> MediaEncoder:
        """The encoder that runs as graph node ``node_name``."""
        return next(e for e in self.encoders if e.node == node_name)

    # -----------------------------------------------------------------------
    # Graph
    # -----------------------------------------------------------------------

    def get_combined_graph_walks(self) -> dict[str, dict[str, set[str]]]:
        return text_combined_walks() if self.TEXT_PROMPTS else {}

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # a safety bound on decode, not the per-request budget
        return media_walks(self.config.max_position_embeddings, self.encoders, text=self.TEXT_PROMPTS)

    def rebuild_policies(self) -> dict[str, RebuildPolicy]:
        # A media prompt is rebuilt from host copies of its encoders' outputs, one embedding per pad. Its layout
        # places them, so a pad id sampled into the rebuild row's tail takes none and needs no `pad_ids`.
        media = media_walk_names(self.encoders)
        text = {PREFILL_TEXT} if self.TEXT_PROMPTS else set()
        return {"LLM": RebuildPolicy(
            walks=frozenset({*text, *media}), ids_input="text_inputs", sample_output="new_token",
            loop="decode_loop", copy_width={walk: self.config.hidden_size for walk in media},
        )}

    # -----------------------------------------------------------------------
    # KV layout
    # -----------------------------------------------------------------------

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        # A text prompt's one `ids` span is keyed for the prefix cache, and its replies extend the
        # keys. Nothing of a media prompt is, since its one step carries tensor inputs, which the
        # prefix cache never cuts.
        if not self.TEXT_PROMPTS:
            return {KV_CACHE: {"main": KVLabel(growth=(("decode", "decode_loop"),))}}
        return {KV_CACHE: {"main": KVLabel(
            growth=(("decode", "decode_loop"),),
            keys=PrefixKeys("text_inputs", (PREFILL_TEXT,), growth=True),
        )}}

    def checkpoint_path(self) -> str:
        return self._resolve_weights()

    def kv_layout(self, layout: dict, length: int) -> dict:
        """`process_prompt`'s ``kv_layout`` for a prompt of ``length`` tokens
        and its ``MEDIA_LAYOUT``: the whole prompt, written by its one prefill
        walk, then a slot per decode step. A media prompt's runs of pads are
        ``rows`` spans, which the encoders' outputs fill, and the text around
        them ``ids`` spans."""
        present = media_encoders(self.encoders, layout)
        if not present:
            span = Span("ids", PREFILL_TEXT, length, length, "text_inputs")
            return {KV_CACHE: {"main": LabelLayout(spans=(span,), growth=(("decode_loop", 1),))}}
        walk, spans, end = media_walk_name(present), [], 0
        for offset, tokens, grid in zip(layout["offset"], layout["tokens"], layout["grid"], strict=True):
            if offset > end:
                spans.append(Span("ids", walk, offset - end, offset - end, "text_inputs"))
            spans.append(Span("rows", walk, tokens, self.media_advance(tokens, grid), "text_inputs"))
            end = offset + tokens
        if length > end:
            spans.append(Span("ids", walk, length - end, length - end, "text_inputs"))
        # one row writes the whole prompt
        spans[1:] = [span._replace(opens_step=False) for span in spans[1:]]
        # Key no span. The default keys the leading text span, and `rebuild_refusal` refuses a prompt keyed only in
        # part.
        return {KV_CACHE: {"main": LabelLayout(spans=tuple(spans), growth=(("decode_loop", 1),), keyed_spans=0)}}

    def media_advance(self, tokens: int, grid: list[int]) -> int:
        """How far a run of ``tokens`` pads moves the position counter: one a
        pad, unless the model places media by their patch ``grid``."""
        return tokens

    # -----------------------------------------------------------------------
    # Intake
    # -----------------------------------------------------------------------

    def _prompt_ids(
        self, parts: list[PromptPart], raw: dict[str, list[torch.Tensor]], **kwargs,
    ) -> torch.Tensor:
        """The rendered prompt's host ids, each attachment's placeholders in
        place, the attachments rendered in the order of ``parts``: intake
        gives the nth placeholder of a pad to the nth attachment of that pad
        in part order (`MediaIntake`). ``raw`` is each modality's
        attachments; ``kwargs`` are `process_prompt`'s."""
        raise NotImplementedError

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> ProcessPromptOutput:
        """The prompt's ids, each attachment's encoder inputs, and the media
        layout (`intake`)."""
        prompt_in = self.intake(prompt, input_modalities, tensors, prompt_parts, **kwargs)
        layout = self.prompt_layout(prompt_in)
        # the raw attachments are spent once prepared (the encoders read their
        # inputs): don't hold them for the request's life
        spent = {
            f"{m.modality}_inputs": [] for m in self.media if (tensors or {}).get(f"{m.modality}_inputs")
        }
        return ProcessPromptOutput(
            {"text_inputs": [prompt_in.pads.ids], **spent, **prompt_in.inputs},
            {MEDIA_LAYOUT: layout, "kv_layout": self.kv_layout(layout, int(prompt_in.pads.ids.numel()))},
        )

    def intake(
        self,
        prompt: str | None,
        input_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> MediaPrompt:
        """Render once, tokenize once (`_prompt_ids`), then prepare each
        attachment in prompt order and expand its placeholders to their
        tokens. A bad request is a ``ValueError`` (a 400)."""
        media = {m.modality: m for m in self.media}
        if prompt is None:
            # An attachment-only message renders with an empty text span. With
            # no attachments either, fail here, where it reaches the client as
            # a 400, not in the conductor, where the request would hang.
            if not any(m != TEXT for m in input_modalities):
                choices = ", ".join(["a prompt", *(f"{'an' if m[0] in 'aeiou' else 'a'} {m}" for m in media)])
                raise ValueError(
                    f"{self.NAME} got a request with no text and no attachments; send {choices}, or a mix"
                )
            prompt = ""
        parts = parts_from_modalities(
            input_modalities,
            [p.text or "" for p in prompt_parts if p.modality == TEXT]
            if prompt_parts is not None else prompt,
            [p.role for p in prompt_parts or ()],
        )
        unsupported = {p.modality for p in parts} - {TEXT, *media}
        if unsupported:
            *rest, last = [TEXT, *media]
            taken = f"{', '.join(rest)} and {last} attachments" if rest else "text"
            raise ValueError(
                f"{self.NAME} here has no {', '.join(sorted(unsupported))} path; {taken} only"
            )
        raw = {m: (tensors or {}).get(f"{m}_inputs", []) for m in media}
        # always, or a named but unsent attachment is quietly dropped
        check_attachments(parts, {m: len(r) for m, r in raw.items()})
        loaded = kwargs.get("input_metadata") or {}

        def metadata(m: str, i: int) -> dict:
            given = loaded.get(f"{m}_inputs") or ()
            return given[i] if i < len(given) else {}

        ids = self._prompt_ids(parts, raw, **kwargs)
        # before any attachment is prepared: user text that spells a placeholder
        # (per pad id, which intakes may share)
        want: dict[int, int] = {}
        for m, intake in media.items():
            want[intake.pad] = want.get(intake.pad, 0) + sum(
                1 if intake.runs is None else intake.runs(raw[m][i], metadata(m, i)) for i in range(len(raw[m]))
            )
        for pad, n in want.items():
            found = int((ids == pad).sum())
            if found != n:
                names = " / ".join(m.modality for m in self.media if m.pad == pad)
                raise ValueError(
                    f"multimodal placement mismatch: the prompt holds {found} {names} "
                    f"placeholders for {n} {names} inputs"
                )
        sentinels: dict[int, tuple[int, int]] = {}
        for m in self.media:
            if m.sentinels is not None and sentinels.setdefault(m.pad, m.sentinels) != m.sentinels:
                raise ValueError(f"{self.NAME}: intakes sharing placeholder {m.pad} wrap it in different sentinels")
        inputs: NameToTensorList = {}
        # per pad id, its runs in prompt order: the nth placeholder of a pad is its nth run
        pad_runs: dict[int, list[MediaRun]] = {m.pad: [] for m in self.media}
        for part in parts:
            intake = media.get(part.modality)
            if intake is None:
                continue
            prepared = intake.prepare(raw[part.modality][part.index], metadata(part.modality, part.index))
            grids = [()] * len(prepared.tokens) if prepared.grids is None else prepared.grids
            pad_runs[intake.pad] += [
                MediaRun(n, intake.encoder.output, grid) for n, grid in zip(prepared.tokens, grids, strict=True)
            ]
            for name, tensor in prepared.inputs.items():
                inputs.setdefault(name, []).append(tensor)
        return MediaPrompt(expand_media_pads(ids, pad_runs, sentinels), inputs)

    def max_input_files(self) -> dict[str, int]:
        # an attachment no intake takes, refused before any file is decoded
        taken = {m.modality for m in self.media}
        return {m: 0 for m in ("image", "audio", VIDEO) if m not in taken}

    def prompt_layout(self, prompt_in: MediaPrompt) -> dict[str, list]:
        """The prompt's ``MEDIA_LAYOUT``. Always set, so a layout the client
        sent never reaches the LLM."""
        return media_layout(prompt_in.pads)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    def text_output(self) -> TextOutput:
        return TextOutput(self._chunk_detokenizer(), self.config.stop_token_ids, self.REASONING_END)

    def postprocess(
        self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None,
    ) -> bytes:
        if modality == "text":
            return super().postprocess(output, modality, request_kwargs)
        raise ValueError(f"Unsupported modality for {self.NAME}: {modality!r}")

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------

    def _sampler_spec(self, vocab_size: int) -> SamplerSpec:
        """The LLM's sampler. Its min-p filter is on only where the checkpoint
        defaults min_p; elsewhere a request's min_p > 0 is refused at
        admission (a 400), not dropped."""
        return SamplerSpec(
            resource_key=SAMPLER, nodes={"LLM"}, vocab_size=vocab_size,
            enable_repetition_penalty=True,
            enable_min_p=bool(self.sampling_defaults.get("min_p")),
        )

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, ResourceReqConfig]:
        model_kwargs = model_kwargs or {}
        sampling = dict(self.sampling_defaults)
        for k in SAMPLING_KEYS:
            if model_kwargs.get(k) is not None:
                sampling[k] = model_kwargs[k]
        return {SAMPLER: SamplingReqConfig(**sampling)}

    def request_kwargs(self) -> frozenset[str]:
        # enable_thinking: what each subclass's chat template reads
        return frozenset({"max_output_tokens", "enable_thinking", *SAMPLING_KEYS})

    # -----------------------------------------------------------------------
    # Walk plan
    # -----------------------------------------------------------------------

    def _prefill_step(
        self, signals: dict[str, list[TensorPointerInfo]], layout: dict | None = None,
    ) -> WalkStep:
        """`media_prefill_step`, carrying `process_prompt`'s media layout."""
        return media_prefill_step(signals, self.encoders, layout, model_name=self.NAME, text=self.TEXT_PROMPTS)

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        # one prefill over the whole prompt, then decode until the stop check ends it
        return [
            self._prefill_step(input_signals, model_kwargs.get(MEDIA_LAYOUT)),
            WalkStep("decode", bind={"text_inputs": "new_token"}),
        ]

    # -----------------------------------------------------------------------
    # Submodules
    # -----------------------------------------------------------------------

    def get_default_sharding_config(self) -> ShardingConfig:
        return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={})

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name not in self._submodule_cache:
            # the weights' dtype, and the fused kernel's: they must agree
            dtype = autocast_dtype or torch.bfloat16
            self._submodule_cache[node_name] = self._create_submodule(
                node_name, device, dtype, tp_group,
            )
            if tp_group is not None and tp_group.world_size > 1:
                # the LLM's norms reduce its row-parallel partial sums in one
                # fused kernel; collective, and every rank builds the node here
                tp_group.init_allreduce_fusion(self.config.hidden_size, dtype)
        return self._submodule_cache[node_name]

    def _create_submodule(
        self, node_name: str, device: str, dtype: torch.dtype, tp_group,
    ) -> NodeSubmodule | None:
        """The node's submodule with its weights loaded in ``dtype``, or None
        for a node this model does not run."""
        raise NotImplementedError

    @staticmethod
    def _build(make, dtype: torch.dtype, device: str) -> torch.nn.Module:
        """Construct under `dtype` so no fp32 copy of the weights is
        allocated. fp32-only parameters restore themselves in `_apply`."""
        torch.set_default_dtype(dtype)
        try:
            return make().to(device)
        finally:
            torch.set_default_dtype(torch.float32)
