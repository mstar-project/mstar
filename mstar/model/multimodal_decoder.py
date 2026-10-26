"""`MultimodalDecoderModel`: the `Model` side of a VLM on the shared scaffold.

An ``LLM`` node (a `SampledTokenDecoder`) behind the model's media encoders,
on `mstar.model.multimodal`'s walks. The base owns what every such model
writes the same way: the walks and the mixed text step, the decode loop's
cap, media intake, the KV label and layout, the prefill schedule, the
forward-pass args, the sampler's request config, sharding, the submodule
cache, and where the files come from. A model sets `NAME` and
`METADATA_FILES`, and writes its own ``__init__`` (calling this one, then
setting ``config``, with ``max_position_embeddings`` and ``stop_token_ids``,
``tokenizer``, ``media``, a `MediaIntake` per attachment modality, and any
``sampling_defaults``), resources, ``_prompt_ids`` (render and tokenize) and
``_create_submodule``.

Not in ``multimodal.py``: ``mstar.model.base`` imports that module, so a
`Model` subclass there would be a circular import.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.resources.spec import CombinedSplit, ResourceReqConfig
from mstar.graph.base import GraphSection, TensorPointerInfo
from mstar.model.base import ForwardPassArgs, KVLabel, LabelLayout, Model, ProcessPromptOutput, Span
from mstar.model.keys import KV_CACHE, MIXED, SAMPLER
from mstar.model.multimodal import (
    MEDIA_LAYOUT,
    TEXT,
    MediaEncoder,
    MediaPads,
    PrefillStep,
    PromptPart,
    check_attachments,
    encoder_combined_walks,
    expand_media_pads,
    initial_prefill_args,
    media_encoders,
    media_layout,
    media_prefill_schedule,
    media_runs,
    media_walk_name,
    media_walks,
    next_prefill_args,
    parts_from_modalities,
    text_combined_walks,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import TextChunkDetokenizer, drop_stop_tokens

if TYPE_CHECKING:
    from mstar.model.components.moe_parallel import MoeParallelConfig

# the sampling knobs a request may set
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty", "ignore_eos")

# A mixed step's prompt rows (as many as `max_rows_in_combined` lets in) take
# the prefill kernels, its decode rows the decode ones; for the ``splits`` of
# the LLM's attention resources.
MIXED_SPLIT = CombinedSplit(MIXED, prefill_walks=("prefill_text",))


class PreparedMedia(NamedTuple):
    """One attachment, ready for its encoder: the encoder's input tensors,
    and the tokens of each run of pads it fills, in prompt order, with each
    run's patch grid for an encoder that reads one (a ViT's grid maths) or
    a model that places media by grid (MRoPE). The grids ride the walk's
    ``MEDIA_LAYOUT``, host values every node reads without a device sync."""

    inputs: dict[str, torch.Tensor]
    tokens: list[int]
    grids: list[torch.Tensor] | None = None


@dataclass(frozen=True)
class MediaIntake:
    """How a prompt's attachments of one ``modality`` reach their encoder.

    Each attachment fills ``runs(metadata)`` placeholders ``pad`` in the
    rendered prompt (one when ``runs`` is None); ``prepare(attachment,
    metadata)`` gives its `PreparedMedia`, and each placeholder is expanded
    to its run's tokens, between ``sentinels`` when given. ``metadata`` is
    the attachment's loader metadata, ``{}`` when it came with none.
    """

    modality: str
    encoder: MediaEncoder
    pad: int
    prepare: Callable[[torch.Tensor, dict], PreparedMedia]
    sentinels: tuple[int, int] | None = None
    runs: Callable[[dict], int] | None = None


class MediaPrompt(NamedTuple):
    """A prompt after intake: its ids with each placeholder expanded, every
    encoder's inputs in prompt order, and the runs' patch grids by pad."""

    pads: MediaPads
    inputs: NameToTensorList
    grids: dict[int, list[torch.Tensor]]


class MultimodalDecoderModel(Model):
    # Image preprocessing is a few small ops per request, and a thread per
    # core only slows them: each op waits on whichever thread a co-tenant
    # descheduled. At 64 threads c=1 TTFT p95 was 152 ms (some preprocesses
    # 35-130 ms); at 4, preprocessing held at 7-10 ms and p95 at 49 ms. One
    # thread is steadier but slower (13 ms).
    PREPROCESS_TORCH_THREADS = 4
    # in errors and logs
    NAME: str
    # what the API and conductor processes download: enough to build the
    # config and tokenize; workers fetch the full snapshot (`_resolve_weights`)
    METADATA_FILES: tuple[str, ...]

    # one per attachment modality the model takes, set by its `__init__`
    media: tuple[MediaIntake, ...] = ()
    _text_detokenizer: TextChunkDetokenizer | None = None

    def __init__(self, model_path_hf: str, cache_dir: str | None = None):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.local_dir = self._resolve_model_metadata()
        # so weights are materialized at most once per node
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        # the checkpoint's sampling defaults, under whatever a request sets
        self.sampling_defaults: dict = {}

    def _resolve_model_metadata(self) -> str:
        local_path = Path(self.model_path_hf)
        if local_path.is_dir():
            return str(local_path)
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=self.model_path_hf, cache_dir=self.cache_dir,
            allow_patterns=list(self.METADATA_FILES),
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
        return {**text_combined_walks(), **encoder_combined_walks(self.encoders)}

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # a safety bound on decode, not the per-request budget
        return media_walks(self.config.max_position_embeddings, self.encoders)

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # the request's output budget: the stop check counts no tokens
        caps = {"decode_loop": self.get_max_output_tokens(**model_kwargs)}
        # A chunked encoder's loop runs each of its runs in whole chunks, so
        # it ends on its last chunk. A request with none of its media leaves
        # it out: its walk has no such loop.
        layout = model_kwargs.get(MEDIA_LAYOUT) or {}
        for e in self.encoders:
            if e.chunk_tokens is not None:
                chunks = sum(map(e.chunks, media_runs(layout, e.output)))
                if chunks:
                    caps[e.loop] = chunks
        return caps

    # -----------------------------------------------------------------------
    # KV layout
    # -----------------------------------------------------------------------

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        # unkeyed: these models leave cross-request prefix reuse off
        return {KV_CACHE: {"main": KVLabel(growth=(("decode", "decode_loop"),))}}

    def kv_layout(self, layout: dict, length: int, advance: int | None = None) -> dict:
        """`process_prompt`'s ``kv_layout`` for a prompt of ``length`` tokens
        and its ``MEDIA_LAYOUT``: the whole prompt, written by its one prefill
        walk, then a slot per decode step. ``advance`` is how far the prompt
        moves the position counter, its length by default."""
        present = media_encoders(self.encoders, layout)
        span = Span(
            "rows" if present else "ids",
            media_walk_name(present) if present else "prefill_text",
            length, length if advance is None else advance, "text_inputs",
        )
        return {KV_CACHE: {"main": LabelLayout(spans=(span,), growth=(("decode_loop", 1),))}}

    # -----------------------------------------------------------------------
    # Intake
    # -----------------------------------------------------------------------

    def _prompt_ids(
        self, parts: list[PromptPart], raw: dict[str, list[torch.Tensor]], **kwargs,
    ) -> torch.Tensor:
        """The rendered prompt's host ids, each attachment's placeholders in
        place. ``raw`` is each modality's attachments; ``kwargs`` are
        `process_prompt`'s."""
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
        return ProcessPromptOutput(
            {"text_inputs": [prompt_in.pads.ids], **prompt_in.inputs},
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
        for m, intake in media.items():
            found = int((ids == intake.pad).sum())
            want = sum(
                1 if intake.runs is None else intake.runs(metadata(m, i)) for i in range(len(raw[m]))
            )
            if found != want:
                raise ValueError(
                    f"multimodal placement mismatch: the prompt holds {found} {m} "
                    f"placeholders for {want} {m} inputs"
                )
        inputs: NameToTensorList = {}
        counts: dict[int, list[int]] = {m.pad: [] for m in self.media}
        grids: dict[int, list[torch.Tensor]] = {}
        for part in parts:
            intake = media.get(part.modality)
            if intake is None:
                continue
            prepared = intake.prepare(raw[part.modality][part.index], metadata(part.modality, part.index))
            counts[intake.pad] += prepared.tokens
            if prepared.grids is not None:
                grids.setdefault(intake.pad, []).extend(prepared.grids)
            for name, tensor in prepared.inputs.items():
                inputs.setdefault(name, []).append(tensor)
        sentinels = {m.pad: m.sentinels for m in self.media if m.sentinels is not None}
        return MediaPrompt(expand_media_pads(ids, counts, sentinels), inputs, grids)

    def prompt_layout(self, prompt_in: MediaPrompt) -> dict[str, list]:
        """The prompt's ``MEDIA_LAYOUT``. Always set, so a layout the client
        sent never reaches the LLM."""
        return media_layout(
            prompt_in.pads, {m.pad: m.encoder.output for m in self.media}, prompt_in.grids,
        )

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    def postprocess(
        self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None,
    ) -> bytes:
        if modality == "text":
            ids = drop_stop_tokens(output, self.config.stop_token_ids, request_kwargs)
            if self._text_detokenizer is None or self._text_detokenizer.tokenizer is not self.tokenizer:
                self._text_detokenizer = TextChunkDetokenizer(self.tokenizer)
            return self._text_detokenizer.to_bytes(ids)
        raise ValueError(f"Unsupported modality for {self.NAME}: {modality!r}")

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs], model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        model_kwargs = model_kwargs or {}
        sampling = dict(self.sampling_defaults)
        for k in SAMPLING_KEYS:
            if model_kwargs.get(k) is not None:
                sampling[k] = model_kwargs[k]
        return {SAMPLER: SamplingReqConfig(**sampling)}

    # -----------------------------------------------------------------------
    # Forward pass args
    # -----------------------------------------------------------------------

    def _prefill_schedule(
        self, input_modalities: list[str], signals: dict[str, list[TensorPointerInfo]],
        layout: dict | None = None,
    ) -> list[PrefillStep]:
        """`media_prefill_schedule`, carrying `process_prompt`'s media layout."""
        return media_prefill_schedule(signals, self.encoders, layout, model_name=self.NAME)

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        schedule = self._prefill_schedule(
            input_modalities, input_signals, (model_kwargs or {}).get(MEDIA_LAYOUT),
        )
        return initial_prefill_args(
            schedule, input_modalities, output_modalities, input_signals, model_name=self.NAME,
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        return next_prefill_args(partition_metadata, persist_signals)

    # -----------------------------------------------------------------------
    # Submodules
    # -----------------------------------------------------------------------

    def get_default_sharding_config(self) -> ShardingConfig:
        return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={})

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        moe_parallel: MoeParallelConfig | None = None,
    ) -> NodeSubmodule | None:
        if node_name not in self._submodule_cache:
            self._submodule_cache[node_name] = self._create_submodule(
                node_name, device, autocast_dtype, tp_group, moe_parallel,
            )
        return self._submodule_cache[node_name]

    def _create_submodule(
        self, node_name: str, device: str, autocast_dtype: torch.dtype | None, tp_group,
        moe_parallel: MoeParallelConfig | None = None,
    ) -> NodeSubmodule | None:
        """The node's submodule with its weights loaded, or None for a node
        this model does not run. ``moe_parallel`` is the node group's
        ``ep_config``, set only for a node in ``ep_enabled_nodes``."""
        raise NotImplementedError

    @staticmethod
    def _build(make, dtype: torch.dtype, device: str) -> torch.nn.Module:
        """Construct on ``device`` under ``dtype``, so no host or fp32 copy of
        the weights is allocated. The ``.to`` copies nothing; it runs each
        module's ``_apply``, where fp32-only parameters restore themselves
        before the weights load."""
        torch.set_default_dtype(dtype)
        try:
            with torch.device(device):
                return make().to(device)
        finally:
            torch.set_default_dtype(torch.float32)
