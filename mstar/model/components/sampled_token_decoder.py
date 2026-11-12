"""An autoregressive LLM node that samples one token per row per step.

`SampledTokenDecoder` serves the VLM walk set of ``mstar.model.multimodal``
(``prefill_text``, the ``mixed`` combined walk, ``decode``, and one media
walk per subset of the model's `MediaEncoder` s, which prefills a whole
multimodal prompt as one row). It owns the engine contract: the capture
configs, device loop-back of the sampled token, the step declaration around
the sampler and the position resource, the forward through the sampler, and
the stop: a `TokenStop` over its prefill and decode walks (a stop id; a
stop, or a one-token budget, on the step the walk plan marks `FINAL_PREFILL`
ends the partition, and the model's loop cap is the decode budget). A model subclasses it with its resource steps
and its attention key, and an MRoPE model adds its own positions.

The decode loop may run under more than one walk: MiniCPM-o's spoken reply
decodes on ``decode_spoken``, whose graph also hands each row's input id and
post-norm hidden state to its TTS. A node takes the walks its model's graph
runs it in the decode loop and the walks that route those two
(`DecoderWalks`, read off the graph by `decoder_walks`) through `bind_walks`,
which the scaffold's `get_submodule` calls. With any of the
latter it returns them as `ROW_IDS` and `ROW_HIDDEN` from every step, one row
per request, so every capture holds them; a row whose walk has no edge for
them leaves them unrouted.

A media walk's row is the whole prompt's ids, each attachment expanded to
one pad per token at intake (`expand_media_pads`). Its encoders' outputs
ride as tensors named by `MediaEncoder.output`. `prepare_inputs` reads the
row's ``MEDIA_LAYOUT`` on the host and lays them out as one prompt-ordered
``MEDIA_EMBEDS`` and a ``MEDIA_MASK`` of the row's tokens, True on each
run's; the forward splices ``where(mask, media[cumsum(mask) - 1],
embed(ids))``. The layout places the rows, not the pad ids, so encoders may
share a placeholder token and a sampled id never takes a media row. The
contract that makes this capture-safe:

- a row's media tensor holds exactly the tokens of its runs, in prompt
  order (`prepare_inputs` checks each encoder's row count against the
  layout), so the packed step's media, concatenated in row order, line up
  with the cumulative mask count;
- every step of the media capture's walks brings the mask, and real tokens
  precede a replay's padding tail, so the tail's stale mask cannot shift a
  real row's index; and the captured media buffer has at least the bucket's
  tokens rows, so a tail index stays in bounds and `splice_media`'s clamp to
  the last row never binds.

A chunked media row (`split_inputs`) cuts its mask with its ids, and its
media by the mask's count before each end.

The decoder module (``self.model``, e.g. ``Qwen3DenseLM``) provides:

- ``embed(ids) -> [tokens, hidden]``
- ``forward(input_embeds, *, label, **decoder_kwargs) -> [tokens, hidden]``,
  the stack through its final norm
- ``logits(hidden) -> [rows, vocab]``
- ``embed_dtype``: the embedding's dtype

and ``config`` has ``hidden_size`` and ``stop_token_ids`` (and the model's
``max_position_embeddings`` bounds the decode loop, see `media_walks`).
The node's sampler is declared under ``mstar.model.keys``' ``SAMPLER`` key and
its positions under ``POS_KEY`` (``ROPE`` by default).
"""

import functools
import logging
import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from time import perf_counter
from typing import Any, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import FINAL_PREFILL, CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    decode_rows_then_prompt,
)
from mstar.engine.resources.position.config import PositionStep
from mstar.engine.resources.sampler.config import SamplerStep
from mstar.engine.resources.step import ResourceStep, Segment, SubmoduleStep
from mstar.graph.base import GraphSection
from mstar.model.components.token_step import TokenStepNode
from mstar.model.keys import MIXED, PREFILL_TEXT, ROPE, SAMPLER
from mstar.model.multimodal import MEDIA_LAYOUT, MediaEncoder, media_rows, media_walk_names, splice_media
from mstar.model.rebuild import loop_walks
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    DeviceLoopback,
    InputSeqLenInfo,
    ModelInputsFromEngine,
    TokenStop,
)
from mstar.utils.h2d import to_device_nonblocking
from mstar.utils.profiler import PHASE_PERIOD, PhaseLaps, phase_record

logger = logging.getLogger(__name__)

# A media row's encoder rows in prompt order, and its tokens' mask of where
# they go (`SampledTokenDecoder._media_inputs`)
MEDIA_EMBEDS = "media_embeds"
MEDIA_MASK = "media_mask"
# A row's last input id and its post-norm hidden state, for a walk whose graph
# routes them (`DecoderWalks.hidden`)
ROW_IDS = "row_ids"
ROW_HIDDEN = "row_hidden"
# The packed mixed capture's key, which a mixed step holding a prompt row
# replays; one of decode rows alone replays the batched decode capture
# (`SampledTokenDecoder.cg_key_info`)
PROMPT_ROWS = "prompt_rows"


class DecoderWalks(NamedTuple):
    """The walks a decoder node treats apart, as its model's graph runs it
    (`decoder_walks`). ``decode``: those running it in the decode loop, one
    token a row, which share the decode capture (captured on ``DECODE_WALK``),
    the stop, the client token inline and the mixed step; ``DECODE_WALK``
    alone keeps its token on the device (`DeviceLoopback`). ``hidden``: those
    whose graph routes its `ROW_IDS` and `ROW_HIDDEN`."""

    decode: frozenset[str]
    hidden: frozenset[str] = frozenset()


def decoder_walks(
    graphs: Mapping[str, GraphSection], node: str = "LLM", loop: str = "decode_loop",
) -> DecoderWalks:
    """``node``'s `DecoderWalks`, read off a model's walk graphs: the walks
    that run it inside Loop ``loop``, and those in which it sends `ROW_IDS`
    and `ROW_HIDDEN`, which go together."""
    hidden = set()
    for walk, section in graphs.items():
        found = section.get_nodes().get(node)
        sent = {ROW_IDS, ROW_HIDDEN} & {e.name for e in found.outputs} if found is not None else set()
        if len(sent) == 1:
            raise ValueError(f"{node} sends {sorted(sent)} alone in {walk!r}: a row's ids and hidden go together")
        if sent:
            hidden.add(walk)
    decode = loop_walks(graphs, loop, node)
    if not decode:
        raise ValueError(f"no walk runs {node} inside Loop {loop!r}: a decoder needs a decode walk")
    return DecoderWalks(decode, frozenset(hidden))


class SampledTokenDecoder(TokenStepNode, ARNodeSubmodule):
    SAMPLER = SAMPLER
    # the text-only prompt's walk; a model whose own graph names it otherwise
    # sets this
    PREFILL_WALK = PREFILL_TEXT
    # Total tokens of a packed text-prefill step, one captured graph per
    # (bucket, batch size); a longer step prefills eagerly. Static buffers are
    # sized by the largest bucket. A deployment sets its own per walk with the
    # serving config's `prefill_token_buckets`.
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]
    # A mixed step carries the decode batch beside its prompt rows, so its
    # captures run to the decode batch: one capped below it splits the batch.
    # Under more than one decode walk its largest is the decode ladder's
    # (`_decode_rows_mix`).
    MIXED_CAPTURE_BATCH_SIZES = [4, 16, 64, 128]
    # The largest is also the scheduler's cap on rows per decode step, and a
    # step pads to the next bucket. MSTAR_DECODE_BUCKETS narrows the ladder.
    DECODE_CAPTURE_BATCH_SIZES = [
        int(x) for x in os.environ.get("MSTAR_DECODE_BUCKETS", "1,2,4,8,16,32,48,64,96,128").split(",")
    ]
    # Text plus media tokens of a whole multimodal prompt, on every media walk
    # (vision, audio or both). Capped well below a long prompt because static
    # buffers are sized by the largest bucket; past it the step runs eagerly.
    PREFILL_MEDIA_TOKEN_BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096]
    # Prompt rows a mixed step may carry beside its decode rows: the
    # attention's prefill kernels are captured for this many, and a recurrent
    # state's prefill kernel gathers and scatters each such row's whole state.
    MIXED_MAX_PREFILL_ROWS = 4
    # Requests per media-walk step, of any medium: the largest caps every
    # media walk, captured or not (`max_batch_size`), so a packed step stays
    # near the top bucket (4 fit ~1000 tokens a request: a ~1000 px square
    # image plus text). Chunked prefill replaces this cap.
    PREFILL_MEDIA_CAPTURE_BATCH_SIZES = [1, 2, 4]
    # A mixed step's token ladder; None runs it on PREFILL_TOKEN_BUCKETS
    MIXED_TOKEN_BUCKETS: list[int] | None = None
    # Whether a mixed step also carries media prompt rows beside its decode rows
    # (a model listing its media walks in its combined walk): its capture then
    # holds the media buffer and its mask, so a replay splices them in.
    MIXED_MEDIA: bool = False
    # Whether the model chunks its media prompt rows (a chunkable
    # ``ChunkingPolicy`` on its media walks): their lengths are then reported
    # before prepare, so a token budget can cut them (`split_inputs`).
    CHUNKS_MEDIA: bool = False

    # The attention resource whose plan picks each prefill row's last token.
    ATTN_KEY: str
    # The position resource the step advances (a layer's ``pos_key``)
    POS_KEY: str = ROPE
    # False: only decode is captured, and every prefill (text, media, mixed)
    # runs eagerly, for a model whose prefill captures are unmeasured
    CAPTURE_PREFILL = True

    # set by `__init__`; class defaults keep a text-only node's walks working
    _encoders: tuple[MediaEncoder, ...] = ()
    _media_walks: frozenset[str] = frozenset()
    _text_prompts = True
    _decode_walks: frozenset[str] = frozenset({TokenStepNode.DECODE_WALK})
    _hidden_walks: frozenset[str] = frozenset()

    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,
        encoders: Sequence[MediaEncoder] = (),
        text_prompts: bool = True,
    ):
        """``encoders`` are the model's media encoders, as its walks declare
        them; a text-only node passes none. ``text_prompts`` is the model's
        `MultimodalDecoderModel.TEXT_PROMPTS`: without it there is no text-only
        prefill walk, so nothing captures one or a mixed step. Its walks are
        ``DECODE_WALK`` alone until `bind_walks` gives it its graph's, as
        `MultimodalDecoderModel.get_submodule` does for every model on the
        scaffold."""
        super().__init__()
        self._text_prompts = text_prompts
        self.model = model
        self.config = config
        self._encoders = tuple(encoders)
        self._media_walks = frozenset(media_walk_names(self._encoders))
        self.bind_walks(DecoderWalks(frozenset({self.DECODE_WALK})))

    def bind_walks(self, walks: DecoderWalks) -> None:
        """Run under the `DecoderWalks` of its model's graph
        (`decoder_walks`): the stop, the decode capture, the mixed step and
        the hidden outputs all read them."""
        if self.DECODE_WALK not in walks.decode:
            raise ValueError(f"{type(self).__name__}'s decode walks {sorted(walks.decode)} miss {self.DECODE_WALK!r}")
        self._decode_walks, self._hidden_walks = walks
        # a prefill walk's token is the reply's first only on the step before
        # decode, which the walk plan marks
        prefills = self._media_walks | ({self.PREFILL_WALK} if self._text_prompts else set())
        self.token_stop = TokenStop(
            loop=self.DECODE_LOOP, buffer="new_token", stop_ids=frozenset(self.config.stop_token_ids),
            sampler=self.SAMPLER, walk=self._decode_walks, prefill_walks=frozenset(prefills),
            last_prefill_flag=FINAL_PREFILL,
            # the graph runtime checks a decode row's stop ids itself, without
            # calling check_stop (measured on Qwen3.5; see TokenStop.runtime_check)
            runtime_check=True,
        )

    @functools.cached_property
    def device_loopback(self) -> DeviceLoopback | None:
        """The decode walk reads its input ids off the sampler's slot masters;
        a model that reads them off the routed token sets this None."""
        return self._decode_loopback()

    def _decode_loopback(self) -> DeviceLoopback:
        return DeviceLoopback(
            walk=self.DECODE_WALK, signal="text_inputs", sampler=self.SAMPLER, buffer="new_token",
        )

    def enable_repeats(self, walks: Collection[str]) -> None:
        """Decode reads its token off the sampler and its rows' positions off the
        position resource (`preprocess`), so a walk asked to repeat may run on
        resident tables and loop its token back, even where the node's own path
        doesn't: Canary keeps main's routed token, a measured choice of its own,
        unless a deployment asks."""
        super().enable_repeats(walks)
        if self.DECODE_WALK in self._repeat_walks and self.device_loopback is None:
            self.device_loopback = self._decode_loopback()

    def resident_walks(self) -> frozenset[str]:
        # decode alone: the walk ``enable_repeats`` vouches for
        return self._repeat_walks & {self.DECODE_WALK}

    # ------------------------------------------------------------------
    # What a model declares
    # ------------------------------------------------------------------

    def layer_steps(self) -> dict[str, ResourceStep]:
        """This step's work on the layers' resources (KV, attention, recurrent
        state, ...), in declaration order; the sampler and position steps are
        added around it."""
        raise NotImplementedError

    def prompt_positions(
        self, start: int, layout: dict, num_tokens: int,
    ) -> tuple[torch.Tensor | None, int | None]:
        """A media prompt's own position ids and how far it advances the
        position counter, for a model whose media do not take one position
        per token (MRoPE). ``start`` is the counter's value, ``layout`` the
        walk's ``MEDIA_LAYOUT`` (each run's offset, tokens and grid). Built on
        the host, from the layout alone: the scaffold moves them to the device,
        and a chunked row's cut rebuilds them (`split_inputs`). The default,
        ``(None, None)``, leaves the position resource to place every token."""
        return None, None

    def node_step(
        self, segments: list[Segment], prefill_tokens: Mapping[int, torch.Tensor], advance: tuple[int, ...] | None,
    ) -> SubmoduleStep:
        """`layer_steps` over ``segments``, with the sampler's step and the position resource's step around it."""
        return SubmoduleStep(
            segments=segments,
            steps={
                **self.layer_steps(),
                SAMPLER: SamplerStep(
                    apply_penalty=True,
                    prefill_tracked_tokens=prefill_tokens,
                ),
                self.POS_KEY: PositionStep(advance=advance),
            },
        )

    def next_position(self, rid_handle: int) -> int:
        """The position resource's counter for the request at worker handle ``rid_handle``."""
        return self.node_resources[self.POS_KEY].position(rid=rid_handle, label="main")

    def loopback_tokens(self, engine_inputs: ModelInputsFromEngine) -> torch.Tensor:
        """Each row's last sampled token, read off the sampler resource's slots."""
        return engine_inputs.resources[SAMPLER].loopback_tokens(engine_inputs.request_ids)

    def last_token_hidden(self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor) -> torch.Tensor:
        """``hidden`` (hidden states or token ids) at each row's last token, picked by the plan of the
        attention resource under ``ATTN_KEY``."""
        return engine_inputs.resources[self.ATTN_KEY].select_last_hidden(hidden)

    def sample_tokens(self, engine_inputs: ModelInputsFromEngine, logits: torch.Tensor) -> torch.Tensor:
        """One token per row of ``logits``, drawn by the sampler resource."""
        return engine_inputs.resources[SAMPLER].sample(engine_inputs.request_ids, logits=logits)

    def decoder_kwargs(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        num_tokens: int, **tensors: torch.Tensor,
    ) -> dict[str, Any]:
        """Extra keyword arguments to the decoder module's forward, from
        ``preprocess``'s other outputs. None by default."""
        return {}

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        """Decode, text prefill, mixed and vision prefill all capture."""
        def ids(n: int) -> ARNodeInputs:
            return ARNodeInputs(
                input_ids=torch.zeros(n, dtype=torch.long, device=device), input_seq_len=n,
            )

        def media(n: int) -> ARNodeInputs:
            # the prompt-ordered media of every encoder and its mask, `n` rows
            # each; the mask on the host, as `_media_inputs` builds it, since a
            # replay pads real rows with these. A row advances the position
            # counter by its own length, so a replay's padding row (n = 0)
            # moves nothing beside real rows that place their own positions.
            return ARNodeInputs(
                input_seq_len=n,
                resource_step_info=n,
                input_ids=torch.zeros(n, dtype=torch.long, device=device),
                tensor_inputs={
                    MEDIA_EMBEDS: torch.zeros(
                        (n, self.config.hidden_size), device=device, dtype=self.model.embed_dtype,
                    ),
                    MEDIA_MASK: torch.zeros(n, dtype=torch.bool),
                },
            )

        # a mixed step of decode rows alone replays the decode capture, keyed apart
        # from the packed one its prompt rows need (`cg_key_info`)
        mix = self._decode_rows_mix
        top = max(self.DECODE_CAPTURE_BATCH_SIZES)
        if mix and (dropped := [b for b in self.MIXED_CAPTURE_BATCH_SIZES if b > top]):
            logger.warning(
                f"{type(self).__name__}: mixed capture sizes {dropped} are above the decode ladder's top "
                f"{top} and are dropped",
            )
        configs: list[AcceleratorGraphConfig] = [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk=self.DECODE_WALK,
                replay_graph_walks=[
                    self.DECODE_WALK, *sorted(self._decode_walks - {self.DECODE_WALK}), *([MIXED] if mix else []),
                ],
                # a decode row, as a pre-plan under another walk declares the template
                single_request_inputs=replace(ids(1), graph_walk=self.DECODE_WALK),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
                compile_mode=self.capture_compile_mode(self.DECODE_WALK),
            ),
        ]
        if self._text_prompts and self.CAPTURE_PREFILL:
            configs += [PackedAcceleratorGraphConfig(
                capture_graph_walk=self.PREFILL_WALK,
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                make_node_input=ids,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
                compile_mode=self.capture_compile_mode(self.PREFILL_WALK),
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk=MIXED,
                capture_token_lengths=(
                    self.PREFILL_TOKEN_BUCKETS if self.MIXED_TOKEN_BUCKETS is None else self.MIXED_TOKEN_BUCKETS
                ),
                make_node_input=media if self._mixes_media else ids,
                # Both captures replay a mixed step, so the scheduler caps one at the
                # larger top (`max_batch_size_for`): topped at the decode ladder's,
                # neither kind of step runs past its own ladder, eager.
                capture_batch_sizes=(
                    [b for b in self.MIXED_CAPTURE_BATCH_SIZES if b < top] + [top] if mix
                    else self.MIXED_CAPTURE_BATCH_SIZES
                ),
                additional_key_info=PROMPT_ROWS if mix else None,
                compile_mode=self.capture_compile_mode(MIXED),
                # the layout the resources split; a replay's padding rows stay empty
                token_layout=decode_rows_then_prompt,
                input_seq_dims={MEDIA_EMBEDS: 0, MEDIA_MASK: 0} if self._mixes_media else None,
            )]
        if self._encoders and self.CAPTURE_PREFILL:
            walks = media_walk_names(self._encoders)
            # one capture every media walk replays
            configs.append(PackedAcceleratorGraphConfig(
                capture_graph_walk=walks[0],
                replay_graph_walks=walks,
                capture_token_lengths=self.PREFILL_MEDIA_TOKEN_BUCKETS,
                make_node_input=media,
                capture_batch_sizes=self.PREFILL_MEDIA_CAPTURE_BATCH_SIZES,
                compile_mode=self.capture_compile_mode(walks[0]),
                # the media buffer is [tokens, hidden], and hidden can equal a bucket
                input_seq_dims={MEDIA_EMBEDS: 0, MEDIA_MASK: 0},
            ))
        return configs

    def capture_compile_mode(self, graph_walk: str) -> str | None:
        """The torch.compile mode of the capture ``graph_walk`` names; None
        takes the deployment's compile level (mstar serve -O)."""
        del graph_walk
        return None

    def max_batch_size(self, graph_walk: str) -> int | None:
        return max(self.PREFILL_MEDIA_CAPTURE_BATCH_SIZES) if graph_walk in self._media_walks else None

    @property
    def _decode_rows_mix(self) -> bool:
        """Whether a mixed step can hold decode rows of more than one walk and no
        prompt row: one decode walk's rows alone are relabelled to it by the
        scheduler, and so replay the decode capture already."""
        return len(self._decode_walks) > 1 and self._text_prompts and self.CAPTURE_PREFILL

    def _step_walk(self, graph_walk: str, rows: Collection[ARNodeInputs | CurrentForwardPassInfo]) -> str:
        """The walk a mixed step of ``rows`` runs as. Under more than one decode
        walk, one of decode rows alone replays the decode capture
        (`cg_key_info`), so it is a decode step: ``DECODE_WALK``, for what is
        keyed on the walk (`preprocess`, `_pack_media`, `decoder_kwargs`).
        Under one, such a step (its prompt row dropped after scheduling)
        replays the packed mixed capture, so it stays ``MIXED`` and no row is
        read."""
        if len(self._decode_walks) == 1 or not rows:
            return graph_walk
        if all((row.graph_walk or graph_walk) in self._decode_walks for row in rows):
            return self.DECODE_WALK
        return graph_walk

    def cg_key_info(
        self, graph_walk: str,
        per_request_info: Mapping[int, CurrentForwardPassInfo],
        per_request_input_metadata=None,
        **kwargs: Any,
    ) -> str | None:
        """A mixed step holding a row of a walk outside the decode loop replays
        the packed mixed capture; one of decode rows alone, the batched decode
        capture, which a pre-plan can lease before the step's inputs exist."""
        del per_request_input_metadata, kwargs
        if graph_walk != MIXED or not self._decode_rows_mix:
            return None
        decode = self._decode_walks
        return PROMPT_ROWS if any(info.graph_walk not in decode for info in per_request_info.values()) else None

    @property
    def _mixes_media(self) -> bool:
        return self.MIXED_MEDIA and bool(self._encoders)

    def max_rows_in_combined(self, graph_walk: str) -> int | None:
        prompt = graph_walk == self.PREFILL_WALK or (self._mixes_media and graph_walk in self._media_walks)
        return self.MIXED_MAX_PREFILL_ROWS if prompt else None

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _media_inputs(
        self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList,
    ) -> ARNodeInputs:
        """A whole multimodal prompt as one row: its ids, and its encoders'
        rows in prompt order (``MEDIA_EMBEDS``) with the mask of the tokens
        they replace (``MEDIA_MASK``), both laid out from the row's
        ``MEDIA_LAYOUT``: slices of the encoders' outputs at host offsets, so
        no device sync.

        Each medium's row count, none when its output is missing, is checked
        against the tokens its runs take, on host values only; a mismatch
        fails this request (a captured replay would otherwise splice in the
        previous step's rows). A model placing its own positions
        (`prompt_positions`) hands them over as ``custom_pos_ids``, and its
        counter advance as the row's ``resource_step_info``; `declare_step`
        passes that to the position resource, whose default (the token
        count) would misplace everything after the first image.
        """
        ids = inputs["text_inputs"][0]
        layout = fwd_info.step_metadata.get(MEDIA_LAYOUT) or {}
        mask, runs = media_rows(layout, ids.shape[0])
        unknown = {r.output for r in runs} - {e.output for e in self._encoders}
        if unknown:
            raise ValueError(f"the prompt's media layout names {sorted(unknown)}, which no encoder of this node sends")
        outputs: dict[str, torch.Tensor] = {}
        for e in self._encoders:
            pads = sum(r.tokens for r in runs if r.output == e.output)
            rows = inputs[e.output][0] if inputs.get(e.output) else None
            found = 0 if rows is None else rows.shape[0]
            if found != pads:
                raise ValueError(f"{e.output} has {found} rows for the prompt's {pads} pads")
            if found:
                outputs[e.output] = rows
        if len(outputs) == 1:
            # one encoder's runs read its output whole, in order
            (embeds,) = outputs.values()
        else:
            # each run takes the next rows of its encoder's output, in prompt
            # order; a run of no tokens takes none (its encoder may send none)
            cursor = dict.fromkeys(outputs, 0)
            pieces = []
            for r in runs:
                if r.tokens:
                    at = cursor[r.output]
                    pieces.append(outputs[r.output][at:at + r.tokens])
                    cursor[r.output] += r.tokens
            embeds = torch.cat(pieces) if pieces else None
        media = {} if embeds is None else {MEDIA_EMBEDS: embeds, MEDIA_MASK: mask}
        # the stream's counter is keyed by the worker's handle (AGENTS.md invariant 14)
        start = self.next_position(fwd_info.rid_handle)
        pos_ids, advance = self.prompt_positions(start, layout, ids.shape[0])
        if pos_ids is not None:
            pos_ids = to_device_nonblocking(pos_ids, self.get_device())
        return ARNodeInputs(
            input_seq_len=ids.shape[0],
            input_ids=ids,
            custom_pos_ids=pos_ids,
            tensor_inputs=media,
            resource_step_info=advance,
        )

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        has_loopback: bool = False,
        **kwargs: Any,
    ) -> ARNodeInputs:
        if graph_walk in self._media_walks:
            return self._media_inputs(fwd_info, inputs)
        if (row := self.uniform_row_inputs(graph_walk, has_loopback)) is not None:
            # one token per row, read off the sampler's slot master in
            # `preprocess`; the routed signal is empty (or ignored) here
            return row
        input_ids = inputs["text_inputs"][0]
        return ARNodeInputs(input_seq_len=input_ids.shape[0], input_ids=input_ids)

    def get_input_sequence_len(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList, **kwargs,
    ) -> InputSeqLenInfo | None:
        if graph_walk in self._decode_walks:
            return InputSeqLenInfo(1)
        # a media row's ids hold a pad per media token, so they measure it; it
        # is left unmeasured, and admitted whole, unless the model chunks it
        media = self.CHUNKS_MEDIA and graph_walk in self._media_walks
        if (graph_walk == self.PREFILL_WALK or media) and "text_inputs" in inputs:
            return InputSeqLenInfo(inputs["text_inputs"][0].shape[0])
        return None

    def split_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo,
        inputs: ARNodeInputs, start: int, end: int,
    ) -> ARNodeInputs:
        """A media row's chunk also takes its part of the row's mask, the
        media rows that part marks (the mask's count before ``start`` to its
        count before ``end``, host values), and the part of the row's position
        advance its tokens place: the advance through ``end`` less that
        through ``start``, so a row's chunks advance by the whole row's."""
        if graph_walk not in self._media_walks:
            return super().split_inputs(graph_walk, fwd_info, inputs, start, end)
        cut = super().split_inputs(
            graph_walk, fwd_info, replace(inputs, tensor_inputs={}, resource_step_info=None), start, end,
        )
        layout = fwd_info.step_metadata.get(MEDIA_LAYOUT) or {}
        mask = inputs.tensor_inputs.get(MEDIA_MASK)
        if mask is not None:
            a, b = int(mask[:start].sum()), int(mask[:end].sum())
            # a chunk without media carries none, as a row without media does
            if b > a:
                cut.tensor_inputs = {MEDIA_EMBEDS: inputs.tensor_inputs[MEDIA_EMBEDS][a:b], MEDIA_MASK: mask[start:end]}
        if inputs.resource_step_info is not None:
            through = self._advance_through(layout, inputs, end) - self._advance_through(layout, inputs, start)
            cut.resource_step_info = through
        return cut

    def _advance_through(self, layout: Mapping, inputs: ARNodeInputs, t: int) -> int:
        """How far a media row's first ``t`` tokens move the position counter:
        one past the largest position they hold, from the row's first. The
        row's own positions are on the device by now, so they are rebuilt on
        the host from the layout (based at 0: the advance doesn't depend on
        where the row starts)."""
        if t == 0:
            return 0
        if t >= inputs.input_seq_len:
            return int(inputs.resource_step_info)
        pos, _ = self.prompt_positions(0, layout, inputs.input_seq_len)
        return int(pos[..., :t].max()) + 1

    def speculative_input_seq_len(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, prev_seq_len: int,
    ) -> int | None:
        if graph_walk in self._decode_walks:
            return 1
        return super().speculative_input_seq_len(graph_walk, fwd_info, prev_seq_len)

    @staticmethod
    def _has_loopback_rows(inputs: Sequence[ARNodeInputs]) -> bool:
        """Whether the step's ids come off the sampler: a row without ids is a
        loop-back decode row."""
        return any(inp.input_ids is None for inp in inputs)

    @staticmethod
    def _splice_row_ids(
        ids: torch.Tensor, inputs: list[ARNodeInputs], decode: bool,
    ) -> torch.Tensor:
        """``ids`` (one loop-back token per row) with each row that brought
        its own ids taking them instead, in row order: a mixed step's prompt
        rows, or a decode row whose last step ran elsewhere. Runs of loop-back
        rows go in as one slice each, and the splice is device ops only: an
        index built from a host list would be a synchronous copy behind the
        step in flight. A decode step keeps the slot's padding rows past
        ``inputs``, so its length stays the captured one."""
        pieces: list[torch.Tensor] = []
        run = None
        for i, inp in enumerate(inputs):
            if inp.input_ids is None or (decode and not inp.input_seq_len):
                if run is None:
                    run = i
                continue
            if run is not None:
                pieces.append(ids[run:i])
                run = None
            pieces.append(inp.input_ids.to(ids.dtype))
        if run is not None:
            pieces.append(ids[run:len(inputs)])
        if decode and ids.shape[0] > len(inputs):
            pieces.append(ids[len(inputs):])
        return torch.cat(pieces) if len(pieces) > 1 else pieces[0]

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        if graph_walk == MIXED:
            graph_walk = self._step_walk(graph_walk, inputs)
        if self._has_loopback_rows(inputs):
            # every row's last sampled token, read by slot (padding rows read
            # slot 0); a row whose last step ran elsewhere brought its own
            t0 = perf_counter() if PHASE_PERIOD else 0.0
            laps = PhaseLaps("engine.preprocess.loopback.")
            ids = self.loopback_tokens(engine_inputs)
            laps.lap("tokens")
            decode = graph_walk in self._decode_walks
            if not decode or any(inp.input_ids is not None and inp.input_seq_len for inp in inputs):
                ids = self._splice_row_ids(ids, inputs, decode)
            laps.lap("splice")
            if PHASE_PERIOD:
                phase_record("engine.preprocess.loopback", perf_counter() - t0)
            out = {"input_ids": ids}
            if not decode:
                # a mixed step's media prompt rows
                self._pack_media(out, graph_walk, inputs)
            return out
        out = {"input_ids": torch.cat([inp.input_ids for inp in inputs], dim=0)}
        self._pack_media(out, graph_walk, inputs)
        return out

    def _pack_media(self, out: dict, graph_walk: str, inputs: list[ARNodeInputs]) -> None:
        """A media step's ``MEDIA_EMBEDS``, its rows' in row order as the
        forward's cumulative mask count reads them, and its ``MEDIA_MASK``: one
        host mask and one copy, a row without media (a decode or text row)
        masking nothing. Every step of a walk a media capture replays brings
        the mask, zeros when no row has media, so a replay never splices in
        the rows a previous step left in the static buffer."""
        if graph_walk not in self._media_walks and not (graph_walk == MIXED and self._mixes_media):
            return
        rows: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        plain = 0
        for inp in inputs:
            mask = inp.tensor_inputs.get(MEDIA_MASK)
            if mask is None:
                plain += inp.input_seq_len
                continue
            if plain:
                masks.append(torch.zeros(plain, dtype=torch.bool))
                plain = 0
            masks.append(mask)
            rows.append(inp.tensor_inputs[MEDIA_EMBEDS])
        if plain or not masks:
            masks.append(torch.zeros(plain, dtype=torch.bool))
        if rows:
            out[MEDIA_EMBEDS] = rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)
        out[MEDIA_MASK] = to_device_nonblocking(
            masks[0] if len(masks) == 1 else torch.cat(masks), out["input_ids"].device,
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        per_request_info: Mapping[int, CurrentForwardPassInfo] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        # A prompt's ids, media pads and sentinels included, feed the
        # repetition penalty, as in transformers and vLLM. A mixed step's
        # decode rows track their token through the sampler as usual.
        prefill_tokens = {
            rid: inp.input_ids for rid, inp in zip(request_ids, inputs, strict=True)
            if (inp.graph_walk or graph_walk) not in self._decode_walks
        } if graph_walk not in self._decode_walks else {}

        # None advances by the span; a media prompt placing its own
        # positions advances by what it says (see `_media_inputs`), and so does
        # each such row of a mixed step, beside rows advancing by their span
        def places_own(inp: ARNodeInputs) -> bool:
            return (inp.graph_walk or graph_walk) in self._media_walks and inp.resource_step_info is not None

        advance = None
        if any(places_own(inp) for inp in inputs):
            advance = tuple(
                int(inp.resource_step_info) if places_own(inp) else int(inp.input_seq_len) for inp in inputs
            )

        step = self.node_step(
            [
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            prefill_tokens,
            advance,
        )
        if not self._decode_rows_mix:
            return step
        # the capture the lease took, from the same rows' walks
        return replace(step, cg_key_info=self.cg_key_info(graph_walk, per_request_info or {}))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        **tensors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """``new_token``, one a row; and with hidden walks, each row's
        `ROW_IDS` and `ROW_HIDDEN` (its last token's)."""
        # a captured graph stages only tensors; anything else would be baked in at capture
        assert all(isinstance(t, torch.Tensor) for t in tensors.values()), (
            f"preprocess outputs the forward reads must be tensors: {sorted(tensors)}"
        )
        if graph_walk == MIXED:
            graph_walk = self._step_walk(graph_walk, engine_inputs.per_request_info.values())
        if input_ids is None:
            input_ids = self.loopback_tokens(engine_inputs)
        media, mask = tensors.pop(MEDIA_EMBEDS, None), tensors.pop(MEDIA_MASK, None)
        decoder_kwargs = self.decoder_kwargs(graph_walk, engine_inputs, input_ids.shape[0], **tensors)
        embeds = self.model.embed(input_ids)
        # a host shape: an eager step without media skips it
        if media is not None and media.shape[0]:
            embeds = splice_media(embeds, mask, media)
        hidden = self.model(embeds, label="main", **decoder_kwargs)
        out = {}
        if graph_walk not in self._decode_walks:
            # only the last token of each prefill row feeds the head
            if self._hidden_walks:
                out[ROW_IDS] = self.last_token_hidden(engine_inputs, input_ids)
            hidden = self.last_token_hidden(engine_inputs, hidden)
        elif self._hidden_walks:
            out[ROW_IDS] = input_ids
        if self._hidden_walks:
            out[ROW_HIDDEN] = hidden
        out["new_token"] = self.sample_tokens(engine_inputs, self.model.logits(hidden))
        return out

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        **tensors: torch.Tensor,
    ) -> NameToTensorList:
        return self._forward(graph_walk, engine_inputs, input_ids, **tensors)

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        **tensors: torch.Tensor,
    ) -> BatchedModelOutput:
        out = self._forward(graph_walk, engine_inputs, input_ids, **tensors)
        new_tokens = out["new_token"]
        # Row i is request i: one clone and one D2H copy per step instead of
        # one per request.
        return BatchedModelOutput(
            row_outputs=out,
            # A keyed label's `PrefixKeys` names the prompt's `text_inputs`; its replies extend the
            # keys from this host copy. One tensor under two names is copied once.
            check_stop_buffers={"new_token": new_tokens, "text_inputs": new_tokens},
        )

    # ------------------------------------------------------------------
    # Post-step
    # ------------------------------------------------------------------

    def step_is_input_free(self, graph_walk: str) -> bool:
        # decode rows are one token each; the prefill walks feed
        # `prefill_tokens` from the inputs
        return graph_walk in self._decode_walks

    def inline_signals(self, graph_walk: str) -> dict[str, str]:
        # every decode walk's client token rides inline, off the stop check's host copy
        loopback = self.device_loopback
        if loopback is not None and graph_walk in self._decode_walks:
            return {loopback.signal: loopback.buffer}
        return super().inline_signals(graph_walk)

    @property
    def runs_batch_postprocess(self) -> bool:
        # this class's tail has rows only where the graph routes hidden rows;
        # a subclass's own tail always runs
        return bool(self._hidden_walks) or type(self).batch_postprocess is not SampledTokenDecoder.batch_postprocess

    def batch_postprocess(
        self,
        outputs: BatchedModelOutput,
        request_infos: Mapping[str, CurrentForwardPassInfo],
        inputs: list[ARNodeInputs],
        piecewise_runners: Mapping[str, Any] | None = None,
    ) -> None:
        """A row whose own walk (in a mixed step too) routes `ROW_IDS` and
        `ROW_HIDDEN` keeps them, each in its own copy: its loop may accumulate
        them for the reply's life, and a view of the step's batch clone, or of
        any copy shared with its step-mates, would keep all their rows alive
        that long. Any other row drops them. A rows-only step routes nothing
        off the device."""
        if not self._hidden_walks or not outputs.per_rid_outputs:
            return
        for rid, info in request_infos.items():
            out = outputs.get(rid)
            if out is None or ROW_HIDDEN not in out:
                continue
            if info.graph_walk in self._hidden_walks:
                for name in (ROW_IDS, ROW_HIDDEN):
                    out[name] = [row.clone() for row in out[name]]
            else:
                del out[ROW_IDS], out[ROW_HIDDEN]
