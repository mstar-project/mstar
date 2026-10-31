"""An autoregressive LLM node that samples one token per row per step.

`SampledTokenDecoder` serves the VLM walk set of ``mstar.model.multimodal``
(``prefill_text``, the ``mixed`` combined walk, ``decode``, and one media
walk per subset of the model's `MediaEncoder` s, which prefills a whole
multimodal prompt as one row). It owns the engine contract: the capture
configs, device loop-back of the sampled token, the step declaration around
the sampler and the position resource, the forward through the sampler, and
the stop checks, which are `TokenStepNode`'s (EOS, or the request's token
budget) on decode rows only. A model subclasses it with its resource steps
and its attention key, and an MRoPE model adds its own positions.

A media walk's row is the whole prompt's ids, each attachment expanded to
one pad per token at intake (`expand_media_pads`). Its encoders' outputs
ride as tensors named by `MediaEncoder.output`, and the forward splices them
in at their pads: ``where(pad, media[cumsum(pad) - 1], embed(ids))``. The
contract that makes this capture-safe:

- a row's media tensor holds exactly the pads in its ids, in prompt order
  (`prepare_inputs` checks the row count against the intake's layout), so
  the packed step's media, concatenated in row order, line up with the
  cumulative pad count;
- real tokens precede a replay's padding tail, so the tail's stale ids
  cannot shift a real row's index; and a captured media buffer has at least
  the bucket's tokens rows, so a tail index stays in bounds and
  `splice_media`'s clamp to the last row never binds.

A step that splits a media row (chunked prefill) must slice its media by
the host pad counts of each piece.

The decoder module (``self.model``) provides (``Qwen3DenseLM`` has the first
three):

- ``embed(ids) -> [tokens, hidden]``
- ``forward(input_embeds, *, label, **decoder_kwargs) -> [tokens, hidden]``,
  the stack through its final norm
- ``logits(hidden) -> [rows, vocab]``
- ``embed_dtype``: the embedding's dtype

and ``config`` has ``hidden_size`` and ``stop_token_ids`` (and the model's
``max_position_embeddings`` bounds the decode loop, see `media_walks`).
The node's resources are declared under ``mstar.model.keys``' ``SAMPLER`` and
``ROPE`` keys.
"""

import os
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    decode_rows_then_prompt,
)
from mstar.engine.resources.position.config import PositionStep
from mstar.engine.resources.sampler.config import SamplerStep
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import ResourceStep, Segment, SubmoduleStep
from mstar.model.components.token_step import TokenStepNode
from mstar.model.keys import MIXED, ROPE, SAMPLER
from mstar.model.multimodal import MEDIA_LAYOUT, MediaEncoder, media_walk_names, splice_media
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    HostRows,
    ModelInputsFromEngine,
    device_loopback_enabled,
)
from mstar.utils.profiler import PHASE_PERIOD, phase_record


class SampledTokenDecoder(TokenStepNode, ARNodeSubmodule):
    SAMPLER = SAMPLER
    # the decode walk reads its input ids off the sampler's slot masters
    reads_device_loopback = True

    # Total tokens of a packed text-prefill step, one captured graph per
    # (bucket, batch size); a longer step prefills eagerly. Static buffers are
    # sized by the largest bucket.
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]
    # A mixed step carries the decode batch beside its prompt rows, so its
    # captures run to the decode batch: one capped below it splits the batch.
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

    # The attention resource whose plan picks each prefill row's last token.
    ATTN_KEY: str

    # set by `__init__`; class defaults keep a text-only node's walks working
    _encoders: tuple[MediaEncoder, ...] = ()
    _media_walks: frozenset[str] = frozenset()

    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,
        encoders: Sequence[MediaEncoder] = (),
    ):
        """``encoders`` are the model's media encoders, as its walks declare
        them; a text-only node passes none."""
        super().__init__()
        self.model = model
        self.config = config
        # read once: a config may build the set on every access
        self.stop_token_ids = config.stop_token_ids
        self._encoders = tuple(encoders)
        self._media_walks = frozenset(media_walk_names(self._encoders))

    # ------------------------------------------------------------------
    # What a model declares
    # ------------------------------------------------------------------

    def layer_steps(self) -> dict[str, ResourceStep]:
        """This step's work on the layers' resources (KV, attention, recurrent
        state, ...), in declaration order; the sampler and position steps are
        added around it."""
        raise NotImplementedError

    def prompt_positions(
        self, fwd_info: CurrentForwardPassInfo, layout: dict, num_tokens: int,
    ) -> tuple[torch.Tensor | None, int | None]:
        """A media prompt's own position ids and how far it advances the
        position counter, for a model whose media do not take one position
        per token (MRoPE). ``layout`` is the walk's ``MEDIA_LAYOUT`` (each
        run's offset, tokens and grid). The default, ``(None, None)``, leaves
        the position resource to place every token."""
        return None, None

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
            # every encoder's buffer, `n` rows: a step without that medium
            # leaves it stale, and no pad reads it
            return ARNodeInputs(
                input_seq_len=n,
                input_ids=torch.zeros(n, dtype=torch.long, device=device),
                tensor_inputs={
                    e.output: torch.zeros(
                        (n, self.config.hidden_size), device=device, dtype=self.model.embed_dtype,
                    )
                    for e in self._encoders
                },
            )

        configs = [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ids(1),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill_text",
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                make_node_input=ids,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk=MIXED,
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                make_node_input=ids,
                capture_batch_sizes=self.MIXED_CAPTURE_BATCH_SIZES,
                # the layout the resources split; a replay's padding rows stay empty
                token_layout=decode_rows_then_prompt,
            ),
        ]
        if self._encoders:
            walks = media_walk_names(self._encoders)
            # one capture every media walk replays
            configs.append(PackedAcceleratorGraphConfig(
                capture_graph_walk=walks[0],
                replay_graph_walks=walks,
                capture_token_lengths=self.PREFILL_MEDIA_TOKEN_BUCKETS,
                make_node_input=media,
                capture_batch_sizes=self.PREFILL_MEDIA_CAPTURE_BATCH_SIZES,
                # a media buffer is [tokens, hidden], and hidden can equal a bucket
                input_seq_dims={e.output: 0 for e in self._encoders},
            ))
        return configs

    def max_batch_size(self, graph_walk: str) -> int | None:
        return max(self.PREFILL_MEDIA_CAPTURE_BATCH_SIZES) if graph_walk in self._media_walks else None

    def max_rows_in_combined(self, graph_walk: str) -> int | None:
        return self.MIXED_MAX_PREFILL_ROWS if graph_walk == "prefill_text" else None

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _media_inputs(
        self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList,
    ) -> ARNodeInputs:
        """A whole multimodal prompt as one row: its ids, and each encoder's
        output for the forward to splice in.

        Each medium's row count, none when its output is missing, is checked
        against the pads the intake laid out, on host values only; a mismatch
        fails this request (a captured replay would otherwise splice in the
        previous step's rows). A model placing its own positions
        (`prompt_positions`) hands them over as ``custom_pos_ids``, and its
        counter advance as the row's ``resource_step_info``; `declare_step`
        passes that to the position resource, whose default (the token
        count) would misplace everything after the first image.
        """
        ids = inputs["text_inputs"][0]
        layout = fwd_info.step_metadata.get(MEDIA_LAYOUT) or {}
        media: dict[str, torch.Tensor] = {}
        for e in self._encoders:
            pads = sum(
                n for n, out in zip(layout.get("tokens", ()), layout.get("output", ()), strict=True)
                if out == e.output
            )
            rows = inputs[e.output][0] if inputs.get(e.output) else None
            found = 0 if rows is None else rows.shape[0]
            if found != pads:
                raise ValueError(f"{e.output} has {found} rows for the prompt's {pads} pads")
            if rows is not None:
                media[e.output] = rows
        pos_ids, advance = self.prompt_positions(fwd_info, layout, ids.shape[0])
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
        if graph_walk == "decode" and has_loopback and self._device_loopback():
            # one token per row, read off the sampler's slot master in
            # `preprocess`; the routed signal is empty (or ignored) here
            return ARNodeInputs(input_seq_len=1)
        input_ids = inputs["text_inputs"][0]
        return ARNodeInputs(input_seq_len=input_ids.shape[0], input_ids=input_ids)

    def uniform_row_inputs(
        self, graph_walk: str, has_loopback: bool = False,
    ) -> ARNodeInputs | None:
        # the loop-back decode row above, built once for the step's rows
        if graph_walk == "decode" and has_loopback and self._device_loopback():
            return ARNodeInputs(input_seq_len=1)
        return None

    def _device_loopback(self) -> bool:
        return device_loopback_enabled() and self.node_resources[SAMPLER].has_slot_masters

    def device_loopback_signals(self, graph_walk: str) -> frozenset[str]:
        if graph_walk == "decode" and self._device_loopback():
            return frozenset({"text_inputs"})
        return frozenset()

    @staticmethod
    def _reads_loopback(inputs: Sequence[ARNodeInputs]) -> bool:
        """Whether the step's ids come off the sampler: a row without ids is a
        loop-back decode row."""
        return any(inp.input_ids is None for inp in inputs)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        if self._reads_loopback(inputs):
            # every row's last sampled token, read by slot (padding rows read
            # slot 0); a row whose last step ran elsewhere brought its own
            sampler: SamplerResource = engine_inputs.resources[SAMPLER]
            t0 = perf_counter() if PHASE_PERIOD else 0.0
            ids = sampler.loopback_tokens(engine_inputs.request_ids)
            routed = [
                i for i, inp in enumerate(inputs)
                if inp.input_ids is not None and inp.input_seq_len
            ]
            if graph_walk != "decode":
                # a mixed step packs its decode rows' single tokens beside the
                # prompts: each row takes its own ids, or its loop-back token
                ids = torch.cat([
                    inp.input_ids.to(ids.dtype) if inp.input_ids is not None else ids[i:i + 1]
                    for i, inp in enumerate(inputs)
                ])
            elif routed:
                sampler.overwrite_rows(routed, (ids, torch.cat([inputs[i].input_ids for i in routed])))
            if PHASE_PERIOD:
                phase_record("engine.preprocess.loopback", perf_counter() - t0)
            return {"input_ids": ids}
        out = {"input_ids": torch.cat([inp.input_ids for inp in inputs], dim=0)}
        for e in self._encoders:
            # in row order, as the forward's cumulative pad count reads them
            rows = [inp.tensor_inputs[e.output] for inp in inputs if e.output in inp.tensor_inputs]
            if rows:
                out[e.output] = torch.cat(rows, dim=0)
        return out

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
            if (inp.graph_walk or graph_walk) != "decode"
        } if graph_walk != "decode" else {}

        # None advances by the span; a media prompt placing its own
        # positions advances by what it says (see `_media_inputs`)
        advance = None
        if graph_walk in self._media_walks and inputs[0].resource_step_info is not None:
            advance = tuple(int(inp.resource_step_info) for inp in inputs)

        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                **self.layer_steps(),
                SAMPLER: SamplerStep(
                    apply_penalty=True,
                    prefill_tracked_tokens=prefill_tokens,
                ),
                ROPE: PositionStep(advance=advance),
            },
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        **tensors: torch.Tensor,
    ) -> torch.Tensor:
        # a captured graph stages only tensors; anything else would be baked in at capture
        assert all(isinstance(t, torch.Tensor) for t in tensors.values()), (
            f"preprocess outputs the forward reads must be tensors: {sorted(tensors)}"
        )
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        if input_ids is None:
            input_ids = sampler.loopback_tokens(engine_inputs.request_ids)
        media = {e.output: tensors.pop(e.output) for e in self._encoders if e.output in tensors}
        decoder_kwargs = self.decoder_kwargs(graph_walk, engine_inputs, input_ids.shape[0], **tensors)
        embeds = self.model.embed(input_ids)
        for e in self._encoders:
            rows = media.get(e.output)
            # a host shape: an eager step without this medium skips it
            if rows is not None and rows.shape[0]:
                embeds = splice_media(embeds, e.pads(input_ids), rows)
        hidden = self.model(embeds, label="main", **decoder_kwargs)
        if graph_walk != "decode":
            # only the last token of each prefill row feeds the head
            hidden = engine_inputs.resources[self.ATTN_KEY].select_last_hidden(hidden)
        return sampler.sample(engine_inputs.request_ids, logits=self.model.logits(hidden))

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        **tensors: torch.Tensor,
    ) -> NameToTensorList:
        return {"new_token": self._forward(graph_walk, engine_inputs, input_ids, **tensors)}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        **tensors: torch.Tensor,
    ) -> BatchedModelOutput:
        new_tokens = self._forward(graph_walk, engine_inputs, input_ids, **tensors)
        # Row i is request i: one clone and one D2H copy per step instead of
        # one per request.
        return BatchedModelOutput(
            row_outputs={"new_token": new_tokens},
            # A keyed label's `PrefixKeys` names the prompt's `text_inputs`; its replies extend the
            # keys from this host copy. One tensor under two names is copied once.
            check_stop_buffers={"new_token": new_tokens, "text_inputs": new_tokens},
        )

    # ------------------------------------------------------------------
    # Post-step
    # ------------------------------------------------------------------

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        # only a decode row's stop can end decode_loop; a prefill is not in it yet
        if request_info.graph_walk != "decode":
            return set()
        return super().check_stop(request_id, request_info, outputs)

    def step_is_input_free(self, graph_walk: str) -> bool:
        # decode rows are one token each; the prefill walks feed
        # `prefill_tokens` from the inputs
        return graph_walk == "decode"

    def inline_client_signals(self, graph_walk: str) -> dict[str, str]:
        # the decode loop's client edge carries the sampled token the stop
        # check already has on the host; prefill's new_token edge is persisted
        return {"text_inputs": "new_token"} if graph_walk == "decode" else {}

    def check_stop_batched(
        self,
        request_ids: list[str],
        request_infos: dict[str, CurrentForwardPassInfo],
        host_rows: HostRows,
        step_walk: str | None = None,
    ) -> dict[str, set[str]] | None:
        """``check_stop`` for every request off one ``tolist``."""
        if step_walk is not None and step_walk != "decode":
            return {}  # as check_stop: only a decode row's stop ends decode_loop
        pairs = host_rows.first_values("new_token", request_ids)
        if pairs is None:
            return None
        # a mixed step reads each row's walk; a decode step need not
        mixed = step_walk is None
        return {
            rid: {"decode_loop"} for rid, token in pairs
            if not (mixed and request_infos[rid].graph_walk != "decode")
            and self._stops(token, request_infos[rid])
        }
