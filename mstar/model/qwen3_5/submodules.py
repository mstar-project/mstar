"""Qwen3.5's node submodules: the hybrid LLM and the vision encoder.

``prefill_vision`` hands the LLM encoder embeddings rather than ids, so steps
are declared off ``input_seq_len``. The rotation is Qwen3.5's interleaved 3D
MRoPE (``components/rope.py``): the position resource is only a per-request
counter, and the 3D ids are built here and passed in as cos/sin.
"""

import logging
import os
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    DETERMINISTIC_COMPILE_MODE,
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    PiecewiseAcceleratorGraphConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewisePackedConfig,
    decode_rows_then_prompt,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv.config import KVStep
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.position.config import PositionStep
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.engine.resources.sampler.config import SamplerStep
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import Segment, SlotLease, SubmoduleStep
from mstar.model.components.mrope import mrope_position_ids
from mstar.model.components.qwen3_vit import (
    vision_interpolation,
    vision_seq_lengths,
)
from mstar.model.components.qwen3_vit import (
    # the tower's 2D patch grid, not `rope`'s 3D MRoPE ids of the same name
    vision_position_ids as vision_grid_position_ids,
)
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
from mstar.model.qwen3_5.components.rope import (
    text_position_ids,
    vision_position_advance,
    vision_position_ids,
)
from mstar.model.qwen3_5.components.vision import Qwen3_5VisionModel
from mstar.model.qwen3_5.config import (
    ATTN,
    GDN_STATE,
    KV_CACHE,
    LINEAR_ATTN,
    LLM_MIXED,
    MIXED_MAX_PREFILL_ROWS,
    ROPE,
    SAMPLER,
    VISION_ATTN,
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from mstar.model.qwen3_5.qwen3_5_model import STEP_GRIDS_KEY, TEXT_PART, VIDEO_PART
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    DeviceLoopback,
    InputSeqLenInfo,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    TokenStop,
)
from mstar.utils.h2d import to_device_nonblocking
from mstar.utils.knobs import compile_deterministic, ingraph_decode_rope
from mstar.utils.profiler import PHASE_PERIOD, PhaseLaps, phase_record

logger = logging.getLogger(__name__)


def token_stop(stop_token_ids) -> TokenStop:
    """The LLM's stop: a decode row's stop id, checked by the graph runtime."""
    return TokenStop(
        loop="decode_loop", buffer="new_token", stop_ids=frozenset(stop_token_ids),
        sampler=SAMPLER, walk="decode", prefill_walks=frozenset({"prefill_text", "prefill_vision"}),
        last_prefill_flag="last_prefill", runtime_check=True,
    )


class LLMSubmodule(ARNodeSubmodule):
    # Total tokens of a packed text-prefill step, one captured graph per
    # (bucket, batch size). A prompt past the top bucket prefills eagerly, and
    # an eager 8k prefill is launch-bound (~1,100 kernel launches, the GPU
    # idle half the time: 78 ms of wall for 36 ms of kernels on 0.8B), so the
    # ladder runs past 8192. The powers of two stay, since a batch of prompts
    # up to a power of two long fills one exactly. 2176, 4352 and 8704 add 1/16
    # above them for the decode rows a mixed step carries beside four prompts
    # and for batches a few tokens over, which would otherwise pad to twice the
    # width. MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS narrows or extends it (static
    # buffers are sized by the largest bucket).
    PREFILL_TOKEN_BUCKETS = [
        int(x) for x in os.environ.get(
            "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS", "32,64,128,256,512,1024,2048,2176,4096,4352,8192,8704",
        ).split(",")
    ]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]
    # A mixed step carries the decode batch beside its prompt rows, so its
    # captures run to the decode batch: one capped below it splits the batch.
    MIXED_CAPTURE_BATCH_SIZES = [4, 16, 64, 128]
    # Four prompts and their decode rows reach 2176, the only slack bucket a
    # mixed step needs. Every (rows, tokens) pair here is one more recompile of
    # the forward all these captures share, and past Dynamo's limit the buckets
    # compiled last run uncompiled inside their graphs.
    MIXED_TOKEN_BUCKETS = [b for b in PREFILL_TOKEN_BUCKETS if b <= 2176 or b & (b - 1) == 0]
    # Capture rows and a replay's padding rows address the pool's sink and
    # hold no slot, so these buckets do not size `gdn_state.max_slots`; that is
    # set by the concurrency a deployment wants (see configs/qwen3_5_*.yaml).
    # The ladder runs to 128 because the largest bucket is also the scheduler's
    # cap on rows per decode step: with 32 as the top, 64 requests in flight
    # ran as two alternating batches of 32 and each paid the whole per-step
    # host cost. A step above 32 rows pads to the next bucket.
    # MSTAR_DECODE_BUCKETS="1,2,4,8,16,32" narrows the ladder for every model
    # that reads it (less graph memory, or the pre-#258 ladder for an A/B);
    # MSTAR_QWEN35_DECODE_BUCKETS overrides it for this model alone.
    DECODE_CAPTURE_BATCH_SIZES = [
        int(x) for x in (
            os.environ.get("MSTAR_QWEN35_DECODE_BUCKETS")
            or os.environ.get("MSTAR_DECODE_BUCKETS", "1,2,4,8,16,32,48,64,96,128")
        ).split(",")
    ]

    # Text plus image tokens of a whole prompt. Capped at 4096, not the
    # processor's 16384, because static buffers are sized by the largest
    # bucket; a longer prompt runs eagerly.
    PREFILL_VISION_TOKEN_BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096]
    # Requests per `prefill_vision` step, so a packed step stays inside the
    # 4096-token top bucket: 4 fit ~1000 tokens a request (a ~1000 px square
    # image plus text). Past the top bucket the step runs eagerly and every
    # new packed length recompiles. Chunked prefill replaces this cap.
    PREFILL_VISION_MAX_BATCH_SIZE = 4
    PREFILL_VISION_CAPTURE_BATCH_SIZES = [1, 2, 4]

    def __init__(
        self,
        model: Qwen3_5ForCausalLM,
        config: Qwen3_5Config,
        vision_config: Qwen3_5VisionConfig | None = None,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.vision_config = vision_config
        self.token_stop = token_stop(config.stop_token_ids)
        # the decode walk reads its input ids off the sampler's slot masters
        self.device_loopback = DeviceLoopback(
            walk="decode", signal="text_inputs", sampler=SAMPLER, buffer="new_token",
        )
        self._vision_sentinels: tuple[torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        """Decode, text prefill and vision prefill all capture."""
        def dummy(n: int) -> ARNodeInputs:
            return ARNodeInputs(
                input_ids=torch.zeros(n, dtype=torch.long, device=device),
                input_seq_len=n,
            )

        def vision_dummy(n: int) -> ARNodeInputs:
            return ARNodeInputs(
                input_seq_len=n,
                input_embeds=torch.zeros(
                    (n, self.config.hidden_size),
                    device=device, dtype=self.model.model.embed_tokens.weight.dtype,
                ),
                custom_pos_ids=torch.zeros(
                    (3, n), dtype=torch.float, device=device,
                ),
                # read by `declare_step`; replay restages the real values
                tensor_inputs={
                    "mrope_advance": max(n - 2, 0),
                    "text_token_ids": torch.zeros(
                        n, dtype=torch.long, device=device,
                    ),
                },
            )

        # Prefill loses nothing to deterministic mode and compiles faster.
        # Decode keeps its tuned kernels, which win at one row, unless the
        # deployment asks for output that reproduces across processes: both
        # the cost and that guarantee sit on this graph.
        decode_compile_mode = DETERMINISTIC_COMPILE_MODE if compile_deterministic() else None

        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=dummy(1),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
                compile_mode=decode_compile_mode,
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill_text",
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                make_node_input=dummy,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
                compile_mode=DETERMINISTIC_COMPILE_MODE,
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk=LLM_MIXED,
                capture_token_lengths=self.MIXED_TOKEN_BUCKETS,
                make_node_input=dummy,
                capture_batch_sizes=self.MIXED_CAPTURE_BATCH_SIZES,
                # the layout its GDN splits; a replay's padding rows stay empty
                token_layout=decode_rows_then_prompt,
                compile_mode=DETERMINISTIC_COMPILE_MODE,
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill_vision",
                capture_token_lengths=self.PREFILL_VISION_TOKEN_BUCKETS,
                make_node_input=vision_dummy,
                capture_batch_sizes=self.PREFILL_VISION_CAPTURE_BATCH_SIZES,
                compile_mode=DETERMINISTIC_COMPILE_MODE,
            ),
        ]

    # ------------------------------------------------------------------
    # Step declaration
    # ------------------------------------------------------------------

    def _sentinel_embeds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``<|vision_start|>`` / ``<|vision_end|>`` embeddings, cached.

        `split_around_spans` drops them with the pad interior, so this walk
        re-adds them.
        """
        if self._vision_sentinels is None:
            ids = torch.tensor(
                [self.config.vision_start_token_id, self.config.vision_end_token_id],
                dtype=torch.long, device=self.get_device(),
            )
            embeds = self.model.model.embed_tokens(ids)
            self._vision_sentinels = (embeds[:1], embeds[1:])
        return self._vision_sentinels

    def _vision_inputs(
        self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList,
    ) -> ARNodeInputs:
        """A whole multimodal prompt as one row, spliced in prompt order.

        `prefill_order` tags each part text, image or video frame group; the
        nth tag of a kind takes the nth tensor of that kind, except that a
        video's tags, one per temporal group, share its one grid and walk
        through it. A group is placed like a one-frame image; the timestamp
        text between groups arrives as ordinary text parts.

        An image spans ``max(h', w')`` MRoPE positions but ``t * h' * w'``
        tokens, so the cursor advances by the real amount and `declare_step`
        hands the total to the position resource; its default (the token
        count) would misplace everything after the first image.
        """
        if self.vision_config is None:
            raise ValueError(
                "prefill_vision needs the vision config for its spatial merge "
                "size; this LLM submodule was built without one"
            )
        device = self.get_device()
        merge = self.vision_config.spatial_merge_size
        texts = inputs.get("text_inputs", [])
        grids = fwd_info.step_metadata[STEP_GRIDS_KEY]
        packed = inputs["vision_embeds"][0].to(device)
        start_embed, end_embed = self._sentinel_embeds()

        pos = self.node_resources[ROPE].position(
            rid=fwd_info.request_id, label="main",
        )
        start_pos = pos
        embeds: list[torch.Tensor] = []
        pos_ids: list[torch.Tensor] = []
        tracked: list[torch.Tensor] = []
        text_i = grid_i = 0
        # frame groups of the video at `grid_i` already placed
        group = 0
        # into `packed`, which holds every image's and video's merged tokens end to end
        cursor = 0

        for kind in fwd_info.step_metadata.get("prefill_order", ()):
            if kind == TEXT_PART:
                ids = texts[text_i].to(device)
                text_i += 1
                span = ids.shape[0]
                embeds.append(self.model.model.embed_tokens(ids))
                pos_ids.append(text_position_ids(span, pos))
                tracked.append(ids)
                pos += span
                continue
            t, h, w = grids[grid_i]
            if kind == VIDEO_PART:
                group += 1
                if group == t:
                    grid_i, group = grid_i + 1, 0
                t = 1
            else:
                grid_i += 1
            grid = torch.tensor((t, h, w))
            span = t * (h // merge) * (w // merge)
            advance = vision_position_advance(grid, merge)
            embeds += [start_embed, packed[cursor:cursor + span], end_embed]
            # built on the host, copied once below
            pos_ids += [
                text_position_ids(1, pos),
                vision_position_ids(grid, merge, pos + 1),
                text_position_ids(1, pos + 1 + advance),
            ]
            cursor += span
            # two sentinels either side, and the image between them
            pos += advance + 2

        input_embeds = torch.cat(embeds, dim=0)
        return ARNodeInputs(
            input_seq_len=input_embeds.shape[0],
            input_embeds=input_embeds,
            custom_pos_ids=to_device_nonblocking(torch.cat(pos_ids, dim=1), device),
            tensor_inputs={
                "mrope_advance": pos - start_pos,
                # Ids cannot ride `input_ids` — that is what `preprocess` keys
                # the embeds-vs-ids branch on — so the repetition penalty
                # takes them from here.
                "text_token_ids": (
                    torch.cat(tracked) if tracked
                    else torch.zeros(0, dtype=torch.long, device=device)
                ),
            },
        )

    # on a prefill capture bucket (PREFILL_TOKEN_BUCKETS), since a step over
    # it pads to the next bucket up. 2048 measured best on 0.8B-9B:
    # 1024 costs 3-10% throughput in steps and speculation, 4096 gives back
    # most of the short-prompt TTFT gain and doubles decode ITL p99 at c48
    MAX_BATCH_TOKENS = 2048

    def get_input_sequence_len(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList, **kwargs,
    ) -> InputSeqLenInfo | None:
        # a vision row's length depends on its image grids; it is admitted whole
        if graph_walk == "decode":
            return InputSeqLenInfo(1)
        if graph_walk == "prefill_text" and "text_inputs" in inputs:
            return InputSeqLenInfo(inputs["text_inputs"][0].shape[0])
        return None

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        has_loopback: bool = False,
        **kwargs: Any,
    ) -> ARNodeInputs:
        if graph_walk == "prefill_vision":
            return self._vision_inputs(fwd_info, inputs)

        if (row := self.uniform_row_inputs(graph_walk, has_loopback)) is not None:
            # one token per row, read off the sampler's slot master in
            # `preprocess`; the routed signal is empty (or ignored) here
            return row
        input_ids = inputs["text_inputs"][0]
        return ARNodeInputs(input_seq_len=input_ids.shape[0], input_ids=input_ids)

    @staticmethod
    def _rope_in_graph(engine_inputs: ModelInputsFromEngine, graph_walk: str) -> bool:
        """A step of a text walk replayed on a leased slot, or being captured
        for one, whose rotary tables come from the planned positions inside
        the graph: always for decode, with ``ingraph_decode_rope`` on for text
        prefill and mixed steps. Only ``prefill_vision`` rows bring their own
        3D positions."""
        if not engine_inputs.captured:
            return False
        return graph_walk == "decode" or (graph_walk in ("prefill_text", LLM_MIXED) and ingraph_decode_rope())

    def _decode_cos_sin(
        self, engine_inputs: ModelInputsFromEngine, num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The rotary tables of a decode step from the position resource's
        planned positions: one token per row, no custom grids."""
        pos_ids = self.node_resources[ROPE].pos_ids("main")
        assert pos_ids is not None, (
            "position resource has no plan for this step; `plan` must run "
            "before the forward"
        )
        pos = pos_ids[:num_tokens].unsqueeze(0).expand(3, -1)
        return self.model.model.build_cos_sin(
            pos, dtype=self.model.model.embed_tokens.weight.dtype,
        )

    def _position_ids_3d(self, inputs: list[ARNodeInputs]) -> torch.Tensor:
        return mrope_position_ids(self.node_resources[ROPE].pos_ids("main"), inputs)

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
        out: dict[str, torch.Tensor | Any] = {}
        in_graph = self._rope_in_graph(engine_inputs, graph_walk)
        first = inputs[0]
        if first.input_embeds is None and (
            first.input_ids is None or any(inp.input_ids is None for inp in inputs)
        ):
            # device loop-back: every row of a decode step is its request's
            # last sampled token, read by slot (padding rows read slot 0)
            sampler: SamplerResource = engine_inputs.resources[SAMPLER]
            t0 = perf_counter() if PHASE_PERIOD else 0.0
            laps = PhaseLaps("engine.preprocess.loopback.")
            ids = sampler.loopback_tokens(engine_inputs.request_ids)
            laps.lap("tokens")
            decode = graph_walk == "decode"
            # rows whose last step ran elsewhere brought their token
            if not decode or any(
                inp.input_ids is not None and inp.input_seq_len for inp in inputs
            ):
                ids = self._splice_row_ids(ids, inputs, decode)
            laps.lap("splice")
            out["input_ids"] = ids
            if PHASE_PERIOD:
                phase_record("engine.preprocess.loopback", perf_counter() - t0)
            if in_graph:
                # the captured forward builds the rotary tables itself
                return out
        elif inputs[0].input_ids is not None:
            out["input_ids"] = torch.cat([inp.input_ids for inp in inputs], dim=0)
            if in_graph:
                assert all(inp.custom_pos_ids is None for inp in inputs), (
                    "a text walk's rows take the planned 1D positions"
                )
                return out
        else:
            out["input_embeds"] = torch.cat(
                [inp.input_embeds for inp in inputs], dim=0,
            )

        laps = PhaseLaps("engine.preprocess.")
        position_ids_3d = self._position_ids_3d(inputs)  # (3, total_tokens)
        laps.lap("position_ids")
        cos, sin = self.model.model.build_cos_sin(
            position_ids_3d, dtype=self.model.model.embed_tokens.weight.dtype,
        )
        laps.lap("cos_sin")
        out["cos_3d"] = cos
        out["sin_3d"] = sin
        return out

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        per_request_info: Mapping[int, CurrentForwardPassInfo] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        prefill_tokens = {}
        if graph_walk in ("prefill_text", LLM_MIXED):
            # a mixed step's decode rows track their token through the sampler as usual
            prefill_tokens = {
                rid: inp.input_ids
                for rid, inp in zip(request_ids, inputs, strict=True)
                if (inp.graph_walk or graph_walk) == "prefill_text"
            }
        elif graph_walk == "prefill_vision":
            # the prompt's text ids still feed the repetition penalty
            prefill_tokens = {
                rid: inp.tensor_inputs["text_token_ids"]
                for rid, inp in zip(request_ids, inputs, strict=True)
            }

        kept_rids = None
        if graph_walk != "decode":
            # decode rows, and the last prefill, keep their token through `postprocess`
            row_walks = [inp.graph_walk or graph_walk for inp in inputs]
            info = per_request_info or {}
            kept_rids = frozenset(
                rid for rid, walk in zip(request_ids, row_walks, strict=True)
                if walk == "decode" or (
                    rid in info and info[rid].step_metadata.get("last_prefill", False)
                )
            )

        # `advance=None` advances by the span, which is wrong for an image:
        # its 3D grid covers fewer positions than tokens.
        advance = None
        if graph_walk == "prefill_vision":
            advance = tuple(
                int(inp.tensor_inputs["mrope_advance"]) for inp in inputs
            )

        return SubmoduleStep(
            segments=[
                Segment(
                    request_id=rid,
                    label="main",
                    span=inp.input_seq_len,
                )
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                KV_CACHE: KVStep(),
                ATTN: AttentionStep(causal=True),
                GDN_STATE: RecurrentStep(),
                LINEAR_ATTN: LinearAttnStep(),
                SAMPLER: SamplerStep(
                    apply_penalty=True,
                    prefill_tracked_tokens=prefill_tokens,
                    kept_rids=kept_rids,
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
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn: AttentionManager = engine_inputs.resources[ATTN]
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        if input_ids is None and input_embeds is None:
            input_ids = sampler.loopback_tokens(engine_inputs.request_ids)
        if self._rope_in_graph(engine_inputs, graph_walk) or cos_3d is None or sin_3d is None:
            # the tables come from the planned positions, inside the graph
            # (off a captured decode's per-step critical path); the static
            # cos/sin stay unread
            num_tokens = (input_ids if input_ids is not None else input_embeds).shape[0]
            cos_3d, sin_3d = self._decode_cos_sin(engine_inputs, num_tokens)

        embeds = (
            self.model.model.embed_tokens(input_ids)
            if input_embeds is None
            else input_embeds
        )
        hidden = self.model.model(embeds, label="main", cos_sin=(cos_3d, sin_3d))

        if graph_walk != "decode":
            # only the last token of each prefill row feeds the head
            hidden = attn.select_last_hidden(hidden)
        logits = self.model.lm_head(hidden)
        return sampler.sample(engine_inputs.request_ids, logits=logits)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {
            "new_token": self._forward(
                graph_walk, engine_inputs, cos_3d, sin_3d,
                input_ids=input_ids, input_embeds=input_embeds,
            )
        }

    def can_batch(
        self, batch: ExecutingBatch, model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        if graph_walk == "prefill_vision":
            return self.PREFILL_VISION_MAX_BATCH_SIZE
        return None

    def max_rows_in_combined(self, graph_walk: str) -> int | None:
        # a mixed step's prompt rows take the chunked GDN path, sized to these
        return MIXED_MAX_PREFILL_ROWS if graph_walk == "prefill_text" else None

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> BatchedModelOutput:
        new_tokens = self._forward(
            graph_walk, engine_inputs, cos_3d, sin_3d,
            input_ids=input_ids, input_embeds=input_embeds,
        )
        # Row i is request i: one clone and one D2H copy per step instead of
        # one per request.
        return BatchedModelOutput(
            row_outputs={"new_token": new_tokens},
            # The prefix cache keys a reply from this host copy, under `text_inputs`, the tensor the
            # label's `PrefixKeys` names.
            check_stop_buffers={"new_token": new_tokens, "text_inputs": new_tokens},
        )

    # ------------------------------------------------------------------
    # Post-step
    # ------------------------------------------------------------------

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        if not self._token_is_real(request_info):
            outputs.pop("new_token", None)
            return
        # Rebind, not copy: the decode loop routes on `text_inputs`. EOS is
        # tested in `check_stop` so the GPU thread never syncs on `.item()`.
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    @staticmethod
    def _token_is_real(request_info: CurrentForwardPassInfo) -> bool:
        """Whether this step's token is the request's: decode, or the last prefill's."""
        return request_info.graph_walk == "decode" or request_info.step_metadata.get(
            "last_prefill", False,
        )

    def step_is_input_free(self, graph_walk: str) -> bool:
        # decode rows are one token each, like the template; the prefill walks
        # feed `prefill_tokens` from the inputs
        return graph_walk == "decode"

    def speculative_input_seq_len(self, graph_walk, fwd_info, prev_seq_len):
        if graph_walk == "decode":
            return 1
        return super().speculative_input_seq_len(graph_walk, fwd_info, prev_seq_len)


# The vision tower's block loop, as a piecewise capture region
VISION_BLOCK_LOOP = "vision_block_loop"


class VisionEncoderSubmodule(NodeSubmodule):
    """Qwen3.5's ViT, which turns pixel patches into LLM-width embeddings.

    A step's images go through packed, in one call. Grid maths runs in
    `prepare_inputs` and attention boundaries are declared as step segments,
    so the compiled tower's only symbolic shape is its token count.

    Eager, the tower is launch-bound (~500 kernels, ~4 ms of GPU work in a
    ~20 ms forward on 9B), so the block loop is captured, and a step's
    requests replay together as one packed row: every image and video frame
    group of every request is its own attention segment.

    A replay's attention plan holds at most the vision attention resource's
    ``max_segments_per_step`` segments. A batch that would overflow it or
    the top patch bucket runs its requests one at a time instead, each
    replaying if it fits alone. A request too large alone (a long video)
    runs the block loop uncaptured; while the block loop is captured for
    other steps, it gets a step of its own (`cg_key_info`).
    """

    # Patch-count buckets of one replay. 16384 patches merge to the LLM's
    # 4096 `prefill_vision` bucket; a larger step runs uncaptured.
    BLOCK_LOOP_PATCH_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384]
    # Requests per encoder step. They share one replay, so a burst of image
    # requests pays the per-step host cost once per this many, not per image.
    MAX_BATCH_SIZE = 8

    # The engine would compile `forward_batched` whole, and its per-step
    # bookkeeping (each request's slice of the output, the segment lengths)
    # guards on the batch size: every new size recompiled (~0.6 s, on a live
    # request). The tensor stages are compiled on their own instead
    # (`compile_regions`), dynamic in the patch count, so one graph serves
    # every capture bucket and the uncaptured steps past the top one.
    # Uncompiled, the blocks' fp32 rope, norms and activations were over half
    # the tower's GPU time on long videos.
    disable_torch_compile = True

    def __init__(
        self, model: Qwen3_5VisionModel, config: Qwen3_5VisionConfig,
        pixel_normalizer: tuple[torch.Tensor, torch.Tensor],
    ):
        super().__init__()
        self.model = model
        self.config = config
        # per-column (mean, std) for byte patches; see `Qwen3_5DenseModel.pixel_normalizer`
        self._pixel_normalizer = pixel_normalizer

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        device = self.get_device()
        merge = self.config.spatial_merge_size
        grid = fwd_info.step_metadata[STEP_GRIDS_KEY]
        # Built here, not in the compiled forward: a `.tolist()` there breaks
        # the graph and makes h, w symints inductor cannot codegen from. And
        # built on the host: on the device, the grid upload and the
        # `repeat_interleave`s each block on the stream.
        indices, weights = vision_interpolation(
            grid, self.config.num_grid_per_side, merge, torch.device("cpu"),
        )
        num_patches = sum(t * h * w for t, h, w in grid)
        return NodeInputs(
            # what the engine leases the captured block loop's bucket by
            input_seq_len=num_patches,
            tensor_inputs={
                "pixel_values": torch.cat(inputs["pixel_values"], dim=0),
                "indices": to_device_nonblocking(indices, device),
                "weights": to_device_nonblocking(weights, device),
                "position_ids": to_device_nonblocking(
                    vision_grid_position_ids(grid, merge, torch.device("cpu")), device,
                ),
                # not forward args: `declare_step` makes segments of the
                # lengths, `forward_batched` cuts the output at the count
                "seq_lengths": vision_seq_lengths(grid),
            },
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep | None:
        # leased, the captured block loop plans its own attention per replay
        if (piecewise_leases or {}).get(VISION_BLOCK_LOOP):
            return None
        # one segment per frame: a frame attends to itself alone
        segments = [
            Segment(request_id=rid, label="main", span=span)
            for rid, inp in zip(request_ids, inputs, strict=True)
            for span in inp.tensor_inputs["seq_lengths"]
        ]
        return SubmoduleStep(
            segments=segments,
            steps={VISION_ATTN: AttentionStep(causal=False)},
        )

    def can_batch(
        self, batch: ExecutingBatch, model_inputs: list[NodeInputs],
    ) -> bool:
        # packed, the requests must fit one replay together. Applied with the
        # block loop uncaptured too (capture off, or a node without the
        # runner), where it only splits a large step into per-request runs.
        return self._fits_capture(model_inputs)

    def split_batches_by_capture_key(self, graph_walk: str) -> bool:
        return True

    def cg_key_info(
        self, graph_walk: str, per_request_info: dict, **kwargs: Any,
    ) -> Any:
        """The scheduler's group (``split_batches_by_capture_key``): requests
        that each fit a replay alone pack together (None); one too large for
        a replay (a long video) is a group of its own, so it gets a step of
        its own. Beside others it would run back to back with them in one
        step (`can_batch`), which holds every request's embeddings, and so
        their prefills, until the last one is encoded."""
        return tuple(
            rid for rid, info in per_request_info.items()
            if not self._fits_replay(
                sum(t * h * w for t, h, w in info.step_metadata[STEP_GRIDS_KEY]),
                sum(t for t, _, _ in info.step_metadata[STEP_GRIDS_KEY]),
            )
        ) or None

    def max_batch_size(self, graph_walk: str) -> int | None:
        return self.MAX_BATCH_SIZE

    def compile_regions(self) -> dict[str, list[tuple]]:
        """Warmed at start-up on the smallest image the processor makes (256 x
        256 pixels), in the dtypes a step brings: float32 pixels out of
        `_normalize`, merger rows in the tower's dtype. Dynamic in the patch
        count from that call, so no image compiles them on a live request. The
        block loop takes no example (it needs a planned attention step): it
        compiles in its capture's warm-up, or on the first image when capture
        is off."""
        device, merge = self.get_device(), self.config.spatial_merge_size
        grid = [(1, 16, 16)]
        indices, weights = vision_interpolation(grid, self.config.num_grid_per_side, merge, device)
        patch = self.config.in_channels * self.config.temporal_patch_size * self.config.patch_size ** 2
        pixels = torch.zeros(16 * 16, patch, device=device)
        rows = torch.zeros(16 * 16, self.config.hidden_size, dtype=next(self.model.merger.parameters()).dtype,
                           device=device)
        return {
            "model.embed": [(pixels, indices, weights, vision_grid_position_ids(grid, merge, device))],
            "model.encode": [],
            "model.merger": [(rows,)],
        }

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype,
        tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """The block loop, one replay per step, by patch-count bucket.

        Patch embed, position resample and rope run eagerly before it, the
        merger after (it quarters the row count, which a region's output view
        cannot express). Batch size 1: a replay is the whole step as one row,
        every request's segments in it.
        """
        hidden, head_dim = self.config.hidden_size, self.config.head_dim

        def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            n = shape.total_tokens
            return {
                "hidden": torch.zeros(n, hidden, dtype=autocast_dtype, device=device),
                "cos": torch.zeros(n, head_dim, dtype=autocast_dtype, device=device),
                "sin": torch.zeros(n, head_dim, dtype=autocast_dtype, device=device),
            }

        return {
            VISION_BLOCK_LOOP: PiecewisePackedConfig(
                capture_fn=self._capture_block_loop,
                make_static_inputs=make_static_inputs,
                attn_key=VISION_ATTN,
                lease_before_step=True,
                one_row_per_step=True,
                can_lease=self._fits_capture,
                total_tokens=self.BLOCK_LOOP_PATCH_BUCKETS,
                capture_batch_sizes=[1],
            )
        }

    def _fits_capture(self, inputs: list[NodeInputs]) -> bool:
        """Whether the batch fits one replay (`_fits_replay`)."""
        return self._fits_replay(
            sum(inp.input_seq_len for inp in inputs),
            sum(len(inp.tensor_inputs["seq_lengths"]) for inp in inputs),
        )

    def _fits_replay(self, patches: int, segments: int) -> bool:
        """Whether a step of this many patches and attention segments fits one
        replay: the top patch bucket, and the attention plan's per-step
        segment ceiling, read from the resource so a YAML override holds.

        TODO: capture more than 64 segments, so long low-res videos replay
        too. The ceiling is sized for images (16384 patches / 256-patch
        floor), and raising it pads every replay's plan to it. Sizing each
        bucket's plan by its own token count over the smallest segment (one
        2x2 merge block) would hold any prompt that fits, at small buckets'
        cost unchanged; that is a ragged-resource change.
        """
        # the whole step replays as one row (`one_row_per_step`)
        return (
            patches <= self.BLOCK_LOOP_PATCH_BUCKETS[-1]
            and segments <= self.node_resources[VISION_ATTN].max_segments_per_step
        )

    def _capture_block_loop(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        """The captured region: reads the runner's buffers, never rebinds."""
        s = inp.static_inputs
        return {"hidden": self.model.encode(s["hidden"], s["cos"], s["sin"])}

    def _replay_block_loop(
        self,
        engine_inputs: ModelInputsFromEngine,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seq_lengths: tuple[int, ...],
    ) -> torch.Tensor | None:
        """The block loop replayed once for the whole step, or None if not leased."""
        if VISION_BLOCK_LOOP not in engine_inputs.piecewise_leases:
            return None
        # one replay for every image, not one each: a second plan overwrites the schedule a queued replay still reads
        return engine_inputs.piecewise_runners[VISION_BLOCK_LOOP].run(
            static_inputs={"hidden": hidden, "cos": cos, "sin": sin},
            request_ids=engine_inputs.request_ids,
            seq_lens=list(seq_lengths),
        ).get_view("hidden")

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        """Every request's images and videos packed end to end, in
        `declare_step`'s segment order."""
        device = self.get_device()

        def packed(name: str) -> torch.Tensor:
            parts = [inp.tensor_inputs[name] for inp in inputs]
            return (parts[0] if len(parts) == 1 else torch.cat(parts)).to(device)

        return {
            "pixel_values": self._normalize(packed("pixel_values")),
            "indices": packed("indices"),
            "weights": packed("weights"),
            "position_ids": packed("position_ids"),
            "seq_lengths": tuple(
                n for inp in inputs for n in inp.tensor_inputs["seq_lengths"]
            ),
            # merged tokens per request, to cut the output back apart
            "row_tokens": tuple(
                inp.input_seq_len // self.config.spatial_merge_size ** 2
                for inp in inputs
            ),
        }

    def _normalize(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Byte patches to the processors' float pixels. Eager, not in the
        compiled embed, so the arithmetic (a subtract, a true divide) is the
        processors' exactly."""
        if pixel_values.dtype != torch.uint8:
            # `process_prompt` ships bytes; anything else would skip normalization
            raise ValueError(f"vision patches must be uint8, got {pixel_values.dtype}")
        mean, std = self._pixel_normalizer
        if mean.device != pixel_values.device:
            mean, std = mean.to(pixel_values.device), std.to(pixel_values.device)
            self._pixel_normalizer = (mean, std)
        return (pixel_values.to(torch.float32) - mean) / std

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs,
    ) -> NameToTensorList:
        rid = engine_inputs.request_ids[0]
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[rid]

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
        seq_lengths: tuple[int, ...] = (),
        row_tokens: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        # One call for every image; the ragged kernel's per-frame segments
        # keep them apart.
        hidden, cos, sin = self.model.embed(pixel_values, indices, weights, position_ids)
        encoded = self._replay_block_loop(engine_inputs, hidden, cos, sin, seq_lengths)
        if encoded is None:
            encoded = self.model.encode(hidden, cos, sin)
        embeds = self.model.merger(encoded)
        return self._split_rows(engine_inputs.request_ids, embeds, row_tokens)

    @staticmethod
    def _split_rows(
        request_ids: list[str], embeds: torch.Tensor, row_tokens: tuple[int, ...],
    ) -> dict[str, NameToTensorList]:
        """Each request's slice of the packed output."""
        if len(request_ids) == 1:
            return {request_ids[0]: {"vision_embeds": [embeds]}}
        out, start = {}, 0
        for rid, n in zip(request_ids, row_tokens, strict=True):
            out[rid] = {"vision_embeds": [embeds[start:start + n]]}
            start += n
        return out
