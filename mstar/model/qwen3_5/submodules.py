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
from mstar.engine.resources.sampler.unfiltered import with_unfiltered_sampling
from mstar.engine.resources.spec_decode.config import SpecDecodeStep, SpecPhase
from mstar.engine.resources.step import Segment, SlotLease, SubmoduleStep
from mstar.model.components.mrope import mrope_position_ids
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
from mstar.model.qwen3_5.components.rope import (
    text_position_ids,
    vision_position_advance,
    vision_position_ids,
)
from mstar.model.qwen3_5.components.vision import (
    Qwen3_5VisionModel,
    vision_interpolation,
    vision_seq_lengths,
)
from mstar.model.qwen3_5.components.vision import (
    # the tower's 2D patch grid, not `rope`'s 3D MRoPE ids of the same name
    vision_position_ids as vision_grid_position_ids,
)
from mstar.model.qwen3_5.config import (
    ATTN,
    GDN_STATE,
    KV_CACHE,
    LINEAR_ATTN,
    LLM_MIXED,
    MIXED_MAX_PREFILL_ROWS,
    ROPE,
    SAMPLER,
    SPEC_DECODE,
    VISION_ATTN,
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from mstar.model.qwen3_5.qwen3_5_model import TEXT_PART, VIDEO_PART
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    HostRows,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    device_loopback_enabled,
)
from mstar.utils.knobs import ingraph_decode_rope
from mstar.utils.profiler import PHASE_PERIOD, PhaseLaps, phase_record

logger = logging.getLogger(__name__)


class LLMSubmodule(ARNodeSubmodule):
    # the decode walk reads its input ids off the sampler's slot masters
    reads_device_loopback = True
    # drafts a speculative decode step verifies (see `__init__`); 0 decodes one
    # token a step
    num_draft_tokens: int = 0
    spec_sampling: bool = False

    # Total tokens of a packed text-prefill step, one captured graph per
    # (bucket, batch size). A prompt past the top bucket prefills eagerly, and
    # an eager 8k prefill is launch-bound (~1,100 kernel launches, the GPU
    # idle half the time: 78 ms of wall for 36 ms of kernels on 0.8B), so the
    # ladder runs to 8192. MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS narrows or
    # extends it (static buffers are sized by the largest bucket).
    PREFILL_TOKEN_BUCKETS = [
        int(x) for x in os.environ.get(
            "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS", "32,64,128,256,512,1024,2048,4096,8192",
        ).split(",")
    ]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]
    # A mixed step carries the decode batch beside its prompt rows, so its
    # captures run to the decode batch: one capped below it splits the batch.
    MIXED_CAPTURE_BATCH_SIZES = [4, 16, 64, 128]
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
        num_draft_tokens: int = 0,
        spec_sampling: bool = False,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.vision_config = vision_config
        self._vision_sentinels: tuple[torch.Tensor, torch.Tensor] | None = None
        # Speculative decoding: a decode step verifies this many drafts of the
        # MTP head (`model.mtp`) and drafts the next ones, each row k + 1
        # tokens; the drafts and lengths live in the SPEC_DECODE resource.
        self.num_draft_tokens = num_draft_tokens
        # verify steps may sample (a second decode capture, keyed by
        # `cg_key_info`); off, they take the target's argmax
        self.spec_sampling = spec_sampling and num_draft_tokens > 0
        if num_draft_tokens and model.mtp is None:
            raise ValueError("a speculative LLM submodule needs the model's MTP head")

    @property
    def _speculative(self) -> bool:
        return self.num_draft_tokens > 0

    @property
    def _decode_tokens(self) -> int:
        """Tokens of one decode row: 1, or a verify row's k + 1."""
        return self.num_draft_tokens + 1

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

        mixed = [] if self._speculative else [
            PackedAcceleratorGraphConfig(
                capture_graph_walk=LLM_MIXED,
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                make_node_input=dummy,
                capture_batch_sizes=self.MIXED_CAPTURE_BATCH_SIZES,
                # the layout its GDN splits; a replay's padding rows stay empty
                token_layout=decode_rows_then_prompt,
            ),
        ]
        # a speculative node that samples captures decode twice: greedy and sampled
        decode_keys = (False, True) if self.spec_sampling else (None,)
        configs = [
            *(BatchedAcceleratorGraphConfig(
                capture_graph_walk="decode",
                # a verify row under speculation
                single_request_inputs=dummy(self._decode_tokens),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
                additional_key_info=key,
            ) for key in decode_keys),
            PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill_text",
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                make_node_input=dummy,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
            *mixed,
            PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill_vision",
                capture_token_lengths=self.PREFILL_VISION_TOKEN_BUCKETS,
                make_node_input=vision_dummy,
                capture_batch_sizes=self.PREFILL_VISION_CAPTURE_BATCH_SIZES,
            ),
        ]
        # a plain decode also captures once more for batches that filter nothing;
        # a verify step doesn't draw through the sampler
        return configs if self._speculative else with_unfiltered_sampling(configs, walks=("decode",))

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
        grids = inputs.get("image_grid_thw", [])
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
                pos_ids.append(text_position_ids(span, pos, device))
                tracked.append(ids)
                pos += span
                continue
            grid = grids[grid_i]
            grid = grid[0] if grid.dim() == 2 else grid
            t, h, w = (int(v) for v in grid.tolist())
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
            pos_ids += [
                text_position_ids(1, pos, device),
                vision_position_ids(grid, merge, pos + 1, device),
                text_position_ids(1, pos + 1 + advance, device),
            ]
            cursor += span
            # two sentinels either side, and the image between them
            pos += advance + 2

        input_embeds = torch.cat(embeds, dim=0)
        return ARNodeInputs(
            input_seq_len=input_embeds.shape[0],
            input_embeds=input_embeds,
            custom_pos_ids=torch.cat(pos_ids, dim=1),
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

        if graph_walk == "decode" and self._speculative:
            # the verify row's tokens are the speculative resource's, on the
            # device; the routed signal is the previous step's output, unread
            return ARNodeInputs(input_seq_len=self._decode_tokens)
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
        if graph_walk == "decode" and self._speculative:
            return ARNodeInputs(input_seq_len=self._decode_tokens)
        if graph_walk == "decode" and has_loopback and self._device_loopback():
            return ARNodeInputs(input_seq_len=1)
        return None

    def _device_loopback(self) -> bool:
        return (
            device_loopback_enabled()
            and self.node_resources[SAMPLER].has_slot_masters
        )

    def device_loopback_signals(self, graph_walk: str) -> frozenset[str]:
        # Not under speculation: a step emits a varying number of tokens, which
        # only the per-request tensor path carries to the client.
        if graph_walk == "decode" and self._device_loopback() and not self._speculative:
            return frozenset({"text_inputs"})
        return frozenset()

    @staticmethod
    def _captured_decode(engine_inputs: ModelInputsFromEngine, graph_walk: str) -> bool:
        """A decode step replayed on a leased slot, or being captured for one.
        The caller's walk decides (a capture's own context carries the
        piecewise walk), the lease says the slot's static buffers are in
        play."""
        if graph_walk != "decode":
            return False
        step = getattr(engine_inputs, "step", None)
        ctx = getattr(step, "ctx", None) if step is not None else None
        return ctx is not None and getattr(ctx, "slot_lease", None) is not None

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

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        out: dict[str, torch.Tensor | Any] = {}
        if graph_walk == "decode" and self._speculative:
            # everything a verify step reads is on the device already
            return out
        captured = self._captured_decode(engine_inputs, graph_walk)
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        if captured and ingraph_decode_rope() and sampler.loopback_in_graph:
            # The captured forward reads each row's last token off the slot
            # masters itself (`_forward`), so nothing is staged. A row that
            # brought its token has it written there first; a capture's rows
            # bring dummies the graph never reads.
            routed = [
                i for i, inp in enumerate(inputs)
                if inp.input_ids is not None and inp.input_seq_len
            ]
            if routed and not engine_inputs.step.ctx.capture:
                sampler.write_loopback_tokens(
                    routed, torch.cat([inputs[i].input_ids for i in routed]),
                )
            return out
        first = inputs[0]
        if first.input_embeds is None and (
            first.input_ids is None or any(inp.input_ids is None for inp in inputs)
        ):
            # device loop-back: every row of a decode step is its request's
            # last sampled token, read by slot (padding rows read slot 0)
            t0 = perf_counter() if PHASE_PERIOD else 0.0
            ids = sampler.loopback_tokens()
            # rows whose last step ran elsewhere brought their token
            routed = [
                i for i, inp in enumerate(inputs)
                if inp.input_ids is not None and inp.input_seq_len
            ]
            if graph_walk != "decode":
                # a mixed step packs its decode rows' single tokens beside the
                # prompts: each row takes its own ids, or its loop-back token
                ids = torch.cat([
                    inp.input_ids.to(ids.dtype) if inp.input_ids is not None
                    else ids[i:i + 1]
                    for i, inp in enumerate(inputs)
                ])
            elif routed:
                sampler.overwrite_rows(routed, (ids, torch.cat([inputs[i].input_ids for i in routed])))
            out["input_ids"] = ids
            if PHASE_PERIOD:
                phase_record("engine.preprocess.loopback", perf_counter() - t0)
            if captured and ingraph_decode_rope():
                # the captured forward builds the rotary tables itself
                return out
        elif inputs[0].input_ids is not None:
            out["input_ids"] = torch.cat([inp.input_ids for inp in inputs], dim=0)
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
        if graph_walk == "decode" and self._speculative:
            lease = kwargs.get("slot_lease")
            key = (
                lease.bucket.cg_key_info if lease is not None
                else self.cg_key_info(graph_walk, per_request_info or {})
            )
            return self._declare_verify(request_ids, key)
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

        steps = {
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
        }
        if self._speculative:
            self._declare_seed(steps, request_ids, per_request_info or {})
        # the leased bucket's key, which `cg_key_info` chose (or the capture's)
        lease = kwargs.get("slot_lease")
        return SubmoduleStep(
            cg_key_info=lease.bucket.cg_key_info if lease is not None else None,
            segments=[
                Segment(
                    request_id=rid,
                    label="main",
                    span=inp.input_seq_len,
                )
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps=steps,
        )

    def cg_key_info(
        self, graph_walk: str,
        per_request_info: Mapping[int, CurrentForwardPassInfo],
        per_request_input_metadata=None,
        **kwargs,
    ):
        """Which capture a decode batch takes. Plain decode: the unfiltered
        draw's, when no row filters. Under speculation with sampling, whether
        the batch samples: any row at a temperature above 0 takes the sampled
        capture (its greedy rows still take the argmax there)."""
        del per_request_input_metadata, kwargs
        if graph_walk != "decode":
            return None
        if not self._speculative:
            return self.node_resources[SAMPLER].unfiltered_key(graph_walk, per_request_info.keys())
        if not self.spec_sampling:
            return None
        return any(
            (cfg := info.resource_configs.get(SAMPLER)) is not None and cfg.temperature > 0
            for info in per_request_info.values()
        )

    def _declare_verify(self, request_ids: list[int], key=None) -> SubmoduleStep:
        """A speculative decode step: k + 1 tokens a row, positions and
        lengths settled on the device. Each row gives back the KV its earlier
        steps over-counted, as far as the host has learned."""
        spec = self.node_resources[SPEC_DECODE]
        span = self._decode_tokens
        retract = {}
        for rid in request_ids:
            given = spec.pending_retract(rid)
            if given:
                retract[rid] = given
        return SubmoduleStep(
            segments=[Segment(request_id=rid, label="main", span=span) for rid in request_ids],
            cg_key_info=key,
            steps={
                KV_CACHE: KVStep(commit_adjust={
                    (rid, "main"): -given for rid, given in retract.items()
                }),
                ATTN: AttentionStep(causal=True, row_queries=span),
                GDN_STATE: RecurrentStep(),
                LINEAR_ATTN: LinearAttnStep(),
                SPEC_DECODE: SpecDecodeStep(phase=SpecPhase.VERIFY, retract=retract),
            },
        )

    def _declare_seed(
        self, steps: dict, request_ids: list[int],
        info: Mapping[int, CurrentForwardPassInfo],
    ) -> None:
        """A prompt step's share of speculation: seed each row's registers,
        and on the last prefill reserve the KV a verify step writes past the
        committed length (`SpecDecodeResource.seed_lookahead`)."""
        spec = self.node_resources[SPEC_DECODE]
        budgets, ignore, adjust, sampling = {}, set(), {}, {}
        for rid in request_ids:
            row = info.get(rid)
            if row is None:
                continue
            # decode runs `max_tokens` steps after the prompt's token (`check_stop`),
            # so a request emits that many past it
            budgets[rid] = row.max_tokens
            # a capture or padding row carries no sampling config
            cfg = row.resource_configs.get(SAMPLER)
            if cfg is not None and cfg.ignore_eos:
                ignore.add(rid)
            if cfg is not None and self.spec_sampling and cfg.temperature > 0:
                sampling[rid] = (
                    float(cfg.temperature), float(cfg.top_p), int(cfg.top_k or 0), int(cfg.seed),
                )
            if row.step_metadata.get("last_prefill", False) and spec.seed_lookahead:
                adjust[(rid, "main")] = spec.seed_lookahead
        steps[KV_CACHE] = KVStep(commit_adjust=adjust)
        steps[SPEC_DECODE] = SpecDecodeStep(
            phase=SpecPhase.SEED, budgets=budgets, ignore_stop=frozenset(ignore),
            sampling=sampling,
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
        if graph_walk == "decode" and self._speculative:
            return self._verify(engine_inputs)
        attn: AttentionManager = engine_inputs.resources[ATTN]
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        captured = self._captured_decode(engine_inputs, graph_walk)
        if input_ids is None and input_embeds is None:
            input_ids = sampler.loopback_tokens()
        if (captured and ingraph_decode_rope()) or cos_3d is None or sin_3d is None:
            # MSTAR_INGRAPH_DECODE_ROPE: the tables come from the planned
            # positions, inside the graph; the static cos/sin stay unread
            num_tokens = (input_ids if input_ids is not None else input_embeds).shape[0]
            cos_3d, sin_3d = self._decode_cos_sin(engine_inputs, num_tokens)

        embeds = (
            self.model.model.embed_tokens(input_ids)
            if input_embeds is None
            else input_embeds
        )
        hidden = self.model.model(embeds, label="main", cos_sin=(cos_3d, sin_3d))

        if graph_walk == "decode":
            return sampler.sample(self.model.lm_head(hidden))
        # only the last token of each prefill row feeds the head
        token = sampler.sample(self.model.lm_head(attn.select_last_hidden(hidden)))
        if self._speculative:
            self._seed_drafts(engine_inputs, hidden, embeds, token, (cos_3d, sin_3d))
        return token

    def _cos_sin_1d(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotary tables for text positions (the three MRoPE grids agree)."""
        lm = self.model.model
        return lm.build_cos_sin(
            positions.unsqueeze(0).expand(3, -1), dtype=lm.embed_tokens.weight.dtype,
        )

    def _seed_drafts(
        self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor,
        embeds: torch.Tensor, token: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Run the MTP head over the prompt, which fills its KV for every
        prompt position, and seed decode with the first token and the head's
        draft after it. Position p reads the next token's embedding: the
        prompt's own, and at each row's last the token just sampled."""
        attn: AttentionManager = engine_inputs.resources[ATTN]
        embed = self.model.model.embed_tokens
        last = (attn.qo_indptr_buf("main")[1:] - 1).long()
        following = embeds.roll(-1, 0).index_copy(0, last, embed(token).to(embeds.dtype))
        drafted = self.model.mtp(hidden, following, label="main", cos_sin=cos_sin)
        draft = self.model.lm_head(drafted.index_select(0, last)).argmax(dim=-1)
        engine_inputs.resources[SPEC_DECODE].seed(token, draft)

    def _verify(self, engine_inputs: ModelInputsFromEngine):
        """A speculative decode step: verify each row's k drafts, then draft
        the next k with the MTP head, every pass over the same (rows, k + 1)
        tokens (see `SpecDecodeResource`)."""
        spec = engine_inputs.resources[SPEC_DECODE]
        lm, head = self.model.model, self.model.lm_head
        k = self.num_draft_tokens
        ids, positions = spec.verify_inputs()
        # draft pass j sits at the verify positions plus j: one table for all
        shifts = torch.arange(k, dtype=positions.dtype, device=positions.device)
        cos, sin = self._cos_sin_1d((positions[None, :] + shifts[:, None]).reshape(-1))
        cos, sin = cos.view(k, positions.shape[0], -1), sin.view(k, positions.shape[0], -1)
        hidden = lm(lm.embed_tokens(ids), label="main", cos_sin=(cos[0], sin[0]))
        step = engine_inputs.step
        sample = bool(step is not None and step.cg_key_info)
        accepted = spec.accept(head(hidden), sample=sample)
        ids = accepted.target.reshape(-1)
        rows = positions.shape[0] // (k + 1)
        for j in range(k):
            spec.begin_draft_pass(j)
            # only each row's last accepted slot is a real query: the head's
            # work after attention, and the lm_head, run on that row alone, and
            # the next pass reads its output in every slot
            out = self.model.mtp(
                hidden, lm.embed_tokens(ids), label="main", cos_sin=(cos[j], sin[j]),
                keep=spec.draft_slots(),
            )
            logits = head(out)
            ids = spec.record_draft(j, spec.draw_draft(j, logits) if sample else logits.argmax(dim=-1))
            hidden = out[:, None, :].expand(rows, k + 1, out.shape[-1]).reshape(rows * (k + 1), -1)
        return accepted

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
        out = self._forward(
            graph_walk, engine_inputs, cos_3d, sin_3d,
            input_ids=input_ids, input_embeds=input_embeds,
        )
        if graph_walk == "decode" and self._speculative:
            out = out.tokens
        return {"new_token": out}

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
        if graph_walk == "decode" and self._speculative:
            # a row's tokens, -1 past what it emits; the stop check reads the
            # accepted count (for the KV the host gives back) and `done`
            return BatchedModelOutput(
                row_outputs={"new_token": new_tokens.tokens},
                check_stop_buffers={
                    "new_token": new_tokens.tokens,
                    "accepted": new_tokens.accepted,
                    "done": new_tokens.done,
                },
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

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        # only a decode row's stop can end decode_loop; a prefill is not in it yet
        if request_info.graph_walk != "decode" or "new_token" not in outputs:
            return set()
        if self.num_draft_tokens:
            # The batched check reads the step's own `done`; this fallback
            # sees only the tokens, so it stops on a stop id and leaves the
            # budget to the loop's cap.
            tokens = outputs["new_token"][0].reshape(-1).tolist()
            if request_info.resource_configs[SAMPLER].ignore_eos:
                return set()
            stop_ids = self.config.stop_token_ids
            return {"decode_loop"} if any(t in stop_ids for t in tokens) else set()
        token = outputs["new_token"][0].item()
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        hit_eos = not ignore_eos and token in self.config.stop_token_ids
        out_of_budget = (
            request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
            >= request_info.max_tokens
        )
        return {"decode_loop"} if hit_eos or out_of_budget else set()

    def step_is_input_free(self, graph_walk: str) -> bool:
        # decode rows are one token each, like the template; the prefill walks
        # feed `prefill_tokens` from the inputs
        return graph_walk == "decode"

    def inline_client_signals(self, graph_walk: str) -> dict[str, str]:
        # the decode loop's client edge carries the sampled token the stop
        # check already has on the host; prefill's new_token edge is persisted.
        # Inline emission carries one token a row, so not under speculation.
        if graph_walk == "decode" and not self._speculative:
            return {"text_inputs": "new_token"}
        return {}

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
        if self.num_draft_tokens:
            return self._check_stop_verify(request_ids, request_infos, host_rows)
        pairs = host_rows.first_values("new_token", request_ids)
        if pairs is None:
            return None
        stop_ids = self.config.stop_token_ids
        stops: dict[str, set[str]] = {}
        mixed = step_walk is None
        for rid, value in pairs:
            info = request_infos[rid]
            # a mixed step reads each row's walk; a decode step need not
            if mixed and info.graph_walk != "decode":
                continue
            # the set test first: a stop id is rare, the config read behind it
            # goes through the wrapper
            if (
                value in stop_ids
                and not info.resource_configs[SAMPLER].ignore_eos
            ) or (
                info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
                >= info.max_tokens
            ):
                stops[rid] = {"decode_loop"}
        return stops


    def _check_stop_verify(
        self,
        request_ids: list[int],
        request_infos: dict[int, CurrentForwardPassInfo],
        host_rows: HostRows,
    ) -> dict[int, set[str]] | None:
        """A verify step's stops, off the flags it computed itself; and what
        each row accepted, which tells the speculative resource how much of
        the KV the step reserved it can give back."""
        accepted = host_rows.buffers.get("accepted")
        done = host_rows.buffers.get("done")
        if not (torch.is_tensor(accepted) and torch.is_tensor(done)):
            return None
        accepted, done = accepted.reshape(-1).tolist(), done.reshape(-1).tolist()
        row_of = {rid: i for i, rid in enumerate(host_rows.request_ids)}
        learned, counts = [], []
        stops: dict[int, set[str]] = {}
        for rid in request_ids:
            i = row_of.get(rid)
            if i is None or i >= len(accepted) or request_infos[rid].graph_walk != "decode":
                continue
            learned.append(rid)
            counts.append(accepted[i])
            if done[i]:
                stops[rid] = {"decode_loop"}
        self.node_resources[SPEC_DECODE].learn(learned, counts)
        return stops


# The vision tower's block loop, as a piecewise capture region
VISION_BLOCK_LOOP = "vision_block_loop"


class VisionEncoderSubmodule(NodeSubmodule):
    """Qwen3.5's ViT, which turns pixel patches into LLM-width embeddings.

    A prompt's images go through packed, in one call. Grid maths runs in
    `prepare_inputs` and attention boundaries are declared as step segments,
    so the compiled tower's only symbolic shape is its token count.

    The block loop is captured one prompt per replay, which beats batching:
    eager, the tower is launch-bound (~500 kernels, ~4 ms of GPU work in a
    ~20 ms forward on 9B). So the encoder takes one request a step.

    A replay's attention plan holds at most the vision attention resource's
    ``max_segments_per_request`` segments (one per image and per video frame
    group). A prompt with more takes no lease and runs the block loop
    eagerly, however few its patches.
    """

    # Patch-count buckets, one prompt per replay. 16384 patches merge to the
    # LLM's 4096 `prefill_vision` bucket; a larger prompt runs eagerly.
    BLOCK_LOOP_PATCH_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384]

    def __init__(
        self, model: Qwen3_5VisionModel, config: Qwen3_5VisionConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        # (num_images, 3): a bare [t, h, w] would read as three images
        device = self.get_device()
        merge = self.config.spatial_merge_size
        grid = [
            (int(t), int(h), int(w))
            for g in inputs["image_grid_thw"]
            for t, h, w in g.reshape(-1, 3).tolist()
        ]
        # Built here, not in the compiled forward: a `.tolist()` there breaks
        # the graph and makes h, w symints inductor cannot codegen from.
        indices, weights = vision_interpolation(
            grid, self.config.num_grid_per_side, merge, device,
        )
        num_patches = sum(t * h * w for t, h, w in grid)
        return NodeInputs(
            # what the engine leases the captured block loop's bucket by
            input_seq_len=num_patches,
            tensor_inputs={
                "pixel_values": torch.cat(inputs["pixel_values"], dim=0),
                "indices": indices,
                "weights": weights,
                "position_ids": vision_grid_position_ids(grid, merge, device),
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
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        # one request a step, so it can take the captured block loop
        return 1

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype,
        tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """The block loop, one prompt per replay, by patch-count bucket.

        Patch embed, position resample and rope run eagerly before it, the
        merger after (it quarters the row count, which a region's output view
        cannot express). Batch size 1: one replay is one prompt's segments.
        """
        hidden, head_dim = self.config.hidden_size, self.config.head_dim

        def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            n = shape.total_tokens
            return {
                "hidden": torch.zeros(n, hidden, dtype=autocast_dtype, device=device),
                "cos": torch.zeros(n, head_dim, dtype=autocast_dtype, device=device),
                "sin": torch.zeros(n, head_dim, dtype=autocast_dtype, device=device),
            }

        def declare_step(request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
            (rid,) = request_ids
            return SubmoduleStep(
                segments=[Segment(request_id=rid, label="main", span=n) for n in seq_lens],
                steps={VISION_ATTN: AttentionStep(causal=False)},
            )

        return {
            VISION_BLOCK_LOOP: PiecewisePackedConfig(
                capture_fn=self._capture_block_loop,
                make_static_inputs=make_static_inputs,
                declare_step=declare_step,
                lease_before_step=True,
                can_lease=self._fits_capture,
                total_tokens=self.BLOCK_LOOP_PATCH_BUCKETS,
                capture_batch_sizes=[1],
            )
        }

    def _fits_capture(self, inputs: list[NodeInputs]) -> bool:
        """Whether the batch's segments fit a replay's attention plan: the
        resource's per-request ceiling, read from it so a YAML override holds.

        TODO: capture more than 64 segments, so long low-res videos replay
        too. The ceiling is sized for images (16384 patches / 256-patch
        floor), and raising it pads every replay's plan to it. Sizing each
        bucket's plan by its own token count over the smallest segment (one
        2x2 merge block) would hold any prompt that fits, at small buckets'
        cost unchanged; that is a ragged-resource change.
        """
        per_request = self.node_resources[VISION_ATTN].max_segments_per_request
        segments = sum(len(inp.tensor_inputs["seq_lengths"]) for inp in inputs)
        return segments <= per_request * len(inputs)

    def _capture_block_loop(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        """The captured region: reads the runner's buffers, never rebinds."""
        s = inp.static_inputs
        return {"hidden": self.model.encode(s["hidden"], s["cos"], s["sin"])}

    @torch.compiler.disable
    def _replay_block_loop(
        self,
        engine_inputs: ModelInputsFromEngine,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seq_lengths: tuple[int, ...],
    ) -> torch.Tensor | None:
        """The block loop replayed once per prompt, or None if not leased.

        A leased step declares nothing of its own (`declare_step`), so
        ``engine_inputs.step`` is None; a bucket that fits is not enough, as
        ``can_lease`` may have refused the batch. The bucket check keeps a
        forward driven outside the engine (no declaration either) eager.
        """
        runner = engine_inputs.piecewise_runners.get(VISION_BLOCK_LOOP)
        if runner is None or engine_inputs.step is not None or not runner.can_run(
            len(engine_inputs.request_ids), int(hidden.shape[0]),
        ):
            return None
        # one replay for every image, not one each: a second plan overwrites the schedule a queued replay still reads
        return runner.run(
            static_inputs={"hidden": hidden, "cos": cos, "sin": sin},
            request_ids=[engine_inputs.request_ids[0]],
            seq_lens=list(seq_lengths),
            real_bs=1,
        ).get_view("hidden")

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        """The batch's one request (see `max_batch_size`), its images packed
        in `declare_step`'s segment order."""
        (inp,) = inputs
        device = self.get_device()
        tensors = inp.tensor_inputs
        return {
            "pixel_values": tensors["pixel_values"].to(device),
            "indices": tensors["indices"].to(device),
            "weights": tensors["weights"].to(device),
            "position_ids": tensors["position_ids"].to(device),
            "seq_lengths": tuple(tensors["seq_lengths"]),
        }

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
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        # One call for every image; the ragged kernel's per-frame segments
        # keep them apart.
        hidden, cos, sin = self.model.embed(pixel_values, indices, weights, position_ids)
        encoded = self._replay_block_loop(engine_inputs, hidden, cos, sin, seq_lengths)
        if encoded is None:
            encoded = self.model.encode(hidden, cos, sin)
        embeds = self.model.merger(encoded)
        return {engine_inputs.request_ids[0]: {"vision_embeds": [embeds]}}
