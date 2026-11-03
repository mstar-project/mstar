"""Node submodules for OmniVoice: reference encode, unmask loop, code2wav.

The unmask node is the interesting one.  Unlike wan22's DiT — the only other
diffusion node in the tree, which caps its batch at one because each request's
latent grid has its own geometry — OmniVoice's canvases differ only in length,
so several requests genuinely share a step.  That is the whole reason to serve
this model on M*: the reference's own server runs one request at a time.

A step packs every request's two CFG documents end to end into one ragged row
and runs the reference's own fused-kernel forward over it, so the per-step
kernel path is the fastest one upstream ships -- and the batch spans concurrent
requests, which theirs cannot.
"""

import logging
from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    PiecewiseAcceleratorGraphConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewisePackedConfig,
)
from mstar.engine.resources import AttentionStep, Segment, SlotLease, SubmoduleStep
from mstar.model.omnivoice.components.backbone import (
    CanvasItem,
    OmniVoiceBackbone,
    PackedCanvas,
    build_packed_canvas,
)
from mstar.model.omnivoice.components.codec import (
    decode_canvas,
    encode_reference,
)
from mstar.model.omnivoice.components.unmask import build_reveal_schedule, unmask_step
from mstar.model.omnivoice.config import OmniVoiceConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)

UNMASK_LOOP_NAME = "unmask_loop"
# Step metadata: the canvas length, in frames
TARGET_LEN = "target_len"
# The backbone's ragged attention resource, and its captured layer stack.
BACKBONE_ATTN = "backbone_attn"
BACKBONE_LAYERS = "backbone_layers"


def document_pairs(lengths: list[tuple[int, int]]) -> list[int]:
    """Two documents per request, conditional then unconditional, in packing
    order; a request with CFG off has a zero-length unconditional one, so every
    request owns two segments whatever its guidance."""
    return [n for pair in lengths for n in pair]


def _loop_step(fwd_info: CurrentForwardPassInfo, step_index: torch.Tensor) -> int:
    """The unmask iteration a step runs, from the engine's loop counter.

    ``step_index`` holds the same value, but it is the previous step's device
    output: reading it on the host waits for that step to finish, which stalls
    the speculative preparation of the next one. The counter is per batch and
    counts speculative iterations too (#351). Callers that drive the
    submodule without a loop counter (tests, harnesses) fall back to the edge.
    """
    counts = getattr(fwd_info, "dynamic_loop_iter_counts", None) or {}
    if UNMASK_LOOP_NAME in counts:
        return int(counts[UNMASK_LOOP_NAME])
    return int(step_index.reshape(-1)[0].item())


class _SingleRequestMixin:
    """Serve one request per step through the engine's batched entry point.

    The engine always dispatches to ``forward_batched``; a submodule that only
    defines ``forward`` never runs.  The codec nodes do not batch — each
    request's waveform has its own length and the codec is small enough that
    padding would cost more than it saves — so they cap at one and hand the row
    to ``forward``, rather than claiming a batching they do not implement.
    """

    def max_batch_size(self, graph_walk: str):
        return 1

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        request_ids = engine_inputs.request_ids
        assert len(request_ids) == 1, (
            f"{type(self).__name__} does not batch; got {len(request_ids)} "
            "requests in one step (max_batch_size should have capped it at 1)"
        )
        return {
            request_ids[0]: self.forward(
                graph_walk, engine_inputs=engine_inputs, **kwargs
            )
        }


# ---------------------------------------------------------------------------
# ref_encoder
# ---------------------------------------------------------------------------


class OmniVoiceRefEncoderSubmodule(_SingleRequestMixin, NodeSubmodule):
    """Higgs-Audio-v2 encode of the cloning reference.

    Consumes ``ref_audio_inputs`` — one mono waveform already resampled to the
    codec's rate by the request seam — and emits ``ref_audio_tokens`` ``[C, T]``.
    The edge is marked ``persist`` so the tokens survive from the
    ``encode_reference`` walk into the generation walk of the *same* request;
    the encode runs once per request, not once per voice. Caching a voice
    across requests, the reference's ``VoiceClonePrompt``, would need a store
    the engine does not have today.
    """

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, codec: nn.Module, config: OmniVoiceConfig):
        super().__init__()
        self.codec = codec
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        waveform = inputs["ref_audio_inputs"][0]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return NodeInputs(tensor_inputs={"waveform": waveform})

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        waveform: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        return {"ref_audio_tokens": [encode_reference(self.codec, waveform)]}


# ---------------------------------------------------------------------------
# backbone (the unmask loop body)
# ---------------------------------------------------------------------------


class OmniVoiceBackboneSubmodule(NodeSubmodule):
    """One unmask iteration for every request in the batch.

    Loop-carried edges, per request:

        audio_tokens  [C, T] int64, the live canvas — all MASK at iteration 0
        step_index    [1] int64, the 0-based iteration counter

    The prefix (``prefix_ids`` / ``prefix_audio_mask``) is fixed for the whole
    loop and arrives persisted from the request seam rather than riding the
    loop, since rebuilding it per step would re-tokenize the text 32 times.

    The reveal schedule is recomputed each step from ``target_len``,
    ``num_step`` and ``t_shift`` rather than shipped: it is a few dozen floats
    and deriving it is cheaper than moving it, the same call wan22 makes with
    its sigma tables.
    """

    disable_torch_compile = True
    # The fused flashinfer kernels this path runs on (RMSNorm, silu_and_mul,
    # ragged attention) require every tensor in a call to share one dtype --
    # unlike ATen, which would quietly promote. Under the engine's autocast the
    # activations arrive at a different dtype from the weights and the fused
    # RMSNorm rejects the call outright:
    #   Mismatched Tensor on argument #1 ... expected dtype=float32
    # So numerics here are governed by the load dtype, not by the engine.
    disable_autocast = True

    def __init__(self, backbone: OmniVoiceBackbone, config: OmniVoiceConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config

    # -- inputs -----------------------------------------------------------

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs | None:
        device = self.get_device()
        meta = fwd_info.step_metadata

        prefix_ids = inputs["prefix_ids"][0].to(device)
        prefix_audio_mask = inputs["prefix_audio_mask"][0].to(device)

        # Cloning walk only: the codec's reference tokens close out the prefix.
        # They are joined here rather than on the data worker because the encode
        # runs on a compute node, and joining at the seam would mean shipping
        # the prefix back and forth.
        ref_tokens = inputs.get("ref_audio_tokens")
        if ref_tokens:
            ref = ref_tokens[0].to(device=device, dtype=prefix_ids.dtype)
            prefix_ids = torch.cat([prefix_ids, ref], dim=-1)
            prefix_audio_mask = torch.cat(
                [
                    prefix_audio_mask,
                    torch.ones(ref.shape[-1], dtype=torch.bool, device=device),
                ],
                dim=-1,
            )

        if "audio_tokens" not in inputs or len(inputs["audio_tokens"]) == 0:
            # Iteration 0: the loop-back edges arrive empty, and the loop starts
            # from an all-MASK canvas. There is no noise and no seed to honour:
            # the stochastic part of this model is the reveal order, not the
            # starting state.
            target_len = int(meta[TARGET_LEN])
            tokens = torch.full(
                (self.config.num_audio_codebook, target_len),
                self.config.audio_mask_id,
                dtype=torch.long,
                device=device,
            )
            # 1-D, not 0-dim: the worker's output fanout reads dims[0] of
            # every routed edge.
            step_index = torch.zeros(1, dtype=torch.int64, device=device)
            k = 0
        else:
            step_index = inputs["step_index"][0]
            k = _loop_step(fwd_info, step_index)
            num_step = int(meta["num_step"])
            if k >= num_step:
                # Async scheduling dispatched an iteration past this request's
                # stop. Veto it — None makes the engine skip the forward.
                logger.info(
                    "OmniVoice backbone: skipping async-overshoot iteration %d "
                    "(request %s runs %d steps)",
                    k, fwd_info.request_id, num_step,
                )
                return None
            tokens = inputs["audio_tokens"][0].to(device)
            target_len = tokens.shape[-1]

        # the row's place in the step's packed canvas, built once here
        item = CanvasItem(
            prefix_ids=prefix_ids,
            prefix_audio_mask=prefix_audio_mask,
            tokens=tokens.unsqueeze(0),
            guidance_scale=float(meta["guidance_scale"]),
        )
        return NodeInputs(
            tensor_inputs={
                "prefix_ids": prefix_ids,
                "prefix_audio_mask": prefix_audio_mask,
                "audio_tokens": tokens,
                "step_index": step_index,
            },
            # the iteration, for forward_batched: at iteration 0 the loop counter
            # is not there yet, and reading the step edge would sync
            kwargs={"item": item, "step": k},
            # The row's packed contribution: its conditional document plus,
            # with CFG on, the target again as the unconditional one.
            input_seq_len=sum(item.doc_pair),
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep | None:
        """The eager path's attention layout; a leased replay plans its own."""
        if (piecewise_leases or {}).get(BACKBONE_LAYERS):
            return None
        return self._attention_step(
            request_ids, document_pairs([inp.kwargs["item"].doc_pair for inp in inputs])
        )

    @staticmethod
    def _attention_step(request_ids: list[str], doc_lens: list[int]) -> SubmoduleStep:
        """Each document attends within itself. ``doc_lens`` holds two entries
        per real request; a padded replay appends one zero per padding row, and
        a capture one length per row, so documents are spread over the ids in
        order -- the ragged plan reads only the spans."""
        per_id = max(1, -(-len(doc_lens) // len(request_ids)))
        return SubmoduleStep(
            segments=[
                Segment(request_id=request_ids[i // per_id], label="main", span=n)
                for i, n in enumerate(doc_lens)
            ],
            steps={BACKBONE_ATTN: AttentionStep(causal=False)},
        )

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs: Any
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """The layer stack (embeddings through the final norm), by request
        count and packed-token bucket. The packed canvas is built eagerly before
        it and the audio head runs after it, on the gathered target rows."""
        del autocast_dtype, tp_world_size, kwargs
        if not self.config.backbone_graphs:
            return {}
        return {
            BACKBONE_LAYERS: PiecewisePackedConfig(
                capture_fn=self._capture_layers,
                make_static_inputs=partial(self._layer_static_inputs, device=device),
                declare_step=self._attention_step,
                lease_before_step=True,
                total_tokens=list(self.config.backbone_token_buckets),
                capture_batch_sizes=list(self.config.backbone_capture_batch_sizes),
            )
        }

    def _layer_static_inputs(self, shape: PiecewiseCaptureShape, device) -> dict[str, torch.Tensor]:
        n = shape.total_tokens
        return {
            "input_ids": torch.zeros(n, self.config.num_audio_codebook, dtype=torch.long, device=device),
            "audio_mask": torch.zeros(n, dtype=torch.bool, device=device),
            "pos_ids": torch.zeros(n, dtype=torch.int32, device=device),
        }

    def _capture_layers(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        s = inp.static_inputs
        return {
            "hidden": self.backbone.encode(
                s["input_ids"], s["audio_mask"], s["pos_ids"],
                self.node_resources[BACKBONE_ATTN],
            )
        }

    @torch.inference_mode()
    def _logits(self, engine_inputs: ModelInputsFromEngine, canvas: PackedCanvas) -> torch.Tensor:
        """The step's target logits: from the captured layer stack when a
        bucket fits, else eagerly through the same ragged attention."""
        position_ids = canvas.position_ids[0]
        static_inputs = {
            "input_ids": canvas.packed_ids[0].t(),
            "audio_mask": canvas.audio_mask[0],
            "pos_ids": position_ids.to(torch.int32),
        }
        num_requests = len(canvas.items)
        if BACKBONE_LAYERS in engine_inputs.piecewise_leases:
            hidden = engine_inputs.piecewise_runners[BACKBONE_LAYERS].run(
                static_inputs=static_inputs,
                request_ids=list(engine_inputs.request_ids),
                seq_lens=document_pairs([item.doc_pair for item in canvas.items]),
                real_bs=num_requests,
            ).get_view("hidden")
        else:
            hidden = self.backbone.encode(**static_inputs, attn=engine_inputs.resources[BACKBONE_ATTN])
        return self.backbone.head(hidden, canvas.tgt_index)

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict:
        """Build the step's packed canvas.

        Canvases differ in length per request, so there is no padded collation
        here; the batch is one packed sequence with a document boundary per
        CFG branch, laid out from each row's ``CanvasItem`` (``prepare_inputs``),
        whose guidance scale decides its documents.
        """
        device = self.get_device()
        items = [row.kwargs["item"] for row in inputs]
        return {
            "items": items,
            "canvas": build_packed_canvas(items, device),
            "steps": [row.kwargs["step"] for row in inputs],
            "step_index": torch.cat([row.tensor_inputs["step_index"] for row in inputs])
            if len(inputs) > 1 else inputs[0].tensor_inputs["step_index"],
        }

    def max_batch_size(self, graph_walk: str):
        return self.config.max_batch_size

    def can_batch(self, batch, model_inputs: list[NodeInputs]) -> bool:
        """Cap the step by packed tokens, not by row count.

        Cost per step is ``sum(doc_lens)`` through a bidirectional attention
        plus a head GEMM over ``2 * sum(target_len)``; eight 30-second requests
        and eight 2-second ones are two very different steps. Row count alone
        would let the first case allocate a logits tensor several hundred MB
        wide, and the unmask step scores those logits in float32, a few copies
        at a time.
        """
        packed = sum(inp.input_seq_len for inp in model_inputs)
        if packed > self.config.max_packed_tokens:
            logger.debug(
                "OmniVoice backbone: %d packed tokens over the %d cap; "
                "the engine runs each request as its own forward",
                packed, self.config.max_packed_tokens,
            )
            return False
        return True

    # -- forward ----------------------------------------------------------

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        items: list[CanvasItem] | None = None,
        canvas: PackedCanvas | None = None,
        steps: list[int] | None = None,
        step_index: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """One unmask iteration for the whole batch: the packed backbone
        forward, then every request's scoring and reveal in one batched pass
        (``unmask_step``), so a step launches the same kernels for 1 request
        as for 16.

        Each request keeps its own reveal schedule, knobs and seeded RNG. The
        iteration index is the one ``prepare_inputs`` took from the loop
        counter (see ``_loop_step``); the knobs come from step metadata.
        """
        assert items is not None and canvas is not None and steps is not None and step_index is not None, (
            "OmniVoice backbone requires preprocess output"
        )
        logits = self._logits(engine_inputs, canvas)
        infos = [engine_inputs.per_request_info[rid] for rid in engine_inputs.request_ids]
        metas = [info.step_metadata for info in infos]
        counts = []
        for item, meta, k in zip(items, metas, steps, strict=True):
            schedule = build_reveal_schedule(
                target_len=item.target_len,
                num_codebook=self.config.num_audio_codebook,
                num_step=int(meta["num_step"]),
                t_shift=float(meta["t_shift"]),
            )
            counts.append(schedule[k] if k < len(schedule) else 0)
        tokens = unmask_step(
            logits,
            canvases=[item.tokens[0] for item in items],
            uncond_rows=canvas.uncond_rows(),
            counts=counts,
            generators=[self._step_generator(info, k, logits.device) for info, k in zip(infos, steps, strict=True)],
            guidance_scale=[item.guidance_scale for item in items],
            class_temperature=[float(meta["class_temperature"]) for meta in metas],
            layer_penalty_factor=[float(meta["layer_penalty_factor"]) for meta in metas],
            position_temperature=[float(meta["position_temperature"]) for meta in metas],
            audio_mask_id=self.config.audio_mask_id,
        )
        next_index = step_index + 1
        # keyed by the engine's ids for the rows (the worker's handles), not the wire ids
        return {
            rid: {
                "audio_tokens": [tokens[:, item.flat_start:item.flat_start + item.target_len]],
                "step_index": [next_index[i:i + 1]],
            }
            for i, (rid, item) in enumerate(zip(engine_inputs.request_ids, items, strict=True))
        }

    @staticmethod
    def _step_generator(
        request_info: CurrentForwardPassInfo, k: int, device: torch.device
    ) -> torch.Generator | None:
        """The request's RNG for iteration ``k``.

        Reseeded per step from the request's seed plus the iteration, so a
        replay matches step for step; one generator for the whole request
        would only match if every step ran in the same order, which async
        scheduling does not promise. The conductor derives a seed per request,
        so this is normally always set and the fallback is only for a caller
        that drives the submodule without one.
        """
        seed = getattr(request_info, "random_seed", None)
        if seed is None:
            return None
        generator = torch.Generator(device=device)
        generator.manual_seed((int(seed) + k) & 0x7FFF_FFFF_FFFF_FFFF)
        return generator

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs,
    ) -> NameToTensorList:
        rid = engine_inputs.request_ids[0]
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[rid]

    # -- stop -------------------------------------------------------------

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        """End the loop after exactly this request's ``num_step`` iterations.

        While iteration k is being postprocessed the counter still reads k, so
        for N steps the stop must fire at ``k == N - 1``.  ``>=`` keeps it
        firing if async scheduling has already advanced the deferred count.
        """
        iter_idx = request_info.dynamic_loop_iter_counts.get(UNMASK_LOOP_NAME, 0)
        requested = int(request_info.step_metadata.get("num_step", 0) or 0)
        if requested > 0 and iter_idx + 1 >= requested:
            return {UNMASK_LOOP_NAME}
        return set()


# ---------------------------------------------------------------------------
# code2wav
# ---------------------------------------------------------------------------


class OmniVoiceCode2WavSubmodule(_SingleRequestMixin, NodeSubmodule):
    """Decode the finished canvas to the codec's float waveform.

    Emits ``audio_output``, 1-D float32 at the codec's sample rate, and nothing
    else happens here: the reference's output shaping (silence trim, level,
    fade, pad) and the PCM16 conversion run in ``OmniVoiceModel.postprocess``
    on the API side. Done in this step they synced the host and held up the
    unmask loop of every other request.

    A canvas that still holds a MASK cell means the loop ended early, which
    would decode to noise -- a fault, not a degraded result. The MASK id is
    outside the codebooks, so the decode reads code 0 there (an out-of-range
    index on the GPU is a device-side assert that kills the worker), and the
    step poisons the waveform with NaN (no host sync here) for
    ``postprocess`` to refuse.
    """

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, codec: nn.Module, config: OmniVoiceConfig):
        super().__init__()
        self.codec = codec
        self.config = config

    def compile_regions(self) -> dict[str, list[tuple]]:
        """The codec's acoustic decoder, compiled for dynamic canvas lengths:
        Inductor fuses each Snake activation and residual add between its convs,
        which eagerly are ~300 of a decode's ~430 kernels. Dynamo builds one
        graph per range of canvas lengths, split where the conv backend choice
        depends on size: on CPU, MKLDNN's numel heuristic splits at 3, 6, 21
        and 81 frames, and the warm-up runs one canvas per range. The CUDA
        ranges (conv backend choice and Inductor's own guards) are unmeasured;
        probably one. Canvases of one and two frames (80 ms at most) get
        graphs of their own, so they are warmed too: six graphs, under
        dynamo's recompile limit of 8 if the CUDA ranges add none. The inputs
        are made as the codec's ``decode`` makes them (``fc2``, then a
        transpose): a tensor of other strides or view base
        fails the guards."""
        fc2 = self.codec.fc2
        latents = [
            torch.zeros(1, frames, fc2.in_features, dtype=fc2.weight.dtype, device=fc2.weight.device)
            for frames in (1, 2, 3, 10, 40, 160)
        ]
        return {"codec.acoustic_decoder": [(fc2(x).transpose(1, 2),) for x in latents]}

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        return NodeInputs(tensor_inputs={"audio_tokens": inputs["audio_tokens"][0]})

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_tokens: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        masked = audio_tokens == self.config.audio_mask_id
        waveform = decode_canvas(self.codec, audio_tokens.masked_fill(masked, 0))
        waveform = waveform.masked_fill(masked.any(), float("nan"))
        return {"audio_output": [waveform]}
