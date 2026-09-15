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

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.omnivoice.components.backbone import (
    CanvasItem,
    OmniVoiceBackbone,
    build_packed_canvas,
)
from mstar.model.omnivoice.components.codec import (
    decode_canvas,
    encode_reference,
    post_process,
)
from mstar.model.omnivoice.components.unmask import (
    apply_reveal,
    build_reveal_schedule,
    predict_tokens_with_scoring,
)
from mstar.model.omnivoice.config import OmniVoiceConfig
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)

UNMASK_LOOP_NAME = "unmask_loop"


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
    The edge persists so a caller cloning the same voice repeatedly pays the
    encode once; that is what the reference's ``VoiceClonePrompt`` is for.
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
        # target_len rides an edge rather than step_metadata: computing it needs
        # the tokenizer and the duration estimator, both data-worker assets.
        target_len = int(inputs["target_len"][0].reshape(-1)[0].item())

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
            # Iteration 0: the loop-back edges arrive empty. Seed a fully
            # masked canvas; there is no noise and no seed to honour, because
            # the stochastic part of this model is the reveal order, not the
            # starting state.
            tokens = torch.full(
                (self.config.num_audio_codebook, target_len),
                self.config.audio_mask_id,
                dtype=torch.long,
                device=device,
            )
            # 1-D, not 0-dim: the worker's output fanout reads dims[0] of
            # every routed edge.
            step_index = torch.zeros(1, dtype=torch.int64, device=device)
        else:
            step_index = inputs["step_index"][0]
            k = int(step_index.reshape(-1)[0].item())
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

        return NodeInputs(
            tensor_inputs={
                "prefix_ids": prefix_ids,
                "prefix_audio_mask": prefix_audio_mask,
                "audio_tokens": tokens,
                "step_index": step_index,
            },
            kwargs={"request_id": fwd_info.request_id},
            # Both CFG documents are packed, so the row's real contribution
            # is the conditional length plus the target again.
            input_seq_len=prefix_ids.shape[-1] + 2 * target_len,
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict:
        """Hand the rows through unchanged.

        Nothing is collated here: canvases differ in length, so the padding
        decision belongs with the batch builder, which needs the per-request
        CFG flags to know how many rows to allocate.
        """
        return {"rows": inputs}

    def max_batch_size(self, graph_walk: str):
        return self.config.max_batch_size

    def can_batch(self, batch, model_inputs: list[NodeInputs]) -> bool:
        """Cap the step by packed tokens, not by row count.

        Cost per step is ``sum(doc_lens)`` through a bidirectional attention
        plus a head GEMM over ``2 * sum(target_len)``; eight 30-second requests
        and eight 2-second ones are two very different steps. Row count alone
        would let the first case allocate a float32 logits tensor several
        hundred MB wide.
        """
        packed = sum(inp.input_seq_len for inp in model_inputs)
        if packed > self.config.max_packed_tokens:
            logger.debug(
                "OmniVoice backbone: %d packed tokens over the %d cap; "
                "the scheduler will split this batch",
                packed, self.config.max_packed_tokens,
            )
            return False
        return True

    # -- forward ----------------------------------------------------------

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        rows: list[NodeInputs] | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        assert rows is not None, "OmniVoice backbone requires preprocess output"
        device = self.get_device()

        items: list[CanvasItem] = []
        for rid, row in zip(engine_inputs.request_ids, rows, strict=True):
            meta = engine_inputs.per_request_info[rid].step_metadata
            items.append(
                CanvasItem(
                    request_id=rid,
                    prefix_ids=row.tensor_inputs["prefix_ids"],
                    prefix_audio_mask=row.tensor_inputs["prefix_audio_mask"],
                    # Cloned, not viewed: apply_reveal writes in place, and
                    # the input tensor is the previous iteration's edge.
                    tokens=row.tensor_inputs["audio_tokens"].clone().unsqueeze(0),
                    guidance_scale=float(meta["guidance_scale"]),
                )
            )

        canvas = build_packed_canvas(items, self.config.audio_mask_id, device)
        logits = self.backbone(canvas).to(torch.float32)

        outputs: dict[str, NameToTensorList] = {}
        for item, row in zip(items, rows, strict=True):
            meta = engine_inputs.per_request_info[item.request_id].step_metadata
            step_index = row.tensor_inputs["step_index"]
            k = int(step_index.reshape(-1)[0].item())

            schedule = build_reveal_schedule(
                target_len=item.target_len,
                num_codebook=self.config.num_audio_codebook,
                num_step=int(meta["num_step"]),
                t_shift=float(meta["t_shift"]),
            )

            c_logits, u_logits = canvas.slice_logits(logits, item)
            pred_tokens, scores = predict_tokens_with_scoring(
                c_logits=c_logits,
                u_logits=u_logits,
                audio_mask_id=self.config.audio_mask_id,
                guidance_scale=item.guidance_scale,
                class_temperature=float(meta["class_temperature"]),
            )
            tokens = apply_reveal(
                tokens=item.tokens,
                pred_tokens=pred_tokens,
                scores=scores,
                reveal_count=schedule[k] if k < len(schedule) else 0,
                audio_mask_id=self.config.audio_mask_id,
                layer_penalty_factor=float(meta["layer_penalty_factor"]),
                position_temperature=float(meta["position_temperature"]),
            )

            outputs[item.request_id] = {
                "audio_tokens": [tokens[0]],
                "step_index": [step_index + 1],
            }
        return outputs

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
    """Decode the finished canvas and apply the reference's output shaping.

    Emits ``audio_output``, 1-D **int16 PCM** at the codec's sample rate: the
    speech handler concatenates node output straight into a WAV container, so
    the conversion belongs here, as it does for orpheus and qwen3_tts.

    A canvas that still holds a MASK cell means the loop ended early, which
    would decode to noise — that is a fault, not a degraded result, so it
    raises rather than emitting.
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
        tensor_inputs = {"audio_tokens": inputs["audio_tokens"][0]}
        ref_rms = inputs.get("ref_rms")
        if ref_rms:
            tensor_inputs["ref_rms"] = ref_rms[0]
        return NodeInputs(tensor_inputs=tensor_inputs)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_tokens: torch.Tensor,
        ref_rms: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        info = engine_inputs.single_request_info
        meta = info.step_metadata

        remaining = int((audio_tokens == self.config.audio_mask_id).sum().item())
        if remaining:
            raise RuntimeError(
                f"OmniVoice canvas for request {info.request_id} still holds "
                f"{remaining} masked cells at decode; the unmask loop ended early."
            )

        waveform = decode_canvas(self.codec, audio_tokens)
        waveform = post_process(
            waveform,
            config=self.config,
            # None means "no reference to match" and selects peak
            # normalisation instead; that is a different code path, not a
            # default value, so it must stay None rather than become 0.
            ref_rms=None if ref_rms is None else float(ref_rms.reshape(-1)[0]),
            pad_duration=float(meta.get("pad_duration", 0.1)),
            fade_duration=float(meta.get("fade_duration", 0.1)),
            enabled=bool(meta.get("postprocess_output", True)),
        )
        pcm16 = (
            torch.from_numpy(waveform).float().clamp(-1, 1).mul(32767).to(torch.int16)
        )
        return {"audio_output": [pcm16]}
