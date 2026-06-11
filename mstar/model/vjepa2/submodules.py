"""NodeSubmodule wrappers for the V-JEPA 2 graph nodes.

Three submodules:

  VJepa2EncoderSubmodule   - ViT 3D-patch video encoder.
  VJepa2PredictorSubmodule - masked latent predictor (single forward per call).
  VJepa2ACPredictorSubmodule - action-conditioned predictor (ditto).

All three are stateless (no KV cache, no per-iteration state) and are
dispatched through ``StatelessEngine``.  ``preprocess`` stacks
per-request tensors into a batch (dim 0) — the single-request case
through ``_execute_sequential`` looks like B=1 and goes through ``forward``;
``_execute_batched`` with B>1 goes through ``forward_batched``.

The masked-vs-AC choice is made by ``VJepa2Model.get_submodule`` based on
``config.predictor_kind``.  Both predictor submodules emit the same output
edge name (``predicted_hidden``) so downstream consumers don't need to
branch.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.base import NodeBatch
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule, ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.model.vjepa2.components.ac_predictor import VisionTransformerPredictorAC
from mstar.model.vjepa2.components.predictor import VJEPA2Predictor
from mstar.model.vjepa2.components.vit_encoder import VJEPA2Encoder
from mstar.model.vjepa2.config import VJepa2Config

logger = logging.getLogger(__name__)


def _normalize_video_frames(frames: torch.Tensor) -> torch.Tensor:
    """Bring a video tensor to ``[1, T, C, H, W]``.

    Accepts ``[T, C, H, W]`` or ``[B, T, C, H, W]``.  Values are assumed to
    already be in the range expected by the encoder (i.e., preprocessing
    happened upstream).  This helper only normalizes the batch dim.
    """
    if frames.dim() == 4:
        frames = frames.unsqueeze(0)
    if frames.dim() != 5:
        raise ValueError(f"Expected video frames of shape [T,C,H,W] or [B,T,C,H,W]; got {tuple(frames.shape)}")
    return frames


def _ensure_lead_batch_dim(tensor: torch.Tensor, target_rank: int) -> torch.Tensor:
    """Return ``tensor`` reshaped so its first dim is a batch dim of 1.

    ``target_rank`` is the expected rank of the per-request tensor *including*
    the batch dim.  Handles two inbound cases:
      * Tensor is already ``[1, ...]`` (rank == target_rank) → return as-is.
      * Tensor is ``[...]`` (rank == target_rank - 1) → unsqueeze(0).

    Any other rank is a shape bug and we fail loudly.  Keeping the helper
    per-field explicit (rather than a generic "heuristic on first dim")
    avoids silent batch-dim-collision surprises (e.g. a `[k, N]` multi-mask
    tensor being mistaken for a `[B, N]` input).
    """
    if tensor.dim() == target_rank:
        if tensor.size(0) != 1:
            raise ValueError(
                f"Per-request tensor has leading dim {tensor.size(0)} != 1; "
                f"expected [1, ...] shape with rank {target_rank}, got {tuple(tensor.shape)}."
            )
        return tensor
    if tensor.dim() == target_rank - 1:
        return tensor.unsqueeze(0)
    raise ValueError(
        f"Tensor rank {tensor.dim()} doesn't match target rank "
        f"{target_rank} or {target_rank - 1}; got shape {tuple(tensor.shape)}."
    )


def _shape_key(tensor: torch.Tensor | None) -> tuple | None:
    """Return a shape tuple for dict-key comparison, or ``None`` if absent."""
    return tuple(tensor.shape) if tensor is not None else None


class VJepa2EncoderSubmodule(NodeSubmodule):
    """ViT 3D-patch video encoder.

    Supports cross-request batching: requests with identical
    ``video_frames`` shapes are stacked on dim 0 and run in a single
    encoder forward.  Shape-heterogeneous batches fall through to the
    engine's sequential path (one forward per request).

    Because ``VJepa2Model.process_prompt`` always resizes + center-crops to
    ``(crop_size, crop_size)`` and uniformly samples to ``frames_per_clip``,
    concurrent requests hitting the same serving config are shape-homogeneous
    by construction.  Cross-config deployments (rare) naturally fall back.
    """

    def __init__(self, encoder: VJEPA2Encoder, config: VJepa2Config):
        super().__init__()
        self.encoder = encoder
        self.config = config

    def can_batch(
        self,
        batch: NodeBatch,
        model_inputs: list[NodeInputs],
    ):
        # Route B=1 through the sequential ``forward`` (proven sequential path).
        # ``forward_batched`` is a distinct torch.compile cache from ``forward``
        # and — empirically on vjepa2-ac-vitg — compiles a much slower kernel
        # than ``forward`` when both are traced with the same [1, ...] shape.
        # Observed: AC warm latency went 1 s → 23 s when B=1 traffic routed
        # through forward_batched.  Keeping B=1 on forward sidesteps that
        # entirely; batched-path cost (and value) only shows up when the
        # scheduler actually co-batches concurrent requests.
        if len(batch.request_ids) < 2:
            return False
        shapes: set[tuple] = {
            tuple(inp.tensor_inputs["video_frames"].shape) for inp in model_inputs
        }
        return len(shapes) == 1

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        return NodeInputs(
            tensor_inputs={
                "video_frames": _normalize_video_frames(inputs["video_frames"][0])
            }
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor]:
        # Handles both sequential (len == 1) and batched (len > 1) — the
        # single-request case yields a [1, T, C, H, W] tensor, identical to
        # the original sequential path.
        stacked = torch.cat(
            [inp.tensor_inputs["video_frames"] for inp in inputs],
            dim=0,
        )
        return {"pixel_values_videos": stacked}

    def _encode(self, graph_walk: str, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """Encode a video clip into the ``encoder_hidden`` context.

        Two distinct schemes:

        * **AC rollout** (``predictor_kind == "ac"`` *and* a ``*_rollout*``
          walk): upstream's self-tubelet replication + post-encoder LayerNorm,
          sliced to the first frame's tokens — see :meth:`_encode_self_tubelet`.
          The AC rollout grows context from a single frame
          (``z_hat = z[:, :tokens_per_frame]``).

        * **Everything else** (masked/anticipative; AC single-pass, encoder-only,
          and MPC walks): the natural encoder forward over the full
          ``[B, T, C, H, W]`` clip, yielding ``[B, grid_depth * grid_size**2,
          D]``.  These predictors attend over the full multi-frame context and
          must NOT have it collapsed to a single frame:

          - the masked anticipative rollout would get ``n_ctxt == n_pred`` and
            trip its ``n_pred >= n_ctxt`` guard;
          - the single-pass AC predictor interleaves per-frame context tokens
            with the per-timestep ``actions``/``states`` (T_action = grid_depth),
            so a single-frame context (t=1) mismatches T_action and fails the
            ``torch.cat([a, s, x], dim=2)`` in ``_prepare_sequence``.
        """
        if self.config.predictor_kind == "ac" and "rollout" in graph_walk:
            return self._encode_self_tubelet(pixel_values_videos)
        return self.encoder(pixel_values_videos)

    def _encode_self_tubelet(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """Encode video via upstream's self-tubelet pattern + post-encoder LayerNorm.

        Mirrors ``forward_target`` from
        ``vjepa2/notebooks/energy_landscape_example.ipynb`` Cell 5 and
        ``vjepa2/app/vjepa_droid/train.py:408-415``: each input frame is
        independently encoded as a 2-frame "self-tubelet" (the frame
        duplicated), the per-frame token outputs are flattened, then
        F.layer_norm is applied to match the AC predictor's training
        distribution. Finally we slice to ``tokens_per_frame`` (the first
        frame's tokens) — matching the notebook's
        ``z_hat = z[:, :tokens_per_frame]`` usage as the AC rollout context.

        Input layout: ``[B, T, C, H, W]`` (HF VJEPA2 convention).
        Output layout: ``[B, tokens_per_frame, D]`` — drop-in for the
        downstream rollout predictor.

        For T=8 (the F-8 / DROID training-config alignment) this runs the
        encoder on a [B*8, 2, C, H, W] batched input — same FLOPs as
        upstream's ``forward_target(clips)`` would do at N=8.
        """
        B, T, C, H, W = pixel_values_videos.shape
        # Self-tubelet: each frame becomes a [frame[i], frame[i]] pair.
        # [B, T, C, H, W] -> [B*T, 1, C, H, W] -> [B*T, 2, C, H, W]
        x = pixel_values_videos.reshape(B * T, 1, C, H, W).repeat(1, 2, 1, 1, 1)
        hidden = self.encoder(x)  # [B*T, tokens_per_frame, D]
        D = hidden.size(-1)
        # Reshape and flatten the temporal axis: [B*T, N, D] -> [B, T*N, D].
        hidden = hidden.view(B, T, -1, D).flatten(1, 2)
        # Post-encoder LayerNorm — matches notebook Cell 5 forward_target
        # and WorldModel.encode (world_model_wrapper.py:50-51).
        hidden = F.layer_norm(hidden, (D,))
        # Slice to the first frame's tokens — the rollout uses single-frame
        # context (notebook Cell 5: z_hat = z[:, :tokens_per_frame]).
        tokens_per_frame = hidden.size(1) // T
        return hidden[:, :tokens_per_frame].contiguous()

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values_videos: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        logger.info(
            "VJepa2EncoderSubmodule.forward: input shape=%s dtype=%s device=%s",
            tuple(pixel_values_videos.shape),
            pixel_values_videos.dtype,
            pixel_values_videos.device,
        )
        hidden = self._encode(graph_walk, pixel_values_videos)
        logger.info("VJepa2EncoderSubmodule.forward: output shape=%s", tuple(hidden.shape))
        return {"encoder_hidden": [hidden]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values_videos: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Batched encoder forward.  Returns per-rid outputs directly.

        The returned per-rid ``encoder_hidden`` slices preserve the leading
        batch dim as size 1 (``hidden[i:i+1]``) so downstream submodules
        see ``[1, N, D]`` — identical shape to what the sequential path
        emits.  That keeps the ``preprocess`` stacking symmetric across
        paths and avoids a shape discrepancy at the predictor boundary.
        """
        request_ids = engine_inputs.request_ids
        b_in = pixel_values_videos.size(0)
        logger.info(
            "VJepa2EncoderSubmodule.forward_batched: input shape=%s rids=%d",
            tuple(pixel_values_videos.shape),
            len(request_ids),
        )
        if b_in != len(request_ids):
            raise ValueError(
                f"pixel_values_videos batch dim {b_in} does not match "
                f"request count {len(request_ids)}."
            )
        hidden = self._encode(graph_walk, pixel_values_videos)  # [B, N, D]
        return {
            rid: {"encoder_hidden": [hidden[i : i + 1]]}
            for i, rid in enumerate(request_ids)
        }


def _build_default_masks(
    encoder_hidden: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Build full-context / full-target masks given encoder hidden states.

    Matches HF's default in ``VJEPA2Model.forward`` when the caller doesn't
    supply masks: ``torch.arange(N).unsqueeze(0).repeat(B, 1)``.
    """
    b, n, _ = encoder_hidden.shape
    ids = torch.arange(n, device=encoder_hidden.device).unsqueeze(0).repeat(b, 1)
    return ids, ids


class VJepa2PredictorSubmodule(ARNodeSubmodule):
    """Masked latent predictor (non-AC).

    Expects ``encoder_hidden`` from the preceding encoder node.  Optionally
    accepts pre-built ``context_mask`` and ``target_mask`` edges; when
    absent, falls back to full-context / full-target defaults.  Output is
    the predicted hidden tensor at the target positions.

    Supports cross-request batching when ``encoder_hidden`` shapes agree
    across the batch AND (if provided) mask shapes agree.  All-defaults is
    the common case — the full-context / full-target masks are derived from
    ``encoder_hidden.shape`` and so are trivially homogeneous when the
    encoder outputs are.
    """

    def __init__(self, predictor: VJEPA2Predictor, config: VJepa2Config):
        super().__init__()
        self.predictor = predictor
        self.config = config

    def can_batch(
        self,
        batch: NodeBatch,
        model_inputs: list[ARNodeInputs],
    ) -> bool:
        # B=1 → sequential forward; see VJepa2EncoderSubmodule.can_batch
        # for the full justification (AC warm-latency regression on
        # forward_batched when it's the only path).
        if len(batch.request_ids) < 2:
            return False
        enc_shapes: set[tuple] = {
            _shape_key(inp.input_embeds) for inp in model_inputs
        }
        ctx_shapes: set[tuple | None] = {
            _shape_key(inp.tensor_inputs.get("context_mask")) for inp in model_inputs
        }
        tgt_shapes: set[tuple | None] = {
            _shape_key(inp.tensor_inputs.get("target_mask")) for inp in model_inputs
        }
        return (
            len(enc_shapes) == 1
            and len(ctx_shapes) == 1
            and len(tgt_shapes) == 1
        )

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> ARNodeInputs:
        """
            parts: list[torch.Tensor] = []
        for inp in per_request_inputs:
            t = inp[field_name][0]
            parts.append(_ensure_lead_batch_dim(t, target_rank))
        return torch.cat(parts, dim=0)

        """
        encoder_hidden = _ensure_lead_batch_dim(inputs["encoder_hidden"][0], 3)
        tensor_inputs = {}

        context_mask = inputs.get("context_mask")
        target_mask = inputs.get("target_mask")

        if context_mask:
            tensor_inputs["context_mask"] = _ensure_lead_batch_dim(context_mask[0], 2)
        if target_mask:
            tensor_inputs["target_mask"] = _ensure_lead_batch_dim(target_mask[0], 2)
        return ARNodeInputs(
            input_embeds=encoder_hidden,
            tensor_inputs=tensor_inputs
        )


    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        # Sequential (len == 1) and batched (len > 1) share this path.
        encoder_hidden = torch.cat([
            inp.input_embeds for inp in inputs
        ], dim=0)
        out: dict[str, torch.Tensor] = {
            "encoder_hidden": encoder_hidden
        }
        has_context_mask = "context_mask" in inputs[0].tensor_inputs
        has_target_mask = "target_mask" in inputs[0].tensor_inputs

        if has_context_mask:
            out["context_mask"] = torch.cat([
                inp.tensor_inputs["context_mask"] for inp in inputs
            ], dim=0)
        if has_target_mask:
            out["target_mask"] = torch.cat([
                inp.tensor_inputs["target_mask"] for inp in inputs
            ], dim=0)
        if (not has_context_mask) or not(has_target_mask):
            out["context_mask"], out["target_mask"] = _build_default_masks(encoder_hidden)
        return out

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        encoder_hidden: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        logger.info(
            "VJepa2PredictorSubmodule.forward: encoder_hidden shape=%s",
            tuple(encoder_hidden.shape),
        )
        if encoder_hidden.dim() == 2:
            encoder_hidden = encoder_hidden.unsqueeze(0)

        ctx_list = [context_mask]
        tgt_list = [target_mask]
        predicted = self.predictor(encoder_hidden, ctx_list, tgt_list)
        logger.info("VJepa2PredictorSubmodule.forward: output shape=%s", tuple(predicted.shape))
        return {"predicted_hidden": [predicted]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        encoder_hidden: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        request_ids = engine_inputs.request_ids

        ctx_list = [context_mask]
        tgt_list = [target_mask]
        logger.info(
            "VJepa2PredictorSubmodule.forward_batched: encoder_hidden=%s rids=%d",
            tuple(encoder_hidden.shape),
            len(request_ids),
        )
        predicted = self.predictor(encoder_hidden, ctx_list, tgt_list)  # [B, N_pred, D]
        return {
            rid: {"predicted_hidden": [predicted[i : i + 1]]}
            for i, rid in enumerate(request_ids)
        }


class VJepa2RolloutPredictorSubmodule(ARNodeSubmodule):
    """Masked-predictor rollout submodule — autoregressive anticipation.

    Parity target: upstream
    ``vjepa2/evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar.py``
    ``AnticipativeWrapper.forward`` (lines 172-224).  That Python loop calls
    the (stateless) predictor once per rollout step, feeding a sliding window
    of the previous step's prediction back as context:

        x_pred_input = x_full                          # initial encoder output
        for _ in range(num_steps):
            x_pred = predictor(x_pred_input, ctxt_positions, tgt_positions)
            x_pred_input = cat([x_pred_input[:, N_pred:], x_pred], dim=1)
            x_accumulate = cat([x_accumulate, x_pred], dim=1)

    In mstar this becomes a ``Loop`` whose section is a single node
    wrapping this submodule.  On each iter:

      * The loop-back ``encoder_hidden`` carries the sliding window state.
        Iter 0 receives it from the preceding ``video_encoder`` node; iter
        k > 0 receives the previous iter's emitted ``encoder_hidden``.
      * ``predicted_hidden`` is emitted both as a (dangling) section output
        — which lets the Loop's cache_outputs machinery pick it up — and
        into the Loop's ``accumulated_outputs`` for client delivery.

    The predictor nn.Module is the same instance used by
    ``VJepa2PredictorSubmodule`` — weights are loaded once and shared.  Only
    the preprocess/forward rollout-awareness differs.

    Uses the single-request sequential path (``can_batch=False`` inherited
    from ``NodeSubmodule``).  Same cross-request batching story as the
    encoder — enabling it requires an engine fix.
    """

    def __init__(
        self,
        predictor: VJEPA2Predictor,
        config: VJepa2Config,
        num_output_frames: int = 2,
        frames_per_second: int = 4,
        anticipation_seconds: float = 1.0,
    ):
        super().__init__()
        self.predictor = predictor
        self.config = config
        # ``num_output_frames`` is rounded up to tubelet_size — below
        # tubelet_size the math (N_pred = grid^2 * (nof // tubelet)) yields
        # zero target tokens, which would make the predictor call degenerate.
        self.num_output_frames = max(int(num_output_frames), config.tubelet_size)
        self.frames_per_second = int(frames_per_second)
        self.anticipation_seconds = float(anticipation_seconds)
        self._piecewise_runner = None

    def set_piecewise_runner(self, runner) -> None:
        """Attach a warmed-up PiecewiseCudaGraphRunner for the layer loop."""
        self._piecewise_runner = runner

    def get_piecewise_runner_config(self) -> dict | None:
        """Return construction args for PiecewiseCudaGraphRunner.

        Called by StatelessEngine.warmup() (masked predictor uses ENC_DEC).
        kv_cache_config / alloc_manager / buffer_manager are all None since the
        masked predictor is stateless — no KV cache between rollout steps.

        The ``position_mask`` buffer is a static [n_seq] float32 tensor holding
        the sorted position IDs for the rollout config.  For sequential context
        (arange(n_ctxt)) + skip target, the positions are already sorted so
        argsort in VJEPA2Predictor.forward is identity; the buffer is filled
        once in fn_factory and never updated at replay.
        """
        # n_ctxt = self.config.grid_depth * self._grid_size ** 2
        # n_pred = self._n_pred
        # skip = n_ctxt + self._grid_size ** 2 * self._anticipation_steps
        # n_seq = n_ctxt + n_pred

        # position_ids = torch.cat([
        #     torch.arange(n_ctxt, dtype=torch.float32),
        #     torch.arange(n_pred, dtype=torch.float32) + skip,
        # ])  # [n_seq], CPU; fn_factory .copy_()s it into the GPU buffer

        # predictor = self.predictor

        # def fn_factory(static_cm, static_pos_bufs):
        #     static_pos_bufs["position_mask"].copy_(position_ids)
        #     return predictor.make_layer_loop_fn(static_cm, static_pos_bufs)

        # return {
        #     "fn_factory": fn_factory,
        #     "embed_dim": self.config.pred_hidden_size,
        #     "capture_seq_len": n_seq,
        #     "capture_batch_sizes": [1, 2, 4, 8],
        #     "pos_buf_shapes": {"position_mask": (n_seq,)},
        #     "cache_labels": ["main"],
        # }

        """Masked predictor rollout does not use piecewise CUDA graphs.

        The non-AC predictor attends over n_ctxt + n_pred ≈ 8448 tokens using
        plain SDPA (no FlashInfer).  At that sequence length the O(N²) attention
        computation dominates completely — Python kernel-launch overhead is
        negligible relative to the ~1.7 GB attention matrix per layer, so CUDA
        graphs provide no meaningful speedup.  Attempting to capture them also
        risks OOM on top of the already-loaded ViT-g encoder weights.

        CUDA graphs are only useful for the AC predictor (capture_seq_len=258,
        FlashInfer per-call overhead worth eliminating).
        """
        return None

    # ------------------------------------------------------------------
    # Rollout geometry (derived from config + hyperparams; shape-static)
    # ------------------------------------------------------------------

    @property
    def _grid_size(self) -> int:
        return self.config.grid_size

    @property
    def _n_pred(self) -> int:
        """Number of predicted tokens per iteration.

        Matches ``AnticipativeWrapper`` line 206:
            N_pred = grid_size**2 * (num_output_frames // tubelet_size)
        """
        return self._grid_size * self._grid_size * (self.num_output_frames // self.config.tubelet_size)

    @property
    def _anticipation_steps(self) -> int:
        """Discrete (tubelet-aligned) time offset between the context window
        and the first predicted token.  From ``AnticipativeWrapper`` line 202.
        """
        return int(self.anticipation_seconds * self.frames_per_second / self.config.tubelet_size)

    # ------------------------------------------------------------------
    # NodeSubmodule ABC
    # ------------------------------------------------------------------

    def can_batch(self, batch: NodeBatch, model_inputs: list[ARNodeInputs]) -> bool:
        """Batch only when every request is at the same rollout iter AND
        their encoder_hidden shapes agree.

        Different iters can't share a forward because the sliding-window
        math ``torch.cat([hidden[:, n_pred:], predicted], dim=1)`` is
        symmetric across the batch dim only if all rids moved through the
        same number of prior iterations.  The scheduler groups same-iter
        requests together, so this is the common case; mixed-iter batches
        fall through to sequential.

        B=1 → sequential forward; see VJepa2EncoderSubmodule.can_batch for
        the rationale (AC warm-latency regression when forward_batched is
        the only path).  Rollout is especially sensitive because it calls
        forward_batched once per iter — amortizing a slow compile over H
        iters would multiply the pain.
        """
        if len(batch.request_ids) < 2:
            return False

        enc_shapes: set[tuple] = {
            _shape_key(inp.input_embeds) for inp in model_inputs
        }
        iters: set[int] = set()
        for rid in batch.request_ids:
            info = batch.per_request_info.get(rid)
            if info is None:
                return False
            iters.add(info.dynamic_loop_iter_counts.get("rollout_loop", 0))
        return (
            len(enc_shapes) == 1
            and len(iters) == 1
        )

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> ARNodeInputs:
        encoder_hidden = _ensure_lead_batch_dim(inputs["encoder_hidden"][0], target_rank=3)
        return ARNodeInputs(
            input_embeds=encoder_hidden,
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:

        return {
            "encoder_hidden": torch.cat([
                inp.input_embeds for inp in inputs
            ], dim=0)
        }

    def _rollout_step(
        self,
        encoder_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute one rollout step for a [B, N, D] encoder_hidden.

        Returns ``(next_encoder_hidden [B, N, D], predicted [B, n_pred, D])``.
        Shape math is symmetric across B.

        CUDA-graph path (when PiecewiseCudaGraphRunner is set): splits the
        predictor forward into preamble (embed+sort), captured layer loop,
        and postamble (layernorm+unsort+proj), with no KV cache involved.
        Eager path calls predictor.forward end-to-end (unchanged).
        """
        b, n_ctxt, _ = encoder_hidden.shape
        device = encoder_hidden.device
        n_pred = self._n_pred
        if n_pred >= n_ctxt:
            raise ValueError(
                f"Rollout requires n_pred ({n_pred}) < encoder context length "
                f"({n_ctxt}); reduce num_output_frames or the anticipation horizon."
            )
        ctxt_positions = torch.arange(n_ctxt, device=device).unsqueeze(0).repeat(b, 1)
        skip = n_ctxt + self._grid_size * self._grid_size * self._anticipation_steps
        tgt_positions = (torch.arange(n_pred, device=device) + skip).unsqueeze(0).repeat(b, 1)

        runner = self._piecewise_runner
        if runner is not None and runner.can_run(b):
            hidden_states, n_ctxt_out, argsort = self.predictor._run_forward_piecewise(
                encoder_hidden, [ctxt_positions], [tgt_positions]
            )
            hidden_states = runner.run(hidden_states)
            predicted = self.predictor._finalize_forward_piecewise(hidden_states, n_ctxt_out, argsort)
        else:
            predicted = self.predictor(encoder_hidden, [ctxt_positions], [tgt_positions])

        next_encoder_hidden = torch.cat([encoder_hidden[:, n_pred:, :], predicted], dim=1)
        return next_encoder_hidden, predicted

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        encoder_hidden: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        """One rollout step.  Runs the masked predictor, then slides the
        encoder_hidden window forward so iter k+1's context is the
        concatenation of the last (N - N_pred) context tokens plus the
        N_pred newly-predicted tokens.
        """
        if encoder_hidden.dim() == 2:
            encoder_hidden = encoder_hidden.unsqueeze(0)
        request_info = engine_inputs.single_request_info

        iter_idx = request_info.dynamic_loop_iter_counts.get("rollout_loop", 0)

        logger.info(
            "VJepa2RolloutPredictorSubmodule.forward: iter=%d encoder_hidden=%s",
            iter_idx,
            tuple(encoder_hidden.shape),
        )
        next_encoder_hidden, predicted = self._rollout_step(encoder_hidden)

        logger.info(
            "VJepa2RolloutPredictorSubmodule.forward: predicted=%s next_encoder_hidden=%s",
            tuple(predicted.shape),
            tuple(next_encoder_hidden.shape),
        )
        return {
            "encoder_hidden": [next_encoder_hidden],
            "predicted_hidden": [predicted],
        }


    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        encoder_hidden: torch.Tensor,
        **kwargs, # coming from preprocess output
    )  -> dict[str, NameToTensorList]: # request_id to tensors
        """Batched rollout step across shape-homogeneous same-iter requests.

        ``can_batch`` guarantees all rids are at the same ``iter_idx`` (else
        we fall to sequential), so the rollout-step math is a single
        ``[B, N, D]`` forward.  Per-request early-exit (``check_stop``)
        still fires independently when each rid's ``rollout_horizon`` is
        reached — individual rids can drop out while others continue, and
        the scheduler re-batches remaining rids on the next iter.
        """
        request_ids = engine_inputs.request_ids
        per_request_info = engine_inputs.per_request_info

        if encoder_hidden.size(0) != len(request_ids):
            raise ValueError(
                f"encoder_hidden batch dim {encoder_hidden.size(0)} does not "
                f"match request count {len(request_ids)}."
            )

        # All rids at same iter (can_batch invariant); read any one for logs.
        iter_idx = per_request_info[request_ids[0]].dynamic_loop_iter_counts.get("rollout_loop", 0)
        logger.info(
            "VJepa2RolloutPredictorSubmodule.forward_batched: iter=%d encoder_hidden=%s rids=%d",
            iter_idx,
            tuple(encoder_hidden.shape),
            len(request_ids),
        )

        next_encoder_hidden, predicted = self._rollout_step(encoder_hidden)

        return {
            rid: {
                "encoder_hidden": [next_encoder_hidden[i : i + 1]],
                "predicted_hidden": [predicted[i : i + 1]],
            }
            for i, rid in enumerate(request_ids)
        }

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        iter_idx = request_info.dynamic_loop_iter_counts.get("rollout_loop", 0)
        horizon = int(request_info.step_metadata.get("rollout_horizon", 0) or 0)
        if horizon > 0 and iter_idx + 1 >= horizon:
            logger.info(
                "VJepa2RolloutPredictorSubmodule.forward: horizon H=%d reached at iter=%d; stopping rollout_loop.",
                horizon,
                iter_idx,
            )
            return {"rollout_loop"}


class VJepa2ACPredictorSubmodule(ARNodeSubmodule):
    """Action-conditioned predictor (V-JEPA 2-AC).

    Additional required inputs vs the masked predictor: ``actions``,
    ``states``, and optionally ``extrinsics``.  Each is per-timestep, shape
    ``[T, action_embed_dim]`` (or ``[T, action_embed_dim - 1]`` for
    extrinsics), delivered as a GraphEdge by the model's ``process_prompt``
    / ``get_initial_forward_pass_args``.  Output is ``predicted_hidden``
    with the same shape as ``encoder_hidden``.
    """

    def __init__(self, predictor: VisionTransformerPredictorAC, config: VJepa2Config):
        super().__init__()
        self.predictor = predictor
        self.config = config

    def can_batch(self, batch: NodeBatch, model_inputs: list[ARNodeInputs]) -> bool:
        # B=1 → sequential forward.  See VJepa2EncoderSubmodule.can_batch
        # for the rationale — AC ViT-g warm-forward_batched measured ~20×
        # slower than warm-forward on the same input shape, so routing
        # single-request traffic through the batched path is a strict
        # regression.  Only opt into forward_batched when there's an
        # actual multi-rid batch to amortize the compile cost over.
        if len(batch.request_ids) < 2:
            return False
        shapes: set[tuple] = set()

        for inp in model_inputs:
            enc = inp.input_embeds
            act = inp.tensor_inputs.get("actions")
            st = inp.tensor_inputs.get("states")
            ext = inp.tensor_inputs.get("extrinsics")
            shapes.add((_shape_key(enc), _shape_key(act), _shape_key(st), _shape_key(ext)))
        return len(shapes) == 1

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> ARNodeInputs:

        encoder_hidden = _ensure_lead_batch_dim(inputs["encoder_hidden"][0], 3)

        tensor_inputs = {}
        tensor_inputs["actions"] = _ensure_lead_batch_dim(inputs["actions"][0], 3)
        tensor_inputs["states"] = _ensure_lead_batch_dim(inputs["states"][0], 3)

        if "extrinsics" in inputs:
            tensor_inputs["extrinsics"] = _ensure_lead_batch_dim(inputs["extrinsics"][0], 3)

        return ARNodeInputs(
            input_embeds=encoder_hidden,
            tensor_inputs=tensor_inputs
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {
            "encoder_hidden": torch.cat([
                inp.input_embeds for inp in inputs
            ], dim=0)
        }
        out["actions"] = torch.cat([
            inp.tensor_inputs["actions"] for inp in inputs
        ], dim=0)
        out["states"] = torch.cat([
            inp.tensor_inputs["states"] for inp in inputs
        ], dim=0)
        if "extrinsics" in inputs[0].tensor_inputs:
            out["extrinsics"] = torch.cat([
                inp.tensor_inputs["extrinsics"] for inp in inputs
            ], dim=0)
        return out

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        encoder_hidden: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if encoder_hidden.dim() == 2:
            encoder_hidden = encoder_hidden.unsqueeze(0)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        if states.dim() == 2:
            states = states.unsqueeze(0)
        if extrinsics is not None and extrinsics.dim() == 2:
            extrinsics = extrinsics.unsqueeze(0)
        predicted = self.predictor(encoder_hidden, actions, states, extrinsics=extrinsics)
        return {"predicted_hidden": [predicted]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        encoder_hidden: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        **kwargs, # coming from preprocess output
    )  -> dict[str, NameToTensorList]:
        request_ids = engine_inputs.request_ids

        if encoder_hidden.size(0) != len(request_ids):
            raise ValueError(
                f"encoder_hidden batch dim {encoder_hidden.size(0)} does not "
                f"match request count {len(request_ids)}."
            )
        logger.info(
            "VJepa2ACPredictorSubmodule.forward_batched: enc=%s act=%s st=%s rids=%d",
            tuple(encoder_hidden.shape),
            tuple(actions.shape),
            tuple(states.shape),
            len(request_ids),
        )
        predicted = self.predictor(
            encoder_hidden, actions, states, extrinsics=extrinsics,
        )  # [B, N, out_dim]
        return {
            rid: {"predicted_hidden": [predicted[i : i + 1]]}
            for i, rid in enumerate(request_ids)
        }


class VJepa2ACRolloutPredictorSubmodule(ARNodeSubmodule):
    """Action-conditioned autoregressive rollout.

    Optionally uses a PiecewiseCudaGraphRunner to accelerate the inner block
    loop. Set via set_piecewise_runner() after construction (typically done by
    the engine after warmup_and_capture).

    **Sliding-window rollout** — diverges from upstream
    ``vjepa2/notebooks/utils/mpc_utils.py::cem`` which uses growing-context
    (``T: 1 → rollout+1``) from a single-tubelet initial encoding.  Our
    encoder's natural output is ``T=grid_depth`` (a full 64-frame clip), so
    growing-context would immediately hit the AC predictor's pre-built
    causal attn-mask cap, so we slide a fixed-size window instead.

    Per iter ``k``:
      * Slice actions/states: ``actions_k = actions[:, k : k + T_ctx]``
        (T_ctx = ``config.grid_depth``; client supplies the full trajectory
        once and we slice per-iter based on ``dynamic_loop_iter_counts``).
      * Run ``VisionTransformerPredictorAC(encoder_hidden, actions_k, states_k)``
        — returns ``[B, N, D]`` predictions at every one of the T_ctx
        timesteps.
      * Take the last tubelet group (``predicted[:, -grid² :]`` — shape
        ``[B, grid², D]``) as the "new" imagined state, matching upstream
        ``world_model``'s ``next_frame = predictor_output[:, -HW:, :]``
        interpretation.
      * Slide the encoder_hidden window: drop oldest ``grid²`` tokens,
        append the new ``grid²`` tokens.  Fixed shape ``[B, N, D]`` every
        iter — torch.compile-friendly, no dynamic-shape recompiles.

    ``actions`` / ``states`` (and optionally ``extrinsics``) are returned
    unchanged as identity loop-back edges so the graph dispatcher can keep
    routing them on every iter.  Per-iter slicing happens inside the
    forward based on ``iter_idx`` — the tensors themselves don't change.

    Shares the sibling rollout submodule's batching contract (``B >= 2``
    + shape + same-iter homogeneity) and ``check_stop`` early-exit
    semantics.
    """

    def __init__(
        self,
        predictor: VisionTransformerPredictorAC,
        config: VJepa2Config,
    ):
        super().__init__()
        self.predictor = predictor
        self.config = config
        self._piecewise_runner = None   # set via set_piecewise_runner() after warmup

    def set_piecewise_runner(self, runner) -> None:
        """Attach a warmed-up PiecewiseCudaGraphRunner for the block loop."""
        self._piecewise_runner = runner

    def get_piecewise_runner_config(self) -> dict | None:
        """Return construction args for PiecewiseCudaGraphRunner, or None.

        Called by KVCacheEngine.warmup() to build and install the runner without
        the engine needing to know the model internals.  The returned dict
        contains everything PiecewiseCudaGraphRunner.__init__ needs except
        device/autocast_dtype/memory_pool (those come from the engine).

        Keys:
          fn_factory    - (static_cm, static_pos_bufs) -> fn(x) -> x
          embed_dim     - hidden dim of the intermediate sequence
          capture_seq_len - tokens per frame (cond_tokens + grid_h * grid_w)
          pos_buf_shapes  - {name: shape} for per-step position tensors
          cache_labels    - KV cache labels (always ["main"] here)
        """
        ac = self.config.ac_predictor
        if ac is None:
            return None
        N = ac.img_size[0] // ac.patch_size   # grid_height (== grid_width for square)
        cond_tokens = 3 if ac.use_extrinsics else 2
        predictor = self.predictor

        def fn_factory(static_cm, static_pos_bufs):
            return predictor.make_block_loop_fn(static_cm, static_pos_bufs, cond_tokens)

        return {
            "fn_factory": fn_factory,
            "embed_dim": ac.predictor_embed_dim,
            "capture_seq_len": cond_tokens + N * N,
            "capture_batch_sizes": [1, 2, 4, 8],
            "pos_buf_shapes": {
                "d_pos":   (N * N,),
                "h_pos":   (N * N,),
                "w_pos":   (N * N,),
                "time_pos": (cond_tokens,),
            },
            "cache_labels": ["main"],
        }

    # ------------------------------------------------------------------
    # Rollout geometry
    # ------------------------------------------------------------------

    @property
    def _grid_size(self) -> int:
        return self.config.grid_size

    @property
    def _grid_depth(self) -> int:
        """T_ctx — number of tubelet-group timesteps the AC predictor sees
        per forward.  Derives from the encoder config; at ViT-g @ 256 this
        is ``64 / 2 = 32``.
        """
        return self.config.grid_depth

    def _window_tokens(self) -> int:
        """Per-iter slide size — one tubelet group = ``grid²`` spatial tokens.

        This matches (a) upstream ``mpc_utils.cem`` which advances 1 frame
        per iter, and (b) our masked ``VJepa2RolloutPredictorSubmodule``
        default ``num_output_frames=2 = 1 tubelet``.
        """
        return self._grid_size * self._grid_size

    # ------------------------------------------------------------------
    # NodeSubmodule ABC
    # ------------------------------------------------------------------

    def can_batch(self, batch: NodeBatch, model_inputs: list[ARNodeInputs]) -> bool:
        """Same rule as masked rollout: B >= 2, shape-homogeneous across
        ``encoder_hidden`` / ``actions`` / ``states`` (+ optional
        ``extrinsics``), and same ``iter_idx`` for every rid.

        B=1 → sequential ``forward`` path (see
        ``VJepa2EncoderSubmodule.can_batch`` for the AC warm-latency
        regression that motivates this gate).
        """
        # if len(batch.request_ids) < 2:
        #     return False

        iters: set[int] = set()

        enc_shapes: set[tuple] = {
            _shape_key(inp.input_embeds) for inp in model_inputs
        }
        act_shapes: set[tuple] = {
            _shape_key(inp.tensor_inputs.get("actions")) for inp in model_inputs
        }
        st_shapes: set[tuple] = {
            _shape_key(inp.tensor_inputs.get("states")) for inp in model_inputs
        }
        ext_shapes: set[tuple | None] = {
            _shape_key(inp.tensor_inputs.get("extrinsics")) for inp in model_inputs
        }
        for rid in batch.request_ids:
            info = batch.per_request_info.get(rid)
            if info is None:
                return False
            iters.add(info.dynamic_loop_iter_counts.get("rollout_loop", 0))
        return (
            len(enc_shapes) == 1
            and len(act_shapes) == 1
            and len(st_shapes) == 1
            and len(ext_shapes) == 1
            and len(iters) == 1
        )

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> ARNodeInputs:
        iter_idx = fwd_info.dynamic_loop_iter_counts.get("rollout_loop", 0)

        encoder_hidden = _ensure_lead_batch_dim(inputs["encoder_hidden"][0], 3)

        tensor_inputs = {}
        tensor_inputs["actions"] = _ensure_lead_batch_dim(inputs["actions"][0], 3)[:, iter_idx:iter_idx+1]
        tensor_inputs["states"] = _ensure_lead_batch_dim(inputs["states"][0], 3)[:, iter_idx:iter_idx+1]

        if "extrinsics" in inputs:
            tensor_inputs["extrinsics"] = _ensure_lead_batch_dim(inputs["extrinsics"][0], 3)[:, iter_idx:iter_idx+1]

        return ARNodeInputs(
            input_embeds=encoder_hidden,
            tensor_inputs=tensor_inputs
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {
            "encoder_hidden": torch.cat([
                inp.input_embeds for inp in inputs
            ], dim=0)
        }
        out["actions"] = torch.cat([
            inp.tensor_inputs["actions"] for inp in inputs
        ], dim=0)
        out["states"] = torch.cat([
            inp.tensor_inputs["states"] for inp in inputs
        ], dim=0)

        per_req_seq_len = out["encoder_hidden"].shape[1] + 2
        if "extrinsics" in inputs[0].tensor_inputs:
            out["extrinsics"] = torch.cat([
                inp.tensor_inputs["extrinsics"] for inp in inputs
            ], dim=0)
            per_req_seq_len += 1

        # plan attention
        engine_inputs.cache_manager.plan_attention(
            seq_lens=[per_req_seq_len] * len(inputs),
            is_causal=False
        )

        return out

    def _rollout_step(
        self,
        encoder_hidden: torch.Tensor,            # [B, H*W, D]
        actions: torch.Tensor,                   # [B, 1, action_embed_dim]
        states: torch.Tensor,                    # [B, 1, action_embed_dim]
        t_0: int,
        cache_handle: BatchedCacheManager,
        extrinsics: torch.Tensor | None = None,  # [B, 1, action_embed_dim - 1] or None
        request_ids: list[str] | None = None,    # needed by PiecewiseCudaGraphRunner
    ) -> torch.Tensor:
        """One AC rollout step using either the CUDA-graph or eager path.

        The CUDA-graph path (when PiecewiseCudaGraphRunner is attached):
          1. Preamble (predictor_embed + action/state concat) runs eagerly.
          2. Position tensors are computed eagerly (hoisted out of the graph).
          3. Block loop replays the captured graph.
          4. advance_seq_len is called by the runner after replay.
          5. Postamble (drop action tokens, norm, proj) runs eagerly.

        The eager path calls predictor.forward end-to-end (same as before).
        """
        p = self.predictor

        runner = self._piecewise_runner
        if runner is not None and runner.can_run(encoder_hidden.size(0)):
            # --- CUDA-graph path ---
            x, cond_tokens, b, t = p._prepare_sequence(
                encoder_hidden, actions, states, extrinsics
            )
            d_pos, h_pos, w_pos, time_pos = p._compute_rope_positions(
                t_0, p.grid_height, p.grid_width, cond_tokens, x.device, x.dtype
            )
            pos_bufs = {
                "d_pos": d_pos, "h_pos": h_pos, "w_pos": w_pos,
            }
            if time_pos is not None:
                pos_bufs["time_pos"] = time_pos

            x = runner.run(
                x=x,
                pos_bufs=pos_bufs,
                request_ids=request_ids,
            )
            # advance_seq_len already called by runner; skip it in _decode_sequence
            new_tg = p._decode_sequence(x, cond_tokens, b, t)
        else:
            # --- Eager path ---
            new_tg = self.predictor(
                encoder_hidden,
                actions,
                states,
                extrinsics=extrinsics,
                t_0=t_0,
                cache_handle=cache_handle,
            )

        # Per-step LayerNorm — matches the upstream notebook's step_predictor
        # body (vjepa2/notebooks/energy_landscape_example.ipynb Cell 5;
        # also world_model_wrapper.py:60-61 inside infer_next_action;
        # also app/vjepa_droid/train.py:421-422 forward_predictions).
        # The AC predictor is trained with normalize_reps=True per the
        # published configs/train/vitg16/droid-256px-8f.yaml, so this
        # F.layer_norm puts each rollout step's output back in the
        # distribution the next step's predictor expects.
        return F.layer_norm(new_tg, (new_tg.size(-1),))

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        encoder_hidden: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        request_info = engine_inputs.single_request_info
        iter_idx = request_info.dynamic_loop_iter_counts.get("rollout_loop", 0)
        logger.info(
            "VJepa2ACRolloutPredictorSubmodule.forward: iter=%d encoder_hidden=%s actions=%s states=%s",
            iter_idx,
            tuple(encoder_hidden.shape),
            tuple(actions.shape),
            tuple(states.shape),
        )

        new_tg = self._rollout_step(
            encoder_hidden, actions, states,
            t_0=iter_idx,
            cache_handle=engine_inputs.cache_manager,
            extrinsics=extrinsics,
            request_ids=engine_inputs.request_ids,
        )

        logger.info(
            "VJepa2ACRolloutPredictorSubmodule.forward: new_tg=%s",
            tuple(new_tg.shape),
        )
        out: NameToTensorList = {
            "encoder_hidden": [new_tg],
            "predicted_hidden": [new_tg],
        }
        return out


    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        encoder_hidden: torch.Tensor, # [B, N, D]
        actions: torch.Tensor, # [B, T_total, action_embed_dim]
        states: torch.Tensor, # [B, T_total, action_embed_dim]
        extrinsics: torch.Tensor | None = None,
        **kwargs, # coming from preprocess output
    )  -> dict[str, NameToTensorList]:

        request_ids = engine_inputs.request_ids
        if encoder_hidden.size(0) != len(request_ids):
            raise ValueError(
                f"encoder_hidden batch dim {encoder_hidden.size(0)} does not "
                f"match request count {len(request_ids)}."
            )

        # Now, can_batch ensures that all requests in a batch have the same loop index
        # TODO: this no longer needs to be the case.
        iter_idx = engine_inputs.first_request_info.dynamic_loop_iter_counts.get(
            "rollout_loop", 0)

        new_tg = self._rollout_step(
            encoder_hidden, actions, states,
            t_0=iter_idx,
            cache_handle=engine_inputs.cache_manager,
            extrinsics=extrinsics,
            request_ids=engine_inputs.request_ids,
        )

        outputs: NameToTensorList = {
            "encoder_hidden": [new_tg],
            "predicted_hidden": [new_tg],
        }

        per_rid = {
            rid: {
                name: tensor[i:i+1] for name, tensor in outputs.items()
                } for i, rid in enumerate(request_ids)
        }
        return per_rid

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        horizon = int(request_info.step_metadata.get("rollout_horizon", 0) or 0)
        iter_idx = request_info.dynamic_loop_iter_counts.get("rollout_loop", 0)
        if horizon > 0 and iter_idx + 1 >= horizon:
            logger.info(
                "VJepa2ACRolloutPredictorSubmodule.forward: horizon H=%d reached at iter=%d; stopping rollout_loop.",
                horizon,
                iter_idx,
            )
            return {"rollout_loop"}


# ---------------------------------------------------------------------------
# MPC submodules (intra-request K-way candidate evaluation)
#
# The AC predictor's ``forward(encoder_hidden, actions, states)`` is already
# batch-dim aware (upstream ``mpc_utils.cem`` uses exactly this pattern via
# ``context_frame.repeat(K, 1, 1, 1)``).  Our MPC walk runs ONE request end-
# to-end: a single encoder forward produces ``encoder_hidden = [1, N, D]``,
# then the MPC predictor broadcasts it to ``[K, N, D]`` via ``.expand`` and
# runs the AC predictor with ``actions/states`` of shape ``[K, T, 7]``.
# The scorer then L1-compares each of K predicted latents against a single
# goal latent (pre-encoded by the client) and emits ``best_index`` + per-
# candidate costs + the full ``[K, N, D]`` predicted stack.
#
# No ``can_batch`` override: MPC is intra-request, so cross-request batching
# is meaningless here — the K-way batch dim is already saturating the
# predictor.  If a future deployment needs cross-request batching of MPC
# requests too (K1 rids × K2 candidates), the natural path is the same
# engine batching used for the single-candidate predictor.
# ---------------------------------------------------------------------------


class VJepa2MPCPredictorSubmodule(ARNodeSubmodule):
    """Single-request K-way AC predictor forward.

    Inputs:
      * ``encoder_hidden``: ``[1, N, D]`` — context-video latent from the
        preceding ``video_encoder`` node.
      * ``actions``:  ``[K, T, 7]`` — K candidate action trajectories.
      * ``states``:   ``[K, T, 7]`` — matching proprioceptive states.
      * ``extrinsics`` (optional, use_extrinsics=True on the config).

    Output: ``predicted_hidden`` of shape ``[K, N, out_dim]`` — all K
    candidates' predicted future latents.

    Parity anchor: ``vjepa2/notebooks/utils/mpc_utils.py::cem`` lines 62-64
    (context expansion) + line 114 (world_model call with K-batched inputs).
    """

    def __init__(self, predictor: VisionTransformerPredictorAC, config: VJepa2Config):
        super().__init__()
        self.predictor = predictor
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> ARNodeInputs:

        enc = inputs["encoder_hidden"][0]
        actions = inputs["actions"][0]
        states = inputs["states"][0]

        tensor_inputs: dict[str, torch.Tensor] = {
            "actions": actions,
            "states": states,
        }
        if "extrinsics" in inputs:
            tensor_inputs["extrinsics"] = inputs["extrinsics"][0]

        return ARNodeInputs(
            input_embeds=enc,
            tensor_inputs=tensor_inputs
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        # Single request — MPC is intra-request K-way.  Enforce explicitly
        # so a future scheduler change that co-batches MPC requests trips
        # this immediately instead of silently mis-broadcasting.
        if len(inputs) != 1:
            raise ValueError(
                f"VJepa2MPCPredictorSubmodule runs one request at a time; "
                f"got batch of {len(inputs)}."
            )
        inputs = inputs[0]
        return {
            "encoder_hidden": inputs.input_embeds,
            **inputs.tensor_inputs
        }


    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        encoder_hidden: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        # Normalize encoder_hidden to [1, N, D].
        if encoder_hidden.dim() == 2:
            encoder_hidden = encoder_hidden.unsqueeze(0)
        if encoder_hidden.size(0) != 1:
            raise ValueError(
                f"MPC predictor expects encoder_hidden [1, N, D]; got "
                f"{tuple(encoder_hidden.shape)} — context should be a single "
                "request's latent before K-way broadcast."
            )
        # Broadcast to [K, N, D].  ``.expand`` is zero-copy; the predictor's
        # internal ops (``predictor_embed`` Linear etc.) will materialize as
        # they consume.  This matches mpc_utils.cem's ``.repeat(K,1,1,1)``
        # at the output level.
        k = actions.size(0)
        enc_k = encoder_hidden.expand(k, -1, -1).contiguous()

        predicted = self.predictor(enc_k, actions, states, extrinsics=extrinsics)
        logger.info(
            "VJepa2MPCPredictorSubmodule.forward: K=%d enc_in=%s actions=%s predicted=%s",
            k,
            tuple(encoder_hidden.shape),
            tuple(actions.shape),
            tuple(predicted.shape),
        )
        return {"predicted_hidden": [predicted]}


class VJepa2MPCScorerSubmodule(NodeSubmodule):
    """Score K candidate predicted latents against a goal latent.

    Inputs:
      * ``predicted_hidden``: ``[K, N, D]`` — from the MPC predictor.
      * ``goal_hidden``:      ``[1, N, D]`` (or ``[N, D]``) — pre-encoded
        by the client via a prior ``prefill_video_encoder_only`` call.

    Outputs emitted to the client:
      * ``best_index`` (int64 scalar): argmin of costs.
      * ``costs`` (``[K]``): per-candidate cost values.
      * ``predicted_hidden`` (``[K, N, D]``): passed through so clients
        that want the imagined trajectories (e.g. for visualization) get
        them without a second request.

    Cost function is selectable via ``config.mpc_cost_fn``:
      * ``"l1"`` (default): ``(pred - goal).abs().mean(dim=[1, 2])`` — matches
        upstream ``mpc_utils.py::l1``.
      * ``"l2"``: squared-error mean.
      * ``"cosine"``: 1 - cosine similarity (so argmin still picks best).
    """

    def __init__(self, config: VJepa2Config):
        super().__init__()
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:

        return NodeInputs(
            tensor_inputs={
                "predicted_hidden": inputs["predicted_hidden"][0],
                "goal_hidden": inputs["goal_hidden"][0],
            }
        )
    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor]:

        if len(inputs) != 1:
            raise ValueError(
                f"VJepa2MPCScorerSubmodule runs one request at a time; "
                f"got batch of {len(inputs)}."
            )
        return inputs[0].tensor_inputs

    def _cost(self, pred: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Compute per-candidate cost for ``pred [K, N, D]`` vs ``goal [1, N, D]``.

        Returns ``[K]``.  Lower cost = closer to goal.
        """
        fn = getattr(self.config, "mpc_cost_fn", "l1")
        # Flatten N*D so shape is [K, N*D] / [1, N*D].
        p = pred.flatten(start_dim=1)
        g = goal.flatten(start_dim=1)
        if fn == "l1":
            return (p - g).abs().mean(dim=-1)
        if fn == "l2":
            return ((p - g) ** 2).mean(dim=-1)
        if fn == "cosine":
            # 1 - cosine_sim so argmin still picks best.
            p_n = torch.nn.functional.normalize(p, dim=-1)
            g_n = torch.nn.functional.normalize(g, dim=-1)
            return 1.0 - (p_n * g_n).sum(dim=-1)
        raise ValueError(
            f"Unknown mpc_cost_fn {fn!r}; expected one of 'l1', 'l2', 'cosine'."
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        predicted_hidden: torch.Tensor,
        goal_hidden: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        # Normalize goal to [1, N, D] if needed (client may send [N, D]).
        if goal_hidden.dim() == 2:
            goal_hidden = goal_hidden.unsqueeze(0)
        if goal_hidden.size(0) != 1:
            raise ValueError(
                f"goal_hidden must have batch dim 1; got shape "
                f"{tuple(goal_hidden.shape)}."
            )
        if predicted_hidden.dim() == 2:
            predicted_hidden = predicted_hidden.unsqueeze(0)

        # Sanity: [K, N, D] vs [1, N, D] agree on N and D.
        if predicted_hidden.shape[1:] != goal_hidden.shape[1:]:
            raise ValueError(
                f"predicted_hidden feature shape {tuple(predicted_hidden.shape[1:])} "
                f"does not match goal_hidden feature shape "
                f"{tuple(goal_hidden.shape[1:])}."
            )

        costs = self._cost(predicted_hidden, goal_hidden)  # [K]
        best = torch.argmin(costs).to(torch.int64)
        logger.info(
            "VJepa2MPCScorerSubmodule.forward: K=%d best=%d min_cost=%.4f max_cost=%.4f",
            int(predicted_hidden.size(0)),
            int(best.item()),
            float(costs.min().item()),
            float(costs.max().item()),
        )
        return {
            "best_index": [best],
            "costs": [costs],
            "predicted_hidden": [predicted_hidden],
        }
