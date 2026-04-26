"""NodeSubmodule wrappers for the V-JEPA 2 graph nodes.

Three submodules:

  VJepa2EncoderSubmodule   - ViT 3D-patch video encoder.
  VJepa2PredictorSubmodule - masked latent predictor (single forward per call).
  VJepa2ACPredictorSubmodule - action-conditioned predictor (ditto).

All three are stateless (no KV cache, no per-iteration state) and are
dispatched through ``EncoderDecoderEngine``.  ``preprocess`` stacks
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

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import NodeBatch
from mminf.model.submodule_base import ARNodeInputs, ARNodeSubmodule, ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mminf.model.vjepa2.components.ac_predictor import VisionTransformerPredictorAC
from mminf.model.vjepa2.components.predictor import VJEPA2Predictor
from mminf.model.vjepa2.components.vit_encoder import VJEPA2Encoder
from mminf.model.vjepa2.config import VJepa2Config

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

    Supports cross-request batching (Phase 3): requests with identical
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
        # Route B=1 through the sequential ``forward`` (proven Phase 2 path).
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
        # Phase 1 behaviour.
        stacked = torch.cat(
            [inp.tensor_inputs["video_frames"] for inp in inputs],
            dim=0,
        )
        return {"pixel_values_videos": stacked}

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
        hidden = self.encoder(pixel_values_videos)
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
        hidden = self.encoder(pixel_values_videos)  # [B, N, D]
        return {
            rid: {"encoder_hidden": [hidden[i : i + 1]]}
            for i, rid in enumerate(request_ids)
        }


def _build_default_masks(
    encoder_hidden: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build full-context / full-target masks given encoder hidden states.

    Matches HF's default in ``VJEPA2Model.forward`` when the caller doesn't
    supply masks: ``torch.arange(N).unsqueeze(0).repeat(B, 1)``.
    """
    b, n, _ = encoder_hidden.shape
    ids = torch.arange(n, device=encoder_hidden.device).unsqueeze(0).repeat(b, 1)
    return [ids], [ids]


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
        if "context_mask" in inputs:
            tensor_inputs["context_mask"] = _ensure_lead_batch_dim(inputs["context_mask"][0], 2)
        if "target_mask" in inputs:
            tensor_inputs["target_mask"] = _ensure_lead_batch_dim(inputs["target_mask"][0], 2)
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
        out: dict[str, torch.Tensor] = {
            "encoder_hidden": torch.cat([
                inp.input_embeds for inp in inputs
            ], dim=0)
        }
        if "context_mask" in inputs[0].tensor_inputs:
            out["context_mask"] = torch.cat([
                inp.tensor_inputs["context_mask"] for inp in inputs
            ], dim=0)
        if "target_mask" in inputs[0].tensor_inputs:
            out["target_mask"] = torch.cat([
                inp.tensor_inputs["target_mask"] for inp in inputs
            ], dim=0)
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
        if context_mask is None or target_mask is None:
            ctx_list, tgt_list = _build_default_masks(encoder_hidden)
        else:
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
        if context_mask is None or target_mask is None:
            ctx_list, tgt_list = _build_default_masks(encoder_hidden)
        else:
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
    """Masked-predictor rollout submodule — Phase 2 autoregressive anticipation.

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

    In mminf this becomes a ``DynamicLoop`` whose section is a single node
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

    Shares Phase 1's single-request sequential path (``can_batch=False``
    inherited from ``NodeSubmodule``).  Same Phase-3 batching story as the
    encoder — enabling it requires the engine fix documented in the plan.
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
        Shape math is symmetric across B, so this function is used by both
        the sequential and batched paths.
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
        # Per-request early-exit.  The DynamicLoop's ``max_iters`` is a
        # config-level upper bound (``max_rollout_horizon``); the caller's
        # ``rollout_horizon`` — snapshotted into ``step_metadata`` by
        # ``VJepa2Model.get_initial_forward_pass_args`` — shortens the loop
        # here.  ``iter_idx`` is the count BEFORE this forward, so after this
        # call the loop will have completed ``iter_idx + 1`` iterations.
        rollout_horizon = int(request_info.step_metadata.get("rollout_horizon", 0) or 0)
        if rollout_horizon > 0 and iter_idx + 1 >= rollout_horizon:
            logger.info(
                "VJepa2RolloutPredictorSubmodule.forward: reached horizon H=%d at iter=%d; stopping rollout_loop.",
                rollout_horizon,
                iter_idx,
            )
            request_info.register_loop_stop("rollout_loop")

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
        ``[B, N, D]`` forward.  Per-request early-exit (``register_loop_stop``)
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

        # Per-rid early exit.
        for rid in request_ids:
            info = per_request_info[rid]
            horizon = int(info.step_metadata.get("rollout_horizon", 0) or 0)
            if horizon > 0 and iter_idx + 1 >= horizon:
                info.register_loop_stop("rollout_loop")

        return {
            rid: {
                "encoder_hidden": [next_encoder_hidden[i : i + 1]],
                "predicted_hidden": [predicted[i : i + 1]],
            }
            for i, rid in enumerate(request_ids)
        }

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
    """Action-conditioned autoregressive rollout — Phase 3.D.

    **Sliding-window rollout** — diverges from upstream
    ``vjepa2/notebooks/utils/mpc_utils.py::cem`` which uses growing-context
    (``T: 1 → rollout+1``) from a single-tubelet initial encoding.  Our
    encoder's natural output is ``T=grid_depth`` (a full 64-frame clip), so
    growing-context would immediately hit the AC predictor's pre-built
    causal attn-mask cap.  See the plan's P3.D "Sliding-window vs upstream
    growing-context" section for the full rationale.

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
    + shape + same-iter homogeneity) and ``register_loop_stop`` early-exit
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
        if len(batch.request_ids) < 2:
            return False

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
    
    def _rollout_step(
        self,
        encoder_hidden: torch.Tensor,   # [B, N, D]
        actions: torch.Tensor,           # [B, T_total, action_embed_dim]
        states: torch.Tensor,            # [B, T_total, action_embed_dim]
        extrinsics: torch.Tensor | None, # [B, T_total, action_embed_dim - 1] or None
        iter_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One sliding-window AC rollout step.

        Returns ``(next_encoder_hidden [B, N, D], predicted_new_tg [B, grid², D])``.
        ``next_encoder_hidden`` is the shifted context for the next iter;
        ``predicted_new_tg`` is the single new tubelet group to emit to the
        client.  Math is symmetric across the batch dim so the same function
        serves both the sequential and batched paths.
        """
        b, n_ctxt, _ = encoder_hidden.shape
        t_ctx = self._grid_depth
        window = self._window_tokens()  # grid² — one tubelet group worth of tokens
        t_total = actions.size(1)

        if n_ctxt != t_ctx * window:
            raise ValueError(
                f"encoder_hidden has {n_ctxt} tokens; expected "
                f"T_ctx * grid² = {t_ctx} * {window} = {t_ctx * window}."
            )
        # Per-iter slice: iter k consumes actions[:, k : k + T_ctx].
        end = iter_idx + t_ctx
        if end > t_total:
            raise ValueError(
                f"actions/states trajectory too short for iter {iter_idx}: "
                f"need at least {end} timesteps (iter_idx + T_ctx={t_ctx}), "
                f"got {t_total}.  Client must send T_total >= T_ctx + H - 1 "
                "when requesting AC rollout with horizon H."
            )
        actions_h = actions[:, iter_idx:end].contiguous()
        states_h = states[:, iter_idx:end].contiguous()
        extrinsics_h = extrinsics[:, iter_idx:end].contiguous() if extrinsics is not None else None

        predicted = self.predictor(
            encoder_hidden,
            actions_h,
            states_h,
            extrinsics=extrinsics_h,
        )  # [B, N, D]

        if predicted.shape[:2] != (b, n_ctxt):
            raise RuntimeError(
                f"AC predictor emitted unexpected shape {tuple(predicted.shape)}; "
                f"expected [{b}, {n_ctxt}, D]."
            )

        new_tg = predicted[:, -window:, :]
        next_encoder_hidden = torch.cat([encoder_hidden[:, window:, :], new_tg], dim=1)
        return next_encoder_hidden, new_tg

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

        if encoder_hidden.dim() == 2:
            encoder_hidden = encoder_hidden.unsqueeze(0)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        if states.dim() == 2:
            states = states.unsqueeze(0)
        if extrinsics is not None and extrinsics.dim() == 2:
            extrinsics = extrinsics.unsqueeze(0)

        iter_idx = request_info.dynamic_loop_iter_counts.get("rollout_loop", 0)
        logger.info(
            "VJepa2ACRolloutPredictorSubmodule.forward: iter=%d encoder_hidden=%s actions=%s states=%s",
            iter_idx,
            tuple(encoder_hidden.shape),
            tuple(actions.shape),
            tuple(states.shape),
        )

        next_encoder_hidden, new_tg = self._rollout_step(
            encoder_hidden, actions, states, extrinsics, iter_idx,
        )

        # Per-request early-exit — same contract as masked rollout.
        rollout_horizon = int(request_info.step_metadata.get("rollout_horizon", 0) or 0)
        if rollout_horizon > 0 and iter_idx + 1 >= rollout_horizon:
            logger.info(
                "VJepa2ACRolloutPredictorSubmodule.forward: horizon H=%d reached at iter=%d; stopping rollout_loop.",
                rollout_horizon,
                iter_idx,
            )
            request_info.register_loop_stop("rollout_loop")

        logger.info(
            "VJepa2ACRolloutPredictorSubmodule.forward: next_encoder_hidden=%s new_tg=%s",
            tuple(next_encoder_hidden.shape),
            tuple(new_tg.shape),
        )
        out: NameToTensorList = {
            "encoder_hidden": [next_encoder_hidden],
            "predicted_hidden": [new_tg],
            # Identity loop-back: actions/states don't change across iters
            # (client sent the full trajectory upfront), but they still need
            # to be routed on every iter so the graph dispatcher finds them.
            "actions": [actions],
            "states": [states],
        }
        if extrinsics is not None:
            out["extrinsics"] = [extrinsics]
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
        per_request_info = engine_inputs.per_request_info

        if encoder_hidden.size(0) != len(request_ids):
            raise ValueError(
                f"encoder_hidden batch dim {encoder_hidden.size(0)} does not "
                f"match request count {len(request_ids)}."
            )
        # can_batch guarantees all rids are at the same iter; read any one.
        iter_idx = per_request_info[request_ids[0]].dynamic_loop_iter_counts.get("rollout_loop", 0)

        logger.info(
            "VJepa2ACRolloutPredictorSubmodule.forward_batched: iter=%d enc=%s act=%s rids=%d",
            iter_idx,
            tuple(encoder_hidden.shape),
            tuple(actions.shape),
            len(request_ids),
        )

        next_encoder_hidden, new_tg = self._rollout_step(
            encoder_hidden, actions, states, extrinsics, iter_idx,
        )

        # Per-rid early-exit (individual rids can drop out while others
        # continue; the scheduler re-batches remaining rids on the next iter).
        for rid in request_ids:
            info = per_request_info[rid]
            horizon = int(info.step_metadata.get("rollout_horizon", 0) or 0)
            if horizon > 0 and iter_idx + 1 >= horizon:
                info.register_loop_stop("rollout_loop")

        per_rid: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(request_ids):
            out_i: NameToTensorList = {
                "encoder_hidden": [next_encoder_hidden[i : i + 1]],
                "predicted_hidden": [new_tg[i : i + 1]],
                "actions": [actions[i : i + 1]],
                "states": [states[i : i + 1]],
            }
            if extrinsics is not None:
                out_i["extrinsics"] = [extrinsics[i : i + 1]]
            per_rid[rid] = out_i
        return per_rid


# ---------------------------------------------------------------------------
# P3.B — MPC submodules (intra-request K-way candidate evaluation)
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
# engine batching that P3.A introduced for the single-candidate predictor.
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
