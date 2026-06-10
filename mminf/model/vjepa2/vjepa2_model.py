"""VJepa2Model: V-JEPA 2 video world model (encoder + predictor).

Architecture (2 nodes):
    video_encoder  (enc_dec) - ViT with 3D tubelet patches + 3D RoPE.
    predictor      (enc_dec) - either masked latent predictor
                               (``predictor_kind="masked"``) or
                               action-conditioned predictor
                               (``predictor_kind="ac"``).

Graph walks (single forward, no Loop):

    prefill_video              - Sequential([video_encoder, predictor]) →
                                 emits ``predicted_hidden``.
    prefill_video_encoder_only - single ``video_encoder`` node →
                                 emits ``encoder_hidden`` directly
                                 (equivalent to HF ``skip_predictor=True``).

Selected via ``model_kwargs={"skip_predictor": True}`` at request time.

Video preprocessing matches HF ``VJEPA2VideoProcessor`` semantics:
resize-shortest-edge to ``crop_size * 256 / 224``, center-crop to
``crop_size``, normalize with ImageNet mean/std.  Implemented inline in
``process_prompt`` (no external dep).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch import nn

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mminf.engine.base import EngineType
from mminf.engine.kv_store import KVCacheConfig
from mminf.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mminf.graph.special_destinations import EMIT_TO_CLIENT
from mminf.model.base import ForwardPassArgs, Model, TensorAndMetadata
from mminf.model.submodule_base import NodeSubmodule
from mminf.model.vjepa2.components.ac_predictor import VisionTransformerPredictorAC
from mminf.model.vjepa2.components.predictor import VJEPA2Predictor
from mminf.model.vjepa2.components.vit_encoder import VJEPA2Encoder
from mminf.model.vjepa2.config import VJepa2ACPredictorConfig, VJepa2Config
from mminf.model.vjepa2.submodules import (
    VJepa2ACPredictorSubmodule,
    VJepa2ACRolloutPredictorSubmodule,
    VJepa2EncoderSubmodule,
    VJepa2MPCPredictorSubmodule,
    VJepa2MPCScorerSubmodule,
    VJepa2PredictorSubmodule,
    VJepa2RolloutPredictorSubmodule,
)
from mminf.model.vjepa2.weight_loader import (
    download_vjepa2_ac_upstream_pt,
    download_vjepa2_snapshot,
    load_vjepa2_ac_upstream_weights,
    load_vjepa2_hf_weights,
)

logger = logging.getLogger(__name__)


# ImageNet normalization constants (match HF ``IMAGENET_DEFAULT_MEAN``/``STD``).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _sample_frames_uniform(frames: torch.Tensor, target_t: int) -> torch.Tensor:
    """Uniformly sample ``target_t`` frames from a ``[T, ...]`` video.

    Strict parity with HuggingFace
    ``BaseVideoProcessor.sample_frames`` (see
    ``transformers/src/transformers/video_processing_utils.py:205-256``):

        indices = torch.arange(0, total, total / num_frames).int()

    This produces ``target_t`` evenly-spaced integer indices in
    ``[0, total)``.  Matching HF's behaviour exactly, we raise when the
    video is shorter than the requested count rather than padding (upstream
    ``vjepa2/src/datasets/video_dataset.py:loadvideo_decord`` is a
    training-time multi-clip window sampler with different semantics — not
    used at inference).
    """
    total = int(frames.size(0))
    if target_t > total:
        raise ValueError(
            f"Video can't be sampled: num_frames={target_t} exceeds total_num_frames={total}. "
            "Either send a longer clip or pass model_kwargs={'num_frames': <=total}."
        )
    if target_t == total:
        return frames
    step = total / target_t
    indices = torch.arange(0, total, step, device=frames.device).to(torch.int64)
    # arange can overshoot by one element due to float accumulation; trim.
    indices = indices[:target_t]
    return frames[indices]


def _preprocess_video(
    frames: torch.Tensor,
    crop_size: int,
    target_frames: int,
) -> torch.Tensor:
    """Apply the HF VJEPA2VideoProcessor transform inline.

    Input: ``[T, C, H, W]`` float in ``[0, 1]`` (matches ``Model.load_video``).
    Output: ``[target_frames, C, crop_size, crop_size]``, ImageNet-normalized.

    Pipeline: temporal subsample → spatial resize (shortest edge) → center
    crop → normalize.  Sampling first keeps intermediate memory small.
    """
    if frames.dim() != 4:
        raise ValueError(f"Expected [T,C,H,W] video; got {tuple(frames.shape)}")
    _, c, h, w = frames.shape
    if c != 3:
        raise ValueError(f"Expected 3-channel RGB video; got {c} channels")

    # Temporal subsample to the model's pretraining clip length.
    frames = _sample_frames_uniform(frames, target_frames)

    resize_short = int(crop_size * 256 / 224)
    if h < w:
        new_h, new_w = resize_short, int(round(w * resize_short / h))
    else:
        new_h, new_w = int(round(h * resize_short / w)), resize_short
    frames = nn.functional.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # Center crop to crop_size x crop_size.
    top = (new_h - crop_size) // 2
    left = (new_w - crop_size) // 2
    frames = frames[:, :, top : top + crop_size, left : left + crop_size]

    mean = torch.tensor(_IMAGENET_MEAN, device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
    return (frames - mean) / std


class VJepa2Model(Model):
    """V-JEPA 2 model (encoder + optional predictor)."""

    PREFILL_VIDEO = "prefill_video"
    PREFILL_VIDEO_ENCODER_ONLY = "prefill_video_encoder_only"
    PREFILL_VIDEO_ROLLOUT = "prefill_video_rollout"
    # Streaming rollout: same rollout but EMIT_TO_CLIENT lives on the section so
    # each iter's ``predicted_hidden`` is delivered as soon as the iter
    # completes (instead of accumulating until loop completion).  Same
    # ``rollout_predictor`` node + same submodule as the batched walk —
    # only the emit topology differs.  Gated via ``stream_rollout=True``
    # in ``model_kwargs`` (selected in ``_initial_walk``).
    PREFILL_VIDEO_ROLLOUT_STREAMING = "prefill_video_rollout_streaming"
    # K-way action-candidate MPC (AC variant only).
    PREFILL_VIDEO_MPC = "prefill_video_mpc"

    # Rollout loop name — referenced by
    # ``VJepa2RolloutPredictorSubmodule.forward`` via
    # ``request_info.dynamic_loop_iter_counts[...]``.
    ROLLOUT_LOOP_NAME = "rollout_loop"

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        predictor_kind: str = "masked",
        ac_predictor_config: dict | None = None,
        **kwargs,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.skip_weight_loading = skip_weight_loading

        self.config = self._load_config(predictor_kind, ac_predictor_config)
        self._repo_dir: Path | None = None
        self._ac_pt_path: Path | None = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

        # Lazily materialized components (populated in get_submodule).
        self.encoder: VJEPA2Encoder | None = None
        self.predictor: VJEPA2Predictor | VisionTransformerPredictorAC | None = None

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    # Architecture override applied on top of whatever HF ``config.json`` we
    # manage to load.  Subclasses use this to force a known-good architecture
    # when the HF repo doesn't ship a ``config.json`` (e.g. the AC repo,
    # which only hosts the upstream ``.pt`` under ``original/model.pth``).
    # Format: ``{VJepa2Config field: value}`` — applied via ``setattr`` after
    # the normal load path.
    _ARCH_OVERRIDES: dict = {}

    def _load_config(
        self,
        predictor_kind: str,
        ac_predictor_config: dict | None,
    ) -> VJepa2Config:
        # AC has no HF repo (the V-JEPA 2 HF collection doesn't
        # ship an AC checkpoint; we pull ``.pt`` from S3).  Skip the HF config
        # lookup entirely and rely on ``_ARCH_OVERRIDES`` to apply ViT-g dims.
        if self.skip_weight_loading or predictor_kind == "ac":
            config = VJepa2Config()
        else:
            try:
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(
                    repo_id=self.model_path_hf,
                    filename="config.json",
                    cache_dir=self.cache_dir,
                )
                with open(config_path) as f:
                    config = VJepa2Config.from_hf_config(json.load(f))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not load V-JEPA 2 config from HF repo %s (%s); using defaults.",
                    self.model_path_hf,
                    exc,
                )
                config = VJepa2Config()

        # Apply class-level arch overrides (e.g. VJepa2ACModel forces ViT-g
        # dims because the HF AC repo doesn't ship a config.json).  Run this
        # BEFORE building ``ac_predictor`` so the AC sub-config picks up the
        # corrected ``hidden_size`` / ``layer_norm_eps`` / etc.
        for field, value in self._ARCH_OVERRIDES.items():
            setattr(config, field, value)

        config.predictor_kind = predictor_kind
        if predictor_kind == "ac":
            if ac_predictor_config is None:
                ac_predictor_config = {}
            # Defaults align with upstream V-JEPA 2-AC (ViT-g @ 256).
            defaults = {
                "img_size": (config.crop_size, config.crop_size),
                "patch_size": config.patch_size,
                "num_frames": config.frames_per_clip,
                "tubelet_size": config.tubelet_size,
                "embed_dim": config.hidden_size,
                "layer_norm_eps": config.layer_norm_eps,
            }
            defaults.update(ac_predictor_config)
            config.ac_predictor = VJepa2ACPredictorConfig(**defaults)
        return config

    def _ensure_repo(self) -> Path:
        if self._repo_dir is not None:
            return self._repo_dir
        self._repo_dir = download_vjepa2_snapshot(self.model_path_hf, self.cache_dir)
        return self._repo_dir

    def _ensure_ac_pt(self) -> Path:
        """Lazily resolve the upstream V-JEPA 2-AC ``.pt`` path.

        AC weights ship as a single ~11.7 GB ``.pt`` at
        ``{model_path_hf}/original/model.pth`` on HuggingFace (mirror of the
        upstream S3 artifact).  We only download that one file — no point
        pulling the whole repo since there are no converted safetensors for
        the AC variant in HF Transformers.
        """
        if self._ac_pt_path is not None:
            return self._ac_pt_path
        self._ac_pt_path = download_vjepa2_ac_upstream_pt(
            model_path_hf=self.model_path_hf,
            cache_dir=self.cache_dir,
        )
        return self._ac_pt_path

    # ------------------------------------------------------------------
    # Model ABC: structure
    # ------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        # KV cache exists for the AC rollout predictor
        if self.config.predictor_kind == "ac":
            return [KVCacheConfig(
                num_layers=self.config.ac_predictor.depth,
                num_kv_heads=self.config.ac_predictor.num_heads,
                head_dim=self.config.ac_predictor.predictor_embed_dim // self.config.ac_predictor.num_heads,
                max_seq_len=16384, # TODO: actually compute this
                num_qo_heads=self.config.ac_predictor.num_heads,
                nodes=["rollout_predictor"]
            )]
        return [] # otherwise no kv cache

    def get_node_engine_types(self) -> dict[str, EngineType]:
        types: dict[str, EngineType] = {
            "video_encoder": EngineType.STATELESS,
            "predictor": EngineType.STATELESS,
        }
        # Rollout uses a distinct node so the single-pass and rollout walks
        # can coexist without branching inside a submodule.  Both node names
        # resolve to wrappers around the same underlying predictor nn.Module
        # (``VJEPA2Predictor`` for masked, ``VisionTransformerPredictorAC``
        # for AC) — no weight duplication.  The AC variant uses a
        # sliding-window autoregressive rollout, so ``rollout_predictor``
        # is registered for both predictor kinds.
        types["rollout_predictor"] = EngineType.STATELESS
        # MPC nodes advertised only for AC (the masked predictor
        # has no action input, so K-way candidate evaluation makes no sense
        # for it).  ``ac_predictor_mpc`` shares the underlying
        # VisionTransformerPredictorAC nn.Module with ``predictor``.
        if self.config.predictor_kind == "ac":
            types["ac_predictor_mpc"] = EngineType.STATELESS
            types["mpc_scorer"] = EngineType.STATELESS
            types["rollout_predictor"] = EngineType.KV_CACHE
        return types

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        predictor_inputs: list[str] = ["encoder_hidden"]
        if self.config.predictor_kind == "ac":
            predictor_inputs += ["actions", "states"]
            if self.config.ac_predictor and self.config.ac_predictor.use_extrinsics:
                predictor_inputs.append("extrinsics")
        else:
            # Masked predictor: ``context_mask`` / ``target_mask`` are
            # optional (submodule builds full-coverage defaults when
            # absent). If absent, an empty ("signal-only") edge is sent
            predictor_inputs += ["context_mask", "target_mask"]

        prefill_video = Sequential(
            [
                GraphNode(
                    name="video_encoder",
                    input_names=["video_frames"],
                    outputs=[GraphEdge(next_node="predictor", name="encoder_hidden")],
                ),
                GraphNode(
                    name="predictor",
                    input_names=predictor_inputs,
                    outputs=[
                        GraphEdge(
                            next_node=EMIT_TO_CLIENT,
                            name="predicted_hidden",
                            output_modality="video",
                            persist=True,
                        )
                    ],
                ),
            ]
        )

        prefill_encoder_only = GraphNode(
            name="video_encoder",
            input_names=["video_frames"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="encoder_hidden",
                    output_modality="video",
                    persist=True,
                )
            ],
        )

        walks: dict[str, GraphSection] = {
            self.PREFILL_VIDEO: prefill_video,
            self.PREFILL_VIDEO_ENCODER_ONLY: prefill_encoder_only,
        }

        # ----------------------------------------------------------------
        # Autoregressive rollout — masked, AC, and streaming variants.
        # ----------------------------------------------------------------
        # The rollout section is a single ``rollout_predictor`` node that
        # consumes a sliding-window ``encoder_hidden`` and emits both the
        # updated window (loop-back) and the fresh ``predicted_hidden``.
        # ``predicted_hidden`` is declared as a section output purely so
        # ``Loop.__post_init__`` recognizes it when filtering the
        # ``accumulated_outputs`` edge list — its ``next_node`` target is
        # the same node, meaning the loop-back value is ignored by the
        # node's ``input_names`` and only survives in the accumulated cache.
        #
        #
        # Two variants of the walk coexist:
        #   * batched (default): ``Loop.accumulated_outputs`` gathers the
        #     per-iter ``predicted_hidden`` and emits a single
        #     ``result_tensors`` message (with ``len==H`` tensor_infos) when
        #     the loop completes.
        #   * streaming: an ``EMIT_TO_CLIENT`` edge lives
        #     directly on the section — the worker routes section outputs
        #     immediately after each node run, so the client sees one
        #     ``result_tensors`` message per iter as soon as it's
        #     produced.  Matches the per-iter emit pattern used by
        #     ``bagel_model.py``'s ``decode`` loop for tokens
        #     (``new_token -> EMIT_TO_CLIENT``) and by
        #     ``qwen3_omni_model.py``'s ``thinker_decode`` loop.  No
        #     partitions, no ``StreamBuffer``, no ``ChunkPolicy`` needed —
        #     those primitives are for cross-partition streaming (e.g.
        #     Orpheus LLM -> SNAC), which isn't what per-iter client
        #     emit requires.
        # Both walks route to the SAME ``rollout_predictor`` node name
        # (same submodule, same engine type, same ``check_stop``
        # semantics) — only the emit topology differs.  Gated per-request
        # via ``model_kwargs["stream_rollout"]`` in ``_initial_walk``.
        rollout_inputs: list[str] = ["encoder_hidden"]
        rollout_loopback_outputs: list[GraphEdge] = [
            GraphEdge(next_node="rollout_predictor", name="encoder_hidden"),
            GraphEdge(next_node="rollout_predictor", name="predicted_hidden"),
        ]
        if self.config.predictor_kind == "ac":
            # NOTE: in the absence of "actions" and "states" loopback outputs,
            # the look primitives always pass in the initial action and state inputs;
            # no additional wiring needed because the actions and states are constant
            # buffers across loop iterations (we just take different indices at each iter)
            rollout_inputs += ["actions", "states"]
            if self.config.ac_predictor and self.config.ac_predictor.use_extrinsics:
                rollout_inputs.append("extrinsics")

        def _build_rollout_encoder_node() -> GraphNode:
            # Fresh instance per walk — GraphNode carries per-request
            # ``ready_inputs`` state populated during graph execution, so
            # reusing one instance across two walks would entangle them.
            return GraphNode(
                name="video_encoder",
                input_names=["video_frames"],
                outputs=[
                    GraphEdge(next_node="rollout_predictor", name="encoder_hidden"),
                ],
            )

        # -- Batched (masked + AC): accumulated_outputs, one message at
        # -- loop completion.  ``max_iters`` is a config-level upper bound
        # -- baked in at graph-build time; the per-request horizon is
        # -- enforced inside the submodule via ``check_stop`` when
        # -- iter_idx + 1 reaches it.
        rollout_section_batched = GraphNode(
            name="rollout_predictor",
            input_names=rollout_inputs,
            outputs=list(rollout_loopback_outputs),
            # dynamic rollout loop; don't want async scheduling to overshoot by one
            enable_async_scheduling=False
        )
        rollout_loop_batched = Loop(
            name=self.ROLLOUT_LOOP_NAME,
            section=rollout_section_batched,
            max_iters=self.config.max_rollout_horizon,
            outputs=[],
            accumulated_outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="predicted_hidden",
                    output_modality="video",
                    persist=True,
                ),
            ],
        )
        walks[self.PREFILL_VIDEO_ROLLOUT] = Sequential(
            [_build_rollout_encoder_node(), rollout_loop_batched]
        )

        # -- Streaming: EMIT_TO_CLIENT on the section itself,
        # -- one message per iter.  ``persist=False`` (default): each emit
        # -- is one-shot per iter; nothing needs to survive across iters.
        rollout_section_streaming = GraphNode(
            name="rollout_predictor",
            input_names=rollout_inputs,
            outputs=list(rollout_loopback_outputs) + [
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="predicted_hidden",
                    output_modality="video",
                ),
            ],
            # dynamic rollout loop; don't want async scheduling to overshoot by one
            enable_async_scheduling=False
        )
        rollout_loop_streaming = Loop(
            name=self.ROLLOUT_LOOP_NAME,
            section=rollout_section_streaming,
            max_iters=self.config.max_rollout_horizon,
            outputs=[],
            accumulated_outputs=[],
        )
        walks[self.PREFILL_VIDEO_ROLLOUT_STREAMING] = Sequential(
            [_build_rollout_encoder_node(), rollout_loop_streaming]
        )

        # ----------------------------------------------------------------
        # K-way action-candidate MPC (AC variant only)
        # ----------------------------------------------------------------
        # Graph shape:
        #   video_encoder -> ac_predictor_mpc -> mpc_scorer
        # Inputs routed per node:
        #   video_frames -> video_encoder
        #   actions [K,T,7] + states [K,T,7] -> ac_predictor_mpc
        #   goal_hidden [1,N,D] (pre-encoded by the client via a prior
        #       ``prefill_video_encoder_only`` call) -> mpc_scorer
        # The AC predictor's forward is natively batch-dim aware; the MPC
        # submodule ``.expand``s encoder_hidden to [K,N,D] and runs the
        # predictor with [K,T,7] actions/states in ONE forward — matches
        # upstream ``mpc_utils.cem`` lines 62-64 + 114.
        if self.config.predictor_kind == "ac":
            mpc_predictor_inputs: list[str] = ["encoder_hidden", "actions", "states"]
            if self.config.ac_predictor and self.config.ac_predictor.use_extrinsics:
                mpc_predictor_inputs.append("extrinsics")

            prefill_video_mpc = Sequential(
                [
                    GraphNode(
                        name="video_encoder",
                        input_names=["video_frames"],
                        outputs=[
                            GraphEdge(next_node="ac_predictor_mpc", name="encoder_hidden"),
                        ],
                    ),
                    GraphNode(
                        name="ac_predictor_mpc",
                        input_names=mpc_predictor_inputs,
                        outputs=[
                            GraphEdge(next_node="mpc_scorer", name="predicted_hidden"),
                        ],
                    ),
                    GraphNode(
                        name="mpc_scorer",
                        input_names=["predicted_hidden", "goal_hidden"],
                        outputs=[
                            GraphEdge(
                                next_node=EMIT_TO_CLIENT,
                                name="best_index",
                                output_modality="scalar",
                                persist=True,
                            ),
                            GraphEdge(
                                next_node=EMIT_TO_CLIENT,
                                name="costs",
                                output_modality="tensor",
                                persist=True,
                            ),
                            GraphEdge(
                                next_node=EMIT_TO_CLIENT,
                                name="predicted_hidden",
                                output_modality="video",
                                persist=True,
                            ),
                        ],
                    ),
                ]
            )
            walks[self.PREFILL_VIDEO_MPC] = prefill_video_mpc

        return walks

    # ------------------------------------------------------------------
    # Model ABC: I/O
    # ------------------------------------------------------------------

    def load_video(self, filepath: str, device: str) -> TensorAndMetadata:
        """Decode a video file into ``[frames_per_clip, C, H, W]`` float in ``[0, 1]``.

        Mirrors HuggingFace ``BaseVideoProcessor`` Path 2
        (``video_processing_utils.py:294``): read metadata, compute the
        uniform sample indices, and hand those indices to the decoder so
        only the sampled frames are materialized.  For long clips this is
        the difference between decoding 64 frames and decoding thousands
        (= seconds, not minutes).

        Device handling mirrors HF too: we don't pass a device to the
        decoder (HF's decode path doesn't either; device placement happens
        downstream in ``_prepare_input_videos``).  The ``device`` argument
        here is kept for signature-compatibility with ``Model.load_video``
        but is intentionally ignored — decode stays on CPU, later steps
        move tensors to GPU as needed.
        """
        from dataclasses import asdict

        from torchcodec.decoders import VideoDecoder

        target_frames = self.config.frames_per_clip
        logger.info("load_video: opening %s", filepath)
        decoder = VideoDecoder(filepath)
        logger.info("load_video: decoder opened; reading metadata")

        metadata_obj = getattr(decoder, "metadata", None)
        total = None
        if metadata_obj is not None:
            total = getattr(metadata_obj, "num_frames", None)
        if total is None:
            total = len(decoder)
        total = int(total)

        # ``frames_per_clip`` is an architectural config (also feeds
        # ``grid_depth`` for the predictor's attention mask) and must stay
        # at the trained value (64). For AC rollout requests the input
        # video may be much shorter — DROIDDataset sends 8-frame clips per
        # the F-8 workload aligned to upstream's droid-256px-8f.yaml. Clamp
        # the decode target to whatever frames are actually present;
        # process_prompt's AC-rollout branch will trim further to
        # AC_ROLLOUT_NUM_FRAMES (=8). A truly empty video still errors.
        if total <= 0:
            raise ValueError(f"Video has no decodable frames: {filepath}")
        target_frames = min(target_frames, total)
        logger.info("load_video: total_frames=%d, target=%d", total, target_frames)

        # HF-parity uniform sampling (``BaseVideoProcessor.sample_frames``
        # at video_processing_utils.py:253):
        #     indices = torch.arange(0, total, total / num_frames).int()
        # Truncated to ``num_frames`` to handle float-accumulation overshoot.
        step = total / target_frames
        indices = torch.arange(0, total, step).to(torch.int64)[:target_frames].tolist()
        logger.info(
            "load_video: sampling indices=%s...%s (step=%.2f)",
            indices[:3],
            indices[-3:],
            step,
        )

        get_at = getattr(decoder, "get_frames_at", None)
        if get_at is not None:
            logger.info("load_video: calling get_frames_at(%d indices)", len(indices))
            frame_batch = get_at(indices=indices)
            frames = getattr(frame_batch, "data", frame_batch)
        else:
            # Older torchcodec: no batched sampled-decode API, per-index
            # lookup still avoids materializing the whole video.
            logger.info("load_video: per-index lookup (no get_frames_at)")
            frames = torch.stack([decoder[i] for i in indices])
        logger.info("load_video: decoded frames shape=%s", tuple(frames.shape))

        video = frames.float() / 255.0

        try:
            metadata = asdict(metadata_obj) if metadata_obj is not None else {}
        except TypeError:
            metadata = {}
        metadata["sampled_indices"] = indices
        metadata["original_num_frames"] = total
        return TensorAndMetadata(data=video, metadata=metadata)

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Pre-process raw video (and optional AC inputs) for the first forward pass.

        Expects the data worker to have populated ``tensors["video_inputs"]``
        via :meth:`Model.load_video` (returns ``[T, C, H, W]`` float in
        ``[0, 1]``).  We run HF-style resize + center crop + ImageNet
        normalization to produce the ``video_frames`` edge the encoder wants.

        For V-JEPA 2-AC, callers pass per-timestep ``actions``, ``states``
        (and optionally ``extrinsics``) through ``**kwargs``.
        """
        out: NameToTensorList = {}

        if tensors and "video_inputs" in tensors and len(tensors["video_inputs"]) > 0:
            logger.info("process_prompt: preprocessing video")
            raw = tensors["video_inputs"][0]
            # Per-request override of the frame budget (e.g. to experiment
            # with longer clips on larger GPUs); defaults to the model's
            # pretraining frames_per_clip. Clamped to whatever frames are
            # actually present (load_video may have decoded fewer than
            # frames_per_clip if the source video was short — e.g. AC
            # rollout requests with the F-8 / droid-256px-8f workload
            # ship 8-frame clips).
            target_frames = int(kwargs.get("num_frames", self.config.frames_per_clip))
            target_frames = min(target_frames, int(raw.size(0)))
            processed = _preprocess_video(
                raw.to(torch.float32),
                crop_size=self.config.crop_size,
                target_frames=target_frames,
            )
            out["video_frames"] = [processed]
            logger.info("process_prompt: video_frames shape=%s", tuple(processed.shape))

        if self.config.predictor_kind == "ac":
            actions = kwargs.get("actions")
            states = kwargs.get("states")
            if actions is None or states is None:
                raise ValueError("V-JEPA 2-AC requires 'actions' and 'states' kwargs (per-timestep tensors).")
            out["actions"] = [torch.as_tensor(actions, dtype=torch.float32)]
            out["states"] = [torch.as_tensor(states, dtype=torch.float32)]
            logger.info(
                "process_prompt: AC actions shape=%s states shape=%s",
                tuple(out["actions"][0].shape),
                tuple(out["states"][0].shape),
            )
            if self.config.ac_predictor and self.config.ac_predictor.use_extrinsics:
                extrinsics = kwargs.get("extrinsics")
                if extrinsics is None:
                    raise ValueError("use_extrinsics=True but no 'extrinsics' kwarg provided.")
                out["extrinsics"] = [torch.as_tensor(extrinsics, dtype=torch.float32)]

            # AC rollout needs T_total >= T_ctx + H - 1.  Fail fast
            # in process_prompt so the client gets a clear error before any
            # forward pass runs.  Sliced per-iter inside the rollout submodule
            # (see ``VJepa2ACRolloutPredictorSubmodule._rollout_step``).
            rollout_horizon = int(kwargs.get("rollout_horizon", 0) or 0)
            if rollout_horizon > 1:
                t_ctx = 1
                required = t_ctx + rollout_horizon - 1
                act_tensor = out["actions"][0]
                t_total = act_tensor.size(0) if act_tensor.dim() == 2 else act_tensor.size(1)
                if t_total < required:
                    raise ValueError(
                        f"AC rollout with rollout_horizon={rollout_horizon} requires "
                        f"actions/states trajectory length >= T_ctx + H - 1 = "
                        f"{t_ctx} + {rollout_horizon} - 1 = {required}; got {t_total}."
                    )

                # AC rollout encoder context: trim video to AC_ROLLOUT_NUM_FRAMES
                # frames per request to match upstream's published DROID training
                # configuration (configs/train/vitg16/droid-256px-8f.yaml has
                # `dataset_fpcs: 8`).  Each frame is then independently encoded
                # via the self-tubelet replication trick inside
                # VJepa2EncoderSubmodule._encode_self_tubelet — mirroring
                # `forward_target` in app/vjepa_droid/train.py:408-415 and
                # notebooks/energy_landscape_example.ipynb Cell 5.  Only the
                # first frame's tokens are used as the rollout starting context
                # (per notebook Cell 5: z_hat = z[:, :tokens_per_frame]); the
                # other 7 frames' tokens are computed and discarded — matched
                # FLOPs to upstream's reference inference path.
                AC_ROLLOUT_NUM_FRAMES = 8
                if "video_frames" in out:
                    out["video_frames"] = [frames[:AC_ROLLOUT_NUM_FRAMES] for frames in out["video_frames"]]


            # MPC walk requires a pre-encoded goal latent.  When
            # the client flags ``mpc=True`` they must also supply EITHER:
            #   * ``goal_hidden`` — the full tensor, shape [1, N, D] or [N, D],
            #     accepted as python list / numpy array / tensor.  In
            #     production use this path: real goals are per-episode, not
            #     shape-broadcast constants.
            #   * ``goal_hidden_fill`` — a scalar that server-side expands
            #     to ``torch.full((1, N, D), fill)``.  Exists purely to make
            #     smoke-testing viable: the full tensor at ViT-g (1×8192×1408
            #     f32 = ~46 MB raw, ~100 MB as JSON list-of-lists) blows past
            #     Starlette's default ``max_part_size`` of 1 MB per form
            #     field, so the request would 400 before reaching this
            #     handler.  A scalar serializes to <20 bytes and hits the
            #     same scorer math end-to-end.
            if kwargs.get("mpc"):
                goal_hidden = kwargs.get("goal_hidden")
                if goal_hidden is None:
                    fill = kwargs.get("goal_hidden_fill")
                    if fill is None:
                        raise ValueError(
                            "model_kwargs['mpc']=True requires 'goal_hidden' "
                            "(pre-encoded goal latent from a prior "
                            "prefill_video_encoder_only call) or "
                            "'goal_hidden_fill' (scalar, for smoke-test only)."
                        )
                    n_tokens = self.config.grid_depth * self.config.grid_size * self.config.grid_size
                    d = self.config.hidden_size
                    out["goal_hidden"] = [
                        torch.full((1, n_tokens, d), float(fill), dtype=torch.float32)
                    ]
                    logger.info(
                        "process_prompt: MPC goal_hidden_fill=%s expanded to shape=%s",
                        fill,
                        tuple(out["goal_hidden"][0].shape),
                    )
                else:
                    out["goal_hidden"] = [torch.as_tensor(goal_hidden, dtype=torch.float32)]
                    logger.info(
                        "process_prompt: MPC goal_hidden shape=%s",
                        tuple(out["goal_hidden"][0].shape),
                    )

        # Optional user-supplied masks (masked predictor only). the submodule
        # builds full-coverage defaults if notpreent.
        if self.config.predictor_kind != "ac":
            for mask_name in ("context_mask", "target_mask"):
                m = kwargs.get(mask_name)
                if m is not None:
                    out[mask_name] = [torch.as_tensor(m, dtype=torch.long)]

        return out

    def postprocess(self, output: torch.Tensor, modality: str) -> bytes:
        if modality == "video":
            # Predicted hidden tensor is emitted as raw float32 bytes.
            # Clients can reshape via (B, N, hidden_size) — shape is
            # communicated separately via the request metadata.
            return output.detach().to(torch.float32).cpu().numpy().tobytes()
        if modality == "scalar":
            # MPC ``best_index`` — int64, 8 bytes.  Clients decode via
            # ``np.frombuffer(raw, dtype=np.int64)[0]``.
            return output.detach().to(torch.int64).cpu().numpy().tobytes()
        if modality == "tensor":
            # MPC ``costs`` ([K] float32) — raw bytes, clients reshape to
            # ``[K]`` based on the request's candidate count.
            return output.detach().to(torch.float32).cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for V-JEPA 2: {modality!r}")

    # ------------------------------------------------------------------
    # Model ABC: forward pass orchestration
    # ------------------------------------------------------------------

    def _initial_walk(self, model_kwargs: dict | None) -> str:
        if model_kwargs and model_kwargs.get("skip_predictor"):
            return self.PREFILL_VIDEO_ENCODER_ONLY
        # MPC walk (AC only).  Requested via ``mpc=True`` with
        # K-way actions/states and a pre-encoded ``goal_hidden``.  Checked
        # before rollout because AC deployments never hit the rollout walk.
        if model_kwargs and model_kwargs.get("mpc") and self.config.predictor_kind == "ac":
            return self.PREFILL_VIDEO_MPC
        # Rollout walk: triggered by ``rollout_horizon > 1``.  H == 1 is
        # equivalent to a single-pass prefill — save a loop's worth of
        # overhead and route through the normal ``prefill_video`` walk.
        # Available for both masked and AC (sliding-window).
        # Streaming: opt into per-iter client emit via ``stream_rollout=True``;
        # default stays batched so existing clients don't break.
        if model_kwargs:
            horizon = int(model_kwargs.get("rollout_horizon", 0) or 0)
            if horizon > 1:
                if model_kwargs.get("stream_rollout"):
                    return self.PREFILL_VIDEO_ROLLOUT_STREAMING
                return self.PREFILL_VIDEO_ROLLOUT
        return self.PREFILL_VIDEO

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        walk = self._initial_walk(model_kwargs)
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=walk,
            is_prefill=True,
            kwargs=(model_kwargs or {}),
        )

        inputs: list[GraphEdge] = []
        if "video_frames" in input_signals:
            edge = GraphEdge(next_node="video_encoder", name="video_frames")
            edge.tensor_info = input_signals["video_frames"]
            inputs.append(edge)

        if walk == self.PREFILL_VIDEO:
            for name in ("actions", "states", "extrinsics"):
                if name in input_signals:
                    edge = GraphEdge(next_node="predictor", name=name)
                    edge.tensor_info = input_signals[name]
                    inputs.append(edge)
            # Optional masks (masked predictor): When absent, the submodule
            # builds full-coverage defaults
            for name in ("context_mask", "target_mask"):
                edge = GraphEdge(next_node="predictor", name=name)
                edge.tensor_info = input_signals.get(name, [])
                inputs.append(edge)

        if walk in (self.PREFILL_VIDEO_ROLLOUT, self.PREFILL_VIDEO_ROLLOUT_STREAMING) \
                and self.config.predictor_kind == "ac":
            # AC rollout: route per-timestep actions/states (and optional
            # extrinsics) to the rollout node.  The submodule identity-
            # loop-backs them across iters and slices per-iter based on
            # dynamic_loop_iter_counts["rollout_loop"].  Streaming variant
            # routes identically — the only downstream difference is the
            # section's EMIT_TO_CLIENT edge.
            for name in ("actions", "states", "extrinsics"):
                if name in input_signals:
                    edge = GraphEdge(next_node="rollout_predictor", name=name)
                    edge.tensor_info = input_signals[name]
                    inputs.append(edge)

        if walk == self.PREFILL_VIDEO_MPC:
            # actions / states / extrinsics go to the MPC predictor node.
            for name in ("actions", "states", "extrinsics"):
                if name in input_signals:
                    edge = GraphEdge(next_node="ac_predictor_mpc", name=name)
                    edge.tensor_info = input_signals[name]
                    inputs.append(edge)
            # goal_hidden goes directly to the scorer (skips the predictor
            # entirely — scorer compares predicted vs goal outside the AC
            # predictor's forward).
            if "goal_hidden" in input_signals:
                edge = GraphEdge(next_node="mpc_scorer", name="goal_hidden")
                edge.tensor_info = input_signals["goal_hidden"]
                inputs.append(edge)

        step_metadata: dict = {"is_prefill": True}
        if walk in (self.PREFILL_VIDEO_ROLLOUT, self.PREFILL_VIDEO_ROLLOUT_STREAMING):
            # Per-request horizon enforced by ``VJepa2RolloutPredictorSubmodule``
            # via ``check_stop`` once iter_idx + 1 reaches it.  The
            # graph's Loop is always built with ``max_iters=config.max_rollout_horizon``.
            # Streaming variant uses the same horizon logic — the submodule
            # doesn't distinguish walks.
            requested = int((model_kwargs or {}).get("rollout_horizon", 2) or 2)
            step_metadata["rollout_horizon"] = max(1, min(requested, self.config.max_rollout_horizon))

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=step_metadata,
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        # Single-walk Phase-1 flow: after prefill_video (or its encoder-only
        # variant), the request is done.
        return ForwardPassArgs(
            full_metadata=partition_metadata,
            inputs=[],
            unpersist_tensors=[],
            request_done=True,
        )

    # ------------------------------------------------------------------
    # Model ABC: submodule loading
    # ------------------------------------------------------------------

    def get_submodule(self, node_name: str, device: str = "cpu", tp_group=None) -> torch.nn.Module | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device)
        self._submodule_cache[node_name] = submodule
        if submodule is not None:
            logger.info("Loaded V-JEPA 2 submodule for %s", node_name)
        return submodule

    def _create_submodule(self, node_name: str, device: str) -> NodeSubmodule | None:
        if node_name == "video_encoder":
            self._init_encoder(device)
            return VJepa2EncoderSubmodule(self.encoder, self.config)
        if node_name == "predictor":
            self._init_predictor(device)
            if self.config.predictor_kind == "ac":
                return VJepa2ACPredictorSubmodule(self.predictor, self.config)
            return VJepa2PredictorSubmodule(self.predictor, self.config)
        if node_name == "rollout_predictor":
            self._init_predictor(device)
            if self.config.predictor_kind == "ac":
                # Sliding-window AC autoregressive rollout.  Shares the
                # predictor nn.Module with the single-shot ``predictor``
                # node.  Sliding-window by 1 tubelet group per iter;
                # diverges from upstream growing-context for encoder-shape
                # reasons.
                return VJepa2ACRolloutPredictorSubmodule(
                    self.predictor,
                    self.config,
                )
            return VJepa2RolloutPredictorSubmodule(
                self.predictor,
                self.config,
                num_output_frames=self.config.rollout_num_output_frames,
                frames_per_second=self.config.rollout_frames_per_second,
                anticipation_seconds=self.config.rollout_anticipation_seconds,
            )
        # MPC predictor + scorer (AC only).  Predictor shares
        # the VisionTransformerPredictorAC nn.Module with ``predictor``.
        if node_name == "ac_predictor_mpc":
            if self.config.predictor_kind != "ac":
                raise NotImplementedError(
                    "ac_predictor_mpc is only available with predictor_kind='ac'."
                )
            self._init_predictor(device)
            return VJepa2MPCPredictorSubmodule(self.predictor, self.config)
        if node_name == "mpc_scorer":
            if self.config.predictor_kind != "ac":
                raise NotImplementedError(
                    "mpc_scorer is only available with predictor_kind='ac'."
                )
            return VJepa2MPCScorerSubmodule(self.config)
        return None

    def _init_encoder(self, device: str) -> None:
        if self.encoder is not None:
            return
        meta = torch.device("meta" if not self.skip_weight_loading else "cpu")
        with meta:
            self.encoder = VJEPA2Encoder(self.config)
        if self.skip_weight_loading:
            self.encoder = self.encoder.to_empty(device=device)
            return
        self.encoder.to_empty(device=device)

        # V-JEPA 2-AC: encoder + predictor come from the same upstream .pt
        # (different key layouts for each).  Loader is idempotent — calling
        # it with one module at a time just reads and discards the other
        # half of the blob, which is cheap once the OS page-cache is warm
        # after the first call on the rank.
        if self.config.predictor_kind == "ac":
            pt_path = self._ensure_ac_pt()
            load_vjepa2_ac_upstream_weights(
                pt_path=pt_path,
                encoder_module=self.encoder,
                predictor_module=None,
                device=device,
                hidden_size=self.config.hidden_size,
            )
            return

        repo_dir = self._ensure_repo()
        load_vjepa2_hf_weights(
            repo_dir=repo_dir,
            encoder_module=self.encoder,
            predictor_module=None,
            device=device,
        )

    def _init_predictor(self, device: str) -> None:
        if self.predictor is not None:
            return
        meta = torch.device("meta" if not self.skip_weight_loading else "cpu")
        if self.config.predictor_kind == "ac":
            assert self.config.ac_predictor is not None
            with meta:
                predictor = VisionTransformerPredictorAC(self.config.ac_predictor)
            self.predictor = predictor
            if self.skip_weight_loading:
                self.predictor = self.predictor.to_empty(device=device)
                return
            self.predictor.to_empty(device=device)
            pt_path = self._ensure_ac_pt()
            load_vjepa2_ac_upstream_weights(
                pt_path=pt_path,
                encoder_module=None,
                predictor_module=self.predictor,
                device=device,
            )
            return

        with meta:
            self.predictor = VJEPA2Predictor(self.config)
        if self.skip_weight_loading:
            self.predictor = self.predictor.to_empty(device=device)
            return
        self.predictor.to_empty(device=device)
        repo_dir = self._ensure_repo()
        load_vjepa2_hf_weights(
            repo_dir=repo_dir,
            encoder_module=None,
            predictor_module=self.predictor,
            device=device,
        )


class VJepa2ACModel(VJepa2Model):
    """V-JEPA 2 action-conditioned variant.

    Thin subclass that hard-codes ``predictor_kind="ac"`` so the serving
    entrypoint — which only forwards ``model_path_hf`` to the constructor
    (see ``api_server/entrypoint.py``) — can select the AC path purely via
    the registry.  Use ``configs/vjepa2_ac.yaml`` with ``model: "vjepa2_ac"``.

    The AC checkpoint ships only as a ViT-g backbone (no vitl-ac / vith-ac
    variants exist on HF), and the HF AC repo does NOT ship an HF-style
    ``config.json`` — it hosts only the upstream ``.pt`` under
    ``original/model.pth``.  To keep the model loadable without a config
    file, we hardcode the ViT-g architecture via ``_ARCH_OVERRIDES``, which
    the base class applies in :meth:`_load_config`.
    """

    # ViT-g @ 256 with AC predictor.  Matches
    # ``vjepa2/src/models/vision_transformer.py::vit_giant_xformers`` +
    # ``transformers/.../convert_vjepa2_to_hf.py::get_vjepa2_config("vit_giant")``.
    _ARCH_OVERRIDES = {
        "hidden_size": 1408,
        "num_hidden_layers": 40,
        "num_attention_heads": 22,
        "mlp_ratio": 48 / 11,
    }

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        ac_predictor_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            model_path_hf=model_path_hf,
            cache_dir=cache_dir,
            skip_weight_loading=skip_weight_loading,
            predictor_kind="ac",
            ac_predictor_config=ac_predictor_config,
            **kwargs,
        )
