"""VJepa2Model: V-JEPA 2 video world model (encoder + predictor).

Architecture (2 nodes):
    video_encoder  (enc_dec) - ViT with 3D tubelet patches + 3D RoPE.
    predictor      (enc_dec) - either masked latent predictor
                               (``predictor_kind="masked"``) or
                               action-conditioned predictor
                               (``predictor_kind="ac"``).

Graph walks (Phase 1 — no Loop, single forward):

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
    Sequential,
    TensorPointerInfo,
)
from mminf.graph.special_destinations import EMIT_TO_CLIENT
from mminf.model.base import ForwardPassArgs, Model, NodeSubmodule
from mminf.model.vjepa2.components.ac_predictor import VisionTransformerPredictorAC
from mminf.model.vjepa2.components.predictor import VJEPA2Predictor
from mminf.model.vjepa2.components.vit_encoder import VJEPA2Encoder
from mminf.model.vjepa2.config import VJepa2ACPredictorConfig, VJepa2Config
from mminf.model.vjepa2.submodules import (
    VJepa2ACPredictorSubmodule,
    VJepa2EncoderSubmodule,
    VJepa2PredictorSubmodule,
)
from mminf.model.vjepa2.weight_loader import (
    download_vjepa2_snapshot,
    load_vjepa2_hf_weights,
)

logger = logging.getLogger(__name__)


# ImageNet normalization constants (match HF ``IMAGENET_DEFAULT_MEAN``/``STD``).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _preprocess_video(
    frames: torch.Tensor,
    crop_size: int,
) -> torch.Tensor:
    """Apply the HF VJEPA2VideoProcessor transform inline.

    Input: ``[T, C, H, W]`` float in ``[0, 1]`` (matches ``Model.load_video``).
    Output: ``[T, C, crop_size, crop_size]``, normalized with ImageNet mean/std.
    """
    if frames.dim() != 4:
        raise ValueError(f"Expected [T,C,H,W] video; got {tuple(frames.shape)}")
    t, c, h, w = frames.shape
    if c != 3:
        raise ValueError(f"Expected 3-channel RGB video; got {c} channels")

    resize_short = int(crop_size * 256 / 224)
    # Resize shortest edge → keep aspect ratio.
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
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

        # Lazily materialized components (populated in get_submodule).
        self.encoder: VJEPA2Encoder | None = None
        self.predictor: VJEPA2Predictor | VisionTransformerPredictorAC | None = None

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(
        self,
        predictor_kind: str,
        ac_predictor_config: dict | None,
    ) -> VJepa2Config:
        if self.skip_weight_loading:
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

    # ------------------------------------------------------------------
    # Model ABC: structure
    # ------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        # V-JEPA 2 has no KV cache (stateless ViT encoder + stateless predictor).
        return []

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            "video_encoder": EngineType.ENC_DEC,
            "predictor": EngineType.ENC_DEC,
        }

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        predictor_inputs: list[str] = ["encoder_hidden"]
        if self.config.predictor_kind == "ac":
            predictor_inputs += ["actions", "states"]
            if self.config.ac_predictor and self.config.ac_predictor.use_extrinsics:
                predictor_inputs.append("extrinsics")
        else:
            # Masked predictor accepts optional context/target masks; when
            # absent, the submodule builds full-coverage defaults.
            predictor_inputs += ["context_mask", "target_mask"]

        prefill_video = Sequential(
            [
                GraphNode(
                    name="video_encoder",
                    input_ids=["video_frames"],
                    outputs=[GraphEdge(next_node="predictor", name="encoder_hidden")],
                ),
                GraphNode(
                    name="predictor",
                    input_ids=predictor_inputs,
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
            input_ids=["video_frames"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="encoder_hidden",
                    output_modality="video",
                    persist=True,
                )
            ],
        )

        return {
            self.PREFILL_VIDEO: prefill_video,
            self.PREFILL_VIDEO_ENCODER_ONLY: prefill_encoder_only,
        }

    # ------------------------------------------------------------------
    # Model ABC: I/O
    # ------------------------------------------------------------------

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
            raw = tensors["video_inputs"][0]
            processed = _preprocess_video(raw.to(torch.float32), crop_size=self.config.crop_size)
            out["video_frames"] = [processed]

        if self.config.predictor_kind == "ac":
            actions = kwargs.get("actions")
            states = kwargs.get("states")
            if actions is None or states is None:
                raise ValueError("V-JEPA 2-AC requires 'actions' and 'states' kwargs (per-timestep tensors).")
            out["actions"] = [torch.as_tensor(actions, dtype=torch.float32)]
            out["states"] = [torch.as_tensor(states, dtype=torch.float32)]
            if self.config.ac_predictor and self.config.ac_predictor.use_extrinsics:
                extrinsics = kwargs.get("extrinsics")
                if extrinsics is None:
                    raise ValueError("use_extrinsics=True but no 'extrinsics' kwarg provided.")
                out["extrinsics"] = [torch.as_tensor(extrinsics, dtype=torch.float32)]

        # Optional custom masks (masked predictor only).
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
        raise ValueError(f"Unsupported modality for V-JEPA 2: {modality!r}")

    # ------------------------------------------------------------------
    # Model ABC: forward pass orchestration
    # ------------------------------------------------------------------

    def _initial_walk(self, model_kwargs: dict | None) -> str:
        if model_kwargs and model_kwargs.get("skip_predictor"):
            return self.PREFILL_VIDEO_ENCODER_ONLY
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
            for name in ("actions", "states", "extrinsics", "context_mask", "target_mask"):
                if name in input_signals:
                    edge = GraphEdge(next_node="predictor", name=name)
                    edge.tensor_info = input_signals[name]
                    inputs.append(edge)

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        new_tokens: dict[str, list[int]],
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

    def get_submodule(self, node_name: str, device: str = "cpu") -> torch.nn.Module | None:
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
            # TODO: AC weight loading (upstream .pt format) — not yet
            # supported.  Users of the AC variant should pass
            # ``skip_weight_loading=True`` and load weights manually, or
            # wait for the follow-up that adds a key-rename loader.
            self.predictor.to_empty(device=device)
            logger.warning(
                "V-JEPA 2-AC predictor instantiated without weights; "
                "load them manually or wait for the upstream-checkpoint "
                "loader follow-up."
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
