"""Cosmos3Model: NVIDIA Cosmos3 omni generator on the mstar engine.

Cosmos3 is a text-conditioned diffusion model: a dual-pathway Mixture-of-
Transformers DiT denoises image/video (and optionally sound) latents, which a
Wan VAE decodes to pixels. An optional action head extends the same backbone to
robot-action generation.

Nodes (2 for image generation):
    dit          (kv_cache)  - dual-pathway DiT. The understanding (text)
                               tower prefills the conditioning K/V; the
                               generation tower runs the denoise loop, reading
                               that frozen K/V each step (it is timestep-
                               independent, so caching it once is exact).
    vae_decoder  (stateless) - Wan VAE: final latents -> pixels.

Graph walks (image generation):
    prefill    - the understanding tower runs over the text prompt and writes
                 its per-layer K/V (causal self-attention over text).
    image_gen  - an N-step denoising loop. Each iteration the generation tower
                 attends to [frozen text K/V | current generation tokens],
                 predicts flow velocity, and applies one scheduler step; the
                 final latents go to the VAE decoder, which emits the image.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.base import EngineType
from mstar.engine.kv_store import KVCacheConfig
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.submodules import (
    Cosmos3DiTSubmodule,
    Cosmos3VAEDecoderSubmodule,
)

logger = logging.getLogger(__name__)

DIT_NODE = "dit"
VAE_DECODER_NODE = "vae_decoder"


class Cosmos3Model(Model):
    """NVIDIA Cosmos3 generator implementation."""

    PREFILL_WALK = "prefill"
    IMAGE_GEN_WALK = "image_gen"
    ACTION_GEN_WALK = "action_gen"

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        **kwargs,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.skip_weight_loading = skip_weight_loading
        self._yaml_config_overrides: dict = dict(kwargs)

        self._repo_dir: Path | None = None
        self.config: Cosmos3Config = self._load_config()
        self.tokenizer = self._load_tokenizer()

        self._submodule_cache: dict[str, torch.nn.Module | None] = {}

    # ------------------------------------------------------------------
    # Config + tokenizer
    # ------------------------------------------------------------------

    def _ensure_repo(self) -> Path:
        if self._repo_dir is not None:
            return self._repo_dir
        candidate = Path(self.model_path_hf)
        if candidate.exists():
            self._repo_dir = candidate
        else:
            from huggingface_hub import snapshot_download

            self._repo_dir = Path(
                snapshot_download(repo_id=self.model_path_hf, cache_dir=self.cache_dir)
            )
        return self._repo_dir

    def _load_config(self) -> Cosmos3Config:
        if self.skip_weight_loading:
            cfg = Cosmos3Config()
        else:
            try:
                cfg = Cosmos3Config.from_pretrained(self._ensure_repo())
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not load Cosmos3 config from %s (%s); using Nano defaults.",
                    self.model_path_hf, exc,
                )
                cfg = Cosmos3Config()

        # Overlay yaml model_kwargs last (so they win over file + defaults).
        if self._yaml_config_overrides:
            valid = {f.name for f in Cosmos3Config.__dataclass_fields__.values()}
            for k, v in self._yaml_config_overrides.items():
                if k in valid:
                    setattr(cfg, k, v)
                else:
                    logger.warning(
                        "Cosmos3Model: yaml model_kwargs key %r is not a Cosmos3Config "
                        "field; ignored.", k,
                    )
        return cfg

    def _load_tokenizer(self):
        if self.skip_weight_loading:
            return None
        from transformers import AutoTokenizer

        repo = self._ensure_repo()
        # The published checkpoint ships the Qwen2 text tokenizer under
        # ``text_tokenizer/``; fall back to the repo root for layouts that
        # keep the tokenizer files at the top level.
        for sub in (repo / "text_tokenizer", repo):
            try:
                return AutoTokenizer.from_pretrained(str(sub), use_fast=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cosmos3 tokenizer load from %s failed (%s).", sub, exc)
        logger.warning("All Cosmos3 tokenizer sources failed; proceeding without one.")
        return None

    # ------------------------------------------------------------------
    # Model ABC: structure
    # ------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        return [
            KVCacheConfig(
                num_layers=self.config.num_hidden_layers,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                max_seq_len=self.config.max_position_embeddings,
                num_qo_heads=self.config.num_attention_heads,
            )
        ]

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            DIT_NODE: EngineType.KV_CACHE,
            VAE_DECODER_NODE: EngineType.STATELESS,
        }

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # prefill: the understanding tower runs over the text prompt and writes
        # its conditioning K/V. No graph output — completion notifies the
        # conductor, and the generation loop reads the K/V from the shared cache.
        prefill = GraphNode(
            name=DIT_NODE,
            input_names=["text_inputs"],
            outputs=[],
        )

        # image_gen: denoising loop -> VAE decode -> emit image. The loop body
        # threads the latents + denoise-step index back to itself each
        # iteration; on the final iteration the latents route to the decoder.
        # max_iters is the number of denoise model evaluations and is
        # reconciled with the scheduler timestep schedule when the step is wired.
        image_gen = Sequential(
            [
                Loop(
                    section=GraphNode(
                        name=DIT_NODE,
                        input_names=["latents", "time_index"],
                        outputs=[
                            GraphEdge(next_node=DIT_NODE, name="latents"),
                            GraphEdge(next_node=DIT_NODE, name="time_index"),
                        ],
                    ),
                    max_iters=self.config.num_inference_steps,
                    outputs=[
                        GraphEdge(next_node=VAE_DECODER_NODE, name="latents"),
                    ],
                ),
                GraphNode(
                    name=VAE_DECODER_NODE,
                    input_names=["latents"],
                    outputs=[
                        GraphEdge(
                            next_node=EMIT_TO_CLIENT,
                            name="image_output",
                            output_modality="image",
                        ),
                    ],
                ),
            ]
        )

        # action_gen: like image_gen but the loop body jointly denoises the video
        # and action latents (threaded as two self-edges), and the predicted
        # action — not a decoded video — is what the request emits.
        action_gen = Sequential(
            [
                Loop(
                    section=GraphNode(
                        name=DIT_NODE,
                        input_names=["latents", "action_latents", "time_index"],
                        outputs=[
                            GraphEdge(next_node=DIT_NODE, name="latents"),
                            GraphEdge(next_node=DIT_NODE, name="action_latents"),
                            GraphEdge(next_node=DIT_NODE, name="time_index"),
                        ],
                    ),
                    max_iters=self.config.num_inference_steps,
                    outputs=[
                        GraphEdge(
                            next_node=EMIT_TO_CLIENT,
                            name="action_output",
                            output_modality="action",
                        ),
                    ],
                ),
            ]
        )

        return {
            self.PREFILL_WALK: prefill,
            self.IMAGE_GEN_WALK: image_gen,
            self.ACTION_GEN_WALK: action_gen,
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
        if prompt is None:
            return {}
        if self.tokenizer is None:
            # Tokenizer-less fallback used by structural unit tests.
            return {
                "text_inputs": [
                    torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long)
                ]
            }
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return {"text_inputs": [torch.tensor(ids, dtype=torch.long)]}

    def postprocess(self, output: torch.Tensor, modality: str) -> bytes:
        if modality == "image":
            import io

            from PIL import Image

            # output: [C, H, W] (or [1, C, H, W]) in [0, 1].
            frame = output[0] if output.ndim == 4 else output
            arr = (frame.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            return buf.getvalue()
        if modality == "action":
            return output.detach().to(torch.float32).cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for Cosmos3: {modality!r}")

    # ------------------------------------------------------------------
    # Model ABC: forward pass orchestration
    # ------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=self.PREFILL_WALK,
            is_prefill=True,
            kwargs={},
        )

        inputs: list[GraphEdge] = []
        if "text_inputs" in input_signals:
            edge = GraphEdge(next_node=DIT_NODE, name="text_inputs")
            edge.tensor_info = input_signals["text_inputs"]
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
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        request_done = False
        inputs: list[GraphEdge] = []

        is_action = "action" in metadata.output_modalities
        if metadata.graph_walk == self.PREFILL_WALK:
            metadata.is_prefill = False
            metadata.graph_walk = self.ACTION_GEN_WALK if is_action else self.IMAGE_GEN_WALK
            # The first denoise iteration's initial noise + step index are
            # sampled inside the DiT submodule's preprocess. Action requests also
            # thread the action latents through the loop.
            inputs = [
                GraphEdge(next_node=DIT_NODE, name="latents"),
                GraphEdge(next_node=DIT_NODE, name="time_index"),
            ]
            if is_action:
                inputs.insert(1, GraphEdge(next_node=DIT_NODE, name="action_latents"))
        elif metadata.graph_walk in (self.IMAGE_GEN_WALK, self.ACTION_GEN_WALK):
            request_done = True

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": metadata.is_prefill},
            request_done=request_done,
        )

    # ------------------------------------------------------------------
    # Model ABC: submodule loading
    # ------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
    ) -> torch.nn.Module | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device)
        self._submodule_cache[node_name] = submodule
        if submodule is not None:
            logger.info("Loaded Cosmos3 submodule for %s", node_name)
        return submodule

    def _create_submodule(self, node_name: str, device: str):
        if node_name == DIT_NODE:
            return Cosmos3DiTSubmodule(
                transformer=self._build_transformer(device),
                config=self.config,
                scheduler=self._build_scheduler(),
            )
        if node_name == VAE_DECODER_NODE:
            return Cosmos3VAEDecoderSubmodule(
                vae=self._build_vae(device), config=self.config
            )
        return None

    def _build_scheduler(self):
        if self.skip_weight_loading:
            return None
        from diffusers import UniPCMultistepScheduler

        return UniPCMultistepScheduler.from_pretrained(str(self._ensure_repo() / "scheduler"))

    def _build_transformer(self, device: str):
        from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
        from mstar.model.cosmos3.loader import load_transformer_weights

        # Build on the meta device (shapes only, no storage), pin the
        # checkpoint's bf16 dtype, then materialize uninitialized tensors on the
        # target device and overwrite with the checkpoint weights — the same
        # path the other model packages use. bf16 matches the published
        # checkpoint exactly and halves resident weight memory vs the float32
        # meta default; the engine additionally runs the forward under a bf16
        # autocast (a no-op here).
        with torch.device("meta" if not self.skip_weight_loading else "cpu"):
            model = Cosmos3OmniTransformer(self.config)
        model = model.to(torch.bfloat16)
        if self.skip_weight_loading:
            return model.to_empty(device=device)

        model.to_empty(device=device)
        load_transformer_weights(model, self._ensure_repo(), device=device)
        # Keep the timestep embedder in fp32, like diffusers'
        # ``_keep_in_fp32_modules=["time_embedder"]`` (the upcast is lossless from
        # the bf16 checkpoint and matches diffusers' numerics).
        model.time_embedder.to(torch.float32)
        model.eval()
        return model

    def _build_vae(self, device: str):
        if self.skip_weight_loading:
            return None
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(str(self._ensure_repo() / "vae"))
        return vae.to(device).eval()
