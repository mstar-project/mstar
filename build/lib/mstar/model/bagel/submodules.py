# ---------------------------------------------------------------------------
# NodeSubmodule wrappers
# ---------------------------------------------------------------------------


import logging
import os
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.base import NodeBatch
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.engine.cuda_graph_config import FlashInferPackedCudaGraphConfig
from mstar.engine.cuda_graph_runner import BasicBatchedCudaGraphConfig
from mstar.engine.kv_store import PositionInfo
from mstar.model.bagel.components.language_model import BagelForCausalLM
from mstar.model.bagel.components.modeling_utils import (
    ImageTransform,
    PositionEmbedding,
    TimestepEmbedder,
    get_flattened_position_ids_extrapolate,
    patchify,
    vllm_vae_resize,
    vllm_vit_resize,
)
from mstar.model.bagel.config import BagelModelConfig
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    StackingMethod,
)
from mstar.utils.sampling import SeenTokenMask

logger = logging.getLogger(__name__)


class ViTEncoderSubmodule(NodeSubmodule):
    """SigLIP2 ViT + connector + vit_pos_embed: pixel patches -> ViT features.

    Receives preprocessed inputs containing packed pixel values, position IDs,
    cumulative sequence lengths, and max sequence length. Both vit_encoder and
    vae_encoder receive "image_inputs" as their graph input name; routing is
    handled by the graph edge's next_node field.
    """

    def __init__(
        self,
        vit_model: nn.Module,
        connector: nn.Module,
        vit_pos_embed: nn.Module,
        vit_patch_size: int,
        vit_max_num_patch_per_side: int,
    ):
        super().__init__()
        self.vit_model = vit_model
        self.connector = connector
        self.vit_pos_embed = vit_pos_embed

        self.vit_patch_size = vit_patch_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.transform = ImageTransform(980, 224, 14)
        self.vae_transform = ImageTransform(1024, 512, 16)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        image_inputs = inputs["image_inputs"]

        image_preprocess = fwd_info.step_metadata.get("image_preprocess", "default")
        if image_preprocess == "vllm":
            # vllm-omni parity: SiglipImageProcessor resizes to a fixed square
            # (size = max_num_patch_per_side * patch_size, e.g. 70*14=980),
            # discarding aspect ratio, then [0.5]/[0.5] normalize.
            square = self.vit_max_num_patch_per_side * self.vit_patch_size
            image_tensor = vllm_vit_resize(image_inputs[0], square)
            image_tensor = self.transform.normalize_transform(image_tensor)
        else:
            image_tensor = self.vae_transform.resize_transform(image_inputs[0])
            image_tensor = self.transform(image_tensor)

        device = image_tensor.device

        position_ids = get_flattened_position_ids_extrapolate(
            image_tensor.size(1), image_tensor.size(2),
            self.vit_patch_size,
            max_num_patches_per_side=self.vit_max_num_patch_per_side,
            device=device,
        )
        pixel_values = patchify(image_tensor, self.vit_patch_size)
        # patchify yields (num_patches, p²·C); pre-patchify image_tensor is
        # (C, H, W), so its .shape[0] is the channel count, not the patch
        # count flashattn / cu_seqlens needs.
        num_tokens = pixel_values.shape[0]

        cu_seqlens = torch.tensor(
            [0, num_tokens], dtype=torch.int32, device=device
        )

        return NodeInputs(
            tensor_inputs={
                "packed_pixel_values": pixel_values,
                "packed_position_ids": position_ids,
            },
            kwargs={
                "cu_seqlens": cu_seqlens,
                "max_seqlen": num_tokens,
                # CPU-side per-image token count, so the batched preprocess
                # can build cu_seqlens without a GPU→CPU sync per request.
                "_num_tokens_per_image_cpu": [num_tokens],
            },
        )

    def forward(
        self,
        engine_inputs: ModelInputsFromEngine,
        packed_pixel_values: torch.Tensor,
        packed_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        **kwargs,
    ) -> NameToTensorList:
        features = self.vit_model(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        features = self.connector(features)
        features = features + self.vit_pos_embed(packed_position_ids)
        return {"img_emb": [features]}

    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        # Opt-in via MSTAR_VIT_BATCHING=1. Off by default because flashattn
        # varlen reductions across packed images produce small bf16 drift,
        # which at greedy temperature=0 can flip downstream LLM argmax.
        if os.environ.get("MSTAR_VIT_BATCHING", "0") != "1":
            return False
        return batch.graph_walk == "prefill_vit" and len(model_inputs) > 1

    def max_batch_size(self, graph_walk: str):
        if graph_walk == "prefill_vit":
            return 32
        return None

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        packed_pixel_values = torch.cat(
            [inp.tensor_inputs["packed_pixel_values"] for inp in inputs], dim=0
        )
        packed_position_ids = torch.cat(
            [inp.tensor_inputs["packed_position_ids"] for inp in inputs], dim=0
        )
        per_image_lens: list[int] = []
        seq_lens: list[int] = []
        for inp in inputs:
            tokens_per_image: list[int] = inp.kwargs["_num_tokens_per_image_cpu"]
            per_image_lens.extend(tokens_per_image)
            seq_lens.append(sum(tokens_per_image))
        cu_lens_cpu = [0]
        for n in per_image_lens:
            cu_lens_cpu.append(cu_lens_cpu[-1] + n)
        cu_seqlens = torch.tensor(
            cu_lens_cpu, dtype=torch.int32, device=packed_pixel_values.device
        )
        return {
            "packed_pixel_values": packed_pixel_values,
            "packed_position_ids": packed_position_ids,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max(per_image_lens) if per_image_lens else 0,
            "seq_lens": seq_lens,
        }

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        packed_pixel_values: torch.Tensor,
        packed_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        seq_lens: list[int],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        features = self.vit_model(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        features = self.connector(features)
        features = features + self.vit_pos_embed(packed_position_ids)

        out: dict[str, NameToTensorList] = {}
        offset = 0
        for rid, n in zip(engine_inputs.request_ids, seq_lens, strict=True):
            out[rid] = {"img_emb": [features[offset:offset + n]]}
            offset += n
        return out


class VAEEncoderSubmodule(NodeSubmodule):
    """VAE encode + patchify + vae2llm + time_embedder + latent_pos_embed.

    Encodes an image tensor to VAE latents, patchifies them, and projects
    into the LLM hidden dimension with positional and timestep embeddings.
    """

    def __init__(
        self,
        vae_model: nn.Module,
        vae2llm: nn.Linear,
        time_embedder: nn.Module,
        latent_pos_embed: nn.Module,
        latent_patch_size: int,
        latent_channel: int,
        latent_downsample: int,
        max_latent_size: int,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.vae2llm = vae2llm
        self.time_embedder = time_embedder
        self.latent_pos_embed = latent_pos_embed
        self.latent_patch_size = latent_patch_size
        self.latent_channel = latent_channel
        self.latent_downsample = latent_downsample
        self.max_latent_size = max_latent_size
        self.transform = ImageTransform(1024, 512, 16)


    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:

        """Convert raw images to VAE encoder input format.

        Computes patchified dimensions as Python ints for CUDA graph
        compatibility (no .item() calls in forward).

        Full implementation should include:
        - Image padding to be divisible by latent_downsample * latent_patch_size
        - VAE position ID computation from latent grid
        - Timestep preparation
        """
        image_inputs = inputs["image_inputs"]

        # [C, H, W]
        image_preprocess = fwd_info.step_metadata.get("image_preprocess", "default")
        img = image_inputs[0].contiguous()
        if image_preprocess == "vllm":
            # vllm-omni parity: _resize_to_stride (aspect-preserving, divisible
            # by latent_downsample, long <= max_latent_size*latent_downsample,
            # short >= 256), then [0.5]/[0.5] normalize.
            image_tensor = vllm_vae_resize(
                img,
                stride=self.latent_downsample,
                max_img_size=self.max_latent_size * self.latent_downsample,
            )
            image_tensor = self.transform.normalize_transform(image_tensor)
        else:
            image_tensor = self.transform(self.transform.resize_transform(img))
        device = image_tensor.device

        # Compute patchified dimensions as ints (CUDA graph compatible)
        p = self.latent_patch_size
        ds = self.latent_downsample
        _, img_h, img_w = image_tensor.shape
        h = (img_h // ds)
        w = (img_w // ds)

        packed_vae_position_ids = get_flattened_position_ids_extrapolate(
            img_h, img_w,
            self.latent_downsample,
            max_num_patches_per_side=self.max_latent_size
        )

        tensor_inputs = {
            "padded_images": image_tensor.unsqueeze(0),
            "packed_vae_position_ids": packed_vae_position_ids,
            "packed_timesteps": torch.tensor([0.0], device=device),
        }
        kwargs = {
            "h": h,
            "w": w,
        }
        return NodeInputs(tensor_inputs=tensor_inputs, kwargs=kwargs)


    def forward(
        self,
        engine_inputs: ModelInputsFromEngine,
        padded_images: torch.Tensor,
        packed_vae_position_ids: torch.Tensor,
        packed_timesteps: torch.Tensor,
        h: int,
        w: int,
        **kwargs,
    ) -> NameToTensorList:
        logger.debug(
            ("Running BAGEL VAE enc with padded_images shape=%s, "
             "packed_vae_position_ids shape=%s, packed_timesteps shape=%s, "
             "h=%d, w=%d"),
            padded_images.shape, packed_vae_position_ids.shape,
            packed_timesteps.shape, h, w
        )

        latent = self.vae_model.encode(padded_images)

        p = self.latent_patch_size
        # h, w are already ints from preprocess (CUDA graph compatible)
        packed_latent = []
        for lat in latent:
            lat = lat[:, :h * p, :w * p].reshape(
                self.latent_channel, h, p, w, p
            )
            lat = torch.einsum("chpwq->hwpqc", lat).reshape(
                -1, p * p * self.latent_channel
            )
            packed_latent.append(lat)
        packed_latent = torch.cat(packed_latent, dim=0)

        # Project to hidden dim with timestep and position embeddings
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        return {"img_emb": [packed_latent]}

def _init_latents_and_time_index(
    config: BagelModelConfig,
    device,
    seed: int,
    H: int,
    W: int,
):

    h, w = (H // config.latent_downsample,
            W // config.latent_downsample)
    num_image_tokens = h * w

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    latents = torch.randn(
        num_image_tokens,
        config.vae_config.z_channels * config.latent_patch_size ** 2,
        generator=g,
        device=device,
    )
    if torch.is_autocast_enabled():
        latents = latents.to(torch.get_autocast_gpu_dtype())
    time_idx = torch.zeros(latents.shape[0], device=device)
    return latents, time_idx


class LLMSubmodule(ARNodeSubmodule):
    """Fat LLM wrapper that dispatches based on graph walk.

    Absorbs text_emb, lm_head, and flow_proj into a single node to avoid
    unnecessary IPC overhead. Graph walk-based dispatch handles:

      - prefill_text: embed_tokens -> LLM forward (causal, mode="und")
      - prefill_vit:  BOI + vit_emb + EOI -> LLM forward (bidirectional)
      - prefill_vae:  BOI + vae_emb + EOI -> LLM forward (bidirectional)
      - decode:       embed_tokens -> LLM forward -> lm_head -> argmax
      - image_gen:    3-pass CFG -> llm2vae -> velocity combine -> Euler step

    BOI/EOI tokens (<|vision_start|>, <|vision_end|>) are structural
    delimiters manually inserted around image embeddings during prefill.
    They are NOT predicted by the model (excluded from CE loss during
    training).

    During image_gen, classifier-free guidance requires 3 LLM forward
    passes with different KV caches (main, cfg_text, cfg_img). The
    velocities are combined via:
        v_final = v_cfg_img + img_scale * (
            v_cfg_text + text_scale * (v_main - v_cfg_text) - v_cfg_img
        )
    followed by an Euler step: x_{t+1} = x_t + v_final * dt.

    Multi-cache orchestration is driven by the requires_cfg flag in
    per-request metadata. When True, graph walk methods manage 3 caches:
      - prefill_text: snapshot main->cfg_text, forward [main, cfg_img]
      - prefill_vit/vae: forward [main], snapshot main->cfg_text
      - decode: forward [main, cfg_img]
      - image_gen: 3-pass CFG with conditional skip and renormalization
    The CacheHandle (provided by KVCacheEngine) manages label switching, page
    allocation, and KV data copying.
    """

    # Node name → cache label mapping for image_gen_cfg
    _NODE_TO_CFG_LABEL = {
        "LLM": "main",
        "LLM_cfg_text": "cfg_text",
        "LLM_cfg_img": "cfg_img",
    }

    def __init__(
        self,
        language_model: BagelForCausalLM,
        llm2vae: nn.Linear,
        vae2llm: nn.Linear,
        time_embedder: TimestepEmbedder,
        latent_pos_embed: PositionEmbedding,
        config: BagelModelConfig,
        boi_token_id: int | None = None,
        eoi_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        node_name: str = "LLM",
    ):
        super().__init__()
        self.node_name = node_name
        self.language_model = language_model
        self.embed_tokens = language_model.model.embed_tokens
        self.lm_head = language_model.lm_head
        self.llm2vae = llm2vae
        self.vae2llm = vae2llm
        self.time_embedder = time_embedder
        self.latent_pos_embed = latent_pos_embed
        self.config = config
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def _get_image_pos_ids(
        self, labels: list[str],
        pos_info: dict[str, PositionInfo],
        device: str,
        seq_len: int,
    ):
        return {
            label: torch.zeros(
                    seq_len, dtype=torch.int32, device=device
                ) + pos_info.get(label, PositionInfo()).position_id_start \
            for label in labels
        }

    def _get_text_vae_idxs(
        self, seq_len: int, device: str
    ):
        tensor_inputs = {}
        tensor_inputs["text_indexes"] = torch.tensor(
            [0, seq_len-1], dtype=torch.long, device=device
        )
        tensor_inputs["vae_token_indexes"] = torch.arange(
            1, seq_len-1,
            dtype=torch.long, device=device
        )
        tensor_inputs["text_mask"] = torch.zeros(
            seq_len, dtype=torch.bool, device=device
        )
        tensor_inputs["text_mask"][0] = True
        tensor_inputs["text_mask"][-1] = True

        return tensor_inputs

    PREFILL_TEXT_TOKEN_BUCKETS = [128, 256, 512, 1024, 2048]
    PREFILL_TEXT_CAPTURE_BATCH_SIZES = [1, 2, 4]

    def _build_prefill_text_packed(
        self, num_tokens: int, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Synthesize a tensor-only post-preprocess packed dict for prefill_text capture.

        BAGEL's ``preprocess`` for prefill_text returns a dict whose only
        capture-relevant tensor is the packed ``input_ids`` (long); the
        rest (``seq_lens``, ``requires_cfg``, ``input_seq_len``) are
        non-tensor entries the runner doesn't intern. ``_forward_prefill_text``
        embeds and forwards inside the captured region.
        """
        return {
            "input_ids": torch.zeros(
                (num_tokens,), dtype=torch.long, device=device,
            ),
        }

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[BasicBatchedCudaGraphConfig | FlashInferPackedCudaGraphConfig]:
        """Declare CUDA graph captures for ``decode`` (cfg-off + cfg-on) and ``prefill_text`` (cfg-off only).

        cfg-on prefill_text is intentionally NOT captured. BAGEL's
        ``preprocess`` for prefill_text+cfg calls
        ``cache_handle.snapshot_all("main", "cfg_text")`` which writes to
        the cache_manager's ``request_ids`` (= ``dummy_rids`` at replay).
        ``cfg_text`` is not in ``config.labels`` (only ``main`` + ``cfg_img``
        get FlashInfer wrappers), so the runner's state-swap doesn't alias
        it onto the real request and the snapshot lands on the dummy slot.
        cfg-on prefill_text continues to use the eager path; downstream
        image_gen / decode+cfg captures are unaffected (they don't depend
        on this capture's snapshot semantics).
        """
        dummy = ARNodeInputs(
            input_ids=torch.zeros(1, dtype=torch.long, device=device),
            input_seq_len=1
        )
        prefill_text_packed = {
            num_tokens: self._build_prefill_text_packed(num_tokens, device)
            for num_tokens in self.PREFILL_TEXT_TOKEN_BUCKETS
        }
        return [
            BasicBatchedCudaGraphConfig(
                capture_graph_walk="decode", requires_cfg=False, labels=["main"],
                single_request_inputs=dummy.clone(),
            ),
            BasicBatchedCudaGraphConfig(
                capture_graph_walk="decode", requires_cfg=True, labels=["main", "cfg_img"],
                single_request_inputs=dummy.clone(),
            ),
            FlashInferPackedCudaGraphConfig(
                capture_graph_walk="prefill_text",
                replay_graph_walks=["prefill_text"],
                packed_seq_len_to_inputs=prefill_text_packed,
                requires_cfg=False,
                labels=["main"],
                compile=True,
                causal_attention=True,
                capture_batch_sizes=self.PREFILL_TEXT_CAPTURE_BATCH_SIZES,
            ),
        ]

    def get_needed_cache_labels(
        self, graph_walk: str, per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str] | None:
        cfg = any([info.requires_cfg for info in per_request_info.values()])
        return self._get_active_labels(graph_walk, cfg)

    def _get_active_labels(
        self, graph_walk: str, cfg: bool
    ):
        if graph_walk in {"prefill_text", "decode"}:
            if cfg:
                return ["main", "cfg_img"]
        elif graph_walk == "image_gen":
            if cfg:
                return ["main", "cfg_text", "cfg_img"]
        elif graph_walk == "image_gen_cfg":
            # Parallel CFG: each LLM node handles one label only
            return [self._NODE_TO_CFG_LABEL.get(self.node_name, "main")]
        return ["main"]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        seen_token_mask: SeenTokenMask,
        pos_info: dict[str, PositionInfo] = {},
        **kwargs
    ) -> ARNodeInputs:

        device = self.get_device()
        node_inputs = ARNodeInputs(input_seq_len=0)

        if graph_walk == "prefill_text":
            node_inputs.input_ids = inputs["text_inputs"][0]
            seen_token_mask.add_tokens(node_inputs.input_ids)
            node_inputs.input_seq_len = node_inputs.input_ids.shape[0]

        elif graph_walk == "decode":
            # NOTE: newly-sampled tokens automatically added to the seen token mask
            node_inputs.input_ids = inputs["text_inputs"][0]
            node_inputs.input_seq_len = 1

        if graph_walk in ["prefill_vit", "prefill_vae"]:
            node_inputs.input_embeds = self._wrap_with_boi_eoi(inputs["img_emb"][0])
            seq_len = node_inputs.input_embeds.shape[0]
            node_inputs.input_seq_len = seq_len

            labels = ["main", "cfg_text", "cfg_img"] # just return all labels since it is cheap

            node_inputs.custom_pos_ids = self._get_image_pos_ids(
                labels, pos_info, device, seq_len
            )

        if graph_walk == "prefill_vae":
            node_inputs.tensor_inputs = self._get_text_vae_idxs(seq_len, device)

        if graph_walk in ("image_gen", "image_gen_cfg"):
            tensor_inputs = {}
            H = fwd_info.step_metadata.get("height", 1024)
            W = fwd_info.step_metadata.get("width", 1024)

            tensor_inputs["vae_position_ids"] = get_flattened_position_ids_extrapolate(
                H, W,
                self.config.latent_downsample,
                max_num_patches_per_side=self.config.max_latent_size
            )
            if "latents" not in inputs or len(inputs["latents"]) == 0:
                node_inputs.input_embeds, tensor_inputs["time_index"] = _init_latents_and_time_index(
                    self.config, device, seed=fwd_info.random_seed, H=H, W=W
                )
            else:
               node_inputs.input_embeds = inputs["latents"][0]
               tensor_inputs["time_index"] = inputs["time_index"][0]

            tensor_inputs["empty_combined_emb"] = self._wrap_with_boi_eoi(
                torch.empty(
                    (node_inputs.input_embeds.shape[0], self.config.hidden_size),
                    dtype=node_inputs.input_embeds.dtype,
                    device=device
                )
            )
            labels = ["main", "cfg_text", "cfg_img"]
            seq_len = tensor_inputs["empty_combined_emb"].shape[0]
            node_inputs.input_seq_len = seq_len
            node_inputs.custom_pos_ids = self._get_image_pos_ids(
                labels, pos_info, device, seq_len
            )
            node_inputs.tensor_inputs = {
                **tensor_inputs,
                **self._get_text_vae_idxs(seq_len, device)
            }

        return node_inputs

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:

        """Data transform + plan attention/rope for all relevant labels.

        When cache_handle is provided (sequential execution), calls
        plan_attention/plan_rope for every cache label needed by this
        graph walk. This must happen outside forward() because plan
        operations are CUDA graph incompatible.

        When cache_handle is None (batched execution preprocesses per-request
        without planning; planning is done separately via preprocess_batched).
        """
        cache_manager = engine_inputs.cache_manager

        requires_cfg = self._batch_get_requires_cfg(
            engine_inputs.per_request_info
        )
        labels = self._get_active_labels(graph_walk, requires_cfg)
        seq_lens = [inp.input_seq_len for inp in inputs]
        per_label_custom_pos_ids = {
            label: [
                inp.custom_pos_ids[label] for inp in inputs \
                    if isinstance(inp.custom_pos_ids, dict) and label in inp.custom_pos_ids
            ] for label in labels
        }

        result = {}


        if graph_walk in ("image_gen", "image_gen_cfg"):
            assert len(inputs) == 1 , "Batching not supported for image gen"

        if graph_walk == "image_gen" and requires_cfg:
            # Batched CFG: plan a single FlashInfer batch across all 3 labels
            # so that image_gen can run one forward pass instead of 3.
            cache_manager.plan_attention_batched_cfg(
                labels=labels,
                seq_lens=seq_lens,
                is_causal=False,
                write_store=False,
            )
            cache_manager.plan_rope_batched_cfg(
                labels=labels,
                seq_lens=seq_lens,
                per_label_pos_ids=per_label_custom_pos_ids,
            )
        else:
            self._plan_for_graph_walk(
                cache_handle=cache_manager,
                seq_lens=seq_lens,
                per_label_custom_pos_ids=per_label_custom_pos_ids,
                is_causal=graph_walk in [
                    "prefill_text", "decode"
                ],
                labels=labels,
                snapshots=[("main", "cfg_text")] if graph_walk == "prefill_text" and requires_cfg else [],
                write_cache=graph_walk not in ("image_gen", "image_gen_cfg")
            )

        # Concatenate lists of tensors into single tensors for each input name
        result = ARNodeInputs.collate(inputs, stacking_method=StackingMethod.CAT)
        result["seq_lens"] = seq_lens
        result["requires_cfg"] =  requires_cfg
        result["sample_token"] = any([
            info.step_metadata.get("sample_prefill_token", True) \
                for info in engine_inputs.per_request_info.values()
        ])
        return result

    def _plan_for_graph_walk(
        self, cache_handle: BatchedCacheManager,
        seq_lens: list[int],
        per_label_custom_pos_ids: dict[str, list[torch.Tensor]] = {},
        is_causal=True,
        labels=["main"],
        snapshots=[],
        write_cache=True
    ) -> None:
        """Plan attention and rope for all cache labels needed by this graph walk."""
        for snap in snapshots:
            cache_handle.snapshot_all(*snap)

        for label in labels:
            pos_ids = per_label_custom_pos_ids.get(label)
            if pos_ids is not None and len(pos_ids) > 0:
                pos_ids = torch.cat(pos_ids)
            else:
                pos_ids = None
            cache_handle.plan_attention(
                seq_lens=seq_lens, is_causal=is_causal, label=label,
                write_store=write_cache
            )
            cache_handle.plan_rope(
                seq_lens=seq_lens, pos_ids=pos_ids, label=label
            )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs
    ) -> NameToTensorList:

        request_info = engine_inputs.single_request_info
        cache_handle = engine_inputs.cache_manager
        kwargs.update(request_info.step_metadata)

        logger.debug("Running BAGEL LLM for graph walk %s", graph_walk)

        if graph_walk == "prefill_text":
            return self._forward_prefill_text(cache_handle=cache_handle, **kwargs)
        elif graph_walk == "prefill_vit":
            return self._forward_prefill_vit(cache_handle=cache_handle, **kwargs)
        elif graph_walk == "prefill_vae":
            return self._forward_prefill_vae(cache_handle=cache_handle, **kwargs)
        elif graph_walk == "decode":
            return self._forward_decode(cache_handle=cache_handle, **kwargs)
        elif graph_walk == "image_gen":
            return self._forward_image_gen(cache_handle=cache_handle, **kwargs)
        elif graph_walk == "image_gen_cfg":
            return self._forward_image_gen_single_branch(cache_handle=cache_handle, **kwargs)
        else:
            raise ValueError(f"Unknown LLM graph walk: {graph_walk!r}")

    def _forward_prefill_text(
        self, input_ids: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs
    ) -> NameToTensorList:
        """embed_tokens -> LLM forward (causal, mode='und') -> KV cache update.

        When requires_cfg is True (image generation mode):
        1. Snapshot main -> cfg_text BEFORE forward (done in preprocess)
        2. Forward for main and cfg_img (both see the text tokens)

        plan_attention/plan_rope are called in preprocess for all needed labels.
        """
        emb = self.embed_tokens(input_ids)
        requires_cfg = kwargs.pop("requires_cfg", False)
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)
        sample_token = kwargs.pop("sample_prefill_token", True)

        if requires_cfg and cache_handle is not None:
            for label in ["main", "cfg_img"]:
                cache_handle.set_active_label(label)
                out = self.language_model(
                    emb, mode="und",
                    cache_handle=cache_handle, **kwargs
                )
                if label == "main":
                    hidden = out
        else:
            if cache_handle is not None:
                cache_handle.set_active_label("main")
            hidden = self.language_model(
                emb, mode="und",
                cache_handle=cache_handle, **kwargs
            )
        if sample_token:
            qo_indptr_buf = cache_handle.get_qo_indptr_buf("main")
            assert qo_indptr_buf is not None
            last_token_indices = (qo_indptr_buf[1:] - 1).long()  # (padded_bs,)
            last_hidden = hidden.index_select(0, last_token_indices)
            logits = self.lm_head(last_hidden)
            return {
                "logits": [logits]
            }
        return {}

    def _forward_prefill_vit(
        self, input_embeds: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs
    ) -> NameToTensorList:
        """Wrap img_emb with BOI/EOI tokens -> LLM forward (bidirectional).

        When requires_cfg is True: forward for main only, then snapshot
        main -> cfg_text (cfg_text = context including this image).

        plan_attention/plan_rope are called in preprocess.
        """
        requires_cfg = kwargs.pop("requires_cfg", False)
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)
        sample_token = kwargs.pop("sample_prefill_token", True)

        cache_handle.set_active_label("main")
        hidden = self.language_model(
            input_embeds, mode="und",
            custom_advance_pos_id=1,
            cache_handle=cache_handle, **kwargs
        )

        if requires_cfg:
            cache_handle.snapshot_all("main", "cfg_text")
        if sample_token:
            qo_indptr_buf = cache_handle.get_qo_indptr_buf("main")
            assert qo_indptr_buf is not None
            last_token_indices = (qo_indptr_buf[1:] - 1).long()  # (padded_bs,)
            last_hidden = hidden.index_select(0, last_token_indices)
            logits = self.lm_head(last_hidden)
            return {
                "logits": [logits]
            }
        return {}

    def _forward_prefill_vae(
        self, input_embeds: torch.Tensor,
        vae_token_indexes: torch.Tensor,
        text_indexes: torch.Tensor,
        text_mask: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs
    ) -> NameToTensorList:
        """VAE image emb -> LLM forward (bidirectional, gen mode).

        When requires_cfg is True: forward for main only, then snapshot
        main -> cfg_text (cfg_text = context including this image).

        plan_attention/plan_rope are called in preprocess.
        """
        requires_cfg = kwargs.pop("requires_cfg", False)
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)

        if cache_handle is not None:
            cache_handle.set_active_label("main")
        self.language_model(
            input_embeds, mode="gen",
            cache_handle=cache_handle,
            vae_token_indexes=vae_token_indexes,
            text_indexes=text_indexes,
            text_mask=text_mask,
            custom_advance_pos_id=1,
            **kwargs
        )

        # NOTE: we will never sample a token from prefill_vae because it is alwaus
        # followed by prefill_vit

        if requires_cfg and cache_handle is not None:
            cache_handle.snapshot_all("main", "cfg_text")
        return {}

    def _forward_decode(
        self, input_ids: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs
    ) -> NameToTensorList:
        """embed_tokens -> LLM forward -> lm_head -> logits.

        Returns logits; token sampling is done by the engine post-forward
        (outside CUDA graph capture).

        When requires_cfg is True: also forward for cfg_img to keep its
        KV cache in sync (cfg_img tracks all text, no images).

        plan_attention/plan_rope are called in preprocess for all needed labels.
        """
        requires_cfg = kwargs.pop("requires_cfg", False)
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)
        kwargs.pop("temperature", None)
        kwargs.pop("top_k", None)
        kwargs.pop("top_p", None)
        emb = self.embed_tokens(input_ids)

        if cache_handle is not None:
            cache_handle.set_active_label("main")
        hidden = self.language_model(
            emb, mode="und",
            cache_handle=cache_handle, **kwargs
        )

        if requires_cfg and cache_handle is not None:
            cache_handle.set_active_label("cfg_img")
            self.language_model(
                emb, mode="und",
                cache_handle=cache_handle, **kwargs
            )

        logits = self.lm_head(hidden[-1:])
        return {
            "logits": [logits],
        }

    @staticmethod
    def _apply_timestep_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
        """Apply BAGEL's non-linear timestep remapping.

        Maps uniform t in [0,1] to shifted t that spends more time
        at higher noise levels (shift > 1).  shift=1 is identity.
        """
        return shift * t / (1 + (shift - 1) * t)

    def _forward_image_gen(
        self,
        input_embeds: torch.Tensor,
        empty_combined_emb: torch.Tensor,
        vae_position_ids: torch.Tensor,
        text_indexes: torch.Tensor,
        vae_token_indexes: torch.Tensor,
        text_mask: torch.Tensor,
        time_index: torch.Tensor,
        cache_handle: BatchedCacheManager,
        requires_cfg: bool = True,
        **kwargs,
    ) -> NameToTensorList:
        """Flow matching Euler step with optional 3-pass CFG.

        Uses cache_handle to switch between the 3 frozen KV caches
        (main, cfg_text, cfg_img). write_cache=False since caches are
        frozen during flow matching.

        When requires_cfg is False, runs a single forward pass (main only)
        without CFG, saving 2/3 of the compute.

        Renormalization modes (cfg_renorm_type in config):
          - "global": single scalar renorm over all dimensions (default)
          - "channel": per-token renorm (independent scale per latent token)
        """
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)

        latents = input_embeds

        # latents, vae_position_ids, time_index and empty_combined_emb are all
        # built with a sequence length derived from the same (H, W), but they
        # arrive as separate inputs. Under torch.compile's dynamic shapes each
        # gets an independent symbolic dim, so Dynamo can't prove the adds below
        # line up. Assert the relationship explicitly.
        n_latent = latents.shape[0]
        torch._check(vae_position_ids.shape[0] == n_latent)
        torch._check(time_index.shape[0] == n_latent)
        torch._check(empty_combined_emb.shape[0] == n_latent + 2)

        N = self.config.num_timesteps
        shift = self.config.timestep_shift

        # Compute shifted timestep and step size for this iteration.
        # time_index goes from 0 to N-2 (N-1 total Euler steps).
        t_uniform = 1.0 - time_index / (N - 1)
        t_uniform_next = 1.0 - (time_index + 1) / (N - 1)
        timestep = self._apply_timestep_shift(t=t_uniform, shift=shift)
        timestep_next = self._apply_timestep_shift(t=t_uniform_next, shift=shift)
        dt = (timestep - timestep_next)[0]  # positive step size

        pos_embed = self.latent_pos_embed(vae_position_ids)
        timestep_embeds = self.time_embedder(timestep)
        latents_ = self.vae2llm(latents) + timestep_embeds \
            + pos_embed

        empty_combined_emb[1:-1] = latents_
        logger.debug(f"packed_seq = {empty_combined_emb}")

        if requires_cfg:
            cfg_text_scale = kwargs.pop("cfg_text_scale", self.config.cfg_text_scale)
            cfg_img_scale = kwargs.pop("cfg_img_scale", self.config.cfg_img_scale)
            renorm_type = kwargs.pop("cfg_renorm_type", self.config.cfg_renorm_type)

            # CFG interval: only apply guidance when timestep is within interval
            cfg_interval = kwargs.pop("cfg_interval", self.config.cfg_interval)
            cfg_lo, cfg_hi = cfg_interval

            # torch.compile compatible logic
            t_val = timestep[0]
            in_cfg_interval = ((t_val > cfg_lo) & (t_val <= cfg_hi)).float()
            effective_text_scale = cfg_text_scale * in_cfg_interval + 1.0 * (1 - in_cfg_interval)
            effective_img_scale = cfg_img_scale * in_cfg_interval + 1.0 * (1 - in_cfg_interval)

            # Batched CFG: run all 3 branches in a single LLM forward.
            # plan_attention_batched_cfg created a single FlashInfer plan
            # treating (main, cfg_text, cfg_img) as 3 "virtual requests".
            S = empty_combined_emb.shape[0]
            batched_emb = empty_combined_emb.repeat(3, 1)  # [3S, hidden]

            # Expand MoT indexes with absolute offsets for each copy
            batched_text_indexes = torch.cat([
                text_indexes + i * S for i in range(3)
            ])
            batched_vae_indexes = torch.cat([
                vae_token_indexes + i * S for i in range(3)
            ])
            batched_text_mask = text_mask.repeat(3)

            cache_handle.set_active_label("_cfg_batched")
            hidden = self.language_model(
                batched_emb, mode="gen",
                cache_handle=cache_handle, write_cache=False,
                vae_token_indexes=batched_vae_indexes,
                text_indexes=batched_text_indexes,
                text_mask=batched_text_mask,
                **kwargs,
            )

            # Split output back to 3 branches and project to VAE space
            v_main = self.llm2vae(hidden[:S])[1:-1]
            v_cfg_text = self.llm2vae(hidden[S:2*S])[1:-1]
            v_cfg_img = self.llm2vae(hidden[2*S:])[1:-1]

            # Two-node CFG velocity combination + renormalization
            cfg_renorm_min = kwargs.pop("cfg_renorm_min", self.config.cfg_renorm_min)

            if renorm_type == "text_channel":
                # text_channel: renorm AFTER text CFG, THEN apply image CFG
                v_text_guided = v_cfg_text + effective_text_scale * (v_main - v_cfg_text)
                norm_v = torch.norm(v_main, dim=-1, keepdim=True)
                norm_v_text = torch.norm(v_text_guided, dim=-1, keepdim=True)
                scale = (norm_v / (norm_v_text + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_text_renormed = v_text_guided * scale
                if effective_img_scale > 1.0:
                    v_final = v_cfg_img + effective_img_scale * (v_text_renormed - v_cfg_img)
                else:
                    v_final = v_text_renormed
            else:
                # global / channel: apply both text+image CFG, THEN renorm
                v_text_guided = v_cfg_text + effective_text_scale * (v_main - v_cfg_text)
                if effective_img_scale > 1.0:
                    v_combined = v_cfg_img + effective_img_scale * (v_text_guided - v_cfg_img)
                else:
                    v_combined = v_text_guided

                if renorm_type == "channel":
                    renorm_scale = (
                        v_main.norm(dim=-1, keepdim=True) /
                        (v_combined.norm(dim=-1, keepdim=True) + 1e-8)
                    ).clamp(min=cfg_renorm_min, max=1.0)
                else:
                    renorm_scale = (
                        v_main.norm() / (v_combined.norm() + 1e-8)
                    ).clamp(min=cfg_renorm_min, max=1.0)
                v_final = v_combined * renorm_scale
        else:
            # No CFG: single forward pass (plan done in preprocess)
            if cache_handle is not None:
                cache_handle.set_active_label("main")
            hidden = self.language_model(
                empty_combined_emb, mode="gen",
                cache_handle=cache_handle, write_cache=False,
                vae_token_indexes=vae_token_indexes,
                text_indexes=text_indexes,
                text_mask=text_mask,
                **kwargs,
            )
            v_final = self.llm2vae(hidden)[1:-1]

        # Euler step: x_{t-dt} = x_t - v * dt  (velocity points data -> noise)
        latents = latents - v_final * dt
        if torch.is_autocast_enabled():
            latents = latents.to(torch.get_autocast_gpu_dtype())
        return {
            "latents": [latents],
            "time_index": [time_index + 1]
        }

    def _forward_image_gen_single_branch(
        self,
        input_embeds: torch.Tensor,
        empty_combined_emb: torch.Tensor,
        vae_position_ids: torch.Tensor,
        text_indexes: torch.Tensor,
        vae_token_indexes: torch.Tensor,
        text_mask: torch.Tensor,
        time_index: torch.Tensor,
        cache_handle: "BatchedCacheManager",
        **kwargs,
    ) -> NameToTensorList:
        """Single-branch LLM forward for parallel CFG (image_gen_cfg walk).

        Each parallel LLM node (LLM, LLM_cfg_text, LLM_cfg_img) runs this
        method with its own cache label. The velocity output is sent to the
        combine_cfg node which applies the CFG formula and Euler step.
        """
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)
        kwargs.pop("requires_cfg", None)
        kwargs.pop("cfg_text_scale", None)
        kwargs.pop("cfg_img_scale", None)
        kwargs.pop("cfg_renorm_type", None)
        kwargs.pop("cfg_interval", None)
        kwargs.pop("cfg_renorm_min", None)

        latents = input_embeds

        # See _forward_image_gen: tie the separately-passed sequence dims so
        # torch.compile's dynamic shapes can prove the adds below line up.
        n_latent = latents.shape[0]
        torch._check(vae_position_ids.shape[0] == n_latent)
        torch._check(time_index.shape[0] == n_latent)
        torch._check(empty_combined_emb.shape[0] == n_latent + 2)

        pos_embed = self.latent_pos_embed(vae_position_ids)

        N = self.config.num_timesteps
        shift = self.config.timestep_shift
        t_uniform = 1.0 - time_index / (N - 1)
        timestep = self._apply_timestep_shift(t=t_uniform, shift=shift)
        timestep_embeds = self.time_embedder(timestep)

        latents_ = self.vae2llm(latents) + timestep_embeds + pos_embed
        empty_combined_emb[1:-1] = latents_

        label = self._NODE_TO_CFG_LABEL.get(self.node_name, "main")
        if cache_handle is not None:
            cache_handle.set_active_label(label)

        hidden = self.language_model(
            empty_combined_emb, mode="gen",
            cache_handle=cache_handle, write_cache=False,
            vae_token_indexes=vae_token_indexes,
            text_indexes=text_indexes,
            text_mask=text_mask,
            **kwargs,
        )

        projected = self.llm2vae(hidden[1:-1])

        # Output velocity. Main LLM also passes through latents/time_index
        # for the combine_cfg node.
        output_name = {
            "LLM": "v_main",
            "LLM_cfg_text": "v_cfg_text",
            "LLM_cfg_img": "v_cfg_img",
        }.get(self.node_name, "v_main")

        result: NameToTensorList = {output_name: [projected]}
        if self.node_name == "LLM":
            result["time_index"] = [time_index]
        return result

    def _batch_get_requires_cfg(self, per_request_info: dict[str, CurrentForwardPassInfo]):
        return any(
            info.requires_cfg
            for info in per_request_info.values()
        )

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None=None,
        input_embeds: torch.Tensor | None=None,
        sample_token: bool=False,
        requires_cfg: bool=False,
        **kwargs
    ) -> dict[str, NameToTensorList]:
        """Batched forward pass for decode and prefill_text.

        Concatenates inputs across requests, runs a single LLM forward with
        the BatchedCacheManager, then splits outputs back per-request.
        """
        request_ids = engine_inputs.request_ids
        cache_manager = engine_inputs.cache_manager

        if graph_walk == "decode":
            return self._forward_decode_batched(
                cache_manager=cache_manager,
                request_ids=request_ids,
                input_ids=input_ids,
                requires_cfg=requires_cfg,
            )
        elif graph_walk == "prefill_text":
            out = self._forward_prefill_text(
                cache_handle=cache_manager,
                input_ids=input_ids,
                requires_cfg=requires_cfg,
                # sample for cuda graph compatibility
                sample_prefill_token=True,
                **kwargs
            )
            if "logits" in out:
                out["__batched_logits__"] = out.pop("logits")[0]
                for i, rid in enumerate(request_ids):
                    out[rid] = {"logits": [out["__batched_logits__"][i]]}
            return out
        elif graph_walk == "prefill_vit":
            out = self._forward_prefill_vit(
                cache_handle=cache_manager,
                input_embeds=input_embeds,
                requires_cfg=requires_cfg,
                sample_prefill_token=sample_token,
                **kwargs
            )
            if "logits" in out:
                out["__batched_logits__"] = out.pop("logits")[0]
                for i, rid in enumerate(request_ids):
                    out[rid] = {"logits": [out["__batched_logits__"][i]]}
            return out
        else:
            raise ValueError(f"Batched forward not supported for graph walk: {graph_walk!r}")

    def _forward_decode_batched(
        self,
        cache_manager: BatchedCacheManager,
        request_ids: list[str],
        input_ids: torch.Tensor,
        requires_cfg: bool = False,
    ) -> dict[str, NameToTensorList]:
        """Batched decode: all requests generate 1 token each.

        1. Concatenate embeddings: [N, hidden] where N = num_requests
        2. Single LLM forward with cache_manager (batched attention)
        3. If any request requires CFG, run a second pass for cfg_img
        4. Per-request lm_head -> logits

        Returns logits per request. Token sampling is done by the engine
        post-forward (outside CUDA graph capture).

        plan_attention/plan_rope are called in preprocess_batched.
        """
        # 1. Embed and concatenate
        embs = self.embed_tokens(input_ids)

        # 2. Single LLM forward (main cache, already planned)
        cache_manager.set_active_label("main")
        hidden = self.language_model(
            embs, mode="und",
            cache_handle=cache_manager,
        )

        # 3. CFG sync pass for cfg_img if needed (already planned)
        if requires_cfg:
            cache_manager.set_active_label("cfg_img")
            self.language_model(
                embs, mode="und",
                cache_handle=cache_manager,
            )

        # 4. Per-request lm_head -> logits (no sampling — done post-forward)
        logits = self.lm_head(hidden)

        # Expose the stacked [B, V] tensor under a sentinel key so the CUDA
        # graph runner can sample directly without concatenating per-rid slices.
        out: dict = {
            rid: {"logits": [logits[i:i+1]]} for i, rid in enumerate(request_ids)
        }
        out["__batched_logits__"] = logits
        return out


    def _wrap_with_boi_eoi(self, emb: torch.Tensor) -> torch.Tensor:
        """Wrap embeddings with <|vision_start|> and <|vision_end|> tokens."""
        assert self.boi_token_id is not None and self.eoi_token_id is not None

        device = emb.device
        boi_ids = torch.tensor([self.boi_token_id], device=device)
        eoi_ids = torch.tensor([self.eoi_token_id], device=device)
        with torch.no_grad():
            boi_emb = self.embed_tokens(boi_ids).to(emb.dtype)
            eoi_emb = self.embed_tokens(eoi_ids).to(emb.dtype)
        return torch.cat([boi_emb, emb, eoi_emb], dim=0)

    def max_batch_size(self, graph_walk: str):
        if graph_walk == "prefill_vit":
            return 2
        return None

    def can_batch(
        self, batch: NodeBatch, model_inputs: list[NodeInputs]
    ):
        return batch.graph_walk in ["decode", "prefill_text", "prefill_vit"]

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        if "new_token" not in outputs:
            return
        if request_info.graph_walk != "decode" and (
            not request_info.step_metadata.get("sample_prefill_token", False)
        ):
            outputs.pop("new_token")
            return
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        token = outputs["new_token"][0].item()
        ignore_eos = request_info.sampling_config["LLM"].ignore_eos
        if (not ignore_eos and self.eos_token_id == token) or \
                (request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 2 >= request_info.max_tokens):
            return {"decode_loop"}
        return set()


class VAEDecoderSubmodule(NodeSubmodule):
    """VAE decoder: latent grid -> pixel image."""

    def __init__(
        self,
        vae_model: nn.Module,
        latent_patch_size: int,
        latent_channel: int,
        latent_downsample: int,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.latent_patch_size = latent_patch_size
        self.latent_channel = latent_channel
        self.latent_downsample = latent_downsample

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        """Prepare VAE decoder inputs.

        Unwraps latents from list. Image dimensions (image_h, image_w)
        are provided via per-request metadata and converted to ints for
        CUDA graph compatibility.
        """
        return NodeInputs(
            tensor_inputs={
                "latents": inputs["latents"][0]
            },
            kwargs={
                "image_h": fwd_info.step_metadata.get("height", 1024),
                "image_w": fwd_info.step_metadata.get("width", 1024)
            }
        )


    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        latents: torch.Tensor,
        image_h: int,
        image_w: int,
        **kwargs,
    ) -> NameToTensorList:
        logger.debug(
            "Running BAGEL VAE dec with latents shape %s, h %d, w %d",
            str(latents.shape), image_h, image_w
        )
        H = image_h
        W = image_w

        p = self.latent_patch_size
        h = H // self.latent_downsample
        w = W // self.latent_downsample

        # Unpatchify: [num_patches, patch_dim] -> [1, C, H_latent, W_latent]
        latent = latents.reshape(1, h, w, p, p, self.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1, self.latent_channel, h * p, w * p
        )
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return {"image_output": [image]}


class CombineCFGSubmodule(NodeSubmodule):
    """Lightweight node: applies CFG formula + Euler step.

    Receives 3 velocity tensors (v_main, v_cfg_text, v_cfg_img) plus
    latents and time_index from the parallel LLM branches. Projects
    velocities to VAE space, applies the 2-node CFG formula with
    renormalization, then performs an Euler step.

    Used in the image_gen_cfg graph walk (parallel CFG architecture).
    Runs on the same GPU as the main LLM branch (enc_dec engine, no KV cache).
    """

    def __init__(
        self,
        llm2vae: nn.Linear,
        config: "BagelModelConfig",
    ):
        super().__init__()
        self.llm2vae = llm2vae
        self.config = config

    @staticmethod
    def _apply_timestep_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * t / (1 + (shift - 1) * t)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        device = self.get_device()

        result = {
            "v_main": inputs["v_main"][0],
            "v_cfg_text": inputs["v_cfg_text"][0],
            "v_cfg_img": inputs["v_cfg_img"][0],
        }
        if "latents" not in inputs or len(inputs["latents"]) == 0:
            H = fwd_info.step_metadata.get("height", 1024)
            W = fwd_info.step_metadata.get("width", 1024)
            result["latents"], result["time_index"] = _init_latents_and_time_index(
                self.config, device=device, seed=fwd_info.random_seed, H=H, W=W
            )
        else:
            result = {
                "latents": inputs["latents"][0],
                "time_index": inputs["time_index"][0],
                **result,
            }
        return NodeInputs(
            tensor_inputs=result,
            kwargs=fwd_info.step_metadata
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        v_main: torch.Tensor,
        v_cfg_text: torch.Tensor,
        v_cfg_img: torch.Tensor,
        latents: torch.Tensor,
        time_index: torch.Tensor,
        cfg_text_scale: float = None,
        cfg_img_scale: float = None,
        cfg_renorm_type: str = None,
        cfg_renorm_min: float = None,
        cfg_interval: tuple = None,
        **kwargs,
    ) -> NameToTensorList:
        if cfg_text_scale is None:
            cfg_text_scale = self.config.cfg_text_scale
        if cfg_img_scale is None:
            cfg_img_scale = self.config.cfg_img_scale
        if cfg_renorm_type is None:
            cfg_renorm_type = self.config.cfg_renorm_type
        if cfg_renorm_min is None:
            cfg_renorm_min = self.config.cfg_renorm_min
        if cfg_interval is None:
            cfg_interval = self.config.cfg_interval

        N = self.config.num_timesteps
        shift = self.config.timestep_shift

        # latents, the three velocity tensors and time_index all share the same
        # (H, W)-derived sequence length but arrive as separate inputs. Tie their
        # symbolic dims so torch.compile can prove the CFG combine / Euler step
        # below line up under dynamic shapes.
        n_latent = latents.shape[0]
        torch._check(v_main.shape[0] == n_latent)
        torch._check(v_cfg_text.shape[0] == n_latent)
        torch._check(v_cfg_img.shape[0] == n_latent)
        torch._check(time_index.shape[0] == n_latent)

        # Compute timestep and step size
        t_uniform = 1.0 - time_index / (N - 1)
        t_uniform_next = 1.0 - (time_index + 1) / (N - 1)
        timestep = self._apply_timestep_shift(t=t_uniform, shift=shift)
        timestep_next = self._apply_timestep_shift(t=t_uniform_next, shift=shift)
        dt = (timestep - timestep_next)[0]

        # Project to VAE space, strip BOI/EOI
        # v_m = self.llm2vae(v_main[1:-1])
        # v_ct = self.llm2vae(v_cfg_text[1:-1])
        # v_ci = self.llm2vae(v_cfg_img[1:-1])
        # Branches now project to VAE space themselves in
        # _forward_image_gen_single_branch (avoids re-projecting the same
        # hidden state on every CFG combine).
        v_m = v_main
        v_ct = v_cfg_text
        v_ci = v_cfg_img

        # CFG interval
        cfg_lo, cfg_hi = cfg_interval
        t_val = timestep[0]
        in_cfg_interval = ((t_val > cfg_lo) & (t_val <= cfg_hi)).float()
        effective_text_scale = cfg_text_scale * in_cfg_interval + 1.0 * (1 - in_cfg_interval)
        effective_img_scale = cfg_img_scale * in_cfg_interval + 1.0 * (1 - in_cfg_interval)

        # CFG formula
        if cfg_renorm_type == "text_channel":
            v_text_guided = v_ct + effective_text_scale * (v_m - v_ct)
            norm_v = torch.norm(v_m, dim=-1, keepdim=True)
            norm_v_text = torch.norm(v_text_guided, dim=-1, keepdim=True)
            scale = (norm_v / (norm_v_text + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
            v_text_renormed = v_text_guided * scale
            if effective_img_scale > 1.0:
                v_final = v_ci + effective_img_scale * (v_text_renormed - v_ci)
            else:
                v_final = v_text_renormed
        else:
            v_text_guided = v_ct + effective_text_scale * (v_m - v_ct)
            if effective_img_scale > 1.0:
                v_combined = v_ci + effective_img_scale * (v_text_guided - v_ci)
            else:
                v_combined = v_text_guided

            if cfg_renorm_type == "channel":
                renorm_scale = (
                    v_m.norm(dim=-1, keepdim=True) /
                    (v_combined.norm(dim=-1, keepdim=True) + 1e-8)
                ).clamp(min=cfg_renorm_min, max=1.0)
            else:
                renorm_scale = (
                    v_m.norm() / (v_combined.norm() + 1e-8)
                ).clamp(min=cfg_renorm_min, max=1.0)
            v_final = v_combined * renorm_scale

        # Euler step
        new_latents = latents - v_final * dt
        if torch.is_autocast_enabled():
            new_latents = new_latents.to(torch.get_autocast_gpu_dtype())

        new_time_index = time_index + 1
        return {
            "latents": [new_latents],
            "time_index": [new_time_index],
        }
