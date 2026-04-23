# ---------------------------------------------------------------------------
# NodeSubmodule wrappers
# ---------------------------------------------------------------------------


from dataclasses import asdict
import logging
from typing import Any

from mminf.engine.kv_store import PositionInfo
from mminf.model.submodule_base import ARNodeInputs, ARNodeSubmodule, ModelInputsFromEngine, NodeInputs
import torch
from torch import nn

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import NodeBatch
from mminf.engine.cache_manager import BatchedCacheManager
from mminf.engine.cuda_graph_runner import CudaGraphConfig
from mminf.model.bagel.components.language_model import BagelForCausalLM
from mminf.model.bagel.components.modeling_utils import (
    ImageTransform,
    PositionEmbedding,
    TimestepEmbedder,
    get_flattened_position_ids_extrapolate,
    patchify,
)
from mminf.model.bagel.config import BagelModelConfig
from mminf.model.submodule_base import NodeSubmodule

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
        """Convert raw images to packed ViT input format.

        Full implementation should include prepare_vit_images logic from BAGEL:
        - Dynamic resolution computation and SigLIP2 image preprocessing
        - Patch splitting and flattening
        - Position ID computation from image grid
        - Packing multiple images with cu_seqlens for FlashAttention
        """

        image_inputs = inputs["image_inputs"]

        image_tensor = self.vae_transform.resize_transform(image_inputs[0])
        image_tensor = self.transform(image_tensor)

        num_tokens = image_tensor.shape[0]
        device = image_tensor.device

        position_ids = get_flattened_position_ids_extrapolate(
            image_tensor.size(1), image_tensor.size(2),
            self.vit_patch_size,
            max_num_patches_per_side=self.vit_max_num_patch_per_side
        ).to(device)
        pixel_values = patchify(image_tensor, self.vit_patch_size)

        # Compute cu_seqlens for FlashAttention
        vit_token_seqlens = torch.tensor(
            [num_tokens], dtype=torch.int32, device=device
        )
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(vit_token_seqlens, dim=0), (1, 0)
        ).to(torch.int32)
        max_seqlen = int(num_tokens)

        tensor_inputs = {
            "packed_pixel_values": pixel_values,
            "packed_position_ids": position_ids,
        }
        kwargs = {
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }
        return NodeInputs(tensor_inputs=tensor_inputs, kwargs=kwargs)

    
    def forward(
        self,
        engine_inputs: ModelInputsFromEngine,
        packed_pixel_values: torch.Tensor,
        packed_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        **kwargs,
    ) -> NameToTensorList:
        logger.debug(
            "Running BAGEL ViT with packed_pixel_values shape=%s, packed_position_ids shape=%s",
            packed_pixel_values.shape, packed_position_ids.shape
        )
        features = self.vit_model(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        features = self.connector(features)
        pos_emb = self.vit_pos_embed(packed_position_ids)
        features = features + pos_emb
        return {"img_emb": [features]}


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
        image_tensor: torch.Tensor = self.transform(self.transform.resize_transform(image_inputs[0].contiguous()))
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
    H: int=1024,
    W: int=1024,
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
    The CacheHandle (provided by AREngine) manages label switching, page
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

    def _preprocess_prefill_text(
        self, text_inputs: torch.Tensor
    ):
        out = text_inputs.new_zeros(text_inputs.shape[0] + 2)
        out[0] = self.bos_token_id
        out[-1] = self.eos_token_id
        out[1:-1] = text_inputs
        return out

    def _get_image_pos_ids(
        self, labels: list[str],
        pos_info: dict[str, PositionInfo],
        device: str,
        seq_len: int,
    ):
        return {
            label: [
                torch.zeros(
                    seq_len, dtype=torch.int32, device=device
                ) + pos_info.get(label, PositionInfo()).position_id_start \
            ] for label in labels
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

    def get_cuda_graph_configs(self, device: torch.device) -> list[CudaGraphConfig]:
        dummy = ARNodeInputs(
            input_ids=torch.zeros(1, dtype=torch.long, device=device),
            input_seq_len=1
        )
        
        return [
            CudaGraphConfig(
                capture_graph_walk="decode", requires_cfg=False, labels=["main"],
                dummy_capture_inputs=dummy.clone(),
            ),
            CudaGraphConfig(
                capture_graph_walk="decode", requires_cfg=True, labels=["main", "cfg_img"],
                dummy_capture_inputs=dummy.clone(),
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
        pos_info: dict[str, PositionInfo] = {},
    ) -> ARNodeInputs:
        
        device = self.device
        node_inputs = ARNodeInputs(input_seq_len=0)

        if graph_walk == "prefill_text":
            node_inputs.input_ids = self._preprocess_prefill_text(inputs["text_inputs"][0])
            node_inputs.input_seq_len = inputs.shape[0]

        elif graph_walk == "decode":
            bos = torch.tensor([self.bos_token_id], device=device)
            node_inputs.input_ids = inputs["text_inputs"][0] if len(inputs["text_inputs"]) > 0 else bos.clone()
            node_inputs.input_seq_len = 1

        if graph_walk in ["prefill_vit", "prefill_vae"]:
            node_inputs.input_embeds = self._wrap_with_boi_eoi(inputs["img_emb"][0])
            seq_len = inputs.shape[0]
            node_inputs.input_seq_len = seq_len

            labels = self._get_active_labels(graph_walk, requires_cfg=True) # just return all labels since it is cheap
            
            node_inputs.custom_pos_ids = self._get_image_pos_ids(
                labels, pos_info, device, seq_len
            )

        if graph_walk == "prefill_vae":
            node_inputs.tensor_inputs = self._get_text_vae_idxs(seq_len, device)

        if graph_walk in ("image_gen", "image_gen_cfg"):
            tensor_inputs = {}
            H, W = 1024, 1024 # TODO: make this configurable?

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
               node_inputs.input_embeds = inputs["latents"]
               tensor_inputs["time_index"] = inputs["time_index"]

            tensor_inputs["empty_combined_emb"] = self._wrap_with_boi_eoi(
                torch.empty(
                    (node_inputs.input_embeds.shape[0], self.config.hidden_size),
                    dtype=node_inputs.input_embeds.dtype,
                    device=device
                )
            )
            seq_len = tensor_inputs["empty_combined_emb"].shape[0]
            node_inputs.input_seq_len = seq_len
            node_inputs.custom_pos_ids = self._get_image_pos_ids(
                labels, pos_info, device, seq_len
            )
            node_inputs.tensor_inputs = {
                **tensor_inputs,
                **self._get_text_vae_idxs(seq_len, device)
            }
        
        return node_inputs #### TODO: "input_embeds" name needs to be matched downstream (not called latents anymore) --- check this
    
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
            label: [inp.custom_pos_ids[label] for inp in inputs if isinstance(inp.custom_pos_ids, dict) and label in inp.custom_pos_ids] for label in labels
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
        result = ARNodeInputs.collate(inputs, stack=True)
        result["seq_lens"] = seq_lens
        result["requires_cfg"] =  requires_cfg

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
            if pos_ids is not None:
                pos_ids = torch.cat(pos_ids)
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
        self, text_inputs: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs
    ) -> NameToTensorList:
        """embed_tokens -> LLM forward (causal, mode='und') -> KV cache update.

        When requires_cfg is True (image generation mode):
        1. Snapshot main -> cfg_text BEFORE forward (done in preprocess)
        2. Forward for main and cfg_img (both see the text tokens)

        plan_attention/plan_rope are called in preprocess for all needed labels.
        """
        emb = self.embed_tokens(text_inputs)
        requires_cfg = kwargs.pop("requires_cfg", False)
        kwargs.pop("cache_labels", None)
        kwargs.pop("snapshot_after", None)
        kwargs.pop("is_prefill", None)

        if requires_cfg and cache_handle is not None:
            for label in ["main", "cfg_img"]:
                cache_handle.set_active_label(label)
                self.language_model(
                    emb, mode="und",
                    cache_handle=cache_handle, **kwargs
                )
        else:
            if cache_handle is not None:
                cache_handle.set_active_label("main")
            self.language_model(
                emb, mode="und",
                cache_handle=cache_handle, **kwargs
            )
        return {}

    def _forward_prefill_vit(
        self, combined_emb: torch.Tensor,
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

        cache_handle.set_active_label("main")
        self.language_model(
            combined_emb, mode="und",
            custom_advance_pos_id=1,
            cache_handle=cache_handle, **kwargs
        )

        if requires_cfg:
            cache_handle.snapshot_all("main", "cfg_text")
        return {}

    def _forward_prefill_vae(
        self, combined_emb: torch.Tensor,
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
            combined_emb, mode="gen",
            cache_handle=cache_handle,
            vae_token_indexes=vae_token_indexes,
            text_indexes=text_indexes,
            text_mask=text_mask,
            custom_advance_pos_id=1,
            **kwargs
        )

        if requires_cfg and cache_handle is not None:
            cache_handle.snapshot_all("main", "cfg_text")
        return {}

    def _forward_decode(
        self, text_inputs: torch.Tensor,
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
        emb = self.embed_tokens(text_inputs)

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
        latents: torch.Tensor,
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
        latents: torch.Tensor,
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

        # Output velocity. Main LLM also passes through latents/time_index
        # for the combine_cfg node.
        output_name = {
            "LLM": "v_main",
            "LLM_cfg_text": "v_cfg_text",
            "LLM_cfg_img": "v_cfg_img",
        }.get(self.node_name, "v_main")

        result: NameToTensorList = {output_name: [hidden]}
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
        packed_inputs: dict[str, torch.Tensor],
    ) -> dict[str, NameToTensorList]:
        """Batched forward pass for decode and prefill_text.

        Concatenates inputs across requests, runs a single LLM forward with
        the BatchedCacheManager, then splits outputs back per-request.
        """
        request_ids = engine_inputs.request_ids
        cache_manager = engine_inputs.cache_manager
        requires_cfg = packed_inputs.get("requires_cfg", True)

        if graph_walk == "decode":
            return self._forward_decode_batched(
                cache_manager=cache_manager,
                request_ids=request_ids,
                packed_inputs=packed_inputs,
                requires_cfg=requires_cfg,
            )
        elif graph_walk == "prefill_text":
            self._forward_prefill_text(
                cache_handle=cache_manager,
                **packed_inputs
            ) # prefill is the same batched and unbatched
            return {rid: [] for rid in request_ids}
        else:
            raise ValueError(f"Batched forward not supported for graph walk: {graph_walk!r}")

    def _forward_decode_batched(
        self,
        cache_manager: BatchedCacheManager,
        request_ids: list[str],
        packed_inputs: dict[str, torch.Tensor],
        requires_cfg: bool = False,
        per_request_info: dict[str, CurrentForwardPassInfo] | None=None
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
        embs = self.embed_tokens(packed_inputs["text_inputs"])

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

    @torch.compiler.disable
    def _batch_get_sampling_param(
        self,
        per_request_info: dict[str, CurrentForwardPassInfo],
        request_ids: list[str],
        key: str,
        default: float | int,
        device,
        dtype: torch.dtype = torch.float32,
    ) -> float | int | torch.Tensor:
        """Extract a sampling param. Returns scalar if uniform, tensor if per-request."""
        values = [
            per_request_info[rid].step_metadata.get(key, default)
            for rid in request_ids
        ]
        if len(set(values)) == 1:
            return values[0]
        return torch.tensor(values, device=device, dtype=dtype)


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

    def can_batch(
        self, batch: NodeBatch, model_inputs: list[NodeInputs]
    ):
        return batch.graph_walk in ["decode", "prefill_text"]

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]
        token = outputs["new_token"][0].item()
        if (self.eos_token_id is not None and self.eos_token_id == token) or \
                (request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1 >= request_info.max_tokens):
            request_info.register_loop_stop("decode_loop")


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
            }
            ## NOTE: we could also add image_h, image_w as kwargs here
        )


    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        latents: torch.Tensor,
        image_h: int | torch.Tensor = 1024, # BAGEL's default image dim
        image_w: int | torch.Tensor = 1024,
        **kwargs,
    ) -> NameToTensorList:
        logger.debug(
            "Running BAGEL VAE dec with latents shape %s, h %d, w %d",
            str(latents.shape), image_h, image_w
        )
        # Convert to int if tensor (CUDA graph compatible when passed as int
        # from metadata; tensor fallback for backwards compatibility)
        H = image_h.item() if isinstance(image_h, torch.Tensor) else int(image_h)
        W = image_w.item() if isinstance(image_w, torch.Tensor) else int(image_w)

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
        device = self.device

        result = {
            "v_main": inputs["v_main"][0],
            "v_cfg_text": inputs["v_cfg_text"][0],
            "v_cfg_img": inputs["v_cfg_img"][0],
        }
        kwargs = fwd_info.step_metadata
        if "latents" not in inputs or len(inputs["latents"]) == 0:
            H, W = 1024, 1024
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
            kwargs=kwargs
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

        # Compute timestep and step size
        t_uniform = 1.0 - time_index / (N - 1)
        t_uniform_next = 1.0 - (time_index + 1) / (N - 1)
        timestep = self._apply_timestep_shift(t=t_uniform, shift=shift)
        timestep_next = self._apply_timestep_shift(t=t_uniform_next, shift=shift)
        dt = (timestep - timestep_next)[0]

        # Project to VAE space, strip BOI/EOI
        v_m = self.llm2vae(v_main)[1:-1]
        v_ct = self.llm2vae(v_cfg_text)[1:-1]
        v_ci = self.llm2vae(v_cfg_img)[1:-1]

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
