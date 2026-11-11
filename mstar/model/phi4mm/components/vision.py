"""Phi-4-multimodal's image tower and projector (``model.embed_tokens_extend.image_embed``).

A SigLIP so400m tower at 448 px (patch 14, a 32 x 32 patch grid per crop) whose
features are ``hidden_states[-2]`` (``Phi4MMImageEmbedding.get_img_features``): the
output of the first ``num_hidden_layers - 1`` blocks, so the last block, the post
LayerNorm and the pooling head never run and are not loaded. Then a 2x2 average
pool per crop, the HD transform (each image's crops laid out on one canvas and cut
to its valid region, a learned ``sub_GN`` closing every row and ``glb_GN`` between
the crops and the global view, in ``sub_glb`` order), and a 2-layer GELU MLP into
the LLM's width.

Served (``Phi4MMVisionEncoderSubmodule`` in ``phi4mm/vision.py``), a step packs the
valid patches of every crop of every image end to end, one ragged attention segment
per crop: ``embed`` / ``encode`` (the block loop) / ``project``.

**Padded patches follow the reference's default attention, FlashAttention-2**
(``config._attn_implementation``, and the snapshot's ``sample_inference_phi4mm.py``):
a padded patch is neither a key nor a query, so its attention output is zero and it
runs only the MLPs, from the same white pixels at position 0 wherever it sits. Its
feature is one constant (`Phi4MMVisionModel.masked_feature`), which the 2x2 pool
mixes into a kept cell at an odd valid edge. Under eager or SDPA attention a padded
query attends the valid keys instead, and those cells differ (``tests/test_vision_parity.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.linear import QKVParallelLinear
from mstar.model.keys import VISION_ATTN

# the checkpoint's image path; everything under it this module loads or names in
# `unused_weight`
CHECKPOINT_PREFIX = "model.embed_tokens_extend.image_embed."


@dataclass(frozen=True)
class Phi4MMVisionConfig:
    """The tower (``get_siglip_vision_model``, whose sizes the remote code fixes),
    the projector (``config.json``'s ``embd_layer.image_embd_layer``) and the
    processor's crop budget (``preprocessor_config.json``'s ``dynamic_hd``)."""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    patch_size: int = 14
    # the crop side, the tower's image size
    crop_size: int = 448
    layer_norm_eps: float = 1e-6
    # ``hidden_states[layer_index]``: -2 runs every block but the last
    layer_index: int = -2
    # the LLM's width, the projection's output
    text_hidden_size: int = 3072
    # most crops an image is cut into, besides its global view
    max_crops: int = 36

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patches_per_side(self) -> int:
        return self.crop_size // self.patch_size

    @property
    def layers_run(self) -> int:
        """Blocks before the feature tap; ``hidden_states[0]`` is the embedding."""
        return self.num_hidden_layers + 1 + self.layer_index if self.layer_index < 0 else self.layer_index

    @classmethod
    def from_hf(cls, hf: dict, preprocessor: dict | None = None) -> Phi4MMVisionConfig:
        """From ``config.json`` and ``preprocessor_config.json``; refuses an image
        embedding this port does not implement rather than running it wrong."""
        layer = hf["embd_layer"]["image_embd_layer"]
        expected = {
            "use_hd_transform": True, "with_learnable_separator": True, "hd_transform_order": "sub_glb",
            "image_token_compression_cls": "avg_pool_2d", "projection_cls": "mlp",
        }
        wrong = {k: layer.get(k) for k, v in expected.items() if layer.get(k) != v}
        if wrong:
            raise ValueError(
                f"Phi-4-multimodal's image embedding is ported for {expected}; this checkpoint has {wrong}"
            )
        processor = hf.get("img_processor") or {}
        if processor.get("type_feature", "patch") != "patch":
            raise ValueError(f"image features of type {processor['type_feature']!r} are not ported")
        return cls(
            crop_size=layer.get("crop_size", 448),
            layer_index=processor.get("layer_idx", -2),
            text_hidden_size=hf["hidden_size"],
            max_crops=(preprocessor or {}).get("dynamic_hd", cls.max_crops),
        )

    @classmethod
    def from_dir(cls, local_dir: str | Path) -> Phi4MMVisionConfig:
        local_dir = Path(local_dir)
        pre = local_dir / "preprocessor_config.json"
        return cls.from_hf(
            json.loads((local_dir / "config.json").read_text()),
            json.loads(pre.read_text()) if pre.exists() else None,
        )


class Phi4MMVisionAttention(nn.Module):
    """Non-causal attention within each crop the step planned on the ragged
    attention resource; q, k and v in one GEMM (the loader stacks the
    checkpoint's three)."""

    def __init__(self, config: Phi4MMVisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.qkv_proj = QKVParallelLinear(CommGroup.trivial(), dim, self.head_dim, self.num_heads, self.num_heads)
        self.out_proj = nn.Linear(dim, dim)
        self.ragged_attn = None

    def bind_resources(self, resources: dict) -> None:
        """See ``NodeSubmodule.bind_node_resources``."""
        self.ragged_attn = resources.get(VISION_ATTN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ragged_attn is None:
            raise ValueError(
                f"the vision tower has no attention plan: declare a RaggedAttentionSpec under {VISION_ATTN!r}"
            )
        n = x.shape[0]
        q, k, v = (t.view(n, self.num_heads, self.head_dim) for t in self.qkv_proj(x).chunk(3, dim=-1))
        return self.out_proj(self.ragged_attn.run(q, k, v).reshape(n, -1))


class Phi4MMVisionMLP(nn.Module):
    def __init__(self, config: Phi4MMVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Phi4MMVisionEncoderLayer(nn.Module):
    def __init__(self, config: Phi4MMVisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Phi4MMVisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Phi4MMVisionMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))

    def forward_unattended(self, x: torch.Tensor) -> torch.Tensor:
        """A padded patch under FlashAttention-2: its attention output is zero,
        so ``out_proj`` gives its bias."""
        x = x + self.self_attn.out_proj.bias
        return x + self.mlp(self.layer_norm2(x))


class Phi4MMVisionEmbeddings(nn.Module):
    def __init__(self, config: Phi4MMVisionConfig):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            3, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size,
        )
        self.position_embedding = nn.Embedding(config.patches_per_side ** 2, config.hidden_size)


class Phi4MMVisionEncoder(nn.Module):
    def __init__(self, config: Phi4MMVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(Phi4MMVisionEncoderLayer(config) for _ in range(config.layers_run))


class Phi4MMVisionTower(nn.Module):
    """``img_processor``, up to the feature tap."""

    def __init__(self, config: Phi4MMVisionConfig):
        super().__init__()
        self.embeddings = Phi4MMVisionEmbeddings(config)
        self.encoder = Phi4MMVisionEncoder(config)


class Phi4MMVisionModel(nn.Module):
    """Tower + HD transform + projection; parameter names are the checkpoint's
    under `CHECKPOINT_PREFIX`, with q / k / v fused."""

    def __init__(self, config: Phi4MMVisionConfig):
        super().__init__()
        self.config = config
        c = config.hidden_size
        self.img_processor = Phi4MMVisionTower(config)
        self.sub_GN = nn.Parameter(torch.zeros(1, 1, 1, c))
        self.glb_GN = nn.Parameter(torch.zeros(1, 1, c))
        width = config.text_hidden_size
        self.img_projection = nn.Sequential(nn.Linear(c, width), nn.GELU(), nn.Linear(width, width))
        self._masked_feature: torch.Tensor | None = None

    @property
    def head_dim(self) -> int:
        return self.config.head_dim

    def _apply(self, fn, recurse=True):
        # derived from the weights, in their dtype and on their device
        self._masked_feature = None
        return super()._apply(fn, recurse)

    def masked_feature(self) -> torch.Tensor:
        """The feature of every padded patch (``[hidden]``): white pixels at
        position 0 through the blocks with a zero attention output. Built on
        first use, after loading, and again after a move or cast."""
        if self._masked_feature is None:
            emb = self.img_processor.embeddings
            weight = emb.patch_embedding.weight
            p = self.config.patch_size
            with torch.no_grad():
                # the processor pads with 255, which normalizes to 1.0
                white = torch.ones(1, 3, p, p, dtype=weight.dtype, device=weight.device)
                x = emb.patch_embedding(white).flatten(1) + emb.position_embedding.weight[:1]
                for layer in self.img_processor.encoder.layers:
                    x = layer.forward_unattended(x)
            self._masked_feature = x[0]
        return self._masked_feature

    # The served path, split around the block loop for `PackedBlockLoopEncoder`

    def embed(
        self,
        pixel_values: torch.Tensor,
        global_pixel_values: torch.Tensor,
        patch_src: torch.Tensor,
        position_ids: torch.Tensor,
        **_: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Patch embedding of the step's global views (fp32, ``[images, 3, S, S]``)
        and crops (bytes, ``[crops, 3, S, S]``), each crop's valid patches in pack
        order (``patch_src`` indexes the conv's rows, globals first), plus their
        positions. The block loop's cos / sin stay zero: SigLIP has no rotary."""
        emb = self.img_processor.embeddings
        weight = emb.patch_embedding.weight
        # ToTensor + Normalize((0.5,), (0.5,)) in fp32, as the processor's ops round. A
        # tensor divisor: CUDA divides by a host scalar as a multiply by its reciprocal.
        crops = pixel_values.float().div(torch.full((), 255.0, device=pixel_values.device)).sub_(0.5).div_(0.5)
        # cast to the tower's dtype first, as `Phi4MMImageEmbedding.forward` does
        images = torch.cat([global_pixel_values.to(weight.dtype), crops.to(weight.dtype)])
        rows = emb.patch_embedding(images).flatten(2).transpose(1, 2).reshape(-1, weight.shape[0])
        hidden = rows[patch_src] + emb.position_embedding(position_ids)
        rope = hidden.new_zeros(()).expand(hidden.shape[0], self.head_dim)
        return hidden, rope, rope

    def encode(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """The blocks before the feature tap, over the crops the step planned."""
        for layer in self.img_processor.encoder.layers:
            hidden = layer(hidden)
        return hidden

    def project(self, encoded: torch.Tensor, pool_index: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        """Every image's rows in prompt order, ``[rows, text_hidden]``.

        ``pool_index`` (``[rows, 4]``) names each row's 2x2 cell in
        ``AvgPool2d``'s order, over the encoded patches followed by the padded
        patches' feature, ``sub_GN`` and ``glb_GN`` (a separator row names its
        separator four times). The sum runs in fp32 and is rounded once, as
        ``avg_pool2d``'s kernel accumulates."""
        c = encoded.shape[-1]
        table = torch.cat([
            encoded, self.masked_feature().to(encoded.dtype)[None],
            self.sub_GN.view(1, c).to(encoded.dtype), self.glb_GN.view(1, c).to(encoded.dtype),
        ])
        cell = table[pool_index].float()
        rows = (((cell[:, 0] + cell[:, 1]) + cell[:, 2]) + cell[:, 3]).div_(4).to(encoded.dtype)
        return self.img_projection(rows)


# q, k and v land in the fused projection
VISION_STACKED = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
]


def unused_weight(name: str, config: Phi4MMVisionConfig) -> bool:
    """Whether a checkpoint tensor (prefix stripped) is skipped on purpose: the
    blocks past the feature tap, the post LayerNorm and the pooling head, none
    of which ``hidden_states[-2]`` reads."""
    if name.startswith(("img_processor.post_layernorm.", "img_processor.head.")):
        return True
    marker = "img_processor.encoder.layers."
    if name.startswith(marker):
        return int(name[len(marker):].split(".", 1)[0]) >= config.layers_run
    return False


def load_vision_weights(model: Phi4MMVisionModel, checkpoint_dir: str | Path, device: str = "cpu") -> set[str]:
    """Stream the image path's tensors into ``model``; raises on a parameter
    left unfilled or a tensor that names none, other than `unused_weight`'s."""
    from mstar.model.loader import StackedParamRule, iter_safetensors_shards, load_weights_into

    def rename(key: str) -> str | None:
        name = key[len(CHECKPOINT_PREFIX):]
        return None if unused_weight(name, model.config) else name

    loaded = load_weights_into(
        model,
        iter_safetensors_shards(checkpoint_dir, device=device, prefix=CHECKPOINT_PREFIX),
        stacked_params=[StackedParamRule(f".{t}", f".{s}", shard) for t, s, shard in VISION_STACKED],
        name_remapper=rename,
        strict=True,
    )
    model._masked_feature = None
    return loaded
