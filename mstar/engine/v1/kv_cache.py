

from dataclasses import dataclass
from enum import Enum

import torch

from mstar.engine.resources.spec import NodeResourceSpec, ResourceType


class KVLayout(Enum):
    NHD = "NHD"
    # TODO: can add more, like HND, MLA

@dataclass
class KVConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    max_num_pages: int = 2048
    page_size: int = 128
    layout: KVLayout=KVLayout.NHD

@dataclass
class KVSpec(NodeResourceSpec):
    config: KVConfig

    @property
    def resource_type(self):
        return ResourceType.KV_CACHE


class KVCache:
    """Owns the KV storage and all layout-dependent addressing, so that
    consumers (KV transfer, attention) stay layout-agnostic."""

    def __init__(
        self,
        cfg: KVConfig,
        device: torch.device,
        dtype=torch.bfloat16
    ):
        self.config = cfg
        if cfg.layout == KVLayout.NHD:
            self.tensor = torch.zeros(
                cfg.num_layers, cfg.max_num_pages, 2,
                cfg.page_size, cfg.num_kv_heads, cfg.head_dim,
                dtype=dtype, device=device,
            ).contiguous()
        else:
            raise NotImplementedError(
                f"KV layout {cfg.layout} is not recognized."
            )

    @property
    def layout(self) -> KVLayout:
        return self.config.layout

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    @property
    def page_size(self) -> int:
        return self.config.page_size

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def nbytes(self) -> int:
        return self.tensor.nbytes

    def data_ptr(self) -> int:
        return self.tensor.data_ptr()

    def chunk_ptrs(
        self, layer_idx: int, page_idx: int,
        token_start: int, token_end: int,
        base_ptr: int | None = None,
    ) -> tuple[list[int], int]:
        """Byte pointers to the contiguous chunks holding tokens
        [token_start, token_end) of one page of one layer, plus the size of
        each chunk. One pointer per chunk (K and V are separate under NHD).

        ``base_ptr`` addresses a remote cache with this same layout/config;
        it defaults to this cache's own storage.
        """
        if self.layout != KVLayout.NHD:
            raise NotImplementedError(
                f"chunk_ptrs is not implemented for layout {self.layout}."
            )

        layer_stride, page_stride, kv_stride, token_stride = self.tensor.stride()[:4]
        element_size = self.tensor.element_size()

        # token_stride = num_kv_heads * head_dim
        nbytes = (token_end - token_start) * token_stride * element_size

        if base_ptr is None:
            base_ptr = self.data_ptr()

        ptrs = [
            base_ptr + (
                layer_idx * layer_stride +
                page_idx * page_stride +
                kv_idx * kv_stride +
                token_start * token_stride
            ) * element_size for kv_idx in [0, 1]
        ]
        return ptrs, nbytes

    def copy_pages(self, src_pages: list[int], dst_pages: list[int]) -> None:
        """Copy whole pages (every layer, both K and V, all tokens) within
        this cache: ``src_pages[i]`` -> ``dst_pages[i]``."""
        if len(src_pages) != len(dst_pages):
            raise ValueError(
                f"copy_pages got {len(src_pages)} src pages and "
                f"{len(dst_pages)} dst pages."
            )
        if not src_pages:
            return
        if self.layout != KVLayout.NHD:
            raise NotImplementedError(
                f"copy_pages is not implemented for layout {self.layout}."
            )

        src = torch.as_tensor(src_pages, dtype=torch.long, device=self.device)
        dst = torch.as_tensor(dst_pages, dtype=torch.long, device=self.device)
        # The gather is materialized before the scatter, so pages may appear
        # in both src and dst.
        self.tensor[:, dst] = self.tensor[:, src]

    def chunk_view(
        self, layer_idx: int, page_idx: int,
        token_start: int, token_end: int,
        tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """View of tokens [token_start, token_end) of one page of one layer,
        covering both K and V. ``tensor`` overrides the backing storage (e.g.
        a tensor rebuilt from another process' cache with this same layout).
        """
        if self.layout != KVLayout.NHD:
            raise NotImplementedError(
                f"chunk_view is not implemented for layout {self.layout}."
            )

        if tensor is None:
            tensor = self.tensor
        return tensor[layer_idx, page_idx, :, token_start:token_end]
