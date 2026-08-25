

import queue
import threading

import torch

from mstar.engine.resources.kv.config import KVConfig, KVLayout


class PageAllocator:
    """Simple page allocator using a FIFO queue of free page indices.

    Thread-safe: a ``threading.Lock`` makes the qsize-then-get sequence in
    ``allocate``/``try_allocate`` atomic against concurrent ``free`` calls.
    Required by the pre-plan path, where the plan thread runs
    ``try_allocate`` while the GPU thread runs ``free`` from
    ``reset_label`` — the unlocked qsize/get pair could false-negative
    (return None when pages are about to be freed) or partially fill the
    output list under multi-consumer contention.
    """

    def __init__(self, max_num_pages: int):
        self.max_num_pages = max_num_pages
        self.free_pages: queue.Queue[int] = queue.Queue()
        self._lock = threading.Lock()
        for i in range(max_num_pages):
            self.free_pages.put(i)

    def allocate(self, n: int) -> list[int]:
        with self._lock:
            if self.free_pages.qsize() < n:
                raise RuntimeError(
                    f"Not enough free pages: requested {n}, "
                    f"available {self.free_pages.qsize()}"
                )
            return [self.free_pages.get() for _ in range(n)]

    def try_allocate(self, n: int) -> list[int] | None:
        """Like allocate() but returns None instead of raising on failure."""
        with self._lock:
            if self.free_pages.qsize() < n:
                return None
            return [self.free_pages.get() for _ in range(n)]

    def free(self, pages: list[int]) -> None:
        with self._lock:
            for page in pages:
                self.free_pages.put(page)

    @property
    def num_free(self) -> int:
        return self.free_pages.qsize()


@torch.library.custom_op("mstar::kv_scatter_nhd", mutates_args={"cache"})
def kv_scatter_nhd(
    cache: torch.Tensor, layer_idx: int,
    k: torch.Tensor, v: torch.Tensor,
    page_idx: torch.Tensor, cache_idx: torch.Tensor,
) -> None:
    """Scatter per-token K/V into one layer's (page, offset) slots.

    A custom op rather than plain indexing because the forward reaches the
    cache through an attribute chain: dynamo lifts it as a graph attribute, so
    tracing the mutation makes AOTAutograd functionalize it into a copy of the
    WHOLE cache (tens of GiB). Declaring the mutation here keeps the write in
    place and in-graph — no copy, and no graph break to recompile the layer
    body once per layer.
    """
    layer = cache[layer_idx]
    layer[page_idx, 0, cache_idx] = k.to(cache.dtype)
    layer[page_idx, 1, cache_idx] = v.to(cache.dtype)


@kv_scatter_nhd.register_fake
def _kv_scatter_nhd_fake(
    cache: torch.Tensor, layer_idx: int,
    k: torch.Tensor, v: torch.Tensor,
    page_idx: torch.Tensor, cache_idx: torch.Tensor,
) -> None:
    return None


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

    def layer_view(self, layer_idx: int) -> torch.Tensor:
        """One layer's pages, in this cache's layout — what an attention
        kernel consumes. NHD: [max_num_pages, 2, page_size, num_kv_heads,
        head_dim]."""
        if self.layout != KVLayout.NHD:
            raise NotImplementedError(
                f"layer_view is not implemented for layout {self.layout}."
            )
        return self.tensor[layer_idx]

    def read_tokens(
        self, layer_idx: int,
        page_idx: torch.Tensor, cache_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Gather the (page, offset-in-page) slots written by ``write_tokens``.
        Returns [num_tokens, 2, num_kv_heads, head_dim] (K at index 0, V at 1);
        it is a gather, so a copy rather than a view."""
        if self.layout != KVLayout.NHD:
            raise NotImplementedError(
                f"read_tokens is not implemented for layout {self.layout}."
            )
        return self.tensor[layer_idx][page_idx, :, cache_idx]

    def write_tokens(
        self, layer_idx: int,
        k: torch.Tensor, v: torch.Tensor,
        page_idx: torch.Tensor, cache_idx: torch.Tensor,
        return_tensor: bool=False
    ) -> None:
        """Scatter per-token K/V ([num_tokens, num_kv_heads, head_dim]) into
        the (page, offset-in-page) slots given by ``page_idx``/``cache_idx``.

        Goes through ``mstar::kv_scatter_nhd`` so the mutation survives being
        traced — see that op for why.
        """
        if self.layout != KVLayout.NHD:
            raise NotImplementedError(
                f"write_tokens is not implemented for layout {self.layout}."
            )
        torch.ops.mstar.kv_scatter_nhd(
            self.tensor, layer_idx, k, v, page_idx, cache_idx,
        )
        if return_tensor:
            return self.read_tokens(layer_idx, page_idx, cache_idx)

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
