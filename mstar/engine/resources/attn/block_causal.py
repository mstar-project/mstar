"""Block-causal attention over a paged KV cache.

A row's new tokens attend the row's cache block-causally: a token in block ``b``
(``block_size`` tokens a block, counted from the row's first cached token) attends
every key of blocks ``0..b``. Earlier blocks can't see later ones, so what a stream
has cached stays valid, and a step runs only its new tokens over the keys of the old.

One step can add several blocks, whose prefixes overlap, so the plan's rows expand
into one paged "request" per block. FlashInfer's prefill kernel takes that as is:
each block's queries, and the pages of its prefix.
"""

import numpy as np
import torch

from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.flashinfer import CGSlotKey, EagerSlotKey, FlashInferManager
from mstar.engine.resources.attn.wrappers import FlashInferPrefillWrapper
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.step import StepContext


def paged_block_causal_layout(
    qo_indptr: list[int],
    kv_indptr: list[int],
    indices: np.ndarray,
    kv_lens: list[int],
    block_size: int,
    page_size: int,
) -> tuple[list[int], list[int], np.ndarray, list[int]]:
    """A KV plan's rows as ``(qo_indptr, kv_indptr, indices, last_page_len)`` of blocks.

    A row with no new tokens plans no block. A row must start on a block: its new
    tokens are the last ``qo`` of its ``kv_len``.
    """
    qo, kv, last = [0], [0], []
    parts = []
    for row in range(len(qo_indptr) - 1):
        new = qo_indptr[row + 1] - qo_indptr[row]
        if new == 0:
            continue
        end = kv_lens[row]
        start = end - new
        if start % block_size:
            raise ValueError(f"a row's {new} new tokens start at {start}, inside a block of {block_size}")
        pages = indices[kv_indptr[row] : kv_indptr[row + 1]]
        for first in range(start, end, block_size):
            stop = min(first + block_size, end)
            num_pages = -(-stop // page_size)
            qo.append(qo[-1] + stop - first)
            kv.append(kv[-1] + num_pages)
            last.append(stop - (num_pages - 1) * page_size)
            parts.append(pages[:num_pages])
    return qo, kv, np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32), last


def max_blocks(num_tokens: int, num_rows: int, block_size: int) -> int:
    """Most blocks a step of these rows and tokens has: every row may end on a partial block."""
    return num_tokens // block_size + num_rows


class PagedBlockCausalWrapper(FlashInferPrefillWrapper):
    """A prefill wrapper planned per block rather than per row.

    Captured, ``batch_size`` is the bucket's most blocks: a plan pads to it with
    empty blocks, and the first plan latches FlashInfer's ceilings at the longest
    prefix, which the capture's own layout (no prefix yet) would leave too low.
    """

    def __init__(self, block_size: int, max_seq_len: int, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.max_prefix_pages = -(-max_seq_len // self.page_size)
        if self.accelerator_graph:
            self._prime()

    def _prime(self) -> None:
        full = self.max_prefix_pages
        n, left, qo = self.batch_size, self.max_total_tokens, [0]
        for _ in range(n):
            qo.append(qo[-1] + min(self.block_size, left))
            left -= qo[-1] - qo[-2]
        self._plan_blocks(
            qo,
            [i * full for i in range(n + 1)],
            np.zeros(n * full, dtype=np.int32),
            [self.page_size] * n,
            torch.bfloat16,
        )

    def _plan_blocks(self, qo, kv, indices, last, dtype) -> None:
        if self.accelerator_graph:
            pad = self.batch_size - (len(qo) - 1)
            if pad < 0:
                raise ValueError(f"{len(qo) - 1} blocks exceed the {self.batch_size} this bucket captures")
            qo = qo + [qo[-1]] * pad
            kv = kv + [kv[-1]] * pad
            last = last + [1] * pad
        super().plan(
            torch.tensor(qo, dtype=torch.int32),
            torch.tensor(kv, dtype=torch.int32),
            torch.from_numpy(np.asarray(indices, dtype=np.int32)),
            torch.tensor(last, dtype=torch.int32),
            causal=False,
            dtype=dtype,
        )

    def plan(
        self, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len,
        causal=False, dtype=torch.bfloat16, kv_lens=None, window_left=-1, **kwargs,
    ):
        del causal, window_left, kwargs
        kv_indptr = paged_kv_indptr.tolist()
        if kv_lens is None:
            last_pages = paged_kv_last_page_len.tolist()
            lens = [
                max(kv_indptr[r + 1] - kv_indptr[r] - 1, 0) * self.page_size + last_pages[r]
                for r in range(len(last_pages))
            ]
        else:
            lens = [int(n) for n in kv_lens]
        qo, kv, indices, last = paged_block_causal_layout(
            qo_indptr.tolist(), kv_indptr, paged_kv_indices.numpy(), lens, self.block_size, self.page_size,
        )
        self._plan_blocks(qo, kv, indices, last, dtype)


class FlashInferBlockCausalManager(FlashInferManager):
    """Attention over a KV pool's rows, block-causally; see ``BlockCausalAttentionSpec``."""

    # its wrappers plan per block, over the pool's whole slots
    supports_context_parallel = False

    def __init__(self, block_size: int, **kwargs):
        super().__init__(**kwargs)
        self._block_size = block_size
        self._wrapper_block_kwargs = dict(block_size=block_size, max_seq_len=self._kv_config.max_seq_len)

    @property
    def supports_preplan(self):
        return False

    def resident_refusal(self) -> str | None:
        return "block-causal attention plans per block on the host"

    def plan(self, step: AttentionStep, ctx: StepContext):
        if step.causal:
            raise ValueError(
                "block-causal attention is its own mask; declare it with AttentionStep(causal=False)"
            )
        super().plan(step, ctx)

    def _cg_wrapper(self, lease, label, num_rows, force_prefill):
        del force_prefill
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        wrapper = self._cg_plan_states.get(key)
        if wrapper is None:
            blocks = max_blocks(lease.bucket.num_tokens, num_rows, self._block_size)
            # a capture's padding rows may run longer than any request's stream
            longest = max(self._kv_config.max_seq_len, lease.bucket.num_tokens)
            page_size = self._wrapper_kv_kwargs["page_size"]
            kv_kwargs = {**self._wrapper_kv_kwargs, "max_num_pages": blocks * -(-longest // page_size)}
            wrapper = self._cg_plan_states[key] = PagedBlockCausalWrapper(
                workspace_buffer=self._workspaces.get(label, lease.slot),
                batch_size=blocks,
                max_total_tokens=lease.bucket.num_tokens,
                accelerator_graph=True,
                block_size=self._block_size,
                max_seq_len=longest,
                **kv_kwargs,
            )
        return wrapper

    def _eager_wrapper(self, label, is_decode, slot):
        del is_decode
        key = EagerSlotKey(label=label, slot=slot, is_decode=False)
        wrapper = self._eager_plan_states.get(key)
        if wrapper is None:
            wrapper = self._eager_plan_states[key] = PagedBlockCausalWrapper(
                workspace_buffer=self._workspaces.get(label, slot),
                **self._wrapper_block_kwargs,
                **self._wrapper_kv_kwargs,
            )
        return wrapper


class BlockCausalAttentionManager:
    """Builds the resource a ``BlockCausalAttentionSpec`` declares."""

    @classmethod
    def build(cls, spec, info):
        kv_config = info.dependency(spec.config.kv_cache).config
        if not isinstance(kv_config, PagedKVConfig) or len(kv_config.groups) > 1:
            raise TypeError(f"block-causal attention needs a plain paged KV, not {spec.config.kv_cache!r}'s config")
        kv_config.shard(info.head_shards)
        return FlashInferBlockCausalManager(
            block_size=spec.block_size,
            kv_cache=spec.config.kv_cache,
            device=info.device,
            dtype=info.kv_dtype,
            kv_config=kv_config,
            backend=spec.config.flashinfer_backend,
            sm_scale=spec.config.sm_scale,
        )
