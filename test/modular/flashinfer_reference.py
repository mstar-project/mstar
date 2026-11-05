"""FlashInfer's attention wrappers as dense torch math on CPU, for tests that run the real
attention managers without a GPU (install with ``sys.modules["flashinfer"] = reference_flashinfer()``).

A helper module rather than ``conftest.py``: several ``conftest.py`` share CPU Core's paths, so a
test can't import one of them by name reliably."""

import importlib.machinery
import itertools
import math
import types

import torch


def reference_attention(q, k, v, causal=True):
    """Attention of ``q`` ``[T, Hq, D]`` over ``k``/``v`` ``[L, Hkv, D]``, the queries the last ``T``
    positions when causal (FlashAttention's bottom-right alignment); fp32 ``(out, lse)``, lse
    natural-log ``[T, Hq]``."""
    group = q.shape[1] // k.shape[1]
    k, v = (t.float().repeat_interleave(group, dim=1) for t in (k, v))
    scores = torch.einsum("thd,lhd->htl", q.float(), k) * q.shape[-1] ** -0.5
    if causal:
        t, length = q.shape[0], k.shape[0]
        seen = torch.arange(length)[None, :] <= torch.arange(t)[:, None] + (length - t)
        scores = scores.masked_fill(~seen.to(scores.device), -math.inf)
    return torch.einsum("htl,lhd->thd", torch.softmax(scores, dim=-1), v), torch.logsumexp(scores, -1).T


def reference_flashinfer() -> types.ModuleType:
    """FlashInfer's paged prefill/decode and ragged wrappers, with their lse bases. A paged plan
    refuses a row FlashInfer's split-KV planner would miscount: no pages, or a last page holding no
    slot (see ``local_indptrs``). The ragged one attends each row's queries over its own keys
    (``kv_indptr``), and reads keys past each row's end as FA3 does."""
    fi = types.ModuleType("flashinfer")
    # a spec, as an imported module has: importlib.util.find_spec (diffusers probes for
    # flashinfer when first imported) raises on a module in sys.modules whose __spec__ is None
    fi.__spec__ = importlib.machinery.ModuleSpec("flashinfer", None)

    class Paged:
        base = "log2"

        def __init__(self, *args, **kwargs):
            pass

        def plan(self, qo_indptr=None, paged_kv_indptr=None, paged_kv_indices=None,
                 paged_kv_last_page_len=None, *, page_size, seq_lens, causal=False, indptr=None,
                 indices=None, last_page_len=None, **kwargs):
            kvp = (paged_kv_indptr if indptr is None else indptr).tolist()
            self.indices = (paged_kv_indices if indices is None else indices).tolist()
            last = (paged_kv_last_page_len if last_page_len is None else last_page_len).tolist()
            self.rows = [(kvp[b], kvp[b + 1], int(seq_lens[b])) for b in range(len(last))]
            for (start, end, length), tail in zip(self.rows, last, strict=True):
                assert end > start and 1 <= tail <= page_size and length == (end - start - 1) * page_size + tail
            self.qo = qo_indptr.tolist() if qo_indptr is not None else list(range(len(last) + 1))
            self.causal = causal

        def run(self, q, kv, return_lse=False, lse_base=None):
            outs, lses = [], []
            # a layer [pages, 2, slots, H, D], or its K and V apart
            caches = kv if isinstance(kv, tuple) else kv.unbind(1)
            for b, (start, end, length) in enumerate(self.rows):
                k, v = (cache[self.indices[start:end]].flatten(0, 1)[:length] for cache in caches)
                out, lse = reference_attention(q[self.qo[b]:self.qo[b + 1]], k, v, self.causal)
                outs.append(out)
                lses.append(lse)
            out, lse = torch.cat(outs).to(q.dtype), torch.cat(lses)
            if (lse_base or self.base) == "log2":
                lse = lse / math.log(2)
            return (out, lse) if return_lse else out

    class Decode(Paged):
        def __init__(self, *args, use_tensor_cores=False, **kwargs):
            self.use_tensor_cores = use_tensor_cores

        def plan(self, **kwargs):
            super().plan(**kwargs)
            self.causal = False

    class Ragged:
        def __init__(self, *args, qo_indptr_buf=None, kv_indptr_buf=None, **kwargs):
            self.bufs = (qo_indptr_buf, kv_indptr_buf)

        def plan(self, qo_indptr, kv_indptr, *args, causal=False, **kwargs):
            self.qo, self.kv, self.causal = qo_indptr.tolist(), kv_indptr.tolist(), causal
            # graph mode: the layouts into the caller's buffers
            for buf, indptr in zip(self.bufs, (qo_indptr, kv_indptr), strict=True):
                if buf is not None:
                    buf.copy_(indptr)

        def run(self, q, k, v, out=None, lse=None, return_lse=False, lse_base="log2"):
            parts = []
            for (a, b), (c, d) in zip(itertools.pairwise(self.qo), itertools.pairwise(self.kv), strict=True):
                part = reference_attention(q[a:b], k[c:d], v[c:d], self.causal)
                # FA3 reads the keys past a row's end, the next rows' and past the plan, to its last
                # 128-key tile and masks them additively: nothing from a finite key, NaN in the row
                # from a NaN one
                over = slice(d, min(k.shape[0], c + -(-(d - c) // 128) * 128))
                poison = (k[over].float().sum() + v[over].float().sum()) * 0
                parts.append((part[0] + poison, part[1] + poison))
            o, got = torch.cat([p[0] for p in parts]).to(q.dtype), torch.cat([p[1] for p in parts])
            got = got if lse_base == "ln" else got / math.log(2)
            # the planned rows alone, into the caller's buffers or q-sized ones of its own (FlashInfer's
            # torch.empty; NaN here)
            if out is None:
                out = torch.full((q.shape[0], *o.shape[1:]), math.nan, dtype=o.dtype, device=q.device)
            if lse is None:
                lse = torch.full((q.shape[0], *got.shape[1:]), math.nan, dtype=got.dtype, device=q.device)
            out[:len(o)], lse[:len(got)] = o, got
            return (out, lse) if return_lse else out

    fi.BatchPrefillWithPagedKVCacheWrapper = Paged
    fi.BatchDecodeWithPagedKVCacheWrapper = Decode
    fi.BatchPrefillWithRaggedKVCacheWrapper = Ragged
    return fi
