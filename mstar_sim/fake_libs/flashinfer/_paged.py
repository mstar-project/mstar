"""What FlashInfer's attention wrappers share: a plan, and a run priced by it.

``plan()`` reads its host arrays once, checks them as FlashInfer does, copies
them where FlashInfer keeps them (the caller's static buffers under
``use_cuda_graph``, else device copies), then calls its planner module
(``_cached_module.plan``, FlashInfer's own signature), which prices the
attention the plan describes from each request's query and key counts. So a
caller that calls the planner module directly, as mstar's captured wrappers do
on a checked FlashInfer version, prices the same step. ``run`` is one kernel
(``clock.kernel_fn``) charged the latest plan's price, so a graph replay
prices whatever the last plan was.

A planner lays its scratch out as FlashInfer's does (``allocate``), sized by FlashInfer's own
arithmetic (``prefill_allocations``, ``fa3_allocations``, ``decode_allocations``; 0.7.2's
``scheduler.cuh``), so a workspace too small for a plan raises FlashInfer's overflow where
FlashInfer would.

FlashInfer's conventions, which the pricing follows: a paged request's length
is ``(num_pages - 1) * page_size + last_page_len``, and a request with ``n``
queries over ``m`` keys aligns its queries to the end of its keys (causal
query ``i`` sees keys ``j <= i + m - n``; ``window_left = w >= 0`` adds
``j >= i + m - n - w``).
"""

from __future__ import annotations

import torch

from mstar_sim import clock
from mstar_sim.device import gpu_error


def canonical_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, dtype)


def check_layout(kv_layout: str) -> str:
    if kv_layout not in ("NHD", "HND"):
        raise gpu_error(KeyError(f"Invalid kv_layout {kv_layout}"))
    return kv_layout


def element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def paged_kv_len(num_pages: int, last_page_len: int, page_size: int) -> int:
    return (num_pages - 1) * page_size + last_page_len if num_pages > 0 else 0


def visible_keys(n: int, m: int, causal: bool, window_left: int) -> int:
    """How many (query, key) pairs a request's mask lets through: what a
    tiled kernel computes, near enough, since it skips fully masked tiles."""
    if n <= 0 or m <= 0:
        return 0
    if causal:
        full = n * m - n * (n - 1) // 2 if m >= n else m * (m + 1) // 2
    else:
        full = n * m
    if window_left >= 0:
        full = min(full, n * (window_left + 1))
    return full


def _outputs(wrapper, out, lse):
    return out, lse


def attention_kernel(name: str):
    """A wrapper's run as kernel ``name`` (a method): priced by the wrapper's
    latest plan, read when charged, so a replay prices its own step's plan."""
    return clock.kernel_fn(name, cost=lambda wrapper, out, lse: lambda: wrapper._cost)(_outputs)


def _not_supported(what: str):
    raise NotImplementedError(f"mstar_sim: flashinfer.{what} has no stand-in yet")


def table_features(rows: list, heads: int, kv_heads: int, head_dim_qk: int, head_dim_vo: int,
                   kv_dtype) -> dict:
    """What :func:`mstar_sim.cost.kernel_seconds` needs to price attention from measured
    tables: ``rows`` [(queries, keys, visible pairs)] of the requests with queries."""
    return {"rows": rows, "heads": heads, "kv_heads": kv_heads,
            "head_dim": head_dim_qk if head_dim_qk == head_dim_vo else None, "kv_dtype": kv_dtype}


# the int workspace each FlashInfer wrapper allocates itself (the float one is the caller's)
INT_WORKSPACE_BYTES = 8 << 20


def buffer_bytes(buffer: torch.Tensor) -> int:
    return buffer.numel() * buffer.element_size()


def allocate(allocations: list, float_bytes: int | None, int_bytes: int | None = INT_WORKSPACE_BYTES):
    """Lay out a plan's ``allocations`` [(buffer, name, bytes, alignment)] as FlashInfer's
    ``AlignedAllocator`` does, one per buffer (``"float"`` of ``float_bytes``, ``"int"`` of
    ``int_bytes``): in order, each at the next multiple of its alignment. One that doesn't fit
    raises FlashInfer's overflow; a size of None only counts. Returns (float, int) bytes used."""
    room = {"float": float_bytes, "int": int_bytes}
    used = {"float": 0, "int": 0}
    for buffer, name, size, alignment in allocations:
        pad = -used[buffer] % alignment
        left = None if room[buffer] is None else room[buffer] - used[buffer]
        if left is not None and pad + size > left:
            raise gpu_error(RuntimeError(
                f"Buffer overflow when allocating memory for {name} with size {size} and alignment "
                f"{alignment}, but only {left} bytes available in AlignedAllocator. Increase the "
                "workspace buffer size."))
        used[buffer] += pad + size
    return used["float"], used["int"]


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def _cta_tile_q(packed_qo_len: int, head_dim_vo: int, head_dim_qk: int, kv_bytes: int) -> int:
    """FlashInfer's ``FA2DetermineCtaTileQ`` on an 8.0-or-newer GPU (every one the emulator
    offers): at 16 packed rows or fewer, tile 16 if one query tile and one KV step of the 1x4
    warp layout fit the block's opt-in shared memory, else 64."""
    if head_dim_vo >= 512:
        return 16 if packed_qo_len <= 32 else 32
    if head_dim_qk >= 512:
        return 16
    if packed_qo_len > 64 and head_dim_vo < 256:
        return 128
    if packed_qo_len > 16:
        return 64
    smem = 16 * head_dim_qk * 2 + (head_dim_qk + head_dim_vo) * 16 * 4 * kv_bytes
    return 64 if smem > torch.cuda.get_device_properties().shared_memory_per_block_optin else 16


def _check_length(name: str, indptr: list[int], i: int) -> None:
    if indptr[i + 1] < indptr[i]:
        raise gpu_error(RuntimeError(f"{name}[{i + 1}]{indptr[i + 1]} - {name}[{i}]{indptr[i]} should be non-negative"))


def prefill_allocations(qo: list[int], kv: list[int], total_num_rows: int, batch: int, num_qo_heads: int,
                        num_kv_heads: int, page_size: int, cuda_graph: bool, head_dim_qk: int,
                        head_dim_vo: int, window_left: int, fixed_split_size: int = -1,
                        disable_split_kv: bool = False, num_colocated_ctas: int = 0,
                        uniform_q_len: int = 0, kv_bytes: int = 2) -> list:
    """What FlashInfer's fa2 prefill planner (``PrefillPlanImpl`` over
    ``PrefillSplitQOKVIndptr``) allocates for ``qo`` rows and ``kv`` pages per request, for a
    module of ``kv_bytes`` KV elements: the float workspace holds the split-KV partials, only when
    it splits."""
    gqa = num_qo_heads // num_kv_heads
    max_grid = max(0, 2 * torch.cuda.get_device_properties().multi_processor_count - num_colocated_ctas)
    max_batch_if_split = max_grid // num_kv_heads
    for i in range(batch):
        _check_length("qo_indptr", qo, i)
        _check_length("kv_indptr", kv, i)
    packed = [(qo[i + 1] - qo[i]) * gqa for i in range(batch)]
    kv_len = [kv[i + 1] - kv[i] for i in range(batch)]
    if cuda_graph and uniform_q_len > 0:
        for i, p in enumerate(packed):
            if p != uniform_q_len * gqa:
                raise gpu_error(RuntimeError(
                    f"qo_indptr gives request {i} a packed qo length of {p}, but "
                    f"uniform_q_len={uniform_q_len} promises {uniform_q_len * gqa}"))
        tile = _cta_tile_q(uniform_q_len * gqa, head_dim_vo, head_dim_qk, kv_bytes)
        tiles_q = batch * _ceil(uniform_q_len * gqa, tile)
    elif cuda_graph:  # sized for the most rows one request could have
        tile = _cta_tile_q((total_num_rows - batch + 1) * gqa, head_dim_vo, head_dim_qk, kv_bytes)
        tiles_q = _ceil(total_num_rows * gqa, tile) + batch - 1
    else:
        tile = _cta_tile_q(sum(packed) // max(batch, 1), head_dim_vo, head_dim_qk, kv_bytes)
        tiles_q = sum(_ceil(p, tile) for p in packed)
    q_tiles = [_ceil(p, tile) for p in packed]
    effective = [min(_ceil(window_left + tile, page_size), n) if window_left >= 0 else n for n in kv_len]
    split, chunk = False, None
    if not disable_split_kv and fixed_split_size > 0:
        chunk = fixed_split_size
    elif not disable_split_kv:  # the shortest kv chunk whose tiles fit the grid
        longest = max([1, *effective])
        low, high = max(128 // page_size, 1), longest
        while low < high:
            mid = (low + high) // 2
            tiles = sum(t * _ceil(max(n, 1), mid) for t, n in zip(q_tiles, effective, strict=True))
            if tiles > max_batch_if_split:
                low = mid + 1
            else:
                high = mid
        split, chunk = cuda_graph or low < longest, low
    chunks = [1 if chunk is None else _ceil(max(n, 1), chunk) for n in effective]
    if fixed_split_size > 0 and not disable_split_kv:
        split = split or any(c > 1 for c in chunks)
    new_batch = sum(t * c for t, c in zip(q_tiles, chunks, strict=True))
    padded = max(max_batch_if_split, tiles_q) if cuda_graph else new_batch
    if new_batch > padded:
        raise gpu_error(RuntimeError(
            "new batch size should not exceed padded batch size. If you are using fixed split size, "
            "please consider disabling cuda graph."))
    out = [("int", "batch_prefill_request_indices", 4 * padded, 16),
           ("int", "batch_prefill_qo_tile_indices", 4 * padded, 16),
           ("int", "batch_prefill_kv_tile_indices", 4 * padded, 16),
           ("int", "batch_prefill_o_indptr", 4 * (batch + 1), 16),
           ("int", "batch_prefill_kv_chunk_size_ptr", 4, 1)]
    if cuda_graph:
        out.append(("int", "batch_prefill_total_num_rows", 4, 16))
    if split:
        rows = _ceil(padded * tile, gqa)
        partial = sum((qo[i + 1] - qo[i]) * chunks[i] for i in range(batch))
        if partial > rows:
            raise gpu_error(RuntimeError(
                f"batch_prefill partial outputs need {partial} rows but the plan reserved {rows}"))
        out += [("float", "batch_prefill_tmp_v", num_qo_heads * rows * head_dim_vo * 4, 16),
                ("float", "batch_prefill_tmp_s", num_qo_heads * rows * 4, 16),
                ("int", "batch_prefill_merge_indptr", 4 * (total_num_rows + 1), 16)]
    if cuda_graph:
        out.append(("int", "batch_prefill_block_valid_mask", padded, 16))
    return out


def fa3_allocations(qo: list[int], total_num_rows: int, batch: int, num_qo_heads: int, cuda_graph: bool,
                    head_dim_vo: int) -> list:
    """What FlashInfer's fa3 prefill planner (``PrefillSM90Plan``) allocates for ``qo`` rows per
    request: it splits no KV, so only work arrays, one work a query tile of one head (past 4096
    tiles a head, the heads share a schedule)."""
    for i in range(batch):
        _check_length("qo_indptr", qo, i)
    tile = 192 if head_dim_vo == 64 else 128
    per_head = _ceil(total_num_rows, tile) + batch - 1
    heads = 1 if per_head > 4096 else num_qo_heads
    works = heads * (per_head if cuda_graph else sum(_ceil(qo[i + 1] - qo[i], tile) for i in range(batch)))
    num_sm = torch.cuda.get_device_properties().multi_processor_count
    return [("int", f"batch_prefill_sm90_{name}", 4 * count, 16) for name, count in (
        ("qo_tile_indices", works), ("qo_offset", works), ("kv_offset", works), ("qo_len", works),
        ("kv_len", works), ("head_indices", works), ("work_indptr", num_sm + 1), ("batch_indices", works))]


def blocks_per_sm(threads: int, smem: int, regs: int = 0) -> int:
    """How many blocks of ``threads`` threads and ``smem`` bytes of dynamic shared memory one SM
    holds: NVIDIA's occupancy calculator (``cuda_occupancy.h``) for a kernel of ``regs`` registers
    a thread and no static shared memory, as a fresh process plans it (no shared-memory opt-in
    yet). ``regs`` 0 bounds nothing; the block cap never binds at 128 threads or more."""
    props = torch.cuda.get_device_properties()
    warps = _ceil(threads, 32)
    limit = props.max_threads_per_multi_processor // 32 // warps
    if regs > 256:
        return 0
    if regs:  # allocated a warp at a time, 256 registers at a time, in 4 sub-partitions
        per_warp = _ceil(regs * 32, 256) * 256
        limit = min(limit, props.regs_per_multiprocessor // 4 // per_warp * 4 // warps)
    block_smem = smem + 1024  # the driver reserves 1 KiB a block; a block's size is its 128-byte unit's multiple
    if block_smem > props.shared_memory_per_block + 1024:
        return 0
    return min(limit, props.shared_memory_per_multiprocessor // block_smem)


def decode_allocations(kv: list[int], batch: int, num_qo_heads: int, num_kv_heads: int, page_size: int,
                       cuda_graph: bool, head_dim: int, kv_dtype, regs: int = 0) -> list:
    """What FlashInfer's CUDA-core decode planner (``DecodePlanImpl`` over its work
    estimation) allocates for ``kv`` pages per request. The grid it sizes for is the decode
    kernel's occupancy (:func:`blocks_per_sm`) for ``regs`` registers a thread: the stand-in
    passes none, as the count is a fact of the compiled kernel, so its grid can only come out
    larger than a GPU's. A kernel past the 48 KiB a block gets before opting in (head dim 512)
    has no stand-in: its grid on a GPU depends on whether a run came before the plan."""
    gqa = num_qo_heads // num_kv_heads
    if gqa not in (1, 2, 3, 4, 6, 8, 16):  # the group sizes its kernels are built for
        raise gpu_error(RuntimeError(f"Unsupported group_size: {gqa}"))
    kv_bytes = element_size(canonical_dtype(kv_dtype))
    vec = max(16 // kv_bytes, head_dim // 32)
    bdx = head_dim // vec
    threads = max(128, bdx * gqa)
    bdz = threads // (bdx * gqa)
    tile = (2 if kv_bytes == 1 else 4) if gqa == 1 else 1
    props = torch.cuda.get_device_properties()
    stages = 2  # every GPU the emulator offers is 8.0 or newer
    smem = 2 * stages * tile * gqa * bdz * head_dim * kv_bytes + max(tile * threads * 8, 2 * gqa * bdz * 4)
    if smem > props.shared_memory_per_block:
        _not_supported(f"BatchDecodeWithPagedKVCacheWrapper (CUDA-core decode) of head dim {head_dim}, whose "
                       f"kernel needs {smem} bytes of shared memory")
    blocks = blocks_per_sm(threads, smem, regs)
    max_grid = blocks * props.multi_processor_count
    pages = [kv[i + 1] - kv[i] for i in range(batch)]
    if batch * num_kv_heads >= max_grid:
        split, new_batch = False, batch
    else:  # the fewest pages per chunk whose chunks fit the grid
        low, high = max(128 // page_size, 1), max(pages, default=0)
        while low < high:
            mid = (low + high) // 2
            if sum(_ceil(p, mid) for p in pages) * num_kv_heads > max_grid:
                low = mid + 1
            else:
                high = mid
        new_batch = sum(_ceil(max(p, 1), low) for p in pages)
        split = cuda_graph or new_batch != batch
    padded = (max_grid // num_kv_heads if split else batch) if cuda_graph else new_batch
    out = [("int", "batch_decode_request_indices", 4 * padded, 16),
           ("int", "batch_decode_kv_tile_indices", 4 * padded, 16),
           ("int", "batch_decode_o_indptr", 4 * (padded + 1), 16),
           ("int", "batch_decode_kv_chunk_size_ptr", 4, 1)]
    if split:
        out += [("float", "batch_decode_tmp_v", num_qo_heads * padded * head_dim * 4, 16),
                ("float", "batch_decode_tmp_s", num_qo_heads * padded * 4, 16),
                ("int", "batch_decode_block_valid_mask", padded, 16)]
    return out


def _copy(src: torch.Tensor, device) -> torch.Tensor:
    """A device copy of a plan array (a snapshot, as FlashInfer's H2D is)."""
    return src.to(device=device, dtype=torch.int32, copy=True)


def _host(x) -> list[int]:
    """A planner's host array, or a kept device buffer read without a sync."""
    return [int(v) for v in getattr(x, "_data", x).tolist()]


class PrefillPlanner:
    """FlashInfer's batch prefill planner module for ``backend`` (fa2/fa3; tensor-core decode
    and the ragged wrapper use it too): ``plan`` takes FlashInfer's arguments, lays out the
    plan's workspace and returns the plan info; ``workspace_size`` counts that layout. A
    wrapper's own module prices its next run."""

    def __init__(self, backend: str, wrapper=None, kv_dtype=None):
        self._backend = backend
        self._wrapper = wrapper
        self._kv_dtype = kv_dtype  # a module is built for one KV dtype; a wrapper's is its plan's

    def _kv_bytes(self) -> int:
        dtype = self._wrapper._cached_kv_data_type if self._wrapper is not None else self._kv_dtype
        return 2 if dtype is None else element_size(canonical_dtype(dtype))  # the binding's default: 2

    def _allocations(self, qo, kv_pages, total_num_rows, batch_size, num_qo_heads, num_kv_heads, page_size,
                     enable_cuda_graph, head_dim_qk, head_dim_vo, window_left, fa2_args) -> list:
        if self._backend == "fa2":
            return prefill_allocations(
                qo, kv_pages, int(total_num_rows), int(batch_size), num_qo_heads, num_kv_heads, page_size,
                bool(enable_cuda_graph), head_dim_qk, head_dim_vo, int(window_left), *fa2_args,
                kv_bytes=self._kv_bytes())
        if self._backend == "fa3":
            return fa3_allocations(qo, int(total_num_rows), int(batch_size), num_qo_heads, bool(enable_cuda_graph),
                                   head_dim_vo)
        return []  # another backend's planner (cudnn, trtllm-gen) isn't laid out

    @clock.computing()
    def plan(self, float_workspace, int_workspace, pinned_int_workspace, qo_indptr, kv_indptr, kv_len_arr,
             total_num_rows, batch_size, num_qo_heads, num_kv_heads, page_size, enable_cuda_graph,
             head_dim_qk, head_dim_vo, causal, window_left, *fa2_args) -> list[int]:
        qo, kv_pages, lens = _host(qo_indptr), _host(kv_indptr), _host(kv_len_arr)[:batch_size]
        allocate(self._allocations(qo, kv_pages, total_num_rows, batch_size, num_qo_heads, num_kv_heads,
                                   page_size, enable_cuda_graph, head_dim_qk, head_dim_vo, window_left,
                                   fa2_args), buffer_bytes(float_workspace))
        if self._wrapper is not None:
            self._wrapper._price(qo, kv_pages, lens, num_qo_heads, num_kv_heads, page_size, head_dim_qk,
                                 head_dim_vo, bool(causal), int(window_left))
        return [int(total_num_rows), int(batch_size), qo[-1], kv_pages[-1], sum(lens)]

    def workspace_size(self, device_buffer, qo_indptr, kv_indptr, kv_len_arr, total_num_rows, batch_size,
                       num_qo_heads, num_kv_heads, page_size, enable_cuda_graph, head_dim_qk, head_dim_vo,
                       causal, window_left, *fa2_args) -> tuple[int, int]:
        """(float, int) workspace bytes a plan of these arguments takes."""
        return allocate(self._allocations(_host(qo_indptr), _host(kv_indptr), total_num_rows, batch_size,
                                          num_qo_heads, num_kv_heads, page_size, enable_cuda_graph,
                                          head_dim_qk, head_dim_vo, window_left, fa2_args), None, None)


class DecodePlanner:
    """FlashInfer's batch decode planner module (CUDA-core decode): its plan sees the
    page counts only; the lengths come from the last-page lengths already copied."""

    def __init__(self, wrapper):
        self._wrapper = wrapper

    @clock.computing()
    def plan(self, float_workspace, int_workspace, pinned_int_workspace, indptr, batch_size, num_qo_heads,
             num_kv_heads, page_size, enable_cuda_graph, window_left, logits_soft_cap, head_dim_qk,
             head_dim_vo, empty_q_data, empty_kv_data) -> list[int]:
        w = self._wrapper
        kv_pages = _host(indptr)
        allocate(decode_allocations(kv_pages, batch_size, num_qo_heads, num_kv_heads, page_size,
                                    bool(enable_cuda_graph), head_dim_qk, empty_kv_data.dtype),
                 buffer_bytes(float_workspace))
        last = _host(w._paged_kv_last_page_len_buf)
        lens = [paged_kv_len(kv_pages[r + 1] - kv_pages[r], last[r], page_size) for r in range(batch_size)]
        w._price(list(range(batch_size + 1)), kv_pages, lens, num_qo_heads, num_kv_heads, page_size,
                 head_dim_qk, head_dim_vo, False, int(window_left))
        return [int(batch_size), kv_pages[-1], sum(lens)]


class PagedAttention:
    """The plan and run of a paged-KV wrapper. Subclasses parse their own
    ``plan`` arguments and call ``_set_plan``."""

    _kernel = attention_kernel("flashinfer.batch_paged_attention")

    def __init__(
        self, float_workspace_buffer: torch.Tensor, kv_layout: str, use_cuda_graph: bool,
        qo_indptr_buf, paged_kv_indptr_buf, paged_kv_indices_buf, paged_kv_last_page_len_buf,
        backend: str,
    ) -> None:
        self._kv_layout = check_layout(kv_layout)
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._backend = backend
        self._use_cuda_graph = use_cuda_graph
        if use_cuda_graph:
            for name, buf in (
                ("paged_kv_indptr_buf", paged_kv_indptr_buf),
                ("paged_kv_indices_buf", paged_kv_indices_buf),
                ("paged_kv_last_page_len_buf", paged_kv_last_page_len_buf),
            ):
                if not torch.is_tensor(buf):
                    raise gpu_error(ValueError(f"{name} should be a torch.Tensor in cuda graph mode"))
            self._fixed_batch_size = len(paged_kv_last_page_len_buf)
            if len(paged_kv_indptr_buf) != self._fixed_batch_size + 1:
                raise gpu_error(ValueError(
                    "The length of paged_kv_indptr_buf should be batch_size + 1."
                ))
            if qo_indptr_buf is not None and len(qo_indptr_buf) != self._fixed_batch_size + 1:
                raise gpu_error(ValueError("The length of qo_indptr_buf should be batch_size + 1."))
        else:
            self._fixed_batch_size = 0
        self._qo_indptr_buf = qo_indptr_buf
        self._paged_kv_indptr_buf = paged_kv_indptr_buf
        self._paged_kv_indices_buf = paged_kv_indices_buf
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buf
        self._max_total_num_rows: int | None = None
        self._planned = False
        self._cost = (0.0, 0.0)
        # FlashInfer's planner workspaces and per-plan state, as its wrappers keep them
        self._int_workspace_buffer = torch.empty(INT_WORKSPACE_BYTES, dtype=torch.uint8, device=self.device)
        self._pin_memory_int_workspace_buffer = torch.empty(INT_WORKSPACE_BYTES, dtype=torch.uint8, device="cpu")
        self._jit_module = None
        self._cached_module = None
        self._plan_info = None
        self._custom_mask_buf = None
        self._prefix_len_ptr = None
        self._pos_encoding_mode = "NONE"
        self._logits_soft_cap = 0.0
        self._window_left = -1

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    # ------------------------------------------------------------------ plan

    def _set_plan(
        self, qo_indptr, indptr, indices, last_page_len, *, num_qo_heads: int, num_kv_heads: int,
        head_dim_qk: int, head_dim_vo: int | None, page_size: int, causal: bool, window_left: int,
        q_data_type, kv_data_type, o_data_type, q_len_per_req: int = 1, fixed_split_size: int | None = None,
        disable_split_kv: bool = False,
    ) -> None:
        """Store the plan. ``qo_indptr`` None means ``q_len_per_req`` queries a row (decode)."""
        if num_qo_heads % num_kv_heads:
            raise gpu_error(ValueError(
                f"num_qo_heads {num_qo_heads} must be a multiple of num_kv_heads {num_kv_heads}"
            ))
        batch = len(last_page_len)
        if len(indptr) != batch + 1:
            raise gpu_error(ValueError(f"indptr has {len(indptr)} entries for a batch of {batch}"))
        qo_host = (
            [i * q_len_per_req for i in range(batch + 1)] if qo_indptr is None
            else [int(x) for x in qo_indptr.tolist()]
        )
        if len(qo_host) != batch + 1:
            raise gpu_error(ValueError(f"qo_indptr has {len(qo_host)} entries for a batch of {batch}"))
        kv_host = [int(x) for x in indptr.tolist()]
        last_host = [int(x) for x in last_page_len.tolist()]
        total_rows = qo_host[-1]

        if self._use_cuda_graph:
            if self._max_total_num_rows is None:
                self._max_total_num_rows = total_rows
            elif total_rows > self._max_total_num_rows:
                raise gpu_error(ValueError(
                    f"The total number of rows in qo_indptr {total_rows} in cuda graph mode cannot "
                    f"exceed the number of rows set during initialization {self._max_total_num_rows}."
                ))
            if batch != self._fixed_batch_size:
                raise gpu_error(ValueError(
                    "The batch size should be fixed during the lifecycle of the wrapper in "
                    f"cuda graph mode, the runtime batch size {batch} mismatches the batch size "
                    f"{self._fixed_batch_size}  set during initialization."
                ))
            if len(indices) > len(self._paged_kv_indices_buf):
                raise gpu_error(ValueError("The length of paged_kv_indices exceeds the allocated buffer size."))
            if qo_indptr is not None:
                self._qo_indptr_buf.copy_(qo_indptr)
            self._paged_kv_indptr_buf.copy_(indptr)
            self._paged_kv_last_page_len_buf.copy_(last_page_len)
            self._paged_kv_indices_buf[: len(indices)].copy_(indices)
        else:
            if qo_indptr is not None:
                self._qo_indptr_buf = _copy(qo_indptr, self.device)
            self._paged_kv_indptr_buf = _copy(indptr, self.device)
            self._paged_kv_indices_buf = _copy(indices, self.device)
            self._paged_kv_last_page_len_buf = _copy(last_page_len, self.device)

        self._batch_size = batch
        self._q_len_per_req = q_len_per_req
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim_qk = head_dim_qk
        self._head_dim_vo = head_dim_vo or head_dim_qk
        self._page_size = page_size
        self._causal = bool(causal)
        self._window_left = int(window_left)
        self._cached_q_data_type = canonical_dtype(q_data_type)
        self._cached_kv_data_type = canonical_dtype(kv_data_type or q_data_type)
        self._cached_o_data_type = canonical_dtype(o_data_type or q_data_type)
        self._qo_indptr_last = total_rows
        lens = [paged_kv_len(kv_host[r + 1] - kv_host[r], last_host[r], page_size) for r in range(batch)]
        self._max_q_len = max((qo_host[r + 1] - qo_host[r] for r in range(batch)), default=0)
        self._max_kv_len = max(lens, default=0)
        self._planned = False  # a planner that raises leaves this half-written plan unrunnable

        as_tensor = lambda x: torch.tensor(x, dtype=torch.int32)  # noqa: E731 - host arrays, as FlashInfer's
        workspaces = (self._float_workspace_buffer, self._int_workspace_buffer, self._pin_memory_int_workspace_buffer)
        if self._cuda_core_decode():
            self._cached_module = self._cached_module or DecodePlanner(self)
            self._plan_info = self._cached_module.plan(
                *workspaces, as_tensor(kv_host), batch, num_qo_heads, num_kv_heads, page_size,
                self._use_cuda_graph, self._window_left, self._logits_soft_cap, head_dim_qk, self._head_dim_vo,
                torch.empty(0, dtype=self._cached_q_data_type), torch.empty(0, dtype=self._cached_kv_data_type))
        else:
            self._cached_module = self._cached_module or PrefillPlanner(self._backend, self)
            args = [*workspaces, as_tensor(qo_host), as_tensor(kv_host), as_tensor(lens),
                    self._max_total_num_rows or total_rows, batch, num_qo_heads, num_kv_heads, page_size,
                    self._use_cuda_graph, head_dim_qk, self._head_dim_vo, self._causal, self._window_left]
            if self._backend == "fa2":  # fixed_split_size, disable_split_kv, num_colocated_ctas, uniform_q_len
                args += [fixed_split_size or -1, disable_split_kv, 0,
                         q_len_per_req if qo_indptr is None and q_len_per_req > 1 else 0]
            self._plan_info = self._cached_module.plan(*args)
        self._planned = True

    def _cuda_core_decode(self) -> bool:
        """Whether this wrapper plans with FlashInfer's batch decode module (CUDA-core
        decode), not the batch prefill module."""
        return False

    def _price(self, qo: list[int], kv_pages: list[int], lens: list[int], num_qo_heads: int,
               num_kv_heads: int, page_size: int, head_dim_qk: int, head_dim_vo: int, causal: bool,
               window_left: int) -> None:
        pairs, pages, rows = 0, 0, []
        for r in range(len(lens)):
            n = qo[r + 1] - qo[r]
            m = lens[r]
            seen = visible_keys(n, m, causal, window_left)
            pairs += seen
            pages += kv_pages[r + 1] - kv_pages[r] if n > 0 else 0
            if n > 0:
                rows.append((n, m, seen))
        q_el = element_size(self._cached_q_data_type)
        kv_el = element_size(self._cached_kv_data_type)
        o_el = element_size(self._cached_o_data_type)
        flops = 2.0 * pairs * num_qo_heads * (head_dim_qk + head_dim_vo)
        nbytes = (
            qo[-1] * num_qo_heads * (head_dim_qk * q_el + head_dim_vo * o_el)
            + pages * page_size * num_kv_heads * (head_dim_qk + head_dim_vo) * kv_el
        )
        self._cost = (flops, float(nbytes), table_features(
            rows, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, self._cached_kv_data_type))

    # ------------------------------------------------------------------- run

    def run(
        self, q: torch.Tensor, paged_kv_cache, *args, q_scale=None, k_scale=None, v_scale=None,
        out: torch.Tensor | None = None, lse: torch.Tensor | None = None, return_lse: bool = False,
        enable_pdl=None, window_left: int | None = None, sinks=None, kv_cache_sf=None,
        lse_base: str = "log2", **kwargs,
    ):
        if not self._planned:
            raise gpu_error(RuntimeError("plan() must be called before run()"))
        if args:
            _not_supported("custom-kernel run(*args)")
        if sinks is not None:
            _not_supported("attention sinks")
        if kv_cache_sf is not None:
            _not_supported("fp4 KV cache scale factors")
        if out is None:
            out = torch.empty(
                (q.shape[0], q.shape[1], self._head_dim_vo), dtype=self._cached_o_data_type,
                device=q.device,
            )
        if return_lse and lse is None:
            lse = torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
        out, lse = self._kernel(out, lse)
        return (out, lse) if return_lse else out
