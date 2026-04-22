"""Code Predictor engine (Qwen3-Omni Talker depth transformer).

Runs the 15-iteration residual-codebook AR loop as a *single, fully-unrolled*
CUDA graph (Phase 2 refactor). This replaces the earlier per-iteration graph
that captured one transformer forward at a time and then re-entered Python
between iterations for the LM head, sampler, and codec embedder.

Why unroll? The code predictor is the critical path for audio TTFT on
Qwen3-Omni. Per-iteration kernel-launch overhead (5 layers × ~10 launches
× 15 iterations ≈ 750 small launches per decode frame) dominated latency.
Capturing the whole depth loop -- attention + LM head + sampler + embedder
-- into one graph eliminates that overhead and matches the vox-serve
pattern (``cuda_graph_worker._initialize_depth_cuda_graphs_unrolled``).

Key design choices (see the phase-2 design doc in
``qwen3-talker-refactor-atindra_phase2_convo.md`` for the full rationale):

  * **Attention backend**: SDPA with a dense KV cache, *not* paged
    FlashInfer. FlashInfer requires a Python-side ``plan()`` call between
    iterations that cannot live inside a captured graph; a dense
    ``[n_layers, bs, 2, n_codebooks, n_kv_heads, head_dim]`` cache is
    stateless and captures cleanly. The cache is tiny (~5 MB per request)
    and transient per decode step.
  * **Unrolling strategy**: Python-for-loop inside ``torch.cuda.graph()``
    (Option P). Each iteration's LM head and codec embedder are fixed-
    address calls resolved at capture time.
  * **KV cache ownership**: A dedicated dense buffer owned by the runner
    (Option Y), *not* mixed into the paged allocator.
  * **Sampling**: A minimal graph-safe depth sampler
    (``sample_depth_gpu``) that reads temperature/top_k/top_p from
    preallocated device buffers and uses FlashInfer's deterministic path.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import asdict, dataclass

import torch

from mminf.engine.ar_engine import AREngine
from mminf.engine.base import EngineType, NodeBatch, NodeOutput
from mminf.model.base import NodeSubmodule
from mminf.utils.profiler import range_pop, range_push
from mminf.utils.sampling import Sampler, sample_depth_gpu

logger = logging.getLogger(__name__)


# TODO: this all feels a bit hacky and not generalizable. Ideally, we would
# like to modify our system to deal with the code predictor paradigm without
# making a code predictor engine (e.g., having an abstraction for injecting
# the cuda graph runner into the submodule execution path).


class CodePredictorSubmodule(NodeSubmodule):
    """Marker base class for submodules driven by ``CodePredictorEngine``.

    Concrete implementations (e.g. ``Qwen3OmniCodePredictorSubmodule``)
    must provide ``forward_batched`` that accepts a ``cuda_graph_runner``
    and returns per-request outputs. The engine does not drive these
    submodules through the generic ``forward`` path.
    """

    def forward(self, request_info, **kwargs):
        raise NotImplementedError(
            "Code predictor submodules must go through forward_batched."
        )


@dataclass
class UnrolledGraphData:
    """State captured for a single batch-size bucket."""

    graph: torch.cuda.CUDAGraph
    # bs-sized views of the shared full-size buffers. Keyed by the same names
    # used when they were allocated; ``pos_dec`` is itself a dict keyed by
    # iteration index (2..n_codebooks-1).
    slices: dict
    bs: int


class CodePredictorCudaGraphRunner:
    """Captures and replays the fully-unrolled depth MTP graph.

    One graph per batch-size bucket in ``CAPTURE_BATCH_SIZES``. All buckets
    share the same underlying buffers (sliced per-bs views), so the extra
    memory cost of adding a bucket is just the CUDAGraph object itself.

    Usage::

        runner = CodePredictorCudaGraphRunner(submodule, sampler, device)
        runner.warmup_and_capture()
        outputs = runner.run(graph_walk, request_ids, last_hidden, layer0_codes)
        # outputs = {"all_codes": [bs, n_codebooks], "codec_emb_sum": [bs, hidden]}
    """

    CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    def __init__(
        self,
        submodule,
        sampler: Sampler,
        device: torch.device,
    ):
        # Duck-typed on Qwen3OmniCodePredictorSubmodule: we rely on the
        # attributes documented below so this runner stays agnostic to the
        # specific subclass.
        self.submodule = submodule
        self.code_predictor = submodule.code_predictor  # Qwen3OmniCodePredictor
        self.talker_code_emb = submodule.talker_code_emb  # nn.Embedding (layer-0 codes)
        self.num_codes = submodule.num_codes              # = num_code_groups
        self.cp_cfg = submodule.cp_cfg                    # CodePredictorConfig
        self.sampler = sampler
        self.device = device
        self.max_batch_size = max(self.CAPTURE_BATCH_SIZES)

        self.graphs: dict[int, UnrolledGraphData] = {}
        self.memory_pool = None
        self._shared_bufs: dict | None = None

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------
    def _allocate_shared_buffers(self) -> None:
        """Allocate the full-size (max_batch_size) buffers shared by all buckets."""
        cp = self.cp_cfg
        max_bs = self.max_batch_size
        n_layers = cp.num_hidden_layers
        n_kv_heads = cp.num_key_value_heads
        head_dim = cp.head_dim
        hidden = cp.hidden_size
        vocab = cp.vocab_size
        n_codes = self.num_codes
        device = self.device

        # Match the dtype of the layer-0 codec embedding (typically fp32 for
        # the code predictor, per the "NO autocast" invariant documented in
        # CodePredictorEngine.has_autocast).
        dtype = self.talker_code_emb.weight.dtype

        # Dense KV cache: [n_layers, max_bs, 2 (K|V), max_seq_len, n_kv_heads, head_dim].
        # max_seq_len == n_codes because the depth sequence packs the
        # two-token prefill + (n_codes - 2) decode iterations into slots
        # 0..n_codes-1.
        kv_cache = torch.zeros(
            n_layers, max_bs, 2, n_codes, n_kv_heads, head_dim,
            dtype=dtype, device=device,
        )

        # Input: [max_bs, 2, hidden] packs last_hidden (slot 0) and c0_embed
        # (slot 1) so the prefill pass processes them in a single 2-token
        # forward. Positions are fixed at [0, 1].
        hidden_buf = torch.zeros(max_bs, 2, hidden, dtype=dtype, device=device)

        # Outputs.
        codec_emb_sum_buf = torch.zeros(max_bs, hidden, dtype=dtype, device=device)
        all_codes_buf = torch.zeros(max_bs, n_codes, dtype=torch.int64, device=device)

        # Per-request sampling config buffers. Defaults: temperature=1,
        # top_k=vocab (disabled), top_p=1 (disabled). The runner copies real
        # values from the Sampler into these before each replay.
        temperature_buf = torch.ones(max_bs, dtype=torch.float32, device=device)
        top_k_buf = torch.full((max_bs,), vocab, dtype=torch.int32, device=device)
        top_p_buf = torch.ones(max_bs, dtype=torch.float32, device=device)
        seed_buf = torch.zeros(max_bs, dtype=torch.long, device=device)
        offset_buf = torch.zeros(max_bs, dtype=torch.long, device=device)

        # Position tensors. These never change across replays -- they hold
        # the hardcoded [0, 1] for prefill and [i] for decode iteration i.
        pos_pf = torch.tensor(
            [0, 1], dtype=torch.int32, device=device,
        ).unsqueeze(0).expand(max_bs, -1).contiguous()
        pos_dec: dict[int, torch.Tensor] = {}
        for i in range(2, n_codes):
            pos_dec[i] = torch.full(
                (max_bs, 1), i, dtype=torch.int32, device=device,
            )

        self._shared_bufs = {
            "kv_cache": kv_cache,
            "hidden_buf": hidden_buf,
            "codec_emb_sum_buf": codec_emb_sum_buf,
            "all_codes_buf": all_codes_buf,
            "temperature_buf": temperature_buf,
            "top_k_buf": top_k_buf,
            "top_p_buf": top_p_buf,
            "offset_buf": offset_buf,
            "pos_pf": pos_pf,
            "pos_dec": pos_dec,
            "seed_buf": seed_buf
        }

    def _slice_for_bs(self, bs: int) -> dict:
        shared = self._shared_bufs
        return {
            "hidden_buf": shared["hidden_buf"][:bs],
            "kv_cache": shared["kv_cache"][:, :bs],
            "codec_emb_sum_buf": shared["codec_emb_sum_buf"][:bs],
            "all_codes_buf": shared["all_codes_buf"][:bs],
            "temperature_buf": shared["temperature_buf"][:bs],
            "top_k_buf": shared["top_k_buf"][:bs],
            "top_p_buf": shared["top_p_buf"][:bs],
            "offset_buf": shared["offset_buf"][:bs],
            "pos_pf": shared["pos_pf"][:bs],
            "seed_buf": shared["seed_buf"][:bs],
            "pos_dec": {i: shared["pos_dec"][i][:bs] for i in shared["pos_dec"]},
        }

    # ------------------------------------------------------------------
    # Captured body
    # ------------------------------------------------------------------
    def _run_unrolled_loop(self, slices: dict) -> None:
        """Execute the full 15-iteration MTP loop on the given bs-sized slices.

        This is the exact body captured by ``torch.cuda.graph()``. It is
        Python-unrolled (``for i in range(2, self.num_codes)``) so that each
        iteration's LM head slice and codec embedder are fixed-address calls
        resolved at capture time. No Python-side state, no allocations beyond
        what ``forward_depth_unrolled`` does internally.
        """
        hidden_buf = slices["hidden_buf"]
        kv_cache = slices["kv_cache"]
        codec_emb_sum_buf = slices["codec_emb_sum_buf"]
        all_codes_buf = slices["all_codes_buf"]
        temperature_buf = slices["temperature_buf"]
        top_k_buf = slices["top_k_buf"]
        top_p_buf = slices["top_p_buf"]
        offset_buf = slices["offset_buf"] 
        pos_pf = slices["pos_pf"]
        pos_dec = slices["pos_dec"]
        seed_buf = slices["seed_buf"]

        cp = self.code_predictor
        codec_embedding = cp.model.codec_embedding
        lm_head_weight = cp.lm_head_weight

        # Zero the KV cache at the start of every replay so stale state from
        # the previous decode step cannot leak into the new one.
        kv_cache.zero_()

        # Prefill pass: forward over [last_hidden, c0_embed] at positions [0, 1].
        # Writes K/V into cache slots [0, 1]; returns hidden at both positions.
        hidden_states = cp.forward_depth_unrolled(
            hidden_buf, pos_pf, kv_cache, cache_pos=0,
        )

        # Sample codebook 1 from the slot-1 hidden (the c0_embed position).
        last_hs = hidden_states[:, -1, :]
        logits = torch.matmul(last_hs, lm_head_weight[0].t())
        tokens = sample_depth_gpu(logits, temperature_buf, top_k_buf, top_p_buf, seed_buf, offset_buf)
        offset_buf += 1
        all_codes_buf[:, 1] = tokens
        embed = codec_embedding[0](tokens)
        codec_emb_sum_buf.add_(embed)

        # Decode iterations for codebooks 2..(num_codes - 1). Each adds one
        # token at position i, attending to the growing dense KV cache.
        for i in range(2, self.num_codes):
            pos_i = pos_dec[i]
            hidden_states = cp.forward_depth_unrolled(
                embed.unsqueeze(1), pos_i, kv_cache, cache_pos=i,
            )
            last_hs = hidden_states[:, 0, :]
            logits = torch.matmul(last_hs, lm_head_weight[i - 1].t())
            tokens = sample_depth_gpu(logits, temperature_buf, top_k_buf, top_p_buf, seed_buf, offset_buf)
            offset_buf += 1
            all_codes_buf[:, i] = tokens
            embed = codec_embedding[i - 1](tokens)
            codec_emb_sum_buf.add_(embed)

    # ------------------------------------------------------------------
    # Warmup + capture
    # ------------------------------------------------------------------
    def warmup_and_capture(self) -> None:
        """Warm up kernels and capture one graph per bucket."""
        if self.device is None or not torch.cuda.is_available():
            logger.warning(
                "CUDA not available, skipping unrolled graph capture for code predictor"
            )
            return

        self._allocate_shared_buffers()
        self.memory_pool = torch.cuda.graphs.graph_pool_handle()
        torch.cuda.set_device(self.device)

        # Capture largest-first so the shared memory pool allocations never
        # need to grow after a smaller bucket has already reserved addresses.
        for bs in reversed(self.CAPTURE_BATCH_SIZES):
            try:
                self._capture_one(bs)
                logger.info(
                    "CodePredictorCudaGraphRunner: captured unrolled graph for bs=%d",
                    bs,
                )
            except Exception:
                logger.warning(
                    "Failed to capture unrolled depth graph for bs=%d",
                    bs, exc_info=True,
                )

    def _capture_one(self, bs: int) -> None:
        slices = self._slice_for_bs(bs)

        # Warmup: 3 full passes to trigger lazy kernel compiles and stabilize
        # the memory pool (matches vox-serve's 3-iter warmup).
        torch.cuda.synchronize()
        for _ in range(3):
            self._run_unrolled_loop(slices)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.memory_pool):
            self._run_unrolled_loop(slices)
        torch.cuda.synchronize()

        # A couple of replay passes to let any pool-internal state settle
        # before the first real replay (matches vox-serve).
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()

        self.graphs[bs] = UnrolledGraphData(
            graph=graph,
            slices=slices,
            bs=bs,
        )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    def _get_padded_batch_size(self, bs: int) -> int | None:
        sizes = sorted(self.CAPTURE_BATCH_SIZES)
        idx = bisect.bisect_left(sizes, bs)
        if idx >= len(sizes):
            return None
        return sizes[idx]

    @torch.compiler.disable()
    def run(
        self,
        graph_walk: str,
        request_ids: list[str],
        last_hidden: torch.Tensor,
        layer0_codes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the full depth loop for one decode step.

        ``graph_walk`` is accepted for API symmetry with the AR graph runner
        but ignored -- the code predictor's KV cache is fresh every call, so
        the captured graph is identical across "talker_last_prefill" and
        "talker_decode".

        Args:
            request_ids: per-request identifiers (used to look up sampling config).
            last_hidden: ``[bs, hidden]`` final Talker hidden state.
            layer0_codes: ``[bs]`` int64 sampled codebook-0 tokens.

        Returns:
            ``{"all_codes": [bs, num_codes] int64, "codec_emb_sum": [bs, hidden] fp32}``
        """
        bs = len(request_ids)
        padded_bs = self._get_padded_batch_size(bs)
        if padded_bs is None or padded_bs not in self.graphs:
            raise RuntimeError(
                f"No captured unrolled graph for batch_size={bs} "
                f"(captured sizes: {sorted(self.graphs.keys())})"
            )

        graph_data = self.graphs[padded_bs]
        shared = self._shared_bufs

        # --- Step 1: eagerly embed the layer-0 codes via the Talker's
        #            shared codec_embedding (a separate module, not the code
        #            predictor's per-layer embedders). ---
        c0_embed = self.talker_code_emb(layer0_codes)  # [bs, hidden]

        # --- Step 2: stage inputs into the static buffers. Padding slots
        #            repeat the last real row so all rows are valid and
        #            kernel divergence stays zero; the padded outputs are
        #            discarded on return. ---
        hidden_buf = shared["hidden_buf"][:padded_bs]
        hidden_buf[:bs, 0].copy_(last_hidden)
        hidden_buf[:bs, 1].copy_(c0_embed)
        if bs < padded_bs:
            hidden_buf[bs:padded_bs, 0].copy_(
                last_hidden[-1:].expand(padded_bs - bs, -1)
            )
            hidden_buf[bs:padded_bs, 1].copy_(
                c0_embed[-1:].expand(padded_bs - bs, -1)
            )

        # Seed codec_emb_sum with c0_embed. The captured graph adds c1..c15
        # on top (via in-place ``.add_()``), producing the full sum at exit.
        codec_emb_sum_buf = shared["codec_emb_sum_buf"][:padded_bs]
        codec_emb_sum_buf[:bs].copy_(c0_embed)
        if bs < padded_bs:
            codec_emb_sum_buf[bs:padded_bs].copy_(
                c0_embed[-1:].expand(padded_bs - bs, -1)
            )

        # Pre-populate all_codes[:, 0] with layer0_codes; the graph fills
        # columns 1..num_codes-1. Zero the rest so previous call's state
        # cannot bleed through.
        all_codes_buf = shared["all_codes_buf"][:padded_bs]
        all_codes_buf[:bs, 0].copy_(layer0_codes)
        if bs < padded_bs:
            all_codes_buf[bs:padded_bs, 0].copy_(
                layer0_codes[-1:].expand(padded_bs - bs)
            )
        all_codes_buf[:, 1:].zero_()

        # --- Step 3: sampling config. ---
        self._update_sampling_buffers(request_ids, padded_bs)

        # --- Step 4: replay. ---
        graph_data.graph.replay()

        # --- Step 5: return bs-sized outputs. Clone so callers don't alias
        #            the static buffers (which will be overwritten on the
        #            next call). ---
        return {
            "all_codes": all_codes_buf[:bs].clone(),
            "codec_emb_sum": codec_emb_sum_buf[:bs].clone(),
        }

    def _update_sampling_buffers(self, request_ids: list[str], padded_bs: int) -> None:
        """Copy per-request sampling config into the static sampling buffers.

        Greedy (``temperature == 0``) is translated to
        ``(temperature=1, top_k=1)`` so the graph-safe sampler can remain
        branch-free. Padding slots repeat the last real entry.
        """
        from mminf.utils.sampling import SamplingConfig

        vocab = self.cp_cfg.vocab_size

        temps: list[float] = []
        top_ks: list[int] = []
        top_ps: list[float] = []
        for rid in request_ids:
            cfg = self.sampler._sampling_config.get(rid, SamplingConfig())
            if cfg.temperature == 0:
                temps.append(1.0)
                top_ks.append(1)
                top_ps.append(1.0)
            else:
                temps.append(float(cfg.temperature))
                top_ks.append(int(cfg.top_k) if cfg.top_k and cfg.top_k > 0 else vocab)
                top_ps.append(float(cfg.top_p) if cfg.top_p else 1.0)

        # Pad to padded_bs with the last entry so every slot is well-defined.
        if not temps:
            # No requests somehow; just use defaults across all padding slots.
            temps = [1.0] * padded_bs
            top_ks = [vocab] * padded_bs
            top_ps = [1.0] * padded_bs
        else:
            while len(temps) < padded_bs:
                temps.append(temps[-1])
                top_ks.append(top_ks[-1])
                top_ps.append(top_ps[-1])

        shared = self._shared_bufs
        shared["temperature_buf"][:padded_bs].copy_(
            torch.tensor(temps, dtype=torch.float32, device=self.device)
        )
        shared["top_k_buf"][:padded_bs].copy_(
            torch.tensor(top_ks, dtype=torch.int32, device=self.device)
        )
        shared["top_p_buf"][:padded_bs].copy_(
            torch.tensor(top_ps, dtype=torch.float32, device=self.device)
        )
        # randomly initialize seed buffer
        shared["seed_buf"][:padded_bs].copy_(
            torch.randint(0, 2**32, (padded_bs,), dtype=torch.long, device=self.device)
        )
        shared["offset_buf"][:] = 0


class CodePredictorEngine(AREngine):
    def engine_type(self) -> EngineType:
        return EngineType.CODE_PREDICTOR

    def has_autocast(self):
        return False

    def warmup(self) -> None:
        """Capture the unrolled depth graph for each registered submodule."""
        for node_name, submodule_mgmt in self.submodule_management.items():
            runner = CodePredictorCudaGraphRunner(
                submodule=submodule_mgmt.submodule,
                sampler=submodule_mgmt.sampler,
                device=self.device,
            )
            runner.warmup_and_capture()
            if runner.graphs:
                submodule_mgmt.cuda_graph_runner = runner
                logger.info(
                    "CodePredictorEngine: unrolled graph runner attached to %s (%d buckets)",
                    node_name, len(runner.graphs),
                )

    def _execute_batched(self, batch: NodeBatch, submodule) -> NodeOutput:
        cache_manager = self._create_cache_manager(
            batch.request_ids, batch.node_name
        )

        for rid in batch.request_ids:
            cache_manager.reset_state(
                request_id=rid,
            )

        rids = list(batch.per_request_input_tensors.keys())
        seq_lens = {
            rid: cache_manager._get_state(rid, "main").seq_len for rid in rids
        }
        logger.debug("Execute batched %s", seq_lens)
        input_tensors = [batch.per_request_input_tensors[rid] for rid in rids]

        if self.enable_nvtx:
            range_push("code_pred.batched.preprocess", synchronize=True)
        preprocessed = submodule.preprocess(
            graph_walk=batch.graph_walk,
            cache_manager=cache_manager,
            per_request_inputs=input_tensors,
            request_ids=rids,
            per_request_info=batch.per_request_info,
        )
        if self.enable_nvtx:
            range_pop(synchronize=True)

        if self.enable_nvtx:
            range_push("code_pred.batched.forward")

        sampler = self.submodule_management[batch.node_name].sampler
        cuda_graph_runner = self.submodule_management[batch.node_name].cuda_graph_runner
        batched_output = submodule.forward_batched(
            graph_walk=batch.graph_walk,
            cache_manager=cache_manager,
            packed_inputs=preprocessed,
            sampler=sampler,
            cuda_graph_runner=cuda_graph_runner,
            request_ids=rids,
            per_request_info=batch.per_request_info,
        )
        output = NodeOutput(per_request_output_tensors=batched_output)
        if self.enable_nvtx:
            range_pop()
        return output

    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        if self.enable_nvtx:
            range_push(
                f"engine.code_pred.{batch.node_name}.{batch.graph_walk}"
                f".bs{len(batch.request_ids)}"
            )

        submod_mgmt = self.submodule_management[batch.node_name]
        submodule = submod_mgmt.submodule
        if self.enable_nvtx:
            range_push("code_pred.sampler_config", synchronize=True)
        for rid, info in batch.per_request_info.items():
            sampling_config = info.sampling_config.get(batch.node_name)
            sampling_config = {} if sampling_config is None else asdict(sampling_config)
            submod_mgmt.sampler.set_config(rid, **sampling_config)
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # NO autocast for float32 Code Predictor inference. HF and
        # vllm-omni found that fused/autocast kernels degrade audio quality
        # for the small (5-layer) Code Predictor.
        with torch.no_grad():
            output = self._execute_batched(batch, submodule)
            for rid, info in batch.per_request_info.items():
                submodule.postprocess(
                    request_id=rid,
                    request_info=info,
                    outputs=output.per_request_output_tensors.get(rid, {}),
                )
            if self.enable_nvtx:
                range_pop(synchronize=True)
            return output

    def check_ready(self, *args, **kwargs):
        return True
