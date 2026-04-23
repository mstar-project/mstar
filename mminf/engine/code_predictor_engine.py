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

from abc import abstractmethod
import bisect
import logging
from dataclasses import asdict, dataclass

import torch

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.ar_engine import AREngine
from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.engine.kv_store import KVCacheConfig, PositionInfo, TransferEngineInfo
from mminf.model.submodule_base import ARNodeInputs, ARNodeSubmodule, CudaGraphConfig, ModelInputsFromEngine
from mminf.utils.profiler import range_pop, range_push
from mminf.utils.sampling import Sampler, SamplingConfig, sample_depth_gpu

logger = logging.getLogger(__name__)


# TODO: this all feels a bit hacky and not generalizable. Ideally, we would
# like to modify our system to deal with the code predictor paradigm without
# making a code predictor engine (e.g., having an abstraction for injecting
# the cuda graph runner into the submodule execution path).


@dataclass
class MTPSampler:
    temperature_buf: torch.Tensor
    top_k_buf: torch.Tensor
    top_p_buf: torch.Tensor
    seed_buf: torch.Tensor
    offset_buf: torch.Tensor

    @torch.compiler.disable
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        codes = sample_depth_gpu(
            logits, self.temperature_buf,
            self.top_k_buf, self.top_p_buf,
            self.seed_buf, self.offset_buf
        )
        self.offset_buf += 1
        return codes


@dataclass
class CodePredictorEngineInputs(ModelInputsFromEngine):
    # These just have defaults so that CodePredictorEngineInputs can inherit
    # from ModelInputsFromEngine, but all of these values should be filled in,
    # and are expected to be downstream
    sampler: MTPSampler | None = None
    kv_cache: torch.Tensor | None = None
    init_pos_ids: torch.Tensor | None = None


class CodePredictorSubmodule(ARNodeSubmodule):
    @abstractmethod
    def get_num_code_groups(self):
        pass

    @abstractmethod
    def get_kv_cache_dtype(self):
        return torch.float32

    @abstractmethod
    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: CodePredictorEngineInputs,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: CodePredictorEngineInputs,
        last_hidden: torch.Tensor | None = None,
        layer0_codes: torch.Tensor | None = None,
        all_codes: torch.Tensor | None = None,
        codec_emb_sum: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        pass

    def forward(
        self,
        *args,
        **kwargs,
    ):
        return next(iter(
            self.forward_batched(*args, **kwargs).values()
        ))
    
    def can_batch(self, batch: NodeBatch, model_inputs: list[ARNodeInputs]) -> bool:
        return True

    def get_needed_cache_labels(
        self,
        graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str]:
        return ["main"]
    

@dataclass
class UnrolledGraphData:
    """State captured for a single batch-size bucket."""

    graph: torch.cuda.CUDAGraph
    bs: int
    kv_cache_slice: torch.Tensor
    pos_id_slice: torch.Tensor
    inputs_slices: dict[str, torch.Tensor]
    static_outputs: list[dict[str, torch.Tensor]]


@dataclass
class MTPSamplerBuffers:
    """Pre-allocated static buffers for graph-safe MTP sampling.

    All tensors are sized to ``max_batch_size``. Slice with
    ``slice_for_bs(bs)`` to get a view suitable for a specific batch.
    """
    max_batch_size: int
    temperature_buf: torch.Tensor   # [max_bs], float32
    top_k_buf: torch.Tensor         # [max_bs], int32
    top_p_buf: torch.Tensor         # [max_bs], float32
    seed_buf: torch.Tensor          # [max_bs], int64
    offset_buf: torch.Tensor        # [max_bs], int64

    @classmethod
    def allocate(
        cls,
        max_batch_size: int,
        device: torch.device,
    ) -> "MTPSamplerBuffers":
        """Allocate zero-initialised sampling buffers for ``max_batch_size``.
        """
        temperature_buf = torch.ones(max_batch_size, dtype=torch.float32, device=device)
        top_k_buf = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        top_p_buf = torch.ones(max_batch_size, dtype=torch.float32, device=device)
        seed_buf = torch.zeros(max_batch_size, dtype=torch.long, device=device)
        offset_buf = torch.zeros(max_batch_size, dtype=torch.long, device=device)
        return cls(
            max_batch_size=max_batch_size,
            temperature_buf=temperature_buf,
            top_k_buf=top_k_buf,
            top_p_buf=top_p_buf,
            seed_buf=seed_buf,
            offset_buf=offset_buf,
        )

    def slice_for_bs(self, bs: int) -> dict[str, torch.Tensor]:
        """Return bs-sized views into each buffer (zero-copy slices)."""
        return {
            "temperature_buf": self.temperature_buf[:bs],
            "top_k_buf": self.top_k_buf[:bs],
            "top_p_buf": self.top_p_buf[:bs],
            "seed_buf": self.seed_buf[:bs],
            "offset_buf": self.offset_buf[:bs],
        }


def _build_sampling_lists(
    request_ids: list[str],
    sampling_configs: dict[str, SamplingConfig],
    padded_bs: int,
    device: torch.device
):
    """Resolve per-request sampling config into flat Python lists of length ``padded_bs``.
    """
    temps = torch.ones(padded_bs, device=device)
    # in the sampler, topk=0 maps to unrestricted top k
    top_ks = torch.zeros(padded_bs, dtype=torch.int32, device=device)
    top_ps = torch.ones(padded_bs, device=device)

    for i, rid in enumerate(request_ids):
        cfg = sampling_configs.get(rid, SamplingConfig())
        if cfg.temperature > 0:
            temps[i] = float(cfg.temperature)
            top_ks[i] = int(cfg.top_k)
            top_ps[i] = float(cfg.top_p) if cfg.top_p else 1.0
        # otherwise, use defaults already filled in
    return temps, top_ks, top_ps, \
        torch.randint(0, 2**32, (padded_bs,), dtype=torch.long, device=device)


def make_mtp_sampler_from_buffers(
    bufs: MTPSamplerBuffers,
    request_ids: list[str],
    sampling_configs: dict[str, SamplingConfig],
    padded_bs: int,

) -> MTPSampler:
    assert padded_bs <= bufs.max_batch_size, (
        f"padded_bs={padded_bs} exceeds MTPSamplerBuffers.max_batch_size={bufs.max_batch_size}"
    )

    temps, top_ks, top_ps, seed = _build_sampling_lists(
        request_ids, sampling_configs, padded_bs,
        device=bufs.temperature_buf.device
    )
    bufs.temperature_buf[:padded_bs].copy_(temps)
    bufs.top_k_buf[:padded_bs].copy_(top_ks)
    bufs.top_p_buf[:padded_bs].copy_(top_ps)
    bufs.seed_buf[:padded_bs].copy_(seed)
    bufs.offset_buf[:padded_bs].zero_()
    slices = bufs.slice_for_bs(padded_bs)
    return MTPSampler(**slices)


def make_mtp_sampler_eager(
    request_ids: list[str],
    sampling_configs: dict,
    device: torch.device,
) -> MTPSampler:
    bs = len(request_ids)
    temps, top_ks, top_ps, seed = _build_sampling_lists(
        request_ids, sampling_configs, padded_bs=bs,
        device=device
    )
    return MTPSampler(
        temperature_buf=temps,
        top_k_buf=top_ks,
        top_p_buf=top_ps,
        seed_buf=seed,
        offset_buf=torch.zeros(bs, dtype=torch.long, device=device),
    )


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

    # CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]
    CAPTURE_BATCH_SIZES = [1, 2] # TODO DEBUG

    def __init__(
        self,
        submodule: CodePredictorSubmodule,
        node_name: str,
        kv_cache: torch.Tensor,
        device: torch.device,
    ):
        # Duck-typed on Qwen3OmniCodePredictorSubmodule: we rely on the
        # attributes documented below so this runner stays agnostic to the
        # specific subclass.
        self.submodule = submodule
        self.node_name = node_name
        self.device = device
        self.max_batch_size = max(self.CAPTURE_BATCH_SIZES)

        self.graphs: dict[int, UnrolledGraphData] = {}
        self.memory_pool = None

        self._shared_bufs: dict | None = None

        # shared buffers
        self.mtp_sampling_buf = MTPSamplerBuffers.allocate(
            max_batch_size=self.max_batch_size, device=device
        )
        self.init_pos_ids_buf = torch.zeros(
            self.max_batch_size, device=device, dtype=torch.long
        )

        self.kv_cache = kv_cache

        # lazily initialized via "preprocess" on max batch size
        self.fwd_input_buffers: dict[str, torch.Tensor] | None = None

    def _slice_inputs_for_bs(self, bs: int) -> dict:
        return {
            key: (
                val[:bs] if isinstance(val, torch.Tensor) else val
            ) for key, val in self.fwd_input_buffers.items()
        }

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

        self.memory_pool = torch.cuda.graphs.graph_pool_handle()
        torch.cuda.set_device(self.device)

        configs = self.submodule.get_cuda_graph_configs(self.device)
        if len(configs) != 1:
            raise NotImplementedError("CodePredictor engine does not support multiple cuda graph configs yet.")
        capture_config = configs[0]

        if len(capture_config.dummy_capture_inputs) != 1:
            raise NotImplementedError(
                "CodePredictor engine currently only supports one dummy input set per graph config."
            )

        # Capture largest-first so the shared memory pool allocations never
        # need to grow after a smaller bucket has already reserved addresses.
        for bs in reversed(self.CAPTURE_BATCH_SIZES):
            try:
                self._capture_one(bs, capture_config)
                logger.info(
                    "CodePredictorCudaGraphRunner: captured unrolled graph for bs=%d",
                    bs,
                )
            except Exception:
                logger.warning(
                    "Failed to capture unrolled depth graph for bs=%d",
                    bs, exc_info=True,
                )
    def _make_dummy_fwd_info(self, bs, graph_walk: str):
        dummy_rids = [
            f"__mtp_{graph_walk}_{i}__" for i in range(bs)
        ]
        return {
            rid: CurrentForwardPassInfo(
                request_id=rid,
                graph_walk=graph_walk,
                requires_cfg=False,
                fwd_index=0,
                random_seed=0,
                max_tokens=1,
                sampling_config={}
            ) for rid in dummy_rids
        }, dummy_rids

    def _capture_one(self, bs: int, config: CudaGraphConfig) -> None:
        dummy_fwd_info, dummy_rids = self._make_dummy_fwd_info(bs, config.capture_graph_walk)
        engine_inputs = CodePredictorEngineInputs(
            request_ids=dummy_rids,
            per_request_info=dummy_fwd_info,
            sampler=make_mtp_sampler_from_buffers(
                bufs=self.mtp_sampling_buf,
                request_ids=[], sampling_configs={},
                padded_bs=bs
            ),
            kv_cache=self.kv_cache[:, :bs],
            init_pos_ids=self.init_pos_ids_buf[:bs]
        )

        if self.fwd_input_buffers is None:
            # allocate buffers on the first capture_one, which is on the max bs
            self.fwd_input_buffers = self.submodule.preprocess(
                graph_walk=config.capture_graph_walk,
                engine_inputs=engine_inputs,
                inputs=[
                    config.dummy_capture_inputs[0].clone() \
                        for _ in range(bs)
                ]
            )
        fwd_inputs = self._slice_inputs_for_bs(bs)

        _run_unrolled_loop = self.submodule.forward_batched
        if config.compile:
            _run_unrolled_loop = torch.compile(
                _run_unrolled_loop,
                mode="max-autotune-no-cudagraphs",
                fullgraph=False,
                dynamic=False,
            )

        # Warmup: 3 full passes to trigger lazy kernel compiles and stabilize
        # the memory pool (matches vox-serve's 3-iter warmup).
        torch.cuda.synchronize()
        for _ in range(3):
            _run_unrolled_loop(
                graph_walk=config.capture_graph_walk,
                engine_inputs=engine_inputs,
                **fwd_inputs,
            )
            self.init_pos_ids_buf.zero_()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.memory_pool):
            static_outputs = _run_unrolled_loop(
                graph_walk=config.capture_graph_walk,
                engine_inputs=engine_inputs,
                **fwd_inputs,
            ) # dummy req_id -> output dict
            self.init_pos_ids_buf.zero_()
        torch.cuda.synchronize()

        # A couple of replay passes to let any pool-internal state settle
        # before the first real replay (matches vox-serve).
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()

        # print out fwd_inputs and static_outputs memory addrs
        for key, tensor in fwd_inputs.items():
            logger.info(
                f"Captured graph input '{key}': shape={tensor.shape}, "
                f"dtype={tensor.dtype}, device={tensor.device}, "
                f"data_ptr={tensor.data_ptr()}"
            )
        for rid in dummy_rids:
            output = static_outputs[rid]
            for key, tensors in output.items():
                tensor = tensors[0]
                logger.info(
                    f"Captured graph output '{key}' for rid '{rid}': "
                    f"shape={tensor.shape}, dtype={tensor.dtype}, "
                    f"device={tensor.device}, data_ptr={tensor.data_ptr()}"
                )

        self.graphs[bs] = UnrolledGraphData(
            graph=graph,
            bs=bs,
            kv_cache_slice=engine_inputs.kv_cache,
            pos_id_slice=engine_inputs.init_pos_ids,
            inputs_slices=fwd_inputs,
            static_outputs=[
                static_outputs[rid] for rid in dummy_rids
            ]
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
        inputs: list[ARNodeInputs],
        per_request_info: dict[str, CurrentForwardPassInfo],
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
        graph_data.pos_id_slice.zero_()

        print(inputs)

        # Build inputs to preprocess and forward_batched
        dummy_fwd_info, dummy_rids = self._make_dummy_fwd_info(padded_bs, graph_walk)
        augmented_request_ids = dummy_rids[:]
        augmented_request_ids[:bs] = request_ids
        augmented_fwd_info = {
            **per_request_info,
            **dummy_fwd_info,
        }
        sampler = make_mtp_sampler_from_buffers(
            bufs=self.mtp_sampling_buf,
            request_ids=request_ids,
            sampling_configs={
                rid: info.sampling_config.get(
                    self.node_name, SamplingConfig()
                ) for rid, info in dummy_fwd_info.items()
            },
            padded_bs=padded_bs,
        )
        engine_inputs = CodePredictorEngineInputs(
            request_ids=augmented_request_ids,
            per_request_info=augmented_fwd_info,
            sampler=sampler,
            kv_cache=graph_data.kv_cache_slice,
            init_pos_ids=graph_data.pos_id_slice,
        )

        # Preprocess and copy inputs into the shared buffers.
        preprocessed = self.submodule.preprocess(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            inputs=inputs,
        )
        buffers = graph_data.inputs_slices
        for key, tensor in preprocessed.items():
            if key not in buffers:
                raise KeyError(f"Preprocess output key '{key}' not found in captured graph inputs")
            buf = buffers[key]
            if tensor.shape != buf.shape:
                raise ValueError(
                    f"Shape mismatch for input '{key}': "
                    f"preprocessed shape {tensor.shape} vs "
                    f"captured buffer shape {buf.shape}"
                )
            buf.copy_(tensor)
        print(buffers)

        # --- Step 4: replay. ---
        graph_data.graph.replay()

        torch.cuda.synchronize()
        # print out fwd_inputs and static_outputs memory addrs
        for key, tensor in buffers.items():
            logger.info(
                f"Replayed graph input '{key}': shape={tensor.shape}, "
                f"dtype={tensor.dtype}, device={tensor.device}, "
                f"data_ptr={tensor.data_ptr()}"
            )
        for i in range(len(dummy_rids)):
            output = graph_data.static_outputs[i]
            for key, tensors in output.items():
                tensor = tensors[0]
                logger.info(
                    f"Received graph output '{key}': "
                    f"shape={tensor.shape}, dtype={tensor.dtype}, "
                    f"device={tensor.device}, data_ptr={tensor.data_ptr()}"
                )

        # return req_id -> output tensor dict for the real batch (ignore padding slots)
        outputs = {}
        for i, rid in enumerate(request_ids):
            outputs[rid] = graph_data.static_outputs[i]
            print(outputs[rid])

            # clone outputs to detach from the static buffers
            # (which will be overwritten on the next call)
            for key, val in outputs[rid].items():
                if isinstance(val, torch.Tensor):
                    outputs[rid][key] = val.clone()
                elif isinstance(val, list):
                    outputs[rid][key] = [
                        v.clone() if isinstance(v, torch.Tensor) \
                            else v for v in val
                    ]
        return outputs


class CodePredictorEngine(BaseEngine):
    def __init__(
        self,
        autocast_dtype=torch.bfloat16,
        enable_nvtx: bool = False,
    ):
        super().__init__(enable_nvtx=enable_nvtx)
        self.cuda_graph_runners: dict[str, CodePredictorCudaGraphRunner] = {}
        self.submodules: dict[str, CodePredictorSubmodule] = {}
        self.kv_caches: dict[str, torch.Tensor] = {}
        self._max_batch_size = max(
            CodePredictorCudaGraphRunner.CAPTURE_BATCH_SIZES
        )

    def engine_type(self) -> EngineType:
        return EngineType.CODE_PREDICTOR

    def has_autocast(self):
        return False
    
    def load_model(
        self,
        submodules: dict[str, CodePredictorSubmodule],
        kv_cache_config: list[KVCacheConfig],
        device: torch.device,
        **kwargs
    ) -> None:
        self.device = device
        node_name_to_kv_cfg: dict[str, KVCacheConfig] = {}
        for cfg in kv_cache_config:
            for node_name in cfg.nodes or submodules.keys():
                node_name_to_kv_cfg[node_name] = cfg
        
        self.submodules = submodules
        for node_name, submodule in submodules.items():
            kv_cfg = node_name_to_kv_cfg.get(node_name)
            if kv_cfg is None:
                raise ValueError(f"No KV cache config found for node '{node_name}'")

            kv_cache = torch.zeros(
                (kv_cfg.num_layers, self._max_batch_size,
                2, submodule.get_num_code_groups(),
                kv_cfg.num_kv_heads, kv_cfg.head_dim),
                dtype=submodule.get_kv_cache_dtype(),
                device=device,
            )
            self.kv_caches[node_name] = kv_cache


    def warmup(self) -> None:
        """Capture the unrolled depth graph for each registered submodule."""
        for node_name, submodule in self.submodules.items():
            runner = CodePredictorCudaGraphRunner(
                submodule=submodule,
                node_name=node_name,
                kv_cache=self.kv_caches[node_name],
                device=self.device,
            )
            runner.warmup_and_capture()
            if runner.graphs:
                self.cuda_graph_runners[node_name] = runner
                logger.info(
                    "CodePredictorEngine: unrolled graph runner attached to %s (%d buckets)",
                    node_name, len(runner.graphs),
                )

    def get_max_batch_size(self):
        return self._max_batch_size

    def _execute_batched(
        self, batch: NodeBatch,
        inputs: list[ARNodeInputs],
        submodule: CodePredictorSubmodule
    ) -> NodeOutput:
        self.kv_caches[batch.node_name].zero_()
        if batch.node_name in self.cuda_graph_runners:
            return NodeOutput(
                per_request_output_tensors=self.cuda_graph_runners[batch.node_name].run(
                    graph_walk=batch.graph_walk,
                    request_ids=batch.request_ids,
                    inputs=inputs,
                    per_request_info=batch.per_request_info
                )
            )
        
        if self.enable_nvtx:
            range_push("code_pred.batched.preprocesss", synchronize=True)
        bs = len(batch.request_ids)
        engine_inputs=CodePredictorEngineInputs(
            request_ids=batch.request_ids,
            per_request_info=batch.per_request_info,
            sampler=make_mtp_sampler_eager(
                batch.request_ids, sampling_configs={
                    rid: info.sampling_config.get(
                        batch.node_name, SamplingConfig()
                    ) for rid, info in batch.per_request_info.items()
                },
                device=self.device
            ),
            kv_cache=self.kv_caches[batch.node_name][:, :bs],
            init_pos_ids=torch.zeros(
                bs, device=self.device,
                dtype=torch.long
            )
        )
        preprocessed = submodule.preprocess(
            graph_walk=batch.graph_walk,
            engine_inputs=engine_inputs,
            inputs=inputs
        )
        print(preprocessed)

        if self.enable_nvtx:
            range_pop(synchronize=True)

        if self.enable_nvtx:
            range_push("code_pred.batched.forward")
        
        batched_output = submodule.forward_batched(
            graph_walk=batch.graph_walk,
            engine_inputs=engine_inputs,
            **preprocessed
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

        submodule = self.submodules[batch.node_name]
        if self.enable_nvtx:
            range_push("code_pred.prepare_iputs", synchronize=True)
        
        node_inputs: list[ARNodeInputs] = []
        for rid in batch.request_ids:
            node_inputs.append(
                submodule.prepare_inputs(
                    graph_walk=batch.graph_walk,
                    fwd_info=batch.per_request_info[rid],
                    inputs=batch.per_request_input_tensors[rid],
                )
            )
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # NO autocast for float32 Code Predictor inference. HF and
        # vllm-omni found that fused/autocast kernels degrade audio quality
        # for the small (5-layer) Code Predictor.
        with torch.no_grad():
            output = self._execute_batched(
                batch, node_inputs, submodule
            )
            for rid, info in batch.per_request_info.items():
                submodule.postprocess(
                    request_id=rid,
                    request_info=info,
                    outputs=output.per_request_output_tensors.get(rid, {}),
                )
            if self.enable_nvtx:
                range_pop(synchronize=True)
            return output

    def remove_request(self, request_id: str) -> None:
        for submodule in self.submodules.values():
            submodule.cleanup_request(request_id)

    def add_request(self, request_id, **kwargs):
        return # no persistent state