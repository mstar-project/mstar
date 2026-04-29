import argparse
import asyncio
import os
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import aiohttp

from benchmark.base import Model, ModelType, RequestType
from benchmark.dataset import (
    BaseDataset,
    Food101Dataset,
    LibriSpeechDataset,
    TxtFileDataset,
    UCF101Dataset,
    VBenchDataset,
)
from benchmark.request import (
    AggregateMetrics,
    InferenceSystem,
    OurSystem,
    RequestInput,
    RequestMetrics,
    SGLangOmni,
    VLLMOmni,
    VoxServe,
    aggregate_metrics,
)


class DatasetType(Enum):
    VBENCH = "vbench"
    TEXT = "text"
    LIBRI = "libri"
    FOOD = "food101"
    UCF = "ucf101"


class InferenceSystemType(Enum):
    OURS = "ours"
    VLLM_OMNI = "vllm_omni"
    VOX_SERVE = "vox_serve"
    SGLANG_OMNI = "sglang_omni"

    def instantiate(self) -> InferenceSystem:
        if self == InferenceSystemType.OURS:
            return OurSystem()
        elif self == InferenceSystemType.VLLM_OMNI:
            return VLLMOmni()
        elif self == InferenceSystemType.VOX_SERVE:
            return VoxServe()
        elif self == InferenceSystemType.SGLANG_OMNI:
            return SGLangOmni()


class ProfilingType(Enum):
    OFFLINE = "offline"  # strict-batch waves of size B (existing)
    CLOSED_LOOP = "closed_loop"  # semaphore-bounded continuous (Fix 9)
    ONLINE = "online"  # Poisson at fixed rate (existing)


@dataclass
class BenchmarkConfig:
    url: str
    model: Model
    dataset: DatasetType
    num_requests: int
    request_type: RequestType
    local_cache_dir: str
    num_warmup: int = 3
    profiling_type: ProfilingType = ProfilingType.OFFLINE
    inference_system: InferenceSystemType = InferenceSystemType.OURS
    verbose: bool = False

    batch_size: Optional[int] = 1
    rate: Optional[float] = 1
    # Max in-flight requests for CLOSED_LOOP mode (semaphore cap). Ignored for
    # OFFLINE / ONLINE. Default 1 = sequential, matching sglang-omni's default.
    max_concurrency: Optional[int] = 1
    output_dir: Optional[str] = None  # Save outputs here (text files / images)
    output_json: Optional[str] = None  # Write metrics JSON here
    max_tokens: Optional[int] = None  # Max output tokens for text generation
    # VBench args
    vbench_cache_dir: str = "./vbench_cache"

    # text dataset args
    request_txt_file: str = "benchmark/assets/simple_text_queries.txt"


class Benchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.inference_system = config.inference_system.instantiate()

    def _get_dataset(self) -> BaseDataset:
        if self.config.dataset == DatasetType.VBENCH:
            return VBenchDataset(
                cache_dir=self.config.vbench_cache_dir,
                task=self.config.request_type,
                num_requests=self.config.num_requests,
            )
        elif self.config.dataset == DatasetType.TEXT:
            return TxtFileDataset(
                filename=self.config.request_txt_file,
                num_requests=self.config.num_requests,
                req_type=self.config.request_type,
            )
        elif self.config.dataset == DatasetType.LIBRI:
            return LibriSpeechDataset(
                num_requests=self.config.num_requests,
                req_type=self.config.request_type,
                local_file_dir=self.config.local_cache_dir,
            )
        elif self.config.dataset == DatasetType.FOOD:
            return Food101Dataset(num_requests=self.config.num_requests, req_type=self.config.request_type)
        elif self.config.dataset == DatasetType.UCF:
            # TODO: this is the dataset that vllm-omni reports using, so we have it as an example,
            # but it only has two videos that we're just alternating between... We should replace
            # this with a better dataset
            return UCF101Dataset(
                num_requests=self.config.num_requests,
                req_type=self.config.request_type,
                local_file_dir=self.config.local_cache_dir,
            )
        raise ValueError(f"Unknown dataset: {self.config.dataset}")

    def _save_outputs(self, metrics: list[RequestMetrics]) -> None:
        """Save outputs to disk (after timing). Text → .txt, images → .png."""
        output_dir = self.config.output_dir
        if output_dir is None:
            return

        os.makedirs(output_dir, exist_ok=True)
        saved = 0
        for m in metrics:
            saved += m.write_files(output_dir)
        if saved:
            print(f"\nSaved {saved} outputs to {output_dir}/")

    def _print_errors(self, metrics: list[RequestMetrics]) -> None:
        errors = [(m.request_id, m.error) for m in metrics if m.error is not None]
        if not errors:
            return
        print(f"\n--- Errors ({len(errors)}/{len(metrics)}) ---")
        for request_id, error in errors:
            print(f"  [{request_id}] {error}")

    def _max_tokens_kwargs(self) -> dict:
        """Build additional_model_kwargs for max_tokens control."""
        if self.config.max_tokens is None:
            return {}
        # mminf uses max_output_tokens; vllm/sglang use max_tokens
        if self.config.inference_system == InferenceSystemType.OURS:
            return {"max_output_tokens": self.config.max_tokens}
        return {"max_tokens": self.config.max_tokens}

    async def _run_batch(
        self,
        session: aiohttp.ClientSession,
        batch: list[tuple[int, RequestInput]],
    ) -> list[RequestMetrics]:
        """Run a single batch of requests concurrently."""
        tasks = [
            asyncio.create_task(
                self.inference_system.send_request(
                    session=session,
                    req_input=req,
                    base_url=self.config.url,
                    request_id=i,
                    model=self.config.model,
                    additional_model_kwargs=self._max_tokens_kwargs(),
                )
            )
            for i, req in batch
        ]
        return list(await asyncio.gather(*tasks))

    async def _run_concurrent_offline(
        self,
        session: aiohttp.ClientSession,
        requests: list[RequestInput],
    ) -> list[RequestMetrics]:
        bs = self.config.batch_size
        all_metrics: list[RequestMetrics] = []

        batches = [
            [(i, requests[i]) for i in range(start, min(start + bs, len(requests)))]
            for start in range(0, len(requests), bs)
        ]

        for batch in batches:
            tic = time.perf_counter()
            metrics = await self._run_batch(session, batch)
            toc = time.perf_counter()
            if self.config.verbose:
                print(toc - tic)
            all_metrics.extend(metrics)
            await asyncio.sleep(0.01)

        return all_metrics

    async def _run_concurrent_online(
        self,
        session: aiohttp.ClientSession,
        requests: list[RequestInput],
    ) -> list[RequestMetrics]:
        tasks: list[asyncio.Task] = []
        for i, req in enumerate(requests):
            task = asyncio.create_task(
                self.inference_system.send_request(
                    session=session,
                    req_input=req,
                    base_url=self.config.url,
                    request_id=i,
                    model=self.config.model,
                    additional_model_kwargs=self._max_tokens_kwargs(),
                )
            )
            tasks.append(task)
            if i < len(requests) - 1:
                # Poisson inter-request times
                interval = random.expovariate(self.config.rate)
                await asyncio.sleep(interval)
        return list(await asyncio.gather(*tasks))

    async def _run_concurrent_closed_loop(
        self,
        session: aiohttp.ClientSession,
        requests: list[RequestInput],
    ) -> list[RequestMetrics]:
        """Closed-loop continuous: keep `max_concurrency` requests in flight at all times.

        Pattern matches `sglang-omni/benchmarks/benchmarker/runner.py:83-110`:
        create all tasks upfront, semaphore-gate each one. As soon as one
        finishes, another is admitted — eliminating the tail-of-wave GPU idle
        that strict-batch (`OFFLINE`) suffers from at high B.
        """
        n = self.config.max_concurrency or 1
        sem = asyncio.Semaphore(n)

        async def _limited(request_id: int, req: RequestInput) -> RequestMetrics:
            async with sem:
                return await self.inference_system.send_request(
                    session=session,
                    req_input=req,
                    base_url=self.config.url,
                    request_id=request_id,
                    model=self.config.model,
                    additional_model_kwargs=self._max_tokens_kwargs(),
                )

        tasks = [asyncio.create_task(_limited(i, r)) for i, r in enumerate(requests)]
        return list(await asyncio.gather(*tasks))

    async def _warmup(
        self,
        session: aiohttp.ClientSession,
        requests: list[RequestInput],
    ) -> None:
        """Fire warmup requests at the same firing pattern used for measurement.

        Sequential warmup on a concurrent measurement path leaves the first
        measured wave hitting cold concurrency code paths (KV-page allocation
        for the bigger shape, scheduler queues, CUDA-graph misses). Warmup
        cadence must match measurement cadence.
        """
        if self.config.num_warmup == 0 or not requests:
            return

        if self.config.profiling_type == ProfilingType.OFFLINE:
            wave_size = max(1, self.config.batch_size or 1)
        elif self.config.profiling_type == ProfilingType.CLOSED_LOOP:
            wave_size = max(1, self.config.max_concurrency or 1)
        else:  # ONLINE
            wave_size = 1

        # Replicate the first `wave_size` requests `num_warmup` times so we
        # always have enough payloads even when the dataset is smaller than
        # the wave size.
        seed = requests[: max(1, wave_size)]
        warmup_total = wave_size * self.config.num_warmup
        warmup_reqs = [seed[i % len(seed)] for i in range(warmup_total)]

        if self.config.verbose:
            print(f"--- Warmup ({self.config.profiling_type.value}, {warmup_total} request(s), wave={wave_size}) ---")

        if self.config.profiling_type == ProfilingType.OFFLINE:
            for w in range(self.config.num_warmup):
                batch = [(-(w * wave_size + i + 1), warmup_reqs[w * wave_size + i]) for i in range(wave_size)]
                await self._run_batch(session, batch)
        elif self.config.profiling_type == ProfilingType.CLOSED_LOOP:
            await self._run_concurrent_closed_loop(session, warmup_reqs)
        else:
            await self._run_concurrent_online(session, warmup_reqs)

        if self.config.verbose:
            print("Warmup complete.")

    async def run(self) -> tuple[list[RequestMetrics], AggregateMetrics]:
        dataset = self._get_dataset()
        if self.config.profiling_type == ProfilingType.OFFLINE:
            bs = self.config.batch_size
            # make even multiple of batch size
            self.config.num_requests = ((self.config.num_requests + bs - 1) // bs) * bs
        requests = dataset.get_requests()[: self.config.num_requests]

        # Bump the connection-pool cap so closed-loop runs at high
        # max_concurrency don't bottleneck on aiohttp's default 100/host limit.
        connector_limit = max(100, (self.config.max_concurrency or 1) + 10)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            connector=aiohttp.TCPConnector(limit=connector_limit),
            read_bufsize=5 * 2**20,  # 1MB read buffer
        ) as session:
            await self._warmup(session, requests)

            wall_start = time.monotonic()

            if self.config.profiling_type == ProfilingType.OFFLINE:
                metrics = await self._run_concurrent_offline(session, requests)
            elif self.config.profiling_type == ProfilingType.CLOSED_LOOP:
                metrics = await self._run_concurrent_closed_loop(session, requests)
            else:
                metrics = await self._run_concurrent_online(session, requests)

            wall_time = time.monotonic() - wall_start

        agg = aggregate_metrics(
            metrics,
            wall_time=wall_time,
            online=self.config.profiling_type == ProfilingType.ONLINE,
            batch_size=self.config.batch_size,
            rate=self.config.rate,
            max_concurrency=self.config.max_concurrency,
            profiling_type=self.config.profiling_type.value,
            model=self.config.model,
        )

        print(f"\n--- Benchmark Results (wall time: {wall_time:.2f}s) ---")
        print(agg)
        self._print_errors(metrics)
        self._save_outputs(metrics)
        self._save_json(metrics, agg, wall_time)

        return metrics, agg

    def _save_json(self, metrics: list[RequestMetrics], agg: AggregateMetrics, wall_time: float) -> None:
        if self.config.output_json is None:
            return
        import json as _json
        result = {
            "system": self.config.inference_system.value,
            "model": self.config.model.get_hf_url(),
            "task": self.config.request_type.value,
            "batch_size": self.config.batch_size,
            "num_requests": self.config.num_requests,
            "num_warmup": self.config.num_warmup,
            "max_tokens": self.config.max_tokens,
            "wall_time": wall_time,
            "n_success": agg.n_success,
            "summary": {
                "ttft": {mod: {"mean": s.mean, "p50": s.p50, "p95": s.p95, "p99": s.p99}
                         for mod, s in agg.ttft.items()},
                "e2e": {"mean": agg.e2e_latency.mean, "p50": agg.e2e_latency.p50,
                        "p95": agg.e2e_latency.p95, "p99": agg.e2e_latency.p99},
                "itl": {mod: {"mean": s.mean, "p50": s.p50, "p95": s.p95, "p99": s.p99}
                        for mod, s in agg.itl.items()},
                "throughput_req_s": agg.n_success / wall_time if wall_time > 0 else 0,
                "throughput_tok_s": agg.total_text_tokens / wall_time if wall_time > 0 else 0,
                "total_text_tokens": agg.total_text_tokens,
                "mean_text_tokens": agg.mean_text_tokens,
            },
        }
        os.makedirs(os.path.dirname(self.config.output_json) or ".", exist_ok=True)
        with open(self.config.output_json, "w") as f:
            _json.dump(result, f, indent=2)
        print(f"\nResults saved to {self.config.output_json}")


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run inference benchmark")
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True, choices=[m.value for m in ModelType])
    parser.add_argument(
        "--inference-system", choices=[s.value for s in InferenceSystemType], default=InferenceSystemType.OURS.value
    )
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument(
        "--profiling-type", choices=[p.value for p in ProfilingType], default=ProfilingType.OFFLINE.value
    )
    parser.add_argument("--request-type", choices=[r.value for r in RequestType])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-concurrency", type=int, default=1, help="Max in-flight requests for closed_loop profiling (default: 1)."
    )
    parser.add_argument("--dataset", default=None, choices=[d.value for d in DatasetType])
    parser.add_argument("--rate", type=float, default=1.0, help="Requests/sec (default: 1.0)")
    parser.add_argument(
        "--output-dir", default=None, help="Directory to save outputs (text files / images). Omit to skip."
    )
    parser.add_argument("--output-json", default=None, help="Write aggregate + per-request metrics to a JSON file.")
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Max output tokens for text generation tasks. Passed as max_output_tokens to mminf and max_tokens to vllm/sglang.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--local-cache", default="./mminf-benchmark-cache/", type=str)

    # specific to image gen
    parser.add_argument("--disable-cfg", action="store_true")

    # VBench args
    vbench = parser.add_argument_group("vbench")
    vbench.add_argument(
        "--vbench-cache-dir",
        default="./vbench_cache",
        help="Directory to cache downloaded VBench data (default: ./vbench_cache)",
    )

    # Text dataset args
    text_dataset = parser.add_argument_group("text_dataset")
    text_dataset.add_argument(
        "--request-txt-file",
        default="benchmark/assets/simple_text_queries.txt",
        help="Text file with one line per prompt",
    )

    args = parser.parse_args()

    dataset = args.dataset
    txtfile = args.request_txt_file
    request_type = RequestType(args.request_type)
    if dataset is None:
        if request_type in {RequestType.T2I, RequestType.I2I}:
            dataset = DatasetType.VBENCH
            txtfile = None
        elif request_type in {RequestType.I2T, RequestType.I2S}:
            dataset = DatasetType.FOOD
            txtfile = None
        elif request_type in {RequestType.V2T, RequestType.V2S}:
            dataset = DatasetType.UCF
            txtfile = None
        elif request_type in {RequestType.A2T, RequestType.A2S}:
            dataset = DatasetType.LIBRI
            txtfile = None
        elif request_type == RequestType.T2T:
            dataset = DatasetType.TEXT
            txtfile = "benchmark/assets/simple_text_queries.txt"
        elif request_type == RequestType.T2S:
            dataset = DatasetType.TEXT
            if args.model == ModelType.ORPHEUS:
                # T2S model, will just transcribe
                txtfile = "benchmark/assets/t2s.txt"
            else:
                # thinker-talker type model that speaks its answer
                txtfile = "benchmark/assets/simple_text_queries.txt"
        print(f"Dataset not specified, setting it to {dataset.value}, txtfile={txtfile}")

    return BenchmarkConfig(
        url=args.url,
        model=ModelType(args.model).inst(disable_cfg=args.disable_cfg),
        dataset=dataset,
        num_requests=args.num_requests,
        request_type=request_type,
        num_warmup=args.num_warmup,
        profiling_type=ProfilingType(args.profiling_type),
        inference_system=InferenceSystemType(args.inference_system),
        batch_size=args.batch_size,
        max_concurrency=args.max_concurrency,
        rate=args.rate,
        verbose=args.verbose,
        output_dir=args.output_dir,
        output_json=args.output_json,
        max_tokens=args.max_tokens,
        vbench_cache_dir=args.vbench_cache_dir,
        request_txt_file=txtfile,
        local_cache_dir=args.local_cache,
    )


async def main():
    config = parse_args()
    benchmark = Benchmark(config)
    await benchmark.run()


if __name__ == "__main__":
    asyncio.run(main())
