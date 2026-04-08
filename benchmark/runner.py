import argparse
import asyncio
import os
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import aiohttp

from benchmark.base import Model, RequestType
from benchmark.dataset import BaseDataset, TxtFileDataset, VBenchDataset
from benchmark.request import (
    AggregateMetrics,
    InferenceSystem,
    OurSystem,
    RequestInput,
    RequestMetrics,
    VLLMOmni,
    aggregate_metrics,
)


class DatasetType(Enum):
    VBENCH = "vbench"
    TEXT = "text"


class InferenceSystemType(Enum):
    OURS = "ours"
    VLLM_OMNI = "vllm_omni"

    def instantiate(self) -> InferenceSystem:
        if self == InferenceSystemType.OURS:
            return OurSystem()
        elif self == InferenceSystemType.VLLM_OMNI:
            return VLLMOmni()


class ProfilingType(Enum):
    OFFLINE = "offline"
    ONLINE = "online"


@dataclass
class BenchmarkConfig:
    url: str
    model: Model
    dataset: DatasetType
    num_requests: int
    request_type: RequestType
    num_warmup: int = 3
    profiling_type: ProfilingType = ProfilingType.OFFLINE
    inference_system: InferenceSystemType = InferenceSystemType.OURS
    verbose: bool = False

    batch_size: Optional[int] = 1
    rate: Optional[float] = 1
    output_dir: Optional[str] = None  # Save outputs here (text files / images)
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
                req_type=self.config.request_type
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
            if m.output_content is None:
                continue
            if isinstance(m.output_content, str):
                path = os.path.join(output_dir, f"req_{m.request_id}.txt")
                with open(path, "w") as f:
                    f.write(m.output_content)
            elif isinstance(m.output_content, bytes):
                path = os.path.join(output_dir, f"req_{m.request_id}.png")
                with open(path, "wb") as f:
                    f.write(m.output_content)
            saved += 1

        if saved:
            print(f"\nSaved {saved} outputs to {output_dir}/")

    def _print_errors(self, metrics: list[RequestMetrics]) -> None:
        errors = [(m.request_id, m.error) for m in metrics if m.error is not None]
        if not errors:
            return
        print(f"\n--- Errors ({len(errors)}/{len(metrics)}) ---")
        for request_id, error in errors:
            print(f"  [{request_id}] {error}")

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
                    base_url=self.config.url,
                    request_id=i,
                    req_type=req.req_type,
                    model=self.config.model,
                    prompt=req.prompt,
                    image_path=req.image_path,
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
                    base_url=self.config.url,
                    request_id=i,
                    req_type=req.req_type,
                    model=self.config.model,
                    prompt=req.prompt,
                    image_path=req.image_path,
                )
            )
            tasks.append(task)
            if i < len(requests) - 1:
                interval = random.expovariate(self.config.rate)
                await asyncio.sleep(interval)
        return list(await asyncio.gather(*tasks))

    async def run(self) -> tuple[list[RequestMetrics], AggregateMetrics]:
        dataset = self._get_dataset()
        if self.config.profiling_type == ProfilingType.OFFLINE:
            bs = self.config.batch_size
            # make even multiple of batch size
            self.config.num_requests = ((self.config.num_requests + bs - 1) // bs) * bs
        requests = dataset.get_requests()[: self.config.num_requests]

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            connector=aiohttp.TCPConnector(),
            read_bufsize=5 * 2**20,  # 1MB read buffer
        ) as session:
            if self.config.verbose:
                print("--- Warmup ---")
            warmup_req = requests[0]
            for i in range(self.config.num_warmup):
                if self.config.verbose:
                    print(f"Warmup {i+1} / {self.config.num_warmup}")
                await self.inference_system.send_request(
                    session=session,
                    base_url=self.config.url,
                    request_id=-1,
                    req_type=warmup_req.req_type,
                    model=self.config.model,
                    prompt=warmup_req.prompt,
                    image_path=warmup_req.image_path,
                )
            if self.config.verbose:
                print("Warmup complete.")

            wall_start = time.monotonic()

            if self.config.profiling_type == ProfilingType.OFFLINE:
                metrics = await self._run_concurrent_offline(session, requests)
            else:
                metrics = await self._run_concurrent_online(session, requests)

            wall_time = time.monotonic() - wall_start

        agg = aggregate_metrics(
            metrics,
            wall_time=wall_time,
            online=self.config.profiling_type == ProfilingType.ONLINE,
            batch_size=self.config.batch_size,
            rate=self.config.rate,
        )

        print(f"\n--- Benchmark Results (wall time: {wall_time:.2f}s) ---")
        print(agg)
        self._print_errors(metrics)
        self._save_outputs(metrics)

        return metrics, agg


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run inference benchmark")
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True, choices=[m.value for m in Model])
    parser.add_argument("--dataset", required=True, choices=[d.value for d in DatasetType])
    parser.add_argument("--inference-system", choices=[s.value for s in InferenceSystemType],
                        default=InferenceSystemType.OURS.value)
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--profiling-type", choices=[p.value for p in ProfilingType],
                        default=ProfilingType.OFFLINE.value)
    parser.add_argument("--request-type", choices=[r.value for r in RequestType])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Requests/sec (default: 1.0)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save outputs (text files / images). Omit to skip.")
    parser.add_argument("--verbose", action="store_true")

    # VBench args
    vbench = parser.add_argument_group("vbench")
    vbench.add_argument("--vbench-cache-dir", default="./vbench_cache",
                        help="Directory to cache downloaded VBench data (default: ./vbench_cache)")

    # Text dataset args
    text_dataset = parser.add_argument_group("text_dataset")
    text_dataset.add_argument(
        "--request-txt-file", default="benchmark/assets/simple_text_queries.txt",
        help="Text file with one line per prompt"
    )

    args = parser.parse_args()

    return BenchmarkConfig(
        url=args.url,
        model=Model(args.model),
        dataset=DatasetType(args.dataset),
        num_requests=args.num_requests,
        request_type=RequestType(args.request_type),
        num_warmup=args.num_warmup,
        profiling_type=ProfilingType(args.profiling_type),
        inference_system=InferenceSystemType(args.inference_system),
        batch_size=args.batch_size,
        rate=args.rate,
        verbose=args.verbose,
        output_dir=args.output_dir,
        vbench_cache_dir=args.vbench_cache_dir,
        request_txt_file=args.request_txt_file,
    )


async def main():
    config = parse_args()
    benchmark = Benchmark(config)
    await benchmark.run()


if __name__ == "__main__":
    asyncio.run(main())
