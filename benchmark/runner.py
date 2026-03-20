import argparse
import asyncio
from dataclasses import dataclass
from enum import Enum
import random
import time
from typing import Optional

import aiohttp

from benchmark.base import Model, RequestType
from benchmark.dataset import BaseDataset, VBenchDataset
from benchmark.request import AggregateMetrics, InferenceSystem, OurSystem, RequestInput, RequestMetrics, VLLMOmni, aggregate_metrics


class DatasetType(Enum):
    VBENCH = "vbench"


class InferenceSystemType(Enum):
    OURS = "ours"
    VLLM_OMNI = "vllm_omni"

    def instantiate(self) -> InferenceSystem:
        if self == InferenceSystemType.OURS:
            return OurSystem()
        elif self == InferenceSystemType.VLLM_OMNI:
            return VLLMOmni()


@dataclass
class BenchmarkConfig:
    url: str
    model: Model
    dataset: DatasetType
    num_requests: int
    request_type: RequestType
    inference_system: InferenceSystemType = InferenceSystemType.OURS
    rate: Optional[float] = None  # None = sequential, >0 = requests/sec
    # VBench args
    vbench_cache_dir: str = "./vbench_cache"


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
        raise ValueError(f"Unknown dataset: {self.config.dataset}")

    def _print_errors(self, metrics: list[RequestMetrics]) -> None:
        errors = [(m.request_id, m.error) for m in metrics if m.error is not None]
        if not errors:
            return
        print(f"\n--- Errors ({len(errors)}/{len(metrics)}) ---")
        for request_id, error in errors:
            print(f"  [{request_id}] {error}")

    async def _run_concurrent(
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

    async def _run_sequential(
        self,
        session: aiohttp.ClientSession,
        requests: list[RequestInput],
    ) -> list[RequestMetrics]:
        results = []
        for i, req in enumerate(requests):
            result = await self.inference_system.send_request(
                session=session,
                base_url=self.config.url,
                request_id=i,
                req_type=req.req_type,
                model=self.config.model,
                prompt=req.prompt,
                image_path=req.image_path,
            )
            results.append(result)
        return results

    async def run(self) -> tuple[list[RequestMetrics], AggregateMetrics]:
        dataset = self._get_dataset()
        requests = dataset.get_requests()[: self.config.num_requests]

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            connector=aiohttp.TCPConnector(),
            read_bufsize=5 * 2**20,  # 1MB read buffer
        ) as session:
            print("--- Warmup ---")
            warmup_req = requests[0]
            for i in range(3):
                print(f"Warmup {i} / 3")
                await self.inference_system.send_request(
                    session=session,
                    base_url=self.config.url,
                    request_id=-1,
                    req_type=warmup_req.req_type,
                    model=self.config.model,
                    prompt=warmup_req.prompt,
                    image_path=warmup_req.image_path,
                )
            print("Warmup complete.\n")

            wall_start = time.monotonic()

            if self.config.rate is None:
                metrics = await self._run_sequential(session, requests)
            else:
                metrics = await self._run_concurrent(session, requests)

            wall_time = time.monotonic() - wall_start

        agg = aggregate_metrics(
            metrics,
            wall_time=wall_time,
            rate=self.config.rate
        )

        print(f"\n--- Benchmark Results (wall time: {wall_time:.2f}s) ---")
        print(agg)
        self._print_errors(metrics)

        return metrics, agg


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run inference benchmark")
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True, choices=[m.value for m in Model])
    parser.add_argument("--dataset", required=True, choices=[d.value for d in DatasetType])
    parser.add_argument("--inference-system", choices=[s.value for s in InferenceSystemType],
                    default=InferenceSystemType.OURS.value)
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--rate", type=float, default=None,
                        help="Requests/sec. Omit for sequential mode.")
    parser.add_argument("--request-type", choices=[r.value for r in RequestType])
    # VBench args
    vbench = parser.add_argument_group("vbench")
    
    vbench.add_argument("--vbench-cache-dir", default="./vbench_cache",
                        help="Directory to cache downloaded VBench data (default: ./vbench_cache)")

    args = parser.parse_args()

    return BenchmarkConfig(
        url=args.url,
        model=Model(args.model),
        request_type=RequestType(args.request_type),
        dataset=DatasetType(args.dataset),
        num_requests=args.num_requests,
        rate=args.rate,
        vbench_cache_dir=args.vbench_cache_dir,
        inference_system=InferenceSystemType(args.inference_system),
    )


async def main():
    config = parse_args()
    benchmark = Benchmark(config)
    await benchmark.run()


if __name__ == "__main__":
    asyncio.run(main())
