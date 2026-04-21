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
from benchmark.dataset import BaseDataset, Food101Dataset, LibriSpeechDataset, TxtFileDataset, UCF101Dataset, VBenchDataset
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

    # def instantiate(self) -> InferenceSystem:
    def instantiate(self, base_url: str = "") -> InferenceSystem:
        if self == InferenceSystemType.OURS:
            return OurSystem()
        elif self == InferenceSystemType.VLLM_OMNI:
            return VLLMOmni(base_url=base_url)
        elif self == InferenceSystemType.VOX_SERVE:
            return VoxServe()
        elif self == InferenceSystemType.SGLANG_OMNI:
            return SGLangOmni()


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
    local_cache_dir: str
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
        # self.inference_system = config.inference_system.instantiate()
        self.inference_system = config.inference_system.instantiate(base_url=config.url)

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
        elif self.config.dataset == DatasetType.LIBRI:
            return LibriSpeechDataset(
                num_requests=self.config.num_requests,
                req_type=self.config.request_type,
                local_file_dir=self.config.local_cache_dir
            )
        elif self.config.dataset == DatasetType.FOOD:
            return Food101Dataset(
                num_requests=self.config.num_requests,
                req_type=self.config.request_type
            )
        elif self.config.dataset == DatasetType.UCF:
            # TODO: this is the dataset that vllm-omni reports using, so we have it as an example,
            # but it only has two videos that we're just alternating between... We should replace
            # this with a better dataset
            return UCF101Dataset(
                num_requests=self.config.num_requests,
                req_type=self.config.request_type,
                local_file_dir=self.config.local_cache_dir
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
                )
            )
            tasks.append(task)
            if i < len(requests) - 1:
                # Poisson inter-request times
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
                    req_input=warmup_req,
                    session=session,
                    base_url=self.config.url,
                    request_id=-1,
                    model=self.config.model,
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
    parser.add_argument("--model", required=True, choices=[m.value for m in ModelType])
    parser.add_argument("--inference-system", choices=[s.value for s in InferenceSystemType],
                        default=InferenceSystemType.OURS.value)
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--profiling-type", choices=[p.value for p in ProfilingType],
                        default=ProfilingType.OFFLINE.value)
    parser.add_argument("--request-type", choices=[r.value for r in RequestType])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataset", default=None, choices=[d.value for d in DatasetType])
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Requests/sec (default: 1.0)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save outputs (text files / images). Omit to skip.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--local-cache",  default="./mminf-benchmark-cache/", type=str)

    # specific to image gen
    parser.add_argument("--disable-cfg", action="store_true")

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

    dataset = args.dataset
    txtfile = args.request_txt_file
    request_type = RequestType(args.request_type)
    if dataset is None:
        if request_type in {
            RequestType.T2I, RequestType.I2I
        }:
            dataset = DatasetType.VBENCH
            txtfile = None
        elif request_type in {
            RequestType.I2T, RequestType.I2S
        }:
            dataset = DatasetType.FOOD
            txtfile = None
        elif request_type in {
            RequestType.V2T, RequestType.V2S
        }:
            dataset = DatasetType.UCF
            txtfile = None
        elif request_type in {
            RequestType.A2T, RequestType.A2S
        }:
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
        rate=args.rate,
        verbose=args.verbose,
        output_dir=args.output_dir,
        vbench_cache_dir=args.vbench_cache_dir,
        request_txt_file=txtfile,
        local_cache_dir=args.local_cache
    )


async def main():
    config = parse_args()
    benchmark = Benchmark(config)
    await benchmark.run()


if __name__ == "__main__":
    asyncio.run(main())
