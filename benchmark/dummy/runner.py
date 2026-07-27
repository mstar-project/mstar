"""Drive the no-op dummy models with offline-batched waves of requests.

Serve one of the tiny-node models first, ideally with ``--log-stats`` so the
server-side per-node tables line up with what this driver measures:

    mstar serve dummy_loop  --log-stats-file loop.log     # loop stays on the worker
    mstar serve dummy_walks --log-stats-file walks.log    # conductor hop per step

then drive it:

    python -m benchmark.dummy.runner --steps 100 --batch-size 16 --num-batches 20

Offline batching, matching ``benchmark/runner.py``'s ``OFFLINE`` mode: each
wave fires ``batch_size`` requests concurrently and waits for all of them
before starting the next wave, so the server sees a clean, bounded batch
rather than a rolling queue.

Every request carries a client-generated ``request_id``, which the server
echoes into its ``--log-stats`` block. That makes the two sides joinable:

    python -m mstar.profile.aggregate loop.log --dump-requests server.jsonl
    # join server.jsonl[].rid against the request_ids in this driver's --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum

import aiohttp

from mstar.profile.aggregate import Stat, summarize

DEFAULT_URL = "http://localhost:8000"


class Mode(str, Enum):
    #: Strict waves of ``batch_size``, each waiting for the previous to drain.
    #: Every wave costs max-of-B, so a single straggler sets the wave's time —
    #: the head-of-line blocking benchmark/runner.py warns about at high B.
    OFFLINE = "offline"
    #: A fixed number of requests in flight, refilled as each finishes. Measures
    #: steady state, which is what a per-step overhead fit wants.
    CLOSED_LOOP = "closed_loop"


@dataclass
class RequestConfig:
    url: str = DEFAULT_URL
    steps: int = 50
    mode: Mode = Mode.OFFLINE
    batch_size: int = 1
    num_batches: int = 10
    warmup_batches: int = 2
    text: str = "dummy"
    output_modalities: str = "tensor"
    input_modalities: str = "text"
    streaming: bool = True
    timeout_s: float = 600.0
    extra_model_kwargs: dict = field(default_factory=dict)

    @property
    def num_requests(self) -> int:
        return self.batch_size * self.num_batches

    @property
    def num_warmup_requests(self) -> int:
        return self.batch_size * self.warmup_batches


@dataclass
class RequestMetrics:
    request_id: str
    wave: int
    steps: int
    start: float
    first_chunk: float | None = None
    end: float | None = None
    num_chunks: int = 0
    num_bytes: int = 0
    error: str | None = None
    is_warmup: bool = False

    @property
    def jct_ms(self) -> float | None:
        if self.end is None:
            return None
        return (self.end - self.start) * 1e3

    @property
    def ttft_ms(self) -> float | None:
        if self.first_chunk is None:
            return None
        return (self.first_chunk - self.start) * 1e3


@dataclass
class WaveMetrics:
    index: int
    is_warmup: bool
    batch_size: int
    start: float
    end: float
    requests: list[RequestMetrics]

    @property
    def makespan_ms(self) -> float:
        """First request submitted → last request finished. This is the JCT of
        the batch as a unit, which is what the a + b·K fit wants at B > 1."""
        return (self.end - self.start) * 1e3


@dataclass
class RunResult:
    config: RequestConfig
    requests: list[RequestMetrics]
    #: Populated in OFFLINE mode only; closed-loop has no wave structure.
    waves: list[WaveMetrics] = field(default_factory=list)

    def measured(self) -> list[WaveMetrics]:
        return [w for w in self.waves if not w.is_warmup]

    def ok_requests(self) -> list[RequestMetrics]:
        return [
            r for r in self.requests
            if not r.is_warmup and r.error is None and r.end is not None
        ]

    def errors(self) -> list[RequestMetrics]:
        return [r for r in self.requests if r.error is not None]

    def measured_rids(self) -> list[str]:
        """Request ids of the measured (non-warmup) requests.

        Feed these to ``mstar.profile.aggregate --rids`` so a server-side
        stats file that spans several runs is narrowed to just this one.
        """
        return [r.request_id for r in self.ok_requests()]

    def jct_stat(self) -> Stat:
        return summarize([r.jct_ms for r in self.ok_requests()])

    def ttft_stat(self) -> Stat:
        return summarize(
            [r.ttft_ms for r in self.ok_requests() if r.ttft_ms is not None]
        )

    def makespan_stat(self) -> Stat:
        return summarize([w.makespan_ms for w in self.measured()])

    def throughput_req_per_s(self) -> float | None:
        ok = self.ok_requests()
        if not ok:
            return None
        span = max(r.end for r in ok) - min(r.start for r in ok)
        return len(ok) / span if span > 0 else None

    def mean_concurrency(self) -> float | None:
        """Requests in flight on average = throughput x mean JCT (Little's law).

        In closed-loop this should sit at the configured concurrency; well
        below it means the client, not the server, was the bottleneck.
        """
        tput = self.throughput_req_per_s()
        jct = self.jct_stat()
        if tput is None or jct.mean is None:
            return None
        return tput * (jct.mean / 1e3)


async def send_request(
    session: aiohttp.ClientSession,
    config: RequestConfig,
    wave: int,
) -> RequestMetrics:
    """Fire one request and time it from submit to last chunk."""
    request_id = str(uuid.uuid4())
    metrics = RequestMetrics(
        request_id=request_id, wave=wave, steps=config.steps, start=0.0
    )
    model_kwargs = {"steps": config.steps, **config.extra_model_kwargs}

    form = aiohttp.FormData()
    form.add_field("text", config.text)
    form.add_field("request_id", request_id)
    form.add_field("model_kwargs", json.dumps(model_kwargs))
    form.add_field("output_modalities", config.output_modalities)
    form.add_field("input_modalities", config.input_modalities)
    form.add_field("streaming", "true" if config.streaming else "false")

    metrics.start = time.perf_counter()
    try:
        async with session.post(
            f"{config.url}/generate", data=form, read_bufsize=2**24
        ) as resp:
            resp.raise_for_status()
            if config.streaming:
                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    data = msg.get("data")
                    if not data:
                        continue
                    if metrics.first_chunk is None:
                        metrics.first_chunk = time.perf_counter()
                    metrics.num_chunks += 1
                    # base64 -> raw byte count, without paying for the decode
                    metrics.num_bytes += (len(data) * 3) // 4
            else:
                payload = await resp.json()
                for chunks in payload.get("outputs", {}).values():
                    for chunk in chunks:
                        if metrics.first_chunk is None:
                            metrics.first_chunk = time.perf_counter()
                        metrics.num_chunks += 1
                        metrics.num_bytes += (len(chunk.get("data", "")) * 3) // 4
    except Exception as exc:  # network, HTTP status, server-side failure
        metrics.error = f"{type(exc).__name__}: {exc}"
    metrics.end = time.perf_counter()
    return metrics


async def run_wave(
    session: aiohttp.ClientSession,
    config: RequestConfig,
    index: int,
    is_warmup: bool,
) -> WaveMetrics:
    start = time.perf_counter()
    requests = await asyncio.gather(
        *(send_request(session, config, index) for _ in range(config.batch_size))
    )
    end = time.perf_counter()
    return WaveMetrics(
        index=index,
        is_warmup=is_warmup,
        batch_size=config.batch_size,
        start=start,
        end=end,
        requests=list(requests),
    )


def _session(config: RequestConfig) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=config.timeout_s),
        connector=aiohttp.TCPConnector(limit=max(config.batch_size * 2, 16)),
    )


async def run_offline(config: RequestConfig, verbose: bool = False) -> RunResult:
    """Strict waves of ``batch_size``, draining between each."""
    waves: list[WaveMetrics] = []
    total = config.warmup_batches + config.num_batches
    async with _session(config) as session:
        for i in range(total):
            is_warmup = i < config.warmup_batches
            wave = await run_wave(session, config, i, is_warmup)
            for req in wave.requests:
                req.is_warmup = is_warmup
            waves.append(wave)
            if verbose:
                tag = "warmup" if is_warmup else "wave  "
                errs = sum(1 for r in wave.requests if r.error)
                print(
                    f"  {tag} {i:>3}  B={wave.batch_size:<3} K={config.steps:<4} "
                    f"makespan={wave.makespan_ms:8.2f} ms"
                    + (f"  errors={errs}" if errs else ""),
                    file=sys.stderr,
                    flush=True,
                )
            # Small gap so the server's queue drains between waves, keeping
            # each wave a clean batch (mirrors benchmark/runner.py).
            await asyncio.sleep(0.01)
    requests = [r for w in waves for r in w.requests]
    return RunResult(config=config, requests=requests, waves=waves)


async def run_closed_loop(config: RequestConfig, verbose: bool = False) -> RunResult:
    """Keep ``batch_size`` requests in flight, refilling as each completes.

    No barrier between requests, so a straggler delays only itself instead of
    the whole wave. The first ``num_warmup_requests`` completions are marked
    warmup and excluded from the stats, which also covers the ramp while the
    pipe fills.
    """
    sem = asyncio.Semaphore(config.batch_size)
    completed = 0
    total = config.num_warmup_requests + config.num_requests
    results: list[RequestMetrics] = []

    async with _session(config) as session:
        async def one() -> None:
            nonlocal completed
            async with sem:
                metrics = await send_request(session, config, wave=-1)
            # Warmup is assigned on completion order, so it covers exactly the
            # requests issued while the pipe was still filling.
            completed += 1
            metrics.is_warmup = completed <= config.num_warmup_requests
            results.append(metrics)
            if verbose and completed % max(config.batch_size, 1) == 0:
                tag = "warmup" if metrics.is_warmup else "steady"
                print(
                    f"  {tag} {completed:>5}/{total}  C={config.batch_size:<3} "
                    f"K={config.steps:<4} jct={metrics.jct_ms:8.2f} ms",
                    file=sys.stderr,
                    flush=True,
                )

        await asyncio.gather(*(one() for _ in range(total)))

    results.sort(key=lambda r: r.start)
    return RunResult(config=config, requests=results)


async def run(config: RequestConfig, verbose: bool = False) -> RunResult:
    if config.mode is Mode.CLOSED_LOOP:
        return await run_closed_loop(config, verbose)
    return await run_offline(config, verbose)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_stat(name: str, stat: Stat, unit: str = "ms") -> str:
    if stat.n == 0:
        return f"  {name:<22} (no samples)"
    return (
        f"  {name:<22} n={stat.n:<5} mean={stat.mean:9.3f}  p50={stat.p50:9.3f}  "
        f"p95={stat.p95:9.3f}  p99={stat.p99:9.3f}  [{unit}]"
    )


def render(result: RunResult) -> str:
    cfg = result.config
    knob = "B" if cfg.mode is Mode.OFFLINE else "C"
    lines = [
        "=" * 88,
        f" dummy run  mode={cfg.mode.value}  K={cfg.steps} steps  "
        f"{knob}={cfg.batch_size}  "
        f"{len(result.ok_requests())} measured requests "
        f"({cfg.num_warmup_requests} warmup dropped)",
        "=" * 88,
    ]
    ok = result.ok_requests()
    errs = result.errors()
    lines.append(f"  requests ok={len(ok)}  failed={len(errs)}")
    if errs:
        seen: dict[str, int] = {}
        for e in errs:
            seen[e.error or "?"] = seen.get(e.error or "?", 0) + 1
        for msg, count in sorted(seen.items(), key=lambda kv: -kv[1])[:5]:
            lines.append(f"    x{count}  {msg}")
    lines.append("")
    lines.append(_fmt_stat("per-request JCT", result.jct_stat()))
    lines.append(_fmt_stat("per-request TTFT", result.ttft_stat()))
    if result.waves:
        lines.append(_fmt_stat("wave makespan", result.makespan_stat()))

    jct = result.jct_stat()
    if jct.n and cfg.steps:
        lines.append("")
        lines.append(
            f"  mean JCT / step        {jct.mean / cfg.steps * 1e3:9.2f} us "
            f"(includes the fixed per-request cost — use sweep.py to remove it)"
        )
        lines.append(
            f"  mean JCT / step / req  "
            f"{jct.mean / cfg.steps / cfg.batch_size * 1e3:9.2f} us"
        )
    tput = result.throughput_req_per_s()
    if tput:
        lines.append(f"  throughput             {tput:9.2f} req/s")
    conc = result.mean_concurrency()
    if conc is not None:
        note = ""
        if cfg.mode is Mode.CLOSED_LOOP and conc < cfg.batch_size * 0.8:
            note = "  <- well below C; the client may be the bottleneck"
        lines.append(f"  mean concurrency       {conc:9.2f}{note}")
    if ok:
        chunks = statistics.mean(r.num_chunks for r in ok)
        nbytes = statistics.mean(r.num_bytes for r in ok)
        lines.append(f"  output chunks/req      {chunks:9.2f}  ({nbytes/1024:.1f} KiB)")
    lines.append("=" * 88)
    return "\n".join(lines) + "\n"


def to_dict(result: RunResult) -> dict:
    config = asdict(result.config)
    config["mode"] = result.config.mode.value
    return {
        "config": config,
        "summary": {
            "num_ok": len(result.ok_requests()),
            "num_failed": len(result.errors()),
            "jct_ms": asdict(result.jct_stat()),
            "ttft_ms": asdict(result.ttft_stat()),
            "makespan_ms": asdict(result.makespan_stat()),
            "throughput_req_per_s": result.throughput_req_per_s(),
            "mean_concurrency": result.mean_concurrency(),
        },
        "waves": [
            {
                "index": w.index,
                "is_warmup": w.is_warmup,
                "batch_size": w.batch_size,
                "makespan_ms": w.makespan_ms,
                "requests": [
                    {
                        **asdict(r),
                        "jct_ms": r.jct_ms,
                        "ttft_ms": r.ttft_ms,
                    }
                    for r in w.requests
                ],
            }
            for w in result.waves
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_args(
    parser: argparse.ArgumentParser, include_batch_size: bool = True
) -> None:
    """Args shared by the single-point runner and the sweep.

    ``include_batch_size=False`` for callers that sweep B themselves and set
    ``args.batch_size`` per point (see ``benchmark/dummy/sweep.py``).
    """
    parser.add_argument("--url", default=DEFAULT_URL, help="mstar server base URL")
    parser.add_argument(
        "--mode", type=Mode, choices=list(Mode), default=Mode.OFFLINE,
        help="offline = strict waves (measures max-of-B, noisy at high B); "
             "closed_loop = fixed in-flight count, refilled (measures steady "
             "state — prefer this for the per-step fit)",
    )
    if include_batch_size:
        parser.add_argument(
            "--batch-size", "-B", type=int, default=1,
            help="requests per wave (offline) or in flight at once (closed_loop)",
        )
    parser.add_argument(
        "--tensor-size", default=None,
        help="dummy input tensor shape as JSON, e.g. '[1,1]'. The 512x512 "
             "fp32 default is 1 MiB per request, whose transfer and "
             "preprocessing dominate at high B and swamp the per-step signal. "
             "Shrink it when measuring dispatch overhead.",
    )
    parser.add_argument(
        "--num-batches", type=int, default=10, help="measured waves"
    )
    parser.add_argument(
        "--warmup-batches", type=int, default=2,
        help="waves run first and excluded from the stats",
    )
    parser.add_argument("--text", default="dummy", help="request prompt text")
    parser.add_argument(
        "--output-modalities", default="tensor",
        help="both dummy models emit a raw 'tensor' chunk",
    )
    parser.add_argument("--input-modalities", default="text")
    parser.add_argument(
        "--no-streaming", dest="streaming", action="store_false",
        help="use the buffered JSON response instead of the NDJSON stream",
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds")
    parser.add_argument(
        "--model-kwargs", default=None,
        help="extra model_kwargs as a JSON object, merged over {'steps': K}",
    )
    parser.add_argument("--verbose", "-v", action="store_true")


def config_from_args(args: argparse.Namespace, steps: int) -> RequestConfig:
    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    if args.tensor_size:
        model_kwargs["tensor_size"] = json.loads(args.tensor_size)
    return RequestConfig(
        url=args.url,
        steps=steps,
        mode=args.mode,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
        text=args.text,
        output_modalities=args.output_modalities,
        input_modalities=args.input_modalities,
        streaming=args.streaming,
        timeout_s=args.timeout,
        extra_model_kwargs=model_kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark.dummy.runner",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps", "-K", type=int, default=50,
        help="per-request loop iterations / graph walks (model_kwargs['steps'])",
    )
    parser.add_argument("--json", default=None, help="write full results here")
    parser.add_argument(
        "--rids-out", default=None,
        help="write this run's measured request ids here, for "
             "`mstar.profile.aggregate --rids`",
    )
    add_common_args(parser)
    args = parser.parse_args(argv)

    config = config_from_args(args, args.steps)
    result = asyncio.run(run(config, verbose=args.verbose))

    sys.stdout.write(render(result))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(to_dict(result), fh, indent=2)
    if args.rids_out:
        with open(args.rids_out, "w") as fh:
            fh.write("\n".join(result.measured_rids()) + "\n")
    return 1 if result.errors() else 0


if __name__ == "__main__":
    sys.exit(main())
