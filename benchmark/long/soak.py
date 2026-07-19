"""Soak / large-tensor stress driver (client half).

Fires a weighted *mixture* of request types at a Poisson arrival rate, capped at
``max_in_flight`` concurrent requests, for ``duration_s`` wall-clock seconds, and
reports rolling moving-average metrics plus failure / timeout / backpressure
signals. Reuses ``benchmark.request`` (the same ``send_request`` adapters and
``RequestMetrics`` the throughput benchmarks use) and ``benchmark.dataset``.

    python -m benchmark.long.soak --config benchmark/long/configs/qwen3omni.yaml \
        --url http://127.0.0.1:8000 --duration-s 7200 --metrics-jsonl soak_qwen.jsonl

Arrivals and admission are decoupled: the arrival process is a true Poisson
stream (independent of service), and the ``max_in_flight`` cap manifests as
growing *admission delay* — the client-side backpressure signal. A separate
backlog cap bounds memory if the server falls behind (arrivals pause and a
warning is logged rather than the client OOMing over a multi-hour run).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import aiohttp

from benchmark.base import Model, ModelType
from benchmark.long.metrics import SoakMetrics
from benchmark.long.mixture import SoakConfig, load_config
from benchmark.request import OursOpenAI, OurSystem, RequestInput

_SYSTEMS = {"ours": OurSystem, "ours_openai": OursOpenAI}


class _GenericModel(Model):
    """Passthrough model for server configs with no registered ``ModelType``
    (e.g. cosmos3). The ``ours`` (``/generate``) path only consults
    ``get_model_kwargs`` (empty here — the mixture entry supplies resolution /
    steps per request) and ``get_openai_system_message`` (None). Not usable with
    ``ours_openai``, which needs ``get_hf_url``."""

    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self._name = name

    def get_hf_url(self):
        return self._name

    def get_supported_modalities(self):
        return set()


def _resolve_model(name: str) -> Model:
    try:
        return ModelType(name).inst()
    except ValueError:
        return _GenericModel(name)


def _pick(rng: random.Random, cfg: SoakConfig):
    return rng.choices(cfg.requests, weights=cfg.weights, k=1)[0]


class Soaker:
    def __init__(self, cfg: SoakConfig, url: str, metrics_jsonl: str | None):
        self.cfg = cfg
        self.url = url.rstrip("/")
        self.metrics_jsonl = metrics_jsonl
        self.model = _resolve_model(cfg.model)
        if cfg.system not in _SYSTEMS:
            raise ValueError(
                f"unknown system {cfg.system!r} (known: {sorted(_SYSTEMS)})"
            )
        self.system = _SYSTEMS[cfg.system]()
        mods = sorted({r.req_type.get_output_modalities() for r in cfg.requests})
        self.metrics = SoakMetrics(window_s=cfg.window_s, modalities=mods)
        self.rng = random.Random(cfg.seed)
        # Admission cap = server-facing concurrency; backlog cap bounds
        # client memory when the server can't keep up.
        self._admit = asyncio.Semaphore(cfg.max_in_flight)
        self._backlog = asyncio.Semaphore(max(cfg.max_in_flight * 16, 64))
        self._stop = asyncio.Event()
        self._backlog_warned = 0.0
        self._id = 0

    async def _run_one(
        self, session: aiohttp.ClientSession, req: RequestInput,
        label: str, req_id: int,
    ) -> None:
        out_mod = req.req_type.get_output_modalities()
        t_arrive = time.monotonic()
        # Wait for an in-flight slot (this wait IS the client-side backpressure).
        await self._admit.acquire()
        self.metrics.on_admit(time.monotonic() - t_arrive)
        timed_out = False
        metrics = None
        try:
            metrics = await asyncio.wait_for(
                self.system.send_request(
                    session=session,
                    req_input=req,
                    base_url=self.url,
                    request_id=req_id,
                    model=self.model,
                ),
                timeout=self.cfg.request_timeout_s,
            )
        except asyncio.TimeoutError:
            timed_out = True
        finally:
            self._admit.release()
            self._backlog.release()
        self.metrics.on_complete(
            label=label, out_mod=out_mod,
            metrics=metrics if metrics is not None else _empty_metrics(req, req_id),
            timed_out=timed_out,
        )

    async def _arrivals(self, session: aiohttp.ClientSession) -> None:
        cfg = self.cfg
        deadline = time.monotonic() + cfg.duration_s
        tasks: set[asyncio.Task] = set()
        while time.monotonic() < deadline:
            await asyncio.sleep(self.rng.expovariate(cfg.rate))
            # Bound outstanding client memory. If we can't get a backlog slot
            # promptly the server is behind — pause arrivals (Poisson breaks,
            # logged) rather than accumulate unbounded tasks.
            if self._backlog.locked():
                now = time.monotonic()
                if now - self._backlog_warned > 5.0:
                    print(
                        f"[soak] WARN backlog cap hit (server behind); pausing "
                        f"arrivals. in_flight={self.metrics.in_flight} "
                        f"backlog={self.metrics.backlog}", flush=True,
                    )
                    self._backlog_warned = now
            await self._backlog.acquire()
            self.metrics.on_arrival()
            entry = _pick(self.rng, cfg)
            req = entry.sample(self.rng)
            self._id += 1
            t = asyncio.create_task(
                self._run_one(session, req, entry.label, self._id))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
        self._stop.set()
        # Drain in-flight (bounded): give the slowest requests one timeout.
        if tasks:
            print(f"[soak] arrivals done; draining {len(tasks)} in-flight…",
                  flush=True)
            await asyncio.wait(tasks, timeout=self.cfg.request_timeout_s + 30)

    async def _reporter(self) -> None:
        jsonl = open(self.metrics_jsonl, "a") if self.metrics_jsonl else None
        try:
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.cfg.report_interval_s)
                except asyncio.TimeoutError:
                    pass
                snap = self.metrics.snapshot()
                print(self.metrics.format_line(snap), flush=True)
                if jsonl:
                    jsonl.write(json.dumps(snap) + "\n")
                    jsonl.flush()
        finally:
            if jsonl:
                jsonl.close()

    async def run(self) -> dict:
        print(
            f"[soak] model={self.cfg.model} system={self.cfg.system} "
            f"rate={self.cfg.rate}/s max_in_flight={self.cfg.max_in_flight} "
            f"duration={self.cfg.duration_s}s window={self.cfg.window_s}s\n"
            f"[soak] mixture: "
            + ", ".join(f"{r.label}={r.weight:g}" for r in self.cfg.requests),
            flush=True,
        )
        reporter = asyncio.create_task(self._reporter())
        # One shared session (like benchmark.runner): reuse connections across
        # requests instead of a connector setup per request. Connection pool
        # sized to the in-flight cap.
        connector = aiohttp.TCPConnector(limit=self.cfg.max_in_flight + 8)
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None),
        ) as session:
            await self._arrivals(session)
        await reporter
        summary = self.metrics.final_summary()
        print("\n[soak] ==== FINAL ====")
        print(json.dumps(summary, indent=2), flush=True)
        return summary


def _empty_metrics(req: RequestInput, req_id: int):
    """A metrics stand-in for a timed-out request (send_request never returned)."""
    from benchmark.request import RequestMetrics

    m = RequestMetrics(
        request_id=str(req_id),
        type=req.req_type,
        expected_output_modalities=[req.req_type.get_output_modalities()],
    )
    m.error = "client_timeout"
    return m


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SHM-arena soak / stress driver (client)")
    p.add_argument("--config", required=True, help="mixture YAML")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="server base URL")
    p.add_argument("--metrics-jsonl", default=None,
                   help="append per-interval snapshots here (align with server SHM series)")
    # Scalar overrides — a single YAML can be swept over these.
    p.add_argument("--rate", type=float, default=None, help="Poisson req/s")
    p.add_argument("--max-in-flight", type=int, default=None)
    p.add_argument("--duration-s", type=float, default=None)
    p.add_argument("--request-timeout-s", type=float, default=None)
    p.add_argument("--pool-size", type=int, default=None)
    p.add_argument("--report-interval-s", type=float, default=None)
    p.add_argument("--window-s", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--system", default=None, choices=list(_SYSTEMS))
    p.add_argument("--cache-dir", default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    overrides = {
        "rate": args.rate,
        "max_in_flight": args.max_in_flight,
        "duration_s": args.duration_s,
        "request_timeout_s": args.request_timeout_s,
        "pool_size": args.pool_size,
        "report_interval_s": args.report_interval_s,
        "window_s": args.window_s,
        "seed": args.seed,
        "system": args.system,
        "cache_dir": args.cache_dir,
    }
    cfg = load_config(args.config, overrides)
    soaker = Soaker(cfg, url=args.url, metrics_jsonl=args.metrics_jsonl)
    asyncio.run(soaker.run())


if __name__ == "__main__":
    main()
