"""``python -m benchmark.eval``: run suites, render reports, build datasets.

Examples::

    # build a dataset once (into HF_HOME + a local directory)
    python -m benchmark.eval data librispeech --out ~/eval-data/ls_clean --n 200

    # launch `mstar serve kokoro`, sweep concurrency, 3 repeats per point
    python -m benchmark.eval run tts --serve kokoro --voice af_heart \\
        --data benchmark/assets/t2s.txt --concurrency 1,4,16 --repeats 3 --out results/kokoro

    # or measure a server that is already running
    python -m benchmark.eval run asr --url http://127.0.0.1:8000 --data ~/eval-data/ls_clean/manifest.jsonl

    # tables and charts from any set of results files
    python -m benchmark.eval report results/kokoro results/qwen3_tts --md table.md \\
        --chart sweep.png --chart-metrics ttfa_ms_p50,audio_s_per_s
"""

from __future__ import annotations

import argparse
import random
import shlex
import sys
from pathlib import Path

from benchmark.eval import datasets
from benchmark.eval.harness import SAVE_AUDIO, run_point
from benchmark.eval.load import MODES, LoadSpec
from benchmark.eval.schema import collect_metadata, served_models
from benchmark.eval.suites import SUITES


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _add_run(sub) -> None:
    run = sub.add_parser("run", help="run a suite against a server")
    suites = run.add_subparsers(dest="suite", required=True)
    for name, cls in SUITES.items():
        p = suites.add_parser(name, help=cls.help)
        srv = p.add_argument_group("server (one of --url / --serve)")
        g = srv.add_mutually_exclusive_group(required=True)
        g.add_argument("--url", help="base URL of a running server")
        g.add_argument("--serve", metavar="MODEL", help="launch `mstar serve MODEL` for the run")
        g.add_argument("--serve-cmd", metavar="CMD",
                       help="launch this command instead ({port} is substituted), e.g. another engine")
        srv.add_argument("--serve-config", default=None, help="deployment YAML for --serve")
        srv.add_argument("--serve-python", default=None,
                         help="Python of the M* environment for --serve (default: this one), so the client "
                              "and its judges can live in a separate venv")
        srv.add_argument("--serve-arg", action="append", default=[], help="extra `mstar serve` argument")
        srv.add_argument("--serve-env", action="append", default=[], metavar="K=V", help="server environment")
        srv.add_argument("--startup-timeout", type=float, default=1800.0)
        p.add_argument("--model-name", default=None, help="'model' field sent (default: --serve MODEL or 'default')")
        p.add_argument("--data", required=True, help="manifest .jsonl or a .txt of sentences")
        p.add_argument("--limit", type=int, default=None, help="first N items")
        p.add_argument("--shuffle-seed", type=int, default=None, help="shuffle items with this seed first")
        p.add_argument("--mode", choices=MODES, default="closed")
        p.add_argument("--concurrency", type=_ints, default=None,
                       help="closed: in-flight requests per point (default 1); open: one in-flight cap")
        p.add_argument("--rate", type=_floats, default=None, help="open loop arrival rates, req/s")
        p.add_argument("--arrival-seed", dest="load_seed", type=int, default=0, help="Poisson arrival seed")
        p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--warmup", type=int, default=3, help="untimed requests before each point")
        p.add_argument("--label", default=None, help="row label (default: model name)")
        p.add_argument("--out", default="eval-results", help="directory for results JSON, audio and server log")
        p.add_argument("--save-audio", choices=SAVE_AUDIO, default="last")
        cls.add_args(p)


def _add_report(sub) -> None:
    p = sub.add_parser("report", help="markdown table and charts from results files")
    p.add_argument("results", nargs="+", help="results JSON files or directories")
    p.add_argument("--md", default=None, help="write the table here (default: stdout)")
    p.add_argument("--chart", default=None, help="PNG path for a sweep chart")
    p.add_argument("--chart-metrics", default=None, help="comma-separated metrics for the chart")
    p.add_argument("--metrics", default=None, help="comma-separated table columns (metric names)")
    p.add_argument("--title", default=None)


def _add_data(sub) -> None:
    p = sub.add_parser("data", help="build a dataset manifest")
    kinds = p.add_subparsers(dest="kind", required=True)
    ls = kinds.add_parser("librispeech", help="LibriSpeech (CC BY 4.0)")
    ls.add_argument("--split", default="test.clean")
    fl = kinds.add_parser("fleurs", help="FLEURS (CC BY 4.0)")
    fl.add_argument("--lang", default="en_us")
    fl.add_argument("--split", default="test")
    st = kinds.add_parser("seedtts", help="Seed-TTS-eval zero-shot set (research use)")
    st.add_argument("--locale", default="en", choices=["en", "zh"])
    st.add_argument("--archive", default=None, help="local seedtts_testset.tar (skips the download)")
    dx = kinds.add_parser("duplex-synth", help="turn-taking and barge-in clips from an ASR manifest")
    dx.add_argument("--from", dest="source", required=True, help="16 kHz ASR manifest (e.g. librispeech)")
    dx.add_argument("--gap-s", type=float, default=2.5, help="pause before the barge-in")
    for k in (ls, fl, st, dx):
        k.add_argument("--out", required=True)
        k.add_argument("--n", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmark.eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run(sub)
    _add_report(sub)
    _add_data(sub)
    return parser


def _kv(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        k, sep, v = p.partition("=")
        if not sep:
            raise SystemExit(f"expected K=V, got {p!r}")
        out[k] = v
    return out


def _points(args) -> list[LoadSpec]:
    if args.mode == "open":
        if not args.rate:
            raise SystemExit("--mode open needs --rate")
        if args.concurrency and len(args.concurrency) > 1:
            raise SystemExit("--mode open takes one --concurrency cap; sweep --rate instead")
        cap = args.concurrency[0] if args.concurrency else 0
        return [LoadSpec("open", cap, r, args.load_seed) for r in args.rate]
    return [LoadSpec("closed", c, None, args.load_seed) for c in args.concurrency or [1]]


def cmd_run(args) -> list[Path]:
    from benchmark.eval.server import AttachedServer, ManagedServer

    cls = SUITES[args.suite]
    args.model_name = args.model_name or args.serve or "default"
    items = datasets.load_manifest(args.data)
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(items)
    items = items[: args.limit] if args.limit else items
    if not items:
        raise SystemExit(f"no items in {args.data}")
    suite = cls(args)
    suite.check_items(items)
    points = _points(args)
    label = args.label or args.model_name
    out = Path(args.out)
    config = {
        "data": {"path": Path(args.data).name, "items": len(items), "limit": args.limit,
                 "shuffle_seed": args.shuffle_seed, "first_id": items[0].id, "last_id": items[-1].id},
        "model_name": args.model_name,
    }
    if args.serve or args.serve_cmd:
        server = ManagedServer(args.serve or "custom", args.serve_config, extra_args=args.serve_arg,
                               env=_kv(args.serve_env), log_path=out / f"server-{label}.log",
                               startup_timeout_s=args.startup_timeout,
                               command=shlex.split(args.serve_cmd) if args.serve_cmd else None,
                               python=args.serve_python)
    else:
        server = AttachedServer(args.url)
    written = []
    with server:
        meta = collect_metadata(server={**server.info(), "models": served_models(server.url)})
        for spec in points:
            tag = spec.describe().replace("=", "").replace("/", "per").replace(",", "_")
            result = run_point(suite, server.url, items, spec, args.repeats, args.warmup, label, config, meta,
                               audio_dir=out / "audio" / f"{label}-{tag}", save_audio=args.save_audio)
            path = result.save(out / f"{args.suite}-{label}-{tag}.json")
            written.append(path)
            print(f"wrote {path}", file=sys.stderr)
    return written


def cmd_report(args) -> None:
    from benchmark.eval.report import COLUMNS, Col, load_results, render_chart, render_table

    results = load_results(args.results)
    if not results:
        raise SystemExit("no results files")
    known = {c.metric: c for cols in COLUMNS.values() for c in cols}
    columns = [known.get(m, Col(m, m, digits=3)) for m in args.metrics.split(",")] if args.metrics else None
    table = render_table(results, columns)
    if args.md:
        Path(args.md).write_text(table + "\n")
    else:
        print(table)
    if args.chart:
        metrics = args.chart_metrics.split(",") if args.chart_metrics else [
            c.metric for c in COLUMNS.get(results[0].suite, [])[:3]]
        render_chart(results, metrics, args.chart, args.title)
        print(f"wrote {args.chart}", file=sys.stderr)


def cmd_data(args) -> None:
    out = Path(args.out).expanduser()
    if args.kind == "librispeech":
        path = datasets.build_librispeech(out, args.split, args.n or 200)
    elif args.kind == "fleurs":
        path = datasets.build_fleurs(out, args.lang, args.split, args.n or 100)
    elif args.kind == "seedtts":
        path = datasets.build_seedtts(out, args.locale, args.n, args.archive)
    else:
        from benchmark.eval.suites.duplex import compose_items

        src = datasets.load_manifest(args.source)
        path = datasets.save_manifest(compose_items(src, out / "audio", args.n or 10, args.gap_s),
                                      out / "manifest.jsonl")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        cmd_data(args)
