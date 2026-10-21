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

    # base vs head on the same GPUs (local, or --hosts over SSH): correctness and perf per model and case
    python -m benchmark.eval compare --base main --head 548 --models tag:tts,qwen3_omni
"""

from __future__ import annotations

import argparse
import random
import re
import shlex
import signal
import sys
from pathlib import Path

from benchmark.eval import datasets
from benchmark.eval.e2e import compare as e2e_compare
from benchmark.eval.e2e import hosts as e2e_hosts
from benchmark.eval.harness import SAVE_AUDIO, run_point
from benchmark.eval.load import MODES, LoadSpec
from benchmark.eval.schema import collect_metadata, served_models
from benchmark.eval.suites import SUITES


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _add_run(sub) -> None:
    run = sub.add_parser("run", allow_abbrev=False, help="run a suite against a server")
    suites = run.add_subparsers(dest="suite", required=True)
    for name, cls in SUITES.items():
        p = suites.add_parser(name, allow_abbrev=False, help=cls.help)
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
        p.add_argument("--data", default=None,
                       help="manifest .jsonl or a .txt of sentences (optional only for suites that build their own)")
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
    p = sub.add_parser("report", allow_abbrev=False, help="markdown table and charts from results files")
    p.add_argument("results", nargs="+", help="results JSON files or directories")
    p.add_argument("--md", default=None, help="write the table here (default: stdout)")
    p.add_argument("--chart", default=None, help="PNG path for a sweep chart")
    p.add_argument("--chart-metrics", default=None, help="comma-separated metrics for the chart")
    p.add_argument("--metrics", default=None, help="comma-separated table columns (metric names)")
    p.add_argument("--title", default=None)
    p.add_argument("--chart-judge", default=None,
                   help="fdb: comma-separated judges whose label shares the chart shows (default: all)")


def _add_rescore(sub) -> None:
    p = sub.add_parser("rescore", allow_abbrev=False,
                       help="recompute WER/CER from stored transcripts with another normalizer")
    p.add_argument("results", nargs="+", help="tts or asr results JSON files or directories")
    p.add_argument("--data", required=True, help="the manifest the run used (references)")
    p.add_argument("--normalizer", required=True)
    p.add_argument("--out", required=True, help="directory for the rescored files")


def _add_data(sub) -> None:
    p = sub.add_parser("data", allow_abbrev=False, help="build a dataset manifest")
    kinds = p.add_subparsers(dest="kind", required=True)
    ls = kinds.add_parser("librispeech", allow_abbrev=False, help="LibriSpeech (CC BY 4.0)")
    ls.add_argument("--split", default="test.clean")
    fl = kinds.add_parser("fleurs", allow_abbrev=False, help="FLEURS (CC BY 4.0)")
    fl.add_argument("--lang", default="en_us")
    fl.add_argument("--split", default="test")
    st = kinds.add_parser("seedtts", allow_abbrev=False, help="Seed-TTS-eval zero-shot set (research use)")
    st.add_argument("--locale", default="en", choices=["en", "zh"])
    st.add_argument("--archive", default=None, help="local seedtts_testset.tar (skips the download)")
    dx = kinds.add_parser("duplex-synth", allow_abbrev=False,
                          help="turn-taking and barge-in clips from an ASR manifest")
    dx.add_argument("--from", dest="source", required=True, help="16 kHz ASR manifest (e.g. librispeech)")
    dx.add_argument("--gap-s", type=float, default=2.5, help="pause before the barge-in")
    for k in (ls, fl, st, dx):
        k.add_argument("--out", required=True)
        k.add_argument("--n", type=int, default=None)
    e2e = kinds.add_parser("e2e", allow_abbrev=False, help="every input `compare` sends (speech sets and media)")
    e2e.add_argument("--archive", default=None, help="local seedtts_testset.tar (skips the download)")
    e2e.add_argument("--out", required=True)
    fdb = kinds.add_parser("fdb-v15", allow_abbrev=False,
                           help="Full-Duplex-Bench v1.5 pairs (CC BY-NC 4.0, non-commercial use only), downloaded "
                                "from the official Google Drive; never vendored")
    fdb.add_argument("--out", required=True)
    fdb.add_argument("--per-category", default="all",
                     help="first N pairs of each category in numeric id order, or 'all' (498 pairs)")
    fdb.add_argument("--zips", default=None,
                     help="directory with the four official zips; a missing one is downloaded into it")


def _judge_args(p) -> None:
    p.add_argument("--judge", action="append", required=True, metavar="NAME=openai|URL",
                   help="judge to add: NAME=openai (needs OPENAI_API_KEY) or NAME=URL of a local endpoint")
    p.add_argument("--judge-revision", action="append", default=[], metavar="NAME=REV")
    p.add_argument("--semantic-judge", default="none", help="also regrade semantic A/F/U with this judge")
    p.add_argument("--retry-failed", action="store_true",
                   help="relabel only the pairs whose label from these (unchanged) judges failed")
    p.add_argument("--semantic-only", action="store_true",
                   help="only regrade semantic A/F/U with --semantic-judge; keep every judge's behavior labels")
    p.add_argument("--judge-concurrency", type=int, default=8)


def _add_fdb_tools(sub) -> None:
    p = sub.add_parser("rejudge", allow_abbrev=False,
                       help="label stored fdb results with more LLM judges, from their transcripts (no GPU)")
    p.add_argument("results", nargs="+", help="fdb results JSON files or directories")
    _judge_args(p)
    p.add_argument("--out", required=True, help="directory for the relabeled files")
    p = sub.add_parser("agreement", allow_abbrev=False,
                       help="label agreement, Cohen's kappa and confusion matrix of two judges on fdb results")
    p.add_argument("results", nargs="+", help="fdb results JSON files or directories")
    p.add_argument("--judges", default=None, help="A,B (default: the first two judges of the run)")
    p.add_argument("--md", default=None, help="write the report here (default: stdout)")
    p.add_argument("--chart", default=None, help="PNG path for the confusion matrix")


def build_parser() -> argparse.ArgumentParser:
    # No abbreviated flags anywhere: results scrub argv by exact flag name (--serve-cmd, --serve-env, ...).
    parser = argparse.ArgumentParser(prog="python -m benchmark.eval", description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run(sub)
    _add_report(sub)
    _add_rescore(sub)
    _add_data(sub)
    e2e_compare.add_args(sub)
    _add_fdb_tools(sub)
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
    suite = cls(args)
    items = datasets.load_manifest(args.data) if args.data else suite.default_items()
    if items is None:
        raise SystemExit(f"suite {args.suite!r} needs --data")
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(items)
    items = items[: args.limit] if args.limit else items
    if not items:
        raise SystemExit(f"no items in {args.data or 'the suite defaults'}")
    suite.check_items(items)
    points = _points(args)
    label = args.label or Path(args.model_name).name
    file_label = re.sub(r"[^\w.-]+", "_", label)
    out = Path(args.out)
    config = {
        "data": {"path": Path(args.data).name if args.data else None, "items": len(items), "limit": args.limit,
                 "shuffle_seed": args.shuffle_seed, "first_id": items[0].id, "last_id": items[-1].id},
        "model_name": args.model_name,
    }
    if args.serve or args.serve_cmd:
        server = ManagedServer(args.serve or "custom", args.serve_config, extra_args=args.serve_arg,
                               env=_kv(args.serve_env), log_path=out / f"server-{file_label}.log",
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
                               audio_dir=out / "audio" / f"{file_label}-{tag}", save_audio=args.save_audio,
                               error_log=out / f"errors-{args.suite}-{file_label}-{tag}.log")
            path = result.save(out / f"{args.suite}-{file_label}-{tag}.json")
            written.append(path)
            print(f"wrote {path}", file=sys.stderr)
    return written


def cmd_report(args) -> None:
    from benchmark.eval.report import COLUMNS, Col, load_results, render_chart, render_table

    results = load_results(args.results)
    if not results:
        raise SystemExit("no results files")
    if all(r.suite == "fdb" for r in results):
        from benchmark.eval import fdb_report

        table = fdb_report.render(results)
        if args.md:
            Path(args.md).write_text(table + "\n")
        else:
            print(table)
        if args.chart:
            judges = args.chart_judge.split(",") if args.chart_judge else None
            fdb_report.render_chart(results, args.chart, judges, args.title)
            print(f"wrote {args.chart}", file=sys.stderr)
        return
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


def cmd_rescore(args) -> None:
    from benchmark.eval.report import load_results
    from benchmark.eval.rescore import rescore

    items = datasets.load_manifest(args.data)
    out = Path(args.out)
    for path, result in zip(_result_paths(args.results), load_results(args.results), strict=True):
        dst = rescore(result, items, args.normalizer).save(out / path.name)
        print(f"wrote {dst}", file=sys.stderr)


def _result_paths(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    return files


def cmd_data(args) -> None:
    out = Path(args.out).expanduser()
    if args.kind == "librispeech":
        path = datasets.build_librispeech(out, args.split, args.n or 200)
    elif args.kind == "fleurs":
        path = datasets.build_fleurs(out, args.lang, args.split, args.n or 100)
    elif args.kind == "seedtts":
        path = datasets.build_seedtts(out, args.locale, args.n, args.archive)
    elif args.kind == "e2e":
        from benchmark.eval.e2e import media

        path = media.build(out, args.archive)
    elif args.kind == "fdb-v15":
        from benchmark.eval.fdb_reference import build_manifest

        per = None
        if args.per_category != "all":
            per = int(args.per_category) if args.per_category.isdigit() else 0
            if per <= 0:
                raise SystemExit(f"--per-category takes a positive count or 'all', not {args.per_category!r}")
        path = build_manifest(out, per, Path(args.zips).expanduser() if args.zips else None)
    else:
        from benchmark.eval.suites.duplex import compose_items

        src = datasets.load_manifest(args.source)
        path = datasets.save_manifest(compose_items(src, out / "audio", args.n or 10, args.gap_s),
                                      out / "manifest.jsonl")
    print(f"wrote {path}")


def cmd_rejudge(args) -> None:
    from benchmark.eval.llm_judge import parse_judges
    from benchmark.eval.report import load_results
    from benchmark.eval.suites.duplex_fdb import Judging, rejudge

    try:
        judges = parse_judges(args.judge, args.judge_revision)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.semantic_only and args.semantic_judge in (None, "none"):
        raise SystemExit("--semantic-only needs --semantic-judge")
    judging = Judging(judges, args.semantic_judge, args.judge_concurrency, behavior=not args.semantic_only)
    if not judging.available:
        raise SystemExit("no judge is available (see the skip reasons above)")
    out = Path(args.out)
    for path, result in zip(_result_paths(args.results), load_results(args.results), strict=True):
        dst = rejudge(result, judging, args.retry_failed).save(out / path.name)
        print(f"wrote {dst}", file=sys.stderr)
    for name, u in judging.usage().items():
        print(f"[fdb] judge {name!r}: {u['requests']} requests ({u['retries']} transport retries), "
              f"{u['prompt']} prompt and {u['completion']} completion tokens", file=sys.stderr)


def cmd_agreement(args) -> None:
    from benchmark.eval import fdb_report
    from benchmark.eval.report import load_results

    results = [r for r in load_results(args.results) if r.suite == "fdb"]
    if not results:
        raise SystemExit("no fdb results files")
    a, b = None, None
    if args.judges:
        names = args.judges.split(",")
        if len(names) != 2 or not all(names):
            raise SystemExit(f"--judges takes two judge names as A,B, not {args.judges!r}")
        a, b = names
    ag = fdb_report.agreement(results, a, b)
    text = fdb_report.render_agreement(ag)
    if args.md:
        Path(args.md).write_text(text)
    else:
        print(text)
    if args.chart and ag["n"]:
        fdb_report.render_confusion_chart(ag, args.chart)
        print(f"wrote {args.chart}", file=sys.stderr)


def _exit_on_signal(signum, frame) -> None:  # noqa: ARG001
    # Unwind through the `with server:` block so a cancelled job never leaves GPU workers behind.
    raise SystemExit(128 + signum)


def _interrupted(e: BaseException | None) -> str:
    """The one line ``compare`` prints on Ctrl-C instead of a traceback."""
    while e is not None and not isinstance(e, e2e_hosts.Interrupted):   # a second Ctrl-C while cleaning up
        e = e.__context__
    if e is None:
        return "interrupted; no job was running"
    if e.stopped:
        return "interrupted; remote job stopped"
    return f"interrupted; a job still running stops itself within {e2e_compare.LEASE_S // 60} min (its lease)"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _exit_on_signal)
    if args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "rescore":
        cmd_rescore(args)
    elif args.command == "compare":
        try:
            code = e2e_compare.cmd_compare(args)
        except KeyboardInterrupt as e:
            print(_interrupted(e), file=sys.stderr)
            code = 128 + signal.SIGINT
        sys.exit(code)
    elif args.command == "calibrate":
        sys.exit(e2e_compare.cmd_calibrate(args))
    elif args.command == "rejudge":
        cmd_rejudge(args)
    elif args.command == "agreement":
        cmd_agreement(args)
    else:
        cmd_data(args)
