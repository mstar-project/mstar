#!/usr/bin/env python3
"""Correctness runner: serve one config, send its sample set, save the outputs, score them
against the reference and write a verdict.

    python benchmark/correctness/run.py --config kokoro --gpus 0 --port 8731 \\
        --worktree $WORK/amd/worktrees/ws-a [--override resources.attn.flashinfer_backend=fa2]

Results go to <results-root>/<config>/<run-id>/ (effective config, versions, server logs, outputs
per restart, node health, report.md and report.json). Exit code 0 on PASS/WARN, 1 on
FAIL/NO_REFERENCE, 2 on ERROR.

References are built once per set under <data-root>/references/ (--make-reference or
--reference-only). The server runs from --worktree with that checkout's configs, so the runner can
sit on any branch. Run it by path.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

_RUNNER_ROOT = Path(__file__).resolve().parents[2]
if str(_RUNNER_ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(_RUNNER_ROOT))

from benchmark.correctness import client, server  # noqa: E402
from benchmark.correctness.families import family_for_config  # noqa: E402
from benchmark.correctness.report import write_reports  # noqa: E402
from benchmark.correctness.sets import DEFAULT_DATA_ROOT, load_set  # noqa: E402

DEFAULT_RESULTS_ROOT = Path("/work1/stephaniewang/atindra/amd/results/correctness")
EXIT = {"PASS": 0, "WARN": 0, "FAIL": 1, "NO_REFERENCE": 1, "ERROR": 2}


def _kv(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        k, sep, v = p.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {p!r}")
        out[k] = v
    return out


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config name (kokoro), file name, or YAML path")
    ap.add_argument("--worktree", type=Path, default=_RUNNER_ROOT,
                    help="mstar checkout to serve (default: the runner's own checkout)")
    ap.add_argument("--gpus", default="0", help="passed to `mstar serve --gpus`")
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--tag", default=None, help="prefix of the per-server /tmp/$USER/<tag>-<key>[-up] socket and "
                                                "upload dirs (default: corr-<worktree directory name>)")
    ap.add_argument("--run-id", default=None, help="default: <UTC timestamp>-<git short sha>")
    ap.add_argument("--override", action="append", default=[],
                    help="dotted YAML override on a copy of the config, e.g. resources.attn.flashinfer_backend=fa2")
    ap.add_argument("--restarts", type=int, default=None,
                    help="server instances to run the set on (default: the family's; >= 2 gives a noise floor)")
    ap.add_argument("--concurrency", type=int, default=None, help="in-flight requests (default: the family's)")
    ap.add_argument("--limit", type=int, default=None, help="first N items only (smoke runs; not a verdict)")
    ap.add_argument("--warmup", type=int, default=0, help="untimed requests before each pass (default 0)")
    ap.add_argument("--family-opt", action="append", default=[], help="family-specific key=value")
    ap.add_argument("--env", action="append", default=[], help="extra server environment KEY=VALUE")
    ap.add_argument("--startup-timeout", type=int, default=2400)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--device", default=None, help="torch device for reference/judge models (default cuda:0)")
    ap.add_argument("--make-reference", action="store_true", help="build the reference first if it is missing")
    ap.add_argument("--reference-only", action="store_true", help="build the reference and exit (no server)")
    ap.add_argument("--rescore", type=Path, default=None, help="re-score an existing run dir (no server)")
    ap.add_argument("--no-node-health", action="store_true")
    return ap.parse_args(argv)


def _config_name(config: str) -> str:
    """Config name from a name, file or path. Only a .yaml or .yml suffix is dropped."""
    name = Path(config).name
    for suffix in (".yaml", ".yml"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _git_short(worktree: Path) -> str:
    import subprocess

    p = subprocess.run(["git", "-C", str(worktree), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                       check=False)
    return p.stdout.strip() or "nogit"


def _save_run(fam, sset, responses, out_dir: Path) -> dict:
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "responses.jsonl", "w") as f:
        for r in responses:
            f.write(json.dumps(r.summary(), default=str) + "\n")
    fam.save(sset, responses, out_dir / "outputs")
    ok = sum(r.ok for r in responses)
    return {"ok": ok, "failed": len(responses) - ok,
            "errors": {r.item_id: r.error for r in responses if not r.ok}}


def main(argv=None) -> int:
    args = parse_args(argv)
    worktree = args.worktree.resolve()
    device = args.device or "cuda:0"
    t_start = _now()

    if args.rescore:
        run_dir = args.rescore.resolve()
        prev = json.loads((run_dir / "report.json").read_text())
        cfg_path = run_dir / "config.yaml"
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text())
        overrides = prev.get("overrides") or []
    else:
        src_cfg = server.resolve_config(worktree, args.config)
        import yaml

        cfg = yaml.safe_load(src_cfg.read_text()) or {}
        overrides = args.override
    config_name = _config_name(args.config) if not args.rescore else prev["config"]
    model_key = cfg.get("model")
    fam = family_for_config(config_name, model_key, cfg, _kv(args.family_opt))
    sset = load_set(fam.set_name, args.data_root)
    sset.items = fam.select_items(sset.items)
    items = sset.items[: args.limit] if args.limit else sset.items
    sset.items = items
    ref_dir = args.data_root / "references" / fam.reference_id() / fam.set_name

    reference_error = None
    if args.reference_only or (args.make_reference and fam.ref_outputs(ref_dir) is None):
        if fam.ref_outputs(ref_dir) is not None and args.reference_only:
            print(f"reference exists: {ref_dir}")
        else:
            if args.limit:
                raise SystemExit("--limit with a reference build would save a partial reference; refusing")
            try:
                _build_reference(fam, sset, ref_dir, device)
            except Exception:
                if args.reference_only:
                    raise
                # The served run still says something (sanity, noise floor), so keep going.
                reference_error = traceback.format_exc()
                print(f"reference build failed, serving without it:\n{reference_error}", file=sys.stderr, flush=True)
        if args.reference_only:
            return 0

    if args.rescore:
        restarts = prev.get("restarts", 1)
        header = {k: prev.get(k) for k in ("config", "run_id", "set", "n_items", "restarts", "concurrency",
                                           "overrides", "versions", "startup_s", "started", "runs", "limit")}
        header.update(reference=fam.reference_description() if fam.ref_outputs(ref_dir) else "missing",
                      protocol=fam.protocol(), rescored=_now())
        header["finished"] = _now()
        runs = [run_dir / f"r{k}" for k in range(1, restarts + 1)]
        return _score_and_report(fam, sset, runs, ref_dir, run_dir, header, device)

    restarts = args.restarts or fam.default_restarts
    concurrency = args.concurrency or fam.default_concurrency
    run_id = args.run_id or f"{datetime.datetime.utcnow():%Y%m%dT%H%M%SZ}-{_git_short(worktree)}"
    run_dir = (args.results_root / config_name / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    run_key = hashlib.sha1(f"{run_dir}:{os.getpid()}".encode()).hexdigest()[:8]
    eff_cfg = run_dir / "config.yaml"
    server.write_effective_config(src_cfg, overrides, eff_cfg)
    print(f"run dir {run_dir}", flush=True)

    header = {"config": config_name, "model": model_key, "run_id": run_id, "set": sset.name, "n_items": len(items),
              "reference_build_error": reference_error,
              "limit": args.limit, "restarts": restarts, "concurrency": concurrency, "overrides": overrides,
              "worktree": str(worktree), "gpus": args.gpus, "port": args.port, "started": t_start,
              "reference": fam.reference_description() if fam.ref_outputs(ref_dir) else
              f"missing ({ref_dir})", "protocol": fam.protocol(), "family_opts": fam.opts}
    error = None
    runs: list[Path] = []
    startup = []
    try:
        if not args.no_node_health:
            header["node_health_before"] = server.node_health_snapshot(run_dir / "node_health_before.txt")
        requests = fam.requests(sset)
        for k in range(1, restarts + 1):
            socket_dir, upload_dir = server.private_dirs(args.tag or f"corr-{worktree.name}", f"{run_key}-r{k}")
            spec = server.ServerSpec(
                name=f"corr-{config_name}-{run_id}-r{k}"[:80], worktree=worktree, model=model_key,
                config_path=eff_cfg, gpus=args.gpus, port=args.port, log_path=run_dir / f"server_r{k}.log",
                socket_dir=socket_dir, upload_dir=upload_dir, timeout_s=args.startup_timeout,
                extra_env=_kv(args.env))
            if k == 1:
                header["versions"] = server.collect_versions(spec)
                (run_dir / "versions.json").write_text(json.dumps(header["versions"], indent=1))
                if not header["versions"].get("serves_worktree"):
                    raise RuntimeError(f"the server would import {header['versions']['stack'].get('mstar_file')}, "
                                       f"not {worktree}")
            print(f"[r{k}] starting server ({spec.name})", flush=True)
            # The checkout can move between instances; a noise floor across two commits isn't one.
            header.setdefault("served_heads", []).append(server.git_short(worktree))
            try:
                startup.append(round(server.start(spec), 1))
                print(f"[r{k}] ready after {startup[-1]}s; sending {len(requests)} requests "
                      f"(concurrency {concurrency})", flush=True)
                base = f"http://127.0.0.1:{args.port}"
                client.warmup(base, requests, args.warmup)
                t0 = time.monotonic()
                responses = client.send_all(base, requests, concurrency=concurrency, timeout_s=fam.request_timeout_s)
                out = run_dir / f"r{k}"
                summary = _save_run(fam, sset, responses, out)
                summary["wall_s"] = round(time.monotonic() - t0, 1)
                header.setdefault("runs", {})[f"r{k}"] = summary
                runs.append(out)
                print(f"[r{k}] {summary['ok']} ok, {summary['failed']} failed in {summary['wall_s']}s", flush=True)
            finally:
                try:
                    server.stop(spec)
                except Exception as e:
                    print(f"[r{k}] stop: {e}", flush=True)
        header["startup_s"] = startup
        if not args.no_node_health:
            header["node_health_after"] = server.node_health_snapshot(run_dir / "node_health_after.txt")
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
    header["startup_s"] = startup
    header["finished"] = _now()
    if error and not runs:
        verdict = write_reports(run_dir, header, None, error)
        print(f"verdict {verdict}: {run_dir / 'report.md'}")
        return EXIT[verdict]
    return _score_and_report(fam, sset, runs, ref_dir, run_dir, header, device, error)


def _build_reference(fam, sset, ref_dir: Path, device: str) -> None:
    print(f"building reference {fam.reference_id()} for {fam.set_name} on {device} -> {ref_dir}", flush=True)
    tmp = ref_dir.with_name(ref_dir.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "outputs").mkdir(parents=True)
    t0 = time.monotonic()
    fam.make_reference(sset, tmp, device)
    print(f"reference built in {time.monotonic() - t0:.0f}s", flush=True)
    if ref_dir.exists():
        shutil.rmtree(ref_dir)
    tmp.rename(ref_dir)


def _score_and_report(fam, sset, runs, ref_dir, run_dir, header, device, error=None) -> int:
    ref = ref_dir if fam.ref_outputs(ref_dir) is not None else None
    work = run_dir / "scoring"
    work.mkdir(exist_ok=True)
    score = None
    try:
        score = fam.score(sset, runs, ref, work, device)
        if header.get("limit"):
            score.notes.append(f"--limit {header['limit']}: a partial set; not a verdict for the config")
    except Exception:
        tb = traceback.format_exc()
        error = (error + "\n" + tb) if error else tb
        print(tb, file=sys.stderr, flush=True)
    verdict = write_reports(run_dir, header, score, error)
    if error and score is not None and verdict != "ERROR":
        verdict = write_reports(run_dir, header, score, error)
    print(f"verdict {verdict}: {run_dir / 'report.md'}", flush=True)
    return EXIT[verdict]


if __name__ == "__main__":
    sys.exit(main())
