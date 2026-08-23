"""Offline step-cost queries against a stepdb — the Stage 0 deliverable.

This is the simulator's cost layer without the simulator: no event loop, no
scheduling, no request lifecycle. You ask "what does one step of this node at
this shape cost?" and get the measured answer, or an honest statement that it
was never measured.

That is enough to answer real capacity questions on its own — how decode step
time scales with batch size, what a KV-length increase costs, whether TP=2
pays for itself on a node — which is why it ships before the DES.

Two entry points:

``step``
    Price one (node, walk, shape) point.
``sweep``
    Price a node across every measured shape, so the batch-size / KV curve
    can be read off directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from mstar.sim.stepdb import Coverage, StepCost, StepDB, StepKey, pad_to_bucket


@dataclass
class StepQuery:
    """One thing to price."""

    node: str
    graph_walk: str
    bs: int
    num_tokens: int | None = None
    kv_len_total: int = 0
    tp_size: int = 1
    sp_size: int = 1
    requires_cfg: bool = False
    mode: str = "graph"


def _shape_buckets(db: StepDB, model: str, node: str, walk: str, mode: str):
    """Measured (padded_bs, padded_num_tokens) points for one node/walk."""
    return sorted({
        (k.padded_bs, k.padded_num_tokens)
        for k in db.keys(model)
        if k.node == node and k.graph_walk == walk and k.mode == mode
    })


def price_step(
    db: StepDB, model: str, q: StepQuery, snap_to_bucket: bool = True
) -> tuple[StepCost, StepKey]:
    """Price one step, snapping the requested shape onto a measured bucket.

    The engine itself pads a batch up to a captured bucket, so a query for
    bs=3 should be answered with the bs=4 row when that is what would run.
    ``snap_to_bucket=False`` disables the snap for callers that already know
    the padded shape (e.g. replaying a recorded step log).
    """
    bs = q.bs
    tokens = q.num_tokens if q.num_tokens is not None else q.bs

    if snap_to_bucket:
        buckets = _shape_buckets(db, model, q.node, q.graph_walk, q.mode)
        if buckets:
            padded_bs = pad_to_bucket(bs, {b for b, _ in buckets})
            if padded_bs is not None:
                token_opts = {t for b, t in buckets if b == padded_bs}
                padded_tokens = pad_to_bucket(tokens, token_opts)
                if padded_tokens is not None:
                    bs, tokens = padded_bs, padded_tokens

    key = StepKey(
        model=model, node=q.node, graph_walk=q.graph_walk,
        padded_bs=bs, padded_num_tokens=tokens,
        tp_size=q.tp_size, sp_size=q.sp_size,
        requires_cfg=q.requires_cfg, mode=q.mode,
    )
    return db.lookup(key, q.kv_len_total), key


def format_step(cost: StepCost, key: StepKey, req_bs: int) -> str:
    pad = "" if key.padded_bs == req_bs else f" (padded from bs={req_bs})"
    flag = "" if cost.coverage == Coverage.EXACT else f"  [{cost.coverage.describe()}]"
    return (
        f"{key.node}/{key.graph_walk} bs={key.padded_bs} "
        f"tok={key.padded_num_tokens} {key.mode}{pad}\n"
        f"  gpu   {cost.gpu_s * 1e3:8.3f} ms{flag}\n"
        f"  cpu   {cost.cpu_s * 1e3:8.3f} ms  "
        f"(prepare {cost.prepare_s * 1e3:.3f} / plan {cost.plan_s * 1e3:.3f} / "
        f"launch {cost.launch_s * 1e3:.3f} / sample {cost.sample_s * 1e3:.3f})\n"
        f"  step  {max(cost.gpu_s, cost.cpu_s) * 1e3:8.3f} ms  "
        f"= max(gpu, cpu)  — the steady-state cadence under speculation"
        + (f"\n  note: {cost.note}" if cost.note else "")
    )


def sweep_node(
    db: StepDB, model: str, node: str, walk: str, mode: str = "graph",
    kv_len_total: int = 0,
) -> list[tuple[StepKey, StepCost]]:
    """Price every measured shape of one node/walk, ascending by batch size."""
    out = []
    for key in db.keys(model):
        if key.node != node or key.graph_walk != walk or key.mode != mode:
            continue
        out.append((key, db.lookup(key, kv_len_total)))
    out.sort(key=lambda kc: (kc[0].padded_bs, kc[0].padded_num_tokens))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="mstar predict",
        description="Query measured per-step costs from a stepdb.",
    )
    p.add_argument("--db", required=True, help="Path to the stepdb SQLite file")
    p.add_argument("--model", default=None, help="Model key (default: the only one present)")
    p.add_argument("--gpu", default=None, help="GPU name to price for (default: this host's)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("step", help="price one step")
    s.add_argument("--node", required=True)
    s.add_argument("--walk", required=True)
    s.add_argument("--bs", type=int, required=True)
    s.add_argument("--tokens", type=int, default=None)
    s.add_argument("--kv", type=int, default=0, help="total KV context across the batch")
    s.add_argument("--tp", type=int, default=1)
    s.add_argument("--sp", type=int, default=1)
    s.add_argument("--cfg", action="store_true")
    s.add_argument("--mode", default="graph", choices=["graph", "eager", "sequential"])
    s.add_argument("--no-snap", action="store_true",
                   help="treat --bs/--tokens as already padded")

    w = sub.add_parser("sweep", help="price every measured shape of a node")
    w.add_argument("--node", required=True)
    w.add_argument("--walk", required=True)
    w.add_argument("--kv", type=int, default=0)
    w.add_argument("--mode", default="graph", choices=["graph", "eager", "sequential"])

    sub.add_parser("list", help="list what the stepdb covers")

    args = p.parse_args(argv)

    db = StepDB(args.db, gpu_name=args.gpu)
    try:
        models = db.models()
        if not models:
            print(f"stepdb {args.db} is empty — run a harvest first")
            return 1
        model = args.model or models[0]
        if model not in models:
            print(f"model {model!r} not in stepdb; have: {', '.join(models)}")
            return 1

        if args.cmd == "list":
            print(f"stepdb: {args.db}")
            print(f"gpu:    {db.gpu_name}")
            print(f"rows:   {db.count()}")
            print(f"models: {', '.join(models)}")
            by_node: dict[tuple[str, str, str], list[StepKey]] = {}
            for k in db.keys(model):
                by_node.setdefault((k.node, k.graph_walk, k.mode), []).append(k)
            for (node, walk, mode), keys in sorted(by_node.items()):
                shapes = sorted({(k.padded_bs, k.padded_num_tokens) for k in keys})
                print(f"\n  {node}/{walk} [{mode}] — {len(shapes)} shapes")
                print("    " + ", ".join(f"bs{b}/tok{t}" for b, t in shapes[:12])
                      + (" ..." if len(shapes) > 12 else ""))
            return 0

        if args.cmd == "step":
            q = StepQuery(
                node=args.node, graph_walk=args.walk, bs=args.bs,
                num_tokens=args.tokens, kv_len_total=args.kv,
                tp_size=args.tp, sp_size=args.sp,
                requires_cfg=args.cfg, mode=args.mode,
            )
            cost, key = price_step(db, model, q, snap_to_bucket=not args.no_snap)
            print(format_step(cost, key, args.bs))
            return 0 if cost.coverage != Coverage.MISSING else 2

        if args.cmd == "sweep":
            rows = sweep_node(db, model, args.node, args.walk, args.mode, args.kv)
            if not rows:
                print(f"no measured shapes for {args.node}/{args.walk} [{args.mode}]")
                return 2
            print(f"{args.node}/{args.walk} [{args.mode}] on {db.gpu_name}, "
                  f"kv={args.kv}")
            print(f"{'bs':>5} {'tokens':>7} {'gpu ms':>9} {'cpu ms':>9} "
                  f"{'step ms':>9} {'ms/req':>9}  coverage")
            for key, cost in rows:
                step_ms = max(cost.gpu_s, cost.cpu_s) * 1e3
                print(
                    f"{key.padded_bs:5d} {key.padded_num_tokens:7d} "
                    f"{cost.gpu_s * 1e3:9.3f} {cost.cpu_s * 1e3:9.3f} "
                    f"{step_ms:9.3f} {step_ms / key.padded_bs:9.3f}  "
                    f"{cost.coverage.describe()}"
                )
            return 0
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
