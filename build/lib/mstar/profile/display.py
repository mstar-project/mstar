
import sys

from mstar.profile.format import RequestProfile

_WIDTH = 60


def _human_bytes(n: int) -> str:
    """Render a byte count with a binary-prefixed, human-friendly unit."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def _ms(start: float, end: float) -> str:
    return f"{(end - start) * 1e3:8.1f} ms"


def _fmt_ms(seconds: float) -> str:
    """Milliseconds with 2 decimals under 10ms, 1 decimal otherwise."""
    ms = seconds * 1e3
    return f"{ms:.2f}" if ms < 10 else f"{ms:.1f}"


def pretty_print_profile(prof: RequestProfile, filename=None):
    """Render a single request's profile.

    Writes to ``filename`` (appended) when given, otherwise to stdout. Stage
    timings are only shown for segments whose endpoints were both recorded, so
    any checkpoint that wasn't stamped is simply skipped rather than reported
    as zero.
    """
    lines: list[str] = []
    sep = "=" * _WIDTH
    rule = "-" * _WIDTH

    lines.append(sep)
    lines.append(f" Request profile: {prof.rid}")
    lines.append(sep)

    # ---- inputs / outputs -------------------------------------------------
    if prof.inputs:
        lines.append(" Inputs:")
        for info in prof.inputs:
            lines.append(
                f"   {info.modality:<12} x{info.count:<4} {_human_bytes(info.total_bytes):>10}"
            )
    if prof.outputs:
        lines.append(" Outputs:")
        for info in prof.outputs:
            lines.append(
                f"   {info.modality:<12} x{info.count:<4} {_human_bytes(info.total_bytes):>10}"
            )

    # ---- timeline ---------------------------------------------------------
    t = prof.timing
    # Ordered checkpoints; consecutive pairs that are both present become a
    # labelled stage. Missing checkpoints (e.g. conductor-side) collapse so the
    # surrounding stages still join up.
    checkpoints = [
        ("recv", t.recv_time),
        ("preprocess done", t.preprocess_finish_time),
        ("conductor ingest", t.conductor_ingest_time),
        ("first chunk", t.first_chunk_time),
        ("last chunk", t.last_chunk_time),
        ("conductor done", t.conductor_finish_time),
        ("finish", t.finish_time),
    ]
    present = [(label, ts) for label, ts in checkpoints if ts is not None]
    # Some checkpoints race — e.g. the api server's ``last chunk`` and the
    # conductor's ``done`` signal arrive in either order — so order segments by
    # the actual timestamp rather than the nominal sequence. This keeps every
    # segment non-negative and reflects what really happened.
    present.sort(key=lambda lt: lt[1])

    lines.append(rule)
    if len(present) >= 2:
        lines.append(" Timeline:")
        for (a_label, a_ts), (b_label, b_ts) in zip(present, present[1:], strict=False):
            lines.append(f"   {a_label + ' → ' + b_label:<40} {_ms(a_ts, b_ts)}")
        # Total spans the first to last recorded checkpoint.
        lines.append(f"   {'total':<40} {_ms(present[0][1], present[-1][1])}")
    else:
        lines.append(" Timeline: (no timing recorded)")

    # ---- per-node graph timings ------------------------------------------
    # Grouped by node with aligned columns so values stack and scan vertically.
    # Each cell is "<total over request> (<avg per exec>)".
    if prof.graph_timings:
        timings = sorted(prof.graph_timings, key=lambda g: (g.node, g.graph_walk))
        walk_w = min(max(len(g.graph_walk) for g in timings), 18)
        prefix_w = 5 + walk_w + 8  # indent + walk + " n=NNNN"

        def _cell(total: float, n: int) -> str:
            return f"  {_fmt_ms(total):>8} ({_fmt_ms(total / n):>7})"

        lines.append(rule)
        lines.append(" Graph timings (CPU, ms) — total over request (avg per exec):")
        lines.append(
            " " * prefix_w + "".join(f"{c:^20}" for c in ("all", "fwd", "pre", "post*"))
        )
        last_node = None
        for gt in timings:
            if gt.node != last_node:
                lines.append(f"   {gt.node}")
                last_node = gt.node
            n = max(gt.exec_count, 1)
            prefix = f"     {gt.graph_walk:<{walk_w}} n={gt.exec_count}"
            lines.append(
                f"{prefix:<{prefix_w}}"
                + _cell(gt.total_time, n)
                + _cell(gt.forward_time, n)
                + _cell(gt.preprocess_time, n)
                + _cell(gt.postprocess_time, n)
            )
        lines.append(
            "\n   * post overlaps the next step / another in-flight batch under"
        )
        lines.append(
            "     speculative scheduling, so it is not necessarily additive."
        )

    # ---- tensor transfer (rx / tx) ---------------------------------------
    if prof.rx_info or prof.tx_info:
        lines.append(rule)
        lines.append(" Tensor transfer   size / transport-time / count:")
        if prof.rx_info:
            lines.append("   rx (received over the wire), by source → dest:")
            last = None
            for rx in sorted(
                prof.rx_info, key=lambda r: (r.source_entity, r.dest_entity, r.edge_name)
            ):
                key = (rx.source_entity, rx.dest_entity)
                if key != last:
                    lines.append(f"     {rx.source_entity} → {rx.dest_entity}")
                    last = key
                lines.append(
                    f"       {rx.edge_name:<22} {_human_bytes(rx.num_bytes):>10}"
                    f"  {_fmt_ms(rx.time):>8} ms  (x{rx.count})"
                )
        if prof.tx_info:
            # No dest: the sender registers/writes data without knowing (a priori)
            # which worker reads it, and may register data that's never sent.
            lines.append("\n   tx (registered/written for send), by source:")
            last = None
            for tx in sorted(prof.tx_info, key=lambda t: (t.source_entity, t.edge_name)):
                if tx.source_entity != last:
                    lines.append(f"     {tx.source_entity}")
                    last = tx.source_entity
                lines.append(
                    f"       {tx.edge_name:<22} {_human_bytes(tx.num_bytes):>10}"
                    f"  {_fmt_ms(tx.time):>8} ms  (x{tx.count})"
                )
    lines.append(sep)

    text = "\n".join(lines) + "\n"

    if filename:
        with open(filename, "a") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
        sys.stdout.flush()
