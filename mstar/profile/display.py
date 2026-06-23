
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

    lines.append(rule)
    if len(present) >= 2:
        lines.append(" Timeline:")
        for (a_label, a_ts), (b_label, b_ts) in zip(present, present[1:]):
            lines.append(f"   {a_label + ' → ' + b_label:<40} {_ms(a_ts, b_ts)}")
        # Total spans the first to last recorded checkpoint.
        lines.append(f"   {'total':<40} {_ms(present[0][1], present[-1][1])}")
    else:
        lines.append(" Timeline: (no timing recorded)")

    # ---- per-node graph timings (summed over the request) -----------------
    if prof.graph_timings:
        lines.append(rule)
        lines.append(" Graph timings (CPU, summed over the request) [ms]:")
        for gt in sorted(prof.graph_timings, key=lambda g: g.total_time, reverse=True):
            label = f"{gt.node}:{gt.graph_walk}"
            if len(label) > 24:
                label = label[:23] + "…"
            lines.append(
                f"   {label:<24} n={gt.exec_count:<5}"
                f" tot {gt.total_time * 1e3:7.1f}"
                f"  fwd {gt.forward_time * 1e3:7.1f}"
                f"  pre {gt.preprocess_time * 1e3:6.1f}"
                f"  post {gt.postprocess_time * 1e3:6.1f}"
            )
    lines.append(sep)

    text = "\n".join(lines) + "\n"

    if filename:
        with open(filename, "a") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
        sys.stdout.flush()
