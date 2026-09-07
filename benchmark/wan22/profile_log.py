"""Parse the server's ``--log-stats-file`` request profiles into phase timings.

The server appends one pretty-printed request profile per finished request. This
module is the inverse of that renderer for the sections the benchmark needs: the
timeline stage spans, and the per-node total/fwd/pre/post ms with exec counts.

It parses text rather than reading JSON because the profiler ships only a
human-readable renderer, and this benchmark must not modify shared profiling code.

wan22's graph nodes map one-to-one onto generation phases: ``text_encoder``,
``vae_encoder`` (I2V only), ``dit`` (denoise — its ``exec_count`` equals
``num_inference_steps``, since the node runs once per UniPC step), and
``vae_decoder`` (latent to pixel). Note the decoder node emits raw frames; the mp4
muxing happens downstream of it, so its time is decode, not encode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Arrow used by display.py timeline rows ("recv → preprocess done").
_ARROW = "→"

# A graph-timing data row: "  <walk> n=<count>  <all> (<avg>)  <fwd> (..)  <pre> (..)  <post> (..)".
# We keep the first number of each of the four "total (avg)" cells (the totals).
_NUM = r"([-\d.]+)"
_CELL = rf"{_NUM}\s*\(\s*{_NUM}\s*\)"
_GRAPH_ROW_RE = re.compile(
    rf"^\s+\S.*?\bn=(\d+)\s+{_CELL}\s+{_CELL}\s+{_CELL}\s+{_CELL}\s*$"
)
# A node header line under "Graph timings": 3-space indent, a bare node token, EOL.
_NODE_HDR_RE = re.compile(r"^   (\S+)\s*$")

# Timeline rows: "   <label>            <val> ms". Label may contain the arrow.
_TIMELINE_RE = re.compile(r"^\s+(.+?)\s{2,}([-\d.]+)\s*ms\s*$")

# Inputs/Outputs rows: "   <modality>   x<count>   <size> <unit>".
_IO_RE = re.compile(r"^\s+(\w+)\s+x(\d+)\s+([\d.]+)\s*(B|KiB|MiB|GiB|TiB)\s*$")

_UNIT_BYTES = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}


def _to_bytes(value: float, unit: str) -> int:
    return int(round(value * _UNIT_BYTES[unit]))


@dataclass
class NodeTiming:
    """Per-node totals over one request (all values in milliseconds)."""

    exec_count: int = 0
    total_ms: float = 0.0
    fwd_ms: float = 0.0
    pre_ms: float = 0.0
    post_ms: float = 0.0

    def merge(self, other: "NodeTiming") -> None:
        """Fold a second walk's row for the same node into this one."""
        self.exec_count += other.exec_count
        self.total_ms += other.total_ms
        self.fwd_ms += other.fwd_ms
        self.pre_ms += other.pre_ms
        self.post_ms += other.post_ms


@dataclass
class RequestProfile:
    """One parsed request profile block."""

    rid: str
    timeline_ms: dict[str, float] = field(default_factory=dict)  # segment label -> ms
    total_ms: float | None = None                                # "total" row
    nodes: dict[str, NodeTiming] = field(default_factory=dict)   # node name -> timing
    output_bytes: dict[str, int] = field(default_factory=dict)   # modality -> bytes

    # --- convenience accessors used by the benchmark schema ---
    def node_ms(self, node: str) -> float | None:
        nt = self.nodes.get(node)
        return nt.total_ms if nt else None

    def denoise_step_mean_ms(self) -> float | None:
        """Mean per-denoise-step wall = dit total / exec_count (== steps)."""
        nt = self.nodes.get("dit")
        if not nt or nt.exec_count == 0:
            return None
        return nt.total_ms / nt.exec_count

    def denoise_steps(self) -> int | None:
        nt = self.nodes.get("dit")
        return nt.exec_count if nt else None

    def timeline_seg(self, label: str) -> float | None:
        """A timeline span by its 'a -> b' label (arrow-normalised)."""
        return self.timeline_ms.get(label)


def _parse_block(text: str, rid: str) -> RequestProfile:
    prof = RequestProfile(rid=rid)
    section = None            # "timeline" | "graph" | "outputs" | "inputs" | None
    current_node: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Timeline:"):
            section, current_node = "timeline", None
            continue
        if stripped.startswith("Graph timings"):
            section, current_node = "graph", None
            continue
        if stripped.startswith("Outputs:"):
            section, current_node = "outputs", None
            continue
        if stripped.startswith("Inputs:"):
            section, current_node = "inputs", None
            continue
        if stripped.startswith("Tensor transfer"):
            section, current_node = None, None
            continue

        if section == "timeline":
            m = _TIMELINE_RE.match(line)
            if m:
                label, val = m.group(1).strip(), float(m.group(2))
                if label == "total":
                    prof.total_ms = val
                else:
                    prof.timeline_ms[label.replace(_ARROW, "->").replace("  ", " ")] = val
            continue

        if section == "outputs":
            m = _IO_RE.match(line)
            if m:
                modality, _count, size, unit = m.group(1), m.group(2), float(m.group(3)), m.group(4)
                prof.output_bytes[modality] = _to_bytes(size, unit)
            continue

        if section == "graph":
            row = _GRAPH_ROW_RE.match(line)
            if row:
                nt = NodeTiming(
                    exec_count=int(row.group(1)),
                    total_ms=float(row.group(2)),
                    fwd_ms=float(row.group(4)),
                    pre_ms=float(row.group(6)),
                    post_ms=float(row.group(8)),
                )
                node = current_node or "?"
                if node in prof.nodes:
                    prof.nodes[node].merge(nt)
                else:
                    prof.nodes[node] = nt
                continue
            hdr = _NODE_HDR_RE.match(line)
            # A node header is a bare token line that is NOT a data row and not the
            # column header (which starts with many spaces, matched by neither).
            if hdr and "n=" not in line and hdr.group(1) not in ("all", "fwd", "pre", "post*"):
                current_node = hdr.group(1)
            continue

    return prof


def parse_profiles(text: str) -> list[RequestProfile]:
    """Parse every request-profile block in a ``--log-stats-file`` dump, in file order.

    Blocks are delimited by the "Request profile: <rid>" header. Ordering is
    preserved so callers can correlate with client-side request order (exact at
    concurrency 1; approximate — by server recv order — at higher concurrency).
    """
    # Find every rid header, keeping each header with its following body.
    header_positions: list[tuple[int, str]] = []
    for m in re.finditer(r"Request profile:\s*(\S+)", text):
        header_positions.append((m.start(), m.group(1)))
    profiles: list[RequestProfile] = []
    for i, (pos, rid) in enumerate(header_positions):
        end = header_positions[i + 1][0] if i + 1 < len(header_positions) else len(text)
        profiles.append(_parse_block(text[pos:end], rid))
    return profiles


def parse_profile_file(path: str) -> list[RequestProfile]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return parse_profiles(f.read())
