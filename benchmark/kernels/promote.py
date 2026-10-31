"""Write a kernel-choice table from the results store.

    python -m benchmark.kernels.promote decode_gemm --store DIR [--margin 0.01] [--root DIR]
        [--device "AMD Instinct MI325X"]

Run it once a store holds a search of every shape the table should cover, and
ship the table after a model-level A/B. ``--device`` names the GPU when run on
a host without one.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from benchmark.kernels.specs import decode_gemv
from benchmark.kernels.store import ResultsStore
from mstar.platforms import current_platform
from mstar.platforms.facts import facts_for

TABLES = {"decode_gemm": decode_gemv.promote}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("table", choices=sorted(TABLES))
    parser.add_argument("--store", required=True)
    parser.add_argument("--margin", type=float, default=0.01, help="what leaving a shape's default must gain")
    parser.add_argument("--root", help="config root to write under (default: the tree's)")
    parser.add_argument("--device", help="a device name, on a host without the GPU")
    parser.add_argument("--arch", default="gfx942", help="the architecture, with --device")
    args = parser.parse_args(argv)
    facts = facts_for("rocm", name=args.device, arch=args.arch) if args.device else current_platform.device_facts()
    path = TABLES[args.table](ResultsStore(args.store), facts, args.margin, root=Path(args.root) if args.root else None,
                              provenance={"store": args.store})
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
