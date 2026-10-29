"""List every kernel op: ``python -m mstar.kernels [--probe] [--json]``.

``--probe`` also resolves each op on this process's platform (imports or
builds its implementation) and says which runs, or why none can.
"""

import argparse
import json

from mstar.kernels.registry import load_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", action="store_true", help="resolve each op on this platform")
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args()
    rows = [op.describe(probe=args.probe) for op in load_all()]
    if args.json:
        print(json.dumps(rows, indent=1, default=str))
        return
    for row in rows:
        print(f"{row['op']}{row['schema']}")
        print(f"  impls: {', '.join(row['impls']) or '-'}   reference: {row['reference']}"
              f"   fallback to reference: {row['fallback_to_reference']}")
        print(f"  tolerance: {row['tolerance']['kind']}   shape classes: {', '.join(row['shape_classes']) or '-'}")
        print(f"  warm shapes: {row['warm_shapes']}")
        if args.probe:
            print(f"  on {row['platform']}: runs {row['runs']}"
                  + (f" ({row['unavailable_reason']})" if row["unavailable_reason"] else ""))


if __name__ == "__main__":
    main()
