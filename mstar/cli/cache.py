"""``mstar cache``: compile-cache bundles (docs/compile_cache.rst).

    mstar cache key                  # this host's stack key, the bundle path a pack writes
    mstar cache dirs                 # the cache dirs a server started here reads and writes
    mstar cache pack ROOT            # after a warm-up: add this host's caches to ROOT/<key>/
    mstar cache unpack SRC           # before serving: the bundle for this stack, into the cache dirs
    mstar cache show ROOT            # the bundles under ROOT
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# unpack's exit status when SRC has no bundle for this stack, so callers can serve cold
EXIT_NO_MATCH = 3


def _key(args: argparse.Namespace) -> None:
    from mstar.utils import compile_bundle

    key = compile_bundle.stack_key()
    print(key.path)
    print(f"  product {key.product}, arch {key.arch}, torch {key.torch}, {key.runtime}, triton {key.triton}")


def _dirs(args: argparse.Namespace) -> None:
    from mstar.utils import compile_bundle

    for kind, path in compile_bundle.cache_dirs().items():
        print(f"{kind:15s} {path}")


def _pack(args: argparse.Namespace) -> None:
    from mstar.utils import compile_bundle

    dirs = compile_bundle.cache_dirs()
    if args.kinds:
        wanted = args.kinds.split(",")
        unknown = sorted(set(wanted) - set(dirs))
        if unknown:
            sys.exit(f"error: no {', '.join(unknown)} cache here; kinds here: {', '.join(dirs)}")
        dirs = {k: v for k, v in dirs.items() if k in wanted}
    t0 = time.monotonic()
    bundle, _ = compile_bundle.pack(Path(args.root), compile_bundle.stack_key(), dirs, note=args.note)
    print(f"packed into {bundle} in {time.monotonic() - t0:.1f} s")


def _unpack(args: argparse.Namespace) -> None:
    from mstar.utils import compile_bundle

    t0 = time.monotonic()
    try:
        done = compile_bundle.unpack(Path(args.src), compile_bundle.stack_key(), compile_bundle.cache_dirs(),
                                     force=args.force)
    except compile_bundle.BundleMismatchError as exc:
        print(f"not unpacked: {exc}", file=sys.stderr)
        sys.exit(EXIT_NO_MATCH)
    kinds = ", ".join(f"{k} -> {d}" for k, d in done.kinds.items())
    print(f"unpacked {done.bundle} in {time.monotonic() - t0:.1f} s: {kinds}")


def _show(args: argparse.Namespace) -> None:
    from mstar.utils import compile_bundle

    root = Path(args.root)
    bundles = compile_bundle.find_bundles(root)
    if not bundles:
        sys.exit(f"no bundles under {root}")
    for b in bundles:
        manifest = compile_bundle.read_manifest(b) or {}
        tars = sorted(b.glob("*.tar"))
        size = sum(t.stat().st_size for t in tars) / 2**30
        print(f"{b}: {size:.2f} GiB in {', '.join(t.stem for t in tars)}; "
              f"updated {manifest.get('updated', 'unknown (no manifest)')}")
        for h in manifest.get("history", [])[-3:]:
            print(f"  {h['time']} {h['host']} mstar {h.get('mstar')}: added {h['added']}"
                  + (f" ({h['note']})" if h.get("note") else ""))


def add_parser(sub: argparse._SubParsersAction) -> None:
    cache = sub.add_parser(
        "cache", help="pack or unpack compile-cache bundles, so a new host or container starts warm",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    csub = cache.add_subparsers(dest="cache_command", required=True)
    csub.add_parser("key", help="this host's stack key (GPU, torch, HIP or CUDA, Triton)").set_defaults(func=_key)
    csub.add_parser("dirs", help="the cache directories a server started here uses").set_defaults(func=_dirs)
    pack = csub.add_parser("pack", help="add this host's caches to the bundle for its stack under ROOT")
    pack.add_argument("root", help="bundle root; the bundle is ROOT/<stack key>/")
    pack.add_argument("--note", default=None, help="recorded in the manifest, e.g. which configs were warmed")
    pack.add_argument("--kinds", default=None, help="comma list of caches to pack (default: all here)")
    pack.set_defaults(func=_pack)
    unpack = csub.add_parser("unpack", help="extract the bundle for this stack into the cache dirs")
    unpack.add_argument("src", help="a bundle, or a root holding one per stack")
    unpack.add_argument("--force", action="store_true", help="unpack a bundle built for another stack")
    unpack.set_defaults(func=_unpack)
    show = csub.add_parser("show", help="list the bundles under ROOT")
    show.add_argument("root")
    show.set_defaults(func=_show)
