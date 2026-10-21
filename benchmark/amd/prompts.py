"""Write the prompts of one of the correctness runner's sample sets as a one-per-line text file,
for the scripts that take a prompt or sentence file (run_tts_benchmark.sh, tts_speech_bench.py,
chatterbox/bench_all.sh, flux2_klein/bench_images.py).

    python -m benchmark.amd.prompts tts_seedtts_v1 --out sentences.txt
    python -m benchmark.amd.prompts image_v1 --out prompts.txt --unique
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SETS_ROOT = Path("/work1/stephaniewang/atindra/amd/results/correctness/data/sets")


def set_prompts(set_dir: Path, unique: bool = False) -> list[str]:
    items = json.loads((set_dir / "manifest.json").read_text(encoding="utf-8"))["items"]
    prompts = [" ".join(it.get("prompt", "").split()) for it in items]
    if any(not p for p in prompts):
        raise ValueError(f"{set_dir} has items without a text prompt")
    if unique:
        prompts = list(dict.fromkeys(prompts))
    return prompts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", help="set name under --sets-root, or a set directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets-root", default=str(SETS_ROOT))
    ap.add_argument("--unique", action="store_true", help="drop repeated prompts (image_v1 repeats each per size)")
    args = ap.parse_args(argv)
    d = Path(args.set) if Path(args.set).is_dir() else Path(args.sets_root) / args.set
    text = "\n".join(set_prompts(d, args.unique)) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    print(f"{args.out}: {text.count(chr(10))} prompts from {d} (sha256 {digest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
