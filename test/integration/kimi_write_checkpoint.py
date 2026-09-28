"""Write the reduced random-weight Kimi checkpoint that the
configs/synthetic/kimi_k2_7_*.yaml deployments load: the ``reduced`` config
variant only, bf16, no quantization. Output tokens are meaningless — this is
plumbing-only, for exercising the served path without a real checkpoint.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kimi_reference import write_checkpoint  # noqa: E402

from mstar.model.kimi_k2_7.config import KimiK2Config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_checkpoint(out, KimiK2Config.reduced(), seed=args.seed)
    print(out / "model.safetensors")


if __name__ == "__main__":
    main()
