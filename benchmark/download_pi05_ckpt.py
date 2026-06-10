#!/usr/bin/env python3
"""Download the pi0.5 checkpoint and print the local path.
    conda activate openpi
    python benchmark/download_pi05_ckpt.py
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_CONFIG = "pi05_droid"
DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_droid"


def main():
    p = argparse.ArgumentParser(description="Download pi0.5 checkpoint")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = p.parse_args()

    try:
        from openpi.shared import download
        from openpi.training import config as _config
    except ImportError as e:
        sys.exit(
            f"\n[ERROR] openpi is not importable ({e}).\n"
            "Run inside the openpi conda env:\n"
            "    conda activate openpi\n"
        )

    _config.get_config(args.config)
    ckpt_dir = download.maybe_download(args.checkpoint)
    print(ckpt_dir)


if __name__ == "__main__":
    main()