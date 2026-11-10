"""Shared pieces of the Step-Audio2 parity harness: paths, the oracle, metrics, loading.

The oracle is recorded once by the reference (``record_oracle.py`` for the bf16 LLM side and
the fp32 token2wav stream, ``record_fp32.py`` for the fp32 third configuration); see
``README`` next to those scripts. Paths default to this pod's layout and can be moved with
``STEPAUDIO2_CKPT`` / ``STEPAUDIO2_ORACLE`` / ``STEPAUDIO2_ASSETS``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NamedTuple

import torch

CKPT = Path(os.environ.get("STEPAUDIO2_CKPT", "/workspace/stepaudio2-ref/hf/Step-Audio-2-mini"))
ORACLE = Path(os.environ.get("STEPAUDIO2_ORACLE", "/workspace/stepaudio2-ref/oracle/rec"))
ASSETS = Path(os.environ.get("STEPAUDIO2_ASSETS", "/workspace/stepaudio2-ref/github/Step-Audio2/assets"))
DEVICE = torch.device("cuda")
CASES = ("t2t", "s2t", "s2s", "s2t_long")
AUDIO_CASES = ("s2t", "s2t_long")


def oracle(name: str) -> dict:
    """``llm_<case>`` (bf16), ``llm_s2t_math``, ``fp32_<case>``, ``token2wav_stream``."""
    return torch.load(ORACLE / f"{name}.pt", weights_only=False)


class Diff(NamedTuple):
    max_abs: float
    rel: float  # ||got - want|| / ||want||

    def __str__(self) -> str:
        return f"max {self.max_abs:9.3e}  rel {self.rel:9.3e}"


def diff(got: torch.Tensor, want: torch.Tensor) -> Diff:
    got, want = got.detach().double().cpu(), want.detach().double().cpu()
    if got.shape != want.shape:
        raise ValueError(f"shape {tuple(got.shape)} vs {tuple(want.shape)}")
    d = got - want
    return Diff(d.abs().max().item(), (d.norm() / want.norm().clamp_min(1e-30)).item())


def setup_torch() -> None:
    if not torch.cuda.is_available():
        sys.exit("the Step-Audio2 parity harness needs a GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    os.environ.setdefault("MSTAR_WORKSPACE_BUFFER_MB", "256")


class Verdict:
    """Collects pass/fail lines; ``finish`` exits nonzero on any failure."""

    def __init__(self):
        self.failures: list[str] = []

    def check(self, ok: bool, what: str) -> None:
        print(("PASS " if ok else "FAIL ") + what, flush=True)
        if not ok:
            self.failures.append(what)

    def finish(self) -> None:
        if self.failures:
            print(f"\n{len(self.failures)} check(s) failed", flush=True)
            sys.exit(1)
        print("\nall checks passed", flush=True)
