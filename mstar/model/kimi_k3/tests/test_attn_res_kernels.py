"""Kimi K3.1's attention-residual read kernels against the torch chain they replace, at the
release's geometry (hidden 7168, blocks of 12: up to 8 stack blocks and the stream), in Triton's
CPU interpreter (``attn_res_case.py``, a subprocess: the interpreter is chosen at import).

The quick group (the widest read on both kernel paths, a verify step, a prefill's sampled rows,
a strided stream, the custom ops under torch.compile, and the tiny model through the Engine on
the kernels) always runs; every depth at decode, verify and prefill sizes is opt-in (minutes)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

_CASE = Path(__file__).with_name("attn_res_case.py")
full = pytest.mark.skipif(not os.environ.get("MSTAR_INTERPRET_FULL_SHAPES"),
                          reason="set MSTAR_INTERPRET_FULL_SHAPES=1 (minutes in Triton's CPU interpreter)")


def _interpret(group: str, timeout: int = 1800) -> str:
    pytest.importorskip("triton")
    root = Path(__file__).resolve().parents[4]
    env = {**os.environ, "TRITON_INTERPRET": "1", "OMP_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")])}
    run = subprocess.run([sys.executable, str(_CASE), group], env=env, capture_output=True, text=True,
                         check=False, timeout=timeout)
    assert run.returncode == 0, (run.stdout[-3000:], run.stderr[-4000:])
    return run.stdout


def test_reads_match_the_torch_chain_and_serve_the_engine():
    out = _interpret("quick")
    assert out.count("\nok ") + out.startswith("ok ") == 10 and "ok engine" in out, out


@full
@pytest.mark.parametrize("group", ["decode", "verify", "prefill"])
def test_every_depth_at_decode_verify_and_prefill_sizes(group):
    _interpret(group)
