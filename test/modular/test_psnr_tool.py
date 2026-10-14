"""CPU tests for ``benchmark/flux2_klein/psnr.py``: pairwise PSNR and the directory-distribution mode."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "benchmark" / "flux2_klein" / "psnr.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("psnr_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pairs(tmp_path: Path):
    """Four indices: exact copy, one-bit change, halved image, and a missing served file."""
    ref_dir, other_dir = tmp_path / "ref", tmp_path / "served"
    ref_dir.mkdir()
    other_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        img = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
        Image.fromarray(img).save(ref_dir / f"sdpa_{i:03d}.png")
        other = img.copy()
        if i == 1:
            other[0, 0, 0] ^= 1
        if i == 2:
            other //= 2
        if i != 3:
            Image.fromarray(other).save(other_dir / f"mstar_{i:03d}.png")
    return ref_dir, other_dir


def test_psnr_values():
    tool = _load_tool()
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    assert tool.psnr(a, a) == math.inf
    b = a.copy()
    b[0, 0, 0] = 255
    expected = 20 * math.log10(255.0) - 10 * math.log10(255.0 ** 2 / 48)
    assert tool.psnr(a, b) == pytest.approx(expected)
    with pytest.raises(ValueError):
        tool.psnr(a, np.zeros((4, 5, 3), dtype=np.uint8))


def test_compare_dirs_and_summary(tmp_path):
    tool = _load_tool()
    ref_dir, other_dir = _write_pairs(tmp_path)
    values, missing = tool.compare_dirs(ref_dir, other_dir, "sdpa_{:03d}.png", "mstar_{:03d}.png", 0, 4)
    assert missing == [3]
    assert values[0] == math.inf
    assert 60 < values[1] < 80
    assert values[2] < 20
    summary = tool.summarize(values, 40.0)
    assert summary["n"] == 3 and summary["exact"] == 1
    assert summary["min"] == values[2] and summary["max"] == math.inf
    assert summary["median"] == values[1]
    assert summary["below"] == [2]
    assert tool.summarize({}, 40.0) == {"n": 0}


def test_cli_dirs_writes_json(tmp_path):
    ref_dir, other_dir = _write_pairs(tmp_path)
    out = tmp_path / "psnr.json"
    result = subprocess.run(
        [sys.executable, str(TOOL), "--dirs", str(ref_dir), str(other_dir), "--count", "4", "--json", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert "missing pairs: [3]" in result.stdout
    assert "below 40 dB: 1 [2]" in result.stdout
    payload = json.loads(out.read_text())
    assert payload["values"]["0"] is None  # inf is stored as null for portable JSON
    assert payload["summary"]["below"] == [2] and payload["missing"] == [3]


def test_cli_pairwise(tmp_path):
    ref_dir, other_dir = _write_pairs(tmp_path)
    result = subprocess.run(
        [sys.executable, str(TOOL), str(ref_dir / "sdpa_000.png"), str(other_dir / "mstar_000.png")],
        capture_output=True, text=True, check=True,
    )
    assert "inf dB" in result.stdout
