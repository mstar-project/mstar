"""DROIDDataset's task-string lookup from a lerobot v3.0 tasks.parquet (CPU)."""

import pytest

pd = pytest.importorskip("pandas")

from benchmark.dataset import _task_strings  # noqa: E402


def test_task_strings_from_index():
    """lerobot v3.0, as droid_100 ships it: the instruction is the index."""
    df = pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(["Put the marker in the pot", "Move the sharpie"]))
    assert _task_strings(df) == {0: "Put the marker in the pot", 1: "Move the sharpie"}
