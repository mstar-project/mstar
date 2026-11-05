"""DROIDDataset over lerobot v3.0 droid_100 metadata (CPU)."""

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from benchmark.dataset import _task_strings  # noqa: E402


def test_task_strings_from_index():
    """lerobot v3.0, as droid_100 ships it: the instruction is the index."""
    df = pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(["Put the marker in the pot", "Move the sharpie"]))
    assert _task_strings(df) == {0: "Put the marker in the pot", 1: "Move the sharpie"}


def test_episode_video_reads_the_episode_segment_of_a_shared_file():
    """lerobot v3.0 packs many episodes per video file; episode 1 starts where episode 0 ends (#1026)."""
    from benchmark.dataset import _episode_video

    info = {"fps": 15, "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    col = "videos/cam/"
    episode = {col + "chunk_index": 0, col + "file_index": 1, col + "from_timestamp": 11.066667}
    frames = [{"frame_index": i} for i in range(3)]
    assert _episode_video(info, episode, "cam", frames) == ("videos/cam/chunk-000/file-001.mp4", [166, 167, 168])


def test_the_dataset_builds_offline_from_a_warm_cache(tmp_path, monkeypatch):
    """With droid_100 cached and HF_HUB_OFFLINE=1, no step may need the Hub's API."""
    datasets = pytest.importorskip("datasets")
    from huggingface_hub import constants

    from benchmark import dataset

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    repo = tmp_path / "hub" / "datasets--lerobot--droid_100"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("0" * 40)
    snapshot = repo / "snapshots" / ("0" * 40)
    (snapshot / "meta" / "episodes").mkdir(parents=True)
    (snapshot / "meta" / "info.json").write_text(
        '{"fps": 15, "features": {"cam": {"dtype": "video"}},'
        ' "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}'
    )
    pd.DataFrame({"episode_index": [0], "videos/cam/chunk_index": [0], "videos/cam/file_index": [0],
                  "videos/cam/from_timestamp": [0.0]}).to_parquet(snapshot / "meta" / "episodes" / "file-000.parquet")
    (snapshot / "videos" / "cam" / "chunk-000").mkdir(parents=True)
    (snapshot / "videos" / "cam" / "chunk-000" / "file-000.mp4").touch()
    rows = [{"episode_index": 0, "frame_index": 0, "action": [0.0] * 7, "observation.state": [1.0] * 8}]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: datasets.Dataset.from_list(rows))
    monkeypatch.setattr(dataset, "_decode_frames_to_png_and_video",
                        lambda video, frames, png_path, mp4_path: Path(png_path).touch())

    ds = dataset.DROIDDataset(str(tmp_path / "out"), num_requests=1, cache_dir=str(tmp_path / "hub"))
    assert [r.model_kwargs["robot_state"] for r in ds.items] == [[1.0] * 8]
