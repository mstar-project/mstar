"""The reference-clip diagnostic must exercise real, intact input files."""

from pathlib import Path

import numpy as np
import pytest

from benchmark.fish_s2_pro_ref_audio_probe import main, run_probe

soundfile = pytest.importorskip("soundfile")


def test_probe_measures_headerless_flac_without_deleting_input(tmp_path: Path):
    rate = 44_100
    wave = (np.sin(np.arange(2 * rate) / 20) * 8_000).astype(np.int16)
    wav, flac = tmp_path / "reference.wav", tmp_path / "reference.flac"
    soundfile.write(wav, wave, rate)
    soundfile.write(flac, wave, rate, format="FLAC", subtype="PCM_16")
    data = bytearray(flac.read_bytes())
    data[21] &= 0xF0
    data[22:42] = bytes(20)
    flac.write_bytes(data)

    report = run_probe(wav, flac, repeat=1)

    assert [sample["format"] for sample in report["samples"]] == ["wav", "flac"]
    assert report["cache_condition"].startswith("warm:")
    assert report["samples"][1]["header_frames"] == 2**63 - 1
    assert report["samples"][1]["decoded_samples"] < len(wave)
    assert all(sample["validation_ms"] >= 0 for sample in report["samples"])
    assert all(sample["load_ms"] >= 0 for sample in report["samples"])
    assert wav.is_file() and flac.is_file()


def test_probe_refuses_invalid_repeat_before_launch(tmp_path: Path):
    wav, flac = tmp_path / "reference.wav", tmp_path / "reference.flac"
    wav.touch()
    flac.touch()
    with pytest.raises(ValueError, match="--repeat"):
        run_probe(wav, flac, repeat=0)


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_output_cannot_overwrite_an_input(tmp_path: Path, alias: str):
    wav, flac = tmp_path / "reference.wav", tmp_path / "reference.flac"
    wav.write_bytes(b"original wav")
    flac.write_bytes(b"original flac")
    output = wav if alias == "same" else tmp_path / "report.json"
    if alias == "symlink":
        output.symlink_to(wav)
    elif alias == "hardlink":
        output.hardlink_to(wav)

    with pytest.raises(SystemExit) as exc:
        main(["--wav", str(wav), "--flac", str(flac), "--output", str(output)])

    assert exc.value.code == 2
    assert wav.read_bytes() == b"original wav"
    assert flac.read_bytes() == b"original flac"
