"""Digest-checked local recordings; never download or substitute an audio corpus."""

import hashlib
import json
import wave
from pathlib import Path

from benchmark.base import PersonaPlex, RequestType
from benchmark.dataset import BaseDataset
from benchmark.request import RequestInput


def read_manifest(filename):
    path = Path(filename).resolve()
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("audio manifest must be a nonempty JSON list")
    checked = []
    for entry in entries:
        audio = (path.parent / entry["path"]).resolve()
        if hashlib.sha256(audio.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"audio SHA256 mismatch: {audio}")
        with wave.open(str(audio), "rb") as wav:
            if wav.getcomptype() != "NONE" or not wav.getnframes():
                raise ValueError(f"expected nonempty uncompressed WAV: {audio}")
            duration = wav.getnframes() / wav.getframerate()
        checked.append({**entry, "path": str(audio), "duration_s": duration})
    return checked


class LocalAudioDataset(BaseDataset):
    def __init__(self, manifest, num_requests, req_type):
        if req_type != RequestType.A2S:
            raise ValueError("local_audio currently supports audio_to_speech")
        self._num_requests = num_requests
        self.entries = read_manifest(manifest)
        self.items = [
            RequestInput(
                req_type=req_type,
                audio_path=e["path"],
                prompt=e.get("persona", PersonaPlex.DEFAULT_PERSONA),
            )
            for e in self.entries
        ]

    @property
    def num_requests(self):
        return self._num_requests

    def __len__(self):
        return self._num_requests

    def __getitem__(self, index):
        return self.items[index % len(self.items)]
