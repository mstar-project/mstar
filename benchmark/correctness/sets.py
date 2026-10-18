"""Fixed sample sets for the correctness runner.

A set is <data_root>/sets/<name>/ with a manifest.json and its media, built once from a pinned
source with seeded picks. Don't rebuild one in place, make a _v2 so references stay valid.
load_set checks every file's sha256.

    python -m benchmark.correctness.sets build all
    python -m benchmark.correctness.sets list
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/work1/stephaniewang/atindra/amd/results/correctness/data")
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Item:
    """One sample with an id, a prompt, media paths relative to the set dir, and params."""

    id: str
    prompt: str = ""
    files: dict[str, str] = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    expected: dict = field(default_factory=dict)


@dataclass
class SampleSet:
    name: str
    root: Path
    manifest: dict
    items: list[Item]

    def path(self, rel: str) -> Path:
        return self.root / rel


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(root: Path, name: str, source: dict, items: list[Item], notes: str = "") -> None:
    files = sorted({rel for it in items for rel in it.files.values()})
    manifest = {
        "name": name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "builder": "benchmark/correctness/sets.py",
        "source": source,
        "notes": notes,
        "items": [it.__dict__ for it in items],
        "sha256": {rel: _sha256(root / rel) for rel in files},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")


def load_set(name: str, data_root: Path = DEFAULT_DATA_ROOT, verify: bool = True) -> SampleSet:
    root = Path(data_root) / "sets" / name
    mpath = root / "manifest.json"
    if not mpath.is_file():
        raise FileNotFoundError(f"sample set {name!r} is not built ({mpath}); run "
                                f"`python -m benchmark.correctness.sets build {name}`")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if verify:
        bad = [rel for rel, digest in manifest.get("sha256", {}).items() if _sha256(root / rel) != digest]
        if bad:
            raise RuntimeError(f"sample set {name!r}: files changed since it was built: {bad[:5]}")
    items = [Item(**it) for it in manifest["items"]]
    return SampleSet(name, root, manifest, items)


# ---------------------------------------------------------------------------
# text: 32 prompts (general, multilingual, structured, edge cases, long context)
# ---------------------------------------------------------------------------

TEXT_PROMPTS: list[tuple[str, str]] = [
    ("en_explain", "Explain how a hash table handles collisions, in three short paragraphs."),
    ("en_math", "A train leaves at 9:40 and travels 210 km at 84 km/h. At what time does it arrive? Show your steps."),
    ("en_code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."),
    ("en_code_explain", "What does this shell command do? find . -name '*.log' -mtime +7 -exec gzip {} \\;"),
    ("en_summary", "Summarize in two sentences: The Industrial Revolution began in Britain in the late 18th century. "
     "It moved production from hand tools to machines, from homes to factories, and drove rapid urban growth. "
     "Steam power, iron making and textile manufacturing were its early engines, and it spread to Europe and "
     "North America during the 19th century."),
    ("en_poem", "Write a four-line rhyming poem about a lighthouse in winter."),
    ("en_facts", "List the eight planets of the solar system in order from the Sun, one per line."),
    ("en_reason", "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Explain."),
    ("en_advice", "Give five practical tips for writing clear technical documentation."),
    ("en_history", "Who was Ada Lovelace and why is she remembered?"),
    ("zh", "请用三句话介绍一下长城的历史。"),
    ("ja", "日本の四季について、それぞれの特徴を簡単に説明してください。"),
    ("ko", "김치를 만드는 기본 과정을 단계별로 설명해 주세요."),
    ("de", "Erkläre in einfachen Worten, wie eine Wärmepumpe funktioniert."),
    ("fr", "Quels sont les principaux monuments à visiter à Paris ? Donne une courte description de chacun."),
    ("es", "Escribe una receta sencilla de tortilla de patatas."),
    ("ru", "Объясни, почему небо голубое, простыми словами."),
    ("ar", "ما هي فوائد القراءة اليومية؟ اذكر ثلاث فوائد."),
    ("hi", "भारत की राजधानी के बारे में तीन वाक्य लिखिए।"),
    ("pt", "Traduza para o inglês: 'O conhecimento é a única riqueza que ninguém pode roubar.'"),
    ("json", "Return a JSON object with keys name, age and city for a fictional person. Output only the JSON."),
    ("table", "Make a markdown table of three programming languages with columns: language, year created, typing."),
    ("count", "Count from 1 to 30, separated by commas."),
    ("arith", "Compute 3847 * 29 step by step."),
    ("short_hi", "Hi"),
    ("one_token", "What is 2+2? Answer with just the number."),
    ("repeat", "Repeat the word 'banana' exactly 40 times, separated by spaces."),
    ("long_story", "Tell a detailed story about a robot who learns to paint, at least 300 words long."),
]

# (id, rough prompt length in characters, needle position as a fraction)
LONG_CONTEXT = [("ctx_1k", 6_000, 0.5), ("ctx_4k", 16_000, 0.3), ("ctx_8k", 32_000, 0.7), ("ctx_16k", 64_000, 0.5)]
NEEDLE = "Note for the reader: the maintenance passphrase for the archive server is VIOLET-ANCHOR-4721."
NEEDLE_QUESTION = ("\n\nQuestion: According to the documentation above, what is the maintenance passphrase for the "
                   "archive server? Then name one topic the documentation covers.")


def _long_context_source() -> str:
    """English text for the long-context prompts. The repo docs, frozen into the manifest."""
    parts = []
    for name in ("models.rst", "adding_models.rst", "serving.rst", "installation.rst"):
        p = REPO_ROOT / "docs" / name
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8"))
    text = "\n\n".join(parts)
    if len(text) < 70_000:
        raise RuntimeError("not enough source text for the long-context prompts")
    return text


def build_text_v1(root: Path) -> None:
    items = [Item(id=f"{i:02d}_{tag}", prompt=p, params={"max_new_tokens": 128}) for i, (tag, p) in
             enumerate(TEXT_PROMPTS)]
    source = _long_context_source()
    for j, (tag, chars, depth) in enumerate(LONG_CONTEXT):
        body = source[:chars]
        cut = body.rfind("\n", 0, int(chars * depth))
        cut = cut if cut > 0 else int(chars * depth)
        prompt = ("Read the following documentation.\n\n" + body[:cut] + "\n" + NEEDLE + "\n" + body[cut:]
                  + NEEDLE_QUESTION)
        items.append(Item(id=f"{len(TEXT_PROMPTS) + j:02d}_{tag}", prompt=prompt,
                          params={"max_new_tokens": 128}, expected={"needle": "VIOLET-ANCHOR-4721"}))
    _write_manifest(root, "text_v1", {"kind": "inline prompts + mstar docs (models, adding_models, serving, "
                                      "installation .rst at build time) for long context"}, items,
                    notes="28 short prompts (10 English, 10 multilingual, 4 structured, 4 edge cases) and 4 "
                          "long-context prompts (~1.5k/4k/8k/16k tokens) with a needle. Greedy, 128 new tokens.")


# ---------------------------------------------------------------------------
# ASR: 100 LibriSpeech test-clean utterances
# ---------------------------------------------------------------------------

LIBRISPEECH_PARQUET = "openslr__librispeech_asr/clean/test/0000.parquet"


def _decode_audio(data: bytes):
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def _librispeech_rows(raw: Path, start: int, count: int) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(raw / LIBRISPEECH_PARQUET, columns=["audio", "text", "id", "speaker_id"])
    rows = table.slice(start, count).to_pylist()
    if len(rows) < count:
        raise RuntimeError(f"LibriSpeech parquet has fewer than {start + count} rows")
    return rows


def build_asr_librispeech_v1(root: Path, raw: Path) -> None:
    from benchmark.correctness.metrics import write_wav

    (root / "audio").mkdir(parents=True, exist_ok=True)
    items = []
    for i, row in enumerate(_librispeech_rows(raw, 0, 100)):
        audio, sr = _decode_audio(row["audio"]["bytes"])
        rel = f"audio/{i:03d}.wav"
        write_wav(root / rel, audio, sr)
        items.append(Item(id=f"{i:03d}", files={"audio": rel},
                          params={"sample_rate": sr, "duration_s": len(audio) / sr},
                          expected={"transcript": row["text"], "utterance_id": row["id"]}))
    _write_manifest(root, "asr_librispeech_v1",
                    {"hf_dataset": "openslr/librispeech_asr", "file": "clean/test/0000.parquet", "rows": "0-99"},
                    items, notes="First 100 rows of LibriSpeech test-clean (parquet order), 16 kHz mono int16 WAV.")


# ---------------------------------------------------------------------------
# TTS: 40 read-aloud sentences from Seed-TTS eval (en), with their prompt utterances
# ---------------------------------------------------------------------------

SEEDTTS_DIR = "seedtts/seedtts_testset/en"


def build_tts_seedtts_v1(root: Path, raw: Path) -> None:
    import shutil

    src = raw / SEEDTTS_DIR
    rows = []
    for line in (src / "meta.lst").read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 4 and parts[3].strip():
            rows.append(parts)
    # Skip very short sentences, WER on 3 words is all or nothing.
    rows = [r for r in rows if 8 <= len(r[3].split()) <= 40]
    picked = random.Random(0).sample(rows, 40)
    (root / "prompt_wavs").mkdir(parents=True, exist_ok=True)
    items = []
    for i, (utt, prompt_text, prompt_wav, target) in enumerate(picked):
        rel = f"prompt_wavs/{i:03d}.wav"
        shutil.copyfile(src / prompt_wav, root / rel)
        items.append(Item(id=f"{i:03d}", prompt=target.strip(), files={"prompt_audio": rel},
                          params={"prompt_text": prompt_text.strip()}, expected={"utterance_id": utt}))
    _write_manifest(root, "tts_seedtts_v1",
                    {"dataset": "BytedanceSpeech seed-tts-eval seedtts_testset/en/meta.lst",
                     "selection": "random.Random(0).sample(40) of rows with 8-40 target words"},
                    items, notes="Target sentence is the prompt; prompt_audio/prompt_text are the Seed-TTS "
                                 "speaker prompt, used only by voice-cloning models.")


# ---------------------------------------------------------------------------
# omni: 20 image+question, 20 audio+question, 10 short videos
# ---------------------------------------------------------------------------

IMAGE_QUESTIONS = ["What food is shown in this image?", "Describe this image in one sentence.",
                   "What colors are most prominent in this picture?", "Is this dish sweet or savory? Why?"]
AUDIO_QUESTIONS = ["Transcribe this audio exactly.", "What is the speaker talking about?",
                   "Summarize what you hear in one sentence.", "Is the speaker male or female?"]
VIDEO_QUESTIONS = ["Describe what happens in this video.", "What objects does the robot arm interact with?"]


def build_omni_v1(root: Path, raw: Path) -> None:
    import pyarrow.parquet as pq

    from benchmark.correctness.metrics import write_wav

    for sub in ("image", "audio", "video"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    items: list[Item] = []
    food = pq.read_table(raw / "ethz__food101/data/validation-00000-of-00003.parquet").to_pylist()
    rng = random.Random(0)
    for i, row in enumerate(rng.sample(food, 20)):
        rel = f"image/{i:02d}.jpg"
        (root / rel).write_bytes(row["image"]["bytes"])
        items.append(Item(id=f"img_{i:02d}", prompt=IMAGE_QUESTIONS[i % len(IMAGE_QUESTIONS)],
                          files={"image": rel}, params={"modality": "image"}, expected={"label": row["label"]}))
    for i, row in enumerate(_librispeech_rows(raw, 100, 20)):
        audio, sr = _decode_audio(row["audio"]["bytes"])
        rel = f"audio/{i:02d}.wav"
        write_wav(root / rel, audio, sr)
        items.append(Item(id=f"aud_{i:02d}", prompt=AUDIO_QUESTIONS[i % len(AUDIO_QUESTIONS)],
                          files={"audio": rel}, params={"modality": "audio"},
                          expected={"transcript": row["text"]}))
    for i, ep in enumerate(range(10, 20)):
        rel = f"video/{i:02d}.mp4"
        _droid_clip_mp4(raw, ep, root / rel, seconds=4.0, fps=8)
        items.append(Item(id=f"vid_{i:02d}", prompt=VIDEO_QUESTIONS[i % len(VIDEO_QUESTIONS)],
                          files={"video": rel}, params={"modality": "video", "droid_episode": ep}))
    _write_manifest(root, "omni_v1",
                    {"images": "ethz/food101 validation-00000-of-00003.parquet, random.Random(0).sample(20)",
                     "audio": "LibriSpeech test-clean rows 100-119", "video": "lerobot/droid_100 episodes 10-19, "
                     "exterior camera 1, first 4 s at 8 fps, re-encoded H.264"},
                    items)


# ---------------------------------------------------------------------------
# image generation: 20 prompts x 2 sizes, seed = prompt index
# ---------------------------------------------------------------------------

IMAGE_PROMPTS = [
    "A red fox sitting in fresh snow at sunrise, photorealistic",
    "A bowl of ramen with a soft-boiled egg, top-down food photography",
    "An astronaut riding a horse on the moon, digital art",
    "A cozy reading nook with a window seat and rain outside",
    "A vintage bicycle leaning against a yellow wall in Lisbon",
    "Macro photograph of a dew-covered spider web at dawn",
    "A futuristic city skyline at night with flying cars, neon lights",
    "A watercolor painting of a lighthouse on a rocky coast",
    "A golden retriever puppy playing with a ball in a garden",
    "A steaming cup of coffee on a wooden table, morning light",
    "An ancient library with towering bookshelves and ladders",
    "A sailboat on a calm lake reflecting mountains, oil painting",
    "A street market in Marrakech with colorful spices",
    "A minimalist logo of a mountain inside a circle, flat design",
    "A sign on a shop door that says OPEN 24 HOURS",
    "A close-up portrait of an elderly man with a white beard, studio lighting",
    "A field of sunflowers under a blue sky with fluffy clouds",
    "A cyberpunk samurai standing in the rain, cinematic",
    "A plate of macarons in pastel colors arranged in a spiral",
    "A snowy mountain cabin with smoke rising from the chimney at dusk",
]
IMAGE_SIZES = [(1024, 1024), (768, 1344)]


def build_image_v1(root: Path) -> None:
    items = [Item(id=f"{i:02d}_{h}x{w}", prompt=p, params={"seed": i, "height": h, "width": w})
             for (h, w) in IMAGE_SIZES for i, p in enumerate(IMAGE_PROMPTS)]
    _write_manifest(root, "image_v1", {"kind": "inline prompts"}, items,
                    notes="20 prompts x (1024x1024, 768x1344 HxW); seed = prompt index. Steps / guidance are "
                          "the model's defaults unless the family adapter pins them.")


# ---------------------------------------------------------------------------
# video generation: 5 prompts (+ a conditioning image for image-to-video)
# ---------------------------------------------------------------------------

VIDEO_PROMPTS = [
    "A cat walks across a sunlit kitchen floor and jumps onto a chair",
    "Waves crash against black volcanic rocks, slow motion, overcast sky",
    "A red car drives along a winding mountain road, aerial drone shot",
    "Steam rises from a cup of tea as a hand stirs it with a spoon",
    "A robot arm picks up a green apple from a table and places it in a bowl",
]


def build_video_v1(root: Path, raw: Path) -> None:
    (root / "image").mkdir(parents=True, exist_ok=True)
    items = []
    for i, p in enumerate(VIDEO_PROMPTS):
        rel = f"image/{i:02d}.png"
        _droid_frame_png(raw, 30 + i, root / rel)
        items.append(Item(id=f"{i:02d}", prompt=p, files={"image": rel}, params={"seed": i}))
    _write_manifest(root, "video_v1", {"prompts": "inline", "i2v images": "lerobot/droid_100 episodes 30-34, "
                                                                          "exterior camera 1, first frame"},
                    items, notes="Resolution / frames / steps are pinned per family adapter.")


# ---------------------------------------------------------------------------
# robotics: 10 DROID episodes (lerobot/droid_100)
# ---------------------------------------------------------------------------

DROID_DIR = "lerobot__droid_100"
DROID_CAMERAS = ["observation.images.exterior_image_1_left", "observation.images.exterior_image_2_left",
                 "observation.images.wrist_image_left"]


def _droid_tables(raw: Path):
    import pyarrow.parquet as pq

    data = pq.read_table(raw / DROID_DIR / "data/chunk-000/file-000.parquet").to_pandas()
    episodes = pq.read_table(raw / DROID_DIR / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    tasks = pq.read_table(raw / DROID_DIR / "meta/tasks.parquet").to_pandas()
    info = json.loads((raw / DROID_DIR / "meta/info.json").read_text())
    return data, episodes, tasks, info


def _droid_video_frames(raw: Path, camera: str, t0: float, t1: float, fps_out: float | None = None):
    """RGB frames [N, H, W, 3] of one camera between file timestamps t0 and t1, in seconds."""
    import av
    import numpy as np

    path = raw / DROID_DIR / "videos" / camera / "chunk-000/file-000.mp4"
    frames, times = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(t0 - 1.0, 0) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < t0 - 1e-3:
                continue
            if t > t1 + 1e-3:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
            times.append(t)
    if not frames:
        raise RuntimeError(f"no frames in {path} between {t0} and {t1}")
    arr = np.stack(frames)
    if fps_out:
        src_fps = (len(times) - 1) / max(times[-1] - times[0], 1e-6) if len(times) > 1 else fps_out
        step = max(1, round(src_fps / fps_out))
        arr = arr[::step]
    return arr


def _droid_episode_window(raw: Path, ep: int, camera: str) -> tuple[float, float]:
    _, episodes, _, _ = _droid_tables(raw)
    row = episodes[episodes["episode_index"] == ep].iloc[0]
    return float(row[f"videos/{camera}/from_timestamp"]), float(row[f"videos/{camera}/to_timestamp"])


def _write_mp4(frames, path: Path, fps: int) -> None:
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames.shape[1] - frames.shape[1] % 2, frames.shape[2] - frames.shape[2] % 2
        stream.pix_fmt = "yuv420p"
        for f in frames:
            frame = av.VideoFrame.from_ndarray(f[: stream.height, : stream.width], format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _droid_clip_mp4(raw: Path, ep: int, path: Path, seconds: float, fps: int) -> None:
    cam = DROID_CAMERAS[0]
    t0, t1 = _droid_episode_window(raw, ep, cam)
    frames = _droid_video_frames(raw, cam, t0, min(t1, t0 + seconds), fps_out=fps)
    _write_mp4(frames, path, fps)


def _droid_frame_png(raw: Path, ep: int, path: Path) -> None:
    from PIL import Image

    cam = DROID_CAMERAS[0]
    t0, _ = _droid_episode_window(raw, ep, cam)
    Image.fromarray(_droid_video_frames(raw, cam, t0, t0 + 0.2)[0]).save(path)


def build_droid_v1(root: Path, raw: Path) -> None:
    import numpy as np
    from PIL import Image

    data, episodes, tasks, info = _droid_tables(raw)
    fps = float(info.get("fps", 15))
    (root / "frames").mkdir(parents=True, exist_ok=True)
    (root / "arrays").mkdir(parents=True, exist_ok=True)
    items = []
    for ep in range(10):
        rows = data[data["episode_index"] == ep].sort_values("frame_index")
        k = len(rows) // 3  # a step a third into the episode: the arm is moving
        state = np.stack(rows["observation.state"].to_numpy())
        action = np.stack(rows["action"].to_numpy())
        task_idx = int(rows["task_index"].iloc[k])
        task = tasks.index[tasks["task_index"] == task_idx][0] if "task_index" in tasks else str(task_idx)
        files = {}
        for cam in DROID_CAMERAS:
            t0, _ = _droid_episode_window(raw, ep, cam)
            t = t0 + k / fps
            frame = _droid_video_frames(raw, cam, t, t + 1.5 / fps)[0]
            rel = f"frames/ep{ep:02d}_{cam.split('.')[-1]}.png"
            Image.fromarray(frame).save(root / rel)
            files[cam.split(".")[-1]] = rel
        # 8-frame exterior clip ending at step k, for V-JEPA2-AC.
        cam = DROID_CAMERAS[0]
        t0, _ = _droid_episode_window(raw, ep, cam)
        clip = _droid_video_frames(raw, cam, t0 + max(k - 7, 0) / fps, t0 + k / fps + 0.5 / fps)[:8]
        np.save(root / f"arrays/ep{ep:02d}_clip.npy", clip)
        _write_mp4(clip, root / f"arrays/ep{ep:02d}_clip.mp4", int(fps))
        np.save(root / f"arrays/ep{ep:02d}_state.npy", state)
        np.save(root / f"arrays/ep{ep:02d}_action.npy", action)
        files.update(clip=f"arrays/ep{ep:02d}_clip.npy", clip_mp4=f"arrays/ep{ep:02d}_clip.mp4",
                     states=f"arrays/ep{ep:02d}_state.npy",
                     actions=f"arrays/ep{ep:02d}_action.npy")
        items.append(Item(id=f"ep{ep:02d}", prompt=str(task), files=files,
                          params={"episode": ep, "step": k, "fps": fps, "seed": ep}))
    _write_manifest(root, "droid_v1", {"hf_dataset": "lerobot/droid_100", "episodes": "0-9",
                                       "step": "len(episode) // 3"}, items,
                    notes="Per episode: the three camera frames at the step, the task string, the full state and "
                          "action trajectories (rows indexed by frame), and an 8-frame exterior clip ending there.")


def build_waypoint_v1(root: Path, raw: Path) -> None:
    from PIL import Image

    (root / "seeds").mkdir(parents=True, exist_ok=True)
    items = []
    for i, ep in enumerate((40, 41, 42)):
        cam = DROID_CAMERAS[0]
        t0, _ = _droid_episode_window(raw, ep, cam)
        img = Image.fromarray(_droid_video_frames(raw, cam, t0, t0 + 0.2)[0])  # 320x180, 16:9
        files = {}
        for key, size in (("seed_720p", (1280, 720)), ("seed_360p", (640, 360))):
            rel = f"seeds/{i}_{size[1]}p.png"
            img.resize(size, Image.BICUBIC).save(root / rel)
            files[key] = rel
        items.append(Item(id=f"{i}", files=files, params={"seed": 1000 + i, "droid_episode": ep}))
    _write_manifest(root, "waypoint_v1", {"seeds": "lerobot/droid_100 episodes 40-42, exterior camera 1, first "
                                                   "frame, bicubic to 1280x720 and 640x360"}, items,
                    notes="Seed images only; the action stream and step count are fixed in the family adapter.")


BUILDERS = {
    "text_v1": lambda root, raw: build_text_v1(root),
    "asr_librispeech_v1": build_asr_librispeech_v1,
    "tts_seedtts_v1": build_tts_seedtts_v1,
    "omni_v1": build_omni_v1,
    "image_v1": lambda root, raw: build_image_v1(root),
    "video_v1": build_video_v1,
    "droid_v1": build_droid_v1,
    "waypoint_v1": build_waypoint_v1,
}


def build(name: str, data_root: Path = DEFAULT_DATA_ROOT, force: bool = False) -> Path:
    root = Path(data_root) / "sets" / name
    if (root / "manifest.json").exists() and not force:
        print(f"{name}: already built at {root} (sets are immutable; make a new version instead)")
        return root
    root.mkdir(parents=True, exist_ok=True)
    BUILDERS[name](root, Path(data_root) / "raw")
    n = len(json.loads((root / "manifest.json").read_text())["items"])
    print(f"{name}: {n} items at {root}")
    return root


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("names", nargs="+", help=f"set names or 'all': {', '.join(BUILDERS)}")
    b.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    b.add_argument("--force", action="store_true", help="rebuild in place (only before any reference exists)")
    ls = sub.add_parser("list")
    ls.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = ap.parse_args(argv)
    if args.cmd == "list":
        for name in BUILDERS:
            try:
                s = load_set(name, args.data_root)
                print(f"{name:22s} {len(s.items):4d} items  built {s.manifest['created']}")
            except FileNotFoundError:
                print(f"{name:22s} (not built)")
        return
    names = list(BUILDERS) if args.names == ["all"] else args.names
    failed = []
    for name in names:
        try:
            build(name, args.data_root, args.force)
        except Exception as e:  # keep building the others; report at the end
            failed.append(name)
            print(f"{name}: FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    if failed:
        sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()
