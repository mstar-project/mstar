import glob
import json
import os
import wave
from abc import ABC, abstractmethod

import numpy as np
import requests

from benchmark.base import RequestType
from benchmark.request import RequestInput


class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> RequestInput:
        pass

    def get_requests(self) -> list[RequestInput]:
        return [
            self[i] for i in range(len(self))
        ]

    @property
    @abstractmethod
    def num_requests(self) -> int:
        pass

    def _resize_data(self, data: list[RequestInput]) -> list[RequestInput]:
        """Resize data to match num_prompts."""
        if not self.num_requests or not data:
            return data

        if len(data) < self.num_requests:
            factor = (self.num_requests // len(data)) + 1
            data = data * factor

        return data[: self.num_requests]


class TxtFileDataset(BaseDataset):
    """
    Dataset loader for text-to-text prompts, coming from a provided text file
    with one line per prompt
    """

    def __init__(
        self,
        filename: str,
        num_requests: int,
        req_type=RequestType.T2T
    ):
        assert req_type.get_input_modalities() == "text"

        self.items = []
        self._num_requests = num_requests
        with open(filename, "r") as f:
            for line in f.readlines():
                self.items.append(RequestInput(
                    req_type=req_type,
                    prompt=line.strip()
                ))
        self.items = self._resize_data(self.items)

    @property
    def num_requests(self):
        return self._num_requests

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestInput:
        return self.items[idx]


VBENCH_UND_PROMPT = "Describe this image in detail."


class VBenchDataset(BaseDataset):
    """
    Dataset loader for VBench prompts.
    Supports t2v, i2v.
    """

    T2V_PROMPT_URL = (
        "https://raw.githubusercontent.com/Vchitect/VBench/master/prompts/prompts_per_dimension/subject_consistency.txt"
    )
    I2V_DOWNLOAD_SCRIPT_URL = (
        "https://raw.githubusercontent.com/Vchitect/VBench/master/vbench2_beta_i2v/download_data.sh"
    )

    def __init__(
        self,
        cache_dir: str,
        task: RequestType,
        num_requests: int,
    ):
        self.cache_dir = cache_dir
        self.task = task
        self._num_requests = num_requests
        self.items = self._load_data()
        self.items = self._resize_data(self.items)

    @property
    def num_requests(self):
        return self._num_requests

    def _load_data(self) -> list[RequestInput]:
        if self.task == RequestType.T2I:
            return self._load_t2v_prompts()
        elif self.task.get_input_modalities() == "image":
            return self._load_i2v_data()
        else:
            raise NotImplementedError(
                f"Vbench does not support request type {self.task}"
            )

    def _download_file(self, url: str, dest_path: str) -> None:
        """Download a file from URL to destination path."""
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        resp = requests.get(url)
        resp.raise_for_status()
        with open(dest_path, "w") as f:
            f.write(resp.text)

    def _load_t2v_prompts(self) -> list[RequestInput]:
        path = os.path.join(self.cache_dir, "vbench_subject_consistency.txt")
        if not os.path.exists(path):
            print(f"Downloading VBench T2V prompts to {path}...")
            self._download_file(self.T2V_PROMPT_URL, path)

        reqs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    reqs.append(RequestInput(
                        req_type=self.task,
                        prompt=line
                    ))
        return reqs

    def _auto_download_i2v_dataset(self) -> str:
        """Auto-download VBench I2V dataset and return the dataset directory."""
        vbench_i2v_dir = os.path.join(self.cache_dir, "vbench_i2v", "vbench2_beta_i2v")
        info_json_path = os.path.join(vbench_i2v_dir, "data", "i2v-bench-info.json")

        if os.path.exists(info_json_path):
            return vbench_i2v_dir

        print(f"Downloading VBench I2V dataset to {vbench_i2v_dir}...")
        try:
            cache_root = os.path.join(self.cache_dir, "vbench_i2v")
            script_path = os.path.join(cache_root, "download_data.sh")

            self._download_file(self.I2V_DOWNLOAD_SCRIPT_URL, script_path)
            os.chmod(script_path, 0o755)

            print("Executing download_data.sh (this may take a while)...")
            import subprocess

            result = subprocess.run(
                ["bash", script_path],
                cwd=cache_root,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Download script failed: {result.stderr}")

            print(f"Successfully downloaded VBench I2V dataset to {vbench_i2v_dir}")
        except Exception as e:
            print(f"Failed to download VBench I2V dataset: {e}")
            print("Please manually download following instructions at:")
            print("https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v#22-download")
            return None

        return vbench_i2v_dir if os.path.exists(info_json_path) else None

    def _load_i2v_data(self) -> list[RequestInput]:
        """Load I2V data from VBench I2V dataset."""
        path = self._auto_download_i2v_dataset()
        if not path:
            raise Exception(
                "Failed to load I2V Data for VBench. Note that you need to pip install gdown to load the data."
            )

        # Try to load from i2v-bench-info.json
        info_json_path = os.path.join(path, "data", "i2v-bench-info.json")
        if os.path.exists(info_json_path):
            try:
                return self._load_from_i2v_json(info_json_path)
            except Exception as e:
                print(f"Failed to load {info_json_path}: {e}")

        # Fallback: scan directory for images
        if os.path.isdir(path):
            data = self._scan_directory_for_images(path)
            if data:
                return data

        raise Exception("Failed to load I2V Datafor VBench")

    def _load_from_i2v_json(self, json_path: str) -> list[RequestInput]:
        with open(json_path) as f:
            items = json.load(f)

        base_dir = os.path.dirname(os.path.dirname(json_path))  # up to vbench2_beta_i2v
        origin_dir = os.path.join(base_dir, "data", "origin")

        reqs = []
        for item in items:
            img_path = os.path.join(origin_dir, item.get("file_name", ""))
            if os.path.exists(img_path):
                reqs.append(RequestInput(
                    req_type=self.task,
                    prompt=item.get("caption", "") \
                        if self.task == RequestType.I2I \
                            else VBENCH_UND_PROMPT,
                    image_path=img_path,
                ))
            else:
                print(f"Warning: Image not found: {img_path}")

        print(f"Loaded {len(reqs)} I2V samples from VBench I2V dataset")
        return reqs

    def _scan_directory_for_images(self, path: str) -> list[RequestInput]:
        exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(path, ext)))
            files.extend(glob.glob(os.path.join(path, ext.upper())))
            origin_dir = os.path.join(path, "data", "origin")
            if os.path.exists(origin_dir):
                files.extend(glob.glob(os.path.join(origin_dir, ext)))
                files.extend(glob.glob(os.path.join(origin_dir, ext.upper())))

        return [
            RequestInput(
                req_type=self.task,
                prompt=os.path.splitext(os.path.basename(f))[0] \
                    if self.task == RequestType.I2I \
                        else VBENCH_UND_PROMPT,
                image_path=f,
            )
            for f in files
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestInput:
        return self.items[idx]


# ---------------------------------------------------------------------------
# Audio – openslr/librispeech_asr
# ---------------------------------------------------------------------------

LIBRISPEECH_AUDIO_PROMPTS = [
    "Transcribe the speech in this audio clip.",
    "What is being said in this audio recording?",
    "Please provide a transcription of the spoken content.",
    "Listen to the audio and write down what you hear.",
    "Convert the spoken words in this audio to text.",
]


class LibriSpeechDataset(BaseDataset):
    """
    Dataset loader for openslr/librispeech_asr.
    Uses the validation split; default request type is A2T.
    Audio files are written to a temp directory and paths are passed as audio_path.
    """

    DEFAULT_PROMPT = LIBRISPEECH_AUDIO_PROMPTS[0]

    def __init__(
        self,
        local_file_dir: str,
        num_requests: int = 100,
        req_type: RequestType = RequestType.A2T,
        prompt: str = DEFAULT_PROMPT,
        split: str = "validation",
        cache_dir: str | None = None,
    ):
        assert req_type.get_input_modalities() == "audio", (
            f"LibriSpeechDataset requires an audio input RequestType, got {req_type}"
        )

        from datasets import load_dataset
        from torchcodec.decoders import AudioDecoder

        os.makedirs(local_file_dir, exist_ok=True)

        self._num_requests = num_requests
        self.prompt = prompt
        self.local_file_dir = local_file_dir

        raw = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        # Take first 100 rows before building items to avoid loading the whole dataset
        raw = raw.select(range(min(100, len(raw))))

        self.items: list[RequestInput] = []

        for i, row in enumerate(raw):
            dec: AudioDecoder = row["audio"]

            # Decode all frames → shape (num_channels, num_samples), float32
            frames = dec.get_all_samples()
            audio_data = frames.data  # torch.Tensor
            sample_rate = frames.sample_rate

            # Convert to int16 PCM for WAV
            audio_np = (audio_data.numpy() * 32767).clip(-32768, 32767).astype(np.int16)

            # WAV expects interleaved (num_samples, num_channels), then flatten
            audio_interleaved = audio_np.T.flatten()

            # Write to disk
            audio_path = os.path.join(local_file_dir, f"librispeech_{i:05d}.wav")
            with wave.open(audio_path, "wb") as wf:
                wf.setnchannels(audio_np.shape[0])       # num channels
                wf.setsampwidth(2)                        # 2 bytes = int16
                wf.setframerate(sample_rate)
                wf.writeframes(audio_interleaved.tobytes())

            self.items.append(RequestInput(
                req_type=req_type,
                prompt=self.prompt,
                audio_path=audio_path,
            ))

        self.items = self._resize_data(self.items)

    @property
    def num_requests(self) -> int:
        return self._num_requests

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestInput:
        return self.items[idx]


# ---------------------------------------------------------------------------
# Image – ethz/food101
# ---------------------------------------------------------------------------

FOOD101_IMAGE_PROMPTS = [
    "What food dish is shown in this image?",
    "Describe the food item pictured here.",
    "Identify the cuisine and dish visible in this photo.",
    "What is the name of this food?",
    "Generate a caption for this food image.",
]

FOOD101_IMAGE_GEN_PROMPTS = [
    "Generate a photorealistic image of a gourmet version of this dish.",
    "Create a stylized illustration inspired by the food shown.",
    "Produce a top-down flat-lay photo of the ingredients for this dish.",
]


class Food101Dataset(BaseDataset):
    """
    Dataset loader for ethz/food101.
    Supports both image understanding (I2T) and image generation (T2I / I2I).
    For T2I the prompt is derived from the class label; for I2T / I2I the raw
    image is passed via image_path.
    """


    def __init__(
        self,
        num_requests: int = 100,
        req_type: RequestType = RequestType.I2T,
        prompts: list[str] = FOOD101_IMAGE_PROMPTS,
        split: str = "validation",
        cache_dir: str | None = None,
    ):
        valid_types = {RequestType.I2T, RequestType.I2I, RequestType.T2I, RequestType.I2S}
        assert req_type in valid_types, (
            f"Food101Dataset supports {valid_types}, got {req_type}"
        )
        from datasets import load_dataset

        self._num_requests = num_requests
        self.prompts = prompts

        raw = load_dataset(
            "ethz/food101",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        # Shuffle with a fixed seed so we get class-diverse images while
        # remaining deterministic across runs (so the same N images appear in
        # the same order on every benchmark invocation).
        raw = raw.shuffle(seed=42)
        raw = raw.select(range(min(100, len(raw))))

        # Build label lookup (int -> class name string)
        label_names: list[str] = raw.features["label"].names

        self.items: list[RequestInput] = []
        for row in raw:
            image = row["image"]
            label: str = label_names[row["label"]]

            if req_type == RequestType.T2I:
                # Text-to-image: prompt is based on the class label, no image input
                item_prompt = f"Generate a photorealistic image of {label.replace('_', ' ')}."
                self.items.append(
                    RequestInput(req_type=req_type, prompt=item_prompt)
                )
            else:
                # Save image to a temp file so downstream code has a stable path
                import os
                import tempfile
                tmp_dir = tempfile.mkdtemp(prefix="food101_")
                img_path = os.path.join(tmp_dir, f"{label}_{len(self.items)}.jpg")
                image.save(img_path)
                self.items.append(
                    RequestInput(
                        req_type=req_type,
                        prompt=self.prompts[len(self.items) % len(self.prompts)],
                        image_path=img_path,
                    )
                )

        self.items = self._resize_data(self.items)

    @property
    def num_requests(self) -> int:
        return self._num_requests

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestInput:
        return self.items[idx]


# ---------------------------------------------------------------------------
# Video – sayakpaul/ucf101-subset
# ---------------------------------------------------------------------------

UCF101_VIDEO_PROMPTS = [
    "Describe the action or activity happening in this video.",
    "What sport or physical activity is being performed in this clip?",
    "Provide a detailed description of the events in this video.",
    "What is the person doing in this video?",
    "Summarize the content of this video clip.",
]


class UCF101Dataset(BaseDataset):
    """
    Dataset loader for sayakpaul/ucf101-subset.
    Default request type is V2T. Each row contains a video file; the path is
    extracted from the HuggingFace cache and passed as video_path.
    """

    DEFAULT_PROMPT = UCF101_VIDEO_PROMPTS[0]

    def __init__(
        self,
        local_file_dir: str,
        num_requests: int = 100,
        req_type: RequestType = RequestType.V2T,
        prompt: str = DEFAULT_PROMPT,
        split: str = "train",
        cache_dir: str | None = None,
    ):
        assert req_type.get_input_modalities() == "video", (
            f"UCF101Dataset requires a video input RequestType, got {req_type}"
        )
        import torch
        from datasets import load_dataset
        from torchcodec.decoders import VideoDecoder
        from torchcodec.encoders import VideoEncoder

        self._num_requests = num_requests
        self.prompt = prompt
        self.local_file_dir = local_file_dir

        raw = load_dataset(
            "sayakpaul/ucf101-subset",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        raw = raw.select(range(min(100, len(raw))))

        self.items: list[RequestInput] = []
        for i, row in enumerate(raw):
            dec: VideoDecoder = row["video"]
            fps = dec.metadata.average_fps
            dec = iter(dec)

            frames = []

            while True:
                try:
                    frame = next(dec)
                    frames.append(frame)
                except RuntimeError as e:
                    print("[WARNING]", e)
                    break
                except StopIteration:
                    break
            assert len(frames) > 0
            frames = torch.stack(frames)

            video_path = os.path.join(self.local_file_dir, f"ucf101_{i:05d}.mp4")

            encoder = VideoEncoder(frames=frames, frame_rate=fps)
            encoder.to_file(video_path)

            self.items.append(RequestInput(
                req_type=req_type,
                prompt=self.prompt,
                video_path=video_path,
            ))

        self.items = self._resize_data(self.items)

    @property
    def num_requests(self) -> int:
        return self._num_requests

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestInput:
        return self.items[idx]


# ---------------------------------------------------------------------------
# Robotics – lerobot/droid_100 (DROID)
#
# lerobot v2 format: images are NOT parquet columns.  Each episode's frames
# for each camera are stored as a separate mp4 at:
#   videos/{video_key}/episode_{episode_id:06d}.mp4
# Camera keys and their dtypes are declared in meta/info.json.
# Language instructions live in meta/tasks.jsonl, indexed by task_index.
# The parquet only carries: action, observation.state, task_index,
# episode_index, frame_index, timestamp.
# ---------------------------------------------------------------------------


def _to_float_list(raw, target_dim: int) -> list[float]:
    """Pad or truncate raw state/action data to exactly target_dim floats."""
    if raw is None:
        return [0.0] * target_dim
    try:
        vals = raw.tolist() if hasattr(raw, "tolist") else [float(v) for v in raw]
    except Exception:
        return [0.0] * target_dim
    if len(vals) >= target_dim:
        return [float(v) for v in vals[:target_dim]]
    return [float(v) for v in vals] + [0.0] * (target_dim - len(vals))


def _decode_frames_to_png_and_video(
    video_path: str,
    frame_indices: list[int],
    png_path: str | None,
    mp4_path: str | None,
    fps: float = 15.0,
) -> None:
    """Decode specific frames from a chunk video.

    If ``png_path`` is given, saves the FIRST decoded frame as a PNG.
    If ``mp4_path`` is given, saves ALL decoded frames as a new mp4.
    ``frame_indices`` are local positions within the chunk file (0-based).
    """
    from PIL import Image as PILImage
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(video_path)
    batch = dec.get_frames_at(indices=frame_indices)
    tensors = batch.data  # [T, C, H, W] uint8

    if png_path is not None:
        arr = tensors[0].permute(1, 2, 0).numpy()  # [H, W, C]
        PILImage.fromarray(arr).save(png_path)

    if mp4_path is not None:
        from torchcodec.encoders import VideoEncoder
        # VideoEncoder expects [T, C, H, W] — same layout as get_frames_at output.
        VideoEncoder(frames=tensors, frame_rate=fps).to_file(mp4_path)


class DROIDDataset(BaseDataset):
    """DROID robotics dataset for evaluating pi0.5 and V-JEPA 2-AC.

    Uses ``lerobot/droid_100`` — a 100-episode subset (~4 GB) of the DROID
    robot-manipulation dataset.  The dataset is in lerobot v2 format: images
    are stored as per-episode mp4 files (one per camera), not as parquet
    columns.  ``hf_hub_download`` is used to fetch individual episode videos
    on demand; they are cached by HuggingFace and not re-downloaded.

    For vjepa2_ac the downloaded mp4 is passed directly to the server — no
    re-encoding.  For pi05 only the first frame of each camera video is
    decoded to a PNG.

    Args:
        local_file_dir:  directory for extracted PNGs (created if absent).
                         mp4s are kept in the HF cache, not copied here.
        num_requests:    number of episodes to use.
        task:            ``"pi05"`` or ``"vjepa2_ac"``.
        rollout_horizon: H for vjepa2_ac; actions/states shape is [H, dim].
        action_dim:      target dim; defaults to 32 (pi05) or 7 (vjepa2_ac).
        cache_dir:       HuggingFace cache root (None → HF default).
    """

    HF_REPO = "lerobot/droid_100"

    def __init__(
        self,
        local_file_dir: str,
        num_requests: int = 10,
        task: str = "pi05",
        rollout_horizon: int = 4,
        action_dim: int | None = None,
        cache_dir: str | None = None,
        num_video_frames: int = 8,
    ):
        assert task in ("pi05", "vjepa2_ac"), \
            f"task must be 'pi05' or 'vjepa2_ac', got {task!r}"
        import json as _json

        from datasets import load_dataset
        from huggingface_hub import hf_hub_download

        os.makedirs(local_file_dir, exist_ok=True)
        self.local_file_dir  = local_file_dir
        self.task            = task
        self._num_requests   = num_requests
        self.action_dim      = action_dim if action_dim is not None else (32 if task == "pi05" else 7)
        self.rollout_horizon = rollout_horizon
        self.cache_dir       = cache_dir
        # Number of video frames to extract per vjepa2_ac request. Default 8
        # matches upstream's published DROID training config
        # (configs/train/vitg16/droid-256px-8f.yaml: dataset_fpcs: 8).
        # Server-side encoder uses self-tubelet replication on each frame,
        # producing 8 token-frames; only the first is used as rollout context.
        self.num_video_frames = num_video_frames

        def _dl(filename):
            return hf_hub_download(
                self.HF_REPO, filename, repo_type="dataset", cache_dir=cache_dir
            )

        # Read meta/info.json for camera keys and chunks_size.
        # lerobot v2 layout: videos/{key}/chunk-{C:03d}/file-{F:03d}.mp4
        # where C = episode_index // chunks_size, F = episode_index % chunks_size.
        with open(_dl("meta/info.json")) as f:
            info = _json.load(f)
        features_info = info.get("features", {})
        camera_keys = [k for k, v in features_info.items()
                       if isinstance(v, dict) and v.get("dtype") == "video"]
        if not camera_keys:
            camera_keys = info.get("video_keys", [])
        if not camera_keys:
            raise RuntimeError(
                f"No video keys found in {self.HF_REPO}/meta/info.json. "
                f"Top-level keys: {list(info.keys())}"
            )
        chunks_size: int = info.get("chunks_size", info.get("chunk_size", 1000))
        print(f"  camera keys : {camera_keys}")
        print(f"  chunks_size : {chunks_size}")

        # Load language instructions from meta/tasks.parquet (lerobot v2).
        tasks: dict[int, str] = {}
        try:
            import pandas as _pd
            tasks_df = _pd.read_parquet(_dl("meta/tasks.parquet"))
            tasks = dict(zip(tasks_df["task_index"].astype(int),
                             tasks_df["task"].astype(str), strict=False))
            print(f"  tasks loaded: {len(tasks)}")
        except Exception as e:
            print(f"  [warn] could not load tasks.parquet: {e}")

        # Load tabular data (state, action, task_index, episode/frame indices).
        print(f"Loading {self.HF_REPO} parquet...")
        raw = load_dataset(self.HF_REPO, split="train", cache_dir=cache_dir)
        feats = raw.features
        state_col    = next((c for c in ["observation.state", "state"] if c in feats), None)
        action_col   = next((c for c in ["action", "actions"]          if c in feats), None)
        task_idx_col = "task_index"   if "task_index"   in feats else None
        ep_col       = "episode_index" if "episode_index" in feats else None
        frame_col    = "frame_index"   if "frame_index"   in feats else None

        if action_col is None:
            raise RuntimeError(
                f"No action column in {self.HF_REPO}. Available: {list(feats.keys())}"
            )
        print(f"  state col  : {state_col}")
        print(f"  action col : {action_col}")

        # Group rows by episode
        episodes: dict[int, list] = {}
        for row in raw:
            ep_id = int(row[ep_col]) if ep_col else 0
            episodes.setdefault(ep_id, []).append(row)
        if frame_col:
            for frames in episodes.values():
                frames.sort(key=lambda r: int(r[frame_col]))

        ep_ids = sorted(episodes.keys())[:num_requests]
        print(f"  using {len(ep_ids)} of {len(episodes)} episodes")

        self.items: list[RequestInput] = []
        for i, ep_id in enumerate(ep_ids):
            frames = episodes[ep_id]
            language = ""
            if task_idx_col and tasks:
                tidx = int(frames[0].get(task_idx_col, 0))
                language = tasks.get(tidx, "")
            try:
                item = (
                    self._make_pi05(i, ep_id, frames, camera_keys, state_col,
                                    action_col, language, _dl, chunks_size)
                    if task == "pi05"
                    else self._make_vjepa2_ac(i, ep_id, frames, camera_keys,
                                              action_col, state_col, language,
                                              _dl, chunks_size)
                )
            except Exception as exc:
                print(f"  [warn] ep{ep_id}: {exc}")
                item = None
            if item is not None:
                self.items.append(item)

        self.items = self._resize_data(self.items)

    # ------------------------------------------------------------------

    def _chunk_video_path(self, ep_id: int, cam_key: str, chunks_size: int) -> str:
        """lerobot v2: videos/{key}/chunk-{C:03d}/file-000.mp4
        Each chunk file contains chunks_size episodes concatenated."""
        chunk = ep_id // chunks_size
        return f"videos/{cam_key}/chunk-{chunk:03d}/file-000.mp4"

    def _local_frame_indices(self, frames: list) -> list[int]:
        """Convert global `index` values to positions within the chunk file.
        The chunk file starts at min(index) for the first episode in the chunk."""
        global_indices = [int(row["index"]) for row in frames]
        chunk_start = global_indices[0] - int(frames[0].get("frame_index", 0))
        return [g - chunk_start for g in global_indices]

    def _make_pi05(self, idx, ep_id, frames, camera_keys, state_col,
                   action_col, language, download_fn, chunks_size) -> RequestInput:
        # Compute frame positions within the chunk video once
        local_indices = self._local_frame_indices(frames)
        first_local   = local_indices[0]

        image_paths: list[str] = []
        for cam_key in camera_keys[:3]:
            chunk_video = download_fn(self._chunk_video_path(ep_id, cam_key, chunks_size))
            png_path    = os.path.join(self.local_file_dir, f"ep{ep_id}_cam{len(image_paths)}.png")
            _decode_frames_to_png_and_video(
                chunk_video, [first_local], png_path=png_path, mp4_path=None
            )
            image_paths.append(png_path)

        if not image_paths:
            raise ValueError("no camera videos found")
        while len(image_paths) < 3:
            image_paths.append(image_paths[0])

        state = _to_float_list(
            frames[0].get(state_col) if state_col else None, self.action_dim
        )
        return RequestInput(
            req_type=RequestType.VLA,
            prompt=language or "manipulate the object",
            image_path=image_paths[0],
            extra_image_paths=image_paths[1:],
            model_kwargs={"robot_state": state},
        )

    def _make_vjepa2_ac(self, idx, ep_id, frames, camera_keys, action_col,
                        state_col, language, download_fn, chunks_size) -> RequestInput:
        # The model requires T_total >= t_ctx + rollout_horizon - 1 actions.
        # t_ctx = grid_depth = frames_per_clip // tubelet_size = 64 // 2 = 32.
        t_ctx     = 32
        n_actions = t_ctx + self.rollout_horizon - 1
        if len(frames) < n_actions:
            raise ValueError(
                f"episode has {len(frames)} frames, need >= {n_actions} "
                f"(t_ctx={t_ctx} + rollout_horizon={self.rollout_horizon} - 1)"
            )

        # Extract the episode's own frames from the chunk video so the server
        # subsamples from the right clip instead of the whole dataset.
        local_indices = self._local_frame_indices(frames)
        # Sample N evenly-spaced frames from the episode for the video input.
        # N defaults to 8, matching upstream's published DROID training config
        # (configs/train/vitg16/droid-256px-8f.yaml: dataset_fpcs: 8).
        n_video = min(self.num_video_frames, len(local_indices))
        step    = max(1, len(local_indices) // n_video)
        video_local_indices = local_indices[::step][:n_video]

        chunk_video = download_fn(
            self._chunk_video_path(ep_id, camera_keys[0], chunks_size)
        )
        mp4_path = os.path.join(self.local_file_dir, f"ep{ep_id}.mp4")
        _decode_frames_to_png_and_video(
            chunk_video, video_local_indices, png_path=None, mp4_path=mp4_path
        )

        actions = [_to_float_list(f.get(action_col), self.action_dim)
                   for f in frames[:n_actions]]
        states  = [_to_float_list(f.get(state_col) if state_col else None, self.action_dim)
                   for f in frames[:n_actions]]
        return RequestInput(
            req_type=RequestType.V2V,
            prompt=language or "world model rollout",
            video_path=mp4_path,
            model_kwargs={
                "actions":         actions,
                "states":          states,
                "rollout_horizon": self.rollout_horizon,
            },
        )

    @property
    def num_requests(self) -> int:
        return self._num_requests

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestInput:
        return self.items[idx]

    # ------------------------------------------------------------------
class VideoMMEDataset(BaseDataset):
    """
    Dataset loader for Video-MME (https://video-mme.github.io/).

    Auto-downloads from the HuggingFace dataset ``zhaochenyang20/Video_MME`` on
    first use, matching the load-from-HF pattern used by Food101/LibriSpeech/
    UCF101. This particular mirror is used because it ships the videos as
    individual mp4s plus per-chunk metadata jsonl, whereas the canonical
    ``lmms-lab/Video-MME`` distributes a single parquet plus 20 video zip
    archives that would need extraction. Caller may override with a local
    path via ``data_dir``.

    Expected layout (matches the HF repo):
        <root>/data/test_part_*.jsonl
        <root>/videos/<videoID>.mp4

    Each jsonl row is one MCQ over a video. The prompt sent to the server
    is the question with its multiple-choice options appended, asking the
    model to answer with a single letter.
    """

    HF_REPO = "zhaochenyang20/Video_MME"

    def __init__(
        self,
        num_requests: int = 100,
        req_type: RequestType = RequestType.V2T,
        data_dir: str | None = None,
        cache_dir: str | None = None,
        jsonl_glob: str = "data/*.jsonl",
    ):
        assert req_type.get_input_modalities() == "video", (
            f"VideoMMEDataset requires a video input RequestType, got {req_type}"
        )
        self._num_requests = num_requests

        if data_dir is None:
            data_dir = self._auto_download(cache_dir)
        self.data_dir = data_dir

        jsonl_paths = sorted(glob.glob(os.path.join(data_dir, jsonl_glob)))
        if not jsonl_paths:
            raise FileNotFoundError(
                f"No Video-MME jsonl files matched {os.path.join(data_dir, jsonl_glob)}"
            )

        rows: list[dict] = []
        for p in jsonl_paths:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))

        self.items: list[RequestInput] = []
        for row in rows:
            video_path = os.path.join(data_dir, row["video_path"])
            if not os.path.exists(video_path):
                continue
            options = "\n".join(row.get("options", []))
            prompt = (
                f"{row['question']}\n{options}\n"
                "Answer with the letter (A, B, C, or D) of the correct option."
            )
            self.items.append(RequestInput(
                req_type=req_type,
                prompt=prompt,
                video_path=video_path,
            ))

        if not self.items:
            raise RuntimeError(
                f"VideoMMEDataset loaded 0 usable rows from {data_dir} -- "
                f"check that videos/ contains the .mp4 files referenced by the jsonl."
            )

        self.items = self._resize_data(self.items)

    @classmethod
    def _auto_download(cls, cache_dir: str | None) -> str:
        from huggingface_hub import snapshot_download
        print(f"Downloading Video-MME from HuggingFace ({cls.HF_REPO})...")
        path = snapshot_download(
            repo_id=cls.HF_REPO,
            repo_type="dataset",
            cache_dir=cache_dir,
            allow_patterns=["data/*.jsonl", "videos/*.mp4"],
        )
        print(f"Video-MME ready at {path}")
        return path

    @property
    def num_requests(self) -> int:
        return self._num_requests

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestInput:
        return self.items[idx]


class SeedTTSDataset(BaseDataset):
    """
    Dataset loader for Seed-TTS — the same TTS benchmark vllm-omni uses
    (vllm-omni CLI flag `--dataset-name seed-tts`; upstream:
    https://github.com/BytedanceSpeech/seed-tts-eval).

    Auto-downloads on first use from Bytedance's official Google Drive link
    (the README of the seed-tts-eval repo points at this same archive). The
    archive is ~1.2 GB and extracts to `seedtts_testset/{en,zh}/...`. Caller
    may override with a local path via `data_dir`.

    Expected post-extraction layout:
        <root>/seedtts_testset/{en,zh}/meta.lst
        <root>/seedtts_testset/{en,zh}/prompt-wavs/<utt_id>.wav

    `meta.lst` is pipe-separated, one row per request:
        utt_id | prompt_transcript | prompt_wav_relative_path | text_to_synthesize

    LIMITATION: Seed-TTS is a *zero-shot voice cloning* benchmark. Each row
    pairs a target sentence with a reference utterance (the WAV + its
    transcript) that the model is supposed to mimic. To stay symmetric with
    `VideoMMEDataset` and the existing `RequestType.T2S` path (which only
    carries text input), this loader emits plain T2S requests using the
    target sentence as the prompt and DROPS the reference WAV / ref_text.
    That exercises mstar's TTS serving path but does NOT evaluate voice
    cloning fidelity (WER vs ground-truth, SIM vs reference speaker).

    For voice-clone eval you would need to either (a) extend `RequestInput`
    + `OurSystem` to carry `ref_audio` / `ref_text` and have mstar's server
    accept them, or (b) point the `VLLMOmni` adapter at vllm-omni's
    `/v1/audio/speech` and pass them in `extra_body` (mirrors what
    `vllm_omni/benchmarks/data_modules/seed_tts_dataset.py` does).
    """

    # Bytedance's official Drive link from the README of github.com/BytedanceSpeech/seed-tts-eval.
    # File is a 1.2GB GNU tar containing `seedtts_testset/`.
    GDRIVE_FILE_ID = "1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP"
    EXTRACTED_ROOT_NAME = "seedtts_testset"

    def __init__(
        self,
        num_requests: int = 100,
        req_type: RequestType = RequestType.T2S,
        locale: str = "en",
        data_dir: str | None = None,
        cache_dir: str | None = None,
    ):
        assert req_type == RequestType.T2S, (
            f"SeedTTSDataset is a TTS benchmark; req_type must be T2S, got {req_type}"
        )
        if locale not in ("en", "zh"):
            raise ValueError(f"locale must be 'en' or 'zh', got {locale!r}")

        self._num_requests = num_requests
        self.locale = locale

        if data_dir is None:
            data_dir = self._auto_download(cache_dir)
        self.data_dir = data_dir

        meta_path = os.path.join(data_dir, locale, "meta.lst")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Seed-TTS meta not found: {meta_path}. "
                f"Expected layout: <root>/{locale}/meta.lst from "
                f"github.com/BytedanceSpeech/seed-tts-eval. If you have a "
                f"local copy, point --seed-tts-dir at the directory that "
                f"contains the {locale}/ subfolder."
            )

        self.items: list[RequestInput] = []
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                # Format: utt_id|prompt_transcript|prompt_wav_rel|text_to_synthesize
                if len(parts) < 4:
                    continue
                target_text = parts[3].strip()
                if not target_text:
                    continue
                self.items.append(RequestInput(
                    req_type=req_type,
                    prompt=target_text,
                ))

        if not self.items:
            raise RuntimeError(
                f"SeedTTSDataset loaded 0 usable rows from {meta_path} -- "
                f"check that the meta.lst follows the 4-pipe seed-tts-eval format."
            )

        self.items = self._resize_data(self.items)

    @classmethod
    def _auto_download(cls, cache_dir: str | None) -> str:
        """Download + extract the Bytedance Seed-TTS test set tarball.

        Returns the path to the extracted `seedtts_testset/` directory. The
        download is skipped if the extracted directory already exists with a
        valid en/meta.lst (so re-runs are instant).
        """
        import tarfile

        root = os.path.expanduser(cache_dir or "~/.cache/mstar-benchmark")
        os.makedirs(root, exist_ok=True)
        extract_root = os.path.join(root, cls.EXTRACTED_ROOT_NAME)

        # Fast path: already extracted.
        if os.path.isfile(os.path.join(extract_root, "en", "meta.lst")):
            return extract_root

        archive_path = os.path.join(root, "seedtts_testset.tar")
        if not os.path.isfile(archive_path):
            try:
                import gdown
            except ImportError as e:
                raise ImportError(
                    "Seed-TTS auto-download needs `gdown` (pip install gdown), "
                    "or pass --seed-tts-dir pointing at a local copy of the "
                    "extracted seedtts_testset/ directory."
                ) from e
            print(
                f"Downloading Seed-TTS from Google Drive "
                f"(id={cls.GDRIVE_FILE_ID}, ~1.2 GB) -> {archive_path} ..."
            )
            url = f"https://drive.google.com/uc?id={cls.GDRIVE_FILE_ID}"
            gdown.download(url, archive_path, quiet=False)

        print(f"Extracting Seed-TTS to {root} ...")
        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(path=root)

        if not os.path.isfile(os.path.join(extract_root, "en", "meta.lst")):
            raise RuntimeError(
                f"Seed-TTS extraction produced unexpected layout at {extract_root}. "
                f"Expected en/meta.lst inside."
            )
        print(f"Seed-TTS ready at {extract_root}")
        return extract_root

    @property
    def num_requests(self) -> int:
        return self._num_requests

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestInput:
        return self.items[idx]
