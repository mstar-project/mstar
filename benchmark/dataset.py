import glob
import json
import os
from abc import ABC, abstractmethod
import wave

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
        if not self.num_requests:
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

    DEFAULT_PROMPT = FOOD101_IMAGE_PROMPTS[0]

    def __init__(
        self,
        num_requests: int = 100,
        req_type: RequestType = RequestType.I2T,
        prompt: str = DEFAULT_PROMPT,
        split: str = "validation",
        cache_dir: str | None = None,
    ):
        valid_types = {RequestType.I2T, RequestType.I2I, RequestType.T2I, RequestType.I2S}
        assert req_type in valid_types, (
            f"Food101Dataset supports {valid_types}, got {req_type}"
        )
        from datasets import load_dataset

        self._num_requests = num_requests
        self.prompt = prompt

        raw = load_dataset(
            "ethz/food101",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
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
                import os, tempfile
                tmp_dir = tempfile.mkdtemp(prefix="food101_")
                img_path = os.path.join(tmp_dir, f"{label}_{len(self.items)}.jpg")
                image.save(img_path)
                self.items.append(
                    RequestInput(
                        req_type=req_type,
                        prompt=self.prompt,
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
        from datasets import load_dataset
        from torchcodec.decoders import VideoDecoder
        from torchcodec.encoders import VideoEncoder
        import torch

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
# DROID is a large-scale robot manipulation dataset with three cameras
# (exterior + two wrists), 7-DOF actions, proprioceptive state, and per-
# episode language instructions.  lerobot/droid_100 is a 100-episode subset
# (~4 GB) suitable for smoke testing without a full download.
#
# Two tasks:
#   "pi05"      – first frame per episode (3 images + state) for pi0.5 VLA.
#   "vjepa2_ac" – video clip + action/state trajectory for V-JEPA 2-AC rollout.
# ---------------------------------------------------------------------------

_CAMERA_CANDIDATES = [
    "observation.images.exterior_image_1_left",
    "observation.images.wrist_image_1_left",
    "observation.images.wrist_image_2_left",
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
    "observation.image",
]
_STATE_CANDIDATES    = ["observation.state", "observation.joint_pos", "state"]
_ACTION_CANDIDATES   = ["action", "actions"]
_LANGUAGE_CANDIDATES = ["language_instruction", "task", "instruction"]


def _detect_columns(features: dict) -> dict:
    def _first(candidates):
        for name in candidates:
            if name in features:
                return name
        return None

    return {
        "cameras":  [c for c in _CAMERA_CANDIDATES if c in features],
        "state":    _first(_STATE_CANDIDATES),
        "action":   _first(_ACTION_CANDIDATES),
        "language": _first(_LANGUAGE_CANDIDATES),
    }


def _extract_image_bytes(img) -> bytes:
    """Convert whatever lerobot provides for an image column to PNG bytes."""
    from PIL import Image as PILImage
    import io as _io

    if isinstance(img, PILImage.Image):
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    if isinstance(img, dict):
        if img.get("bytes"):
            return img["bytes"]
        if img.get("path"):
            return open(img["path"], "rb").read()
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)
    try:
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        buf = _io.BytesIO()
        PILImage.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        pass
    raise TypeError(f"Cannot convert image of type {type(img)} to bytes")


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


class DROIDDataset(BaseDataset):
    """DROID robotics dataset for evaluating pi0.5 and V-JEPA 2-AC.

    Downloads ``lerobot/droid_100`` (~4 GB, 100 episodes) on first use via
    the HuggingFace ``datasets`` library and caches it at ``cache_dir``.
    Images and videos are extracted to ``local_file_dir`` as files so the
    server can receive them as multipart uploads.

    Args:
        local_file_dir:  directory for extracted images/videos (created if absent).
        num_requests:    number of episodes to use (capped at available).
        task:            ``"pi05"`` – first frame per episode → 3 images + state.
                         ``"vjepa2_ac"`` – video clip + action/state trajectory.
        rollout_horizon: rollout horizon H for vjepa2_ac; actions/states are
                         [H, action_dim].
        video_frames:    frames written to the mp4 clip for vjepa2_ac (model
                         subsamples to frames_per_clip=64 internally).
        action_dim:      target dimensionality; DROID data is padded/truncated.
                         Defaults to 32 for pi05 and 7 for vjepa2_ac.
        cache_dir:       HuggingFace cache root (None → HF default).
    """

    HF_REPO = "lerobot/droid_100"

    def __init__(
        self,
        local_file_dir: str,
        num_requests: int = 10,
        task: str = "pi05",
        rollout_horizon: int = 4,
        video_frames: int = 80,
        action_dim: int | None = None,
        cache_dir: str | None = None,
    ):
        assert task in ("pi05", "vjepa2_ac"), \
            f"task must be 'pi05' or 'vjepa2_ac', got {task!r}"
        from datasets import load_dataset

        os.makedirs(local_file_dir, exist_ok=True)
        self.local_file_dir = local_file_dir
        self.task           = task
        self._num_requests  = num_requests
        self.action_dim     = action_dim if action_dim is not None else (32 if task == "pi05" else 7)
        self.rollout_horizon = rollout_horizon
        self.video_frames   = video_frames

        print(f"Loading {self.HF_REPO} (first run downloads ~4 GB)...")
        raw = load_dataset(
            self.HF_REPO, split="train",
            cache_dir=cache_dir, trust_remote_code=True,
        )

        cols = _detect_columns(raw.features)
        if not cols["cameras"]:
            raise RuntimeError(
                f"No camera columns found in {self.HF_REPO}. "
                f"Available: {list(raw.features.keys())}"
            )
        if cols["action"] is None:
            raise RuntimeError(f"No action column found in {self.HF_REPO}")

        print(f"  cameras : {cols['cameras']}")
        print(f"  state   : {cols['state']}")
        print(f"  action  : {cols['action']}")
        print(f"  language: {cols['language']}")

        ep_col    = "episode_index" if "episode_index" in raw.features else None
        frame_col = "frame_index"   if "frame_index"   in raw.features else None

        episodes: dict[int, list] = {}
        for row in raw:
            ep_id = int(row[ep_col]) if ep_col else 0
            episodes.setdefault(ep_id, []).append(row)

        if frame_col:
            for ep_id in episodes:
                episodes[ep_id].sort(key=lambda r: int(r[frame_col]))

        ep_ids = sorted(episodes.keys())[:num_requests]
        print(f"  using {len(ep_ids)} episodes (of {len(episodes)} available)")

        self.items: list[RequestInput] = []
        for i, ep_id in enumerate(ep_ids):
            frames = episodes[ep_id]
            try:
                item = (self._make_pi05_request(i, ep_id, frames, cols)
                        if task == "pi05"
                        else self._make_vjepa2_ac_request(i, ep_id, frames, cols))
            except Exception as exc:
                print(f"  [warn] ep{ep_id}: {exc}")
                item = None
            if item is not None:
                self.items.append(item)

        self.items = self._resize_data(self.items)

    # ------------------------------------------------------------------

    def _make_pi05_request(self, idx, ep_id, frames, cols) -> RequestInput:
        first = frames[0]

        image_paths: list[str] = []
        for cam_col in cols["cameras"][:3]:
            img = first.get(cam_col)
            if img is None:
                continue
            path = os.path.join(self.local_file_dir, f"ep{ep_id}_cam{len(image_paths)}.png")
            with open(path, "wb") as f:
                f.write(_extract_image_bytes(img))
            image_paths.append(path)

        if not image_paths:
            raise ValueError("no camera images extracted")
        while len(image_paths) < 3:
            image_paths.append(image_paths[0])  # pad with first camera

        language = ""
        if cols["language"]:
            language = str(first.get(cols["language"]) or "")
        state = _to_float_list(
            first.get(cols["state"]) if cols["state"] else None,
            self.action_dim,
        )

        return RequestInput(
            req_type=RequestType.VLA,
            prompt=language or "manipulate the object",
            image_path=image_paths[0],
            extra_image_paths=image_paths[1:],
            model_kwargs={"robot_state": state},
        )

    def _make_vjepa2_ac_request(self, idx, ep_id, frames, cols) -> RequestInput:
        if len(frames) < self.rollout_horizon:
            raise ValueError(
                f"episode has {len(frames)} frames, need >= {self.rollout_horizon}"
            )

        n_video = min(self.video_frames, len(frames))
        video_path = os.path.join(self.local_file_dir, f"ep{ep_id}.mp4")
        self._save_video(frames[:n_video], cols["cameras"][0], video_path)

        act_col = cols["action"]
        st_col  = cols["state"]
        actions = [_to_float_list(f.get(act_col), self.action_dim)
                   for f in frames[:self.rollout_horizon]]
        states  = [_to_float_list(f.get(st_col) if st_col else None, self.action_dim)
                   for f in frames[:self.rollout_horizon]]

        language = ""
        if cols["language"]:
            language = str(frames[0].get(cols["language"]) or "")

        return RequestInput(
            req_type=RequestType.V2V,
            prompt=language or "world model rollout",
            video_path=video_path,
            model_kwargs={
                "actions":          actions,
                "states":           states,
                "rollout_horizon":  self.rollout_horizon,
            },
        )

    def _save_video(self, frames, cam_col: str, output_path: str) -> None:
        import io as _io
        import torch
        from PIL import Image as PILImage
        from torchcodec.encoders import VideoEncoder

        tensors = []
        for row in frames:
            img = row.get(cam_col)
            if img is None:
                continue
            pil = PILImage.open(_io.BytesIO(_extract_image_bytes(img))).convert("RGB")
            tensors.append(torch.from_numpy(np.array(pil, dtype=np.uint8)))

        if not tensors:
            raise ValueError(f"no frames for camera column {cam_col!r}")

        VideoEncoder(frames=torch.stack(tensors), frame_rate=15).to_file(output_path)

    # ------------------------------------------------------------------

    @property
    def num_requests(self) -> int:
        return self._num_requests

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestInput:
        return self.items[idx]
