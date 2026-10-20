"""GR00T N1.7's request preprocessing: one robot observation -> the tensors
the model consumes. Mirrors the eval path of Isaac-GR00T's
``Gr00tN1d7Processor.process_observation`` (``processing_gr00t_n1d7.py``),
driven by the checkpoint's ``processor_config.json``, ``statistics.json`` and
``embodiment_id.json``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

# The one reference processor variant this port implements; any other value
# is different preprocessing, so it fails here rather than running the wrong one.
_REQUIRED = {
    "use_mean_std": False,
    "apply_sincos_state_encoding": False,
    "exclude_state": False,
    "use_percentiles": True,
    "clip_outliers": True,
    "use_albumentations": True,  # the eval transform below; torchvision's differs
    "letter_box_transform": False,
    "formalize_language": True,
}


class Gr00tN17Preprocessor:
    def __init__(
        self,
        processor_kwargs: dict,
        statistics: dict,
        embodiment_ids: dict[str, int],
        max_state_dim: int,
        max_action_horizon: int,
        processor=None,
    ):
        bad = {k: processor_kwargs.get(k) for k, v in _REQUIRED.items() if processor_kwargs.get(k, v) != v}
        modality_configs = processor_kwargs["modality_configs"]
        for tag, mc in modality_configs.items():
            for key in ("mean_std_embedding_keys", "sin_cos_embedding_keys"):
                if mc["state"].get(key):
                    bad[f"{tag}.state.{key}"] = mc["state"][key]
        if bad:
            raise ValueError(f"unsupported GR00T N1.7 processor variant: {bad}")

        lo, hi = "q01", "q99"
        self.video_keys = {tag: mc["video"]["modality_keys"] for tag, mc in modality_configs.items()}
        self.state_keys = {tag: mc["state"]["modality_keys"] for tag, mc in modality_configs.items()}
        self.state_bounds = {
            tag: {
                key: (np.array(statistics[tag]["state"][key][lo]), np.array(statistics[tag]["state"][key][hi]))
                for key in keys
            }
            for tag, keys in self.state_keys.items()
        }
        self.action_horizon = {tag: len(mc["action"]["delta_indices"]) for tag, mc in modality_configs.items()}
        self.embodiment_ids = {tag: embodiment_ids[tag] for tag in modality_configs}
        self.image_edge = processor_kwargs["shortest_image_edge"]
        self.crop_fraction = processor_kwargs["crop_fraction"]
        self.max_state_dim = max_state_dim
        # the backbone's Qwen3VLProcessor; left padding as the reference sets it
        self.processor = processor
        if processor is not None:
            processor.tokenizer.padding_side = "left"
        self.max_action_horizon = max_action_horizon
        too_long = {t: h for t, h in self.action_horizon.items() if h > max_action_horizon}
        if too_long:
            raise ValueError(f"action horizons {too_long} exceed the model's {max_action_horizon}")

    @classmethod
    def from_pretrained(
        cls, path: str | Path, max_state_dim: int, max_action_horizon: int,
        backbone_path: str | Path | None = None, cache_dir: str | None = None,
    ):
        """``backbone_path`` (Cosmos-Reason2-2B) loads the text/image processor;
        without it only ``state``/``image``/``embodiment_id``/``action_mask`` work.

        It may be a hub id, which is what registration passes, so ``cache_dir``
        goes with it: a deployment that sets one expects nothing written to the
        default cache."""
        processor = None
        if backbone_path is not None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(backbone_path, cache_dir=cache_dir)
        path = Path(path)
        return cls(
            processor_kwargs=json.loads((path / "processor_config.json").read_text())["processor_kwargs"],
            statistics=json.loads((path / "statistics.json").read_text()),
            embodiment_ids=json.loads((path / "embodiment_id.json").read_text()),
            max_state_dim=max_state_dim,
            max_action_horizon=max_action_horizon,
            processor=processor,
        )

    def _tag(self, embodiment_tag: str) -> str:
        # the checkpoint's own spelling only, not the reference's short enum
        # names (``XDOF``): those exist only in its Python code
        tag = embodiment_tag.lower()
        if tag not in self.state_keys:
            raise ValueError(
                f"GR00T embodiment_tag {embodiment_tag!r} is not in this checkpoint; "
                f"expected one of {sorted(self.state_keys)}"
            )
        return tag

    def state(self, embodiment_tag: str, state: dict[str, np.ndarray]) -> torch.Tensor:
        """Raw state per joint group ``(..., D)`` -> ``(..., max_state_dim)``:
        min-max to [-1, 1] per group, clipped, concatenated, zero-padded."""
        tag = self._tag(embodiment_tag)
        parts = []
        for key in self.state_keys[tag]:
            # ponytail: float32 like the reference's dataset path; float64
            # input would round differently from what NVIDIA serves.
            x = np.asarray(state[key], dtype=np.float32)
            lo, hi = self.state_bounds[tag][key]
            if lo.size == 0:
                # no stats for this group (xdof's eef here): the reference
                # writes zeros as wide as the input, the layout it trained on
                parts.append(np.zeros_like(x))
                continue
            if x.shape[-1] != lo.size:
                raise ValueError(f"state[{key!r}] has {x.shape[-1]} values, expected {lo.size}")
            # the reference's normalize_values_minmax, op for op: a constant
            # dimension (isclose(lo, hi)) stays 0. Two roundings, as there: the
            # float64 quotient is stored float32, then 2n - 1 runs in float32;
            # folding them into one expression moves results by an ulp.
            out = np.zeros_like(x)
            live = ~np.isclose(hi, lo)
            out[..., live] = (x[..., live] - lo[live]) / (hi[live] - lo[live])
            out[..., live] = 2 * out[..., live] - 1
            parts.append(np.clip(out, -1.0, 1.0))
        s = torch.from_numpy(np.concatenate(parts, axis=-1))
        if s.shape[-1] > self.max_state_dim:
            raise ValueError(f"state has {s.shape[-1]} dims, more than max_state_dim {self.max_state_dim}")
        return torch.cat([s, torch.zeros(*s.shape[:-1], self.max_state_dim - s.shape[-1])], dim=-1)

    def embodiment_id(self, embodiment_tag: str, batch: int) -> torch.Tensor:
        # long, not the reference's int32: it indexes the per-embodiment weights
        return torch.full((batch,), self.embodiment_ids[self._tag(embodiment_tag)], dtype=torch.long)

    def action_mask(self, embodiment_tag: str, batch: int) -> torch.Tensor:
        mask = torch.zeros((batch, self.max_action_horizon), dtype=torch.float32)
        mask[:, : self.action_horizon[self._tag(embodiment_tag)]] = 1.0
        return mask

    def image(self, img: np.ndarray) -> np.ndarray:
        """One uint8 ``(H, W, 3)`` frame through the reference's eval transform:
        shortest edge to 256 (area), centre crop 95%, shortest edge to 256 again
        (``build_image_transformations_albumentations``)."""
        img = self._shortest_edge_to(img)
        h, w = img.shape[:2]
        # FractionalCenterCrop's arithmetic
        ch, cw = max(1, int(h * self.crop_fraction)), max(1, int(w * self.crop_fraction))
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return self._shortest_edge_to(img[y0 : y0 + ch, x0 : x0 + cw])

    def _shortest_edge_to(self, img: np.ndarray) -> np.ndarray:
        # albumentations' SmallestMaxSize: round() each side, skip at scale 1.
        # cv2's INTER_AREA is the reference's filter; torchvision's bilinear
        # moved actions ~5x the whole port's error (D05).
        import cv2

        h, w = img.shape[:2]
        scale = self.image_edge / float(min(h, w))
        if scale == 1.0:
            return img
        return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)

    def __call__(
        self,
        embodiment_tag: str,
        video: dict[str, np.ndarray],
        language: list[str],
        state: dict[str, np.ndarray],
    ) -> dict[str, torch.Tensor]:
        """A batch of observations -> the model's inputs, as the reference's
        ``process_observation``. ``video[view]`` is uint8 ``(B, T, H, W, 3)``,
        ``language`` one instruction per item, ``state[key]`` ``(B, T, D)``."""
        tag = self._tag(embodiment_tag)
        views = [np.asarray(video[v]) for v in self.video_keys[tag]]
        batch, steps = views[0].shape[:2]
        if len(language) != batch:
            raise ValueError(f"{len(language)} instructions for a batch of {batch}")
        texts, frames = [], []
        for b in range(batch):
            # time-major, views in modality order: (t0, v0), (t0, v1), (t1, v0), ...
            item = [
                torch.from_numpy(np.ascontiguousarray(self.image(v[b, t]))).permute(2, 0, 1)
                for t in range(steps)
                for v in views
            ]
            # formalize_language: lowercase, strip punctuation
            text = re.sub(r"[^\w\s]", "", language[b].lower())
            conversation = [{
                "role": "user",
                "content": [*({"type": "image", "image": f} for f in item), {"type": "text", "text": text}],
            }]
            texts.append(self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False))
            frames.extend(item)
        # one call over the batch, so padding (left) is shared as in the reference
        out = dict(self.processor(text=texts, images=frames, return_tensors="pt", padding=True))
        out["state"] = self.state(tag, state)
        out["embodiment_id"] = self.embodiment_id(tag, batch)
        out["action_mask"] = self.action_mask(tag, batch)
        return out
