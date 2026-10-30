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
    # the decode's relative->absolute step is only correct under this
    "use_relative_action": True,
    "use_albumentations": True,  # the eval transform below; torchvision's differs
    "letter_box_transform": False,
    "formalize_language": True,
}


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """A 6D rotation -- a rotation matrix's first two rows -- back to the matrix.

    ``pose.py:436-455``'s Gram-Schmidt, **in the input's own dtype** (the raw
    state's float32 for a reference frame); upcasting is worse, see `unapply`.
    The reference refuses a degenerate row (scipy ``LinAlgError``) where plain
    Gram-Schmidt would return NaN, so the norms are checked.
    """
    rows = rot6d.reshape(2, 3)
    if not np.all(np.isfinite(rows)):
        # the reference's rejection too, as a bare scipy `LinAlgError`
        raise ValueError(
            "GR00T action decode: the rotation has a non-finite component, so no "
            "frame can be built from it"
        )
    row1, row2 = rows[0], rows[1]
    n1 = np.linalg.norm(row1)
    # finite and `> 0`: a float32 row near 1e20 overflows the norm to `inf`, the
    # division gives zero, and a `== 0` test on the raw row would pass it through
    # as a silent identity rotation. The reference rejects it ("null coordinate frame").
    if not (np.isfinite(n1) and n1 > 0.0):
        raise ValueError(
            "GR00T action decode: the rotation's first row has norm "
            f"{float(n1)!r}, so it does not define a direction"
        )
    row1 = row1 / n1
    row2 = row2 - np.dot(row1, row2) * row1
    n2 = np.linalg.norm(row2)
    if not (np.isfinite(n2) and n2 > 0.0):
        raise ValueError(
            "GR00T action decode: the rotation's two rows are parallel, so they "
            "do not span a plane"
        )
    row2 = row2 / n2
    return _nearest_rotation(np.vstack([row1, row2, np.cross(row1, row2)]))


def _nearest_rotation(m: np.ndarray) -> np.ndarray:
    """The nearest true rotation to ``m``, by SVD: ``U @ Vt``.

    The reference stores every rotation as a scipy ``Rotation``, whose
    ``from_matrix`` re-orthogonalises (``pose.py:500-501``, ``:424``). **Not a
    no-op**: a float32 Gram-Schmidt is orthonormal only to ~1e-07. numpy's SVD
    reproduces scipy's snap to 8e-16, so scipy is not a dependency.
    """
    u, _, vt = np.linalg.svd(np.asarray(m, dtype=np.float64))
    return u @ vt


def _homogeneous(xyz_rot6d: np.ndarray) -> np.ndarray:
    """``(9,)`` translation + 6D rotation -> a 4x4 transform (``pose.py:576-580``)."""
    h = np.eye(4)
    h[:3, :3] = _rot6d_to_matrix(xyz_rot6d[3:])
    h[:3, 3] = xyz_rot6d[:3]
    return h


def _compose_eef(relative: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``(T, 9)`` relative poses on a ``(9,)`` reference -> ``(T, 9)`` absolute.

    ``T_abs = T_ref @ T_rel`` per pose (``action_chunking.py:641-653``), not the
    joints' addition: composing rigid transforms rotates the relative translation
    into the reference's frame. A loop like the reference's, not a batched
    ``matmul``, which could pick a different BLAS kernel and move the last bits.
    """
    t_ref = _homogeneous(reference)
    out = np.empty_like(relative)
    for i, rel in enumerate(relative):
        t_abs = t_ref @ _homogeneous(rel)
        out[i, :3] = t_abs[:3, 3]
        # snapped again on the way out: the reference reads the composed pose
        # back through `EndEffectorPose(homogeneous=...)`, which re-orthogonalises
        out[i, 3:] = _nearest_rotation(t_abs[:3, :3])[:2, :].flatten()
    return out


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
            # the action side has its own mean/std opt-in, and the decode below
            # only implements min-max; every tag in this checkpoint has it unset
            if mc["action"].get("mean_std_embedding_keys"):
                bad[f"{tag}.action.mean_std_embedding_keys"] = mc["action"]["mean_std_embedding_keys"]
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
        # frames per view, 2 for every tag in this checkpoint
        self.video_steps = {tag: len(mc["video"]["delta_indices"]) for tag, mc in modality_configs.items()}
        self.action_keys = {tag: mc["action"]["modality_keys"] for tag, mc in modality_configs.items()}
        # The decode's tables follow the reference's rules
        # (`state_action_processor.py:156-204`), which are not the state's:
        #  * `action_dims` is the *absolute* stats' width even for a relative key:
        #    the relative override keeps the `dim` computed before it (`:200-204`).
        #    It is 0 where a group has no absolute stats; see `unapply`.
        #  * a relative key reads `min`/`max` off `relative_action`, an absolute
        #    one `q01`/`q99` off `action`: the override bypasses `use_percentiles`.
        #    Percentiles for a relative key would silently warp the trajectory.
        #  * relative bounds are per timestep, `(T, D)`, absolute ones per
        #    dimension, `(D,)`, so the unnormalise broadcasts over the batch axis
        #    only, and a horizon that disagrees raises rather than mis-scales.
        #  * the `1e-8` range floor (`:169`) only sizes `dim` and the stored bounds
        #    are unfloored, so it is not carried: a constant dimension decodes to
        #    that constant.
        self.action_configs = {}
        self.action_bounds = {}
        self.action_dims = {}
        for tag, mc in modality_configs.items():
            keys = mc["action"]["modality_keys"]
            cfgs = mc["action"].get("action_configs") or [None] * len(keys)
            self.action_configs[tag] = dict(zip(keys, cfgs, strict=True))
            self.action_bounds[tag] = {}
            self.action_dims[tag] = {}
            for key, cfg in zip(keys, cfgs, strict=True):
                absolute = statistics[tag]["action"][key]
                self.action_dims[tag][key] = len(absolute["q99"])
                if cfg is not None and cfg["rep"] == "RELATIVE":
                    src, lo, hi = statistics[tag]["relative_action"][key], "min", "max"
                else:
                    src, lo, hi = absolute, "q01", "q99"
                self.action_bounds[tag][key] = (
                    np.array(src[lo], dtype=np.float64), np.array(src[hi], dtype=np.float64),
                )
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
        It may be a hub id, so ``cache_dir`` goes with it: a deployment that sets
        one expects nothing in the default cache."""
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
            # float32 like the reference's dataset path; float64
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

    # --------------------------------------------------------------- decode
    def unapply(
        self,
        embodiment_tag: str,
        action: np.ndarray,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """The model's normalised chunk -> raw absolute actions per joint group.

        Mirrors ``Gr00tN1d7Processor.unapply`` (``processing_gr00t_n1d7.py:406``)
        and ``StateActionProcessor.unapply_action`` (``state_action_processor.py:421``).

        ``action`` is ``(B, T, max_action_dim)`` off the action decoder; the
        horizon and each group's width are cut out of the padding. ``state`` is
        the request's **raw** state per group, required when a group is relative.

        Dtypes are the reference's, not the widest: the chunk and the reference
        frame widen only where the reference does, on the multiply by the float64
        bounds. ``eef_9d`` is not bit-exact at any dtype: ``np.dot`` on float32
        3-vectors can differ between the reference's numpy 1.26 and ours, 2.5,
        which moves the Gram-Schmidt by one float32 ulp of the reference frame;
        that ulp is the bound, and a float64 Gram-Schmidt is worse everywhere.
        """
        tag = self._tag(embodiment_tag)
        # **Not** upcast: the reference runs `(clip(x) + 1) / 2` in the chunk's
        # own float32. A bf16 value plus one is exact there, but a tiny one
        # rounds, by up to `2**-24 / 2 * range`, which upcasting would not reproduce.
        chunk = np.asarray(action)
        if chunk.ndim != 3:
            raise ValueError(f"GR00T action decode wants (B, T, D); got {chunk.shape}")
        horizon = self.action_horizon[tag]
        width = sum(self.action_dims[tag].values())
        # the padding would hide both: a chunk one group short decodes every later
        # group from the wrong offset, one short of the horizon returns fewer steps
        if chunk.shape[1] < horizon or chunk.shape[2] < width:
            raise ValueError(
                f"GR00T {tag!r} decodes {width} dims over {horizon} steps, but the "
                f"chunk is {chunk.shape[1]} x {chunk.shape[2]}"
            )
        out: dict[str, np.ndarray] = {}
        start = 0
        for key in self.action_keys[tag]:
            dim = self.action_dims[tag][key]
            if dim == 0:
                # no absolute stats, so no width to decode into: the checkpoint's
                # hole (only xdof's two wrist groups). The reference fails too, when
                # its unnormalise meets the empty slice (`state_action_processor.py:451-469`).
                raise ValueError(
                    f"GR00T embodiment {tag!r} has no action statistics for joint "
                    f"group {key!r}, so its decoded width is 0 and the chunk cannot "
                    "be decoded; the reference implementation fails here too"
                )
            cfg = self.action_configs[tag][key]
            lo, hi = self.action_bounds[tag][key]
            group = chunk[:, :horizon, start:start + dim]
            start += dim
            # `utils.py:146-152`: clip into [-1, 1] first -- the decoder's output
            # is unbounded, and the bound it is mapped onto is a percentile
            group = (np.clip(group, -1.0, 1.0) + 1.0) / 2.0 * (hi - lo) + lo
            if cfg is not None and cfg["rep"] == "RELATIVE":
                group = self._to_absolute(tag, key, cfg, group, state)
            out[key] = group
        return out

    def _to_absolute(
        self,
        tag: str,
        key: str,
        cfg: dict,
        relative: np.ndarray,
        state: dict[str, np.ndarray] | None,
    ) -> np.ndarray:
        """One relative group ``(B, T, D)`` onto the raw state's last timestep."""
        state_key = cfg.get("state_key") or key
        if state is None or state_key not in state:
            raise ValueError(
                f"GR00T embodiment {tag!r} decodes joint group {key!r} relative to "
                f"state {state_key!r}, so the request's raw state is required to "
                "turn it into an absolute action"
            )
        # float32, not upcast: the reference builds the frame from the raw
        # state's float32 and runs its Gram-Schmidt there (see `unapply`). JSON
        # delivers Python floats, which `asarray` alone would make float64.
        ref = np.asarray(state[state_key], dtype=np.float32)
        if ref.ndim == 2:  # (T, D) -- one request, as the reference also allows
            ref = ref[None]
        # the reference frame is the state's **last** timestep
        # (`state_action_processor.py:511-515`), per request
        ref = ref[:, -1, :]
        if ref.shape[0] != relative.shape[0] or ref.shape[-1] != relative.shape[-1]:
            raise ValueError(
                f"GR00T {tag!r} {key!r}: state {state_key!r} is {ref.shape} after "
                f"taking the last timestep, which does not match the action's "
                f"{relative.shape}"
            )
        if cfg["type"] == "NON_EEF":
            # joints add (`action_chunking.py:383-387`)
            return ref[:, None, :] + relative
        if cfg["type"] != "EEF":
            raise ValueError(f"GR00T {tag!r} {key!r}: unsupported action type {cfg['type']!r}")
        if cfg["format"] != "XYZ_ROT6D":
            raise ValueError(f"GR00T {tag!r} {key!r}: unsupported action format {cfg['format']!r}")
        return np.stack([
            _compose_eef(rel, frame) for rel, frame in zip(relative, ref, strict=True)
        ])

    def frame_layout(self, embodiment_tag: str) -> tuple[tuple[str, ...], int]:
        """This embodiment's camera views, in order, and its frames per view.

        The order is the request's contract: the engine keys media by *modality*,
        not name (`PreprocessWorkerThread._load_media`), so cameras arrive as one
        positional list.
        """
        tag = self._tag(embodiment_tag)
        return tuple(self.video_keys[tag]), self.video_steps[tag]

    def warmup_chat_template(self) -> None:
        """Compile the chat template up front, rather than on the first observation."""
        if self.processor is None:
            return
        self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": ""}]}],
            tokenize=False, add_generation_prompt=False,
        )

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
        # moves the actions measurably.
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
        # The names are the checkpoint's, so a client cannot guess them; a bare
        # `KeyError` would be a 500 whose body is just the key
        # (`PreprocessWorkerThread._fail_request`).
        for name, want, got in (("video", self.video_keys[tag], video),
                                ("state", self.state_keys[tag], state)):
            missing = [k for k in want if k not in got]
            if missing:
                raise ValueError(
                    f"GR00T {tag!r} needs {name} {list(want)}; missing {missing}"
                )
        views = [np.asarray(video[v]) for v in self.video_keys[tag]]
        # uint8, as the reference asserts (`gr00t_policy.py:276-279`): the HF image
        # processor rescales by 1/255 whatever it gets, so pre-normalised frames
        # become a black image with no error and plausible-looking actions.
        bad = [v for v, a in zip(self.video_keys[tag], views, strict=True) if a.dtype != np.uint8]
        if bad:
            raise ValueError(
                f"GR00T video must be uint8 (0-255); {bad} came as "
                f"{[views[list(self.video_keys[tag]).index(v)].dtype for v in bad]}"
            )
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
            # the reference stacks every view's transformed frames
            # (`processing_gr00t_n1d7.py:746`), so cameras must share an aspect ratio
            sizes = {tuple(f.shape[1:]) for f in item}
            if len(sizes) > 1:
                raise ValueError(
                    f"GR00T needs every camera at one aspect ratio, as the reference stacks "
                    f"their transformed frames; these transform to {sorted(sizes)}"
                )
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
