"""Quality of a cached clip against the dense clip of the same prompt and seed.

Every function takes clips as uint8 frames ``[T, H, W, 3]`` (torch tensors or numpy arrays). The pixel metrics
(PSNR, SSIM, LPIPS) say how far the cache moved the output; the CLIP score says whether the clip still shows the
prompt; the flicker metrics say whether the cache added change between frames that the dense clip does not have.
LPIPS and CLIP need packages or weights a box may lack: without them the metric is None and one line says why.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# PSNR of identical frames, where the ratio has no finite value.
PSNR_CAP = 100.0
_CLIP_FRAMES = 8
CLIP_MODEL = "openai/clip-vit-base-patch32"
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def as_frames(clip: Any) -> torch.Tensor:
    """``clip`` as a uint8 ``[T, H, W, 3]`` tensor."""
    frames = torch.as_tensor(clip)
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"a clip is uint8 [T, H, W, 3]; got {frames.dtype} {tuple(frames.shape)}")
    return frames


def decode_mp4(mp4: bytes | str) -> torch.Tensor:
    """The video stream of an mp4 (bytes, or a path) as uint8 ``[T, H, W, 3]``."""
    import av
    import numpy as np

    with av.open(io.BytesIO(mp4) if isinstance(mp4, bytes) else mp4) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if not frames:
        raise ValueError("the mp4 has no video frames")
    return torch.from_numpy(np.stack(frames))


def frames_md5(clip: Any) -> str:
    """A digest of the clip's shape and pixels: equal for two clips exactly when they are bit-identical."""
    frames = as_frames(clip).contiguous().cpu()
    digest = hashlib.md5(str(tuple(frames.shape)).encode())
    digest.update(frames.numpy().tobytes())
    return digest.hexdigest()


def _psnr(mse: torch.Tensor) -> torch.Tensor:
    return (10.0 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-10))).clamp_max(PSNR_CAP)


def psnr_per_frame(ref: Any, test: Any) -> torch.Tensor:
    """PSNR of each frame in dB, ``PSNR_CAP`` where the two frames are identical."""
    ref, test = _pair(ref, test)
    return _psnr((ref.double() - test.double()).square().mean(dim=(1, 2, 3)))


def _window(kind: str) -> tuple[torch.Tensor, float]:
    """The SSIM window as 1-D taps, and the factor that turns its biased covariance into the estimator ``kind``
    uses."""
    if kind == "gaussian":
        # Wang et al.'s 11 taps at sigma 1.5, biased moments
        taps = torch.exp(-torch.arange(-5, 6, dtype=torch.float64).square() / (2 * 1.5 ** 2))
        return taps / taps.sum(), 1.0
    if kind == "uniform":
        # scikit-image's default: a 7x7 box and the sample covariance
        return torch.full((7,), 1.0 / 7, dtype=torch.float64), 49.0 / 48.0
    raise ValueError(f"SSIM window must be 'gaussian' or 'uniform'; got {kind!r}")


def ssim_per_frame(ref: Any, test: Any, window: str = "gaussian", device: torch.device | str = "cpu") -> torch.Tensor:
    """SSIM of each frame, the mean over channels and over the positions where the window fits.

    ``window`` "gaussian" is the standard form (scikit-image's ``gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False``); "uniform" is scikit-image's default, which earlier quality tables used."""
    ref, test = _pair(ref, test)
    taps, cov_norm = _window(window)
    if min(ref.shape[1:3]) < taps.numel():
        raise ValueError(f"SSIM's {taps.numel()}-tap window does not fit {tuple(ref.shape[1:3])} frames")
    taps = taps.to(device)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.conv2d(x, taps.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)
        return F.conv2d(x, taps.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)

    scores = []
    for start in range(0, ref.shape[0], 8):
        x = ref[start:start + 8].to(device).permute(0, 3, 1, 2).double()
        y = test[start:start + 8].to(device).permute(0, 3, 1, 2).double()
        mu_x, mu_y = blur(x), blur(y)
        var_x = cov_norm * (blur(x * x) - mu_x * mu_x)
        var_y = cov_norm * (blur(y * y) - mu_y * mu_y)
        cov = cov_norm * (blur(x * y) - mu_x * mu_y)
        ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2))
        scores.append(ssim.mean(dim=(1, 2, 3)).cpu())
    return torch.cat(scores)


def temporal_flicker(clip: Any) -> float:
    """VBench's temporal-flickering score, ``1 - mean|f[t+1] - f[t]| / 255``: 1 for a still clip, lower the more
    consecutive frames differ. It counts real motion too, so read it next to the dense clip's."""
    frames = as_frames(clip)
    if frames.shape[0] < 2:
        return 1.0
    return 1.0 - float((frames[1:].double() - frames[:-1].double()).abs().mean()) / 255.0


def _second_difference_energy(frames: torch.Tensor) -> float:
    x = frames.double()
    return float((x[2:] - 2 * x[1:-1] + x[:-2]).square().mean())


def flicker_excess(ref: Any, test: Any) -> float:
    """How much more the test clip changes back and forth between frames than the dense clip: the mean energy of
    the second temporal difference ``f[t+1] - 2 f[t] + f[t-1]`` of the test clip over the dense clip's, minus 1.
    Smooth motion has a small second difference and frame-to-frame jitter a large one, so 0 is as steady as dense
    and 0.5 is half again as much jitter; a clip under three frames scores 0."""
    ref, test = _pair(ref, test)
    if ref.shape[0] < 3:
        return 0.0
    # one grey level squared of slack, so a still dense clip does not divide by zero
    return (_second_difference_energy(test) + 1.0) / (_second_difference_energy(ref) + 1.0) - 1.0


def motion_psnr(ref: Any, test: Any) -> float:
    """PSNR between the two clips' frame differences ``f[t+1] - f[t]``, the mean over frame pairs: whether the
    cache changed how the clip moves, apart from an error that sits still in every frame (which cancels here and
    shows in the per-frame PSNR instead)."""
    ref, test = _pair(ref, test)
    if ref.shape[0] < 2:
        return PSNR_CAP
    d_ref = ref[1:].double() - ref[:-1].double()
    d_test = test[1:].double() - test[:-1].double()
    return float(_psnr((d_ref - d_test).square().mean(dim=(1, 2, 3))).mean())


def _pair(ref: Any, test: Any) -> tuple[torch.Tensor, torch.Tensor]:
    ref, test = as_frames(ref), as_frames(test)
    if ref.shape != test.shape:
        raise ValueError(f"the clips differ in shape: {tuple(ref.shape)} against {tuple(test.shape)}")
    return ref, test


class Lpips:
    """LPIPS (AlexNet) per frame, from the ``lpips`` package; ``available`` is False when it or its weights are
    missing."""

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = device
        self.note: str | None = None
        self._net = None
        try:
            import lpips

            self._net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as error:  # noqa: BLE001 - an optional metric: any failure to load it turns it off
            self.note = f"LPIPS off: {type(error).__name__}: {error}"
            logger.warning(self.note)

    @property
    def available(self) -> bool:
        return self._net is not None

    @torch.no_grad()
    def per_frame(self, ref: Any, test: Any) -> torch.Tensor | None:
        if self._net is None:
            return None
        ref, test = _pair(ref, test)
        scores = []
        for start in range(0, ref.shape[0], 16):
            a = ref[start:start + 16].to(self.device).permute(0, 3, 1, 2).float() / 127.5 - 1
            b = test[start:start + 16].to(self.device).permute(0, 3, 1, 2).float() / 127.5 - 1
            scores.append(self._net(a, b).flatten().cpu())
        return torch.cat(scores)


class ClipScore:
    """CLIP text-video similarity: the mean over evenly spaced frames of the cosine between the prompt's and the
    frame's embeddings (``openai/clip-vit-base-patch32`` through ``transformers``). It moves when the clip stops
    showing the prompt, which the pixel metrics cannot tell from a harmless change of detail. ``available`` is
    False when the package or the weights are missing."""

    def __init__(self, device: torch.device | str = "cpu", model: str = CLIP_MODEL):
        self.device = device
        self.note: str | None = None
        self._model = self._tokenizer = None
        try:
            from transformers import AutoTokenizer, CLIPModel

            self._tokenizer = AutoTokenizer.from_pretrained(model)
            self._model = CLIPModel.from_pretrained(model).to(device).eval()
        except Exception as error:  # noqa: BLE001 - an optional metric: any failure to load it turns it off
            self._model = self._tokenizer = None
            self.note = f"CLIP score off: {type(error).__name__}: {str(error).splitlines()[0]}"
            logger.warning(self.note)

    @property
    def available(self) -> bool:
        return self._model is not None

    def _pixels(self, frames: torch.Tensor) -> torch.Tensor:
        """CLIP's preprocessing: the short side to 224 (bicubic), a center crop, its mean and std."""
        x = frames.to(self.device).permute(0, 3, 1, 2).float() / 255.0
        height, width = x.shape[-2:]
        scale = 224 / min(height, width)
        size = (max(224, round(height * scale)), max(224, round(width * scale)))
        x = F.interpolate(x, size=size, mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
        top, left = (size[0] - 224) // 2, (size[1] - 224) // 2
        x = x[..., top:top + 224, left:left + 224]
        mean, std = (torch.tensor(v, device=x.device).view(1, 3, 1, 1) for v in (_CLIP_MEAN, _CLIP_STD))
        return (x - mean) / std

    @torch.no_grad()
    def score(self, clip: Any, prompt: str) -> float | None:
        if self._model is None:
            return None
        frames = as_frames(clip)
        picks = torch.linspace(0, frames.shape[0] - 1, min(_CLIP_FRAMES, frames.shape[0])).round().long()
        tokens = self._tokenizer([prompt], padding=True, truncation=True, max_length=77, return_tensors="pt")
        text = self._model.get_text_features(**{k: v.to(self.device) for k, v in tokens.items()})
        image = self._model.get_image_features(pixel_values=self._pixels(frames[picks]))
        text, image = (getattr(v, "pooler_output", v) for v in (text, image))
        return float(F.cosine_similarity(image.float(), text.float().expand_as(image), dim=-1).mean())


@dataclass(frozen=True)
class Quality:
    """One clip's scores against its dense reference. A None is a metric that was not available."""

    psnr: float
    # the worst frame's
    psnr_min: float
    ssim: float
    lpips: float | None
    lpips_max: float | None
    # CLIP text-video similarity of the clip, and of the dense clip
    clip: float | None
    clip_ref: float | None
    # VBench temporal flickering of the clip, and of the dense clip
    flicker: float
    flicker_ref: float
    flicker_excess: float
    motion_psnr: float
    identical: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Scorer:
    """The metrics a run computes, with the optional ones loaded once."""

    device: torch.device | str = "cpu"
    ssim_window: str = "gaussian"
    lpips: Lpips | None = None
    clip: ClipScore | None = None

    @classmethod
    def build(
        cls, device: torch.device | str = "cpu", ssim_window: str = "gaussian", optional: bool = True,
        clip_model: str = CLIP_MODEL,
    ) -> Scorer:
        """A scorer on ``device``; ``optional`` False leaves LPIPS and CLIP off without trying to load them.
        ``clip_model`` is the CLIP checkpoint's hub id or a local snapshot directory."""
        _window(ssim_window)
        if not optional:
            return cls(device=device, ssim_window=ssim_window)
        return cls(
            device=device, ssim_window=ssim_window, lpips=Lpips(device), clip=ClipScore(device, clip_model),
        )

    @property
    def notes(self) -> list[str]:
        """Why an optional metric is missing, one line each."""
        parts = {"LPIPS": self.lpips, "CLIP score": self.clip}
        return [
            f"{name} off: not loaded" if part is None else part.note
            for name, part in parts.items() if part is None or not part.available
        ]

    def clip_score(self, clip: Any, prompt: str) -> float | None:
        return None if self.clip is None else self.clip.score(clip, prompt)

    def compare(self, ref: Any, test: Any, prompt: str, clip_ref: float | None = None) -> Quality:
        """``test`` against the dense ``ref``; ``clip_ref`` is the dense clip's CLIP score when already known."""
        ref, test = _pair(ref, test)
        psnr = psnr_per_frame(ref, test)
        lpips = None if self.lpips is None else self.lpips.per_frame(ref, test)
        if clip_ref is None:
            clip_ref = self.clip_score(ref, prompt)
        return Quality(
            psnr=float(psnr.mean()), psnr_min=float(psnr.min()),
            ssim=float(ssim_per_frame(ref, test, self.ssim_window, self.device).mean()),
            lpips=None if lpips is None else float(lpips.mean()),
            lpips_max=None if lpips is None else float(lpips.max()),
            clip=self.clip_score(test, prompt), clip_ref=clip_ref,
            flicker=temporal_flicker(test), flicker_ref=temporal_flicker(ref),
            flicker_excess=flicker_excess(ref, test), motion_psnr=motion_psnr(ref, test),
            identical=bool(torch.equal(ref, test)),
        )
