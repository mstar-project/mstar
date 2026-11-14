"""The Wan2.2-VAE decoder (TI2V-5B: 48 latent channels, 16x spatial and 4x temporal) as Blackwell kernels.

Vendored from FastVideo (hao-ai-lab/FastVideo ``fastvideo-kernel/wan22_vae_decode``, ec9145a, Apache-2.0; ``csrc/``
keeps its sources and comments). Upstream decodes one grid, latents 44x78 (704x1248); here the C++ driver takes the grid
from the latents, which is all the kernels needed, and adds one-GPU entry points without a process group and a
``StreamDecoder`` that keeps the causal history between calls.

What makes it fast (upstream's design, ``csrc/binding.cpp``):

- the 18 3x3x3 convolutions of up blocks 1-3 (most of the FLOPs) are F(4x4, 3x3) Winograd in space with the causal
  3-tap convolution in time: an input transform, 36 CUTLASS SM100 implicit-GEMM products, and an inverse transform
  that also applies the bias, the residual and the next convolution's RMS norm, SiLU and causal padding;
- nearest-x2 + 3x3 upsampling as four 2x2 convolutions at the low resolution, interleaved in one pass;
- several latent frames per pass (``frames_per_pass``), so the low-resolution GEMMs fill the GPU.

It is not bit-exact with ``vae.decode``: Winograd's transforms and the fused norms round differently, so it is an
opt-in near-lossless path (``Wan22Config.vae_decode_kernel``) with a quality table in its PR, never a default. The
video does not depend on ``frames_per_pass`` or on how a stream is cut into calls.

Runs on compute capability 10.0 and 10.3 (B200, GB200, B300): tcgen05 MMA, TMA, 2-CTA clusters. The extension is
built on first use (about a minute, then cached under torch's extensions root) against FlashInfer's bundled CUTLASS
headers, or ``$CUTLASS_DIR``.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import shutil

import torch
from torch import nn

logger = logging.getLogger(__name__)

_CSRC = os.path.join(os.path.dirname(__file__), "csrc")
_SOURCES = (".cu", ".cpp")
LATENT_CHANNELS = 48


def _weight_shapes() -> tuple[tuple[str, tuple[int, ...]], ...]:
    """(name, shape) of the decoder's tensors in the order the kernels take them: diffusers' registration order of
    ``post_quant_conv`` and ``decoder`` of the Wan2.2-TI2V-5B VAE."""
    out = []

    def conv(name, shape):
        out.extend([(f"{name}.weight", tuple(shape)), (f"{name}.bias", (shape[0],))])

    def resnet(name, cin, cout):
        out.append((f"{name}.norm1.gamma", (cin, 1, 1, 1)))
        conv(f"{name}.conv1", (cout, cin, 3, 3, 3))
        out.append((f"{name}.norm2.gamma", (cout, 1, 1, 1)))
        conv(f"{name}.conv2", (cout, cout, 3, 3, 3))
        if cin != cout:
            conv(f"{name}.conv_shortcut", (cout, cin, 1, 1, 1))

    conv("post_quant_conv", (48, 48, 1, 1, 1))
    conv("decoder.conv_in", (1024, 48, 3, 3, 3))
    out.append(("decoder.mid_block.attentions.0.norm.gamma", (1024, 1, 1)))
    conv("decoder.mid_block.attentions.0.to_qkv", (3072, 1024, 1, 1))
    conv("decoder.mid_block.attentions.0.proj", (1024, 1024, 1, 1))
    for j in range(2):
        resnet(f"decoder.mid_block.resnets.{j}", 1024, 1024)
    up_blocks = ((1024, 1024, True, True), (1024, 1024, True, True), (1024, 512, False, True), (512, 256, False, False))
    for i, (cin, cout, temporal, spatial) in enumerate(up_blocks):
        for j in range(3):
            resnet(f"decoder.up_blocks.{i}.resnets.{j}", cin if j == 0 else cout, cout)
        if spatial:
            conv(f"decoder.up_blocks.{i}.upsampler.resample.1", (cout, cout, 3, 3))
        if temporal:
            conv(f"decoder.up_blocks.{i}.upsampler.time_conv", (2 * cout, cout, 3, 1, 1))
    out.append(("decoder.norm_out.gamma", (256, 1, 1, 1)))
    conv("decoder.conv_out", (12, 256, 3, 3, 3))
    return tuple(out)


WEIGHT_SHAPES = _weight_shapes()


def unsupported_reason(device: torch.device | None = None, vae: nn.Module | None = None) -> str | None:
    """Why the kernels cannot decode for ``vae`` on ``device`` (the current CUDA device), or None when they can."""
    if vae is not None:
        params = {name: tuple(p.shape) for name, p in vae.named_parameters()}
        for name, shape in WEIGHT_SHAPES:
            if params.get(name) != shape:
                return f"not the Wan2.2-TI2V-5B VAE ({name} is {params.get(name)}, not {shape})"
    if not torch.cuda.is_available():
        return "no CUDA device"
    from mstar.platforms import current_platform

    if current_platform.is_rocm():
        return "the kernels are CUDA (tcgen05) only"
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) not in ((10, 0), (10, 3)):
        return f"the kernels need compute capability 10.0 or 10.3 (B200 / GB200 / B300), not {major}.{minor}"
    if _cutlass_dir() is None:
        return "no CUTLASS headers (FlashInfer's bundled copy, or $CUTLASS_DIR)"
    if shutil.which("nvcc") is None and not os.environ.get("CUDA_HOME"):
        return "no nvcc to build the extension with"
    return None


def _cutlass_dir() -> str | None:
    for path in (os.environ.get("CUTLASS_DIR"),):
        if path and os.path.isfile(os.path.join(path, "include", "cutlass", "cutlass.h")):
            return path
    try:
        import flashinfer
    except ImportError:
        return None
    path = os.path.join(os.path.dirname(flashinfer.__file__), "data", "cutlass")
    return path if os.path.isfile(os.path.join(path, "include", "cutlass", "cutlass.h")) else None


@functools.lru_cache(maxsize=None)
def extension(arch: str):
    """The built extension for ``arch`` (``sm_100a`` / ``sm_103a``), once per process.

    The sources are copied under torch's extensions root into a directory named by their content, so every checkout
    of one version shares a build and each version gets its own (``mstar.utils.fused_moe.align`` does the same)."""
    from torch.utils.cpp_extension import _get_build_directory, load

    files = sorted(f for f in os.listdir(_CSRC) if os.path.isfile(os.path.join(_CSRC, f)))
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.encode())
        with open(os.path.join(_CSRC, f), "rb") as fh:
            digest.update(fh.read())
    cutlass = _cutlass_dir()
    digest.update(arch.encode())
    digest.update(str(cutlass).encode())
    tag = digest.hexdigest()[:12]
    name = f"_mstar_wan22_vae_{tag}"
    src_dir = os.path.join(os.path.dirname(_get_build_directory(name, verbose=False)), "_mstar_wan22_vae_src", tag)
    if not os.path.isdir(src_dir):
        tmp = f"{src_dir}.{os.getpid()}.tmp"
        shutil.copytree(_CSRC, tmp)
        try:
            os.replace(tmp, src_dir)  # atomic: a racing process sees the whole tree or none
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
    sources = [os.path.join(src_dir, f) for f in files if f.endswith(_SOURCES)]
    logger.info("Wan2.2 VAE kernels: loading %s for %s (built on first use, about a minute)", name, arch)
    return load(
        name=name,
        sources=sources,
        extra_include_paths=[
            src_dir, os.path.join(cutlass, "include"), os.path.join(cutlass, "tools", "util", "include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[f"-gencode=arch=compute_{arch[3:]},code={arch}", "-O3", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


def _arch(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}a"


def decoder_weights(vae: nn.Module, device: torch.device) -> list[torch.Tensor]:
    """The decoder's tensors in ``WEIGHT_SHAPES`` order, as contiguous bf16 copies on ``device`` (whatever layout and
    dtype the VAE holds them in). Copies, so a later layout switch of the VAE (``wan_vae.FastDecoder``) never aliases
    what the kernels read."""
    params = dict(vae.named_parameters())
    weights = []
    for name, shape in WEIGHT_SHAPES:
        if name not in params:
            raise ValueError(f"the VAE has no {name}: the Wan2.2 VAE kernels decode the TI2V-5B VAE only")
        w = params[name].detach()
        if tuple(w.shape) != shape:
            raise ValueError(f"{name} is {tuple(w.shape)}, the Wan2.2-TI2V-5B VAE's is {shape}")
        weights.append(w.to(device=device, dtype=torch.bfloat16).clone(memory_format=torch.contiguous_format))
    return weights


class Wan22VaeKernel:
    """The Wan2.2-TI2V-5B VAE's decoder with its weights prepared once (layouts, Winograd and phase transforms).

    ``decode(z)`` is ``vae.decode(z)``'s video for raw latents ``z`` ``[1, 48, T, h, w]`` (de-normalized, as
    ``vae.decode`` takes them): ``[1, 3, 4T - 3, 16h, 16w]`` bf16 in [-1, 1]. ``stream(h, w)`` returns a decoder whose
    ``step`` calls continue one clip, so a clip decodes in pieces (``stream_decode``) with the whole decode's frames.
    """

    def __init__(self, vae: nn.Module, device: torch.device, frames_per_pass: int = 4):
        reason = unsupported_reason(device, vae)
        if reason is not None:
            raise RuntimeError(f"Wan2.2 VAE kernels unavailable: {reason}")
        if frames_per_pass < 1:
            raise ValueError(f"frames_per_pass must be at least 1, got {frames_per_pass}")
        self.device = torch.device(device)
        self.frames_per_pass = int(frames_per_pass)
        self._ext = extension(_arch(self.device))
        with torch.cuda.device(self.device), torch.inference_mode(False), torch.no_grad():
            self._weights = self._ext.prepare_local(decoder_weights(vae, self.device))

    @staticmethod
    def _latents(z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 5 or z.shape[0] != 1 or z.shape[1] != LATENT_CHANNELS:
            raise ValueError(f"latents {tuple(z.shape)}: expected [1, {LATENT_CHANNELS}, T, h, w]")
        return z.to(dtype=torch.bfloat16).contiguous()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self._latents(z)
        _, _, t, h, w = z.shape
        out = torch.empty((1, 3, 4 * t - 3, 16 * h, 16 * w), dtype=torch.bfloat16, device=z.device)
        self._ext.decode_local(z, self._weights, out, self.frames_per_pass)
        return out

    def stream(self, h: int, w: int) -> "KernelStream":
        return KernelStream(self._ext.StreamDecoder(self._weights, h, w))


class KernelStream:
    """One clip's decoder state between pieces: ``step(z)`` decodes the next latent frames of the clip."""

    def __init__(self, handle):
        self._handle = handle
        self._started = False

    def step(self, z: torch.Tensor) -> torch.Tensor:
        z = Wan22VaeKernel._latents(z)
        _, _, t, h, w = z.shape
        frames = 4 * t if self._started else 4 * t - 3
        out = torch.empty((1, 3, frames, 16 * h, 16 * w), dtype=torch.bfloat16, device=z.device)
        self._handle.step(z, out)
        self._started = True
        return out
