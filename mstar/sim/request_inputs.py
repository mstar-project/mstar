"""Synthetic request inputs for the simulator.

A real request arrives as tensors: the api server tokenizes the prompt and
decodes any media, and the conductor hands the model a dict of
``TensorPointerInfo`` describing them. Models read that dict to decide what
to do — Bagel builds its prefill schedule from which modalities are present
and sizes the image it will generate from the input image's dims; Qwen3-Omni
routes to a vision or audio prefill walk depending on what arrived.

The simulator has no tensors, but it still has to answer those questions, so
it fabricates *descriptors*: correct names, correct shapes, no data. That is
enough because the transition functions read dims and modality names, never
values.

Shapes come from the workload spec, so "a 512×512 image in, 1024×1024 out"
is something you state rather than something the simulator guesses.
"""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field

from mstar.graph.base import TensorPointerInfo

#: Bytes per element, by dtype name. Only used to fill ``nbytes``, which
#: feeds the transfer model.
_DTYPE_BYTES = {
    "int64": 8, "int32": 4, "float32": 4, "bfloat16": 2, "float16": 2, "uint8": 1,
}


def _ptr(dims: list[int], dtype: str = "float32") -> TensorPointerInfo:
    n = 1
    for d in dims:
        n *= max(1, d)
    return TensorPointerInfo(
        dims=list(dims),
        dtype=dtype,
        nbytes=n * _DTYPE_BYTES.get(dtype, 4),
        address=0,
        stride=[1] * len(dims),
        uuid=str(_uuid.uuid4()),
        source_session_id="sim",
        source_entity="api_server",
    )


@dataclass
class InputSpec:
    """What a simulated request carries in, and what it asks for.

    Defaults describe a plain text-in/text-out request. Anything richer —
    an image to caption, a video to roll out, an audio clip to transcribe —
    is stated explicitly, because the model's own transition logic branches
    on it and guessing would silently simulate a different request than the
    one you meant.
    """

    input_modalities: list[str] = field(default_factory=lambda: ["text"])
    output_modalities: list[str] = field(default_factory=lambda: ["text"])

    prompt_tokens: int = 64
    #: Generated length in autoregressive steps (see the README on why this
    #: differs from client-visible chunks for codec models).
    output_tokens: int = 128

    #: (height, width) for image inputs and generated images.
    image_size: tuple[int, int] = (1024, 1024)
    #: Number of input images.
    num_images: int = 1
    #: Audio input length in samples (16 kHz assumed by most encoders).
    audio_samples: int = 16000 * 5
    #: Video input: frames × (height, width).
    video_frames: int = 16
    video_size: tuple[int, int] = (256, 256)
    #: Robot state/action dims for VLA policies.
    state_dim: int = 32
    #: Action-trajectory length for action-conditioned rollouts.
    action_horizon: int = 4

    #: Passed verbatim to the model's transition functions, exactly as the
    #: api server passes a request's ``model_kwargs``.
    model_kwargs: dict = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"{'+'.join(self.input_modalities)} → {'+'.join(self.output_modalities)}, "
            f"prompt~{self.prompt_tokens} tok, output~{self.output_tokens} steps"
        )


def build_input_signals(spec: InputSpec) -> dict[str, list[TensorPointerInfo]]:
    r"""Descriptors for the tensors ``process_prompt`` would have produced.

    The *names* here are not arbitrary — models look up specific keys, and a
    key they cannot find reads to them as "that modality was not supplied".
    The list below is every key the shipped models actually read from
    ``input_signals`` (``rg 'input_signals(\.get\(|\[)"' mstar/model/``),
    grouped by the modality that produces it. Several names per modality is
    deliberate: Whisper wants ``audio_features``, Qwen3-Omni also wants
    ``audio_seqlens``, V-JEPA wants ``video_frames`` where Qwen wants
    ``pixel_values_videos``.

    Calling the real ``process_prompt`` would avoid the table, but it needs a
    tokenizer and decoded media — exactly the things the simulator exists to
    do without. When a new model reads a key that isn't here, its transition
    function will route as though that input were missing; add the name.
    """
    sig: dict[str, list[TensorPointerInfo]] = {}
    mods = set(spec.input_modalities)

    if "text" in mods:
        sig["text_inputs"] = [_ptr([spec.prompt_tokens], "int64")]

    if "image" in mods:
        h, w = spec.image_size
        n = max(1, spec.num_images)
        images = [_ptr([3, h, w]) for _ in range(n)]
        sig["image_inputs"] = images
        sig["pixel_values"] = images
        # (t, h, w) in patch units — what the processor reports alongside.
        sig["image_grid_thw"] = [_ptr([n, 3], "int64")]

    if "audio" in mods:
        # Mel features, not raw samples: ~100 fps before the encoder's 2x
        # conv stack, which is the shape encoders are handed.
        frames = max(1, spec.audio_samples // 160)
        sig["audio_features"] = [_ptr([128, frames])]
        sig["audio_seqlens"] = [_ptr([1], "int64")]
        sig["audio_feature_lens"] = [_ptr([1], "int64")]

    if "video" in mods:
        f, (h, w) = spec.video_frames, spec.video_size
        frames = [_ptr([f, 3, h, w])]
        sig["video_frames"] = frames
        sig["video_inputs"] = frames
        sig["pixel_values_videos"] = frames
        sig["video_grid_thw"] = [_ptr([1, 3], "int64")]
        sig["video_second_per_grid"] = [_ptr([1])]

    if "state" in mods or "action" in mods:
        # Robot policies read a proprioceptive state and, for action-
        # conditioned rollouts, a per-timestep action trajectory.
        sig["states"] = [_ptr([1, spec.state_dim])]
        sig["actions"] = [_ptr([spec.action_horizon, spec.state_dim])]
        sig["state_inputs"] = sig["states"]

    return sig


def token_count_for(spec: InputSpec) -> int:
    """How many tokens the prompt is worth to an AR backbone.

    Media contributes tokens too — a vision encoder turns an image into
    patch embeddings the backbone then attends over — so a caption request
    prefills far more than its text length suggests. The per-patch counts
    here are coarse (a 16×16 patch grid is the common case); a model whose
    tokenizer disagrees should have its prompt length stated directly.
    """
    total = spec.prompt_tokens if "text" in spec.input_modalities else 0
    h, w = spec.image_size
    if "image" in spec.input_modalities:
        total += spec.num_images * (h // 16) * (w // 16)
    if "video" in spec.input_modalities:
        vh, vw = spec.video_size
        total += spec.video_frames * (vh // 16) * (vw // 16)
    if "audio" in spec.input_modalities:
        # ~50 frames/s after a 2× conv stack on 100 fps mel features.
        total += spec.audio_samples // 320
    return max(1, total)
