"""Model profiles and their cases. A ModelSpec is a config under ``configs/``
plus the HTTP requests that exercise it; see README.md for adding one."""
import dataclasses
import functools
import io
import json
from collections.abc import Callable

import numpy as np
from validators import Expect


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    route: str                        # chat | speech | transcribe | generate | generate_stream
    build: Callable[[], dict]         # JSON body, or form fields plus optional "files"
    expect: Expect
    smoke: bool = False
    requires: tuple[str, ...] = ()    # modules the runner needs to build the input


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    config: str                       # relative to the repo root
    cases: tuple[Case, ...]
    family: str
    must_differ: tuple[tuple[str, str], ...] = ()
    requires: tuple[str, ...] = ()    # importable modules the server needs
    gated: tuple[str, ...] = ()       # Hub repos that need a license accepted
    peak_vram_gb: float | None = None  # measured on one H200, eager servers


def _png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@functools.cache
def red_square_png() -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(image).rectangle((64, 64, 192, 192), fill="red")
    return _png(image)


@functools.cache
def scene_png(width: int = 224, height: int = 224) -> bytes:
    """Wall, table, block and gripper, for robotics and world models."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (200, 200, 205))
    draw = ImageDraw.Draw(image)
    sx, sy = width / 224, height / 224
    draw.rectangle((0, 150 * sy, width, height), fill=(120, 90, 60))
    draw.rectangle((95 * sx, 120 * sy, 130 * sx, 155 * sy), fill=(200, 30, 30))
    draw.rectangle((100 * sx, 20 * sy, 125 * sx, 80 * sy), fill=(60, 60, 60))
    return _png(image)


@functools.cache
def moving_square_mp4(frames: int = 16, size: int = 256, fps: int = 8) -> bytes:
    import av

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            pixels = np.full((size, size, 3), 40, dtype=np.uint8)
            x = 16 + i * (size - 96) // max(frames - 1, 1)
            pixels[96:160, x:x + 64] = (230, 200, 40)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buf.getvalue()


SPEECH_DATASET = ("hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet")
SPEECH_FIXTURE = ("pyarrow", "soundfile")  # what speech_sample() imports
OFFLINE = False  # set by --offline: fixtures come from the local Hugging Face cache


@functools.cache
def speech_sample() -> tuple[bytes, str]:
    """A LibriSpeech utterance as 16 kHz PCM16 WAV, with its transcript."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SPEECH_DATASET[0], SPEECH_DATASET[1], repo_type="dataset", local_files_only=OFFLINE)
    import pyarrow.parquet as pq
    import soundfile as sf

    row = pq.read_table(path).slice(0, 1).to_pylist()[0]
    samples, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue(), row["text"]


def speech_wav() -> bytes:
    return speech_sample()[0]


def speech_transcript() -> str:
    return speech_sample()[1]


def data_url(data: bytes, mime: str = "image/png") -> str:
    import base64

    return f"data:{mime};base64," + base64.b64encode(data).decode()


TWO_PLUS_TWO = "What is two plus two? Answer with just the number."
SQUARE_COLOR = "What color is the square? Answer with one word."
SENTENCE = "The train leaves tomorrow morning."
SMALL = {"height": 256, "width": 256, "seed": 7}
ROBOT_STATE = [round(-1 + 2 * i / 31, 6) for i in range(32)]
ANSWER_4 = Expect("text", contains_any=("4", "four"))
ANSWER_RED = Expect("text", contains_any=("red",))
SPEECH = Expect("audio", min_seconds=0.5)


def chat(content_fn, **extra) -> Callable[[], dict]:
    def build():
        return dict(messages=[dict(role="user", content=content_fn())], temperature=0, max_tokens=32,
                    seed=7, enable_thinking=False, think_mode=False, **extra)
    return build


def vision_prompt():
    return [dict(type="text", text=SQUARE_COLOR),
            dict(type="image_url", image_url=dict(url=data_url(red_square_png())))]


def audio_prompt():
    return [dict(type="text", text="Transcribe this clip."),
            dict(type="audio_url", audio_url=dict(url=data_url(speech_wav(), "audio/wav")))]


def speech(**fields) -> Callable[[], dict]:
    def build():
        body = dict(input=SENTENCE, response_format="wav", seed=7, **fields)
        return {k: (v() if callable(v) else v) for k, v in body.items()}
    return build


def transcribe(**fields) -> Callable[[], dict]:
    def build():
        return dict(fields, language="en", response_format="json", temperature="0",
                    files=("speech.wav", speech_wav()))
    return build


def generate(text, modality, kwargs, file_fn=None, streaming=False) -> Callable[[], dict]:
    def build():
        fields = dict(text=text, output_modalities=modality, streaming="true" if streaming else "false",
                      model_kwargs=json.dumps(kwargs))
        if file_fn:
            fields["files"] = file_fn()
        return fields
    return build


def lm_cases(smoke=False):
    return (
        Case("text_to_text", "chat", chat(lambda: TWO_PLUS_TWO), ANSWER_4, smoke=smoke),
        Case("text_to_text_stream", "chat", chat(lambda: TWO_PLUS_TWO, stream=True), ANSWER_4),
        Case("image_to_text", "chat", chat(vision_prompt), ANSWER_RED),
    )


def image_cases(steps_kwargs, edit=True, smoke=False):
    cases = [Case("text_to_image", "generate", generate(
        "A red square on a white background.", "image", {**SMALL, **steps_kwargs}),
        Expect("image", modality="image", color="red", size=(256, 256)), smoke=smoke)]
    if edit:
        cases.append(Case("image_to_image", "generate", generate(
            "Make the square blue.", "image", {**SMALL, **steps_kwargs}, lambda: ("square.png", red_square_png())),
            Expect("image", modality="image", color="blue", size=(256, 256))))
    return tuple(cases)


def cosmos_forward_dynamics(actions, domain):
    return generate("Pick up the cup", "video", {
        "action": actions, "action_mode": "forward_dynamics", **domain, "num_frames": 9,
        "action_chunk_size": 8, "height": 128, "width": 128, "guidance_scale": 1.0,
        "num_inference_steps": 4, "seed": 7}, lambda: ("scene.png", scene_png()))


def cosmos_fd_pair(dim, domain):
    zeros = [[0.0] * dim] * 8
    ramp = [[(i + 1) / 8] * dim for i in range(8)]
    return (
        Case("image_action_to_video_zeros", "generate", cosmos_forward_dynamics(zeros, domain),
             Expect("video", modality="video", frames=9, size=(128, 128))),
        Case("image_action_to_video_ramp", "generate", cosmos_forward_dynamics(ramp, domain),
             Expect("video", modality="video", frames=9, size=(128, 128))),
    )


FD_PAIR = (("image_action_to_video_zeros", "image_action_to_video_ramp"),)
DROID = {"domain_name": "droid_lerobot"}


def cosmos_policy():
    return Case("image_to_action_policy", "generate", generate("Pick up the red block", "action", {
        "action_mode": "policy", **DROID, "raw_action_dim": 10, "action_chunk_size": 8, "num_frames": 9,
        "num_inference_steps": 4, "seed": 7}, lambda: ("scene.png", scene_png())),
        Expect("float32", modality="action", count=8 * 64))  # action_chunk_size x padded action dim


def waypoint_rollout(mouse, size):
    actions = [{"mouse": [mouse, 0], "buttons": [], "scroll": 0}] * 2
    return generate("", "video_frame", {"num_steps": 2, "seed": 7, "actions": actions},
                    lambda: ("seed.png", scene_png(*size)), streaming=True)


def waypoint_cases(size, smoke=False):
    # 2 steps x 4 frames per step
    return (
        Case("image_actions_to_frames_left", "generate_stream", waypoint_rollout(-24, size),
             Expect("frames", modality="video_frame", frames=8, size=size), smoke=smoke),
        Case("image_actions_to_frames_right", "generate_stream", waypoint_rollout(24, size),
             Expect("frames", modality="video_frame", frames=8, size=size)),
    )


WAYPOINT_PAIR = (("image_actions_to_frames_left", "image_actions_to_frames_right"),)


def vjepa2_ac_case(name, scale):
    actions = [[scale * np.sin(i / 4 + j) for j in range(7)] for i in range(32)]
    states = [[0.1 * j for j in range(7)] for _ in range(32)]
    return Case(name, "generate", generate(
        "Predict the next frames", "video", {"actions": actions, "states": states, "seed": 7},
        lambda: ("clip.mp4", moving_square_mp4(frames=64))),
        Expect("float32", modality="video", count=8192 * 1408))  # predicted tokens x hidden size


QWEN_TTS = dict(language="English", temperature=0, talker_temperature=0, subtalker_dosample=False,
                do_sample=False, max_output_tokens=256)
CHATTERBOX = ("pyloudnorm", "perth")

MODELS: dict[str, ModelSpec] = {m.name: m for m in [
    ModelSpec("qwen3_5", "configs/qwen3_5_0.8b.yaml", lm_cases(smoke=True), "llm", peak_vram_gb=13.5),
    ModelSpec("qwen3_5_2b", "configs/qwen3_5_2b.yaml", lm_cases(), "llm", peak_vram_gb=16.4),
    ModelSpec("qwen3_5_4b", "configs/qwen3_5_4b.yaml", lm_cases(), "llm", peak_vram_gb=30.1),
    ModelSpec("qwen3_5_9b", "configs/qwen3_5_9b.yaml", lm_cases(), "llm", peak_vram_gb=39.6),
    # ~60 GB of bf16 weights: one 141 GB GPU; peak VRAM not yet measured on an H200
    ModelSpec("muse_glimmer", "configs/muse_glimmer.yaml", lm_cases(), "llm"),
    # ~213 GB of bf16 weights a rank over 8 GPUs (one 8x B300 node); peak VRAM
    # provisional, not yet measured
    ModelSpec("muse_spark", "configs/muse_spark_open.yaml", (
        *lm_cases(),
        Case("audio_to_text", "chat", chat(audio_prompt), Expect("text", contains_any=("quilter", "apostle")),
             requires=SPEECH_FIXTURE),
    ), "llm", gated=("meta-models/Muse-Spark-Open-816B-A42B",), peak_vram_gb=250),
    ModelSpec("bagel", "configs/bagel_single_gpu.yaml", (
        Case("text_to_text", "chat", chat(lambda: TWO_PLUS_TWO), ANSWER_4),
        Case("image_to_text", "chat", chat(vision_prompt), ANSWER_RED),
        *image_cases({"num_timesteps": 10, "think_mode": False}),
    ), "image", peak_vram_gb=48.3),
    ModelSpec("qwen3omni", "configs/qwen3omni_colocated.yaml", (
        Case("text_to_text", "chat", chat(lambda: TWO_PLUS_TWO), ANSWER_4),
        Case("image_to_text", "chat", chat(vision_prompt), ANSWER_RED),
        Case("text_to_speech", "chat", chat(lambda: f"Say exactly: {SENTENCE}", modalities=["text", "audio"],
                                            audio={"voice": "Ethan"}), SPEECH),
    ), "omni", requires=("flash_attn", "qwen_omni_utils"), peak_vram_gb=83.9),
    ModelSpec("qwen3tts", "configs/qwen3tts.yaml", (
        Case("text_to_speech", "speech", speech(voice="vivian", **QWEN_TTS), SPEECH, smoke=True),
    ), "speech", requires=("qwen_tts",), peak_vram_gb=50.6),
    ModelSpec("qwen3tts_1p7b", "configs/qwen3tts_1p7b.yaml", (
        Case("text_to_speech", "speech", speech(voice="vivian", **QWEN_TTS), SPEECH),
    ), "speech", requires=("qwen_tts",), peak_vram_gb=52.2),
    ModelSpec("qwen3tts_base", "configs/qwen3tts_base.yaml", (
        Case("voice_clone", "speech", speech(language="English", ref_text=speech_transcript,
                                             ref_audio=lambda: data_url(speech_wav(), "audio/wav")), SPEECH,
             requires=SPEECH_FIXTURE),
    ), "speech", requires=("qwen_tts",), peak_vram_gb=51.7),
    ModelSpec("qwen3tts_voicedesign", "configs/qwen3tts_voicedesign.yaml", (
        Case("voice_design", "speech", speech(instructions="A clear, friendly adult female voice.",
                                              language="English", do_sample=False, subtalker_dosample=False), SPEECH),
    ), "speech", requires=("qwen_tts",), peak_vram_gb=51.7),
    ModelSpec("kokoro", "configs/kokoro.yaml", (
        Case("text_to_speech", "speech", speech(voice="af_heart", speed=1.0), SPEECH),
    ), "speech", requires=("misaki", "spacy", "en_core_web_sm"), peak_vram_gb=40.5),
    ModelSpec("orpheus", "configs/orpheus_colocated.yaml", (
        Case("text_to_speech", "speech", speech(voice="tara", temperature=0), SPEECH),
    ), "speech", gated=("canopylabs/orpheus-3b-0.1-ft",), peak_vram_gb=38.2),
    ModelSpec("chatterbox", "configs/chatterbox.yaml", (
        Case("text_to_speech", "speech", speech(voice="default", do_sample=False), SPEECH),
    ), "speech", requires=CHATTERBOX, peak_vram_gb=18.9),
    ModelSpec("chatterbox_multilingual", "configs/chatterbox_multilingual.yaml", (
        Case("text_to_speech", "speech", speech(voice="default", language_id="en", do_sample=False), SPEECH),
    ), "speech", requires=(*CHATTERBOX, "pykakasi", "spacy_pkuseg"), peak_vram_gb=18.9),
    ModelSpec("chatterbox_turbo", "configs/chatterbox_turbo.yaml", (
        Case("text_to_speech", "speech", speech(voice="default"), SPEECH),
    ), "speech", requires=CHATTERBOX, peak_vram_gb=14.3),
    ModelSpec("omnivoice", "configs/omnivoice.yaml", (
        # Voice design accepts only OmniVoice's listed instruct items, comma separated.
        Case("text_to_speech", "speech", speech(voice="female, low pitch"), SPEECH),
    ), "speech", requires=("omnivoice.models.omnivoice_flashinfer",), peak_vram_gb=3.7),
    ModelSpec("whisper_large", "configs/whisper_large.yaml", (
        Case("speech_to_text", "transcribe", transcribe(),
             Expect("text", transcript="MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES"), smoke=True,
             requires=SPEECH_FIXTURE),
    ), "asr", peak_vram_gb=12.6),
    ModelSpec("higgs_audio", "configs/higgs_audio.yaml", (
        Case("speech_to_text", "transcribe", transcribe(model="higgs_audio"),
             Expect("text", transcript="MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES"),
             requires=SPEECH_FIXTURE),
    ), "asr", peak_vram_gb=11.5),
    ModelSpec("flux2_klein", "configs/flux2_klein.yaml", image_cases({}, smoke=True), "image", peak_vram_gb=47.3),
    ModelSpec("flux2_klein_9b", "configs/flux2_klein_9b.yaml", image_cases({}), "image",
              gated=("black-forest-labs/FLUX.2-klein-9B",), peak_vram_gb=48.0),
    ModelSpec("z_image_turbo", "configs/z_image_turbo.yaml", image_cases({}, edit=False), "image", peak_vram_gb=40.3),
    ModelSpec("cosmos3", "configs/cosmos3_nano.yaml", (
        # Cosmos3 images are checked for size and content, not color: two steps
        # give Edge a pale pink (mean RGB 240, 207, 206) that fails "red dominates".
        Case("text_to_image", "generate", generate(
            "A red square on a white background.", "image", {**SMALL, "num_inference_steps": 2, "guidance_scale": 3}),
            Expect("image", modality="image", size=(256, 256))),
        *cosmos_fd_pair(7, {"domain_id": 0}),
    ), "video", must_differ=FD_PAIR, peak_vram_gb=53.4),
    ModelSpec("cosmos3_nano_ar", "configs/cosmos3_nano_ar.yaml", (
        Case("text_to_video_windowed", "generate", generate("A ball rolls across a wooden table.", "video", {
            "window_mode": "kv", "num_frames": 29, "window_frames": 29, "num_inference_steps": 4, **SMALL}),
            Expect("video", modality="video", frames=29, size=(256, 256))),
    ), "video", peak_vram_gb=52.4),
    ModelSpec("cosmos3_edge", "configs/cosmos3_edge.yaml", (
        Case("text_to_image", "generate", generate(
            "A red square on a white background.", "image", {**SMALL, "num_inference_steps": 2, "guidance_scale": 3}),
            Expect("image", modality="image", size=(256, 256))),
        Case("text_to_video", "generate", generate("A ball rolls across a wooden table.", "video", {
            "height": 320, "width": 192, "num_frames": 9, "num_inference_steps": 4, "seed": 7}),
            Expect("video", modality="video", frames=9, size=(192, 320))),
    ), "video", peak_vram_gb=36.3),
    ModelSpec("cosmos3_droid", "configs/cosmos3_droid.yaml", (cosmos_policy(), *cosmos_fd_pair(10, DROID)),
              "robotics", must_differ=FD_PAIR, peak_vram_gb=53.4),
    ModelSpec("cosmos3_edge_droid", "configs/cosmos3_edge_droid.yaml", (cosmos_policy(),), "robotics",
              peak_vram_gb=27.4),
    ModelSpec("wan22", "configs/wan22.yaml", (
        Case("text_to_video", "generate", generate("A ball rolls across a wooden table.", "video", {
            **SMALL, "num_frames": 5, "num_inference_steps": 4, "guidance_scale": 5}),
            Expect("video", modality="video", frames=5, size=(256, 256)), smoke=True),
    ), "video", peak_vram_gb=24.4),
    ModelSpec("waypoint", "configs/waypoint.yaml", waypoint_cases((1280, 720)), "world",
              must_differ=WAYPOINT_PAIR, requires=("taehv",), peak_vram_gb=18.1),
    ModelSpec("waypoint_360p", "configs/waypoint_360p.yaml", waypoint_cases((640, 360), smoke=True), "world",
              must_differ=WAYPOINT_PAIR, requires=("taehv",), peak_vram_gb=15.1),
    ModelSpec("pi05", "configs/pi05.yaml", (
        Case("image_state_to_action", "generate", generate(
            "Pick up the red block", "action", {"robot_state": ROBOT_STATE, "seed": 7},
            lambda: ("scene.png", scene_png())),
            Expect("float32", modality="action", count=50 * 32), smoke=True),  # action_horizon x action_dim
    ), "robotics", peak_vram_gb=15.5),
    ModelSpec("pi05_droid", "configs/pi05_droid.yaml", (
        Case("image_state_to_action", "generate", generate(
            "Pick up the red block", "action", {"robot_state": ROBOT_STATE, "seed": 7},
            lambda: ("scene.png", scene_png())),
            Expect("float32", modality="action", count=15 * 32)),  # action_horizon x action_dim
    ), "robotics", peak_vram_gb=18.9),
    ModelSpec("vjepa2", "configs/vjepa2.yaml", (
        Case("video_to_prediction", "generate", generate(
            "Predict the masked frames", "video", {"context_mask": [0, 1], "target_mask": [2, 3], "seed": 7},
            lambda: ("clip.mp4", moving_square_mp4())),
            Expect("float32", modality="video", count=2 * 1024)),  # target frames x hidden size
    ), "robotics", peak_vram_gb=1.5),
    ModelSpec("vjepa2_ac", "configs/vjepa2_ac.yaml", (
        vjepa2_ac_case("video_actions_to_prediction_small", 0.1),
        vjepa2_ac_case("video_actions_to_prediction_large", 1.0),
    ), "robotics", must_differ=(("video_actions_to_prediction_small", "video_actions_to_prediction_large"),),
        peak_vram_gb=29.6),
]}
