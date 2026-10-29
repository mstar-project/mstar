"""Table rows (a model and its GPU layout) and engine arms.

Competitor settings are the vendors' documented AMD recipes; ``Arm.recipe`` says where each one
comes from. mstar runs its deployment config unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

CONTAINERS = "/work1/stephaniewang/atindra/amd/containers"
REFERENCES = "/work1/stephaniewang/atindra/amd/results/correctness/data/references"
# Deploy overlays for vllm-omni (the files inherit the model's deploy YAML inside the image).
VLLM_OMNI_DEPLOY = Path(__file__).resolve().parent / "vllm_omni"
# Packages the correctness runner installs outside the serving venv (kokoro, snac).
REFENV = "/work1/stephaniewang/atindra/amd/results/correctness/data/refenv"


@dataclass(frozen=True)
class Row:
    """One model served on ``gpus`` GPUs. ``name`` is also the mstar config name, unless ``mstar_configs``
    names another one for an arm."""

    name: str
    family: str  # text, tts, asr, omni, image or video
    hf_id: str
    runner_model: str  # benchmark.runner --model
    gpus: int = 1
    tp: int = 1
    # Context length every engine is given (mstar's max_seq_len), so KV sizing starts from the same place.
    max_len: int | None = None
    runner_model_id: bool = False  # pass --model-id (the runner's Qwen3.5 class serves several sizes)
    # Flags and environment this model needs, keyed by arm name or engine name (the arm's entry wins).
    engine_args: dict = field(default_factory=dict)
    engine_env: dict = field(default_factory=dict)
    cost: float = 1.0  # rough relative run time, so the sweep starts the long rows first
    # How long the launcher waits for /health when a cold start of this model runs past its default.
    startup_timeout: int | None = None
    # mstar's fused-MoE layer sizes, as config file names start ("E=..,N=..,K=.."). The table shows
    # whether each one loaded a tuned table, since a missing one silently runs the default launch config.
    moe_shapes: tuple[str, ...] = ()
    # Deployment configs (configs/<name>.yaml) of mstar arms that serve this row with another one, by arm name.
    mstar_configs: dict = field(default_factory=dict)
    # Media rows: the device the reference draws its noise on, sent to engines that take ``generator_device`` so
    # their outputs can match the reference pixel for pixel ("cpu" for diffusers' FLUX, Z-Image and Wan, "cuda" for
    # Cosmos3).
    generator_device: str | None = None
    # Workloads of the table this row runs (empty: every workload of its family), and per-workload concurrency
    # where the row's differs from the workload's.
    workloads: tuple[str, ...] = ()
    concurrency: dict = field(default_factory=dict)
    # A directory of first frames (<item id>.png) that image-to-video requests send instead of the set's own, so
    # every arm conditions on the frames the reference used.
    i2v_frames: str | None = None
    # Options of the row's correctness family (its mode, a pinned size), which also set the requests' settings.
    correctness_opts: dict = field(default_factory=dict)


def mstar_config_name(row: Row, arm: str) -> str:
    return row.mstar_configs.get(arm, row.name)


def family_config_name(row: Row, arm: str) -> str:
    """The mstar config whose correctness family scores an arm on this row: the arm's own for mstar arms, the mstar
    arm's for every other engine."""
    return mstar_config_name(row, arm if arm in ARMS and ARMS[arm].engine == "mstar" else "mstar")


def _qwen(size: str, gpus: int = 1, tp: int = 1, name: str | None = None) -> Row:
    return Row(name or f"qwen3_5_{size.lower()}", "text", f"Qwen/Qwen3.5-{size}", "qwen3.5", gpus=gpus, tp=tp,
               max_len=32768, runner_model_id=True, cost=1 + float(size.rstrip("B")) / 4)


QWEN3_OMNI_MOE = ("E=128,N=1536,K=2048", "E=128,N=768,K=1024")  # Thinker, Talker


def _qwen3tts(name: str, size: str, runner_model: str, sglang_omni_config: str, **kw) -> Row:
    # sglang-omni's pipeline configs are paths relative to its checkout (the arm runs from there).
    tuned = str(VLLM_OMNI_DEPLOY / "qwen3_tts_rocm_tuned.yaml")
    return Row(name, "tts", f"Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice", runner_model,
               engine_args={"sglang-omni": ("--config", f"examples/configs/{sglang_omni_config}"),
                            "vllm-omni-tuned": ("--deploy-config", tuned)}, **kw)


ROWS: dict[str, Row] = {r.name: r for r in [
    _qwen("0.8B"),
    _qwen("2B"),
    _qwen("4B"),
    _qwen("9B"),
    _qwen("9B", gpus=2, tp=2, name="qwen3_5_9b_tp2"),
    Row("kokoro", "tts", "hexgrad/Kokoro-82M", "kokoro"),
    Row("orpheus", "tts", "canopylabs/orpheus-3b-0.1-ft", "orpheus", gpus=2),
    Row("orpheus_colocated", "tts", "canopylabs/orpheus-3b-0.1-ft", "orpheus"),
    _qwen3tts("qwen3tts", "0.6B", "qwen3_tts", "qwen3_tts_0_6b_customvoice.yaml"),
    # mstar serves the codec on its own worker in bf16; mstar-colocated keeps the single-worker float32 config.
    _qwen3tts("qwen3tts_1p7b", "1.7B", "qwen3_tts_1p7b", "qwen3_tts_1_7b_customvoice.yaml",
              mstar_configs={"mstar": "qwen3tts_1p7b_split_bf16", "mstar-colocated": "qwen3tts_1p7b"}),
    # Qwen3-Omni speaks its answers. vllm-omni's own layout is two GPUs; the one-GPU row places its stages
    # on one device to sit next to mstar's colocated config. mstar's cold start ran past 2400 s on an MI300X
    # slice with an empty MIOpen DB (correctness campaign), hence the longer wait.
    Row("qwen3omni_colocated", "omni", "Qwen/Qwen3-Omni-30B-A3B-Instruct", "qwen3omni",
        engine_args={"vllm-omni": ("--deploy-config", str(VLLM_OMNI_DEPLOY / "qwen3_omni_one_gpu.yaml")),
                     "vllm-omni-tuned": ("--deploy-config",
                                         str(VLLM_OMNI_DEPLOY / "qwen3_omni_one_gpu_rocm_tuned.yaml"))},
        engine_env={"vllm-omni": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}}, cost=3.0, startup_timeout=5400,
        moe_shapes=QWEN3_OMNI_MOE),
    Row("qwen3omni_2gpu", "omni", "Qwen/Qwen3-Omni-30B-A3B-Instruct", "qwen3omni", gpus=2,
        engine_args={"vllm-omni-tuned": ("--deploy-config", str(VLLM_OMNI_DEPLOY / "qwen3_omni_rocm_tuned.yaml"))},
        engine_env={"vllm-omni": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}}, cost=3.0, startup_timeout=5400,
        moe_shapes=QWEN3_OMNI_MOE),
    Row("whisper_large", "asr", "openai/whisper-large-v3", "whisper_large", max_len=448,
        mstar_configs={"mstar-split": "whisper_large_split"}),
    Row("flux2_klein", "image", "black-forest-labs/FLUX.2-klein-4B", "flux2_klein", generator_device="cpu"),
    Row("flux2_klein_9b", "image", "black-forest-labs/FLUX.2-klein-9B", "flux2_klein_9b", generator_device="cpu"),
    Row("z_image_turbo", "image", "Tongyi-MAI/Z-Image-Turbo", "z_image_turbo", generator_device="cpu"),
    # Cosmos3-Nano's text-to-image mode, every prompt once at 480x832 (the family's pinned size); mstar serves it
    # from its cosmos3_nano config with the family's request_extra.
    Row("cosmos3_nano_t2i", "image", "nvidia/Cosmos3-Nano", "cosmos3_nano", generator_device="cuda",
        mstar_configs={"mstar": "cosmos3_nano", "mstar-base": "cosmos3_nano"},
        correctness_opts={"mode": "t2i", "size": "480x832"},
        engine_args={"vllm-omni": ("--no-guardrails",), "vllm-omni-rc": ("--no-guardrails",)}),
    # mstar's wan22 takes only a frame of the request's size: i2v sends the reference's 480x832 conditioning frames.
    Row("wan22", "video", "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "wan22", cost=4.0, generator_device="cpu",
        concurrency={"video-t2v": (1, 2, 4)},
        i2v_frames=f"{REFERENCES}/diffusers-Wan-AI--Wan2.2-TI2V-5B-Diffusers-480x832x33-s20-i2v/video_v1/conditioning"),
    # vllm-omni's Cosmos3 recipe serves without the guardrail models (recipes/cosmos3/Cosmos3-Nano.md).
    # i2v sends the reference's 480x832 conditioning frames (diffusers' cover-crop of the 320x180 set frames):
    # engines bring a frame of another size to the request size their own way (vllm-omni's LANCZOS stretch, mstar's
    # and SGLang's crop), and one already at the request size passes through all of them unchanged.
    Row("cosmos3_nano", "video", "nvidia/Cosmos3-Nano", "cosmos3_nano", cost=4.0, generator_device="cuda",
        engine_args={"vllm-omni": ("--no-guardrails",), "vllm-omni-rc": ("--no-guardrails",)},
        i2v_frames=f"{REFERENCES}/diffusers-nvidia--Cosmos3-Nano-480x832x17-s35-i2v/video_v1/conditioning"),
    Row("cosmos3_edge", "video", "nvidia/Cosmos3-Edge", "cosmos3_edge", cost=3.0, generator_device="cuda",
        workloads=("video-t2v",),
        engine_args={"vllm-omni": ("--no-guardrails",), "vllm-omni-rc": ("--no-guardrails",)}),
]}


@dataclass(frozen=True)
class Arm:
    name: str
    engine: str  # mstar, vllm, sglang, vllm-omni, sglang-omni, reference
    families: tuple[str, ...]
    client: str = ""  # benchmark.runner --inference-system
    image: str | None = None  # ctr.sh image name
    env: dict = field(default_factory=dict)
    args: tuple[str, ...] = ()
    tp_env: dict = field(default_factory=dict)  # added when tp > 1
    tp_args: tuple[str, ...] = ()
    recipe: str = ""
    # Why the arm does not compute the same thing as mstar, for the table's "not comparable because" column,
    # for every family or per family.
    differs: str = ""
    differs_for: dict = field(default_factory=dict)
    models: tuple[str, ...] = ()  # rows (by runner model) this arm serves; empty means any row of its families
    rows: tuple[str, ...] = ()  # rows (by name) this arm serves, where one runner model covers several rows
    # The server takes the first ``server_gpus`` of the row's GPUs; the client gets the next one (or the
    # first if none is left) for work it does on a GPU, such as decoding audio.
    server_gpus: int | None = None
    client_gpu: bool = False
    # How long the launcher waits for /health when the engine's own startup limits run past its default.
    startup_timeout: int | None = None
    # Extra PYTHONPATH entries for the benchmark client (packages the client needs outside mstar's venv).
    client_path: tuple[str, ...] = ()
    # A named mstar tree from the sweep's --tree NAME=PATH instead of its --worktree; the sweep skips the arm when
    # that tree isn't given.
    tree: str = ""
    # Families whose requests to this engine carry the row's generator_device (where its protocol has the field).
    generator_device_for: tuple[str, ...] = ()
    # mstar arms: entries merged into the row's serving config's model_kwargs: section (model constructor args).
    model_kwargs: dict = field(default_factory=dict)
    # Runs right after the named arm on each row, outside the per-row rotation of arm order.
    follows: str = ""


SGLANG_NO_OFFLOAD = ("--dit-cpu-offload", "false", "--text-encoder-cpu-offload", "false", "--vae-cpu-offload", "false")
SGLANG_DIFFUSION_MODELS = ("flux2_klein", "flux2_klein_9b", "z_image_turbo", "wan22", "cosmos3_nano", "cosmos3_edge")
VLLM_GUIDE = ("AMD vLLM guide https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/"
              "vllm-optimization.html; vllm/platforms/rocm.py in the image")
SGLANG_COOKBOOK = "sglang v0.5.22 docs/cookbook/autoregressive/Qwen/Qwen3.5.mdx, AMD section"
SGLANG_CI = "sglang v0.5.22 test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py (AMD CI launch flags)"

ARMS: dict[str, Arm] = {a.name: a for a in [
    # GLOO_SOCKET_IFNAME: on these nodes the hostname resolves to an IPv6 link-local address and gloo's connect
    # between TP ranks times out; a host network setting, not a tuning knob.
    Arm("mstar", "mstar", ("text", "tts", "asr", "omni", "image", "video"), client="ours_openai",
        env={"MSTAR_SHM_ARENA": "1"}, tp_env={"GLOO_SOCKET_IFNAME": "lo"},
        recipe="the deployment config named in the engine column, unchanged; MSTAR_SHM_ARENA=1 as "
               ".claude/skills/benchmarking asks for numbers"),
    # TP rows only: the worker's async scheduling for lockstep TP nodes, which defaults to off
    # (MSTAR_TP_ASYNC_SCHED, mstar/worker/worker.py).
    Arm("mstar-tpasync", "mstar", ("text",), client="ours_openai", rows=("qwen3_5_9b_tp2",),
        env={"MSTAR_SHM_ARENA": "1", "MSTAR_TP_ASYNC_SCHED": "1"}, tp_env={"GLOO_SOCKET_IFNAME": "lo"},
        recipe="as the mstar arm, plus MSTAR_TP_ASYNC_SCHED=1 (async scheduling on the TP node)"),
    Arm("mstar-colocated", "mstar", ("tts",), client="ours_openai", rows=("qwen3tts_1p7b",),
        env={"MSTAR_SHM_ARENA": "1"}, recipe="as the mstar arm, on the single-worker config"),
    # The encoder and the decoder as two worker processes on one GPU (configs/whisper_large_split.yaml, its
    # mps: true is a no-op on AMD GPUs: the hardware scheduler runs both workers' queues side by side).
    Arm("mstar-split", "mstar", ("asr",), client="ours_openai", rows=("whisper_large",),
        env={"MSTAR_SHM_ARENA": "1"}, recipe="as the mstar arm, on the encoder/decoder split config"),
    # The Thinker's and Talker's routed experts on AITER's fused MoE instead of mstar's Triton kernels
    # (MoEBackend in mstar/utils/fused_moe/backend.py), right after the mstar arm on the same GPUs.
    Arm("mstar-aitermoe", "mstar", ("omni",), client="ours_openai", env={"MSTAR_SHM_ARENA": "1"},
        tp_env={"GLOO_SOCKET_IFNAME": "lo"}, model_kwargs={"moe_backend": "aiter"}, follows="mstar",
        recipe="as the mstar arm, with model_kwargs moe_backend: aiter (AITER's fused MoE for the Thinker and "
               "Talker experts; mstar's tuned Triton MoE tables don't apply)"),
    # The same arms on the base tree (--tree base=PATH), back to back with the head's on each row, so the head's
    # change is measured in the same session. Each serves its own tree's configs.
    Arm("mstar-base", "mstar", ("text", "tts", "asr", "omni", "image", "video"), client="ours_openai", tree="base",
        env={"MSTAR_SHM_ARENA": "1"}, tp_env={"GLOO_SOCKET_IFNAME": "lo"},
        recipe="the mstar arm on the base tree (its head in the engine column)"),
    Arm("mstar-base-tpasync", "mstar", ("text",), client="ours_openai", rows=("qwen3_5_9b_tp2",), tree="base",
        env={"MSTAR_SHM_ARENA": "1", "MSTAR_TP_ASYNC_SCHED": "1"}, tp_env={"GLOO_SOCKET_IFNAME": "lo"},
        recipe="the mstar-tpasync arm on the base tree"),
    Arm("vllm", "vllm", ("text",), client="vllm", image="vllm",
        recipe="image defaults (v0.31.0 picks ROCM_ATTN on gfx942, AITER off)"),
    Arm("vllm-aiter", "vllm", ("text",), client="vllm", image="vllm", env={"VLLM_ROCM_USE_AITER": "1"},
        args=("--attention-backend", "ROCM_AITER_FA"), tp_env={"NCCL_MIN_NCHANNELS": "112"},
        recipe=VLLM_GUIDE + ": VLLM_ROCM_USE_AITER=1, --attention-backend ROCM_AITER_FA, "
                            "NCCL_MIN_NCHANNELS=112 for TP"),
    Arm("vllm-aiter-unified", "vllm", ("text",), client="vllm", image="vllm", env={"VLLM_ROCM_USE_AITER": "1"},
        args=("--attention-backend", "ROCM_AITER_UNIFIED_ATTN"), tp_env={"NCCL_MIN_NCHANNELS": "112"},
        recipe=VLLM_GUIDE + ": the AITER unified attention backend, the same kernel mstar uses for paged attention"),
    Arm("vllm-whisper", "vllm", ("asr",), client="openai_transcribe", image="vllm",
        env={"VLLM_ROCM_USE_AITER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
             "PYTHONPATH": f"{CONTAINERS}/pyext/vllm-audio"},
        args=("--attention-backend", "TRITON_ATTN"),
        recipe="vLLM ROCm CI speech-to-text job; TRITON_ATTN because ROCM_ATTN and ROCM_AITER_FA are decoder-only",
        differs="eager: vLLM turns off CUDA graphs for encoder-decoder models on ROCm"),
    Arm("sglang", "sglang", ("text",), client="sglang", image="sglang",
        recipe="image defaults (SGLANG_USE_AITER=1 set by the image)"),
    Arm("sglang-aiter", "sglang", ("text",), client="sglang", image="sglang",
        args=("--attention-backend", "aiter", "--trust-remote-code"), recipe=SGLANG_CI),
    Arm("sglang-recipe", "sglang", ("text",), client="sglang", image="sglang",
        env={"SGLANG_USE_AITER": "1", "SGLANG_USE_AITER_UNIFIED_ATTN": "1", "SGLANG_MAMBA_SSM_DTYPE": "bfloat16"},
        args=("--attention-backend", "aiter", "--page-size", "16", "--mem-fraction-static", "0.8",
              "--trust-remote-code"),
        tp_args=("--enable-aiter-allreduce-fusion",), recipe=SGLANG_COOKBOOK,
        differs="bf16 GDN state (mstar and vLLM keep the checkpoint's fp32)"),
    Arm("sglang-whisper", "sglang", ("asr",), client="openai_transcribe", image="sglang",
        args=("--attention-backend", "aiter"),
        recipe="sglang v0.5.22: aiter attention named explicitly, since its auto-selector picks FlashInfer for "
               "Whisper (DESIGN.md section 11; research/competitors_matrix.md 2.2)"),
    Arm("sglang-diffusion", "sglang-diffusion", ("image", "video"), client="sglang_diffusion", image="sglang",
        models=SGLANG_DIFFUSION_MODELS, args=SGLANG_NO_OFFLOAD, generator_device_for=("image", "video"),
        recipe="SGLang Diffusion with every CPU offload off (DESIGN.md section 11; benchmark/wan22/reproduce.sh)"),
    # SGLang Diffusion's own speedups that keep the output: breakable CUDA graphs for the image DiTs (validated on
    # CUDA), with both image_v1 sizes warmed at startup, and torch.compile for video.
    Arm("sglang-diffusion-bcg", "sglang-diffusion", ("image",), client="sglang_diffusion", image="sglang",
        models=("flux2_klein", "flux2_klein_9b", "z_image_turbo"), generator_device_for=("image",),
        args=SGLANG_NO_OFFLOAD + ("--enable-breakable-cuda-graph", "--warmup-resolutions", "1024x1024", "1344x768"),
        recipe="as sglang-diffusion, plus --enable-breakable-cuda-graph and --warmup-resolutions for image_v1's sizes"),
    Arm("sglang-diffusion-compile", "sglang-diffusion", ("video",), client="sglang_diffusion", image="sglang",
        models=("wan22", "cosmos3_nano", "cosmos3_edge"), generator_device_for=("video",),
        args=SGLANG_NO_OFFLOAD + ("--enable-torch-compile",),
        recipe="as sglang-diffusion, plus --enable-torch-compile"),
    Arm("vllm-omni", "vllm-omni", ("tts", "omni", "image", "video"), client="vllm_omni", image="vllm-omni",
        models=("qwen3_tts", "qwen3_tts_1p7b", "qwen3omni", "flux2_klein", "flux2_klein_9b", "z_image_turbo", "wan22",
                "cosmos3_nano", "cosmos3_edge"),
        args=("--omni", "--init-timeout", "2700", "--stage-init-timeout", "2700"), generator_device_for=("image",),
        recipe="vllm-omni v0.30.0 defaults on ROCm: the model's deploy YAML with its platforms.rocm section "
               "(AR stages on TRITON_ATTN, Code2Wav eager)",
        differs_for={"tts": "Code2Wav runs eager on ROCm", "omni": "Code2Wav runs eager on ROCm"}),
    # v0.32.0rc1 for the Cosmos3 rows: its Cosmos3 loop keeps the sampling state in fp32 (v0.30.0's is bf16).
    Arm("vllm-omni-rc", "vllm-omni", ("image", "video"), client="vllm_omni", image="vllm-omni-rc",
        models=("cosmos3_nano", "cosmos3_edge"), generator_device_for=("image",),
        args=("--omni", "--init-timeout", "2700", "--stage-init-timeout", "2700"),
        recipe="vllm-omni v0.32.0rc1 (rocm/vllm-omni:da572cdd...) defaults on ROCm, as the vllm-omni arm"),
    Arm("vllm-omni-tuned", "vllm-omni", ("tts", "omni"), client="vllm_omni", image="vllm-omni",
        models=("qwen3_tts", "qwen3_tts_1p7b", "qwen3omni"),
        env={"VLLM_ROCM_USE_AITER": "1", "MIOPEN_FIND_MODE": "FAST"},
        args=("--omni", "--init-timeout", "2700", "--stage-init-timeout", "2700"),
        recipe="vllm-omni recipes/Qwen/Qwen3-TTS.md, 1x AMD MI300X (Code2Wav graphs allowed, MIOPEN_FIND_MODE="
               "FAST) and vLLM's AMD guide (VLLM_ROCM_USE_AITER=1, ROCM_AITER_FA on the AR stages), applied by the "
               "deploy overlays in benchmark/amd/vllm_omni/"),
    # Startup and warmup limits from the launch line in containers/README.md (the 600 s defaults are too short
    # for a cold start); the launcher waits out both back to back, plus the container start. STRICT_PORT makes a
    # taken port an error instead of a silent move to another one.
    Arm("sglang-omni", "sglang-omni", ("tts",), client="sglang_omni", image="sglang-omni",
        models=("qwen3_tts", "qwen3_tts_1p7b"),
        env={"CTR_PWD": "/workspace/sglang-omni", "SGLANG_OMNI_STARTUP_TIMEOUT": "2700",
             "SGLANG_OMNI_WARMUP_TIMEOUT": "1800", "SGLANG_OMNI_STRICT_PORT": "1"},
        startup_timeout=2700 + 1800 + 300,
        recipe="sglang-omni v0.1.8 defaults with the model's examples/configs pipeline, built on "
               "lmsysorg/sglang:v0.5.21-rocm724-mi30x (sha256:d139bdfc); on ROCm it picks AITER attention and "
               "breakable prefill graphs (AITER cannot capture a full one)"),
    # In-process references at c=1, each request timed alone: no HTTP, no output encoding, no batching.
    Arm("ref-diffusers", "reference", ("image", "video"),
        models=("flux2_klein", "flux2_klein_9b", "z_image_turbo", "wan22", "cosmos3_nano", "cosmos3_edge"),
        recipe="the correctness family's diffusers reference pipeline (bf16, the family's attention and generator), "
               "timed in-process",
        differs="in-process reference: c=1 only, no HTTP, base64 or output encoding"),
    Arm("ref-xdit", "reference", ("image", "video"), image="xdit",
        models=("flux2_klein", "flux2_klein_9b", "z_image_turbo", "wan22", "cosmos3_nano"),
        recipe="xDiT (rocm/pytorch-xdit:v26.9) with --use_torch_compile --attention_backend aiter, bf16, timed "
               "in-process",
        differs="in-process reference: c=1 only, no HTTP, base64 or output encoding"),
    Arm("ref-kokoro", "reference", ("tts",), models=("kokoro",),
        recipe="`kokoro` 0.9.4 KPipeline, eager, one sentence at a time",
        differs="reference implementation: single stream, no server"),
    Arm("ref-orpheus-vllm", "vllm", ("tts",), client="orpheus_vllm", image="vllm", models=("orpheus",),
        env={"VLLM_ROCM_USE_AITER": "1"}, args=("--attention-backend", "ROCM_AITER_FA"), server_gpus=1,
        client_gpu=True, client_path=(f"{REFENV}",),
        recipe="the orpheus-speech design: vLLM serves the LM (AMD guide flags), the client decodes SNAC "
               "in 4-frame windows; " + VLLM_GUIDE,
        differs="reference serving: SNAC decodes one window at a time on the client, unbatched"),
]}


def arms_for(row: Row, names: list[str]) -> list[Arm]:
    """The named arms that can serve this row's family."""
    out = []
    for n in names:
        if n not in ARMS:
            raise KeyError(f"unknown arm {n!r}; known: {', '.join(ARMS)}")
        a = ARMS[n]
        if (row.family in a.families and (not a.models or row.runner_model in a.models)
                and (not a.rows or row.name in a.rows)):
            out.append(a)
    return out


def server_args(arm: Arm, row: Row) -> tuple[list[str], dict]:
    """Engine flags and environment for one (arm, row), without port or GPU selection."""
    args = list(arm.args) + list(row.engine_args.get(arm.name, row.engine_args.get(arm.engine, ())))
    env = {**arm.env, **row.engine_env.get(arm.engine, {}), **row.engine_env.get(arm.name, {})}
    if row.tp > 1:
        args += list(arm.tp_args)
        env.update(arm.tp_env)
    if arm.engine == "vllm":
        if row.tp > 1:
            args += ["--tensor-parallel-size", str(row.tp)]
        if row.max_len:
            args += ["--max-model-len", str(row.max_len)]
    elif arm.engine == "sglang":
        if row.tp > 1:
            args += ["--tp", str(row.tp)]
        if row.max_len:
            args += ["--context-length", str(row.max_len)]
    return args, env
