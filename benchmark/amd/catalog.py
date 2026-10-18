"""Table rows (a model and its GPU layout) and engine arms.

Competitor settings are the vendors' documented AMD recipes; ``Arm.recipe`` says where each one
comes from. mstar runs its deployment config unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CONTAINERS = "/work1/stephaniewang/atindra/amd/containers"
# Packages the correctness runner installs outside the serving venv (kokoro, snac).
REFENV = "/work1/stephaniewang/atindra/amd/results/correctness/data/refenv"


@dataclass(frozen=True)
class Row:
    """One model served on ``gpus`` GPUs. ``name`` is also the mstar config name."""

    name: str
    family: str  # text, tts or asr
    hf_id: str
    runner_model: str  # benchmark.runner --model
    gpus: int = 1
    tp: int = 1
    # Context length every engine is given (mstar's max_seq_len), so KV sizing starts from the same place.
    max_len: int | None = None
    runner_model_id: bool = False  # pass --model-id (the runner's Qwen3.5 class serves several sizes)


def _qwen(size: str, gpus: int = 1, tp: int = 1, name: str | None = None) -> Row:
    return Row(name or f"qwen3_5_{size.lower()}", "text", f"Qwen/Qwen3.5-{size}", "qwen3.5", gpus=gpus, tp=tp,
               max_len=32768, runner_model_id=True)


ROWS: dict[str, Row] = {r.name: r for r in [
    _qwen("0.8B"),
    _qwen("2B"),
    _qwen("4B"),
    _qwen("9B"),
    _qwen("9B", gpus=2, tp=2, name="qwen3_5_9b_tp2"),
    Row("kokoro", "tts", "hexgrad/Kokoro-82M", "kokoro"),
    Row("orpheus", "tts", "canopylabs/orpheus-3b-0.1-ft", "orpheus", gpus=2),
    Row("orpheus_colocated", "tts", "canopylabs/orpheus-3b-0.1-ft", "orpheus"),
    Row("qwen3tts", "tts", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "qwen3_tts"),
    Row("whisper_large", "asr", "openai/whisper-large-v3", "whisper_large", max_len=448),
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
    # Why the arm does not compute the same thing as mstar, for the table's "not comparable because" column.
    differs: str = ""
    models: tuple[str, ...] = ()  # rows (by runner model) this arm serves; empty means any row of its families
    # The server takes the first ``server_gpus`` of the row's GPUs; the client gets the next one (or the
    # first if none is left) for work it does on a GPU, such as decoding audio.
    server_gpus: int | None = None
    client_gpu: bool = False
    # Extra PYTHONPATH entries for the benchmark client (packages the client needs outside mstar's venv).
    client_path: tuple[str, ...] = ()


VLLM_GUIDE = ("AMD vLLM guide https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/"
              "vllm-optimization.html; vllm/platforms/rocm.py in the image")
SGLANG_COOKBOOK = "sglang v0.5.22 docs/cookbook/autoregressive/Qwen/Qwen3.5.mdx, AMD section"
SGLANG_CI = "sglang v0.5.22 test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py (AMD CI launch flags)"

ARMS: dict[str, Arm] = {a.name: a for a in [
    Arm("mstar", "mstar", ("text", "tts", "asr"), client="ours_openai", env={"MSTAR_SHM_ARENA": "1"},
        recipe="configs/<row>.yaml unchanged; MSTAR_SHM_ARENA=1 as .claude/skills/benchmarking asks for numbers"),
    Arm("vllm", "vllm", ("text",), client="vllm", image="vllm",
        recipe="image defaults (v0.31.0 picks ROCM_ATTN on gfx942, AITER off)"),
    Arm("vllm-aiter", "vllm", ("text",), client="vllm", image="vllm", env={"VLLM_ROCM_USE_AITER": "1"},
        args=("--attention-backend", "ROCM_AITER_FA"), tp_env={"NCCL_MIN_NCHANNELS": "112"},
        recipe=VLLM_GUIDE + ": VLLM_ROCM_USE_AITER=1, --attention-backend ROCM_AITER_FA, "
                            "NCCL_MIN_NCHANNELS=112 for TP"),
    Arm("vllm-aiter-unified", "vllm", ("text",), client="vllm", image="vllm", env={"VLLM_ROCM_USE_AITER": "1"},
        args=("--attention-backend", "ROCM_AITER_UNIFIED_ATTN"), tp_env={"NCCL_MIN_NCHANNELS": "112"},
        recipe=VLLM_GUIDE + ": the AITER unified attention backend, the same kernel mstar uses for paged attention"),
    Arm("vllm-whisper", "vllm", ("asr",), client="vllm_transcribe", image="vllm",
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
    Arm("vllm-omni", "vllm-omni", ("tts",), client="vllm_omni", image="vllm-omni", models=("qwen3_tts",),
        args=("--omni", "--init-timeout", "2700", "--stage-init-timeout", "2700"),
        recipe="vllm-omni v0.30.0 defaults on ROCm (AR stages on TRITON_ATTN, Code2Wav eager)",
        differs="Code2Wav runs eager on ROCm"),
    Arm("vllm-omni-tuned", "vllm-omni", ("tts",), client="vllm_omni", image="vllm-omni", models=("qwen3_tts",),
        env={"VLLM_ROCM_USE_AITER": "1", "MIOPEN_FIND_MODE": "FAST"},
        args=("--omni", "--init-timeout", "2700", "--stage-init-timeout", "2700"),
        recipe="vllm-omni recipes (VLLM_ROCM_USE_AITER=1, MIOPEN_FIND_MODE=FAST); per-stage ROCM_AITER_FA needs "
               "a --deploy-config"),
    Arm("sglang-omni", "sglang-omni", ("tts",), client="sglang_omni", image="sglang-omni", models=("qwen3_tts",),
        recipe="sglang-omni v0.1.8 defaults, self-built on lmsysorg/sglang:v0.5.21-rocm720-mi30x",
        differs="prefill and Code2Wav graphs are off on ROCm"),
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
        if row.family in a.families and (not a.models or row.runner_model in a.models):
            out.append(a)
    return out


def server_args(arm: Arm, row: Row) -> tuple[list[str], dict]:
    """Engine flags and environment for one (arm, row), without port or GPU selection."""
    args, env = list(arm.args), dict(arm.env)
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
