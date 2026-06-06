"""MingFlashOmniModel: native mminf port of Ming-flash-omni-2.0.

WIP SCAFFOLD — does not run end-to-end yet.

Until this port is complete, benchmark Ming-flash-omni-2.0 via the
``vllm_omni`` inference system against a vllm-omni server (see
``benchmark/vllm_omni_instructions.md``).

The released checkpoint (``inclusionAI/Ming-flash-omni-2.0``, 2026-02-11) is a
Ling-2.0 sparse-MoE omni model: 100B total / 6B active params, ~238 GB / 42
shards. The vllm-omni reference port (~6,500 LOC) lives at::

    /sgl-workspace/vllm-omni/vllm_omni/model_executor/models/ming_flash_omni/

That tree is the source of truth for the architecture; this scaffold mirrors
mminf's class shape (``mminf/model/qwen3_omni/qwen3_omni_model.py``) and
leaves each abstractmethod raising ``NotImplementedError`` with a pointer to
the corresponding upstream file/symbol.

Target partition layout (mirrors vllm-omni's deploy yamls):

    Thinker   — Ling-2.0 MoE LLM + vision/audio encoders -> text out
    Talker    — CFM head + small LLM -> audio waveform via AudioVAE
    ImageGen  — ByT5 + ZImage DiT -> image out (separate deploy)

Mapping to vllm-omni source (use these as the porting cribsheet):

    Thinker       -> ming_flash_omni_thinker.py            (1,164 LOC)
    Talker        -> ming_flash_omni_talker.py + talker_module.py
    AudioVAE      -> audio_vae.py
    AudioEncoder  -> audio_encoder.py
    Vision        -> vision_encoder.py + projectors.py
    Ling MoE LLM  -> modeling_bailing_moe_v2.py            (892 LOC)
    ImageGen      -> /sgl-workspace/vllm-omni/vllm_omni/diffusion/models/ming_flash_omni/
    Pipeline glue -> pipeline.py + ming_flash_omni.py
    Prompt tokens -> prompt_utils.py (IMAGE_PATCH_TOKEN, BASE_CAPTION_TEMPLATE)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mminf.engine.base import EngineType
from mminf.engine.kv_store import KVCacheConfig
from mminf.graph.base import GraphSection, TensorPointerInfo
from mminf.model.base import ForwardPassArgs, Model
from mminf.model.ming_omni_flash.config import MingFlashOmniModelConfig

logger = logging.getLogger(__name__)


_NOT_PORTED = (
    "MingFlashOmniModel is a scaffold; the native mminf port is incomplete. "
    "Benchmark via `--inference-system vllm_omni` against a vllm-omni server "
    "(see benchmark/vllm_omni_instructions.md) until this lands. Reference "
    "implementation: /sgl-workspace/vllm-omni/vllm_omni/model_executor/models/ming_flash_omni/."
)


# Files in the Ming GitHub repo (https://github.com/inclusionAI/Ming) that
# the HF AutoTokenizer / AutoProcessor for Ming-flash-omni-2.0 needs to find
# adjacent to the snapshot's ``config.json``. The HF checkpoint ships only
# weights + sub-dir configs; the modeling/processing/tokenization Python
# modules live in the source repo. ``_prepare_tokenizer_dir`` symlinks these
# alongside the snapshot when both are available.
_MING_CODE_FILES = (
    # Python modules (configs, modeling, processing)
    "configuration_audio.py",
    "configuration_bailing_moe_v2.py",
    "configuration_bailing_talker.py",
    "configuration_bailingmm2.py",
    "configuration_whisper_encoder.py",
    "audio_processing_bailingmm2.py",
    "bailingmm_utils.py",
    "bailingmm_utils_video.py",
    "chat_format.py",
    "image_processing_bailingmm2.py",
    "modeling_bailing_moe_v2.py",
    "modeling_bailing_talker.py",
    "modeling_bailingmm2.py",
    "modeling_utils.py",
    "modeling_whisper_encoder.py",
    "processing_bailingmm2.py",
    "qwen2_5_vit.py",
    "qwen3_moe_vit.py",
    "s3bpe_tokenizer.py",
    "tokenization_bailing.py",
    # JSON assets the processor / tokenizer load from disk
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
)


def _resolve_local_hf_snapshot(repo_id: str, cache_dir: str | None = None) -> str:
    """Resolve a HF repo id to a local snapshot path (downloading if needed).

    Mirrors mminf/model/qwen3_omni/qwen3_omni_model.py:_resolve_local_hf_snapshot.
    Returns the repo id unchanged if the download fails — that way an
    air-gapped environment with a pre-populated cache (or a local-path repo
    id) still resolves.
    """
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=False,
        )
    except Exception as e:
        logger.warning("Error downloading from HuggingFace: %s", str(e))
        return repo_id
    return str(Path(local_dir))


def _find_ming_code_dir() -> str | None:
    """Locate a clone of https://github.com/inclusionAI/Ming on disk.

    Lookup order:
      1. ``MING_CODE_DIR`` environment variable (explicit override).
      2. ``./Ming`` or ``/tmp/ming_repo`` (common dev locations).
      3. Any directory on ``sys.path`` containing ``configuration_bailingmm2.py``.

    Returns ``None`` if nothing is found. Caller is responsible for surfacing
    a clear error/warning in that case.
    """
    override = os.environ.get("MING_CODE_DIR")
    candidates: list[str] = []
    if override:
        candidates.append(override)
    candidates.extend(["./Ming", "/tmp/ming_repo"])
    candidates.extend(sys.path)

    for c in candidates:
        if c and (Path(c) / "configuration_bailingmm2.py").exists():
            return str(Path(c).resolve())
    return None


def _prepare_tokenizer_dir(snapshot_dir: str, ming_code_dir: str) -> None:
    """Symlink Ming source files alongside the snapshot's ``config.json``.

    ``transformers.AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)``
    resolves ``auto_map`` references (e.g. ``configuration_bailingmm2.py``)
    by file path adjacent to ``config.json`` — not via PYTHONPATH. We bridge
    that by symlinking the .py files from ``ming_code_dir`` into the snapshot
    dir. Idempotent: existing files (and existing symlinks) are skipped, so
    re-running on a populated snapshot is a no-op.
    """
    snap = Path(snapshot_dir)
    src = Path(ming_code_dir)
    for name in _MING_CODE_FILES:
        target = snap / name
        if target.exists() or target.is_symlink():
            continue
        source = src / name
        if not source.exists():
            continue
        try:
            target.symlink_to(source)
        except OSError as e:
            # Snapshot may be on a filesystem without symlink support, or
            # may be read-only. Don't crash — the loader below will surface
            # a clearer error if the file is still missing.
            logger.debug("Failed to symlink %s -> %s: %s", target, source, e)


class MingFlashOmniModel(Model):
    """Thinker + Talker + ImageGen native port of Ming-flash-omni-2.0.

    See module docstring for the target partition layout and a cribsheet
    mapping each abstractmethod to the upstream vllm-omni reference file.
    """

    def __init__(
        self,
        model_path_hf: str = "inclusionAI/Ming-flash-omni-2.0",
        cache_dir: str | None = None,
        ming_code_dir: str | None = None,
        **kwargs,
    ):
        """Load config + (best-effort) tokenizer + processor.

        Args:
            model_path_hf: HF repo id or local path to the Ming snapshot.
            cache_dir: Override HF Hub cache for snapshot_download.
            ming_code_dir: Path to a clone of github.com/inclusionAI/Ming
                (must contain ``configuration_bailingmm2.py`` etc.). Required
                for the tokenizer + processor — the HF checkpoint ships only
                weights, the Python modules live in the source repo. Falls
                back to MING_CODE_DIR env var, then to ``./Ming``,
                ``/tmp/ming_repo``, and sys.path.

        Subclasses' abstractmethods all still raise NotImplementedError; this
        constructor only stages config / tokenizer / processor so the
        verification tests for step-1/step-2 can exercise the load path.
        """
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir

        local_dir = _resolve_local_hf_snapshot(model_path_hf, cache_dir=cache_dir)
        self.local_dir = local_dir
        self.config = MingFlashOmniModelConfig.from_pretrained(local_dir)

        # Tokenizer + processor. The released checkpoint ships only weights
        # and sub-dir configs — no top-level tokenizer.json / vocab.json, and
        # none of the .py modules that AutoTokenizer / AutoProcessor's
        # ``trust_remote_code`` path expects to find next to config.json.
        # We resolve those from a separately-cloned Ming source repo and
        # symlink them in. If neither is available, we warn loudly and
        # leave self.tokenizer / self._processor as None — process_prompt
        # (step 7) will raise a clearer error then.
        code_dir = ming_code_dir or _find_ming_code_dir()
        if code_dir is not None:
            _prepare_tokenizer_dir(local_dir, code_dir)
            # transformers' trust_remote_code loader resolves sibling imports
            # (e.g. ``configuration_bailing_moe_v2``) via ``sys.path``, not by
            # scanning the snapshot dir. Push the snapshot onto sys.path so
            # those imports succeed during dynamic module loading.
            if local_dir not in sys.path:
                sys.path.insert(0, local_dir)
        self.ming_code_dir = code_dir

        self.tokenizer = None
        self._processor = None
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_dir, cache_dir=cache_dir, trust_remote_code=True,
            )
        except Exception as e:
            self._warn_tokenizer_unavailable("tokenizer", e)

        try:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                local_dir, cache_dir=cache_dir, trust_remote_code=True,
            )
        except Exception as e:
            self._warn_tokenizer_unavailable("processor", e)

        # Lazy submodule cache — empty until later porting steps land.
        self._submodule_cache: dict[str, object] = {}

        raise NotImplementedError(_NOT_PORTED)

    @staticmethod
    def _warn_tokenizer_unavailable(what: str, err: Exception) -> None:
        """Single-place explanation of how to make the tokenizer/processor load.

        Tokenizer + processor live in the Ming source repo, not the HF
        checkpoint. Without them ``process_prompt`` can't run; the rest of
        the model loads fine.
        """
        logger.warning(
            "Ming-flash-omni-2.0 %s could not be loaded (%s: %s). "
            "To enable it: (1) git clone https://github.com/inclusionAI/Ming "
            "(2) pip install opencv-python-headless openai-whisper "
            "(3) set MING_CODE_DIR=<path/to/Ming>. The snapshot ships only "
            "weights; the tokenizer/processor Python modules live in the "
            "source repo.",
            what, type(err).__name__, str(err)[:200],
        )

    # ------------------------------------------------------------------
    # Model ABC — every method below is a stub. Implement by mirroring
    # mminf/model/qwen3_omni/qwen3_omni_model.py and the upstream
    # vllm-omni files listed in the module docstring.
    # ------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        # Port: separate KVCacheConfig for Thinker (Ling MoE) and Talker.
        # Pull dims from MingFlashOmniModelConfig.thinker / .talker after
        # the config port is done. Cribsheet: qwen3_omni_model.get_kv_cache_config.
        raise NotImplementedError(_NOT_PORTED)

    def get_node_engine_types(self) -> dict[str, EngineType]:
        # Likely shape (mirrors Qwen3-Omni's set):
        #   "audio_encoder": STATELESS
        #   "vision_encoder": STATELESS
        #   "Thinker": KV_CACHE
        #   "Talker": KV_CACHE   (CFM still runs autoregressively token-side)
        #   "AudioVAE": STATELESS
        #   "ImageGen": STATELESS  (DiT, no KV cache)
        raise NotImplementedError(_NOT_PORTED)

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # Walks to port:
        #   prefill_text / prefill_audio / prefill_vision / prefill_video
        #   thinker_decode
        #   talker_prefill / talker_decode
        #   audio_vae_decode  (codec tokens -> waveform)
        #   image_gen         (ImageGen partition, separate deploy yaml)
        raise NotImplementedError(_NOT_PORTED)

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        raise NotImplementedError(_NOT_PORTED)

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        new_tokens: dict[str, list[int]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        raise NotImplementedError(_NOT_PORTED)

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        # Build the chat-template prompt and (when output is image) append
        # the <image><imagePatch>*N</image> query-token block via
        # ``vllm_omni/.../prompt_utils.py:maybe_expand_image_gen_prompt``.
        # OpenAI roles (user/assistant/system) map to Ming's uppercase
        # HUMAN/ASSISTANT/SYSTEM inside the HF processor's chat_template.
        raise NotImplementedError(_NOT_PORTED)

    def postprocess(self, output: torch.Tensor, modality: str) -> bytes:
        # Text -> utf-8; image -> PNG; audio -> 16-bit PCM @ get_output_sample_rate().
        raise NotImplementedError(_NOT_PORTED)

    def get_submodule(self, node_name: str, device="cpu", tp_group=None):
        # Per-node nn.Module factory. Lazy-cache like qwen3_omni does.
        raise NotImplementedError(_NOT_PORTED)
