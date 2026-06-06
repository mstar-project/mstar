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

logger = logging.getLogger(__name__)


_NOT_PORTED = (
    "MingFlashOmniModel is a scaffold; the native mminf port is incomplete. "
    "Benchmark via `--inference-system vllm_omni` against a vllm-omni server "
    "(see benchmark/vllm_omni_instructions.md) until this lands. Reference "
    "implementation: /sgl-workspace/vllm-omni/vllm_omni/model_executor/models/ming_flash_omni/."
)


class MingFlashOmniModel(Model):
    """Thinker + Talker + ImageGen native port of Ming-flash-omni-2.0.

    See module docstring for the target partition layout and a cribsheet
    mapping each abstractmethod to the upstream vllm-omni reference file.
    """

    def __init__(
        self,
        model_path_hf: str = "inclusionAI/Ming-flash-omni-2.0",
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        # Deliberately fail loudly on instantiation: every method below also
        # raises, but stopping at __init__ avoids triggering a half-loaded
        # 238 GB snapshot download for a model whose graph isn't ready.
        raise NotImplementedError(_NOT_PORTED)

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
