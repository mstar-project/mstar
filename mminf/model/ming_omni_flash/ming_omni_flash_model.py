"""MingFlashOmniModel: native mminf port of Ming-flash-omni-2.0.

Step 3d: text-only thinker path is wired end-to-end. Vision / audio /
talker / image-gen are step 4+.

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
    PartitionDefinition,
    StreamingConnectionState,
)
from mminf.engine.base import EngineType
from mminf.engine.kv_store import KVCacheConfig
from mminf.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    TensorPointerInfo,
)
from mminf.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mminf.model.base import ForwardPassArgs, Model
from mminf.model.ming_omni_flash.components.model import LingMoeModel
from mminf.model.ming_omni_flash.config import MingFlashOmniModelConfig
from mminf.model.ming_omni_flash.loader import load_thinker_weights
from mminf.model.ming_omni_flash.submodules import (
    AudioEncoderSubmodule,
    BailingMoeV2ThinkerSubmodule,
    VisionEncoderSubmodule,
)
from mminf.streaming.topology import PartitionTopology

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


def _patch_bailing_tokenizer_for_transformers5() -> None:
    """Make BailingTokenizer load under transformers >= 5.0.

    Two upstream incompatibilities, both in
    ``tokenization_bailing.BailingTokenizer``:

    (1) transformers 5.x removed ``PreTrainedTokenizerBase.verbose``, but
    Ming's accessor properties (``gmask_token`` etc.) still reference
    ``self.verbose`` in their not-set fallback paths.  Backport a class-level
    default so ``check_special_tokens`` doesn't blow up.

    (2) ``BailingTokenizer.__init__`` sets ``self.add_bos_token = ...``
    BEFORE calling ``super().__init__()``.  In transformers 5.x the
    ``PreTrainedTokenizerFast.add_bos_token`` setter immediately calls
    ``update_post_processor()``, which dereferences ``self._tokenizer`` —
    but that attribute is only created inside the deferred ``super``
    call.  Wrap ``update_post_processor`` to no-op when ``_tokenizer``
    isn't built yet; the deferred super call runs it for real.

    The module is loaded dynamically by ``transformers``' trust_remote_code
    machinery; look it up in ``sys.modules`` rather than importing it.
    """
    import sys as _sys
    for mod_name, mod in list(_sys.modules.items()):
        if mod is None or not mod_name.endswith("tokenization_bailing"):
            continue
        cls = getattr(mod, "BailingTokenizer", None)
        if cls is None:
            continue
        if not hasattr(cls, "verbose"):
            cls.verbose = False

    # (2) — patch update_post_processor on the parent fast-tokenizer class
    # once. Guard against re-patching across multiple model instantiations.
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        return
    if getattr(PreTrainedTokenizerFast.update_post_processor, "_mminf_patched", False):
        return
    _orig_upp = PreTrainedTokenizerFast.update_post_processor

    def _safe_update_post_processor(self):
        if getattr(self, "_tokenizer", None) is None:
            return
        return _orig_upp(self)

    _safe_update_post_processor._mminf_patched = True
    PreTrainedTokenizerFast.update_post_processor = _safe_update_post_processor


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
        except AttributeError as e:
            # Two BailingTokenizer/transformers-5.x incompats — see
            # _patch_bailing_tokenizer_for_transformers5 for the full story.
            # Patch once and retry; surface only the second error.
            if "verbose" in str(e) or "post_processor" in str(e):
                _patch_bailing_tokenizer_for_transformers5()
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        local_dir, cache_dir=cache_dir, trust_remote_code=True,
                    )
                except Exception as e2:
                    self._warn_tokenizer_unavailable("tokenizer", e2)
            else:
                self._warn_tokenizer_unavailable("tokenizer", e)
        except Exception as e:
            self._warn_tokenizer_unavailable("tokenizer", e)

        try:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                local_dir, cache_dir=cache_dir, trust_remote_code=True,
            )
        except Exception as e:
            self._warn_tokenizer_unavailable("processor", e)

        # Lazy submodule cache — populated on first get_submodule call.
        self._submodule_cache: dict[str, object] = {}

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
    # Model ABC: KV cache config (thinker only for step 3d)
    # ------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        llm = self.config.thinker_llm
        return [KVCacheConfig(
            num_layers=llm.num_hidden_layers,
            num_kv_heads=llm.num_key_value_heads,
            head_dim=llm.head_dim,
            max_seq_len=llm.max_position_embeddings,
            num_qo_heads=llm.num_attention_heads,
            nodes=["Thinker"],
        )]

    def get_node_engine_types(self) -> dict[str, EngineType]:
        # Step 5a: vision + audio encoders are stateless graph nodes
        # alongside the Thinker. Talker / AudioVAE / ImageGen fold in
        # at step 6+. The encoders only register as nodes here when
        # the snapshot ships the corresponding sub-configs — a
        # thinker-only config (configs/ming_flash_omni_thinker_only.yaml)
        # will still want only Thinker, so callers wire encoder nodes
        # in their yaml only when needed.
        return {
            "Thinker": EngineType.KV_CACHE,
            "vision_encoder": EngineType.STATELESS,
            "audio_encoder": EngineType.STATELESS,
        }

    # ------------------------------------------------------------------
    # Graph walks: prefill + decode loop, text-only
    # ------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = GraphNode(
            name="Thinker",
            input_names=["text_inputs"],
            outputs=[GraphEdge(
                next_node=EMIT_TO_CLIENT,
                name="new_token",
                output_modality="text",
                persist=True,
            )],
        )
        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="Thinker",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                    ),
                    GraphEdge(
                        next_node="Thinker",
                        name="text_inputs",
                        output_modality="text",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )
        return {"prefill": prefill, "decode": decode}

    def get_partition_topology(self) -> PartitionTopology:
        return PartitionTopology(partitions=["Thinker"], connections=[])

    def get_partitions(self) -> list[PartitionDefinition]:
        return [PartitionDefinition(
            name="Thinker",
            graph_walks={"prefill", "decode"},
            initial_walk="prefill",
            producer_partitions=[],
        )]

    # ------------------------------------------------------------------
    # Forward-pass arg builders — mirrors Orpheus's LLM-partition flow
    # ------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        if partition_name != "Thinker":
            raise ValueError(f"Unknown partition: {partition_name!r}")
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk="prefill",
            is_prefill=True,
        )
        graph_edge = GraphEdge(next_node="Thinker", name="text_inputs")
        graph_edge.tensor_info = input_signals.get("text_inputs", [])
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=[graph_edge],
            unpersist_tensors=list(graph_edge.tensor_info),
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        new_tokens: dict[str, list[int]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Thinker partition: prefill → decode loop until EOS or max tokens.

        Same shape as Orpheus's _get_llm_partition_forward.
        """
        if partition_name != "Thinker":
            raise ValueError(f"Unknown partition: {partition_name!r}")

        request_done = False
        if partition_metadata.is_prefill:
            partition_metadata.is_prefill = False
            partition_metadata.graph_walk = "decode"
        elif partition_metadata.graph_walk == "decode":
            request_done = True
            partition_metadata.kwargs["decode_finished"] = True

        if request_done:
            return ForwardPassArgs(
                full_metadata=partition_metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        graph_edge = GraphEdge(next_node="Thinker", name="text_inputs")
        graph_edge.tensor_info = persist_signals.get("new_token", [])
        return ForwardPassArgs(
            full_metadata=partition_metadata,
            inputs=[graph_edge],
            unpersist_tensors=list(graph_edge.tensor_info),
            step_metadata={"is_prefill": partition_metadata.is_prefill},
        )

    # ------------------------------------------------------------------
    # Prompt / output handling
    # ------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Tokenize a text prompt via the chat template.

        The jinja chat_template in ``tokenizer_config.json`` accepts
        OpenAI-standard ``user``/``assistant``/``system`` roles and
        remaps them to Ming's internal HUMAN/ASSISTANT/SYSTEM. We
        send a plain ``{"role": "user", "content": <prompt>}`` and
        let the template handle the rest.
        """
        if prompt is None:
            return {}
        if self.tokenizer is None:
            raise RuntimeError(
                "MingFlashOmniModel.process_prompt called but tokenizer "
                "is not loaded. See _warn_tokenizer_unavailable for setup."
            )
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
        return {"text_inputs": [input_ids]}

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        if modality != "text":
            raise ValueError(
                f"Unsupported modality for Ming-flash-omni-2.0 step 3d: "
                f"{modality!r}. Audio/image lands in step 4+."
            )
        if self.tokenizer is None:
            return b""
        if output.numel() == 0:
            return b""
        text = self.tokenizer.decode(output.tolist(), skip_special_tokens=True)
        return text.encode("utf-8")

    # ------------------------------------------------------------------
    # Submodule construction
    # ------------------------------------------------------------------

    def get_default_sharding_config(self):
        """Thinker is TP-capable; engine's worker maps `tp_size` from
        the yaml's node_group to the rank's comm_group."""
        from mminf.distributed.base import ShardingConfig

        return ShardingConfig(
            groups=[],
            tp_enabled_nodes={"Thinker"},
            shard_dim={},
        )

    def get_submodule(self, node_name: str, device="cpu", tp_group=None):
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        if node_name == "vision_encoder":
            submodule = self._create_vision_encoder_submodule(device)
            self._submodule_cache[node_name] = submodule
            return submodule
        if node_name == "audio_encoder":
            submodule = self._create_audio_encoder_submodule(device)
            self._submodule_cache[node_name] = submodule
            return submodule
        if node_name != "Thinker":
            raise ValueError(
                f"Unknown node: {node_name!r}. Step 5a registers "
                f"'Thinker', 'vision_encoder', 'audio_encoder'; Talker / "
                f"AudioVAE / ImageGen follow in steps 6+."
            )

        # Build LingMoeModel on the meta device first so the constructor's
        # `torch.empty(...)` allocations don't materialise on the target
        # device. Then `.to_empty(device=device)` reallocates each Parameter
        # in real memory, and the loader streams weights into them.
        llm = self.config.thinker_llm
        mrope = llm.mrope_section
        with torch.device("meta"):
            model = LingMoeModel(
                vocab_size=llm.vocab_size,
                hidden_size=llm.hidden_size,
                intermediate_size=llm.intermediate_size,
                moe_intermediate_size=llm.moe_intermediate_size,
                num_hidden_layers=llm.num_hidden_layers,
                num_attention_heads=llm.num_attention_heads,
                num_kv_heads=llm.num_key_value_heads,
                head_dim=llm.head_dim,
                rms_norm_eps=llm.rms_norm_eps,
                rope_theta=llm.rope_theta,
                max_position_embeddings=llm.max_position_embeddings,
                partial_rotary_factor=llm.partial_rotary_factor,
                mrope_section=mrope,
                num_experts=llm.num_experts,
                num_experts_per_tok=llm.num_experts_per_tok,
                num_shared_experts=llm.num_shared_experts,
                n_group=llm.n_group,
                topk_group=llm.topk_group,
                routed_scaling_factor=llm.moe_router_topk_scaling_factor,
                first_k_dense_replace=llm.first_k_dense_replace,
                tie_word_embeddings=llm.tie_word_embeddings,
                use_qkv_bias=llm.use_qkv_bias,
                use_bias=llm.use_bias,
                comm_group=tp_group,
            )
        # Materialise + cast to bf16 (matches the released ckpt's torch_dtype).
        model.to_empty(device=device)
        model.to(self.get_autocast_dtype())

        load_thinker_weights(model, self.local_dir, device=device, strict=True)
        model.eval()

        submodule = BailingMoeV2ThinkerSubmodule(
            model=model,
            config=self.config,
            eos_token_id=llm.eos_token_id,
        )
        self._submodule_cache[node_name] = submodule
        return submodule

    # ------------------------------------------------------------------
    # Encoder construction helpers (step 5a)
    # ------------------------------------------------------------------

    def _create_vision_encoder_submodule(self, device: str):
        """Build Qwen3MoeVisionTransformer + MingVisionProjector, load weights.

        The vision encoder lives on a single rank (no TP) per the
        typical topology. Uses bf16 to match the released ckpt's dtype.
        ``attn_implementation`` defaults to ``flash_attention_2`` for
        video performance (same gotcha as qwen3_omni:1508-1519); fall
        back to eager only when explicitly disabled via env var.
        """
        from mminf.model.ming_omni_flash.components.projectors import (
            MingVisionProjector,
        )
        from mminf.model.ming_omni_flash.components.vision_encoder import (
            build_vision_encoder,
        )
        from mminf.model.ming_omni_flash.loader import (
            load_vision_encoder_weights,
            load_vision_projector_weights,
        )

        dtype = self.get_autocast_dtype()
        attn = os.environ.get("MING_VISION_ATTN_IMPL", "flash_attention_2")

        vision_encoder = build_vision_encoder(
            config=self.config.vision,
            dtype=dtype,
            device=device,
            attn_implementation=attn,
            local_dir=self.local_dir,
        )
        load_vision_encoder_weights(
            vision_encoder, self.local_dir, device=device, strict=True,
        )

        vision_projector = MingVisionProjector(
            vision_dim=self.config.vision.out_hidden_size,
            llm_dim=self.config.thinker_llm.hidden_size,
            mlp_depth=self.config.mlp_depth,
        )
        vision_projector = vision_projector.to(dtype=dtype, device=device)
        load_vision_projector_weights(
            vision_projector, self.local_dir, device=device, strict=True,
        )
        vision_projector.eval()

        return VisionEncoderSubmodule(
            vision_encoder=vision_encoder,
            vision_projector=vision_projector,
            config=self.config,
        )

    def _create_audio_encoder_submodule(self, device: str):
        """Build MingAudioEncoder + MingAudioProjector, load weights.

        Audio encoder is the self-contained Whisper port from step 4a
        (no openai-whisper runtime dep). Uses bf16 to match the
        released ckpt's dtype. Flash-attn varlen kicks in when
        available; otherwise the manual padded-attention fallback runs.
        """
        from mminf.model.ming_omni_flash.components.audio_encoder import (
            build_audio_encoder,
        )
        from mminf.model.ming_omni_flash.components.projectors import (
            MingAudioProjector,
        )
        from mminf.model.ming_omni_flash.loader import (
            load_audio_encoder_weights,
            load_audio_projector_weights,
        )

        dtype = self.get_autocast_dtype()

        audio_encoder = build_audio_encoder(
            audio_config=self.config.audio_encoder,
            dtype=dtype,
            device=device,
            use_flash_attn=True,
        )
        load_audio_encoder_weights(
            audio_encoder, self.local_dir, device=device, strict=True,
        )

        audio_projector = MingAudioProjector(
            audio_dim=self.config.audio_encoder.d_model,
            llm_dim=self.config.thinker_llm.hidden_size,
            ds_kernel_size=self.config.audio_encoder.ds_kernel_size,
            ds_stride=self.config.audio_encoder.ds_stride,
            mlp_depth=self.config.mlp_depth,
        )
        audio_projector = audio_projector.to(dtype=dtype, device=device)
        load_audio_projector_weights(
            audio_projector, self.local_dir, device=device, strict=True,
        )
        audio_projector.eval()

        return AudioEncoderSubmodule(
            audio_encoder=audio_encoder,
            audio_projector=audio_projector,
            config=self.config,
        )
