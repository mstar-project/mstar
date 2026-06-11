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
    Sequential,
    TensorPointerInfo,
)
from mminf.graph.special_destinations import EMIT_TO_CLIENT
from mminf.model.base import ForwardPassArgs, Model
from mminf.model.ming_omni_flash.components.model import LingMoeModel
from mminf.model.ming_omni_flash.config import MingFlashOmniModelConfig
from mminf.model.ming_omni_flash.loader import load_thinker_weights
from mminf.model.ming_omni_flash.submodules import (
    AudioEncoderSubmodule,
    BailingMoeV2ThinkerSubmodule,
    VisionEncoderSubmodule,
)
from mminf.streaming.chunk_policy import FixedChunkPolicy
from mminf.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge

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

        # Talker tokenizer (talker/llm/) — separate from the thinker
        # tokenizer. The Thinker->Talker bridge (step 6e-3) decodes the
        # thinker's text output and re-encodes it here. Loaded lazily on
        # first use via `_get_talker_tokenizer` so thinker-only configs
        # don't pay for it.
        self._talker_tokenizer = None

        # Lazy submodule cache — populated on first get_submodule call.
        self._submodule_cache: dict[str, object] = {}

    def _get_talker_tokenizer(self):
        """Load + cache the talker's own Qwen2 tokenizer (talker/llm/).

        The talker re-tokenizes the thinker's detokenized text with this
        tokenizer (vocab_size 151936, distinct from the thinker's
        BailingTokenizer). Returns None if the talker subdir / tokenizer
        is unavailable.
        """
        if self._talker_tokenizer is not None:
            return self._talker_tokenizer
        talker_dir = Path(self.local_dir) / "talker" / "llm"
        if not (talker_dir / "tokenizer_config.json").exists():
            return None
        try:
            from transformers import AutoTokenizer
            self._talker_tokenizer = AutoTokenizer.from_pretrained(
                str(talker_dir), trust_remote_code=True,
            )
        except Exception as e:
            logger.warning("Talker tokenizer (talker/llm/) failed to load: %s", e)
            return None
        return self._talker_tokenizer

    def thinker_text_to_talker_inputs(self, thinker_token_ids) -> "torch.Tensor":
        """Bridge: thinker output token ids -> talker_text_inputs token ids.

        Ming's thinker->talker handoff passes detokenized TEXT, not
        hidden states (see vllm-omni pipeline.py `thinker2talker`). We
        decode the thinker's generated ids with the thinker tokenizer,
        then re-encode with the talker's own `talker/llm` tokenizer.

        Returns a 1-D long tensor of talker token ids. Raises if either
        tokenizer is unavailable.
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "thinker_text_to_talker_inputs: thinker tokenizer not loaded."
            )
        talker_tok = self._get_talker_tokenizer()
        if talker_tok is None:
            raise RuntimeError(
                "thinker_text_to_talker_inputs: talker tokenizer (talker/llm/) "
                "not available — cannot bridge to the Talker partition."
            )
        if isinstance(thinker_token_ids, torch.Tensor):
            ids_list = thinker_token_ids.flatten().tolist()
        else:
            ids_list = list(thinker_token_ids)
        text = self.tokenizer.decode(ids_list, skip_special_tokens=True)
        talker_ids = talker_tok(text, return_tensors="pt").input_ids[0]
        return talker_ids

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
        types = {
            "Thinker": EngineType.KV_CACHE,
            "vision_encoder": EngineType.STATELESS,
            "audio_encoder": EngineType.STATELESS,
        }
        # Step 6e-2: register the Talker as a stateless TTS node when the
        # snapshot ships a talker/ subdir. The talker runs its full
        # AR-decode + VAE-decode internally (the CFM step count is
        # stop_head-determined, not a conductor decode loop), so a single
        # STATELESS node suffices. Thinker-only configs leave this off.
        if self.config.talker is not None:
            types["Talker"] = EngineType.STATELESS
        # Step 9b: register ImageGen as a stateless diffusion node when the
        # snapshot ships an imagegen tree (transformer/ + vae/ + connector/).
        # Its full denoise loop + VAE decode run internally (the step count is
        # scheduler-determined, not a conductor decode loop), so a single
        # STATELESS node suffices. Thinker-only / talker-only configs leave it
        # off.
        if self.config.image_gen is not None:
            types["ImageGen"] = EngineType.STATELESS
        return types

    # ------------------------------------------------------------------
    # Graph walks: text + audio + vision/video prefill + AR decode (step 5c)
    # ------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        """Five graph walks covering all modality inputs + autoregressive decode.

        * ``prefill_text`` — Thinker only; text tokens → first sampled
          token (also the legacy ``prefill`` walk in step 3f).
        * ``prefill_audio`` — ``audio_encoder`` → Thinker. Audio encoder
          emits ``audio_embeds`` that the Thinker splices between
          ``audio_start``/``audio_end`` sentinels (step 5b).
        * ``prefill_vision`` — ``vision_encoder`` → Thinker. Image
          inputs; the Thinker splices between ``image_start``/``image_end``.
        * ``prefill_video`` — ``vision_encoder`` → Thinker. Video inputs
          (same encoder; the Thinker dispatch reads
          ``video_second_per_grid`` and switches to video sentinels).
        * ``thinker_decode`` — single-step AR loop (also the legacy
          ``decode`` walk in step 3f).

        Each prefill walk's final Thinker node emits the first sampled
        token to the client (``EMIT_TO_CLIENT`` + ``output_modality="text"``)
        and the decode loop emits + loops each subsequent token, exactly
        like step 3f's text-only path.
        """
        max_decode = self.get_max_output_tokens()

        imagegen_enabled = self.config.image_gen is not None

        def _thinker_prefill_node(input_names: list[str]) -> GraphNode:
            outputs = [GraphEdge(
                next_node=EMIT_TO_CLIENT,
                name="new_token",
                output_modality="text",
                persist=True,
            )]
            if imagegen_enabled:
                # Image-generation requests carry an <imagePatch> query-token
                # block in the prompt; the thinker computes hidden states at
                # those positions during prefill and streams them to the
                # ImageGen partition. The submodule only populates this output
                # when the request actually asked for an image (otherwise the
                # edge carries nothing and the FixedChunkPolicy keeps the
                # consumer idle until producer-done → request_done).
                outputs.append(
                    StreamingGraphEdge(
                        next_node="ImageGen",
                        name="thinker_hidden_states",
                        target_partition="ImageGen",
                    )
                )
            return GraphNode(
                name="Thinker",
                input_names=input_names,
                outputs=outputs,
            )

        prefill_text = _thinker_prefill_node(["text_inputs"])

        # Audio prefill: encoder consumes (audio_features, audio_seqlens)
        # and emits ``audio_embeds`` → Thinker. The Thinker submodule's
        # prefill_audio dispatch wraps that with audio_start/audio_end
        # sentinel embeds and builds text-like 3D MRoPE positions.
        prefill_audio = Sequential([
            GraphNode(
                name="audio_encoder",
                input_names=["audio_features", "audio_seqlens"],
                outputs=[GraphEdge(next_node="Thinker", name="audio_embeds")],
            ),
            _thinker_prefill_node(["audio_embeds"]),
        ])

        # Vision prefill (image): encoder takes (pixel_values,
        # image_grid_thw) and emits ``vision_embeds``. The Thinker still
        # needs the grid for its 3D MRoPE math, so route grid_thw
        # straight into the Thinker via a parallel edge from the
        # conductor's initial inputs (see _get_thinker_prefill_inputs).
        prefill_vision = Sequential([
            GraphNode(
                name="vision_encoder",
                input_names=["pixel_values", "image_grid_thw"],
                outputs=[GraphEdge(next_node="Thinker", name="vision_embeds")],
            ),
            _thinker_prefill_node(["vision_embeds", "image_grid_thw"]),
        ])

        # Video prefill: same encoder, plus video_second_per_grid for the
        # timestamp-scaled temporal positions. The Thinker dispatches on
        # walk name (prefill_video) so it picks video_start/video_end
        # sentinels instead of image_*.
        prefill_video = Sequential([
            GraphNode(
                name="vision_encoder",
                input_names=["pixel_values", "image_grid_thw"],
                outputs=[GraphEdge(next_node="Thinker", name="vision_embeds")],
            ),
            _thinker_prefill_node([
                "vision_embeds", "image_grid_thw", "video_second_per_grid",
            ]),
        ])

        # Thinker decode loop — same shape as step 3f's `decode` walk,
        # renamed for symmetry with the prefill walks. When the talker is
        # available, each decoded token additionally streams to the Talker
        # partition as ``thinker_tokens`` (the Talker accumulates the full
        # text then re-tokenizes + generates audio in one shot — Ming's
        # bridge passes detokenized text, not hidden states).
        talker_enabled = self.config.talker is not None
        decode_outputs = [
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
        ]
        if talker_enabled:
            decode_outputs.append(
                StreamingGraphEdge(
                    next_node="Talker",
                    name="thinker_tokens",
                    target_partition="Talker",
                )
            )
        thinker_decode = Loop(
            name="thinker_decode_loop",
            section=GraphNode(
                name="Thinker",
                input_names=["text_inputs"],
                outputs=decode_outputs,
            ),
            max_iters=max_decode,
            outputs=[],
        )
        walks: dict[str, GraphSection] = {
            "prefill_text": prefill_text,
            "prefill_audio": prefill_audio,
            "prefill_vision": prefill_vision,
            "prefill_video": prefill_video,
            "thinker_decode": thinker_decode,
        }
        if talker_enabled:
            # Single Talker node: consume the streamed thinker tokens,
            # run the full AR-decode + VAE-decode internally, emit one
            # audio chunk to the client.
            walks["talker"] = GraphNode(
                name="Talker",
                input_names=["thinker_tokens"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="audio_chunk",
                        output_modality="audio",
                    ),
                ],
            )
        if self.config.image_gen is not None:
            # Single ImageGen node: consume the thinker hidden states at the
            # <imagePatch> query-token positions (streamed from the Thinker as
            # ``thinker_hidden_states``), run the full diffusion denoise + VAE
            # decode internally, emit one image to the client. Like the Talker,
            # the per-request work is one shot (scheduler-determined step
            # count), so no conductor decode loop is needed.
            walks["imagegen"] = GraphNode(
                name="ImageGen",
                input_names=["thinker_hidden_states"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="image",
                        output_modality="image",
                    ),
                ],
            )
        return walks

    def get_partition_topology(self) -> PartitionTopology:
        partitions = ["Thinker"]
        connections = []
        if self.config.talker is not None:
            partitions.append("Talker")
            connections.append(
                Connection(
                    from_partition="Thinker",
                    to_partition="Talker",
                    edge_name="thinker_tokens",
                    # The talker needs the FULL text before it generates.
                    # continue_after_done=True keeps the Talker partition
                    # alive past the Thinker's text-EOS so it can fire its
                    # single generation once all tokens have arrived.
                    chunk_policy_factory=lambda: FixedChunkPolicy(
                        chunk_size=1, continue_after_done=True,
                    ),
                )
            )
        if self.config.image_gen is not None:
            partitions.append("ImageGen")
            connections.append(
                Connection(
                    from_partition="Thinker",
                    to_partition="ImageGen",
                    edge_name="thinker_hidden_states",
                    # The imagegen node fires once, after the thinker has
                    # produced the query-token hidden states. continue_after_done
                    # keeps the partition alive until that single handoff lands.
                    chunk_policy_factory=lambda: FixedChunkPolicy(
                        chunk_size=1, continue_after_done=True,
                    ),
                )
            )
        if not connections:
            return PartitionTopology(partitions=["Thinker"], connections=[])
        return PartitionTopology(partitions=partitions, connections=connections)

    def get_partitions(self) -> list[PartitionDefinition]:
        thinker = PartitionDefinition(
            name="Thinker",
            graph_walks={
                "prefill_text", "prefill_audio",
                "prefill_vision", "prefill_video",
                "thinker_decode",
            },
            initial_walk="prefill_text",
            producer_partitions=[],
        )
        partitions = [thinker]
        if self.config.talker is not None:
            partitions.append(
                PartitionDefinition(
                    name="Talker",
                    graph_walks={"talker"},
                    initial_walk=None,
                    producer_partitions=["Thinker"],
                )
            )
        if self.config.image_gen is not None:
            partitions.append(
                PartitionDefinition(
                    name="ImageGen",
                    graph_walks={"imagegen"},
                    initial_walk=None,
                    producer_partitions=["Thinker"],
                )
            )
        return partitions

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        """Talker AudioVAE sample rate (44.1 kHz on the released ckpt)."""
        if modality == "audio" and self.config.talker is not None:
            return self.config.talker.vae_sample_rate
        return super().get_output_sample_rate(modality)

    # ------------------------------------------------------------------
    # Prefill scheduling — mirrors qwen3_omni's _build_thinker_prefill_schedule
    # ------------------------------------------------------------------

    def _build_thinker_prefill_schedule(
        self,
        input_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[tuple[str, dict[str, TensorPointerInfo]]]:
        """Walk-name + per-input tensor map per modality, in input order.

        Mirrors qwen3_omni's helper: each ``input_modalities`` entry
        yields one schedule step. The audio walk needs
        ``audio_features`` (+ optional ``audio_seqlens``); image / video
        walks need ``pixel_values`` + ``image_grid_thw``; video walks
        also take ``video_second_per_grid``. Steps the conductor's
        ``input_signals`` does not actually have (e.g. ``audio`` listed
        but no ``audio_features`` provided) are silently skipped.
        """
        texts = input_signals.get("text_inputs", [])
        audio_features = input_signals.get("audio_features", [])
        audio_seqlens = input_signals.get("audio_seqlens", [])
        pixel_values = input_signals.get("pixel_values", [])
        image_grid_thws = input_signals.get("image_grid_thw", [])
        # Video uses pixel_values_videos in HF; accept both keys
        # for parity with qwen3_omni's helper.
        pixel_values_videos = input_signals.get("pixel_values_videos", [])
        video_grid_thws = input_signals.get("video_grid_thw", [])
        video_second_per_grid = input_signals.get("video_second_per_grid", [])

        schedule: list[tuple[str, dict[str, TensorPointerInfo]]] = []
        text_idx = audio_idx = vision_idx = video_idx = 0
        for mod in input_modalities:
            if mod == "text":
                if text_idx < len(texts):
                    schedule.append((
                        "prefill_text",
                        {"text_inputs": texts[text_idx]},
                    ))
                    text_idx += 1
            elif mod == "audio":
                if audio_idx < len(audio_features):
                    entry: dict[str, TensorPointerInfo] = {
                        "audio_features": audio_features[audio_idx],
                    }
                    if audio_idx < len(audio_seqlens):
                        entry["audio_seqlens"] = audio_seqlens[audio_idx]
                    schedule.append(("prefill_audio", entry))
                    audio_idx += 1
            elif mod == "image":
                if vision_idx < len(pixel_values):
                    entry = {"pixel_values": pixel_values[vision_idx]}
                    if vision_idx < len(image_grid_thws):
                        entry["image_grid_thw"] = image_grid_thws[vision_idx]
                    schedule.append(("prefill_vision", entry))
                    vision_idx += 1
            elif mod == "video":
                if video_idx < len(pixel_values_videos):
                    entry = {"pixel_values": pixel_values_videos[video_idx]}
                    if video_idx < len(video_grid_thws):
                        entry["image_grid_thw"] = video_grid_thws[video_idx]
                    if video_idx < len(video_second_per_grid):
                        entry["video_second_per_grid"] = video_second_per_grid[video_idx]
                    schedule.append(("prefill_video", entry))
                    video_idx += 1
        return schedule

    def _get_thinker_prefill_inputs(
        self,
        metadata: CurrentForwardConductorMetadata,
        input_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[GraphEdge]:
        """Build the GraphEdges for the current prefill step.

        For audio/vision/video walks the encoder is the first graph
        node, so each ``input_name`` from the schedule entry routes
        to that encoder; ``image_grid_thw`` and ``video_second_per_grid``
        also need to reach the Thinker (for the 3D MRoPE math) and
        get their own parallel edges to ``Thinker``.
        """
        schedule = metadata.kwargs["prefill_schedule"]
        step = metadata.kwargs["prefill_step"]
        walk_name, tensor_dict = schedule[step]

        if walk_name == "prefill_text":
            target_node = "Thinker"
        elif walk_name == "prefill_audio":
            target_node = "audio_encoder"
        elif walk_name in ("prefill_vision", "prefill_video"):
            target_node = "vision_encoder"
        else:
            raise ValueError(f"Unrecognized prefill walk: {walk_name!r}")

        edges: list[GraphEdge] = []
        for input_name, tensor_info in tensor_dict.items():
            if input_name in ("image_grid_thw", "video_second_per_grid"):
                # These go to the Thinker, not the encoder — handled below.
                continue
            edge = GraphEdge(next_node=target_node, name=input_name)
            edge.tensor_info = [tensor_info]
            edges.append(edge)

        if walk_name in ("prefill_vision", "prefill_video"):
            # Vision encoder needs image_grid_thw, AND the Thinker needs
            # it for 3D position math. Emit a duplicate edge to each.
            if "image_grid_thw" in tensor_dict:
                enc_edge = GraphEdge(next_node="vision_encoder", name="image_grid_thw")
                enc_edge.tensor_info = [tensor_dict["image_grid_thw"]]
                edges.append(enc_edge)
                thinker_edge = GraphEdge(next_node="Thinker", name="image_grid_thw")
                thinker_edge.tensor_info = [tensor_dict["image_grid_thw"]]
                edges.append(thinker_edge)
            if walk_name == "prefill_video" and "video_second_per_grid" in tensor_dict:
                vspg_edge = GraphEdge(next_node="Thinker", name="video_second_per_grid")
                vspg_edge.tensor_info = [tensor_dict["video_second_per_grid"]]
                edges.append(vspg_edge)

        return edges

    # ------------------------------------------------------------------
    # Forward-pass arg builders — multimodal prefill scheduling (step 5c)
    # ------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        if partition_name == "Talker":
            # Talker is a consumer partition: it has no initial inputs of
            # its own — it self-triggers when the Thinker's streamed
            # ``thinker_tokens`` arrive. Audio output only.
            audio_output = "audio" in output_modalities
            full_metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="talker",
                is_prefill=False,
            )
            return ForwardPassArgs(
                full_metadata=full_metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=not audio_output,
            )
        if partition_name == "ImageGen":
            # ImageGen is a consumer partition: it self-triggers when the
            # Thinker's streamed ``thinker_hidden_states`` arrive. Image output
            # only.
            image_output = "image" in output_modalities
            full_metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="imagegen",
                is_prefill=False,
            )
            return ForwardPassArgs(
                full_metadata=full_metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=not image_output,
            )
        if partition_name != "Thinker":
            raise ValueError(f"Unknown partition: {partition_name!r}")
        schedule = self._build_thinker_prefill_schedule(
            input_modalities, input_signals,
        )
        if not schedule:
            # No modalities provided — fall through to decode immediately.
            # The conductor will report request_done after the first decode
            # step returns nothing. Useful for empty-prompt smoke tests.
            full_metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="thinker_decode",
                is_prefill=False,
            )
            return ForwardPassArgs(
                full_metadata=full_metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        first_walk, _ = schedule[0]
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=first_walk,
            is_prefill=True,
            kwargs={
                "prefill_schedule": schedule,
                "prefill_step": 0,
            },
        )
        inputs = self._get_thinker_prefill_inputs(full_metadata, input_signals)
        unpersist_tensors = sum(
            (inp.tensor_info for inp in inputs), start=[],
        )
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={
                "is_prefill": True,
                "is_last_prefill": len(schedule) == 1,
            },
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        new_tokens: dict[str, list[int]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Thinker state machine: walk schedule → thinker_decode → done.

        Each prefill step pops the next walk from
        ``metadata.kwargs["prefill_schedule"]``. When all prefill steps
        are done we transition to ``thinker_decode``; when the decode
        loop unwinds (the loop's max_iters or check_stop fired) we
        return ``request_done=True``.

        For the Talker partition the state machine is trivial: it runs
        its single ``talker`` walk (one Talker node consuming the streamed
        thinker tokens, generating audio internally) and is then done.

        Thinker shape mirrors ``mminf/model/qwen3_omni/qwen3_omni_model.py:765+``.
        """
        if partition_name == "Talker":
            return self._get_talker_forward(partition_metadata, incoming_connections)
        if partition_name == "ImageGen":
            return self._get_imagegen_forward(partition_metadata, incoming_connections)
        if partition_name != "Thinker":
            raise ValueError(f"Unknown partition: {partition_name!r}")

        if partition_metadata.is_prefill:
            step = partition_metadata.kwargs["prefill_step"] + 1
            schedule = partition_metadata.kwargs["prefill_schedule"]
            if step < len(schedule):
                partition_metadata.kwargs["prefill_step"] = step
                partition_metadata.graph_walk = schedule[step][0]
            else:
                partition_metadata.is_prefill = False
                partition_metadata.graph_walk = "thinker_decode"
        elif partition_metadata.graph_walk == "thinker_decode":
            # Decode loop unwound — Thinker is fully done with this request.
            return ForwardPassArgs(
                full_metadata=partition_metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        if partition_metadata.is_prefill:
            schedule = partition_metadata.kwargs["prefill_schedule"]
            step = partition_metadata.kwargs["prefill_step"]
            is_last_prefill = step == len(schedule) - 1
            inputs = self._get_thinker_prefill_inputs(
                partition_metadata, persist_signals,
            )
        else:
            is_last_prefill = False
            edge = GraphEdge(next_node="Thinker", name="text_inputs")
            edge.tensor_info = persist_signals.get("new_token", [])
            inputs = [edge]

        unpersist_tensors = sum(
            (inp.tensor_info for inp in inputs), start=[],
        )
        return ForwardPassArgs(
            full_metadata=partition_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={
                "is_prefill": partition_metadata.is_prefill,
                "is_last_prefill": is_last_prefill,
            },
        )

    def _get_talker_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        incoming_connections: list[StreamingConnectionState] | None,
    ) -> ForwardPassArgs:
        """Talker partition state machine — runs once, then done.

        The Talker is a single stateless node: it consumes the full
        stream of ``thinker_tokens`` (gated by the FixedChunkPolicy with
        continue_after_done=True so it stays alive past the Thinker's
        text EOS), re-tokenizes the accumulated text, and generates one
        audio chunk inside ``TalkerSubmodule.forward``. We fire that walk
        once the producer (Thinker) is done, then report request_done.
        """
        conn = incoming_connections[0] if incoming_connections else None
        producer_done = conn.producer_done if conn else True

        # Wait until the Thinker has finished emitting all its tokens — the
        # talker needs the FULL text before it can generate. Until then,
        # return an empty no-op step (the conductor re-invokes us as more
        # tokens stream in).
        if not producer_done:
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
            )

        if metadata.kwargs.get("talker_fired"):
            # Already generated — the request is complete for this partition.
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        metadata.kwargs["talker_fired"] = True
        metadata.graph_walk = "talker"
        edge = GraphEdge(next_node="Talker", name="thinker_tokens")
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=[edge],
            unpersist_tensors=[],
        )

    def _get_imagegen_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        incoming_connections: list[StreamingConnectionState] | None,
    ) -> ForwardPassArgs:
        """ImageGen partition state machine — runs once, then done.

        Mirrors :meth:`_get_talker_forward`: the ImageGen node is a single
        stateless diffusion node consuming the Thinker's streamed
        ``thinker_hidden_states`` (the query-token hidden states sliced at the
        ``<imagePatch>`` positions). The FixedChunkPolicy with
        continue_after_done=True keeps the partition alive until the Thinker has
        emitted them; we then fire the ``imagegen`` walk once (full denoise +
        VAE decode happen inside ``ImageGenSubmodule.forward``) and report
        request_done.
        """
        conn = incoming_connections[0] if incoming_connections else None
        producer_done = conn.producer_done if conn else True

        # Wait until the Thinker has produced the query-token hidden states.
        if not producer_done:
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
            )

        if metadata.kwargs.get("imagegen_fired"):
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        metadata.kwargs["imagegen_fired"] = True
        metadata.graph_walk = "imagegen"
        edge = GraphEdge(next_node="ImageGen", name="thinker_hidden_states")
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=[edge],
            unpersist_tensors=[],
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
        """Build text_inputs + modality tensors for the prefill schedule.

        Strategy mirrors qwen3_omni's process_prompt (step 7 of porting
        notes): apply the chat template to TEXT-ONLY messages (so the
        tokenizer doesn't insert placeholder tokens we'd later have to
        strip), then run the image / audio sub-processors separately
        on each modality input.

        The Ming chat template (`tokenizer.apply_chat_template`) is the
        jinja path that accepts OpenAI roles (user / assistant /
        system) and rewrites them to Ming's HUMAN / ASSISTANT / SYSTEM.
        The processor's Python `apply_chat_template` (`BailingMM2Processor`)
        is stricter and asserts on lowercase roles — see PORTING_NOTES
        "Role-handling nuance". Using the tokenizer path keeps the
        interface OpenAI-compatible.

        Input shape (`tensors`):

          * ``image_inputs`` — list of CHW float32 [0, 1] tensors (one
            per image). Converted to HWC uint8 [0, 255] before the
            image processor (the upstream BailingMM2ImageProcessor
            assumes uint8; double-rescaling near-zeros the tensor).
          * ``audio_inputs`` — list of ``(waveform, sampling_rate)``
            tuples OR list of 1-D float tensors (sample rate inferred
            from the processor's default — 16 kHz on the released ckpt).
          * ``video_inputs`` — list of 4-D (T, C, H, W) float tensors.
            Currently treated like a stack of images via the image
            processor's video path; per-frame timestamp scaffolding
            (``video_second_per_grid``) defaults to 1.0 unless an
            ``input_metadata["video"][i]["second_per_grid"]`` override
            is supplied via ``**kwargs``.

        Output shape — keys consumed by
        ``_build_thinker_prefill_schedule`` in step 5c:

          * ``text_inputs`` — list of 1-D long tensors.
          * ``pixel_values``, ``image_grid_thw`` — one entry per image.
          * ``pixel_values_videos``, ``video_grid_thw``,
            ``video_second_per_grid`` — one entry per video clip.
          * ``audio_features``, ``audio_seqlens`` — one entry per
            audio clip; ``audio_features`` is (n_mels, T) and
            ``audio_seqlens`` is a length-1 int tensor.
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "MingFlashOmniModel.process_prompt called but tokenizer "
                "is not loaded. See _warn_tokenizer_unavailable for setup."
            )

        result: NameToTensorList = {
            "text_inputs": [],
            "pixel_values": [],
            "image_grid_thw": [],
            "pixel_values_videos": [],
            "video_grid_thw": [],
            "video_second_per_grid": [],
            "audio_features": [],
            "audio_seqlens": [],
        }

        # ----- Text path (always present, even for image-/audio-only
        # turns since the chat template emits role markers + an
        # assistant-prompt suffix the model needs to start decoding).
        if prompt is not None:
            # Image-generation requests append the learnable query-token
            # block (<image><imagePatch>*N</image>) to the prompt — the
            # thinker substitutes its image-gen query embeddings at those
            # positions during forward (step 9). Only when the deploy
            # actually ships an imagegen sub-config and the caller asked
            # for an image output.
            prompt_for_template = prompt
            if "image" in output_modalities and self.config.image_gen is not None:
                from mminf.model.ming_omni_flash.components.prompt_utils import (
                    maybe_expand_image_gen_prompt,
                )
                prompt_for_template = maybe_expand_image_gen_prompt(
                    prompt, num_query_tokens=self.config.image_gen.num_query_tokens,
                )
            messages = [{"role": "user", "content": prompt_for_template}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
            result["text_inputs"].append(input_ids)

        if tensors is None:
            return result

        # ----- Image path
        raw_images = tensors.get("image_inputs", []) or []
        if raw_images:
            self._process_image_inputs(raw_images, result)

        # ----- Video path
        raw_videos = tensors.get("video_inputs", []) or []
        if raw_videos:
            video_metadata = kwargs.get("input_metadata", {}).get("video", [])
            self._process_video_inputs(raw_videos, video_metadata, result)

        # ----- Audio path
        raw_audios = tensors.get("audio_inputs", []) or []
        if raw_audios:
            self._process_audio_inputs(raw_audios, result)

        return result

    # ------------------------------------------------------------------
    # Per-modality helpers (split out so process_prompt stays readable)
    # ------------------------------------------------------------------

    @staticmethod
    def _image_to_processor_input(img: "torch.Tensor"):
        """Convert a CHW float [0,1] tensor to HWC uint8 numpy for HF.

        BailingMM2ImageProcessor (and most HF image processors)
        assume PIL/uint8 inputs with ``do_rescale=True`` by default.
        Passing a float [0,1] tensor would double-rescale it to
        near-zero. Mirror qwen3_omni's conversion (qwen3_omni_model.py:
        1027-1039).
        """
        import numpy as np
        x = img
        if x.dtype.is_floating_point:
            x = (x * 255.0).clamp(0, 255).to(torch.uint8)
        if x.dim() == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)  # CHW -> HWC
        arr = x.cpu().contiguous().numpy()
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr

    def _process_image_inputs(
        self,
        raw_images: list["torch.Tensor"],
        result: NameToTensorList,
    ) -> None:
        if self._processor is None:
            raise RuntimeError(
                "process_prompt: image inputs supplied but processor is None. "
                "See PORTING_NOTES 'Ming source dependency' for setup."
            )
        img_proc = self._processor.image_processor
        for img in raw_images:
            arr = self._image_to_processor_input(img)
            out = img_proc(images=[arr], return_tensors="pt")
            # ``pixel_values`` is (n_patches, C, ph, pw); the encoder
            # consumes it directly. ``image_grid_thw`` is (1, 3).
            result["pixel_values"].append(out["pixel_values"])
            grid = out["image_grid_thw"]
            if not isinstance(grid, torch.Tensor):
                grid = torch.as_tensor(grid)
            result["image_grid_thw"].append(grid[0])

    def _process_video_inputs(
        self,
        raw_videos: list["torch.Tensor"],
        video_metadata: list[dict],
        result: NameToTensorList,
    ) -> None:
        if self._processor is None:
            raise RuntimeError(
                "process_prompt: video inputs supplied but processor is None."
            )
        img_proc = self._processor.image_processor
        # Per-frame timestamp override; default 1.0 second/frame so the
        # Thinker's temporal positions advance once per grid step
        # (matches modeling_bailing_moe_v2.get_rope_index's `else: 1.0`).
        for i, video in enumerate(raw_videos):
            # Convert (T, C, H, W) float [0,1] to (T, H, W, C) uint8.
            frames = []
            for t in range(video.shape[0]):
                frames.append(self._image_to_processor_input(video[t]))
            out = img_proc(
                images=None,
                videos=[frames],
                **({} if not video_metadata else {}),
            )
            result["pixel_values_videos"].append(out["pixel_values_videos"])
            grid = out["video_grid_thw"]
            if not isinstance(grid, torch.Tensor):
                grid = torch.as_tensor(grid)
            result["video_grid_thw"].append(grid[0])
            spg = 1.0
            if i < len(video_metadata):
                spg = float(video_metadata[i].get("second_per_grid", 1.0))
            result["video_second_per_grid"].append(torch.tensor(spg))

    def _process_audio_inputs(
        self,
        raw_audios: list,
        result: NameToTensorList,
    ) -> None:
        if self._processor is None:
            raise RuntimeError(
                "process_prompt: audio inputs supplied but processor is None."
            )
        audio_proc = self._processor.audio_processor
        # Normalise each input into the (waveform, sampling_rate) tuple
        # the processor expects. Accept either:
        #   * raw 1-D float tensor (assume the processor's default SR)
        #   * (waveform_tensor, int sr) tuple
        default_sr = getattr(audio_proc, "sampling_rate", 16000)
        for audio in raw_audios:
            if isinstance(audio, tuple) and len(audio) == 2:
                waveform, sr = audio
            else:
                waveform, sr = audio, default_sr
            if isinstance(waveform, torch.Tensor):
                waveform_np = waveform.detach().cpu().numpy()
            else:
                waveform_np = waveform
            out = audio_proc([(waveform_np, sr)])
            # `audio_feats` is (B, T, n_mels); transpose to (n_mels, T)
            # per clip — that's what the AudioEncoderSubmodule expects
            # for a single-clip prepare_inputs.
            feats = out["audio_feats"]
            if not isinstance(feats, torch.Tensor):
                feats = torch.as_tensor(feats)
            # B=1 per clip in our loop.
            mel = feats[0].transpose(0, 1).contiguous()  # (n_mels, T)
            length = out["audio_feats_lengths"]
            if not isinstance(length, torch.Tensor):
                length = torch.as_tensor(length)
            result["audio_features"].append(mel)
            result["audio_seqlens"].append(length.to(torch.long))

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
        if node_name == "Talker":
            submodule = self._create_talker_submodule(device)
            self._submodule_cache[node_name] = submodule
            return submodule
        if node_name == "ImageGen":
            submodule = self._create_imagegen_submodule(device)
            self._submodule_cache[node_name] = submodule
            return submodule
        if node_name != "Thinker":
            raise ValueError(
                f"Unknown node: {node_name!r}. Registers "
                f"'Thinker', 'vision_encoder', 'audio_encoder', 'Talker', "
                f"'ImageGen'."
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

    def _create_talker_submodule(self, device: str):
        """Build the full talker stack + load weights, wrap in a submodule.

        Assembles Qwen2 LLM + CFM(DiT) + Aggregator + stop/spk heads +
        AudioVAE via the step-6b/6c/6d factories, loads each subtree
        with the step-6f loaders, and wraps the lot in a
        :class:`TalkerSubmodule` around a :class:`TalkerGenerator`.

        The talker colocates on a single rank (no TP) — bf16 to match
        the released ckpt's torch_dtype.
        """
        if self.config.talker is None:
            raise RuntimeError(
                "MingFlashOmniModel: 'Talker' node requested but the snapshot "
                "has no talker/ subdir (thinker-only checkpoint)."
            )
        from mminf.model.ming_omni_flash.components.audio_vae import (
            build_audio_vae,
        )
        from mminf.model.ming_omni_flash.components.talker_dit import (
            build_aggregator,
            build_talker_cfm,
            build_talker_heads,
            build_talker_llm,
        )
        from mminf.model.ming_omni_flash.components.talker_generator import (
            TalkerGenerator,
        )
        from mminf.model.ming_omni_flash.loader import (
            load_talker_aggregator_weights,
            load_talker_audio_vae_weights,
            load_talker_cfm_weights,
            load_talker_heads_weights,
            load_talker_llm_weights,
        )
        from mminf.model.ming_omni_flash.submodules import TalkerSubmodule

        talker = self.config.talker
        dtype = self.get_autocast_dtype()

        llm = build_talker_llm(talker.llm, dtype=dtype, device=device)
        load_talker_llm_weights(llm, self.local_dir, device=device, strict=True)

        cfm = build_talker_cfm(talker, dtype=dtype, device=device)
        load_talker_cfm_weights(cfm, self.local_dir, device=device, strict=True)

        aggregator = build_aggregator(talker, dtype=dtype, device=device)
        load_talker_aggregator_weights(
            aggregator, self.local_dir, device=device, strict=True,
        )

        heads = build_talker_heads(talker, dtype=dtype, device=device)
        load_talker_heads_weights(heads, self.local_dir, device=device, strict=True)

        audio_vae = build_audio_vae(talker.vae, dtype=dtype, device=device)
        load_talker_audio_vae_weights(
            audio_vae, self.local_dir, device=device, strict=True,
        )

        generator = TalkerGenerator(
            talker_config=talker,
            llm=llm,
            cfm=cfm,
            aggregator=aggregator,
            stop_head=heads["stop_head"],
            audio_vae=audio_vae,
        )
        return TalkerSubmodule(
            generator=generator,
            config=self.config,
            text_bridge=self.thinker_text_to_talker_inputs,
        )

    def _create_imagegen_submodule(self, device: str):
        """Build the imagegen diffusion stack + load weights, wrap in a submodule.

        Assembles the ZImage DiT + VAE + scheduler + Qwen2 condition encoder
        (+ optional ByT5) via :meth:`MingImagePipeline.from_checkpoint`, then
        wraps it in an :class:`ImageGenSubmodule`. The imagegen stack colocates
        on a single rank (no TP) — bf16 to match the released ckpt dtype.

        ``from_checkpoint`` lazily imports diffusers (for the VAE + scheduler),
        so this factory only runs on a box where diffusers is healthy and the
        snapshot ships the imagegen tree.
        """
        if self.config.image_gen is None:
            raise RuntimeError(
                "MingFlashOmniModel: 'ImageGen' node requested but the snapshot "
                "has no imagegen tree (no transformer/ + vae/ + connector/)."
            )
        from mminf.model.ming_omni_flash.components.imagegen_pipeline import (
            MingImagePipeline,
        )
        from mminf.model.ming_omni_flash.submodules import ImageGenSubmodule

        pipeline = MingImagePipeline.from_checkpoint(
            self.local_dir,
            self.config.image_gen,
            device=device,
            dtype=self.get_autocast_dtype(),
        )
        return ImageGenSubmodule(pipeline=pipeline, config=self.config)
