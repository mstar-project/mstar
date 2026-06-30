"""
Qwen3OmniModel: 3-partition streaming model for Qwen3-Omni-Moe.

Qwen3-Omni is a dual-AR multimodal model with a Thinker (30B-A3B MoE)
that reasons over text/audio/vision inputs and a Talker (3B-A0.3B MoE)
that converts Thinker hidden states into streaming codec tokens.  A
Code2Wav vocoder converts codec tokens to 24 kHz PCM audio.

Architecture (3 async partitions):
    Thinker  — multimodal encoder + MoE LLM (text, audio, vision prefill -> decode)
    Talker   — smaller MoE LLM that predicts codec tokens from Thinker hidden states
    Code2Wav — vocoder that converts codec tokens to audio waveform

Streaming topology:
    Thinker --[thinker_states, FixedChunkPolicy(1)]--> Talker
    Talker  --[codec_tokens,  FixedChunkPolicy(25)]--> Code2Wav

Conductor-triggered pipelined prefill (Approach C):
    After each Thinker walk completes (prefill_text, prefill_audio,
    prefill_vision, thinker_decode), the conductor sends a
    ``talker_trigger`` to the Talker partition.  During prefill each
    trigger extends the Talker KV cache with the new Thinker hidden
    states.  The final trigger (when thinker_decode starts) tells the
    Talker to sample its first codec token and transition to decode.

Text-only mode:
    When output_modalities does not include "audio", only the Thinker
    partition runs.  Talker and Code2Wav are idle.
"""

import logging
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    StreamingConnectionState,
)
from mstar.engine.base import EngineType
from mstar.engine.kv_store import KVCacheConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import MAX_OUTPUT_TOKENS, ForwardPassArgs, Model, TensorAndMetadata
from mstar.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import Operation, WeightConverter
from mstar.streaming.chunk_policy import FixedChunkPolicy, LeftContextChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge
from mstar.utils.sampling import SamplingConfig

logger = logging.getLogger(__name__)

# Performance defaults (ON), validated vs HF to cos>=0.9999. Opt out for the
# HF-identical baseline: MSTAR_GPU_MEL=0 / MSTAR_GPU_IMAGE_PREPROCESS=0 /
# MSTAR_VLLM_PROMPT_LAYOUT=0 / MSTAR_QWEN3_NATIVE_{AUDIO,VISION}_ENCODER=0.
# (MSTAR_VLLM_AUDIO_SENTINELS, MSTAR_BATCH_VISION_PREFILL stay opt-in.)
# GPU log-mel: mel spectrograms on GPU instead of HF's CPU WhisperFeatureExtractor.
_GPU_MEL = os.environ.get("MSTAR_GPU_MEL", "1") in ("1", "true", "True")


def gpu_log_mel(waveform, mel_filters, window, n_fft, hop):
    """GPU log-mel matching HF ``WhisperFeatureExtractor._np_extract_fbank_features``.

    ``waveform`` (any 1-D-reshapable tensor) -> ``(n_mel, T)`` float32 on the input
    device, ``T = floor(len/hop)`` (== HF's valid, un-padded frame count). Same hann
    window (periodic), center+reflect STFT, power spectrogram, drop-last-frame, log10,
    per-clip max-8 clamp, and (x+4)/4 normalization. Module-level so the parity test
    (test_qwen3_omni_gpu_mel_parity.py) guards the exact production transform.
    """
    wav = waveform.reshape(-1)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window,
                      center=True, pad_mode="reflect", return_complex=True)  # (n_freq, T+1)
    mag = stft[..., :-1].abs().pow(2)                 # drop last frame -> (n_freq, T)
    mel = mel_filters.T @ mag                         # (n_mel, T)
    log = torch.clamp(mel, min=1e-10).log10()
    log = torch.maximum(log, log.max() - 8.0)
    log = (log + 4.0) / 4.0
    return log.to(torch.float32)


def _envflag(name: str, default: bool = False) -> bool:
    """Read a boolean env flag. Accepts 1/true/yes/on; returns ``default`` if unset."""
    import os as _os

    raw = _os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def vllm_prompt_layout_enabled() -> bool:
    """Replicate vLLM-Omni's prompt layout: position the audio block INSIDE the
    user turn BEFORE the instruction text, so the effective Thinker sequence is
    system-turn, then user-turn = [audio][instruction], then assistant. ON by
    default (the benchmarked config; also gives audio M-RoPE h/w parity vs vLLM).
    Set MSTAR_VLLM_PROMPT_LAYOUT=0 for the legacy bare-block layout.
    """
    return _envflag("MSTAR_VLLM_PROMPT_LAYOUT", default=True)


def vllm_audio_sentinels_enabled() -> bool:
    """When ON, wrap the audio span with the real Qwen3-Omni audio marker token
    IDs (151669 ``<|audio_start|>`` / 151670 ``<|audio_end|>``, what vLLM uses)
    instead of the legacy 151647/151648 (mislabeled ``<|audio_bos|>``/
    ``<|audio_eos|>``). Default OFF. Independent of MSTAR_VLLM_PROMPT_LAYOUT so
    both can be tested separately."""
    return _envflag("MSTAR_VLLM_AUDIO_SENTINELS")


def batch_vision_prefill_enabled() -> bool:
    """When ON, allow the Thinker ``prefill_vision`` walk to batch more than
    one request per step (like ``prefill_audio`` / ``prefill_text`` already
    do): the vision tower runs once over the concatenated multi-image batch
    and all requests are prefilled together, cutting Image-to-Text TTFT.

    Default OFF -> ``preprocess`` keeps the single-request assert and the
    eager single-request side-channels, so behavior is byte-identical to the
    pre-batching path. See ``ThinkerSubmodule.preprocess`` /
    ``get_cuda_graph_configs`` in ``submodules.py``.
    """
    return _envflag("MSTAR_BATCH_VISION_PREFILL")


def _tensor_dump_dir() -> str | None:
    """Directory for env-gated intermediate-tensor / token dumps, or None."""
    import os as _os

    return _os.environ.get("MSTAR_DUMP_DIR") or None


def _dump_obj(name: str, obj) -> None:
    """Best-effort dump of a tensor / python object to MSTAR_DUMP_DIR."""
    d = _tensor_dump_dir()
    if not d:
        return
    try:
        import os as _os

        _os.makedirs(d, exist_ok=True)
        path = _os.path.join(d, name)
        if isinstance(obj, torch.Tensor):
            torch.save(obj.detach().cpu(), path)
        else:
            import json as _json

            with open(path, "w") as f:
                _json.dump(obj, f, indent=2)
        logger.info("MSTAR_DUMP: wrote %s", path)
    except Exception as e:  # never let instrumentation break a run
        logger.warning("MSTAR_DUMP failed for %s: %s", name, e)


def _hf_encoder_attn_impl() -> str:
    """Pick the attention implementation for the HF-wrapper encoder fallback.

    The HF ``Qwen3OmniMoe{Audio,Vision}Encoder`` classes hard-fail at init if
    asked for ``flash_attention_2`` without the ``flash_attn`` package present
    (transformers raises ImportError rather than degrading). The native encoder
    path already degrades gracefully to torch SDPA when flash_attn is missing
    (see bagel vit_encoder), so the HF path must do the same to keep both
    variants benchmarkable on the *same* hardware footing. We only request FA2
    when flash_attn actually imports.
    """
    import importlib.util

    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    logger.warning(
        "flash_attn is not available; HF-wrapper encoders will fall back to "
        "torch SDPA (slower than FA2 varlen, but matches the native path's "
        "fallback so the M*-old vs M*-new comparison stays on equal footing)."
    )
    return "sdpa"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_local_hf_snapshot(repo_id: str, cache_dir: str | None = None) -> str:
    """Download (or locate) a HuggingFace snapshot and return the local path."""
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


# GPU image preprocessing: the CPU round-trip to HF's Qwen2VLImageProcessor is the
# biggest I2T TTFT cost (~175 ms). MSTAR_GPU_IMAGE_PREPROCESS=1 runs the identical
# algorithm on-GPU (torchvision bicubic resize, same kernel HF calls); grid_thw is
# bit-exact, pixel_values cos>0.9999. =0 restores the byte-identical HF CPU path.


def _gpu_image_preprocess_enabled() -> bool:
    import os

    # ON by default (benchmarked canonical config); MSTAR_GPU_IMAGE_PREPROCESS=0
    # falls back to the byte-identical HF CPU image processor.
    return os.environ.get("MSTAR_GPU_IMAGE_PREPROCESS", "1") != "0"


def _smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Port of HF ``smart_resize`` (Qwen2VLImageProcessor).  Pure python ints."""
    import math

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _gpu_image_preprocess(
    img: "torch.Tensor",
    *,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean,
    image_std,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Resize + rescale + normalize + patchify a single image on its device.

    ``img`` is a (C, H, W) tensor on the GPU, float in [0, 1] (as produced by
    data_worker) or uint8 in [0, 255].  Returns ``(pixel_values, grid_thw)``
    matching HF's ``Qwen2VLImageProcessor`` output for one image:
    ``pixel_values`` is 2-D ``(grid_h*grid_w, C*temporal*patch*patch)`` and
    ``grid_thw`` is ``(1, 3)`` long ``[[1, grid_h, grid_w]]``.
    """
    from torchvision.transforms.v2.functional import InterpolationMode
    from torchvision.transforms.v2.functional import resize as tv_resize

    # Normalise layout to (C, H, W) and dtype to uint8 in [0, 255], exactly as
    # the CPU path does before handing the array to HF (which then casts to
    # uint8 -> tvF.resize).
    if img.dim() == 3 and img.shape[-1] in (1, 3) and img.shape[0] not in (1, 3):
        img = img.permute(2, 0, 1)  # HWC -> CHW
    if img.dtype.is_floating_point:
        img_u8 = (img * 255.0).clamp(0, 255).to(torch.uint8)
    else:
        img_u8 = img.to(torch.uint8)
    img_u8 = img_u8.contiguous()

    C, H, W = img_u8.shape
    factor = patch_size * merge_size
    h_bar, w_bar = _smart_resize(H, W, factor, min_pixels, max_pixels)

    # Resize on-device with the same torchvision kernel HF's fast backend uses
    # (bicubic + antialias on uint8).
    resized = tv_resize(
        img_u8,
        [h_bar, w_bar],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )

    # Fused rescale (1/255) + normalize, matching HF's
    # ``_fuse_mean_std_and_rescale_factor``: mean *= 255, std *= 255, then
    # (x - mean) / std on the float32 image.
    dev = resized.device
    mean_t = torch.as_tensor(image_mean, device=dev, dtype=torch.float32) * 255.0
    std_t = torch.as_tensor(image_std, device=dev, dtype=torch.float32) * 255.0
    patches = resized.to(torch.float32)
    patches = (patches - mean_t[:, None, None]) / std_t[:, None, None]
    patches = patches.unsqueeze(0)  # (1, C, h_bar, w_bar)

    grid_h, grid_w = h_bar // patch_size, w_bar // patch_size
    patches = patches.reshape(
        1,
        C,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    # [batch, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    flatten_patches = (
        patches.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
        .reshape(
            grid_h * grid_w,
            C * temporal_patch_size * patch_size * patch_size,
        )
    )
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    return flatten_patches, grid_thw


# ---------------------------------------------------------------------------
# Qwen3OmniModel
# ---------------------------------------------------------------------------

class Qwen3OmniModel(Model):
    """Qwen3-Omni: Thinker + Talker + Code2Wav 3-partition streaming model."""

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.cache_dir = cache_dir
        self.model_path_hf = model_path_hf

        self.CONVERTER = [
            WeightConverter(
                source_patterns=[
                    "mlp.experts.*.gate_proj.weight",
                    "mlp.experts.*.up_proj.weight",
                ],
                target_patterns="mlp.experts.gate_up_proj",
                operations=[
                    Operation("MergeModulelist",  dim=0),
                    Operation("Concatenate", dim=1)
                ]
            ),
            WeightConverter(
                source_patterns=["mlp.experts.*.down_proj.weight"],
                target_patterns="mlp.experts.down_proj",
                operations=[Operation("MergeModulelist",  dim=0)],
            ),
        ]

        # Load config from pretrained checkpoint
        from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig

        local_dir = _resolve_local_hf_snapshot(model_path_hf, cache_dir=cache_dir)
        self.config = Qwen3OmniModelConfig.from_pretrained(local_dir)
        self.local_dir = local_dir

        # Allow yaml model_kwargs / constructor kwargs to toggle native encoders
        # (e.g. model_kwargs: {native_audio_encoder: true, native_vision_encoder: true}).
        for _flag in ("native_audio_encoder", "native_vision_encoder"):
            if _flag in kwargs:
                setattr(self.config, _flag, bool(kwargs[_flag]))

        # Tokenizer (Thinker uses a Qwen-family tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir, cache_dir=cache_dir, trust_remote_code=True,
        )

        # Full multimodal processor: combines tokenizer + image_processor +
        # video_processor + audio feature_extractor + chat template support.
        # Used by process_prompt to build the full ChatML prompt with the
        # correct image_pad / audio_pad / video_pad expansion.
        try:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                local_dir, cache_dir=cache_dir, trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(
                "Could not load Qwen3-Omni AutoProcessor (%s); "
                "process_prompt will fall back to raw tokenizer.encode.",
                e,
            )
            self._processor = None

        # Lazy submodule cache -- each worker only loads what it needs
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

        # GPU log-mel state (MSTAR_GPU_MEL=1): cached filterbank + window per
        # device, built lazily on first audio request. Default OFF -> HF path.
        self._gpu_mel_state: dict | None = None

    # -----------------------------------------------------------------------
    # Model ABC: KV cache config
    # -----------------------------------------------------------------------

    def get_kv_cache_config(self) -> list[KVCacheConfig]:
        """Return separate KV cache configs for Thinker and Talker."""
        thinker_cfg = KVCacheConfig(
            num_layers=self.config.thinker_text.num_hidden_layers,
            num_kv_heads=self.config.thinker_text.num_key_value_heads,
            head_dim=self.config.thinker_head_dim,
            max_seq_len=self.config.thinker_text.max_position_embeddings,
            num_qo_heads=self.config.thinker_text.num_attention_heads,
            nodes=["Thinker"]
        )
        talker_cfg = KVCacheConfig(
            num_layers=self.config.talker_text.num_hidden_layers,
            num_kv_heads=self.config.talker_text.num_key_value_heads,
            head_dim=self.config.talker_head_dim,
            max_seq_len=self.config.thinker_text.max_position_embeddings,
            num_qo_heads=self.config.talker_text.num_attention_heads,
            nodes=["Talker"]
        )
        return [thinker_cfg, talker_cfg]

    # -----------------------------------------------------------------------
    # Model ABC: node engine types
    # -----------------------------------------------------------------------

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            "audio_encoder": EngineType.STATELESS,
            "vision_encoder": EngineType.STATELESS,
            "Thinker": EngineType.KV_CACHE,
            "Talker": EngineType.KV_CACHE,
            "Code2Wav": EngineType.STATELESS,
        }

    def get_max_talker_output_tokens(self, **model_kwargs):
        return model_kwargs.get("talker_max_output_tokens", MAX_OUTPUT_TOKENS)

    # -----------------------------------------------------------------------
    # Model ABC: graph walk definitions
    # -----------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphNode | Sequential]:
        """Define all graph walks for the 3-partition architecture.

        Thinker walks:
            prefill_text   - text token embedding + Thinker prefill
            prefill_audio  - audio feature encoding + Thinker prefill
            prefill_vision - vision feature encoding + Thinker prefill
            thinker_decode - autoregressive text token generation

        Talker walks:
            talker_prefill - prefill Talker KV cache from Thinker states
            talker_decode  - autoregressive codec token generation

        Code2Wav walks:
            code2wav_chunk - vocoder streaming decode
        """
        # -- Thinker prefill walks: process inputs and stream hidden states
        #    to the Talker partition via StreamingGraphEdge --
        prefill_text = GraphNode(
            name="Thinker",
            input_names=["text_inputs"],
            outputs=[
                GraphEdge( # last prefill samples a token
                    next_node=EMIT_TO_CLIENT,
                    name="new_token",
                    output_modality="text",
                    persist=True,
                ),
                StreamingGraphEdge(
                    next_node="Talker",
                    name="thinker_states",
                    target_partition="Talker",
                ),
                # The thinker_mask tensor includes two masks: one for multimodal inputs,
                # and one for text inputs (allowing us to cut out the system prompt and
                # assistant history from the talker input)
                StreamingGraphEdge(
                    next_node="Talker",
                    name="thinker_mask",
                    target_partition="Talker",
                ),
            ],
        )

        prefill_audio = Sequential([
            GraphNode(
                name="audio_encoder",
                # audio_seqlens carries the original (pre-padding) length of
                # each audio clip, used by the encoder to compute attention
                # masks and output position IDs.
                input_names=["audio_features", "audio_seqlens"],
                outputs=[GraphEdge(next_node="Thinker", name="audio_embeds")],
            ),
            GraphNode(
                name="Thinker",
                input_names=["audio_embeds"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        persist=True,
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_states",
                        target_partition="Talker",
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_mask",
                        target_partition="Talker",
                    ),
                ],
            ),
        ])

        prefill_vision = Sequential([
            GraphNode(
                name="vision_encoder",
                # image_grid_thw / video_grid_thw carries the (T, H, W) grid
                # dimensions per image/video, used by the encoder to compute
                # spatial position IDs and patch counts.
                input_names=["pixel_values", "image_grid_thw"],
                outputs=[
                    GraphEdge(next_node="Thinker", name="vision_embeds"),
                    GraphEdge(next_node="Thinker", name="deepstack")
                ],
            ),
            GraphNode(
                name="Thinker",
                input_names=["vision_embeds", "deepstack", "video_second_per_grid", "image_grid_thw"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        persist=True,
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_states",
                        target_partition="Talker",
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_mask",
                        target_partition="Talker",
                    ),
                ],
            ),
        ])

        # -- Thinker decode: produces new_token (persist) + thinker_states
        #    (streaming to Talker) --
        thinker_decode = Loop(
            name="thinker_decode_loop",
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
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_states",
                        target_partition="Talker",
                    ),
                    StreamingGraphEdge(
                        next_node="Talker",
                        name="thinker_mask",
                        target_partition="Talker",
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        # -- Talker prefill: receives thinker_states + talker_trigger --
        # Dual-input gating: both thinker_states from streaming and
        # talker_trigger from conductor cross-partition trigger must be
        # present for a prefill step.
        talker_prefill = GraphNode(
            name="Talker",
            input_names=["thinker_states", "thinker_mask", "talker_trigger"],
            outputs=[],
        )

        talker_last_prefill = Sequential(
            sections=[
                GraphNode(
                    name="Talker",
                    input_names=["thinker_states", "thinker_mask", "talker_trigger"],
                    outputs=[
                        GraphEdge(
                            next_node=EMPTY_DESTINATION,
                            name="talker_input_embeds",
                            persist=True
                        ),
                        StreamingGraphEdge(
                            next_node="Code2Wav",
                            name="codec_tokens",
                            target_partition="Code2Wav",
                        ),
                    ]
                )
            ]
        )

        # -- Talker decode: autoregressive codec token generation --
        talker_decode = Loop(
            name="talker_decode_loop",
            section=Sequential(
                sections=[
                    GraphNode(
                        name="Talker",
                        input_names=["thinker_states", "thinker_mask", "talker_input_embeds"],
                        outputs=[
                            GraphEdge(
                                next_node="Talker",
                                name="talker_input_embeds",
                            ),
                            StreamingGraphEdge(
                                next_node="Code2Wav",
                                name="codec_tokens",
                                target_partition="Code2Wav",
                            ),
                        ],
                    )
                ]
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        # -- Code2Wav chunk: vocoder streaming decode --
        code2wav_chunk = GraphNode(
            name="Code2Wav",
            input_names=["codec_tokens"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="audio_chunk",
                    output_modality="audio",
                ),
            ],
        )

        return {
            "prefill_text": prefill_text,
            "prefill_audio": prefill_audio,
            "prefill_vision": prefill_vision,
            "thinker_decode": thinker_decode,
            "talker_prefill": talker_prefill,
            "talker_last_prefill": talker_last_prefill,
            "talker_decode": talker_decode,
            "code2wav_chunk": code2wav_chunk,
        }

    # -----------------------------------------------------------------------
    # Partition API: 3-partition streaming topology
    # -----------------------------------------------------------------------

    def get_partitions(self) -> list[PartitionDefinition]:
        return [
            PartitionDefinition(
                name="Thinker",
                graph_walks={
                    "prefill_text", "prefill_audio",
                    "prefill_vision", "thinker_decode",
                },
                initial_walk="prefill_text",
                producer_partitions=[],
            ),
            PartitionDefinition(
                name="Talker",
                graph_walks={"talker_prefill", "talker_last_prefill", "talker_decode"},
                initial_walk="talker_prefill",
                producer_partitions=["Thinker"],
            ),
            PartitionDefinition(
                name="Code2Wav",
                graph_walks={"code2wav_chunk"},
                initial_walk="code2wav_chunk",
                producer_partitions=["Talker"],
            ),
        ]

    def get_partition_topology(self) -> PartitionTopology:
        return PartitionTopology(
            partitions=["Thinker", "Talker", "Code2Wav"],
            connections=[
                Connection(
                    from_partition="Thinker",
                    to_partition="Talker",
                    edge_name="thinker_states",
                    chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1, continue_after_done=True),
                ),
                Connection(
                    from_partition="Thinker",
                    to_partition="Talker",
                    edge_name="thinker_mask",
                    chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1, continue_after_done=True),
                ),
                Connection(
                    from_partition="Talker",
                    to_partition="Code2Wav",
                    edge_name="codec_tokens",
                    chunk_policy_factory=lambda: LeftContextChunkPolicy(
                        chunk=self.config.code2wav.codec_chunk_frames,
                        left_context=self.config.code2wav.codec_left_context_frames,
                    ),
                ),
            ],
        )

    # -----------------------------------------------------------------------
    # Model ABC: sampling config
    # -----------------------------------------------------------------------
    def get_sampling_config(
        self, node_name: str,
        model_kwargs: dict | None = None,
    )  -> SamplingConfig | None:
        if model_kwargs is None:
            model_kwargs = {}

        if node_name == "Thinker":
            temperature = model_kwargs.get("thinker_temperature", 0.7)
            top_p = model_kwargs.get("thinker_top_p", 0.9)
            # only apply ignore_eos to the thinker
            ignore_eos = model_kwargs.get("ignore_eos", False)
            return SamplingConfig(
                vocab_size=self.config.thinker_text.vocab_size,
                temperature=temperature, top_p=top_p,
                ignore_eos=ignore_eos
            )
        if node_name == "Talker":
            temperature = model_kwargs.get("talker_temperature", 0.9)
            top_k = model_kwargs.get("talker_top_k", 50)
            top_p = model_kwargs.get("talker_top_p", 1.0)
            repetition_penalty = model_kwargs.get("talker_repetition_penalty", 1.05)
            return SamplingConfig(
                vocab_size=self.config.talker_text.vocab_size,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty
            )
        # fallback to default config
        return SamplingConfig()

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        # Qwen3-Omni's Code2Wav vocoder emits speech at 24 kHz.
        return 24000

    # -----------------------------------------------------------------------
    # Model ABC: initial forward pass args
    # -----------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        audio_output = "audio" in output_modalities

        if model_kwargs is None:
            model_kwargs = {}

        if partition_name == "Thinker":
            return self._get_thinker_initial_args(
                input_modalities, output_modalities,
                input_signals, model_kwargs or {},
            )
        elif partition_name == "Talker":
            # Talker starts in prefill mode
            full_metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="talker_prefill",
                is_prefill=True,
                kwargs={
                    "audio_output": audio_output,
                    "talker_prefill_done": False,
                    # The Talker consumes one thinker_states chunk per Thinker
                    # prefill walk, so this MUST equal the actual number of
                    # Thinker prefill walks -- not len(input_modalities).  The
                    # vLLM-layout path (MSTAR_VLLM_PROMPT_LAYOUT=1) splits text
                    # into prefix+suffix around the audio, producing an extra
                    # prefill_text walk; deriving the count from the real
                    # schedule keeps the Talker's last-prefill detection aligned.
                    "num_thinker_prefill_steps": len(
                        self._build_thinker_prefill_schedule(
                            input_modalities, input_signals,
                        )
                    ),
                    "prefill_chunks_processed": 0,
                    "voice": model_kwargs.get("voice", "Ethan"),
                    "talker_max_tokens": self.get_max_talker_output_tokens(**model_kwargs),
                },
            )
            return ForwardPassArgs(
                full_metadata=full_metadata,
                inputs=[GraphEdge(next_node="Talker", name="talker_trigger")] if audio_output else [],
                unpersist_tensors=[],
                request_done="audio" not in output_modalities,
                step_metadata={
                    "voice": model_kwargs.get("voice", "Ethan"),
                    "talker_max_tokens": full_metadata.kwargs.get("talker_max_tokens")
                }
            )
        elif partition_name == "Code2Wav":
            # Code2Wav starts with code2wav_chunk walk but no inputs --
            # it self-triggers via StreamBuffer when codec tokens arrive.
            full_metadata = CurrentForwardConductorMetadata(
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                graph_walk="code2wav_chunk",
                is_prefill=False,
            )
            return ForwardPassArgs(
                full_metadata=full_metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done="audio" not in output_modalities,
            )
        raise ValueError(f"Unknown partition: {partition_name!r}")

    def _get_thinker_initial_args(
        self,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> ForwardPassArgs:
        """Build initial ForwardPassArgs for the Thinker partition.

        Constructs a prefill schedule from the input modalities, then
        begins the first walk in that schedule (always prefill_text).
        """
        audio_output = "audio" in output_modalities

        # Build prefill schedule: list of (graph_walk_name, tensor_info)
        schedule = self._build_thinker_prefill_schedule(
            input_modalities, input_signals,
        )

        first_walk = schedule[0][0] if schedule else "thinker_decode"
        is_last_prefill = (schedule and len(schedule) == 1)

        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=first_walk,
            is_prefill=bool(schedule),
            kwargs={
                "prefill_schedule": schedule,
                "prefill_step": 0,
                "audio_output": audio_output,
            },
        )

        # First walk inputs
        inputs = self._get_thinker_prefill_inputs(full_metadata, input_signals)
        unpersist_tensors = sum(
            [inp.tensor_info for inp in inputs], start=[]
        )

        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={
                "is_prefill": True,
                # Tell the Thinker whether to emit thinker_states.  Text only
                # requests skip it to save cross-partition bandwidth.
                "audio_output": audio_output,
                "is_last_prefill": is_last_prefill
            },
        )

    def _build_thinker_prefill_schedule(
        self,
        input_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[tuple[str, dict[str, TensorPointerInfo]]]:
        """Build the sequential prefill schedule for the Thinker.

        Order: [prefill_text] + [prefill_audio if audio inputs] + [prefill_vision if vision inputs]

        Each schedule entry is ``(walk_name, {input_name: tensor_info})``,
        capturing all tensors needed by that step's first node.  For audio
        and vision walks, this includes auxiliary tensors like
        ``audio_seqlens`` and ``image_grid_thw`` that the encoder nodes
        require alongside the primary feature tensor.
        """
        schedule: list[tuple[str, dict[str, TensorPointerInfo]]] = []

        texts = input_signals.get("text_inputs", [])
        audio_features = input_signals.get("audio_features", [])
        audio_seqlens = input_signals.get("audio_seqlens", [])
        pixel_values = input_signals.get("pixel_values", [])
        image_grid_thws = input_signals.get("image_grid_thw", [])
        # video uses pixel_values_videos in HF; we accept both keys here
        pixel_values_videos = input_signals.get("pixel_values_videos", [])
        video_grid_thws = input_signals.get("video_grid_thw", [])
        video_second_per_grid = input_signals.get("video_second_per_grid", [])

        # --- vLLM prompt-layout schedule -----------------------------------
        # process_prompt split text_inputs into [prefix, suffix] and wants the
        # audio interleaved: prefill_text(prefix) -> prefill_audio ->
        # prefill_text(suffix).  This puts the audio block INSIDE the user turn
        # before the instruction, matching vLLM.  Only triggers when the flag
        # is on AND there are exactly the expected two text spans + audio.
        if (
            vllm_prompt_layout_enabled()
            and len(texts) >= 2
            and len(audio_features) >= 1
        ):
            audio_entry: dict[str, TensorPointerInfo] = {
                "audio_features": audio_features[0],
            }
            if len(audio_seqlens) >= 1:
                audio_entry["audio_seqlens"] = audio_seqlens[0]
            schedule.append(("prefill_text", {"text_inputs": texts[0]}))
            schedule.append(("prefill_audio", audio_entry))
            schedule.append(("prefill_text", {"text_inputs": texts[1]}))
            return schedule

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
                # Video uses pixel_values_videos + video_grid_thw, but the
                # graph node still consumes them under the "pixel_values" /
                # "image_grid_thw" input names (the vision encoder is shared).
                if video_idx < len(pixel_values_videos):
                    entry = {"pixel_values": pixel_values_videos[video_idx]}
                    if video_idx < len(video_grid_thws):
                        entry["image_grid_thw"] = video_grid_thws[video_idx]
                    if video_idx < len(video_second_per_grid):
                        entry["video_second_per_grid"] = video_second_per_grid[video_idx]
                    schedule.append(("prefill_vision", entry))
                    video_idx += 1

        # Robustness guard: an empty user prompt (e.g. greedy T2S/I2S/A2S with no
        # caption) arrives with templated text_inputs present but "text" absent from
        # input_modalities, so the loop above schedules no prefill_text -> empty
        # schedule (IndexError) and the Talker expects zero thinker_states. Prefill
        # every templated text_inputs regardless; a no-op for normal requests.
        while text_idx < len(texts):
            schedule.append(("prefill_text", {"text_inputs": texts[text_idx]}))
            text_idx += 1

        return schedule

    def _get_thinker_prefill_inputs(
        self,
        metadata: CurrentForwardConductorMetadata,
        input_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[GraphEdge]:
        """Construct input GraphEdges for the current Thinker prefill step.

        Each schedule entry maps an ``(walk_name, {input_name: tensor_info})``.
        We emit one GraphEdge per input so that auxiliary tensors like
        ``audio_seqlens`` and ``image_grid_thw`` reach the encoder node
        alongside the primary feature tensor.
        """
        schedule = metadata.kwargs["prefill_schedule"]
        step = metadata.kwargs["prefill_step"]
        walk_name, tensor_dict = schedule[step]

        # Determine the target node — for audio/vision, the first node in
        # the Sequential walk is the encoder (not the Thinker).
        if walk_name == "prefill_text":
            target_node = "Thinker"
        elif walk_name == "prefill_audio":
            target_node = "audio_encoder"
        elif walk_name == "prefill_vision":
            target_node = "vision_encoder"
        else:
            raise ValueError(f"Unrecognized prefill walk: {walk_name}")

        edges: list[GraphEdge] = []
        for input_name, tensor_info in tensor_dict.items():
            if input_name == "video_second_per_grid":
                continue # goes directly to Thinker
            edge = GraphEdge(next_node=target_node, name=input_name)
            edge.tensor_info = [tensor_info]
            edges.append(edge)

        if walk_name == "prefill_vision":
            for key in ["image_grid_thw", "video_second_per_grid"]:
                edge = GraphEdge(next_node="Thinker", name=key)
                if key in tensor_dict:
                    edge.tensor_info = [tensor_dict[key]]
                edges.append(edge)
        return edges

    # -----------------------------------------------------------------------
    # Model ABC: partition forward pass args (STATE MACHINE)
    # -----------------------------------------------------------------------

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        if partition_name == "Thinker":
            return self._get_thinker_forward(
                partition_metadata, persist_signals,
            )
        elif partition_name == "Talker":
            return self._get_talker_forward(
                partition_metadata, persist_signals,
                incoming_connections,
            )
        elif partition_name == "Code2Wav":
            conn = incoming_connections[0] if incoming_connections else None
            return self._get_code2wav_forward(
                partition_metadata, conn,
            )
        raise ValueError(f"Unknown partition: {partition_name!r}")

    # -- Thinker state machine ---------------------------------------------

    def _get_thinker_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
    ) -> ForwardPassArgs:
        """Thinker partition state machine.

        1. Build prefill schedule: [prefill_text] + [prefill_audio] + [prefill_vision]
        2. Pop walks from schedule until done
        3. Transition to thinker_decode
        4. Each decode step: check new_token for EOS (im_end_token_id)
        5. On EOS: request_done=True for Thinker
        """

        if metadata.is_prefill:
            # Advance prefill schedule
            step = metadata.kwargs["prefill_step"] + 1
            schedule = metadata.kwargs["prefill_schedule"]

            if step < len(schedule):
                # More prefill steps remaining
                metadata.kwargs["prefill_step"] = step
                metadata.graph_walk = schedule[step][0]
            else:
                # All prefill done -- transition to thinker_decode
                metadata.is_prefill = False
                metadata.graph_walk = "thinker_decode"

        elif metadata.graph_walk == "thinker_decode":
            # if the decode loop returns to conductor, the thinker is fully done
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )


        if metadata.is_prefill:
            # Still in prefill -- delegate to _get_thinker_prefill_inputs
            # which handles the (walk_name, {input_name: tensor_info}) schedule
            # entry format and emits one GraphEdge per input (so auxiliary
            # tensors like image_grid_thw / audio_seqlens reach the encoder
            # alongside the primary feature tensor).
            schedule = metadata.kwargs["prefill_schedule"]
            step = metadata.kwargs["prefill_step"]
            is_last_prefill = (step == len(schedule) - 1)
            inputs = self._get_thinker_prefill_inputs(metadata, persist_signals)
        else:
            # Decode: previous token feeds back as text_inputs
            is_last_prefill = False
            edge = GraphEdge(next_node="Thinker", name="text_inputs")
            edge.tensor_info = persist_signals.get("new_token", [])
            inputs = [edge]

        unpersist_tensors = sum(
            [inp.tensor_info for inp in inputs], start=[]
        )

        step_metadata = {
            "is_prefill": metadata.is_prefill,
            "is_last_prefill": is_last_prefill,
            # Persist the audio_output flag across every Thinker step so
            # the submodule can gate thinker_states emission.  Default True
            # for backwards compatibility with callers that never set it.
            "audio_output": metadata.kwargs.get("audio_output", True),
        }

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=step_metadata,
        )

    # -- Talker state machine ----------------------------------------------

    def _get_talker_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Talker partition state machine.

        1. While prefill: return empty inputs (wait for cross-partition trigger)
           - When trigger arrives with is_last_prefill=False:
             extend KV cache only, no outputs
           - When trigger arrives with is_last_prefill=True:
             sample first codec token, produce all_codes
        2. After last prefill produces all_codes: transition to talker_decode
           - Set graph_walk="talker_decode", is_prefill=False
           - Return all_codes as input edge (conductor-driven)
        3. Each decode step: check all_codes for codec_eos
           - If codec_eos: request_done=True for Talker
           - Else: return all_codes as input again (loop)
        """
        if metadata.graph_walk == "talker_prefill":
            metadata.kwargs["prefill_chunks_processed"] += 1
            is_last_prefill = metadata.kwargs["num_thinker_prefill_steps"] == \
                 metadata.kwargs["prefill_chunks_processed"]
            metadata.graph_walk = "talker_last_prefill" if is_last_prefill else "talker_prefill"
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[GraphEdge(next_node="Talker", name="talker_trigger")],
                unpersist_tensors=[],
                step_metadata={
                    "is_prefill": True,
                    # voice is used for the last prefill
                    "voice": metadata.kwargs.get("voice", "Ethan"),
                    "talker_max_tokens": metadata.kwargs.get("talker_max_tokens")
                },
            )
        elif metadata.graph_walk == "talker_last_prefill":
            metadata.is_prefill = False
            metadata.graph_walk = "talker_decode"
            metadata.kwargs["talker_prefill_done"] = True

            # Feed talker_input_embeds back as input for first decode step
            edge = GraphEdge(next_node="Talker", name="talker_input_embeds")
            edge.tensor_info = persist_signals["talker_input_embeds"]
            inputs = [edge]
            unpersist_tensors = sum(
                [inp.tensor_info for inp in inputs], start=[]
            )

            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=inputs,
                unpersist_tensors=unpersist_tensors,
                step_metadata={
                    "is_prefill": False,
                    "talker_max_tokens": metadata.kwargs.get("talker_max_tokens")
                },
            )

        elif metadata.graph_walk == "talker_decode":
            # If the decode dynamic loop reaches the conductor, we can end the request.
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        raise ValueError(
            f"Talker in unexpected state: walk={metadata.graph_walk!r}, "
            f"is_prefill={metadata.is_prefill}"
        )

    # -- Code2Wav state machine --------------------------------------------

    def _get_code2wav_forward(
        self,
        metadata: CurrentForwardConductorMetadata,
        conn: StreamingConnectionState | None,
    ) -> ForwardPassArgs:
        """Code2Wav partition: streaming vocoder, self-triggered by StreamBuffer.

        Same pattern as Orpheus SNAC -- the conductor just tracks whether
        there are more codec tokens to process.
        """
        chunk_size = self.config.code2wav.codec_chunk_frames
        metadata.graph_walk = "code2wav_chunk"
        step_metadata = {"consumed_tokens": chunk_size}

        # Don't predict the last chunk from token counts: LeftContextChunkPolicy
        # emits an extra flush pass for the retained overlap, so a count-based
        # guess completes the request before that final chunk is emitted. The
        # `available <= 0` check above fires once consumption actually catches up.
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=[],
            unpersist_tensors=[],
            step_metadata=step_metadata,
            request_done=False,
        )

    # -----------------------------------------------------------------------
    # Model ABC: prompt processing
    # -----------------------------------------------------------------------

    def load_video(
        self, filepath: str, device: str
    ) -> TensorAndMetadata:
        # TODO: support audio in video
        from qwen_omni_utils.v2_5.vision_process import fetch_video
        video_input, video_sample_fps = fetch_video(
            {"video": filepath},
            return_video_sample_fps=True,
            image_patch_size=14,
            return_video_metadata=False
        )
        return TensorAndMetadata(
            data=video_input.to(device),
            metadata=dict(
                video_sample_fps=video_sample_fps
            )
        )

    def _user_turn_audio_split_index(
        self, input_ids: torch.Tensor
    ) -> int | None:
        """Index in ``input_ids`` right after ``<|im_start|>user\\n`` where the
        audio block must be inserted to match vLLM's layout.

        The Qwen ChatML user turn tokenizes as
        ``[<|im_start|>(151644), user(872), \\n(198), <prompt...>]``.  We locate
        the ``[im_start, user]`` pair and return the index just past the newline
        that follows it.  Returns None if no user turn is found.
        """
        im_start = self.config.im_start_token_id
        user_tok = self.config.user_token_id
        ids = input_ids.tolist()
        for i in range(len(ids) - 1):
            if ids[i] == im_start and ids[i + 1] == user_tok:
                j = i + 2
                # Skip the single newline token that the template emits after
                # the role name (id 198 for "\n"); guard against absence.
                if j < len(ids) and ids[j] == 198:
                    j += 1
                return j
        return None

    def _audio_mel_gpu(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """GPU log-mel matching HF ``WhisperFeatureExtractor`` (MSTAR_GPU_MEL=1).

        The HF feature_extractor runs the STFT + mel filterbank + log on the CPU
        (numpy); for a 30 s clip that is ~tens of ms on the TTFT critical path and
        is amplified under host-CPU contention (the same poll-loop sensitivity that
        inflates M*'s TTFT). This computes the identical transform on the GPU.

        Returns ``(input_features (n_mel, T) float32 CPU, audio_seqlen (1,) long)``
        — byte-compatible with the HF path's per-audio output so everything
        downstream is unchanged. Numerically matches HF to cos>=0.9999 / max-abs
        ~1e-5 (test_qwen3_omni_gpu_mel_parity.py): same hann window (periodic),
        center+reflect STFT, power spectrogram, drop-last-frame, log10, max-8 clamp,
        (x+4)/4. ``T = floor(len/hop)`` == HF's valid (un-padded) frame count.
        """
        fe = self._processor.feature_extractor
        dev = waveform.device if waveform.is_cuda else torch.device("cuda")
        st = self._gpu_mel_state
        if st is None or st["dev"] != dev:
            import numpy as np
            st = {
                "dev": dev,
                "filters": torch.tensor(np.asarray(fe.mel_filters),
                                        dtype=torch.float32, device=dev),  # (n_freq, n_mel)
                "window": torch.hann_window(fe.n_fft, periodic=True, device=dev),
                "n_fft": fe.n_fft, "hop": fe.hop_length,
            }
            self._gpu_mel_state = st
        wav = waveform.to(dev, torch.float32)
        log = gpu_log_mel(wav, st["filters"], st["window"], st["n_fft"], st["hop"])
        feat = log.cpu()                                  # CPU float32 == HF contract
        seqlen = torch.tensor([feat.shape[1]], dtype=torch.long)
        return feat, seqlen

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        input_metadata: dict[str, dict] = {},
        **kwargs,
    ) -> NameToTensorList:
        """Build the full ChatML prompt + derived multimodal tensors.

        Uses HF's full ``AutoProcessor`` (combines tokenizer + image_processor
        + video_processor + feature_extractor + chat template) to:

        1. Build a ChatML-formatted prompt from ``prompt`` and any
           multimodal inputs in ``tensors``.
        2. Apply ``add_generation_prompt=True`` so the model receives the
           ``<|im_start|>assistant\\n`` suffix and knows to start the
           assistant response.
        3. Run the image_processor / feature_extractor on the raw modality
           tensors to produce ``pixel_values`` / ``image_grid_thw`` /
           ``audio_features`` / ``audio_seqlens``.
        4. Expand the single ``<|image_pad|>`` / ``<|audio_pad|>`` /
           ``<|video_pad|>`` placeholder in the tokenized text to N copies
           where N = number of patches after spatial merge (this is what
           ``Qwen3OmniMoeProcessor.replace_multimodal_special_tokens`` does
           internally).

        The result has ``text_inputs`` containing the FULL templated +
        expanded token IDs, plus the per-modality tensor outputs needed by
        the Thinker's prefill walks.
        """
        result: NameToTensorList = {}

        if tensors is None:
            tensors = {}

        # ----- Convert raw modality tensors to PIL/numpy form for HF -----
        raw_image_inputs = tensors.get("image_inputs", [])
        raw_audio_inputs = tensors.get("audio_inputs", [])
        raw_video_inputs = tensors.get("video_inputs", [])

        # When GPU image preprocessing is enabled we keep the raw GPU tensors
        # and never round-trip through CPU/numpy (see _gpu_image_preprocess).
        gpu_img_preprocess = _gpu_image_preprocess_enabled()

        pil_images: list = []
        if not gpu_img_preprocess:
            for img in raw_image_inputs:
                # data_worker.py provides images as (C, H, W) float32 in [0, 1]
                # on the GPU.  HF processors expect PIL/numpy uint8 (H, W, C)
                # in [0, 255] -- otherwise the default do_rescale=True double-
                # rescales and the model sees a near-zero (essentially black)
                # tensor regardless of the actual image content.
                if img.dtype.is_floating_point:
                    img_u8 = (img * 255.0).clamp(0, 255).to(torch.uint8)
                else:
                    img_u8 = img
                if img_u8.dim() == 3 and img_u8.shape[0] in (1, 3):
                    img_u8 = img_u8.permute(1, 2, 0)  # CHW -> HWC
                pil_images.append(img_u8.cpu().contiguous().numpy())

        # GPU log-mel is opt-in (MSTAR_GPU_MEL=1) and only when CUDA is present in
        # this worker; otherwise the raw audio is converted to numpy for the HF
        # (CPU) feature_extractor exactly as before. Default OFF = byte-identical.
        _use_gpu_mel = (
            _GPU_MEL and self._processor is not None and torch.cuda.is_available()
        )
        np_audios: list = []
        if not _use_gpu_mel:
            for waveform in raw_audio_inputs:
                np_audios.append(waveform.cpu().numpy())

        # ----- Preferred path: text-only chat template + separate modality processors -----
        #
        # We deliberately DO NOT include image/audio/video content blocks in
        # the messages list passed to apply_chat_template.  HF's chat template
        # would otherwise insert ``<|vision_start|><|image_pad|>...<|vision_end|>``
        # placeholders into text_inputs, which we don't want because:
        #
        #   1. Our prefill_vision / prefill_audio walks already wrap the
        #      modality content in their own start/end tokens before pushing
        #      it into the Thinker's KV cache.  Having the same wrapping in
        #      text_inputs would make the model see each modality twice
        #      (once as actual encoder embeddings via the modality walks,
        #      once as generic token embeddings via prefill_text), which is
        #      noise.
        #
        #   2. Unlike HF's single-shot prefill (which masked-scatter's the
        #      vision embeds INTO the placeholder positions in input_embeds),
        #      our multi-walk prefill builds up the same final KV cache via
        #      sequential walks.  The modality placeholders in text_inputs
        #      would never be replaced by real content in our flow.
        #
        # Functionally, both approaches end up with the same set of
        # embeddings in the KV cache (text + modality content).  Stripping
        # the placeholders avoids noise from the unfilled embeddings.
        # if self._processor is not None:
            # try:
        system_text = (
            "You are Qwen, a virtual human developed by the "
            "Qwen team, Alibaba Group, capable of perceiving "
            "auditory and visual inputs, as well as generating "
            "text and speech."
        )
        messages = [
            {"role": "system", "content": system_text},
        ]
        # vLLM token parity: flatten_messages folds the system text into the prompt
        # blob, which we re-wrap in our own system turn -> double-counted system text.
        # Under MSTAR_VLLM_PROMPT_LAYOUT, strip the leading system-text copy so the
        # user turn is instruction-only and tokens match vLLM. OFF path untouched.
        user_prompt = prompt
        if vllm_prompt_layout_enabled() and prompt is not None:
            for sep in ("\n", ""):
                dup = system_text + sep
                if prompt.startswith(dup):
                    user_prompt = prompt[len(dup):]
                    break
        if user_prompt is not None:
            messages.append(
                {"role": "user", "content": user_prompt},
            )

        # apply_chat_template with TEXT-ONLY content -> no modality
        # placeholders are inserted.  add_generation_prompt=True
        # appends the trailing ``<|im_start|>assistant\n`` so the
        # model knows to start the assistant response.
        if self._processor is None:
            # __init__ sets _processor=None and warns if AutoProcessor fails to
            # load. Fail fast with a clear message instead of a cryptic
            # AttributeError on the first request (the old commented-out guard
            # promised a tokenizer fallback that was never wired up).
            raise RuntimeError(
                "Qwen3-Omni processor failed to load at init; cannot build the "
                "chat-template prompt. Check the checkpoint/processor files."
            )
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer(
            text, return_tensors="pt"
        )["input_ids"][0]

        # --- vLLM prompt-layout: audio INSIDE the user turn, BEFORE instr ---
        #
        # Legacy M* layout prefills audio as a separate bare block (schedule
        # [prefill_audio, prefill_text]) so the audio sits OUTSIDE any turn and
        # the instruction governs -> the model TRANSCRIBES.  vLLM-Omni applies
        # the stock HF chat template which puts the audio inside the user turn
        # before the instruction -> the trained "spoken-query -> reply" layout
        # -> the model ANSWERS.
        #
        # To replicate that without retokenizing across the boundary (which can
        # shift BPE merges), we slice the ALREADY-tokenized full sequence right
        # after ``<|im_start|>user\n`` into a prefix (system turn + user-turn
        # opener) and a suffix (instruction + ``<|im_end|>`` + assistant
        # prompt).  The schedule builder then runs
        # [prefill_text(prefix), prefill_audio, prefill_text(suffix)], so the
        # audio walk's BOS/AUDIO/EOS embeddings land between them -> exactly the
        # vLLM token layout (modulo sentinel IDs + M-RoPE, tracked separately).
        if (
            vllm_prompt_layout_enabled()
            and len(np_audios) > 0
            and prompt is not None
        ):
            split = self._user_turn_audio_split_index(input_ids)
            if split is not None:
                prefix_ids = input_ids[:split]
                suffix_ids = input_ids[split:]
                result["text_inputs"] = [prefix_ids, suffix_ids]
                _dump_obj("mstar_thinker_prefix_ids.pt", prefix_ids)
                _dump_obj("mstar_thinker_suffix_ids.pt", suffix_ids)
                _dump_obj(
                    "mstar_thinker_layout.json",
                    {
                        "layout": "vllm_user_turn_audio_before_instruction",
                        "prefix_ids": prefix_ids.tolist(),
                        "suffix_ids": suffix_ids.tolist(),
                        "split_index": int(split),
                        "audio_sentinels": (
                            [151669, 151670]
                            if vllm_audio_sentinels_enabled()
                            else [
                                self.config.thinker.audio_start_token_id,
                                self.config.thinker.audio_end_token_id,
                            ]
                        ),
                    },
                )
            else:
                logger.warning(
                    "MSTAR_VLLM_PROMPT_LAYOUT=1 but could not locate the user "
                    "turn in the tokenized prompt; falling back to legacy "
                    "layout for this request."
                )
                result["text_inputs"] = [input_ids]
        else:
            result["text_inputs"] = [input_ids]
            _dump_obj("mstar_thinker_input_ids.pt", input_ids)
            _dump_obj(
                "mstar_thinker_layout.json",
                {
                    "layout": "legacy_audio_separate_block",
                    "input_ids": input_ids.tolist(),
                },
            )

        result["pixel_values"] = []
        result["image_grid_thw"] = []
        result["audio_seqlens"] = []
        result["audio_features"] = []
        result["video_second_per_grid"] = []
        result["video_grid_thw"] = []
        result["pixel_values_videos"] = []

        # Run image_processor / feature_extractor SEPARATELY for the
        # modality outputs.  These don't touch text_inputs.
        if gpu_img_preprocess:
            # GPU path: process each image fully on-device (no CPU round-trip).
            img_proc = self._processor.image_processor
            for img in raw_image_inputs:
                pv, grid_thw = _gpu_image_preprocess(
                    img,
                    patch_size=img_proc.patch_size,
                    temporal_patch_size=img_proc.temporal_patch_size,
                    merge_size=img_proc.merge_size,
                    min_pixels=img_proc.size["shortest_edge"],
                    max_pixels=img_proc.size["longest_edge"],
                    image_mean=img_proc.image_mean,
                    image_std=img_proc.image_std,
                )
                result["pixel_values"].append(pv)
                result["image_grid_thw"] += list(grid_thw)
        else:
            for img in pil_images:
                img_proc = self._processor.image_processor
                img_out = img_proc(images=[img], return_tensors="pt")
                result["pixel_values"].append(img_out["pixel_values"])
                result["image_grid_thw"] += img_out["image_grid_thw"]

        if _use_gpu_mel:
            for waveform in raw_audio_inputs:
                feat, seqlen = self._audio_mel_gpu(waveform)   # (n_mel, T), (1,)
                result["audio_seqlens"].append(seqlen)
                result["audio_features"].append(feat)
        else:
            for audio in np_audios:
                feat_extractor = self._processor.feature_extractor
                sr = getattr(feat_extractor, "sampling_rate", 16000)
                aud_out = feat_extractor(
                    audio, sampling_rate=sr,
                    padding=True,
                    truncation=False,
                    return_attention_mask=True,
                    return_tensors="pt"
                )
                aud_out["input_features"] = (
                    aud_out["input_features"]
                    .permute(0, 2, 1)[aud_out["attention_mask"].bool()]
                    .permute(1, 0)
                )
                result["audio_seqlens"].append(
                    aud_out["attention_mask"].sum(-1).to(torch.long)
                )
                result["audio_features"].append(
                    aud_out["input_features"]
                )

        # Video uses the video_processor; left as TODO since our
        # prefill_vision walk doesn't yet handle video frame stacks.
        for video, meta in zip(raw_video_inputs, input_metadata.get("video_inputs", []), strict=True):
            fps = meta.get(
                "video_sample_fps", 2.0
            )
            vid_out = self._processor.video_processor(
                videos=video,
                size={
                    "shortest_edge": 128 * 32 * 32,
                    "longest_edge": 768 * 32 * 32,
                }
            )
            result["video_second_per_grid"].append(
                torch.tensor([self._processor.video_processor.temporal_patch_size / fps])
            )
            result["video_grid_thw"] += vid_out["video_grid_thw"]
            result["pixel_values_videos"].append(vid_out["pixel_values_videos"])

        return result

    # -----------------------------------------------------------------------
    # Model ABC: postprocess
    # -----------------------------------------------------------------------

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,
    ) -> bytes:
        if modality == "text":
            detok = self.tokenizer.decode(output)
            return detok.encode("utf-8")
        elif modality == "audio":
            if output.numel() == 0:
                return b""
            return output.cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for Qwen3-Omni: {modality!r}")

    # -----------------------------------------------------------------------
    # Model ABC: sharding
    # -----------------------------------------------------------------------

    def get_default_sharding_config(self):
        from mstar.distributed.base import ShardingConfig

        # Talker LLM (attention + MoE-with-shared-expert) is TP-capable
        # via the same ``ParallelAttention`` / ``ParallelSparseMoeBlock*``
        # parts as the Thinker. The internal CodePredictor is intentionally
        # left at TP=1 (replicated weights, deterministic sampler) — see
        # ``_create_talker_submodule``. ``shard_dim`` stays empty because
        # every cross-edge signal (``thinker_states``, ``thinker_mask``,
        # ``codec_tokens``, ``talker_input_embeds``, ``new_token``) is
        # already replicated by the upstream all-reduce or sampler
        # broadcast before it leaves its producing node.
        return ShardingConfig(
            groups=[], tp_enabled_nodes={"Thinker", "Talker"}, shard_dim={},
        )

    # -----------------------------------------------------------------------
    # Model ABC: submodule loading
    # -----------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(
            node_name, device, tp_group=tp_group, autocast_dtype=autocast_dtype,
        )
        logger.info("Successfully loaded Qwen3-Omni submodule for %s", node_name)
        self._submodule_cache[node_name] = submodule

        # W3: If the Thinker was just loaded and the Talker already exists
        # (but TTS embeds were not initialized because Thinker wasn't
        # available at Talker creation time), initialize them now.
        if node_name == "Thinker":
            talker_sub = self._submodule_cache.get("Talker")
            if (
                talker_sub is not None
                and hasattr(talker_sub, '_tts_pad_embed_cached')
                and talker_sub._tts_pad_embed_cached is None
                and hasattr(submodule, 'model')
            ):
                try:
                    talker_sub.init_tts_embeds(submodule.model.embed_tokens)
                except Exception as e:
                    logger.warning(
                        "Deferred TTS embed init failed: %s", e,
                    )

        return submodule

    def _create_submodule(
        self, node_name: str, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name == "Thinker":
            return self._create_thinker_submodule(
                device, tp_group=tp_group, autocast_dtype=autocast_dtype,
            )
        elif node_name == "Talker":
            return self._create_talker_submodule(
                device, tp_group=tp_group, autocast_dtype=autocast_dtype,
            )
        elif node_name == "Code2Wav":
            return self._create_code2wav_submodule(device)
        elif node_name == "audio_encoder":
            return self._create_audio_encoder_submodule(device)
        elif node_name == "vision_encoder":
            return self._create_vision_encoder_submodule(device)
        return None

    @staticmethod
    def _thinker_remap(name: str) -> str | None:
        """Map HF checkpoint keys (after ``thinker.`` prefix strip) to model param paths.

        Handles the ``block_sparse_moe`` → ``mlp`` rename and the per-expert
        weight fusion: ``experts.{N}.{gate,up,down}_proj.weight`` becomes a
        shard_id-carrying key that the MoE weight_loaders consume via
        ``StackedParamRule``.
        """
        import re

        if "rotary_emb" in name:
            return None
        name = name.replace("block_sparse_moe.", "mlp.")
        # Per-expert weights: experts.N.{gate,up,down}_proj.weight
        # → experts.{gate_up_proj,down_proj} with shard_id handled by stacked rules.
        # We rewrite the name so StackedParamRule suffix matching works:
        # "experts.N.gate_proj.weight" → "experts.gate_proj.__N__.weight"
        # The weight_loader on the fused param extracts the expert index from shard_id.
        m = re.match(r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", name)
        if m:
            prefix, expert_idx, proj = m.groups()
            return f"{prefix}.experts.{proj}.__expert{expert_idx}__.weight"
        return name

    # MoE stacked param rules: route per-expert projections into fused params.
    # The shard_id encodes both the projection type AND the expert index.
    # The __expertN__ marker is injected by _thinker_remap; weight_loaders
    # parse it to determine the expert slot.
    _THINKER_STACKED_PARAMS: list = []  # populated lazily below

    def _get_thinker_stacked_params(self):
        from mstar.model.loader.base import StackedParamRule

        if self._THINKER_STACKED_PARAMS:
            return self._THINKER_STACKED_PARAMS
        E = self.config.thinker_text.num_experts
        # MoE expert rules MUST come before dense MLP rules because
        # the dense ".gate_proj" suffix would also match the remapped
        # MoE key "experts.gate_proj.__expertN__.weight". _apply_stacked
        # returns on first match, so longer/more-specific rules go first.
        rules = []
        for i in range(E):
            # source_suffix includes ".weight" so the replacement strips it —
            # the target params (experts.gate_up_proj, experts.down_proj) are
            # bare nn.Parameters, not Linear submodules, so they have no
            # ".weight" suffix in named_parameters().
            rules.append(StackedParamRule(
                target_suffix=".experts.gate_up_proj",
                source_suffix=f".experts.gate_proj.__expert{i}__.weight",
                shard_id=f"gate:{i}",
            ))
            rules.append(StackedParamRule(
                target_suffix=".experts.gate_up_proj",
                source_suffix=f".experts.up_proj.__expert{i}__.weight",
                shard_id=f"up:{i}",
            ))
            rules.append(StackedParamRule(
                target_suffix=".experts.down_proj",
                source_suffix=f".experts.down_proj.__expert{i}__.weight",
                shard_id=f"down:{i}",
            ))
        # Dense MLP gate/up fusion and attention qkv fusion.
        rules.append(StackedParamRule(".gate_up_proj", ".gate_proj", 0))
        rules.append(StackedParamRule(".gate_up_proj", ".up_proj", 1))
        rules.append(StackedParamRule(".qkv_proj", ".q_proj", "q"))
        rules.append(StackedParamRule(".qkv_proj", ".k_proj", "k"))
        rules.append(StackedParamRule(".qkv_proj", ".v_proj", "v"))
        self._THINKER_STACKED_PARAMS = rules
        return rules

    def _create_thinker_submodule(
        self, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.loader import load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_omni.components.thinker import Qwen3OmniThinkerModel

        with torch.device("meta"):
            thinker_model = Qwen3OmniThinkerModel(self.config, comm_group=tp_group)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            thinker_model = thinker_model.to(autocast_dtype)
        thinker_model.to_empty(device=device)

        weights = iter_safetensors_shards(
            self.local_dir, device=device,
            prefix="thinker."
        )
        # Strip the "thinker." prefix from checkpoint keys.
        weights = ((k.removeprefix("thinker."), v) for k, v in weights)

        load_hf_weights(
            thinker_model, weights,
            stacked_params=self._get_thinker_stacked_params(),
            name_remapper=self._thinker_remap,
        )
        thinker_model.eval()


        from mstar.model.qwen3_omni.submodules import ThinkerSubmodule
        return ThinkerSubmodule(
            thinker_model=thinker_model,
            config=self.config,
        )

    def _create_talker_submodule(
        self, device: str, tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule:
        from mstar.model.loader import load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_omni.components.talker import (
            Qwen3OmniTalkerModel,
        )

        with torch.device("meta"):
            # ``tp_group`` shards the Talker LLM's attention + MoE. The
            # CodePredictor stays TP=1 (separate construction below) —
            # its compute is small and the deterministic FlashInfer sampler
            # produces bit-equal codes on every rank, so per-rank
            # replication is cheaper than 150+ NCCL all-reduces per
            # decode step (5 layers x 15 unrolled iterations).
            talker_model = Qwen3OmniTalkerModel(self.config, comm_group=tp_group)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            talker_model = talker_model.to(autocast_dtype)
        talker_model.to_empty(device=device)

        # Talker and CodePredictor share the "talker." prefix. We stream
        # once and split: code_predictor keys go to the CodePredictor,
        # everything else goes to the TalkerModel.
        talker_weights = []
        code_pred_weights = []
        _CP_PREFIX = "talker.code_predictor."
        _TALKER_PREFIX = "talker."

        initialized_thinker_embed_tokens = False
        text_config = self.config.thinker_text
        embed_tokens = torch.nn.Embedding(
            text_config.vocab_size, text_config.hidden_size,
            device=device
        ).eval()
        for k, v in iter_safetensors_shards(self.local_dir, device=device, prefix=_TALKER_PREFIX):
            if k.startswith(_CP_PREFIX):
                code_pred_weights.append((k.removeprefix(_CP_PREFIX), v))
            else:
                talker_weights.append((k.removeprefix(_TALKER_PREFIX), v))
        for _k, v in iter_safetensors_shards(
            self.local_dir, device=device, prefix="thinker.model.embed_tokens"
        ):
            initialized_thinker_embed_tokens = True
            with torch.no_grad():
                embed_tokens.weight.copy_(v)

        assert initialized_thinker_embed_tokens, \
            "thinker.model.embed_tokens not found to initialize talker TTS embeds"

        stacked = self._get_thinker_stacked_params()
        load_hf_weights(
            talker_model, iter(talker_weights),
            stacked_params=stacked,
            name_remapper=self._thinker_remap,
        )
        talker_model.eval()

        with torch.device("meta"):
            code_predictor = Qwen3OmniCodePredictor(self.config)
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            code_predictor = code_predictor.to(autocast_dtype)
        code_predictor.to_empty(device=device)
        load_hf_weights(
            code_predictor, iter(code_pred_weights),
            stacked_params=stacked,
            name_remapper=self._thinker_remap,
        )
        code_predictor.consolidate_stacked_weights()
        code_predictor.eval()

        from mstar.model.qwen3_omni.submodules import TalkerSubmodule
        talker_sub = TalkerSubmodule(
            talker_model=talker_model,
            code_predictor=code_predictor,
            config=self.config,
        )
        talker_sub.init_tts_embeds(embed_tokens)
        del embed_tokens

        return talker_sub

    def _create_code2wav_submodule(self, device: str) -> NodeSubmodule:
        # Code2Wav is the vocoder that converts codec tokens to audio waveform.
        # The actual model class will be defined in components.
        from mstar.model.qwen3_omni.components.code2wav import Qwen3OmniMoeCode2Wav
        from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        # The vocoder is dominated by Conv1d/ConvTranspose1d at small channel
        # counts where cuDNN's default heuristic picks a sub-optimal algo.
        # benchmark=True autotunes per shape on the warm-up call, before
        # CUDA-graph capture, so the chosen algo is baked into the graph.
        torch.backends.cudnn.benchmark = True

        code2wav_model = Qwen3OmniMoeCode2Wav(self.config.code2wav)
        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[
                ModuleAndPrefix(code2wav_model, prefix="code2wav"),
            ],
            device=device,
        )
        code2wav_model.eval()
        code2wav_model.consolidate()

        from mstar.model.qwen3_omni.submodules import Code2WavSubmodule
        return Code2WavSubmodule(
            code2wav_model=code2wav_model,
            config=self.config,
        )

    def _create_audio_encoder_submodule(self, device: str) -> NodeSubmodule:
        """Load the audio encoder (AuT) from HF weights.

        Two paths, selected by ``config.native_audio_encoder``:
          * native (default on): batched, transformers-decoupled mstar module
            (``NativeAudioEncoderSubmodule``). Numerically matches HF (fp32
            exact; bf16 within the parity bar). Throughput gain over the HF
            wrapper is modest — ~1.2-1.7x in the SDPA microbenchmark, peaking at
            batch 4-8 (see benchmark/artifacts/README_qwen3_omni_encoders.md);
            the win is cross-request batching, not a faster attention kernel.
          * HF wrapper (fallback/reference, kept for one release).
        """
        from transformers import AutoConfig

        from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        config = AutoConfig.from_pretrained(self.local_dir, trust_remote_code=True)
        audio_config = config.thinker_config.audio_config

        if getattr(self.config, "native_audio_encoder", False):
            from mstar.model.qwen3_omni.components.audio_encoder import (
                NativeQwen3OmniAudioEncoder,
            )
            from mstar.model.qwen3_omni.submodules import NativeAudioEncoderSubmodule
            audio_encoder = NativeQwen3OmniAudioEncoder(audio_config).to(device)
            load_weights_from_hf_shards(
                repo_dir=self.local_dir,
                modules=[ModuleAndPrefix(audio_encoder, prefix="thinker.audio_tower")],
                device=device,
            )
            audio_encoder.eval()
            return NativeAudioEncoderSubmodule(audio_encoder=audio_encoder, config=self.config)

        # ---- HF-wrapper fallback path ----
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
        )

        # Build the audio encoder from config.
        # IMPORTANT: pass attn_implementation="flash_attention_2" so the
        # encoder uses the cu_seqlens FA2 path. With the HF default
        # (which resolves to "sdpa"), Qwen3OmniMoeAudioAttention runs
        # SDPA on the full packed sequence (no per-segment fusion),
        # which is significantly slower than FA2's varlen path.
        audio_encoder = Qwen3OmniMoeAudioEncoder._from_config(
            audio_config, attn_implementation=_hf_encoder_attn_impl()
        )

        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[ModuleAndPrefix(audio_encoder, prefix="thinker.audio_tower")],
            device=device,
        )
        audio_encoder.eval()

        from mstar.model.qwen3_omni.submodules import AudioEncoderSubmodule
        return AudioEncoderSubmodule(audio_encoder=audio_encoder, config=self.config)

    def _create_vision_encoder_submodule(self, device: str) -> NodeSubmodule:
        """Load the vision encoder (SigLIP2 ViT) from HF weights.

        Two paths, selected by ``config.native_vision_encoder``:
          * native (default on): batched, varlen mstar module
            (``NativeVisionEncoderSubmodule``). Numerically matches HF (fp32
            exact; bf16 within bar) for the pooler output and every DeepStack
            level. The large per-image speedup comes almost entirely from
            computing the patch embed as an ``F.linear`` instead of HF's bf16
            ``Conv3d`` (kernel==stride), which hits a cuDNN low-precision cliff
            (~3.3 s/image on H100) — the same swap could in principle be applied
            to the HF path. Attention is the same ``flash_attn_varlen_func``
            primitive HF uses, not a shape-specialized kernel.
          * HF wrapper (fallback/reference, kept for one release).
        """
        from transformers import AutoConfig

        from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        config = AutoConfig.from_pretrained(self.local_dir, trust_remote_code=True)
        vision_config = config.thinker_config.vision_config

        if getattr(self.config, "native_vision_encoder", False):
            from mstar.model.qwen3_omni.components.vision_encoder import (
                NativeQwen3OmniVisionEncoder,
            )
            from mstar.model.qwen3_omni.submodules import NativeVisionEncoderSubmodule
            vision_encoder = NativeQwen3OmniVisionEncoder(vision_config).to(device)
            load_weights_from_hf_shards(
                repo_dir=self.local_dir,
                modules=[ModuleAndPrefix(vision_encoder, prefix="thinker.visual")],
                device=device,
            )
            vision_encoder.eval()
            return NativeVisionEncoderSubmodule(vision_encoder=vision_encoder, config=self.config)

        # ---- HF-wrapper fallback path ----
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeVisionEncoder,
        )

        # Build the vision encoder.
        # CRITICAL: pass attn_implementation="flash_attention_2". Without
        # this, vision_config._attn_implementation defaults to None and is
        # resolved to "sdpa" at runtime (modeling_utils.py:1889). With
        # "sdpa", Qwen3OmniMoeVisionAttention.forward falls into the
        # per-segment Python loop (modeling_qwen3_omni_moe.py:892-913),
        # which issues N sequential attention calls per layer for an
        # N-frame video. This causes the 10× V2T/V2S TTFT regression vs
        # vllm-omni. With "flash_attention_2", a single varlen FA2 call
        # per layer handles all frames at once via cu_seqlens.
        vision_encoder = Qwen3OmniMoeVisionEncoder._from_config(
            vision_config, attn_implementation=_hf_encoder_attn_impl()
        )

        load_weights_from_hf_shards(
            repo_dir=self.local_dir,
            modules=[ModuleAndPrefix(vision_encoder, prefix="thinker.visual")],
            device=device,
        )
        vision_encoder.eval()

        from mstar.model.qwen3_omni.submodules import VisionEncoderSubmodule
        return VisionEncoderSubmodule(vision_encoder=vision_encoder, config=self.config)
