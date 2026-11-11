"""
Qwen3ASRModel: speech-to-text LLM (Qwen/Qwen3-ASR-1.7B, Qwen/Qwen3-ASR-0.6B).

An AuT audio encoder turns a log-mel clip of up to 20 minutes into ~13
tokens per second in LLM space; a dense Qwen3 decoder reads them inside a
ChatML prompt and writes ``language {Name}<asr_text>{transcript}``. Forcing
the language pre-fills ``language {Name}<asr_text>`` into the assistant
turn, so the model emits the transcript alone.

Two nodes on the shared decoder scaffold (`MultimodalDecoderModel`):
    audio_encoder  — packed AuT, 8 s attention windows through the ragged
                     attention resource (no cache)
    LLM            — dense Qwen3 (`SampledTokenDecoder`); paged KV cache;
                     prefill captured on packed token buckets, decode per
                     batch size

Graph walks:
    prefill — audio_encoder -> LLM over the whole prompt as one row, the
              audio embeddings spliced in at the ``<|audio_pad|>``
              placeholders; samples the first token
    decode  — LLM loop; each step feeds the sampled token back

Every request carries its one clip, so the scaffold's text-only walk and
mixed step are not built (``TEXT_PROMPTS``).

Prompt (the reference SDK's chat template, system turn included):
    <|im_start|>system\\n{context}<|im_end|>\\n
    <|im_start|>user\\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\\n
    <|im_start|>assistant\\n[language {Name}<asr_text>]
"""

import logging
from collections.abc import Mapping
from typing import Any

import torch

from mstar.engine.state import (
    Array,
    Attend,
    Block,
    Causal,
    KVEntry,
    Layout,
    Positions,
    Read,
    Rope,
    Sampler,
    SampleSettings,
)
from mstar.model.base import ProcessPromptOutput
from mstar.model.components.audio_features import LogMelSpectrogram
from mstar.model.components.request_kwargs import (
    GREEDY_SAMPLING,
    SAMPLING_KEYS,
    checked_request_kwargs,
    sampler_capabilities,
    sampling_config,
    sampling_defaults,
)
from mstar.model.multimodal import TEXT, MediaEncoder, PromptPart
from mstar.model.multimodal_decoder import MediaIntake, MultimodalDecoderModel, PreparedMedia
from mstar.model.qwen3_asr.config import (
    ASR_TEXT_TAG,
    ATTN,
    AUT_ATTN,
    AUT_TOKENS,
    ENCODER_NODE,
    KV_CACHE,
    LANGUAGE_PREFIX,
    LLM_NODE,
    SAMPLER,
    Qwen3ASRModelConfig,
)
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import ByteLevelDetokenizer, TextOutput

logger = logging.getLogger(__name__)


def audio_encoder() -> MediaEncoder:
    """The ``audio_encoder`` node: a clip's samples in, one LLM-space row per
    audio pad out, on the ``prefill`` walk (configs and metrics read walk names)."""
    return MediaEncoder("audio", ENCODER_NODE, ("audio",), "audio_embeds", walk="prefill")


class Qwen3ASRModel(MultimodalDecoderModel):
    """Qwen3-ASR: AuT encoder + dense Qwen3 decoder."""

    NAME = "Qwen3-ASR"
    # the API server and conductor fetch the whole snapshot too, as they always have
    METADATA_FILES = ("*",)
    # every request carries its one clip
    TEXT_PROMPTS = False

    # Concurrency the default cache sizing targets: 20 minutes of audio is
    # ~16k tokens, a LibriSpeech utterance ~300, so the pool is sized for a
    # mix (1024 pages of 128 = 131k tokens). Deployments retune under
    # ``resources:`` in their YAML.
    KV_PAGES = 1024

    # greedy, as the reference decodes, under the checkpoint's generation_config.json; set by __init__
    sampling_defaults: Mapping[str, object] = GREEDY_SAMPLING

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(model_path_hf, cache_dir)
        self.config = Qwen3ASRModelConfig.from_pretrained(self.local_dir)
        self.sampling_defaults = sampling_defaults(self.local_dir)
        self.AUDIO_INPUT_RATE = self.config.sampling_rate
        self.MAX_AUDIO_SECONDS = self.config.max_audio_seconds

        from transformers import AutoTokenizer

        # The checkpoint's tokenizer.json carries the pre-tokenizer regex bug
        # transformers warns about. The fix flag is what the reference uses.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_dir, cache_dir=cache_dir, fix_mistral_regex=True,
        )
        self.log_mel = LogMelSpectrogram(
            num_mel_bins=self.config.num_mel_bins,
            sampling_rate=self.config.sampling_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        self._detokenizer = ByteLevelDetokenizer(self.tokenizer)
        pad = self.config.audio_token_id
        self.media = (MediaIntake("audio", audio_encoder(), pad, self._audio_intake),)

    # -------------------------------------------------------------------
    # Model ABC: resources
    # -------------------------------------------------------------------

    def rebuild_policies(self) -> dict[str, RebuildPolicy]:
        # not yet dropped: no drop and rebuild has run through its captured prefill
        return {}

    @property
    def max_windows_per_request(self) -> int:
        """Attention windows (``n_window_infer`` mel frames, 8 s) in the longest
        clip a request may carry: 150 for 20 minutes."""
        frames = self.config.max_audio_samples // self.config.hop_length
        return -(-frames // self.config.audio.n_window_infer)

    def declare_state(self) -> list[Array | Sampler]:
        text, audio = self.config.text, self.config.audio
        capabilities = sampler_capabilities(self.sampling_defaults)
        return [
            Array(
                AUT_TOKENS, nodes={ENCODER_NODE}, layers=frozenset(range(audio.encoder_layers)), max_len=0,
                entry=KVEntry(heads=audio.encoder_attention_heads, head_dim=audio.head_dim),
                reads={AUT_ATTN: Read(mask=Block(), op=Attend(q_heads=audio.encoder_attention_heads))},
            ),
            Array(
                KV_CACHE, nodes={LLM_NODE}, layers=frozenset(range(text.num_hidden_layers)),
                entry=KVEntry(heads=text.num_key_value_heads, head_dim=text.head_dim),
                max_len=text.max_position_embeddings, positions=Positions(rope=Rope(theta=text.rope_theta)),
                reads={ATTN: Read(mask=Causal(), op=Attend(q_heads=text.num_attention_heads))},
            ),
            Sampler(
                SAMPLER, nodes={LLM_NODE}, vocab_size=text.vocab_size,
                penalty=capabilities["enable_repetition_penalty"], min_p=capabilities["enable_min_p"],
            ),
        ]

    def kernel_settings(self) -> dict[str, dict[str, Any]]:
        return {
            # every 8 s window of a request is its own segment
            AUT_ATTN: {"max_segments_per_request": self.max_windows_per_request},
            # two query heads per KV head: the CUDA-core decode kernel wins
            ATTN: {"decode_tensor_cores": False},
            KV_CACHE: {"max_num_pages": self.KV_PAGES},
        }

    def request_kwargs(self) -> frozenset[str]:
        return frozenset({"language", "initial_prompt", "assistant_prefix", "max_output_tokens", *SAMPLING_KEYS})

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, SampleSettings]:
        del input_modalities, output_modalities
        config = sampling_config(self.sampling_defaults, model_kwargs)
        return {SAMPLER: SampleSettings(**{key: getattr(config, key) for key in SAMPLING_KEYS})}

    def process_prompt(self, *args: Any, **kwargs: Any) -> ProcessPromptOutput:
        out = super().process_prompt(*args, **kwargs)
        kv = out.metadata.pop("kv_layout")[KV_CACHE]["main"]
        out.metadata["state_layout"] = {KV_CACHE: {"main": Layout(spans=kv.spans, growth=kv.growth)}}
        if kv.keyed_spans is not None:
            out.metadata["keyed_spans"] = {KV_CACHE: {"main": kv.keyed_spans}}
        return out

    # -------------------------------------------------------------------
    # Serving config and budget
    # -------------------------------------------------------------------

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Requests past the cap wait in the conductor. A 20 min file holds
        about 122 pages of 128 tokens for its prompt and up to 32 for its
        transcript, so the page count, not the cap, bounds how many long files
        decode at once."""
        cap = config.get("max_concurrent_requests")
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise ValueError(f"{config_path}: set max_concurrent_requests to a positive int")
        pages = (config.get("resources") or {}).get("kv_cache", {}).get("max_num_pages", self.KV_PAGES)
        if cap > pages:
            raise ValueError(
                f"{config_path}: max_concurrent_requests={cap} but kv_cache.max_num_pages={pages}, "
                "every admitted request needs at least one page"
            )

    def get_max_output_tokens(self, **model_kwargs):
        return checked_request_kwargs(model_kwargs).get("max_output_tokens", self.config.max_new_tokens)

    # -------------------------------------------------------------------
    # Media loading + intake
    # -------------------------------------------------------------------

    def prompt_text(
        self, num_audio_tokens: int, context: str = "", language: str | None = None,
        assistant_prefix: str = "",
    ) -> str:
        """The reference SDK's ChatML prompt, with ``num_audio_tokens``
        placeholders and, for a forced language, the ``language X<asr_text>``
        assistant prefix. ``assistant_prefix`` is raw model output the
        assistant turn continues from (streaming re-decodes the audio so far
        with the stable part of the previous hypothesis prefilled)."""
        prompt = (
            f"<|im_start|>system\n{context}<|im_end|>\n"
            f"<|im_start|>user\n<|audio_start|>{'<|audio_pad|>' * num_audio_tokens}<|audio_end|><|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        if language:
            prompt += f"{LANGUAGE_PREFIX}{language}{ASR_TEXT_TAG}"
        return prompt + assistant_prefix

    def max_input_files(self) -> dict[str, int]:
        # none refused before decoding: intake refuses a second clip and ignores other files
        return {}

    def intake(self, prompt, input_modalities, tensors=None, prompt_parts=None, **kwargs):
        """The one clip, checked first, then the prompt around it. The text is
        the system-turn context; the request's layout and any other
        attachment are ignored, as they always were."""
        clips = (tensors or {}).get("audio_inputs", [])
        if len(clips) != 1:
            raise ValueError(f"{self.NAME} expects one audio input per request, got {len(clips)}.")
        clip = clips[0].reshape(-1).to(torch.float32).cpu()
        if clip.numel() == 0:
            raise ValueError(f"{self.NAME} received an empty audio input.")
        if clip.numel() > self.config.max_audio_samples:
            raise ValueError(
                f"{self.NAME} takes at most {self.config.max_audio_seconds:.0f} s of audio per request; "
                f"got {clip.numel() / self.config.sampling_rate:.1f} s. Split the file first."
            )
        # the encoder's first conv needs a few frames. The reference pads
        # sub-0.5 s clips with silence too
        min_samples = self.config.sampling_rate // 2
        if clip.numel() < min_samples:
            clip = torch.nn.functional.pad(clip, (0, min_samples - clip.numel()))
        return super().intake(prompt, [TEXT, "audio"], {"audio_inputs": [clip]}, None, **kwargs)

    def _prompt_ids(
        self, parts: list[PromptPart], raw: dict[str, list[torch.Tensor]], **kwargs,
    ) -> torch.Tensor:
        """The prompt with its one ``<|audio_pad|>``. The text (or
        ``initial_prompt``) is the system-turn context the reference SDK calls
        ``context`` (hot words, domain hints); ``language`` (ISO code or name)
        forces the output language and turns off the model's own language
        line; ``assistant_prefix`` is model output to continue from (the
        streaming route's stable hypothesis)."""
        kw = checked_request_kwargs(kwargs)
        context = kw.get("initial_prompt") or "".join(p.text or "" for p in parts if p.modality == TEXT)
        language = self.config.language_name(kw.get("language"))
        # One placeholder in the text, expanded in ids by intake: tokenizing
        # thousands of repeated special tokens is slow and adds nothing.
        ids = self.tokenizer.encode(
            self.prompt_text(1, context, language, kw.get("assistant_prefix") or ""),
            add_special_tokens=False,
        )
        if ids.count(self.config.audio_token_id) != 1:
            raise ValueError("the prompt must not contain <|audio_pad|>")
        return torch.tensor(ids, dtype=torch.long)

    def _audio_intake(self, clip: torch.Tensor, metadata: dict) -> PreparedMedia:
        """The samples go to the encoder as they are. The log-mel is its first
        step on the GPU (spectrograms on the data worker's CPU threads were
        the serving bottleneck). The prompt only needs the frame count."""
        tokens = self.config.audio.tokens_for_frames(self.log_mel.num_frames(clip.numel()))
        return PreparedMedia({"audio": clip.contiguous()}, [tokens])

    def text_output(self) -> TextOutput:
        return TextOutput(self._detokenizer)

    # -------------------------------------------------------------------
    # Submodules
    # -------------------------------------------------------------------

    def _create_submodule(
        self, node_name: str, device: str, dtype: torch.dtype, tp_group,
    ) -> NodeSubmodule | None:
        if node_name == ENCODER_NODE:
            submodule = self._create_encoder_submodule(device, dtype)
        elif node_name == LLM_NODE:
            submodule = self._create_llm_submodule(device, tp_group, dtype)
        else:
            return None
        logger.info("Successfully loaded Qwen3-ASR submodule for %s", node_name)
        return submodule

    def _create_encoder_submodule(self, device: str, dtype: torch.dtype) -> NodeSubmodule:
        from mstar.model.components.aut_encoder import AuTEncoder
        from mstar.model.loader.iterators import iter_safetensors_shards

        with torch.device("meta"):
            encoder = AuTEncoder(self.config.audio, array=AUT_TOKENS)
        encoder = encoder.to(dtype)
        encoder.to_empty(device=device)
        prefix = "thinker.audio_tower."
        weights = iter_safetensors_shards(self._resolve_weights(), device=device, prefix=prefix)
        encoder.load_weights((k.removeprefix(prefix), v) for k, v in weights)
        encoder.eval()

        from mstar.model.qwen3_asr.submodules import Qwen3ASREncoderSubmodule
        return Qwen3ASREncoderSubmodule(encoder=encoder, config=self.config)

    @staticmethod
    def _llm_remap(name: str) -> str | None:
        if name.startswith("thinker.model."):
            return name.removeprefix("thinker.model.")
        if name == "thinker.lm_head.weight":
            return "lm_head.weight"
        return None

    def _create_llm_submodule(
        self, device: str, tp_group, dtype: torch.dtype,
    ) -> NodeSubmodule:
        from mstar.model.components.qwen3_lm import Qwen3DenseLM
        from mstar.model.loader.iterators import iter_safetensors_shards

        with torch.device("meta"):
            llm = Qwen3DenseLM(self.config.text, array=KV_CACHE, comm_group=tp_group)
        llm = llm.to(dtype)
        llm.to_empty(device=device)
        weights = iter_safetensors_shards(self._resolve_weights(), device=device, prefix="thinker.")
        llm.load_weights(
            (mapped, v) for k, v in weights
            if (mapped := self._llm_remap(k)) is not None
        )
        llm.eval()

        from mstar.model.qwen3_asr.submodules import Qwen3ASRLLMSubmodule
        return Qwen3ASRLLMSubmodule(llm, self.config, self.encoders, text_prompts=self.TEXT_PROMPTS)
