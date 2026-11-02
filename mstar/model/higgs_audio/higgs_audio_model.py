"""
HiggsAudioModel: speech-to-text model (bosonai/higgs-audio-v3-stt).

Higgs-Audio STT couples a Whisper-style audio encoder ("audio_tower")
and MLP projector ("audio_encoder_proj") with a dense Qwen3-1.7B text
LLM. Audio is split into fixed 4 s chunks, each chunk is encoded to
12.5 embeddings/s in LLM space, and the embeddings are spliced into the
ChatML prompt between ``<|audio_bos|>`` and ``<|audio_eos|>``:

    <|im_start|>user\\n{prompt}<|audio_bos|>[audio embeds]<|audio_eos|><|im_end|>\\n
    <|im_start|>assistant\\n -> transcript

Two nodes on the shared VLM scaffold (`MultimodalDecoderModel`):
``audio_encoder`` (tower + projector) and ``LLM`` (dense Qwen3, paged KV).
The ``prefill_audio`` walk encodes the clip and prefills the whole prompt
as one row, its placeholder expanded to one pad per audio embedding; then
the ``decode`` loop. Every request carries one clip, so the scaffold's
text-only ``prefill_text`` walk, and its ``mixed`` step with decode, never
run here.

Not implemented from the reference pipeline: silero-VAD chunk
boundaries (fixed 4 s chunks only) and the client-side n-gram loop fix
(``ngram_loop_fix.py``), which is post-hoc string cleanup.
"""

import logging
from dataclasses import asdict, replace

import torch

from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVSpec,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
)
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.model.higgs_audio.config import ATTN, HiggsAudioModelConfig
from mstar.model.keys import KV_CACHE, ROPE, SAMPLER
from mstar.model.multimodal import TEXT, MediaEncoder, PromptPart
from mstar.model.multimodal_decoder import MediaIntake, MultimodalDecoderModel, PreparedMedia
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import ByteLevelDetokenizer, TextOutput

logger = logging.getLogger(__name__)


def audio_encoder(pad_id: int) -> MediaEncoder:
    """The ``audio_encoder`` node: a clip's mel chunks in, one LLM-space
    row per audio pad out."""
    return MediaEncoder(
        "audio", "audio_encoder", ("audio_features", "audio_feature_lens"), "audio_embeds", (pad_id,),
    )


class HiggsAudioModel(MultimodalDecoderModel):
    """Higgs-Audio STT: audio_tower + projector + dense Qwen3 LLM."""

    NAME = "Higgs-Audio"
    # every process fetches the whole snapshot, as before the scaffold
    METADATA_FILES = ("*",)
    # The reference pipeline extracts mel features with the
    # whisper-large-v3 processor (not from the higgs checkpoint).
    WHISPER_PROCESSOR_ID = "openai/whisper-large-v3"
    # every request carries its one clip
    TEXT_PROMPTS = False

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(model_path_hf, cache_dir)
        self.config = HiggsAudioModelConfig.from_pretrained(self.local_dir)

        from transformers import AutoFeatureExtractor, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.WHISPER_PROCESSOR_ID, cache_dir=cache_dir,
        )
        self._detokenizer = ByteLevelDetokenizer(self.tokenizer)
        pad = self.config.audio_in_token_idx
        self.media = (MediaIntake("audio", audio_encoder(pad), pad, self._audio_intake),)

    # -------------------------------------------------------------------
    # Resources
    # -------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """LLM paged KV + attention/positions/sampler. audio_encoder is
        stateless: it runs once per request and holds nothing."""
        kv_config = PagedKVConfig(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            num_qo_heads=self.config.num_attention_heads,
            # ~2k tokens per ASR request (prompt + 12.5 audio embeds/s +
            # transcript); 256 pages = 32k tokens ≈ 16 concurrent
            # requests at ~3.7 GB.
            max_num_pages=256,
        )
        return [
            KVSpec(resource_key=KV_CACHE, nodes={"LLM"}, config=kv_config),
            AttentionSpec(
                resource_key=ATTN, nodes={"LLM"},
                config=AttentionConfig(kv_cache=KV_CACHE),
            ),
            PositionSpec(
                resource_key=ROPE, nodes={"LLM"},
                config=PositionConfig(
                    kv_cache=KV_CACHE,
                    rope_theta=self.config.rope_theta,
                ),
            ),
            # ASR transcription: the reference decodes greedily and exposes
            # no penalty knob, so no seen-token buffers.
            replace(self._sampler_spec(self.config.vocab_size), enable_repetion_penalty=False),
        ]

    def get_max_output_tokens(self, **model_kwargs):
        # Reference transcribe.py generates up to 1024 new tokens.
        return model_kwargs.get("max_output_tokens", 1024)

    # -------------------------------------------------------------------
    # Intake
    # -------------------------------------------------------------------

    def max_input_files(self) -> dict[str, int]:
        return {"audio": 1}

    def intake(self, prompt, input_modalities, tensors=None, prompt_parts=None, **kwargs):
        # the reference prompt has a place for the one clip only: other
        # attachments are ignored, as they always were
        keep = {TEXT, "audio"}
        return super().intake(
            prompt, [m for m in input_modalities if m in keep], tensors,
            None if prompt_parts is None else [p for p in prompt_parts if p.modality in keep], **kwargs,
        )

    def get_request_resource_configs(
        self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
    ) -> dict[str, ResourceReqConfig]:
        # the request knobs higgs has always read; the rest are ignored, as before
        model_kwargs = model_kwargs or {}
        return {SAMPLER: SamplingReqConfig(
            # Reference eval decodes greedily (do_sample=False).
            temperature=model_kwargs.get("temperature", 0.0),
            top_p=model_kwargs.get("top_p", 1.0),
            ignore_eos=model_kwargs.get("ignore_eos", False),
        )}

    def _prompt_ids(
        self, parts: list[PromptPart], raw: dict[str, list[torch.Tensor]], **kwargs,
    ) -> torch.Tensor:
        """The reference ``transcribe.py``'s ChatML prompt: the transcription
        instruction (the model's default when the request sends none), then
        the clip between ``<|audio_bos|>`` and ``<|audio_eos|>``, each piece
        tokenized on its own as there. The clip is one placeholder, which
        intake expands to its tokens. ``enable_thinking=False`` (default)
        appends the empty think block so the transcript starts immediately.
        """
        clips = len(raw["audio"])
        if clips != 1:
            raise ValueError(f"{self.NAME} expects exactly one audio input per request; got {clips}.")
        instruction = "".join(p.text or "" for p in parts if p.modality == TEXT) or self.config.default_prompt

        def enc(s: str) -> list[int]:
            return self.tokenizer.encode(s, add_special_tokens=False)

        ids = [
            *enc("<|im_start|>user\n"), *enc(instruction), *enc("<|audio_bos|>"),
            self.config.audio_in_token_idx,
            *enc("<|audio_eos|>"), *enc("<|im_end|>\n<|im_start|>assistant\n"),
        ]
        if not kwargs.get("enable_thinking", False):
            ids += enc("<think>\n\n</think>\n\n")
        return torch.tensor(ids, dtype=torch.long)

    def _audio_intake(self, clip: torch.Tensor, metadata: dict) -> PreparedMedia:
        """The reference's front end: fixed ``chunk_size_seconds`` chunks (no
        VAD), whisper-large-v3 mels padded to the longest chunk. The clip
        fills a pad per row the projector keeps."""
        waveform = clip.reshape(-1).cpu().numpy()
        n = self.config.chunk_size_samples
        chunks = [waveform[i:i + n] for i in range(0, max(len(waveform), 1), n)]
        feat = self.feature_extractor(
            chunks,
            sampling_rate=self.config.sampling_rate,
            padding="longest",
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        lens = feat["attention_mask"].sum(-1).to(torch.long)
        tokens = sum(self.config.encoder_output_length(int(m)) for m in lens.tolist())
        return PreparedMedia(
            {"audio_features": feat["input_features"], "audio_feature_lens": lens}, [tokens],
        )

    def text_output(self) -> TextOutput:
        # a transcript, its special tokens dropped
        return TextOutput(self._detokenizer)

    # -------------------------------------------------------------------
    # Submodules
    # -------------------------------------------------------------------

    def _create_submodule(
        self, node_name: str, device: str, autocast_dtype: torch.dtype | None, tp_group,
    ) -> NodeSubmodule | None:
        if node_name == "audio_encoder":
            submodule = self._create_encoder_submodule(device)
        elif node_name == "LLM":
            submodule = self._create_llm_submodule(device, tp_group, autocast_dtype)
        else:
            return None
        logger.info("Loaded Higgs-Audio %s onto %s", node_name, device)
        return submodule

    def _create_encoder_submodule(self, device: str) -> NodeSubmodule:
        from mstar.model.higgs_audio.components.audio_tower import (
            HiggsAudioFeatureProjector,
            HiggsAudioTower,
        )
        from mstar.model.higgs_audio.submodules import HiggsAudioEncoderSubmodule
        from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        audio_tower = HiggsAudioTower(self.config)
        projector = HiggsAudioFeatureProjector(self.config)

        load_weights_from_hf_shards(
            repo_dir=self._resolve_weights(),
            modules=[
                ModuleAndPrefix(audio_tower, prefix="audio_tower"),
                ModuleAndPrefix(projector, prefix="audio_encoder_proj"),
            ],
            device=device,
        )
        audio_tower.eval()
        projector.eval()
        return HiggsAudioEncoderSubmodule(
            audio_tower=audio_tower, projector=projector, config=self.config,
        )

    @staticmethod
    def _llm_remap(name: str) -> str | None:
        # Generation-only components (audio codec embeddings, audio LM
        # head) and the encoder/projector are not part of the LLM.
        if name.startswith((
            "audio_tower.", "audio_encoder_proj.",
            "audio_codebook_embeddings.",
        )):
            return None
        if name == "audio_decoder_proj.text_lm_head.weight":
            return "lm_head.weight"
        if name.startswith("audio_decoder_proj."):
            return None
        return name

    def _create_llm_submodule(
        self, device: str, tp_group, autocast_dtype: torch.dtype | None,
    ) -> NodeSubmodule:
        from mstar.model.components.qwen3_lm import Qwen3DenseLM, Qwen3LMConfig
        from mstar.model.higgs_audio.submodules import HiggsAudioLLMSubmodule
        from mstar.model.loader.iterators import iter_safetensors_shards

        # the checkpoint's flat layout (embed_tokens.*, layers.N.*, norm.*) is
        # Qwen3DenseLM's; only the text head is renamed
        with torch.device("meta"):
            llm = Qwen3DenseLM(
                Qwen3LMConfig.from_hf(asdict(self.config)),
                attn_key=ATTN, kv_key=KV_CACHE, pos_key=ROPE, comm_group=tp_group,
            )
        # Cast on meta (no allocation) so to_empty allocates directly in the
        # target dtype instead of fp32-then-downcast.
        if autocast_dtype is not None:
            llm = llm.to(autocast_dtype)
        llm.to_empty(device=device)
        weights = iter_safetensors_shards(self._resolve_weights(), device=device)
        llm.load_weights(
            (mapped, v) for k, v in weights if (mapped := self._llm_remap(k)) is not None
        )
        llm.eval()
        return HiggsAudioLLMSubmodule(llm, self.config, self.encoders, text_prompts=self.TEXT_PROMPTS)
