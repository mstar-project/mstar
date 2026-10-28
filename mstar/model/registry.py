from importlib import import_module

from mstar.model.base import Model

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "bagel": ("mstar.model.bagel.bagel_model", "BagelModel"),
    "canary": ("mstar.model.canary.canary_model", "CanaryModel"),
    "canary_qwen": ("mstar.model.canary_qwen.canary_qwen_model", "CanaryQwenModel"),
    "chatterbox": ("mstar.model.chatterbox.chatterbox_model", "ChatterboxModel"),
    "chatterbox_multilingual": ("mstar.model.chatterbox.chatterbox_model", "ChatterboxModel"),
    "chatterbox_turbo": ("mstar.model.chatterbox.chatterbox_model", "ChatterboxModel"),
    "command_a_plus": ("mstar.model.command_a_plus.command_a_plus_model", "CommandAPlusModel"),
    "cosmos3": ("mstar.model.cosmos3.cosmos3_model", "Cosmos3Model"),
    "cosmos3_droid": ("mstar.model.cosmos3.cosmos3_model", "Cosmos3Model"),
    "cosmos3_edge": ("mstar.model.cosmos3.cosmos3_model", "Cosmos3Model"),
    "cosmos3_edge_droid": ("mstar.model.cosmos3.cosmos3_model", "Cosmos3Model"),
    "cosmos3_super": ("mstar.model.cosmos3.cosmos3_model", "Cosmos3Model"),
    "cosmos3_super_i2v_4step": ("mstar.model.cosmos3.cosmos3_model", "Cosmos3Model"),
    "cosmos3_super_t2i_4step": ("mstar.model.cosmos3.cosmos3_model", "Cosmos3Model"),
    "flux2_klein": ("mstar.model.flux2_klein.flux2_klein_model", "Flux2KleinModel"),
    "flux2_klein_9b": ("mstar.model.flux2_klein.flux2_klein_model", "Flux2KleinModel"),
    "fish_s2_pro": ("mstar.model.fish_s2_pro.fish_s2_pro_model", "FishS2ProModel"),
    "higgs_audio": ("mstar.model.higgs_audio.higgs_audio_model", "HiggsAudioModel"),
    "nemotron_duplex": ("mstar.model.nemotron_duplex.nemotron_duplex_model", "NemotronDuplexModel"),
    "omnilingual_asr_ctc_300m": ("mstar.model.omnilingual_asr.omnilingual_asr_model", "OmnilingualASRModel"),
    "omnilingual_asr_ctc_1b": ("mstar.model.omnilingual_asr.omnilingual_asr_model", "OmnilingualASRModel"),
    "omnilingual_asr_ctc_3b": ("mstar.model.omnilingual_asr.omnilingual_asr_model", "OmnilingualASRModel"),
    "omnilingual_asr_ctc_7b": ("mstar.model.omnilingual_asr.omnilingual_asr_model", "OmnilingualASRModel"),
    "omnivoice": ("mstar.model.omnivoice.omnivoice_model", "OmniVoiceModel"),
    "kokoro": ("mstar.model.kokoro.kokoro_model", "KokoroModel"),
    "ltx2_5": ("mstar.model.ltx2_5.ltx2_5_model", "LTX25Model"),
    "lingbot_world_1_3b": ("mstar.model.lingbot_world.lingbot_world_model", "LingBotWorldModel"),
    "nemotron_asr": ("mstar.model.parakeet.parakeet_model", "ParakeetModel"),
    "orpheus": ("mstar.model.orpheus.orpheus_model", "OrpheusModel"),
    "parakeet": ("mstar.model.parakeet.parakeet_model", "ParakeetModel"),
    "pi05": ("mstar.model.pi05.pi05_model", "Pi05Model"),
    "qwen3_omni": ("mstar.model.qwen3_omni.qwen3_omni_model", "Qwen3OmniModel"),
    "qwen3_asr": ("mstar.model.qwen3_asr.qwen3_asr_model", "Qwen3ASRModel"),
    "qwen3_asr_realtime": ("mstar.model.qwen3_asr.qwen3_asr_model", "Qwen3ASRModel"),
    "qwen3_tts": ("mstar.model.qwen3_tts.qwen3_tts_model", "Qwen3TTSModel"),
    "qwen3_tts_1p7b": ("mstar.model.qwen3_tts.qwen3_tts_model", "Qwen3TTSModel"),
    "qwen3_tts_voicedesign": ("mstar.model.qwen3_tts.qwen3_tts_model", "Qwen3TTSModel"),
    "qwen3_tts_base": ("mstar.model.qwen3_tts.qwen3_tts_model", "Qwen3TTSModel"),
    "vjepa2": ("mstar.model.vjepa2.vjepa2_model", "VJepa2Model"),
    "vjepa2_ac": ("mstar.model.vjepa2.vjepa2_model", "VJepa2ACModel"),
    "wan22": ("mstar.model.wan22.wan22_model", "Wan22Model"),
    "waypoint": ("mstar.model.waypoint.waypoint_model", "WaypointModel"),
    "whisper_large": ("mstar.model.whisper.whisper_model", "WhisperModel"),
    "whisper_large_v3_turbo": ("mstar.model.whisper.whisper_model", "WhisperModel"),
    "z_image_turbo": ("mstar.model.z_image.z_image_model", "ZImageModel"),
}

HF_MODELS: dict[str, dict] = {
    "bagel": {"model_path_hf": "ByteDance-Seed/BAGEL-7B-MoT"},
    # NVIDIA Canary-1B-v2: FastConformer encoder + Transformer decoder, speech
    # recognition in 25 European languages and translation to/from English.
    "canary": {"model_path_hf": "nvidia/canary-1b-v2"},
    # NVIDIA Canary-Qwen-2.5B: Canary's FastConformer encoder feeding a
    # LoRA-adapted Qwen3-1.7B (NeMo SALM); English speech recognition.
    "canary_qwen": {"model_path_hf": "nvidia/canary-qwen-2.5b"},
    # Resemble AI Chatterbox: Llama-520M T3 speech-token LM + S3Gen flow
    # matching decoder + HiFT vocoder, zero-shot voice cloning with CFG and
    # exaggeration control. The Turbo checkpoint is a GPT-2-medium T3 with a
    # distilled two-step decoder; same class, variant picked from the repo id.
    # The 23-language checkpoint lives in the same repo as the English one
    # (a T3 with a 2454-token grapheme vocabulary), so its variant is named.
    "chatterbox": {"model_path_hf": "ResembleAI/chatterbox"},
    "chatterbox_multilingual": {"model_path_hf": "ResembleAI/chatterbox", "variant": "multilingual"},
    "chatterbox_turbo": {"model_path_hf": "ResembleAI/chatterbox-turbo"},
    "command_a_plus": {"model_path_hf": "CohereLabs/command-a-plus-05-2026-bf16"},
    # NVIDIA Cosmos3-Nano generator (diffusers transformer/ + Wan VAE + UniPC).
    "cosmos3": {"model_path_hf": "nvidia/Cosmos3-Nano"},
    # Cosmos3-Nano-Policy-DROID — Nano-sized action-policy fine-tune for the
    # DROID robot platform (domain droid_lerobot, 10-dim raw actions). Same
    # class; the checkpoint's config disables the sound pathway (sound_gen
    # false, no sound_tokenizer/), so the model self-serves without audio.
    "cosmos3_droid": {"model_path_hf": "nvidia/Cosmos3-Nano-Policy-DROID"},
    # Cosmos3-Edge (4B) — same class; the dense Nemotron backbone family
    # (relu2 MLPs, Nemotron norms, no text QK-norm, k_norm_und_for_gen),
    # 480p-native generation and the reasoner (understanding tower + SigLIP2
    # vision encoder served as a VLM) all load from the checkpoint's configs.
    "cosmos3_edge": {"model_path_hf": "nvidia/Cosmos3-Edge"},
    # Edge action-policy fine-tune for DROID (domain droid_lerobot); Edge
    # backbone, no reasoner weights beyond the shared text tower.
    "cosmos3_edge_droid": {"model_path_hf": "nvidia/Cosmos3-Edge-Policy-DROID"},
    # Cosmos3-Super (64B) — same architecture + class; dims (64 layers / 5120
    # hidden / 25600 intermediate) load from the checkpoint's config.json, so it
    # needs tensor parallelism (it does not fit on one GPU).
    "cosmos3_super": {"model_path_hf": "nvidia/Cosmos3-Super"},
    # 4-step distilled Super task checkpoints (guidance baked in; a fixed
    # 4-sigma stochastic Euler sampler instead of UniPC). Same class + TP
    # deployment as Super.
    "cosmos3_super_i2v_4step": {"model_path_hf": "nvidia/Cosmos3-Super-Image2Video-4Step"},
    "cosmos3_super_t2i_4step": {"model_path_hf": "nvidia/Cosmos3-Super-Text2Image-4Step"},
    # FLUX.2 [klein]: step-distilled rectified-flow text-to-image + multi-reference
    # editing (Qwen3 hidden-state text encoder, FLUX.2 VAE). 4B is Apache-2.0; 9B is
    # under the FLUX Non-Commercial License. Same class, all dims from the checkpoint.
    "flux2_klein": {"model_path_hf": "black-forest-labs/FLUX.2-klein-4B"},
    # Lightricks LTX-2.5: 22B joint audio-video DiT (distilled), Gemma-4 12B text
    # encoder, video + audio VAEs and a 48 kHz vocoder. Gated on Hugging Face.
    "ltx2_5": {"model_path_hf": "Lightricks/LTX-2.5-Diffusers"},
    "flux2_klein_9b": {"model_path_hf": "black-forest-labs/FLUX.2-klein-9B"},
    # Fish Audio S2-Pro: dual-AR TTS (Qwen3-4B Slow AR + 4-layer Fast AR over
    # 10 codebooks) with a 44.1 kHz causal DAC codec and reference-audio voice
    # cloning. Weights under the Fish Audio Research License (non-commercial).
    "fish_s2_pro": {"model_path_hf": "fishaudio/s2-pro"},
    # Higgs-Audio v3 STT: Whisper-style audio tower + Qwen3-1.7B LLM.
    # (The v2 checkpoints are TTS/generation models, not ASR.)
    "higgs_audio": {"model_path_hf": "bosonai/higgs-audio-v3-stt"},
    # NVIDIA NemotronLabs VoiceChat-11B: full-duplex S2S -- Fast-Conformer STT
    # encoder + Nemotron-H hybrid Mamba-2/attn/MLP backbone (9B) + EarTTS
    # (Gemma3 talker + RVQ codec). Single composite ``model.safetensors``.
    "nemotron_duplex": {"model_path_hf": "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"},
    # Meta Omnilingual ASR, CTC variants: wav2vec 2.0 encoder + CTC head, 1,600+
    # languages, no language code needed. Apache-2.0. Meta's own .pt files: v2
    # from dl.fbaipublicfiles.com (pinned sha256), or v1 from the
    # facebook/omniASR-CTC-* Hub repos with weights_version: v1 (config.py).
    # No HF repo id: the size picks the checkpoint.
    "omnilingual_asr_ctc_300m": {"model_path_hf": None, "size": "300m"},
    "omnilingual_asr_ctc_1b": {"model_path_hf": None, "size": "1b"},
    "omnilingual_asr_ctc_3b": {"model_path_hf": None, "size": "3b"},
    "omnilingual_asr_ctc_7b": {"model_path_hf": None, "size": "7b"},
    # OmniVoice — masked-diffusion TTS, 600+ languages. A Qwen3-0.6B body read
    # bidirectionally over an 8-codebook canvas, plus the Higgs-Audio-v2 codec
    # shipped in the checkpoint's audio_tokenizer/ subfolder (~3.3 GB total).
    "omnivoice": {"model_path_hf": "k2-fsa/OmniVoice"},
    # Kokoro-82M TTS: misaki G2P + PL-BERT prosody + iSTFTNet decoder, 54 bundled voices.
    "kokoro": {"model_path_hf": "hexgrad/Kokoro-82M"},
    # NVIDIA Nemotron-3.5-ASR: cache-aware streaming FastConformer + RNN-T with
    # a language prompt, 40 language-locales; served here over whole clips.
    "nemotron_asr": {"model_path_hf": "nvidia/nemotron-3.5-asr-streaming-0.6b"},
    "orpheus": {"model_path_hf": "canopylabs/orpheus-3b-0.1-ft"},
    # NVIDIA Parakeet-TDT-0.6B-v3: FastConformer + token-and-duration
    # transducer, 25 European languages, language detected from the speech.
    "parakeet": {"model_path_hf": "nvidia/parakeet-tdt-0.6b-v3"},
    # Pi0.5 PyTorch port published by lerobot — single safetensors blob
    # (~14 GB). mstar/model/pi05/weight_loader.py handles the lerobot->mstar
    # state-dict remap inside Pi05Model.get_submodule().
    "pi05": {"model_path_hf": "lerobot/pi05_base"},
    "qwen3_omni": {"model_path_hf": "Qwen/Qwen3-Omni-30B-A3B-Instruct"},
    # Qwen3-ASR: AuT audio encoder + dense Qwen3 decoder. ``qwen3_asr`` is
    # the 1.7B transcription model, ``qwen3_asr_realtime`` the 0.6B variant
    # trained for chunked streaming, served by the same class.
    "qwen3_asr": {"model_path_hf": "Qwen/Qwen3-ASR-1.7B"},
    "qwen3_asr_realtime": {"model_path_hf": "Qwen/Qwen3-ASR-0.6B"},
    # Qwen3-TTS 12 Hz family: one class, the variant is read from config.json.
    # CustomVoice = built-in speakers (1.7B also takes style instructions),
    # VoiceDesign = voice described by an instruction, Base = voice cloned
    # from reference audio.
    "qwen3_tts": {"model_path_hf": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"},
    "qwen3_tts_1p7b": {"model_path_hf": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"},
    "qwen3_tts_voicedesign": {"model_path_hf": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"},
    "qwen3_tts_base": {"model_path_hf": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"},
    # V-JEPA 2 standard (encoder + masked predictor).  Default is ViT-L @ 256
    # (~300M); the same class loads vitl/h/g at 256 or 384 by reading
    # config.json.
    "vjepa2": {"model_path_hf": "facebook/vjepa2-vitl-fpc64-256"},
    # V-JEPA 2-AC (encoder + action-conditioned predictor).  HF doesn't host
    # an AC checkpoint; weights come from the public S3 mirror
    # ``https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`` via
    # ``download_vjepa2_ac_upstream_pt`` — the ``model_path_hf`` string is
    # kept as a logical identifier but isn't resolved against HuggingFace.
    "vjepa2_ac": {"model_path_hf": "vjepa2-ac-vitg"},
    # Wan2.2-TI2V-5B (dense video DiT + UMT5-XXL + Wan2.2-VAE).  TI2V-5B
    # only; the A14B MoE variants are a separate follow-up.
    "wan22": {"model_path_hf": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"},
    # LingBot-World-V2 1.3B causal-fast (weights CC-BY-NC-SA-4.0): the DiT repo only;
    # UMT5 and the Wan2.1 VAE load from the diffusers repos mstar already uses.
    "lingbot_world_1_3b": {"model_path_hf": "robbyant/lingbot-world-v2-1.3b-causal-fast"},
    # Waypoint owns a variant -> Hub repository mapping. None is intentional:
    # it lets WaypointModel distinguish the registry default from an explicit
    # local path or Hub ID, then select the 720P or 360P repository named by the
    # YAML `variant`. TAEHV is resolved independently from config.ae_uri.
    "waypoint": {"model_path_hf": None},
    # Whisper works for any size (dims and token ids come from the
    # checkpoint's configs). ``whisper_large`` pins large-v3, the standard
    # ASR-benchmark checkpoint. ``whisper_large_v3_turbo`` is the 4-decoder-
    # layer distillation that serves the same encoder several times faster.
    "whisper_large": {"model_path_hf": "openai/whisper-large-v3"},
    "whisper_large_v3_turbo": {"model_path_hf": "openai/whisper-large-v3-turbo"},
    # Z-Image-Turbo: 8-step distilled single-stream flow DiT (Qwen3-4B caption encoder,
    # FLUX.1 VAE), text-to-image only. Apache-2.0.
    "z_image_turbo": {"model_path_hf": "Tongyi-MAI/Z-Image-Turbo"},
}

# qwen 3.5 family: dense (MoE TODO)
qwen_3_5_dense_sizes = ("0.8", "2", "4", "9", "27")
MODEL_REGISTRY.update({
    f"qwen3_5_{size}b": ("mstar.model.qwen3_5.qwen3_5_model", "Qwen3_5DenseModel") \
        for size in qwen_3_5_dense_sizes
})
HF_MODELS.update({
    f"qwen3_5_{size}b": {"model_path_hf": f"Qwen/Qwen3.5-{size}B"} \
        for size in qwen_3_5_dense_sizes
})
# Qwen3.8-27B is the Qwen3.5 architecture (same config.json geometry)
MODEL_REGISTRY["qwen3_8_27b"] = ("mstar.model.qwen3_5.qwen3_5_model", "Qwen3_5DenseModel")
HF_MODELS["qwen3_8_27b"] = {"model_path_hf": "Qwen/Qwen3.8-27B"}


def model_init_kwargs(name: str) -> dict:
    """Constructor kwargs the registry pins for ``name``: the checkpoint repo
    and, where one repo serves several models, the variant."""
    return {"model_path_hf": "", **HF_MODELS.get(name, {})}


def get_model_class(name: str) -> type[Model]:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model name: {name!r}. Available: {list(MODEL_REGISTRY.keys())}")
    module_name, class_name = MODEL_REGISTRY[name]
    return getattr(import_module(module_name), class_name)
