//! Speech models (`/v1/audio/speech`), and the speech translations they share.

use super::*;

/// A speech's model kwargs: the passthrough fields, `voice` when sent, and the
/// standard sampling fields under the model's names.
pub(super) fn speech_kwargs(req: &SpeechRequest, temperature_key: &str, top_p_key: &str) -> Map<String, Value> {
    let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
    if let Some(v) = &req.voice {
        mk.insert("voice".to_string(), Value::from(v.clone()));
    }
    sampling(&mut mk, req, temperature_key, top_p_key);
    mk
}

/// `_apply_sampling` for a speech: `temperature`, `top_p` and `seed`, no token cap.
pub(super) fn sampling(mk: &mut Map<String, Value>, req: &SpeechRequest, temperature_key: &str, top_p_key: &str) {
    apply_sampling(
        mk,
        Sampling {
            temperature: req.temperature.clone(),
            top_p: req.top_p.clone(),
            seed: req.seed,
            max_tokens: None,
            temperature_key,
            top_p_key,
            max_tokens_key: None,
        },
    );
}

/// An audio-out submission of the speech's input.
pub(super) fn speech_submission(req: &SpeechRequest, model_kwargs: Map<String, Value>) -> SubmitArgs {
    SubmitArgs {
        text: Some(req.input.clone()),
        input_modalities: vec!["text".to_string()],
        output_modalities: vec!["audio".to_string()],
        model_kwargs,
        ..SubmitArgs::default()
    }
}

/// `SpeechRequest.speed` as Python reads it: unsent is its default 1.0, `null` None.
pub(super) fn speech_speed(req: &SpeechRequest) -> Option<f64> {
    match req.speed {
        None => Some(1.0),
        Some(sent) => sent,
    }
}

/// OpenAI's `instructions` as the model's `instruct`, when non-empty.
pub(super) fn instructions_as_instruct(mk: &mut Map<String, Value>) {
    if let Some(instructions) = mk.remove("instructions").filter(truthy) {
        set_default(mk, "instruct", instructions);
    }
}

/// Orpheus: speech.
pub struct Orpheus;

impl OpenAIAdapter for Orpheus {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk = speech_kwargs(req, "temperature", "top_p");
        forward_sent_speed(&mut mk, req);
        Ok(speech_submission(req, mk))
    }
}

/// Qwen3-TTS (CustomVoice, VoiceDesign, Base): speech, with `instructions` as
/// `instruct`, long input chunked, and a Base model's reference clip first.
pub struct Qwen3TTS;

impl OpenAIAdapter for Qwen3TTS {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_chunking(&self) -> SpeechChunking {
        // each Talker request stays near the lengths the model was trained on
        SpeechChunking { min_chars: Some(600), ..SPEECH_CHUNKING }
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        if let Some(v) = &req.voice {
            mk.insert("voice".to_string(), Value::from(v.clone()));
        }
        // OpenAI's field is `instructions`; the model reads `instruct`.
        instructions_as_instruct(&mut mk);
        sampling(&mut mk, req, "temperature", "top_p");
        forward_sent_speed(&mut mk, req);
        Ok(speech_submission(req, mk))
    }
    fn ref_audio(&self) -> Option<RefAudio> {
        Some(RefAudio { decoder: "soundfile", clip_first: true })
    }
}

/// Kokoro: speech, forwarding `voice` and `speed`, which are real model knobs.
pub struct Kokoro;

impl OpenAIAdapter for Kokoro {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        if let Some(voice) = &req.voice {
            set_default(&mut mk, "voice", Value::from(voice.clone()));
        }
        if let Some(speed) = speech_speed(req) {
            set_default(&mut mk, "speed", json_num(speed));
        }
        set_seed(&mut mk, req.seed);
        Ok(speech_submission(req, mk))
    }
}

/// Chatterbox: speech with the standard sampling fields, cloning a voice from
/// a clip its loader opens with soundfile, else FFmpeg.
pub struct Chatterbox;

impl OpenAIAdapter for Chatterbox {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        if let Some(voice) = &req.voice {
            set_default(&mut mk, "voice", Value::from(voice.clone()));
        }
        sampling(&mut mk, req, "temperature", "top_p");
        forward_sent_speed(&mut mk, req);
        Ok(speech_submission(req, mk))
    }
    fn ref_audio(&self) -> Option<RefAudio> {
        Some(RefAudio { decoder: "soundfile+torchcodec", clip_first: false })
    }
}

/// Fish Audio S2-Pro: speech with the standard sampling fields, long input
/// chunked; `voice` and `speed` are no controls of it.
pub struct FishS2Pro;

impl OpenAIAdapter for FishS2Pro {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_chunking(&self) -> SpeechChunking {
        SpeechChunking { min_chars: Some(600), ..SPEECH_CHUNKING }
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        sampling(&mut mk, req, "temperature", "top_p");
        // a transcript without its clip
        if !mk.get("ref_audio").is_some_and(truthy) {
            mk.remove("ref_text");
        }
        Ok(speech_submission(req, mk))
    }
    fn ref_audio(&self) -> Option<RefAudio> {
        Some(RefAudio { decoder: "soundfile", clip_first: false })
    }
}

/// Fun-CosyVoice3: speech that always clones a voice, cut by its own splitter.
pub struct CosyVoice3;

impl OpenAIAdapter for CosyVoice3 {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_chunking(&self) -> SpeechChunking {
        // 80 tokens at ~4 characters each; nothing under 80 characters of Chinese is split
        SpeechChunking {
            min_chars: Some(81),
            max_chars: 320,
            splitter: crate::speech_chunking::Splitter::CosyVoice3,
            ..SPEECH_CHUNKING
        }
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        if let Some(voice) = req.voice.as_ref().filter(|v| !v.is_empty()) {
            set_default(&mut mk, "voice", Value::from(voice.clone()));
        }
        instructions_as_instruct(&mut mk);
        sampling(&mut mk, req, "temperature", "top_p");
        Ok(speech_submission(req, mk))
    }
    fn ref_audio(&self) -> Option<RefAudio> {
        Some(RefAudio { decoder: "soundfile", clip_first: false })
    }
}

/// OmniVoice: speech whose mode follows from what is sent; `voice` stands in
/// for `instruct` and `speed` is a real knob.
pub struct OmniVoice;

impl OpenAIAdapter for OmniVoice {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        // is-not-None, so an empty voice reaches the model and is refused there
        if let Some(voice) = &req.voice {
            if mk.get("instruct").is_none_or(Value::is_null) {
                mk.insert("instruct".to_string(), Value::from(voice.clone()));
            }
        }
        mk.remove("voice");
        if let Some(speed) = speech_speed(req) {
            set_default(&mut mk, "speed", json_num(speed));
        }
        set_seed(&mut mk, req.seed);
        Ok(speech_submission(req, mk))
    }
    fn ref_audio(&self) -> Option<RefAudio> {
        Some(RefAudio { decoder: "soundfile+torchcodec", clip_first: false })
    }
}

/// VoxCPM2: speech designing a voice from `instructions` or cloning one; it has
/// no named voices and no speed control.
pub struct VoxCPM2;

impl OpenAIAdapter for VoxCPM2 {
    fn supports_speech(&self) -> bool {
        true
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        instructions_as_instruct(&mut mk);
        set_seed(&mut mk, req.seed);
        Ok(speech_submission(req, mk))
    }
    fn ref_audio(&self) -> Option<RefAudio> {
        Some(RefAudio { decoder: "soundfile", clip_first: false })
    }
}
