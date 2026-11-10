//! Chat models: Qwen3.5, Qwen3-Omni (chat and speech), and text-only LLMs.

use super::*;

/// Qwen3.5: multimodal chat.
pub struct Qwen3_5;

impl OpenAIAdapter for Qwen3_5 {
    fn supports_chat(&self) -> bool {
        true
    }
    fn chat_args(&self, req: &ChatCompletionRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        template_kwargs_chat(req, upload_dir, allow_remote)
    }
}

/// Qwen3-Omni: chat (text, or text and audio) and speech, sampling the Thinker
/// for a chat and the Talker for a speech.
pub struct Qwen3Omni;

impl OpenAIAdapter for Qwen3Omni {
    fn supports_chat(&self) -> bool {
        true
    }
    fn supports_speech(&self) -> bool {
        true
    }
    fn chat_args(&self, req: &ChatCompletionRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        let (text, file_paths, in_mods, parts) = flatten_messages(&req.messages, upload_dir, allow_remote)?;
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        // Speech output also emits text, so request both when audio asked.
        let want_audio = req.modalities.as_ref().is_some_and(|m| m.iter().any(|x| x == "audio"));
        let out_mods = if want_audio { vec!["text", "audio"] } else { vec!["text"] };
        apply_sampling(
            &mut mk,
            Sampling {
                temperature: req.temperature.clone(),
                top_p: req.top_p.clone(),
                seed: req.seed,
                max_tokens: chat_max_tokens(req),
                temperature_key: "thinker_temperature",
                top_p_key: "thinker_top_p",
                max_tokens_key: Some("max_output_tokens"),
            },
        );
        if let Some(voice) = chat_voice(req) {
            mk.insert("voice".to_string(), Value::from(voice));
        }
        Ok(SubmitArgs {
            tokens: None,
            text,
            file_paths,
            input_modalities: in_mods,
            output_modalities: out_mods.into_iter().map(str::to_string).collect(),
            model_kwargs: mk,
            parts,
            ..SubmitArgs::default()
        })
    }
    fn speech_args(&self, req: &SpeechRequest) -> Result<SubmitArgs, String> {
        let mut mk = speech_kwargs(req, "talker_temperature", "talker_top_p");
        Ok(SubmitArgs { output_modalities: vec!["text".to_string(), "audio".to_string()], ..speech_submission(req, mk) })
    }
}

/// Text-only LLMs (GLM-5.x): chat, refusing what a text-only model cannot serve.
pub struct TextChat;

impl OpenAIAdapter for TextChat {
    fn supports_chat(&self) -> bool {
        true
    }
    fn chat_args(&self, req: &ChatCompletionRequest, upload_dir: &Path, _allow_remote: bool) -> Result<SubmitArgs, String> {
        text_only_chat(req, upload_dir)
    }
}
