//! Video models: the Cosmos3 family.

use super::*;

/// Cosmos3: images and videos, `size` passed on as the model's own kwarg.
pub struct Cosmos3;

impl OpenAIAdapter for Cosmos3 {
    fn supports_images(&self) -> bool {
        true
    }
    fn supports_videos(&self) -> bool {
        true
    }
    fn image_to_request(&self, req: &ImageGenerationRequest) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        if let Some(size) = &req.size {
            set_default(&mut mk, "size", Value::from(size.clone()));
        }
        set_seed(&mut mk, req.seed);
        Ok(SubmitArgs {
            text: Some(req.prompt.clone()),
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["image".to_string()],
            model_kwargs: mk,
            ..SubmitArgs::default()
        })
    }
    /// Text-to-video, or image/video-to-video when a conditioning reference is
    /// supplied, as `Cosmos3Adapter.video_to_request`.
    fn video_to_request(&self, req: &VideoGenerationRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        if let Some(size) = &req.size {
            set_default(&mut mk, "size", Value::from(size.clone()));
        }
        set_seed(&mut mk, req.seed);
        // num_frames / fps are first-class video fields (not extra_body).
        if let Some(nf) = req.num_frames {
            set_default(&mut mk, "num_frames", Value::from(nf));
        }
        if let Some(fps) = req.fps {
            set_default(&mut mk, "fps", json_num(fps));
        }
        let mut args = SubmitArgs {
            text: Some(req.prompt.clone()),
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["video".to_string()],
            model_kwargs: mk,
            ..SubmitArgs::default()
        };
        // The conditioning frame (image-to-video) or clip (video-to-video) is
        // persisted and VAE-encoded by the worker into the clean frame-0 anchor /
        // pinned latent prefix. At most one may be given.
        let reference = match (req.image.as_deref(), req.video.as_deref()) {
            (Some(_), Some(_)) => return Err("Provide either 'image' or 'video' conditioning, not both.".to_string()),
            (Some(image), None) => Some(("image", image)),
            (None, Some(video)) => Some(("video", video)),
            (None, None) => None,
        };
        if let Some((modality, reference)) = reference {
            let (_, path) = media::resolve_media_ref(reference, upload_dir, allow_remote)?;
            args.file_paths.insert(modality.to_string(), vec![path]);
            args.input_modalities = vec![modality.to_string(), "text".to_string()];
        }
        Ok(args)
    }
}

/// Cosmos3-Edge: Cosmos3's images and videos, and chat.
pub struct Cosmos3Edge;

impl OpenAIAdapter for Cosmos3Edge {
    fn supports_chat(&self) -> bool {
        true
    }
    fn supports_images(&self) -> bool {
        true
    }
    fn supports_videos(&self) -> bool {
        true
    }
    fn chat_args(&self, req: &ChatCompletionRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        // How the vLLM recipe toggles thinking; a flat enable_thinking wins.
        // Read before the attachments are saved, since a malformed value
        // refuses the request.
        let thinking = template_enable_thinking(req.extra.get("chat_template_kwargs"))?;
        let mut args = text_chat(req, upload_dir, allow_remote)?;
        args.model_kwargs.remove("chat_template_kwargs");
        if let Some(on) = thinking {
            set_default(&mut args.model_kwargs, "enable_thinking", on);
        }
        Ok(args)
    }
    fn image_to_request(&self, req: &ImageGenerationRequest) -> Result<SubmitArgs, String> {
        Cosmos3.image_to_request(req)
    }
    fn video_to_request(&self, req: &VideoGenerationRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        Cosmos3.video_to_request(req, upload_dir, allow_remote)
    }
}

/// `chat_template_kwargs.enable_thinking` read as `Cosmos3EdgeAdapter` does:
/// `(kwargs or {})`, then `kwargs["enable_thinking"]` as sent when it has the
/// key, so the model refuses a non-boolean rather than reading `"false"` as
/// true. A falsy value is no kwargs; a value Python cannot look the key up in
/// is the `TypeError` that answers 400 there.
pub(super) fn template_enable_thinking(kwargs: Option<&Value>) -> Result<Option<Value>, String> {
    const KEY: &str = "enable_thinking";
    match kwargs {
        None => Ok(None),
        Some(v) if !truthy(v) => Ok(None),
        Some(Value::Object(m)) => Ok(m.get(KEY).cloned()),
        Some(Value::Array(items)) if items.iter().any(|i| i.as_str() == Some(KEY)) => {
            Err("list indices must be integers or slices, not str".to_string())
        }
        Some(Value::String(s)) if s.contains(KEY) => {
            Err("string indices must be integers, not 'str'".to_string())
        }
        Some(Value::Array(_) | Value::String(_)) => Ok(None),
        Some(Value::Bool(_)) => Err("argument of type 'bool' is not iterable".to_string()),
        Some(Value::Number(n)) => Err(format!(
            "argument of type '{}' is not iterable",
            if n.is_f64() { "float" } else { "int" }
        )),
        Some(Value::Null) => Ok(None),
    }
}
