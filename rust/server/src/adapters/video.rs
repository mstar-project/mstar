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
        template_kwargs_chat(req, upload_dir, allow_remote)
    }
    fn image_to_request(&self, req: &ImageGenerationRequest) -> Result<SubmitArgs, String> {
        Cosmos3.image_to_request(req)
    }
    fn video_to_request(&self, req: &VideoGenerationRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        Cosmos3.video_to_request(req, upload_dir, allow_remote)
    }
}
