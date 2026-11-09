//! Video models: Cosmos3, Wan2.2, LTX-2.5 and MiniMax-H3.

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
        if let Some(size) = nonempty(&req.size) {
            set_default(&mut mk, "size", Value::from(size));
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
        if let Some(size) = nonempty(&req.size) {
            set_default(&mut mk, "size", Value::from(size));
        }
        video_fields(&mut mk, req);
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
        let reference = match (nonempty(&req.image), nonempty(&req.video)) {
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

/// Wan2.2-TI2V-5B: text- or image-to-video, `size` as the model's width and height.
pub struct Wan22;

impl OpenAIAdapter for Wan22 {
    fn supports_videos(&self) -> bool {
        true
    }
    fn video_to_request(&self, req: &VideoGenerationRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        super::images::size_kwargs(&mut mk, nonempty(&req.size), "832x480")?;
        video_fields(&mut mk, req);
        if nonempty(&req.video).is_some() {
            return Err("Wan2.2 does not support 'video' conditioning; provide 'image' or neither.".to_string());
        }
        let mut args = SubmitArgs {
            text: Some(req.prompt.clone()),
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["video".to_string()],
            model_kwargs: mk,
            ..SubmitArgs::default()
        };
        if let Some(image) = nonempty(&req.image) {
            let (_, path) = media::resolve_media_ref(image, upload_dir, allow_remote)?;
            args.file_paths.insert("image".to_string(), vec![path]);
            args.input_modalities = vec!["image".to_string(), "text".to_string()];
        }
        Ok(args)
    }
}

/// LTX-2.5: text to video, with its generated audio unless `audio` is false.
pub struct LTX25;

impl OpenAIAdapter for LTX25 {
    fn supports_videos(&self) -> bool {
        true
    }
    fn video_to_request(&self, req: &VideoGenerationRequest, _upload_dir: &Path, _allow_remote: bool) -> Result<SubmitArgs, String> {
        if nonempty(&req.image).is_some() || nonempty(&req.video).is_some() {
            return Err("LTX-2.5 (this port) is text-to-video only; omit 'image' and 'video'.".to_string());
        }
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        // a bool, not truthiness: "false" would be sound
        let with_audio = match mk.remove("audio") {
            None | Some(Value::Null) => true,
            Some(Value::Bool(b)) => b,
            Some(other) => return Err(format!("audio must be true or false, got {}", py_value_repr(&other))),
        };
        super::images::size_kwargs(&mut mk, nonempty(&req.size), "960x544")?;
        video_fields(&mut mk, req);
        let output = if with_audio { vec!["video", "audio"] } else { vec!["video"] };
        Ok(SubmitArgs {
            text: Some(req.prompt.clone()),
            input_modalities: vec!["text".to_string()],
            output_modalities: output.into_iter().map(str::to_string).collect(),
            model_kwargs: mk,
            ..SubmitArgs::default()
        })
    }
}

/// MiniMax-H3: one mp4 with its 32 kHz stereo audio as AAC (`audio: false` for
/// the video alone), `image` its first frame and `last_image` its last. Only 24 fps.
pub struct MiniMaxH3;

impl OpenAIAdapter for MiniMaxH3 {
    fn supports_videos(&self) -> bool {
        true
    }
    fn video_to_request(&self, req: &VideoGenerationRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        if nonempty(&req.video).is_some() {
            return Err("MiniMax-H3 takes keyframes ('image', extra_body 'last_image'), not a 'video'.".to_string());
        }
        if let Some(fps) = req.fps.filter(|f| *f != 24.0) {
            return Err(format!("MiniMax-H3 generates at 24 fps; got fps={}", py_float_repr(fps)));
        }
        if let Some(n) = req.n.filter(|n| *n != 1) {
            return Err(format!("MiniMax-H3 generates one video per request; got n={n}"));
        }
        let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
        let with_audio = match mk.remove("audio") {
            None | Some(Value::Null) => true,
            Some(Value::Bool(b)) => b,
            Some(other) => return Err(format!("extra_body 'audio' must be true or false; got {}", py_value_repr(&other))),
        };
        let last_image = mk.remove("last_image").filter(|v| !v.is_null());
        super::images::size_kwargs(&mut mk, nonempty(&req.size), "1344x768")?;
        set_seed(&mut mk, req.seed);
        if let Some(nf) = req.num_frames {
            set_default(&mut mk, "num_frames", Value::from(nf));
        }
        let refs: Vec<(&str, Value)> = [("first", req.image.clone().map(Value::String)), ("last", last_image)]
            .into_iter()
            .filter_map(|(anchor, r)| r.map(|r| (anchor, r)))
            .collect();
        for (anchor, r) in &refs {
            if !r.is_string() {
                let name = if *anchor == "first" { "image" } else { "extra_body 'last_image'" };
                return Err(format!("{name} must be a data URL; got {}", py_type_name(r)));
            }
        }
        let mut paths: Vec<String> = Vec::new();
        for (_, r) in &refs {
            match media::resolve_media_ref(r.as_str().unwrap_or(""), upload_dir, allow_remote) {
                Ok((_, path)) => paths.push(path),
                Err(e) => {
                    // a refused request is never submitted, so no deferred cleanup
                    // covers the keyframe saved before the refusal
                    remove_under(upload_dir, &paths);
                    return Err(e);
                }
            }
        }
        if !paths.is_empty() {
            mk.insert("keyframe_anchors".to_string(), refs.iter().map(|(a, _)| Value::from(*a)).collect());
        }
        let input = if paths.is_empty() { vec!["text"] } else { vec!["image", "text"] };
        let output = if with_audio { vec!["video", "audio"] } else { vec!["video"] };
        Ok(SubmitArgs {
            text: Some(req.prompt.clone()),
            file_paths: if paths.is_empty() { BTreeMap::new() } else { BTreeMap::from([("image".to_string(), paths)]) },
            input_modalities: input.into_iter().map(str::to_string).collect(),
            output_modalities: output.into_iter().map(str::to_string).collect(),
            model_kwargs: mk,
            ..SubmitArgs::default()
        })
    }
}

/// `type(value).__name__` of a JSON value as `json.loads` gives it.
fn py_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// `media_io.remove_uploads`: the files under `upload_dir` among `paths`.
fn remove_under(upload_dir: &Path, paths: &[String]) {
    let Ok(root) = upload_dir.canonicalize() else { return };
    for p in paths {
        if Path::new(p).canonicalize().is_ok_and(|c| c.starts_with(&root)) {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// A string field as Python's truthiness reads it: `None` when unset or empty.
fn nonempty(field: &Option<String>) -> Option<&str> {
    field.as_deref().filter(|s| !s.is_empty())
}

/// The first-class video fields every video adapter forwards when set.
fn video_fields(mk: &mut Map<String, Value>, req: &VideoGenerationRequest) {
    set_seed(mk, req.seed);
    if let Some(nf) = req.num_frames {
        set_default(mk, "num_frames", Value::from(nf));
    }
    if let Some(fps) = req.fps {
        set_default(mk, "fps", json_num(fps));
    }
}
