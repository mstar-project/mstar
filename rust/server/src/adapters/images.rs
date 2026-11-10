//! Image models: BAGEL (chat and images) and the DiT-scaffold image models.

use super::*;

/// Python's `int(s)` of a string: surrounding whitespace, a sign, and `_`
/// between digits.
fn py_int(s: &str) -> Option<i64> {
    static INT: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let int = INT.get_or_init(|| regex::Regex::new(r"^[+-]?[0-9](?:_?[0-9])*$").unwrap());
    let t = py_strip(s);
    int.is_match(t).then(|| t.replace('_', "").parse().ok()).flatten()
}

/// OpenAI `size` ("WxH") onto the model's `width` / `height`: the adapters'
/// `(int(v) for v in size.lower().split("x"))` into two names, refused with
/// the adapter's own `example`.
pub(super) fn size_kwargs(mk: &mut Map<String, Value>, size: Option<&str>, example: &str) -> Result<(), String> {
    let Some(size) = size.filter(|s| !s.is_empty()) else {
        return Ok(());
    };
    let lowered = size.to_lowercase();
    let parts: Vec<&str> = lowered.split('x').collect();
    let parsed = match parts.as_slice() {
        [w, h] => py_int(w).zip(py_int(h)),
        _ => None,
    };
    let Some((width, height)) = parsed else {
        return Err(format!("size must be 'WxH' (e.g. '{example}'); got {}", media::py_repr(size)));
    };
    set_default(mk, "width", Value::from(width));
    set_default(mk, "height", Value::from(height));
    Ok(())
}

/// `BagelAdapter.image_to_request` (`DiffusionImageAdapter`'s is the same, line for line).
pub(super) fn sized_image(req: &ImageGenerationRequest) -> Result<SubmitArgs, String> {
    let mut mk: Map<String, Value> = req.extra.clone().into_iter().collect();
    size_kwargs(&mut mk, req.size.as_deref(), "1024x1024")?;
    set_seed(&mut mk, req.seed);
    Ok(SubmitArgs {
        text: Some(req.prompt.clone()),
        input_modalities: vec!["text".to_string()],
        output_modalities: vec!["image".to_string()],
        model_kwargs: mk,
        ..SubmitArgs::default()
    })
}

/// `BagelAdapter.image_edit_to_request` (and `DiffusionImageAdapter`'s).
pub(super) fn sized_image_edit(prompt: &str, image_path: &str, mut extra_kwargs: Map<String, Value>) -> Result<SubmitArgs, String> {
    // a form field that parsed as JSON (e.g. "512") is still the size text
    let size = extra_kwargs
        .remove("size")
        .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string));
    size_kwargs(&mut extra_kwargs, size.as_deref(), "1024x1024")?;
    let mut file_paths = BTreeMap::new();
    file_paths.insert("image".to_string(), vec![image_path.to_string()]);
    Ok(SubmitArgs {
        text: Some(prompt.to_string()),
        file_paths,
        input_modalities: vec!["image".to_string(), "text".to_string()],
        output_modalities: vec!["image".to_string()],
        model_kwargs: extra_kwargs,
        ..SubmitArgs::default()
    })
}

/// BAGEL: chat (it reads temperature / top_p from its model config; only
/// max_output_tokens and seed are honored), images and edits.
pub struct Bagel;

impl OpenAIAdapter for Bagel {
    fn supports_chat(&self) -> bool {
        true
    }
    fn supports_images(&self) -> bool {
        true
    }
    fn chat_args(&self, req: &ChatCompletionRequest, upload_dir: &Path, allow_remote: bool) -> Result<SubmitArgs, String> {
        text_chat(req, upload_dir, allow_remote)
    }
    fn image_to_request(&self, req: &ImageGenerationRequest) -> Result<SubmitArgs, String> {
        sized_image(req)
    }
    fn image_edit_to_request(&self, prompt: &str, image_path: &str, extra_kwargs: Map<String, Value>) -> Result<SubmitArgs, String> {
        sized_image_edit(prompt, image_path, extra_kwargs)
    }
}

/// The DiT-scaffold image models (FLUX.2 klein, Z-Image): BAGEL's image translations.
pub struct DiffusionImage;

impl OpenAIAdapter for DiffusionImage {
    fn supports_images(&self) -> bool {
        true
    }
    fn image_to_request(&self, req: &ImageGenerationRequest) -> Result<SubmitArgs, String> {
        sized_image(req)
    }
    fn image_edit_to_request(&self, prompt: &str, image_path: &str, extra_kwargs: Map<String, Value>) -> Result<SubmitArgs, String> {
        sized_image_edit(prompt, image_path, extra_kwargs)
    }
}
