use super::*;
use crate::protocol::ChatMessage;

type Flattened = (Option<String>, BTreeMap<String, Vec<String>>, Vec<String>, Vec<Part>);

fn flatten(messages: &str, upload_dir: &Path) -> Result<Flattened, String> {
    let messages: Vec<ChatMessage> = serde_json::from_str(messages).unwrap();
    flatten_messages(&messages, upload_dir, false)
}

fn described(parts: &[Part]) -> Vec<(&str, &str, Option<&str>, usize)> {
    parts
        .iter()
        .map(|p| (p.modality.as_str(), p.role.as_deref().unwrap_or(""), p.text.as_deref(), p.index))
        .collect()
}

#[test]
fn a_reply_stays_its_own_part() {
    let (text, _, in_mods, parts) = flatten(
        r#"[{"role":"system","content":"Be brief."},
            {"role":"user","content":"Name a color."},
            {"role":"assistant","content":"Blue."},
            {"role":"user","content":"Another."}]"#,
        Path::new("/unused"),
    )
    .unwrap();
    assert_eq!(text.as_deref(), Some("Be brief.\nName a color.\nBlue.\nAnother."));
    assert_eq!(in_mods, ["text", "text", "text", "text"]);
    assert_eq!(
        described(&parts),
        [
            ("text", "system", Some("Be brief."), 0),
            ("text", "user", Some("Name a color."), 0),
            ("text", "assistant", Some("Blue."), 0),
            ("text", "user", Some("Another."), 0),
        ],
        "text merged across a role change, so the reply lands inside a user turn"
    );
}

#[test]
fn messages_of_one_role_stay_one_part() {
    let (_, _, in_mods, parts) = flatten(
        r#"[{"role":"user","content":"First."},{"role":"user","content":"Second."}]"#,
        Path::new("/unused"),
    )
    .unwrap();
    assert_eq!(in_mods, ["text"]);
    assert_eq!(
        described(&parts),
        [("text", "user", Some("First.\nSecond."), 0)],
        "two user messages no longer render as the one turn they do on the default server"
    );
}

#[test]
fn a_developer_message_is_a_system_message() {
    let (_, _, _, parts) = flatten(
        r#"[{"role":"developer","content":"Be brief."},{"role":"user","content":"Hi."}]"#,
        Path::new("/unused"),
    )
    .unwrap();
    assert_eq!(
        described(&parts),
        [("text", "system", Some("Be brief."), 0), ("text", "user", Some("Hi."), 0)],
        "a developer message reached the templates under a role they do not render"
    );
}

#[test]
fn a_role_a_chat_cannot_render_is_refused() {
    let err = flatten(r#"[{"role":"tool","content":"42"}]"#, Path::new("/unused")).unwrap_err();
    assert_eq!(err, "a message's role must be system, developer, user or assistant, not 'tool'");
}

/// The surfaces a class serves: chat, speech, images, videos.
fn surfaces(adapter: &dyn OpenAIAdapter) -> [bool; 5] {
    [
        adapter.supports_chat(),
        adapter.supports_speech(),
        adapter.supports_images(),
        adapter.supports_videos(),
        adapter.supports_transcriptions(),
    ]
}

#[test]
fn each_ported_python_adapter_class_is_found_by_its_name_once() {
    let mut names: Vec<&str> = ADAPTERS.iter().map(|(name, _)| *name).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), ADAPTERS.len(), "a class listed twice");
    for (class, adapter) in ADAPTERS {
        assert_eq!(surfaces(from_python(class).unwrap()), surfaces(*adapter), "{class}");
    }
    for class in ["", "Pi05Adapter", "qwen3_5_0.8b", "bagel"] {
        assert!(from_python(class).is_none(), "{class:?}");
    }
    // the surfaces of a few classes, as their Python attributes declare them
    assert_eq!(surfaces(&Bagel), [true, false, true, false, false]);
    assert_eq!(surfaces(&Qwen3Omni), [true, true, false, false, false]);
    assert_eq!(surfaces(&Qwen3_5), [true, false, false, false, false]);
    assert_eq!(surfaces(&Cosmos3), [false, false, true, true, false]);
    assert_eq!(surfaces(&Cosmos3Edge), [true, false, true, true, false]);
    assert_eq!(surfaces(&Qwen3TTS), [false, true, false, false, false]);
    assert_eq!(surfaces(&Orpheus), [false, true, false, false, false]);
    assert_eq!(surfaces(&Whisper), [false, false, false, false, true]);
}

fn speech(body: &str) -> SpeechRequest {
    serde_json::from_str(body).unwrap()
}

#[test]
fn a_qwen3_tts_speech_maps_instructions_and_sampling() {
    let args = Qwen3TTS
        .speech_args(&speech(
            r#"{"input":"Hello.","voice":"ryan","language":"English","instructions":"calm",
                "temperature":0.7,"top_p":0.8,"seed":4,"speed":1.25,"top_k":20,"stream":true}"#,
        ))
        .unwrap();
    assert_eq!(args.text.as_deref(), Some("Hello."));
    assert_eq!((args.input_modalities, args.output_modalities), (vec!["text".to_string()], vec!["audio".to_string()]));
    assert!(args.file_paths.is_empty());
    assert_eq!(Value::Object(args.model_kwargs), serde_json::json!({
        "voice": "ryan", "language": "English", "instruct": "calm", "temperature": 0.7,
        "top_p": 0.8, "seed": 4, "top_k": 20,
    }));
}

#[test]
fn a_qwen3_tts_speech_keeps_an_explicit_instruct_and_drops_empty_instructions() {
    let mk = |body: &str| Value::Object(Qwen3TTS.speech_args(&speech(body)).unwrap().model_kwargs);
    assert_eq!(mk(r#"{"input":"x","instructions":"calm","instruct":"bright"}"#), serde_json::json!({"instruct": "bright"}));
    assert_eq!(mk(r#"{"input":"x","instructions":""}"#), serde_json::json!({}));
    // an empty clip is dropped, as the Python adapter pops it
    let speech = Qwen3TTS.speech_to_request(&speech(r#"{"input":"x","ref_audio":""}"#), Path::new("/unused"), false);
    assert_eq!(Value::Object(speech.unwrap().args.model_kwargs), serde_json::json!({}));
}

#[test]
fn no_speech_adapter_forwards_speed_itself() {
    // `audio_speech` forwards a sent speed once for every adapter, as Python's speech_request does
    for (name, adapter) in super::ADAPTERS.iter().filter(|(_, adapter)| adapter.supports_speech()) {
        for body in [r#"{"input":"x"}"#, r#"{"input":"x","speed":"2"}"#] {
            // an adapter that needs more than `input` (a voice, a clip) refuses it; nothing to check
            let Ok(args) = adapter.speech_args(&speech(body)) else { continue };
            assert!(!args.model_kwargs.contains_key("speed"), "{name}: {body}");
        }
    }
}

#[test]
fn a_qwen3_tts_voice_clone_saves_its_clip_for_the_loader_to_check() {
    let dir = std::env::temp_dir().join(format!("mstar_ref_{}", uuid::Uuid::new_v4().simple()));
    let clone = |body: &str| Qwen3TTS.speech_to_request(&speech(body), &dir, false);
    for (body, bare) in [
        (r#"{"input":"x","ref_audio":"data:audio/wav;base64,UklGRg=="}"#, false),
        (r#"{"input":"x","ref_audio":"UklGRg"}"#, true), // bare and unpadded, as Python reads it
    ] {
        let Speech { args, ref_audio } = clone(body).unwrap();
        let check = ref_audio.unwrap();
        assert_eq!((check.decoder, check.bare), ("soundfile", bare), "{body}");
        assert_eq!(std::fs::read(&check.path).unwrap(), b"RIFF");
        assert_eq!(args.file_paths["audio"], vec![check.path.clone()]);
        assert_eq!(args.input_modalities, ["audio", "text"]);
        assert!(!args.model_kwargs.contains_key("ref_audio"));
    }
    for (body, want) in [
        (r#"{"input":"x","ref_audio":"/etc/passwd!"}"#, (400, crate::media::BARE_REF_AUDIO_REFUSAL.to_string())),
        (r#"{"input":"x","ref_audio":7}"#, (400, "ref_audio must be a data URL or base64 string, got int".into())),
        (r#"{"input":"x","ref_audio":"http://h/a.wav"}"#, (400, "Remote media fetch is disabled on this server".into())),
    ] {
        assert_eq!(clone(body).unwrap_err(), want, "{body}");
    }
    // no clip: a plain speech, nothing to check
    assert!(clone(r#"{"input":"x","ref_audio":""}"#).unwrap().ref_audio.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn qwen3_tts_and_orpheus_chunk_long_speech_by_default() {
    assert_eq!(Qwen3TTS.speech_chunking().min_chars, Some(600));
    assert_eq!(Qwen3Omni.speech_chunking().min_chars, None);
    for (name, adapter) in [("Qwen3TTS", &Qwen3TTS as &dyn OpenAIAdapter), ("Qwen3Omni", &Qwen3Omni)] {
        let c = adapter.speech_chunking();
        assert_eq!((c.max_chars, c.lookahead, c.max_pieces), (400, 2, 32), "{name}");
    }
    // 2,048 output tokens are about 25 s of speech: a longer input came back cut short with a 200
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(22);
    let pieces = crate::speech_chunking::plan_chunks(None, &Orpheus.speech_chunking(), &text).unwrap();
    assert!(pieces.len() > 1, "{pieces:?}");
}

#[test]
fn a_qwen3_5_chat_maps_the_standard_fields_and_passes_the_rest_through() {
    let req: ChatCompletionRequest = serde_json::from_str(
        r#"{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"海について"}],
            "temperature":0.7,"top_p":0.9,"max_tokens":32,"seed":5,
            "enable_thinking":false,"repetition_penalty":1.1,"max_output_tokens":8}"#,
    )
    .unwrap();
    let args = Qwen3_5.chat_to_request(&req, Path::new("/unused"), false).unwrap();
    assert_eq!(args.text.as_deref(), Some("Be brief.\n海について"));
    assert_eq!(args.input_modalities, ["text", "text"]);
    assert_eq!(args.output_modalities, ["text"]);
    let mk = Value::Object(args.model_kwargs);
    assert_eq!(mk, serde_json::json!({
        "temperature": 0.7, "top_p": 0.9, "seed": 5, "enable_thinking": false,
        "repetition_penalty": 1.1, "max_output_tokens": 8,
    }), "an explicit extra_body max_output_tokens wins over max_tokens");
}

#[test]
fn a_thinking_chat_reads_the_template_kwargs_flat() {
    for (name, adapter) in [("Qwen3_5", &Qwen3_5 as &dyn OpenAIAdapter), ("Cosmos3Edge", &Cosmos3Edge)] {
        let chat = |fields: &str| adapter.chat_to_request(&chat_with(fields), Path::new("/unused"), false);
        let thinking = |fields: &str| chat(fields).unwrap().model_kwargs.get("enable_thinking").cloned();
        for (fields, want) in [
            (r#""chat_template_kwargs":{"enable_thinking":false}"#, Some(Value::Bool(false))),
            // as sent: the model refuses what is not a boolean
            (r#""chat_template_kwargs":{"enable_thinking":"false"}"#, Some(serde_json::json!("false"))),
            (r#""chat_template_kwargs":{"enable_thinking":false},"enable_thinking":true"#, Some(Value::Bool(true))),
            (r#""chat_template_kwargs":{"other":1}"#, None),
            (r#""chat_template_kwargs":{}"#, None),
            (r#""chat_template_kwargs":null"#, None),
            (r#""chat_template_kwargs":0"#, None),
        ] {
            assert_eq!(thinking(fields), want, "{fields} on {name}");
            assert!(!chat(fields).unwrap().model_kwargs.contains_key("chat_template_kwargs"), "{fields} on {name}");
        }
        for fields in [
            r#""chat_template_kwargs":["enable_thinking"]"#,
            r#""chat_template_kwargs":"enable_thinking=1""#,
            r#""chat_template_kwargs":true"#,
            r#""chat_template_kwargs":1.5"#,
        ] {
            assert_eq!(chat(fields).unwrap_err(), "chat_template_kwargs must be an object", "{fields} on {name}");
        }
        let mk = chat(
            r#""chat_template_kwargs":{"reasoning_strength":"low","current_date":"2026-11-08","enable_thinking":false},
               "reasoning_strength":"high""#,
        )
        .unwrap()
        .model_kwargs;
        assert_eq!(
            Value::Object(mk),
            serde_json::json!({"reasoning_strength": "high", "current_date": "2026-11-08", "enable_thinking": false}),
            "a template variable was dropped, or won over the flat key on {name}"
        );
    }
}

fn chat_with(fields: &str) -> ChatCompletionRequest {
    let body = format!(r#"{{"messages":[{{"role":"user","content":"Hi."}}],{fields}}}"#);
    serde_json::from_str(&body).unwrap()
}

#[test]
fn a_chat_that_asks_for_a_tool_call_is_refused() {
    for fields in [
        r#""tools":[{"type":"function","function":{"name":"get_weather"}}]"#,
        r#""tool_choice":"required""#,
        r#""tool_choice":{"type":"function","function":{"name":"get_weather"}}"#,
        r#""functions":[{"name":"get_weather"}]"#,
        r#""function_call":{"name":"get_weather"}"#,
    ] {
        for (name, adapter) in ADAPTERS.iter().filter(|(_, a)| a.supports_chat()) {
            let err = adapter.chat_to_request(&chat_with(fields), Path::new("/unused"), false).unwrap_err();
            assert_eq!(err, "tool calling is not supported for this model", "{fields} reached {name}");
        }
    }
}

#[test]
fn a_chat_asking_for_more_than_one_plain_reply_is_refused() {
    for (fields, err) in [
        (r#""n":2"#, "n=2: this model returns one choice"),
        (r#""stop":".""#, "stop sequences are not supported"),
        (r#""stop":["END"]"#, "stop sequences are not supported"),
        (r#""logprobs":true"#, "logprobs is not supported"),
        (r#""presence_penalty":1.5"#, "presence_penalty is not supported"),
        (r#""best_of":3"#, "best_of is not supported"),
        (r#""response_format":{"type":"yaml"}"#, "response_format type must be"),
    ] {
        for (name, adapter) in ADAPTERS.iter().filter(|(_, a)| a.supports_chat()) {
            let got = adapter.chat_to_request(&chat_with(fields), Path::new("/unused"), false).unwrap_err();
            assert!(got.starts_with(err), "{fields} on {name}: {got}");
        }
    }
}

#[test]
fn a_json_response_format_becomes_structured_outputs() {
    for (fields, want) in [
        (r#""response_format":{"type":"json_object"}"#, serde_json::json!({"json_object": true})),
        (
            r#""response_format":{"type":"json_schema","json_schema":{"name":"n","schema":{"type":"object"}}}"#,
            serde_json::json!({"json": {"type": "object"}}),
        ),
    ] {
        for (name, adapter) in ADAPTERS.iter().filter(|(_, a)| a.supports_chat()) {
            let mk = adapter.chat_to_request(&chat_with(fields), Path::new("/unused"), false).unwrap().model_kwargs;
            assert_eq!(mk.get("structured_outputs"), Some(&want), "{fields} on {name}");
            assert!(!mk.contains_key("response_format"), "{fields} on {name}");
        }
    }
}

#[test]
fn the_tool_fields_clients_send_by_default_pass() {
    for fields in [
        r#""n":1"#,
        r#""n":null"#,
        r#""stop":null"#,
        r#""stop":[]"#,
        r#""tools":[]"#,
        r#""tools":null"#,
        r#""tool_choice":"none""#,
        r#""tool_choice":"auto""#,
        r#""tool_choice":null"#,
        r#""functions":[]"#,
        r#""functions":null"#,
        r#""function_call":"none""#,
        r#""function_call":"auto""#,
        r#""function_call":null"#,
        r#""tools":[{"type":"function","function":{"name":"get_weather"}}],"tool_choice":"none""#,
        r#""functions":[{"name":"get_weather"}],"function_call":"none""#,
        r#""logprobs":false"#,
        r#""presence_penalty":0"#,
        r#""best_of":1"#,
        r#""response_format":{"type":"text"}"#,
        r#""stream_options":{"include_usage":true},"user":"u1","store":false,"metadata":{"k":"v"},"service_tier":"auto","parallel_tool_calls":true"#,
    ] {
        for (name, adapter) in ADAPTERS.iter().filter(|(_, a)| a.supports_chat()) {
            let args = adapter.chat_to_request(&chat_with(fields), Path::new("/unused"), false);
            assert!(args.is_ok(), "{fields} refused a plain chat on {name}: {args:?}");
            let mk = args.unwrap().model_kwargs;
            let sent: serde_json::Value = serde_json::from_str(&format!("{{{fields}}}")).unwrap();
            for key in sent.as_object().unwrap().keys() {
                assert!(!mk.contains_key(key), "{key} reached the backend from {fields} on {name}");
            }
        }
    }
}

/// A fresh upload directory, removed on drop. Request media must be inline
/// (a data URL), so the server persists it here.
struct UploadDir(std::path::PathBuf);

impl UploadDir {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!("mstar-adapters-test-{}", uuid::Uuid::new_v4())))
    }
}

impl Drop for UploadDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// What each persisted upload holds, in `file_paths` order.
fn contents(paths: &[String]) -> Vec<Vec<u8>> {
    paths.iter().map(|p| std::fs::read(p).unwrap()).collect()
}

// The bytes "a" and "b", so a test can tell which image went where.
const IMAGE_A: &str = "data:image/png;base64,YQ==";
const IMAGE_B: &str = "data:image/png;base64,Yg==";

#[test]
fn an_attachment_keeps_its_place_between_text() {
    let dir = UploadDir::new();
    let (text, file_paths, in_mods, parts) = flatten(
        &format!(
            r#"[{{"role":"user","content":[
            {{"type":"text","text":"A"}},
            {{"type":"image_url","image_url":{{"url":"{IMAGE_A}"}}}},
            {{"type":"text","text":"B"}}]}}]"#
        ),
        &dir.0,
    )
    .unwrap();
    assert_eq!(text.as_deref(), Some("A\nB"));
    assert_eq!(contents(&file_paths["image"]), [b"a"]);
    assert_eq!(in_mods, ["text", "image", "text"], "the layout lost the image's place");
    assert_eq!(
        described(&parts),
        [
            ("text", "user", Some("A"), 0),
            ("image", "user", None, 0),
            ("text", "user", Some("B"), 0),
        ],
        "text written after the image moved ahead of it"
    );
}

#[test]
fn an_image_in_a_reply_lands_in_the_next_user_turn() {
    let dir = UploadDir::new();
    let (_, file_paths, in_mods, parts) = flatten(
        &format!(
            r#"[{{"role":"user","content":"Draw a red cube."}},
            {{"role":"assistant","content":[
                {{"type":"text","text":"Here it is."}},
                {{"type":"image_url","image_url":{{"url":"{IMAGE_A}"}}}}]}},
            {{"role":"user","content":"Make it blue."}}]"#
        ),
        &dir.0,
    )
    .unwrap();
    assert_eq!(contents(&file_paths["image"]), [b"a"]);
    assert_eq!(in_mods, ["text", "text", "image", "text"]);
    assert_eq!(
        described(&parts),
        [
            ("text", "user", Some("Draw a red cube."), 0),
            ("text", "assistant", Some("Here it is."), 0),
            ("image", "user", None, 0),
            ("text", "user", Some("Make it blue."), 0),
        ],
        "the reply's image did not land in the next user turn"
    );
}

#[test]
fn two_images_are_indexed_in_order() {
    let dir = UploadDir::new();
    let (_, file_paths, in_mods, parts) = flatten(
        &format!(
            r#"[{{"role":"user","content":[
            {{"type":"image_url","image_url":{{"url":"{IMAGE_A}"}}}},
            {{"type":"image_url","image_url":{{"url":"{IMAGE_B}"}}}},
            {{"type":"text","text":"Compare."}}]}}]"#
        ),
        &dir.0,
    )
    .unwrap();
    assert_eq!(contents(&file_paths["image"]), [b"a", b"b"]);
    assert_eq!(in_mods, ["image", "image", "text"]);
    assert_eq!(
        parts.iter().map(|p| p.index).collect::<Vec<_>>(),
        [0, 1, 0],
        "the second image does not address the second upload"
    );
}

#[test]
fn fish_forwards_a_sent_voice_to_be_logged_as_ignored() {
    let mk = |body: &str| FishS2Pro.speech_args(&speech(body)).unwrap().model_kwargs;
    assert_eq!(mk(r#"{"input":"x","voice":"ryan"}"#)["voice"], serde_json::json!("ryan"));
    assert!(!mk(r#"{"input":"x"}"#).contains_key("voice"));
}
