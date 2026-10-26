"""Resolve pinned metadata without ever downloading model weights."""

from pathlib import Path

MODEL_ID = "CohereLabs/command-a-plus-05-2026-bf16"
CHECKPOINT_REVISION = "5fb6fde5fd12ff89356aae552e11883bc49f069b"
METADATA_FILES = (
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "chat_template.jinja", "generation_config.json",
    "model.safetensors.index.json",
)


def resolve_metadata(source: str, cache_dir: str | None = None) -> Path:
    local = Path(source)
    if (local / "config.json").is_file():
        return local
    if source != MODEL_ID:
        raise ValueError("Command A+ requires its supported model ID or a local checkpoint directory with config.json")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=MODEL_ID, revision=CHECKPOINT_REVISION, cache_dir=cache_dir,
        allow_patterns=list(METADATA_FILES),
    ))
