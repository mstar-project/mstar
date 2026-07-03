"""Pydantic request models for the OpenAI-compatible endpoints.

Requests are validated here; unknown fields are allowed (``extra="allow"``) so
the OpenAI client's ``extra_body`` flows through as model_kwargs. Responses are
built as plain dicts in the serving handlers to keep the multimodal shapes
(audio in ``message.audio``, images as data URLs) flexible.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Allow a field literally named ``model`` without pydantic's protected-namespace
# warning, and accept unknown fields as passthrough model_kwargs.
_CFG = ConfigDict(extra="allow", protected_namespaces=())


class ChatMessage(BaseModel):
    model_config = _CFG
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = _CFG

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = 1
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = False

    # Multimodal output (vllm-omni / sglang-omni style)
    modalities: list[str] | None = None
    audio: dict[str, Any] | None = None  # {"voice": ..., "format": "wav"}


class SpeechRequest(BaseModel):
    """OpenAI ``/v1/audio/speech`` (text-to-speech)."""

    model_config = _CFG

    input: str
    model: str | None = None
    voice: str | None = None
    response_format: str = "wav"
    speed: float | None = 1.0
    stream: bool | None = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None


class ImageGenerationRequest(BaseModel):
    """OpenAI ``/v1/images/generations``."""

    model_config = _CFG

    prompt: str
    model: str | None = None
    n: int | None = 1
    size: str | None = None
    response_format: str = "b64_json"
    seed: int | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "mstar"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)
