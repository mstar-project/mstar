"""GLM-5.2 / GLM-5.3 chat: the ``glm47`` tool-call format and the template's
switches (the parser and the request side are ``tool_calls``').

The template ends the prompt inside ``<think>`` when the model reasons, so the
reply opens with reasoning and ``</think>`` hands over to the answer. A call is
``<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>``:
a string value arrives bare and any other as JSON, as the template renders them.
"""

from __future__ import annotations

from mstar.api_server.openai.tool_calls import (  # noqa: F401 (the names GLM's callers read)
    GLM47,
    THINK_END,
    THINK_START,
    ReplyParser,
    template_messages,
    tool_parameters,
)
from mstar.api_server.openai.tool_calls import arg_value as _arg_value  # noqa: F401

CALL_START, CALL_END = GLM47.call_start, GLM47.call_end
ARG_KEY, ARG_KEY_END = GLM47.key_start, GLM47.key_end
ARG_VALUE, ARG_VALUE_END = GLM47.value_start, GLM47.value_end

# what the templates read; others are dropped, as some would collide with
# apply_chat_template's own arguments
TEMPLATE_KWARGS = ("enable_thinking", "reasoning_effort", "clear_thinking")


class GlmReplyParser(ReplyParser):
    """``ReplyParser`` over GLM's markup (see there)."""

    def __init__(self, *, thinking: bool, tools: dict[str, dict] | None, forced_call: str | None = None,
                 single_call: bool = False):
        super().__init__(GLM47, thinking=thinking, tools=tools, forced_call=forced_call, single_call=single_call)
