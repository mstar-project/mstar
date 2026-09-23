"""Unit tests for the Kimi-K2.7 output parser."""

from mstar.model.kimi_k2_7.output_parser import (
    TOOL_ARG_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    TOOL_SECTION_END,
    TOOL_SECTION_START,
    KimiOutputParser,
    _merge_deltas,
    parse_full,
)

REFERENCE = (
    "<think>plan...</think>Answer"
    + TOOL_SECTION_START
    + TOOL_CALL_START + "functions.get_weather:0" + TOOL_ARG_START
    + '{"city": "Tokyo"}' + TOOL_CALL_END
    + TOOL_SECTION_END
)


def test_full_single_shot_parse():
    result = parse_full(REFERENCE)
    assert result["reasoning_content"] == "plan..."
    assert result["content"] == "Answer"
    assert result["tool_calls"] == [{
        "id": "functions.get_weather:0", "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
    }]

    parser = KimiOutputParser()
    deltas = parser.feed(REFERENCE) + parser.finish()
    assert _merge_deltas(deltas) == result
    assert parser.finish_reason == "tool_calls"


def test_char_by_char_matches_single_shot_and_header_precedes_arguments():
    parser = KimiOutputParser()
    deltas = []
    for ch in REFERENCE:
        deltas.extend(parser.feed(ch))
    deltas.extend(parser.finish())

    assert _merge_deltas(deltas) == parse_full(REFERENCE)

    header_idx = next(
        i for i, d in enumerate(deltas)
        if "tool_calls" in d and "id" in d["tool_calls"][0]
    )
    arg_idx = next(
        i for i, d in enumerate(deltas)
        if "tool_calls" in d and "arguments" in d["tool_calls"][0].get("function", {})
        and "id" not in d["tool_calls"][0]
    )
    assert header_idx < arg_idx


def test_two_tool_calls_get_sequential_indices():
    text = (
        TOOL_SECTION_START
        + TOOL_CALL_START + "functions.get_weather:0" + TOOL_ARG_START
        + '{"city": "Tokyo"}' + TOOL_CALL_END
        + TOOL_CALL_START + "functions.get_time:1" + TOOL_ARG_START
        + "{}" + TOOL_CALL_END
        + TOOL_SECTION_END
    )
    result = parse_full(text)
    assert [tc["function"]["name"] for tc in result["tool_calls"]] == [
        "get_weather", "get_time",
    ]
    assert result["tool_calls"][0]["function"]["arguments"] == '{"city": "Tokyo"}'
    assert result["tool_calls"][1]["function"]["arguments"] == "{}"


def test_thinking_false_starts_in_content():
    result = parse_full("Hello there", thinking=False)
    assert result["reasoning_content"] is None
    assert result["content"] == "Hello there"


def test_reasoning_only_no_close_before_finish():
    parser = KimiOutputParser()
    deltas = parser.feed("<think>still thinking...") + parser.finish()
    result = _merge_deltas(deltas)
    assert result["reasoning_content"] == "still thinking..."
    assert result["content"] is None
    assert parser.finish_reason == "stop"


def test_whitespace_only_content_before_tool_section_is_dropped():
    text = (
        "<think>plan</think>   "
        + TOOL_SECTION_START
        + TOOL_CALL_START + "functions.f:0" + TOOL_ARG_START
        + "{}" + TOOL_CALL_END
        + TOOL_SECTION_END
    )
    result = parse_full(text)
    assert result["reasoning_content"] == "plan"
    assert result["content"] is None
    assert result["tool_calls"][0]["function"]["name"] == "f"


def test_bad_header_yields_no_tool_call():
    text = (
        TOOL_SECTION_START
        + TOOL_CALL_START + "not_a_valid_header" + TOOL_ARG_START
        + "{}" + TOOL_CALL_END
        + TOOL_SECTION_END
    )
    parser = KimiOutputParser()
    deltas = parser.feed(text) + parser.finish()
    result = _merge_deltas(deltas)
    assert result["tool_calls"] is None
    assert parser.finish_reason == "stop"


def test_text_after_section_end_is_suppressed():
    text = (
        TOOL_SECTION_START
        + TOOL_CALL_START + "functions.f:0" + TOOL_ARG_START
        + "{}" + TOOL_CALL_END
        + TOOL_SECTION_END
        + "trailing garbage here"
    )
    result = parse_full(text)
    assert result["content"] is None
    assert result["tool_calls"][0]["function"]["name"] == "f"


def test_marker_split_across_feed_calls():
    part1 = "Hello" + TOOL_SECTION_START[:10]
    part2 = (
        TOOL_SECTION_START[10:]
        + TOOL_CALL_START + "functions.f:0" + TOOL_ARG_START
        + "{}" + TOOL_CALL_END
        + TOOL_SECTION_END
    )
    parser = KimiOutputParser(thinking=False)
    deltas = parser.feed(part1) + parser.feed(part2) + parser.finish()
    result = _merge_deltas(deltas)

    assert result["content"] == "Hello"
    assert result["tool_calls"][0]["function"]["name"] == "f"
    assert parser.finish_reason == "tool_calls"
