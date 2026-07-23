"""Transport-mocked tests for the Ollama `LLMClient` (Task C4).

No test here talks to a real server: the client takes an injectable
`transport` callable, so every test asserts on the exact request payload the
client would POST to ``/api/chat`` and feeds back a canned response dict.
"""

from __future__ import annotations

import pytest

from agentic_analyst.agent import TOOLS, LLMClient, TextBlock, ToolUseBlock
from agentic_analyst.ollama_client import (
    OllamaClient,
    OllamaTransportError,
    _translate_messages,
    _translate_tools,
)


def _canned(response: dict):
    """A transport that records payloads and always returns `response`."""
    sent: list[dict] = []

    def transport(payload: dict) -> dict:
        sent.append(payload)
        return response

    return transport, sent


def _tool_call_response(name: str, arguments: object) -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
        },
        "done_reason": "stop",
    }


TEXT_ONLY_RESPONSE = {
    "message": {"role": "assistant", "content": "All done."},
    "done_reason": "stop",
}


def test_client_satisfies_the_llm_client_protocol() -> None:
    transport, _ = _canned(TEXT_ONLY_RESPONSE)
    assert isinstance(OllamaClient(transport=transport), LLMClient)


def test_tools_translate_to_ollama_function_format_with_schema_passthrough() -> None:
    translated = _translate_tools(TOOLS)
    assert [t["function"]["name"] for t in translated] == [t["name"] for t in TOOLS]
    for original, converted in zip(TOOLS, translated, strict=True):
        assert converted["type"] == "function"
        assert converted["function"]["parameters"] == original["input_schema"]
        assert converted["function"]["description"] == original["description"]


def test_full_round_trip_message_translation() -> None:
    """One complete loop round: user prompt, assistant tool call, tool result."""
    messages = [
        {"role": "user", "content": "Profile the dataset."},
        {
            "role": "assistant",
            "content": [
                TextBlock(text="Let me look at the schema."),
                ToolUseBlock(id="toolu_1", name="read_schema", input={"n_preview": 3}),
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "row_count: 7043"}
            ],
        },
    ]
    translated = _translate_messages("SYSTEM", messages)

    assert translated[0] == {"role": "system", "content": "SYSTEM"}
    assert translated[1] == {"role": "user", "content": "Profile the dataset."}
    assert translated[2]["role"] == "assistant"
    assert translated[2]["content"] == "Let me look at the schema."
    assert translated[2]["tool_calls"] == [
        {
            "type": "function",
            "function": {"name": "read_schema", "arguments": {"n_preview": 3}},
        }
    ]
    # the tool_result is paired back to the call's NAME via its tool_use_id
    assert translated[3] == {
        "role": "tool",
        "content": "row_count: 7043",
        "tool_name": "read_schema",
    }


def test_error_tool_result_is_visibly_labeled() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [ToolUseBlock(id="toolu_9", name="query_sql", input={"query": "SELEC"})],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_9",
                    "content": "syntax error at SELEC",
                    "is_error": True,
                }
            ],
        },
    ]
    translated = _translate_messages("s", messages)
    assert translated[-1]["content"] == "ERROR: syntax error at SELEC"
    assert translated[-1]["tool_name"] == "query_sql"


def test_tool_call_response_maps_to_tool_use_stop_reason() -> None:
    transport, sent = _canned(_tool_call_response("query_sql", {"query": "SELECT 1"}))
    client = OllamaClient(model="qwen2.5:7b", transport=transport)
    response = client.create_message(system="s", messages=[], tools=TOOLS)

    assert response.stop_reason == "tool_use"
    (block,) = response.content
    assert isinstance(block, ToolUseBlock)
    assert block.name == "query_sql"
    assert block.input == {"query": "SELECT 1"}
    assert block.id  # synthetic but present — run_agent keys results on it

    payload = sent[0]
    assert payload["model"] == "qwen2.5:7b"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0
    assert payload["options"]["num_ctx"] == client.num_ctx


def test_string_arguments_are_parsed_and_junk_degrades_to_empty_dict() -> None:
    transport, _ = _canned(_tool_call_response("read_schema", '{"n_preview": 2}'))
    client = OllamaClient(transport=transport)
    (block,) = client.create_message(system="s", messages=[], tools=TOOLS).content
    assert isinstance(block, ToolUseBlock)
    assert block.input == {"n_preview": 2}

    transport_junk, _ = _canned(_tool_call_response("read_schema", "{not json"))
    client_junk = OllamaClient(transport=transport_junk)
    (junk_block,) = client_junk.create_message(system="s", messages=[], tools=TOOLS).content
    assert isinstance(junk_block, ToolUseBlock)
    assert junk_block.input == {}


def test_text_only_response_is_end_turn() -> None:
    transport, _ = _canned(TEXT_ONLY_RESPONSE)
    response = OllamaClient(transport=transport).create_message(
        system="s", messages=[], tools=TOOLS
    )
    assert response.stop_reason == "end_turn"
    (block,) = response.content
    assert isinstance(block, TextBlock)
    assert block.text == "All done."


def test_synthetic_tool_use_ids_are_unique_across_calls() -> None:
    transport, _ = _canned(_tool_call_response("read_schema", {}))
    client = OllamaClient(transport=transport)
    first = client.create_message(system="s", messages=[], tools=TOOLS)
    second = client.create_message(system="s", messages=[], tools=TOOLS)
    ids = [b.id for b in first.content + second.content if isinstance(b, ToolUseBlock)]
    assert len(ids) == 2
    assert len(ids) == len(set(ids))


def test_transcript_records_every_request_response_pair() -> None:
    transport, _ = _canned(TEXT_ONLY_RESPONSE)
    client = OllamaClient(transport=transport)
    client.create_message(system="s", messages=[], tools=TOOLS)
    client.create_message(system="s", messages=[], tools=TOOLS)
    assert len(client.transcript) == 2
    assert client.transcript[0]["response"] == TEXT_ONLY_RESPONSE
    assert client.transcript[0]["request"]["model"] == client.model


def test_transport_error_propagates_as_ollama_transport_error() -> None:
    def broken(_payload: dict) -> dict:
        raise OllamaTransportError("cannot reach Ollama at http://localhost:11434: refused")

    client = OllamaClient(transport=broken)
    with pytest.raises(OllamaTransportError):
        client.create_message(system="s", messages=[], tools=TOOLS)
