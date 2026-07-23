"""Ollama tool-calling client implementing the agent loop's `LLMClient` protocol.

Task C4: put a *real* model at the wheel of the agent loop without an API key.
This client speaks Ollama's native ``/api/chat`` tools API (loopback only, no
cloud) and translates between the Anthropic-shaped message/tool format the
loop uses (`agent.py`) and Ollama's chat format:

- Anthropic ``tools`` (``{name, description, input_schema}``) become Ollama
  function tools (``{"type": "function", "function": {..., "parameters":
  input_schema}}``) -- both sides are plain JSON Schema, so the schema passes
  through unchanged.
- Anthropic ``tool_use`` blocks in assistant turns become ``tool_calls``;
  Anthropic ``tool_result`` blocks in user turns become ``role: "tool"``
  messages, paired back to their call's tool *name* via ``tool_use_id`` (the
  translation keeps an id->name map because Ollama identifies results by name,
  not id). An ``is_error`` result is prefixed ``ERROR:`` -- Ollama has no
  error flag, and the model handles a visibly-labeled failure better than a
  silent one.
- An Ollama response with ``tool_calls`` maps to ``stop_reason="tool_use"``
  plus one `ToolUseBlock` per call (synthetic ids -- Ollama does not assign
  any); otherwise ``stop_reason="end_turn"``. Tool-call ``arguments`` are
  accepted both as a dict (the documented shape) and as a JSON string (seen
  from some models/versions), defensively.

The HTTP transport is injectable (`transport=` in ``__init__``) so every test
runs without a server; the default transport is stdlib ``urllib`` -- this
module deliberately adds no dependency. Every request/response pair is
recorded in ``self.transcript`` so a failed run can be committed as an honest
failure artifact (see ``scripts/run_real_llm.py``).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable

from agentic_analyst.agent import LLMResponse, TextBlock, ToolUseBlock

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_HOST = "http://localhost:11434"
# qwen2.5 supports 32k; Ollama's default num_ctx (4096) silently truncates the
# conversation once tool results (schema previews, query tables) accumulate,
# which manifests as the model "forgetting" its tools mid-loop -- so the
# context window is pinned explicitly.
DEFAULT_NUM_CTX = 16384

Transport = Callable[[dict], dict]


class OllamaTransportError(RuntimeError):
    """The Ollama server could not be reached or returned a non-JSON/HTTP error.

    Deliberately distinct from the model failing the task (the loop's
    `AgentIncompleteError`): a transport error is an environment problem, not
    a finding about the model's tool-calling ability.
    """


def _urllib_transport(url: str, timeout_s: float) -> Transport:
    def send(payload: dict) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - exercised via mock
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaTransportError(f"Ollama HTTP {exc.code} at {url}: {body[:500]}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaTransportError(f"cannot reach Ollama at {url}: {exc}") from exc

    return send


def _block_attr(block: object, name: str) -> object:
    """Read a content-block field from either the loop's dataclass blocks
    (assistant turns) or plain dicts (the tool_result turns run_agent builds)."""
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def _translate_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for tool in tools
    ]


def _translate_messages(system: str, messages: list[dict]) -> list[dict]:
    """Anthropic-shaped loop messages -> Ollama chat messages.

    Keeps an id->name map from tool_use blocks so each tool_result can be
    emitted as a ``role: "tool"`` message carrying the tool's *name* (Ollama's
    pairing mechanism -- it has no tool_use_id concept).
    """
    ollama_messages: list[dict] = [{"role": "system", "content": system}]
    tool_names_by_id: dict[str, str] = {}

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if isinstance(content, str):
            ollama_messages.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content or []:
                block_type = _block_attr(block, "type")
                if block_type == "text":
                    text_parts.append(str(_block_attr(block, "text") or ""))
                elif block_type == "tool_use":
                    block_id = _block_attr(block, "id")
                    name = str(_block_attr(block, "name") or "")
                    if block_id:
                        tool_names_by_id[str(block_id)] = name
                    tool_calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": _block_attr(block, "input") or {},
                            },
                        }
                    )
            entry: dict = {"role": "assistant", "content": "\n".join(text_parts)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            ollama_messages.append(entry)
            continue

        # user turn with block content: tool_results (and possibly text)
        for block in content or []:
            block_type = _block_attr(block, "type")
            if block_type == "tool_result":
                result_text = str(_block_attr(block, "content") or "")
                if _block_attr(block, "is_error"):
                    result_text = f"ERROR: {result_text}"
                tool_use_id = str(_block_attr(block, "tool_use_id") or "")
                ollama_messages.append(
                    {
                        "role": "tool",
                        "content": result_text,
                        "tool_name": tool_names_by_id.get(tool_use_id, ""),
                    }
                )
            elif block_type == "text":
                ollama_messages.append(
                    {"role": "user", "content": str(_block_attr(block, "text") or "")}
                )

    return ollama_messages


def _parse_arguments(arguments: object) -> dict:
    """Tool-call arguments arrive as a dict per the docs, but some
    models/versions emit a JSON string -- accept both, degrade to {} on junk
    (the loop then feeds the tool's own validation error back to the model)."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class OllamaClient:
    """`LLMClient` implementation over Ollama's native ``/api/chat`` tools API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        num_ctx: int = DEFAULT_NUM_CTX,
        # 30 min per call: a 7B model generating a long code block on CPU can
        # legitimately take >10 min (measured: the baseline-training turn blew
        # a 600s timeout at ~5 tok/s while shorter turns took 27-47s). The
        # per-call timeout only needs to catch a truly dead server, not pace
        # the model.
        timeout_s: float = 1800.0,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self._call_counter = 0
        self.transcript: list[dict] = []
        self._transport = transport or _urllib_transport(f"{host}/api/chat", timeout_s)

    def create_message(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": _translate_messages(system, messages),
            "tools": _translate_tools(tools),
            "stream": False,
            # temperature 0: as deterministic as the runtime allows; the run
            # is still labeled with model+date, never claimed reproducible.
            "options": {"num_ctx": self.num_ctx, "num_predict": max_tokens, "temperature": 0},
        }
        raw = self._transport(payload)
        self.transcript.append({"request": payload, "response": raw})
        return self._parse_response(raw)

    def _parse_response(self, raw: dict) -> LLMResponse:
        message = raw.get("message") or {}
        content: list[TextBlock | ToolUseBlock] = []

        text = message.get("content") or ""
        if text.strip():
            content.append(TextBlock(text=text))

        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            self._call_counter += 1
            content.append(
                ToolUseBlock(
                    id=f"toolu_ollama_{self._call_counter}",
                    name=str(function.get("name") or ""),
                    input=_parse_arguments(function.get("arguments")),
                )
            )

        has_tool_calls = any(block.type == "tool_use" for block in content)
        return LLMResponse(
            content=content,
            stop_reason="tool_use" if has_tool_calls else "end_turn",
        )
