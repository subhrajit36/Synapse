"""Phase F6: a chat assistant that answers by calling the MCP tools.

The assistant is a thin loop around Gemini function calling, and the functions
it may call are *the MCP server's own tools*, reached through an in-process
FastMCP client. Their names, descriptions and JSON schemas are read from the
server at call time, so the chatbot cannot drift from what an external MCP
client sees - there is no second, hand-written copy of any tool.

Three rules, all from the plan:

  * No memory. Each request is one user message; nothing is stored between
    requests. A follow-up must restate what it refers to.
  * No streaming. The response arrives whole.
  * Tool calls are returned, not hidden. Every call the model made comes back
    with its arguments and its result or error, so the page can show them. The
    deterministic ranking panel stays the source of truth; the chat is a way to
    ask questions of it, and it must be checkable against it.

Session scoping is enforced here, not requested in the prompt: for a tool that
takes `batch_id`, the parameter is removed from the schema the model sees and
the session's id is injected on every call. A model cannot be talked into
reading another session's candidates, because it has no way to name one.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Bounds the loop, not the answer: a question that needs more than this many
# rounds of tool calls is reported as truncated rather than left to spin
# against the free-tier rate limit.
MAX_TOOL_ROUNDS = 4

# Parameters the server fills in itself. Hidden from the model.
SCOPED_PARAMS = ("batch_id",)

SYSTEM_INSTRUCTION = """\
You are the assistant inside Synapse, a candidate-ranking tool that scores
résumés against a job description using a skill knowledge graph.

Answer ONLY from the results of the tools you call. Never invent a score, a
skill, a candidate or a path. If a tool fails or returns nothing relevant, say
so plainly.

How Synapse scores, so you can explain results correctly:
- A JD skill the candidate holds is a direct match.
- A JD skill the candidate lacks but can reach through a short weighted path
  in the skill graph is "bridged": it earns partial credit, and the result
  names the held skill it is reached `via`, the `hops` and the `distance`.
- A skill with no such path is an unreachable gap, with a `reason`.
- `total` = (direct + bridge - penalty) / total demand.

Tools act only on the current upload session's candidates; you cannot see any
other session. Skill names are canonicalised by the tools, so pass them as the
user wrote them. Keep answers short and concrete, and quote the numbers the
tools returned.
"""


# ------------------------------------------------------------------ contracts


class ToolCall(BaseModel):
    """One function call the model made, exactly as executed."""

    name: str
    arguments: dict = Field(..., description="As sent to the tool, after session scoping.")
    ok: bool
    result: Any = Field(None, description="The tool's structured output when ok.")
    error: str | None = None


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    batch_id: str | None = None
    model: str = ""
    rounds: int = Field(0, description="Model calls made to produce this reply.")
    truncated: bool = Field(
        False, description="The tool-round limit was hit before a final answer."
    )


# ---------------------------------------------------------------- the client


_CLIENT = None


def get_chat_client():
    """Gemini client, built on first use. Missing key -> RuntimeError (a 503)."""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Chat unavailable: no GEMINI_API_KEY is set."
            )
        from google import genai

        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def set_chat_client(client) -> None:
    """Swap the client. Tests use this to script the model's turns."""
    global _CLIENT
    _CLIENT = client


def chat_model() -> str:
    from ..ingest.extractor import DEFAULT_MODEL

    return os.getenv("SYNAPSE_CHAT_MODEL", DEFAULT_MODEL)


# --------------------------------------------------------------- tool bridge


def _declaration(tool) -> dict:
    """An MCP tool as a Gemini function declaration, minus scoped params."""
    schema = copy.deepcopy(tool.inputSchema or {"type": "object", "properties": {}})
    props = schema.get("properties", {})
    for name in SCOPED_PARAMS:
        props.pop(name, None)
    if "required" in schema:
        schema["required"] = [r for r in schema["required"] if r not in SCOPED_PARAMS]
    return {
        "name": tool.name,
        "description": tool.description or "",
        "parameters_json_schema": schema,
    }


def _scope(name: str, args: dict, batch_id: str | None, scoped_tools: set[str]) -> dict:
    """Inject the session's batch id; never trust one from the model."""
    args = dict(args or {})
    for param in SCOPED_PARAMS:
        args.pop(param, None)
    if name in scoped_tools and batch_id:
        # rank_candidates over explicit candidates is a stateless call, and
        # combining it with a batch is an error by design (F2).
        if not (name == "rank_candidates" and args.get("candidates") is not None):
            args["batch_id"] = batch_id
    return args


async def _call(client, name: str, args: dict) -> ToolCall:
    try:
        result = await client.call_tool(name, args, raise_on_error=False)
    except Exception as exc:  # noqa: BLE001 - unknown tool, bad args, transport
        return ToolCall(name=name, arguments=args, ok=False,
                        error=f"{type(exc).__name__}: {exc}")
    if result.is_error:
        text = " ".join(getattr(c, "text", "") for c in result.content).strip()
        return ToolCall(name=name, arguments=args, ok=False, error=text or "tool error")
    return ToolCall(name=name, arguments=args, ok=True, result=result.structured_content)


# ---------------------------------------------------------------------- loop


async def run_chat(
    message: str,
    batch_id: str | None,
    mcp_server,
    client=None,
    model: str | None = None,
    max_rounds: int = MAX_TOOL_ROUNDS,
) -> ChatResponse:
    """Answer one message, calling the server's MCP tools as needed.

    Raises ValueError for an empty message and RuntimeError when the model
    cannot be reached; the route maps those to 400 and 503.
    """
    from fastmcp import Client
    from google.genai import types

    message = (message or "").strip()
    if not message:
        raise ValueError("Message is empty.")
    batch_id = batch_id or None
    client = client or get_chat_client()
    model = model or chat_model()

    calls: list[ToolCall] = []
    async with Client(mcp_server) as mcp_client:
        tools = await mcp_client.list_tools()
        scoped_tools = {
            t.name for t in tools
            if any(p in (t.inputSchema or {}).get("properties", {}) for p in SCOPED_PARAMS)
        }
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION
            + (f"\nCurrent session: {batch_id}." if batch_id else
               "\nNo upload session is active; pool tools cover every candidate."),
            tools=[types.Tool(function_declarations=[_declaration(t) for t in tools])],
            # We run the loop ourselves so every call is recorded and bounded.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.2,
        )
        contents: list = [types.Content(role="user", parts=[types.Part(text=message)])]

        for round_no in range(1, max_rounds + 1):
            try:
                # The SDK call is blocking; keep it off the event loop.
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model, contents=contents, config=config,
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Chat model failed ({type(exc).__name__}): {exc}"
                ) from exc

            function_calls = response.function_calls or []
            if not function_calls:
                return ChatResponse(
                    reply=(response.text or "").strip() or "(no answer)",
                    tool_calls=calls, batch_id=batch_id, model=model,
                    rounds=round_no,
                )

            contents.append(response.candidates[0].content)
            parts = []
            for fc in function_calls:
                args = _scope(fc.name, dict(fc.args or {}), batch_id, scoped_tools)
                call = await _call(mcp_client, fc.name, args)
                calls.append(call)
                logger.info("chat tool %s ok=%s", fc.name, call.ok)
                payload = {"result": call.result} if call.ok else {"error": call.error}
                parts.append(types.Part.from_function_response(name=fc.name, response=payload))
            contents.append(types.Content(role="user", parts=parts))

    return ChatResponse(
        reply=(f"Stopped after {max_rounds} rounds of tool calls without a final "
               "answer. The calls made so far are listed below."),
        tool_calls=calls, batch_id=batch_id, model=model,
        rounds=max_rounds, truncated=True,
    )
