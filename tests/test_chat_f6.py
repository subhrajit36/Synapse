"""Phase F6: the chat assistant over the MCP tools.

The model is scripted: each test supplies the sequence of turns Gemini would
produce, built from the SDK's real response types. Everything after that is
real - the tool schemas are read from the live MCP server, the calls go through
an in-process FastMCP client, and the tools run against a stubbed pool.
"""

from __future__ import annotations

import asyncio
import pickle

import networkx as nx
import pytest

# Imported at collection time on purpose: other test modules install a stub
# `google.genai` when the real one is absent, and this file needs the real
# response types.
from google.genai import types

from synapse.graph.neo4j_client import Neo4jConfig
from synapse.matching.entity_linker import EntityLinker
from synapse.mcp import chat as chat_mod
from synapse.mcp.chat import ChatResponse, run_chat, set_chat_client
from synapse.mcp.engine import MatchEngine, set_engine
from synapse.mcp.server import mcp

NODES = ["Docker", "Kubernetes", "AWS", "Terraform"]

POOL = [
    {"candidate_id": "c1", "name": "devops", "batch_ids": ["b1"],
     "skills": {"Docker": 1.0, "AWS": 1.0}, "unresolved": []},
    {"candidate_id": "c2", "name": "frontend", "batch_ids": ["b2"],
     "skills": {"Kubernetes": 1.0}, "unresolved": []},
]


class StubClient:
    def __init__(self):
        self.config = Neo4jConfig(uri="neo4j+s://stub", password="x")

    @staticmethod
    def _in(row, batch_id):
        return batch_id is None or batch_id in row["batch_ids"]

    def load_candidate_pool(self, batch_id=None):
        return [dict(c) for c in POOL if self._in(c, batch_id)]

    def list_candidates(self, batch_id=None):
        return [{"candidate_id": c["candidate_id"], "name": c["name"],
                 "skill_count": len(c["skills"]), "batch_ids": c["batch_ids"]}
                for c in POOL if self._in(c, batch_id)]


# --------------------------------------------------------- scripted model


def call(name, **args):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(name=name, args=args))
        ]))])


def say(text):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=text)]))])


class ScriptedGemini:
    def __init__(self, *turns, error: Exception | None = None):
        self.turns = list(turns)
        self.error = error
        self.requests: list[dict] = []
        self.models = self

    def generate_content(self, **kwargs):
        # Snapshot: the caller keeps appending to the same `contents` list.
        self.requests.append({**kwargs, "contents": list(kwargs["contents"])})
        if self.error:
            raise self.error
        return self.turns.pop(0) if self.turns else say("done")


# -------------------------------------------------------------- fixtures


@pytest.fixture
def engine(tmp_path):
    G = nx.Graph()
    for n in NODES:
        G.add_node(n, node_type="skill", category="t")
    G.add_edge("Docker", "Kubernetes", relation="similar", weight=0.8)
    path = tmp_path / "g.pkl"
    path.write_bytes(pickle.dumps(G))
    eng = MatchEngine(graph_path=path, neo4j_client=StubClient())
    eng._linker = EntityLinker(NODES, use_embeddings=False)
    set_engine(eng)
    yield eng
    set_engine(None)
    set_chat_client(None)


def chat(message, batch_id, gemini, **kw) -> ChatResponse:
    return asyncio.run(run_chat(message, batch_id, mcp, client=gemini, model="m", **kw))


def function_responses(request) -> list:
    return [p.function_response for c in request["contents"] for p in (c.parts or [])
            if p.function_response is not None]


# -------------------------------------------------------------- the loop


def test_plain_answer_needs_no_tools(engine):
    g = ScriptedGemini(say("Synapse ranks by graph distance."))
    r = chat("what is this?", "b1", g)
    assert r.reply == "Synapse ranks by graph distance."
    assert r.tool_calls == [] and r.rounds == 1 and not r.truncated


def test_tool_is_called_through_mcp_and_its_result_goes_back_to_the_model(engine):
    g = ScriptedGemini(call("list_candidates"), say("One candidate: frontend."))
    r = chat("who is in this session?", "b2", g)

    (tc,) = r.tool_calls
    assert tc.name == "list_candidates" and tc.ok
    assert tc.arguments == {"batch_id": "b2"}, "session id injected server-side"
    assert tc.result["count"] == 1
    assert tc.result["candidates"][0]["name"] == "frontend"

    # Round 2 carried the tool's output back to the model.
    (fr,) = function_responses(g.requests[1])
    assert fr.name == "list_candidates"
    assert fr.response["result"]["count"] == 1
    assert r.reply == "One candidate: frontend." and r.rounds == 2


def test_model_cannot_choose_another_sessions_batch(engine):
    """Scoping is enforced, not requested: a model-supplied id is discarded."""
    g = ScriptedGemini(call("list_candidates", batch_id="b1"), say("ok"))
    (tc,) = chat("list everyone", "b2", g).tool_calls
    assert tc.arguments == {"batch_id": "b2"}
    assert [c["name"] for c in tc.result["candidates"]] == ["frontend"]


def test_batch_id_is_hidden_from_the_declared_schemas(engine):
    g = ScriptedGemini(say("hi"))
    chat("hi", "b1", g)
    decls = {d.name: d for d in g.requests[0]["config"].tools[0].function_declarations}
    assert set(decls) == {"rank_candidates", "get_bridgeable_gaps", "explain_score",
                          "list_candidates", "graph_stats"}, "every MCP tool is bound"
    for d in decls.values():
        assert "batch_id" not in d.parameters_json_schema.get("properties", {})
    assert "jd_skills" in decls["rank_candidates"].parameters_json_schema["properties"]


def test_ranking_runs_on_the_session_pool(engine):
    g = ScriptedGemini(
        call("rank_candidates", jd_skills=[{"skill": "Kubernetes"}]),
        say("frontend holds Kubernetes outright."),
    )
    (tc,) = chat("rank for kubernetes", "b2", g).tool_calls
    assert tc.ok and tc.arguments["batch_id"] == "b2"
    assert tc.result["batch_id"] == "b2"
    assert [c["name"] for c in tc.result["candidates"]] == ["frontend"]


def test_explicit_candidates_are_not_combined_with_the_batch(engine):
    """F2 forbids batch_id + candidates; the scoper must not create that."""
    g = ScriptedGemini(
        call("rank_candidates", jd_skills=[{"skill": "Kubernetes"}],
             candidates=[{"name": "x", "skills": [{"skill": "Docker"}]}]),
        say("x bridges to Kubernetes via Docker."),
    )
    (tc,) = chat("score x", "b2", g).tool_calls
    assert "batch_id" not in tc.arguments
    assert tc.ok, tc.error
    assert tc.result["candidate_source"] == "request"


def test_failed_tool_is_reported_and_the_model_is_told(engine):
    g = ScriptedGemini(call("no_such_tool"), say("That tool does not exist."))
    r = chat("do something odd", "b1", g)
    (tc,) = r.tool_calls
    assert tc.ok is False and tc.error
    (fr,) = function_responses(g.requests[1])
    assert "error" in fr.response
    assert r.reply == "That tool does not exist."


def test_loop_is_bounded(engine):
    g = ScriptedGemini(*[call("graph_stats") for _ in range(10)])
    r = chat("loop forever", "b1", g, max_rounds=3)
    assert r.truncated is True
    assert len(r.tool_calls) == 3 and len(g.requests) == 3


def test_no_memory_each_request_starts_from_one_message(engine):
    g1, g2 = ScriptedGemini(say("a")), ScriptedGemini(say("b"))
    chat("first question", "b1", g1)
    chat("second question", "b1", g2)
    (only,) = g2.requests[0]["contents"]
    assert only.parts[0].text == "second question"


def test_session_id_is_in_the_system_instruction(engine):
    g = ScriptedGemini(say("hi"))
    chat("hi", "b-7f3a", g)
    assert "b-7f3a" in g.requests[0]["config"].system_instruction


def test_empty_message_is_a_bad_request(engine):
    with pytest.raises(ValueError, match="empty"):
        chat("   ", "b1", ScriptedGemini())


def test_model_failure_is_a_runtime_error(engine):
    with pytest.raises(RuntimeError, match="Chat model failed"):
        chat("hi", "b1", ScriptedGemini(error=TimeoutError("timeout")))


def test_missing_api_key_is_a_runtime_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    set_chat_client(None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        chat_mod.get_chat_client()


# ------------------------------------------------------------------ route


@pytest.fixture
def web(engine):
    from starlette.testclient import TestClient

    with TestClient(mcp.http_app()) as client:
        yield client


def test_route_returns_reply_and_visible_tool_calls(web):
    set_chat_client(ScriptedGemini(call("list_candidates"), say("frontend.")))
    res = web.post("/api/chat", json={"message": "who?", "batch_id": "b2"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["reply"] == "frontend."
    assert body["batch_id"] == "b2"
    (tc,) = body["tool_calls"]
    assert tc["name"] == "list_candidates" and tc["arguments"] == {"batch_id": "b2"}


def test_route_rejects_bad_requests(web):
    assert web.post("/api/chat", json={}).status_code == 400
    assert web.post("/api/chat", json={"message": "  "}).status_code == 400
    assert web.post("/api/chat", content=b"nope").status_code == 400


def test_route_is_503_when_the_model_fails(web):
    set_chat_client(ScriptedGemini(error=ConnectionError("503 unavailable")))
    res = web.post("/api/chat", json={"message": "hi", "batch_id": "b1"})
    assert res.status_code == 503
    assert "Chat model failed" in res.json()["error"]


def test_page_has_the_ask_tab(web):
    body = web.get("/").text
    for needle in ("/api/chat", 'id="tab-ask"', "tool_calls", "renderCall"):
        assert needle in body


def test_chat_route_does_not_add_an_mcp_tool():
    """The chat is a consumer of the tools, not one of them."""
    tools = asyncio.run(mcp._list_tools())
    assert len(tools) == 5
