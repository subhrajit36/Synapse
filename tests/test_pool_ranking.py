"""Phase E4: ranking the stored pool.

The pool is stubbed at the client boundary, so these exercise the engine, the
tools and the routes without a database.
"""

from __future__ import annotations

import asyncio
import pickle

import networkx as nx
import pytest

from synapse.graph.neo4j_client import Neo4jConfig
from synapse.matching.entity_linker import EntityLinker
from synapse.mcp.engine import CandidateInput, MatchEngine, SkillWeight, set_engine
from synapse.mcp.server import mcp

EDGES = [
    ("Docker", "Kubernetes", 0.80),
    ("AWS", "Terraform", 0.75),
    ("React", "JavaScript", 0.90),   # a disconnected component
]


def build_graph() -> nx.Graph:
    G = nx.Graph()
    for n in ["Docker", "Kubernetes", "AWS", "Terraform", "React", "JavaScript"]:
        G.add_node(n, node_type="skill", category="test")
    for a, b, w in EDGES:
        G.add_edge(a, b, relation="similar", weight=w)
    return G


POOL = [
    {"candidate_id": "c1", "name": "devops",
     "skills": {"Docker": 1.5, "AWS": 1.0}, "unresolved": []},
    {"candidate_id": "c2", "name": "frontend",
     "skills": {"React": 1.0, "JavaScript": 1.0}, "unresolved": ["vibes"]},
]

LISTING = [
    {"candidate_id": "c1", "name": "devops", "doc_type": "resume",
     "model": "gemini", "skill_count": 2, "unresolved_count": 0,
     "content_hash": "abc", "extracted_at": "2026-09-12T00:00:00Z"},
    {"candidate_id": "c2", "name": "frontend", "doc_type": "resume",
     "model": "gemini", "skill_count": 2, "unresolved_count": 1,
     "content_hash": "def", "extracted_at": "2026-09-12T00:00:00Z"},
]


class StubClient:
    def __init__(self, pool=None, listing=None, configured=True):
        self.config = Neo4jConfig(uri="neo4j+s://stub", password="x" if configured else "")
        self._pool = POOL if pool is None else pool
        self._listing = LISTING if listing is None else listing
        self.pool_loads = 0

    def load_candidate_pool(self):
        self.pool_loads += 1
        return [dict(c) for c in self._pool]

    def list_candidates(self):
        return [dict(r) for r in self._listing]


@pytest.fixture
def engine(tmp_path):
    path = tmp_path / "g.pkl"
    path.write_bytes(pickle.dumps(build_graph()))
    eng = MatchEngine(graph_path=path, neo4j_client=StubClient())
    eng._linker = EntityLinker(eng.skill_names, use_embeddings=False)
    return eng


@pytest.fixture
def served(engine):
    set_engine(engine)
    yield engine
    set_engine(None)


def sw(*names) -> list[SkillWeight]:
    return [SkillWeight(skill=n) for n in names]


JD = sw("Docker", "Kubernetes", "AWS")


# ------------------------------------------------------------------ engine


def test_omitting_candidates_ranks_the_pool(engine):
    response = engine.rank_candidates(jd_skills=JD)

    assert response.candidate_source == "pool"
    assert [c.name for c in response.candidates] == ["devops", "frontend"]
    assert response.candidates[0].total > response.candidates[1].total


def test_explicit_candidates_still_bypass_the_pool(engine):
    """The stateless path the eval harness relies on must keep working."""
    response = engine.rank_candidates(
        jd_skills=JD,
        candidates=[CandidateInput(name="adhoc", skills=sw("Docker"))],
    )
    assert response.candidate_source == "request"
    assert [c.name for c in response.candidates] == ["adhoc"]
    assert engine.neo4j.pool_loads == 0, "the pool must not be touched"


def test_empty_list_means_nothing_not_everything(engine):
    """`candidates=[]` must not silently rank the entire pool."""
    response = engine.rank_candidates(jd_skills=JD, candidates=[])
    assert response.candidates == []
    assert response.candidate_source == "request"


def test_stored_weights_are_used_verbatim(engine):
    """Profiles are already canonical and already weighted - no re-linking.

    Re-linking would let a linking change move the score of a candidate whose
    document has not been touched since ingestion.
    """
    response = engine.rank_candidates(jd_skills=sw("Docker"))
    devops = next(c for c in response.candidates if c.name == "devops")
    # Docker stored at 1.5; proficiency credit caps at 1.0 of a demand of 1.0.
    assert devops.matched_skills == ["Docker"]
    assert devops.total == pytest.approx(1.0)


def test_unresolved_from_ingestion_reaches_the_ranking(engine):
    response = engine.rank_candidates(jd_skills=JD)
    frontend = next(c for c in response.candidates if c.name == "frontend")
    assert frontend.unresolved_skills == ["vibes"]


def test_pool_is_loaded_once_and_cached(engine):
    engine.rank_candidates(jd_skills=JD)
    engine.rank_candidates(jd_skills=JD)
    assert engine.neo4j.pool_loads == 1, "ranking must not re-query per call"


def test_invalidate_pool_forces_a_reload(engine):
    engine.rank_candidates(jd_skills=JD)
    engine.invalidate_pool()
    engine.rank_candidates(jd_skills=JD)
    assert engine.neo4j.pool_loads == 2


def test_top_k_applies_to_the_pool(engine):
    response = engine.rank_candidates(jd_skills=JD, top_k=1)
    assert len(response.candidates) == 1


def test_unconfigured_database_fails_with_an_actionable_message(tmp_path):
    path = tmp_path / "g.pkl"
    path.write_bytes(pickle.dumps(build_graph()))
    eng = MatchEngine(graph_path=path, neo4j_client=StubClient(configured=False))
    with pytest.raises(RuntimeError, match="NEO4J_PASSWORD"):
        eng.rank_candidates(jd_skills=JD)


def test_list_pool_summarises_without_skill_payloads(engine):
    listing = engine.list_pool()
    assert listing.count == 2
    assert [c.name for c in listing.candidates] == ["devops", "frontend"]
    frontend = listing.candidates[1]
    assert frontend.skill_count == 2
    assert frontend.unresolved_count == 1


# ------------------------------------------------------------------- tools


def call_tool(name, arguments):
    from fastmcp import Client

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def test_rank_candidates_tool_defaults_to_the_pool(served):
    result = call_tool("rank_candidates", {
        "jd_skills": [{"skill": "Docker"}, {"skill": "Kubernetes"}],
    })
    assert result.data.candidate_source == "pool"
    assert [c.name for c in result.data.candidates] == ["devops", "frontend"]


def test_list_candidates_tool(served):
    result = call_tool("list_candidates", {})
    assert result.data.count == 2
    assert {c.candidate_id for c in result.data.candidates} == {"c1", "c2"}


def test_rank_candidates_tool_still_accepts_explicit_candidates(served):
    result = call_tool("rank_candidates", {
        "jd_skills": [{"skill": "Docker"}],
        "candidates": [{"name": "adhoc", "skills": [{"skill": "Docker"}]}],
    })
    assert result.data.candidate_source == "request"
    assert [c.name for c in result.data.candidates] == ["adhoc"]


# ------------------------------------------------------------------ routes


@pytest.fixture
def web(engine):
    from starlette.testclient import TestClient

    set_engine(engine)
    try:
        with TestClient(mcp.http_app()) as client:
            yield client
    finally:
        set_engine(None)


def test_api_candidates_lists_the_pool(web):
    body = web.get("/api/candidates").json()
    assert body["count"] == 2
    assert [c["name"] for c in body["candidates"]] == ["devops", "frontend"]


def test_api_rank_pool_ranks(web):
    body = web.post("/api/rank_pool", json={
        "jd_skills": [{"skill": "Docker", "weight": 1.5}, {"skill": "Kubernetes"}],
    }).json()
    assert body["candidate_source"] == "pool"
    assert [c["name"] for c in body["candidates"]] == ["devops", "frontend"]


def test_api_rank_pool_accepts_bare_skill_strings(web):
    body = web.post("/api/rank_pool", json={"jd_skills": ["Docker"]}).json()
    assert body["candidates"], "a plain list of names must work"


def test_api_rank_pool_rejects_bad_requests(web):
    assert web.post("/api/rank_pool", json={}).status_code == 400
    assert web.post("/api/rank_pool", content=b"not json").status_code == 400


def test_pool_routes_report_unavailable_when_the_database_is_not(tmp_path):
    from starlette.testclient import TestClient

    path = tmp_path / "g.pkl"
    path.write_bytes(pickle.dumps(build_graph()))
    set_engine(MatchEngine(graph_path=path, neo4j_client=StubClient(configured=False)))
    try:
        with TestClient(mcp.http_app()) as client:
            assert client.post(
                "/api/rank_pool", json={"jd_skills": ["Docker"]}
            ).status_code == 503
    finally:
        set_engine(None)
