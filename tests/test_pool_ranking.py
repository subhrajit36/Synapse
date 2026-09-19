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


# F2: c1 was uploaded in one session, c2 in two - the dedupe case where the
# same résumé is re-uploaded later and must belong to both batches.
POOL = [
    {"candidate_id": "c1", "name": "devops", "batch_ids": ["b1"],
     "skills": {"Docker": 1.5, "AWS": 1.0}, "unresolved": []},
    {"candidate_id": "c2", "name": "frontend", "batch_ids": ["b1", "b2"],
     "skills": {"React": 1.0, "JavaScript": 1.0}, "unresolved": ["vibes"]},
]

LISTING = [
    {"candidate_id": "c1", "name": "devops", "doc_type": "resume",
     "model": "gemini", "skill_count": 2, "unresolved_count": 0,
     "batch_ids": ["b1"],
     "content_hash": "abc", "extracted_at": "2026-09-12T00:00:00Z"},
    {"candidate_id": "c2", "name": "frontend", "doc_type": "resume",
     "model": "gemini", "skill_count": 2, "unresolved_count": 1,
     "batch_ids": ["b1", "b2"],
     "content_hash": "def", "extracted_at": "2026-09-12T00:00:00Z"},
]


class StubClient:
    def __init__(self, pool=None, listing=None, configured=True):
        self.config = Neo4jConfig(uri="neo4j+s://stub", password="x" if configured else "")
        self._pool = POOL if pool is None else pool
        self._listing = LISTING if listing is None else listing
        self.pool_loads = 0
        self.pool_batches: list[str | None] = []   # which slices were fetched

    @staticmethod
    def _in_batch(row, batch_id):
        # Same semantics as the Cypher: null = everyone, else list membership.
        return batch_id is None or batch_id in row.get("batch_ids", [])

    def load_candidate_pool(self, batch_id=None):
        self.pool_loads += 1
        self.pool_batches.append(batch_id)
        return [dict(c) for c in self._pool if self._in_batch(c, batch_id)]

    def list_candidates(self, batch_id=None):
        return [dict(r) for r in self._listing if self._in_batch(r, batch_id)]


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
    assert listing.batch_id is None
    assert [c.name for c in listing.candidates] == ["devops", "frontend"]
    frontend = listing.candidates[1]
    assert frontend.skill_count == 2
    assert frontend.unresolved_count == 1
    assert frontend.batch_ids == ["b1", "b2"]


# ------------------------------------------------------- F2: batch scoping


def test_batch_id_scopes_the_pool_to_one_session(engine):
    """The product page ranks only what this user uploaded."""
    response = engine.rank_candidates(jd_skills=JD, batch_id="b2")
    assert response.candidate_source == "pool"
    assert response.batch_id == "b2"
    assert [c.name for c in response.candidates] == ["frontend"]


def test_a_candidate_in_two_batches_is_visible_from_both(engine):
    """Dedupe: a re-uploaded résumé belongs to every session that sent it."""
    b1 = engine.rank_candidates(jd_skills=JD, batch_id="b1")
    b2 = engine.rank_candidates(jd_skills=JD, batch_id="b2")
    assert "frontend" in [c.name for c in b1.candidates]
    assert "frontend" in [c.name for c in b2.candidates]


def test_unknown_batch_ranks_nobody_rather_than_everybody(engine):
    """A wrong id must never quietly widen to the whole pool."""
    response = engine.rank_candidates(jd_skills=JD, batch_id="no-such-batch")
    assert response.candidates == []
    assert response.batch_id == "no-such-batch"


def test_empty_batch_id_means_the_whole_pool(engine):
    """An unset form field arrives as ''; treat it as 'not scoped'."""
    response = engine.rank_candidates(jd_skills=JD, batch_id="")
    assert len(response.candidates) == 2
    assert response.batch_id is None


def test_batch_id_cannot_be_combined_with_explicit_candidates(engine):
    with pytest.raises(ValueError, match="batch_id"):
        engine.rank_candidates(
            jd_skills=JD,
            candidates=[CandidateInput(name="adhoc", skills=sw("Docker"))],
            batch_id="b1",
        )


def test_pool_cache_is_keyed_by_batch(engine):
    """Two sessions ranking in turn must not evict each other's slice."""
    engine.rank_candidates(jd_skills=JD, batch_id="b1")
    engine.rank_candidates(jd_skills=JD, batch_id="b2")
    engine.rank_candidates(jd_skills=JD, batch_id="b1")
    engine.rank_candidates(jd_skills=JD)             # unscoped is its own slice
    assert engine.neo4j.pool_batches == ["b1", "b2", None]


def test_invalidating_one_batch_leaves_the_others_cached(engine):
    engine.rank_candidates(jd_skills=JD)              # None
    engine.rank_candidates(jd_skills=JD, batch_id="b1")
    engine.rank_candidates(jd_skills=JD, batch_id="b2")
    assert engine.neo4j.pool_loads == 3

    engine.invalidate_pool("b1")
    engine.rank_candidates(jd_skills=JD, batch_id="b2")   # untouched: cached
    assert engine.neo4j.pool_loads == 3
    engine.rank_candidates(jd_skills=JD, batch_id="b1")   # dropped: reloads
    assert engine.neo4j.pool_loads == 4
    # The unscoped view is a superset of every batch, so it was stale too.
    engine.rank_candidates(jd_skills=JD)
    assert engine.neo4j.pool_loads == 5


def test_invalidating_without_a_batch_drops_everything(engine):
    engine.rank_candidates(jd_skills=JD, batch_id="b1")
    engine.rank_candidates(jd_skills=JD, batch_id="b2")
    engine.invalidate_pool()
    engine.rank_candidates(jd_skills=JD, batch_id="b1")
    engine.rank_candidates(jd_skills=JD, batch_id="b2")
    assert engine.neo4j.pool_loads == 4


def test_list_pool_scoped_to_a_batch(engine):
    listing = engine.list_pool(batch_id="b2")
    assert listing.batch_id == "b2"
    assert listing.count == 1
    assert listing.candidates[0].candidate_id == "c2"


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


def test_rank_candidates_tool_accepts_a_batch_id(served):
    result = call_tool("rank_candidates", {
        "jd_skills": [{"skill": "Docker"}], "batch_id": "b2",
    })
    assert result.data.batch_id == "b2"
    assert [c.name for c in result.data.candidates] == ["frontend"]


def test_list_candidates_tool_accepts_a_batch_id(served):
    result = call_tool("list_candidates", {"batch_id": "b2"})
    assert result.data.count == 1
    assert result.data.candidates[0].candidate_id == "c2"


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


def test_api_candidates_filters_by_batch(web):
    body = web.get("/api/candidates", params={"batch_id": "b2"}).json()
    assert body["batch_id"] == "b2"
    assert [c["candidate_id"] for c in body["candidates"]] == ["c2"]


def test_api_rank_pool_scopes_by_batch(web):
    body = web.post("/api/rank_pool", json={
        "jd_skills": ["Docker"], "batch_id": "b2",
    }).json()
    assert body["batch_id"] == "b2"
    assert [c["name"] for c in body["candidates"]] == ["frontend"]


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
