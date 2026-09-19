"""Phase F4: a free-text JD becomes rankable skills in one call.

No Gemini: the extractor is given a fake client that returns canned JSON, so
what is under test is the one-call contract, the linking, and the route.
"""

from __future__ import annotations

import pickle

import networkx as nx
import pytest

from synapse.graph.neo4j_client import Neo4jConfig
from synapse.ingest.extractor import SkillExtractor
from synapse.matching.entity_linker import EntityLinker
from synapse.mcp.engine import MatchEngine, set_engine
from synapse.mcp.server import mcp

NODES = ["Docker", "Kubernetes", "Python", "React"]

# What the model "says" about the JD: a core skill, an alias, a duplicate
# mention, and one thing the graph has never heard of.
PAYLOAD = (
    '[{"skill": "Kubernetes", "weight": 1.5, "context": "must have"},'
    ' {"skill": "k8s", "weight": 1.0, "context": "clusters"},'
    ' {"skill": "Docker", "weight": 1.0, "context": "containers"},'
    ' {"skill": "Docker", "weight": 0.5, "context": "mentioned again"},'
    ' {"skill": "quantum basket weaving", "weight": 0.5, "context": "bonus"}]'
)

JD_TEXT = """
Senior platform engineer.   Must have Kubernetes; k8s at scale.
Docker daily.   Bonus: quantum basket weaving.
"""


class FakeGemini:
    def __init__(self, payload=PAYLOAD, error: Exception | None = None):
        self.calls: list[str] = []
        self.payload, self.error = payload, error
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs["contents"])
        if self.error:
            raise self.error

        class R:
            text = self.payload

        return R()


class StubClient:
    """One stored candidate, so the JD -> rank round trip has something to score."""

    def __init__(self, configured=True):
        self.config = Neo4jConfig(uri="neo4j+s://stub", password="x" if configured else "")

    def load_candidate_pool(self, batch_id=None):
        return [{"candidate_id": "c1", "name": "devops", "batch_ids": [],
                 "skills": {"Docker": 1.0}, "unresolved": []}]

    def list_candidates(self, batch_id=None):
        return []


def engine_with(tmp_path, gemini: FakeGemini | None = None, extractor="fake"):
    G = nx.Graph()
    for n in NODES:
        G.add_node(n, node_type="skill", category="t")
    G.add_edge("Docker", "Kubernetes", relation="similar", weight=0.8)
    path = tmp_path / "g.pkl"
    path.write_bytes(pickle.dumps(G))

    ext = SkillExtractor(client=gemini or FakeGemini(), rpm=0) if extractor == "fake" else extractor
    eng = MatchEngine(graph_path=path, neo4j_client=StubClient(), extractor=ext)
    eng._linker = EntityLinker(NODES, use_embeddings=False)
    return eng


# ------------------------------------------------------------------ engine


def test_one_call_and_the_whole_text_goes_in_it(tmp_path):
    gemini = FakeGemini()
    eng = engine_with(tmp_path, gemini)
    r = eng.extract_jd_skills(JD_TEXT)

    assert len(gemini.calls) == 1, "one chunk, one call"
    sent = gemini.calls[0]
    assert "Kubernetes" in sent and "quantum basket weaving" in sent
    assert "   " not in sent, "text is normalised before it is sent"
    assert r.text_words == len(sent.split())


def test_skills_are_canonical_weighted_and_rankable(tmp_path):
    eng = engine_with(tmp_path)
    r = eng.extract_jd_skills(JD_TEXT)

    by_node = {s.skill: s.weight for s in r.skills}
    # 'Kubernetes' (1.5) and its alias 'k8s' (1.0) collapse to one node at the
    # stronger demand; the two Docker mentions collapse at extraction time.
    assert by_node == {"Kubernetes": 1.5, "Docker": 1.0}
    assert [s.skill for s in r.skills] == ["Kubernetes", "Docker"], "demand-desc order"
    assert r.unresolved == ["quantum basket weaving"]

    # The contract that matters: the output ranks without any translation.
    ranked = eng.rank_candidates(
        jd_skills=r.skills,
        candidates=[],
        link=False,
    )
    assert ranked.jd_skills == ["Docker", "Kubernetes"]
    assert ranked.jd_unresolved == []


def test_linking_trace_and_raw_extraction_are_exposed(tmp_path):
    """NFR6: the page can show 'k8s -> Kubernetes (alias)' next to the JD."""
    r = engine_with(tmp_path).extract_jd_skills(JD_TEXT)

    assert {s.skill for s in r.extracted} == {"Kubernetes", "k8s", "Docker",
                                              "quantum basket weaving"}
    trace = {l.surface: (l.node, l.method) for l in r.linking}
    assert trace["k8s"] == ("Kubernetes", "alias")
    assert trace["quantum basket weaving"] == (None, "unresolved")


def test_empty_text_is_a_bad_request(tmp_path):
    eng = engine_with(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        eng.extract_jd_skills("   \n  ")


def test_model_failure_is_reported_not_retried(tmp_path):
    """Synchronous: the user has a Retry button; the engine must not sleep."""
    gemini = FakeGemini(error=ConnectionError("503 unavailable"))
    eng = engine_with(tmp_path, gemini)
    with pytest.raises(RuntimeError, match="JD extraction failed"):
        eng.extract_jd_skills(JD_TEXT)
    assert len(gemini.calls) == 1


def test_bad_model_output_is_a_runtime_error_too(tmp_path):
    eng = engine_with(tmp_path, FakeGemini(payload="not json at all"))
    with pytest.raises(RuntimeError, match="JD extraction failed"):
        eng.extract_jd_skills(JD_TEXT)


def test_missing_api_key_surfaces_as_a_service_error(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    eng = engine_with(tmp_path, extractor=None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        eng.extract_jd_skills(JD_TEXT)


def test_extractor_is_shared_with_the_upload_pipeline(tmp_path):
    """One model, one prompt, for both sides of the match."""
    eng = engine_with(tmp_path)
    eng.checkpoint_path = None
    assert eng.pipeline.extractor is eng.extractor


# ------------------------------------------------------------------- route


@pytest.fixture
def web(tmp_path):
    from starlette.testclient import TestClient

    set_engine(engine_with(tmp_path))
    try:
        with TestClient(mcp.http_app()) as client:
            yield client
    finally:
        set_engine(None)


def test_route_returns_rankable_skills(web):
    res = web.post("/api/jd_skills", json={"text": JD_TEXT})
    assert res.status_code == 200, res.text
    body = res.json()
    assert [s["skill"] for s in body["skills"]] == ["Kubernetes", "Docker"]
    assert body["unresolved"] == ["quantum basket weaving"]

    # Round trip: the response feeds the ranking route verbatim, and the
    # stored candidate's Docker bridges to the JD's Kubernetes.
    ranked = web.post("/api/rank_pool", json={"jd_skills": body["skills"]})
    assert ranked.status_code == 200, ranked.text
    (devops,) = ranked.json()["candidates"]
    assert devops["matched_skills"] == ["Docker"]
    assert [g["skill"] for g in devops["bridged_skills"]] == ["Kubernetes"]


def test_route_rejects_bad_requests(web):
    assert web.post("/api/jd_skills", json={}).status_code == 400
    assert web.post("/api/jd_skills", json={"text": "   "}).status_code == 400
    assert web.post("/api/jd_skills", json={"text": 42}).status_code == 400
    assert web.post("/api/jd_skills", content=b"nope").status_code == 400


def test_route_is_503_when_the_model_fails(tmp_path):
    from starlette.testclient import TestClient

    set_engine(engine_with(tmp_path, FakeGemini(error=TimeoutError("timeout"))))
    try:
        with TestClient(mcp.http_app()) as client:
            res = client.post("/api/jd_skills", json={"text": JD_TEXT})
            assert res.status_code == 503
            assert "JD extraction failed" in res.json()["error"]
    finally:
        set_engine(None)
