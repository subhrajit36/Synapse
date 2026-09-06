"""AuraDB wiring: config, loader, engine source selection, dynamic MERGE.

No network. A stub client returns the same row shapes the real Cypher returns -
those statements were validated against the live instance separately, so what is
under test here is the wiring around them.
"""

from __future__ import annotations

import pickle

import networkx as nx
import pytest

from synapse.graph.neo4j_client import (
    DEFAULT_MAX_HOPS,
    MAX_HOPS_CEILING,
    Neo4jClient,
    Neo4jConfig,
)
from synapse.graph.neo4j_loader import GraphParityError, load_graph_from_neo4j
from synapse.mcp.engine import (
    GRAPH_SOURCE_NEO4J,
    GRAPH_SOURCE_PICKLE,
    MatchEngine,
)

# Mirrors the real graph's shape: two connected skills plus an isolated one.
STUB_NODES = [
    {"name": "Docker", "category": "Application server software",
     "embed_category": None, "source": "onet"},
    {"name": "Kubernetes", "category": "Application server software",
     "embed_category": None, "source": "onet"},
    {"name": "Python", "category": "Development environment software",
     "embed_category": None, "source": "onet"},
]
STUB_EDGES = [{"lo": "Docker", "hi": "Kubernetes", "weight": 0.77}]
STUB_ROLES = [("DevOps Engineer", "Docker"), ("DevOps Engineer", "Kubernetes")]
# One role deliberately has no REQUIRES edges, mirroring the two such roles in
# the real O*NET data that an edge-derived loader would silently drop.
STUB_ROLE_NODES = [{"name": "DevOps Engineer", "soc": "15-1244"},
                   {"name": "Computer Occupations, All Other", "soc": "15-1299"}]


class StubClient:
    """Row shapes match what the validated Cypher actually returns."""

    def __init__(self, nodes=None, edges=None, roles=None, configured=True):
        # Copied, not aliased: a MERGE appends to this list, and sharing the
        # module-level default would leak writes between tests.
        self._nodes = list(STUB_NODES if nodes is None else nodes)
        self._edges = list(STUB_EDGES if edges is None else edges)
        self._roles = list(STUB_ROLES if roles is None else roles)
        self.config = Neo4jConfig(uri="neo4j+s://stub.example", password="x" if configured else "")
        self.merged: list[tuple] = []
        self.dedup_hit: str | None = None

    def iter_skill_graph(self):
        return list(self._nodes), list(self._edges)

    def iter_roles(self):
        return list(STUB_ROLE_NODES)

    def iter_role_requirements(self):
        return list(self._roles)

    def get_or_create_skill_with_dedup(self, name, embedding, category=None,
                                       min_similarity=0.82):
        self.merged.append((name, min_similarity))
        if self.dedup_hit:
            return self.dedup_hit, False
        self._nodes.append({"name": name, "category": category,
                            "embed_category": None, "source": "dynamic"})
        return name, True


# ------------------------------------------------------------------- config


def test_config_reads_env_at_instantiation_not_import(monkeypatch):
    """The bug that made a misconfiguration look like a network failure."""
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://late.example")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    cfg = Neo4jConfig()
    assert cfg.uri == "neo4j+s://late.example"
    assert cfg.is_configured


def test_config_reports_missing_password_without_leaking_it(monkeypatch):
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    cfg = Neo4jConfig()
    assert cfg.is_configured is False
    assert "MISSING" in cfg.describe()

    cfg2 = Neo4jConfig(password="hunter2")
    assert "hunter2" not in cfg2.describe()
    assert "password=set" in cfg2.describe()


def test_ping_returns_false_instead_of_raising(monkeypatch):
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    assert Neo4jClient(Neo4jConfig()).ping() is False


def test_hop_bounds_are_enforced():
    """`max_hops=None` must never become an unbounded traversal."""
    assert DEFAULT_MAX_HOPS >= 1
    assert MAX_HOPS_CEILING >= DEFAULT_MAX_HOPS
    client = Neo4jClient(Neo4jConfig(password="x"))
    with pytest.raises(ValueError, match="max_hops"):
        client.shortest_path_distance(["A"], "B", max_hops=0)


# ------------------------------------------------------------------- loader


def test_loader_builds_a_matcher_compatible_graph():
    G = load_graph_from_neo4j(StubClient())

    # Attribute names Matcher and EntityLinker select on.
    assert G.nodes["Docker"]["node_type"] == "skill"
    assert G.nodes["Docker"]["category"] == "Application server software"
    assert G["Docker"]["Kubernetes"]["relation"] == "similar"
    assert G["Docker"]["Kubernetes"]["weight"] == pytest.approx(0.77)
    assert G.nodes["DevOps Engineer"]["node_type"] == "role"
    assert G["DevOps Engineer"]["Docker"]["relation"] == "requires"


def test_loaded_graph_drives_the_matcher_unchanged():
    from synapse.matching.matcher import Matcher, ScoringParams

    G = load_graph_from_neo4j(StubClient())
    m = Matcher(G, params=ScoringParams(bridge_cutoff=0.7, max_hops=2))
    result = m.match({"Kubernetes": 1.0}, {"Docker": 1.0})

    (bridge,) = result.bridged_skills
    assert (bridge.skill, bridge.via, bridge.hops) == ("Kubernetes", "Docker", 1)
    assert bridge.distance == pytest.approx(0.23)


def test_roles_can_be_omitted():
    G = load_graph_from_neo4j(StubClient(), with_roles=False)
    assert all(d.get("node_type") == "skill" for _, d in G.nodes(data=True))


def test_parity_mismatch_aborts_startup():
    with pytest.raises(GraphParityError, match="expected 213 skills"):
        load_graph_from_neo4j(StubClient(), expected_skills=213)
    with pytest.raises(GraphParityError, match="similarity pairs"):
        load_graph_from_neo4j(StubClient(), expected_pairs=15459)


def test_empty_graph_is_an_error_not_a_silent_success():
    with pytest.raises(GraphParityError, match="no Skill nodes"):
        load_graph_from_neo4j(StubClient(nodes=[], edges=[], roles=[]))


def test_parity_passes_on_the_expected_shape():
    G = load_graph_from_neo4j(StubClient(), expected_skills=3, expected_pairs=1)
    assert G.number_of_nodes() == 5  # 3 skills + 2 roles


def test_roles_without_requirements_are_still_loaded():
    """Regression: deriving roles from the REQUIRES list dropped edgeless ones.

    The real graph has two such roles, which made a Neo4j-loaded graph report
    249 nodes against the pickle's 251 - a silent shape mismatch.
    """
    G = load_graph_from_neo4j(StubClient())
    assert "Computer Occupations, All Other" in G
    assert G.nodes["Computer Occupations, All Other"]["node_type"] == "role"
    assert G.degree("Computer Occupations, All Other") == 0


# ------------------------------------------------------- engine source wiring


def test_engine_defaults_to_pickle_and_never_dials(tmp_path):
    path = tmp_path / "g.pkl"
    G = nx.Graph()
    G.add_node("Docker", node_type="skill", category="c")
    path.write_bytes(pickle.dumps(G))

    engine = MatchEngine(graph_path=path, graph_source=GRAPH_SOURCE_PICKLE)
    assert engine.skill_names == ["Docker"]
    assert engine.stats().graph_source == GRAPH_SOURCE_PICKLE


def test_engine_loads_from_neo4j_when_told_to():
    engine = MatchEngine(graph_source=GRAPH_SOURCE_NEO4J, neo4j_client=StubClient(), expected_skills=None, expected_pairs=None)
    assert sorted(engine.skill_names) == ["Docker", "Kubernetes", "Python"]

    stats = engine.stats()
    assert stats.graph_source == GRAPH_SOURCE_NEO4J
    assert stats.graph_path == "neo4j+s://stub.example"   # not a filesystem path
    assert stats.skill_nodes == 3
    assert stats.similar_edges == 1
    assert stats.orphan_skills == 1                       # Python has no edges


def test_neo4j_source_without_credentials_fails_loudly():
    """It must not quietly fall back to the pickle - that would break NFR4."""
    engine = MatchEngine(graph_source=GRAPH_SOURCE_NEO4J, neo4j_client=StubClient(configured=False), expected_skills=None, expected_pairs=None)
    with pytest.raises(RuntimeError, match="NEO4J_PASSWORD"):
        _ = engine.graph


def test_unknown_source_is_rejected():
    engine = MatchEngine(graph_source="postgres")
    with pytest.raises(ValueError, match="Unknown SYNAPSE_GRAPH_SOURCE"):
        _ = engine.graph


def test_engine_is_still_lazy():
    """Constructing must not query. Render cold starts pay only on first use."""
    stub = StubClient()
    engine = MatchEngine(graph_source=GRAPH_SOURCE_NEO4J, neo4j_client=stub, expected_skills=None, expected_pairs=None)
    assert engine._graph is None and engine._matcher is None


# --------------------------------------------------------- C2.5 dynamic MERGE


class FakeEmbedder:
    def encode(self, texts):
        import numpy as np
        return np.ones((len(texts), 384), dtype="float32")


def _engine_with_embedder(stub):
    from synapse.matching.entity_linker import EntityLinker

    engine = MatchEngine(graph_source=GRAPH_SOURCE_NEO4J, neo4j_client=stub, expected_skills=None, expected_pairs=None)
    engine._linker = EntityLinker(engine.skill_names, embedder=FakeEmbedder())
    return engine


def test_register_skills_merges_and_reports():
    stub = StubClient()
    engine = _engine_with_embedder(stub)

    (result,) = engine.register_skills(["Rust"])
    assert (result.surface, result.node, result.created) == ("Rust", "Rust", True)
    assert stub.merged == [("Rust", 0.82)]


def test_register_skills_reuses_a_near_duplicate_instead_of_creating():
    """The C2.5 guard: don't create 'Javascript' beside 'JavaScript'."""
    stub = StubClient()
    stub.dedup_hit = "Kubernetes"
    engine = _engine_with_embedder(stub)

    (result,) = engine.register_skills(["k8s"])
    assert result.node == "Kubernetes"
    assert result.created is False


def test_creating_a_node_invalidates_the_cached_graph():
    stub = StubClient()
    engine = _engine_with_embedder(stub)
    assert len(engine.skill_names) == 3

    engine.register_skills(["Rust"])
    assert engine._graph is None, "graph must be reloaded after a write"
    assert "Rust" in engine.skill_names


def test_dedup_hit_does_not_invalidate():
    stub = StubClient()
    stub.dedup_hit = "Kubernetes"
    engine = _engine_with_embedder(stub)
    _ = engine.graph

    engine.register_skills(["k8s"])
    assert engine._graph is not None, "nothing was written; no reload needed"


def test_writes_are_refused_against_a_pickle(tmp_path):
    path = tmp_path / "g.pkl"
    G = nx.Graph()
    G.add_node("Docker", node_type="skill", category="c")
    path.write_bytes(pickle.dumps(G))

    engine = MatchEngine(graph_path=path, graph_source=GRAPH_SOURCE_PICKLE)
    with pytest.raises(RuntimeError, match="requires SYNAPSE_GRAPH_SOURCE=neo4j"):
        engine.register_skills(["Rust"])


