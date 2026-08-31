"""Phase C4 tests: the MCP tool layer.

Everything runs against a small synthetic graph whose right answers are known
by construction (the A3.3 discipline), so a failure here means the serving layer
is wrong, not that the real graph shifted.
"""

from __future__ import annotations

import asyncio
import pickle

import networkx as nx
import pytest

from synapse.matching.entity_linker import EntityLinker
from synapse.matching.matcher import TUNED_PARAMS
from synapse.mcp.engine import (
    CandidateInput,
    MatchEngine,
    SkillWeight,
    get_engine,
    set_engine,
)
from synapse.mcp.server import mcp

# Two disconnected components, so "bridgeable" and "no path" are both reachable
# states. Weights are similarities; the matcher converts to distance = 1 - w.
EDGES = [
    ("Docker", "Kubernetes", 0.80),      # distance 0.20
    ("Kubernetes", "Helm", 0.90),        # distance 0.10  -> Docker..Helm = 0.30, 2 hops
    ("AWS", "Terraform", 0.75),          # distance 0.25
    ("Docker", "AWS", 0.65),             # distance 0.35
    ("React", "JavaScript", 0.90),       # the other component
]


def build_graph() -> nx.Graph:
    G = nx.Graph()
    for name in ["Docker", "Kubernetes", "Helm", "AWS", "Terraform", "React",
                 "JavaScript", "Python"]:
        G.add_node(name, node_type="skill", category="test")
    for a, b, w in EDGES:
        G.add_edge(a, b, relation="similar", weight=w)
    return G


@pytest.fixture
def engine(tmp_path) -> MatchEngine:
    """An engine over the synthetic graph, with embeddings switched off.

    Alias and surface linking still run - those are deterministic. Only the
    embedding fallback is disabled, so no test downloads an ONNX model.
    """
    path = tmp_path / "skill_graph.pkl"
    path.write_bytes(pickle.dumps(build_graph()))
    eng = MatchEngine(graph_path=path)
    eng._linker = EntityLinker(eng.skill_names, use_embeddings=False)
    return eng


@pytest.fixture
def served(engine):
    """Point the MCP singleton at the synthetic engine for the duration."""
    set_engine(engine)
    yield engine
    set_engine(None)


def sw(*pairs) -> list[SkillWeight]:
    """('Docker', 1.5) or plain 'Docker' -> SkillWeight list."""
    out = []
    for p in pairs:
        skill, weight = p if isinstance(p, tuple) else (p, 1.0)
        out.append(SkillWeight(skill=skill, weight=weight))
    return out


JD = sw(("Kubernetes", 1.5), ("Docker", 1.5), ("AWS", 1.0), ("Terraform", 1.0))


# --------------------------------------------------------------------- FR5


def test_ranking_orders_by_fit(engine):
    response = engine.rank_candidates(
        jd_skills=JD,
        candidates=[
            CandidateInput(name="frontend", skills=sw("React", "JavaScript")),
            CandidateInput(name="devops", skills=sw("Docker", "AWS", "Python")),
        ],
    )

    assert [c.name for c in response.candidates] == ["devops", "frontend"]
    assert response.candidates[0].total > response.candidates[1].total
    assert response.jd_skills == ["AWS", "Docker", "Kubernetes", "Terraform"]


def test_score_components_are_returned_and_consistent(engine):
    response = engine.rank_candidates(
        jd_skills=JD,
        candidates=[CandidateInput(name="devops", skills=sw("Docker", "AWS"))],
    )
    score = response.candidates[0]

    assert sorted(score.matched_skills) == ["AWS", "Docker"]
    # Kubernetes (via Docker) and Terraform (via AWS) are both one hop away.
    assert {g.skill for g in score.bridged_skills} == {"Kubernetes", "Terraform"}
    assert score.missing_skills == []
    # NFR6: the reported total must be reconstructible from its parts.
    expected = (
        score.direct_match_score + score.bridge_score - score.gap_penalty
    ) / score.total_demand
    assert score.total == pytest.approx(expected, abs=5e-4)


def test_top_k_truncates(engine):
    response = engine.rank_candidates(
        jd_skills=JD,
        candidates=[
            CandidateInput(name="a", skills=sw("Docker")),
            CandidateInput(name="b", skills=sw("React")),
            CandidateInput(name="c", skills=sw("AWS")),
        ],
        top_k=2,
    )
    assert len(response.candidates) == 2


def test_served_params_are_the_tuned_ones_not_the_defaults(engine):
    response = engine.rank_candidates(jd_skills=JD, candidates=[])
    assert response.params["bridge_cutoff"] == TUNED_PARAMS.bridge_cutoff
    assert response.params["max_hops"] == TUNED_PARAMS.max_hops
    assert response.params["bridge_credit_scale"] == TUNED_PARAMS.bridge_credit_scale
    # The ceiling must reach the wire too - it is what keeps a served ranking
    # from putting a candidate who holds nothing above one who holds everything.
    assert response.params["max_bridge_credit"] == TUNED_PARAMS.max_bridge_credit
    assert response.params["max_bridge_credit"] < 1.0


# --------------------------------------------------------------------- FR4


def test_bridgeable_gaps_report_the_path(engine):
    response = engine.get_bridgeable_gaps(
        candidate_skills=sw("Docker"), jd_skills=sw("Kubernetes")
    )

    assert response.gaps == []
    (bridge,) = response.bridgeable
    assert (bridge.skill, bridge.via, bridge.hops) == ("Kubernetes", "Docker", 1)
    assert bridge.distance == pytest.approx(0.20)
    assert bridge.bridgeable


def test_unreachable_gap_is_json_safe_and_reasoned(engine):
    """Infinity does not survive JSON; -1 plus `reason` does."""
    response = engine.get_bridgeable_gaps(
        candidate_skills=sw("React"), jd_skills=sw("Kubernetes")
    )

    assert response.bridgeable == []
    (gap,) = response.gaps
    assert gap.distance == -1.0
    assert gap.hops is None
    assert gap.reason == "no_path"


def test_max_hops_override_actually_reaches_the_matcher(engine):
    """Docker -> Kubernetes -> Helm is 2 hops at distance 0.30.

    Under the tuned max_hops=2 it bridges; at max_hops=1 it must not. This is
    the exact override path that was silently a no-op before the matcher fix,
    so it is worth pinning at the serving layer too.
    """
    two = engine.get_bridgeable_gaps(sw("Docker"), sw("Helm"), max_hops=2)
    assert [g.skill for g in two.bridgeable] == ["Helm"]
    assert two.bridgeable[0].hops == 2

    one = engine.get_bridgeable_gaps(sw("Docker"), sw("Helm"), max_hops=1)
    assert one.bridgeable == []
    assert one.gaps[0].reason == "beyond_hops(2>1)"


def test_bridging_can_be_disabled(engine):
    response = engine.rank_candidates(
        jd_skills=sw("Kubernetes"),
        candidates=[CandidateInput(name="x", skills=sw("Docker"))],
        enable_bridging=False,
    )
    score = response.candidates[0]
    assert score.bridged_skills == []
    assert score.missing_skills[0].reason == "bridging_disabled"


# ------------------------------------------------------------ linking / NFR6


def test_aliases_resolve_before_scoring(engine):
    response = engine.explain_score(jd_skills=sw("Kubernetes"), candidate_skills=sw("k8s"))
    assert response.score.matched_skills == ["Kubernetes"]
    assert response.candidate_linking[0].node == "Kubernetes"
    assert response.candidate_linking[0].method == "alias"


def test_unresolved_surfaces_are_surfaced_not_dropped(engine):
    """A skill that reaches no node leaves the denominator - callers must see it."""
    response = engine.rank_candidates(
        jd_skills=sw("Docker", "quantum basket weaving"),
        candidates=[CandidateInput(name="x", skills=sw("Docker", "underwater welding"))],
    )

    assert response.jd_unresolved == ["quantum basket weaving"]
    assert response.jd_skills == ["Docker"]
    assert response.candidates[0].unresolved_skills == ["underwater welding"]


def test_explain_returns_a_readable_derivation(engine):
    response = engine.explain_score(
        jd_skills=JD, candidate_skills=sw("Docker", "AWS"), candidate_name="devops"
    )
    assert response.score.name == "devops"
    assert "fit=" in response.explanation and "bridged" in response.explanation
    assert {l.surface for l in response.jd_linking} == {
        "Kubernetes", "Docker", "AWS", "Terraform"
    }


def test_link_false_treats_names_as_nodes_and_flags_strangers(engine):
    response = engine.rank_candidates(
        jd_skills=sw("Docker"),
        candidates=[CandidateInput(name="x", skills=sw("Docker", "k8s"))],
        link=False,
    )
    # 'k8s' is an alias, not a node: without linking it must not resolve.
    assert response.candidates[0].unresolved_skills == ["k8s"]
    assert response.candidates[0].matched_skills == ["Docker"]


def test_weights_change_the_score(engine):
    weighted = engine.rank_candidates(
        jd_skills=sw(("Docker", 1.5), ("React", 0.5)),
        candidates=[CandidateInput(name="x", skills=sw(("Docker", 1.5)))],
    ).candidates[0]
    uniform = engine.rank_candidates(
        jd_skills=sw(("Docker", 1.5), ("React", 0.5)),
        candidates=[CandidateInput(name="x", skills=sw(("Docker", 1.5)))],
        use_weights=False,
    ).candidates[0]

    assert weighted.total_demand == pytest.approx(2.0)
    assert uniform.total_demand == pytest.approx(2.0 - 0.0)  # 2 skills, 1.0 each
    assert weighted.total != uniform.total


# ---------------------------------------------------------------- diagnostics


def test_graph_stats_describes_the_loaded_graph(engine):
    stats = engine.stats()
    assert stats.skill_nodes == 8
    assert stats.similar_edges == len(EDGES)
    assert stats.orphan_skills == 1          # Python has no 'similar' edge
    assert stats.scoring_params["max_hops"] == TUNED_PARAMS.max_hops


def test_missing_graph_fails_with_an_actionable_message(tmp_path):
    eng = MatchEngine(graph_path=tmp_path / "nope.pkl")
    with pytest.raises(FileNotFoundError, match="build_graph_artifact"):
        _ = eng.graph


def test_engine_is_lazy(tmp_path):
    """Importing or constructing must not read the pickle (Render cold start)."""
    eng = MatchEngine(graph_path=tmp_path / "absent.pkl")
    assert eng._graph is None and eng._matcher is None and eng._linker is None


def test_singleton_round_trips():
    original = MatchEngine(graph_path="data/skill_graph.pkl")
    set_engine(original)
    try:
        assert get_engine() is original
    finally:
        set_engine(None)


# ------------------------------------------------------------ MCP round-trip


def call_tool(name: str, arguments: dict):
    """Drive the real server in-memory, exactly as an MCP client would."""
    from fastmcp import Client

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def test_all_four_tools_are_registered():
    tools = asyncio.run(mcp._list_tools())
    assert {t.name for t in tools} == {
        "rank_candidates", "get_bridgeable_gaps", "explain_score", "graph_stats"
    }
    for tool in tools:
        assert tool.description, f"{tool.name} has no description for the client to read"


def test_rank_candidates_over_mcp(served):
    result = call_tool("rank_candidates", {
        "jd_skills": [{"skill": "Kubernetes", "weight": 1.5},
                      {"skill": "Docker", "weight": 1.5}],
        "candidates": [
            {"name": "devops", "skills": [{"skill": "Docker", "weight": 1.5}]},
            {"name": "frontend", "skills": [{"skill": "React", "weight": 1.0}]},
        ],
    })

    data = result.data
    assert [c.name for c in data.candidates] == ["devops", "frontend"]
    assert data.candidates[0].bridged_skills[0].skill == "Kubernetes"


def test_bridgeable_gaps_over_mcp(served):
    result = call_tool("get_bridgeable_gaps", {
        "candidate_skills": [{"skill": "Docker"}],
        "jd_skills": [{"skill": "Kubernetes"}, {"skill": "React"}],
    })

    data = result.data
    assert [g.skill for g in data.bridgeable] == ["Kubernetes"]
    assert [g.skill for g in data.gaps] == ["React"]


def test_tool_output_is_json_serializable(served):
    """No inf/nan may escape: MCP payloads are strict JSON."""
    import json

    result = call_tool("get_bridgeable_gaps", {
        "candidate_skills": [{"skill": "React"}],
        "jd_skills": [{"skill": "Kubernetes"}],
    })
    text = result.content[0].text
    json.loads(text)  # raises on Infinity
    assert "Infinity" not in text


def test_bad_arguments_are_rejected_by_the_schema(served):
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError):
        call_tool("rank_candidates", {"jd_skills": [{"skill": "Docker"}]})  # no candidates


def test_health_route_does_not_touch_the_graph():
    """The probe must answer while the graph is still unloaded (C6.3)."""
    from starlette.testclient import TestClient

    set_engine(MatchEngine(graph_path="does/not/exist.pkl"))
    try:
        with TestClient(mcp.http_app()) as client:
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    finally:
        set_engine(None)
