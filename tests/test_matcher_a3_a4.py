"""Phase A3/A4 tests.

Per CLAUDE.md A3.3 these run on tiny graphs whose correct answers are known by
construction, so a logic error surfaces here rather than hiding inside a
10,000-node graph.

    Chain graph (edge label = similarity, distance = 1 - similarity):

        A --0.9-- B --0.9-- C --0.5-- D        E (isolated)

    distances from A:  B=0.1, C=0.2, D=0.7
    hops      from A:  B=1,   C=2,   D=3
"""

from __future__ import annotations

import networkx as nx
import pytest

from synapse.matching.matcher import Matcher, ScoringParams


@pytest.fixture
def chain() -> nx.Graph:
    G = nx.Graph()
    for n in "ABCDE":
        G.add_node(n, node_type="skill", category="test")
    G.add_edge("A", "B", relation="similar", weight=0.9)
    G.add_edge("B", "C", relation="similar", weight=0.9)
    G.add_edge("C", "D", relation="similar", weight=0.5)
    G.add_node("Role", node_type="role")
    G.add_edge("Role", "A", relation="requires")  # must be ignored by bridging
    return G


# ------------------------------------------------------------------- graph build


def test_only_similar_edges_enter_the_skill_graph(chain):
    m = Matcher(chain)
    assert "Role" not in m.skill_graph
    assert m.skill_graph["A"]["B"]["distance"] == pytest.approx(0.1)


# ------------------------------------------------------------------- reachability


def test_distances_and_hops_are_computed_independently(chain):
    m = Matcher(chain, params=ScoringParams(bridge_cutoff=None))
    dist, hops, via = m._reachability(["A"])
    assert dist["C"] == pytest.approx(0.2)
    assert hops["C"] == 2
    assert via["C"] == "A"
    assert "E" not in dist  # unreachable, never invented


def test_isolated_skill_is_a_true_gap(chain):
    r = Matcher(chain).match(["E"], ["A"])
    assert [g.skill for g in r.missing_skills] == ["E"]
    assert r.missing_skills[0].distance == float("inf")
    assert r.missing_skills[0].via is None


def test_candidate_with_no_graph_skills_bridges_nothing(chain):
    r = Matcher(chain).match(["B"], ["not_in_graph"])
    assert r.total == 0.0
    assert len(r.missing_skills) == 1


# ---------------------------------------------------------------------- bridging


def test_distance_cutoff_decides_bridgeability(chain):
    m = Matcher(chain, bridge_cutoff=0.25)
    r = m.match(["C", "D"], ["A"])
    assert [g.skill for g in r.bridged_skills] == ["C"]   # 0.2 <= 0.25
    assert [g.skill for g in r.missing_skills] == ["D"]   # 0.7 >  0.25


def test_max_hops_ablation_arm(chain):
    """B4.3: hop radius must change the answer independently of distance."""
    one = Matcher(chain, params=ScoringParams(bridge_cutoff=None, max_hops=1))
    two = Matcher(chain, params=ScoringParams(bridge_cutoff=None, max_hops=2))
    assert [g.skill for g in one.match(["C"], ["A"]).bridged_skills] == []
    assert [g.skill for g in two.match(["C"], ["A"]).bridged_skills] == ["C"]


def test_bridging_disabled_arm(chain):
    """B4.2: direct-match-only must score strictly lower here."""
    on = Matcher(chain).match(["A", "B"], ["A"])
    off = Matcher(chain, params=ScoringParams(enable_bridging=False)).match(["A", "B"], ["A"])
    assert off.bridge_score == 0.0
    assert off.total < on.total
    assert [g.bridgeable for g in off.gaps] == [False]


def test_via_reports_the_nearest_held_skill(chain):
    # A->C costs 0.2, D->C costs 0.5, so A is the bridge even though D is closer in hops.
    r = Matcher(chain, bridge_cutoff=1.0).match(["C"], ["A", "D"])
    assert r.bridged_skills[0].via == "A"
    assert r.bridged_skills[0].distance == pytest.approx(0.2)


# ----------------------------------------------------------------------- scoring


def test_perfect_match_scores_one(chain):
    assert Matcher(chain).match(["A", "B"], ["A", "B"]).total == 1.0


def test_empty_jd_is_zero_not_a_crash(chain):
    assert Matcher(chain).match([], ["A"]).total == 0.0


def test_score_components_reconstruct_the_total(chain):
    r = Matcher(chain).match(["A", "C", "E"], ["A"])
    expected = (r.direct_match_score + r.bridge_score - r.gap_penalty) / r.total_demand
    assert r.total == pytest.approx(round(expected, 3))


def test_bridge_credit_is_one_minus_distance(chain):
    r = Matcher(chain).match(["C"], ["A"])
    assert r.bridge_score == pytest.approx(0.8)   # 1 - 0.2


# ------------------------------------------------------------------------ weights


def test_lists_reproduce_uniform_weight_behaviour(chain):
    m = Matcher(chain)
    assert m.match(["A", "C"], ["A"]).total == m.match({"A": 1.0, "C": 1.0}, {"A": 1.0}).total


def test_jd_demand_weights_shift_the_score(chain):
    m = Matcher(chain)
    critical_a = m.match({"A": 1.5, "C": 0.5}, {"A": 1.0})
    critical_c = m.match({"A": 0.5, "C": 1.5}, {"A": 1.0})
    # A is held outright; weighting it higher must score better than weighting
    # the merely-bridgeable C higher.
    assert critical_a.total > critical_c.total


def test_low_proficiency_earns_partial_credit(chain):
    m = Matcher(chain)
    strong = m.match({"A": 1.0}, {"A": 1.0})
    weak = m.match({"A": 1.0}, {"A": 0.5})
    assert weak.total == pytest.approx(0.5)
    assert strong.total == 1.0


def test_proficiency_above_reference_is_capped(chain):
    r = Matcher(chain).match({"A": 1.0}, {"A": 1.5})
    assert r.total == 1.0  # no runaway score from one over-weighted skill


def test_uniform_weight_ablation_arm(chain):
    """B4.1: turning weights off must ignore proficiency entirely."""
    p = ScoringParams(use_weights=False)
    m = Matcher(chain, params=p)
    assert m.match({"A": 1.0}, {"A": 0.5}).total == m.match({"A": 1.0}, {"A": 1.5}).total == 1.0


def test_unreachable_penalty_can_be_turned_on(chain):
    base = Matcher(chain).match(["E"], ["A"])
    harsh = Matcher(chain, params=ScoringParams(unreachable_penalty=1.0)).match(["E"], ["A"])
    assert base.total == 0.0
    assert harsh.total == -1.0  # explicit penalty, not silently clipped


# ------------------------------------------------------------- legacy compatibility


def test_legacy_dict_access_still_works(chain):
    r = Matcher(chain).match(["A", "C"], ["A"])
    assert r["score"] == r.total
    assert r["matched"] == ["A"]
    g = r["gaps"][0]
    assert g["skill"] == "C" and g["bridgeable"] is True
    assert isinstance(g["distance"], float) and g["via"] == "A"


def test_rank_sets_name_and_orders_by_score(chain):
    ranked = Matcher(chain).rank(["A", "B"], {
        "strong": ["A", "B"],
        "weak": ["D"],
        "middling": ["A"],
    })
    assert [r.name for r in ranked] == ["strong", "middling", "weak"]
    assert ranked[0]["name"] == "strong"


def test_rank_is_deterministic_on_ties(chain):
    ranked = Matcher(chain).rank(["A"], {"zeta": ["A"], "alpha": ["A"]})
    assert [r.name for r in ranked] == ["alpha", "zeta"]


def test_top_k_truncates(chain):
    ranked = Matcher(chain).rank(["A"], {"a": ["A"], "b": ["B"], "c": ["E"]}, top_k=2)
    assert len(ranked) == 2


def test_get_bridgeable_gaps_signature(chain):
    out = Matcher(chain, bridge_cutoff=None).get_bridgeable_gaps(["A"], ["C", "E"], max_hops=2)
    assert [g["skill"] for g in out["bridgeable"]] == ["C"]
    assert [g["skill"] for g in out["gaps"]] == ["E"]