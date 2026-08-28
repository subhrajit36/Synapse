"""Tests for Work Item 2: Typed directed edges."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import networkx as nx
import pytest

from synapse.graph.typed_edges import (
    TypedEdge,
    ClassificationResult,
    TYPE_COST,
    EDGE_TYPES,
    get_candidate_pairs,
    filter_contaminated_pairs,
    load_cache,
    save_to_cache,
    run_sanity_checks,
    build_typed_graph,
)


def test_edge_types_constant():
    assert EDGE_TYPES == ("substitute", "complement", "prerequisite", "unrelated")


def test_type_cost_semantics():
    """Traversal costs follow the spec in §2.5."""
    assert TYPE_COST["substitute"] == 0.2
    assert TYPE_COST["prerequisite"] == 0.6
    assert TYPE_COST["complement"] is None
    assert TYPE_COST["unrelated"] is None


def test_typed_edge_weight_calculation():
    """Weight = TYPE_COST * (2.0 - confidence)."""
    # substitute at confidence 1.0 -> 0.2 * 1.0 = 0.2
    e = TypedEdge(
        a="A", b="B",
        edge_type="substitute",
        direction="symmetric",
        confidence=1.0,
        source="llm",
        rationale="test",
    )
    assert e.weight == pytest.approx(0.2)

    # substitute at confidence 0.5 -> 0.2 * 1.5 = 0.3
    e2 = TypedEdge(
        a="A", b="B",
        edge_type="substitute",
        direction="symmetric",
        confidence=0.5,
        source="llm",
        rationale="test",
    )
    assert e2.weight == pytest.approx(0.3)

    # prerequisite at confidence 1.0 -> 0.6 * 1.0 = 0.6
    e3 = TypedEdge(
        a="A", b="B",
        edge_type="prerequisite",
        direction="a_to_b",
        confidence=1.0,
        source="llm",
        rationale="test",
    )
    assert e3.weight == pytest.approx(0.6)

    # complement is non-traversable (inf weight)
    e4 = TypedEdge(
        a="A", b="B",
        edge_type="complement",
        direction="symmetric",
        confidence=1.0,
        source="llm",
        rationale="test",
    )
    assert e4.weight == float("inf")
    assert not e4.is_traversable

    # unrelated is non-traversable
    e5 = TypedEdge(
        a="A", b="B",
        edge_type="unrelated",
        direction="symmetric",
        confidence=1.0,
        source="llm",
        rationale="test",
    )
    assert e5.weight == float("inf")
    assert not e5.is_traversable


def test_typed_edge_is_directed():
    """Only prerequisite with non-symmetric direction is directed."""
    assert TypedEdge(a="A", b="B", edge_type="substitute", direction="symmetric",
                     confidence=1.0, source="llm", rationale="").is_directed is False
    assert TypedEdge(a="A", b="B", edge_type="complement", direction="symmetric",
                     confidence=1.0, source="llm", rationale="").is_directed is False
    assert TypedEdge(a="A", b="B", edge_type="prerequisite", direction="symmetric",
                     confidence=1.0, source="llm", rationale="").is_directed is False
    assert TypedEdge(a="A", b="B", edge_type="prerequisite", direction="a_to_b",
                     confidence=1.0, source="llm", rationale="").is_directed is True
    assert TypedEdge(a="A", b="B", edge_type="prerequisite", direction="b_to_a",
                     confidence=1.0, source="llm", rationale="").is_directed is True


def test_candidate_pairs_generation():
    """Test candidate pair generation on a synthetic graph."""
    G = nx.Graph()
    # Add skills with categories
    skills = [
        ("Docker", "Container"),
        ("Kubernetes", "Container"),
        ("PyTorch", "ML"),
        ("TensorFlow", "ML"),
        ("Git", "VCS"),
        ("GitHub", "VCS"),
    ]
    for skill, cat in skills:
        G.add_node(skill, node_type="skill", category=cat)

    pairs = get_candidate_pairs(G, k_cross_category=2, min_similarity=0.1)
    # Within-category: Container: C(2,2)=1, ML: C(2,2)=1, VCS: C(2,2)=1 => 3 pairs
    # Cross-category: up to 2 per skill, but filtered by min_similarity
    assert len(pairs) >= 3  # at least within-category pairs
    assert ("Docker", "Kubernetes") in pairs or ("Kubernetes", "Docker") in pairs


def test_filter_contaminated_pairs():
    """Curated substitution pairs from eval dataset are removed."""
    # These are from SUBSTITUTION_GROUPS in dataset.py
    pairs = [
        ("Docker", "Kubernetes"),       # in substitution groups -> removed
        ("PyTorch", "TensorFlow"),      # in substitution groups -> removed
        ("Git", "GitHub"),              # NOT in substitution groups -> kept
        ("React", "Vue.js"),            # in substitution groups -> removed
        ("FastAPI", "Django"),          # NOT in substitution groups -> kept
    ]
    filtered = filter_contaminated_pairs(pairs)
    assert ("Git", "GitHub") in filtered
    assert ("FastAPI", "Django") in filtered
    assert ("Docker", "Kubernetes") not in filtered
    assert ("PyTorch", "TensorFlow") not in filtered
    assert ("React", "Vue.js") not in filtered


def test_cache_roundtrip():
    """Cache save and load preserves data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache.jsonl"

        results = [
            ClassificationResult(
                a="A", b="B",
                edge_type="substitute",
                direction="symmetric",
                confidence=0.9,
                rationale="test rationale",
            ),
            ClassificationResult(
                a="C", b="D",
                edge_type="prerequisite",
                direction="a_to_b",
                confidence=0.8,
                rationale="another rationale",
            ),
        ]

        save_to_cache(cache_path, results)
        loaded = load_cache(cache_path)

        assert len(loaded) == 2
        assert ("A", "B") in loaded
        assert loaded[("A", "B")].edge_type == "substitute"
        assert loaded[("A", "B")].confidence == 0.9
        assert loaded[("C", "D")].edge_type == "prerequisite"
        assert loaded[("C", "D")].direction == "a_to_b"


def test_sanity_checks_type_distribution():
    """Sanity checks report type distribution correctly."""
    edges = [
        TypedEdge(a="A", b="B", edge_type="substitute", direction="symmetric",
                  confidence=0.9, source="llm", rationale=""),
        TypedEdge(a="C", b="D", edge_type="substitute", direction="symmetric",
                  confidence=0.8, source="llm", rationale=""),
        TypedEdge(a="E", b="F", edge_type="complement", direction="symmetric",
                  confidence=0.7, source="llm", rationale=""),
        TypedEdge(a="G", b="H", edge_type="prerequisite", direction="a_to_b",
                  confidence=0.6, source="llm", rationale=""),
        TypedEdge(a="I", b="J", edge_type="unrelated", direction="symmetric",
                  confidence=0.5, source="llm", rationale=""),
    ]

    G = nx.Graph()
    for e in edges:
        G.add_node(e.a, node_type="skill")
        G.add_node(e.b, node_type="skill")

    report = run_sanity_checks(edges, G)

    dist = report["type_distribution"]
    assert dist["substitute"]["count"] == 2
    assert dist["substitute"]["pct"] == 40.0
    assert dist["complement"]["count"] == 1
    assert dist["prerequisite"]["count"] == 1
    assert dist["unrelated"]["count"] == 1


def test_sanity_checks_directed_consistency():
    """Sanity checks detect mutual prerequisite violations."""
    edges = [
        TypedEdge(a="A", b="B", edge_type="prerequisite", direction="a_to_b",
                  confidence=0.9, source="llm", rationale=""),
        TypedEdge(a="B", b="A", edge_type="prerequisite", direction="b_to_a",  # mutual!
                  confidence=0.8, source="llm", rationale=""),
    ]

    G = nx.Graph()
    for e in edges:
        G.add_node(e.a, node_type="skill")
        G.add_node(e.b, node_type="skill")

    report = run_sanity_checks(edges, G)
    assert report["directed_consistency"]["mutual_prerequisite_violations"] == 1
    assert len(report["directed_consistency"]["details"]) == 1


def test_sanity_checks_probe_checks():
    """Sanity checks validate known-good probe pairs."""
    # Updated expectations per system instruction:
    # - complement = coherent stack (Docker/Kubernetes, Git/GitHub)
    # - substitute = truly interchangeable (PyTorch/TensorFlow, MySQL/PostgreSQL)
    edges = [
        TypedEdge(a="Docker", b="Kubernetes", edge_type="complement", direction="symmetric",
                  confidence=0.9, source="llm", rationale=""),
        TypedEdge(a="PyTorch", b="TensorFlow", edge_type="substitute", direction="symmetric",
                  confidence=0.9, source="llm", rationale=""),
        TypedEdge(a="MySQL", b="PostgreSQL", edge_type="substitute", direction="symmetric",
                  confidence=0.9, source="llm", rationale=""),
        TypedEdge(a="Git", b="GitHub", edge_type="complement", direction="symmetric",
                  confidence=0.9, source="llm", rationale=""),
    ]

    G = nx.Graph()
    for e in edges:
        G.add_node(e.a, node_type="skill")
        G.add_node(e.b, node_type="skill")

    report = run_sanity_checks(edges, G)

    probe_results = {p["pair"]: p for p in report["probe_checks"]}
    assert probe_results["Docker↔Kubernetes"]["pass"] is True
    assert probe_results["PyTorch↔TensorFlow"]["pass"] is True
    assert probe_results["MySQL↔PostgreSQL"]["pass"] is True
    assert probe_results["Git↔GitHub"]["pass"] is True
    assert report.get("critical_warning") is None


def test_build_typed_graph():
    """Typed edges are added to graph with correct attributes."""
    G = nx.Graph()
    G.add_node("Docker", node_type="skill")
    G.add_node("Kubernetes", node_type="skill")
    G.add_node("Python", node_type="skill")
    G.add_node("FastAPI", node_type="skill")

    edges = [
        TypedEdge(a="Docker", b="Kubernetes", edge_type="substitute", direction="symmetric",
                  confidence=0.9, source="llm", rationale="container orchestration"),
        TypedEdge(a="Python", b="FastAPI", edge_type="prerequisite", direction="a_to_b",
                  confidence=0.8, source="llm", rationale="FastAPI requires Python"),
        TypedEdge(a="Docker", b="Python", edge_type="complement", direction="symmetric",
                  confidence=0.7, source="llm", rationale="commonly used together"),
        TypedEdge(a="Docker", b="FastAPI", edge_type="unrelated", direction="symmetric",
                  confidence=0.5, source="llm", rationale="no direct relationship"),
    ]

    H = build_typed_graph(G, edges, traversable_types={"substitute", "prerequisite"})

    # substitute edge added and traversable
    assert H.has_edge("Docker", "Kubernetes")
    assert H["Docker"]["Kubernetes"]["edge_type"] == "substitute"
    assert H["Docker"]["Kubernetes"]["traversable"] is True
    assert H["Docker"]["Kubernetes"]["weight"] == pytest.approx(0.22)  # 0.2 * (2.0 - 0.9) = 0.22

    # prerequisite edge added directed and traversable
    assert H.has_edge("Python", "FastAPI")
    assert not H.has_edge("FastAPI", "Python")  # directed
    assert H["Python"]["FastAPI"]["edge_type"] == "prerequisite"
    assert H["Python"]["FastAPI"]["direction"] == "a_to_b"
    assert H["Python"]["FastAPI"]["traversable"] is True
    assert H["Python"]["FastAPI"]["weight"] == pytest.approx(0.72)  # 0.6 * (2.0 - 0.8) = 0.72

    # complement edge added but NOT traversable
    assert H.has_edge("Docker", "Python")
    assert H["Docker"]["Python"]["edge_type"] == "complement"
    assert H["Docker"]["Python"]["traversable"] is False

    # unrelated edge NOT added
    assert not H.has_edge("Docker", "FastAPI")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])