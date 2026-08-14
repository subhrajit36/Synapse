"""Phase A2 tests. No torch, no network."""

from __future__ import annotations

import json

import numpy as np
import pytest

from synapse.matching.aliases import build_surface_index, core_form, normalize, validate_alias_table
from synapse.matching.entity_linker import (
    METHOD_ALIAS,
    METHOD_EMBEDDING,
    METHOD_SURFACE,
    METHOD_UNRESOLVED,
    EntityLinker,
)

NODES = [
    "Docker",
    "Kubernetes",
    "Amazon Web Services AWS software",
    "Structured query language SQL",
    "Apache Kafka",
    "Microsoft SQL Server",
]


class FakeEmbedder:
    """Deterministic toy embedder: vector = normalized char-bigram histogram."""

    DIM = 64

    def encode(self, texts):
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            t = t.lower()
            for a, b in zip(t, t[1:]):
                out[i, (ord(a) * 31 + ord(b)) % self.DIM] += 1.0
        return out


# ----------------------------------------------------------------- normalization


def test_core_form_strips_vendor_and_suffix():
    assert core_form("Microsoft SQL Server") == "sql server"
    assert core_form("Amazon Web Services AWS software") == "web services aws"


def test_normalize_preserves_meaningful_symbols():
    assert normalize("  C++ / C#  ") == "c++ c#"
    assert normalize("Node.js") == "node.js"


def test_surface_index_drops_ambiguous_surfaces():
    # Both nodes reduce to a core containing 'sql'; nothing may silently win.
    index = build_surface_index(["Structured query language SQL", "SQL"])
    assert index.get("sql") in (None, "SQL")  # never the long O*NET string by accident


def test_validate_alias_table_reports_dead_targets():
    report = validate_alias_table(["Docker"], {"k8s": "Kubernetes"})
    assert report["missing_targets"] == ["k8s"]


# ---------------------------------------------------------------------- cascade


def test_alias_beats_embedding():
    linker = EntityLinker(NODES, embedder=FakeEmbedder())
    r = linker.link("k8s")
    assert r.node == "Kubernetes"
    assert r.method == METHOD_ALIAS
    assert r.score == 1.0


def test_exact_surface_match_is_deterministic():
    linker = EntityLinker(NODES, embedder=FakeEmbedder())
    r = linker.link("  docker ")
    assert (r.node, r.method) == ("Docker", METHOD_SURFACE)


def test_embedded_acronym_resolves():
    linker = EntityLinker(NODES, embedder=FakeEmbedder())
    assert linker.link("aws").node == "Amazon Web Services AWS software"


def test_below_threshold_is_unresolved_not_forced():
    linker = EntityLinker(NODES, embedder=FakeEmbedder(), min_score=0.99)
    r = linker.link("stakeholder management")
    assert r.node is None
    assert r.method == METHOD_UNRESOLVED
    assert r.score < 0.99  # near-miss score still recorded for inspection


def test_above_threshold_uses_embedding_path():
    linker = EntityLinker(NODES, embedder=FakeEmbedder(), min_score=0.0)
    r = linker.link("some unseen phrase")
    assert r.method == METHOD_EMBEDDING and r.node is not None


def test_embeddings_can_be_disabled():
    linker = EntityLinker(NODES, use_embeddings=False)
    assert linker.link("docker").node == "Docker"
    assert linker.link("anything else").node is None


# ----------------------------------------------------------------------- weights


def test_weights_survive_linking():
    linker = EntityLinker(NODES, embedder=FakeEmbedder(), min_score=0.99)
    profile = linker.link_many([("Docker", 1.5), ("k8s", 0.5)])
    assert profile.skills == {"Docker": 1.5, "Kubernetes": 0.5}


def test_duplicate_targets_keep_highest_weight():
    linker = EntityLinker(NODES, embedder=FakeEmbedder(), min_score=0.99)
    profile = linker.link_many([("k8s", 0.5), ("kubernetes", 1.5)])
    assert profile.skills == {"Kubernetes": 1.5}


def test_resolution_rate_and_unresolved():
    linker = EntityLinker(NODES, embedder=FakeEmbedder(), min_score=0.99)
    profile = linker.link_many(["Docker", "quantum basket weaving"])
    assert len(profile.unresolved) == 1
    assert profile.resolution_rate == 0.5


def test_legacy_extract_still_returns_node_names():
    linker = EntityLinker(NODES, embedder=FakeEmbedder(), min_score=0.99)
    assert linker.extract(["Docker", "k8s"]) == ["Docker", "Kubernetes"]


def test_link_extraction_consumes_a1_output():
    from synapse.ingest.schemas import ExtractedSkill, ExtractionResult

    extraction = ExtractionResult(
        source_id="cand_01",
        skills=[
            ExtractedSkill(skill="Docker", weight=1.5, context="ran Docker"),
            ExtractedSkill(skill="K8s", weight=1.0, context="k8s clusters"),
        ],
    )
    linker = EntityLinker(NODES, embedder=FakeEmbedder(), min_score=0.99)
    profile = linker.link_extraction(extraction)
    assert profile.source_id == "cand_01"
    assert profile.skills == {"Docker": 1.5, "Kubernetes": 1.0}


# ----------------------------------------------------------------------- logging


def test_unresolved_are_logged_to_jsonl(tmp_path):
    log = tmp_path / "eval" / "unresolved.jsonl"
    linker = EntityLinker(
        NODES, embedder=FakeEmbedder(), min_score=0.99, unresolved_log=log
    )
    linker.link_many(["definitely not a skill"], source_id="jd_07")
    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["source_id"] == "jd_07"
    assert record["surface"] == "definitely not a skill"
    assert "best_score" in record


def test_no_log_file_written_when_all_resolved(tmp_path):
    log = tmp_path / "unresolved.jsonl"
    linker = EntityLinker(NODES, embedder=FakeEmbedder(), unresolved_log=log)
    linker.link_many(["Docker"])
    assert not log.exists()