"""Phase A2: canonicalize extracted skill strings onto graph nodes.

Resolution cascade, most deterministic first:

    1. alias table      - hand-maintained overrides ('k8s' -> 'Kubernetes')
    2. surface index    - exact / vendor-stripped / embedded-acronym match
    3. embedding        - cosine similarity, accepted only above `min_score`
    4. unresolved       - logged, never force-linked

Two deliberate changes from the previous version:

  * The embedder is injected rather than hard-constructed. Swapping
    sentence-transformers for FastEmbed in Phase C1 becomes a constructor
    argument, and these tests run without torch installed.
  * `link_many` carries proficiency weights through from extraction. The old
    `extract()` dropped them, which made A4's weighted scoring impossible.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np

from .aliases import active_alias_table, build_surface_index, normalize, validate_alias_table

logger = logging.getLogger(__name__)

# NOTE: this default is a starting point, NOT a validated value. Canonical node
# names are long O*NET strings, so a short surface like "aws" scores far lower
# against "Amazon Web Services AWS software" than the 0.82 the plan assumes.
# Run scripts/calibrate_link_threshold.py and set this from the data.
DEFAULT_MIN_SCORE = 0.60

METHOD_ALIAS = "alias"
METHOD_SURFACE = "surface"
METHOD_EMBEDDING = "embedding"
METHOD_UNRESOLVED = "unresolved"


class Embedder(Protocol):
    """Minimal contract satisfied by both SentenceTransformer and FastEmbed wrappers."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Dev-phase embedder. Phase C1 replaces this with a FastEmbed equivalent."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts)), dtype=np.float32)


@dataclass(frozen=True)
class LinkResult:
    surface: str
    node: str | None
    score: float
    method: str
    weight: float = 1.0

    @property
    def resolved(self) -> bool:
        return self.node is not None


@dataclass
class LinkedProfile:
    """Canonicalized skills for one document, with full linking provenance."""

    source_id: str = ""
    skills: dict[str, float] = field(default_factory=dict)  # node -> weight
    results: list[LinkResult] = field(default_factory=list)

    @property
    def nodes(self) -> list[str]:
        return sorted(self.skills)

    @property
    def unresolved(self) -> list[LinkResult]:
        return [r for r in self.results if not r.resolved]

    @property
    def resolution_rate(self) -> float:
        return (len(self.skills) / len(self.results)) if self.results else 0.0


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class EntityLinker:
    """Map free-text skill phrases onto canonical graph skill nodes."""

    def __init__(
        self,
        skill_names: Iterable[str],
        embedder: Embedder | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        min_score: float = DEFAULT_MIN_SCORE,
        node_texts: dict[str, str] | None = None,
        alias_table: dict[str, str] | None = None,
        unresolved_log: str | Path | None = None,
        use_embeddings: bool = True,
    ) -> None:
        self.skills = list(skill_names)
        self.min_score = min_score
        self.unresolved_log = Path(unresolved_log) if unresolved_log else None

        self.alias_index = active_alias_table(self.skills, alias_table)
        self.surface_index = build_surface_index(self.skills)
        self.alias_report = validate_alias_table(self.skills, alias_table)

        self._embedder = embedder
        self._model_name = model_name
        self._use_embeddings = use_embeddings
        self._node_emb: np.ndarray | None = None
        # Context-enriched node text (e.g. "Docker (Development environment
        # software)") sharpens the embedding, mirroring build_graph.py.
        self._node_texts = [
            (node_texts or {}).get(name, name) for name in self.skills
        ]

    # ------------------------------------------------------------- embeddings

    def _ensure_embeddings(self) -> bool:
        """Lazily build the node embedding matrix. False if unavailable."""
        if not self._use_embeddings:
            return False
        if self._node_emb is not None:
            return True
        if self._embedder is None:
            try:
                self._embedder = SentenceTransformerEmbedder(self._model_name)
            except ImportError:
                logger.warning(
                    "No embedder available; linking is alias/surface-only. "
                    "Unmatched surfaces will be reported as unresolved."
                )
                self._use_embeddings = False
                return False
        self._node_emb = _l2_normalize(self._embedder.encode(self._node_texts))
        return True

    # ------------------------------------------------------------------ linking

    def link(self, phrase: str, weight: float = 1.0) -> LinkResult:
        """Resolve one surface form. Never guesses below the threshold."""
        surface = normalize(phrase)
        if not surface:
            return LinkResult(phrase, None, 0.0, METHOD_UNRESOLVED, weight)

        if surface in self.alias_index:
            return LinkResult(phrase, self.alias_index[surface], 1.0, METHOD_ALIAS, weight)

        if surface in self.surface_index:
            return LinkResult(phrase, self.surface_index[surface], 1.0, METHOD_SURFACE, weight)

        if not self._ensure_embeddings():
            return LinkResult(phrase, None, 0.0, METHOD_UNRESOLVED, weight)

        query = _l2_normalize(self._embedder.encode([phrase]))  # type: ignore[union-attr]
        sims = (self._node_emb @ query[0])  # type: ignore[operator]
        best = int(np.argmax(sims))
        score = float(sims[best])

        if score < self.min_score:
            # Record the near-miss so the score is inspectable later; the node
            # is still withheld.
            return LinkResult(phrase, None, score, METHOD_UNRESOLVED, weight)
        return LinkResult(phrase, self.skills[best], score, METHOD_EMBEDDING, weight)

    def link_many(
        self,
        phrases: Iterable[str] | Iterable[tuple[str, float]],
        source_id: str = "",
    ) -> LinkedProfile:
        """Link a batch, preserving weights and collapsing duplicate targets.

        Accepts bare strings or (phrase, weight) pairs. When two surfaces land on
        the same node the higher weight wins, matching merge_skills' rule that the
        strongest evidence in a document is the best proficiency estimate.
        """
        profile = LinkedProfile(source_id=source_id)
        for item in phrases:
            phrase, weight = item if isinstance(item, tuple) else (item, 1.0)
            result = self.link(phrase, weight)
            profile.results.append(result)
            if result.node is not None:
                current = profile.skills.get(result.node)
                if current is None or result.weight > current:
                    profile.skills[result.node] = result.weight

        self._log_unresolved(profile)
        return profile

    def link_extraction(self, extraction) -> LinkedProfile:
        """Bridge from Phase A1's ExtractionResult straight into linking."""
        return self.link_many(
            [(s.skill, s.weight) for s in extraction.skills],
            source_id=extraction.source_id,
        )

    # ------------------------------------------------------------------ logging

    def _log_unresolved(self, profile: LinkedProfile) -> None:
        """Append unresolved surfaces to JSONL for ontology-expansion decisions.

        Per CLAUDE.md A2.5 these are recorded only; auto-MERGE into the graph is
        deferred to C2.5, where it gets a deduplication check first.
        """
        if self.unresolved_log is None or not profile.unresolved:
            return
        self.unresolved_log.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat()
        with self.unresolved_log.open("a", encoding="utf-8") as fh:
            for r in profile.unresolved:
                fh.write(json.dumps({
                    "timestamp": stamp,
                    "source_id": profile.source_id,
                    "surface": r.surface,
                    "best_score": round(r.score, 4),
                    "min_score": self.min_score,
                }) + "\n")

    # -------------------------------------------------------- backwards compat

    def extract(self, phrases: Iterable[str]) -> list[str]:
        """Legacy API: resolved node names only. Prefer link_many for weights."""
        return self.link_many(phrases).nodes