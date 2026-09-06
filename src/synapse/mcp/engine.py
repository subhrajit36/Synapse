"""Phase C4 support: one loaded scoring engine plus the tools' typed contracts.

`server.py` is deliberately a set of one-line delegations to this module, per
C4.4: the MCP layer must not hold business logic, so everything testable lives
here and is callable without a running server.

The engine owns exactly three things - the graph, the matcher and the entity
linker - and it constructs them the same way `app.py` does, on purpose. A
serving surface that links or scores differently from the one Phase B measured
is a surface whose numbers mean nothing (C6.4).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel, Field

from ..matching.entity_linker import (
    DEFAULT_MIN_SCORE,
    METHOD_SURFACE,
    METHOD_UNRESOLVED,
    EntityLinker,
    LinkedProfile,
    LinkResult,
)
from ..matching.matcher import TUNED_PARAMS, Gap, Matcher, MatchResult, ScoringParams

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_PATH = os.getenv("SYNAPSE_GRAPH_PATH", "data/skill_graph.pkl")

# Where the scoring graph comes from. AuraDB is the system of record (NFR4);
# the pickle exists for tests and for offline development, and is NOT a
# production fallback - a service that silently serves a stale local graph
# violates NFR4 while looking healthy, so `neo4j` fails loudly instead.
#   neo4j  - load from AuraDB, fail startup if unavailable
#   pickle - load from data/skill_graph.pkl, never touch the network
GRAPH_SOURCE_NEO4J = "neo4j"
GRAPH_SOURCE_PICKLE = "pickle"
DEFAULT_GRAPH_SOURCE = os.getenv("SYNAPSE_GRAPH_SOURCE", GRAPH_SOURCE_PICKLE)

# Shape the deployed graph must have, asserted at load. These are the counts the
# Phase B evaluation ran against; a mismatch means the deployed artifact is not
# the evaluated one, which is exactly what C6.4 exists to catch.
EXPECTED_SKILLS = int(os.getenv("SYNAPSE_EXPECTED_SKILLS", "213"))
EXPECTED_SIMILAR_PAIRS = int(os.getenv("SYNAPSE_EXPECTED_PAIRS", "15459"))

# Phase D reads its JDs from the versioned eval snapshot rather than an ad-hoc
# demo fixture, so the page shows the same pairs the reported metrics came from.
DEFAULT_EVAL_DATASET = os.getenv("SYNAPSE_EVAL_DATASET", "data/eval/v2/dataset.json")

# Phase C1's production embedder. FastEmbed builds it lazily on the first
# surface that reaches the embedding fallback, which keeps a cold start off the
# 512MB ceiling until something actually needs it (NFR1).
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


# ------------------------------------------------------------------ contracts


class SkillWeight(BaseModel):
    """One skill and how much it counts.

    On a JD the weight is demand ("how badly this role needs it"); on a
    candidate it is proficiency. The matcher treats them symmetrically, so one
    model serves both.
    """

    skill: str = Field(..., min_length=1, description="Skill as written, e.g. 'K8s'.")
    weight: float = Field(
        1.0, ge=0.0, le=10.0,
        description="Demand (JD) or proficiency (candidate). 1.0 = ordinary.",
    )


class CandidateInput(BaseModel):
    name: str = Field(..., min_length=1, description="Identifier echoed back in results.")
    skills: list[SkillWeight] = Field(default_factory=list)


class LinkedSkill(BaseModel):
    """How one input surface was canonicalized. NFR6: linking is never opaque."""

    surface: str
    node: str | None = Field(None, description="Canonical graph node, null if unresolved.")
    score: float
    method: str = Field(..., description="alias | surface | embedding | unresolved")
    weight: float


class GapInfo(BaseModel):
    skill: str
    via: str | None = Field(None, description="Nearest held skill the path starts from.")
    distance: float = Field(
        ..., description="Weighted path distance; -1 when no path exists."
    )
    hops: int | None = None
    bridgeable: bool
    demand: float
    reason: str = Field(..., description="Why this landed where it did, e.g. 'no_path'.")


class CandidateScore(BaseModel):
    """FR5's explainable score object, one per ranked candidate."""

    name: str
    total: float
    direct_match_score: float
    bridge_score: float
    gap_penalty: float
    total_demand: float
    matched_skills: list[str]
    bridged_skills: list[GapInfo]
    missing_skills: list[GapInfo]
    unresolved_skills: list[str] = Field(
        default_factory=list,
        description="Candidate surfaces that reached no graph node and so scored nothing.",
    )


class RankingResponse(BaseModel):
    jd_skills: list[str] = Field(..., description="Canonical JD skills actually scored.")
    jd_unresolved: list[str] = Field(
        default_factory=list,
        description="JD surfaces that reached no node. These are excluded from the "
                    "demand denominator, so a long list means the score is over a "
                    "narrower job than the caller asked about.",
    )
    params: dict = Field(..., description="Scoring parameters this ranking used.")
    candidates: list[CandidateScore]


class GapResponse(BaseModel):
    """FR4."""

    jd_skills: list[str]
    candidate_skills: list[str]
    jd_unresolved: list[str] = Field(default_factory=list)
    candidate_unresolved: list[str] = Field(default_factory=list)
    bridgeable: list[GapInfo]
    gaps: list[GapInfo]


class ExplainResponse(BaseModel):
    score: CandidateScore
    explanation: str = Field(..., description="Human-readable component breakdown.")
    jd_linking: list[LinkedSkill]
    candidate_linking: list[LinkedSkill]


class JDSummary(BaseModel):
    """One row in the Phase D JD selector."""

    jd_id: str
    domain: str
    split: str = Field(..., description="'train' or 'heldout'.")
    n_skills: int
    n_candidates: int


class JDListResponse(BaseModel):
    dataset: str
    version: str
    jds: list[JDSummary]


class EvalCandidateScore(CandidateScore):
    """A scored candidate plus the label the eval set assigns it.

    The ground-truth tier travels with the score so the page can show the
    ranking against the labels rather than asking the reader to take the order
    on trust.
    """

    tier: str = Field(..., description="strong | bridgeable | weak | irrelevant")
    grade: int = Field(..., description="3 | 2 | 1 | 0, matching the tier.")
    exact_overlap: int = Field(
        ..., description="Count of JD skills held outright, from the dataset."
    )


class EvalRankResponse(BaseModel):
    jd_id: str
    domain: str
    split: str
    jd_skills: list[SkillWeight]
    params: dict
    candidates: list[EvalCandidateScore]


class RegisteredSkill(BaseModel):
    """Outcome of a C2.5 dynamic MERGE attempt for one unresolved surface."""

    surface: str
    node: str = Field(..., description="Canonical node the surface now maps to.")
    created: bool = Field(..., description="True if a new node was written to Aura.")


class GraphStats(BaseModel):
    graph_source: str = Field(
        ..., description="'neo4j' or 'pickle' - where this graph was actually loaded from."
    )
    graph_path: str
    skill_nodes: int
    total_nodes: int
    total_edges: int
    similar_edges: int
    orphan_skills: int = Field(
        ..., description="Skills with no 'similar' edge; nothing can bridge to them."
    )
    embeddings_loaded: bool
    scoring_params: dict


# ------------------------------------------------------------------ adapters


def _to_gap_info(gap: Gap) -> GapInfo:
    return GapInfo(
        skill=gap.skill,
        via=gap.via,
        # JSON has no infinity. An unreachable skill reports -1, which is
        # unambiguous next to `reason='no_path'` and survives strict parsers.
        distance=-1.0 if gap.distance == float("inf") else round(gap.distance, 4),
        hops=gap.hops,
        bridgeable=gap.bridgeable,
        demand=gap.demand,
        reason=gap.reason,
    )


def _to_linked(profile: LinkedProfile) -> list[LinkedSkill]:
    return [
        LinkedSkill(
            surface=r.surface, node=r.node, score=round(r.score, 4),
            method=r.method, weight=r.weight,
        )
        for r in profile.results
    ]


def _params_dict(params: ScoringParams) -> dict:
    return asdict(params)


def _to_candidate_score(result: MatchResult, unresolved: Iterable[str]) -> CandidateScore:
    return CandidateScore(
        name=result.name,
        total=result.total,
        direct_match_score=round(result.direct_match_score, 4),
        bridge_score=round(result.bridge_score, 4),
        gap_penalty=round(result.gap_penalty, 4),
        total_demand=round(result.total_demand, 4),
        matched_skills=result.matched_skills,
        bridged_skills=[_to_gap_info(g) for g in result.bridged_skills],
        missing_skills=[_to_gap_info(g) for g in result.missing_skills],
        unresolved_skills=list(unresolved),
    )


# --------------------------------------------------------------------- engine


class MatchEngine:
    """Loads the graph once and answers scoring questions against it.

    Construction is lazy: importing this module (or the MCP server) must not
    read a pickle or spin up an embedder, so a Render cold start pays for the
    graph only when the first real request arrives.
    """

    def __init__(
        self,
        graph_path: str | Path = DEFAULT_GRAPH_PATH,
        params: ScoringParams | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
        embed_model: str = DEFAULT_EMBED_MODEL,
        eval_dataset: str | Path = DEFAULT_EVAL_DATASET,
        graph_source: str = DEFAULT_GRAPH_SOURCE,
        neo4j_client=None,
        expected_skills: int | None = EXPECTED_SKILLS,
        expected_pairs: int | None = EXPECTED_SIMILAR_PAIRS,
    ) -> None:
        self.graph_path = Path(graph_path)
        self.params = params or TUNED_PARAMS
        self.min_score = min_score
        self.embed_model = embed_model
        self.eval_dataset = Path(eval_dataset)
        self.graph_source = graph_source
        # None disables the check; the defaults are the shape Phase B ran on.
        self.expected_skills = expected_skills
        self.expected_pairs = expected_pairs
        self._neo4j_client = neo4j_client   # injectable for tests
        self._graph = None
        self._matcher: Matcher | None = None
        self._linker: EntityLinker | None = None
        self._dataset: dict | None = None

    # -- lazy resources ----------------------------------------------------

    @property
    def neo4j(self):
        """The Neo4j client, constructed on demand.

        Constructed here rather than in __init__ so that importing the engine
        never reads credentials or opens a socket.
        """
        if self._neo4j_client is None:
            from ..graph.neo4j_client import Neo4jClient

            self._neo4j_client = Neo4jClient()
        return self._neo4j_client

    @property
    def graph(self):
        if self._graph is None:
            if self.graph_source == GRAPH_SOURCE_NEO4J:
                self._graph = self._load_from_neo4j()
            elif self.graph_source == GRAPH_SOURCE_PICKLE:
                self._graph = self._load_from_pickle()
            else:
                raise ValueError(
                    f"Unknown SYNAPSE_GRAPH_SOURCE {self.graph_source!r}; "
                    f"expected {GRAPH_SOURCE_NEO4J!r} or {GRAPH_SOURCE_PICKLE!r}."
                )
        return self._graph

    def _load_from_pickle(self):
        if not self.graph_path.exists():
            raise FileNotFoundError(
                f"Missing {self.graph_path}. Build it with "
                "`python scripts/build_graph_artifact.py`, or point "
                "SYNAPSE_GRAPH_PATH at an existing artifact."
            )
        logger.info("Loading skill graph from %s", self.graph_path)
        with self.graph_path.open("rb") as fh:
            return pickle.load(fh)

    def _load_from_neo4j(self):
        """Load from AuraDB, failing loudly rather than degrading silently."""
        from ..graph.neo4j_loader import load_graph_from_neo4j

        client = self.neo4j
        if not client.config.is_configured:
            raise RuntimeError(
                "SYNAPSE_GRAPH_SOURCE=neo4j but no NEO4J_PASSWORD is set "
                f"({client.config.describe()}). Set the Neo4j environment "
                "variables, or use SYNAPSE_GRAPH_SOURCE=pickle for offline work."
            )
        logger.info("Loading skill graph from Neo4j (%s)", client.config.describe())
        return load_graph_from_neo4j(
            client,
            expected_skills=self.expected_skills,
            expected_pairs=self.expected_pairs,
        )

    def invalidate_graph(self) -> None:
        """Drop the cached graph so the next access reloads it.

        Needed by the C2.5 dynamic-MERGE path: a skill added to Aura after
        startup is invisible to the in-process graph until it is rebuilt.
        """
        self._graph = None
        self._matcher = None
        self._linker = None

    @property
    def skill_names(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "skill"]

    @property
    def matcher(self) -> Matcher:
        if self._matcher is None:
            self._matcher = Matcher(self.graph, params=self.params)
        return self._matcher

    @property
    def linker(self) -> EntityLinker:
        if self._linker is None:
            skills = self.skill_names
            node_texts = {
                n: f"{n} ({self.graph.nodes[n].get('category', '')})" for n in skills
            }
            self._linker = EntityLinker(
                skills,
                node_texts=node_texts,
                model_name=self.embed_model,
                min_score=self.min_score,
                use_embeddings=True,
            )
        return self._linker

    # -- linking -----------------------------------------------------------

    def resolve(
        self,
        skills: Sequence[SkillWeight],
        source_id: str = "",
        link: bool = True,
    ) -> tuple[dict[str, float], LinkedProfile]:
        """Canonicalize input surfaces onto graph nodes.

        `link=False` is the escape hatch for a caller that already holds
        canonical node names (an eval harness, or a second MCP call reusing the
        first's output) and does not want to pay for linking again. Names that
        are not nodes are still reported unresolved rather than assumed valid.
        """
        pairs = [(s.skill, s.weight) for s in skills]
        if link:
            profile = self.linker.link_many(pairs, source_id=source_id)
            return profile.skills, profile

        profile = LinkedProfile(source_id=source_id)
        known = set(self.skill_names)
        for surface, weight in pairs:
            hit = surface in known
            profile.results.append(LinkResult(
                surface, surface if hit else None, 1.0 if hit else 0.0,
                METHOD_SURFACE if hit else METHOD_UNRESOLVED, weight,
            ))
            if hit:
                profile.skills[surface] = max(profile.skills.get(surface, 0.0), weight)
        return profile.skills, profile

    # -- tools -------------------------------------------------------------

    def _params_for(
        self,
        max_hops: int | None = None,
        use_weights: bool | None = None,
        enable_bridging: bool | None = None,
    ) -> ScoringParams:
        """Overlay per-request overrides on the tuned defaults.

        Only the three knobs Phase B4 actually studied are exposed. Leaving the
        rest fixed keeps every served score comparable to the evaluated ones.
        """
        overrides = {
            k: v for k, v in {
                "max_hops": max_hops,
                "use_weights": use_weights,
                "enable_bridging": enable_bridging,
            }.items() if v is not None
        }
        return replace(self.params, **overrides) if overrides else self.params

    def rank_candidates(
        self,
        jd_skills: Sequence[SkillWeight],
        candidates: Sequence[CandidateInput],
        top_k: int | None = None,
        max_hops: int | None = None,
        use_weights: bool | None = None,
        enable_bridging: bool | None = None,
        link: bool = True,
    ) -> RankingResponse:
        """FR5: rank a candidate pool against a JD with explainable components."""
        params = self._params_for(max_hops, use_weights, enable_bridging)
        jd_map, jd_profile = self.resolve(jd_skills, "jd", link)

        resolved: dict[str, dict[str, float]] = {}
        unresolved: dict[str, list[str]] = {}
        for candidate in candidates:
            skills, profile = self.resolve(candidate.skills, candidate.name, link)
            resolved[candidate.name] = skills
            unresolved[candidate.name] = [r.surface for r in profile.unresolved]

        ranked = self.matcher.rank(jd_map, resolved, params=params, top_k=top_k)

        return RankingResponse(
            jd_skills=sorted(jd_map),
            jd_unresolved=[r.surface for r in jd_profile.unresolved],
            params=_params_dict(params),
            candidates=[
                _to_candidate_score(r, unresolved.get(r.name, [])) for r in ranked
            ],
        )

    def get_bridgeable_gaps(
        self,
        candidate_skills: Sequence[SkillWeight],
        jd_skills: Sequence[SkillWeight],
        max_hops: int | None = None,
        link: bool = True,
    ) -> GapResponse:
        """FR4. Candidate first, matching CLAUDE.md's signature."""
        params = self._params_for(max_hops=max_hops)
        cand_map, cand_profile = self.resolve(candidate_skills, "candidate", link)
        jd_map, jd_profile = self.resolve(jd_skills, "jd", link)

        result = self.matcher.match(jd_map, cand_map, params=params)
        return GapResponse(
            jd_skills=sorted(jd_map),
            candidate_skills=sorted(cand_map),
            jd_unresolved=[r.surface for r in jd_profile.unresolved],
            candidate_unresolved=[r.surface for r in cand_profile.unresolved],
            bridgeable=[_to_gap_info(g) for g in result.bridged_skills],
            gaps=[_to_gap_info(g) for g in result.missing_skills],
        )

    def explain_score(
        self,
        jd_skills: Sequence[SkillWeight],
        candidate_skills: Sequence[SkillWeight],
        candidate_name: str = "candidate",
        max_hops: int | None = None,
        link: bool = True,
    ) -> ExplainResponse:
        """NFR6: the full derivation of one score, linking included."""
        params = self._params_for(max_hops=max_hops)
        jd_map, jd_profile = self.resolve(jd_skills, "jd", link)
        cand_map, cand_profile = self.resolve(candidate_skills, candidate_name, link)

        result = self.matcher.match(jd_map, cand_map, params=params)
        result.name = candidate_name
        return ExplainResponse(
            score=_to_candidate_score(result, [r.surface for r in cand_profile.unresolved]),
            explanation=result.explain(),
            jd_linking=_to_linked(jd_profile),
            candidate_linking=_to_linked(cand_profile),
        )

    # -- Phase D: the eval snapshot behind the page -------------------------

    @property
    def dataset(self) -> dict:
        if self._dataset is None:
            if not self.eval_dataset.exists():
                raise FileNotFoundError(
                    f"Missing eval dataset {self.eval_dataset}. Point "
                    "SYNAPSE_EVAL_DATASET at a versioned snapshot."
                )
            self._dataset = json.loads(self.eval_dataset.read_text(encoding="utf-8"))
        return self._dataset

    def list_eval_jds(self) -> JDListResponse:
        """D1: `GET /api/jds`."""
        data = self.dataset
        return JDListResponse(
            dataset=str(self.eval_dataset),
            version=str(data.get("version", "")),
            jds=[
                JDSummary(
                    jd_id=jd["jd_id"],
                    domain=jd.get("domain", ""),
                    split=jd.get("split", ""),
                    n_skills=len(jd.get("jd_skills", {})),
                    n_candidates=len(jd.get("candidates", [])),
                )
                for jd in data.get("jds", [])
            ],
        )

    def rank_eval_jd(self, jd_id: str, top_k: int | None = None) -> EvalRankResponse:
        """D1: `POST /api/rank`. Full MatchResult per candidate, one round trip.

        Eval-set skills are already canonical node names - they were drawn from
        the graph - so linking is skipped. Running them back through the linker
        would let a linking change silently alter numbers that are supposed to
        be reproducible from the snapshot (NFR7).
        """
        jd = next((j for j in self.dataset.get("jds", []) if j["jd_id"] == jd_id), None)
        if jd is None:
            raise KeyError(jd_id)

        labels = {
            c["cand_id"]: (c.get("tier", ""), c.get("grade", 0), c.get("exact_overlap", 0))
            for c in jd["candidates"]
        }
        candidates = {c["cand_id"]: c["skills"] for c in jd["candidates"]}
        ranked = self.matcher.rank(
            jd["jd_skills"], candidates, params=self.params, top_k=top_k
        )

        scored: list[EvalCandidateScore] = []
        for result in ranked:
            tier, grade, overlap = labels.get(result.name, ("", 0, 0))
            base = _to_candidate_score(result, [])
            scored.append(EvalCandidateScore(
                **base.model_dump(), tier=tier, grade=grade, exact_overlap=overlap
            ))

        return EvalRankResponse(
            jd_id=jd["jd_id"],
            domain=jd.get("domain", ""),
            split=jd.get("split", ""),
            jd_skills=[
                SkillWeight(skill=s, weight=w)
                for s, w in sorted(jd["jd_skills"].items(), key=lambda kv: (-kv[1], kv[0]))
            ],
            params=_params_dict(self.params),
            candidates=scored,
        )

    # -- C2.5: dynamic graph building --------------------------------------

    def register_skills(
        self,
        surfaces: Sequence[str],
        min_similarity: float = 0.82,
    ) -> list[RegisteredSkill]:
        """MERGE unresolved surfaces into Aura, deduplicated first (C2.5).

        Each surface is embedded, vector-searched against existing nodes, and
        only written if nothing at or above `min_similarity` already exists -
        the guard that stops "JS"/"JavaScript"/"Javascript" becoming three nodes.

        Deliberately NOT called automatically from the linking path. Doing so
        would let any MCP caller write into the shared ontology by sending a
        typo, and an unresolved surface is far more often a bad input than a
        genuinely missing skill. Wire it to a reviewed ingestion flow, not to
        request handling.

        A newly created node has no SIMILAR edges, so it is isolated and cannot
        bridge to anything until edges are computed for it. Creating the node is
        the ontology decision; edge construction is a separate one.
        """
        if self.graph_source != GRAPH_SOURCE_NEO4J:
            raise RuntimeError(
                "register_skills writes to the graph and requires "
                f"SYNAPSE_GRAPH_SOURCE=neo4j (currently {self.graph_source!r}). "
                "Writes must not go to a local pickle that Aura will never see."
            )

        embedder = self.linker._embedder
        if embedder is None:
            self.linker._ensure_embeddings()
            embedder = self.linker._embedder
        if embedder is None:
            raise RuntimeError("No embedder available; cannot deduplicate before MERGE.")

        results: list[RegisteredSkill] = []
        created_any = False
        for surface in surfaces:
            vector = (embedder.encode_queries([surface])[0]
                      if hasattr(embedder, "encode_queries")
                      else embedder.encode([surface])[0])
            node, created = self.neo4j.get_or_create_skill_with_dedup(
                surface, vector, min_similarity=min_similarity
            )
            created_any = created_any or created
            results.append(RegisteredSkill(surface=surface, node=node, created=created))

        if created_any:
            # The in-process graph is now stale; next access reloads it.
            self.invalidate_graph()
        return results

    def stats(self) -> GraphStats:
        """Cheap enough to serve as a readiness probe (C6.3)."""
        graph = self.graph
        similar = sum(
            1 for _, _, d in graph.edges(data=True) if d.get("relation") == "similar"
        )
        return GraphStats(
            graph_source=self.graph_source,
            graph_path=(str(self.graph_path) if self.graph_source == GRAPH_SOURCE_PICKLE
                        else self.neo4j.config.uri),
            skill_nodes=len(self.skill_names),
            total_nodes=graph.number_of_nodes(),
            total_edges=graph.number_of_edges(),
            similar_edges=similar,
            orphan_skills=len(self.matcher.orphan_skills),
            embeddings_loaded=(
                self._linker is not None and self._linker._node_emb is not None
            ),
            scoring_params=_params_dict(self.params),
        )


# ------------------------------------------------------------------ singleton


_ENGINE: MatchEngine | None = None


def get_engine() -> MatchEngine:
    """Process-wide singleton. One graph in RAM, not one per request (NFR1)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = MatchEngine()
    return _ENGINE


def set_engine(engine: MatchEngine | None) -> None:
    """Swap the singleton. Tests use this to serve a small synthetic graph."""
    global _ENGINE
    _ENGINE = engine
