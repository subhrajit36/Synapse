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
# 773, not 15459. The 15459-pair graph was built with an absolute similarity
# threshold that stopped being selective when the embedder changed (see
# build_graph.add_semantic_edges). AuraDB still holds that dense graph until it
# is re-migrated, so this assertion will - correctly - refuse to start against
# it until the migration is re-run from the rebuilt artifact.
EXPECTED_SIMILAR_PAIRS = int(os.getenv("SYNAPSE_EXPECTED_PAIRS", "773"))

# Phase D reads its JDs from the versioned eval snapshot rather than an ad-hoc
# demo fixture, so the page shows the same pairs the reported metrics came from.
DEFAULT_EVAL_DATASET = os.getenv("SYNAPSE_EVAL_DATASET", "data/eval/v2/dataset.json")

# Phase C1's production embedder. FastEmbed builds it lazily on the first
# surface that reaches the embedding fallback, which keeps a cold start off the
# 512MB ceiling until something actually needs it (NFR1).
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# F3: where the upload pipeline checkpoints. SQLite so a retry after a Gemini
# rate-limit pause resumes instead of re-billing the document (C3.3).
DEFAULT_CHECKPOINT_PATH = os.getenv("SYNAPSE_CHECKPOINT", "data/checkpoints/ingest.sqlite")

# Upload outcomes. `reused` is the dedupe hit: no pipeline ran at all.
UPLOAD_REUSED = "reused"


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
    candidate_source: str = Field(
        "request",
        description="'pool' when the stored candidate pool was ranked, "
                    "'request' when candidates were supplied in the call.",
    )
    batch_id: str | None = Field(
        None,
        description="Upload session the pool was scoped to; null = the whole pool.",
    )
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


class PoolCandidate(BaseModel):
    """One row in the stored candidate pool (E4). No skill payload - a listing
    is polled and must stay cheap."""

    candidate_id: str
    name: str
    doc_type: str = ""
    model: str = Field("", description="Extractor that produced this profile.")
    skill_count: int
    unresolved_count: int = Field(
        0, description="Surfaces that reached no graph node and so score nothing."
    )
    batch_ids: list[str] = Field(
        default_factory=list,
        description="Every upload session this candidate was part of (F2).",
    )
    extracted_at: str = ""


class PoolListResponse(BaseModel):
    count: int
    batch_id: str | None = Field(None, description="Session filter applied; null = all.")
    candidates: list[PoolCandidate]


class UploadResponse(BaseModel):
    """What one `POST /api/candidates` produced (F3).

    `in_pool` is the field the page should act on: True means `rank_candidates`
    with this `batch_id` will see the candidate. `status` says why not when it
    is False, and `reused` says whether the extraction was paid for by this
    request or an earlier one.
    """

    candidate_id: str
    name: str
    batch_id: str
    content_hash: str
    status: str = Field(..., description="reused | complete | partial | link_failed | persist_failed")
    reused: bool = Field(..., description="Existing profile attached to the batch; no LLM call.")
    in_pool: bool
    skill_count: int
    unresolved: list[str] = Field(default_factory=list)
    failed_chunks: int = 0


class JDSkillsResponse(BaseModel):
    """F4: a free-text JD turned into something `rank_candidates` accepts.

    `skills` is already canonical and weighted - it can be posted straight to
    `/api/rank_pool` as `jd_skills`. `unresolved` is what the model named but
    the graph does not know; the page should show it, because a JD whose key
    requirement is unresolved is being ranked on a narrower job than written.
    """

    skills: list[SkillWeight] = Field(
        ..., description="Canonical graph nodes with demand weights, ready to rank on."
    )
    unresolved: list[str] = Field(
        default_factory=list, description="Extracted surfaces that reached no node."
    )
    extracted: list[SkillWeight] = Field(
        default_factory=list, description="What the model returned, before linking."
    )
    linking: list[LinkedSkill] = Field(
        default_factory=list, description="Surface -> node trace, NFR6."
    )
    model: str = ""
    text_words: int = 0


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
        pipeline=None,
        checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
        extractor=None,
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
        self._pipeline = pipeline           # injectable for tests
        self._extractor = extractor         # injectable for tests
        self.checkpoint_path = checkpoint_path
        self._graph = None
        self._matcher: Matcher | None = None
        self._linker: EntityLinker | None = None
        self._dataset: dict | None = None
        # Keyed by batch id (None = the whole pool) so that two upload sessions
        # ranking at the same time do not evict each other's cached slice.
        self._pool: dict[str | None, list[dict]] = {}

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

    @property
    def extractor(self):
        """The Gemini extractor, built on first use and shared by the upload
        pipeline (F3) and JD extraction (F4), so a résumé and the JD it is
        ranked against are read by the same model under the same prompt."""
        if self._extractor is None:
            from ..ingest.extractor import SkillExtractor

            try:
                self._extractor = SkillExtractor()
            except ValueError as exc:
                # A missing GEMINI_API_KEY is a deployment problem, not a bad
                # request; surface it as such so the routes map it to 503.
                raise RuntimeError(f"Extraction unavailable: {exc}") from exc
        return self._extractor

    @property
    def pipeline(self):
        """The ingestion graph, built on first upload (F3).

        Lazy for the same reason as the graph: constructing it opens the Gemini
        client and the checkpoint database, neither of which a ranking-only
        process should pay for. Reuses the engine's own linker and Neo4j client
        so an uploaded candidate is canonicalized exactly as a ranked one is.
        """
        if self._pipeline is None:
            from ..ingest.pipeline import IngestionPipeline

            self._pipeline = IngestionPipeline(
                extractor=self.extractor,
                checkpoint_path=self.checkpoint_path,
                linker=self.linker,
                store=self.neo4j,
            )
        return self._pipeline

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

    # -- E4: the persistent candidate pool ---------------------------------

    def pool_for(self, batch_id: str | None = None) -> list[dict]:
        """Stored candidate profiles for one upload session, cached per batch.

        Ranking is the fast half of the system and must stay that way, so a
        slice is fetched in one query rather than one per candidate. It is
        small - a profile is a name and a few dozen skill weights. None is the
        whole pool.
        """
        key = batch_id or None
        if key not in self._pool:
            client = self.neo4j
            if not client.config.is_configured:
                raise RuntimeError(
                    "The candidate pool lives in AuraDB, but no NEO4J_PASSWORD "
                    f"is set ({client.config.describe()}). Ingest candidates "
                    "first, and configure the Neo4j environment variables."
                )
            self._pool[key] = client.load_candidate_pool(batch_id=key)
            logger.info("Loaded %d candidates from the pool (batch=%s)",
                        len(self._pool[key]), key)
        return self._pool[key]

    @property
    def pool(self) -> list[dict]:
        """The whole pool. Kept for callers that predate batch scoping."""
        return self.pool_for(None)

    def invalidate_pool(self, batch_id: str | None = None) -> None:
        """Drop cached slices so the next ranking sees new ingestions.

        A write to one batch also stales the unscoped view, which is a superset
        of it; other batches are untouched, since a candidate is only ever
        added to the batch it was uploaded under. No batch given = drop all.
        """
        if batch_id is None:
            self._pool.clear()
        else:
            self._pool.pop(batch_id, None)
            self._pool.pop(None, None)

    def list_pool(self, batch_id: str | None = None) -> PoolListResponse:
        """What is in the pool, without the skill payloads."""
        batch_id = batch_id or None
        rows = self.neo4j.list_candidates(batch_id=batch_id)
        return PoolListResponse(
            count=len(rows),
            batch_id=batch_id,
            candidates=[
                PoolCandidate(
                    candidate_id=r["candidate_id"],
                    name=r.get("name") or r["candidate_id"],
                    doc_type=r.get("doc_type") or "",
                    model=r.get("model") or "",
                    skill_count=r.get("skill_count") or 0,
                    unresolved_count=r.get("unresolved_count") or 0,
                    batch_ids=list(r.get("batch_ids") or []),
                    extracted_at=r.get("extracted_at") or "",
                )
                for r in rows
            ],
        )

    # -- F3: one résumé in -------------------------------------------------

    def ingest_upload(
        self,
        data: bytes,
        filename: str,
        batch_id: str,
        doc_type: str = "resume",
    ) -> UploadResponse:
        """`POST /api/candidates`: one document into the pool, under a batch.

        Hash first. If the pool already holds this exact text, the existing
        profile is attached to the batch and nothing else runs - the extraction
        was paid for once and is not paid for again (locked decision). Only a
        genuinely new document goes through the LangGraph pipeline.

        Raises `ValueError` for a bad request (no batch, unsupported or
        unparseable file) and `RuntimeError` when the service is not configured
        to accept uploads; the route maps those to 400 and 503.
        """
        from ..ingest.reader import read_bytes
        from ..ingest.pipeline import content_hash_of

        if not batch_id:
            raise ValueError("batch_id is required: an upload belongs to a session.")

        client = self.neo4j
        if not client.config.is_configured:
            raise RuntimeError(
                "Uploads write to the AuraDB candidate pool, but no NEO4J_PASSWORD "
                f"is set ({client.config.describe()})."
            )

        document = read_bytes(data, filename, doc_type=doc_type)   # ValueError on a bad file
        if not document.chunks:
            # The common case is a scanned PDF with no text layer. The graph
            # would finalize it with zero skills and never persist it, so say
            # so now, before an LLM call or a checkpoint is spent on nothing.
            raise ValueError(
                f"No readable text in {filename!r}. A scanned or image-only "
                "document has no text layer to extract skills from."
            )
        content_hash = content_hash_of(document.text)

        existing = client.find_candidate_by_hash(content_hash)
        if existing:
            client.add_candidate_to_batch(existing, batch_id)
            self.invalidate_pool(batch_id)
            profile = client.get_candidate(existing) or {}
            logger.info("Upload %s reused candidate %s (batch=%s)", filename, existing, batch_id)
            return UploadResponse(
                candidate_id=existing,
                name=profile.get("name") or existing,
                batch_id=batch_id,
                content_hash=content_hash,
                status=UPLOAD_REUSED,
                reused=True,
                in_pool=True,
                skill_count=len(profile.get("skills") or []),
                unresolved=list(profile.get("unresolved") or []),
                failed_chunks=len(profile.get("failed_chunks") or []),
            )

        # The file stem alone is not an identity: two people uploading
        # `resume.pdf` must not MERGE into one node. The hash makes it unique;
        # the stem keeps it readable.
        candidate_id = f"{document.source_id}-{content_hash[:8]}"
        final = self.pipeline.run_document(
            document, batch_id=batch_id, candidate_id=candidate_id
        )

        in_pool = bool(final.get("persisted"))
        if in_pool:
            self.invalidate_pool(batch_id)
        return UploadResponse(
            candidate_id=final.get("candidate_id") or candidate_id,
            name=document.source_id,
            batch_id=batch_id,
            content_hash=content_hash,
            status=final.get("status") or "",
            reused=False,
            in_pool=in_pool,
            skill_count=len(final.get("linked") or []),
            unresolved=list(final.get("unresolved") or []),
            failed_chunks=len(final.get("failed_chunks") or []),
        )

    # -- F4: JD text -> skills ---------------------------------------------

    def extract_jd_skills(self, text: str) -> JDSkillsResponse:
        """`POST /api/jd_skills`: one chunk, one call, synchronous.

        A JD is short and the page is waiting, so it does not go through the
        checkpointed pipeline: no chunking, no backoff node, one `extract_once`.
        A failure is returned to the caller to retry, not retried here - the
        pipeline's retry policy exists for unattended batches, and a user with
        a Retry button is a better policy for an interactive call.

        Extraction weights become demand weights unchanged: the model's "core
        requirement" (1.5) versus "nice to have" (0.5) is exactly the signal
        `ScoringParams.use_weights` scores on.
        """
        from ..ingest.reader import normalize_text
        from ..ingest.schemas import merge_skills

        cleaned = normalize_text(text or "")
        if not cleaned:
            raise ValueError("JD text is empty.")

        extractor = self.extractor
        try:
            extracted = merge_skills(extractor.extract_once(cleaned))
        except Exception as exc:  # noqa: BLE001 - one attempt; the caller retries
            raise RuntimeError(
                f"JD extraction failed ({type(exc).__name__}): {exc}"
            ) from exc

        pairs = [SkillWeight(skill=s.skill, weight=s.weight) for s in extracted]
        skills, profile = self.resolve(pairs, source_id="jd", link=True)

        return JDSkillsResponse(
            skills=[
                SkillWeight(skill=node, weight=weight)
                for node, weight in sorted(skills.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
            unresolved=[r.surface for r in profile.unresolved],
            extracted=pairs,
            linking=_to_linked(profile),
            model=getattr(extractor, "model", ""),
            text_words=len(cleaned.split()),
        )

    def _pool_as_candidates(
        self, batch_id: str | None = None
    ) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
        """Pool profiles in the shape the matcher wants.

        Stored skills are already canonical node names and already carry their
        extracted proficiency weight, so nothing is re-linked here. Re-linking
        would let a linking change silently move the score of a candidate whose
        document has not been touched since ingestion.
        """
        pool = self.pool_for(batch_id)
        resolved = {c["name"] or c["candidate_id"]: c["skills"] for c in pool}
        unresolved = {
            (c["name"] or c["candidate_id"]): list(c.get("unresolved") or [])
            for c in pool
        }
        return resolved, unresolved

    def rank_candidates(
        self,
        jd_skills: Sequence[SkillWeight],
        candidates: Sequence[CandidateInput] | None = None,
        top_k: int | None = None,
        max_hops: int | None = None,
        use_weights: bool | None = None,
        enable_bridging: bool | None = None,
        link: bool = True,
        batch_id: str | None = None,
    ) -> RankingResponse:
        """FR5: rank candidates against a JD with explainable components.

        `candidates` omitted ranks the stored pool (E4), narrowed to one upload
        session when `batch_id` is given (F2). Passing candidates explicitly
        keeps the original stateless behaviour, which the evaluation harness
        and the tests rely on.
        """
        batch_id = batch_id or None
        if candidates is not None and batch_id is not None:
            # Not silently ignorable: the caller asked for two different sets.
            raise ValueError(
                "batch_id scopes the stored pool and cannot be combined with "
                "explicit candidates; pass one or the other."
            )

        params = self._params_for(max_hops, use_weights, enable_bridging)
        jd_map, jd_profile = self.resolve(jd_skills, "jd", link)

        # `None` means "use the pool"; an explicitly empty list means "score
        # nothing". Collapsing the two would make `candidates=[]` silently rank
        # the entire stored pool, which is not what an empty list asks for.
        if candidates is not None:
            resolved: dict[str, dict[str, float]] = {}
            unresolved: dict[str, list[str]] = {}
            for candidate in candidates:
                skills, profile = self.resolve(candidate.skills, candidate.name, link)
                resolved[candidate.name] = skills
                unresolved[candidate.name] = [r.surface for r in profile.unresolved]
            source = "request"
        else:
            resolved, unresolved = self._pool_as_candidates(batch_id)
            source = "pool"

        ranked = self.matcher.rank(jd_map, resolved, params=params, top_k=top_k)

        return RankingResponse(
            jd_skills=sorted(jd_map),
            jd_unresolved=[r.surface for r in jd_profile.unresolved],
            params=_params_dict(params),
            candidate_source=source,
            batch_id=batch_id if source == "pool" else None,
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
