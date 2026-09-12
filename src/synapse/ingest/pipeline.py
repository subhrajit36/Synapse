"""Phase C3: the LangGraph ingestion pipeline (Reader -> Extractor), cloud-wired.

What this adds over calling `SkillExtractor.extract_from_document()` directly:

  * C3.1 - Reader and Extractor are real graph nodes over an explicit, typed
    state schema (`IngestState`), not a loose dict passed hand to hand.
  * C3.2 - Retry/backoff is its own node with its own edges. A 429 or a 503 is
    a routing decision the graph makes and records in state, not a `try/except`
    buried inside a helper. That is what makes "why did this document take four
    minutes?" answerable after the fact (NFR5, NFR6).
  * C3.3 - Extraction advances one chunk per superstep, so the checkpointer
    persists a `cursor` after every chunk. A batch killed during a rate-limit
    pause resumes at the chunk it was on instead of re-billing the whole
    document against the 15 RPM free tier.

The graph:

        read ──▶ extract ──chunk ok, more left──▶ extract
                    │  │
                    │  └──all chunks done──▶ finalize ──▶ END
                    ▼
                 backoff ──retries left──▶ extract
                    │
                    └──gave up (chunk recorded as failed)──▶ extract / finalize

Business logic still lives in `reader.py` and `extractor.py`; this module only
wires them, exactly as C4 will keep the MCP layer thin over `scoring`/`graph`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypedDict

from langgraph.graph import END, StateGraph

from .extractor import PROMPT_VERSION, SkillExtractor, backoff_delay
from .reader import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    SUPPORTED_SUFFIXES,
    read_document,
)
from .schemas import ExtractedSkill, ExtractionResult, merge_skills

logger = logging.getLogger(__name__)

STATUS_READING = "reading"
STATUS_EXTRACTING = "extracting"
STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"      # finished, but some chunks never validated
STATUS_READ_FAILED = "read_failed"
STATUS_LINK_FAILED = "link_failed"        # E3: canonicalization could not run
STATUS_PERSIST_FAILED = "persist_failed"  # E3: extracted but not stored

# Which step the backoff node is currently protecting. One backoff node serves
# both because the policy - wait, retry, eventually give up and record why - is
# identical; only what "give up" means differs.
STAGE_EXTRACT = "extract"
STAGE_PERSIST = "persist"


# -------------------------------------------------------------------------- state


class ChunkPayload(TypedDict):
    """A chunk flattened to primitives so it survives checkpoint serialization."""

    index: int
    text: str


class IngestState(TypedDict, total=False):
    """C3.1: the pipeline's state contract.

    `total=False` because nodes emit partial updates; every key below is
    populated by `initial_state()` before the graph runs, so a node never reads
    an absent key.
    """

    # --- inputs
    source_path: str
    doc_type: str

    # --- reader output
    source_id: str
    chunks: list[ChunkPayload]

    # --- extraction progress (the checkpointed part that makes resume work)
    cursor: int                 # index of the next chunk to extract
    skills: list[dict]          # ExtractedSkill dicts accumulated so far
    failed_chunks: list[int]

    # --- retry bookkeeping owned by the backoff node
    attempt: int                # consecutive failures on the current group
    last_error: str
    fatal: bool                 # error class says retrying cannot help
    stage: str                  # which step backoff is protecting

    # --- E3: canonicalization and persistence
    candidate_id: str
    content_hash: str           # dedupe key; a re-upload must not re-extract
    linked: list[dict]          # {node, weight, context, method, link_score}
    unresolved: list[str]       # surfaces that reached no graph node
    persisted: bool

    # --- terminal
    status: str
    result: dict | None         # ExtractionResult.model_dump()


def initial_state(
    source_path: str | Path,
    doc_type: str = "unknown",
    candidate_id: str = "",
) -> IngestState:
    return IngestState(
        source_path=str(source_path),
        doc_type=doc_type,
        source_id="",
        chunks=[],
        cursor=0,
        skills=[],
        failed_chunks=[],
        attempt=0,
        last_error="",
        fatal=False,
        stage=STAGE_EXTRACT,
        candidate_id=candidate_id,
        content_hash="",
        linked=[],
        unresolved=[],
        persisted=False,
        status=STATUS_READING,
        result=None,
    )


@dataclass(frozen=True)
class IngestionConfig:
    """Every tunable in one place, mirroring `ScoringParams` in the matcher."""

    chunk_words: int = DEFAULT_CHUNK_WORDS
    overlap_words: int = DEFAULT_OVERLAP_WORDS
    max_attempts: int = 4          # per group, across the backoff node
    backoff_base: float = 2.0
    backoff_cap: float = 30.0
    sleep: bool = True             # tests turn the real sleeping off
    # E2: chunks sent per Gemini call. The free tier caps requests per minute,
    # not tokens, so grouping chunks divides a batch's wall-clock cost by this
    # factor. Raising it coarsens checkpoint granularity and widens the blast
    # radius of one failed call; 1 restores the original chunk-at-a-time
    # behaviour exactly. Tune against real documents.
    chunks_per_call: int = 3


# --------------------------------------------------------------------- nodes


def _read_node(state: IngestState, config: IngestionConfig) -> IngestState:
    """Node 1. Load and chunk the document.

    A read failure is terminal for this document but not for the batch: it is
    recorded in state and routed straight to END, so one unreadable file cannot
    take down a 200-document run.
    """
    path = state["source_path"]
    try:
        document = read_document(
            path,
            doc_type=state.get("doc_type", "unknown"),
            chunk_words=config.chunk_words,
            overlap_words=config.overlap_words,
        )
    except (OSError, ValueError) as exc:
        logger.error("Read failed for %s: %s", path, exc)
        return {"status": STATUS_READ_FAILED, "last_error": f"{type(exc).__name__}: {exc}"}

    return {
        "source_id": document.source_id,
        "candidate_id": state.get("candidate_id") or document.source_id,
        # Hash the normalised text, not the raw bytes: the same résumé saved as
        # .txt and .docx should dedupe, and a trailing-whitespace edit should not
        # cost another extraction.
        "content_hash": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
        "chunks": [{"index": c.index, "text": c.text} for c in document.chunks],
        "cursor": 0,
        "skills": [],
        "failed_chunks": [],
        "attempt": 0,
        "last_error": "",
        "fatal": False,
        "stage": STAGE_EXTRACT,
        "status": STATUS_EXTRACTING,
    }


def _group_size(config: IngestionConfig) -> int:
    return max(1, config.chunks_per_call)


def _extract_node(
    state: IngestState, extractor: SkillExtractor, config: IngestionConfig
) -> IngestState:
    """Node 2. Extract one GROUP of chunks per superstep (E2).

    One group per superstep is still what gives the checkpointer something to
    save between Gemini calls (C3.3); the group is simply larger than one chunk
    now. Extracting the whole document in a single node would make its
    checkpoint all-or-nothing, which is the thing this design exists to avoid.

    The cursor counts chunks, not groups, so a checkpoint written under one
    `chunks_per_call` setting stays meaningful if the setting changes.
    """
    cursor = state["cursor"]
    chunks = state["chunks"]
    group = chunks[cursor:cursor + _group_size(config)]
    texts = [c["text"] for c in group]

    try:
        extracted = extractor.extract_batch(texts)
    except Exception as exc:  # noqa: BLE001 - classification happens in the router
        fatal = not extractor.is_retryable(exc)
        logger.warning(
            "Chunks %d-%d of %s failed (%s, attempt %d/%d, fatal=%s)",
            cursor, cursor + len(group) - 1,
            state.get("source_id") or state["source_path"],
            type(exc).__name__, state["attempt"] + 1, config.max_attempts, fatal,
        )
        return {
            "attempt": state["attempt"] + 1,
            "last_error": f"{type(exc).__name__}: {exc}",
            "fatal": fatal,
        }

    return {
        "skills": state["skills"] + [s.model_dump() for s in extracted],
        "cursor": cursor + len(group),
        "attempt": 0,
        "last_error": "",
        "fatal": False,
    }


def _link_node(state: IngestState, linker) -> IngestState:
    """E3. Canonicalize extracted surfaces onto graph nodes.

    Runs once per document, after every chunk group, rather than per group: the
    linker deduplicates by target node, so linking the merged set is both
    cheaper and more correct than linking each group in isolation.

    Linking is local computation. A failure here means the embedder is broken or
    absent, which retrying will not fix, so it is terminal for the document
    rather than routed to backoff - and terminal *without* losing the extraction,
    which still reaches `finalize` and the caller.
    """
    merged = merge_skills([ExtractedSkill.model_validate(s) for s in state["skills"]])

    try:
        profile = linker.link_many(
            [(s.skill, s.weight) for s in merged],
            source_id=state.get("source_id", ""),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Linking failed for %s: %s: %s",
                     state.get("source_id"), type(exc).__name__, exc)
        return {
            "status": STATUS_LINK_FAILED,
            "last_error": f"{type(exc).__name__}: {exc}",
        }

    # Keep the justifying context from extraction alongside the link provenance,
    # so a stored profile can still answer "why does this candidate have this
    # skill" without re-reading the document (NFR6).
    context_by_surface = {s.skill: s.context for s in merged}
    linked = [
        {
            "node": r.node,
            "weight": r.weight,
            "context": context_by_surface.get(r.surface, ""),
            "method": r.method,
            "link_score": round(r.score, 4),
        }
        for r in profile.results if r.node is not None
    ]
    unresolved = [r.surface for r in profile.unresolved]

    logger.info("Linked %s: %d nodes, %d unresolved",
                state.get("source_id"), len(linked), len(unresolved))
    return {"linked": linked, "unresolved": unresolved, "last_error": ""}


def _persist_node(state: IngestState, store, model: str = "") -> IngestState:
    """E3. Write the candidate profile to AuraDB.

    A network failure routes to the same backoff node the extractor uses, via
    `stage`. Giving up is NOT silent: the document finishes with
    `persist_failed`, the extraction is still returned, and the checkpoint means
    a re-run resumes here rather than re-extracting anything.
    """
    try:
        store.upsert_candidate(
            candidate_id=state["candidate_id"],
            skills=state["linked"],
            name=state.get("source_id", ""),
            source_id=state.get("source_id", ""),
            doc_type=state.get("doc_type", "resume"),
            content_hash=state.get("content_hash", ""),
            model=model,
            prompt_version=PROMPT_VERSION,
            chunk_count=len(state["chunks"]),
            failed_chunks=sorted(state["failed_chunks"]),
            unresolved=state["unresolved"],
        )
    except Exception as exc:  # noqa: BLE001 - classified by the router
        logger.warning(
            "Persist failed for %s (%s, attempt %d)",
            state["candidate_id"], type(exc).__name__, state["attempt"] + 1,
        )
        return {
            "stage": STAGE_PERSIST,
            "attempt": state["attempt"] + 1,
            "last_error": f"{type(exc).__name__}: {exc}",
            "fatal": _persist_is_fatal(exc),
        }

    return {"persisted": True, "attempt": 0, "last_error": "", "fatal": False,
            "stage": STAGE_EXTRACT}


def _persist_is_fatal(exc: Exception) -> bool:
    """Auth and schema errors will not fix themselves; network errors might."""
    text = str(exc).lower()
    return any(k in text for k in (
        "unauthorized", "authentication", "forbidden", "constraint",
        "syntaxerror", "invalid input",
    ))


def _backoff_node(state: IngestState, config: IngestionConfig) -> IngestState:
    """C3.2. The retry policy, as a node.

    Two outcomes, both explicit in state:
      * retries remain and the error is transient -> wait, then re-run the group
        with `cursor` untouched.
      * the error is fatal, or attempts are exhausted -> record EVERY chunk index
        in the group as failed and step over the whole group. A chunk is never
        silently dropped (A1.3), and one poisoned group never stalls the
        document forever. Recording the whole group is the honest accounting:
        one call covered N chunks, so its failure cost all N of them.
    """
    cursor = state["cursor"]
    give_up = state["fatal"] or state["attempt"] >= config.max_attempts

    if state.get("stage") == STAGE_PERSIST:
        if give_up:
            logger.error(
                "Giving up on persisting %s after %d attempt(s): %s",
                state.get("candidate_id"), state["attempt"], state["last_error"],
            )
            # The extraction is not lost - it still reaches finalize and the
            # caller. Only the write to the pool failed, and the checkpoint
            # means a re-run retries the write without re-extracting.
            return {"status": STATUS_PERSIST_FAILED, "attempt": 0, "fatal": False}
        delay = backoff_delay(
            state["attempt"] - 1, base=config.backoff_base, cap=config.backoff_cap
        )
        logger.info("Backing off %.1fs before retrying persist", delay)
        if config.sleep:
            time.sleep(delay)
        return {}

    if give_up:
        span = len(state["chunks"][cursor:cursor + _group_size(config)]) or 1
        logger.error(
            "Giving up on chunks %d-%d of %s after %d attempt(s): %s",
            cursor, cursor + span - 1,
            state.get("source_id") or state["source_path"],
            state["attempt"], state["last_error"],
        )
        return {
            "failed_chunks": state["failed_chunks"] + list(range(cursor, cursor + span)),
            "cursor": cursor + span,
            "attempt": 0,
            "fatal": False,
        }

    delay = backoff_delay(
        state["attempt"] - 1, base=config.backoff_base, cap=config.backoff_cap
    )
    logger.info("Backing off %.1fs before retrying chunk %d", delay, cursor)
    if config.sleep:
        time.sleep(delay)
    return {}


def _finalize_node(state: IngestState, extractor: SkillExtractor) -> IngestState:
    """Collapse per-chunk output into one audited `ExtractionResult`."""
    skills = [ExtractedSkill.model_validate(s) for s in state["skills"]]
    failed = sorted(state["failed_chunks"])
    result = ExtractionResult(
        source_id=state["source_id"],
        doc_type=state.get("doc_type", "unknown"),
        skills=merge_skills(skills),
        model=extractor.model,
        prompt_version=PROMPT_VERSION,
        chunk_count=len(state["chunks"]),
        failed_chunks=failed,
    )
    # A terminal failure already recorded upstream wins: the extraction may be
    # complete while the link or the write was not, and reporting "complete"
    # would hide that the profile never reached the pool.
    status = state.get("status")
    if status not in (STATUS_LINK_FAILED, STATUS_PERSIST_FAILED):
        status = STATUS_COMPLETE if not failed else STATUS_PARTIAL

    return {"result": result.model_dump(), "status": status}


# -------------------------------------------------------------------- routers


def _route_after_read(state: IngestState) -> str:
    if state["status"] == STATUS_READ_FAILED:
        return END
    # An empty document is a legitimate outcome, not an error: finalize it so the
    # caller still gets a result object recording zero skills over zero chunks.
    return "extract" if state["chunks"] else "finalize"


def _route_after_extract(state: IngestState) -> str:
    if state["last_error"]:
        return "backoff"
    if state["cursor"] < len(state["chunks"]):
        return "extract"
    return "link"


def _route_after_backoff(state: IngestState) -> str:
    if state.get("stage") == STAGE_PERSIST:
        # Gave up: status was set, so stop retrying and finish.
        return "finalize" if state.get("status") == STATUS_PERSIST_FAILED else "persist"
    if state["cursor"] < len(state["chunks"]):
        return "extract"
    return "link"


def _route_after_link(state: IngestState) -> str:
    return "finalize" if state.get("status") == STATUS_LINK_FAILED else "persist"


def _route_after_persist(state: IngestState) -> str:
    return "finalize" if state.get("persisted") else "backoff"


# --------------------------------------------------------------------- graph


def build_ingestion_graph(
    extractor: SkillExtractor,
    config: IngestionConfig | None = None,
    checkpointer=None,
    linker=None,
    store=None,
):
    """Compile the ingestion graph.

        read -> extract -> [link -> persist] -> finalize

    `linker` and `store` are optional. Without them the graph is exactly the
    Phase C3 pipeline: read, extract, write JSON. With them it is the Phase E
    pool ingestion. Keeping both shapes in one graph avoids a second, drifting
    copy of the retry and checkpoint logic.

    All three collaborators are closed over rather than carried in state: a live
    SDK client, an embedder and a database driver are none of them serializable,
    and putting them in state would break checkpointing.
    """
    config = config or IngestionConfig()

    builder = StateGraph(IngestState)
    builder.add_node("read", lambda s: _read_node(s, config))
    builder.add_node("extract", lambda s: _extract_node(s, extractor, config))
    builder.add_node("backoff", lambda s: _backoff_node(s, config))
    builder.add_node("finalize", lambda s: _finalize_node(s, extractor))

    # "link" and "persist" always exist as nodes so the routers have one shape;
    # when a collaborator is absent the node is a pass-through and the router
    # skips straight to finalize.
    builder.add_node(
        "link",
        (lambda s: _link_node(s, linker)) if linker is not None else (lambda s: {}),
    )
    builder.add_node(
        "persist",
        (lambda s: _persist_node(s, store, extractor.model)) if store is not None
        else (lambda s: {"persisted": False}),
    )

    def route_after_extract(state: IngestState) -> str:
        nxt = _route_after_extract(state)
        return "finalize" if (nxt == "link" and linker is None) else nxt

    def route_after_backoff(state: IngestState) -> str:
        nxt = _route_after_backoff(state)
        return "finalize" if (nxt == "link" and linker is None) else nxt

    def route_after_link(state: IngestState) -> str:
        nxt = _route_after_link(state)
        return "finalize" if (nxt == "persist" and store is None) else nxt

    builder.set_entry_point("read")
    builder.add_conditional_edges(
        "read", _route_after_read, {"extract": "extract", "finalize": "finalize", END: END}
    )
    builder.add_conditional_edges(
        "extract", route_after_extract,
        {"extract": "extract", "backoff": "backoff", "link": "link",
         "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "backoff", route_after_backoff,
        {"extract": "extract", "link": "link", "persist": "persist",
         "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "link", route_after_link, {"persist": "persist", "finalize": "finalize"}
    )
    builder.add_conditional_edges(
        "persist", _route_after_persist,
        {"backoff": "backoff", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


def default_thread_id(path: str | Path) -> str:
    """Stable checkpoint thread id for a document.

    The stem alone is not enough: `ravi_backend.txt` and `ravi_backend.docx`
    share a stem, and sharing a thread would let one document resume the other's
    pending work. The path digest keeps ids unique while the stem keeps logs
    readable, and the same file always maps to the same id so resume works
    across process restarts.
    """
    path = Path(path)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{path.stem}-{digest}"


def make_checkpointer(path: str | Path | None = None):
    """C3.3. SQLite-backed saver when given a path, in-memory otherwise.

    Only the SQLite saver survives process death, which is the case that
    actually matters here - a run interrupted during a rate-limit pause.
    """
    if path is None:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: LangGraph may touch the connection from a worker
    # thread. Access stays serialized by the saver itself.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


# ------------------------------------------------------------------- runner


class IngestionPipeline:
    """Run the graph over one document or a directory, with resume."""

    def __init__(
        self,
        extractor: SkillExtractor | None = None,
        config: IngestionConfig | None = None,
        checkpoint_path: str | Path | None = None,
        checkpointer=None,
        linker=None,
        store=None,
    ) -> None:
        self.extractor = extractor or SkillExtractor()
        self.config = config or IngestionConfig()
        self.checkpointer = checkpointer or make_checkpointer(checkpoint_path)
        self.linker = linker
        self.store = store
        self.graph = build_ingestion_graph(
            self.extractor, self.config, self.checkpointer,
            linker=linker, store=store,
        )

    # -- resume ------------------------------------------------------------

    def _thread_config(self, thread_id: str, chunk_estimate: int) -> dict:
        # Each chunk costs one `extract` superstep plus, worst case, one
        # extract+backoff pair per retry. The +20 covers read/finalize and a
        # short document's rounding.
        limit = 20 + chunk_estimate * (1 + 2 * self.config.max_attempts)
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": limit}

    def _pending(self, config: dict) -> bool:
        """True if this thread has an interrupted run waiting to be resumed."""
        snapshot = self.graph.get_state(config)
        return bool(snapshot.next)

    # -- single document ---------------------------------------------------

    def run(
        self,
        path: str | Path,
        doc_type: str = "unknown",
        thread_id: str | None = None,
        force_restart: bool = False,
    ) -> ExtractionResult | None:
        """Ingest one document. Returns None if the file could not be read.

        If a previous run on the same `thread_id` was interrupted, this resumes
        it (`invoke(None, ...)`) instead of re-extracting chunks that already
        succeeded and were already paid for against the RPM budget.
        """
        path = Path(path)
        thread_id = thread_id or default_thread_id(path)
        # Cheap upper bound on chunk count, used only to size the recursion
        # limit; the reader node does the real chunking.
        estimate = max(1, path.stat().st_size // (self.config.chunk_words * 3) + 1)
        config = self._thread_config(thread_id, estimate)

        resuming = not force_restart and self._pending(config)
        if resuming:
            logger.info("Resuming interrupted ingestion of %s", thread_id)
            final = self.graph.invoke(None, config)
        else:
            final = self.graph.invoke(initial_state(path, doc_type), config)

        if final.get("status") == STATUS_READ_FAILED:
            return None
        result = final.get("result")
        return ExtractionResult.model_validate(result) if result else None

    # -- batch -------------------------------------------------------------

    def run_batch(
        self,
        paths: Sequence[str | Path],
        doc_type: str = "unknown",
        thread_prefix: str = "",
    ) -> list[ExtractionResult]:
        """Ingest many documents, one checkpoint thread each.

        Per-document threads mean an interrupted batch resumes only the document
        it died on; everything already finished is skipped by its own checkpoint.
        """
        results: list[ExtractionResult] = []
        for path in paths:
            path = Path(path)
            try:
                result = self.run(
                    path,
                    doc_type=doc_type,
                    thread_id=f"{thread_prefix}{default_thread_id(path)}",
                )
            except Exception as exc:  # noqa: BLE001
                # State is checkpointed; re-running the batch picks this one up.
                logger.error("Ingestion of %s aborted: %s", path, exc)
                continue
            if result is not None:
                results.append(result)
        return results

    def run_directory(self, directory: str | Path, doc_type: str = "unknown", **kw):
        paths = sorted(
            p for p in Path(directory).iterdir()
            if p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        return self.run_batch(paths, doc_type=doc_type, **kw)


# ---------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase C3 LangGraph ingestion.")
    parser.add_argument("source", help="A document, or a directory of documents.")
    parser.add_argument("--doc-type", default="unknown", choices=["resume", "jd", "unknown"])
    parser.add_argument(
        "--checkpoint",
        default="data/checkpoints/ingest.sqlite",
        help="SQLite checkpoint file; 'none' for a non-resumable in-memory run.",
    )
    parser.add_argument("--out", default="data/extractions", help="Where to write JSON results.")
    parser.add_argument("--rpm", type=int, default=15, help="Gemini free-tier RPM ceiling.")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument(
        "--chunks-per-call", type=int, default=IngestionConfig.chunks_per_call,
        help="Chunks per Gemini call. Higher = fewer calls, coarser checkpoints.",
    )
    parser.add_argument(
        "--to-pool", action="store_true",
        help="E5: canonicalize and write each candidate into the AuraDB pool.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    linker = store = None
    if args.to_pool:
        # Imported here so a plain JSON run never constructs an embedder or a
        # database client.
        from ..mcp.engine import MatchEngine

        engine = MatchEngine()
        client = engine.neo4j
        if not client.config.is_configured:
            parser.error(
                "--to-pool writes to AuraDB but no NEO4J_PASSWORD is set "
                f"({client.config.describe()})."
            )
        client.ensure_candidate_schema()
        linker, store = engine.linker, client
        print(f"writing to pool: {client.config.uri}")

    pipeline = IngestionPipeline(
        extractor=SkillExtractor(rpm=args.rpm),
        config=IngestionConfig(max_attempts=args.max_attempts,
                               chunks_per_call=args.chunks_per_call),
        checkpoint_path=None if args.checkpoint == "none" else args.checkpoint,
        linker=linker,
        store=store,
    )

    source = Path(args.source)
    if source.is_dir():
        results = pipeline.run_directory(source, doc_type=args.doc_type)
    else:
        one = pipeline.run(source, doc_type=args.doc_type)
        results = [one] if one else []

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # source_id is the file stem, so a directory holding both `x.txt` and
    # `x.docx` yields two results with the same name. Disambiguate rather than
    # letting the second silently overwrite the first.
    written: set[str] = set()
    for result in results:
        name = result.source_id
        if name in written:
            name = f"{name}-{len(written)}"
        written.add(name)
        (out_dir / f"{name}.json").write_text(
            json.dumps(result.model_dump(), indent=2), encoding="utf-8"
        )
        flag = "" if result.is_complete else f"  [failed chunks: {result.failed_chunks}]"
        print(f"{name}: {len(result.skills)} skills "
              f"from {result.chunk_count} chunks{flag}")

    print(f"\n{len(results)} document(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
