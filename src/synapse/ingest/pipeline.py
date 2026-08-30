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
    attempt: int                # consecutive failures on the current chunk
    last_error: str
    fatal: bool                 # error class says retrying cannot help

    # --- terminal
    status: str
    result: dict | None         # ExtractionResult.model_dump()


def initial_state(source_path: str | Path, doc_type: str = "unknown") -> IngestState:
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
        status=STATUS_READING,
        result=None,
    )


@dataclass(frozen=True)
class IngestionConfig:
    """Every tunable in one place, mirroring `ScoringParams` in the matcher."""

    chunk_words: int = DEFAULT_CHUNK_WORDS
    overlap_words: int = DEFAULT_OVERLAP_WORDS
    max_attempts: int = 4          # per chunk, across the backoff node
    backoff_base: float = 2.0
    backoff_cap: float = 30.0
    sleep: bool = True             # tests turn the real sleeping off


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
        "chunks": [{"index": c.index, "text": c.text} for c in document.chunks],
        "cursor": 0,
        "skills": [],
        "failed_chunks": [],
        "attempt": 0,
        "last_error": "",
        "fatal": False,
        "status": STATUS_EXTRACTING,
    }


def _extract_node(
    state: IngestState, extractor: SkillExtractor, config: IngestionConfig
) -> IngestState:
    """Node 2. Extract exactly one chunk.

    One chunk per superstep is the whole point: it is what gives the
    checkpointer something to save between Gemini calls (C3.3). Batching the
    document into a single node would make the checkpoint all-or-nothing.
    """
    cursor = state["cursor"]
    chunk = state["chunks"][cursor]

    try:
        extracted = extractor.extract_once(chunk["text"])
    except Exception as exc:  # noqa: BLE001 - classification happens in the router
        fatal = not extractor.is_retryable(exc)
        logger.warning(
            "Chunk %d of %s failed (%s, attempt %d/%d, fatal=%s)",
            cursor, state.get("source_id") or state["source_path"],
            type(exc).__name__, state["attempt"] + 1, config.max_attempts, fatal,
        )
        return {
            "attempt": state["attempt"] + 1,
            "last_error": f"{type(exc).__name__}: {exc}",
            "fatal": fatal,
        }

    return {
        "skills": state["skills"] + [s.model_dump() for s in extracted],
        "cursor": cursor + 1,
        "attempt": 0,
        "last_error": "",
        "fatal": False,
    }


def _backoff_node(state: IngestState, config: IngestionConfig) -> IngestState:
    """C3.2. The retry policy, as a node.

    Two outcomes, both explicit in state:
      * retries remain and the error is transient -> wait, then re-run the chunk
        with `cursor` untouched.
      * the error is fatal, or attempts are exhausted -> record the chunk index
        in `failed_chunks` and step over it. A chunk is never silently dropped
        (A1.3), and one poisoned chunk never stalls the document forever.
    """
    cursor = state["cursor"]
    give_up = state["fatal"] or state["attempt"] >= config.max_attempts

    if give_up:
        logger.error(
            "Giving up on chunk %d of %s after %d attempt(s): %s",
            cursor, state.get("source_id") or state["source_path"],
            state["attempt"], state["last_error"],
        )
        return {
            "failed_chunks": state["failed_chunks"] + [cursor],
            "cursor": cursor + 1,
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
    return {
        "result": result.model_dump(),
        "status": STATUS_COMPLETE if not failed else STATUS_PARTIAL,
    }


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
    return "extract" if state["cursor"] < len(state["chunks"]) else "finalize"


def _route_after_backoff(state: IngestState) -> str:
    return "extract" if state["cursor"] < len(state["chunks"]) else "finalize"


# --------------------------------------------------------------------- graph


def build_ingestion_graph(
    extractor: SkillExtractor,
    config: IngestionConfig | None = None,
    checkpointer=None,
):
    """Compile the Reader -> Extractor graph.

    `extractor` is closed over rather than carried in state: a live SDK client is
    not serializable, and putting it in state would break checkpointing.
    """
    config = config or IngestionConfig()

    builder = StateGraph(IngestState)
    builder.add_node("read", lambda s: _read_node(s, config))
    builder.add_node("extract", lambda s: _extract_node(s, extractor, config))
    builder.add_node("backoff", lambda s: _backoff_node(s, config))
    builder.add_node("finalize", lambda s: _finalize_node(s, extractor))

    builder.set_entry_point("read")
    builder.add_conditional_edges(
        "read", _route_after_read, {"extract": "extract", "finalize": "finalize", END: END}
    )
    builder.add_conditional_edges(
        "extract", _route_after_extract,
        {"extract": "extract", "backoff": "backoff", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "backoff", _route_after_backoff, {"extract": "extract", "finalize": "finalize"}
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
    ) -> None:
        self.extractor = extractor or SkillExtractor()
        self.config = config or IngestionConfig()
        self.checkpointer = checkpointer or make_checkpointer(checkpoint_path)
        self.graph = build_ingestion_graph(
            self.extractor, self.config, self.checkpointer
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
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    pipeline = IngestionPipeline(
        extractor=SkillExtractor(rpm=args.rpm),
        config=IngestionConfig(max_attempts=args.max_attempts),
        checkpoint_path=None if args.checkpoint == "none" else args.checkpoint,
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
