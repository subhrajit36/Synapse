"""Phase C3 tests: the LangGraph ingestion pipeline. No network, no API key.

The extractor is faked throughout - what is under test here is the graph's
routing, its retry/backoff node and its checkpoint/resume behaviour, not
Gemini's output.
"""

from __future__ import annotations

import json

import pytest

from synapse.ingest.extractor import SkillExtractor
from synapse.ingest.pipeline import (
    STATUS_PARTIAL,
    STATUS_READ_FAILED,
    IngestionConfig,
    IngestionPipeline,
    build_ingestion_graph,
    default_thread_id,
    initial_state,
    make_checkpointer,
)

VALID = json.dumps([{"skill": "Python", "weight": 1.5, "context": "led Python work"}])
VALID_DOCKER = json.dumps([{"skill": "Docker", "weight": 1.0, "context": "containerised"}])

# Fast config: no real sleeping, so a backoff path costs nothing in test time.
FAST = IngestionConfig(chunk_words=10, overlap_words=2, max_attempts=3, sleep=False)


class FakeClient:
    """Queued payloads, same shape as the A1 tests' fake."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0
        self.models = self

    def generate_content(self, **kwargs):
        self.calls += 1
        payload = self.payloads.pop(0) if self.payloads else VALID
        if isinstance(payload, Exception):
            raise payload

        class R:
            text = payload

        return R()


@pytest.fixture(autouse=True)
def _stub_types(monkeypatch):
    """Keep the SDK's types module out of the way (mirrors test_ingest_a1)."""
    import sys
    import types as pytypes

    if "google.genai" not in sys.modules:
        google = sys.modules.setdefault("google", pytypes.ModuleType("google"))
        genai = pytypes.ModuleType("google.genai")
        gtypes = pytypes.ModuleType("google.genai.types")
        gtypes.GenerateContentConfig = lambda **kw: kw
        genai.types = gtypes
        google.genai = genai
        sys.modules["google.genai"] = genai
        sys.modules["google.genai.types"] = gtypes


def make_extractor(payloads) -> SkillExtractor:
    # rpm=0 disables the client-side limiter; the fake client has no quota.
    return SkillExtractor(client=FakeClient(payloads), rpm=0)


def write_doc(tmp_path, name="cand_01.txt", words=45):
    path = tmp_path / name
    path.write_text(" ".join(f"w{i}" for i in range(words)), encoding="utf-8")
    return path


# ------------------------------------------------------------------ happy path


def test_single_chunk_document_produces_result(tmp_path):
    path = write_doc(tmp_path, words=5)
    pipeline = IngestionPipeline(make_extractor([VALID]), FAST)

    result = pipeline.run(path, doc_type="resume")

    assert result.source_id == "cand_01"
    assert result.doc_type == "resume"
    assert [s.skill for s in result.skills] == ["Python"]
    assert result.chunk_count == 1
    assert result.is_complete


def test_multi_chunk_document_extracts_every_chunk_and_merges(tmp_path):
    # 45 words at chunk_words=10 / overlap=2 -> 6 chunks.
    path = write_doc(tmp_path, words=45)
    extractor = make_extractor([VALID, VALID_DOCKER] * 3)
    pipeline = IngestionPipeline(extractor, FAST)

    result = pipeline.run(path)

    assert result.chunk_count == 6
    assert extractor._client.calls == 6
    # merge_skills collapses the repeats across chunks into two skills.
    assert sorted(s.skill for s in result.skills) == ["Docker", "Python"]
    assert result.is_complete


def test_empty_document_finalizes_instead_of_erroring(tmp_path):
    path = tmp_path / "blank.txt"
    path.write_text("   \n\n  ", encoding="utf-8")
    extractor = make_extractor([])
    result = IngestionPipeline(extractor, FAST).run(path)

    assert result is not None
    assert result.chunk_count == 0
    assert result.skills == []
    assert extractor._client.calls == 0


# ------------------------------------------------------- retry / backoff node


def test_transient_failure_is_retried_then_succeeds(tmp_path):
    path = write_doc(tmp_path, words=5)
    extractor = make_extractor([ConnectionError("503 unavailable"), VALID])
    pipeline = IngestionPipeline(extractor, FAST)

    result = pipeline.run(path)

    assert extractor._client.calls == 2
    assert [s.skill for s in result.skills] == ["Python"]
    assert result.is_complete


def test_exhausted_retries_record_the_chunk_and_continue(tmp_path):
    """A poisoned chunk must not stall the document or vanish silently."""
    path = write_doc(tmp_path, words=17)  # -> 2 chunks
    extractor = make_extractor(["not json", "not json", "not json", VALID])
    pipeline = IngestionPipeline(extractor, FAST)

    result = pipeline.run(path)

    assert result.failed_chunks == [0]          # recorded, per A1.3
    assert [s.skill for s in result.skills] == ["Python"]  # chunk 1 still ran
    assert result.is_complete is False


def test_fatal_error_is_not_retried(tmp_path):
    """An auth failure burns one call, not `max_attempts` of them."""
    path = write_doc(tmp_path, words=5)
    extractor = make_extractor([PermissionError("401 unauthorized: bad api_key")])
    pipeline = IngestionPipeline(extractor, FAST)

    result = pipeline.run(path)

    assert extractor._client.calls == 1
    assert result.failed_chunks == [0]


def test_backoff_delay_grows_between_attempts(tmp_path, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("synapse.ingest.pipeline.time.sleep", slept.append)

    path = write_doc(tmp_path, words=5)
    config = IngestionConfig(chunk_words=10, overlap_words=2, max_attempts=3, sleep=True)
    extractor = make_extractor([TimeoutError("timeout"), TimeoutError("timeout"), VALID])

    IngestionPipeline(extractor, config).run(path)

    assert len(slept) == 2
    assert slept[1] > slept[0]  # exponential, jitter is small enough not to invert


# ---------------------------------------------------------------- read failure


def test_unreadable_file_returns_none_without_killing_the_batch(tmp_path):
    good = write_doc(tmp_path, "cand_ok.txt", words=5)
    bad = tmp_path / "cand_bad.pdf"
    bad.write_bytes(b"%PDF-1.4")

    pipeline = IngestionPipeline(make_extractor([VALID]), FAST)

    assert pipeline.run(bad) is None
    results = pipeline.run_batch([bad, good])
    assert [r.source_id for r in results] == ["cand_ok"]


def test_read_failure_is_recorded_in_state(tmp_path):
    bad = tmp_path / "x.pdf"
    bad.write_bytes(b"%PDF")
    graph = build_ingestion_graph(make_extractor([]), FAST)

    final = graph.invoke(initial_state(bad))

    assert final["status"] == STATUS_READ_FAILED
    assert "ValueError" in final["last_error"]
    assert final["result"] is None


# ------------------------------------------------------------ C3.3 checkpoints


def test_progress_is_checkpointed_and_resumed_after_a_crash(tmp_path):
    """The C3.3 guarantee: chunks already paid for are not re-extracted."""
    path = write_doc(tmp_path, words=45)  # 6 chunks
    checkpointer = make_checkpointer(tmp_path / "ck.sqlite")

    # First run dies on chunk 2 with an error the extractor cannot swallow.
    boom = RuntimeError("process killed mid-batch")

    class Exploding(FakeClient):
        def generate_content(self, **kwargs):
            if self.calls == 2:
                self.calls += 1
                raise boom
            return super().generate_content(**kwargs)

    def explode_check(exc):
        raise exc  # surface it out of the node, like a hard crash would

    first = SkillExtractor(client=Exploding([VALID] * 6), rpm=0)
    first.is_retryable = explode_check
    p1 = IngestionPipeline(first, FAST, checkpointer=checkpointer)
    with pytest.raises(RuntimeError):
        p1.run(path)

    # Two chunks committed before the crash.
    state = p1.graph.get_state(
        {"configurable": {"thread_id": default_thread_id(path)}}
    )
    assert state.values["cursor"] == 2
    assert state.next, "an interrupted run must leave pending work to resume"

    # Second run reuses the same checkpoint and only extracts what remains.
    second = make_extractor([VALID] * 6)
    p2 = IngestionPipeline(second, FAST, checkpointer=checkpointer)
    result = p2.run(path)

    assert second._client.calls == 4, "chunks 0 and 1 must not be re-extracted"
    assert result.chunk_count == 6
    assert result.is_complete


def test_force_restart_ignores_the_checkpoint(tmp_path):
    path = write_doc(tmp_path, words=17)  # 2 chunks
    checkpointer = make_checkpointer(tmp_path / "ck.sqlite")

    p1 = IngestionPipeline(make_extractor([VALID] * 2), FAST, checkpointer=checkpointer)
    p1.run(path)

    second = make_extractor([VALID] * 2)
    IngestionPipeline(second, FAST, checkpointer=checkpointer).run(
        path, force_restart=True
    )
    assert second._client.calls == 2


def test_sqlite_checkpoint_survives_a_new_saver_instance(tmp_path):
    """A restarted *process* (not just a new pipeline) must still resume."""
    path = write_doc(tmp_path, words=45)
    db = tmp_path / "ck.sqlite"

    class Exploding(FakeClient):
        def generate_content(self, **kwargs):
            if self.calls == 3:
                self.calls += 1
                raise RuntimeError("killed")
            return super().generate_content(**kwargs)

    first = SkillExtractor(client=Exploding([VALID] * 6), rpm=0)
    first.is_retryable = lambda exc: (_ for _ in ()).throw(exc)
    with pytest.raises(RuntimeError):
        IngestionPipeline(first, FAST, checkpointer=make_checkpointer(db)).run(path)

    # Fresh saver over the same file, as a new process would build.
    second = make_extractor([VALID] * 6)
    result = IngestionPipeline(second, FAST, checkpointer=make_checkpointer(db)).run(path)

    assert second._client.calls == 3
    assert result.chunk_count == 6


# ------------------------------------------------------------------- batching


def test_run_batch_gives_each_document_its_own_thread(tmp_path):
    a = write_doc(tmp_path, "a.txt", words=5)
    b = write_doc(tmp_path, "b.txt", words=5)
    pipeline = IngestionPipeline(
        make_extractor([VALID, VALID_DOCKER]), FAST,
        checkpointer=make_checkpointer(tmp_path / "ck.sqlite"),
    )

    results = pipeline.run_batch([a, b], doc_type="resume")

    assert [r.source_id for r in results] == ["a", "b"]
    assert [s.skill for s in results[0].skills] == ["Python"]
    assert [s.skill for s in results[1].skills] == ["Docker"]


def test_same_stem_different_suffix_get_separate_threads(tmp_path):
    """data/samples ships ravi_backend.txt AND ravi_backend.docx - the stem alone
    would make them share a checkpoint thread and resume each other's work."""
    txt = tmp_path / "ravi_backend.txt"
    md = tmp_path / "ravi_backend.md"
    txt.write_text("react", encoding="utf-8")
    md.write_text("react", encoding="utf-8")

    assert default_thread_id(txt) != default_thread_id(md)
    assert default_thread_id(txt).startswith("ravi_backend-")
    # Stable across calls, so a restarted process resumes the same thread.
    assert default_thread_id(txt) == default_thread_id(str(txt))


def test_partial_status_when_a_chunk_never_validates(tmp_path):
    path = write_doc(tmp_path, words=5)
    graph = build_ingestion_graph(make_extractor(["nope"] * 3), FAST)

    final = graph.invoke(initial_state(path, "resume"))

    assert final["status"] == STATUS_PARTIAL
    assert final["result"]["failed_chunks"] == [0]
