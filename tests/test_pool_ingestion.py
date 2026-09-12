"""Phase E3: the link and persist nodes.

No network and no database. A stub store records what would have been written,
so what is under test is the graph's routing and failure accounting rather than
Cypher (which is covered separately).
"""

from __future__ import annotations

import json

import pytest

from synapse.ingest.extractor import SkillExtractor
from synapse.ingest.pipeline import (
    STATUS_COMPLETE,
    STATUS_LINK_FAILED,
    STATUS_PERSIST_FAILED,
    IngestionConfig,
    IngestionPipeline,
    build_ingestion_graph,
    initial_state,
    make_checkpointer,
)
from synapse.matching.entity_linker import EntityLinker

NODES = ["Docker", "Kubernetes", "Python", "React"]

PAYLOAD = json.dumps([
    {"skill": "Docker", "weight": 1.5, "context": "ran Docker in production"},
    {"skill": "k8s", "weight": 1.0, "context": "managed clusters"},
    {"skill": "quantum basket weaving", "weight": 1.0, "context": "hobby"},
])

FAST = IngestionConfig(chunk_words=50, overlap_words=5, max_attempts=2,
                       sleep=False, chunks_per_call=3)


class FakeClient:
    def __init__(self, payloads=None):
        self.payloads = list(payloads or [])
        self.calls = 0
        self.models = self

    def generate_content(self, **kwargs):
        self.calls += 1
        payload = self.payloads.pop(0) if self.payloads else PAYLOAD
        if isinstance(payload, Exception):
            raise payload

        class R:
            text = payload

        return R()


class StubStore:
    """Records writes; can be told to fail a given number of times first."""

    def __init__(self, fail_times: int = 0, error: Exception | None = None):
        self.written: list[dict] = []
        self.fail_times = fail_times
        self.error = error or ConnectionError("503 service unavailable")
        self.attempts = 0

    def upsert_candidate(self, **kwargs):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise self.error
        self.written.append(kwargs)


@pytest.fixture(autouse=True)
def _stub_types(monkeypatch):
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


@pytest.fixture
def linker():
    # Alias and surface layers only - deterministic, and no model download.
    return EntityLinker(NODES, use_embeddings=False)


def extractor(payloads=None):
    return SkillExtractor(client=FakeClient(payloads), rpm=0)


def write_doc(tmp_path, name="cand_01.txt"):
    path = tmp_path / name
    path.write_text("Docker and k8s and quantum basket weaving.", encoding="utf-8")
    return path


# ------------------------------------------------------------ happy path


def test_extract_link_persist(tmp_path, linker):
    store = StubStore()
    path = write_doc(tmp_path)

    result = IngestionPipeline(extractor(), FAST, linker=linker, store=store).run(
        path, doc_type="resume")

    assert result is not None
    assert len(store.written) == 1
    written = store.written[0]
    assert written["candidate_id"] == "cand_01"
    assert written["doc_type"] == "resume"
    assert written["content_hash"], "dedupe key must be populated"

    nodes = {s["node"] for s in written["skills"]}
    assert nodes == {"Docker", "Kubernetes"}, "'k8s' must canonicalize"
    # The unresolvable surface is recorded, not invented as a node.
    assert written["unresolved"] == ["quantum basket weaving"]


def test_link_provenance_travels_with_each_skill(tmp_path, linker):
    store = StubStore()
    IngestionPipeline(extractor(), FAST, linker=linker, store=store).run(
        write_doc(tmp_path))

    by_node = {s["node"]: s for s in store.written[0]["skills"]}
    assert by_node["Kubernetes"]["method"] == "alias"      # via 'k8s'
    assert by_node["Docker"]["method"] == "surface"
    # Extraction's justifying context survives into the stored profile.
    assert "Docker" in by_node["Docker"]["context"]
    assert by_node["Kubernetes"]["weight"] == 1.0


def test_content_hash_is_stable_for_identical_text(tmp_path, linker):
    a, b = write_doc(tmp_path, "a.txt"), write_doc(tmp_path, "b.txt")
    store = StubStore()
    pipe = IngestionPipeline(extractor(), FAST, linker=linker, store=store)
    pipe.run(a)
    pipe.run(b)

    hashes = {w["content_hash"] for w in store.written}
    assert len(hashes) == 1, "same text must hash the same, whatever the filename"


def test_without_a_store_nothing_is_written(tmp_path, linker):
    """The Phase C3 shape must survive: read, extract, return, no database."""
    result = IngestionPipeline(extractor(), FAST, linker=linker).run(write_doc(tmp_path))
    assert result is not None
    assert result.is_complete


def test_without_a_linker_the_graph_skips_straight_to_finalize(tmp_path):
    store = StubStore()
    result = IngestionPipeline(extractor(), FAST, store=store).run(write_doc(tmp_path))
    assert result is not None
    assert store.written == [], "no linking means nothing canonical to persist"


# --------------------------------------------------------- failure handling


def test_transient_persist_failure_is_retried(tmp_path, linker):
    store = StubStore(fail_times=1)
    result = IngestionPipeline(extractor(), FAST, linker=linker, store=store).run(
        write_doc(tmp_path))

    assert store.attempts == 2, "one failure, then a successful retry"
    assert len(store.written) == 1
    assert result is not None


def test_exhausted_persist_retries_keep_the_extraction(tmp_path, linker):
    """The write failed; the extraction must not be thrown away with it."""
    store = StubStore(fail_times=99)
    graph = build_ingestion_graph(extractor(), FAST, linker=linker, store=store)

    final = graph.invoke(initial_state(write_doc(tmp_path), "resume"))

    assert final["status"] == STATUS_PERSIST_FAILED
    assert final["persisted"] is False
    assert store.written == []
    # ...but the result object exists and holds the skills.
    assert final["result"] is not None
    assert final["result"]["skills"], "extraction survives a failed write"


def test_fatal_persist_failure_is_not_retried(tmp_path, linker):
    store = StubStore(fail_times=99, error=PermissionError("unauthorized"))
    graph = build_ingestion_graph(extractor(), FAST, linker=linker, store=store)

    final = graph.invoke(initial_state(write_doc(tmp_path), "resume"))

    assert final["status"] == STATUS_PERSIST_FAILED
    assert store.attempts == 1, "auth failures must not burn the retry budget"


def test_link_failure_is_terminal_but_keeps_the_extraction(tmp_path):
    class BrokenLinker:
        def link_many(self, *a, **kw):
            raise RuntimeError("embedder exploded")

    store = StubStore()
    graph = build_ingestion_graph(extractor(), FAST, linker=BrokenLinker(), store=store)

    final = graph.invoke(initial_state(write_doc(tmp_path), "resume"))

    assert final["status"] == STATUS_LINK_FAILED
    assert store.written == [], "nothing canonical to write"
    assert final["result"]["skills"], "extraction survives a failed link"


def test_status_is_complete_on_the_full_path(tmp_path, linker):
    store = StubStore()
    graph = build_ingestion_graph(extractor(), FAST, linker=linker, store=store)
    final = graph.invoke(initial_state(write_doc(tmp_path), "resume"))
    assert final["status"] == STATUS_COMPLETE
    assert final["persisted"] is True


# --------------------------------------------------------------- checkpoints


def test_a_failed_write_resumes_without_re_extracting(tmp_path, linker):
    """The cost asymmetry made concrete: never re-pay the LLM for a DB blip."""
    path = write_doc(tmp_path)
    ck = make_checkpointer(tmp_path / "ck.sqlite")

    class Killed(StubStore):
        """KeyboardInterrupt is a BaseException, so it escapes the persist
        node's `except Exception` and kills the run the way a real SIGINT or an
        OOM would - which is the scenario the checkpoint exists for."""

        def upsert_candidate(self, **kwargs):
            self.attempts += 1
            raise KeyboardInterrupt("process killed during the write")

    first_ex = extractor()
    first = IngestionPipeline(first_ex, FAST, checkpointer=ck,
                              linker=linker, store=Killed())
    with pytest.raises(KeyboardInterrupt):
        first.run(path)

    assert first_ex._client.calls >= 1, "the first run did pay for extraction"

    second_ex = extractor()
    second_store = StubStore()
    IngestionPipeline(second_ex, FAST, checkpointer=ck,
                      linker=linker, store=second_store).run(path)

    assert second_ex._client.calls == 0, "resume must not call the LLM again"
    assert len(second_store.written) == 1
