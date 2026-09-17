"""Phase F3: one résumé into the pool over HTTP.

Three layers, no network, no Gemini, no database:

  * the pipeline's pre-read entry point and its hash-keyed checkpoint thread,
  * the engine's hash-first / reuse-on-hit decision,
  * the multipart route and its status codes.
"""

from __future__ import annotations

import io
import pickle

import networkx as nx
import pytest

from synapse.graph.neo4j_client import Neo4jConfig
from synapse.ingest.extractor import SkillExtractor
from synapse.ingest.pipeline import (
    STATUS_COMPLETE,
    IngestionConfig,
    IngestionPipeline,
    content_hash_of,
    make_checkpointer,
    upload_thread_id,
)
from synapse.ingest.reader import read_bytes
from synapse.matching.entity_linker import EntityLinker
from synapse.mcp.engine import UPLOAD_REUSED, MatchEngine, set_engine
from synapse.mcp.server import mcp

TEXT = b"Docker and k8s and quantum basket weaving."
NODES = ["Docker", "Kubernetes", "Python", "React"]
PAYLOAD = (
    '[{"skill": "Docker", "weight": 1.5, "context": "ran Docker"},'
    ' {"skill": "k8s", "weight": 1.0, "context": "clusters"},'
    ' {"skill": "quantum basket weaving", "weight": 1.0, "context": "hobby"}]'
)
FAST = IngestionConfig(chunk_words=50, overlap_words=5, max_attempts=2,
                       sleep=False, chunks_per_call=3)


# ------------------------------------------------------------------ fakes


class FakeGemini:
    def __init__(self):
        self.calls = 0
        self.models = self

    def generate_content(self, **kwargs):
        self.calls += 1

        class R:
            text = PAYLOAD

        return R()


class StubStore:
    """Enough of Neo4jClient for the upload path: hash lookup, batch attach,
    profile write and read-back."""

    def __init__(self, configured=True, existing=None):
        self.config = Neo4jConfig(uri="neo4j+s://stub", password="x" if configured else "")
        self.by_hash: dict[str, str] = dict(existing or {})
        self.profiles: dict[str, dict] = {}
        self.attached: list[tuple[str, str]] = []
        self.written: list[dict] = []
        self.pool_loads: list[str | None] = []

    def find_candidate_by_hash(self, content_hash):
        return self.by_hash.get(content_hash)

    def add_candidate_to_batch(self, candidate_id, batch_id):
        self.attached.append((candidate_id, batch_id))
        return True

    def get_candidate(self, candidate_id):
        return self.profiles.get(candidate_id, {
            "candidate_id": candidate_id, "name": "stored",
            "skills": [{"node": "Docker", "weight": 1.0}], "unresolved": ["x"],
            "failed_chunks": [],
        })

    def upsert_candidate(self, **kwargs):
        self.written.append(kwargs)
        self.by_hash[kwargs["content_hash"]] = kwargs["candidate_id"]

    def load_candidate_pool(self, batch_id=None):
        self.pool_loads.append(batch_id)
        return []

    def list_candidates(self, batch_id=None):
        return []


class FakePipeline:
    """Records what the engine asked for and returns a canned final state."""

    def __init__(self, status=STATUS_COMPLETE, persisted=True):
        self.calls: list[dict] = []
        self.status, self.persisted = status, persisted

    def run_document(self, document, batch_id="", candidate_id="", **kw):
        self.calls.append({"document": document, "batch_id": batch_id,
                           "candidate_id": candidate_id})
        return {
            "status": self.status, "persisted": self.persisted,
            "candidate_id": candidate_id, "linked": [{"node": "Docker"}],
            "unresolved": ["quantum basket weaving"], "failed_chunks": [],
        }


def extractor():
    return SkillExtractor(client=FakeGemini(), rpm=0)


def real_pipeline(store, checkpointer=None):
    return IngestionPipeline(
        extractor(), FAST, checkpointer=checkpointer,
        linker=EntityLinker(NODES, use_embeddings=False), store=store,
    )


def engine_with(store, pipeline, tmp_path):
    G = nx.Graph()
    for n in NODES:
        G.add_node(n, node_type="skill", category="t")
    path = tmp_path / "g.pkl"
    path.write_bytes(pickle.dumps(G))
    eng = MatchEngine(graph_path=path, neo4j_client=store, pipeline=pipeline)
    eng._linker = EntityLinker(NODES, use_embeddings=False)
    return eng


@pytest.fixture(autouse=True)
def _stub_genai(monkeypatch):
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


# --------------------------------------------------- pipeline: pre-read path


def test_run_document_never_touches_disk():
    """The upload's `path` is a bare filename that does not exist."""
    store = StubStore()
    doc = read_bytes(TEXT, "ravi.txt", doc_type="resume",
                     chunk_words=50, overlap_words=5)
    final = real_pipeline(store).run_document(doc, batch_id="b1", candidate_id="ravi-x")

    assert final["status"] == STATUS_COMPLETE
    assert final["persisted"] is True
    written = store.written[0]
    assert written["candidate_id"] == "ravi-x"
    assert written["batch_id"] == "b1"
    assert written["content_hash"] == content_hash_of(doc.text)
    assert {s["node"] for s in written["skills"]} == {"Docker", "Kubernetes"}


def test_thread_is_keyed_on_content_not_filename():
    a = read_bytes(TEXT, "ravi.txt")
    b = read_bytes(TEXT, "totally_different_name.md")
    assert content_hash_of(a.text) == content_hash_of(b.text)
    assert upload_thread_id(content_hash_of(a.text)) == upload_thread_id(content_hash_of(b.text))
    assert upload_thread_id(content_hash_of(a.text)).startswith("upload-")


def test_an_interrupted_upload_resumes_under_a_new_filename(tmp_path):
    """The locked decision, end to end: the LLM is never paid twice for one
    document, even when the retry arrives with a different name."""
    ck = make_checkpointer(tmp_path / "ck.sqlite")

    class Killed(StubStore):
        def upsert_candidate(self, **kwargs):
            raise KeyboardInterrupt("process died during the write")

    first = real_pipeline(Killed(), checkpointer=ck)
    with pytest.raises(KeyboardInterrupt):
        first.run_document(read_bytes(TEXT, "first.txt"), batch_id="b1")
    assert first.extractor._client.calls >= 1

    store = StubStore()
    second = real_pipeline(store, checkpointer=ck)
    final = second.run_document(read_bytes(TEXT, "second_try.txt"), batch_id="b1")

    assert second.extractor._client.calls == 0, "resume must not call the LLM"
    assert final["persisted"] is True
    assert len(store.written) == 1


def test_empty_document_finalizes_without_reaching_the_pool():
    """Existing graph behaviour, pinned: zero chunks routes read -> finalize,
    so nothing is linked or persisted. The engine screens this out earlier."""
    store = StubStore()
    doc = read_bytes(b"   \n  ", "blank.txt")
    final = real_pipeline(store).run_document(doc, batch_id="b1")
    assert final["status"] == STATUS_COMPLETE
    assert final["result"]["skills"] == []
    assert final["persisted"] is False
    assert store.written == []


# ------------------------------------------------------------ engine: dedupe


def test_new_document_runs_the_pipeline_under_the_batch(tmp_path):
    store, pipe = StubStore(), FakePipeline()
    eng = engine_with(store, pipe, tmp_path)

    r = eng.ingest_upload(TEXT, "ravi.txt", "b1")

    assert len(pipe.calls) == 1
    call = pipe.calls[0]
    assert call["batch_id"] == "b1"
    assert call["document"].source_id == "ravi"
    # Stem plus hash: two strangers' `resume.pdf` must not collide.
    assert call["candidate_id"].startswith("ravi-")
    assert call["candidate_id"] != "ravi"
    assert r.reused is False and r.in_pool is True
    assert r.status == STATUS_COMPLETE
    assert r.skill_count == 1
    assert r.unresolved == ["quantum basket weaving"]
    assert r.content_hash == content_hash_of(read_bytes(TEXT, "ravi.txt").text)


def test_known_document_is_attached_not_re_extracted(tmp_path):
    """Hash first, reuse on a hit: no pipeline run, no LLM call."""
    h = content_hash_of(read_bytes(TEXT, "x.txt").text)
    store, pipe = StubStore(existing={h: "ravi-old"}), FakePipeline()
    eng = engine_with(store, pipe, tmp_path)

    r = eng.ingest_upload(TEXT, "renamed.txt", "b2")

    assert pipe.calls == [], "the pipeline must not run on a hit"
    assert store.attached == [("ravi-old", "b2")]
    assert r.reused is True and r.in_pool is True
    assert r.status == UPLOAD_REUSED
    assert r.candidate_id == "ravi-old"
    assert r.skill_count == 1 and r.unresolved == ["x"]


def test_successful_upload_invalidates_the_batch_cache(tmp_path):
    store, pipe = StubStore(), FakePipeline()
    eng = engine_with(store, pipe, tmp_path)
    eng.pool_for("b1")
    eng.pool_for(None)
    eng.ingest_upload(TEXT, "ravi.txt", "b1")
    assert eng._pool == {}, "the batch and the unscoped view are both stale now"


def test_failed_persist_is_reported_and_not_in_pool(tmp_path):
    store, pipe = StubStore(), FakePipeline(status="persist_failed", persisted=False)
    eng = engine_with(store, pipe, tmp_path)
    eng.pool_for("b1")

    r = eng.ingest_upload(TEXT, "ravi.txt", "b1")
    assert r.in_pool is False
    assert r.status == "persist_failed"
    assert "b1" in eng._pool, "nothing changed in the pool; keep the cache"


def test_missing_batch_is_a_bad_request(tmp_path):
    eng = engine_with(StubStore(), FakePipeline(), tmp_path)
    with pytest.raises(ValueError, match="batch_id"):
        eng.ingest_upload(TEXT, "ravi.txt", "")


def test_unsupported_file_is_a_bad_request(tmp_path):
    eng = engine_with(StubStore(), FakePipeline(), tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        eng.ingest_upload(TEXT, "ravi.exe", "b1")


def test_textless_document_is_rejected_before_the_pipeline_runs(tmp_path):
    """A scanned PDF is a bad input, not a server failure, and must cost no
    LLM call and no checkpoint."""
    store, pipe = StubStore(), FakePipeline()
    eng = engine_with(store, pipe, tmp_path)
    with pytest.raises(ValueError, match="No readable text"):
        eng.ingest_upload(b"  \n\n ", "scan.txt", "b1")
    assert pipe.calls == []


def test_unconfigured_database_is_a_service_problem(tmp_path):
    eng = engine_with(StubStore(configured=False), FakePipeline(), tmp_path)
    with pytest.raises(RuntimeError, match="NEO4J_PASSWORD"):
        eng.ingest_upload(TEXT, "ravi.txt", "b1")


def test_pipeline_is_built_lazily_and_missing_api_key_is_a_503_not_a_400(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    eng = engine_with(StubStore(), None, tmp_path)
    assert eng._pipeline is None, "constructing the engine must not build it"
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        _ = eng.pipeline


# ------------------------------------------------------------------- route


@pytest.fixture
def web(tmp_path):
    from starlette.testclient import TestClient

    store, pipe = StubStore(), FakePipeline()
    eng = engine_with(store, pipe, tmp_path)
    set_engine(eng)
    try:
        with TestClient(mcp.http_app()) as client:
            yield client, store, pipe
    finally:
        set_engine(None)


def post(client, filename="ravi.txt", data=TEXT, **fields):
    files = {"file": (filename, io.BytesIO(data), "text/plain")}
    return client.post("/api/candidates", files=files, data=fields)


def test_upload_route_returns_the_outcome(web):
    client, store, pipe = web
    res = post(client, batch_id="b1")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["batch_id"] == "b1"
    assert body["in_pool"] is True and body["reused"] is False
    assert pipe.calls[0]["batch_id"] == "b1"


def test_upload_route_dedupes(web):
    client, store, pipe = web
    store.by_hash[content_hash_of(read_bytes(TEXT, "a.txt").text)] = "ravi-old"
    body = post(client, batch_id="b7").json()
    assert body["reused"] is True and body["status"] == UPLOAD_REUSED
    assert pipe.calls == []
    assert store.attached == [("ravi-old", "b7")]


def test_upload_route_rejects_missing_fields(web):
    client, _, _ = web
    assert post(client).status_code == 400                        # no batch_id
    assert client.post("/api/candidates", data={"batch_id": "b1"}).status_code == 400  # no file
    assert client.post("/api/candidates", json={"batch_id": "b1"}).status_code == 400  # not multipart


def test_upload_route_rejects_unsupported_files(web):
    client, _, _ = web
    res = post(client, filename="ravi.exe", batch_id="b1")
    assert res.status_code == 400
    assert "Unsupported" in res.json()["error"]


def test_upload_route_reports_a_candidate_that_missed_the_pool(tmp_path):
    from starlette.testclient import TestClient

    eng = engine_with(StubStore(), FakePipeline(status="persist_failed", persisted=False), tmp_path)
    set_engine(eng)
    try:
        with TestClient(mcp.http_app()) as client:
            res = post(client, batch_id="b1")
            assert res.status_code == 502
            assert res.json()["in_pool"] is False
    finally:
        set_engine(None)


def test_upload_route_is_503_without_a_database(tmp_path):
    from starlette.testclient import TestClient

    set_engine(engine_with(StubStore(configured=False), FakePipeline(), tmp_path))
    try:
        with TestClient(mcp.http_app()) as client:
            assert post(client, batch_id="b1").status_code == 503
    finally:
        set_engine(None)


def test_get_on_the_same_path_still_lists(web):
    """Adding POST must not shadow the E4 listing on the same URL."""
    client, _, _ = web
    assert client.get("/api/candidates").status_code == 200
