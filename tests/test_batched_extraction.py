"""Phase E2: N chunks per Gemini call.

The free tier caps requests per minute, not tokens, so what matters is call
*count*. These tests pin that grouping actually reduces it, and that the failure
accounting stays honest when one call covers several chunks.
"""

from __future__ import annotations

import json

import pytest

from synapse.ingest.extractor import SkillExtractor
from synapse.ingest.pipeline import IngestionConfig, IngestionPipeline

VALID = json.dumps([{"skill": "Python", "weight": 1.5, "context": "led Python work"}])
TWO = json.dumps([
    {"skill": "Docker", "weight": 1.0, "context": "containerised services"},
    {"skill": "Kubernetes", "weight": 1.5, "context": "ran clusters"},
])


class RecordingClient:
    """Records the prompt of every call so grouping is directly observable."""

    def __init__(self, payloads=None):
        self.payloads = list(payloads or [])
        self.calls = 0
        self.prompts: list[str] = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls += 1
        self.prompts.append(kwargs.get("contents", ""))
        payload = self.payloads.pop(0) if self.payloads else VALID
        if isinstance(payload, Exception):
            raise payload

        class R:
            text = payload

        return R()


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


def extractor(payloads=None):
    return SkillExtractor(client=RecordingClient(payloads), rpm=0)


def write_doc(tmp_path, words=45, name="cand.txt"):
    path = tmp_path / name
    path.write_text(" ".join(f"w{i}" for i in range(words)), encoding="utf-8")
    return path


def config(chunks_per_call, **kw):
    return IngestionConfig(chunk_words=10, overlap_words=2, max_attempts=3,
                           sleep=False, chunks_per_call=chunks_per_call, **kw)


# ------------------------------------------------------- extract_batch itself


def test_batch_of_one_is_a_plain_call():
    ex = extractor([VALID])
    ex.extract_batch(["only chunk"])
    assert ex._client.calls == 1
    # No section headers for a single chunk - nothing to delimit.
    assert "SECTION" not in ex._client.prompts[0]


def test_batch_sends_all_chunks_in_one_call():
    ex = extractor([TWO])
    skills = ex.extract_batch(["alpha text", "beta text", "gamma text"])

    assert ex._client.calls == 1, "three chunks must cost one call"
    prompt = ex._client.prompts[0]
    for fragment in ("alpha text", "beta text", "gamma text"):
        assert fragment in prompt
    assert prompt.count("--- SECTION") == 3
    assert {s.skill for s in skills} == {"Docker", "Kubernetes"}


def test_empty_batch_makes_no_call():
    ex = extractor()
    assert ex.extract_batch([]) == []
    assert ex._client.calls == 0


def test_batch_output_is_still_schema_validated():
    ex = extractor(["not json at all"])
    with pytest.raises(Exception):
        ex.extract_batch(["a", "b"])


# ------------------------------------------------------ through the pipeline


def test_grouping_reduces_call_count(tmp_path):
    """The whole point of E2, stated as a number."""
    path = write_doc(tmp_path, words=45)          # 6 chunks

    one = extractor()
    IngestionPipeline(one, config(1)).run(path)

    three = extractor()
    IngestionPipeline(three, config(3)).run(path)

    assert one._client.calls == 6
    assert three._client.calls == 2, "6 chunks at 3 per call"
    assert three._client.calls < one._client.calls


@pytest.mark.parametrize("per_call,expected", [(1, 6), (2, 3), (3, 2), (4, 2), (6, 1), (99, 1)])
def test_call_count_for_various_group_sizes(tmp_path, per_call, expected):
    path = write_doc(tmp_path, words=45)          # 6 chunks
    ex = extractor()
    IngestionPipeline(ex, config(per_call)).run(path)
    assert ex._client.calls == expected


def test_ragged_final_group_is_handled(tmp_path):
    """6 chunks at 4 per call leaves a group of 2 - it must still be extracted."""
    path = write_doc(tmp_path, words=45)
    ex = extractor()
    result = IngestionPipeline(ex, config(4)).run(path)

    assert result.chunk_count == 6
    assert result.is_complete
    assert ex._client.calls == 2
    assert ex._client.prompts[1].count("--- SECTION") == 2


def test_zero_or_negative_group_size_falls_back_to_one(tmp_path):
    """A misconfiguration must not produce an empty group and spin forever."""
    path = write_doc(tmp_path, words=17)          # 2 chunks
    ex = extractor()
    result = IngestionPipeline(ex, config(0)).run(path)
    assert result.chunk_count == 2
    assert ex._client.calls == 2


def test_skills_still_merge_across_groups(tmp_path):
    path = write_doc(tmp_path, words=45)
    ex = extractor([TWO, TWO])
    result = IngestionPipeline(ex, config(3)).run(path)

    assert sorted(s.skill for s in result.skills) == ["Docker", "Kubernetes"]
    assert result.is_complete


# ----------------------------------------------------- failure accounting


def test_a_failed_group_records_every_chunk_in_it(tmp_path):
    """One call covered 3 chunks, so its failure cost all 3 - say so."""
    path = write_doc(tmp_path, words=45)          # 6 chunks, 2 groups
    # First group fails all its attempts; second group succeeds.
    ex = extractor(["bad", "bad", "bad", VALID])
    result = IngestionPipeline(ex, config(3)).run(path)

    assert result.failed_chunks == [0, 1, 2], "the whole group is accounted for"
    assert [s.skill for s in result.skills] == ["Python"]   # group 2 still ran
    assert result.is_complete is False


def test_transient_failure_retries_the_whole_group(tmp_path):
    path = write_doc(tmp_path, words=45)
    ex = extractor([ConnectionError("503 unavailable"), VALID, VALID])
    result = IngestionPipeline(ex, config(3)).run(path)

    assert result.is_complete
    assert result.failed_chunks == []
    # one failed call, then the same group retried, then the second group
    assert ex._client.calls == 3


def test_fatal_failure_skips_the_group_without_retrying(tmp_path):
    path = write_doc(tmp_path, words=45)
    ex = extractor([PermissionError("401 unauthorized: bad api_key"), VALID])
    result = IngestionPipeline(ex, config(3)).run(path)

    assert result.failed_chunks == [0, 1, 2]
    assert ex._client.calls == 2, "no retry on a fatal error"


def test_checkpoint_resumes_at_group_granularity(tmp_path):
    """Groups already paid for must not be re-extracted after a crash."""
    from synapse.ingest.pipeline import make_checkpointer

    path = write_doc(tmp_path, words=45)          # 6 chunks, 2 groups of 3
    ck = make_checkpointer(tmp_path / "ck.sqlite")

    class Exploding(RecordingClient):
        def generate_content(self, **kwargs):
            if self.calls == 1:                    # blow up on the SECOND group
                self.calls += 1
                raise RuntimeError("killed mid-batch")
            return super().generate_content(**kwargs)

    first = SkillExtractor(client=Exploding(), rpm=0)
    first.is_retryable = lambda exc: (_ for _ in ()).throw(exc)
    with pytest.raises(RuntimeError):
        IngestionPipeline(first, config(3), checkpointer=ck).run(path)

    second = extractor()
    result = IngestionPipeline(second, config(3), checkpointer=ck).run(path)

    assert second._client.calls == 1, "the first group must not be re-extracted"
    assert result.chunk_count == 6
    assert result.is_complete
