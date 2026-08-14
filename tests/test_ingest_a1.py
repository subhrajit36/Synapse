"""Phase A1 unit tests. No network, no API key required."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from synapse.ingest.extractor import ExtractionError, SkillExtractor
from synapse.ingest.reader import chunk_text, normalize_text, read_document
from synapse.ingest.schemas import ExtractedSkill, merge_skills


# ------------------------------------------------------------------ schemas


def test_weight_bounds_are_enforced():
    ExtractedSkill(skill="Python", weight=1.5, context="5 years of Python")
    for bad in (0.4, 1.6, -1.0):
        with pytest.raises(ValidationError):
            ExtractedSkill(skill="Python", weight=bad, context="x")


def test_blank_skill_rejected():
    with pytest.raises(ValidationError):
        ExtractedSkill(skill="   ", weight=1.0, context="x")


def test_whitespace_is_normalized():
    s = ExtractedSkill(skill="  Apache   Kafka \n", weight=1.0, context=" ran  Kafka ")
    assert s.skill == "Apache Kafka"
    assert s.context == "ran Kafka"


def test_merge_keeps_highest_weight_and_its_context():
    merged = merge_skills(
        [
            ExtractedSkill(skill="Python", weight=0.5, context="listed in skills"),
            ExtractedSkill(skill="python", weight=1.5, context="led Python migration"),
            ExtractedSkill(skill="Docker", weight=1.0, context="containerised services"),
        ]
    )
    assert len(merged) == 2
    python = next(s for s in merged if s.key == "python")
    assert python.weight == 1.5
    assert "migration" in python.context
    # sorted by descending weight
    assert merged[0].weight >= merged[1].weight


# ------------------------------------------------------------------- reader


def test_normalize_preserves_line_structure():
    assert normalize_text("a  b\n\n\n\nc   d") == "a b\n\nc d"


def test_chunking_overlaps_and_covers():
    text = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_text(text, "doc", chunk_words=400, overlap_words=50)
    assert len(chunks) == 3
    assert chunks[0].text.split()[-50:] == chunks[1].text.split()[:50]
    assert chunks[-1].text.split()[-1] == "w999"
    assert [c.index for c in chunks] == [0, 1, 2]


def test_empty_text_produces_no_chunks():
    assert chunk_text("   ", "doc") == []


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text("a b c", "doc", chunk_words=10, overlap_words=10)


def test_read_document_txt(tmp_path):
    p = tmp_path / "cand_01.txt"
    p.write_text("Senior engineer.  Built   Kubernetes platforms.", encoding="utf-8")
    doc = read_document(p, doc_type="resume")
    assert doc.source_id == "cand_01"
    assert doc.doc_type == "resume"
    assert len(doc.chunks) == 1
    assert "Kubernetes platforms" in doc.text


def test_unsupported_suffix_rejected(tmp_path):
    p = tmp_path / "resume.pdf"
    p.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError):
        read_document(p)


# ---------------------------------------------------------------- extractor


class FakeClient:
    """Stands in for genai.Client; returns queued payloads in order."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0
        self.models = self

    def generate_content(self, **kwargs):  # noqa: D401
        self.calls += 1
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload

        class R:
            text = payload

        return R()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("synapse.ingest.extractor.time.sleep", lambda *_: None)


@pytest.fixture(autouse=True)
def _stub_types(monkeypatch):
    """Avoid importing google.genai.types when the SDK is absent."""
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


VALID = json.dumps([{"skill": "Python", "weight": 1.5, "context": "led Python work"}])


def test_valid_payload_parses():
    ex = SkillExtractor(client=FakeClient([VALID]))
    skills = ex.extract_from_text("...")
    assert skills[0].skill == "Python"


def test_out_of_range_weight_is_retried_not_accepted():
    bad = json.dumps([{"skill": "Python", "weight": 9.0, "context": "x"}])
    ex = SkillExtractor(client=FakeClient([bad, VALID]), max_retries=3)
    skills = ex.extract_from_text("...")
    assert skills[0].weight == 1.5
    assert ex._client.calls == 2


def test_dict_wrapped_payload_tolerated():
    wrapped = json.dumps({"skills": json.loads(VALID)})
    ex = SkillExtractor(client=FakeClient([wrapped]))
    assert ex.extract_from_text("...")[0].skill == "Python"


def test_exhausted_retries_raise():
    ex = SkillExtractor(client=FakeClient(["not json"] * 3), max_retries=3)
    with pytest.raises(ExtractionError):
        ex.extract_from_text("...")


def test_failed_chunk_is_recorded_not_silently_dropped():
    from synapse.ingest.reader import Chunk

    chunks = [Chunk(0, "a", "doc"), Chunk(1, "b", "doc")]
    ex = SkillExtractor(client=FakeClient([VALID, "bad", "bad"]), max_retries=2)
    result = ex.extract_from_chunks(chunks, "doc", "resume")
    assert result.failed_chunks == [1]
    assert result.is_complete is False
    assert result.prompt_version and result.model


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        SkillExtractor()