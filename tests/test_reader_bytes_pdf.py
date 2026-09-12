"""Phase F1: the upload entry point, and PDF support.

An HTTP upload hands you bytes and a filename, never a path. These pin that the
bytes path produces exactly what the disk path produces, that PDFs are readable
at all, and that one broken file cannot take down an upload batch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synapse.ingest.reader import (
    SUPPORTED_SUFFIXES,
    read_bytes,
    read_document,
)
from synapse.ingest.resume import docx_text, read_resume


# --------------------------------------------------------------- fixtures


def make_docx_with_table(path: Path) -> Path:
    """A .docx whose skills live in a TABLE, as real résumés often do."""
    import docx

    document = docx.Document()
    document.add_paragraph("Ravi Kumar — Backend Engineer")
    document.add_paragraph("Built services and shipped them.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Languages"
    table.cell(0, 1).text = "Python, Go"
    table.cell(1, 0).text = "Infrastructure"
    table.cell(1, 1).text = "Docker, Kubernetes"
    document.save(str(path))
    return path


def make_pdf(path: Path, text: str = "Docker and Kubernetes and Python") -> Path:
    """A minimal single-page PDF with a real text layer.

    Built by hand rather than with a PDF library because none is a dependency
    here and the project bans heavy ones. Byte offsets for the xref table are
    computed as the file is assembled - a PDF without a valid xref is exactly
    what `test_corrupt_pdf_is_a_value_error` covers, so this one has to be real.
    """
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    bodies = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj" + body + b"endobj\n"

    xref_at = len(out)
    size = len(bodies) + 1
    out += b"xref\n0 " + str(size).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (b"trailer<</Size " + str(size).encode() + b"/Root 1 0 R>>\nstartxref\n"
            + str(xref_at).encode() + b"\n%%EOF\n")

    path.write_bytes(bytes(out))
    return path


# ------------------------------------------------------------ docx tables


def test_docx_tables_reach_the_bytes_path(tmp_path):
    """The F1 fix: `read_resume` used to read paragraphs only.

    A résumé that keeps its skills in a table would have arrived at the
    extractor with the skills section missing, but only when uploaded - the
    disk path always read tables. Which code path a file came in on decided
    what the model got to see.
    """
    path = make_docx_with_table(tmp_path / "cand.docx")
    text = read_resume(path.read_bytes(), "cand.docx")

    assert "Kubernetes" in text, "table content must be extracted"
    assert "Languages | Python, Go" in text


def test_docx_bytes_and_disk_agree(tmp_path):
    path = make_docx_with_table(tmp_path / "cand.docx")
    assert read_bytes(path.read_bytes(), "cand.docx").text == read_document(path).text


def test_docx_text_accepts_a_path_or_a_stream(tmp_path):
    from io import BytesIO

    path = make_docx_with_table(tmp_path / "cand.docx")
    assert docx_text(str(path)) == docx_text(BytesIO(path.read_bytes()))


# -------------------------------------------------------------------- pdf


def test_pdf_is_supported():
    assert ".pdf" in SUPPORTED_SUFFIXES


def test_pdf_text_is_extracted(tmp_path):
    path = make_pdf(tmp_path / "cand.pdf")
    document = read_bytes(path.read_bytes(), "cand.pdf", doc_type="resume")

    assert "Docker" in document.text
    assert document.chunks, "a PDF with text must produce chunks"


def test_pdf_bytes_and_disk_agree(tmp_path):
    path = make_pdf(tmp_path / "cand.pdf")
    assert read_bytes(path.read_bytes(), "cand.pdf").text == read_document(path).text


def test_pdf_flows_through_the_pipeline(tmp_path):
    """End to end: a PDF on disk must be ingestable, not just parseable."""
    path = make_pdf(tmp_path / "cand.pdf")
    document = read_document(path, doc_type="resume")
    assert document.source_id == "cand"
    assert document.doc_type == "resume"
    assert len(document.chunks) >= 1


# --------------------------------------------------------- read_bytes shape


def test_read_bytes_matches_read_document_for_plain_text(tmp_path):
    path = tmp_path / "cand_01.txt"
    path.write_text("Senior engineer.  Built   Kubernetes platforms.", encoding="utf-8")

    from_disk = read_document(path, doc_type="resume")
    from_bytes = read_bytes(path.read_bytes(), "cand_01.txt", doc_type="resume")

    assert from_bytes.text == from_disk.text
    assert from_bytes.source_id == from_disk.source_id == "cand_01"
    assert from_bytes.doc_type == "resume"
    assert [c.text for c in from_bytes.chunks] == [c.text for c in from_disk.chunks]


def test_source_id_comes_from_the_filename_stem():
    doc = read_bytes(b"hello world", "Ravi Kumar CV.txt")
    assert doc.source_id == "Ravi Kumar CV"


def test_read_bytes_writes_nothing_to_disk(tmp_path, monkeypatch):
    """No temp file: a spooled path would also break checkpoint identity."""
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    read_bytes(b"Docker and Python", "cand.txt")
    assert set(tmp_path.iterdir()) == before


def test_chunking_options_are_honoured():
    text = " ".join(f"w{i}" for i in range(100)).encode()
    doc = read_bytes(text, "big.txt", chunk_words=20, overlap_words=5)
    assert len(doc.chunks) > 1
    assert all(len(c.text.split()) <= 20 for c in doc.chunks)


# ------------------------------------------------------------- bad input


def test_unsupported_suffix_rejected_on_the_bytes_path():
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_bytes(b"PK\x03\x04", "sheet.xlsx")


def test_corrupt_pdf_is_a_value_error(tmp_path):
    """One broken upload must not abort a batch of fifty."""
    with pytest.raises(ValueError):
        read_bytes(b"%PDF-1.4 definitely not a pdf", "broken.pdf")


def test_corrupt_docx_is_a_value_error():
    with pytest.raises(ValueError):
        read_bytes(b"not a zip archive at all", "broken.docx")


def test_empty_upload_is_a_clean_failure_not_a_crash():
    with pytest.raises(ValueError):
        read_bytes(b"", "empty.pdf")


def test_text_with_undecodable_bytes_survives():
    """`replace`, not `ignore` - a bad byte leaves a mark instead of vanishing."""
    doc = read_bytes(b"Docker \xff\xfe Kubernetes", "cand.txt")
    assert "Docker" in doc.text and "Kubernetes" in doc.text
