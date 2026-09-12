"""Document text extraction. The single place a resume becomes a string.

Both entry points into the pipeline go through here: `reader.read_document`
opens a path, `reader.read_bytes` takes an upload, and both need identical
parsing. Keeping one implementation is the point - there were previously two
`.docx` readers, and only one of them read tables, so which code path a résumé
arrived on decided whether its skills section was seen at all.

Every function accepts a path *or* a file-like object, because `python-docx` and
`pypdf` both do, and an upload only ever has bytes.
"""

from io import BytesIO

# Import lazily inside each reader so the module stays importable on a machine
# without python-docx or pypdf installed; only opening that format needs them.


def pdf_text(source) -> str:
    """Text from a PDF. `source` is a path or a file-like object."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "Reading .pdf requires pypdf. Install it with `pip install pypdf`."
        ) from exc

    reader = PdfReader(source)
    # extract_text() returns None for image-only pages, so guard with "".
    # A scanned résumé therefore yields an empty document rather than crashing -
    # the caller sees zero chunks and can say so, which is the honest outcome
    # when there is no text layer to read.
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def docx_text(source) -> str:
    """Text from a .docx, INCLUDING tables.

    Résumés routinely put the entire skills section inside a table, and
    `python-docx`'s paragraph iteration skips table content completely. Reading
    paragraphs alone silently drops the most important part of a large fraction
    of real documents.
    """
    try:
        import docx  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "Reading .docx requires python-docx. Install it with `pip install python-docx`."
        ) from exc

    document = docx.Document(source)
    parts: list[str] = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def read_resume(data: bytes, filename: str) -> str:
    """Extract raw text from uploaded bytes. Supports .pdf, .docx, .txt/.md."""
    name = filename.lower()

    if name.endswith(".pdf"):
        return pdf_text(BytesIO(data))

    if name.endswith(".docx"):
        return docx_text(BytesIO(data))

    # Fallback: plain text. `replace` rather than `ignore` so a mis-encoded byte
    # leaves a visible placeholder instead of silently vanishing mid-word.
    return data.decode("utf-8", errors="replace")


if __name__ == "__main__":
    import glob
    for path in sorted(glob.glob("data/samples/ravi_backend.*")):
        with open(path, "rb") as f:
            text = read_resume(f.read(), path)
        print(f"\n=== {path}  ({len(text)} chars) ===")
        print(text[:220])
