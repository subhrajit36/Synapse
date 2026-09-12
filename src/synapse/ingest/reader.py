"""Phase A1, Node 1: load a resume or JD and chunk it for extraction.

Two entry points, one behaviour:

  * `read_document(path)` - the batch/CLI path, reads a file from disk.
  * `read_bytes(data, filename)` - the upload path (F1), takes what an HTTP
    request hands you. A web upload has bytes and a filename, never a path.

Both delegate parsing to `resume.py` so the two can never disagree about what a
document contains, and both then normalise and chunk identically.

Deliberately dependency-light: `.txt`/`.md` need nothing; `.docx` and `.pdf`
import their parsers only when a file of that type is actually opened, so the
rest of the pipeline stays importable without them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .resume import docx_text, pdf_text

# ~500 tokens; English prose runs roughly 0.75 words per token, so ~375 words.
DEFAULT_CHUNK_WORDS = 375
DEFAULT_OVERLAP_WORDS = 40

# `.pdf` is here because it is what résumés actually arrive as. Parsing lives in
# `resume.py`; this set only decides what the pipeline will accept.
SUPPORTED_SUFFIXES = {".txt", ".md", ".docx", ".pdf"}

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANKLINES = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    source_id: str


@dataclass
class Document:
    source_id: str
    path: Path
    text: str
    doc_type: str = "unknown"
    chunks: list[Chunk] = field(default_factory=list)


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_text(source, suffix: str) -> str:
    """Dispatch one source to the right parser in `resume.py`.

    `source` is a path for the disk entry point and a BytesIO for the upload
    one; both parsers accept either, so the two paths share this dispatch and
    cannot drift apart.

    A parser failure is re-raised as `ValueError`. That is not cosmetic: a
    corrupt or password-protected PDF raises `pypdf.errors.PdfReadError`, which
    nothing upstream catches, so one bad file would abort an entire upload
    batch. As a `ValueError` it lands in the read node's existing handler and
    becomes a recorded `read_failed` for that document alone. Users will upload
    broken PDFs; that has to be survivable.

    `ImportError` passes through untouched - a missing parser is a deployment
    problem, not a bad document, and must not look like one.
    """
    try:
        if suffix == ".pdf":
            return pdf_text(source)
        if suffix == ".docx":
            return docx_text(source)
        if isinstance(source, Path):
            return _read_txt(source)
        return source.read().decode("utf-8", errors="replace")
    except ImportError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalised for the caller
        raise ValueError(
            f"Could not parse {suffix or 'file'}: {type(exc).__name__}: {exc}"
        ) from exc


def _check_suffix(suffix: str) -> None:
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type {suffix!r}; supported: {sorted(SUPPORTED_SUFFIXES)}"
        )


def normalize_text(raw: str) -> str:
    """Collapse intra-line whitespace and runs of blank lines; keep line structure.

    Line breaks are load-bearing in resumes (they separate bullets), so they are
    preserved rather than flattened into a single paragraph.
    """
    lines = [_WHITESPACE.sub(" ", line).strip() for line in raw.splitlines()]
    return _BLANKLINES.sub("\n\n", "\n".join(lines)).strip()


def chunk_text(
    text: str,
    source_id: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Split into overlapping word windows.

    Overlap exists so a skill named at a chunk boundary is not severed from the
    sentence that evidences it.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must be >= 0 and < chunk_words")

    words = text.split()
    if not words:
        return []

    stride = chunk_words - overlap_words
    chunks: list[Chunk] = []
    for start in range(0, len(words), stride):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(
            Chunk(index=len(chunks), text=" ".join(window), source_id=source_id)
        )
        if start + chunk_words >= len(words):
            break
    return chunks


def read_document(
    path: str | Path,
    doc_type: str = "unknown",
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> Document:
    """Load one file from disk and return it normalized and chunked."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    _check_suffix(suffix)

    text = normalize_text(_extract_text(path, suffix))
    source_id = path.stem

    return Document(
        source_id=source_id,
        path=path,
        text=text,
        doc_type=doc_type,
        chunks=chunk_text(text, source_id, chunk_words, overlap_words),
    )


def read_bytes(
    data: bytes,
    filename: str,
    doc_type: str = "unknown",
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> Document:
    """F1: the upload entry point. Same result as `read_document`, no disk.

    An HTTP upload hands you bytes and a filename, never a path, and spooling to
    a temp file just to read it back would also break checkpoint identity - the
    temp path differs on every request, so a path-derived thread id would never
    resume. `filename` is used only for its suffix (which parser) and its stem
    (the `source_id`); nothing is written anywhere.

    `Document.path` is set to the bare filename so the object stays printable
    and the field keeps meaning something, but it is not a real location and
    must not be opened.
    """
    from io import BytesIO

    name = Path(filename)
    suffix = name.suffix.lower()
    _check_suffix(suffix)

    text = normalize_text(_extract_text(BytesIO(data), suffix))
    source_id = name.stem

    return Document(
        source_id=source_id,
        path=name,
        text=text,
        doc_type=doc_type,
        chunks=chunk_text(text, source_id, chunk_words, overlap_words),
    )


def read_directory(
    directory: str | Path, doc_type: str = "unknown", **kwargs: object
) -> list[Document]:
    """Read every supported file in a directory, sorted for reproducibility."""
    directory = Path(directory)
    paths = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return [read_document(p, doc_type=doc_type, **kwargs) for p in paths]  # type: ignore[arg-type]
