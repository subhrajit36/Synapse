from io import BytesIO


def read_resume(data: bytes, filename: str) -> str:
    """Extract raw text from an uploaded resume. Supports .pdf, .docx, .txt."""
    name = filename.lower()

    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        # extract_text() can return None for image-only pages, so guard with "".
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if name.endswith(".docx"):
        import docx
        document = docx.Document(BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)

    # Fallback: treat as plain text.
    return data.decode("utf-8", errors="ignore")


if __name__ == "__main__":
    import glob
    for path in sorted(glob.glob("data/samples/ravi_backend.*")):
        with open(path, "rb") as f:
            text = read_resume(f.read(), path)
        print(f"\n=== {path}  ({len(text)} chars) ===")
        print(text[:220])
