"""Extract text from plain text, Markdown, PDF, and Word documents."""

from io import BytesIO
from pathlib import Path


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.casefold()
    if suffix in {".txt", ".md"}:
        text = content.decode("utf-8-sig")
    elif suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".docx":
        from docx import Document

        document = Document(BytesIO(content))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    else:
        raise ValueError("Supported document types are .txt, .md, .pdf, and .docx")
    text = text.strip()
    if not text:
        raise ValueError("No extractable text was found in this document")
    return text
