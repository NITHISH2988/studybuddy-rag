from io import BytesIO

import pytest
from docx import Document

from studybuddy.ingestion import extract_text


def test_extracts_plain_text_and_markdown():
    assert extract_text("notes.txt", b"  vector search  ") == "vector search"
    assert extract_text("notes.md", b"# Heading\nText") == "# Heading\nText"


def test_rejects_unsupported_and_empty_documents():
    with pytest.raises(ValueError, match="Supported document"):
        extract_text("notes.csv", b"a,b")
    with pytest.raises(ValueError, match="No extractable text"):
        extract_text("empty.txt", b"  ")


def _one_page_pdf(text: str) -> bytes:
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, item in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode() + item + b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    )
    return bytes(output)


def test_extracts_pdf_and_docx_text():
    assert "Solar panels" in extract_text(
        "notes.pdf", _one_page_pdf("Solar panels use sunlight")
    )
    document = Document()
    document.add_paragraph("Vector indexes store embeddings")
    buffer = BytesIO()
    document.save(buffer)
    assert (
        extract_text("notes.docx", buffer.getvalue())
        == "Vector indexes store embeddings"
    )
