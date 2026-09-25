from pathlib import Path

import pytest

from rag.chunking import chunk_text
from rag.gemini import Gemini
from rag.loaders import find_documents, load_text

SAMPLES = Path(__file__).parent.parent / "docs" / "samples"


def test_finds_supported_files_only(tmp_path):
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.png").write_bytes(b"\x89PNG")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("world")
    assert [p.name for p in find_documents(tmp_path)] == ["a.md", "c.txt"]


def test_loads_pdf(tmp_path):
    from pypdf import PdfWriter

    pdf = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with pdf.open("wb") as f:
        writer.write(f)
    assert load_text(pdf).strip() == ""


def test_short_text_is_one_chunk():
    assert chunk_text("Just one sentence.", size=100, overlap=10) == ["Just one sentence."]


def test_long_text_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, size=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # Consecutive chunks share some words thanks to the overlap.
    assert set(chunks[0].split()[-5:]) & set(chunks[1].split()[:15])


def test_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("x", size=10, overlap=10)


@pytest.mark.gemini
def test_sample_chunks_embed_cleanly(gemini_settings):
    chunks = [c for p in find_documents(SAMPLES) for c in chunk_text(load_text(p), 300, 50)]
    vectors = Gemini(gemini_settings).embed_documents(chunks)
    assert vectors.shape[0] == len(chunks) > 2
