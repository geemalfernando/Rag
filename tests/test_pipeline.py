from dataclasses import replace

import pytest

from rag.pipeline import RAG

pytestmark = pytest.mark.gemini


def test_answers_from_documents_and_follows_edits(gemini_settings, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    rag = RAG(replace(gemini_settings, store_dir=tmp_path / "store"))

    (docs / "policy.md").write_text("Employees get 18 days of paid annual leave per year.")
    (docs / "parking.md").write_text("Visitor parking is on level B2 of the garage.")
    rag.sync(docs)
    first = rag.ask("How many days of annual leave do employees get?", k=2)
    assert "18" in first.text
    assert first.sources[0].chunk.doc.endswith("policy.md")

    (docs / "policy.md").write_text("From 2026, employees get 25 days of paid annual leave per year.")
    rag.sync(docs)
    second = rag.ask("How many days of annual leave do employees get?", k=2)
    assert "25" in second.text and "18" not in second.text
