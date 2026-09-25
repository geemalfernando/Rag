import pytest

from rag.agents import Planner
from rag.gemini import Gemini

pytestmark = pytest.mark.gemini


def test_planner_splits_multi_part_questions(gemini_settings):
    plan = Planner(Gemini(gemini_settings)).run("How long should coffee brew, and which planet is the largest?")
    assert plan.needs_documents
    assert len(plan.queries) >= 2
    joined = " ".join(plan.queries).lower()
    assert "coffee" in joined and "planet" in joined


def test_planner_skips_retrieval_for_small_talk(gemini_settings):
    plan = Planner(Gemini(gemini_settings)).run("hi there, thanks!")
    assert not plan.needs_documents and plan.direct_reply


@pytest.fixture
def team(gemini_settings, tmp_path):
    from rag.gemini import EMBED_DIM
    from rag.indexer import Indexer
    from rag.store import VectorStore

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "hr.md").write_text("Employees get 25 days of paid annual leave per year.")
    (docs / "it.md").write_text("Laptops are replaced every three years by the IT team.")
    (docs / "food.md").write_text("The canteen serves vegetarian curry on Fridays.")
    gem = Gemini(gemini_settings)
    store = VectorStore(tmp_path / "store", EMBED_DIM)
    Indexer(store, gem).sync(docs)
    return gem, store


def test_researcher_covers_every_query(team):
    from rag.agents import Researcher

    gem, store = team
    hits = Researcher(gem, store).run(["annual leave days", "laptop replacement"], k=2)
    assert {h.chunk.doc.split("/")[-1] for h in hits} == {"hr.md", "it.md"}


def test_verifier_catches_unsupported_claims_and_writer_fixes_them(team):
    from rag.agents import Researcher, Verifier, Writer

    gem, store = team
    question = "How many leave days do employees get?"
    hits = Researcher(gem, store).run([question], k=2)

    bad = "Employees get 30 days of annual leave [1], plus free gym membership [1]."
    verdict = Verifier(gem).run(question, hits, bad)
    assert not verdict.approved and verdict.issues

    fixed = Writer(gem).run(question, hits, draft=bad, feedback=verdict.issues)
    assert "25" in fixed and "gym" not in fixed.lower()
    assert Verifier(gem).run(question, hits, fixed).approved


def test_orchestrator_answers_multi_part_question_with_trace(team):
    from rag.agents import Orchestrator

    gem, store = team
    result = Orchestrator(gem, store).run("How many leave days do we get, and how often are laptops replaced?", k=4)
    agents = [s.agent for s in result.trace.steps]
    assert agents[:4] == ["Planner", "Researcher", "Writer", "Verifier"]
    assert "25" in result.text and ("three" in result.text.lower() or "3" in result.text)
    assert {"hr.md", "it.md"} <= {h.chunk.doc.split("/")[-1] for h in result.sources}
    assert result.verified is not None
