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
