from pydantic import BaseModel, Field

from rag.gemini import Gemini

SYSTEM = """You are the Planner in a team that answers questions from a user's private documents.
Decide whether the documents are needed, then write search queries for the Researcher.

- Split multi-part questions into one focused query per part (at most 3).
- Rewrite vague wording into the terms a document would likely use.
- Only set needs_documents to false for small talk like greetings or thanks; then put a short friendly
  reply in direct_reply. Anything factual needs the documents, even if you think you know the answer."""


class Plan(BaseModel):
    needs_documents: bool
    queries: list[str] = Field(default_factory=list, description="1-3 focused search queries")
    direct_reply: str = Field(default="", description="only when needs_documents is false")
    reasoning: str = Field(default="", description="one short sentence")


class Planner:
    name = "Planner"

    def __init__(self, gemini: Gemini):
        self.gemini = gemini

    def run(self, question: str) -> Plan:
        plan = self.gemini.generate_json(f"Question: {question}", Plan, system=SYSTEM)
        plan.queries = [q.strip() for q in plan.queries if q.strip()][:3]
        if plan.needs_documents and not plan.queries:
            plan.queries = [question]
        return plan
