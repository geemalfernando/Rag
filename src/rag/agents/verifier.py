from pydantic import BaseModel, Field

from rag.agents.writer import format_context
from rag.gemini import Gemini
from rag.store import Hit

SYSTEM = """You are the Verifier in a team that answers questions from a user's private documents.
Check the draft answer against the numbered context, claim by claim.

- A claim is supported only if the cited context block actually says it.
- Flag claims that are missing a citation, cite the wrong block, or aren't in the context at all.
- Saying the answer couldn't be found is fine when the context really lacks it.
Set approved to true only if there are no issues."""


class Verdict(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list, description="one sentence per problem")


class Verifier:
    name = "Verifier"

    def __init__(self, gemini: Gemini):
        self.gemini = gemini

    def run(self, question: str, hits: list[Hit], draft: str) -> Verdict:
        prompt = f"Context:\n{format_context(hits)}\n\nQuestion: {question}\n\nDraft answer:\n{draft}"
        verdict = self.gemini.generate_json(prompt, Verdict, system=SYSTEM)
        if verdict.issues:
            verdict.approved = False
        return verdict
