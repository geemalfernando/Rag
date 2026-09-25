from pathlib import Path

from rag.gemini import Gemini
from rag.store import Hit

SYSTEM = """You are the Writer in a team that answers questions from a user's private documents.
Answer using only the numbered context blocks. Cite every fact inline like [1] or [2].
If the context doesn't contain the answer, say you couldn't find it in the documents instead of guessing.
Be direct: lead with the answer, keep it short."""


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] (from {Path(h.chunk.doc).name})\n{h.chunk.text}" for i, h in enumerate(hits, 1))


class Writer:
    name = "Writer"

    def __init__(self, gemini: Gemini):
        self.gemini = gemini

    def run(self, question: str, hits: list[Hit], draft: str | None = None, feedback: list[str] | None = None) -> str:
        prompt = f"Context:\n{format_context(hits)}\n\nQuestion: {question}"
        if draft and feedback:
            issues = "\n".join(f"- {issue}" for issue in feedback)
            prompt += (
                f"\n\nYour previous draft:\n{draft}\n\nThe Verifier found these problems:\n{issues}\n\n"
                "Rewrite the answer fixing them. Drop anything the context doesn't support."
            )
        return self.gemini.generate(prompt, system=SYSTEM).strip()
