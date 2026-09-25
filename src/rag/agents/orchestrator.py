from dataclasses import dataclass, field
from pathlib import Path

from rag.agents.base import Trace
from rag.agents.planner import Planner
from rag.agents.researcher import Researcher
from rag.agents.verifier import Verifier
from rag.agents.writer import Writer
from rag.gemini import Gemini
from rag.store import Hit, VectorStore


@dataclass
class TeamAnswer:
    text: str
    sources: list[Hit]
    trace: Trace
    verified: bool | None = None  # None when the Verifier never ran (small talk, nothing found)
    notes: list[str] = field(default_factory=list)


class Orchestrator:
    """Runs Planner -> Researcher -> Writer -> Verifier, looping back to the Writer when claims don't check out."""

    def __init__(self, gemini: Gemini, store: VectorStore, max_revisions: int = 1):
        self.planner = Planner(gemini)
        self.researcher = Researcher(gemini, store)
        self.writer = Writer(gemini)
        self.verifier = Verifier(gemini)
        self.max_revisions = max_revisions

    def run(self, question: str, k: int = 5) -> TeamAnswer:
        trace = Trace()

        with trace.step(self.planner.name) as step:
            plan = self.planner.run(question)
            step.summary = f"{len(plan.queries)} search quer{'y' if len(plan.queries) == 1 else 'ies'}" if plan.needs_documents else "no documents needed"
            step.detail = {"queries": plan.queries, "reasoning": plan.reasoning}
        if not plan.needs_documents:
            return TeamAnswer(plan.direct_reply or "Hi! Ask me anything about your documents.", [], trace)

        with trace.step(self.researcher.name) as step:
            hits = self.researcher.run(plan.queries, k=k)
            docs = sorted({Path(h.chunk.doc).name for h in hits})
            step.summary = f"{len(hits)} chunks from {len(docs)} document{'s' if len(docs) != 1 else ''}"
            step.detail = {"documents": docs}
        if not hits:
            return TeamAnswer("There's nothing indexed yet, so I couldn't look anything up. Add some documents first.", [], trace)

        with trace.step(self.writer.name) as step:
            draft = self.writer.run(question, hits)
            step.summary = "drafted an answer"

        verified = False
        for attempt in range(self.max_revisions + 1):
            with trace.step(self.verifier.name) as step:
                verdict = self.verifier.run(question, hits, draft)
                verified = verdict.approved
                step.summary = "approved" if verified else f"found {len(verdict.issues)} issue{'s' if len(verdict.issues) != 1 else ''}"
                step.detail = {"issues": verdict.issues}
            if verified or attempt == self.max_revisions:
                break
            with trace.step(self.writer.name) as step:
                draft = self.writer.run(question, hits, draft=draft, feedback=verdict.issues)
                step.summary = "revised the answer"

        notes = [] if verified else ["The Verifier couldn't confirm every claim in this answer; check the sources."]
        return TeamAnswer(draft, hits, trace, verified=verified, notes=notes)
