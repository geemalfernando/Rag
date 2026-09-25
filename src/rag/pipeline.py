"""Ties the pieces together: sync documents, retrieve relevant chunks, answer with Gemini."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rag.agents import Orchestrator
from rag.config import Settings
from rag.gemini import EMBED_DIM, Gemini
from rag.indexer import Indexer, SyncReport
from rag.store import Hit, VectorStore

SYSTEM_PROMPT = """You answer questions using only the provided context from the user's documents.
Cite the sources you used inline like [1] or [2], matching the numbered context blocks.
If the context doesn't contain the answer, say you couldn't find it in the documents instead of guessing."""


@dataclass
class Answer:
    text: str
    sources: list[Hit]
    mode: str = "simple"
    trace: list[dict] = field(default_factory=list)
    verified: bool | None = None
    notes: list[str] = field(default_factory=list)


Mode = Literal["agents", "simple"]


class RAG:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.gemini = Gemini(self.settings)
        self.store = VectorStore.load(self.settings.store_dir, EMBED_DIM)
        self.indexer = Indexer(self.store, self.gemini, self.settings.chunk_size, self.settings.chunk_overlap)
        self.team = Orchestrator(self.gemini, self.store)

    def sync(self, path: Path, prune: bool = True) -> SyncReport:
        return self.indexer.sync(Path(path), prune=prune)

    def remove(self, path: Path) -> bool:
        return self.indexer.remove(Path(path))

    def retrieve(self, question: str, k: int = 5) -> list[Hit]:
        return self.store.search(self.gemini.embed_query(question), k=k)

    def ask(self, question: str, k: int = 5, mode: Mode = "agents") -> Answer:
        if mode == "agents":
            result = self.team.run(question, k=k)
            return Answer(result.text, result.sources, "agents", result.trace.to_list(), result.verified, result.notes)
        return self.ask_simple(question, k)

    def ask_simple(self, question: str, k: int = 5) -> Answer:
        """One retrieval and one generation call, no agents. Cheaper on quota."""
        hits = self.retrieve(question, k)
        if not hits:
            return Answer("The index is empty. Run `rag sync <folder>` first.", [])
        context = "\n\n".join(f"[{i}] (from {Path(h.chunk.doc).name})\n{h.chunk.text}" for i, h in enumerate(hits, 1))
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        return Answer(self.gemini.generate(prompt, system=SYSTEM_PROMPT), hits)
