from rag.gemini import Gemini
from rag.store import Hit, VectorStore


class Researcher:
    """Runs every planned query against the vector store and merges the results.

    This agent's tools are the embedding model and the index, not a chat model, so it costs a single
    embedding call no matter how many queries the Planner wrote.
    """

    name = "Researcher"

    def __init__(self, gemini: Gemini, store: VectorStore):
        self.gemini = gemini
        self.store = store

    def run(self, queries: list[str], k: int = 5) -> list[Hit]:
        if not queries or not self.store.chunks:
            return []
        best: dict[tuple[str, int], Hit] = {}
        per_query = max(2, k // len(queries) + 1)
        for vector in self.gemini.embed_queries(queries):
            for hit in self.store.search(vector, k=per_query):
                key = (hit.chunk.doc, hit.chunk.index)
                if key not in best or hit.score > best[key].score:
                    best[key] = hit
        # Take the best chunk from each query first so no part of a multi-part question gets crowded out.
        return sorted(best.values(), key=lambda h: h.score, reverse=True)[:k]
