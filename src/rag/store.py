"""A small on-disk vector store: chunk metadata in JSON, vectors in a .npy file."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Chunk:
    doc: str
    index: int
    text: str


@dataclass
class DocRecord:
    hash: str
    chunks: int


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, directory: Path, dim: int):
        self.directory = Path(directory)
        self.dim = dim
        self.docs: dict[str, DocRecord] = {}
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    @property
    def _meta_path(self) -> Path:
        return self.directory / "index.json"

    @property
    def _vectors_path(self) -> Path:
        return self.directory / "vectors.npy"

    @classmethod
    def load(cls, directory: Path, dim: int) -> "VectorStore":
        store = cls(directory, dim)
        if store._meta_path.exists():
            meta = json.loads(store._meta_path.read_text())
            store.docs = {k: DocRecord(**v) for k, v in meta["docs"].items()}
            store.chunks = [Chunk(**c) for c in meta["chunks"]]
            store.vectors = np.load(store._vectors_path)
        return store

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        meta = {"docs": {k: asdict(v) for k, v in self.docs.items()}, "chunks": [asdict(c) for c in self.chunks]}
        # Write to temp files first so a crash mid-save can't leave a half-written index.
        tmp_meta = self._meta_path.with_suffix(".json.tmp")
        tmp_vec = self.directory / "vectors.tmp.npy"
        tmp_meta.write_text(json.dumps(meta, indent=1))
        np.save(tmp_vec, self.vectors)
        tmp_vec.replace(self._vectors_path)
        tmp_meta.replace(self._meta_path)

    def remove_doc(self, doc: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c.doc != doc]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep]
        self.docs.pop(doc, None)

    def upsert_doc(self, doc: str, doc_hash: str, texts: list[str], vectors: np.ndarray) -> None:
        self.remove_doc(doc)
        self.chunks.extend(Chunk(doc=doc, index=i, text=t) for i, t in enumerate(texts))
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        self.docs[doc] = DocRecord(hash=doc_hash, chunks=len(texts))

    def search(self, query: np.ndarray, k: int = 5) -> list[Hit]:
        if not self.chunks:
            return []
        scores = self.vectors @ query
        top = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]
