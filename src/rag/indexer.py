"""Keep the vector store in sync with documents on disk."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from rag.chunking import chunk_text
from rag.loaders import find_documents, load_text
from rag.store import VectorStore


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def doc_key(path: Path) -> str:
    return str(path.resolve())


class Indexer:
    def __init__(self, store: VectorStore, embedder: Embedder, chunk_size: int = 800, chunk_overlap: int = 150):
        self.store = store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def index_file(self, path: Path, report: SyncReport) -> None:
        key = doc_key(path)
        digest = file_hash(path)
        existing = self.store.docs.get(key)
        if existing and existing.hash == digest:
            report.unchanged.append(key)
            return

        chunks = chunk_text(load_text(path), self.chunk_size, self.chunk_overlap)
        if chunks:
            self.store.upsert_doc(key, digest, chunks, self.embedder.embed_documents(chunks))
        else:
            self.store.remove_doc(key)
        (report.updated if existing else report.added).append(key)

    def sync(self, root: Path, prune: bool = True) -> SyncReport:
        """Index new or edited files under root; with prune, drop ones that were deleted."""
        report = SyncReport()
        found = find_documents(root)
        for path in found:
            self.index_file(path, report)

        if prune:
            present = {doc_key(p) for p in found}
            scope = doc_key(root)
            for key in list(self.store.docs):
                inside = key == scope or key.startswith(scope.rstrip("/") + "/")
                if inside and key not in present:
                    self.store.remove_doc(key)
                    report.removed.append(key)

        if report.changed:
            self.store.save()
        return report

    def remove(self, path: Path) -> bool:
        key = doc_key(path)
        if key not in self.store.docs:
            return False
        self.store.remove_doc(key)
        self.store.save()
        return True
