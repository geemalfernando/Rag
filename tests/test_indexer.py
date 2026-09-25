import numpy as np
import pytest

from rag.gemini import EMBED_DIM, Gemini
from rag.indexer import Indexer
from rag.store import VectorStore


class FakeEmbedder:
    """Deterministic bag-of-words embedder so update logic can be tested offline."""

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        v = np.zeros(EMBED_DIM, dtype=np.float32)
        for word in text.lower().split():
            v[hash(word.strip(".,")) % EMBED_DIM] += 1
        return v / max(np.linalg.norm(v), 1e-12)

    def embed_documents(self, texts):
        self.calls += 1
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def setup(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    store = VectorStore(tmp_path / "store", EMBED_DIM)
    embedder = FakeEmbedder()
    return docs, store, embedder, Indexer(store, embedder, chunk_size=200, chunk_overlap=40)


def test_sync_adds_updates_and_removes(setup, tmp_path):
    docs, store, embedder, indexer = setup
    (docs / "a.md").write_text("Apples are red.")
    (docs / "b.txt").write_text("Bananas are yellow.")

    report = indexer.sync(docs)
    assert len(report.added) == 2 and not report.updated

    # Nothing changed: no re-embedding.
    calls = embedder.calls
    report = indexer.sync(docs)
    assert len(report.unchanged) == 2 and not report.changed
    assert embedder.calls == calls

    # Edit one file, delete the other.
    (docs / "a.md").write_text("Apples are green now.")
    (docs / "b.txt").unlink()
    report = indexer.sync(docs)
    assert [p.split("/")[-1] for p in report.updated] == ["a.md"]
    assert [p.split("/")[-1] for p in report.removed] == ["b.txt"]
    assert [c.text for c in store.chunks] == ["Apples are green now."]

    # Changes survive a reload from disk.
    reloaded = VectorStore.load(tmp_path / "store", EMBED_DIM)
    assert reloaded.vectors.shape == (1, EMBED_DIM)
    assert list(reloaded.docs) == list(store.docs)


def test_prune_only_touches_the_synced_folder(setup):
    docs, store, _, indexer = setup
    (docs / "one").mkdir()
    (docs / "two").mkdir()
    (docs / "one" / "x.md").write_text("inside one")
    (docs / "two" / "y.md").write_text("inside two")
    indexer.sync(docs)

    (docs / "one" / "x.md").unlink()
    report = indexer.sync(docs / "two")
    assert not report.removed
    assert len(store.docs) == 2


def test_search_finds_updated_content(setup):
    docs, store, embedder, indexer = setup
    (docs / "fruit.md").write_text("Apples are red.")
    indexer.sync(docs)
    (docs / "fruit.md").write_text("Mangoes are sweet and tropical.")
    indexer.sync(docs)
    hit = store.search(embedder.embed_query("tropical mangoes"), k=1)[0]
    assert "Mangoes" in hit.chunk.text


@pytest.mark.gemini
def test_updates_are_reflected_in_gemini_search(gemini_settings, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    gem = Gemini(gemini_settings)
    store = VectorStore(tmp_path / "store", EMBED_DIM)
    indexer = Indexer(store, gem)

    (docs / "office.md").write_text("The office wifi password is sunflower42.")
    (docs / "lunch.md").write_text("Lunch is served in the cafeteria at noon.")
    indexer.sync(docs)
    query = gem.embed_query("what is the wifi password?")
    assert "sunflower42" in store.search(query, k=1)[0].chunk.text

    (docs / "office.md").write_text("The office wifi password was changed to bluebird77.")
    report = indexer.sync(docs)
    assert len(report.updated) == 1 and len(report.unchanged) == 1
    top = store.search(query, k=1)[0].chunk.text
    assert "bluebird77" in top and all("sunflower42" not in c.text for c in store.chunks)


def test_empty_files_are_tracked_not_readded(setup):
    docs, store, _, indexer = setup
    (docs / "empty.md").write_text("   ")
    assert len(indexer.sync(docs).added) == 1
    assert not indexer.sync(docs).changed
    (docs / "empty.md").write_text("Now it has words.")
    assert len(indexer.sync(docs).updated) == 1
    assert [c.text for c in store.chunks] == ["Now it has words."]
