import importlib

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.gemini


@pytest.fixture
def client(gemini_settings, tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DOCS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("RAG_STORE_DIR", str(tmp_path / "store"))
    import rag.api

    api = importlib.reload(rag.api)
    with TestClient(api.app) as c:
        yield c


def test_document_lifecycle_and_answers(client):
    # Starts seeded with the sample docs.
    names = {d["name"] for d in client.get("/api/documents").json()}
    assert {"coffee.txt", "solar_system.md"} <= names

    r = client.put("/api/documents/menu.md", json={"content": "The cafe's soup of the day is pumpkin."})
    assert r.json()["added"] == ["menu.md"]
    assert "pumpkin" in client.post("/api/ask", json={"question": "What is the soup of the day?", "k": 3}).json()["answer"].lower()

    r = client.put("/api/documents/menu.md", json={"content": "The cafe's soup of the day is tomato basil."})
    assert r.json()["updated"] == ["menu.md"]
    answer = client.post("/api/ask", json={"question": "What is the soup of the day?", "k": 3}).json()
    assert "tomato" in answer["answer"].lower() and answer["sources"][0]["doc"] == "menu.md"

    assert client.delete("/api/documents/menu.md").json()["removed"] == ["menu.md"]

    r = client.post("/api/documents", files=[("files", ("notes.txt", b"Standup is at 9:15 every weekday.", "text/plain"))])
    assert r.json()["added"] == ["notes.txt"]


def test_rejects_unsafe_names(client, tmp_path):
    # Traversal attempts must never write outside the docs folder, whichever layer rejects them.
    for name in ("..%2Fescape.md", "%2E%2E%2Fescape.md", "sub%2Fescape.md"):
        assert client.put(f"/api/documents/{name}", json={"content": "x"}).status_code >= 400
    assert not list(tmp_path.rglob("escape.md"))
    assert client.put("/api/documents/evil.sh", json={"content": "x"}).status_code == 400
