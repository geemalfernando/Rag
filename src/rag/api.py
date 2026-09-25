"""HTTP API + static frontend. Documents live in a folder on disk; every change re-syncs the index."""

import os
import re
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

from rag.indexer import SyncReport
from rag.loaders import SUPPORTED_SUFFIXES, TEXT_SUFFIXES
from rag.pipeline import RAG

DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", "data/docs")).resolve()
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "docs" / "samples"
WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_DOCS = 100
_NAME_RE = re.compile(r"^[\w\- .()]+$")

_lock = threading.Lock()
_rag: RAG | None = None


def rag() -> RAG:
    assert _rag is not None, "app not started"
    return _rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    # Fresh deploys (and free hosts that wipe the disk) start with the sample docs so there's something to ask about.
    if not any(DOCS_DIR.iterdir()) and SAMPLES_DIR.exists():
        for sample in SAMPLES_DIR.iterdir():
            shutil.copy(sample, DOCS_DIR / sample.name)
    _rag = RAG()
    with _lock:
        _rag.sync(DOCS_DIR)
    yield


app = FastAPI(title="Rag", lifespan=lifespan)


@app.exception_handler(genai_errors.APIError)
async def gemini_error(request: Request, exc: genai_errors.APIError):
    status = 429 if exc.code == 429 else 502
    return JSONResponse({"detail": f"Gemini error {exc.code}: {exc.message}"}, status_code=status)


def _doc_path(name: str) -> Path:
    if not _NAME_RE.match(name) or name.startswith("."):
        raise HTTPException(400, "Use a plain file name with letters, numbers, spaces, dashes or dots.")
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"Supported types: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
    return DOCS_DIR / name


def _report(report: SyncReport) -> dict:
    names = lambda items: [Path(p).name for p in items]
    return {
        "added": names(report.added),
        "updated": names(report.updated),
        "removed": names(report.removed),
        "unchanged": len(report.unchanged),
    }


def _sync() -> dict:
    return _report(rag().sync(DOCS_DIR))


class TextDoc(BaseModel):
    content: str = Field(max_length=MAX_UPLOAD_BYTES)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=10)
    mode: Literal["agents", "simple"] = "agents"


@app.get("/api/health")
def health():
    return {"ok": True, "chat_model": rag().settings.chat_model, "embed_model": rag().settings.embed_model}


@app.get("/api/documents")
def list_documents():
    indexed = {Path(k).name: v for k, v in rag().store.docs.items()}
    docs = []
    for path in sorted(DOCS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            record = indexed.get(path.name)
            docs.append({
                "name": path.name,
                "size": path.stat().st_size,
                "chunks": record.chunks if record else 0,
                "editable": path.suffix.lower() in TEXT_SUFFIXES,
            })
    return docs


@app.get("/api/documents/{name}")
def read_document(name: str):
    path = _doc_path(name)
    if not path.exists():
        raise HTTPException(404, "No such document.")
    if path.suffix.lower() not in TEXT_SUFFIXES:
        raise HTTPException(400, "Only text documents can be opened in the editor.")
    return {"name": name, "content": path.read_text(encoding="utf-8", errors="replace")}


@app.put("/api/documents/{name}")
def save_document(name: str, body: TextDoc):
    path = _doc_path(name)
    if path.suffix.lower() not in TEXT_SUFFIXES:
        raise HTTPException(400, "Only text documents can be written from the editor.")
    with _lock:
        if not path.exists() and len(list(DOCS_DIR.iterdir())) >= MAX_DOCS:
            raise HTTPException(400, f"Limit of {MAX_DOCS} documents reached.")
        path.write_text(body.content, encoding="utf-8")
        return _sync()


@app.post("/api/documents")
async def upload_documents(files: list[UploadFile]):
    staged = []
    for file in files:
        path = _doc_path(Path(file.filename or "").name)
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{file.filename} is larger than 5 MB.")
        staged.append((path, data))
    with _lock:
        existing = {p.name for p in DOCS_DIR.iterdir()}
        if len(existing | {p.name for p, _ in staged}) > MAX_DOCS:
            raise HTTPException(400, f"Limit of {MAX_DOCS} documents reached.")
        for path, data in staged:
            path.write_bytes(data)
        return _sync()


@app.delete("/api/documents/{name}")
def delete_document(name: str):
    path = _doc_path(name)
    with _lock:
        if not path.exists():
            raise HTTPException(404, "No such document.")
        path.unlink()
        return _sync()


@app.post("/api/ask")
def ask(body: Question):
    answer = rag().ask(body.question, k=body.k, mode=body.mode)
    return {
        "answer": answer.text,
        "mode": answer.mode,
        "verified": answer.verified,
        "notes": answer.notes,
        "trace": answer.trace,
        "sources": [
            {"doc": Path(h.chunk.doc).name, "chunk": h.chunk.index, "score": round(h.score, 3), "text": h.chunk.text}
            for h in answer.sources
        ],
    }


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
