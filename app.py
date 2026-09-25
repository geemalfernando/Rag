"""Vercel entrypoint; local development continues to use rag.api:app."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

if os.getenv("VERCEL"):
    os.environ.setdefault("RAG_DOCS_DIR", "/tmp/rag/docs")
    os.environ.setdefault("RAG_STORE_DIR", "/tmp/rag/store")

from rag.api import app  # noqa: E402, F401
