import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    embed_model: str
    chat_model: str
    store_dir: Path
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            embed_model=os.getenv("RAG_EMBED_MODEL", "gemini-embedding-001"),
            chat_model=os.getenv("RAG_CHAT_MODEL", "gemini-flash-latest"),
            store_dir=Path(os.getenv("RAG_STORE_DIR", ".rag_store")),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "150")),
        )

    def require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return self.api_key
