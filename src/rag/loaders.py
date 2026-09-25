"""Turn files on disk into plain text."""

from pathlib import Path

from pypdf import PdfReader

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path}")


def find_documents(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if is_supported(root) else []
    return sorted(p for p in root.rglob("*") if p.is_file() and is_supported(p) and not p.name.startswith("."))
