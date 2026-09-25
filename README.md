# Rag

Ask questions about your own documents with Gemini, and keep the index up to date as those documents change.

- **Supports** `.txt`, `.md`, `.rst` and `.pdf`
- **Incremental updates**: files are fingerprinted with sha256, so a sync only re-embeds what was added or edited and drops what was deleted
- **Grounded answers**: Gemini answers only from the retrieved chunks and cites them as `[1]`, `[2]`, …
- **No database needed**: the index is a JSON file plus a NumPy array in `.rag_store/`

## Setup

```bash
uv sync
cp .env.example .env   # then put your key in GEMINI_API_KEY
```

Get a key at <https://aistudio.google.com/apikey>.

## Usage

```bash
uv run rag sync docs/samples          # index a folder (re-run any time files change)
uv run rag ask "What temperature should the water be for pour-over?"
uv run rag search "largest planet"    # just show the matching chunks
uv run rag list                       # what's indexed
uv run rag remove docs/samples/coffee.txt
uv run rag watch docs/                # re-sync automatically while you edit
```

`sync` prints what happened:

```
  updated  docs/samples/coffee.txt
  removed  docs/samples/solar_system.md
0 added, 1 updated, 1 removed, 0 unchanged
```

Use `--keep-deleted` if you don't want files that disappeared to be removed from the index.
Pruning only applies inside the folder you sync, so syncing `docs/a` never touches `docs/b`.

## Configuration

Set these in `.env` or the environment:

| Variable | Default | |
|---|---|---|
| `GEMINI_API_KEY` | — | required |
| `RAG_CHAT_MODEL` | `gemini-flash-lite-latest` | model that writes the answers |
| `RAG_EMBED_MODEL` | `gemini-embedding-001` | 768-dim embeddings |
| `RAG_STORE_DIR` | `.rag_store` | where the index lives |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `800` / `150` | characters |

On the free tier, the full `gemini-flash-latest` model is capped at 20 requests a day, which is why the lite model is the default.
Busy (503) and rate-limit (429) responses are retried automatically with backoff.

## How it works

```
files ──load──▶ text ──chunk──▶ chunks ──embed (Gemini)──▶ vector store
                                                               │
question ──embed (Gemini)──▶ top-k cosine search ◀─────────────┘
                                   │
                                   ▼
                     Gemini answers from those chunks
```

| Module | Job |
|---|---|
| `loaders.py` | read txt/md/pdf into text |
| `chunking.py` | split on paragraphs → lines → sentences → words, with overlap |
| `gemini.py` | embeddings + generation, with retries |
| `store.py` | chunks + vectors on disk, cosine search |
| `indexer.py` | hash-based add/update/remove sync |
| `pipeline.py` | retrieve and answer |
| `cli.py` | the `rag` command |

## Tests

```bash
uv run pytest -m "not gemini"   # offline, uses a fake embedder
uv run pytest -m gemini         # live checks against the Gemini API (needs a key)
```

The live tests check that edits actually show up: they change a document, re-sync, and make sure the answer follows the change.
