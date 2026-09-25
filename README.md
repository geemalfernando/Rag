# Rag

Ask questions about your own documents with Gemini, and keep the index up to date as those documents change.

- **Supports** `.txt`, `.md`, `.rst` and `.pdf`
- **Incremental updates**: files are fingerprinted with sha256, so a sync only re-embeds what was added or edited and drops what was deleted
- **Multi-agent answers**: a Planner, Researcher, Writer and Verifier work together, and every claim is fact-checked against your docs
- **Grounded answers**: Gemini answers only from the retrieved chunks and cites them as `[1]`, `[2]`, …
- **No database needed**: the index is a JSON file plus a NumPy array in `.rag_store/`
- **Web app**: upload, edit and delete docs in the browser, then ask questions with clickable citations

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
uv run rag ask "..." --trace             # show what each agent did
uv run rag ask "..." --simple            # skip the agents: one search, one Gemini call
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

## Web app

```bash
uv run uvicorn rag.api:app --reload
```

Open <http://localhost:8000>. Documents live in `data/docs/` (set `RAG_DOCS_DIR` to change it). A fresh
install starts with the sample docs. Every upload, edit or delete re-syncs just that change.

| Endpoint | |
|---|---|
| `GET /api/documents` | list docs with size and chunk count |
| `GET /api/documents/{name}` | read a text doc |
| `PUT /api/documents/{name}` | create or overwrite a text doc (`{"content": "..."}`) |
| `POST /api/documents` | upload files (multipart, field `files`) |
| `DELETE /api/documents/{name}` | delete a doc |
| `POST /api/ask` | `{"question": "...", "k": 5, "mode": "agents"}` → answer, sources, `verified`, `trace` |

Each write returns a sync report like `{"added": [], "updated": ["coffee.txt"], "removed": [], "unchanged": 1}`.

## Deploying to Vercel

The repo includes a FastAPI entrypoint (`app.py`) and `vercel.json`.

1. Import this repository in Vercel, or run `npx vercel link` locally.
2. Set `GEMINI_API_KEY` in the project's environment variables.
3. Deploy with `npx vercel --prod` or through the dashboard.

On Vercel, documents and the index use `/tmp/rag/`. This storage is temporary
and is not shared between function instances: uploads and edits can disappear
on a cold start or be unavailable on another instance. This deployment is suitable
for a demo; durable document management requires shared external storage.
Fresh instances index the sample documents using your Gemini API key.

The app has no login: anyone with the URL can access documents and use your Gemini quota.

## Deploying to Render

The repo includes a `render.yaml` blueprint.

1. In the Render dashboard choose **New → Blueprint** and pick this repo.
2. Paste your `GEMINI_API_KEY` when asked.
3. Deploy. Pushes to `main` redeploy automatically.

Heads-up for the free plan: the service sleeps after ~15 minutes idle (first request afterwards takes
~30s), and the disk is wiped on every restart or deploy, so uploaded docs reset to the samples.
Attach a persistent disk (paid) and point `RAG_DOCS_DIR` / `RAG_STORE_DIR` at it to keep them.

The app has no login: anyone with the URL can upload documents and use your Gemini quota.

## Configuration

Set these in `.env` or the environment:

| Variable | Default | |
|---|---|---|
| `GEMINI_API_KEY` | — | required |
| `RAG_CHAT_MODEL` | `gemini-flash-lite-latest` | model that writes the answers |
| `RAG_EMBED_MODEL` | `gemini-embedding-001` | 768-dim embeddings |
| `RAG_STORE_DIR` | `.rag_store` | where the index lives |
| `RAG_DOCS_DIR` | `data/docs` | where the web app keeps documents |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `800` / `150` | characters |

On the free tier, the full `gemini-flash-latest` model is capped at 20 requests a day, which is why the lite model is the default.
Busy (503) and rate-limit (429) responses are retried automatically with backoff.

## The agent team

Questions are answered by four agents, each with one job:

```
question
   │
   ▼
Planner ─────── needs the docs? split into focused search queries (max 3)
   │            └─ small talk ("hi", "thanks") gets a direct reply, no search
   ▼
Researcher ──── embeds all queries in one call, searches the index, merges the best chunks
   ▼
Writer ──────── drafts an answer with [n] citations, using only those chunks
   ▼
Verifier ────── checks each claim against its cited chunk
   │   ├─ approved ─────────────▶ answer (✓ verified)
   │   └─ issues found ─▶ Writer revises once ─▶ Verifier checks again
   ▼
answer + sources + trace
```

| Agent | Uses | Output |
|---|---|---|
| Planner (`agents/planner.py`) | Gemini, structured JSON | `Plan`: queries, or a direct reply |
| Researcher (`agents/researcher.py`) | embeddings + vector store | merged chunks, best per query |
| Writer (`agents/writer.py`) | Gemini | cited answer, or a revision from feedback |
| Verifier (`agents/verifier.py`) | Gemini, structured JSON | `Verdict`: approved + list of issues |

The `Orchestrator` (`agents/orchestrator.py`) runs them and records a trace, which is printed with `--trace`,
returned by `/api/ask`, and shown as a timeline in the web UI.

**Quota:** an agent answer uses 3 Gemini calls (up to 5 when a revision is needed), versus 1 in simple mode.
On the free tier, switch the web UI to **Simple** or pass `--simple` when you're running low.

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
| `agents/` | Planner, Researcher, Writer, Verifier and the Orchestrator |
| `pipeline.py` | sync, retrieve and answer (agent team or simple mode) |
| `cli.py` | the `rag` command |
| `api.py` + `web/` | FastAPI backend and the browser frontend |

## Tests

```bash
uv run pytest -m "not gemini"   # offline, uses a fake embedder
uv run pytest -m gemini         # live checks against the Gemini API (needs a key)
```

The live tests check that edits actually show up: they change a document, re-sync, and make sure the answer follows the change.

## License

MIT, see [LICENSE](LICENSE).
