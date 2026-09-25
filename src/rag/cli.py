import argparse
import sys
import time
from pathlib import Path

from rag.indexer import SyncReport
from rag.pipeline import RAG


def _short(path: str) -> str:
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except ValueError:
        return path


def _print_report(report: SyncReport) -> None:
    for label, items in (("added", report.added), ("updated", report.updated), ("removed", report.removed)):
        for item in items:
            print(f"  {label:<8} {_short(item)}")
    print(
        f"{len(report.added)} added, {len(report.updated)} updated, "
        f"{len(report.removed)} removed, {len(report.unchanged)} unchanged"
    )


def cmd_sync(rag: RAG, args) -> None:
    _print_report(rag.sync(Path(args.path), prune=not args.keep_deleted))


def cmd_watch(rag: RAG, args) -> None:
    print(f"Watching {args.path} every {args.interval}s (Ctrl+C to stop)")
    try:
        while True:
            report = rag.sync(Path(args.path))
            if report.changed:
                print(time.strftime("[%H:%M:%S]"), end=" ")
                _print_report(report)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


def cmd_ask(rag: RAG, args) -> None:
    answer = rag.ask(args.question, k=args.k)
    print(answer.text.strip())
    if answer.sources:
        print("\nSources:")
        for i, hit in enumerate(answer.sources, 1):
            print(f"  [{i}] {_short(hit.chunk.doc)} (chunk {hit.chunk.index}, score {hit.score:.3f})")


def cmd_search(rag: RAG, args) -> None:
    for hit in rag.retrieve(args.query, k=args.k):
        print(f"{hit.score:.3f}  {_short(hit.chunk.doc)}#{hit.chunk.index}")
        print("   " + hit.chunk.text[:200].replace("\n", " ") + "\n")


def cmd_list(rag: RAG, args) -> None:
    if not rag.store.docs:
        print("No documents indexed yet.")
    for doc, record in sorted(rag.store.docs.items()):
        print(f"{record.chunks:>4} chunks  {record.hash[:10]}  {_short(doc)}")


def cmd_remove(rag: RAG, args) -> None:
    print("Removed." if rag.remove(Path(args.path)) else "That file isn't in the index.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Ask questions about your documents with Gemini.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sync", help="index new/edited files and drop deleted ones")
    p.add_argument("path")
    p.add_argument("--keep-deleted", action="store_true", help="don't remove files that no longer exist")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("watch", help="keep re-syncing a folder as files change")
    p.add_argument("path")
    p.add_argument("--interval", type=float, default=5.0)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("ask", help="answer a question using the indexed documents")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5, help="number of chunks to retrieve")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("search", help="show the most relevant chunks without generating an answer")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    sub.add_parser("list", help="show indexed documents").set_defaults(func=cmd_list)

    p = sub.add_parser("remove", help="drop a document from the index")
    p.add_argument("path")
    p.set_defaults(func=cmd_remove)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(RAG(), args)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
