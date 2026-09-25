import pytest

from rag.cli import build_parser


def test_parses_commands():
    parser = build_parser()
    args = parser.parse_args(["ask", "what is up?", "-k", "3"])
    assert args.question == "what is up?" and args.k == 3
    assert parser.parse_args(["sync", "docs", "--keep-deleted"]).keep_deleted


def test_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
