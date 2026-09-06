"""Command-line tools for the refactored Fenix core."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fenix")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_command = commands.add_parser(
        "inspect", help="Extract and summarize a PDF without external API calls"
    )
    inspect_command.add_argument("input_pdf", type=Path)
    return parser


async def run(args: argparse.Namespace) -> int:
    if args.command == "inspect":
        document = await Pipeline().process(args.input_pdf)
        print(json.dumps(document.summary(), ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"Unknown command: {args.command}")


def main() -> int:
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

