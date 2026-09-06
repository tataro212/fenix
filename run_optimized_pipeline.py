#!/usr/bin/env python3
"""Command-line entry point for the Fenix optimized translation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate a PDF while preserving its document structure."
    )
    parser.add_argument("input_pdf", type=Path, help="PDF file to translate")
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--target", "-t", default="el",
        help="Target language code (default: el)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=5,
        help="Maximum concurrent translation batches (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


async def run(args: argparse.Namespace) -> int:
    if not args.input_pdf.is_file() or args.input_pdf.suffix.lower() != ".pdf":
        logging.error("Input must be an existing PDF: %s", args.input_pdf)
        return 2
    if args.workers < 1:
        logging.error("--workers must be at least 1")
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    # Import after validation so --help works without optional ML packages.
    from optimized_document_pipeline import OptimizedDocumentPipeline

    pipeline = OptimizedDocumentPipeline(max_workers=args.workers)
    result = await pipeline.process_pdf_with_optimized_pipeline(
        str(args.input_pdf.resolve()),
        str(args.output.resolve()),
        args.target,
    )

    summary = {
        "success": result.success,
        "outputs": result.output_files,
        "error": result.error,
        "processing_time_seconds": result.statistics.processing_time,
        "pages": result.statistics.total_pages,
        "translation_success_rate": result.statistics.translation_success_rate,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result.success else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

