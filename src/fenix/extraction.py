"""Single-pass PyMuPDF extraction into the canonical document model."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pymupdf

from .domain import BoundingBox, Document, ImageBlock, Page, TextBlock, TextStyle


class PyMuPDFExtractor:
    """Extract text and image positions while opening the PDF exactly once."""

    def extract(self, pdf_path: str | Path) -> Document:
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {path}")

        pages: list[Page] = []
        with pymupdf.open(path) as pdf:
            metadata = {key: value for key, value in pdf.metadata.items() if value}
            for page_index, pdf_page in enumerate(pdf):
                pages.append(self._extract_page(pdf_page, page_index + 1))

        return Document(
            source_path=path.resolve(),
            title=metadata.get("title", path.stem),
            pages=pages,
            metadata=metadata,
        )

    def _extract_page(self, pdf_page: pymupdf.Page, page_number: int) -> Page:
        raw = pdf_page.get_text("dict", sort=True)
        blocks = []

        for order, raw_block in enumerate(raw.get("blocks", [])):
            bbox = BoundingBox.from_value(raw_block.get("bbox", (0, 0, 0, 0)))
            block_id = f"p{page_number:04d}-b{order:05d}"
            block_type = raw_block.get("type", 0)

            if block_type == 0:
                text_block = self._text_block(raw_block, block_id, page_number, order, bbox)
                if text_block is not None:
                    blocks.append(text_block)
            elif block_type == 1:
                blocks.append(
                    ImageBlock(
                        block_id=block_id,
                        page_number=page_number,
                        order=order,
                        bbox=bbox,
                        asset_id=f"page-{page_number}-image-{raw_block.get('number', order)}",
                        mime_type=self._image_mime_type(raw_block.get("ext")),
                        width=raw_block.get("width"),
                        height=raw_block.get("height"),
                    )
                )

        return Page(
            page_number=page_number,
            width=float(pdf_page.rect.width),
            height=float(pdf_page.rect.height),
            blocks=blocks,
        )

    def _text_block(
        self,
        raw_block: dict[str, Any],
        block_id: str,
        page_number: int,
        order: int,
        bbox: BoundingBox,
    ) -> TextBlock | None:
        lines: list[str] = []
        spans: list[dict[str, Any]] = []
        for line in raw_block.get("lines", []):
            line_spans = line.get("spans", [])
            spans.extend(line_spans)
            line_text = "".join(str(span.get("text", "")) for span in line_spans).strip()
            if line_text:
                lines.append(line_text)

        text = self._join_lines(lines)
        if not text:
            return None

        primary = max(spans, key=lambda span: len(str(span.get("text", ""))), default={})
        flags = int(primary.get("flags", 0))
        return TextBlock(
            block_id=block_id,
            page_number=page_number,
            order=order,
            bbox=bbox,
            source_text=text,
            style=TextStyle(
                font_family=str(primary.get("font", "")),
                font_size=float(primary.get("size", 0.0)),
                bold=bool(flags & 16),
                italic=bool(flags & 2),
                color=primary.get("color"),
            ),
        )

    @staticmethod
    def _join_lines(lines: Iterable[str]) -> str:
        result = ""
        for line in lines:
            clean = line.strip()
            if not clean:
                continue
            if result.endswith("-") and clean[:1].islower():
                result = result[:-1] + clean
            elif result:
                result += " " + clean
            else:
                result = clean
        return result

    @staticmethod
    def _image_mime_type(extension: object) -> str | None:
        ext = str(extension or "").lower().lstrip(".")
        return f"image/{'jpeg' if ext in {'jpg', 'jpeg'} else ext}" if ext else None

