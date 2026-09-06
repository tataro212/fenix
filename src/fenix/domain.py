"""Canonical, dependency-light document model used by every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator, TypeAlias


class TranslationStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    TRANSLATED = "translated"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("Bounding-box coordinates are inverted")

    @classmethod
    def from_value(cls, value: object) -> "BoundingBox":
        coordinates = tuple(float(item) for item in value)  # type: ignore[arg-type]
        if len(coordinates) != 4:
            raise ValueError("A bounding box requires four coordinates")
        return cls(*coordinates)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1


@dataclass(slots=True)
class TextStyle:
    font_family: str = ""
    font_size: float = 0.0
    bold: bool = False
    italic: bool = False
    color: int | None = None


@dataclass(slots=True)
class BaseBlock:
    block_id: str
    page_number: int
    order: int
    bbox: BoundingBox
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("block_id cannot be empty")
        if self.page_number < 1:
            raise ValueError("page_number must be one-based")
        if self.order < 0:
            raise ValueError("order cannot be negative")


@dataclass(slots=True)
class TextBlock(BaseBlock):
    source_text: str = ""
    translated_text: str | None = None
    style: TextStyle = field(default_factory=TextStyle)
    translation_status: TranslationStatus = TranslationStatus.NOT_REQUESTED
    translation_error: str | None = None

    @property
    def display_text(self) -> str:
        return self.translated_text if self.translated_text is not None else self.source_text


@dataclass(slots=True)
class ImageBlock(BaseBlock):
    asset_id: str = ""
    asset_path: Path | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass(slots=True)
class TableBlock(BaseBlock):
    rows: list[list[str]] = field(default_factory=list)
    translated_rows: list[list[str]] | None = None


Block: TypeAlias = TextBlock | ImageBlock | TableBlock


@dataclass(slots=True)
class Page:
    page_number: int
    width: float
    height: float
    blocks: list[Block] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must be one-based")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Page dimensions must be positive")
        self.validate()

    def validate(self) -> None:
        orders: set[int] = set()
        ids: set[str] = set()
        for block in self.blocks:
            if block.page_number != self.page_number:
                raise ValueError(f"{block.block_id} belongs to the wrong page")
            if block.order in orders:
                raise ValueError(f"Duplicate block order {block.order} on page {self.page_number}")
            if block.block_id in ids:
                raise ValueError(f"Duplicate block ID {block.block_id}")
            orders.add(block.order)
            ids.add(block.block_id)

    def ordered_blocks(self) -> list[Block]:
        return sorted(self.blocks, key=lambda block: block.order)


@dataclass(slots=True)
class Document:
    source_path: Path
    pages: list[Page]
    title: str = ""
    source_language: str | None = None
    target_language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_path = Path(self.source_path)
        self.validate()

    def validate(self) -> None:
        numbers = [page.page_number for page in self.pages]
        if numbers != list(range(1, len(self.pages) + 1)):
            raise ValueError("Document pages must be contiguous and one-based")
        seen_ids: set[str] = set()
        for page in self.pages:
            page.validate()
            for block in page.blocks:
                if block.block_id in seen_ids:
                    raise ValueError(f"Duplicate document block ID {block.block_id}")
                seen_ids.add(block.block_id)

    def iter_blocks(self) -> Iterator[Block]:
        for page in self.pages:
            yield from page.ordered_blocks()

    def text_blocks(self) -> Iterator[TextBlock]:
        for block in self.iter_blocks():
            if isinstance(block, TextBlock):
                yield block

    def summary(self) -> dict[str, Any]:
        blocks = list(self.iter_blocks())
        return {
            "source": str(self.source_path),
            "pages": len(self.pages),
            "blocks": len(blocks),
            "text_blocks": sum(isinstance(block, TextBlock) for block in blocks),
            "image_blocks": sum(isinstance(block, ImageBlock) for block in blocks),
            "table_blocks": sum(isinstance(block, TableBlock) for block in blocks),
            "characters": sum(len(block.source_text) for block in blocks if isinstance(block, TextBlock)),
        }

