from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import pymupdf

from fenix.domain import BoundingBox, Document, Page, TextBlock, TranslationStatus
from fenix.extraction import PyMuPDFExtractor
from fenix.translation import BatchTranslator, TranslationSegment


class UppercaseProvider:
    async def translate(
        self, segments: list[TranslationSegment], target_language: str
    ) -> dict[str, str]:
        await asyncio.sleep(0)
        return {segment.segment_id: segment.text.upper() for segment in segments}


class IncompleteProvider:
    async def translate(
        self, segments: list[TranslationSegment], target_language: str
    ) -> dict[str, str]:
        return {}


class CorePipelineTests(unittest.TestCase):
    def test_document_rejects_duplicate_block_order(self) -> None:
        box = BoundingBox(0, 0, 10, 10)
        with self.assertRaises(ValueError):
            Page(
                page_number=1,
                width=100,
                height=100,
                blocks=[
                    TextBlock("one", 1, 0, box, source_text="one"),
                    TextBlock("two", 1, 0, box, source_text="two"),
                ],
            )

    def test_extractor_preserves_page_and_block_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "sample.pdf")
            with pymupdf.open() as pdf:
                page = pdf.new_page()
                page.insert_text((72, 72), "First paragraph")
                page.insert_text((72, 110), "Second paragraph")
                pdf.save(path)

            document = PyMuPDFExtractor().extract(path)
            self.assertEqual(document.summary()["pages"], 1)
            self.assertGreaterEqual(document.summary()["text_blocks"], 1)
            orders = [block.order for block in document.pages[0].ordered_blocks()]
            self.assertEqual(orders, sorted(orders))

    def test_batch_translation_maps_results_by_stable_id(self) -> None:
        page = Page(
            page_number=1,
            width=100,
            height=100,
            blocks=[
                TextBlock(
                    block_id="p0001-b00000",
                    page_number=1,
                    order=0,
                    bbox=BoundingBox(0, 0, 10, 10),
                    source_text="alpha",
                ),
                TextBlock(
                    block_id="p0001-b00001",
                    page_number=1,
                    order=1,
                    bbox=BoundingBox(0, 20, 10, 30),
                    source_text="beta",
                ),
            ],
        )
        document = Document(source_path=Path("sample.pdf"), pages=[page])
        translator = BatchTranslator(
            UppercaseProvider(), max_characters=5, max_segments=1, concurrency=2
        )

        asyncio.run(translator.translate_document(document, "el"))

        blocks = list(document.text_blocks())
        self.assertEqual([block.translated_text for block in blocks], ["ALPHA", "BETA"])
        self.assertTrue(
            all(block.translation_status is TranslationStatus.TRANSLATED for block in blocks)
        )

    def test_missing_translation_is_an_explicit_failure(self) -> None:
        block = TextBlock(
            block_id="p0001-b00000",
            page_number=1,
            order=0,
            bbox=BoundingBox(0, 0, 10, 10),
            source_text="alpha",
        )
        document = Document(
            source_path=Path("sample.pdf"),
            pages=[Page(page_number=1, width=100, height=100, blocks=[block])],
        )
        translator = BatchTranslator(IncompleteProvider(), retries=0)

        asyncio.run(translator.translate_document(document, "el"))

        self.assertIs(block.translation_status, TranslationStatus.FAILED)
        self.assertIsNone(block.translated_text)
        self.assertEqual(block.display_text, "alpha")


if __name__ == "__main__":
    unittest.main()

