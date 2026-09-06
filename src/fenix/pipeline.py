"""The single production orchestration path for the refactored package."""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .domain import Document
from .extraction import PyMuPDFExtractor
from .translation import BatchTranslator, TranslationProvider


class Pipeline:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        extractor: PyMuPDFExtractor | None = None,
        translation_provider: TranslationProvider | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.extractor = extractor or PyMuPDFExtractor()
        self.translation_provider = translation_provider

    async def process(
        self, pdf_path: str | Path, *, target_language: str | None = None
    ) -> Document:
        document = self.extractor.extract(pdf_path)
        if target_language is None:
            return document
        if self.translation_provider is None:
            raise RuntimeError("A translation provider is required when target_language is set")

        translator = BatchTranslator(
            self.translation_provider,
            max_characters=self.settings.translation_batch_characters,
            max_segments=self.settings.translation_batch_segments,
            concurrency=self.settings.translation_concurrency,
            retries=self.settings.translation_retries,
        )
        return await translator.translate_document(document, target_language)

