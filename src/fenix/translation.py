"""Provider-neutral translation batching with bounded concurrency."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from .domain import Document, TextBlock, TranslationStatus


@dataclass(frozen=True, slots=True)
class TranslationSegment:
    segment_id: str
    text: str


class TranslationProvider(Protocol):
    async def translate(
        self, segments: list[TranslationSegment], target_language: str
    ) -> dict[str, str]:
        """Return one translated string for every supplied segment ID."""


class BatchTranslator:
    def __init__(
        self,
        provider: TranslationProvider,
        *,
        max_characters: int = 12_000,
        max_segments: int = 32,
        concurrency: int = 5,
        retries: int = 2,
    ) -> None:
        if min(max_characters, max_segments, concurrency) < 1 or retries < 0:
            raise ValueError("Invalid translation batching configuration")
        self.provider = provider
        self.max_characters = max_characters
        self.max_segments = max_segments
        self.concurrency = concurrency
        self.retries = retries

    async def translate_document(self, document: Document, target_language: str) -> Document:
        blocks = [block for block in document.text_blocks() if block.source_text.strip()]
        for block in blocks:
            block.translation_status = TranslationStatus.PENDING

        semaphore = asyncio.Semaphore(self.concurrency)
        batches = self._create_batches(blocks)

        async def run_batch(batch: list[TextBlock]) -> None:
            async with semaphore:
                await self._translate_batch(batch, target_language)

        await asyncio.gather(*(run_batch(batch) for batch in batches))
        document.target_language = target_language
        document.validate()
        return document

    def _create_batches(self, blocks: list[TextBlock]) -> list[list[TextBlock]]:
        batches: list[list[TextBlock]] = []
        current: list[TextBlock] = []
        characters = 0

        for block in blocks:
            size = len(block.source_text)
            if current and (
                len(current) >= self.max_segments
                or characters + size > self.max_characters
            ):
                batches.append(current)
                current = []
                characters = 0
            current.append(block)
            characters += size

        if current:
            batches.append(current)
        return batches

    async def _translate_batch(self, batch: list[TextBlock], target_language: str) -> None:
        segments = [
            TranslationSegment(segment_id=block.block_id, text=block.source_text)
            for block in batch
        ]
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                translations = await self.provider.translate(segments, target_language)
                expected = {segment.segment_id for segment in segments}
                unexpected = set(translations) - expected
                if unexpected:
                    raise ValueError(f"Provider returned unknown segment IDs: {sorted(unexpected)}")

                for block in batch:
                    translated = translations.get(block.block_id)
                    if translated is None or not translated.strip():
                        block.translation_status = TranslationStatus.FAILED
                        block.translation_error = "Provider omitted this segment"
                    else:
                        block.translated_text = translated
                        block.translation_status = TranslationStatus.TRANSLATED
                        block.translation_error = None
                return
            except Exception as error:
                last_error = error
                if attempt < self.retries:
                    await asyncio.sleep(0.25 * (2**attempt))

        for block in batch:
            block.translation_status = TranslationStatus.FAILED
            block.translation_error = str(last_error)

