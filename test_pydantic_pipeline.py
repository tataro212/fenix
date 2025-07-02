import pytest
import asyncio
from pymupdf_yolo_processor import PyMuPDFYOLOProcessor, PageModel
from processing_strategies import ProcessingStrategyExecutor

class MockGeminiService:
    async def translate_text(self, text, target_language, timeout=30.0):
        # Mock translation: just append language code
        return f"{text} [{target_language}]"

def test_process_page_schema():
    processor = PyMuPDFYOLOProcessor()
    # Use a sample PDF from the workspace
    pdf_path = 'test_document_with_text.pdf'
    page_num = 0
    # Run the async process_page in a synchronous test
    result = asyncio.run(processor.process_page(pdf_path, page_num))
    # Validate schema
    page_model = PageModel(**result)
    assert page_model.page_number == page_num
    assert 'width' in page_model.dimensions and 'height' in page_model.dimensions
    assert isinstance(page_model.elements, list)
    assert all(hasattr(el, 'type') for el in page_model.elements)

def test_translate_direct_text_with_pydantic_schema():
    processor = PyMuPDFYOLOProcessor()
    pdf_path = 'test_document_with_text.pdf'
    page_num = 0
    result = asyncio.run(processor.process_page(pdf_path, page_num))
    # Prepare the strategy executor with a mock translation service
    executor = ProcessingStrategyExecutor(gemini_service=MockGeminiService())
    # Run the async translate_direct_text in a synchronous test
    translated = asyncio.run(executor.direct_text_processor.translate_direct_text(result, target_language='fr'))
    assert 'translated_content' in translated
    assert '[fr]' in translated['translated_content']
    assert translated['type'] == 'translated_text_document' 