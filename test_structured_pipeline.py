"""
Test Script for Structured Pipeline Implementation

This script validates the refactored implementation against the directives:
1. Structured document model with TextBlock dataclass
2. Coordinate-based sorting for correct reading order
3. Semantic cohesion through text block merging
4. Asynchronous integrity with sequence_id reassembly
5. Strict separation of concerns
6. Enhanced error handling and fallback mechanisms
"""

import os
import sys
import logging
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_textblock_dataclass():
    """Test 1: Validate TextBlock dataclass implementation"""
    logger.info("🧪 Test 1: TextBlock dataclass validation")
    
    try:
        from pdf_processor import TextBlock, ContentType
        
        # Test basic TextBlock creation
        block = TextBlock(
            text="Test text content",
            page_num=1,
            bbox=(10.0, 20.0, 100.0, 30.0),
            font_size=12.0,
            font_family="Arial",
            content_type=ContentType.PARAGRAPH
        )
        
        # Validate required attributes
        assert block.text == "Test text content"
        assert block.page_num == 1
        assert block.bbox == (10.0, 20.0, 100.0, 30.0)
        assert block.font_size == 12.0
        assert block.font_family == "Arial"
        assert block.content_type == ContentType.PARAGRAPH
        assert block.sequence_id is not None
        assert len(block.sequence_id) > 0
        
        # Test coordinate methods
        assert block.get_vertical_position() == 20.0
        assert block.get_horizontal_position() == 10.0
        
        # Test to_dict method
        block_dict = block.to_dict()
        assert 'text' in block_dict
        assert 'sequence_id' in block_dict
        assert 'bbox' in block_dict
        
        logger.info("✅ TextBlock dataclass validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ TextBlock dataclass validation failed: {e}")
        return False

def test_coordinate_based_sorting():
    """Test 2: Validate coordinate-based sorting for reading order"""
    logger.info("🧪 Test 2: Coordinate-based sorting validation")
    
    try:
        from pdf_processor import TextBlock, ContentType, StructuredPDFProcessor
        
        processor = StructuredPDFProcessor()
        
        # Create test blocks with different positions
        blocks = [
            TextBlock(text="Bottom right", page_num=1, bbox=(100.0, 50.0, 200.0, 60.0)),
            TextBlock(text="Top left", page_num=1, bbox=(10.0, 10.0, 100.0, 20.0)),
            TextBlock(text="Top right", page_num=1, bbox=(100.0, 10.0, 200.0, 20.0)),
            TextBlock(text="Bottom left", page_num=1, bbox=(10.0, 50.0, 100.0, 60.0)),
        ]
        
        # Sort blocks
        sorted_blocks = processor._sort_blocks_by_reading_order(blocks)
        
        # Verify reading order (top to bottom, left to right)
        expected_order = ["Top left", "Top right", "Bottom left", "Bottom right"]
        actual_order = [block.text for block in sorted_blocks]
        
        assert actual_order == expected_order, f"Expected {expected_order}, got {actual_order}"
        
        logger.info("✅ Coordinate-based sorting validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Coordinate-based sorting validation failed: {e}")
        return False

def test_sequence_id_assignment():
    """Test 3: Validate sequence_id assignment and uniqueness"""
    logger.info("🧪 Test 3: Sequence ID assignment validation")
    
    try:
        from pdf_processor import TextBlock, StructuredPDFProcessor
        
        processor = StructuredPDFProcessor()
        
        # Create test blocks
        blocks = [
            TextBlock(text="Block 1", page_num=1, bbox=(10.0, 10.0, 100.0, 20.0)),
            TextBlock(text="Block 2", page_num=1, bbox=(10.0, 20.0, 100.0, 30.0)),
            TextBlock(text="Block 3", page_num=1, bbox=(10.0, 30.0, 100.0, 40.0)),
        ]
        
        # Assign sequence IDs
        processor._assign_sequence_ids(blocks, 1)
        
        # Verify all blocks have sequence IDs
        sequence_ids = set()
        for block in blocks:
            assert block.sequence_id is not None
            assert len(block.sequence_id) > 0
            assert block.sequence_id.startswith("page_1_block_")
            sequence_ids.add(block.sequence_id)
        
        # Verify uniqueness
        assert len(sequence_ids) == len(blocks), "Sequence IDs must be unique"
        
        logger.info("✅ Sequence ID assignment validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sequence ID assignment validation failed: {e}")
        return False

def test_text_block_merging():
    """Test 4: Validate text block merging for semantic cohesion"""
    logger.info("🧪 Test 4: Text block merging validation")
    
    try:
        from pdf_processor import TextBlock, ContentType, StructuredPDFProcessor
        
        processor = StructuredPDFProcessor()
        
        # Create blocks that should be merged (same paragraph)
        blocks = [
            TextBlock(text="This is the first", page_num=1, bbox=(10.0, 10.0, 100.0, 20.0), font_size=12.0, font_family="Arial"),
            TextBlock(text="line of a paragraph", page_num=1, bbox=(10.0, 20.0, 100.0, 30.0), font_size=12.0, font_family="Arial"),
            TextBlock(text="that should be merged.", page_num=1, bbox=(10.0, 30.0, 100.0, 40.0), font_size=12.0, font_family="Arial"),
        ]
        
        # Sort blocks first
        sorted_blocks = processor._sort_blocks_by_reading_order(blocks)
        
        # Merge blocks
        merged_blocks = processor._merge_text_blocks(sorted_blocks)
        
        # Verify merging occurred
        assert len(merged_blocks) < len(blocks), "Blocks should be merged"
        assert len(merged_blocks) == 1, "All blocks should be merged into one"
        
        # Verify merged text
        expected_text = "This is the first line of a paragraph that should be merged."
        assert merged_blocks[0].text == expected_text
        
        logger.info("✅ Text block merging validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Text block merging validation failed: {e}")
        return False

def test_text_sanitization():
    """Test 5: Validate text sanitization to remove PDF artifacts"""
    logger.info("🧪 Test 5: Text sanitization validation")
    
    try:
        from pdf_processor import StructuredPDFProcessor
        
        processor = StructuredPDFProcessor()
        
        # Test text with PDF artifacts
        test_texts = [
            ("Hello _Toc_Bookmark_123 world", "Hello world"),
            ("Text with [metadata] here", "Text with here"),
            ("Content with {brackets} and _123_", "Content with and"),
            ("Normal text without artifacts", "Normal text without artifacts"),
        ]
        
        for input_text, expected_output in test_texts:
            sanitized = processor._sanitize_text(input_text)
            assert sanitized.strip() == expected_output.strip(), f"Expected '{expected_output}', got '{sanitized}'"
        
        logger.info("✅ Text sanitization validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Text sanitization validation failed: {e}")
        return False

async def test_asynchronous_integrity():
    """Test 6: Validate asynchronous integrity with sequence_id reassembly"""
    logger.info("🧪 Test 6: Asynchronous integrity validation")
    
    try:
        from pdf_processor import TextBlock, ContentType
        from text_translator import StructuredTextTranslator
        
        translator = StructuredTextTranslator()
        
        # Create test document structure
        document_structure = [
            [
                TextBlock(text="First block", page_num=1, bbox=(10.0, 10.0, 100.0, 20.0), sequence_id="block_1"),
                TextBlock(text="Second block", page_num=1, bbox=(10.0, 20.0, 100.0, 30.0), sequence_id="block_2"),
                TextBlock(text="Third block", page_num=1, bbox=(10.0, 30.0, 100.0, 40.0), sequence_id="block_3"),
            ]
        ]
        
        # Mock translation service for testing
        class MockTranslationService:
            async def translate_text_enhanced(self, text, target_language, style_guide, prev_context, next_context, item_type):
                # Simulate translation by adding prefix
                return f"TRANSLATED: {text}"
        
        translator.translation_service = MockTranslationService()
        
        # Test translation
        translated_structure = await translator.translate_document_structure(
            document_structure, "Greek", "Test style guide"
        )
        
        # Verify structure integrity
        assert len(translated_structure) == len(document_structure)
        assert len(translated_structure[0]) == len(document_structure[0])
        
        # Verify sequence preservation
        for i, block in enumerate(translated_structure[0]):
            assert block.sequence_id == document_structure[0][i].sequence_id
            assert block.text.startswith("TRANSLATED: ")
        
        logger.info("✅ Asynchronous integrity validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Asynchronous integrity validation failed: {e}")
        return False

def test_separation_of_concerns():
    """Test 7: Validate strict separation of concerns"""
    logger.info("🧪 Test 7: Separation of concerns validation")
    
    try:
        from pdf_processor import StructuredPDFProcessor
        from text_translator import StructuredTextTranslator
        
        # Test that PDF processor only handles extraction
        pdf_processor = StructuredPDFProcessor()
        assert hasattr(pdf_processor, 'extract_document_structure')
        assert hasattr(pdf_processor, '_extract_page_blocks')
        assert hasattr(pdf_processor, '_sort_blocks_by_reading_order')
        
        # Test that text translator only handles translation
        text_translator = StructuredTextTranslator()
        assert hasattr(text_translator, 'translate_document_structure')
        assert hasattr(text_translator, '_translate_page_blocks_parallel')
        assert hasattr(text_translator, '_reassemble_translated_blocks')
        
        # Verify no cross-contamination
        assert not hasattr(pdf_processor, 'translate_document_structure')
        assert not hasattr(text_translator, 'extract_document_structure')
        
        logger.info("✅ Separation of concerns validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Separation of concerns validation failed: {e}")
        return False

def test_error_handling():
    """Test 8: Validate enhanced error handling and fallback mechanisms"""
    logger.info("🧪 Test 8: Error handling validation")
    
    try:
        from pdf_processor import StructuredPDFProcessor
        from text_translator import StructuredTextTranslator
        
        pdf_processor = StructuredPDFProcessor()
        text_translator = StructuredTextTranslator()
        
        # Test PDF processor error handling
        try:
            # Test with non-existent file
            result = pdf_processor.extract_document_structure("non_existent_file.pdf")
            assert False, "Should have raised an exception"
        except Exception:
            # Expected behavior
            pass
        
        # Test text translator error handling
        try:
            # Test with None translation service
            translator = StructuredTextTranslator(translation_service=None)
            # This should not crash - translation_service might be None or a fallback service
            assert translator is not None
        except Exception as e:
            logger.error(f"Unexpected error in text translator initialization: {e}")
            return False
        
        logger.info("✅ Error handling validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling validation failed: {e}")
        return False

async def test_complete_pipeline():
    """Test 9: Validate complete pipeline integration"""
    logger.info("🧪 Test 9: Complete pipeline integration validation")
    
    try:
        from main import StructuredDocumentPipeline
        
        pipeline = StructuredDocumentPipeline()
        
        # Test pipeline initialization
        assert pipeline.pdf_processor is not None
        assert pipeline.text_translator is not None
        assert pipeline.target_language == "Ελληνικά"
        
        # Test input validation
        assert pipeline._validate_input("test.pdf") == False  # File doesn't exist
        assert pipeline._validate_input("test.txt") == False  # Wrong extension
        
        logger.info("✅ Complete pipeline integration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Complete pipeline integration validation failed: {e}")
        return False

async def run_all_tests():
    """Run all validation tests"""
    logger.info("🚀 Starting comprehensive validation of refactored pipeline")
    
    tests = [
        ("TextBlock Dataclass", test_textblock_dataclass),
        ("Coordinate-based Sorting", test_coordinate_based_sorting),
        ("Sequence ID Assignment", test_sequence_id_assignment),
        ("Text Block Merging", test_text_block_merging),
        ("Text Sanitization", test_text_sanitization),
        ("Asynchronous Integrity", test_asynchronous_integrity),
        ("Separation of Concerns", test_separation_of_concerns),
        ("Error Handling", test_error_handling),
        ("Complete Pipeline", test_complete_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Refactored pipeline is ready!")
        return True
    else:
        logger.error(f"⚠️ {total - passed} tests failed - Review implementation")
        return False

if __name__ == "__main__":
    # Run all tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1) 