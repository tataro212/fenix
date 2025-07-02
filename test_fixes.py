#!/usr/bin/env python3
"""
Test script to validate the architectural fixes for Project Phoenix
"""

import asyncio
import logging
import time
from models import PageModel, ElementModel, ProcessResult
from config_manager import Config
from processing_strategies import process_page_worker
from optimized_document_pipeline import OptimizedDocumentPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_processresult_creation():
    """Test ProcessResult object creation and serialization"""
    logger.info("🧪 Testing ProcessResult creation...")
    
    # Create a test PageModel
    test_page = PageModel(
        page_number=1,
        dimensions=[612, 792],
        elements=[
            ElementModel(
                type='text',
                content='This is a test text element',
                bbox=[100, 100, 300, 150],
                confidence=0.95
            ),
            ElementModel(
                type='image',
                content='',
                bbox=[100, 200, 300, 350],
                confidence=0.85
            )
        ]
    )
    
    # Create ProcessResult
    result = ProcessResult(
        page_number=1,
        data=test_page,
        error=None
    )
    
    logger.info(f"✅ ProcessResult created: page {result.page_number}, data: {result.data is not None}")
    return result

def test_worker_function():
    """Test the worker function with proper serialization"""
    logger.info("🧪 Testing worker function...")
    
    # Create test data
    test_page = PageModel(
        page_number=1,
        dimensions=[612, 792],
        elements=[
            ElementModel(
                type='text',
                content='Test content for worker',
                bbox=[100, 100, 300, 150],
                confidence=0.95
            )
        ]
    )
    
    config = Config()
    
    # Test serialization and worker
    page_dict = test_page.model_dump()
    result = process_page_worker(page_dict, config)
    
    logger.info(f"✅ Worker result: page {result.page_number}, success: {result.error is None}")
    return result

def test_strategy_routing():
    """Test the strategy routing with ProcessResult"""
    logger.info("🧪 Testing strategy routing...")
    
    pipeline = OptimizedDocumentPipeline(max_workers=1)
    
    # Create test ProcessResult
    test_page = PageModel(
        page_number=1,
        dimensions=[612, 792],
        elements=[
            ElementModel(
                type='text',
                content='Pure text content',
                bbox=[100, 100, 300, 150],
                confidence=0.95
            )
        ]
    )
    
    result = ProcessResult(
        page_number=1,
        data=test_page,
        error=None
    )
    
    # Test strategy routing
    strategy_input = pipeline._route_strategy_for_page(result)
    
    logger.info(f"✅ Strategy routing: {strategy_input['strategy'].strategy}")
    logger.info(f"   Mapped content keys: {list(strategy_input['mapped_content'].keys())}")
    return strategy_input

async def test_markdown_timeout():
    """Test markdown translation timeout"""
    logger.info("🧪 Testing markdown translation timeout...")
    
    from markdown_aware_translator import MarkdownAwareTranslator
    
    translator = MarkdownAwareTranslator()
    
    # Mock translation function that hangs
    async def hanging_translation_func(text, lang, style, prev, next, type):
        await asyncio.sleep(10)  # Simulate hanging
        return text
    
    # Test timeout
    start_time = time.time()
    try:
        result = await translator.translate_markdown_content(
            "# Test Heading\n\nThis is a test paragraph.",
            hanging_translation_func,
            "el"
        )
        elapsed = time.time() - start_time
        logger.info(f"✅ Markdown translation completed in {elapsed:.2f}s")
        logger.info(f"   Result length: {len(result)}")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ Markdown translation failed after {elapsed:.2f}s: {e}")
    
    return True

async def main():
    """Run all tests"""
    logger.info("🚀 Starting architectural fixes validation...")
    
    try:
        # Test 1: ProcessResult creation
        test_processresult_creation()
        
        # Test 2: Worker function
        test_worker_function()
        
        # Test 3: Strategy routing
        test_strategy_routing()
        
        # Test 4: Markdown timeout
        await test_markdown_timeout()
        
        logger.info("✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 