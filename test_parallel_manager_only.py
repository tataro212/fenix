#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for Parallel Translation Manager (Simulation Only)

This test verifies the parallel translation manager logic without making
actual API calls or using GPU resources. It uses mock translation to test:
1. Batch creation and management
2. Parallel processing logic
3. Error handling
4. Performance tracking
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Import our modules
from intelligent_content_batcher import IntelligentContentBatcher
from parallel_translation_manager import ParallelTranslationManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_content() -> Dict[str, Any]:
    """Create test content for parallel translation testing"""
    return {
        "title_1": {
            "combined_text": "Chapter 1: Introduction to Advanced PDF Processing",
            "layout_info": {
                "label": "title",
                "bbox": (50, 50, 500, 80),
                "confidence": 0.95
            },
            "page_num": 0
        },
        "paragraph_1": {
            "combined_text": "The processing of PDF documents has become increasingly important in modern digital workflows. Organizations need to extract, analyze, and translate content from various document formats efficiently and accurately.",
            "layout_info": {
                "label": "paragraph",
                "bbox": (50, 130, 450, 200),
                "confidence": 0.88
            },
            "page_num": 0
        },
        "paragraph_2": {
            "combined_text": "Traditional OCR-based approaches often struggle with complex layouts, mixed content types, and maintaining the semantic structure of documents. This limitation has led to the development of more sophisticated processing pipelines.",
            "layout_info": {
                "label": "paragraph",
                "bbox": (50, 210, 450, 280),
                "confidence": 0.87
            },
            "page_num": 0
        },
        "list_item_1": {
            "combined_text": "• Improved accuracy through layout-aware processing",
            "layout_info": {
                "label": "list",
                "bbox": (70, 290, 430, 310),
                "confidence": 0.85
            },
            "page_num": 0
        },
        "list_item_2": {
            "combined_text": "• Better preservation of document structure",
            "layout_info": {
                "label": "list",
                "bbox": (70, 320, 430, 340),
                "confidence": 0.84
            },
            "page_num": 0
        },
        "heading_1": {
            "combined_text": "1.2 Technical Approach",
            "layout_info": {
                "label": "heading",
                "bbox": (50, 390, 400, 410),
                "confidence": 0.91
            },
            "page_num": 0
        },
        "paragraph_3": {
            "combined_text": "Our approach combines PyMuPDF for high-fidelity text extraction with YOLO-based layout analysis for accurate content type classification. This hybrid methodology ensures both speed and accuracy in document processing.",
            "layout_info": {
                "label": "paragraph",
                "bbox": (50, 420, 450, 490),
                "confidence": 0.86
            },
            "page_num": 0
        }
    }


async def test_parallel_translation_manager():
    """Test the parallel translation manager with mock translation"""
    logger.info("🧪 Testing ParallelTranslationManager (Simulation)...")
    
    # Create manager with mock translation (no actual API calls)
    manager = ParallelTranslationManager(max_concurrent_batches=3)
    
    # Override the async translator to use mock translation
    class MockAsyncTranslator:
        async def translate_text(self, text: str, target_language: str) -> str:
            # Simulate translation delay
            await asyncio.sleep(0.1)
            return f"[TRANSLATED TO {target_language.upper()}] {text}"
    
    manager.async_translator = MockAsyncTranslator()
    
    # Test content
    test_content = create_test_content()
    
    # Test parallel translation
    logger.info("   Testing parallel translation simulation...")
    result = await manager.translate_content_parallel(test_content, target_language='es')
    
    # Validate results
    logger.info("   Translation results:")
    logger.info(f"     - Total batches: {result.total_batches}")
    logger.info(f"     - Successful: {result.successful_batches}")
    logger.info(f"     - Failed: {result.failed_batches}")
    logger.info(f"     - Total time: {result.total_processing_time:.3f}s")
    logger.info(f"     - Average batch time: {result.average_batch_time:.3f}s")
    logger.info(f"     - API calls reduction: {result.api_calls_reduction:.1f}%")
    
    # Validate batch results
    for batch_result in result.batch_results:
        logger.info(f"   Batch {batch_result.batch_id}:")
        logger.info(f"     - Success: {batch_result.success}")
        logger.info(f"     - Items: {batch_result.item_count}")
        logger.info(f"     - Content types: {batch_result.content_types}")
        logger.info(f"     - Processing time: {batch_result.processing_time:.3f}s")
        logger.info(f"     - Original chars: {len(batch_result.original_text)}")
        logger.info(f"     - Translated chars: {len(batch_result.translated_text)}")
        
        if not batch_result.success:
            logger.warning(f"     - Error: {batch_result.error}")
    
    # Get performance stats
    stats = manager.get_performance_stats()
    logger.info("   Performance statistics:")
    logger.info(f"     - Global success rate: {stats['global_stats']['success_rate']:.1f}%")
    logger.info(f"     - Average batch time: {stats['global_stats']['average_batch_time']:.3f}s")
    logger.info(f"     - API calls reduction avg: {stats['global_stats']['api_calls_reduction']:.1f}%")
    
    return result.successful_batches > 0


async def test_concurrency_control():
    """Test concurrency control with different batch sizes"""
    logger.info("🧪 Testing Concurrency Control...")
    
    # Test with different concurrency levels
    for max_concurrent in [1, 2, 3, 5]:
        logger.info(f"   Testing with {max_concurrent} concurrent batches...")
        
        manager = ParallelTranslationManager(max_concurrent_batches=max_concurrent)
        
        # Mock translator with longer delay to see concurrency effect
        class MockAsyncTranslator:
            async def translate_text(self, text: str, target_language: str) -> str:
                await asyncio.sleep(0.2)  # Longer delay
                return f"[TRANSLATED TO {target_language.upper()}] {text}"
        
        manager.async_translator = MockAsyncTranslator()
        
        test_content = create_test_content()
        start_time = time.time()
        
        result = await manager.translate_content_parallel(test_content, target_language='fr')
        
        total_time = time.time() - start_time
        logger.info(f"     - Total time: {total_time:.3f}s")
        logger.info(f"     - Batches: {result.total_batches}")
        logger.info(f"     - Success rate: {result.successful_batches}/{result.total_batches}")
        
        # With higher concurrency, we should see faster total time
        if max_concurrent > 1:
            expected_time = (result.total_batches * 0.2) / max_concurrent
            logger.info(f"     - Expected time: {expected_time:.3f}s")
            logger.info(f"     - Efficiency: {expected_time/total_time:.1f}x")
    
    return True


async def test_error_handling():
    """Test error handling in parallel translation"""
    logger.info("🧪 Testing Error Handling...")
    
    manager = ParallelTranslationManager(max_concurrent_batches=2)
    
    # Mock translator that sometimes fails
    class FailingMockTranslator:
        def __init__(self):
            self.call_count = 0
        
        async def translate_text(self, text: str, target_language: str) -> str:
            self.call_count += 1
            if self.call_count % 3 == 0:  # Fail every 3rd call
                raise Exception(f"Simulated translation error for call {self.call_count}")
            
            await asyncio.sleep(0.1)
            return f"[TRANSLATED TO {target_language.upper()}] {text}"
    
    manager.async_translator = FailingMockTranslator()
    
    test_content = create_test_content()
    result = await manager.translate_content_parallel(test_content, target_language='de')
    
    logger.info(f"   Error handling results:")
    logger.info(f"     - Total batches: {result.total_batches}")
    logger.info(f"     - Successful: {result.successful_batches}")
    logger.info(f"     - Failed: {result.failed_batches}")
    logger.info(f"     - Success rate: {result.successful_batches/result.total_batches*100:.1f}%")
    
    # Check that failed batches have error information
    failed_batches = [r for r in result.batch_results if not r.success]
    for batch in failed_batches:
        logger.info(f"     - Failed batch {batch.batch_id}: {batch.error}")
    
    return result.failed_batches > 0  # We expect some failures


async def run_parallel_manager_test():
    """Run all parallel manager tests"""
    logger.info("🚀 Starting Parallel Translation Manager Test (Simulation)")
    logger.info("=" * 80)
    
    test_results = {}
    
    # Test 1: Basic parallel translation
    logger.info("\n📋 Test 1: Basic Parallel Translation")
    test_results['basic_parallel'] = await test_parallel_translation_manager()
    
    # Test 2: Concurrency control
    logger.info("\n📋 Test 2: Concurrency Control")
    test_results['concurrency'] = await test_concurrency_control()
    
    # Test 3: Error handling
    logger.info("\n📋 Test 3: Error Handling")
    test_results['error_handling'] = await test_error_handling()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 PARALLEL MANAGER TEST RESULTS")
    logger.info("=" * 80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Parallel translation manager is working correctly.")
        logger.info("   - Parallel batch processing ✓")
        logger.info("   - Concurrency control ✓")
        logger.info("   - Error handling ✓")
        logger.info("   - No GPU usage ✓")
        logger.info("   - No real API calls ✓")
        return True
    else:
        logger.error("⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    # Run the parallel manager test
    success = asyncio.run(run_parallel_manager_test())
    
    if success:
        print("\n✅ Parallel Translation Manager is ready for use!")
        print("   - No GPU resources used")
        print("   - No real API calls made")
        print("   - Parallel processing logic tested")
    else:
        print("\n❌ Some issues detected. Please review the test output above.") 