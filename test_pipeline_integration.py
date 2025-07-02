#!/usr/bin/env python3
"""
Test Pipeline Integration with Intelligent Batching

This script tests the integration of intelligent batching with the main pipeline.
"""

import asyncio
import logging
import time
from typing import List, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MockLayoutInfo:
    """Mock layout info for testing"""
    label: str
    bbox: tuple
    confidence: float


@dataclass
class MockTextArea:
    """Mock text area for testing"""
    combined_text: str
    layout_info: MockLayoutInfo
    text_blocks: List[Any]
    image_blocks: List[Any]


@dataclass
class MockNonTextArea:
    """Mock non-text area for testing"""
    layout_info: MockLayoutInfo
    image_blocks: List[Any]


class MockGeminiService:
    """Mock Gemini service for testing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.call_count = 0
    
    async def translate_text(self, text: str, target_language: str) -> str:
        """Mock translation that simulates API delay and format preservation"""
        self.call_count += 1
        
        # Simulate API delay (shorter for testing)
        await asyncio.sleep(0.05)
        
        # Simple mock translation (just add prefix)
        if target_language == 'Greek':
            return f"[TRANSLATED TO GREEK] {text}"
        else:
            return f"[TRANSLATED] {text}"


def create_mock_processing_result():
    """Create mock processing result for testing"""
    
    # Create mock text areas
    text_areas = [
        MockTextArea(
            combined_text="TITLE OF THE DOCUMENT",
            layout_info=MockLayoutInfo("title", (50, 100, 400, 130), 0.9),
            text_blocks=[],
            image_blocks=[]
        ),
        MockTextArea(
            combined_text="This is the first paragraph of the document. It contains some important information about the topic being discussed.",
            layout_info=MockLayoutInfo("paragraph", (50, 140, 400, 180), 0.8),
            text_blocks=[],
            image_blocks=[]
        ),
        MockTextArea(
            combined_text="• First item in the list",
            layout_info=MockLayoutInfo("list", (50, 190, 400, 210), 0.8),
            text_blocks=[],
            image_blocks=[]
        ),
        MockTextArea(
            combined_text="• Second item in the list",
            layout_info=MockLayoutInfo("list", (50, 220, 400, 240), 0.8),
            text_blocks=[],
            image_blocks=[]
        ),
        MockTextArea(
            combined_text="This is another paragraph that follows the list. It provides additional context and information.",
            layout_info=MockLayoutInfo("paragraph", (50, 250, 400, 290), 0.8),
            text_blocks=[],
            image_blocks=[]
        )
    ]
    
    # Create mock non-text areas
    non_text_areas = [
        MockNonTextArea(
            layout_info=MockLayoutInfo("figure", (450, 100, 600, 300), 0.9),
            image_blocks=[{}]
        )
    ]
    
    # Create mapped content
    mapped_content = {}
    for i, area in enumerate(text_areas + non_text_areas):
        mapped_content[f'area_{i}'] = area
    
    return {
        'mapped_content': mapped_content,
        'text_blocks': [],
        'image_blocks': [],
        'layout_areas': [],
        'page_num': 0
    }


async def test_pipeline_integration():
    """Test the integration of intelligent batching with the main pipeline"""
    
    logger.info("🧪 Testing Pipeline Integration with Intelligent Batching...")
    
    try:
        # Import the processing strategy executor
        from processing_strategies import ProcessingStrategyExecutor
        
        # Create mock content
        processing_result = create_mock_processing_result()
        
        # Create mock Gemini service
        mock_gemini = MockGeminiService()
        
        # Create strategy executor
        strategy_executor = ProcessingStrategyExecutor(mock_gemini)
        
        logger.info(f"📄 Created mock processing result with {len(processing_result['mapped_content'])} areas")
        
        # Test coordinate-based extraction with intelligent batching
        start_time = time.time()
        result = await strategy_executor._process_coordinate_based_extraction(
            processing_result, 'Greek'
        )
        total_time = time.time() - start_time
        
        # Analyze results
        if result.success:
            logger.info("✅ Pipeline integration test PASSED")
            logger.info(f"   Total processing time: {total_time:.3f}s")
            logger.info(f"   Strategy: {result.strategy}")
            logger.info(f"   Gemini API calls: {mock_gemini.call_count}")
            logger.info(f"   Processing time: {result.processing_time:.3f}s")
            
            # Check statistics
            stats = result.statistics
            logger.info(f"   Total areas: {stats.get('total_areas', 0)}")
            logger.info(f"   Text areas: {stats.get('text_areas_count', 0)}")
            logger.info(f"   Non-text areas: {stats.get('non_text_areas_count', 0)}")
            logger.info(f"   Total batches: {stats.get('total_batches', 0)}")
            logger.info(f"   Successful batches: {stats.get('successful_batches', 0)}")
            logger.info(f"   Total tokens: {stats.get('total_tokens', 0)}")
            logger.info(f"   Average batch time: {stats.get('average_batch_time', 0):.3f}s")
            logger.info(f"   Coordinate precision: {stats.get('coordinate_precision', 'N/A')}")
            logger.info(f"   Processing efficiency: {stats.get('processing_efficiency', 'N/A')}")
            
            # Check content structure
            content = result.content
            if 'final_content' in content:
                final_content = content['final_content']
                logger.info(f"   Final content items: {len(final_content)}")
                
                # Show some content
                for i, item in enumerate(final_content[:3]):
                    if item['type'] == 'translated_text':
                        logger.info(f"   Content {i+1}: {item['content'][:100]}...")
            
            # Performance analysis
            expected_sequential_time = len([k for k, v in processing_result['mapped_content'].items() 
                                          if hasattr(v, 'combined_text') and v.combined_text]) * 0.05
            actual_time = total_time
            speedup = expected_sequential_time / actual_time if actual_time > 0 else 0
            
            logger.info(f"   Expected sequential time: {expected_sequential_time:.3f}s")
            logger.info(f"   Actual parallel time: {actual_time:.3f}s")
            logger.info(f"   Speedup: {speedup:.2f}x")
            
            if speedup > 1.5:
                logger.info("🚀 Significant performance improvement achieved!")
            else:
                logger.warning("⚠️ Performance improvement less than expected")
            
            return True
            
        else:
            logger.error(f"❌ Pipeline integration test FAILED: {result.error}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fallback_method():
    """Test the fallback method works correctly"""
    
    logger.info("🧪 Testing Fallback Method...")
    
    try:
        from processing_strategies import ProcessingStrategyExecutor
        
        # Create mock content
        processing_result = create_mock_processing_result()
        
        # Create mock Gemini service
        mock_gemini = MockGeminiService()
        
        # Create strategy executor
        strategy_executor = ProcessingStrategyExecutor(mock_gemini)
        
        # Test fallback method directly
        text_areas = [v for v in processing_result['mapped_content'].values() 
                     if hasattr(v, 'combined_text') and v.combined_text]
        non_text_areas = [v for v in processing_result['mapped_content'].values() 
                         if not hasattr(v, 'combined_text') or not v.combined_text]
        
        start_time = time.time()
        result = await strategy_executor._fallback_coordinate_extraction(
            text_areas, non_text_areas, processing_result['mapped_content'], 'Greek', start_time
        )
        
        if result.success:
            logger.info("✅ Fallback method test PASSED")
            logger.info(f"   Processing time: {result.processing_time:.3f}s")
            logger.info(f"   Strategy: {result.strategy}")
            return True
        else:
            logger.error(f"❌ Fallback method test FAILED: {result.error}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Fallback test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    
    logger.info("🚀 Starting Pipeline Integration Tests")
    logger.info("=" * 60)
    
    # Test 1: Pipeline integration with intelligent batching
    test1_passed = await test_pipeline_integration()
    
    # Test 2: Fallback method
    test2_passed = await test_fallback_method()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info(f"   Pipeline integration test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    logger.info(f"   Fallback method test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests PASSED! Pipeline integration is ready for production.")
        return True
    else:
        logger.error("❌ Some tests FAILED. Please fix issues before production use.")
        return False


if __name__ == "__main__":
    asyncio.run(main()) 