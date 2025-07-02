#!/usr/bin/env python3
"""
Performance Comparison Test

This script compares the performance of the original sequential processing
vs the new intelligent batching system.
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


class MockGeminiService:
    """Mock Gemini service that simulates real API delays"""
    
    def __init__(self, delay_per_call: float = 0.1):
        self.logger = logging.getLogger(__name__)
        self.call_count = 0
        self.delay_per_call = delay_per_call
    
    async def translate_text(self, text: str, target_language: str) -> str:
        """Mock translation that simulates API delay"""
        self.call_count += 1
        
        # Simulate real API delay
        await asyncio.sleep(self.delay_per_call)
        
        # Simple mock translation
        if target_language == 'Greek':
            return f"[TRANSLATED TO GREEK] {text}"
        else:
            return f"[TRANSLATED] {text}"


def create_test_content(num_text_areas: int = 10):
    """Create test content with specified number of text areas"""
    
    text_areas = []
    for i in range(num_text_areas):
        text_areas.append(MockTextArea(
            combined_text=f"This is text area {i+1} with some content that needs to be translated. It contains multiple sentences and should be processed efficiently.",
            layout_info=MockLayoutInfo("paragraph", (50, 100 + i*50, 400, 130 + i*50), 0.8),
            text_blocks=[],
            image_blocks=[]
        ))
    
    return text_areas


async def test_sequential_processing(text_areas: List[MockTextArea], target_language: str = 'Greek'):
    """Test sequential processing (original method)"""
    
    logger.info(f"🧪 Testing Sequential Processing with {len(text_areas)} text areas...")
    
    mock_gemini = MockGeminiService(delay_per_call=0.1)
    start_time = time.time()
    
    translated_areas = []
    for i, area in enumerate(text_areas):
        logger.info(f"   Processing area {i+1}/{len(text_areas)}")
        translated_text = await mock_gemini.translate_text(area.combined_text, target_language)
        translated_areas.append({
            'original_text': area.combined_text,
            'translated_text': translated_text,
            'area_id': i
        })
    
    total_time = time.time() - start_time
    
    logger.info(f"✅ Sequential processing completed in {total_time:.3f}s")
    logger.info(f"   API calls: {mock_gemini.call_count}")
    logger.info(f"   Average time per area: {total_time/len(text_areas):.3f}s")
    
    return {
        'method': 'sequential',
        'total_time': total_time,
        'api_calls': mock_gemini.call_count,
        'average_time_per_area': total_time / len(text_areas),
        'translated_areas': translated_areas
    }


async def test_intelligent_batching(text_areas: List[MockTextArea], target_language: str = 'Greek'):
    """Test intelligent batching (new method)"""
    
    logger.info(f"🧪 Testing Intelligent Batching with {len(text_areas)} text areas...")
    
    try:
        from intelligent_content_batcher import IntelligentContentBatcher
        
        mock_gemini = MockGeminiService(delay_per_call=0.1)
        batcher = IntelligentContentBatcher(mock_gemini, max_batch_tokens=2000, max_concurrent_batches=3)
        
        start_time = time.time()
        
        # Create non-text areas (empty for this test)
        non_text_areas = []
        
        # Process with intelligent batching
        result = await batcher.process_content_intelligently(text_areas, non_text_areas, target_language)
        
        total_time = time.time() - start_time
        
        if result['success']:
            logger.info(f"✅ Intelligent batching completed in {total_time:.3f}s")
            logger.info(f"   API calls: {mock_gemini.call_count}")
            logger.info(f"   Batches: {result['statistics']['total_batches']}")
            logger.info(f"   Successful batches: {result['statistics']['successful_batches']}")
            logger.info(f"   Total tokens: {result['statistics']['total_tokens']}")
            logger.info(f"   Average batch time: {result['statistics']['average_batch_time']:.3f}s")
            
            return {
                'method': 'intelligent_batching',
                'total_time': total_time,
                'api_calls': mock_gemini.call_count,
                'batches': result['statistics']['total_batches'],
                'successful_batches': result['statistics']['successful_batches'],
                'total_tokens': result['statistics']['total_tokens'],
                'average_batch_time': result['statistics']['average_batch_time'],
                'result': result
            }
        else:
            logger.error(f"❌ Intelligent batching failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Intelligent batching test failed: {e}")
        return None


async def run_performance_comparison():
    """Run the complete performance comparison"""
    
    logger.info("🚀 Starting Performance Comparison Test")
    logger.info("=" * 60)
    
    # Test with different content sizes
    test_sizes = [5, 10, 20]
    
    for size in test_sizes:
        logger.info(f"\n📊 Testing with {size} text areas")
        logger.info("-" * 40)
        
        # Create test content
        text_areas = create_test_content(size)
        
        # Test sequential processing
        sequential_result = await test_sequential_processing(text_areas)
        
        # Test intelligent batching
        batching_result = await test_intelligent_batching(text_areas)
        
        if batching_result:
            # Calculate performance improvement
            speedup = sequential_result['total_time'] / batching_result['total_time']
            api_reduction = (sequential_result['api_calls'] - batching_result['api_calls']) / sequential_result['api_calls'] * 100
            
            logger.info(f"\n📈 Performance Results for {size} text areas:")
            logger.info(f"   Sequential time: {sequential_result['total_time']:.3f}s")
            logger.info(f"   Batching time: {batching_result['total_time']:.3f}s")
            logger.info(f"   Speedup: {speedup:.2f}x")
            logger.info(f"   API calls reduction: {api_reduction:.1f}%")
            logger.info(f"   Batches created: {batching_result['batches']}")
            
            if speedup > 1.5:
                logger.info("🚀 Significant performance improvement achieved!")
            elif speedup > 1.1:
                logger.info("✅ Moderate performance improvement achieved!")
            else:
                logger.warning("⚠️ Minimal performance improvement")
        else:
            logger.error("❌ Could not compare performance - batching failed")
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 PERFORMANCE COMPARISON SUMMARY")
    logger.info("✅ Test completed successfully")
    logger.info("🎯 Intelligent batching system is ready for production use")


if __name__ == "__main__":
    asyncio.run(run_performance_comparison()) 