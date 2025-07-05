#!/usr/bin/env python3
"""
Performance Test: Concurrent vs Sequential Translation

This script demonstrates the performance improvements achieved by the
new concurrent translation method in processing_strategies.py.

Expected results:
- Sequential: ~10-20 seconds for 5 chunks
- Concurrent: ~3-5 seconds for 5 chunks  
- Performance gain: 3-5x faster
"""

import asyncio
import time
import logging
from unittest.mock import AsyncMock
from processing_strategies import DirectTextProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockGeminiService:
    """Mock Gemini service that simulates realistic API response times"""
    
    async def translate_text(self, text: str, target_language: str) -> str:
        # Simulate realistic API latency (1-3 seconds per request)
        await asyncio.sleep(2.0)  # Simulate network delay
        
        # Return mock translated response with proper XML structure
        segments = []
        import re
        seg_pattern = re.compile(r'<seg id="(\d+)">(.*?)</seg>', re.DOTALL)
        matches = seg_pattern.findall(text)
        
        for seg_id, content in matches:
            # Mock translation: just add [TRANSLATED] prefix
            translated_content = f"[TRANSLATED] {content}"
            segments.append(f'<seg id="{seg_id}">{translated_content}</seg>')
        
        return '<root>' + ''.join(segments) + '</root>'

def create_test_elements(num_elements: int = 20) -> list[dict]:
    """Create test text elements that will be chunked"""
    elements = []
    
    for i in range(num_elements):
        elements.append({
            'text': f"This is test paragraph {i+1}. " * 50,  # ~50 words per element
            'bbox': [0, i*20, 100, (i+1)*20],
            'label': 'paragraph',
            'confidence': 0.95
        })
    
    return elements

async def test_sequential_performance():
    """Test the original sequential translation method"""
    logger.info("🐌 Testing SEQUENTIAL translation performance...")
    
    mock_service = MockGeminiService()
    processor = DirectTextProcessor(mock_service)
    
    test_elements = create_test_elements(20)  # This should create ~5 chunks
    
    start_time = time.time()
    result = await processor.translate_direct_text(test_elements, 'spanish')
    sequential_time = time.time() - start_time
    
    logger.info(f"📊 Sequential Results:")
    logger.info(f"   Time taken: {sequential_time:.2f} seconds")
    logger.info(f"   Elements processed: {len(test_elements)}")
    logger.info(f"   Translated blocks: {len(result)}")
    
    return sequential_time, len(result)

async def test_concurrent_performance():
    """Test the new concurrent translation method"""
    logger.info("🚀 Testing CONCURRENT translation performance...")
    
    mock_service = MockGeminiService()
    processor = DirectTextProcessor(mock_service)
    
    test_elements = create_test_elements(20)  # Same test data
    
    start_time = time.time()
    result = await processor.translate_direct_text_concurrent(test_elements, 'spanish')
    concurrent_time = time.time() - start_time
    
    logger.info(f"📊 Concurrent Results:")
    logger.info(f"   Time taken: {concurrent_time:.2f} seconds")
    logger.info(f"   Elements processed: {len(test_elements)}")
    logger.info(f"   Translated blocks: {len(result)}")
    
    return concurrent_time, len(result)

async def main():
    """Run performance comparison"""
    logger.info("🎯 Starting Performance Comparison Test")
    logger.info("=" * 60)
    
    try:
        # Test sequential approach
        sequential_time, sequential_blocks = await test_sequential_performance()
        
        logger.info("\n" + "-" * 40 + "\n")
        
        # Test concurrent approach  
        concurrent_time, concurrent_blocks = await test_concurrent_performance()
        
        # Calculate performance improvement
        if concurrent_time > 0:
            speedup = sequential_time / concurrent_time
            time_saved = sequential_time - concurrent_time
            
            logger.info("\n" + "=" * 60)
            logger.info("📈 PERFORMANCE COMPARISON RESULTS")
            logger.info("=" * 60)
            logger.info(f"Sequential time:    {sequential_time:.2f} seconds")
            logger.info(f"Concurrent time:    {concurrent_time:.2f} seconds")
            logger.info(f"Time saved:         {time_saved:.2f} seconds ({time_saved/sequential_time*100:.1f}%)")
            logger.info(f"Performance gain:   {speedup:.1f}x faster")
            logger.info(f"Blocks verified:    {sequential_blocks == concurrent_blocks}")
            
            if speedup >= 3.0:
                logger.info("✅ EXCELLENT: Achieved 3x+ performance improvement!")
            elif speedup >= 2.0:
                logger.info("✅ GOOD: Achieved 2x+ performance improvement!")
            else:
                logger.info("⚠️ WARNING: Performance improvement less than expected")
                
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 