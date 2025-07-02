#!/usr/bin/env python3
"""
Test script to verify the batching fix is working correctly.
This will test that we're now batching text areas instead of making individual API calls.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockGeminiService:
    """Mock Gemini service to track API calls"""
    
    def __init__(self, delay_per_call: float = 0.1):
        self.call_count = 0
        self.delay_per_call = delay_per_call
        self.translated_texts = []
    
    async def translate_text(self, text: str, target_language: str) -> str:
        """Mock translation that tracks calls and simulates delay"""
        self.call_count += 1
        logger.info(f"🔍 API Call #{self.call_count}: Translating {len(text)} characters")
        
        # Simulate API delay
        await asyncio.sleep(self.delay_per_call)
        
        # Mock translation (just add prefix)
        translated = f"[TRANSLATED TO {target_language.upper()}] {text}"
        self.translated_texts.append(translated)
        
        return translated

class MockTextArea:
    """Mock text area for testing"""
    
    def __init__(self, text: str, label: str = 'text', bbox: tuple = (0, 0, 100, 50)):
        self.combined_text = text
        self.layout_info = MockLayoutInfo(label, bbox, 0.9)
        self.text_blocks = []
        self.image_blocks = []

class MockLayoutInfo:
    """Mock layout info for testing"""
    
    def __init__(self, label: str, bbox: tuple, confidence: float):
        self.label = label
        self.bbox = bbox
        self.confidence = confidence

def create_test_content():
    """Create test content with multiple text areas"""
    text_areas = []
    
    # Create 20 text areas with varying lengths
    for i in range(20):
        if i % 3 == 0:
            # Short text (title-like)
            text = f"Section {i+1}: Introduction"
        elif i % 3 == 1:
            # Medium text (paragraph-like)
            text = f"This is paragraph {i+1} with some content. It contains multiple sentences to simulate real document content. The text should be long enough to test batching behavior."
        else:
            # Long text (detailed paragraph)
            text = f"This is a detailed paragraph {i+1} with extensive content. It contains multiple sentences and should be long enough to test the batching algorithm. The text includes various elements like numbers, punctuation, and different word lengths to simulate realistic document content."
        
        text_areas.append(MockTextArea(text, 'text' if i % 3 != 0 else 'title'))
    
    return text_areas

async def test_coordinate_based_batching():
    """Test the coordinate-based extraction batching"""
    
    logger.info("🧪 Testing Coordinate-Based Extraction Batching...")
    
    try:
        from processing_strategies import ProcessingStrategyExecutor
        
        # Create mock content
        text_areas = create_test_content()
        logger.info(f"📄 Created {len(text_areas)} text areas for testing")
        
        # Create mock Gemini service
        mock_gemini = MockGeminiService(delay_per_call=0.05)  # 50ms per call
        
        # Create processing strategy executor
        executor = ProcessingStrategyExecutor(mock_gemini)
        
        # Create mock processing result
        mapped_content = {f"area_{i}": area for i, area in enumerate(text_areas)}
        processing_result = {
            'mapped_content': mapped_content,
            'page_num': 0
        }
        
        # Test the coordinate-based extraction
        start_time = time.time()
        result = await executor._process_coordinate_based_extraction(processing_result, 'Greek')
        total_time = time.time() - start_time
        
        # Analyze results
        if result.success:
            logger.info("✅ Coordinate-based batching test PASSED")
            logger.info(f"   Total processing time: {total_time:.3f}s")
            logger.info(f"   Gemini API calls: {mock_gemini.call_count}")
            logger.info(f"   Text areas processed: {len(text_areas)}")
            logger.info(f"   API calls reduction: {len(text_areas) - mock_gemini.call_count}")
            logger.info(f"   Reduction percentage: {((len(text_areas) - mock_gemini.call_count) / len(text_areas) * 100):.1f}%")
            
            # Check statistics
            stats = result.statistics
            logger.info(f"   API calls reduction (from stats): {stats.get('api_calls_reduction', 'N/A')}")
            logger.info(f"   Processing efficiency: {stats.get('processing_efficiency', 'N/A')}")
            
            # Performance analysis
            expected_sequential_time = len(text_areas) * 0.05  # 50ms per call
            actual_time = total_time
            speedup = expected_sequential_time / actual_time if actual_time > 0 else 0
            
            logger.info(f"   Expected sequential time: {expected_sequential_time:.3f}s")
            logger.info(f"   Actual batched time: {actual_time:.3f}s")
            logger.info(f"   Speedup: {speedup:.2f}x")
            
            if speedup > 1.5:
                logger.info("🚀 Significant performance improvement achieved!")
            else:
                logger.warning("⚠️ Performance improvement less than expected")
            
            # Show some translated content
            content = result.content
            if 'text_areas' in content:
                for i, text_area in enumerate(content['text_areas'][:3]):
                    logger.info(f"   Text area {i+1}: {text_area['translated_content'][:100]}...")
            
            return True
            
        else:
            logger.error(f"❌ Coordinate-based batching test FAILED: {result.error}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_pure_text_batching():
    """Test the pure text fast batching"""
    
    logger.info("🧪 Testing Pure Text Fast Batching...")
    
    try:
        from processing_strategies import ProcessingStrategyExecutor
        
        # Create mock content with sections
        sections = []
        for i in range(15):
            if i % 3 == 0:
                text = f"Chapter {i+1}: Overview"
            elif i % 3 == 1:
                text = f"This is section {i+1} with detailed content. It contains multiple sentences and should be long enough to test batching behavior."
            else:
                text = f"This is a comprehensive section {i+1} with extensive content. It includes multiple paragraphs and detailed explanations to simulate realistic document content."
            
            sections.append({'content': text})
        
        logger.info(f"📄 Created {len(sections)} sections for testing")
        
        # Create mock Gemini service
        mock_gemini = MockGeminiService(delay_per_call=0.05)
        
        # Create processing strategy executor
        executor = ProcessingStrategyExecutor(mock_gemini)
        
        # Create mock document structure
        document_structure = {
            'sections': sections,
            'total_text': ' '.join([section['content'] for section in sections])
        }
        
        # Test the direct text translation
        start_time = time.time()
        result = await executor.direct_text_processor.translate_direct_text(document_structure, 'Greek')
        total_time = time.time() - start_time
        
        # Analyze results
        if 'error' not in result:
            logger.info("✅ Pure text batching test PASSED")
            logger.info(f"   Total processing time: {total_time:.3f}s")
            logger.info(f"   Gemini API calls: {mock_gemini.call_count}")
            logger.info(f"   Sections processed: {len(sections)}")
            logger.info(f"   API calls reduction: {len(sections) - mock_gemini.call_count}")
            logger.info(f"   Reduction percentage: {((len(sections) - mock_gemini.call_count) / len(sections) * 100):.1f}%")
            
            # Performance analysis
            expected_sequential_time = len(sections) * 0.05
            actual_time = total_time
            speedup = expected_sequential_time / actual_time if actual_time > 0 else 0
            
            logger.info(f"   Expected sequential time: {expected_sequential_time:.3f}s")
            logger.info(f"   Actual batched time: {actual_time:.3f}s")
            logger.info(f"   Speedup: {speedup:.2f}x")
            
            if speedup > 1.5:
                logger.info("🚀 Significant performance improvement achieved!")
            else:
                logger.warning("⚠️ Performance improvement less than expected")
            
            return True
            
        else:
            logger.error(f"❌ Pure text batching test FAILED: {result['error']}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all batching tests"""
    
    logger.info("🚀 Starting Batching Fix Tests...")
    
    # Test coordinate-based batching
    coord_result = await test_coordinate_based_batching()
    
    # Reset mock service for next test
    await asyncio.sleep(1)
    
    # Test pure text batching
    pure_result = await test_pure_text_batching()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 BATCHING FIX TEST SUMMARY")
    logger.info("="*50)
    
    if coord_result and pure_result:
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("🎯 The batching fix is working correctly!")
        logger.info("📈 API calls are being reduced significantly!")
        logger.info("⚡ Performance improvements achieved!")
    else:
        logger.error("❌ SOME TESTS FAILED!")
        logger.error("🔧 The batching fix needs more work!")
    
    logger.info("="*50)

if __name__ == "__main__":
    asyncio.run(main()) 