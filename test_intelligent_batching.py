#!/usr/bin/env python3
"""
Test script for Intelligent Content Batcher

This script tests the intelligent batching system to ensure it works correctly
before integrating with the main pipeline.
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
        
        # Simulate API delay
        await asyncio.sleep(0.1)
        
        # Simple mock translation (just add prefix)
        if target_language == 'Greek':
            return f"[TRANSLATED TO GREEK] {text}"
        else:
            return f"[TRANSLATED] {text}"


def create_mock_content():
    """Create mock content for testing"""
    
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
        ),
        MockTextArea(
            combined_text="SUBSECTION HEADER",
            layout_info=MockLayoutInfo("title", (50, 300, 400, 330), 0.9),
            text_blocks=[],
            image_blocks=[]
        ),
        MockTextArea(
            combined_text="This paragraph discusses the subsection in detail. It contains more specific information about the topic.",
            layout_info=MockLayoutInfo("paragraph", (50, 340, 400, 380), 0.8),
            text_blocks=[],
            image_blocks=[]
        )
    ]
    
    # Create mock non-text areas
    non_text_areas = [
        MockNonTextArea(
            layout_info=MockLayoutInfo("figure", (450, 100, 600, 300), 0.9),
            image_blocks=[{}]
        ),
        MockNonTextArea(
            layout_info=MockLayoutInfo("table", (50, 400, 400, 500), 0.8),
            image_blocks=[{}]
        )
    ]
    
    return text_areas, non_text_areas


async def test_intelligent_batching():
    """Test the intelligent batching system"""
    
    logger.info("🧪 Testing Intelligent Content Batcher...")
    
    try:
        # Import the intelligent batcher
        from intelligent_content_batcher import IntelligentContentBatcher
        
        # Create mock content
        text_areas, non_text_areas = create_mock_content()
        
        # Create mock Gemini service
        mock_gemini = MockGeminiService()
        
        # Create intelligent batcher
        batcher = IntelligentContentBatcher(
            gemini_service=mock_gemini,
            max_batch_tokens=2000,  # Smaller for testing
            max_concurrent_batches=2  # Smaller for testing
        )
        
        logger.info(f"📄 Created {len(text_areas)} text areas and {len(non_text_areas)} non-text areas")
        
        # Test intelligent processing
        start_time = time.time()
        result = await batcher.process_content_intelligently(
            text_areas, non_text_areas, 'Greek'
        )
        total_time = time.time() - start_time
        
        # Analyze results
        if result['success']:
            logger.info("✅ Intelligent batching test PASSED")
            logger.info(f"   Total processing time: {total_time:.3f}s")
            logger.info(f"   Gemini API calls: {mock_gemini.call_count}")
            logger.info(f"   Batches created: {result['statistics']['total_batches']}")
            logger.info(f"   Successful batches: {result['statistics']['successful_batches']}")
            logger.info(f"   Total tokens: {result['statistics']['total_tokens']}")
            logger.info(f"   Average batch time: {result['statistics']['average_batch_time']:.3f}s")
            
            # Check reconstructed content
            reconstructed = result['reconstructed_content']
            logger.info(f"   Reconstructed content items: {len(reconstructed)}")
            
            # Show some translated content
            for i, item in enumerate(reconstructed[:3]):
                if item['type'] == 'translated_text':
                    logger.info(f"   Item {i+1}: {item['content'][:100]}...")
            
            # Performance analysis
            original_expected_time = len(text_areas) * 0.1  # 0.1s per item
            actual_time = total_time
            speedup = original_expected_time / actual_time if actual_time > 0 else 0
            
            logger.info(f"   Expected sequential time: {original_expected_time:.3f}s")
            logger.info(f"   Actual parallel time: {actual_time:.3f}s")
            logger.info(f"   Speedup: {speedup:.2f}x")
            
            if speedup > 1.5:
                logger.info("🚀 Significant performance improvement achieved!")
            else:
                logger.warning("⚠️ Performance improvement less than expected")
            
            return True
            
        else:
            logger.error(f"❌ Intelligent batching test FAILED: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_batch_creation():
    """Test batch creation logic"""
    
    logger.info("🧪 Testing batch creation logic...")
    
    try:
        from intelligent_content_batcher import IntelligentContentBatcher
        
        # Create mock content
        text_areas, non_text_areas = create_mock_content()
        
        # Create batcher
        batcher = IntelligentContentBatcher(None, max_batch_tokens=2000, max_concurrent_batches=2)
        
        # Test batch creation
        batches = batcher._create_semantic_batches(text_areas, non_text_areas)
        
        logger.info(f"✅ Created {len(batches)} batches")
        
        for i, batch in enumerate(batches):
            logger.info(f"   Batch {i+1}: {batch.batch_type} ({batch.total_tokens} tokens, {len(batch.content_items)} items)")
            
            # Show content types in batch
            content_types = [item['content_type'].value for item in batch.content_items]
            logger.info(f"      Content types: {content_types}")
            
            # Show first few words of each item
            for j, item in enumerate(batch.content_items[:3]):
                text_preview = item['text'][:50] + "..." if len(item['text']) > 50 else item['text']
                logger.info(f"      Item {j+1}: {text_preview}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Batch creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    
    logger.info("🚀 Starting Intelligent Content Batcher Tests")
    logger.info("=" * 60)
    
    # Test 1: Batch creation
    test1_passed = await test_batch_creation()
    
    # Test 2: Full intelligent batching
    test2_passed = await test_intelligent_batching()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info(f"   Batch creation test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    logger.info(f"   Intelligent batching test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests PASSED! Intelligent batching system is ready for integration.")
        return True
    else:
        logger.error("❌ Some tests FAILED. Please fix issues before integration.")
        return False


if __name__ == "__main__":
    asyncio.run(main()) 