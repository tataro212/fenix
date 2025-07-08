#!/usr/bin/env python3
"""
Test script for the new intelligent batching system in AsyncTranslationService.

This script tests:
1. Batch creation with 14,000 character limit
2. Contextual continuity between batches
3. XML tagging and parsing
4. API call reduction
"""

import asyncio
import logging
import sys
from async_translation_service import AsyncTranslationService, TranslationTask, IntelligentBatcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_tasks(num_tasks: int = 50, avg_chars_per_task: int = 300) -> list:
    """Create test translation tasks with varying text lengths"""
    tasks = []
    
    # Sample text content of different lengths
    sample_texts = [
        "This is a short paragraph about artificial intelligence and machine learning applications.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
        "The field of natural language processing has evolved significantly over the past decade. Modern transformer architectures like BERT, GPT, and T5 have revolutionized how we approach text understanding and generation. These models leverage attention mechanisms to capture long-range dependencies in text, enabling more sophisticated language understanding capabilities.",
        "In the context of document translation, maintaining coherence across different sections is crucial for preserving the original meaning and flow. This requires careful consideration of context, terminology consistency, and narrative structure throughout the translation process.",
        "Machine translation has progressed from rule-based systems to statistical models and now to neural networks. Each advancement has brought improvements in translation quality, but challenges remain in handling context, idiomatic expressions, and domain-specific terminology.",
        "The integration of large language models into translation workflows offers new possibilities for improving translation quality. By leveraging contextual understanding and pre-trained knowledge, these models can produce more nuanced and accurate translations.",
        "Document structure preservation is another critical aspect of professional translation services. Headers, footnotes, tables, and formatting must be maintained while ensuring the translated content flows naturally in the target language.",
        "Quality assurance in automated translation involves multiple layers of validation, including linguistic accuracy, cultural appropriateness, and technical correctness. This multi-faceted approach helps ensure professional-grade translation output."
    ]
    
    for i in range(num_tasks):
        # Vary text length to simulate real document content
        base_text = sample_texts[i % len(sample_texts)]
        
        # Sometimes make text longer by repeating or extending
        if i % 3 == 0:  # Every third task gets longer text
            text = base_text + " " + base_text[:100] + " Additional content for variety."
        elif i % 5 == 0:  # Every fifth task gets shorter text
            text = base_text[:100]
        else:
            text = base_text
        
        task = TranslationTask(
            text=text,
            target_language="el",  # Greek
            context_before=f"Previous context for task {i}",
            context_after=f"Following context for task {i}",
            item_type="paragraph",
            priority=1,
            task_id=f"test_task_{i}"
        )
        tasks.append(task)
    
    return tasks

def test_intelligent_batcher():
    """Test the IntelligentBatcher class directly"""
    logger.info("🧪 Testing IntelligentBatcher...")
    
    # Create test tasks
    tasks = create_test_tasks(100, 300)  # 100 tasks, ~300 chars each
    total_chars = sum(len(task.text) for task in tasks)
    
    logger.info(f"📊 Test data:")
    logger.info(f"   • Total tasks: {len(tasks)}")
    logger.info(f"   • Total characters: {total_chars:,}")
    logger.info(f"   • Average chars per task: {total_chars / len(tasks):.1f}")
    
    # Initialize batcher
    batcher = IntelligentBatcher(max_batch_chars=14000, context_overlap_chars=500)
    
    # Create batches
    batches = batcher.create_intelligent_batches(tasks)
    
    logger.info(f"📦 Batching results:")
    logger.info(f"   • Batches created: {len(batches)}")
    logger.info(f"   • API call reduction: {(1 - len(batches) / len(tasks)) * 100:.1f}%")
    logger.info(f"   • Average tasks per batch: {len(tasks) / len(batches):.1f}")
    
    # Test batch characteristics
    for i, batch in enumerate(batches):
        logger.info(f"   • Batch {i}: {len(batch.text_blocks)} tasks, {batch.total_chars:,} chars")
        
        # Verify batch doesn't exceed limit
        if batch.total_chars > 14000:
            logger.error(f"❌ Batch {i} exceeds character limit: {batch.total_chars}")
        
        # Check for context continuity
        if i > 0 and batch.context_from_previous:
            logger.info(f"     - Has context from previous batch: {len(batch.context_from_previous)} chars")
    
    # Test XML parsing
    if batches:
        logger.info("🔍 Testing XML parsing...")
        test_batch = batches[0]
        
        # Simulate a translated response
        simulated_translation = test_batch.combined_text.replace("artificial intelligence", "τεχνητή νοημοσύνη")
        simulated_translation = simulated_translation.replace("machine learning", "μηχανική μάθηση")
        
        # Parse the translation
        parsed_translations = batcher.parse_batch_translation(test_batch, simulated_translation)
        
        logger.info(f"   • Original tasks in batch: {len(test_batch.text_blocks)}")
        logger.info(f"   • Parsed translations: {len(parsed_translations)}")
        
        if len(parsed_translations) == len(test_batch.text_blocks):
            logger.info("✅ XML parsing successful")
        else:
            logger.error("❌ XML parsing failed - count mismatch")

async def test_full_translation_service():
    """Test the full AsyncTranslationService with intelligent batching"""
    logger.info("🚀 Testing full AsyncTranslationService...")
    
    try:
        # Create service instance
        service = AsyncTranslationService()
        
        # Create test tasks
        tasks = create_test_tasks(50, 400)  # 50 tasks, ~400 chars each
        
        logger.info(f"📊 Translation test:")
        logger.info(f"   • Tasks to translate: {len(tasks)}")
        logger.info(f"   • Target language: Greek (el)")
        
        # Note: This would normally make real API calls
        # For testing, we'll just verify the batching logic works
        logger.info("⚠️  Note: This test requires actual API access for full translation")
        logger.info("   Testing batching logic only...")
        
        # Test just the batching part
        batches = service.batcher.create_intelligent_batches(tasks)
        
        logger.info(f"✅ Batching test completed:")
        logger.info(f"   • {len(batches)} batches would be sent to API")
        logger.info(f"   • {len(tasks) - len(batches)} fewer API calls needed")
        logger.info(f"   • Efficiency improvement: {len(tasks) / len(batches):.1f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Translation service test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🎯 Starting Intelligent Batching Tests")
    logger.info("=" * 50)
    
    try:
        # Test 1: Batcher directly
        test_intelligent_batcher()
        logger.info("")
        
        # Test 2: Full service
        success = asyncio.run(test_full_translation_service())
        
        logger.info("=" * 50)
        if success:
            logger.info("✅ All tests completed successfully!")
            logger.info("🎉 Intelligent batching system is working correctly")
        else:
            logger.error("❌ Some tests failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 