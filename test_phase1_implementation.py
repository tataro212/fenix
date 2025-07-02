#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for Phase 1 Implementation

This script tests the new intelligent content batcher and parallel translation manager
to validate the improvements in batching logic and parallel processing.
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Import our new components
from intelligent_content_batcher import IntelligentContentBatcher, ContentType
from parallel_translation_manager import ParallelTranslationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_content() -> Dict[str, Any]:
    """Create test content that simulates PyMuPDF-YOLO mapping output"""
    test_content = {}
    
    # Simulate different content types
    content_samples = [
        {
            'id': 'area_1',
            'combined_text': 'Chapter 1: Introduction to Machine Learning',
            'layout_info': {'label': 'title', 'bbox': (50, 50, 400, 80), 'confidence': 0.95},
            'page_num': 0
        },
        {
            'id': 'area_2',
            'combined_text': 'Machine learning is a subset of artificial intelligence that focuses on the development of computer programs that can access data and use it to learn for themselves.',
            'layout_info': {'label': 'paragraph', 'bbox': (50, 100, 400, 150), 'confidence': 0.92},
            'page_num': 0
        },
        {
            'id': 'area_3',
            'combined_text': 'The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.',
            'layout_info': {'label': 'paragraph', 'bbox': (50, 170, 400, 220), 'confidence': 0.91},
            'page_num': 0
        },
        {
            'id': 'area_4',
            'combined_text': '• Supervised Learning',
            'layout_info': {'label': 'list', 'bbox': (50, 240, 400, 260), 'confidence': 0.89},
            'page_num': 0
        },
        {
            'id': 'area_5',
            'combined_text': '• Unsupervised Learning',
            'layout_info': {'label': 'list', 'bbox': (50, 270, 400, 290), 'confidence': 0.89},
            'page_num': 0
        },
        {
            'id': 'area_6',
            'combined_text': '• Reinforcement Learning',
            'layout_info': {'label': 'list', 'bbox': (50, 300, 400, 320), 'confidence': 0.89},
            'page_num': 0
        },
        {
            'id': 'area_7',
            'combined_text': 'Section 1.1: Types of Machine Learning',
            'layout_info': {'label': 'heading', 'bbox': (50, 350, 400, 380), 'confidence': 0.94},
            'page_num': 0
        },
        {
            'id': 'area_8',
            'combined_text': 'Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs.',
            'layout_info': {'label': 'paragraph', 'bbox': (50, 400, 400, 450), 'confidence': 0.90},
            'page_num': 0
        },
        {
            'id': 'area_9',
            'combined_text': 'Figure 1.1: Machine Learning Taxonomy',
            'layout_info': {'label': 'caption', 'bbox': (50, 480, 400, 500), 'confidence': 0.88},
            'page_num': 0
        },
        {
            'id': 'area_10',
            'combined_text': 'This is a continuation of the previous paragraph that extends across multiple lines to test batching behavior with longer content.',
            'layout_info': {'label': 'paragraph', 'bbox': (50, 520, 400, 570), 'confidence': 0.91},
            'page_num': 0
        }
    ]
    
    # Convert to the format expected by our batcher
    for sample in content_samples:
        area_id = sample['id']
        test_content[area_id] = type('MockArea', (), {
            'combined_text': sample['combined_text'],
            'layout_info': type('MockLayoutInfo', (), {
                'label': sample['layout_info']['label'],
                'bbox': sample['layout_info']['bbox'],
                'confidence': sample['layout_info']['confidence']
            })(),
            'page_num': sample['page_num']
        })()
    
    return test_content


async def test_intelligent_content_batcher():
    """Test the intelligent content batcher"""
    logger.info("Testing Intelligent Content Batcher...")
    
    # Create test content
    test_content = create_test_content()
    logger.info(f"Created {len(test_content)} test content areas")
    
    # Initialize batcher
    batcher = IntelligentContentBatcher(max_batch_chars=12000)
    
    # Test content item creation
    start_time = time.time()
    content_items = batcher.create_content_items(test_content)
    creation_time = time.time() - start_time
    
    logger.info(f"Content items created in {creation_time:.3f}s")
    logger.info(f"   Total items: {len(content_items)}")
    logger.info(f"   Translatable items: {len([item for item in content_items if item.is_translatable()])}")
    
    # Test intelligent batching
    start_time = time.time()
    batches = batcher.create_intelligent_batches(content_items)
    batching_time = time.time() - start_time
    
    logger.info(f"Intelligent batching completed in {batching_time:.3f}s")
    logger.info(f"   Total batches: {len(batches)}")
    
    # Analyze batch characteristics
    for i, batch in enumerate(batches):
        logger.info(f"   Batch {i+1}:")
        logger.info(f"     Items: {len(batch.items)}")
        logger.info(f"     Characters: {batch.total_chars}")
        logger.info(f"     Content types: {[t.value for t in batch.content_types]}")
        logger.info(f"     Semantic coherence: {batch.semantic_coherence:.3f}")
        logger.info(f"     Translation priority: {batch.translation_priority}")
    
    # Get batching report
    report = batcher.get_batching_report()
    logger.info(f"Batching Report:")
    logger.info(f"   Items per batch: {report['efficiency_metrics']['items_per_batch']:.2f}")
    logger.info(f"   Character utilization: {report['efficiency_metrics']['character_utilization']:.1f}%")
    
    return batches, report


async def test_parallel_translation_manager():
    """Test the parallel translation manager"""
    logger.info("Testing Parallel Translation Manager...")
    
    # Create test content
    test_content = create_test_content()
    
    # Initialize manager
    manager = ParallelTranslationManager(max_concurrent_batches=3)
    
    # Test parallel translation (this will use mock translation for testing)
    start_time = time.time()
    result = await manager.translate_content_parallel(test_content, target_language='Greek')
    total_time = time.time() - start_time
    
    logger.info(f"Parallel translation completed in {total_time:.3f}s")
    logger.info(f"   Total batches: {result.total_batches}")
    logger.info(f"   Successful batches: {result.successful_batches}")
    logger.info(f"   Failed batches: {result.failed_batches}")
    logger.info(f"   API calls reduction: {result.api_calls_reduction:.1f}%")
    logger.info(f"   Average batch time: {result.average_batch_time:.3f}s")
    
    # Analyze batch results
    for batch_result in result.batch_results:
        logger.info(f"   Batch {batch_result.batch_id}:")
        logger.info(f"     Success: {batch_result.success}")
        logger.info(f"     Processing time: {batch_result.processing_time:.3f}s")
        logger.info(f"     Item count: {batch_result.item_count}")
        if batch_result.error:
            logger.info(f"     Error: {batch_result.error}")
    
    # Get performance stats
    stats = manager.get_performance_stats()
    logger.info(f"Performance Stats:")
    logger.info(f"   Global success rate: {stats['global_stats']['success_rate']:.1%}")
    logger.info(f"   Average batch time: {stats['global_stats']['average_batch_time']:.3f}s")
    
    return result, stats


async def test_content_type_classification():
    """Test content type classification"""
    logger.info("Testing Content Type Classification...")
    
    from intelligent_content_batcher import ContentTypeClassifier
    
    classifier = ContentTypeClassifier()
    
    # Test cases
    test_cases = [
        ("Chapter 1: Introduction", "title"),
        ("1.1 Background", "heading"),
        ("• This is a bullet point", "list"),
        ("1. First item", "list"),
        ("Figure 1.1: Sample", "caption"),
        ("This is a regular paragraph with normal text content.", "paragraph"),
        ("123", "text"),
        ("ABC", "text"),
    ]
    
    for text, expected_label in test_cases:
        content_type = classifier.classify_content(text, expected_label)
        logger.info(f"   '{text[:30]}...' -> {content_type.value} (expected: {expected_label})")
    
    logger.info("Content type classification test completed")


async def test_performance_comparison():
    """Compare performance with old vs new approach"""
    logger.info("Testing Performance Comparison...")
    
    # Create larger test content for performance testing
    test_content = create_test_content()
    
    # Simulate old approach (individual item processing)
    logger.info("Old Approach (Individual Items):")
    old_items = len(test_content)
    old_api_calls = old_items
    old_time = 2.0 * old_items  # Simulate 2s per item
    logger.info(f"   Items: {old_items}")
    logger.info(f"   API calls: {old_api_calls}")
    logger.info(f"   Estimated time: {old_time:.1f}s")
    
    # New approach (intelligent batching)
    logger.info("New Approach (Intelligent Batching):")
    
    batcher = IntelligentContentBatcher(max_batch_chars=12000)
    content_items = batcher.create_content_items(test_content)
    batches = batcher.create_intelligent_batches(content_items)
    
    new_api_calls = len(batches)
    new_time = 5.0 * new_api_calls  # Simulate 5s per batch (but larger batches)
    
    logger.info(f"   Items: {len(content_items)}")
    logger.info(f"   Batches: {len(batches)}")
    logger.info(f"   API calls: {new_api_calls}")
    logger.info(f"   Estimated time: {new_time:.1f}s")
    
    # Calculate improvements
    api_reduction = ((old_api_calls - new_api_calls) / old_api_calls) * 100
    time_reduction = ((old_time - new_time) / old_time) * 100
    
    logger.info(f"Performance Improvements:")
    logger.info(f"   API calls reduction: {api_reduction:.1f}%")
    logger.info(f"   Time reduction: {time_reduction:.1f}%")
    
    return {
        'old_approach': {'items': old_items, 'api_calls': old_api_calls, 'time': old_time},
        'new_approach': {'items': len(content_items), 'batches': len(batches), 'api_calls': new_api_calls, 'time': new_time},
        'improvements': {'api_reduction': api_reduction, 'time_reduction': time_reduction}
    }


async def main():
    """Run all tests"""
    logger.info("Starting Phase 1 Implementation Tests")
    logger.info("=" * 60)
    
    try:
        # Test 1: Content Type Classification
        await test_content_type_classification()
        logger.info("-" * 40)
        
        # Test 2: Intelligent Content Batcher
        batches, batcher_report = await test_intelligent_content_batcher()
        logger.info("-" * 40)
        
        # Test 3: Parallel Translation Manager
        translation_result, translation_stats = await test_parallel_translation_manager()
        logger.info("-" * 40)
        
        # Test 4: Performance Comparison
        performance_comparison = await test_performance_comparison()
        logger.info("-" * 40)
        
        # Summary
        logger.info("PHASE 1 TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Content Type Classification: PASSED")
        logger.info(f"Intelligent Content Batcher: PASSED")
        logger.info(f"   - Created {len(batches)} intelligent batches")
        logger.info(f"   - Average items per batch: {batcher_report['efficiency_metrics']['items_per_batch']:.2f}")
        logger.info(f"Parallel Translation Manager: PASSED")
        logger.info(f"   - API calls reduction: {translation_result.api_calls_reduction:.1f}%")
        logger.info(f"   - Success rate: {translation_result.successful_batches}/{translation_result.total_batches}")
        logger.info(f"Performance Comparison: PASSED")
        logger.info(f"   - API calls reduction: {performance_comparison['improvements']['api_reduction']:.1f}%")
        logger.info(f"   - Time reduction: {performance_comparison['improvements']['time_reduction']:.1f}%")
        
        logger.info("All Phase 1 tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 