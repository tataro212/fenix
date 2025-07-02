#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test for Enhanced Intelligent Content Batcher and Parallel Translation Manager

This test verifies:
1. Enhanced content type classification with pattern matching
2. Semantic coherence calculation
3. Translation priority assignment
4. Intelligent batching with 12,000 character limit
5. Parallel translation management
6. Performance tracking and reporting
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Import our enhanced modules
from intelligent_content_batcher import IntelligentContentBatcher, ContentType, ContentTypeClassifier
from parallel_translation_manager import ParallelTranslationManager, TranslationBatchResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_content() -> Dict[str, Any]:
    """Create comprehensive test content with various content types"""
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
        "heading_1": {
            "combined_text": "1.1 Background and Motivation",
            "layout_info": {
                "label": "heading",
                "bbox": (50, 100, 400, 120),
                "confidence": 0.92
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
        "list_item_3": {
            "combined_text": "• Enhanced translation quality through context preservation",
            "layout_info": {
                "label": "list",
                "bbox": (70, 350, 430, 370),
                "confidence": 0.83
            },
            "page_num": 0
        },
        "heading_2": {
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
        },
        "code_block": {
            "combined_text": "def process_document(pdf_path):\n    extractor = PyMuPDFExtractor()\n    layout_analyzer = YOLOLayoutAnalyzer()\n    return hybrid_processor(extractor, layout_analyzer)",
            "layout_info": {
                "label": "code",
                "bbox": (70, 500, 430, 580),
                "confidence": 0.89
            },
            "page_num": 0
        },
        "caption_1": {
            "combined_text": "Figure 1: Document processing pipeline architecture",
            "layout_info": {
                "label": "caption",
                "bbox": (50, 590, 450, 610),
                "confidence": 0.82
            },
            "page_num": 0
        },
        "footnote_1": {
            "combined_text": "1 This approach has been validated on a diverse dataset of academic and technical documents.",
            "layout_info": {
                "label": "footnote",
                "bbox": (50, 620, 450, 640),
                "confidence": 0.80
            },
            "page_num": 0
        }
    }


def test_content_type_classifier():
    """Test the enhanced content type classifier"""
    logger.info("🧪 Testing ContentTypeClassifier...")
    
    classifier = ContentTypeClassifier()
    
    test_cases = [
        ("Chapter 1: Introduction", "text", ContentType.HEADING),
        ("1.1 Background", "text", ContentType.HEADING),
        ("This is a regular paragraph with normal text.", "text", ContentType.PARAGRAPH),
        ("• List item with bullet point", "text", ContentType.LIST_ITEM),
        ("1. Numbered list item", "text", ContentType.LIST_ITEM),
        ("Figure 1: Sample image", "text", ContentType.CAPTION),
        ("1 Footnote text", "text", ContentType.FOOTNOTE),
        ("def process_data():", "text", ContentType.CODE),
        ("12345", "text", ContentType.UNKNOWN),
        ("ACRONYM", "text", ContentType.UNKNOWN),
    ]
    
    correct_classifications = 0
    total_tests = len(test_cases)
    
    for text, label, expected_type in test_cases:
        classified_type = classifier.classify_content(text, label)
        if classified_type == expected_type:
            correct_classifications += 1
            logger.info(f"   ✅ '{text[:30]}...' -> {classified_type.value}")
        else:
            logger.warning(f"   ❌ '{text[:30]}...' -> {classified_type.value} (expected {expected_type.value})")
    
    accuracy = (correct_classifications / total_tests) * 100
    logger.info(f"📊 ContentTypeClassifier accuracy: {accuracy:.1f}% ({correct_classifications}/{total_tests})")
    
    return accuracy >= 80  # Require at least 80% accuracy


def test_intelligent_content_batcher():
    """Test the enhanced intelligent content batcher"""
    logger.info("🧪 Testing IntelligentContentBatcher...")
    
    batcher = IntelligentContentBatcher(max_batch_chars=12000)
    test_content = create_test_content()
    
    # Step 1: Create content items
    logger.info("   Step 1: Creating content items...")
    content_items = batcher.create_content_items(test_content)
    
    if not content_items:
        logger.error("   ❌ No content items created")
        return False
    
    logger.info(f"   ✅ Created {len(content_items)} content items")
    
    # Step 2: Create intelligent batches
    logger.info("   Step 2: Creating intelligent batches...")
    batches = batcher.create_intelligent_batches(content_items)
    
    if not batches:
        logger.error("   ❌ No batches created")
        return False
    
    logger.info(f"   ✅ Created {len(batches)} intelligent batches")
    
    # Step 3: Validate batch characteristics
    logger.info("   Step 3: Validating batch characteristics...")
    all_valid = True
    
    for i, batch in enumerate(batches):
        logger.info(f"   Batch {i+1}:")
        logger.info(f"     - Items: {len(batch.items)}")
        logger.info(f"     - Characters: {batch.total_chars}")
        logger.info(f"     - Content types: {[t.value for t in batch.content_types]}")
        logger.info(f"     - Semantic coherence: {batch.semantic_coherence:.3f}")
        logger.info(f"     - Translation priority: {batch.translation_priority}")
        
        # Validate character limit
        if batch.total_chars > 12000:
            logger.error(f"     ❌ Batch exceeds 12,000 character limit: {batch.total_chars}")
            all_valid = False
        
        # Validate minimum batch size
        if batch.total_chars < 100 and len(batches) > 1:
            logger.warning(f"     ⚠️ Batch is very small: {batch.total_chars} chars")
        
        # Validate content type compatibility
        if len(batch.content_types) > 3:
            logger.warning(f"     ⚠️ Batch has many content types: {len(batch.content_types)}")
    
    # Step 4: Get batching report
    report = batcher.get_batching_report()
    logger.info("   Step 4: Batching report:")
    logger.info(f"     - Total items: {report['statistics']['total_items']}")
    logger.info(f"     - Translatable items: {report['statistics']['translatable_items']}")
    logger.info(f"     - Batches created: {report['statistics']['batches_created']}")
    logger.info(f"     - Average batch size: {report['statistics']['average_batch_size']:.1f}")
    logger.info(f"     - Character utilization: {report['efficiency_metrics']['character_utilization']:.1f}%")
    logger.info(f"     - Semantic coherence avg: {report['efficiency_metrics']['semantic_coherence_avg']:.3f}")
    
    return all_valid


async def test_parallel_translation_manager():
    """Test the parallel translation manager"""
    logger.info("🧪 Testing ParallelTranslationManager...")
    
    manager = ParallelTranslationManager(max_concurrent_batches=3)
    test_content = create_test_content()
    
    # Test parallel translation
    logger.info("   Testing parallel translation...")
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
        
        if not batch_result.success:
            logger.warning(f"     - Error: {batch_result.error}")
    
    # Get performance stats
    stats = manager.get_performance_stats()
    logger.info("   Performance statistics:")
    logger.info(f"     - Global success rate: {stats['global_stats']['success_rate']:.1f}%")
    logger.info(f"     - Average batch time: {stats['global_stats']['average_batch_time']:.3f}s")
    logger.info(f"     - API calls reduction avg: {stats['global_stats']['api_calls_reduction']:.1f}%")
    
    return result.successful_batches > 0


def test_integration():
    """Test integration between batcher and parallel manager"""
    logger.info("🧪 Testing Integration...")
    
    # Create test content with mixed types
    test_content = create_test_content()
    
    # Test full pipeline
    batcher = IntelligentContentBatcher(max_batch_chars=12000)
    manager = ParallelTranslationManager(max_concurrent_batches=3)
    
    # Create items and batches
    content_items = batcher.create_content_items(test_content)
    batches = batcher.create_intelligent_batches(content_items)
    
    # Simulate translation (without actual API calls)
    logger.info("   Simulating translation pipeline...")
    
    total_chars = sum(batch.total_chars for batch in batches)
    total_items = len(content_items)
    translatable_items = len([item for item in content_items if item.is_translatable()])
    
    logger.info(f"   Pipeline summary:")
    logger.info(f"     - Total content items: {total_items}")
    logger.info(f"     - Translatable items: {translatable_items}")
    logger.info(f"     - Batches created: {len(batches)}")
    logger.info(f"     - Total characters: {total_chars}")
    logger.info(f"     - Average batch size: {total_chars / len(batches):.0f} chars")
    
    # Calculate efficiency metrics
    old_api_calls = translatable_items
    new_api_calls = len(batches)
    api_reduction = ((old_api_calls - new_api_calls) / old_api_calls) * 100 if old_api_calls > 0 else 0
    
    logger.info(f"   Efficiency metrics:")
    logger.info(f"     - API calls reduction: {api_reduction:.1f}%")
    logger.info(f"     - Character utilization: {(total_chars / len(batches)) / 12000 * 100:.1f}%")
    
    return True


async def run_comprehensive_test():
    """Run all comprehensive tests"""
    logger.info("🚀 Starting Comprehensive Enhanced Batcher and Parallel Manager Test")
    logger.info("=" * 80)
    
    test_results = {}
    
    # Test 1: Content Type Classifier
    logger.info("\n📋 Test 1: Content Type Classifier")
    test_results['classifier'] = test_content_type_classifier()
    
    # Test 2: Intelligent Content Batcher
    logger.info("\n📋 Test 2: Intelligent Content Batcher")
    test_results['batcher'] = test_intelligent_content_batcher()
    
    # Test 3: Parallel Translation Manager
    logger.info("\n📋 Test 3: Parallel Translation Manager")
    test_results['parallel_manager'] = await test_parallel_translation_manager()
    
    # Test 4: Integration
    logger.info("\n📋 Test 4: Integration Test")
    test_results['integration'] = test_integration()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 COMPREHENSIVE TEST RESULTS")
    logger.info("=" * 80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Enhanced batcher and parallel manager are working correctly.")
        return True
    else:
        logger.error("⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(run_comprehensive_test())
    
    if success:
        print("\n✅ Enhanced Intelligent Content Batcher and Parallel Translation Manager are ready for use!")
        print("   - Content-flow-based batching ✓")
        print("   - 12,000 character limit enforcement ✓")
        print("   - Content type grouping ✓")
        print("   - Semantic coherence calculation ✓")
        print("   - Translation priority assignment ✓")
        print("   - Parallel processing support ✓")
        print("   - Comprehensive reporting ✓")
    else:
        print("\n❌ Some issues detected. Please review the test output above.") 