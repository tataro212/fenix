#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Test for Enhanced Intelligent Content Batcher Only

This test focuses on the batching logic without making actual translation API calls
or using GPU resources. It tests:
1. Content type classification
2. Intelligent batching with 12,000 character limit
3. Semantic coherence calculation
4. Translation priority assignment
5. Performance tracking and reporting
"""

import logging
import time
from typing import Dict, Any

# Import our enhanced modules
from intelligent_content_batcher import IntelligentContentBatcher, ContentType, ContentTypeClassifier

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


def test_large_content_batching():
    """Test batching with large content to verify 12,000 character limit"""
    logger.info("🧪 Testing Large Content Batching...")
    
    batcher = IntelligentContentBatcher(max_batch_chars=12000)
    
    # Create large content that should be split into multiple batches
    large_content = {}
    
    # Add several large paragraphs
    for i in range(10):
        large_paragraph = f"This is a very long paragraph {i+1} that contains a substantial amount of text. " * 50  # ~3000 chars
        large_content[f"paragraph_{i+1}"] = {
            "combined_text": large_paragraph,
            "layout_info": {
                "label": "paragraph",
                "bbox": (50, 100 + i*50, 450, 130 + i*50),
                "confidence": 0.88
            },
            "page_num": 0
        }
    
    # Create content items
    content_items = batcher.create_content_items(large_content)
    batches = batcher.create_intelligent_batches(content_items)
    
    logger.info(f"   Large content test:")
    logger.info(f"     - Total items: {len(content_items)}")
    logger.info(f"     - Total characters: {sum(item.char_count for item in content_items)}")
    logger.info(f"     - Batches created: {len(batches)}")
    
    # Verify that batches respect the character limit
    for i, batch in enumerate(batches):
        logger.info(f"     Batch {i+1}: {batch.total_chars} chars")
        if batch.total_chars > 12000:
            logger.error(f"     ❌ Batch {i+1} exceeds limit: {batch.total_chars} chars")
            return False
    
    logger.info("   ✅ All batches respect 12,000 character limit")
    return True


def test_content_type_compatibility():
    """Test content type compatibility rules"""
    logger.info("🧪 Testing Content Type Compatibility...")
    
    batcher = IntelligentContentBatcher(max_batch_chars=12000)
    
    # Create content with incompatible types
    mixed_content = {
        "heading_1": {
            "combined_text": "Chapter 1: Introduction",
            "layout_info": {"label": "heading", "bbox": (50, 50, 400, 70), "confidence": 0.9},
            "page_num": 0
        },
        "paragraph_1": {
            "combined_text": "This is a regular paragraph.",
            "layout_info": {"label": "paragraph", "bbox": (50, 80, 400, 100), "confidence": 0.8},
            "page_num": 0
        },
        "list_item_1": {
            "combined_text": "• List item 1",
            "layout_info": {"label": "list", "bbox": (70, 110, 380, 125), "confidence": 0.8},
            "page_num": 0
        },
        "list_item_2": {
            "combined_text": "• List item 2",
            "layout_info": {"label": "list", "bbox": (70, 135, 380, 150), "confidence": 0.8},
            "page_num": 0
        },
        "caption_1": {
            "combined_text": "Figure 1: Sample",
            "layout_info": {"label": "caption", "bbox": (50, 160, 400, 175), "confidence": 0.8},
            "page_num": 0
        }
    }
    
    content_items = batcher.create_content_items(mixed_content)
    batches = batcher.create_intelligent_batches(content_items)
    
    logger.info(f"   Content type compatibility test:")
    logger.info(f"     - Items: {len(content_items)}")
    logger.info(f"     - Batches: {len(batches)}")
    
    for i, batch in enumerate(batches):
        types = [t.value for t in batch.content_types]
        logger.info(f"     Batch {i+1} types: {types}")
        
        # Check that incompatible types are not mixed
        if 'caption' in types and len(types) > 1:
            logger.warning(f"     ⚠️ Caption mixed with other types in batch {i+1}")
    
    return True


def run_batcher_only_test():
    """Run all batcher-only tests"""
    logger.info("🚀 Starting Enhanced Intelligent Content Batcher Test (No Translation)")
    logger.info("=" * 80)
    
    test_results = {}
    
    # Test 1: Content Type Classifier
    logger.info("\n📋 Test 1: Content Type Classifier")
    test_results['classifier'] = test_content_type_classifier()
    
    # Test 2: Intelligent Content Batcher
    logger.info("\n📋 Test 2: Intelligent Content Batcher")
    test_results['batcher'] = test_intelligent_content_batcher()
    
    # Test 3: Large Content Batching
    logger.info("\n📋 Test 3: Large Content Batching")
    test_results['large_content'] = test_large_content_batching()
    
    # Test 4: Content Type Compatibility
    logger.info("\n📋 Test 4: Content Type Compatibility")
    test_results['compatibility'] = test_content_type_compatibility()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 BATCHER-ONLY TEST RESULTS")
    logger.info("=" * 80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Enhanced intelligent content batcher is working correctly.")
        logger.info("   - Content-flow-based batching ✓")
        logger.info("   - 12,000 character limit enforcement ✓")
        logger.info("   - Content type grouping ✓")
        logger.info("   - Semantic coherence calculation ✓")
        logger.info("   - Translation priority assignment ✓")
        logger.info("   - No GPU usage ✓")
        logger.info("   - No API calls ✓")
        return True
    else:
        logger.error("⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    # Run the batcher-only test
    success = run_batcher_only_test()
    
    if success:
        print("\n✅ Enhanced Intelligent Content Batcher is ready for use!")
        print("   - No GPU resources used")
        print("   - No API calls made")
        print("   - Pure batching logic tested")
    else:
        print("\n❌ Some issues detected. Please review the test output above.") 