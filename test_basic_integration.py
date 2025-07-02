#!/usr/bin/env python3
"""
Basic Integration Test for Strategic Implementation

Simple end-to-end test to validate the strategic PyMuPDF+YOLO implementation
works with the main workflow.
"""

import os
import sys
import asyncio
import tempfile
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_basic_integration():
    """Basic integration test for the strategic implementation"""
    logger.info("🧪 Starting Basic Integration Test")
    
    try:
        # Import our optimized pipeline
        from optimized_document_pipeline import OptimizedDocumentPipeline
        
        # Check if sample PDF exists
        sample_pdf = "sample_page.pdf"
        if not os.path.exists(sample_pdf):
            logger.warning(f"Sample PDF not found: {sample_pdf}")
            logger.info("Creating a simple test PDF...")
            
            # Try to create a simple test PDF using reportlab if available
            try:
                from reportlab.pdfgen import canvas
                from reportlab.lib.pagesizes import letter
                
                c = canvas.Canvas(sample_pdf, pagesize=letter)
                c.drawString(100, 750, "This is a test document for strategic implementation.")
                c.drawString(100, 720, "It contains simple text to test the pure text fast path.")
                c.drawString(100, 690, "The PyMuPDF processor should detect this as pure text.")
                c.drawString(100, 660, "No YOLO overhead should be involved in processing this.")
                c.save()
                
                logger.info(f"✅ Created test PDF: {sample_pdf}")
                
            except ImportError:
                logger.error("❌ Cannot create test PDF - reportlab not available")
                logger.info("Please place a sample PDF named 'sample_page.pdf' in the current directory")
                return False
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"📁 Using temporary output directory: {temp_dir}")
            
            # Initialize pipeline
            pipeline = OptimizedDocumentPipeline(max_workers=2)
            
            # Process the PDF
            logger.info(f"🔄 Processing PDF with strategic implementation...")
            result = await pipeline.process_pdf_with_optimized_pipeline(
                pdf_path=sample_pdf,
                output_dir=temp_dir,
                target_language='Greek'
            )
            
            # Validate results
            if result.success:
                logger.info("✅ Processing completed successfully!")
                
                # Check output files
                output_files = result.output_files
                logger.info(f"📄 Generated {len(output_files)} output files:")
                
                for file_type, file_path in output_files.items():
                    exists = os.path.exists(file_path)
                    logger.info(f"   {file_type}: {os.path.basename(file_path)} {'✅' if exists else '❌'}")
                
                # Check statistics
                stats = result.statistics
                logger.info(f"📊 Processing Statistics:")
                logger.info(f"   Total pages: {stats.total_pages}")
                logger.info(f"   Processing time: {stats.processing_time:.3f}s")
                logger.info(f"   Average page time: {stats.average_page_time:.3f}s")
                logger.info(f"   Strategy distribution: {stats.strategy_distribution}")
                logger.info(f"   Content type distribution: {stats.content_type_distribution}")
                
                # Check if strategic implementation is working
                strategies_used = stats.strategy_distribution
                if 'pure_text_fast' in strategies_used:
                    logger.info("⚡ Pure text fast path was used - strategic implementation working!")
                elif 'coordinate_based_extraction' in strategies_used:
                    logger.info("🎯 Coordinate-based extraction was used - strategic implementation working!")
                else:
                    logger.warning("⚠️ Strategic implementation may not be working - check strategy distribution")
                
                return True
                
            else:
                logger.error(f"❌ Processing failed: {result.error}")
                return False
    
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_processor_components():
    """Test individual processor components"""
    logger.info("🔧 Testing Processor Components...")
    
    try:
        from pymupdf_yolo_processor import PyMuPDFYOLOProcessor, ContentType
        from processing_strategies import ProcessingStrategyExecutor
        
        # Test processor initialization
        processor = PyMuPDFYOLOProcessor()
        executor = ProcessingStrategyExecutor()
        
        logger.info("✅ Processor components initialized successfully")
        
        # Test content type classification
        from pymupdf_yolo_processor import ContentTypeClassifier
        classifier = ContentTypeClassifier()
        
        # Test pure text strategy
        pure_text_strategy = classifier.get_processing_strategy(ContentType.PURE_TEXT, {})
        if pure_text_strategy.strategy == 'pure_text_fast':
            logger.info("✅ Pure text routing working correctly")
        else:
            logger.warning(f"⚠️ Pure text routing issue - got {pure_text_strategy.strategy}")
        
        # Test mixed content strategy
        mixed_content_strategy = classifier.get_processing_strategy(ContentType.MIXED_CONTENT, {})
        if mixed_content_strategy.strategy == 'coordinate_based_extraction':
            logger.info("✅ Mixed content routing working correctly")
        else:
            logger.warning(f"⚠️ Mixed content routing issue - got {mixed_content_strategy.strategy}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Component test failed: {e}")
        return False

async def main():
    """Run basic integration tests"""
    print("🧪 Basic Integration Test for Strategic Implementation")
    print("=" * 60)
    
    # Test 1: Component functionality
    logger.info("🔧 Test 1: Component Functionality")
    component_success = await test_processor_components()
    
    print()
    
    # Test 2: End-to-end integration
    logger.info("🔄 Test 2: End-to-End Integration")
    integration_success = await test_basic_integration()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    
    if component_success and integration_success:
        print("🎉 All tests passed! Strategic implementation is working.")
        return True
    else:
        print("❌ Some tests failed. Check the logs above for details.")
        if not component_success:
            print("   - Component functionality issues detected")
        if not integration_success:
            print("   - End-to-end integration issues detected")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 