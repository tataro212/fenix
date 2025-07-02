#!/usr/bin/env python3
"""
Simple Test Script for Optimized Document Pipeline
"""

import os
import sys
import asyncio
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_pipeline():
    """Test the optimized document pipeline"""
    
    print("Testing Optimized Document Pipeline")
    print("=" * 50)
    
    # Test 1: Imports
    print("\nTest 1: Component Imports")
    try:
        from optimized_document_pipeline import OptimizedDocumentPipeline, process_pdf_optimized
        from pymupdf_yolo_processor import PyMuPDFYOLOProcessor
        from processing_strategies import ProcessingStrategyExecutor
        
        print("All components imported successfully")
        
        # Test initialization
        pipeline = OptimizedDocumentPipeline(max_workers=6)
        print(f"Pipeline initialized with max_workers={pipeline.max_workers}")
        
    except ImportError as e:
        print(f"Import failed: {e}")
        return False
    except Exception as e:
        print(f"Initialization failed: {e}")
        return False
    
    # Test 2: Create simple test file
    print("\nTest 2: Create Test File")
    test_content = """
Sample Document Title

This is a sample paragraph with some text content. It contains multiple sentences to test the text extraction and processing capabilities of our enhanced pipeline.

This is a second paragraph with different content. It helps test the layout analysis and content mapping features.

First list item
Second list item
Third list item
    """
    
    test_file = "test_document.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    print(f"Test file created: {test_file}")
    
    # Test 3: Test processor
    print("\nTest 3: PyMuPDF-YOLO Processor")
    try:
        processor = PyMuPDFYOLOProcessor()
        page_result = await processor.process_page(test_file, 0)
        
        if 'error' in page_result:
            print(f"Processor failed: {page_result['error']}")
            return False
        
        print(f"Processor completed successfully")
        print(f"  Content type: {page_result.get('content_type', 'Unknown')}")
        print(f"  Text blocks: {page_result.get('statistics', {}).get('text_blocks', 0)}")
        
    except Exception as e:
        print(f"Processor test failed: {e}")
        return False
    
    # Test 4: Test strategy execution
    print("\nTest 4: Processing Strategies")
    try:
        strategy_executor = ProcessingStrategyExecutor()
        strategy_result = await strategy_executor.execute_strategy(page_result, 'es')
        
        if not strategy_result.success:
            print(f"Strategy failed: {strategy_result.error}")
            return False
        
        print(f"Strategy '{strategy_result.strategy}' completed successfully")
        print(f"  Processing time: {strategy_result.processing_time:.2f}s")
        
    except Exception as e:
        print(f"Strategy test failed: {e}")
        return False
    
    # Test 5: Test full pipeline
    print("\nTest 5: Full Pipeline")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using output directory: {temp_dir}")
            
            pipeline_result = await process_pdf_optimized(
                test_file, 
                temp_dir, 
                target_language='es',
                max_workers=6
            )
            
            if not pipeline_result.success:
                print(f"Pipeline failed: {pipeline_result.error}")
                return False
            
            print(f"Pipeline completed successfully")
            print(f"  Output files: {len(pipeline_result.output_files)}")
            print(f"  Processing time: {pipeline_result.statistics.processing_time:.2f}s")
            print(f"  Translation success rate: {pipeline_result.statistics.translation_success_rate:.1%}")
            
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        return False
    
    # Cleanup
    print("\nCleanup")
    try:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Removed test file: {test_file}")
    except Exception as e:
        print(f"Cleanup warning: {e}")
    
    print("\nAll tests completed successfully!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_pipeline())
    
    if success:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed.")
        sys.exit(1) 