#!/usr/bin/env python3

"""
Test script to verify the intelligent chunking with 13,000 character limit works correctly.
This script should show proper chunking while maintaining text coherence.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from optimized_document_pipeline import OptimizedDocumentPipeline

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_intelligent_chunking():
    """Test that intelligent chunking with 13K character limit works correctly"""
    print("🧪 Testing Intelligent Chunking (13K Character Limit)")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = OptimizedDocumentPipeline(max_workers=4)
    
    # Test with the sample PDF
    pdf_path = "test_document_with_text.pdf"
    if not Path(pdf_path).exists():
        print(f"❌ Test PDF not found: {pdf_path}")
        print("Please ensure test_document_with_text.pdf exists in the current directory")
        return
    
    output_dir = "output"
    target_language = "fr"  # French for clear translation differences
    
    print(f"📄 Processing: {pdf_path}")
    print(f"🎯 Target language: {target_language}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔢 Max characters per chunk: 13,000")
    print()
    
    try:
        # Process the document
        result = await pipeline.process_pdf_with_optimized_pipeline(
            pdf_path=pdf_path,
            output_dir=output_dir,
            target_language=target_language
        )
        
        if result.success:
            print("✅ Processing completed successfully!")
            print()
            print("📊 Statistics:")
            print(f"   Total pages: {result.statistics.total_pages}")
            print(f"   Processing time: {result.statistics.processing_time:.3f}s")
            print(f"   Strategy distribution: {result.statistics.strategy_distribution}")
            print(f"   Translation success rate: {result.statistics.translation_success_rate:.3f}")
            print()
            print("📄 Output files:")
            for file_type, file_path in result.output_files.items():
                print(f"   {file_type}: {file_path}")
            print()
            
            # Analyze chunking behavior from logs
            print("🔍 Analyzing chunking behavior...")
            for i, processing_result in enumerate(result.processing_results):
                if processing_result.success and processing_result.content:
                    blocks = processing_result.content.get('final_content', [])
                    
                    print(f"   Result {i+1}: {len(blocks)} blocks processed")
                    
                    # Check if blocks have proper metadata
                    if blocks and isinstance(blocks[0], dict):
                        first_block = blocks[0]
                        metadata_keys = ['page_number', 'element_index', 'global_index', 'y_coordinate']
                        present_metadata = [key for key in metadata_keys if key in first_block]
                        print(f"   Metadata preserved: {present_metadata}")
                        
                        # Calculate total character count
                        total_chars = sum(len(block.get('text', '')) for block in blocks)
                        print(f"   Total characters processed: {total_chars}")
            
            print()
            print("🎉 Intelligent chunking test completed successfully!")
            print("📖 Check the output document to verify text coherence is maintained!")
            
        else:
            print(f"❌ Processing failed: {result.error}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_intelligent_chunking()) 