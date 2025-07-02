#!/usr/bin/env python3
"""
Test script to verify the optimized pipeline fixes
"""

import asyncio
import os
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_optimized_pipeline():
    """Test the optimized pipeline with fixes"""
    
    print("🧪 Testing Optimized Pipeline Fixes")
    print("=" * 50)
    
    try:
        # Import the optimized pipeline
        from optimized_document_pipeline import process_pdf_optimized
        
        # Create a simple test PDF or use an existing one
        test_pdf = "sample_page.jpg"  # We'll use an image for testing
        
        if not os.path.exists(test_pdf):
            print(f"❌ Test file not found: {test_pdf}")
            print("Please ensure you have a test PDF file available")
            return False
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Using temporary output directory: {temp_dir}")
            
            # Test the pipeline with Greek translation and 4 workers
            print("🚀 Running optimized pipeline test...")
            print(f"📄 Input: {test_pdf}")
            print(f"📁 Output: {temp_dir}")
            print(f"🌍 Language: el (Greek)")
            print(f"⚡ Workers: 4")
            print("-" * 50)
            
            result = await process_pdf_optimized(
                pdf_path=test_pdf,
                output_dir=temp_dir,
                target_language="el",
                max_workers=4
            )
            
            # Check results
            if result.success:
                print("✅ Pipeline completed successfully!")
                print(f"📊 Statistics:")
                print(f"   • Total pages: {result.statistics.total_pages}")
                print(f"   • Processing time: {result.statistics.processing_time:.2f}s")
                print(f"   • Translation success rate: {result.statistics.translation_success_rate:.1%}")
                print(f"   • Memory usage: {result.statistics.memory_usage_mb:.1f}MB")
                
                print(f"📄 Output files:")
                for file_type, file_path in result.output_files.items():
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        print(f"   • {file_type}: {file_path} ({file_size} bytes)")
                    else:
                        print(f"   • {file_type}: {file_path} (NOT FOUND)")
                
                return True
            else:
                print(f"❌ Pipeline failed: {result.error}")
                return False
                
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_translation_service():
    """Test the translation service fixes"""
    
    print("\n🧪 Testing Translation Service Fixes")
    print("=" * 50)
    
    try:
        # Test the enhanced translation service
        from translation_service_enhanced import enhanced_translation_service
        
        test_text = "Hello world, this is a test."
        print(f"📝 Testing translation: '{test_text}'")
        
        result = await enhanced_translation_service.translate_text_enhanced(
            test_text, target_language="el"
        )
        
        print(f"✅ Translation successful!")
        print(f"📝 Original: {test_text}")
        print(f"🌍 Translated: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Translation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    
    print("🚀 Starting Optimized Pipeline Fix Tests")
    print("=" * 60)
    
    # Test translation service first
    translation_success = await test_translation_service()
    
    # Test full pipeline
    pipeline_success = await test_optimized_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Translation Service: {'✅ PASS' if translation_success else '❌ FAIL'}")
    print(f"Optimized Pipeline: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
    
    if translation_success and pipeline_success:
        print("\n🎉 All tests passed! The fixes are working correctly.")
        return True
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 