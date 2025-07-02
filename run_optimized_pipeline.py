#!/usr/bin/env python3
"""
Optimized Document Pipeline Entry Point

This script uses the new optimized document pipeline with:
- PyMuPDF content extraction
- YOLO layout analysis (confidence threshold 0.15)
- Content-to-layout mapping
- Intelligent processing strategies
- Parallel processing with concurrency limit of 6
- Progress tracking
"""

import os
import sys
import asyncio
import logging
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    """Main entry point for the optimized document pipeline"""
    
    print("🚀 OPTIMIZED DOCUMENT PIPELINE")
    print("=" * 50)
    print("Features:")
    print("  ✅ PyMuPDF content extraction")
    print("  ✅ YOLO layout analysis (0.15 confidence)")
    print("  ✅ Content-to-layout mapping")
    print("  ✅ Intelligent processing strategies")
    print("  ✅ Parallel processing (6 workers)")
    print("  ✅ Progress tracking")
    print("=" * 50)
    
    try:
        # Import our optimized pipeline
        from optimized_document_pipeline import process_pdf_optimized
        from utils import choose_input_path, choose_base_output_directory
        
        print("✅ Optimized pipeline components loaded successfully")
        
        # Get input file
        print("\n📄 Select input PDF file:")
        input_file_result = choose_input_path()
        if not input_file_result:
            print("❌ No input file selected")
            return
        
        # Handle tuple return from choose_input_path
        if isinstance(input_file_result, tuple):
            input_file = input_file_result[0]  # Extract the file path from the tuple
        else:
            input_file = input_file_result
        
        print(f"📄 Selected file: {input_file}")
        
        # Get output directory
        print("\n📁 Select output directory:")
        output_base_dir_result = choose_base_output_directory()
        if not output_base_dir_result:
            print("❌ No output directory selected")
            return
        
        # Handle tuple return from choose_base_output_directory
        if isinstance(output_base_dir_result, tuple):
            output_base_dir = output_base_dir_result[0]  # Extract the directory path from the tuple
        else:
            output_base_dir = output_base_dir_result
        
        # Create specific output directory for this file
        file_name = Path(input_file).stem
        output_dir = os.path.join(output_base_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 Output directory: {output_dir}")
        
        # Set target language to Greek (el)
        target_language = "el"
        print(f"🌍 Target language: {target_language}")
        
        # Set concurrency to 4 workers
        max_workers = 4
        print(f"⚡ Max workers: {max_workers}")
        
        # Process the document
        print(f"\n🚀 Starting optimized pipeline processing...")
        print(f"📄 Input: {input_file}")
        print(f"📁 Output: {output_dir}")
        print(f"🌍 Language: {target_language}")
        print(f"⚡ Workers: {max_workers}")
        print("-" * 50)
        
        # Run the optimized pipeline
        result = await process_pdf_optimized(
            pdf_path=input_file,
            output_dir=output_dir,
            target_language=target_language,
            max_workers=max_workers
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("📊 PROCESSING RESULTS")
        print("=" * 50)
        
        if result.success:
            print("✅ Processing completed successfully!")
            print(f"📄 Total pages: {result.statistics.total_pages}")
            print(f"⏱️ Processing time: {result.statistics.processing_time:.2f}s")
            print(f"📊 Average page time: {result.statistics.average_page_time:.2f}s")
            print(f"🌍 Translation success rate: {result.statistics.translation_success_rate:.1%}")
            print(f"💾 Memory usage: {result.statistics.memory_usage_mb:.1f} MB")
            
            # Show strategy distribution
            print(f"\n📈 Strategy distribution:")
            for strategy, count in result.statistics.strategy_distribution.items():
                print(f"  {strategy}: {count}")
            
            # Show content type distribution
            if result.statistics.content_type_distribution:
                print(f"\n📋 Content type distribution:")
                for content_type, count in result.statistics.content_type_distribution.items():
                    print(f"  {content_type}: {count}")
            
            # Show output files
            print(f"\n📁 Generated files:")
            for file_type, file_path in result.output_files.items():
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"  {file_type}: {file_path} ({file_size} bytes)")
                else:
                    print(f"  {file_type}: {file_path} (not found)")
            
            # Show performance tips
            print(f"\n💡 Performance tips:")
            tips = [
                "Consider reducing max_workers if system is overloaded",
                "Use SSD storage for better I/O performance",
                "Ensure sufficient RAM for large documents",
                "Monitor GPU memory usage for YOLO processing"
            ]
            for i, tip in enumerate(tips, 1):
                print(f"  {i}. {tip}")
            
        else:
            print("❌ Processing failed!")
            print(f"Error: {result.error}")
            
            # Show partial results if available
            if hasattr(result, 'statistics') and result.statistics:
                print(f"\n📊 Partial results:")
                print(f"  Pages processed: {getattr(result.statistics, 'total_pages', 0)}")
                print(f"  Processing time: {getattr(result.statistics, 'processing_time', 0):.2f}s")
        
        print("\n🎉 Optimized pipeline processing completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all optimized pipeline components are available")
        return False
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Run the optimized pipeline
    success = asyncio.run(main())
    
    if success:
        print("\n✅ Optimized pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Optimized pipeline failed.")
        sys.exit(1) 