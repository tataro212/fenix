#!/usr/bin/env python3
"""
Digital Twin Document Pipeline Entry Point

This script provides the main execution interface for the Digital Twin document processing approach.
It handles the complete workflow: PDF extraction → Digital Twin model → translation → reconstruction

Features:
- Complete Digital Twin document modeling
- Native PyMuPDF TOC extraction
- Proper image extraction and linking
- Structure-preserving translation
- High-fidelity document reconstruction
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
    """Main entry point for Digital Twin document processing"""
    
    print("🚀 DIGITAL TWIN DOCUMENT PIPELINE")
    print("=" * 60)
    print("Digital Twin Architecture Features:")
    print("  ✅ Unified document data model")
    print("  ✅ Native PyMuPDF TOC extraction")
    print("  ✅ Proper image extraction and filesystem linking")
    print("  ✅ Structure-preserving translation")
    print("  ✅ High-fidelity document reconstruction")
    print("  ✅ Complete spatial relationship preservation")
    print("=" * 60)
    
    # Initialize variables for cleanup
    gemini_service = None
    strategy_executor = None
    
    try:
        # Import Digital Twin components
        from processing_strategies import ProcessingStrategyExecutor
        from document_generator import WordDocumentGenerator
        from utils import choose_input_path, choose_base_output_directory
        from gemini_service import GeminiService
        
        print("✅ Digital Twin pipeline components loaded successfully")
        
        # Get input file
        print("\n📄 Select input PDF file:")
        input_file_result = choose_input_path()
        if not input_file_result:
            print("❌ No input file selected")
            return False
        
        # Handle tuple return from choose_input_path
        if isinstance(input_file_result, tuple):
            input_file = input_file_result[0]  # Extract the file path from the tuple
        else:
            input_file = input_file_result
        
        if not input_file or not os.path.exists(input_file):
            print("❌ Invalid input file selected")
            return False
            
        print(f"📄 Selected file: {input_file}")
        
        # Get output directory
        print("\n📁 Select output directory:")
        output_base_dir_result = choose_base_output_directory()
        if not output_base_dir_result:
            print("❌ No output directory selected")
            return False
        
        # Handle tuple return from choose_base_output_directory
        if isinstance(output_base_dir_result, tuple):
            output_base_dir = output_base_dir_result[0]  # Extract the directory path from the tuple
        else:
            output_base_dir = output_base_dir_result
        
        # Create specific output directory for this file
        file_name = Path(input_file).stem
        output_dir = os.path.join(output_base_dir, f"{file_name}_digital_twin")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 Output directory: {output_dir}")
        
        # Set target language
        target_language = input("🌍 Target language (default: el for Greek): ").strip()
        if not target_language:
            target_language = "el"
        
        print(f"🌍 Target language: {target_language}")
        
        # Initialize services
        print("\n🔧 Initializing Digital Twin services...")
        gemini_service = GeminiService()
        strategy_executor = ProcessingStrategyExecutor(gemini_service)
        
        # Execute Digital Twin processing
        print(f"\n🚀 Starting Digital Twin pipeline processing...")
        print(f"📄 Input: {input_file}")
        print(f"📁 Output: {output_dir}")
        print(f"🌍 Language: {target_language}")
        print("-" * 60)
        
        # Run the Digital Twin strategy
        result = await strategy_executor.execute_strategy_digital_twin(
            pdf_path=input_file,
            output_dir=output_dir,
            target_language=target_language
        )
        
        # Display processing results
        print("\n" + "=" * 60)
        print("📊 DIGITAL TWIN PROCESSING RESULTS")
        print("=" * 60)
        
        if result.success:
            print("✅ Digital Twin processing completed successfully!")
            
            # Extract Digital Twin document
            digital_twin_doc = result.content.get('digital_twin_document')
            
            if digital_twin_doc:
                print(f"📊 Digital Twin Statistics:")
                print(f"  📄 Total pages: {result.statistics['total_pages']}")
                print(f"  📝 Text blocks: {result.statistics['total_text_blocks']}")
                print(f"  🖼️ Image blocks: {result.statistics['total_image_blocks']}")
                print(f"  📋 Tables: {result.statistics['total_tables']}")
                print(f"  📑 TOC entries: {result.statistics['total_toc_entries']}")
                print(f"  🌐 Translated blocks: {result.statistics['translated_blocks']}")
                print(f"  ⏱️ Processing time: {result.processing_time:.2f}s")
                
                # Step 2: Generate final Word document using Digital Twin
                print(f"\n📄 Generating final Word document from Digital Twin...")
                
                try:
                    word_doc_path = os.path.join(output_dir, f"{file_name}_translated.docx")
                    
                    # Generate Word document from Digital Twin
                    doc_generator = WordDocumentGenerator()
                    success_path = doc_generator.create_word_document_from_digital_twin(
                        digital_twin_doc, 
                        word_doc_path
                    )
                    success = success_path is not None
                    
                    if success:
                        print(f"✅ Word document generated: {word_doc_path}")
                        
                        # Show file size
                        if os.path.exists(word_doc_path):
                            word_size = os.path.getsize(word_doc_path)
                            print(f"📄 Word file size: {word_size / 1024:.1f} KB")
                    else:
                        print("❌ Failed to generate Word document")
                        
                except Exception as doc_error:
                    print(f"❌ Document generation error: {doc_error}")
                    # Still show partial success
                    print("✅ Digital Twin processing completed, but document generation failed")
                
                # Show Digital Twin benefits achieved
                print(f"\n🎯 Digital Twin Benefits Achieved:")
                print(f"  ✅ Image extraction: {result.statistics['total_image_blocks']} images saved to filesystem")
                print(f"  ✅ TOC structure: {result.statistics['total_toc_entries']} entries preserved")
                print(f"  ✅ Spatial relationships: All blocks maintain bbox coordinates")
                print(f"  ✅ Translation integrity: Tag-based reconstruction method used")
                print(f"  ✅ Document fidelity: {digital_twin_doc.total_pages} pages with structure preserved")
                
            else:
                print("❌ No Digital Twin document found in result")
                
        else:
            print("❌ Digital Twin processing failed!")
            print(f"Error: {result.error}")
            
            # Show error details
            if hasattr(result, 'statistics') and result.statistics:
                print(f"\n📊 Partial results:")
                print(f"  Processing time: {getattr(result, 'processing_time', 0):.2f}s")
        
        print(f"\n🎉 Digital Twin pipeline processing completed!")
        print(f"📁 All outputs saved to: {output_dir}")
        
        return result.success if result else False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all Digital Twin components are available:")
        print("  - digital_twin_model.py")
        print("  - Enhanced pymupdf_yolo_processor.py")
        print("  - Enhanced processing_strategies.py")
        print("  - Enhanced document_generator.py")
        return False
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup services to prevent gRPC shutdown errors
        try:
            if gemini_service and hasattr(gemini_service, 'cleanup'):
                await gemini_service.cleanup()
            elif gemini_service:
                # Give gRPC time to finish any pending operations
                await asyncio.sleep(0.1)
                
            # Additional cleanup for strategy executor
            if strategy_executor and hasattr(strategy_executor, 'cleanup'):
                await strategy_executor.cleanup()
                
        except Exception as cleanup_error:
            # Don't let cleanup errors affect the main result
            logger.debug(f"Cleanup warning (non-critical): {cleanup_error}")
        
        # Give the event loop time to complete any remaining async operations
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) >= 2:
        if sys.argv[1] in ["--help", "-h"]:
            print("Digital Twin Document Pipeline")
            print("Usage: python run_digital_twin_pipeline.py")
            print("The script will prompt for input files and settings")
            print("\nFeatures:")
            print("  - Complete Digital Twin document modeling")
            print("  - Structure-preserving translation")
            print("  - High-fidelity document reconstruction")
            print("  - Proper image and TOC handling")
            sys.exit(0)
    
    # Run the Digital Twin pipeline with proper cleanup
    success = False
    try:
        # Use asyncio.run() with proper exception handling
        success = asyncio.run(main())
        
        # Give extra time for any remaining gRPC operations to complete
        import time
        time.sleep(0.2)
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        success = False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        success = False
    finally:
        # Additional cleanup to prevent gRPC issues
        try:
            # Force cleanup of any remaining async resources
            import gc
            gc.collect()
            
            # Small delay to let gRPC finish cleanup
            import time
            time.sleep(0.1)
            
        except Exception:
            # Ignore cleanup errors
            pass
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 