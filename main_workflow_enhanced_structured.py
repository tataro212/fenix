"""
Enhanced Main Workflow Module with Structured Pipeline Integration

This module integrates the new structured pipeline components:
1. StructuredPDFProcessor (pdf_processor.py) - Structure-aware PDF processing
2. StructuredTextTranslator (text_translator.py) - Sequence-preserving translation
3. StructuredDocumentPipeline (main.py) - Complete orchestration

While maintaining compatibility with existing enhanced components.
"""

import os
import asyncio
import time
import logging
logger = logging.getLogger(__name__)
import sys
from pathlib import Path

# Import NEW structured pipeline components
try:
    from pdf_processor import StructuredPDFProcessor, TextBlock
    from text_translator import StructuredTextTranslator
    from main import StructuredDocumentPipeline
    STRUCTURED_PIPELINE_AVAILABLE = True
    logger.info("✅ Structured pipeline components available")
except ImportError as e:
    STRUCTURED_PIPELINE_AVAILABLE = False
    logger.warning(f"⚠️ Structured pipeline components not available: {e}")

# Import enhanced components (for fallback)
from document_generator import WordDocumentGenerator as EnhancedWordDocumentGenerator
from translation_service_enhanced import enhanced_translation_service
from pdf_parser_enhanced import enhanced_pdf_parser

# Import PyMuPDF-YOLO integration components
try:
    from optimized_document_pipeline import OptimizedDocumentPipeline, process_pdf_optimized
    PYMUPDF_YOLO_AVAILABLE = True
    logger.info("✅ PyMuPDF-YOLO integration available")
except ImportError as e:
    PYMUPDF_YOLO_AVAILABLE = False
    logger.warning(f"⚠️ PyMuPDF-YOLO integration not available: {e}")

# Import original components for compatibility
from config_manager import config_manager

class EnhancedPDFTranslator:
    """Enhanced PDF translator with structured pipeline integration"""
    
    def __init__(self):
        self.document_generator = EnhancedWordDocumentGenerator()
        self.translation_service = enhanced_translation_service
        self.pdf_parser = enhanced_pdf_parser
        self.settings = config_manager.word_output_settings
        
        # Initialize structured pipeline if available
        self.structured_pipeline = None
        if STRUCTURED_PIPELINE_AVAILABLE:
            self.structured_pipeline = StructuredDocumentPipeline()
            logger.info("🚀 Structured document pipeline initialized")
        
        # Initialize PyMuPDF-YOLO pipeline if available
        self.optimized_pipeline = None
        if PYMUPDF_YOLO_AVAILABLE:
            self.optimized_pipeline = OptimizedDocumentPipeline()
            logger.info("🚀 PyMuPDF-YOLO optimized pipeline initialized")
        
        logger.info("Enhanced PDF Translator initialized with structured pipeline integration")
    
    async def translate_document_enhanced(self, input_path: str, output_dir: str, 
                                        use_structured_pipeline: bool = True,
                                        use_optimized_pipeline: bool = True) -> bool:
        """
        Enhanced document translation workflow with structured pipeline integration
        Priority: Structured Pipeline → PyMuPDF-YOLO → Standard Enhanced
        """
        try:
            start_time = time.time()
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            logger.info(f"Starting enhanced translation: {base_name}")
            
            # Priority 1: Use structured pipeline (new refactored components)
            if use_structured_pipeline and self.structured_pipeline:
                logger.info("🏗️ Using structured document pipeline (new refactored components)")
                return await self._translate_with_structured_pipeline(input_path, output_dir, base_name)
            
            # Priority 2: Use PyMuPDF-YOLO optimized pipeline
            elif use_optimized_pipeline and self.optimized_pipeline:
                logger.info("🚀 Using PyMuPDF-YOLO optimized pipeline")
                return await self._translate_with_optimized_pipeline(input_path, output_dir, base_name)
            
            # Priority 3: Fall back to standard enhanced pipeline
            else:
                logger.info("📄 Using standard enhanced pipeline")
                return await self._translate_with_standard_pipeline(input_path, output_dir, base_name)
            
        except Exception as e:
            logger.error(f"Enhanced translation failed: {e}")
            return False
    
    async def _translate_with_structured_pipeline(self, input_path: str, output_dir: str, base_name: str) -> bool:
        """Translate using the new structured pipeline (refactored components)"""
        try:
            start_time = time.time()
            logger.info("🏗️ Structured pipeline: Phase 1 - Document structure extraction")
            
            # Use the structured pipeline for complete processing
            success = await self.structured_pipeline.process_document(
                input_path=input_path,
                output_dir=output_dir,
                target_language=config_manager.translation_enhancement_settings['target_language']
            )
            
            if success:
                total_time = time.time() - start_time
                logger.info(f"✅ Structured pipeline completed in {total_time:.2f} seconds")
                logger.info("   • Document structure preserved")
                logger.info("   • Sequence integrity maintained")
                logger.info("   • Pure text payloads used")
                return True
            else:
                logger.error("❌ Structured pipeline failed")
                # Fall back to optimized pipeline
                logger.info("🔄 Falling back to PyMuPDF-YOLO optimized pipeline")
                return await self._translate_with_optimized_pipeline(input_path, output_dir, base_name)
                
        except Exception as e:
            logger.error(f"❌ Structured pipeline error: {e}")
            # Fall back to optimized pipeline
            logger.info("🔄 Falling back to PyMuPDF-YOLO optimized pipeline")
            return await self._translate_with_optimized_pipeline(input_path, output_dir, base_name)
    
    async def _translate_with_optimized_pipeline(self, input_path: str, output_dir: str, base_name: str) -> bool:
        """Translate using PyMuPDF-YOLO optimized pipeline"""
        try:
            start_time = time.time()
            
            # Use the optimized pipeline
            target_language = config_manager.translation_enhancement_settings['target_language']
            result = await self.optimized_pipeline.process_pdf_with_optimized_pipeline(
                input_path, output_dir, target_language
            )
            
            if result.success:
                total_time = time.time() - start_time
                logger.info(f"✅ Optimized pipeline completed in {total_time:.2f} seconds")
                logger.info(f"   Pages processed: {result.statistics.total_pages}")
                logger.info(f"   Strategy distribution: {result.statistics.strategy_distribution}")
                logger.info(f"   Graph overhead: {result.statistics.graph_overhead_total:.3f}s")
                
                # Log performance improvements
                if result.statistics.strategy_distribution.get('direct_text', 0) > 0:
                    logger.info("🎯 Direct text processing used - 20-100x faster than graph-based approaches")
                
                return True
            else:
                logger.error(f"❌ Optimized pipeline failed: {result.error}")
                # Fall back to standard pipeline
                logger.info("🔄 Falling back to standard enhanced pipeline")
                return await self._translate_with_standard_pipeline(input_path, output_dir, base_name)
                
        except Exception as e:
            logger.error(f"❌ Optimized pipeline error: {e}")
            # Fall back to standard pipeline
            logger.info("🔄 Falling back to standard enhanced pipeline")
            return await self._translate_with_standard_pipeline(input_path, output_dir, base_name)
    
    async def _translate_with_standard_pipeline(self, input_path: str, output_dir: str, base_name: str) -> bool:
        """Translate using standard enhanced pipeline (legacy fallback)"""
        try:
            start_time = time.time()
            
            # PHASE 1: Enhanced PDF Parsing with TOC awareness
            logger.info("Phase 1: Enhanced PDF parsing with TOC awareness")
            extracted_pages = self.pdf_parser.extract_pdf_with_enhanced_structure(input_path)
            
            if not extracted_pages:
                logger.error("No content extracted from PDF")
                return False
            
            # PHASE 2: Content Processing and Translation
            logger.info("Phase 2: Enhanced content processing and translation")
            processed_content = await self._process_content_enhanced(extracted_pages)
            
            # PHASE 3: Document Generation with Unicode Support and Two-Pass TOC
            logger.info("Phase 3: Enhanced document generation with Unicode support")
            docx_path = os.path.join(output_dir, f"{base_name}_translated_enhanced.docx")
            
            # Use enhanced document generator with Unicode font support
            success = self.document_generator.create_word_document_with_structure(
                processed_content, docx_path, 
                image_folder_path=os.path.join(output_dir, "images"),
                cover_page_data=None
            )
            
            if not success:
                logger.error("Document generation failed")
                return False
            
            # PHASE 4: Enhanced PDF Conversion (if enabled)
            if self.settings.get('convert_to_pdf', False):
                logger.info("Phase 4: Enhanced PDF conversion with Unicode support")
                pdf_path = os.path.join(output_dir, f"{base_name}_translated_enhanced.pdf")
                
                # Import the enhanced PDF conversion function
                from document_generator import convert_word_to_pdf
                pdf_success = convert_word_to_pdf(docx_path, pdf_path)
                
                if pdf_success:
                    logger.info(f"Enhanced PDF created: {pdf_path}")
                else:
                    logger.warning("Enhanced PDF conversion failed, but DOCX was created successfully")
            
            total_time = time.time() - start_time
            logger.info(f"Enhanced translation completed in {total_time:.2f} seconds")
            logger.info(f"Output: {docx_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Standard enhanced translation failed: {e}")
            return False
    
    async def _process_content_enhanced(self, extracted_pages):
        """
        Enhanced content processing with proper noun handling and missing letter fixes
        Now with parallel translation of content blocks.
        """
        try:
            # Convert extracted pages to translatable content
            content_blocks = []
            
            for page_data in extracted_pages:
                if page_data.get('type') == 'toc_page':
                    # Skip TOC pages for translation, they will be regenerated
                    logger.info(f"Skipping TOC page {page_data.get('page_num', 'unknown')}")
                    continue
                
                if page_data.get('type') == 'content_page':
                    # Process content blocks from enhanced extraction
                    for block in page_data.get('content_blocks', []):
                        if block.get('type') == 'text' and block.get('content'):
                            content_blocks.append({
                                'text': block['content'],
                                'page_num': block.get('page_num', 0),
                                'bbox': block.get('bbox', [0, 0, 0, 0])
                            })
                
                elif page_data.get('type') == 'simple_text':
                    # Handle simple text extraction fallback
                    if page_data.get('content'):
                        content_blocks.append({
                            'text': page_data['content'],
                            'page_num': page_data.get('page_num', 0),
                            'bbox': [0, 0, 0, 0]
                        })
            
            # Translate content blocks in parallel using asyncio.gather
            async def translate_block(block):
                try:
                    translated_text = await self.translation_service.translate_text_enhanced(
                        block['text'],
                        target_language=config_manager.translation_enhancement_settings['target_language'],
                        prev_context="",  # Could be enhanced with actual context
                        next_context="",  # Could be enhanced with actual context
                        item_type="text"
                    )
                    return {
                        'original_text': block['text'],
                        'translated_text': translated_text,
                        'page_num': block['page_num'],
                        'bbox': block['bbox']
                    }
                except Exception as e:
                    logger.warning(f"Translation failed for block: {e}")
                    return {
                        'original_text': block['text'],
                        'translated_text': block['text'],  # Keep original on failure
                        'page_num': block['page_num'],
                        'bbox': block['bbox']
                    }
            
            # Execute parallel translation
            translation_tasks = [translate_block(block) for block in content_blocks]
            translated_blocks = await asyncio.gather(*translation_tasks)
            
            return translated_blocks
            
        except Exception as e:
            logger.error(f"Enhanced content processing failed: {e}")
            return []

async def translate_pdf_with_all_fixes(input_path: str, output_dir: str, 
                                     use_structured_pipeline: bool = True) -> bool:
    """
    Main entry point for PDF translation with all fixes and structured pipeline integration
    """
    translator = EnhancedPDFTranslator()
    return await translator.translate_document_enhanced(
        input_path, output_dir, 
        use_structured_pipeline=use_structured_pipeline
    )

async def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 3:
        print("Usage: python main_workflow_enhanced_structured.py <input_pdf> <output_dir>")
        print("Options:")
        print("  --no-structured: Skip structured pipeline, use PyMuPDF-YOLO or standard")
        print("  --no-optimized: Skip PyMuPDF-YOLO, use standard pipeline")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Parse command line options
    use_structured_pipeline = "--no-structured" not in sys.argv
    use_optimized_pipeline = "--no-optimized" not in sys.argv
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Start translation
    success = await translate_pdf_with_all_fixes(
        input_path, output_dir, 
        use_structured_pipeline=use_structured_pipeline
    )
    
    if success:
        print(f"✅ Translation completed successfully!")
        print(f"Output directory: {output_dir}")
    else:
        print("❌ Translation failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 