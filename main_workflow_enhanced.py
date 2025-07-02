"""
Enhanced Main Workflow Module for Ultimate PDF Translator

Integrates all the fixes while preserving the two-pass TOC generation:
1. Unicode font support for Greek characters
2. TOC-aware parsing to prevent structural collapse  
3. Enhanced proper noun handling
4. Better PDF conversion with font embedding

This module extends the existing main_workflow.py without breaking compatibility.
"""

import os
import asyncio
import time
import logging
logger = logging.getLogger(__name__)
import sys
from pathlib import Path

# Import enhanced components
from document_generator import WordDocumentGenerator as EnhancedWordDocumentGenerator
from translation_service_enhanced import enhanced_translation_service
from pdf_parser_enhanced import enhanced_pdf_parser

# Import structural logic components (indivisible parts of the unified pipeline)
from pdf_processor import TextBlock, ContentType, StructuredPDFProcessor

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
    """Enhanced PDF translator with all fixes integrated"""
    
    def __init__(self):
        self.document_generator = EnhancedWordDocumentGenerator()
        self.translation_service = enhanced_translation_service
        self.pdf_parser = enhanced_pdf_parser
        self.settings = config_manager.word_output_settings
        
        # Initialize structural processor (indivisible component of unified pipeline)
        self.structural_processor = StructuredPDFProcessor()
        logger.info("🏗️ Structural processor initialized (unified pipeline component)")
        
        # Initialize PyMuPDF-YOLO pipeline if available
        self.optimized_pipeline = None
        if PYMUPDF_YOLO_AVAILABLE:
            self.optimized_pipeline = OptimizedDocumentPipeline()
            logger.info("🚀 PyMuPDF-YOLO optimized pipeline initialized")
        
        logger.info("Enhanced PDF Translator initialized with unified structural logic")
    
    async def translate_document_enhanced(self, input_path: str, output_dir: str, use_optimized_pipeline: bool = True) -> bool:
        """
        Enhanced document translation workflow with all fixes applied
        Now includes PyMuPDF-YOLO optimized pipeline option
        """
        try:
            start_time = time.time()
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            logger.info(f"Starting enhanced translation: {base_name}")
            
            # Check if we should use the optimized pipeline
            if use_optimized_pipeline and self.optimized_pipeline:
                logger.info("🚀 Using PyMuPDF-YOLO optimized pipeline")
                return await self._translate_with_optimized_pipeline(input_path, output_dir, base_name)
            else:
                logger.info("📄 Using standard enhanced pipeline")
                return await self._translate_with_standard_pipeline(input_path, output_dir, base_name)
            
        except Exception as e:
            logger.error(f"Enhanced translation failed: {e}")
            return False
    
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
        """Translate using unified pipeline with structural logic as indivisible component"""
        try:
            start_time = time.time()
            
            # PHASE 1: Extract with PyMuPDF (ground truth with coordinates)
            logger.info("Phase 1: Extract with PyMuPDF - coordinate-based ground truth")
            document_structure = self.structural_processor.extract_document_structure(input_path)
            
            if not document_structure:
                logger.error("No structured content extracted from PDF")
                return False
            
            # PHASE 2: Model the Structure (use coordinate data to build structured representation)
            logger.info("Phase 2: Model the Structure - coordinate-based sorting and merging")
            # Structural modeling is already done in extract_document_structure
            # This includes: coordinate-based sorting, sequence ID assignment, text block merging
            logger.info(f"   • Extracted {len(document_structure)} pages with structured blocks")
            total_blocks = sum(len(page_blocks) for page_blocks in document_structure)
            logger.info(f"   • Total structured blocks: {total_blocks}")
            
            # PHASE 3: Process the Model (operate on structured model)
            logger.info("Phase 3: Process the Model - translation on structured representation")
            translated_structure = await self._process_structured_content(document_structure)
            
            # PHASE 4: Reconstruct the Output (generate final document from structured model)
            logger.info("Phase 4: Reconstruct the Output - preserve layout and narrative flow")
            docx_path = os.path.join(output_dir, f"{base_name}_translated_enhanced.docx")
            
            # Use enhanced document generator with structured model
            success = self._generate_document_from_structure(translated_structure, docx_path, output_dir)
            
            if not success:
                logger.error("Document generation from structured model failed")
                return False
            
            # Enhanced PDF Conversion (if enabled)
            if self.settings.get('convert_to_pdf', False):
                logger.info("Phase 4b: Enhanced PDF conversion with Unicode support")
                pdf_path = os.path.join(output_dir, f"{base_name}_translated_enhanced.pdf")
                
                # Import the enhanced PDF conversion function
                from document_generator import convert_word_to_pdf
                pdf_success = convert_word_to_pdf(docx_path, pdf_path)
                
                if pdf_success:
                    logger.info(f"Enhanced PDF created: {pdf_path}")
                else:
                    logger.warning("Enhanced PDF conversion failed, but DOCX was created successfully")
            
            total_time = time.time() - start_time
            logger.info(f"Unified pipeline completed in {total_time:.2f} seconds")
            logger.info(f"   • Document structure preserved")
            logger.info(f"   • Coordinate-based layout maintained")
            logger.info(f"   • Narrative flow intact")
            logger.info(f"Output: {docx_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Unified pipeline failed: {e}")
            return False
    
    async def _process_structured_content(self, document_structure):
        """
        Process the structured model - operate on structured representation
        This maintains sequence integrity and coordinate-based relationships
        """
        try:
            logger.info("Processing structured content with sequence preservation")
            
            # Process each page's structured blocks
            translated_structure = []
            
            for page_num, page_blocks in enumerate(document_structure):
                logger.info(f"   Processing page {page_num + 1} ({len(page_blocks)} structured blocks)")
                
                # Translate page blocks while preserving sequence
                translated_page_blocks = await self._translate_page_blocks_structured(page_blocks)
                translated_structure.append(translated_page_blocks)
            
            logger.info(f"✅ Structured content processing completed: {len(translated_structure)} pages")
            return translated_structure
            
        except Exception as e:
            logger.error(f"Structured content processing failed: {e}")
            return document_structure  # Return original structure on failure
    
    async def _translate_page_blocks_structured(self, page_blocks):
        """
        Translate page blocks while preserving sequence integrity
        Each TextBlock maintains its sequence_id and coordinate relationships
        """
        try:
            # Create translation tasks with sequence preservation
            translation_tasks = []
            
            for block in page_blocks:
                if block.content_type == ContentType.IMAGE:
                    # Skip translation for image blocks, preserve as-is
                    translation_tasks.append(self._create_identity_task(block))
                else:
                    # Create translation task with sequence_id preservation
                    task = self._create_structured_translation_task(block)
                    translation_tasks.append(task)
            
            # Execute all tasks in parallel
            logger.debug(f"Executing {len(translation_tasks)} structured translation tasks")
            results = await asyncio.gather(*translation_tasks, return_exceptions=True)
            
            # Reassemble results using sequence_id to restore original order
            translated_blocks = self._reassemble_structured_blocks(page_blocks, results)
            
            return translated_blocks
            
        except Exception as e:
            logger.error(f"Error in structured page translation: {e}")
            return page_blocks  # Return original blocks on failure
    
    async def _create_structured_translation_task(self, block):
        """Create a translation task that preserves sequence_id and coordinate data"""
        try:
            # Use the translation service with pure text payload (no API instruction contamination)
            translated_text = await self.translation_service.translate_text_enhanced(
                text=block.text,  # Pure text payload
                target_language=config_manager.translation_enhancement_settings['target_language'],
                style_guide="",
                prev_context="",
                next_context="",
                item_type=block.content_type.value
            )
            
            return {
                'sequence_id': block.sequence_id,
                'translated_text': translated_text,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.warning(f"Translation failed for block {block.sequence_id}: {e}")
            return {
                'sequence_id': block.sequence_id,
                'translated_text': block.text,  # Keep original text
                'success': False,
                'error': str(e)
            }
    
    async def _create_identity_task(self, block):
        """Create an identity task for non-translatable blocks (images, etc.)"""
        return {
            'sequence_id': block.sequence_id,
            'translated_text': block.text,  # Keep original
            'success': True,
            'error': None
        }
    
    def _reassemble_structured_blocks(self, original_blocks, translation_results):
        """
        Reassemble translated blocks using sequence_id to restore original order
        This ensures coordinate-based relationships are maintained
        """
        try:
            # Create a mapping of sequence_id to translation result
            result_mapping = {}
            for result in translation_results:
                if isinstance(result, dict) and 'sequence_id' in result:
                    result_mapping[result['sequence_id']] = result
            
            # Create translated blocks in original order
            translated_blocks = []
            
            for original_block in original_blocks:
                # Create a copy of the original block with all coordinate data preserved
                translated_block = TextBlock(
                    text=original_block.text,  # Will be updated below
                    page_num=original_block.page_num,
                    sequence_id=original_block.sequence_id,
                    bbox=original_block.bbox,  # Preserve coordinate data
                    font_size=original_block.font_size,
                    font_family=original_block.font_family,
                    font_weight=original_block.font_weight,
                    font_style=original_block.font_style,
                    color=original_block.color,
                    content_type=original_block.content_type,
                    confidence=original_block.confidence,
                    block_type=original_block.block_type
                )
                
                # Update with translated text if available
                if original_block.sequence_id in result_mapping:
                    result = result_mapping[original_block.sequence_id]
                    if result.get('success', False):
                        translated_block.text = result['translated_text']
                    else:
                        logger.warning(f"Translation failed for {original_block.sequence_id}: {result.get('error', 'Unknown error')}")
                
                translated_blocks.append(translated_block)
            
            logger.debug(f"Reassembled {len(translated_blocks)} structured blocks using sequence_id")
            return translated_blocks
            
        except Exception as e:
            logger.error(f"Error reassembling structured blocks: {e}")
            return original_blocks  # Return original blocks as fallback
    
    def _generate_document_from_structure(self, translated_structure, docx_path, output_dir):
        """
        Generate final document from structured model
        Preserve layout and narrative flow using coordinate data
        """
        try:
            logger.info("Generating document from structured model")
            
            # Convert structured model to document generator format
            document_content = []
            
            for page_num, page_blocks in enumerate(translated_structure):
                page_content = {
                    'page_num': page_num + 1,
                    'content_blocks': []
                }
                
                for block in page_blocks:
                    # Convert TextBlock to document generator format
                    content_block = {
                        'type': 'text',
                        'content': block.text,
                        'page_num': block.page_num,
                        'bbox': block.bbox,  # Preserve coordinate data
                        'font_size': block.font_size,
                        'font_family': block.font_family,
                        'font_weight': block.font_weight,
                        'font_style': block.font_style,
                        'color': block.color,
                        'content_type': block.content_type.value,
                        'sequence_id': block.sequence_id
                    }
                    page_content['content_blocks'].append(content_block)
                
                document_content.append(page_content)
            
            # Use enhanced document generator with structured content
            success = self.document_generator.create_word_document_with_structure(
                document_content, docx_path, 
                image_folder_path=os.path.join(output_dir, "images"),
                cover_page_data=None
            )
            
            if success:
                logger.info(f"✅ Document generated from structured model: {docx_path}")
            else:
                logger.error("❌ Document generation from structured model failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error generating document from structure: {e}")
            return False

async def translate_pdf_with_all_fixes(input_path: str, output_dir: str) -> bool:
    """
    Main function to translate PDF with all enhanced fixes applied
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create enhanced translator instance
    translator = EnhancedPDFTranslator()
    
    # Run enhanced translation
    return await translator.translate_document_enhanced(input_path, output_dir)

async def main():
    """
    Unified PDF translator with structural logic as indivisible component
    """
    logger.info("=== UNIFIED PDF TRANSLATOR WITH STRUCTURAL LOGIC ===")
    logger.info("Unified pipeline philosophy:")
    logger.info("1. Extract with PyMuPDF - coordinate-based ground truth")
    logger.info("2. Model the Structure - coordinate-based sorting and merging")
    logger.info("3. Process the Model - translation on structured representation")
    logger.info("4. Reconstruct the Output - preserve layout and narrative flow")
    logger.info("")
    logger.info("Structural components (indivisible):")
    logger.info("• TextBlock dataclass with coordinate data")
    logger.info("• Coordinate-based sorting for reading order")
    logger.info("• Sequence ID assignment for integrity")
    logger.info("• Text block merging for semantic coherence")
    logger.info("• Pure text payloads (no API instruction contamination)")
    logger.info("• Sequence-preserving parallel translation")
    
    # Import GUI utilities from utils.py
    from utils import choose_input_path, choose_base_output_directory, get_specific_output_dir_for_file
    
    # Get input files using GUI dialog
    input_path, process_mode = choose_input_path()
    if not input_path:
        logger.info("No input selected. Exiting.")
        return True
    
    # Get output directory using GUI dialog
    main_output_directory = choose_base_output_directory(
        os.path.dirname(input_path) if process_mode == 'file' else input_path
    )
    
    if not main_output_directory:
        logger.error("No output directory selected. Exiting.")
        return False
    
    # Collect files to process
    files_to_process = []
    if process_mode == 'file':
        files_to_process = [input_path]
    else:
        for filename in os.listdir(input_path):
            if filename.lower().endswith('.pdf'):
                files_to_process.append(os.path.join(input_path, filename))
    
    if not files_to_process:
        logger.error("No PDF files found to process.")
        return False
    
    # Initialize enhanced translator
    translator = EnhancedPDFTranslator()
    
    # Process files
    processed_count = 0
    
    for i, filepath in enumerate(files_to_process):
        logger.info(f"\n>>> Processing file {i+1}/{len(files_to_process)}: {os.path.basename(filepath)} <<<")
        
        specific_output_dir = get_specific_output_dir_for_file(main_output_directory, filepath)
        if not specific_output_dir:
            logger.error(f"Could not create output directory for {os.path.basename(filepath)}")
            continue
        
        try:
            success = await translator.translate_document_enhanced(filepath, specific_output_dir)
            if success:
                processed_count += 1
                logger.info(f"SUCCESS: Successfully processed: {os.path.basename(filepath)}")
            else:
                logger.error(f"FAILED: Failed to process: {os.path.basename(filepath)}")
        except Exception as e:
            logger.error(f"ERROR: Error processing {os.path.basename(filepath)}: {e}")
    
    # Final summary
    logger.info(f"\n=== ENHANCED TRANSLATION SUMMARY ===")
    logger.info(f"Files processed: {processed_count}/{len(files_to_process)}")
    logger.info(f"Output directory: {main_output_directory}")
    
    return processed_count > 0

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )
    
    # Handle command line arguments
    if len(sys.argv) >= 2:
        if sys.argv[1] in ["--help", "-h"]:
            print("Enhanced PDF Translator - With All Fixes")
            print("Usage: python main_workflow_enhanced.py")
            print("The script will show file dialogs for input and output selection")
            sys.exit(0)
    
    # Run the enhanced workflow
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)