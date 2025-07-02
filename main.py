"""
Main Application - Refactored Pipeline Implementation

This module implements the complete refactored pipeline based on the new
core philosophy of structure-aware document reconstruction. It replaces
the brute-force, unordered extraction model with an intelligent,
sequence-preserving approach.

Key Features:
- Structured document model with TextBlock dataclass
- Coordinate-based sorting for correct reading order
- Semantic cohesion through text block merging
- Asynchronous integrity with sequence_id reassembly
- Strict separation of concerns
- Enhanced error handling and fallback mechanisms
"""

import os
import sys
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import the structured components
from pdf_processor import structured_pdf_processor, TextBlock
from text_translator import structured_text_translator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('structured_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

class StructuredDocumentPipeline:
    """
    Main pipeline class that orchestrates the complete document processing workflow.
    
    This class implements the new philosophy of structure-aware document reconstruction,
    ensuring that semantic and sequential integrity are preserved throughout the process.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Structured Document Pipeline initialized")
        
        # Initialize components
        self.pdf_processor = structured_pdf_processor
        self.text_translator = structured_text_translator
        
        # Configuration
        self.target_language = "Ελληνικά"  # Greek
        self.style_guide = "Academic, formal, technical"
        
    async def process_document(self, input_path: str, output_dir: str) -> bool:
        """
        Main entry point for document processing.
        
        This function implements the complete refactored pipeline:
        1. Extract document structure with sequence preservation
        2. Translate content while maintaining integrity
        3. Export results in proper order
        """
        try:
            start_time = time.time()
            
            self.logger.info(f"📄 Starting structured document processing: {input_path}")
            
            # Validate input
            if not self._validate_input(input_path):
                return False
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # PHASE 1: Extract document structure
            self.logger.info("Phase 1: Extracting document structure with sequence preservation")
            document_structure = self.pdf_processor.extract_document_structure(input_path)
            
            if not document_structure:
                self.logger.error("No document structure extracted")
                return False
            
            # Get document statistics
            stats = self.pdf_processor.get_document_statistics(document_structure)
            self.logger.info(f"Document statistics: {stats}")
            
            # PHASE 2: Translate document structure
            self.logger.info("Phase 2: Translating document structure with integrity preservation")
            translated_structure = await self.text_translator.translate_document_structure(
                document_structure, self.target_language, self.style_guide
            )
            
            if not translated_structure:
                self.logger.error("Translation failed")
                return False
            
            # PHASE 3: Export results
            self.logger.info("Phase 3: Exporting translated document")
            output_path = os.path.join(output_dir, f"{Path(input_path).stem}_translated.txt")
            
            export_success = self.text_translator.export_translated_document(
                translated_structure, output_path
            )
            
            if not export_success:
                self.logger.error("Document export failed")
                return False
            
            # Get translation statistics
            translation_stats = self.text_translator.get_translation_statistics(
                document_structure, translated_structure
            )
            self.logger.info(f"Translation statistics: {translation_stats}")
            
            total_time = time.time() - start_time
            self.logger.info(f"✅ Structured document processing completed in {total_time:.2f} seconds")
            self.logger.info(f"Output: {output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in document processing: {e}")
            return False
    
    def _validate_input(self, input_path: str) -> bool:
        """Validate input file"""
        if not os.path.exists(input_path):
            self.logger.error(f"Input file does not exist: {input_path}")
            return False
        
        if not input_path.lower().endswith('.pdf'):
            self.logger.error(f"Input file must be a PDF: {input_path}")
            return False
        
        return True
    
    async def process_multiple_documents(self, input_files: List[str], output_dir: str) -> Dict[str, bool]:
        """
        Process multiple documents in parallel while maintaining individual integrity.
        """
        try:
            self.logger.info(f"📚 Processing {len(input_files)} documents in parallel")
            
            # Create tasks for parallel processing
            tasks = []
            for input_file in input_files:
                task = self.process_document(input_file, output_dir)
                tasks.append(task)
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            success_count = 0
            results_dict = {}
            
            for i, (input_file, result) in enumerate(zip(input_files, results)):
                if isinstance(result, Exception):
                    self.logger.error(f"Error processing {input_file}: {result}")
                    results_dict[input_file] = False
                else:
                    results_dict[input_file] = result
                    if result:
                        success_count += 1
            
            self.logger.info(f"📊 Batch processing completed: {success_count}/{len(input_files)} successful")
            return results_dict
            
        except Exception as e:
            self.logger.error(f"❌ Error in batch processing: {e}")
            return {file: False for file in input_files}

async def main():
    """
    Main application entry point.
    
    This function demonstrates the complete refactored pipeline
    and can be used for testing and development.
    """
    try:
        # Initialize the pipeline
        pipeline = StructuredDocumentPipeline()
        
        # Example usage
        if len(sys.argv) > 1:
            # Command line usage
            input_path = sys.argv[1]
            output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
            
            success = await pipeline.process_document(input_path, output_dir)
            
            if success:
                logger.info("✅ Document processing completed successfully")
                return 0
            else:
                logger.error("❌ Document processing failed")
                return 1
        else:
            # Interactive usage
            logger.info("🔧 Structured Document Pipeline - Interactive Mode")
            logger.info("Enter the path to a PDF file to process:")
            
            input_path = input("PDF file path: ").strip()
            if not input_path:
                logger.info("No file specified, exiting")
                return 0
            
            output_dir = input("Output directory (default: output): ").strip()
            if not output_dir:
                output_dir = "output"
            
            success = await pipeline.process_document(input_path, output_dir)
            
            if success:
                logger.info("✅ Document processing completed successfully")
                return 0
            else:
                logger.error("❌ Document processing failed")
                return 1
                
    except KeyboardInterrupt:
        logger.info("🛑 Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    # Run the main application
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 