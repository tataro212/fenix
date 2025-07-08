"""
Contextual Translation Initializer

This module provides convenient functions to initialize contextual priming
for translation workflows. It integrates with existing translation services
and can be easily added to any translation pipeline.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
import os

# Import existing services
from contextual_priming_service import contextual_priming_service, DocumentContext

logger = logging.getLogger(__name__)

class ContextualTranslationInitializer:
    """
    Convenient wrapper for initializing contextual priming in translation workflows
    """
    
    def __init__(self):
        self.is_initialized = False
        self.current_context: Optional[DocumentContext] = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize_from_text(self, text_sample: str, 
                                 document_title: str = "",
                                 force_reanalysis: bool = False) -> DocumentContext:
        """
        Initialize contextual priming from a text sample
        
        Args:
            text_sample: Sample text from the document (first few pages recommended)
            document_title: Optional document title for better analysis
            force_reanalysis: Force new analysis even if cached version exists
            
        Returns:
            DocumentContext: The analyzed document context
        """
        self.logger.info("🚀 Initializing contextual translation system...")
        
        try:
            # Initialize contextual priming service
            context = await contextual_priming_service.initialize_document_context(
                text_sample, document_title, force_reanalysis
            )
            
            self.current_context = context
            self.is_initialized = True
            
            # Log initialization summary
            self.logger.info("✅ Contextual translation system initialized successfully!")
            self.logger.info(f"📊 Context Summary: {contextual_priming_service.get_context_summary()}")
            
            return context
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize contextual priming: {e}")
            self.is_initialized = False
            raise
    
    async def initialize_from_file(self, file_path: str, 
                                 max_sample_size: int = 8000,
                                 force_reanalysis: bool = False) -> DocumentContext:
        """
        Initialize contextual priming from a file
        
        Args:
            file_path: Path to the document file
            max_sample_size: Maximum number of characters to analyze
            force_reanalysis: Force new analysis even if cached version exists
            
        Returns:
            DocumentContext: The analyzed document context
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text sample from file
        text_sample = await self._extract_text_sample(file_path, max_sample_size)
        document_title = os.path.splitext(os.path.basename(file_path))[0]
        
        return await self.initialize_from_text(text_sample, document_title, force_reanalysis)
    
    async def _extract_text_sample(self, file_path: str, max_size: int) -> str:
        """Extract text sample from various file formats"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return await self._extract_from_pdf(file_path, max_size)
        elif file_ext in ['.txt', '.md']:
            return await self._extract_from_text_file(file_path, max_size)
        elif file_ext in ['.docx', '.doc']:
            return await self._extract_from_word(file_path, max_size)
        else:
            # Try to read as text file
            try:
                return await self._extract_from_text_file(file_path, max_size)
            except Exception as e:
                raise ValueError(f"Unsupported file format: {file_ext}. Error: {e}")
    
    async def _extract_from_pdf(self, file_path: str, max_size: int) -> str:
        """Extract text from PDF file"""
        try:
            # Try multiple PDF extraction methods
            text_sample = ""
            
            # Method 1: Try PyMuPDF (fitz)
            try:
                import fitz
                doc = fitz.open(file_path)
                
                # Extract text from first few pages
                for page_num in range(min(3, doc.page_count)):
                    page = doc.page(page_num)
                    page_text = page.get_text()
                    text_sample += page_text + "\n"
                    
                    if len(text_sample) >= max_size:
                        break
                
                doc.close()
                
                if text_sample.strip():
                    return text_sample[:max_size]
                    
            except ImportError:
                self.logger.debug("PyMuPDF not available, trying alternative methods")
            except Exception as e:
                self.logger.debug(f"PyMuPDF extraction failed: {e}")
            
            # Method 2: Try existing PDF processors
            try:
                from pymupdf_yolo_processor import PyMuPDFYoloProcessor
                processor = PyMuPDFYoloProcessor()
                
                # Extract text blocks from first few pages
                for page_num in range(min(3, 10)):  # Limit to 3 pages
                    try:
                        page_blocks = processor.extract_text_blocks_from_page(file_path, page_num)
                        for block in page_blocks:
                            if hasattr(block, 'content'):
                                text_sample += block.content + "\n"
                            elif hasattr(block, 'text'):
                                text_sample += block.text + "\n"
                            else:
                                text_sample += str(block) + "\n"
                            
                            if len(text_sample) >= max_size:
                                break
                        
                        if len(text_sample) >= max_size:
                            break
                            
                    except Exception:
                        break  # Stop if we can't process more pages
                
                if text_sample.strip():
                    return text_sample[:max_size]
                    
            except Exception as e:
                self.logger.debug(f"PyMuPDF processor extraction failed: {e}")
            
            # Method 3: Try PDF parser
            try:
                from pdf_parser import PDFParser
                parser = PDFParser()
                
                # Try different extraction methods
                if hasattr(parser, 'extract_text_simple'):
                    text_sample = parser.extract_text_simple(file_path, max_pages=3)
                elif hasattr(parser, 'extract_text'):
                    text_sample = parser.extract_text(file_path)
                else:
                    # Fallback: try to read the file directly
                    with open(file_path, 'rb') as f:
                        # This is a very basic fallback - won't work well for PDFs
                        content = f.read(max_size)
                        text_sample = content.decode('utf-8', errors='ignore')
                
                if text_sample.strip():
                    return text_sample[:max_size]
                    
            except Exception as e:
                self.logger.debug(f"PDF parser extraction failed: {e}")
            
            # If all methods fail, provide a fallback
            if not text_sample.strip():
                self.logger.warning("Could not extract text from PDF, using filename as context")
                filename = os.path.basename(file_path)
                text_sample = f"Document filename: {filename}\nUnable to extract text content for context analysis."
            
            return text_sample[:max_size]
            
        except Exception as e:
            self.logger.error(f"Failed to extract text from PDF: {e}")
            # Return filename as fallback
            filename = os.path.basename(file_path)
            return f"Document filename: {filename}\nContext analysis limited due to extraction error."
    
    async def _extract_from_text_file(self, file_path: str, max_size: int) -> str:
        """Extract text from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(max_size)
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read(max_size)
    
    async def _extract_from_word(self, file_path: str, max_size: int) -> str:
        """Extract text from Word document"""
        try:
            import docx
            doc = docx.Document(file_path)
            
            text_sample = ""
            for paragraph in doc.paragraphs:
                text_sample += paragraph.text + "\n"
                if len(text_sample) >= max_size:
                    break
            
            return text_sample[:max_size]
            
        except ImportError:
            raise ImportError("python-docx package required for Word document support")
        except Exception as e:
            self.logger.error(f"Failed to extract text from Word document: {e}")
            raise
    
    def get_context_summary(self) -> str:
        """Get a summary of the current context"""
        if not self.is_initialized:
            return "Contextual priming not initialized"
        
        return contextual_priming_service.get_context_summary()
    
    def get_current_context(self) -> Optional[DocumentContext]:
        """Get the current document context"""
        return self.current_context
    
    def is_ready(self) -> bool:
        """Check if contextual priming is ready to use"""
        return self.is_initialized and self.current_context is not None

# Convenient functions for easy integration

async def initialize_contextual_translation(text_sample: str, 
                                          document_title: str = "",
                                          force_reanalysis: bool = False) -> DocumentContext:
    """
    Convenient function to initialize contextual priming from text
    
    This is the main function to call at the start of any translation workflow.
    
    Example usage:
        # At the start of your translation script
        context = await initialize_contextual_translation(
            document_text[:8000],  # First 8000 characters
            "Research Paper on Machine Learning"
        )
        
        # Now all translations will use contextual priming
    """
    initializer = ContextualTranslationInitializer()
    return await initializer.initialize_from_text(text_sample, document_title, force_reanalysis)

async def initialize_contextual_translation_from_file(file_path: str,
                                                    force_reanalysis: bool = False) -> DocumentContext:
    """
    Convenient function to initialize contextual priming from a file
    
    Example usage:
        # At the start of your translation script
        context = await initialize_contextual_translation_from_file(
            "research_paper.pdf"
        )
        
        # Now all translations will use contextual priming
    """
    initializer = ContextualTranslationInitializer()
    return await initializer.initialize_from_file(file_path, force_reanalysis=force_reanalysis)

def get_contextual_translation_status() -> Dict[str, Any]:
    """
    Get the current status of contextual translation system
    
    Returns:
        Dict with status information
    """
    if not hasattr(contextual_priming_service, 'current_context') or not contextual_priming_service.current_context:
        return {
            'initialized': False,
            'context_available': False,
            'summary': 'Contextual priming not initialized'
        }
    
    context = contextual_priming_service.current_context
    return {
        'initialized': True,
        'context_available': True,
        'summary': contextual_priming_service.get_context_summary(),
        'document_type': context.document_type,
        'domain': context.domain,
        'technical_level': context.technical_level,
        'confidence': context.analysis_confidence,
        'key_terms_count': len(context.key_terminology),
        'creation_time': context.creation_timestamp
    }

# Example integration with existing workflows
async def enhance_existing_translation_workflow(original_workflow_func, 
                                              text_sample: str,
                                              document_title: str = "",
                                              *args, **kwargs):
    """
    Wrapper function to enhance any existing translation workflow with contextual priming
    
    Example usage:
        # Enhance your existing translation function
        result = await enhance_existing_translation_workflow(
            my_translation_function,
            document_text[:8000],  # Context sample
            "Technical Manual",
            # ... other arguments to your function
        )
    """
    # Initialize contextual priming first
    await initialize_contextual_translation(text_sample, document_title)
    
    # Run the original workflow (now with contextual priming active)
    return await original_workflow_func(*args, **kwargs)

# Global initializer instance for convenience
contextual_initializer = ContextualTranslationInitializer() 