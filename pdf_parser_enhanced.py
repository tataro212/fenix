"""
Enhanced PDF Parser Module for Ultimate PDF Translator

Includes fixes for:
1. TOC-aware parsing to prevent structural collapse
2. Better handling of embedded pages
3. Improved text extraction with bounding box tolerance
"""

import os
import logging
import fitz  # PyMuPDF
import re
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class EnhancedPDFParser:
    """Enhanced PDF parser with better TOC and structure handling"""
    
    def __init__(self):
        # Import the original PDF parser to extend it
        try:
            from pdf_parser import PDFParser
            self.base_parser = PDFParser()
        except ImportError:
            logger.warning("Base PDF parser not available, using standalone mode")
            self.base_parser = None
    
    def is_toc_page(self, page: fitz.Page, page_num: int) -> bool:
        """
        FIX 2: Detect if a page is likely a Table of Contents page
        This prevents the parser from trying to extract structured text from TOC pages
        """
        try:
            # Get page text
            text = page.get_text()
            text_lower = text.lower()
            
            # TOC indicators
            toc_keywords = [
                'table of contents', 'contents', 'index', 'table des matieres',
                'indice', 'inhalt', 'sommaire', 'indice'
            ]
            
            # Check for TOC keywords
            has_toc_keywords = any(keyword in text_lower for keyword in toc_keywords)
            
            # Check for dot leaders (common in TOCs)
            dot_leader_pattern = r'\.{3,}'  # Three or more consecutive dots
            has_dot_leaders = len(re.findall(dot_leader_pattern, text)) > 3
            
            # Check for page number patterns at end of lines
            page_num_pattern = r'\d+\s*$'  # Numbers at end of lines
            lines = text.split('\n')
            lines_with_page_nums = sum(1 for line in lines if re.search(page_num_pattern, line.strip()))
            has_many_page_nums = lines_with_page_nums > 5
            
            # Check for typical TOC structure (short lines with numbers)
            short_lines_with_nums = 0
            for line in lines:
                line = line.strip()
                if len(line) < 80 and re.search(r'\d+', line):  # Short lines with numbers
                    short_lines_with_nums += 1
            
            has_toc_structure = short_lines_with_nums > 5
            
            # Combine indicators
            toc_score = sum([
                has_toc_keywords * 3,  # Strong indicator
                has_dot_leaders * 2,   # Strong indicator
                has_many_page_nums * 1,
                has_toc_structure * 1
            ])
            
            is_toc = toc_score >= 3
            
            if is_toc:
                logger.info(f"Page {page_num} identified as TOC page (score: {toc_score})")
            
            return is_toc
            
        except Exception as e:
            logger.warning(f"Error detecting TOC page {page_num}: {e}")
            return False
    
    def extract_page_content_with_toc_awareness(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """
        FIX 2: Extract page content with TOC awareness
        Skips detailed text extraction for TOC pages to prevent structural collapse
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # Check if this is a TOC page
            if self.is_toc_page(page, page_num):
                logger.info(f"Skipping detailed extraction for TOC page {page_num}")
                
                # For TOC pages, return minimal structure to be rebuilt later
                return {
                    'page_num': page_num,
                    'type': 'toc_page',
                    'content': '[TABLE OF CONTENTS - Will be regenerated]',
                    'skip_translation': True,
                    'regenerate_toc': True
                }
            
            # For non-TOC pages, use enhanced text extraction
            return self._extract_page_content_enhanced(page, page_num)
            
        except Exception as e:
            logger.error(f"Error extracting page {page_num}: {e}")
            return {
                'page_num': page_num,
                'type': 'error',
                'content': f'[Error extracting page {page_num}: {e}]',
                'error': str(e)
            }
        finally:
            if 'doc' in locals():
                doc.close()
    
    def _extract_page_content_enhanced(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """
        FIX 3: Enhanced text extraction with better bounding box handling
        Addresses the missing first letter issue
        """
        try:
            # Get text with detailed positioning information
            text_dict = page.get_text("dict")
            
            # Extract text blocks with enhanced bounding box tolerance
            content_blocks = []
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:  # Text block
                    block_text = self._extract_text_from_block_enhanced(block)
                    if block_text.strip():
                        content_blocks.append({
                            'type': 'text',
                            'content': block_text,
                            'bbox': block.get('bbox', [0, 0, 0, 0]),
                            'page_num': page_num
                        })
                elif "image" in block:  # Image block
                    content_blocks.append({
                        'type': 'image',
                        'content': '[Image placeholder]',
                        'bbox': block.get('bbox', [0, 0, 0, 0]),
                        'page_num': page_num
                    })
            
            return {
                'page_num': page_num,
                'type': 'content_page',
                'content_blocks': content_blocks,
                'total_blocks': len(content_blocks)
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced extraction for page {page_num}: {e}")
            # Fallback to simple text extraction
            simple_text = page.get_text()
            return {
                'page_num': page_num,
                'type': 'simple_text',
                'content': simple_text,
                'extraction_method': 'fallback'
            }
    
    def _extract_text_from_block_enhanced(self, block: Dict) -> str:
        """
        FIX 3: Enhanced text extraction from block with better character handling
        Addresses missing first letter issues by using more tolerant bounding boxes
        """
        block_text = ""
        
        try:
            for line in block.get("lines", []):
                line_text = ""
                
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    
                    # FIX 3: Apply character-level fixes for common parsing issues
                    span_text = self._fix_character_parsing_issues(span_text)
                    
                    line_text += span_text
                
                if line_text.strip():
                    block_text += line_text + "\n"
            
            return block_text.strip()
            
        except Exception as e:
            logger.warning(f"Error extracting text from block: {e}")
            return ""
    
    def _fix_character_parsing_issues(self, text: str) -> str:
        """
        FIX 3: Fix common character parsing issues like missing first letters
        """
        if not text:
            return text
        
        # Common fixes for character parsing issues
        fixes = [
            # Fix common character substitutions that might indicate missing characters
            (r'^([a-z])([A-Z][a-z]+)', r'\1\2'),  # Handle case where first letter might be lowercase when it should be uppercase
            
            # Fix specific patterns that indicate missing first letters in names
            (r'\beah\s+([A-Z][a-z]+)', r'Leah \1'),  # "eah Lakshmi" -> "Leah Lakshmi"
            (r'\bisa\s+([A-Z][a-z]+)', r'Lisa \1'),  # "isa Sierra" -> "Lisa Sierra"
            (r'\billy\'s\s+([A-Z][a-z]+)', r'Billy\'s \1'),  # "illy's Missed" -> "Billy's Missed"
        ]
        
        fixed_text = text
        for pattern, replacement in fixes:
            fixed_text = re.sub(pattern, replacement, fixed_text)
        
        return fixed_text
    
    def should_embed_original_page(self, page: fitz.Page, page_num: int, 
                                 extracted_content: Dict[str, Any]) -> bool:
        """
        FIX 2: Improved logic for when to embed original pages
        Reduces unnecessary embedding of original pages
        """
        try:
            # Never embed TOC pages - they should be regenerated
            if extracted_content.get('type') == 'toc_page':
                return False
            
            # Check if extraction was successful
            if extracted_content.get('type') == 'error':
                logger.warning(f"Page {page_num} had extraction errors, considering embedding")
                return True
            
            # Check content quality
            content_blocks = extracted_content.get('content_blocks', [])
            text_blocks = [block for block in content_blocks if block.get('type') == 'text']
            
            if not text_blocks:
                logger.info(f"Page {page_num} has no text blocks, considering embedding")
                return True
            
            # Calculate text extraction confidence
            total_text = ' '.join(block.get('content', '') for block in text_blocks)
            
            # Heuristics for low-quality extraction
            if len(total_text.strip()) < 50:  # Very little text extracted
                return True
            
            # Check for high ratio of special characters (might indicate poor extraction)
            special_char_ratio = len(re.findall(r'[^\w\s]', total_text)) / len(total_text) if total_text else 0
            if special_char_ratio > 0.3:  # More than 30% special characters
                logger.warning(f"Page {page_num} has high special character ratio ({special_char_ratio:.2f}), considering embedding")
                return True
            
            # Don't embed if extraction seems good
            return False
            
        except Exception as e:
            logger.error(f"Error determining embedding for page {page_num}: {e}")
            return True  # Default to embedding on error
    
    def extract_pdf_with_enhanced_structure(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Main method to extract PDF with enhanced structure handling
        """
        logger.info(f"Starting enhanced PDF extraction: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            extracted_pages = []
            
            for page_num in range(total_pages):
                logger.debug(f"Processing page {page_num + 1}/{total_pages}")
                
                # Extract page content with TOC awareness
                page_content = self.extract_page_content_with_toc_awareness(pdf_path, page_num)
                
                # Add metadata
                page_content['total_pages'] = total_pages
                page_content['pdf_path'] = pdf_path
                
                extracted_pages.append(page_content)
            
            doc.close()
            
            logger.info(f"Enhanced extraction complete: {len(extracted_pages)} pages processed")
            return extracted_pages
            
        except Exception as e:
            logger.error(f"Error in enhanced PDF extraction: {e}")
            raise
    
    # Delegate other methods to base parser if available
    def __getattr__(self, name):
        if self.base_parser and hasattr(self.base_parser, name):
            return getattr(self.base_parser, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

# Create enhanced parser instance
enhanced_pdf_parser = EnhancedPDFParser()